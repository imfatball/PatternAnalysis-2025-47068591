"""
modules.py
-----------
Self-built ConvNeXt-like (Tiny) for 1-channel input and binary output.

Key pieces:
- ConvNeXtBlock: DW-7x7 -> LN (channels-last) -> 1x1 MLP (expand 4x) -> GELU -> 1x1 (project) -> layer scale -> DropPath -> residual
- Downsample: LN (channels-last) + Conv2d (stride=2)
- ConvNeXtTiny1C: stem (4x4/4) + stages with depths [3,3,9,3], dims [96,192,384,768]
- Head: GAP -> LN -> Dropout -> Linear(1)  => single logit

Use with BCE-with-logits.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Utilities ------------------------------------- #

class DropPath(nn.Module):
    """Stochastic depth (per-sample) — as in ConvNeXt/DeiT. Set drop_prob=0 to disable."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with (N, ...) shaped input. Broadcast along all but batch.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask


class LayerNorm2d(nn.Module):
    """
    Convenience wrapper: apply LayerNorm expecting channels-last.
    We permute (N,C,H,W) -> (N,H,W,C), LN over C, then back.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


# --------------------------- ConvNeXt Building Blocks ------------------------- #

class ConvNeXtBlock(nn.Module):
    """
    One ConvNeXt block:
      DWConv7x7 -> LN (channels-last) -> Linear(4x) -> GELU -> Linear(1x) -> LayerScale(gamma) -> DropPath -> Residual
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise 7x7
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # LN expects channels-last
        self.pwconv1 = nn.Linear(dim, 4 * dim)   # pointwise (expand)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)   # pointwise (project)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.dwconv(x)                # (N,C,H,W)
        x = x.permute(0, 2, 3, 1)         # -> (N,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)         # -> (N,C,H,W)

        x = shortcut + self.drop_path(x)
        return x


class Downsample(nn.Module):
    """
    ConvNeXt downsample layer:
      LN (channels-last) -> Conv2d stride=2
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = LayerNorm2d(in_ch, eps=1e-6)
        self.reduction = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ------------------------------ Full Model ----------------------------------- #

class ConvNeXtTiny1C(nn.Module):
    """
    ConvNeXt-Tiny-like network for 1-channel input.
    - Stem: Conv(1->96, k=4,s=4) + LN
    - Stages: depths [3,3,9,3], dims [96,192,384,768]
    - Head: GAP -> LN -> Dropout -> Linear(1)  (single logit for BCE-with-logits)
    """
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 1,            # 1 => single logit for binary
        depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        head_drop: float = 0.1,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()

        # Stem: patch embedding (4x4, stride 4) then LN
        self.stem_conv = nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4)
        self.stem_ln = LayerNorm2d(dims[0], eps=1e-6)

        # stochastic depth decay rule across all blocks
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, steps=total_blocks).tolist()
        dp_iter = 0

        # Stage 0 (no downsample before first stage)
        stage0 = []
        for _ in range(depths[0]):
            stage0.append(ConvNeXtBlock(dims[0], drop_path=dp_rates[dp_iter], layer_scale_init_value=layer_scale_init_value))
            dp_iter += 1
        self.stage0 = nn.Sequential(*stage0)

        # Stage 1
        self.down1 = Downsample(dims[0], dims[1])
        stage1 = []
        for _ in range(depths[1]):
            stage1.append(ConvNeXtBlock(dims[1], drop_path=dp_rates[dp_iter], layer_scale_init_value=layer_scale_init_value))
            dp_iter += 1
        self.stage1 = nn.Sequential(*stage1)

        # Stage 2
        self.down2 = Downsample(dims[1], dims[2])
        stage2 = []
        for _ in range(depths[2]):
            stage2.append(ConvNeXtBlock(dims[2], drop_path=dp_rates[dp_iter], layer_scale_init_value=layer_scale_init_value))
            dp_iter += 1
        self.stage2 = nn.Sequential(*stage2)

        # Stage 3
        self.down3 = Downsample(dims[2], dims[3])
        stage3 = []
        for _ in range(depths[3]):
            stage3.append(ConvNeXtBlock(dims[3], drop_path=dp_rates[dp_iter], layer_scale_init_value=layer_scale_init_value))
            dp_iter += 1
        self.stage3 = nn.Sequential(*stage3)

        # Head
        self.head_ln = nn.LayerNorm(dims[3], eps=1e-6)  # applied after GAP, so vector LN
        self.dropout = nn.Dropout(head_drop)
        self.fc = nn.Linear(dims[3], num_classes)

        # Weight init 
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem_conv(x)        # (N, C0, H/4, W/4)
        x = self.stem_ln(x)          # channels-last LN wrapper

        # Stages
        x = self.stage0(x)
        x = self.down1(x); x = self.stage1(x)
        x = self.down2(x); x = self.stage2(x)
        x = self.down3(x); x = self.stage3(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)  # (N, C3)
        x = self.head_ln(x)            # vector LN (no permute needed)
        x = self.dropout(x)
        x = self.fc(x)                 # (N, 1) for binary
        return x.squeeze(1)            # (N,) single logit


# ---------------------------- Loss & Metrics --------------------------------- #

def bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[float] = None
) -> torch.Tensor:
    """
    Stable BCE-with-logits for binary classification (targets in {0,1}).
    Set pos_weight>1.0 if AD is the minority class.
    """
    targets = targets.float()
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """
    Returns:
      acc: mean accuracy
      mpt: mean probability for the positive class (on positive samples), NaN if none.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    acc = (preds == targets).float().mean().item()
    mpt = probs[targets == 1].mean().item() if (targets == 1).any() else float('nan')
    return acc, mpt
