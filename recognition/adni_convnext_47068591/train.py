"""
train.py (improved, no subject sampler)
---------------------------------------
Set CONFIG below (especially ROOT) and run:
    python train.py
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt
import numpy as np

from dataset import ADNIJPEGSlicesDataset
from modules import ConvNeXtTiny1C, bce_with_logits_loss, binary_metrics

# ====================== USER CONFIG ============================= #
CONFIG = dict(
    ROOT=r"C:\Users\harri\UQ\COMP3710\COMP3710_A3\PatternAnalysis-2025-47068591\data\ADNI\AD_NC",
    EPOCHS=60,
    BATCH=16,
    LR=2e-4,
    WEIGHT_DECAY=2e-3,
    OUT="runs",
    WORKERS=4,
    IMAGE_SIZE=224,
    LIMIT_SLICES_PER_SUBJECT=12, 
    SUBJECT_EVAL=True,
    SEED=42,

    # Regularization
    DROP_PATH_RATE=0.15,
    HEAD_DROP=0.25,
    WARMUP_EPOCHS=5,
    ETA_MIN=1e-5,

    # Stability & Generalization
    CLIP_NORM=1.0,       # gradient clipping
    MIXUP_ALPHA=0.2,     # set 0.0 to disable
    EMA=False,           # enable for smoother eval
    EMA_DECAY=0.999,
    EVAL_TTA=False,      # test-time augmentation (flip)
)
# ============================================================================ #


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Saved checkpoint: {path}")


def plot_history(history: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "loss_curve.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "acc_curve.png", dpi=150); plt.close()

    if "val_subj_acc" in history and len(history["val_subj_acc"]) > 0:
        plt.figure()
        plt.plot(history["val_subj_acc"], label="Val Subject Acc")
        plt.xlabel("Epoch"); plt.ylabel("Subject Acc"); plt.legend()
        plt.tight_layout(); plt.savefig(outdir / "subject_acc_curve.png", dpi=150); plt.close()


# ---------- MixUp ----------
def do_mixup(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y = y.float()
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix, lam


# ---------- EMA ----------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.collected = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self.collected = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.collected[n] = p.detach().clone()
            p.data.copy_(self.shadow[n].data)

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.collected[n].data)
        self.collected = {}


# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, device, scaler=None, mixup_alpha=0.2, clip_norm=1.0, ema: EMA=None):
    model.train()
    running_loss, running_acc, n_samples = 0.0, 0.0, 0
    autocast_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()

    for imgs, labels, _sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # MixUp augmentation
        imgs, y_soft, _ = do_mixup(imgs, labels, alpha=mixup_alpha)

        if scaler is not None:
            with autocast_ctx:
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), y_soft.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_ctx:
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), y_soft.view(-1))
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        acc, _ = binary_metrics(logits.detach(), labels)
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc  += acc * bs
        n_samples    += bs

    return running_loss / n_samples, running_acc / n_samples


# ---------- Evaluation ----------
@torch.inference_mode()
def evaluate_slice_level(model, loader, device, tta=False):
    model.eval()
    total_loss, total_acc, n_samples = 0.0, 0.0, 0
    autocast_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()

    for imgs, labels, _sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx:
            logits = model(imgs)
            if tta:
                logits = 0.5 * (logits + model(torch.flip(imgs, dims=[-1])))
            loss = bce_with_logits_loss(logits, labels)
        acc, _ = binary_metrics(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc  += acc * bs
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples


@torch.inference_mode()
def evaluate_subject_level(model, loader, device, tta=False):
    model.eval()
    bucket = defaultdict(list)
    autocast_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()

    for imgs, labels, sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx:
            logits = model(imgs)
            if tta:
                logits = 0.5 * (logits + model(torch.flip(imgs, dims=[-1])))

        for lg, lab, sid in zip(logits.detach().cpu(), labels.cpu(), sids):
            bucket[sid].append((float(lg), int(lab)))

    correct, total = 0, 0
    for sid, entries in bucket.items():
        mean_logit = sum(lg for lg, _ in entries) / len(entries)
        pred = 1 if mean_logit >= 0.0 else 0
        true = entries[0][1]
        correct += int(pred == true)
        total   += 1

    return correct / max(total, 1)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=str, default=CONFIG["ROOT"])
    args, _ = parser.parse_known_args()

    set_seed(CONFIG["SEED"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(CONFIG["OUT"]); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Root:   {args.root}")

    # Datasets
    train_ds = ADNIJPEGSlicesDataset(
        root=args.root, split="train",
        image_size=CONFIG["IMAGE_SIZE"], augment=True,
        limit_slices_per_subject=CONFIG["LIMIT_SLICES_PER_SUBJECT"]
    )
    val_ds = ADNIJPEGSlicesDataset(
        root=args.root, split="test",
        image_size=CONFIG["IMAGE_SIZE"], augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH"], shuffle=True,
                              num_workers=CONFIG["WORKERS"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["BATCH"], shuffle=False,
                              num_workers=CONFIG["WORKERS"], pin_memory=True)

    # Model
    model = ConvNeXtTiny1C(
        in_ch=1, num_classes=1,
        drop_path_rate=CONFIG["DROP_PATH_RATE"],
        head_drop=CONFIG["HEAD_DROP"]
    ).to(device)

    # Optimizer (decoupled WD)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias") or ("norm" in n.lower()):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = AdamW(
        [{"params": decay, "weight_decay": CONFIG["WEIGHT_DECAY"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=CONFIG["LR"], betas=(0.9, 0.999)
    )

    # Scheduler: warmup + cosine
    warmup_epochs = CONFIG["WARMUP_EPOCHS"]
    main_epochs   = CONFIG["EPOCHS"] - warmup_epochs
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=CONFIG["ETA_MIN"])
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    ema = EMA(model, decay=CONFIG["EMA_DECAY"]) if CONFIG["EMA"] else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_subj_acc": []}
    best_metric, best_path = -1.0, outdir / "best_model.pt"

    print("\n=== Training ===")
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler, mixup_alpha=CONFIG["MIXUP_ALPHA"],
            clip_norm=CONFIG["CLIP_NORM"], ema=ema
        )

        if ema: ema.apply_to(model)
        val_loss, val_acc = evaluate_slice_level(model, val_loader, device, tta=CONFIG["EVAL_TTA"])
        subj_metric = evaluate_subject_level(model, val_loader, device, tta=CONFIG["EVAL_TTA"]) if CONFIG["SUBJECT_EVAL"] else val_acc
        if ema: ema.restore(model)

        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(val_acc)
        history["val_subj_acc"].append(subj_metric)
        scheduler.step()

        line = f"Epoch {epoch:03d}/{CONFIG['EPOCHS']} | Train {tr_loss:.4f}/{tr_acc:.3f} | Val {val_loss:.4f}/{val_acc:.3f} | Subj {subj_metric:.3f} | LR={optimizer.param_groups[0]['lr']:.6g} | {time.time()-t0:.1f}s"
        print(line)

        if subj_metric > best_metric:
            best_metric = subj_metric
            save_checkpoint(model, best_path)

    plot_history(history, outdir)
    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n🏁 Done. Best subject acc: {best_metric:.3f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
