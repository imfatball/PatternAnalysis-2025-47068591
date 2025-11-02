"""
train.py (no-CLI needed)
------------------------
Set CONFIG below (especially ROOT) and run:
    python train.py
You can still override with flags (e.g., --root PATH), but it's optional.
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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
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
    DROP_PATH_RATE=0.15,
    HEAD_DROP=0.25,
    WARMUP_EPOCHS=5,
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

def do_mixup(x, y, alpha=0.2):
    if alpha <= 0: 
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y = y.float()
    y_mix = lam * y + (1 - lam) * y[idx]  # soft targets in [0,1]
    return x_mix, y_mix, lam


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    running_loss, running_acc, n_samples = 0.0, 0.0, 0
    autocast_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()

    for imgs, labels, _sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # 🔸 Apply MixUp (only for training)
        imgs, y_soft, lam = do_mixup(imgs, labels, alpha=0.2)

        if scaler is not None:
            with autocast_ctx:
                logits = model(imgs)
                # use soft targets (MixUp already smooths)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), y_soft.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_ctx:
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), y_soft.view(-1))
            loss.backward()
            optimizer.step()
            
        acc, _ = binary_metrics(logits.detach(), labels)
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc  += acc * bs
        n_samples    += bs

    return running_loss / n_samples, running_acc / n_samples


@torch.no_grad()
def evaluate_slice_level(model, loader, device):
    model.eval()
    total_loss, total_acc, n_samples = 0.0, 0.0, 0
    autocast_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()

    for imgs, labels, _sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx:
            logits = model(imgs)
            loss = bce_with_logits_loss(logits, labels)
        acc, _ = binary_metrics(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc  += acc * bs
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples


@torch.no_grad()
def evaluate_subject_level(model, loader, device):
    model.eval()
    bucket = defaultdict(list)
    autocast_ctx = torch.amp.autocast('cuda') if device == "cuda" else nullcontext()

    for imgs, labels, sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx:
            logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        labs  = labels.cpu().numpy()
        for p, l, sid in zip(probs, labs, sids):
            bucket[sid].append((float(p), int(l)))

    correct, total = 0, 0
    for sid, entries in bucket.items():
        mean_prob = sum(p for p, _ in entries) / len(entries)
        pred = 1 if mean_prob >= 0.5 else 0
        true = entries[0][1]
        correct += int(pred == true)
        total   += 1

    return correct / max(total, 1)


def main():
    # Optional CLI overrides (but all have defaults from CONFIG)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=str, default=CONFIG["ROOT"])
    parser.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"])
    parser.add_argument("--batch", type=int, default=CONFIG["BATCH"])
    parser.add_argument("--lr", type=float, default=CONFIG["LR"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["WEIGHT_DECAY"])
    parser.add_argument("--out", type=str, default=CONFIG["OUT"])
    parser.add_argument("--workers", type=int, default=CONFIG["WORKERS"])
    parser.add_argument("--image_size", type=int, default=CONFIG["IMAGE_SIZE"])
    parser.add_argument("--limit_slices_per_subject", type=int, default=CONFIG["LIMIT_SLICES_PER_SUBJECT"])
    parser.add_argument("--subject_eval", action="store_true" if CONFIG["SUBJECT_EVAL"] else "store_false")
    parser.add_argument("--seed", type=int, default=CONFIG["SEED"])
    args, _ = parser.parse_known_args()

    # If CONFIG["SUBJECT_EVAL"] is True but flag not passed, enforce it
    args.subject_eval = CONFIG["SUBJECT_EVAL"] or args.subject_eval

    # Seed / device
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Root:   {args.root}")

    # Datasets
    train_ds = ADNIJPEGSlicesDataset(
        root=args.root, split="train",
        image_size=args.image_size, augment=True,
        limit_slices_per_subject=args.limit_slices_per_subject
    )
    val_ds = ADNIJPEGSlicesDataset(
        root=args.root, split="test",
        image_size=args.image_size, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model / Optim / Sched
    model = ConvNeXtTiny1C(
        in_ch=1, num_classes=1,
        drop_path_rate=CONFIG["DROP_PATH_RATE"],
        head_drop=CONFIG["HEAD_DROP"]
    ).to(device)
    warmup_epochs = CONFIG["WARMUP_EPOCHS"]
    

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # after warmup, delegate to cosine by keeping lambda=1 and stepping base_scheduler
        return 1.0

    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    base_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    schedulers = (warmup, base_scheduler)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if args.subject_eval:
        history["val_subj_acc"] = []

    best_metric = -1.0
    best_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = evaluate_slice_level(model, val_loader, device)
        if epoch <= warmup_epochs:
            schedulers[0].step()
        else:
            schedulers[1].step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(val_acc)

        line = f"Epoch {epoch:03d}/{args.epochs} | Train {tr_loss:.4f}/{tr_acc:.3f} | Val {val_loss:.4f}/{val_acc:.3f}"
        subj_metric = None
        if args.subject_eval:
            subj_metric = evaluate_subject_level(model, val_loader, device)
            history["val_subj_acc"].append(subj_metric)
            line += f" | Val-Subject {subj_metric:.3f}"
        line += f" | {time.time()-t0:.1f}s"
        print(line)

        monitor = subj_metric if (args.subject_eval and subj_metric is not None) else val_acc
        if monitor > best_metric:
            best_metric = monitor
            save_checkpoint(model, best_path)

    # Save logs
    plot_history(history, outdir)
    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"🏁 Done. Best metric ({'subject' if args.subject_eval else 'slice'}): {best_metric:.3f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
