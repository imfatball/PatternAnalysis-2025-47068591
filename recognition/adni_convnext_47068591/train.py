"""
train.py
---------
Train a self-built ConvNeXtTiny1C on ADNI JPEG slices (AD vs NC).

Example (local):
    python train.py --root "D:/ADNI/AD_NC" --epochs 10 --batch 16 --image_size 224

Example (Rangpur):
    python train.py --root /home/groups/comp3710/ADNI/AD_NC --epochs 30 --batch 64 --workers 8 --subject_eval
"""

import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from dataset import ADNIJPEGSlicesDataset
from modules import ConvNeXtTiny1C, bce_with_logits_loss, binary_metrics


# ------------------------------ Utilities ------------------------------------ #

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
        plt.xlabel("Epoch"); plt.ylabel("Subject Accuracy"); plt.legend()
        plt.tight_layout(); plt.savefig(outdir / "subject_acc_curve.png", dpi=150); plt.close()


# --------------------------- Train / Eval Loops ------------------------------- #

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    running_loss, running_acc, n_samples = 0.0, 0.0, 0

    for imgs, labels, _sids in loader:  # sids not needed for training
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = bce_with_logits_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = bce_with_logits_loss(logits, labels)
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

    for imgs, labels, _sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
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
    """
    Aggregates predictions per subject by averaging slice probabilities.
    Returns subject-level accuracy.
    """
    model.eval()
    bucket = defaultdict(list)  # sid -> list of (prob, label)

    for imgs, labels, sids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        labs  = labels.cpu().numpy()
        for p, l, sid in zip(probs, labs, sids):
            bucket[sid].append((float(p), int(l)))

    correct, total = 0, 0
    for sid, entries in bucket.items():
        mean_prob = sum(p for p, _ in entries) / len(entries)
        pred = 1 if mean_prob >= 0.5 else 0
        true = entries[0][1]  # all slices of same subject share label
        correct += int(pred == true)
        total   += 1

    return correct / max(total, 1)


# ---------------------------------- Main ------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to ADNI/AD_NC directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--limit_slices_per_subject", type=int, default=None)
    parser.add_argument("--subject_eval", action="store_true", help="Also compute subject-level accuracy on val/test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Root:   {args.root}")

    # Datasets: use 'train' for training and 'test' for validation (as per your layout)
    train_ds = ADNIJPEGSlicesDataset(
        root=args.root,
        split="train",
        image_size=args.image_size,
        augment=True,
        limit_slices_per_subject=args.limit_slices_per_subject
    )
    val_ds = ADNIJPEGSlicesDataset(
        root=args.root,
        split="test",
        image_size=args.image_size,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model / Optimizer / Scheduler
    model = ConvNeXtTiny1C(in_ch=1, num_classes=1, drop_path_rate=0.1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if args.subject_eval:
        history["val_subj_acc"] = []

    best_metric = -1.0  # track best (subject-level if enabled, else slice-level)
    best_path = outdir / "best_model.pt"

    # ------------------------------ Training Loop ---------------------------- #
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = evaluate_slice_level(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        line = f"Epoch {epoch:03d}/{args.epochs} | " \
               f"Train {tr_loss:.4f}/{tr_acc:.3f} | " \
               f"Val {val_loss:.4f}/{val_acc:.3f}"

        # Optional subject-level validation
        subj_metric = None
        if args.subject_eval:
            subj_metric = evaluate_subject_level(model, val_loader, device)
            history["val_subj_acc"].append(subj_metric)
            line += f" | Val-Subject {subj_metric:.3f}"

        line += f" | {time.time() - t0:.1f}s"
        print(line)

        # Choose which metric to monitor for "best"
        monitor = subj_metric if (args.subject_eval and subj_metric is not None) else val_acc
        if monitor > best_metric:
            best_metric = monitor
            save_checkpoint(model, best_path)

    # Save logs & plots
    plot_history(history, outdir)
    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"🏁 Done. Best metric ({'subject' if args.subject_eval else 'slice'}): {best_metric:.3f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
