"""
predict.py — demo/report artifacts from the saved checkpoint

What this script does (no training here):
  1) Loads the TEST split via ADNIJPEGSlicesDataset.
  2) Builds the ConvNeXtTiny1C model and loads weights from a checkpoint.
  3) Computes slice-level and subject-level metrics (acc, AUC, sensitivity, specificity).
  4) Saves a confusion matrix image, ROC curve image, an example grid image, and a JSON with metrics.

Usage:
    python predict.py --root <ADNI/AD_NC root> --ckpt runs/best_model.pt
    example:python recognition/adni_convnext_47068591/predict.py --root /home/groups/comp3710/ADNI/AD_NC --ckpt runs/best_model.pt

Outputs (in --out, default "runs/"):
    demo_confusion.png, demo_roc.png, demo_examples.png, demo_metrics.json
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

from dataset import ADNIJPEGSlicesDataset
from modules import ConvNeXtTiny1C


def compute_slice_metrics(model, loader, device):
    """
    Evaluate model performance at the SLICE level.

    Returns a dict with:
      - acc: accuracy over slices
      - auc: ROC AUC over slices (probabilities vs labels)
      - sens/spec: sensitivity/specificity from a 0.5 threshold
      - y_true, y_prob, y_pred: arrays for downstream plots (ROC/confusion)
    """
    model.eval()
    y_true, y_prob = [], []

    # Disable grad; iterate over all test batches
    with torch.inference_mode():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)                    # raw logits (shape [N])
            probs = torch.sigmoid(logits).cpu().numpy()  # convert to p(AD)
            y_prob.append(probs)
            y_true.append(labels.numpy())

    # Concatenate across all batches
    y_prob = np.concatenate(y_prob).astype(np.float64)
    y_true = np.concatenate(y_true).astype(np.int64)

    # Threshold at 0.5 for hard predictions
    y_pred = (y_prob >= 0.5).astype(np.int64)

    # Slice accuracy
    acc = (y_pred == y_true).mean().item()

    # ROC AUC can fail if there is only one class present → guard with try/except
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    # Confusion matrix and derived metrics at the 0.5 threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)  # recall for AD
    spec = tn / max(tn + fp, 1)  # recall for NC

    return dict(acc=acc, auc=auc, sens=sens, spec=spec,
                y_true=y_true, y_prob=y_prob, y_pred=y_pred)


def compute_subject_metrics(model, loader, device):
    """
    Evaluate model performance at the SUBJECT level.

    Approach:
      - Accumulate slice probabilities per subject_id in a bucket.
      - Average probabilities per subject (mean p(AD) across its slices).
      - Threshold averaged probability at 0.5 to get a subject prediction.
    """
    model.eval()
    bucket = defaultdict(list)  # sid -> list of (p, label)

    with torch.inference_mode():
        for imgs, labels, sids in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            # Group each slice's prob with its subject id
            for p, lab, sid in zip(probs, labels.numpy(), sids):
                bucket[sid].append((float(p), int(lab)))

    # Aggregate per subject (mean probability); labels are consistent for a subject
    y_true, y_prob = [], []
    for sid, items in bucket.items():
        mean_prob = sum(p for p, _ in items) / len(items)
        y_prob.append(mean_prob)
        y_true.append(items[0][1])

    # Convert to arrays for metrics
    y_true = np.array(y_true, dtype=np.int64)
    y_prob = np.array(y_prob, dtype=np.float64)

    # Subject-level hard predictions at 0.5
    y_pred = (y_prob >= 0.5).astype(np.int64)

    # Subject accuracy
    acc = (y_pred == y_true).mean().item()

    # Subject ROC AUC (may be NaN if only one class present)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    # Subject confusion matrix-derived metrics (using the 0.5 threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    return dict(acc=acc, auc=auc, sens=sens, spec=spec,
                y_true=y_true, y_prob=y_prob, y_pred=y_pred)


def plot_subject_confusion(cm, outpath):
    """
    Plot a labeled 2x2 confusion matrix for SUBJECT-LEVEL classification (NC vs AD).

    Args:
        cm (np.ndarray): 2x2 array in the form [[TN, FP], [FN, TP]]
        outpath (Path or str): path to save the figure (PNG)

    Conventions:
        - Rows are TRUE subject diagnoses (NC row, AD row)
        - Columns are PREDICTED subject diagnoses (NC col, AD col)
    """
    tn, fp, fn, tp = cm.ravel()
    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted NC', 'Predicted AD'])
    ax.set_yticklabels(['True NC', 'True AD'])

    # Axis titles (explicitly labeled as subject-level)
    ax.set_xlabel("Predicted Diagnosis (Subject-Level)", fontsize=11)
    ax.set_ylabel("True Diagnosis (Subject-Level)", fontsize=11)

    # Value annotations
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=13, color="black")

    # Add gridlines between cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Updated title
    ax.set_title("Confusion Matrix (Subject-Level)", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)



def plot_roc(y_true, y_prob, outpath, title="ROC"):
    """
    Plot a standard ROC curve with diagonal (chance) line and AUC in legend.

    Args:
        y_true (array-like): Ground-truth labels (0/1)
        y_prob (array-like): Predicted probabilities for the positive class (AD)
        outpath (Path/str): Where to save the PNG
        title (str): Figure title
    """
    # ROC points: varying threshold from 1 → 0
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # AUC can be undefined when only one class is present
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # chance line
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def make_examples_grid(ds, model, device, outpath, n_per_class=6):
    """
    Create a grid of TEST subjects with their subject-level predicted probability p(AD).

    Each subject's probability is computed as the mean of all slice probabilities.
    The grid shows one representative slice (middle slice) for each subject, with a
    banner that explicitly labels the following:
        - Subject ID
        - True Diagnosis (NC/AD)
        - Model's predicted probability for AD (subject-level mean)

    Args:
        ds: ADNIJPEGSlicesDataset (split="test").
        model: Trained model (in eval mode).
        device: "cuda" or "cpu".
        outpath: Path to save the resulting image grid.
        n_per_class: Max subjects per class (NC/AD) to show.
    """
    import numpy as np
    from collections import defaultdict
    from PIL import Image, ImageOps, ImageDraw, ImageFont

    model.eval()

    subj_to_indices = defaultdict(list)
    subj_to_label = {}
    for idx, (_path, y, sid) in enumerate(ds.samples):
        subj_to_indices[sid].append(idx)
        subj_to_label.setdefault(sid, y)

    subj_probs = {}
    with torch.inference_mode():
        for sid, idx_list in subj_to_indices.items():
            slice_probs = []
            for idx in idx_list:
                path, _y, _sid = ds.samples[idx]
                pil = Image.open(path).convert("L")
                x = ds.tf(pil).unsqueeze(0).to(device)
                p = torch.sigmoid(model(x)).item()
                slice_probs.append(p)
            subj_probs[sid] = float(np.mean(slice_probs)) if slice_probs else float("nan")

    ad_sids = [sid for sid, y in subj_to_label.items() if y == 1][:n_per_class]
    nc_sids = [sid for sid, y in subj_to_label.items() if y == 0][:n_per_class]
    chosen_sids = ad_sids + nc_sids
    if len(chosen_sids) == 0:
        return

    tiles = []
    with torch.inference_mode():
        for sid in chosen_sids:
            idx_list = subj_to_indices[sid]
            label = subj_to_label[sid]
            p_subj = subj_probs.get(sid, float("nan"))

            idx_list_sorted = sorted(idx_list)
            rep_idx = idx_list_sorted[len(idx_list_sorted) // 2]
            path, _y, _sid = ds.samples[rep_idx]

            pil = Image.open(path).convert("L")
            disp = ImageOps.equalize(pil.resize((224, 224)))
            draw = ImageDraw.Draw(disp)

            # Use two-line banner for clarity and space efficiency
            text_line1 = f"Subject ID: {sid}"
            text_line2 = f"True: {'AD' if label==1 else 'NC'}  |  Pred p(AD): {p_subj:.2f}"

            # Draw a taller banner rectangle
            banner_height = 40
            draw.rectangle([0, 0, 223, banner_height], fill=0)

            # Use a slightly smaller font if available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = None  # fallback to default

            # Write each line separately
            draw.text((4, 4), text_line1, fill=255, font=font)
            draw.text((4, 20), text_line2, fill=255, font=font)

            tiles.append(disp.convert("RGB"))

    cols = 6
    rows = int(np.ceil(len(tiles) / cols))
    w, h = tiles[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for idx, im in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid.paste(im, (c * w, r * h))
    grid.save(outpath)




def main():
    """
    Entrypoint:
      - Parse CLI args
      - Construct test dataset/loader
      - Load checkpointed model
      - Compute and save metrics/plots/examples
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)                # ADNI/AD_NC root
    ap.add_argument("--ckpt", type=str, default="runs/best_model.pt") # checkpoint path
    ap.add_argument("--batch", type=int, default=32)                  # test-time batch size
    ap.add_argument("--image_size", type=int, default=224)            # must match training/eval tf
    ap.add_argument("--workers", type=int, default=4)                 # dataloader workers
    ap.add_argument("--out", type=str, default="runs")                # output directory
    args = ap.parse_args()

    # Device + output directory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Dataset/loader (TEST split only — no training here)
    test_ds = ADNIJPEGSlicesDataset(root=args.root, split="test",
                                    image_size=args.image_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Model (same architecture as training) + checkpoint load
    model = ConvNeXtTiny1C(in_ch=1, num_classes=1, drop_path_rate=0.15, head_drop=0.25).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Compute metrics (slice- and subject-level)
    slice_res = compute_slice_metrics(model, test_loader, device)
    subj_res  = compute_subject_metrics(model, test_loader, device)

    # Save a compact JSON with the numeric metrics (arrays omitted)
    metrics = {
        "slice": {k: float(v) if not isinstance(v, (list, np.ndarray)) else None
                  for k, v in slice_res.items() if k not in ("y_true", "y_prob", "y_pred")},
        "subject": {k: float(v) if not isinstance(v, (list, np.ndarray)) else None
                    for k, v in subj_res.items() if k not in ("y_true", "y_prob", "y_pred")}
    }
    with open(outdir / "demo_metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    print("Demo metrics:", metrics)

    # Plots (slice-level confusion + ROC)
    cm = confusion_matrix(slice_res["y_true"], slice_res["y_pred"], labels=[0, 1])
    plot_subject_confusion(cm, outdir / "demo_confusion.png")
    plot_roc(slice_res["y_true"], slice_res["y_prob"], outdir / "demo_roc.png", title="ROC (slice-level)")

    # Qualitative example grid
    make_examples_grid(test_ds, model, device, outdir / "demo_examples.png", n_per_class=6)

    print("Demo complete.")
    print(f"Artifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
