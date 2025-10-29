# Alzheimer’s Disease Classification using ConvNeXt on ADNI Dataset

*Author: Man Hin Lai (s47068591)*
*Course: COMP3710 Pattern Analysis (Topic-Recognition Branch)*
*Difficulty: Hard*

---

## 1. Project Overview

This project aims to classify **Alzheimer’s Disease (AD)** vs **Cognitively Normal (CN)** brain scans from the **ADNI dataset** using a **ConvNeXt** vision transformer-style CNN.

---

## 2. Problem Description

Alzheimer’s disease is a progressive neurodegenerative disorder identifiable in brain MRI scans.
The **goal** is to train a model that distinguishes AD from CN subjects using volumetric MRI slices.

---

## 3. How the Algorithm Works

The model is based on **ConvNeXt-Tiny**, a convolutional architecture that adapts Transformer-like design principles into pure CNNs.The network was fine-tuned on preprocessed 2D slices from ADNI MRI volumes using transfer learning from ImageNet weights.The pipeline includes:

1. **Data Preprocessing:** Intensity normalization, skull stripping (via preprocessed ADNI dataset on Rangpur), and slice extraction.
2. **Augmentation:** Random flips, rotations, and intensity scaling to improve generalization.
3. **Training:** Binary cross-entropy loss with AdamW optimizer and cosine-annealing learning-rate scheduling.
4. **Evaluation:** Accuracy, ROC-AUC, confusion matrix, and Grad-CAM visualization for interpretability.

---

## 4. Dataset and Preprocessing

- **Dataset Source:** `/home/groups/comp3710/ADNI` (on Rangpur HPC)
- **Classes:** Alzheimer’s Disease (AD) and Cognitive Normal (CN)
- **Format:** Preprocessed NIfTI volumes (`.nii.gz`)
- **Splits:** 70 % training / 15 % validation / 15 % test
- **Justification:** Stratified splitting maintains class balance and ensures unseen subjects per split.
- **Tools Used:** `nibabel` for I/O, `torchvision.transforms` for augmentations.

*Preprocessing steps and rationale:*

- Converted 3D volumes into mid-axial 2D slices to balance training time and memory.
- Normalized intensities to [0, 1].
- Removed non-brain tissue using provided preprocessed dataset.

---

## 5. Dependencies and Environment

| Package      | Version        | Purpose                 |
| ------------ | -------------- | ----------------------- |
| Python       | 3.11.9         | Environment             |
| PyTorch      | 2.5.1 + cu124  | Deep learning (GPU)     |
| Torchvision  | 0.20.1 + cu124 | Model zoo / transforms  |
| Nibabel      | 5.2+           | MRI I/O                 |
| Scikit-learn | 1.4+           | Metrics / preprocessing |
| Matplotlib   | 3.8+           | Plotting                |
| Pandas       | 2.2+           | Data handling           |
| TQDM         | 4.66+          | Progress bars           |
| Pillow       | 10.3+          | Image utilities         |

> Reproducibility: All results were obtained on Windows 10 + RTX 4070 Ti (CUDA 12.4 build).
> To replicate:
>
> ```bash
> git clone https://github.com/imfatball/PatternAnalysis-2025-47068591.git
> cd PatternAnalysis-2025-47068591/recognition/adni_convnext_47068591
> python -m venv .venv && .venv\Scripts\activate
> pip install -r ../../requirements.txt
> python train.py
> ```

---

## 6. Example Usage

### Training

```bash
python train.py --epochs 50 --batch_size 16 --lr 1e-4
```
