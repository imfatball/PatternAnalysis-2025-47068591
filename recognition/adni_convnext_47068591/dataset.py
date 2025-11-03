"""
dataset.py
-----------
Loads grayscale JPEG slices for AD vs NC classification (ADNI dataset).

"""

import os
import glob
from typing import List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IM


def _parse_subject_id(filename: str) -> str:
    """
    Extract subject ID from filenames like '123456_78.jpeg' → '123456'.
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    return stem.split("_")[0]


class ADNIJPEGSlicesDataset(Dataset):
    """
    Dataset for grayscale JPEG MRI slices (AD vs NC).

    Returns:
        img: FloatTensor [1, H, W]  (grayscale)
        label: LongTensor 0 (NC) or 1 (AD)
        subject_id: str
    """

    def __init__(
        self,
        root: str,                # e.g. "path/to/ADNI/AD_NC"
        split: str,               # "train" or "test"
        image_size: int = 224,
        augment: bool = True,
        limit_slices_per_subject: Optional[int] = None,
    ):
        super().__init__()
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.root = root
        self.split = split
        self.image_size = image_size
        self.limit_slices_per_subject = limit_slices_per_subject

        # Map class names to labels
        self.class_to_label = {"AD": 1, "NC": 0}

        split_dir = os.path.join(root, split)
        self.samples: List[Tuple[str, int, str]] = []  # (path, label, subject_id)

        # Collect all JPEGs
        for cls in ("AD", "NC"):
            cls_dir = os.path.join(split_dir, cls)
            paths = sorted(glob.glob(os.path.join(cls_dir, "*.jpeg"))) + \
                    sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))

            label = self.class_to_label[cls]
            by_subject = defaultdict(list)
            for p in paths:
                sid = _parse_subject_id(p)
                by_subject[sid].append(p)

            for sid, plist in by_subject.items():
                if self.limit_slices_per_subject and len(plist) > self.limit_slices_per_subject:
                    plist = plist[: self.limit_slices_per_subject]
                for p in plist:
                    self.samples.append((p, label, sid))

        # data augmentations / transforms
        if split == "train" and augment:
            self.tf = T.Compose([
                # --- geometric (PIL space) ---
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.80, 1.00),   
                    ratio=(0.90, 1.10),
                    interpolation=IM.BICUBIC
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=8,             
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        shear=(-5, 5),
                        interpolation=IM.BILINEAR
                    )
                ], p=0.7),
                T.RandomPerspective(distortion_scale=0.20, p=0.3),
                T.ColorJitter(brightness=0.18, contrast=0.18),

                # --- tensor space ---
                T.ToTensor(),                      # → [1, H, W]
                T.Normalize(mean=[0.5], std=[0.5]),
                T.RandomApply([
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
                ], p=0.3),
                T.RandomErasing(
                    p=0.25,
                    scale=(0.01, 0.05),
                    ratio=(0.4, 2.5),
                    value='random'
                ),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((image_size, image_size), interpolation=IM.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        img = self.tf(img)                  # tensor [1,H,W]
        return img, torch.tensor(label, dtype=torch.long), sid
