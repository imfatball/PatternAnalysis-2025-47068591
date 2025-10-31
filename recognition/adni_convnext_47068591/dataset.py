"""
dataset.py
-----------
Loads grayscale JPEG slices for AD vs NC classification (ADNI dataset).

Expected directory structure:
    ADNI/
        AD_NC/
            train/
                AD/
                    123456_78.jpeg
                    123456_79.jpeg
                    ...
                NC/
                    654321_81.jpeg
                    654321_82.jpeg
                    ...
            test/
                AD/
                NC/
"""

import os
import glob
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def _parse_subject_id(filename: str) -> str:
    """
    Extract subject ID from filenames like '123456_78.jpeg' → '123456'.
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    return stem.split("_")[0]

def build_transforms(image_size=224, train=True):
    pad_to_square = T.Pad((8, 0, 8, 0), padding_mode="reflect")  # 256x240 -> 256x256
    if train:
        return T.Compose([
            pad_to_square,
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(3),
            T.RandomAffine(degrees=0, translate=(0.02,0.02), scale=(0.98,1.02)),
            T.RandomCrop((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5],[0.5]),
            T.RandomErasing(p=0.15, scale=(0.005,0.02), ratio=(0.4,2.0), value='random'),
        ])
    else:
        return T.Compose([
            pad_to_square,
            T.CenterCrop((image_size,image_size)),
            T.ToTensor(),
            T.Normalize([0.5],[0.5]),
        ])


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

        # Define transforms
        self.tf = build_transforms(self.image_size, train=(split=="train" and augment))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        img = self.tf(img)                  # tensor [1,H,W]
        return img, torch.tensor(label, dtype=torch.long), sid
