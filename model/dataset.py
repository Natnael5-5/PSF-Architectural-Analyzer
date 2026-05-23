"""
dataset.py - StudioPSFDataset
==============================
PyTorch Dataset that loads architectural studio renderings and their
three PSF ground-truth scores (Creativity, Belonging, EaseOfUse).

Expected directory layout
--------------------------
<root>/
  images/              ← folder of .png renderings  (configurable via image_dir)
  labels.csv           ← CSV with columns: Name, Creativity, Belonging, EaseOfUse

CSV format example
------------------
Name,Creativity,Belonging,EaseOfUse
image_001,3.4,2.9,3.7
image_002,4.1,3.5,2.8
...

Notes
-----
* DINOv2 ViT-B/14 native resolution is 518×518 (patch 14 → 37×37 grid).
* ImageNet mean/std normalization is applied in all transforms.
* The dataset filters out any CSV rows whose image file cannot be found.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  Transform factory
# ──────────────────────────────────────────────────────────────────────────────


def get_transforms(
    img_size: int = 518,
    is_train: bool = True,
    aug_cfg: Optional[dict] = None,
) -> transforms.Compose:
    """
    Return a torchvision transform pipeline.

    DINOv2 ViT-B/14 expects 518×518 images (14×37 = 518 patches).
    Images are resized/cropped to *img_size* and normalized with ImageNet stats.

    Parameters
    ----------
    img_size : int
        Target spatial resolution (height = width).
    is_train : bool
        If True, applies data-augmentation transforms; otherwise deterministic.
    aug_cfg : dict | None
        Augmentation config from the YAML config file. Ignored when is_train=False.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # ── Inference / Validation transforms (deterministic) ────────────────────
    if not is_train:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ── Training transforms (with augmentation) ───────────────────────────────
    tf_list = []
    aug = aug_cfg if aug_cfg is not None else {}
    enabled = aug.get("enabled", True)

    if enabled:
        # 1. Geometric augmentations
        rc_cfg = aug.get("random_resized_crop", {"scale": (0.8, 1.0)})
        tf_list.append(transforms.RandomResizedCrop(
            img_size,
            scale=tuple(rc_cfg["scale"]),
            interpolation=transforms.InterpolationMode.BICUBIC
        ))
        tf_list.append(transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip_prob", 0.5)))

        rot = aug.get("rotation_degrees", 10)
        if rot > 0:
            tf_list.append(transforms.RandomRotation(degrees=rot))

        persp = aug.get("random_perspective", {})
        if persp.get("prob", 0) > 0:
            tf_list.append(transforms.RandomPerspective(
                distortion_scale=persp.get("distortion_scale", 0.5),
                p=persp.get("prob", 0.2)
            ))

        affine = aug.get("random_affine", {})
        if affine.get("prob", 0) > 0:
            tf_list.append(transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=affine.get("degrees", 0),
                    translate=tuple(affine.get("translate", (0.1, 0.1))),
                    scale=tuple(affine.get("scale", (0.9, 1.1)))
                )], p=affine.get("prob", 0.2)
            ))

        # 2. Photometric augmentations
        cj = aug.get("color_jitter", {})
        if cj:
            tf_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.05)
            ))

        gb = aug.get("gaussian_blur", {})
        if gb.get("prob", 0) > 0:
            tf_list.append(transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=gb.get("kernel_size", 3),
                    sigma=tuple(gb.get("sigma", (0.1, 2.0)))
                )], p=gb.get("prob", 0.3)
            ))
    else:
        # Augmentation disabled: deterministic resize + crop
        tf_list.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
        tf_list.append(transforms.CenterCrop(img_size))

    # 3. Final conversion + normalization
    tf_list.append(transforms.ToTensor())
    tf_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(tf_list)


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────


class StudioPSFDataset(Dataset):
    """
    PyTorch Dataset for the PSF Architectural Rendering corpus.

    Parameters
    ----------
    root_dir : str | Path
        Folder that contains *image_dir* and *labels_csv*.
    image_dir : str
        Sub-folder name inside *root_dir* containing the images.
    labels_csv : str
        CSV file name (relative to *root_dir*).
    image_col : str
        Column in the CSV that holds the image stem (e.g. ``image_1``).
    label_cols : list[str]
        Columns to read as regression targets.
    image_ext : str
        File extension to append when the CSV stem has none (default ``.png``).
    transform : torchvision.transforms.Compose | None
        If *None*, the default deterministic transforms are applied.
    indices : list[int] | None
        Subset of row indices to use (for train/val splitting).
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_dir: str = "images",
        labels_csv: str = "labels.csv",
        image_col: str = "Name",
        label_cols: Optional[List[str]] = None,
        image_ext: str = ".png",
        transform: Optional[transforms.Compose] = None,
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.root_dir  = Path(root_dir)
        self.img_dir   = self.root_dir / image_dir
        self.image_ext = image_ext
        self.transform = transform or get_transforms(is_train=False)

        # Read CSV
        csv_path = self.root_dir / labels_csv
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        if label_cols is None:
            label_cols = ["Creativity", "Belonging", "EaseOfUse"]
        self.label_cols = label_cols

        # Optional subsetting (for train/val splits)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        # Filter to images that actually exist on disk
        valid_rows = [
            idx for idx, row in df.iterrows()
            if self._resolve_image_path(row[image_col]) is not None
        ]
        df = df.iloc[valid_rows].reset_index(drop=True)

        self.image_names: List[str] = df[image_col].tolist()
        self.labels: np.ndarray    = df[label_cols].values.astype(np.float32)

        print(f"[StudioPSFDataset] Loaded {len(self)} samples from {csv_path}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_image_path(self, stem: str) -> Optional[Path]:
        """Try common extensions if the file is not found directly."""
        for ext in [self.image_ext, ".png", ".jpg", ".jpeg"]:
            candidate = self.img_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            # Maybe stem already contains an extension
            candidate2 = self.img_dir / stem
            if candidate2.exists():
                return candidate2
        return None

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns
        -------
        image  : torch.Tensor  shape (3, H, W)
        labels : torch.Tensor  shape (num_outputs,)   — Creativity, Belonging, EaseOfUse
        name   : str           — image stem (useful for result logging)
        """
        stem     = self.image_names[idx]
        img_path = self._resolve_image_path(stem)
        assert img_path is not None, f"Image not found for stem={stem!r}"

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, stem
