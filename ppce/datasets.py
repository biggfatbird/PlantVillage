"""Manifest rows and torchvision transforms.

Augmentations are applied only when ``train`` is true. Evaluation uses a
deterministic resize and center crop. Normalization uses ImageNet statistics
because the baselines start from ImageNet weights; those statistics are not
computed from the evaluation images.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ppce.models import IMAGENET_MEAN, IMAGENET_STD


POLICIES = ("none", "weak", "strong", "ppce")


def build_transform(policy: str, *, train: bool, image_size: int = 224) -> transforms.Compose:
    if policy not in POLICIES:
        raise ValueError(f"unknown augmentation policy {policy!r}")
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train or policy == "none":
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    color = []
    if policy == "strong":
        color.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            *color,
            transforms.ToTensor(),
            normalize,
        ]
    )


class ManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        policy: str,
        train: bool,
        image_size: int = 224,
        ppce_probability: float = 0.0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.train = train
        self.policy = policy
        self.ppce_probability = ppce_probability if train else 0.0
        self.transform = build_transform(policy, train=train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.train and self.ppce_probability > 0:
            import random

            from ppce.editing import lesion_preserving_edit

            if random.random() < self.ppce_probability:
                edited = lesion_preserving_edit(image, seed=index)
                if edited.segmentation_ok:
                    image = Image.fromarray(edited.edited)
        tensor = self.transform(image)
        return tensor, int(row["label_index"])


def load_manifest(path: str | Path, split: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if split is not None:
        frame = frame[frame["split"] == split].reset_index(drop=True)
    required = {"path", "label_index", "split", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return frame
