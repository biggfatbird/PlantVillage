"""Figures drawn only from measured tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_domain_shift(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.model}\n{row.augmentation}" for row in metrics.itertuples()]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.3), 4.5))
    ax.bar(x - width / 2, metrics["id_test_accuracy"], width, label="PlantVillage in-domain test", color="#2a6f4e")
    ax.bar(x + width / 2, metrics["plantdoc_accuracy"], width, label="PlantDoc external test", color="#c47b2b")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("In-domain accuracy and PlantDoc accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, group in history.groupby("run"):
        ax.plot(group["epoch"], group["val_macro_f1"], marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Validation macro-F1 by epoch")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_edit_panel(example: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lesion = example["lesion_mask"]
    overlay = example["original"].copy()
    overlay[lesion] = (0.5 * overlay[lesion] + np.array([255, 40, 40]) * 0.5).astype(np.uint8)
    panels = [
        (example["original"], "Original"),
        (overlay, "Lesion mask"),
        (example["ppce"], "Lesion-preserving edit"),
        (example["destroy"], "Lesion-destroying control"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(example["label"], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
