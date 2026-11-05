"""Apply the editor to real images and score a trained classifier.

The classifier is already selected. This study does not tune it. Images whose
heuristic finds no lesion are counted and left out of the probability comparison.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ppce.editing import lesion_destroying_control, lesion_preserving_edit
from ppce.models import IMAGENET_MEAN, IMAGENET_STD


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def _true_class_probability(model: nn.Module, image: Image.Image, label: int, transform, device: torch.device) -> float:
    batch = transform(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(batch), dim=1)[0, label]
    return float(probabilities.item())


def run_editing_study(
    model: nn.Module,
    frame: pd.DataFrame,
    *,
    device: torch.device,
    image_size: int = 224,
    max_images: int = 48,
    seed: int = 42,
) -> dict:
    model.eval()
    diseased = frame[frame["kind"] == "disease"].reset_index(drop=True)
    if diseased.empty:
        diseased = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if len(diseased) > max_images:
        chosen = rng.choice(len(diseased), size=max_images, replace=False)
        diseased = diseased.iloc[np.sort(chosen)].reset_index(drop=True)

    transform = _eval_transform(image_size)
    rows = []
    for index, record in diseased.iterrows():
        image = Image.open(record["path"]).convert("RGB")
        preserved = lesion_preserving_edit(image, seed=seed + int(index))
        destroyed = lesion_destroying_control(image)
        label = int(record["label_index"])
        row = {
            "path": record["path"],
            "label": record["label"],
            "label_index": label,
            "segmentation_ok": bool(preserved.segmentation_ok and destroyed.segmentation_ok),
            "lesion_fraction": preserved.lesion_fraction,
            "ppce_lesion_pixel_match": preserved.lesion_pixel_match,
            "ppce_outside_mae": preserved.outside_lesion_mae,
            "destroy_lesion_pixel_match": destroyed.lesion_pixel_match,
            "destroy_outside_mae": destroyed.outside_lesion_mae,
        }
        if row["segmentation_ok"]:
            original_p = _true_class_probability(model, image, label, transform, device)
            preserved_image = Image.fromarray(preserved.edited)
            destroyed_image = Image.fromarray(destroyed.edited)
            preserved_p = _true_class_probability(model, preserved_image, label, transform, device)
            destroyed_p = _true_class_probability(model, destroyed_image, label, transform, device)
            row.update(
                {
                    "p_original": original_p,
                    "p_ppce": preserved_p,
                    "p_destroy": destroyed_p,
                    "abs_delta_ppce": abs(preserved_p - original_p),
                    "abs_delta_destroy": abs(destroyed_p - original_p),
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    ok = table[table["segmentation_ok"]] if not table.empty else table
    summary = {
        "n_images": int(len(table)),
        "n_segmented": int(len(ok)),
        "mean_lesion_fraction": _mean(table, "lesion_fraction"),
        "mean_ppce_lesion_pixel_match": _mean(ok, "ppce_lesion_pixel_match"),
        "mean_ppce_outside_mae": _mean(ok, "ppce_outside_mae"),
        "mean_destroy_lesion_pixel_match": _mean(ok, "destroy_lesion_pixel_match"),
        "mean_destroy_outside_mae": _mean(ok, "destroy_outside_mae"),
        "mean_abs_delta_ppce": _mean(ok, "abs_delta_ppce"),
        "mean_abs_delta_destroy": _mean(ok, "abs_delta_destroy"),
    }
    if not ok.empty and "abs_delta_destroy" in ok:
        summary["fraction_destroy_moves_prob_more"] = float((ok["abs_delta_destroy"] > ok["abs_delta_ppce"]).mean())
    else:
        summary["fraction_destroy_moves_prob_more"] = float("nan")
    return {"summary": summary, "rows": table.to_dict(orient="records"), "example": _example_payload(diseased, seed)}


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame or frame[column].dropna().empty:
        return float("nan")
    return float(frame[column].mean())


def _example_payload(frame: pd.DataFrame, seed: int) -> dict | None:
    if frame.empty:
        return None
    record = frame.iloc[0]
    image = Image.open(record["path"]).convert("RGB")
    preserved = lesion_preserving_edit(image, seed=seed)
    destroyed = lesion_destroying_control(image)
    return {
        "label": record["label"],
        "original": np.asarray(image),
        "leaf_mask": preserved.leaf_mask,
        "lesion_mask": preserved.lesion_mask,
        "ppce": preserved.edited,
        "destroy": destroyed.edited,
        "meta": {key: value for key, value in asdict(preserved).items() if key not in {"edited", "leaf_mask", "lesion_mask"}},
    }
