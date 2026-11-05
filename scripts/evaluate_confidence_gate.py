#!/usr/bin/env python3
"""Score the confidence gate on the held-out PlantVillage images and, when
present, on the Zenodo apple masks.

This does not train a model and does not modify results/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppce.confidence_edit import confidence_gated_edit, lesion_pixels_from_three_level_mask


ZENODO_URL = "https://zenodo.org/api/records/18659728/files/Restricted_Plant_Village_Dataset.zip/content"
MODES = ("unconstrained", "constrained", "gated")


def _ensure_zenodo(destination: Path) -> Path | None:
    root = destination / "Restricted_Plant_Village_Dataset"
    if (root / "images").is_dir() and (root / "masks").is_dir():
        return root
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "Restricted_Plant_Village_Dataset.zip"
    if not archive.exists():
        print("downloading Zenodo apple masks", flush=True)
        urllib.request.urlretrieve(ZENODO_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)
    if (root / "images").is_dir():
        return root
    return None


def _nanmean(values: list) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return float(np.mean(numbers))


def _summarize(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n": 0}
    summary = {"n": int(len(frame))}
    summary["emitted_rate"] = float(frame["emitted_edit"].mean())
    summary["accepted_rate"] = float(frame["gate_accepted"].mean())
    reasons = Counter()
    for value in frame["reasons"]:
        for reason in str(value).split("|"):
            if reason:
                reasons[reason] += 1
    summary["reason_counts"] = dict(reasons)
    for column in (
        "core_pixel_match",
        "uncertain_pixel_match",
        "environment_mae",
        "leaf_body_mae",
        "gt_lesion_pixel_match",
        "resegmentation_core_recall",
    ):
        if column in frame and frame[column].notna().any():
            summary[f"mean_{column}"] = float(frame[column].mean())
    emitted = frame[frame["emitted_edit"]]
    if "gt_lesion_pixel_match" in frame and not emitted.empty and emitted["gt_lesion_pixel_match"].notna().any():
        summary["mean_gt_lesion_pixel_match_when_emitted"] = float(emitted["gt_lesion_pixel_match"].mean())
        summary["n_emitted_with_gt"] = int(emitted["gt_lesion_pixel_match"].notna().sum())
    if "environment_mae" in emitted and not emitted.empty and emitted["environment_mae"].notna().any():
        summary["mean_environment_mae_when_emitted"] = float(emitted["environment_mae"].mean())
    return summary


def _gt_scores(decision_image: np.ndarray, source: np.ndarray, lesion: np.ndarray, core: np.ndarray | None) -> dict:
    if lesion.any():
        match = float(np.all(decision_image[lesion] == source[lesion], axis=-1).mean())
    else:
        match = None
    scores = {"gt_lesion_pixel_match": match, "gt_lesion_fraction": float(lesion.mean())}
    return scores


def evaluate_manifest(manifest: Path, split: str) -> list[dict]:
    frame = pd.read_csv(manifest)
    frame = frame[(frame["split"] == split) & (frame["kind"] == "disease")]
    rows = []
    for index, record in frame.reset_index(drop=True).iterrows():
        image = np.asarray(Image.open(record["path"]).convert("RGB"))
        for mode in MODES:
            decision = confidence_gated_edit(image, seed=1000 + int(index), mode=mode)
            rows.append(
                {
                    "source": "plantvillage_manifest",
                    "split": split,
                    "label": record["label"],
                    "path": record["path"],
                    "mode": mode,
                    "emitted_edit": decision.emitted_edit,
                    "gate_accepted": decision.gate_accepted,
                    "reasons": "|".join(decision.reasons),
                    "core_fraction": decision.core_fraction,
                    "uncertain_fraction": decision.uncertain_fraction,
                    "core_pixel_match": decision.core_pixel_match,
                    "uncertain_pixel_match": decision.uncertain_pixel_match,
                    "environment_mae": decision.environment_mae,
                    "leaf_body_mae": decision.leaf_body_mae,
                    "resegmentation_core_recall": decision.resegmentation_core_recall,
                    "gt_lesion_pixel_match": None,
                    "gt_lesion_fraction": None,
                }
            )
    return rows


def evaluate_zenodo(root: Path) -> list[dict]:
    rows = []
    pairs = []
    for split in ("train", "val", "test"):
        for image_path in sorted((root / "images" / split).glob("*")):
            mask_candidates = list((root / "masks" / split).glob(image_path.stem + ".*"))
            if not mask_candidates:
                continue
            pairs.append((split, image_path, mask_candidates[0]))
    for index, (split, image_path, mask_path) in enumerate(pairs):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        lesion = lesion_pixels_from_three_level_mask(np.asarray(Image.open(mask_path).convert("L")))
        for mode in MODES:
            decision = confidence_gated_edit(image, seed=2000 + index, mode=mode)
            # Core is not returned. Recompute agreement coverage against the lesion
            # from the unchanged source only once per image via gated decision fractions.
            gt = _gt_scores(decision.image, image, lesion, None)
            rows.append(
                {
                    "source": "zenodo_apple_masks",
                    "split": split,
                    "label": image_path.stem.rsplit("_", 1)[0],
                    "path": str(image_path),
                    "mode": mode,
                    "emitted_edit": decision.emitted_edit,
                    "gate_accepted": decision.gate_accepted,
                    "reasons": "|".join(decision.reasons),
                    "core_fraction": decision.core_fraction,
                    "uncertain_fraction": decision.uncertain_fraction,
                    "core_pixel_match": decision.core_pixel_match,
                    "uncertain_pixel_match": decision.uncertain_pixel_match,
                    "environment_mae": decision.environment_mae,
                    "leaf_body_mae": decision.leaf_body_mae,
                    "resegmentation_core_recall": decision.resegmentation_core_recall,
                    **gt,
                }
            )
    # Mask quality of the heuristic core against the manual spots, measured on the source.
    from ppce.confidence_edit import separate_regions

    quality = []
    for _split, image_path, mask_path in pairs:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        lesion = lesion_pixels_from_three_level_mask(np.asarray(Image.open(mask_path).convert("L")))
        separation = separate_regions(image)
        if lesion.any():
            recall = float((separation.core & lesion).sum()) / float(lesion.sum())
        else:
            recall = None
        if separation.core.any():
            precision = float((separation.core & lesion).sum()) / float(separation.core.sum())
        else:
            precision = None
        quality.append({"recall": recall, "precision": precision, "lesion_fraction": float(lesion.mean())})
    return rows, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    parser.add_argument("--zenodo-dir", default="data/external")
    parser.add_argument("--out", default="results/confidence_gate")
    parser.add_argument("--skip-zenodo", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = evaluate_manifest(Path(args.manifest), "id_test")
    quality = []
    if not args.skip_zenodo:
        root = _ensure_zenodo(Path(args.zenodo_dir))
        if root is not None:
            zenodo_rows, quality = evaluate_zenodo(root)
            rows.extend(zenodo_rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "edit_decisions.csv", index=False)
    summary = {
        "plantvillage_id_test_disease": {
            mode: _summarize(frame[(frame.source == "plantvillage_manifest") & (frame["mode"] == mode)].to_dict("records"))
            for mode in MODES
        },
        "zenodo_apple": {
            mode: _summarize(frame[(frame.source == "zenodo_apple_masks") & (frame["mode"] == mode)].to_dict("records"))
            for mode in MODES
        },
        "zenodo_mask_encoding": "0 background, 255 leaf, any other value lesion. Checked visually on test/scab_1 and test/cedar_rust_1.",
        "heuristic_core_against_zenodo_lesion": {
            "n": len(quality),
            "mean_recall": _nanmean([row["recall"] for row in quality]),
            "mean_precision": _nanmean([row["precision"] for row in quality]),
            "mean_lesion_fraction": _nanmean([row["lesion_fraction"] for row in quality]),
        },
        "zenodo_url": ZENODO_URL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
