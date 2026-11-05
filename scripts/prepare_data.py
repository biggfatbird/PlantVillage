#!/usr/bin/env python3
"""Build a leakage-safe manifest and extract the images it names.

PlantVillage bytes come from the color images inside ``data.zip`` of
``mohanty/PlantVillage``. PlantDoc bytes come from ``geraldmc/plantdoc-full``.
Nothing in this script trains a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ppce.grouping import (
    ImageRow,
    assign_group_ids,
    lookup_leaf_map_id,
    normalized_capture_id,
    official_split_collisions,
)
from ppce.mapping import ALIGNED, PD_TO_PV, PV_TO_INDEX, assert_alignment_is_bijective
from ppce.splits import cap_groups, stratified_group_split


def _hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def _rows_from_split_files(split_root: Path, leaf_map: dict, classes: set[str] | None) -> list[ImageRow]:
    rows: list[ImageRow] = []
    for official in ("train", "test"):
        for line in (split_root / f"color_{official}.txt").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            if len(parts) < 4:
                continue
            class_name, filename = parts[2], parts[3]
            if classes is not None and class_name not in classes:
                continue
            capture = normalized_capture_id(filename)
            rows.append(
                ImageRow(
                    path=line,
                    class_name=class_name,
                    filename=filename,
                    official_split=official,
                    capture_id=capture,
                    leaf_map_id=lookup_leaf_map_id(capture, class_name, leaf_map),
                )
            )
    return rows


def _kind(label: str) -> str:
    for pair in ALIGNED:
        if pair.plantvillage == label:
            return pair.kind
    return "unknown"


def prepare(args: argparse.Namespace) -> None:
    assert_alignment_is_bijective()
    out = Path(args.out)
    image_root = out / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    leaf_map_path = _hf_file("mohanty/PlantVillage", "leaf_grouping/leaf-map.json")
    split_train = _hf_file("mohanty/PlantVillage", "splits/color_train.txt")
    _hf_file("mohanty/PlantVillage", "splits/color_test.txt")
    zip_path = _hf_file("mohanty/PlantVillage", "data.zip")
    leaf_map = json.loads(leaf_map_path.read_text())
    split_root = split_train.parent

    all_rows = _rows_from_split_files(split_root, leaf_map, classes=None)
    all_groups = assign_group_ids(all_rows)
    full_collisions = official_split_collisions(all_rows, all_groups)

    aligned = {pair.plantvillage for pair in ALIGNED}
    rows = _rows_from_split_files(split_root, leaf_map, classes=aligned)
    group_of = assign_group_ids(rows)
    aligned_collisions = official_split_collisions(rows, group_of)
    assignment, notes = stratified_group_split(
        rows,
        group_of,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    kept: set[str] = set()
    for split, cap in (
        ("train", args.max_train),
        ("val", args.max_val),
        ("id_test", args.max_id_test),
    ):
        kept |= cap_groups(rows, group_of, assignment, split=split, max_images=cap, seed=args.seed)

    our_sides: dict[str, set[str]] = {}
    records = []
    missing = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for index, row in enumerate(rows):
            group_id = group_of[index]
            if group_id not in kept:
                continue
            split = assignment[group_id]
            our_sides.setdefault(group_id, set()).add(split)
            if row.path not in names:
                missing += 1
                continue
            destination = image_root / "plantvillage" / f"{len(records):05d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row.path) as source, destination.open("wb") as target:
                target.write(source.read())
            records.append(
                {
                    "path": str(destination),
                    "source": "plantvillage",
                    "split": split,
                    "label": row.class_name,
                    "label_index": PV_TO_INDEX[row.class_name],
                    "kind": _kind(row.class_name),
                    "group_id": group_id,
                    "official_split": row.official_split,
                    "capture_id": row.capture_id,
                    "filename": row.filename,
                }
            )

    plantdoc_counts: Counter[str] = Counter()
    skipped_plantdoc: Counter[str] = Counter()
    from datasets import load_dataset

    dataset = load_dataset("geraldmc/plantdoc-full", split="train", streaming=True)
    for row in dataset:
        raw_label = row["class_label"]
        canonical = PD_TO_PV.get(raw_label)
        plantdoc_counts[raw_label] += 1
        if canonical is None:
            skipped_plantdoc[raw_label] += 1
            continue
        image = row["image"].convert("RGB")
        destination = image_root / "plantdoc" / f"{len(records):05d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=95)
        records.append(
            {
                "path": str(destination),
                "source": "plantdoc",
                "split": "plantdoc",
                "label": canonical,
                "label_index": PV_TO_INDEX[canonical],
                "kind": _kind(canonical),
                "group_id": f"plantdoc:{row['filename']}",
                "official_split": row.get("split", ""),
                "capture_id": normalized_capture_id(str(row["filename"])),
                "filename": row["filename"],
            }
        )

    frame = pd.DataFrame(records)
    manifest = out / "manifest.csv"
    frame.to_csv(manifest, index=False)
    our_collisions = sum(1 for sides in our_sides.values() if len(sides) > 1)
    counts = frame.groupby(["split", "label"]).size().reset_index(name="n")
    audit = {
        "plantvillage_repo": "mohanty/PlantVillage",
        "plantdoc_repo": "geraldmc/plantdoc-full",
        "zip_path": str(zip_path),
        "n_aligned_classes": len(ALIGNED),
        "n_color_images_listed": len(all_rows),
        "official_split_groups_on_both_sides": len(full_collisions),
        "official_split_groups_on_both_sides_aligned": len(aligned_collisions),
        "official_collision_examples": full_collisions[:8],
        "our_split_group_collisions": our_collisions,
        "split_notes": notes,
        "missing_zip_entries": missing,
        "n_train": int((frame["split"] == "train").sum()) if len(frame) else 0,
        "n_val": int((frame["split"] == "val").sum()) if len(frame) else 0,
        "n_id_test": int((frame["split"] == "id_test").sum()) if len(frame) else 0,
        "n_plantdoc": int((frame["split"] == "plantdoc").sum()) if len(frame) else 0,
        "plantdoc_raw_counts": dict(plantdoc_counts),
        "plantdoc_unmapped_counts": dict(skipped_plantdoc),
        "per_class_counts": counts.to_dict(orient="records"),
        "caps": {
            "max_train": args.max_train,
            "max_val": args.max_val,
            "max_id_test": args.max_id_test,
        },
        "seed": args.seed,
    }
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: audit[key] for key in ("n_train", "n_val", "n_id_test", "n_plantdoc", "official_split_groups_on_both_sides", "our_split_group_collisions")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/prepared")
    parser.add_argument("--max-train", type=int, default=48)
    parser.add_argument("--max-val", type=int, default=16)
    parser.add_argument("--max-id-test", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
