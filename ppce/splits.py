"""Stratified group splits and per-class caps.

Validation and in-domain test groups are chosen from the PlantVillage pool.
PlantDoc is never part of this split. Caps keep whole groups, so a leaf's
copies stay together even after subsampling.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from ppce.grouping import ImageRow


def _majority(labels: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return max(sorted(counts), key=counts.get)


def stratified_group_split(
    rows: list[ImageRow],
    group_of: dict[int, str],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign each group to train, val, or id_test.

    Returns the assignment and a note for groups whose files have more than
    one class. Those groups are kept intact and stratified by majority class.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in group_of.items():
        by_group[group_id].append(index)

    labels: dict[str, str] = {}
    notes: dict[str, str] = {}
    for group_id, indexes in by_group.items():
        classes = [rows[i].class_name for i in indexes]
        labels[group_id] = _majority(classes)
        if len(set(classes)) > 1:
            notes[group_id] = "mixed-class group assigned by majority label"

    by_class: dict[str, list[str]] = defaultdict(list)
    for group_id, label in labels.items():
        by_class[label].append(group_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        n = len(groups)
        n_test = math.floor(n * test_fraction)
        n_val = math.floor(n * val_fraction)
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            overflow = n_test + n_val - (n - 1)
            n_val = max(0, n_val - overflow)
        if n_test + n_val > n:
            n_test = min(n_test, n)
            n_val = min(n_val, n - n_test)
        for group_id in groups[:n_test]:
            assignment[group_id] = "id_test"
        for group_id in groups[n_test : n_test + n_val]:
            assignment[group_id] = "val"
        for group_id in groups[n_test + n_val :]:
            assignment[group_id] = "train"
        if n < 3:
            notes[f"class:{label}"] = (
                f"only {n} group(s); could not put one group in each of train, val, and id_test"
            )
    return assignment, notes


def cap_groups(
    rows: list[ImageRow],
    group_of: dict[int, str],
    assignment: dict[str, str],
    *,
    split: str,
    max_images: int | None,
    seed: int,
) -> set[str]:
    """Keep whole groups in one split until the image cap is reached.

    ``max_images`` None or a number below 1 means no cap. The first group of
    a class is kept even when it alone exceeds the cap, so a class is not
    dropped because one leaf has many copies.
    """
    if max_images is None or max_images < 1:
        return {group_id for group_id, name in assignment.items() if name == split}

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in group_of.items():
        if assignment.get(group_id) == split:
            by_group[group_id].append(index)

    by_class: dict[str, list[str]] = defaultdict(list)
    for group_id, indexes in by_group.items():
        classes = [rows[i].class_name for i in indexes]
        label = max(set(classes), key=classes.count)
        by_class[label].append(group_id)

    rng = random.Random(seed)
    kept: set[str] = set()
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        count = 0
        for position, group_id in enumerate(groups):
            size = len(by_group[group_id])
            if position == 0 or count + size <= max_images:
                kept.add(group_id)
                count += size
            if count >= max_images:
                break
    return kept


def assert_groups_are_pure(assignment: dict[str, str], group_of: dict[int, str], rows: list[ImageRow]) -> None:
    seen: dict[str, str] = {}
    for index, group_id in group_of.items():
        split = assignment[group_id]
        previous = seen.setdefault(group_id, split)
        if previous != split:
            raise AssertionError(f"group {group_id} is in more than one split")
        _ = rows[index]
