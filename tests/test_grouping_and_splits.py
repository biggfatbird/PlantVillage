from ppce.grouping import (
    ImageRow,
    assign_group_ids,
    normalized_capture_id,
    official_split_collisions,
)
from ppce.splits import cap_groups, stratified_group_split


def _row(path, filename, official, leaf=None, label="Tomato___healthy"):
    return ImageRow(path, label, filename, official, normalized_capture_id(filename), leaf)


def test_copy_filenames_share_a_capture_id():
    original = normalized_capture_id("abc___RS_GLSp 4620.JPG")
    copy = normalized_capture_id("def___RS_GLSp 4620 copy.jpg")
    second = normalized_capture_id("ghi___RS_GLSp 4620 copy 2.jpg")
    assert original == copy == second


def test_official_split_collision_on_copies():
    rows = [
        _row("a", "abc___RS_GLSp 4620.JPG", "train"),
        _row("b", "def___RS_GLSp 4620 copy.jpg", "test"),
        _row("c", "zzz___other 1.JPG", "train"),
    ]
    groups = assign_group_ids(rows)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    collisions = official_split_collisions(rows, groups)
    assert len(collisions) == 1


def test_same_stem_in_different_classes_stays_separate():
    rows = [
        _row("a", "abc___RS_HL 5376.JPG", "train", label="Blueberry___healthy"),
        _row("b", "def___RS_HL 5376.JPG", "test", label="Soybean___healthy"),
    ]
    groups = assign_group_ids(rows)
    assert groups[0] != groups[1]
    assert official_split_collisions(rows, groups) == []


def test_leaf_map_links_distinct_stems():
    rows = [
        _row("a", "one___alpha 1.JPG", "train", leaf="leaf-7"),
        _row("b", "two___beta 2.JPG", "train", leaf="leaf-7"),
    ]
    groups = assign_group_ids(rows)
    assert groups[0] == groups[1]


def test_group_split_is_disjoint_and_deterministic():
    rows = []
    for label_index, label in enumerate(("Apple___healthy", "Tomato___healthy")):
        for item in range(10):
            filename = f"leaf{label_index}-{item}.jpg"
            rows.append(_row(filename, filename, "train", label=label))
    groups = assign_group_ids(rows)
    first, notes = stratified_group_split(rows, groups, seed=1)
    second, _ = stratified_group_split(rows, groups, seed=1)
    assert first == second
    assert notes == {}
    sides = {}
    for index, group_id in groups.items():
        sides.setdefault(group_id, set()).add(first[group_id])
    assert all(len(seen) == 1 for seen in sides.values())
    used = set(first.values())
    assert used == {"train", "val", "id_test"}


def test_cap_keeps_a_group_whole():
    rows = [
        _row("a1", "stem-a.jpg", "train"),
        _row("a2", "stem-a copy.jpg", "train"),
        _row("b1", "stem-b.jpg", "train"),
    ]
    groups = assign_group_ids(rows)
    assignment = {group_id: "train" for group_id in groups.values()}
    kept = cap_groups(rows, groups, assignment, split="train", max_images=1, seed=0)
    kept_rows = [index for index, group_id in groups.items() if group_id in kept]
    # The two copies are one group. A cap of one image still keeps that whole group
    # or the other singleton, never one copy without the other.
    if groups[0] in kept:
        assert groups[1] in kept
        assert 0 in kept_rows and 1 in kept_rows
