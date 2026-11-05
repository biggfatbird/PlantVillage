"""Group images that are copies or the same mapped leaf.

The published PlantVillage color split keeps most leaf-map groups on one side,
but copies whose stem is missing from the leaf map still cross the official
train and test files. Grouping is therefore based on two links:

- the same class plus the same normalized filename stem, which is what
  "... copy" and "... copy 2" share;
- the same non-ambiguous leaf-map identifier, when the stem is in the map.

Stems are qualified by class. Codes such as ``RS_HL 5376`` are reused for
different crops, so a global stem match would chain unrelated leaves together.

A group is a connected component of those links. Splits are assigned to groups,
never to individual files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRow:
    path: str
    class_name: str
    filename: str
    official_split: str
    capture_id: str
    leaf_map_id: str | None


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def normalized_capture_id(filename: str) -> str:
    """Stem shared by an image and its 'copy' duplicates."""
    name = filename.replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = name.split("copy")[0]
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return " ".join(name.strip().lower().split())


def lookup_leaf_map_id(
    capture_id: str,
    class_name: str,
    leaf_map: dict[str, list[str]] | None,
) -> str | None:
    if not leaf_map or capture_id not in leaf_map:
        return None
    suggestions = leaf_map[capture_id]
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return None


def assign_group_ids(rows: list[ImageRow]) -> dict[int, str]:
    """Return a map from row index to a stable group id."""
    uf = UnionFind()
    for index, row in enumerate(rows):
        node = f"row:{index}"
        uf.add(node)
        uf.union(node, f"capture:{row.class_name}:{row.capture_id}")
        if row.leaf_map_id:
            uf.union(node, f"leaf:{row.leaf_map_id}")

    members: dict[str, list[int]] = {}
    for index in range(len(rows)):
        root = uf.find(f"row:{index}")
        members.setdefault(root, []).append(index)

    group_of: dict[int, str] = {}
    for indexes in members.values():
        capture_ids = sorted({rows[i].capture_id for i in indexes})
        leaf_ids = sorted({rows[i].leaf_map_id for i in indexes if rows[i].leaf_map_id})
        class_key = "|".join(sorted({rows[i].class_name for i in indexes}))
        if leaf_ids:
            group_id = class_key + "|leaf:" + "|".join(leaf_ids)
        else:
            group_id = class_key + "|cap:" + capture_ids[0]
        for index in indexes:
            group_of[index] = group_id
    return group_of


def official_split_collisions(rows: list[ImageRow], group_of: dict[int, str]) -> list[str]:
    """Group ids that the published train/test files place on both sides."""
    sides: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        sides.setdefault(group_of[index], set()).add(row.official_split)
    return sorted(group_id for group_id, seen in sides.items() if len(seen) > 1)
