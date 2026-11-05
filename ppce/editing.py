"""Lesion-preserving counterfactual edit, and a lesion-destroying control.

This is a classical mask-and-composite prototype. It is not a trained
generative editor. The lesion mask is a color heuristic: green tissue is
found with the excess-green index, holes inside that tissue are filled so
brown or yellow spots count as part of the leaf, and those non-green pixels
are the lesion.

The preserving edit may change background and non-lesion leaf color, then
copies the original lesion pixels back so those pixels match exactly. The
destroying control replaces lesion pixels with the median color of the
remaining leaf and leaves every other pixel unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class EditResult:
    edited: np.ndarray
    leaf_mask: np.ndarray
    lesion_mask: np.ndarray
    lesion_pixel_match: float | None
    outside_lesion_mae: float | None
    lesion_fraction: float
    segmentation_ok: bool


def _as_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("expected an RGB image")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def segment_leaf_and_lesion(rgb: np.ndarray, *, max_side: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean leaf and lesion masks at the original resolution."""
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        small = np.asarray(
            Image.fromarray(rgb).resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.Resampling.BILINEAR,
            )
        )
    else:
        small = rgb

    red = small[:, :, 0].astype(np.int16)
    green = small[:, :, 1].astype(np.int16)
    blue = small[:, :, 2].astype(np.int16)
    excess_green = 2 * green - red - blue
    green_tissue = excess_green > 20
    if int(green_tissue.sum()) < 16:
        leaf_small = np.zeros(small.shape[:2], dtype=bool)
        lesion_small = np.zeros(small.shape[:2], dtype=bool)
    else:
        grown = ndimage.binary_dilation(green_tissue, iterations=8)
        filled = ndimage.binary_fill_holes(grown)
        leaf_small = _largest_component(filled)
        interior = ndimage.binary_erosion(leaf_small, iterations=1)
        lesion_small = interior & ~green_tissue
        lesion_small = ndimage.binary_opening(lesion_small, iterations=1)
        labeled, count = ndimage.label(lesion_small)
        if count:
            sizes = np.bincount(labeled.ravel())
            small_ids = np.where(sizes < 12)[0]
            if len(small_ids):
                lesion_small = lesion_small.copy()
                lesion_small[np.isin(labeled, small_ids)] = False

    if scale < 1:
        leaf = _upsample_mask(leaf_small, (width, height))
        lesion = _upsample_mask(lesion_small, (width, height))
        lesion = lesion & leaf
    else:
        leaf = leaf_small
        lesion = lesion_small
    return leaf, lesion


def _upsample_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def lesion_preserving_edit(
    image: Image.Image | np.ndarray,
    *,
    seed: int = 0,
) -> EditResult:
    """Change background and healthy-leaf color without changing lesion pixels."""
    rgb = _as_uint8(image)
    leaf, lesion = segment_leaf_and_lesion(rgb)
    edited = rgb.copy()
    segmentation_ok = bool(leaf.any() and lesion.any())
    if segmentation_ok:
        rng = np.random.default_rng(seed)
        soil_color = np.array(
            [rng.integers(100, 150), rng.integers(70, 110), rng.integers(30, 60)],
            dtype=np.int16,
        )
        noise = rng.integers(-12, 13, size=rgb.shape, dtype=np.int16)
        soil = np.clip(soil_color.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
        edited[~leaf] = soil[~leaf]
        healthy = leaf & ~lesion
        if healthy.any():
            shifted = edited.astype(np.int16)
            shifted[healthy, 0] += 16
            shifted[healthy, 1] -= 6
            shifted[healthy, 2] -= 14
            edited = np.clip(shifted, 0, 255).astype(np.uint8)
        edited[lesion] = rgb[lesion]
    return _pack(rgb, edited, leaf, lesion, segmentation_ok)


def lesion_destroying_control(image: Image.Image | np.ndarray) -> EditResult:
    """Replace lesion pixels with the median healthy-leaf color."""
    rgb = _as_uint8(image)
    leaf, lesion = segment_leaf_and_lesion(rgb)
    edited = rgb.copy()
    healthy = leaf & ~lesion
    segmentation_ok = bool(lesion.any() and healthy.any())
    if segmentation_ok:
        fill = np.median(rgb[healthy], axis=0).astype(np.uint8)
        edited[lesion] = fill
    return _pack(rgb, edited, leaf, lesion, segmentation_ok)


def _pack(
    original: np.ndarray,
    edited: np.ndarray,
    leaf: np.ndarray,
    lesion: np.ndarray,
    segmentation_ok: bool,
) -> EditResult:
    lesion_fraction = float(lesion.mean())
    if lesion.any():
        match = float(np.all(edited[lesion] == original[lesion], axis=-1).mean())
    else:
        match = None
    outside = ~lesion
    if outside.any():
        outside_mae = float(np.abs(edited[outside].astype(np.int16) - original[outside].astype(np.int16)).mean())
    else:
        outside_mae = None
    return EditResult(
        edited=edited,
        leaf_mask=leaf,
        lesion_mask=lesion,
        lesion_pixel_match=match,
        outside_lesion_mae=outside_mae,
        lesion_fraction=lesion_fraction,
        segmentation_ok=segmentation_ok,
    )
