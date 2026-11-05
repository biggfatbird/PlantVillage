"""Confidence-gated environmental edit.

The image is split into environment, leaf body, an uncertain disease band, and a
high-agreement disease core. Agreement is the fraction of fixed color rules that
call a leaf pixel disease. It is not a trained probability and it is not a
biological measurement.

An environmental edit replaces non-leaf pixels with a soil-colored field and
shifts the color of unprotected leaf pixels. The constrained edit copies the
core and the uncertain band from the source. The gated edit emits that result
only when every preservation check below passes. Otherwise it emits the source
and records the failed checks.

Checks:

- no_leaf: consensus leaf covers less than the configured fraction.
- no_high_confidence_evidence: the core has fewer pixels than the configured minimum.
- segmentation_unstable: the uncertain band is too large a share of the leaf.
- no_editable_region: there is nowhere an environmental change can be made.
- core_modified: a core pixel differs from the source.
- uncertain_modified: an uncertain pixel differs from the source.
- domain_change_too_small: editable pixels barely changed.
- evidence_not_recovered: re-running the rules on the edited image recalls too
  little of the original core.

Self-consistency of the core is not biological preservation. A pixel the rules
miss can still be edited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class GateConfig:
    core_agreement: float = 0.6
    uncertain_agreement: float = 0.4
    min_leaf_fraction: float = 0.05
    min_core_pixels: int = 20
    max_uncertain_fraction_of_leaf: float = 0.55
    min_domain_mae: float = 5.0
    min_environment_fraction: float = 0.02
    min_resegmentation_recall: float = 0.90
    max_side: int = 384


DEFAULT_CONFIG = GateConfig()


@dataclass(frozen=True)
class Separation:
    leaf: np.ndarray
    agreement: np.ndarray
    core: np.ndarray
    uncertain: np.ndarray
    leaf_body: np.ndarray
    environment: np.ndarray


@dataclass(frozen=True)
class EditDecision:
    image: np.ndarray
    emitted_edit: bool
    gate_accepted: bool
    reasons: tuple[str, ...]
    mode: str
    leaf_fraction: float
    core_fraction: float
    uncertain_fraction: float
    leaf_body_fraction: float
    environment_fraction: float
    mean_core_agreement: float | None
    core_pixel_match: float | None
    uncertain_pixel_match: float | None
    environment_mae: float | None
    leaf_body_mae: float | None
    resegmentation_core_recall: float | None


def confidence_gated_edit(
    image: Image.Image | np.ndarray,
    *,
    seed: int = 0,
    mode: str = "gated",
    config: GateConfig = DEFAULT_CONFIG,
) -> EditDecision:
    if mode not in {"unconstrained", "constrained", "gated"}:
        raise ValueError(f"unknown edit mode {mode!r}")
    source = _as_uint8(image)
    separation = separate_regions(source, config)
    if float(separation.leaf.mean()) < config.min_leaf_fraction:
        return _decision(
            source,
            source,
            separation,
            emitted_edit=False,
            gate_accepted=False,
            reasons=("no_leaf",),
            mode=mode,
            recall=None,
        )
    protect = mode != "unconstrained"
    edited = _apply_environment_edit(source, separation, seed=seed, protect=protect)
    if mode == "unconstrained":
        return _decision(
            source,
            edited,
            separation,
            emitted_edit=True,
            gate_accepted=False,
            reasons=("unconstrained_no_preservation_check",),
            mode=mode,
            recall=None,
        )
    recall = _core_recall(source, edited, separation, config)
    reasons = _failed_checks(source, edited, separation, recall, config)
    accepted = not reasons
    if mode == "constrained":
        return _decision(
            source,
            edited,
            separation,
            emitted_edit=True,
            gate_accepted=accepted,
            reasons=reasons or ("constrained_emitted",),
            mode=mode,
            recall=recall,
        )
    if accepted:
        return _decision(
            source,
            edited,
            separation,
            emitted_edit=True,
            gate_accepted=True,
            reasons=("accepted",),
            mode=mode,
            recall=recall,
        )
    return _decision(
        source,
        source,
        separation,
        emitted_edit=False,
        gate_accepted=False,
        reasons=reasons,
        mode=mode,
        recall=recall,
    )


def separate_regions(rgb: np.ndarray, config: GateConfig = DEFAULT_CONFIG) -> Separation:
    small, full_size = _downscale(rgb, config.max_side)
    leaf_small = _leaf_consensus(small)
    agreement_small = _disease_agreement(small, leaf_small)
    leaf = _upsample_mask(leaf_small, full_size)
    agreement = _upsample_float(agreement_small, full_size)
    agreement = agreement * leaf
    core = leaf & (agreement >= config.core_agreement)
    uncertain = leaf & (agreement >= config.uncertain_agreement) & (agreement < config.core_agreement)
    leaf_body = leaf & ~core & ~uncertain
    environment = ~leaf
    return Separation(leaf, agreement, core, uncertain, leaf_body, environment)


def _disease_agreement(rgb: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    excess = 2 * green - red - blue
    rules = (
        excess <= 0,
        excess <= 12,
        (red - green > 18) & (red > 50),
        (red > 110) & (green > 90) & (blue + 25 < np.minimum(red, green)),
        ((red + green + blue) < 150) & (excess < 25),
    )
    votes = np.zeros(leaf.shape, dtype=np.float32)
    for rule in rules:
        votes += rule
    agreement = np.zeros(leaf.shape, dtype=np.float32)
    agreement[leaf] = votes[leaf] / float(len(rules))
    return agreement


def _leaf_consensus(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    excess = 2 * green - red - blue
    votes = np.zeros(excess.shape, dtype=np.uint8)
    for threshold in (12, 24, 40):
        tissue = excess > threshold
        if int(tissue.sum()) < 16:
            continue
        grown = ndimage.binary_dilation(tissue, iterations=6)
        filled = ndimage.binary_fill_holes(grown)
        votes += _largest_component(filled)
    return votes >= 2


def _apply_environment_edit(
    rgb: np.ndarray,
    separation: Separation,
    *,
    seed: int,
    protect: bool,
) -> np.ndarray:
    edited = rgb.copy()
    rng = np.random.default_rng(seed)
    soil_color = np.array(
        [int(rng.integers(100, 150)), int(rng.integers(70, 110)), int(rng.integers(30, 60))],
        dtype=np.int16,
    )
    noise = rng.integers(-12, 13, size=rgb.shape, dtype=np.int16)
    soil = np.clip(soil_color.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
    edited[separation.environment] = soil[separation.environment]
    shifted = edited.astype(np.int16)
    shifted[separation.leaf_body, 0] += 16
    shifted[separation.leaf_body, 1] -= 6
    shifted[separation.leaf_body, 2] -= 14
    if not protect:
        shifted[separation.core, 0] += 16
        shifted[separation.core, 1] -= 6
        shifted[separation.core, 2] -= 14
        shifted[separation.uncertain, 0] += 16
        shifted[separation.uncertain, 1] -= 6
        shifted[separation.uncertain, 2] -= 14
    edited = np.clip(shifted, 0, 255).astype(np.uint8)
    if protect:
        edited[separation.core] = rgb[separation.core]
        edited[separation.uncertain] = rgb[separation.uncertain]
    return edited


def _failed_checks(
    source: np.ndarray,
    edited: np.ndarray,
    separation: Separation,
    recall: float | None,
    config: GateConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    leaf_fraction = float(separation.leaf.mean())
    core_count = int(separation.core.sum())
    if leaf_fraction < config.min_leaf_fraction:
        reasons.append("no_leaf")
        return tuple(reasons)
    if core_count < config.min_core_pixels:
        reasons.append("no_high_confidence_evidence")
    uncertain_share = float(separation.uncertain.sum()) / float(separation.leaf.sum())
    if uncertain_share > config.max_uncertain_fraction_of_leaf:
        reasons.append("segmentation_unstable")
    environment_fraction = float(separation.environment.mean())
    leaf_body_fraction = float(separation.leaf_body.mean())
    if environment_fraction < config.min_environment_fraction and leaf_body_fraction < config.min_environment_fraction:
        reasons.append("no_editable_region")
    if separation.core.any() and _match(source, edited, separation.core) < 1.0:
        reasons.append("core_modified")
    if separation.uncertain.any() and _match(source, edited, separation.uncertain) < 1.0:
        reasons.append("uncertain_modified")
    environment_mae = _mae(source, edited, separation.environment)
    leaf_body_mae = _mae(source, edited, separation.leaf_body)
    if environment_fraction >= config.min_environment_fraction:
        changed = environment_mae is not None and environment_mae >= config.min_domain_mae
    else:
        changed = leaf_body_mae is not None and leaf_body_mae >= config.min_domain_mae
    if not changed:
        reasons.append("domain_change_too_small")
    if recall is None or recall < config.min_resegmentation_recall:
        reasons.append("evidence_not_recovered")
    return tuple(reasons)


def _core_recall(
    source: np.ndarray,
    edited: np.ndarray,
    separation: Separation,
    config: GateConfig,
) -> float | None:
    if not separation.core.any():
        return None
    revised = separate_regions(edited, config)
    kept = separation.core & (revised.agreement >= config.core_agreement)
    return float(kept.sum()) / float(separation.core.sum())


def _decision(
    source: np.ndarray,
    emitted: np.ndarray,
    separation: Separation,
    *,
    emitted_edit: bool,
    gate_accepted: bool,
    reasons: tuple[str, ...],
    mode: str,
    recall: float | None,
) -> EditDecision:
    core_values = separation.agreement[separation.core]
    return EditDecision(
        image=emitted,
        emitted_edit=emitted_edit,
        gate_accepted=gate_accepted,
        reasons=reasons,
        mode=mode,
        leaf_fraction=float(separation.leaf.mean()),
        core_fraction=float(separation.core.mean()),
        uncertain_fraction=float(separation.uncertain.mean()),
        leaf_body_fraction=float(separation.leaf_body.mean()),
        environment_fraction=float(separation.environment.mean()),
        mean_core_agreement=None if core_values.size == 0 else float(core_values.mean()),
        core_pixel_match=_match(source, emitted, separation.core),
        uncertain_pixel_match=_match(source, emitted, separation.uncertain),
        environment_mae=_mae(source, emitted, separation.environment),
        leaf_body_mae=_mae(source, emitted, separation.leaf_body),
        resegmentation_core_recall=recall,
    )


def _match(source: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(np.all(source[mask] == edited[mask], axis=-1).mean())


def _mae(source: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    delta = np.abs(source[mask].astype(np.int16) - edited[mask].astype(np.int16))
    return float(delta.mean())


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


def _downscale(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
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
    return small, (width, height)


def _upsample_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def _upsample_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    return np.asarray(image).astype(np.float32) / 255.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def lesion_pixels_from_three_level_mask(mask: np.ndarray) -> np.ndarray:
    """Lesion channel used by the Zenodo apple subset.

    Inspected files use 0 for background, 255 for the leaf, and one other
    value for the marked spots. The spot value is 150, 76, or 29 depending on
    the disease name. Pixels with that other value are treated as the lesion.
    """
    gray = np.asarray(mask)
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    return (gray != 0) & (gray != 255)
