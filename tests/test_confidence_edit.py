import numpy as np

from ppce.confidence_edit import GateConfig, confidence_gated_edit, lesion_pixels_from_three_level_mask


def _leaf_with_spot():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 180)
    yy, xx = np.ogrid[:96, :96]
    leaf = (yy - 48) ** 2 + (xx - 48) ** 2 <= 36**2
    spot = (yy - 48) ** 2 + (xx - 48) ** 2 <= 10**2
    image[leaf] = (30, 160, 40)
    image[spot] = (150, 60, 30)
    return image, spot


def _green_leaf():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 180)
    yy, xx = np.ogrid[:96, :96]
    leaf = (yy - 48) ** 2 + (xx - 48) ** 2 <= 36**2
    image[leaf] = (30, 170, 40)
    return image


def test_gated_edit_keeps_high_agreement_spot_and_changes_background():
    image, spot = _leaf_with_spot()
    decision = confidence_gated_edit(image, seed=4, mode="gated")
    assert decision.gate_accepted
    assert decision.emitted_edit
    assert decision.reasons == ("accepted",)
    assert decision.core_pixel_match == 1.0
    assert not np.array_equal(decision.image[0, 0], image[0, 0])
    # The hand-drawn brown disk is inside the frozen core.
    assert decision.core_fraction > 0
    core_covers_spot = None
    # Reconstruct core as pixels that were not allowed to change and lie on the spot.
    unchanged = np.all(decision.image == image, axis=-1)
    assert unchanged[spot].mean() > 0.8


def test_gate_rejects_a_leaf_with_no_high_agreement_evidence():
    image = _green_leaf()
    decision = confidence_gated_edit(image, seed=1, mode="gated")
    assert not decision.gate_accepted
    assert not decision.emitted_edit
    assert "no_high_confidence_evidence" in decision.reasons
    assert np.array_equal(decision.image, image)


def test_unconstrained_edit_changes_the_spot():
    image, spot = _leaf_with_spot()
    decision = confidence_gated_edit(image, seed=4, mode="unconstrained")
    assert decision.emitted_edit
    assert decision.reasons == ("unconstrained_no_preservation_check",)
    assert not np.array_equal(decision.image[spot], image[spot])


def test_constrained_emits_when_the_gate_would_reject():
    image, _ = _leaf_with_spot()
    strict = GateConfig(max_uncertain_fraction_of_leaf=-1.0)
    gated = confidence_gated_edit(image, seed=4, mode="gated", config=strict)
    constrained = confidence_gated_edit(image, seed=4, mode="constrained", config=strict)
    assert "segmentation_unstable" in gated.reasons
    assert np.array_equal(gated.image, image)
    assert constrained.emitted_edit
    assert constrained.core_pixel_match == 1.0
    assert not np.array_equal(constrained.image[0, 0], image[0, 0])


def test_three_level_mask_uses_the_non_leaf_foreground():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
    mask[1, 1] = 150
    lesion = lesion_pixels_from_three_level_mask(mask)
    assert lesion[1, 1]
    assert not lesion[0, 0]
    assert int(lesion.sum()) == 1
