import numpy as np

from ppce.editing import lesion_destroying_control, lesion_preserving_edit


def _synthetic_leaf():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 180)
    yy, xx = np.ogrid[:80, :80]
    leaf = (yy - 40) ** 2 + (xx - 40) ** 2 <= 28**2
    lesion = (yy - 40) ** 2 + (xx - 40) ** 2 <= 8**2
    image[leaf] = (30, 160, 40)
    image[lesion] = (150, 60, 30)
    return image, lesion


def test_preserving_edit_keeps_lesion_pixels_and_changes_background():
    image, lesion = _synthetic_leaf()
    result = lesion_preserving_edit(image, seed=3)
    assert result.segmentation_ok
    assert result.lesion_mask[lesion].mean() > 0.8
    assert np.array_equal(result.edited[result.lesion_mask], image[result.lesion_mask])
    assert result.lesion_pixel_match == 1.0
    background = ~result.leaf_mask
    assert background.any()
    assert not np.array_equal(result.edited[background], image[background])
    assert result.outside_lesion_mae > 0


def test_destroying_control_changes_only_the_lesion():
    image, _ = _synthetic_leaf()
    result = lesion_destroying_control(image)
    assert result.segmentation_ok
    assert result.lesion_pixel_match < 1.0
    outside = ~result.lesion_mask
    assert np.array_equal(result.edited[outside], image[outside])
    assert result.outside_lesion_mae == 0
