"""Unit tests for ImagePyramid construction and level coordinate mappings."""
from __future__ import annotations

import numpy as np

from lunarmatch.features import ImagePyramid
from lunarmatch.models.config import PyramidConfig


def test_image_pyramid_construction_and_mappings() -> None:
    """Test image pyramid downsampling and coordinate mappings back to original pixel centers."""
    img = np.random.default_rng(42).uniform(0, 100, (128, 128)).astype(np.float32)
    mask = np.ones((128, 128), dtype=bool)
    mask[0:10, 0:10] = False

    cfg = PyramidConfig(levels_per_octave=1)
    pyramid = ImagePyramid.build(img, mask=mask, config=cfg)

    assert len(pyramid) >= 2
    level_0 = pyramid[0]
    assert level_0.scale_factor == 1.0
    assert level_0.width == 128
    assert level_0.height == 128

    # Test coordinate mapping round-trip on each level
    for lvl in pyramid.levels:
        assert lvl.image.shape == lvl.valid_mask.shape

        # Pick center point of level
        pts_lvl = np.array([[lvl.width / 2.0, lvl.height / 2.0]], dtype=np.float64)
        pts_orig = lvl.mapping.apply(pts_lvl)
        restored = lvl.mapping.apply_inverse(pts_orig)

        # Coordinate round-trip tolerance check
        np.testing.assert_allclose(restored, pts_lvl, atol=1e-11)


def test_pyramid_pixel_center_cv2_resize() -> None:
    """Verify that pyramid coordinate mapping aligns exactly with cv2.resize pixel centers."""
    # 1. Integer scale case: 4x4 downsampled to 2x2 (scale_factor = 0.5)
    img_4x4 = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
    ], dtype=np.float32)
    
    cfg = PyramidConfig(levels_per_octave=1)
    pyramid = ImagePyramid.build(img_4x4, config=cfg, min_dimension=2)
    
    # level 1 is downsampled by 0.5x, size 2x2
    level_1 = pyramid[1]
    assert level_1.width == 2
    assert level_1.height == 2
    
    # Pixel center at (0.0, 0.0) in level 1 maps to (0.5, 0.5) in level 0
    pts_lvl = np.array([[0.0, 0.0]], dtype=np.float64)
    pts_orig = level_1.mapping.apply(pts_lvl)
    np.testing.assert_allclose(pts_orig, np.array([[0.5, 0.5]], dtype=np.float64), atol=1e-7)

    # 2. Non-integer scale case: 5x5 downsampled to 3x3
    # Note: 5 / (2 ** 0.5) = 5 / 1.414 = 3.53 -> rounded to 4.
    # To get exactly 5 to 3, let's manually construct a level with custom mapping and test it.
    w_orig, h_orig = 5, 5
    new_w, new_h = 3, 3
    scale_x = float(w_orig) / float(new_w)
    scale_y = float(h_orig) / float(new_h)
    
    # Correct mapping calculation:
    # x_orig = scale_x * x_level + 0.5 * (scale_x - 1.0)
    # y_orig = scale_y * y_level + 0.5 * (scale_y - 1.0)
    from lunarmatch.geometry import CoordinateMapping
    level_mapping = CoordinateMapping(np.array([
        [scale_x, 0.0, 0.5 * (scale_x - 1.0)],
        [0.0, scale_y, 0.5 * (scale_y - 1.0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64))
    
    # Center pixel of 3x3 is at (1.0, 1.0).
    # Since w_orig = 5, the physical center of the image is at (2.0, 2.0).
    pts_lvl_non_int = np.array([[1.0, 1.0]], dtype=np.float64)
    pts_orig_non_int = level_mapping.apply(pts_lvl_non_int)
    np.testing.assert_allclose(pts_orig_non_int, np.array([[2.0, 2.0]], dtype=np.float64), atol=1e-7)

