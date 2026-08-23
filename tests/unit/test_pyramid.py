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
