"""Unit tests for source raster warping into reference image grid."""
import numpy as np
import pytest

from lunarmatch.export.warper import warp_source_to_reference
from lunarmatch.models.domain import TransformEstimate


def test_warp_source_shape_matches_reference():
    """Test that warped output dimensions match target reference grid shape."""
    src = np.ones((100, 150), dtype=np.float32)
    trans = TransformEstimate.from_matrix("affine", np.eye(3), inlier_count=10, inlier_ratio=1.0, rmse_px=0.1)

    warped = warp_source_to_reference(src, trans, reference_shape=(200, 250))
    assert warped.shape == (200, 250)


def test_warp_known_translation():
    """Test feature alignment under known translation transform."""
    # Source image with a bright feature at (50, 50)
    src = np.zeros((100, 100), dtype=np.float32)
    src[45:55, 45:55] = 1.0

    # Ground truth matrix mapping source -> reference: shift by (+20, +10)
    # p_ref = p_src + (20, 10)
    M = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    trans = TransformEstimate.from_matrix("affine", M, inlier_count=10, inlier_ratio=1.0, rmse_px=0.0)

    warped = warp_source_to_reference(src, trans, reference_shape=(100, 100))

    # The feature originally at (50, 50) should now appear at (70, 60) in reference grid
    assert warped[60, 70] > 0.8
    assert warped[50, 50] < 0.1


def test_warp_interpolation_modes():
    """Test warping with different interpolation algorithms."""
    src = np.random.default_rng(42).uniform(0, 255, (80, 80)).astype(np.float32)
    M = np.eye(3, dtype=np.float64)

    for mode in ("nearest", "bilinear", "cubic", "lanczos"):
        w = warp_source_to_reference(src, M, (80, 80), interpolation=mode)
        assert w.shape == (80, 80)


def test_warp_nodata_border_fill():
    """Test border filling with nodata value."""
    src = np.ones((50, 50), dtype=np.float32)
    # Shift source right by 30 px -> left border of reference will be unmapped
    M = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    warped = warp_source_to_reference(src, M, (50, 50), nodata_value=-999.0)
    assert warped[25, 5] == -999.0


def test_singular_transform_matrix_raises():
    """Test that singular transform matrix raises ValueError."""
    src = np.ones((50, 50), dtype=np.float32)
    singular_M = np.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="singular"):
        warp_source_to_reference(src, singular_M, (50, 50))
