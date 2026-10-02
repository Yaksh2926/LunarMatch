"""Unit tests for composable 3x3 CoordinateMapping."""
from __future__ import annotations

import numpy as np
import pytest

from lunarmatch.geometry import CoordinateMapping


def test_identity_mapping() -> None:
    """Test identity mapping preserves points exactly."""
    mapping = CoordinateMapping.identity()
    pts = np.array([[0.0, 0.0], [10.5, 20.25], [100.0, 200.0]], dtype=np.float64)
    res = mapping.apply(pts)
    np.testing.assert_allclose(res, pts, atol=1e-12)

    inv_res = mapping.apply_inverse(pts)
    np.testing.assert_allclose(inv_res, pts, atol=1e-12)


def test_crop_mapping() -> None:
    """Test crop offset mapping and inverse round-trip."""
    offset_x, offset_y = 12.5, 34.75
    crop_map = CoordinateMapping.from_crop(offset_x, offset_y)

    pts_crop = np.array([[0.5, 0.5], [10.0, 15.0]], dtype=np.float64)
    pts_orig = crop_map.apply(pts_crop)
    expected = np.array([[13.0, 35.25], [22.5, 49.75]], dtype=np.float64)
    np.testing.assert_allclose(pts_orig, expected, atol=1e-12)

    # Inverse round-trip
    restored = crop_map.apply_inverse(pts_orig)
    np.testing.assert_allclose(restored, pts_crop, atol=1e-12)


def test_scale_mapping() -> None:
    """Test non-integer scaling mapping and inverse round-trip."""
    scale_x, scale_y = 2.5, 1.75
    scale_map = CoordinateMapping.from_scale(scale_x, scale_y)

    pts_scaled = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float64)
    pts_orig = scale_map.apply(pts_scaled)
    expected = np.array([[2.5, 3.5], [10.0, 14.0]], dtype=np.float64)
    np.testing.assert_allclose(pts_orig, expected, atol=1e-12)

    # Inverse round-trip
    restored = scale_map.apply_inverse(pts_orig)
    np.testing.assert_allclose(restored, pts_scaled, atol=1e-12)


def test_composition_crop_and_non_integer_scale() -> None:
    """Test multi-stage composition of non-integer scale and crop operations."""
    scale_map = CoordinateMapping.from_scale(2.5, 3.2)
    crop_map = CoordinateMapping.from_crop(15.2, 8.7)

    # Apply crop then scale: M_composed = M_scale @ M_crop
    composed = scale_map.compose(crop_map)

    pts_working = np.array([[0.5, 0.5], [100.25, 200.75]], dtype=np.float64)
    pts_mapped = composed.apply(pts_working)

    # Manual verification: first crop, then scale
    pts_cropped = crop_map.apply(pts_working)
    pts_scaled = scale_map.apply(pts_cropped)
    np.testing.assert_allclose(pts_mapped, pts_scaled, atol=1e-12)

    # Inverse round-trip
    restored = composed.apply_inverse(pts_mapped)
    np.testing.assert_allclose(restored, pts_working, atol=1e-12)


def test_numerical_roundtrip_tolerance() -> None:
    """Test round-trip accuracy across random points and 3-tier pyramid/crop composition."""
    rng = np.random.default_rng(42)
    random_pts = rng.uniform(0.0, 1000.0, size=(100, 2))

    m_pyramid = CoordinateMapping.from_scale(1.75, 1.75)
    m_crop = CoordinateMapping.from_crop(100.5, 200.25)
    m_pyramid2 = CoordinateMapping.from_scale(2.0, 2.0)

    composed = m_pyramid.compose(m_crop).compose(m_pyramid2)

    forward = composed.apply(random_pts)
    inverse = composed.apply_inverse(forward)

    max_err = np.max(np.abs(random_pts - inverse))
    assert max_err < 1e-11


def test_invalid_matrix_and_inputs() -> None:
    """Test error handling for invalid matrices, singular matrices, and malformed point arrays."""
    # Singular matrix
    with pytest.raises(ValueError, match="singular"):
        CoordinateMapping(np.zeros((3, 3)))

    # Non-3x3 matrix
    with pytest.raises(ValueError, match="shape"):
        CoordinateMapping(np.eye(4))

    # Negative scale
    with pytest.raises(ValueError, match="positive"):
        CoordinateMapping.from_scale(-1.0, 2.0)

    mapping = CoordinateMapping.identity()

    # 1D points array
    with pytest.raises(ValueError, match="shape"):
        mapping.apply(np.array([1.0, 2.0]))

    # Non-finite points (NaN / Inf)
    with pytest.raises(ValueError, match="non-finite"):
        mapping.apply(np.array([[1.0, np.nan]]))


def test_serialization() -> None:
    """Test to_dict and from_dict serialization."""
    crop_map = CoordinateMapping.from_crop(10.0, 20.0)
    data = crop_map.to_dict()
    assert "matrix" in data

    reconstructed = CoordinateMapping.from_dict(data)
    assert reconstructed == crop_map


def test_pyramid_center_convention() -> None:
    """Verify that the coordinate mapping matches cv2.resize pixel-center convention."""
    scale_x, scale_y = 5.0 / 3.0, 5.0 / 3.0
    mapping = CoordinateMapping(np.array([
        [scale_x, 0.0, 0.5 * (scale_x - 1.0)],
        [0.0, scale_y, 0.5 * (scale_y - 1.0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64))

    # Pixel center at (1.0, 1.0) maps to (2.0, 2.0)
    pt = np.array([[1.0, 1.0]], dtype=np.float64)
    mapped = mapping.apply(pt)
    np.testing.assert_allclose(mapped, np.array([[2.0, 2.0]], dtype=np.float64), atol=1e-7)

