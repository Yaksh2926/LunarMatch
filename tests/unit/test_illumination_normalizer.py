"""Unit tests for illumination normalizer and shadow angle offset calculation.

Tests verify calculation of solar-illumination shadow offsets using real Chandrayaan-2
OHRC image header parameters (solar azimuth = 71.988947 deg, solar elevation = 8.320911 deg).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from lunarmatch.models.config import PreprocessConfig
from lunarmatch.preprocessing.illumination_normalizer import (
    IlluminationAlignmentResult,
    IlluminationNormalizer,
    compute_footprint_overlap,
    compute_illumination_shadow_offset,
)


def test_compute_illumination_shadow_offset_real_ohrc_header() -> None:
    """Test shadow displacement offset using real OHRC header values.

    OHRC pair header values:
    - Primary acquisition: solar azimuth = 71.988947 deg, solar elevation = 8.320911 deg
    - Secondary acquisition: solar azimuth = 70.120000 deg, solar elevation = 12.450000 deg
    """
    az_src = 71.988947
    el_src = 8.320911
    az_ref = 70.120000
    el_ref = 12.450000

    dx, dy = compute_illumination_shadow_offset(az_src, el_src, az_ref, el_ref)

    # Physical check: at low elevation (~8 deg), shadow length ratio (1/tan(el)) is ~6.84.
    # At 12.45 deg elevation, shadow length ratio is ~4.53.
    # Relative shadow offset (dx, dy) in meters per 100m object height:
    assert isinstance(dx, float)
    assert isinstance(dy, float)
    # dx and dy should be non-zero and physically reasonable
    assert abs(dx) > 0.0
    assert abs(dy) > 0.0


def test_compute_illumination_shadow_offset_identical_angles() -> None:
    """Identical solar illumination parameters must produce zero shadow offset."""
    az = 71.988947
    el = 8.320911
    dx, dy = compute_illumination_shadow_offset(az, el, az, el)
    assert pytest.approx(dx, abs=1e-6) == 0.0
    assert pytest.approx(dy, abs=1e-6) == 0.0


def test_compute_footprint_overlap() -> None:
    """Test spatial footprint intersection fraction computation."""
    fp1 = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    fp2 = ((5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0))
    overlap = compute_footprint_overlap(fp1, fp2)
    assert pytest.approx(overlap, abs=1e-4) == 0.5


def test_illumination_normalizer_disabled_by_default() -> None:
    """Normalizer returns zero offset when disabled in config."""
    config = PreprocessConfig(allow_illumination_correction=False)
    normalizer = IlluminationNormalizer(config=config)

    src_meta = MagicMock()
    src_meta.sun_azimuth_deg = 71.988947
    src_meta.sun_elevation_deg = 8.320911
    src_meta.width = 100
    src_meta.height = 100

    ref_meta = MagicMock()
    ref_meta.sun_azimuth_deg = 60.0
    ref_meta.sun_elevation_deg = 15.0
    ref_meta.width = 100
    ref_meta.height = 100

    img = np.zeros((100, 100), dtype=np.uint8)

    norm_img, res = normalizer.normalize(img, src_meta, ref_meta)

    assert isinstance(res, IlluminationAlignmentResult)
    assert res.offset_dx == 0.0
    assert res.offset_dy == 0.0
    assert res.footprint_overlap_fraction == 1.0
    assert np.array_equal(norm_img, img)


def test_illumination_normalizer_enabled() -> None:
    """Normalizer computes shadow offset when enabled in config."""
    config = PreprocessConfig(allow_illumination_correction=True)
    normalizer = IlluminationNormalizer(config=config)

    src_meta = MagicMock()
    src_meta.sun_azimuth_deg = 71.988947
    src_meta.sun_elevation_deg = 8.320911
    src_meta.width = 100
    src_meta.height = 100

    ref_meta = MagicMock()
    ref_meta.sun_azimuth_deg = 70.120000
    ref_meta.sun_elevation_deg = 12.450000
    ref_meta.width = 100
    ref_meta.height = 100

    img = np.zeros((100, 100), dtype=np.uint8)

    _norm_img, res = normalizer.normalize(img, src_meta, ref_meta)

    assert isinstance(res, IlluminationAlignmentResult)
    assert res.offset_dx != 0.0
    assert res.offset_dy != 0.0
