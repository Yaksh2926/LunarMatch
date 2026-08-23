"""Unit tests for geometry-preserving preprocessing and structural representations."""
from __future__ import annotations

import numpy as np
import pytest

from lunarmatch.models.config import PreprocessConfig
from lunarmatch.preprocessing import (
    create_preprocessing_quicklook,
    percentile_normalize,
    preprocess_raster,
)


@pytest.fixture
def synthetic_lunar_terrain() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic crater-like heightfield terrain image and valid mask."""
    y, x = np.ogrid[:64, :64]
    r = np.sqrt((x - 32) ** 2 + (y - 32) ** 2)
    terrain = np.exp(-((r - 15) ** 2) / 20.0) * 100.0 + np.random.default_rng(42).normal(0, 2, (64, 64))

    mask = np.ones((64, 64), dtype=bool)
    mask[r > 30] = False  # Invalid border mask
    terrain[~mask] = -9999.0

    return terrain.astype(np.float32), mask


@pytest.mark.parametrize("rep_name", ["gradient", "clahe", "edge", "phase", "raw_contrast"])
def test_all_representation_modes(
    synthetic_lunar_terrain: tuple[np.ndarray, np.ndarray], rep_name: str
) -> None:
    """Test preprocessing produces finite outputs and preserves image geometry/shape."""
    terrain, mask = synthetic_lunar_terrain
    cfg = PreprocessConfig(representation=rep_name, percentile_clip=(1.0, 99.0))  # type: ignore[arg-type]

    res = preprocess_raster(terrain, mask=mask, config=cfg)

    # Acceptance checks:
    # 1. Geometry/shape is unchanged
    assert res.processed_image.shape == terrain.shape
    assert res.valid_mask.shape == mask.shape

    # 2. Outputs are finite on valid pixels
    assert np.all(np.isfinite(res.processed_image[res.valid_mask]))
    # Invalid pixels are zeroed out
    assert np.all(res.processed_image[~res.valid_mask] == 0.0)

    # Values in [0.0, 1.0]
    assert np.min(res.processed_image) >= 0.0
    assert np.max(res.processed_image) <= 1.0

    # Provenance recorded
    assert res.provenance["representation"] == rep_name


def test_constant_image_edge_case() -> None:
    """Test constant image does not crash or produce NaNs."""
    constant_img = np.full((32, 32), 42.0, dtype=np.float32)
    mask = np.ones((32, 32), dtype=bool)

    norm, _prov = percentile_normalize(constant_img, mask=mask)
    assert np.all(norm == 0.0)
    assert np.all(np.isfinite(norm))

    res = preprocess_raster(constant_img, mask=mask)
    assert res.processed_image.shape == (32, 32)
    assert np.all(np.isfinite(res.processed_image))


def test_nans_infs_and_sparse_mask() -> None:
    """Test handling of NaN/Inf values and sparse valid pixel masks."""
    arr = np.random.default_rng(42).uniform(0, 100, (32, 32)).astype(np.float32)
    arr[0, 0] = np.nan
    arr[0, 1] = np.inf
    arr[0, 2] = -np.inf

    # Sparse mask (only 5 valid pixels)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:15, 10] = True

    res = preprocess_raster(arr, mask=mask)
    assert res.processed_image.shape == (32, 32)
    assert np.all(np.isfinite(res.processed_image[res.valid_mask]))
    assert np.all(res.processed_image[~res.valid_mask] == 0.0)


def test_denoise_and_clahe_parameters(
    synthetic_lunar_terrain: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test Gaussian denoising and CLAHE configuration options."""
    terrain, mask = synthetic_lunar_terrain
    cfg = PreprocessConfig(representation="clahe", denoise_sigma=1.2, clahe_clip_limit=3.0)

    res = preprocess_raster(terrain, mask=mask, config=cfg)
    assert res.provenance["denoise_sigma"] == 1.2
    assert res.provenance["clahe_clip_limit"] == 3.0
    assert np.all(np.isfinite(res.processed_image))


def test_preprocessing_quicklook_utility(
    synthetic_lunar_terrain: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test before/after side-by-side quick-look visualization utility."""
    terrain, mask = synthetic_lunar_terrain
    res = preprocess_raster(terrain, mask=mask)

    quicklook = create_preprocessing_quicklook(terrain, res.processed_image, mask=mask)

    # Side-by-side RGB image: Height x (Width * 2) x 3
    assert quicklook.shape == (64, 128, 3)
    assert quicklook.dtype == np.uint8


def test_unknown_representation_error() -> None:
    """Test error handling when requesting an unknown representation."""
    with pytest.raises(ValueError, match="Unknown representation"):
        # Create dummy config with unknown representation bypassing Pydantic validation via raw call
        cfg = PreprocessConfig()
        object.__setattr__(cfg, "representation", "unknown_magic_rep")
        preprocess_raster(np.zeros((10, 10)), config=cfg)
