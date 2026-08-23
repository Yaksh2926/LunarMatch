"""Unit tests for raster reading, windowed slicing, and nodata mask generation using synthetic fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from lunarmatch.data import (
    RasterWindow,
    is_rasterio_available,
    read_raster,
    read_raster_metadata,
)


@pytest.fixture
def synthetic_uint8_tiff(tmp_path: Path) -> Path:
    """Generate a tiny 64x64 uint8 synthetic TIFF raster."""
    path = tmp_path / "synthetic_uint8.tif"
    arr = np.arange(64 * 64, dtype=np.uint8).reshape((64, 64))
    tifffile.imwrite(str(path), arr)
    return path


@pytest.fixture
def synthetic_float32_nan_tiff(tmp_path: Path) -> Path:
    """Generate a tiny 32x32 float32 TIFF raster containing NaNs and explicit nodata (-9999.0)."""
    path = tmp_path / "synthetic_float32.tif"
    arr = np.linspace(0.0, 100.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    arr[5, 5] = np.nan
    arr[10, 10] = -9999.0
    tifffile.imwrite(str(path), arr)
    return path


@pytest.fixture
def synthetic_png_debug(tmp_path: Path) -> Path:
    """Generate a tiny 16x16 uint8 debug PNG raster."""
    path = tmp_path / "debug_raster.png"
    arr = np.full((16, 16), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def test_full_vs_windowed_read(synthetic_uint8_tiff: Path) -> None:
    """Test windowed read equals exact slice of full-image array."""
    full_res = read_raster(synthetic_uint8_tiff)
    assert full_res.data.shape == (64, 64)
    assert full_res.metadata.width == 64
    assert full_res.metadata.height == 64

    # Perform windowed read
    win = RasterWindow(col_off=10, row_off=15, width=20, height=25)
    win_res = read_raster(synthetic_uint8_tiff, window=win)

    assert win_res.data.shape == (25, 20)
    assert win_res.valid_mask.shape == (25, 20)

    # Acceptance check: window read equals slice of full-image array
    expected_slice = full_res.data[15 : 15 + 25, 10 : 10 + 20]
    np.testing.assert_array_equal(win_res.data, expected_slice)


def test_dtypes_and_nodata_masks(synthetic_float32_nan_tiff: Path) -> None:
    """Test float32 raster reading, NaN detection, and nodata value mask creation."""
    res = read_raster(synthetic_float32_nan_tiff, nodata=-9999.0)
    assert res.data.dtype == np.float32

    # Verify mask excludes NaN and nodata pixels
    assert not res.valid_mask[5, 5]  # NaN
    assert not res.valid_mask[10, 10]  # Nodata -9999.0
    assert res.valid_mask[0, 0]  # Valid pixel


def test_png_debug_raster_reading(synthetic_png_debug: Path) -> None:
    """Test debug PNG reading and metadata extraction."""
    meta = read_raster_metadata(synthetic_png_debug)
    assert meta.width == 16
    assert meta.height == 16

    res = read_raster(synthetic_png_debug)
    assert res.data.shape == (16, 16)
    assert np.all(res.valid_mask)


def test_rasterio_availability_flag() -> None:
    """Test is_rasterio_available function returns bool."""
    avail = is_rasterio_available()
    assert isinstance(avail, bool)


def test_invalid_band_and_window_bounds(synthetic_uint8_tiff: Path) -> None:
    """Test error handling for invalid band index and out-of-bounds window."""
    with pytest.raises(ValueError, match="exceeds available bands"):
        read_raster(synthetic_uint8_tiff, band=99)

    with pytest.raises(ValueError, match="exceeds image dimensions"):
        read_raster(synthetic_uint8_tiff, window=RasterWindow(col_off=50, row_off=50, width=30, height=30))


def test_missing_raster_file() -> None:
    """Test read_raster error handling for missing file."""
    with pytest.raises(FileNotFoundError, match="Raster file not found"):
        read_raster("non_existent_lunar_image.tif")
