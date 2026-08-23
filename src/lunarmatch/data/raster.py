"""Raster I/O operations, windowed slicing, and metadata parsing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from lunarmatch.models.domain import RasterMetadata

# Try importing rasterio
try:
    import rasterio  # type: ignore[import-untyped]
    import rasterio.windows  # type: ignore[import-untyped]
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


class RasterReadError(Exception):
    """Exception raised when a raster file cannot be read."""


@dataclass(frozen=True)
class RasterWindow:
    """Window specification for reading a subset of a raster."""
    col_off: int
    row_off: int
    width: int
    height: int


@dataclass
class RasterReadResult:
    """Result of a raster read operation containing data, mask, and metadata."""
    data: np.ndarray
    valid_mask: np.ndarray
    metadata: RasterMetadata


def is_rasterio_available() -> bool:
    """Check if the rasterio library is installed and available."""
    return _RASTERIO_AVAILABLE


def read_raster_metadata(path: str | Path) -> RasterMetadata:
    """Read metadata from a raster file without loading the image array.

    Args:
        path: Path to the raster file.

    Returns:
        RasterMetadata instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        RasterReadError: If metadata cannot be read.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Raster file not found: {p}")

    try:
        if _RASTERIO_AVAILABLE:
            with rasterio.open(p) as src:
                width = src.width
                height = src.height
                bands = src.count
                dtype = str(src.dtypes[0])
                crs = src.crs.to_string() if src.crs else None
                transform = tuple(src.transform)[:6] if src.transform else None
                nodata_value = src.nodata

                res = src.res
                pixel_scale_x = float(res[0]) if res else None
                pixel_scale_y = float(res[1]) if res else None

                tags = src.tags()
                sensor = tags.get("SENSOR") or tags.get("sensor")
                acquisition_time = tags.get("ACQUISITION_TIME") or tags.get("acquisition_time")

                sun_az = tags.get("SUN_AZIMUTH_DEG") or tags.get("sun_azimuth")
                sun_el = tags.get("SUN_ELEVATION_DEG") or tags.get("sun_elevation")
                sun_azimuth_deg = float(sun_az) if sun_az else None
                sun_elevation_deg = float(sun_el) if sun_el else None

                return RasterMetadata(
                    path=p,
                    width=width,
                    height=height,
                    bands=bands,
                    dtype=dtype,
                    pixel_scale_x=pixel_scale_x,
                    pixel_scale_y=pixel_scale_y,
                    crs=crs,
                    transform=transform,
                    nodata_value=nodata_value,
                    sensor=sensor,
                    acquisition_time=acquisition_time,
                    sun_azimuth_deg=sun_azimuth_deg,
                    sun_elevation_deg=sun_elevation_deg,
                )
        else:
            if p.suffix.lower() in (".tif", ".tiff"):
                with tifffile.TiffFile(p) as tif:
                    page = tif.pages[0]
                    height, width = page.shape[:2]
                    bands = len(page.axes) if len(page.shape) > 2 else 1
                    dtype = str(page.dtype)
                    return RasterMetadata(
                        path=p,
                        width=width,
                        height=height,
                        bands=bands,
                        dtype=dtype,
                    )
            else:
                with Image.open(p) as img:
                    width, height = img.size
                    bands = len(img.getbands())
                    dtype = "uint8"
                    return RasterMetadata(
                        path=p,
                        width=width,
                        height=height,
                        bands=bands,
                        dtype=dtype,
                    )
    except Exception as exc:
        raise RasterReadError(f"Error reading metadata from {p}: {exc}") from exc


def read_raster(
    path: str | Path,
    band: int = 1,
    window: RasterWindow | None = None,
    nodata: float | None = None,
) -> RasterReadResult:
    """Read a raster band data and construct a valid data mask.

    Args:
        path: Path to the raster file.
        band: 1-indexed band number to read.
        window: Optional window specifying the bounding box to read.
        nodata: Optional nodata value to override metadata.

    Returns:
        RasterReadResult instance containing array data, boolean mask, and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If band or window bounds are invalid.
        RasterReadError: If data reading fails.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Raster file not found: {p}")

    meta = read_raster_metadata(p)
    if band < 1 or band > meta.bands:
        raise ValueError(f"Band index {band} exceeds available bands (1 to {meta.bands})")

    if window is not None and (
        window.col_off + window.width > meta.width
        or window.row_off + window.height > meta.height
    ):
        raise ValueError("Window exceeds image dimensions")

    try:
        effective_nodata = nodata if nodata is not None else meta.nodata_value

        if _RASTERIO_AVAILABLE:
            with rasterio.open(p) as src:
                if window is not None:
                    rio_win = rasterio.windows.Window(
                        col_off=window.col_off,
                        row_off=window.row_off,
                        width=window.width,
                        height=window.height,
                    )
                    data = src.read(band, window=rio_win)
                else:
                    data = src.read(band)
        else:
            if p.suffix.lower() in (".tif", ".tiff"):
                with tifffile.TiffFile(p) as tif:
                    page = tif.pages[0]
                    arr = page.asarray()
                    if len(arr.shape) > 2:
                        arr = arr[..., band - 1]
                    if window is not None:
                        arr = arr[
                            window.row_off : window.row_off + window.height,
                            window.col_off : window.col_off + window.width,
                        ]
                    data = arr
            else:
                with Image.open(p) as img:
                    arr = np.array(img)
                    if len(arr.shape) > 2:
                        arr = arr[..., band - 1]
                    if window is not None:
                        arr = arr[
                            window.row_off : window.row_off + window.height,
                            window.col_off : window.col_off + window.width,
                        ]
                    data = arr

        valid_mask = np.ones(data.shape, dtype=bool)

        if np.issubdtype(data.dtype, np.floating):
            valid_mask &= ~np.isnan(data)

        if effective_nodata is not None:
            valid_mask &= (data == effective_nodata) == False

        return RasterReadResult(data=data, valid_mask=valid_mask, metadata=meta)
    except Exception as exc:
        raise RasterReadError(f"Failed to read raster data: {exc}") from exc
