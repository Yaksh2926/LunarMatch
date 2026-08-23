"""Overlapping tile planning and memory management with coordinate mapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarmatch.data.raster import RasterWindow
from lunarmatch.geometry import CoordinateMapping


@dataclass
class TileInfo:
    """Tile metadata containing window bounds, valid pixel count, and tile-to-original coordinate mapping."""

    tile_id: int
    window: RasterWindow
    level_idx: int
    level_mapping: CoordinateMapping
    tile_mapping: CoordinateMapping  # Maps tile local (x, y) to original (x, y)
    valid_pixel_count: int


def plan_tiles(
    image_shape: tuple[int, int],
    mask: np.ndarray | None = None,
    tile_size: int = 2048,
    tile_overlap: int = 128,
    level_idx: int = 0,
    level_mapping: CoordinateMapping | None = None,
    max_megapixels_in_memory: int | None = 256,
) -> list[TileInfo]:
    """Plan overlapping tiles covering valid regions of an image.

    Args:
        image_shape: Tuple of (height, width) in pixels.
        mask: Optional 2D boolean valid mask.
        tile_size: Tile dimension in pixels.
        tile_overlap: Overlap between adjacent tiles in pixels.
        level_idx: Pyramid level index.
        level_mapping: Mapping from level coordinates to original coordinates.
        max_megapixels_in_memory: Max allowed image/tile budget in megapixels.

    Returns:
        Deterministic list of TileInfo objects.

    Raises:
        ValueError: If tile dimensions are invalid or memory limits are exceeded.
    """
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got shape ({height}, {width})")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_overlap < 0:
        raise ValueError(f"tile_overlap must be non-negative, got {tile_overlap}")
    if tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap ({tile_overlap}) must be strictly less than tile_size ({tile_size})")

    # Check memory budget
    megapixels = (width * height) / 1.0e6
    if max_megapixels_in_memory is not None and megapixels > max_megapixels_in_memory:
        raise ValueError(
            f"Image memory size ({megapixels:.1f} MP) exceeds memory limit ({max_megapixels_in_memory} MP)"
        )

    lvl_map = level_mapping if level_mapping is not None else CoordinateMapping.identity()

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (height, width):
            raise ValueError(f"Mask shape {mask_arr.shape} does not match image shape ({height}, {width})")
    else:
        mask_arr = np.ones((height, width), dtype=bool)

    stride = tile_size - tile_overlap

    # Calculate tile top-left origin coordinates
    if width <= tile_size:
        x_offsets = [0]
    else:
        x_offsets = list(range(0, width - tile_size + 1, stride))
        if x_offsets[-1] + tile_size < width:
            x_offsets.append(width - tile_size)

    if height <= tile_size:
        y_offsets = [0]
    else:
        y_offsets = list(range(0, height - tile_size + 1, stride))
        if y_offsets[-1] + tile_size < height:
            y_offsets.append(height - tile_size)

    tiles: list[TileInfo] = []
    seen_windows: set[tuple[int, int, int, int]] = set()

    for col_off in x_offsets:
        for row_off in y_offsets:
            w_curr = min(tile_size, width - col_off)
            h_curr = min(tile_size, height - row_off)

            window_tuple = (col_off, row_off, w_curr, h_curr)
            if window_tuple in seen_windows:
                continue
            seen_windows.add(window_tuple)

            # Count valid pixels inside tile window
            tile_mask_slice = mask_arr[row_off : row_off + h_curr, col_off : col_off + w_curr]
            valid_cnt = int(np.count_nonzero(tile_mask_slice))

            # Filter out tiles with 0 valid pixels if mask has valid pixels
            if valid_cnt == 0 and np.any(mask_arr):
                continue

            window = RasterWindow(col_off=col_off, row_off=row_off, width=w_curr, height=h_curr)
            crop_map = CoordinateMapping.from_crop(float(col_off), float(row_off))
            tile_to_orig_map = lvl_map.compose(crop_map)

            tile_info = TileInfo(
                tile_id=len(tiles),
                window=window,
                level_idx=level_idx,
                level_mapping=lvl_map,
                tile_mapping=tile_to_orig_map,
                valid_pixel_count=valid_cnt,
            )
            tiles.append(tile_info)

    # Fallback if all tiles were filtered out but mask has valid pixels
    if not tiles and np.any(mask_arr):
        window = RasterWindow(col_off=0, row_off=0, width=width, height=height)
        tile_info = TileInfo(
            tile_id=0,
            window=window,
            level_idx=level_idx,
            level_mapping=lvl_map,
            tile_mapping=lvl_map.compose(CoordinateMapping.from_crop(0.0, 0.0)),
            valid_pixel_count=int(np.count_nonzero(mask_arr)),
        )
        tiles.append(tile_info)

    return tiles
