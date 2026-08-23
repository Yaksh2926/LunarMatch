"""Unit tests for overlapping tile planning, coverage guarantees, and memory limits."""
from __future__ import annotations

import numpy as np
import pytest

from lunarmatch.features import plan_tiles


def test_tile_planning_bounds_and_coverage() -> None:
    """Test tile planning satisfies coverage, no bounds overflow, and no duplicate tiles."""
    h, w = 3000, 5000
    tile_size = 2048
    tile_overlap = 128

    mask = np.ones((h, w), dtype=bool)
    mask[0:100, 0:100] = False  # Mask out top-left corner

    tiles = plan_tiles((h, w), mask=mask, tile_size=tile_size, tile_overlap=tile_overlap)

    assert len(tiles) > 0

    # Acceptance checks:
    # 1. No tile exceeds bounds
    seen_windows: set[tuple[int, int, int, int]] = set()
    coverage_map = np.zeros((h, w), dtype=bool)

    for t in tiles:
        win = t.window
        assert win.col_off >= 0
        assert win.row_off >= 0
        assert win.col_off + win.width <= w
        assert win.row_off + win.height <= h

        # Check duplicate avoidance
        window_tuple = (win.col_off, win.row_off, win.width, win.height)
        assert window_tuple not in seen_windows
        seen_windows.add(window_tuple)

        # Mark coverage
        coverage_map[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width] = True

    # 2. All valid pixels are covered as configured
    assert np.all(coverage_map[mask])


def test_small_image_tiling() -> None:
    """Test small image smaller than tile size returns a single tile matching image size."""
    h, w = 100, 100
    tiles = plan_tiles((h, w), tile_size=2048, tile_overlap=128)

    assert len(tiles) == 1
    win = tiles[0].window
    assert win.col_off == 0
    assert win.row_off == 0
    assert win.width == 100
    assert win.height == 100


def test_extreme_aspect_ratios() -> None:
    """Test tile planning on extreme aspect ratio images (wide and tall)."""
    # Wide image
    tiles_wide = plan_tiles((200, 5000), tile_size=1024, tile_overlap=64)
    for t in tiles_wide:
        assert t.window.col_off + t.window.width <= 5000
        assert t.window.row_off + t.window.height <= 200

    # Tall image
    tiles_tall = plan_tiles((5000, 200), tile_size=1024, tile_overlap=64)
    for t in tiles_tall:
        assert t.window.col_off + t.window.width <= 200
        assert t.window.row_off + t.window.height <= 5000


def test_tile_coordinate_mapping_roundtrip() -> None:
    """Test tile-local (x, y) to original (x, y) coordinate mapping round-trip."""
    h, w = 2500, 3500
    tiles = plan_tiles((h, w), tile_size=1024, tile_overlap=64)

    for t in tiles:
        # Tile local center point
        pts_local = np.array([[t.window.width / 2.0, t.window.height / 2.0]], dtype=np.float64)
        pts_orig = t.tile_mapping.apply(pts_local)

        # Original point must fall within image bounds
        assert 0.0 <= pts_orig[0, 0] <= w
        assert 0.0 <= pts_orig[0, 1] <= h

        # Round-trip back to local tile coordinates
        restored_local = t.tile_mapping.apply_inverse(pts_orig)
        np.testing.assert_allclose(restored_local, pts_local, atol=1e-12)


def test_memory_limit_enforcement() -> None:
    """Test memory limit enforcement raises ValueError when budget is exceeded."""
    with pytest.raises(ValueError, match="exceeds memory limit"):
        plan_tiles((5000, 5000), max_megapixels_in_memory=10)  # 25 MP image > 10 MP limit
