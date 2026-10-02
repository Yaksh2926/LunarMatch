"""Unit tests for uniform spatial match selection and coverage metrics."""
import numpy as np

from lunarmatch.matching import (
    GridPartition,
    SpatialSelector,
    select_uniform_matches,
)
from lunarmatch.models.config import CoverageConfig
from lunarmatch.models.domain import MatchSet


def test_grid_partition():
    """Test 2D grid partitioning coordinate mapping and cell IDs."""
    grid = GridPartition(min_x=0.0, min_y=0.0, max_x=800.0, max_y=800.0, grid_rows=8, grid_cols=8)

    assert grid.total_cells == 64
    assert grid.cell_width == 100.0
    assert grid.cell_height == 100.0

    # Top-left corner -> cell (0, 0) -> cell_id = 0
    assert grid.get_cell_id(10.0, 10.0) == 0

    # Bottom-right corner -> cell (7, 7) -> cell_id = 63
    assert grid.get_cell_id(790.0, 790.0) == 63

    # Vectorized test
    pts = np.array([[50, 50], [150, 50], [50, 150], [750, 750]], dtype=np.float64)
    cids = grid.get_cell_ids(pts)
    np.testing.assert_array_equal(cids, np.array([0, 1, 8, 63], dtype=np.int64))


def test_clustered_points_honest_low_coverage():
    """Test that clustered inputs yield low coverage and respect cell caps."""
    # 100 points tightly clustered in top-left cell [0, 50] x [0, 50]
    rng = np.random.default_rng(42)
    src_pts = rng.uniform(0.0, 50.0, size=(100, 2))
    ref_pts = rng.uniform(0.0, 50.0, size=(100, 2))
    scores = rng.uniform(0.5, 1.0, size=100)

    matches = MatchSet(source_points=src_pts, reference_points=ref_pts, scores=scores)

    res = select_uniform_matches(
        matches=matches,
        image_bounds=(0.0, 0.0, 800.0, 800.0),
        grid_rows=8,
        grid_cols=8,
        max_matches_per_cell=10,
        min_required_matches=10,
    )

    # 1 cell occupied out of 64
    assert len(res.selected_matches) == 10
    assert res.occupied_grid_fraction == 1.0 / 64.0
    assert res.convex_hull_coverage_fraction < 0.01
    assert res.selected_matches.cell_ids is not None
    assert all(cid == 0 for cid in res.selected_matches.cell_ids)


def test_uniform_points_high_coverage():
    """Test that evenly distributed points yield high coverage and zero CV."""
    # 64 points, exactly 1 per cell center in an 8x8 grid of size 800x800
    pts_list = []
    for r in range(8):
        for c in range(8):
            pts_list.append([c * 100.0 + 50.0, r * 100.0 + 50.0])

    ref_pts = np.array(pts_list, dtype=np.float64)
    src_pts = ref_pts.copy()
    scores = np.ones(64, dtype=np.float64)

    matches = MatchSet(source_points=src_pts, reference_points=ref_pts, scores=scores)

    res = select_uniform_matches(
        matches=matches,
        image_bounds=(0.0, 0.0, 800.0, 800.0),
        grid_rows=8,
        grid_cols=8,
        max_matches_per_cell=5,
        min_required_matches=10,
    )

    assert len(res.selected_matches) == 64
    assert res.occupied_grid_fraction == 1.0
    assert res.convex_hull_coverage_fraction > 0.70
    assert res.count_uniformity_cv == 0.0  # Exactly 1 match per cell -> std = 0


def test_partially_masked_points():
    """Test that invalid nodata cells are excluded from total cells denominator."""
    # 800x800 image, top half (r in [0, 3]) is nodata (False), bottom half (r in [4, 7]) is valid (True)
    mask = np.zeros((800, 800), dtype=bool)
    mask[400:800, 0:800] = True  # Bottom 32 cells are valid

    # Place 32 points uniformly across bottom 32 cells
    pts_list = []
    for r in range(4, 8):
        for c in range(8):
            pts_list.append([c * 100.0 + 50.0, r * 100.0 + 50.0])

    ref_pts = np.array(pts_list, dtype=np.float64)
    src_pts = ref_pts.copy()

    matches = MatchSet(source_points=src_pts, reference_points=ref_pts)

    res = select_uniform_matches(
        matches=matches,
        valid_mask=mask,
        grid_rows=8,
        grid_cols=8,
        max_matches_per_cell=5,
        min_required_matches=10,
    )

    assert res.provenance["total_valid_cells"] == 32
    assert res.provenance["occupied_valid_cells"] == 32
    # 32 occupied / 32 valid cells = 1.0 (not 32/64 = 0.5)
    assert res.occupied_grid_fraction == 1.0


def test_min_required_matches_preservation():
    """Test that min_required_matches tops up selection when cell capping is tight."""
    # 50 points in 1 cell, max_matches_per_cell=5, min_required_matches=20
    src_pts = np.zeros((50, 2), dtype=np.float64)
    ref_pts = np.zeros((50, 2), dtype=np.float64)
    scores = np.linspace(0.1, 1.0, 50)

    matches = MatchSet(source_points=src_pts, reference_points=ref_pts, scores=scores)

    res = select_uniform_matches(
        matches=matches,
        image_bounds=(0.0, 0.0, 800.0, 800.0),
        grid_rows=8,
        grid_cols=8,
        max_matches_per_cell=5,
        min_required_matches=20,
    )

    assert len(res.selected_matches) == 20


def test_deterministic_tie_breaking():
    """Test that match selection is strictly deterministic for tied scores."""
    src_pts = np.array([[10, 10], [10, 10], [10, 10]], dtype=np.float64)
    ref_pts = np.array([[15, 20], [12, 10], [14, 15]], dtype=np.float64)
    scores = np.array([0.9, 0.9, 0.9], dtype=np.float64)  # Tied scores

    matches = MatchSet(source_points=src_pts, reference_points=ref_pts, scores=scores)

    res1 = select_uniform_matches(matches, image_bounds=(0, 0, 100, 100), max_matches_per_cell=2, min_required_matches=1)
    res2 = select_uniform_matches(matches, image_bounds=(0, 0, 100, 100), max_matches_per_cell=2, min_required_matches=1)

    assert np.array_equal(res1.selected_indices, res2.selected_indices)


def test_spatial_selector_class_with_config():
    """Test SpatialSelector initialization with CoverageConfig."""
    cfg = CoverageConfig(grid_rows=4, grid_cols=4, max_matches_per_cell=10)
    selector = SpatialSelector(config=cfg, min_required_matches=5)

    src_pts = np.random.default_rng(1).uniform(0, 400, size=(30, 2))
    ref_pts = src_pts.copy()
    matches = MatchSet(source_points=src_pts, reference_points=ref_pts)

    res = selector.select(matches, image_bounds=(0, 0, 400, 400))
    assert res.provenance["grid_rows"] == 4
    assert res.provenance["grid_cols"] == 4
    assert res.provenance["max_matches_per_cell"] == 10


def test_registration_quality_gate_clustered_inlier_fixture():
    """Test RegistrationQualityGate correctly flags Phase 1 audit clustered-inlier case as UNDER_CONSTRAINED.

    Phase 1 Audit OHRC<->TMC-2 classical SIFT result:
      - 4 inliers (low)
      - occupied_grid_fraction = 0.016 (1.6% = 1/64 cells)
      - convex_hull_coverage_fraction = 0.01 (1.0%)
      - reprojection_rmse = 0.9224 px (passing low RMSE)
    Assert gate evaluates to UNDER_CONSTRAINED and False coverage pass.
    """
    from lunarmatch.matching.spatial_selector import RegistrationQualityGate

    gate = RegistrationQualityGate(
        min_occupied_grid_fraction=0.15,
        min_convex_hull_coverage=0.05,
        min_inliers=15,
        max_rmse_px=5.0,
    )

    status, passed = gate.evaluate(
        inlier_count=4,
        occupied_grid_fraction=0.016,
        convex_hull_coverage_fraction=0.01,
        rmse_px=0.9224,
    )

    assert status == "UNDER_CONSTRAINED"
    assert passed is False

