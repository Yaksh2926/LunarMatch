"""Uniform spatial match selection and coverage metric calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

from lunarmatch.matching.grid import GridPartition
from lunarmatch.models.config import CoverageConfig
from lunarmatch.models.domain import MatchSet


@dataclass
class SpatialSelectionResult:
    """Result of spatial match selection and coverage metrics."""

    selected_matches: MatchSet
    selected_indices: np.ndarray  # (M,) int64 indices into input match set
    occupied_grid_fraction: float  # Fraction of valid grid cells with >= 1 selected match
    convex_hull_coverage_fraction: float  # Ratio of convex hull area to total valid area
    count_uniformity_cv: float | None  # CV (std / mean) of match counts per valid cell
    per_cell_counts: dict[int, int]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "selected_matches": self.selected_matches.to_dict(),
            "selected_indices": self.selected_indices.tolist(),
            "occupied_grid_fraction": self.occupied_grid_fraction,
            "convex_hull_coverage_fraction": self.convex_hull_coverage_fraction,
            "count_uniformity_cv": self.count_uniformity_cv,
            "per_cell_counts": self.per_cell_counts,
            "provenance": self.provenance,
        }


def compute_convex_hull_area(points: np.ndarray) -> float:
    """Compute 2D convex hull area for a set of points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0

    # Check for duplicate or collinear points to avoid QhullError
    centered = pts - pts.mean(axis=0)
    try:
        s = np.linalg.svd(centered, compute_uv=False)
        if s[0] < 1e-12 or (s[1] / s[0]) < 1e-5:
            return 0.0
    except np.linalg.LinAlgError:
        return 0.0

    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # In 2D, hull.volume is the area
    except (ValueError, RuntimeError):
        return 0.0


class SpatialSelector:
    """Grid-based uniform spatial match selector."""

    def __init__(
        self,
        config: CoverageConfig | None = None,
        grid_rows: int = 8,
        grid_cols: int = 8,
        max_matches_per_cell: int = 20,
        min_required_matches: int = 15,
        use_reference_coords: bool = True,
    ) -> None:
        if config is not None:
            self._grid_rows = config.grid_rows
            self._grid_cols = config.grid_cols
            self._max_matches_per_cell = config.max_matches_per_cell
            self._min_required_matches = min_required_matches
        else:
            self._grid_rows = grid_rows
            self._grid_cols = grid_cols
            self._max_matches_per_cell = max_matches_per_cell
            self._min_required_matches = min_required_matches

        self._use_reference_coords = use_reference_coords

    def select(
        self,
        matches: MatchSet,
        image_bounds: tuple[float, float, float, float] | tuple[int, int] | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> SpatialSelectionResult:
        """Select spatially uniform subset of matches and calculate coverage metrics."""
        n_pts = len(matches)

        if self._use_reference_coords:
            pts = matches.reference_points
        else:
            pts = matches.source_points

        scores = matches.scores
        prov = {
            "grid_rows": self._grid_rows,
            "grid_cols": self._grid_cols,
            "max_matches_per_cell": self._max_matches_per_cell,
            "min_required_matches": self._min_required_matches,
            "input_match_count": n_pts,
        }

        if n_pts == 0:
            empty_ms = MatchSet(
                source_points=np.empty((0, 2), dtype=np.float64),
                reference_points=np.empty((0, 2), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
                cell_ids=[],
            )
            return SpatialSelectionResult(
                selected_matches=empty_ms,
                selected_indices=np.empty((0,), dtype=np.int64),
                occupied_grid_fraction=0.0,
                convex_hull_coverage_fraction=0.0,
                count_uniformity_cv=None,
                per_cell_counts={},
                provenance=prov,
            )

        # 1. Bounding box calculation
        if image_bounds is not None:
            if len(image_bounds) == 2:
                w_b, h_b = image_bounds
                min_x, min_y, max_x, max_y = 0.0, 0.0, float(w_b), float(h_b)
            else:
                min_x, min_y, max_x, max_y = (
                    float(image_bounds[0]),
                    float(image_bounds[1]),
                    float(image_bounds[2]),
                    float(image_bounds[3]),
                )
        elif valid_mask is not None:
            h_m, w_m = valid_mask.shape
            min_x, min_y, max_x, max_y = 0.0, 0.0, float(w_m), float(h_m)
        else:
            min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
            min_y, max_y = float(pts[:, 1].min()), float(pts[:, 1].max())
            if max_x <= min_x:
                max_x = min_x + 1.0
            if max_y <= min_y:
                max_y = min_y + 1.0

        # 2. Partition points into grid
        partition = GridPartition(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            grid_rows=self._grid_rows,
            grid_cols=self._grid_cols,
        )

        match_cell_ids = partition.get_cell_ids(pts)

        # 3. Group and rank matches by cell
        cell_match_map: dict[int, list[int]] = {}
        for i in range(n_pts):
            cid = int(match_cell_ids[i])
            cell_match_map.setdefault(cid, []).append(i)

        selected_indices_set: set[int] = set()

        for cid, idx_list in cell_match_map.items():
            # Deterministic sort: score descending, x ascending, y ascending, index ascending
            idx_list.sort(key=lambda i: (-scores[i], pts[i, 0], pts[i, 1], i))

            # Select top max_matches_per_cell
            kept = idx_list[: self._max_matches_per_cell]
            selected_indices_set.update(kept)

        # 4. Top up to min_required_matches if capping reduced match count below threshold
        target_min = min(n_pts, self._min_required_matches)
        if len(selected_indices_set) < target_min:
            unselected = [i for i in range(n_pts) if i not in selected_indices_set]
            unselected.sort(key=lambda i: (-scores[i], pts[i, 0], pts[i, 1], i))
            needed = target_min - len(selected_indices_set)
            selected_indices_set.update(unselected[:needed])

        selected_indices = np.array(sorted(selected_indices_set), dtype=np.int64)

        # 5. Calculate coverage and uniformity metrics
        valid_cells_mask = partition.compute_valid_cells(valid_mask)
        valid_cell_indices = np.where(valid_cells_mask)[0]
        n_valid_cells = len(valid_cell_indices)
        if n_valid_cells == 0:
            n_valid_cells = partition.total_cells

        # Compute per-cell counts for selected matches
        per_cell_counts: dict[int, int] = {int(cid): 0 for cid in range(partition.total_cells)}
        for idx in selected_indices:
            cid = int(match_cell_ids[idx])
            per_cell_counts[cid] = per_cell_counts.get(cid, 0) + 1

        # Occupied fraction over valid cells
        occupied_valid_cells = sum(1 for cid in valid_cell_indices if per_cell_counts.get(int(cid), 0) > 0)
        occupied_grid_fraction = float(occupied_valid_cells / n_valid_cells)

        # Convex hull coverage
        selected_pts = pts[selected_indices]
        hull_area = compute_convex_hull_area(selected_pts)

        if valid_mask is not None:
            total_valid_area = float(np.count_nonzero(valid_mask))
        else:
            total_valid_area = partition.width * partition.height

        if total_valid_area > 0:
            convex_hull_coverage_fraction = float(min(1.0, hull_area / total_valid_area))
        else:
            convex_hull_coverage_fraction = 0.0

        # Per-cell count uniformity CV (std / mean across valid cells)
        valid_counts = np.array([per_cell_counts[int(cid)] for cid in valid_cell_indices], dtype=np.float64)
        mean_count = float(valid_counts.mean())
        if mean_count > 0.0:
            std_count = float(valid_counts.std())
            count_uniformity_cv: float | None = float(std_count / mean_count)
        else:
            count_uniformity_cv = None

        # 6. Construct selected MatchSet
        selected_src_pts = matches.source_points[selected_indices]
        selected_ref_pts = matches.reference_points[selected_indices]
        selected_scores = matches.scores[selected_indices]
        selected_inliers = matches.inliers[selected_indices]
        selected_cids = [int(c) for c in match_cell_ids[selected_indices]]

        selected_match_set = MatchSet(
            source_points=selected_src_pts,
            reference_points=selected_ref_pts,
            scores=selected_scores,
            inliers=selected_inliers,
            cell_ids=selected_cids,
        )

        prov["selected_match_count"] = len(selected_indices)
        prov["occupied_valid_cells"] = occupied_valid_cells
        prov["total_valid_cells"] = n_valid_cells

        return SpatialSelectionResult(
            selected_matches=selected_match_set,
            selected_indices=selected_indices,
            occupied_grid_fraction=occupied_grid_fraction,
            convex_hull_coverage_fraction=convex_hull_coverage_fraction,
            count_uniformity_cv=count_uniformity_cv,
            per_cell_counts=per_cell_counts,
            provenance=prov,
        )


def select_uniform_matches(
    matches: MatchSet,
    config: CoverageConfig | None = None,
    image_bounds: tuple[float, float, float, float] | tuple[int, int] | None = None,
    valid_mask: np.ndarray | None = None,
    grid_rows: int = 8,
    grid_cols: int = 8,
    max_matches_per_cell: int = 20,
    min_required_matches: int = 15,
    use_reference_coords: bool = True,
) -> SpatialSelectionResult:
    """Convenience function for uniform spatial match selection and coverage metrics."""
    selector = SpatialSelector(
        config=config,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        max_matches_per_cell=max_matches_per_cell,
        min_required_matches=min_required_matches,
        use_reference_coords=use_reference_coords,
    )
    return selector.select(
        matches=matches, image_bounds=image_bounds, valid_mask=valid_mask
    )


class RegistrationQualityGate:
    """Classifies registration results as WELL_CONSTRAINED vs UNDER_CONSTRAINED based on spatial coverage and inliers."""

    def __init__(
        self,
        min_occupied_grid_fraction: float = 0.15,
        min_convex_hull_coverage: float = 0.05,
        min_inliers: int = 15,
        max_rmse_px: float | None = 5.0,
    ) -> None:
        self.min_occupied_grid_fraction = min_occupied_grid_fraction
        self.min_convex_hull_coverage = min_convex_hull_coverage
        self.min_inliers = min_inliers
        self.max_rmse_px = max_rmse_px

    def evaluate(
        self,
        inlier_count: int,
        occupied_grid_fraction: float,
        convex_hull_coverage_fraction: float,
        rmse_px: float | None = None,
    ) -> tuple[Literal["WELL_CONSTRAINED", "UNDER_CONSTRAINED"], bool]:
        """Evaluate spatial coverage and metrics.

        Returns:
            ("WELL_CONSTRAINED", True) if all spatial coverage and metric thresholds pass.
            ("UNDER_CONSTRAINED", False) if spatial coverage is clustered or under-constrained.
        """
        coverage_passed = (
            occupied_grid_fraction >= self.min_occupied_grid_fraction
            and convex_hull_coverage_fraction >= self.min_convex_hull_coverage
        )
        inliers_passed = inlier_count >= self.min_inliers
        rmse_passed = (self.max_rmse_px is None) or (rmse_px is not None and rmse_px <= self.max_rmse_px)

        well_constrained = bool(coverage_passed and inliers_passed and rmse_passed)
        status: Literal["WELL_CONSTRAINED", "UNDER_CONSTRAINED"] = (
            "WELL_CONSTRAINED" if well_constrained else "UNDER_CONSTRAINED"
        )
        return status, coverage_passed

