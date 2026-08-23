"""Independent ground-truth control-point CSV ingestion and residual evaluation."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lunarmatch.evaluation.models import ControlPoint, ControlPointMetrics
from lunarmatch.models.domain import TransformEstimate


def load_control_points(
    csv_path: str | Path,
    source_bounds: tuple[float, float] | None = None,
    reference_bounds: tuple[float, float] | None = None,
) -> list[ControlPoint]:
    """Ingest control points from CSV file with strict coordinate validation.

    Supported CSV column names (case-insensitive):
    - Source X: 'source_x', 'src_x', 'x_source', 'x_src'
    - Source Y: 'source_y', 'src_y', 'y_source', 'y_src'
    - Reference X: 'reference_x', 'ref_x', 'x_reference', 'x_ref'
    - Reference Y: 'reference_y', 'ref_y', 'y_reference', 'y_ref'
    - Optional ID: 'point_id', 'id', 'name'

    Args:
        csv_path: Path to control points CSV file.
        source_bounds: Optional (width, height) boundary tuple for source image.
        reference_bounds: Optional (width, height) boundary tuple for reference image.

    Returns:
        List of validated ControlPoint instances.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing, values are non-numeric or non-finite,
                   or points fall outside image boundaries.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Control points file not found: {path}")

    control_points: list[ControlPoint] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Control points CSV file is empty: {path}")

        # Normalize column header mapping
        field_map = {name.strip().lower(): name for name in reader.fieldnames}

        def get_col(possible_names: list[str]) -> str:
            for name in possible_names:
                if name in field_map:
                    return field_map[name]
            raise ValueError(
                f"Missing required control point column (expected one of {possible_names}). Found columns: {reader.fieldnames}"
            )

        src_x_col = get_col(["source_x", "src_x", "x_source", "x_src"])
        src_y_col = get_col(["source_y", "src_y", "y_source", "y_src"])
        ref_x_col = get_col(["reference_x", "ref_x", "x_reference", "x_ref"])
        ref_y_col = get_col(["reference_y", "ref_y", "y_reference", "y_ref"])

        id_col = None
        for name in ["point_id", "id", "name"]:
            if name in field_map:
                id_col = field_map[name]
                break

        for row_idx, row in enumerate(reader, start=1):
            try:
                sx = float(row[src_x_col].strip())
                sy = float(row[src_y_col].strip())
                rx = float(row[ref_x_col].strip())
                ry = float(row[ref_y_col].strip())
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid numeric coordinate at line {row_idx} of '{path.name}': {exc}"
                ) from exc

            if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(rx) and np.isfinite(ry)):
                raise ValueError(
                    f"Non-finite control point coordinate at line {row_idx}: ({sx}, {sy}) -> ({rx}, {ry})"
                )

            if source_bounds is not None:
                w_src, h_src = source_bounds
                if not (0.0 <= sx <= w_src and 0.0 <= sy <= h_src):
                    raise ValueError(
                        f"Source control point ({sx}, {sy}) at line {row_idx} is out of bounds (0..{w_src}, 0..{h_src})"
                    )

            if reference_bounds is not None:
                w_ref, h_ref = reference_bounds
                if not (0.0 <= rx <= w_ref and 0.0 <= ry <= h_ref):
                    raise ValueError(
                        f"Reference control point ({rx}, {ry}) at line {row_idx} is out of bounds (0..{w_ref}, 0..{h_ref})"
                    )

            pt_id = row[id_col].strip() if id_col and row.get(id_col) else f"cp_{row_idx}"
            control_points.append(
                ControlPoint(
                    source_x=sx,
                    source_y=sy,
                    reference_x=rx,
                    reference_y=ry,
                    point_id=pt_id,
                )
            )

    if not control_points:
        raise ValueError(f"No valid control point rows found in '{path.name}'")

    return control_points


def evaluate_control_points(
    control_points: list[ControlPoint],
    transform: TransformEstimate | np.ndarray,
) -> ControlPointMetrics:
    """Evaluate predicted vs reference pixel locations for independent control points.

    Args:
        control_points: List of ground-truth ControlPoint instances.
        transform: TransformEstimate instance or 3x3 transformation matrix.

    Returns:
        ControlPointMetrics containing RMSE, median, P90, P95, max error, and per-point residuals.

    Raises:
        ValueError: If control_points is empty or transform matrix is invalid.
    """
    if not control_points:
        raise ValueError("Cannot evaluate control points: control_points list is empty.")

    if isinstance(transform, TransformEstimate):
        M = transform.get_matrix_array()
    else:
        M = np.asarray(transform, dtype=np.float64)
        if M.shape != (3, 3):
            raise ValueError(f"Transform matrix must be 3x3, got shape {M.shape}")

    src_pts = np.array([[pt.source_x, pt.source_y] for pt in control_points], dtype=np.float64)
    ref_pts = np.array([[pt.reference_x, pt.reference_y] for pt in control_points], dtype=np.float64)

    # Convert source points to homogeneous coordinates (N, 3)
    n_pts = len(src_pts)
    ones = np.ones((n_pts, 1), dtype=np.float64)
    src_homo = np.hstack([src_pts, ones])

    # Apply transform: p_pred = (M @ src_homo.T).T
    pred_homo = (M @ src_homo.T).T  # (N, 3)

    # Normalize by scale factor W (row 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = pred_homo[:, 2:3]
        w[w == 0] = 1e-12
        pred_pts = pred_homo[:, :2] / w

    # Compute Euclidean distance residual per point: sqrt((x_pred - x_ref)^2 + (y_pred - y_ref)^2)
    diffs = pred_pts - ref_pts
    residuals_px = np.linalg.norm(diffs, axis=1)

    rmse = float(np.sqrt(np.mean(residuals_px**2)))
    med = float(np.median(residuals_px))
    p90 = float(np.percentile(residuals_px, 90))
    p95 = float(np.percentile(residuals_px, 95))
    max_err = float(np.max(residuals_px))

    return ControlPointMetrics(
        cp_rmse_px=rmse,
        cp_median_px=med,
        cp_p90_px=p90,
        cp_p95_px=p95,
        cp_max_px=max_err,
        count=n_pts,
        residuals_px=[float(r) for r in residuals_px],
    )
