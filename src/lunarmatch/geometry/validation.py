"""Validation and plausibility checks for geometric transformation estimates."""
from __future__ import annotations

import numpy as np

MIN_POINTS_MAP: dict[str, int] = {
    "similarity": 2,
    "affine": 3,
    "homography": 4,
}


def check_minimum_points(n_points: int, model_type: str) -> str | None:
    """Check if point count satisfies minimum requirement for model type.

    Returns:
        Failure reason string if point count is insufficient, else None.
    """
    model_clean = model_type.lower().strip()
    min_required = MIN_POINTS_MAP.get(model_clean, 3)
    if n_points < min_required:
        return "insufficient_points"
    return None


def check_collinearity(points: np.ndarray, tol: float = 1e-5) -> bool:
    """Check if 2D point set is degenerate (e.g. collinear or coincident).

    Args:
        points: Nx2 float64 array of point coordinates.
        tol: Tolerance for normalized singular value or triangle area.

    Returns:
        True if points are collinear or degenerate, False otherwise.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_pts = len(pts)

    if n_pts < 3:
        return False  # 2 points cannot be collinear in a degenerate 2D polygon sense for similarity

    # Center points
    centered = pts - pts.mean(axis=0)

    # Compute SVD of centered coordinates
    try:
        s = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return True

    # If the smaller singular value relative to the larger is below tol, points lie on a line
    if s[0] < 1e-12:
        return True

    ratio = s[1] / s[0]
    return bool(ratio < tol)


def check_matrix_plausibility(
    matrix: np.ndarray,
    model_type: str,
    allowed_scale: tuple[float, float] = (0.01, 100.0),
) -> str | None:
    """Check if estimated 3x3 transformation matrix is physically plausible.

    Verifies:
    1. Matrix elements are finite.
    2. Matrix is non-singular.
    3. Matrix preserves orientation (no mirror reflections, det > 0).
    4. Matrix scale factor is within allowed_scale range.

    Args:
        matrix: 3x3 float64 transformation matrix mapping source to reference.
        model_type: Transformation model type ('similarity', 'affine', 'homography').
        allowed_scale: Tuple of (min_scale, max_scale).

    Returns:
        Failure reason string if matrix fails any plausibility check, else None.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        return "invalid_matrix_shape"

    if not np.all(np.isfinite(mat)):
        return "non_finite_matrix"

    min_scale, max_scale = allowed_scale

    # 1. Non-singularity check
    try:
        det_full = np.linalg.det(mat)
    except np.linalg.LinAlgError:
        return "singular_matrix"

    if abs(det_full) < 1e-12:
        return "singular_matrix"

    # Extract 2x2 linear portion
    a22 = mat[:2, :2]
    det_2d = float(a22[0, 0] * a22[1, 1] - a22[0, 1] * a22[1, 0])

    if abs(det_2d) < 1e-12:
        return "singular_matrix"

    # 2. Reflection check (orientation preservation)
    # For similarity and affine, det_2d > 0. If det_2d < 0, it is a mirror flip (reflection).
    if det_2d <= 0.0:
        return "reflection_detected"

    # 3. Scale plausibility check
    # Scale factor S is sqrt(det_2d) for 2D area expansion ratio
    scale_factor = float(np.sqrt(det_2d))
    if scale_factor < min_scale or scale_factor > max_scale:
        return "scale_out_of_bounds"

    # Model specific check for homography
    if model_type.lower().strip() == "homography":
        # Ensure w-normalization element is reasonable
        w_val = abs(mat[2, 2])
        if w_val < 1e-8:
            return "singular_matrix"

    return None
