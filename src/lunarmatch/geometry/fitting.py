"""Robust geometric model fitting and least-squares refitting functions."""
from __future__ import annotations

import cv2
import numpy as np


def get_robust_flag(robust_method: str, model_type: str = "homography") -> int:
    """Get OpenCV method flag for RANSAC / USAC_MAGSAC."""
    method_clean = robust_method.lower().strip()
    model_clean = model_type.lower().strip()
    if (
        method_clean == "usac_magsac"
        and model_clean == "homography"
        and hasattr(cv2, "USAC_MAGSAC")
    ):
        return int(cv2.USAC_MAGSAC)
    return int(cv2.RANSAC)


def estimate_initial_transform(
    src_points: np.ndarray,
    ref_points: np.ndarray,
    model_type: str,
    robust_method: str = "usac_magsac",
    reprojection_threshold_px: float = 3.0,
    confidence: float = 0.999,
    max_iterations: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate initial 3x3 transformation matrix mapping source to reference using OpenCV RANSAC/USAC_MAGSAC.

    Returns:
        Tuple of (3x3 float64 matrix or None, 1D boolean inlier mask or None).
    """
    src = np.asarray(src_points, dtype=np.float64)
    ref = np.asarray(ref_points, dtype=np.float64)
    n_pts = len(src)

    if n_pts == 0 or len(ref) != n_pts:
        return None, None

    model_clean = model_type.lower().strip()
    flag = get_robust_flag(robust_method, model_clean)

    # Set deterministic RNG seed
    cv2.setRNGSeed(seed)

    mat_3x3: np.ndarray | None = None
    inliers_mask: np.ndarray | None = None

    if model_clean == "similarity":
        if n_pts < 2:
            return None, None
        est_func = getattr(cv2, "estimateAffinePartial2D", None)
        if est_func is not None:
            res_mat, inliers = est_func(
                src,
                ref,
                method=flag,
                ransacReprojThreshold=reprojection_threshold_px,
                maxIters=max_iterations,
                confidence=confidence,
            )
            if res_mat is not None and res_mat.shape == (2, 3):
                mat_3x3 = np.vstack([res_mat, [0.0, 0.0, 1.0]])
                inliers_mask = (inliers.ravel() > 0) if inliers is not None else np.ones(n_pts, dtype=bool)

    elif model_clean == "affine":
        if n_pts < 3:
            return None, None
        est_func = getattr(cv2, "estimateAffine2D", None)
        if est_func is not None:
            res_mat, inliers = est_func(
                src,
                ref,
                method=flag,
                ransacReprojThreshold=reprojection_threshold_px,
                maxIters=max_iterations,
                confidence=confidence,
            )
            if res_mat is not None and res_mat.shape == (2, 3):
                mat_3x3 = np.vstack([res_mat, [0.0, 0.0, 1.0]])
                inliers_mask = (inliers.ravel() > 0) if inliers is not None else np.ones(n_pts, dtype=bool)

    elif model_clean == "homography":
        if n_pts < 4:
            return None, None
        res_mat, inliers = cv2.findHomography(
            src,
            ref,
            method=flag,
            ransacReprojThreshold=reprojection_threshold_px,
            maxIters=max_iterations,
            confidence=confidence,
        )
        if res_mat is not None and res_mat.shape == (3, 3):
            # Normalize homography matrix by m[2,2]
            w = res_mat[2, 2]
            if abs(w) > 1e-12:
                res_mat = res_mat / w
            mat_3x3 = res_mat.astype(np.float64)
            inliers_mask = (inliers.ravel() > 0) if inliers is not None else np.ones(n_pts, dtype=bool)

    return mat_3x3, inliers_mask


def refit_similarity(src_inliers: np.ndarray, ref_inliers: np.ndarray) -> np.ndarray | None:
    """Least-squares refit for similarity transformation (4 DOFs: s, theta, tx, ty)."""
    n_pts = len(src_inliers)
    if n_pts < 2:
        return None

    # Solve A * x = b where x = [a, b, tx, ty]^T, a = s*cos(theta), b = s*sin(theta)
    # x_ref = a * x_src - b * y_src + tx
    # y_ref = b * x_src + a * y_src + ty
    a_mat = np.zeros((2 * n_pts, 4), dtype=np.float64)
    b_vec = np.zeros(2 * n_pts, dtype=np.float64)

    for i in range(n_pts):
        xs, ys = src_inliers[i]
        xr, yr = ref_inliers[i]

        a_mat[2 * i] = [xs, -ys, 1.0, 0.0]
        b_vec[2 * i] = xr

        a_mat[2 * i + 1] = [ys, xs, 0.0, 1.0]
        b_vec[2 * i + 1] = yr

    try:
        x, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b_val, tx, ty = x
    mat = np.array(
        [
            [a, -b_val, tx],
            [b_val, a, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mat


def refit_affine(src_inliers: np.ndarray, ref_inliers: np.ndarray) -> np.ndarray | None:
    """Least-squares refit for affine transformation (6 DOFs)."""
    n_pts = len(src_inliers)
    if n_pts < 3:
        return None

    # Solve A * h_row = b for each output row (x_ref and y_ref)
    a_mat = np.hstack([src_inliers, np.ones((n_pts, 1), dtype=np.float64)])  # Nx3

    try:
        row1, _, _, _ = np.linalg.lstsq(a_mat, ref_inliers[:, 0], rcond=None)
        row2, _, _, _ = np.linalg.lstsq(a_mat, ref_inliers[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return None

    mat = np.array(
        [
            [row1[0], row1[1], row1[2]],
            [row2[0], row2[1], row2[2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mat


def refit_homography(src_inliers: np.ndarray, ref_inliers: np.ndarray) -> np.ndarray | None:
    """Least-squares refit for homography transformation using DLT SVD."""
    n_pts = len(src_inliers)
    if n_pts < 4:
        return None

    d_mat = np.zeros((2 * n_pts, 9), dtype=np.float64)
    for i in range(n_pts):
        xs, ys = src_inliers[i]
        xr, yr = ref_inliers[i]

        d_mat[2 * i] = [-xs, -ys, -1.0, 0.0, 0.0, 0.0, xr * xs, xr * ys, xr]
        d_mat[2 * i + 1] = [0.0, 0.0, 0.0, -xs, -ys, -1.0, yr * xs, yr * ys, yr]

    try:
        _, _, vh = np.linalg.svd(d_mat)
    except np.linalg.LinAlgError:
        return None

    h = vh[-1].reshape((3, 3))
    w = float(h[2, 2])
    if abs(w) > 1e-12:
        h = h / w

    res: np.ndarray = np.asarray(h, dtype=np.float64)
    return res


def refit_transform(
    src_inliers: np.ndarray, ref_inliers: np.ndarray, model_type: str
) -> np.ndarray | None:
    """Refit transformation parameters using only inlier points."""
    model_clean = model_type.lower().strip()
    if model_clean == "similarity":
        return refit_similarity(src_inliers, ref_inliers)
    elif model_clean == "affine":
        return refit_affine(src_inliers, ref_inliers)
    elif model_clean == "homography":
        return refit_homography(src_inliers, ref_inliers)
    return None
