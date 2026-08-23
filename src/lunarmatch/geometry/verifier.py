"""Robust geometric verifier for point correspondence sets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from lunarmatch.geometry.fitting import estimate_initial_transform, refit_transform
from lunarmatch.geometry.validation import (
    check_collinearity,
    check_matrix_plausibility,
    check_minimum_points,
)
from lunarmatch.models.config import GeometryConfig
from lunarmatch.models.domain import MatchSet, TransformEstimate


@dataclass
class GeometricVerificationResult:
    """Typed result of geometric verification fitting and inlier filtering."""

    success: bool
    transform: TransformEstimate | None
    inlier_mask: np.ndarray  # (N,) boolean array
    inliers: MatchSet | None
    reprojection_residuals: np.ndarray  # (N,) float64 pixel errors
    failure_reason: str | None
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "success": self.success,
            "transform": self.transform.to_dict() if self.transform is not None else None,
            "inlier_mask": self.inlier_mask.tolist(),
            "inliers": self.inliers.to_dict() if self.inliers is not None else None,
            "reprojection_residuals": self.reprojection_residuals.tolist(),
            "failure_reason": self.failure_reason,
            "provenance": self.provenance,
        }


def compute_reprojection_residuals(
    matrix: np.ndarray, src_points: np.ndarray, ref_points: np.ndarray
) -> np.ndarray:
    """Compute per-point reprojection errors in pixels mapping source to reference.

    Args:
        matrix: 3x3 float64 transformation matrix mapping source to reference.
        src_points: Nx2 float64 source point coordinates (x, y).
        ref_points: Nx2 float64 reference point coordinates (x, y).

    Returns:
        N float64 array of Euclidean pixel distances between projected and reference points.
    """
    src = np.asarray(src_points, dtype=np.float64)
    ref = np.asarray(ref_points, dtype=np.float64)

    if len(src) == 0:
        return np.empty((0,), dtype=np.float64)

    homo = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])  # Nx3
    proj = (matrix @ homo.T).T  # Nx3

    w = proj[:, 2:3]
    # Guard against division by near-zero
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    proj_xy = proj[:, :2] / w_safe

    diffs = proj_xy - ref
    residuals = np.sqrt(np.sum(diffs**2, axis=1))
    return np.asarray(residuals, dtype=np.float64)


class GeometricVerifier:
    """Robust geometric verifier for similarity, affine, and homography models."""

    def __init__(
        self,
        config: GeometryConfig | None = None,
        model: Literal["similarity", "affine", "homography"] = "affine",
        robust_method: Literal["usac_magsac", "ransac"] = "usac_magsac",
        reprojection_threshold_px: float = 3.0,
        confidence: float = 0.999,
        max_iterations: int = 10000,
        allowed_scale: tuple[float, float] = (0.01, 100.0),
    ) -> None:
        if config is not None:
            self._model = config.model
            self._robust_method = config.robust_method
            self._reprojection_threshold = config.reprojection_threshold_px
            self._confidence = config.confidence
            self._max_iterations = config.max_iterations
            self._allowed_scale = config.allowed_scale
        else:
            self._model = model
            self._robust_method = robust_method
            self._reprojection_threshold = reprojection_threshold_px
            self._confidence = confidence
            self._max_iterations = max_iterations
            self._allowed_scale = allowed_scale

    def verify(
        self,
        matches: MatchSet | tuple[np.ndarray, np.ndarray],
        seed: int = 42,
    ) -> GeometricVerificationResult:
        """Fit geometric model and verify correspondences.

        Args:
            matches: MatchSet instance or tuple of (source_points, reference_points).
            seed: Random seed for RANSAC / USAC_MAGSAC reproducibility.

        Returns:
            GeometricVerificationResult containing typed status, transform, and failure details.
        """
        if isinstance(matches, MatchSet):
            src = matches.source_points
            ref = matches.reference_points
            scores = matches.scores
        else:
            src = np.asarray(matches[0], dtype=np.float64)
            ref = np.asarray(matches[1], dtype=np.float64)
            scores = np.ones(len(src), dtype=np.float64)

        n_pts = len(src)
        empty_mask = np.zeros(n_pts, dtype=bool)
        empty_residuals = np.full(n_pts, np.nan, dtype=np.float64)

        prov = {
            "model_type": self._model,
            "robust_method": self._robust_method,
            "reprojection_threshold_px": self._reprojection_threshold,
            "seed": seed,
            "input_match_count": n_pts,
        }

        # 1. Minimum point count check
        min_fail = check_minimum_points(n_pts, self._model)
        if min_fail is not None:
            return GeometricVerificationResult(
                success=False,
                transform=None,
                inlier_mask=empty_mask,
                inliers=None,
                reprojection_residuals=empty_residuals,
                failure_reason=min_fail,
                provenance=prov,
            )

        # 2. Degeneracy / Collinearity check
        if check_collinearity(src) or check_collinearity(ref):
            return GeometricVerificationResult(
                success=False,
                transform=None,
                inlier_mask=empty_mask,
                inliers=None,
                reprojection_residuals=empty_residuals,
                failure_reason="degenerate_points",
                provenance=prov,
            )

        # 3. Initial robust model fitting
        init_matrix, _ = estimate_initial_transform(
            src_points=src,
            ref_points=ref,
            model_type=self._model,
            robust_method=self._robust_method,
            reprojection_threshold_px=self._reprojection_threshold,
            confidence=self._confidence,
            max_iterations=self._max_iterations,
            seed=seed,
        )

        if init_matrix is None:
            return GeometricVerificationResult(
                success=False,
                transform=None,
                inlier_mask=empty_mask,
                inliers=None,
                reprojection_residuals=empty_residuals,
                failure_reason="estimation_failed",
                provenance=prov,
            )

        # 4. Compute initial residuals and inlier mask
        residuals = compute_reprojection_residuals(init_matrix, src, ref)
        inlier_mask = residuals <= self._reprojection_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))

        # Check minimum inliers after initial fit
        min_inlier_fail = check_minimum_points(inlier_count, self._model)
        if min_inlier_fail is not None:
            return GeometricVerificationResult(
                success=False,
                transform=None,
                inlier_mask=inlier_mask,
                inliers=None,
                reprojection_residuals=residuals,
                failure_reason="insufficient_inliers",
                provenance=prov,
            )

        # 5. Inlier refitting using least squares
        refitted_matrix = refit_transform(src[inlier_mask], ref[inlier_mask], self._model)
        final_matrix = init_matrix

        if refitted_matrix is not None:
            refitted_residuals = compute_reprojection_residuals(refitted_matrix, src, ref)
            refitted_inliers = refitted_residuals <= self._reprojection_threshold
            # Keep refitted matrix if it retains equal or greater inlier count
            if np.count_nonzero(refitted_inliers) >= inlier_count:
                final_matrix = refitted_matrix
                residuals = refitted_residuals
                inlier_mask = refitted_inliers
                inlier_count = int(np.count_nonzero(inlier_mask))

        # 6. Matrix plausibility check (singularity, reflection, scale)
        plaus_fail = check_matrix_plausibility(final_matrix, self._model, self._allowed_scale)
        if plaus_fail is not None:
            return GeometricVerificationResult(
                success=False,
                transform=None,
                inlier_mask=inlier_mask,
                inliers=None,
                reprojection_residuals=residuals,
                failure_reason=plaus_fail,
                provenance=prov,
            )

        # 7. Construct final typed transform and inliers MatchSet
        inlier_ratio = inlier_count / n_pts if n_pts > 0 else 0.0
        inlier_residuals = residuals[inlier_mask]
        rmse_px = float(np.sqrt(np.mean(inlier_residuals**2))) if len(inlier_residuals) > 0 else 0.0

        prov["inlier_count"] = inlier_count
        prov["inlier_ratio"] = inlier_ratio
        prov["rmse_px"] = rmse_px

        transform = TransformEstimate.from_matrix(
            model_type=self._model,
            matrix=final_matrix,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_px=rmse_px,
            provenance=prov,
        )

        inliers_match_set = MatchSet(
            source_points=src[inlier_mask],
            reference_points=ref[inlier_mask],
            scores=scores[inlier_mask],
            inliers=np.ones(inlier_count, dtype=bool),
        )

        return GeometricVerificationResult(
            success=True,
            transform=transform,
            inlier_mask=inlier_mask,
            inliers=inliers_match_set,
            reprojection_residuals=residuals,
            failure_reason=None,
            provenance=prov,
        )


def verify_matches(
    matches: MatchSet | tuple[np.ndarray, np.ndarray],
    config: GeometryConfig | None = None,
    model: Literal["similarity", "affine", "homography"] = "affine",
    robust_method: Literal["usac_magsac", "ransac"] = "usac_magsac",
    reprojection_threshold_px: float = 3.0,
    confidence: float = 0.999,
    max_iterations: int = 10000,
    allowed_scale: tuple[float, float] = (0.01, 100.0),
    seed: int = 42,
) -> GeometricVerificationResult:
    """Convenience function to perform robust geometric verification on point matches."""
    verifier = GeometricVerifier(
        config=config,
        model=model,
        robust_method=robust_method,
        reprojection_threshold_px=reprojection_threshold_px,
        confidence=confidence,
        max_iterations=max_iterations,
        allowed_scale=allowed_scale,
    )
    return verifier.verify(matches=matches, seed=seed)
