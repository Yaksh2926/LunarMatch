"""Sub-pixel patch refinement using ECC and phase correlation fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from lunarmatch.geometry.verifier import compute_reprojection_residuals, verify_matches
from lunarmatch.models.config import GeometryConfig, SubpixelConfig
from lunarmatch.models.domain import MatchSet, TransformEstimate


@dataclass
class SubpixelRefinementResult:
    """Typed result of patch-based sub-pixel correspondence refinement."""

    refined_matches: MatchSet
    refit_transform: TransformEstimate | None
    refine_status: list[str]  # Per-match status
    shifts_px: np.ndarray  # (N, 2) sub-pixel displacement vectors (dx, dy)
    pre_residuals_px: np.ndarray  # (N,) pre-refinement reprojection errors
    post_residuals_px: np.ndarray  # (N,) post-refinement reprojection errors
    accepted_count: int
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "refined_matches": self.refined_matches.to_dict(),
            "refit_transform": self.refit_transform.to_dict() if self.refit_transform is not None else None,
            "refine_status": self.refine_status,
            "shifts_px": self.shifts_px.tolist(),
            "pre_residuals_px": self.pre_residuals_px.tolist(),
            "post_residuals_px": self.post_residuals_px.tolist(),
            "accepted_count": self.accepted_count,
            "provenance": self.provenance,
        }


def extract_subpixel_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    patch_radius: int,
) -> np.ndarray | None:
    """Extract a square sub-pixel patch centered at (center_x, center_y).

    Args:
        image: 2D float32 or uint8 grayscale image array.
        center_x: Sub-pixel X center coordinate.
        center_y: Sub-pixel Y center coordinate.
        patch_radius: Patch radius in pixels. Patch size is (2*r+1, 2*r+1).

    Returns:
        Float32 patch array of shape (2*r+1, 2*r+1) or None if out of bounds.
    """
    img_arr = np.asarray(image, dtype=np.float32)
    h, w = img_arr.shape
    patch_size = 2 * patch_radius + 1

    margin = float(patch_radius)
    if (
        center_x < margin
        or center_x >= w - margin
        or center_y < margin
        or center_y >= h - margin
    ):
        return None

    patch = cv2.getRectSubPix(
        img_arr,
        patchSize=(patch_size, patch_size),
        center=(float(center_x), float(center_y)),
    )
    return patch


class SubpixelRefiner:
    """Patch-based sub-pixel correspondence refiner.

    Note:
        Returning fractional coordinate outputs alone does NOT prove physical sub-pixel accuracy.
        Sub-pixel accuracy must be verified empirically on synthetic held fixtures with known ground-truth shifts.
    """

    def __init__(
        self,
        config: SubpixelConfig | None = None,
        method: Literal["ecc_patch", "phase_correlation", "none"] = "ecc_patch",
        patch_radius: int = 15,
        max_iterations: int = 50,
        epsilon: float = 0.0001,
        max_shift_px: float = 3.0,
        min_quality: float = 0.2,
    ) -> None:
        if config is not None:
            self._method = config.method
            self._patch_radius = config.patch_radius
            self._max_iterations = config.max_iterations
            self._epsilon = config.epsilon
            self._max_shift_px = config.max_shift_px
            self._min_quality = config.min_quality
        else:
            self._method = method
            self._patch_radius = patch_radius
            self._max_iterations = max_iterations
            self._epsilon = epsilon
            self._max_shift_px = max_shift_px
            self._min_quality = min_quality

    def refine(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        matches: MatchSet,
        initial_transform: TransformEstimate | None = None,
        source_mask: np.ndarray | None = None,
        reference_mask: np.ndarray | None = None,
        geometry_config: GeometryConfig | None = None,
    ) -> SubpixelRefinementResult:
        """Refine match correspondence coordinates to sub-pixel precision."""
        src_img = np.asarray(source_image, dtype=np.float32)
        ref_img = np.asarray(reference_image, dtype=np.float32)
        n_pts = len(matches)

        src_pts = matches.source_points.copy()
        ref_pts = matches.reference_points.copy()
        scores = matches.scores.copy()
        inliers = matches.inliers.copy()

        prov = {
            "method": self._method,
            "patch_radius": self._patch_radius,
            "max_iterations": self._max_iterations,
            "epsilon": self._epsilon,
            "max_shift_px": self._max_shift_px,
            "min_quality": self._min_quality,
            "input_match_count": n_pts,
        }

        if n_pts == 0 or self._method == "none":
            empty_shifts = np.zeros((n_pts, 2), dtype=np.float64)
            empty_res = np.zeros(n_pts, dtype=np.float64)
            return SubpixelRefinementResult(
                refined_matches=matches,
                refit_transform=initial_transform,
                refine_status=["none"] * n_pts if self._method == "none" else [],
                shifts_px=empty_shifts,
                pre_residuals_px=empty_res,
                post_residuals_px=empty_res,
                accepted_count=n_pts if self._method == "none" else 0,
                provenance=prov,
            )

        # Pre-refinement reprojection residuals
        if initial_transform is not None:
            pre_residuals = compute_reprojection_residuals(
                initial_transform.get_matrix_array(), src_pts, ref_pts
            )
        else:
            pre_residuals = np.zeros(n_pts, dtype=np.float64)

        refined_ref_pts = ref_pts.copy()
        shifts = np.zeros((n_pts, 2), dtype=np.float64)
        statuses: list[str] = []
        accepted_count = 0

        h_src, w_src = src_img.shape
        h_ref, w_ref = ref_img.shape

        margin = float(self._patch_radius + int(np.ceil(self._max_shift_px)) + 1)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self._max_iterations,
            self._epsilon,
        )

        for i in range(n_pts):
            xs, ys = src_pts[i]
            xr, yr = ref_pts[i]

            # 1. Border check
            if (
                xs < margin
                or xs >= w_src - margin
                or ys < margin
                or ys >= h_src - margin
                or xr < margin
                or xr >= w_ref - margin
                or yr < margin
                or yr >= h_ref - margin
            ):
                statuses.append("border_rejected")
                continue

            # 2. Validity mask check
            if source_mask is not None:
                sm = np.asarray(source_mask, dtype=bool)
                r_start, r_end = int(ys - margin), int(ys + margin)
                c_start, c_end = int(xs - margin), int(xs + margin)
                if not np.all(sm[r_start:r_end, c_start:c_end]):
                    statuses.append("mask_rejected")
                    continue

            if reference_mask is not None:
                rm = np.asarray(reference_mask, dtype=bool)
                r_start, r_end = int(yr - margin), int(yr + margin)
                c_start, c_end = int(xr - margin), int(xr + margin)
                if not np.all(rm[r_start:r_end, c_start:c_end]):
                    statuses.append("mask_rejected")
                    continue

            # 3. Extract sub-pixel patches
            patch_src = extract_subpixel_patch(src_img, xs, ys, self._patch_radius)
            patch_ref = extract_subpixel_patch(ref_img, xr, yr, self._patch_radius)

            if patch_src is None or patch_ref is None:
                statuses.append("border_rejected")
                continue

            # Check patch variance / non-uniformity
            std_src = patch_src.std()
            std_ref = patch_ref.std()
            if std_src < 1e-4 or std_ref < 1e-4:
                statuses.append("low_quality")
                continue

            # Normalize patches to zero mean and unit variance for correlation alignment
            p_src = (patch_src - patch_src.mean()) / (std_src + 1e-6)
            p_ref = (patch_ref - patch_ref.mean()) / (std_ref + 1e-6)

            dx, dy = 0.0, 0.0
            refined_ok = False

            # 4. Primary refinement: ECC
            # findTransformECC(p_ref, p_src) finds warp mapping template (p_ref) to input (p_src)
            # warp_matrix[0, 2], warp_matrix[1, 2] gives (dx, dy) translation offset
            if self._method == "ecc_patch":
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                try:
                    cc, ecc_mat = cv2.findTransformECC(
                        p_ref,
                        p_src,
                        warp_matrix,
                        cv2.MOTION_TRANSLATION,
                        criteria,
                    )
                    if cc >= self._min_quality:
                        warp_arr = np.asarray(ecc_mat, dtype=np.float32)
                        dx = float(warp_arr[0, 2])
                        dy = float(warp_arr[1, 2])
                        refined_ok = True
                    else:
                        refined_ok = False
                except cv2.error:
                    refined_ok = False

            # 5. Fallback refinement: Phase Correlation
            if not refined_ok and self._method in ("ecc_patch", "phase_correlation"):
                try:
                    (pc_shift, resp) = cv2.phaseCorrelate(p_ref, p_src)
                    if resp >= self._min_quality:
                        dx = float(pc_shift[0])
                        dy = float(pc_shift[1])
                        refined_ok = True
                    else:
                        statuses.append("low_quality")
                        continue
                except cv2.error:
                    statuses.append("ecc_failed")
                    continue

            if not refined_ok:
                statuses.append("ecc_failed")
                continue

            # 6. Maximum shift rejection check
            shift_mag = np.sqrt(dx**2 + dy**2)
            if shift_mag > self._max_shift_px:
                statuses.append("shift_too_large")
                continue

            # Accepted refinement!
            # Refined reference coordinate is xr - dx, yr - dy
            refined_x = xr - dx
            refined_y = yr - dy

            shifts[i] = [-dx, -dy]
            refined_ref_pts[i] = [refined_x, refined_y]
            statuses.append("accepted")
            accepted_count += 1

        # 7. Post-refinement refitting & post-residuals
        refit_transform: TransformEstimate | None = initial_transform
        accepted_mask = np.array([s == "accepted" for s in statuses], dtype=bool)

        if accepted_count >= 3:
            accepted_match_set = MatchSet(
                source_points=src_pts[accepted_mask],
                reference_points=refined_ref_pts[accepted_mask],
                scores=scores[accepted_mask],
                inliers=inliers[accepted_mask],
            )
            geom_res = verify_matches(
                accepted_match_set,
                config=geometry_config,
                model=initial_transform.model_type if initial_transform else "affine",
            )
            if geom_res.success and geom_res.transform is not None:
                refit_transform = geom_res.transform

        if refit_transform is not None:
            post_residuals = compute_reprojection_residuals(
                refit_transform.get_matrix_array(), src_pts, refined_ref_pts
            )
        else:
            post_residuals = pre_residuals.copy()

        refined_match_set = MatchSet(
            source_points=src_pts,
            reference_points=refined_ref_pts,
            scores=scores,
            inliers=inliers,
            refine_status=statuses,
        )

        prov["accepted_count"] = accepted_count
        prov["acceptance_ratio"] = accepted_count / n_pts if n_pts > 0 else 0.0

        return SubpixelRefinementResult(
            refined_matches=refined_match_set,
            refit_transform=refit_transform,
            refine_status=statuses,
            shifts_px=shifts,
            pre_residuals_px=pre_residuals,
            post_residuals_px=post_residuals,
            accepted_count=accepted_count,
            provenance=prov,
        )


def refine_subpixel_matches(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    matches: MatchSet,
    initial_transform: TransformEstimate | None = None,
    config: SubpixelConfig | None = None,
    source_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    geometry_config: GeometryConfig | None = None,
    method: Literal["ecc_patch", "phase_correlation", "none"] = "ecc_patch",
    patch_radius: int = 15,
    max_iterations: int = 50,
    epsilon: float = 0.0001,
    max_shift_px: float = 3.0,
    min_quality: float = 0.2,
) -> SubpixelRefinementResult:
    """Convenience function for patch-based sub-pixel correspondence refinement."""
    refiner = SubpixelRefiner(
        config=config,
        method=method,
        patch_radius=patch_radius,
        max_iterations=max_iterations,
        epsilon=epsilon,
        max_shift_px=max_shift_px,
        min_quality=min_quality,
    )
    return refiner.refine(
        source_image=source_image,
        reference_image=reference_image,
        matches=matches,
        initial_transform=initial_transform,
        source_mask=source_mask,
        reference_mask=reference_mask,
        geometry_config=geometry_config,
    )
