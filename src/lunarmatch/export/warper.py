"""Source raster warping into reference image grid."""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from lunarmatch.models.domain import TransformEstimate

INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def warp_source_to_reference(
    source_image: np.ndarray,
    transform: TransformEstimate | np.ndarray,
    reference_shape: tuple[int, int],
    interpolation: Literal["nearest", "linear", "bilinear", "cubic", "lanczos"] = "bilinear",
    nodata_value: float = 0.0,
) -> np.ndarray:
    """Warp source image array into reference coordinate grid using estimated transformation.

    Args:
        source_image: 2D or 3D numeric source image array.
        transform: TransformEstimate instance or 3x3 transformation matrix mapping source to reference.
        reference_shape: (Height, Width) of target reference grid.
        interpolation: Interpolation mode ('nearest', 'linear', 'bilinear', 'cubic', 'lanczos').
        nodata_value: Fill value for unmapped / border pixels.

    Returns:
        Warped image array matching reference_shape (or (H_ref, W_ref, C) if multi-channel).
    """
    src_arr = np.asarray(source_image)
    h_ref, w_ref = reference_shape

    if h_ref <= 0 or w_ref <= 0:
        raise ValueError(f"Reference shape dimensions must be positive, got ({h_ref}, {w_ref})")

    if isinstance(transform, TransformEstimate):
        M_mat = transform.get_matrix_array()
        model_type = transform.model_type
    else:
        M_mat = np.asarray(transform, dtype=np.float64)
        if M_mat.shape != (3, 3):
            raise ValueError(f"Transform matrix must be 3x3, got shape {M_mat.shape}")
        model_type = "homography"

    # Singularity check
    det = np.linalg.det(M_mat)
    if np.abs(det) < 1e-12:
        raise ValueError("Cannot warp source image: transform matrix is singular")

    if interpolation not in INTERPOLATION_MAP:
        raise ValueError(
            f"Unsupported interpolation '{interpolation}'. Available: {sorted(INTERPOLATION_MAP.keys())}"
        )
    interp_flag = INTERPOLATION_MAP[interpolation]

    # Perform warping using warpAffine or warpPerspective
    if model_type in ("similarity", "affine") or np.allclose(M_mat[2, :], [0, 0, 1]):
        M_2x3 = M_mat[:2, :].astype(np.float32)
        warped = cv2.warpAffine(
            src_arr,
            M_2x3,
            (w_ref, h_ref),
            flags=interp_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(nodata_value),
        )
    else:
        M_3x3 = M_mat.astype(np.float32)
        warped = cv2.warpPerspective(
            src_arr,
            M_3x3,
            (w_ref, h_ref),
            flags=interp_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(nodata_value),
        )

    return warped
