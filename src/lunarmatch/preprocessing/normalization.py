"""Geometry-preserving radiometric normalization, CLAHE, and denoising functions."""
from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np


def percentile_normalize(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize image intensity using valid-mask-aware percentile clipping.

    Args:
        image: 2D numeric numpy array.
        mask: Optional 2D boolean array (True=valid).
        percentile_clip: Tuple of (low_percentile, high_percentile) in range [0, 100].

    Returns:
        Tuple of (normalized_float32_array, provenance_dict).
        Output array values are scaled to [0.0, 1.0] on valid pixels; non-valid pixels are 0.0.
    """
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]

    if mask is None:
        valid_mask = np.isfinite(arr)
    else:
        valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)

    valid_pixels = arr[valid_mask]

    low_p, high_p = percentile_clip
    if valid_pixels.size == 0:
        p_min, p_max = 0.0, 1.0
        normalized = np.zeros((h, w), dtype=np.float32)
    else:
        p_min = float(np.percentile(valid_pixels, low_p))
        p_max = float(np.percentile(valid_pixels, high_p))

        if np.isclose(p_min, p_max):
            normalized = np.zeros((h, w), dtype=np.float32)
        else:
            clipped = np.clip(arr, p_min, p_max)
            normalized = (clipped - p_min) / (p_max - p_min)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

    normalized[~valid_mask] = 0.0
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    prov = {
        "percentile_clip": list(percentile_clip),
        "percentile_min_val": p_min,
        "percentile_max_val": p_max,
        "valid_pixel_count": int(valid_pixels.size),
        "total_pixel_count": int(h * w),
    }
    return cast(np.ndarray, normalized), prov


def apply_clahe(
    image_float01: np.ndarray,
    mask: np.ndarray | None = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image_float01: 2D float32 array in range [0.0, 1.0].
        mask: Optional 2D boolean valid mask.
        clip_limit: CLAHE clip limit parameter.
        tile_grid_size: Tile grid resolution for histogram equalisation.

    Returns:
        2D float32 array in range [0.0, 1.0].
    """
    img_uint8 = np.clip(image_float01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    equalized_uint8 = clahe.apply(img_uint8)
    res = (equalized_uint8.astype(np.float32) / 255.0)

    if mask is not None:
        res[~mask] = 0.0

    return cast(np.ndarray, np.clip(res, 0.0, 1.0).astype(np.float32))


def apply_denoise(
    image_float01: np.ndarray,
    sigma: float = 0.8,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply optional Gaussian denoising.

    Args:
        image_float01: 2D float32 array in range [0.0, 1.0].
        sigma: Gaussian kernel standard deviation (0 to disable).
        mask: Optional 2D boolean valid mask.

    Returns:
        Denoised 2D float32 array in range [0.0, 1.0].
    """
    if sigma <= 0.0:
        return image_float01.copy()

    ksize = int(2 * round(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    blurred = cv2.GaussianBlur(image_float01, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))
    if mask is not None:
        blurred[~mask] = 0.0

    return cast(np.ndarray, np.clip(blurred, 0.0, 1.0).astype(np.float32))
