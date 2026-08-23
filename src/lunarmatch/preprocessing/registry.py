"""Preprocessing registry, configuration-driven orchestrator, and quick-look visualization utility."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from lunarmatch.models.config import PreprocessConfig
from lunarmatch.preprocessing.normalization import apply_denoise, percentile_normalize
from lunarmatch.preprocessing.representations import (
    PreprocessingResult,
    compute_clahe_representation,
    compute_edge_representation,
    compute_gradient_representation,
    compute_phase_representation,
    compute_raw_contrast,
)

REPRESENTATIONS: dict[
    str, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, Any]]]
] = {
    "gradient": compute_gradient_representation,
    "clahe": lambda img, m: compute_clahe_representation(img, m, clip_limit=2.0),
    "edge": compute_edge_representation,
    "phase": compute_phase_representation,
    "raw_contrast": compute_raw_contrast,
}


def preprocess_raster(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    config: PreprocessConfig | None = None,
) -> PreprocessingResult:
    """Preprocess raster image according to configuration settings.

    Args:
        image: Input 2D numeric image array.
        mask: Optional 2D boolean valid pixel mask (True=valid).
        config: Optional PreprocessConfig instance.

    Returns:
        PreprocessingResult containing processed float32 image in [0.0, 1.0], valid_mask, and provenance.
    """
    cfg = config if config is not None else PreprocessConfig()

    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]

    if mask is None:
        valid_mask = np.isfinite(arr)
    else:
        valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)

    # Stage 1: Percentile normalization to [0.0, 1.0]
    image_norm, norm_prov = percentile_normalize(
        arr, mask=valid_mask, percentile_clip=cfg.percentile_clip
    )

    # Stage 2: Optional Gaussian denoise
    if cfg.denoise_sigma > 0.0:
        image_norm = apply_denoise(image_norm, sigma=cfg.denoise_sigma, mask=valid_mask)

    # Stage 3: Select modality representation from registry
    rep_name = cfg.representation
    if rep_name not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation '{rep_name}'. Available representations: {sorted(REPRESENTATIONS.keys())}"
        )

    if rep_name == "clahe":
        processed, rep_prov = compute_clahe_representation(image_norm, valid_mask, clip_limit=cfg.clahe_clip_limit)
    else:
        rep_func = REPRESENTATIONS[rep_name]
        processed, rep_prov = rep_func(image_norm, valid_mask)

    # Guarantee geometry preservation and finiteness on valid pixels
    assert processed.shape == (h, w), f"Shape mismatch: {processed.shape} vs ({h}, {w})"
    processed[~valid_mask] = 0.0
    processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=0.0)
    processed = np.clip(processed, 0.0, 1.0).astype(np.float32)

    combined_provenance = {
        "representation": rep_name,
        "normalization": norm_prov,
        "representation_provenance": rep_prov,
        "denoise_sigma": cfg.denoise_sigma,
        "clahe_clip_limit": cfg.clahe_clip_limit,
        "percentile_clip": list(cfg.percentile_clip),
        "input_shape": [h, w],
    }

    return PreprocessingResult(
        processed_image=processed,
        valid_mask=valid_mask,
        representation=rep_name,
        provenance=combined_provenance,
    )


def create_preprocessing_quicklook(
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generate side-by-side uint8 RGB quick-look image comparing original vs preprocessed.

    Returns:
        3D uint8 array (Height x (Width * 2) x 3) for visual report generation.
    """
    orig_norm, _ = percentile_normalize(original, mask=mask)
    orig_uint8 = (orig_norm * 255.0).astype(np.uint8)
    proc_uint8 = (np.clip(processed, 0.0, 1.0) * 255.0).astype(np.uint8)

    orig_rgb = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
    proc_rgb = cv2.cvtColor(proc_uint8, cv2.COLOR_GRAY2RGB)

    if mask is not None:
        invalid = ~mask
        orig_rgb[invalid] = [180, 40, 40]

    quicklook = np.hstack([orig_rgb, proc_rgb])
    return quicklook
