"""Geometry-preserving preprocessing and modality-invariant structural representations."""
from lunarmatch.preprocessing.normalization import (
    apply_clahe,
    apply_denoise,
    percentile_normalize,
)
from lunarmatch.preprocessing.registry import (
    REPRESENTATIONS,
    create_preprocessing_quicklook,
    preprocess_raster,
)
from lunarmatch.preprocessing.representations import (
    PreprocessingResult,
    compute_clahe_representation,
    compute_edge_representation,
    compute_gradient_representation,
    compute_phase_representation,
    compute_raw_contrast,
)

__all__ = [
    "REPRESENTATIONS",
    "PreprocessingResult",
    "apply_clahe",
    "apply_denoise",
    "compute_clahe_representation",
    "compute_edge_representation",
    "compute_gradient_representation",
    "compute_phase_representation",
    "compute_raw_contrast",
    "create_preprocessing_quicklook",
    "percentile_normalize",
    "preprocess_raster",
]
