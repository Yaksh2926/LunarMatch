"""Modality-invariant structural representations for multi-modal lunar image correspondence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from lunarmatch.preprocessing.normalization import apply_clahe, percentile_normalize


@dataclass
class PreprocessingResult:
    """Container for processed image representation, valid mask, and parameter provenance."""

    processed_image: np.ndarray  # 2D float32 array in [0.0, 1.0]
    valid_mask: np.ndarray  # 2D bool array
    representation: str
    provenance: dict[str, Any] = field(default_factory=dict)


def compute_raw_contrast(
    image_norm: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Raw percentile-normalized intensity representation."""
    res = image_norm.copy()
    res[~mask] = 0.0
    return res, {"representation": "raw_contrast"}


def compute_clahe_representation(
    image_norm: np.ndarray, mask: np.ndarray, clip_limit: float = 2.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """CLAHE contrast-enhanced representation."""
    equalized = apply_clahe(image_norm, mask=mask, clip_limit=clip_limit)
    return equalized, {"representation": "clahe", "clahe_clip_limit": clip_limit}


def compute_gradient_representation(
    image_norm: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gradient magnitude structural representation using Sobel operators."""
    sobel_x = cv2.Sobel(image_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_norm, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_norm, prov = percentile_normalize(grad_mag, mask=mask, percentile_clip=(0.5, 99.5))
    prov["representation"] = "gradient"
    return grad_norm, prov


def compute_edge_representation(
    image_norm: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Edge-based structural representation combining Canny and Sobel responses."""
    img_uint8 = np.clip(image_norm * 255.0, 0, 255).astype(np.uint8)
    canny = cv2.Canny(img_uint8, threshold1=30, threshold2=100)
    canny_float = (canny.astype(np.float32) / 255.0)

    grad_mag, _ = compute_gradient_representation(image_norm, mask=mask)
    edge_combined = np.maximum(canny_float, grad_mag)
    edge_combined[~mask] = 0.0
    return np.clip(edge_combined, 0.0, 1.0).astype(np.float32), {"representation": "edge"}


def compute_phase_representation(
    image_norm: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Phase-friendly high-pass / Difference-of-Gaussians (DoG) structural representation."""
    blur_small = cv2.GaussianBlur(image_norm, (3, 3), sigmaX=0.5, sigmaY=0.5)
    blur_large = cv2.GaussianBlur(image_norm, (9, 9), sigmaX=2.0, sigmaY=2.0)

    dog = blur_small - blur_large
    dog_norm, prov = percentile_normalize(dog, mask=mask, percentile_clip=(1.0, 99.0))
    prov["representation"] = "phase"
    return dog_norm, prov
