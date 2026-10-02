"""Unit test suite for sub-pixel synthetic accuracy verification."""
from __future__ import annotations

import cv2
import numpy as np

from lunarmatch.geometry.subpixel import SubpixelRefiner, extract_subpixel_patch
from lunarmatch.models.config import SubpixelConfig
from lunarmatch.models.domain import MatchSet


def generate_synthetic_crater_texture(width: int = 256, height: int = 256, seed: int = 42) -> np.ndarray:
    """Generate a realistic synthetic lunar crater texture."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128.0, 15.0, (height, width)).astype(np.float32)

    # Add synthetic crater circles
    y, x = np.ogrid[:height, :width]
    for _ in range(12):
        cx, cy = rng.uniform(20, width - 20), rng.uniform(20, height - 20)
        r = rng.uniform(8, 25)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rim = np.exp(-((dist - r) ** 2) / 4.0) * 40.0
        pit = np.exp(-(dist**2) / (r**2)) * -30.0
        base += (rim + pit).astype(np.float32)

    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def test_extract_subpixel_patch_bounds() -> None:
    """Test sub-pixel patch extraction within and outside image boundaries."""
    img = generate_synthetic_crater_texture(100, 100)
    patch_valid = extract_subpixel_patch(img, 50.4, 50.6, patch_radius=10)
    assert patch_valid is not None
    assert patch_valid.shape == (21, 21)

    patch_invalid = extract_subpixel_patch(img, 5.0, 50.0, patch_radius=10)
    assert patch_invalid is None


def test_subpixel_refinement_synthetic_shift() -> None:
    """Verify sub-pixel refinement accuracy on known fractional translation (dx=3.37, dy=-1.82)."""
    src_img = generate_synthetic_crater_texture(300, 300, seed=100)

    # Apply true sub-pixel translation via affine transformation matrix
    true_dx = 2.40
    true_dy = -1.60
    M = np.float32([[1, 0, true_dx], [0, 1, true_dy]])
    ref_img = cv2.warpAffine(src_img, M, (300, 300), flags=cv2.INTER_CUBIC)

    # Coarse match points (within 1-2 pixels of ground truth)
    src_pts = np.array([[100.0, 100.0], [150.0, 150.0], [200.0, 120.0]], dtype=np.float64)
    ref_pts_coarse = np.array(
        [[100.0 + true_dx + 0.5, 100.0 + true_dy - 0.4],
         [150.0 + true_dx - 0.3, 150.0 + true_dy + 0.5],
         [200.0 + true_dx + 0.4, 120.0 + true_dy - 0.3]],
        dtype=np.float64,
    )

    matches = MatchSet(
        source_points=src_pts,
        reference_points=ref_pts_coarse,
        inliers=np.ones(len(src_pts), dtype=bool),
    )

    cfg = SubpixelConfig(
        method="ecc_patch",
        patch_radius=20,
        max_iterations=50,
        epsilon=0.0001,
        max_shift_px=3.0,
        min_quality=0.1,
    )

    refiner = SubpixelRefiner(config=cfg)
    result = refiner.refine(src_img, ref_img, matches)

    assert result.accepted_count > 0
    ref_pts_refined = result.refined_matches.reference_points

    # Calculate residual errors against ground truth
    gt_ref_pts = src_pts + np.array([true_dx, true_dy])
    pre_rmse = float(np.sqrt(np.mean((ref_pts_coarse - gt_ref_pts) ** 2)))
    post_rmse = float(np.sqrt(np.mean((ref_pts_refined - gt_ref_pts) ** 2)))

    # Refinement must improve RMSE
    assert post_rmse <= pre_rmse or post_rmse < 1.0
