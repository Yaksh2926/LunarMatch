"""Unit tests for patch-based sub-pixel correspondence refinement."""
import cv2
import numpy as np

from lunarmatch.geometry import (
    SubpixelRefiner,
    refine_subpixel_matches,
)
from lunarmatch.models.config import SubpixelConfig
from lunarmatch.models.domain import MatchSet, TransformEstimate


def create_synthetic_texture(width: int = 256, height: int = 256, seed: int = 42) -> np.ndarray:
    """Create synthetic textured image with Gaussian blur for smooth gradients."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.0, 255.0, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (15, 15), sigmaX=3.0)
    return blurred


def test_synthetic_fractional_translation_recovery():
    """Test sub-pixel refinement on synthetic image with known fractional translation."""
    ref_img = create_synthetic_texture(256, 256, seed=123)

    # Known fractional translation shift
    gt_dx = 0.35
    gt_dy = -0.65

    # Generate source image by warping reference image with exact shift (dx, dy)
    trans_mat = np.array([[1.0, 0.0, gt_dx], [0.0, 1.0, gt_dy]], dtype=np.float32)
    src_img = cv2.warpAffine(ref_img, trans_mat, (256, 256), flags=cv2.INTER_CUBIC)

    # Add Gaussian noise and contrast shift
    rng = np.random.default_rng(99)
    src_img = np.clip(src_img * 0.95 + rng.normal(0.0, 2.0, size=src_img.shape), 0, 255).astype(np.float32)

    # Initial matches at integer centers (initial error ~ 0.74 px)
    pts = np.array(
        [
            [60.0, 60.0],
            [120.0, 80.0],
            [80.0, 140.0],
            [180.0, 180.0],
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )

    matches = MatchSet(source_points=pts, reference_points=pts.copy())

    init_transform = TransformEstimate.from_matrix(
        model_type="similarity",
        matrix=np.eye(3, dtype=np.float64),
        inlier_count=5,
        inlier_ratio=1.0,
        rmse_px=0.74,
    )

    res = refine_subpixel_matches(
        source_image=src_img,
        reference_image=ref_img,
        matches=matches,
        initial_transform=init_transform,
        patch_radius=15,
        max_shift_px=3.0,
    )

    assert res.accepted_count > 0
    assert all(status == "accepted" for status in res.refine_status)

    # Verify estimated shifts match ground truth within 0.10 px
    for shift in res.shifts_px:
        np.testing.assert_allclose(shift[0], -gt_dx, atol=0.10)
        np.testing.assert_allclose(shift[1], -gt_dy, atol=0.10)

    # Initial unrefined localization error is magnitude of ground-truth shift (~0.74 px)
    gt_error_px = float(np.hypot(gt_dx, gt_dy))
    post_median_err = float(np.median(res.post_residuals_px))
    assert post_median_err < 0.10
    assert post_median_err < gt_error_px


def test_border_rejection_safety():
    """Test that points near image edges (< patch_radius) are safely rejected."""
    src_img = create_synthetic_texture(100, 100)
    ref_img = src_img.copy()

    # Point at (5.0, 5.0) with patch_radius=15 is out of bounds
    src_pts = np.array([[5.0, 5.0], [50.0, 50.0]], dtype=np.float64)
    matches = MatchSet(source_points=src_pts, reference_points=src_pts.copy())

    res = refine_subpixel_matches(
        source_image=src_img,
        reference_image=ref_img,
        matches=matches,
        patch_radius=15,
    )

    assert res.refine_status[0] == "border_rejected"
    assert res.refine_status[1] == "accepted"
    assert res.accepted_count == 1


def test_mask_rejection_safety():
    """Test that patches containing nodata/zero mask values are rejected."""
    src_img = create_synthetic_texture(100, 100)
    ref_img = src_img.copy()

    mask = np.ones((100, 100), dtype=bool)
    mask[40:60, 40:60] = False  # Mask out center

    src_pts = np.array([[50.0, 50.0]], dtype=np.float64)
    matches = MatchSet(source_points=src_pts, reference_points=src_pts.copy())

    res = refine_subpixel_matches(
        source_image=src_img,
        reference_image=ref_img,
        matches=matches,
        source_mask=mask,
        patch_radius=15,
    )

    assert res.refine_status[0] == "mask_rejected"
    assert res.accepted_count == 0


def test_max_shift_rejection_safety():
    """Test that shift exceeding max_shift_px is rejected."""
    src_img = create_synthetic_texture(200, 200, seed=55)

    # Shift by 4.0 pixels
    gt_shift_x = 4.0
    trans_mat = np.array([[1.0, 0.0, gt_shift_x], [0.0, 1.0, 0.0]], dtype=np.float32)
    ref_img = cv2.warpAffine(src_img, trans_mat, (200, 200), flags=cv2.INTER_CUBIC)

    src_pts = np.array([[100.0, 100.0]], dtype=np.float64)
    matches = MatchSet(source_points=src_pts, reference_points=src_pts.copy())

    res = refine_subpixel_matches(
        source_image=src_img,
        reference_image=ref_img,
        matches=matches,
        patch_radius=25,
        max_shift_px=2.0,  # Max shift set to 2.0 px (smaller than 4.0 px shift)
    )

    assert res.refine_status[0] == "shift_too_large"
    assert res.accepted_count == 0


def test_subpixel_refiner_class_config():
    """Test SubpixelRefiner class initialization with SubpixelConfig."""
    cfg = SubpixelConfig(method="ecc_patch", patch_radius=10, max_shift_px=2.5)
    refiner = SubpixelRefiner(config=cfg)

    src_img = create_synthetic_texture(100, 100)
    ref_img = src_img.copy()

    src_pts = np.array([[50.0, 50.0]], dtype=np.float64)
    matches = MatchSet(source_points=src_pts, reference_points=src_pts.copy())

    res = refiner.refine(src_img, ref_img, matches)
    assert res.provenance["patch_radius"] == 10
    assert res.provenance["max_shift_px"] == 2.5
    assert res.accepted_count == 1
