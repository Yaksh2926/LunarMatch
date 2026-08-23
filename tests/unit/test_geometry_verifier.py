"""Unit tests for robust geometric verification, fitting, and plausibility gates."""
import numpy as np

from lunarmatch.geometry import GeometricVerifier, verify_matches
from lunarmatch.models.config import GeometryConfig
from lunarmatch.models.domain import MatchSet


def generate_synthetic_points(
    n_points: int = 30, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D point cloud in [0, 1000] space."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(50.0, 950.0, size=n_points)
    y = rng.uniform(50.0, 950.0, size=n_points)
    pts = np.column_stack([x, y])
    return pts, pts.copy()


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 3x3 matrix to Nx2 float64 points."""
    homo = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    proj = (matrix @ homo.T).T
    return proj[:, :2] / proj[:, 2:3]


def test_known_similarity_recovery():
    """Test exact recovery of similarity transform (rotation, scale, translation)."""
    src_pts, _ = generate_synthetic_points(30, seed=10)

    # 2D similarity: rotation 30 deg, scale 1.25, translation (50, -30)
    theta = np.deg2rad(30.0)
    scale = 1.25
    tx, ty = 50.0, -30.0

    a = scale * np.cos(theta)
    b = scale * np.sin(theta)

    gt_matrix = np.array(
        [
            [a, -b, tx],
            [b, a, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ref_pts = apply_transform(gt_matrix, src_pts)

    res = verify_matches((src_pts, ref_pts), model="similarity", robust_method="ransac")

    assert res.success is True
    assert res.failure_reason is None
    assert res.transform is not None
    assert res.transform.inlier_count == 30
    assert res.transform.rmse_px is not None
    assert res.transform.rmse_px < 1e-4

    # Check matrix recovery
    est_mat = res.transform.get_matrix_array()
    np.testing.assert_allclose(est_mat, gt_matrix, atol=1e-4)


def test_known_affine_recovery():
    """Test exact recovery of affine transform (rotation, non-uniform scale, shear, translation)."""
    src_pts, _ = generate_synthetic_points(40, seed=20)

    gt_matrix = np.array(
        [
            [1.2, 0.15, 120.0],
            [-0.1, 0.9, -45.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ref_pts = apply_transform(gt_matrix, src_pts)

    res = verify_matches((src_pts, ref_pts), model="affine")

    assert res.success is True
    assert res.transform is not None
    assert res.transform.inlier_count == 40
    assert res.transform.rmse_px is not None
    assert res.transform.rmse_px < 1e-4

    # Direction check: est_mat @ [src, 1] ~ ref
    est_mat = res.transform.get_matrix_array()
    reprojected = apply_transform(est_mat, src_pts)
    np.testing.assert_allclose(reprojected, ref_pts, atol=1e-3)


def test_known_homography_recovery():
    """Test exact recovery of homography transform."""
    src_pts, _ = generate_synthetic_points(50, seed=30)

    gt_matrix = np.array(
        [
            [1.05, 0.02, 30.0],
            [-0.01, 0.98, -15.0],
            [0.0001, -0.00005, 1.0],
        ],
        dtype=np.float64,
    )

    ref_pts = apply_transform(gt_matrix, src_pts)

    res = verify_matches((src_pts, ref_pts), model="homography")

    assert res.success is True
    assert res.transform is not None
    assert res.transform.inlier_count == 50
    assert res.transform.rmse_px is not None
    assert res.transform.rmse_px < 1e-3


def test_outlier_heavy_matches():
    """Test robust fitting with 50% inliers and 50% random noise outliers."""
    n_inliers = 30
    n_outliers = 30

    src_inliers, _ = generate_synthetic_points(n_inliers, seed=100)
    gt_matrix = np.array(
        [
            [1.1, -0.05, 25.0],
            [0.05, 1.1, -10.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ref_inliers = apply_transform(gt_matrix, src_inliers)

    rng = np.random.default_rng(200)
    src_outliers = rng.uniform(0.0, 1000.0, size=(n_outliers, 2))
    ref_outliers = rng.uniform(0.0, 1000.0, size=(n_outliers, 2))

    src_all = np.vstack([src_inliers, src_outliers])
    ref_all = np.vstack([ref_inliers, ref_outliers])

    match_set = MatchSet(source_points=src_all, reference_points=ref_all)

    res = verify_matches(match_set, model="affine", reprojection_threshold_px=3.0)

    assert res.success is True
    assert res.transform is not None
    assert res.transform.inlier_count >= n_inliers
    assert res.transform.rmse_px is not None
    assert res.transform.rmse_px < 1.0

    # Verify that inliers are correctly identified for first n_inliers
    assert np.all(res.inlier_mask[:n_inliers])


def test_insufficient_points_failure():
    """Test that fewer than minimum required points returns insufficient_points failure."""
    src_pts = np.array([[10, 10], [20, 20]], dtype=np.float64)
    ref_pts = np.array([[15, 15], [25, 25]], dtype=np.float64)

    # Affine requires at least 3 points
    res_affine = verify_matches((src_pts, ref_pts), model="affine")
    assert res_affine.success is False
    assert res_affine.failure_reason == "insufficient_points"
    assert res_affine.transform is None

    # Homography requires at least 4 points
    res_homo = verify_matches((src_pts, ref_pts), model="homography")
    assert res_homo.success is False
    assert res_homo.failure_reason == "insufficient_points"


def test_collinear_points_failure():
    """Test that collinear points return degenerate_points failure."""
    # Points lying on y = 2*x + 1
    x = np.linspace(0.0, 100.0, 10)
    y = 2.0 * x + 1.0
    src_pts = np.column_stack([x, y])
    ref_pts = src_pts + 5.0

    res = verify_matches((src_pts, ref_pts), model="affine")
    assert res.success is False
    assert res.failure_reason == "degenerate_points"


def test_reflection_rejection():
    """Test that mirror reflection (orientation flip, det < 0) is rejected."""
    src_pts, _ = generate_synthetic_points(20, seed=50)

    # Reflection matrix: flip X axis (det = -1)
    refl_matrix = np.array(
        [
            [-1.0, 0.0, 500.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ref_pts = apply_transform(refl_matrix, src_pts)

    res = verify_matches((src_pts, ref_pts), model="affine")
    assert res.success is False
    assert res.failure_reason == "reflection_detected"


def test_scale_out_of_bounds_rejection():
    """Test that extreme scale expansion is rejected."""
    src_pts, _ = generate_synthetic_points(20, seed=60)

    # Scale 50x
    scale_matrix = np.array(
        [
            [50.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ref_pts = apply_transform(scale_matrix, src_pts)

    cfg = GeometryConfig(model="affine", allowed_scale=(0.1, 10.0))
    res = verify_matches((src_pts, ref_pts), config=cfg)

    assert res.success is False
    assert res.failure_reason == "scale_out_of_bounds"


def test_geometry_verifier_class():
    """Test GeometricVerifier class instantiation and config integration."""
    cfg = GeometryConfig(model="similarity", reprojection_threshold_px=2.0)
    verifier = GeometricVerifier(config=cfg)

    src_pts, _ = generate_synthetic_points(25, seed=70)
    gt_mat = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ref_pts = apply_transform(gt_mat, src_pts)

    res = verifier.verify((src_pts, ref_pts))
    assert res.success is True
    assert res.transform is not None
    assert res.transform.model_type == "similarity"
