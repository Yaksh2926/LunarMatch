"""Unit tests for metrics, independent control-point evaluation, and bootstrap confidence intervals."""
from pathlib import Path

import numpy as np
import pytest

from lunarmatch.evaluation.bootstrap import compute_bootstrap_ci
from lunarmatch.evaluation.control_points import evaluate_control_points, load_control_points
from lunarmatch.evaluation.evaluator import evaluate_registration
from lunarmatch.evaluation.models import ControlPoint
from lunarmatch.models.domain import RegistrationMetrics, RunManifest, TransformEstimate


def test_load_control_points_valid_csv(tmp_path: Path):
    """Test loading valid control points from CSV."""
    csv_file = tmp_path / "control_points.csv"
    csv_file.write_text(
        "source_x,source_y,reference_x,reference_y,point_id\n"
        "10.0,20.0,15.0,25.0,cp1\n"
        "30.0,40.0,35.0,45.0,cp2\n"
    )

    cpts = load_control_points(csv_file)
    assert len(cpts) == 2
    assert cpts[0] == ControlPoint(source_x=10.0, source_y=20.0, reference_x=15.0, reference_y=25.0, point_id="cp1")
    assert cpts[1].point_id == "cp2"


def test_load_control_points_invalid_and_out_of_bounds(tmp_path: Path):
    """Test error handling for non-finite and out-of-bounds control point coordinates."""
    # 1. Non-finite coordinate
    nan_csv = tmp_path / "nan.csv"
    nan_csv.write_text("src_x,src_y,ref_x,ref_y\n10.0,NaN,15.0,25.0\n")
    with pytest.raises(ValueError, match="Non-finite"):
        load_control_points(nan_csv)

    # 2. Out of bounds
    oob_csv = tmp_path / "oob.csv"
    oob_csv.write_text("src_x,src_y,ref_x,ref_y\n150.0,20.0,15.0,25.0\n")
    with pytest.raises(ValueError, match="out of bounds"):
        load_control_points(oob_csv, source_bounds=(100.0, 100.0))


def test_evaluate_control_points_known_offset():
    """Test control point evaluation with known translation transform."""
    cpts = [
        ControlPoint(source_x=10.0, source_y=10.0, reference_x=13.0, reference_y=14.0),  # Error = sqrt(3^2 + 4^2) = 5.0
        ControlPoint(source_x=20.0, source_y=20.0, reference_x=20.0, reference_y=20.0),  # Error = 0.0
    ]

    # Identity transform (predicts (10, 10) and (20, 20))
    M = np.eye(3, dtype=np.float64)
    trans = TransformEstimate.from_matrix("similarity", M, inlier_count=10, inlier_ratio=1.0, rmse_px=0.0)

    metrics = evaluate_control_points(cpts, trans)

    assert metrics.count == 2
    assert pytest.approx(metrics.residuals_px[0], abs=1e-5) == 5.0
    assert pytest.approx(metrics.residuals_px[1], abs=1e-5) == 0.0
    assert pytest.approx(metrics.cp_max_px, abs=1e-5) == 5.0
    assert pytest.approx(metrics.cp_median_px, abs=1e-5) == 2.5
    # RMSE = sqrt((25 + 0) / 2) = sqrt(12.5) ≈ 3.53553
    assert pytest.approx(metrics.cp_rmse_px, abs=1e-4) == np.sqrt(12.5)


def test_control_point_and_self_consistency_separation(tmp_path: Path):
    """Test explicit separation of self-consistency metrics and ground-truth control points."""
    csv_file = tmp_path / "cpts.csv"
    csv_file.write_text("src_x,src_y,ref_x,ref_y\n" + "\n".join([f"{i}.0,{i}.0,{i}.2,{i}.3" for i in range(10)]))

    self_metrics = RegistrationMetrics(
        rmse_px=0.10,
        median_residual_px=0.08,
        inlier_count=50,
        total_matches=60,
        inlier_ratio=0.833,
        occupied_grid_fraction=0.8,
        convex_hull_coverage_fraction=0.85,
        runtime_seconds=0.2,
    )
    transform = TransformEstimate.from_matrix("affine", np.eye(3), inlier_count=50, inlier_ratio=0.833, rmse_px=0.10)

    manifest = RunManifest(
        pair_id="pair_001",
        source_path="src.tif",
        reference_path="ref.tif",
        config_hash="hash123",
        timestamp_utc="2026-08-23T11:00:00Z",
        package_version="0.1.0",
        seed=42,
        metrics=self_metrics,
        transform=transform,
        quality_gate_passed=True,
    )

    summary = evaluate_registration(manifest, control_points_path=csv_file)

    assert summary.self_consistency_metrics.rmse_px == 0.10
    assert summary.control_point_metrics is not None
    assert summary.control_point_metrics.evaluation_type == "independent_control_points"
    assert summary.subpixel_accuracy_proven is True


def test_bootstrap_confidence_intervals():
    """Test bootstrap 95% confidence interval estimation on synthetic metric values."""
    # Generate 50 sample RMSE values centered around 0.50 px
    rng = np.random.default_rng(42)
    rmse_samples = rng.normal(loc=0.50, scale=0.05, size=50)

    mean_val, ci_lower, ci_upper = compute_bootstrap_ci(rmse_samples, confidence_level=0.95, n_resamples=1000, seed=42)

    assert pytest.approx(mean_val, abs=0.02) == 0.50
    assert ci_lower < mean_val < ci_upper
    assert 0.45 <= ci_lower <= 0.52
    assert 0.48 <= ci_upper <= 0.55
