"""Unit tests for LunarMatch domain models and validations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunarmatch.models import (
    ImagePair,
    KeypointSet,
    MatchSet,
    RasterMetadata,
    RegistrationMetrics,
    RunManifest,
    TransformEstimate,
)


def test_raster_metadata_serialization() -> None:
    """Test RasterMetadata creation and serialization round-trip."""
    meta = RasterMetadata(
        path=Path("data/ohrc.tif"),
        width=1024,
        height=1024,
        bands=1,
        dtype="float32",
        pixel_scale_x=0.25,
        pixel_scale_y=0.25,
        sensor="OHRC",
        sun_azimuth_deg=45.0,
        sun_elevation_deg=30.0,
    )
    d = meta.to_dict()
    assert d["width"] == 1024
    assert d["sensor"] == "OHRC"

    reconstructed = RasterMetadata.from_dict(d)
    assert reconstructed.path == meta.path
    assert reconstructed.pixel_scale_x == 0.25


def test_image_pair_serialization() -> None:
    """Test ImagePair creation and dictionary serialization."""
    src_meta = RasterMetadata(path=Path("src.tif"), width=500, height=500)
    ref_meta = RasterMetadata(path=Path("ref.tif"), width=500, height=500)
    pair = ImagePair(pair_id="pair_001", source_meta=src_meta, reference_meta=ref_meta)

    d = pair.to_dict()
    assert d["pair_id"] == "pair_001"
    reconstructed = ImagePair.from_dict(d)
    assert reconstructed.source_meta.width == 500


def test_keypoint_set_validations_and_serialization() -> None:
    """Test KeypointSet validations and dictionary round-trip."""
    coords = np.array([[10.5, 20.25], [30.0, 40.0]], dtype=np.float64)
    desc = np.ones((2, 128), dtype=np.float32)
    kp_set = KeypointSet(coordinates=coords, descriptors=desc)

    assert len(kp_set) == 2
    np.testing.assert_allclose(kp_set.coordinates, coords)

    # Serialization
    d = kp_set.to_dict()
    reconstructed = KeypointSet.from_dict(d)
    assert len(reconstructed) == 2
    np.testing.assert_allclose(reconstructed.coordinates, coords)

    # Invalid coordinates (1D array)
    with pytest.raises(ValueError, match="shape"):
        KeypointSet(coordinates=np.array([1.0, 2.0]))

    # Invalid coordinates (NaN value)
    with pytest.raises(ValueError, match="non-finite"):
        KeypointSet(coordinates=np.array([[1.0, np.nan]]))

    # Descriptor length mismatch
    with pytest.raises(ValueError, match="Descriptors must be 2D array"):
        KeypointSet(coordinates=coords, descriptors=np.ones((3, 128)))


def test_match_set_validations_and_serialization() -> None:
    """Test MatchSet validations (Nx2 float64, finite values, matching counts) and serialization."""
    src_pts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float64)
    ref_pts = np.array([[12.0, 22.0], [32.0, 42.0], [52.0, 62.0]], dtype=np.float64)
    scores = np.array([0.95, 0.88, 0.91], dtype=np.float64)
    inliers = np.array([True, True, False], dtype=bool)

    matches = MatchSet(
        source_points=src_pts,
        reference_points=ref_pts,
        scores=scores,
        inliers=inliers,
        cell_ids=[0, 1, 2],
        refine_status=["accepted", "accepted", "rejected"],
    )

    assert len(matches) == 3
    np.testing.assert_allclose(matches.source_points, src_pts)
    np.testing.assert_allclose(matches.reference_points, ref_pts)

    # Serialization round-trip
    d = matches.to_dict()
    reconstructed = MatchSet.from_dict(d)
    assert len(reconstructed) == 3
    np.testing.assert_allclose(reconstructed.source_points, src_pts)
    np.testing.assert_allclose(reconstructed.inliers, inliers)

    # Rejection of length mismatch
    with pytest.raises(ValueError, match="must match reference points count"):
        MatchSet(source_points=src_pts, reference_points=ref_pts[:2])

    # Rejection of 1D shape
    with pytest.raises(ValueError, match="Source points must have shape"):
        MatchSet(source_points=np.array([1.0, 2.0]), reference_points=np.array([1.0, 2.0]))

    # Rejection of non-finite points (NaN)
    with pytest.raises(ValueError, match="non-finite"):
        MatchSet(
            source_points=np.array([[10.0, np.nan]]),
            reference_points=np.array([[10.0, 20.0]]),
        )


def test_transform_estimate() -> None:
    """Test TransformEstimate creation from matrix, dictionary round-trip, and validation."""
    mat = np.eye(3, dtype=np.float64)
    mat[0, 2] = 5.0
    mat[1, 2] = -10.0

    t_est = TransformEstimate.from_matrix(
        model_type="affine",
        matrix=mat,
        inlier_count=50,
        inlier_ratio=0.85,
        rmse_px=0.45,
    )

    assert t_est.model_type == "affine"
    assert t_est.inlier_count == 50
    np.testing.assert_allclose(t_est.get_matrix_array(), mat)

    # Dictionary round-trip
    d = t_est.to_dict()
    reconstructed = TransformEstimate.from_dict(d)
    assert reconstructed.model_type == "affine"
    assert reconstructed.rmse_px == 0.45


def test_registration_metrics_and_manifest() -> None:
    """Test RegistrationMetrics and RunManifest serialization."""
    metrics = RegistrationMetrics(
        rmse_px=0.35,
        median_residual_px=0.25,
        p90_residual_px=0.55,
        p95_residual_px=0.65,
        max_residual_px=0.95,
        inlier_count=45,
        total_matches=50,
        inlier_ratio=0.90,
        occupied_grid_fraction=0.75,
        convex_hull_coverage_fraction=0.68,
        runtime_seconds=1.25,
    )
    d_metrics = metrics.to_dict()
    assert d_metrics["inlier_count"] == 45
    reconstructed_metrics = RegistrationMetrics.from_dict(d_metrics)
    assert reconstructed_metrics.rmse_px == 0.35

    mat = np.eye(3, dtype=np.float64)
    t_est = TransformEstimate.from_matrix(model_type="affine", matrix=mat, inlier_count=45, inlier_ratio=0.90)

    manifest = RunManifest(
        pair_id="pair_001",
        source_path="data/src.tif",
        reference_path="data/ref.tif",
        config_hash="abc123hash",
        timestamp_utc="2026-08-23T00:00:00Z",
        package_version="0.1.0",
        seed=42,
        metrics=metrics,
        transform=t_est,
        quality_gate_passed=True,
    )

    d_manifest = manifest.to_dict()
    assert d_manifest["quality_gate_passed"] is True
    reconstructed_manifest = RunManifest.from_dict(d_manifest)
    assert reconstructed_manifest.config_hash == "abc123hash"
