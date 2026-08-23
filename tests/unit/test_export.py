"""Unit tests for atomic product export, overwrite control, and geospatial metadata handling."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lunarmatch.export.writer import export_registered_products
from lunarmatch.models.domain import (
    MatchSet,
    RasterMetadata,
    RegistrationMetrics,
    RunManifest,
    TransformEstimate,
)


@pytest.fixture
def sample_manifest_and_matches(tmp_path: Path):
    """Fixture returning sample RunManifest, MatchSet, and RasterMetadata objects."""
    src_meta = RasterMetadata(
        path=tmp_path / "source.tif",
        width=100,
        height=100,
        bands=1,
        dtype="uint8",
    )
    ref_meta = RasterMetadata(
        path=tmp_path / "reference.tif",
        width=100,
        height=100,
        bands=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=(0.0, 1.0, 0.0, 100.0, 0.0, -1.0),
        nodata_value=0.0,
    )

    transform = TransformEstimate.from_matrix(
        model_type="similarity",
        matrix=np.eye(3, dtype=np.float64),
        inlier_count=10,
        inlier_ratio=0.8,
        rmse_px=0.25,
    )

    metrics = RegistrationMetrics(
        rmse_px=0.25,
        median_residual_px=0.20,
        p90_residual_px=0.35,
        p95_residual_px=0.40,
        max_residual_px=0.50,
        inlier_count=10,
        total_matches=12,
        inlier_ratio=0.8,
        occupied_grid_fraction=0.75,
        convex_hull_coverage_fraction=0.80,
        count_uniformity_cv=0.15,
        runtime_seconds=0.5,
        warnings=[],
    )

    manifest = RunManifest(
        pair_id="source_reference",
        source_path=str(src_meta.path),
        reference_path=str(ref_meta.path),
        config_hash="abc123hash",
        timestamp_utc="2026-08-23T11:00:00Z",
        package_version="0.1.0",
        seed=42,
        metrics=metrics,
        transform=transform,
        quality_gate_passed=True,
    )

    matches = MatchSet(
        source_points=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        reference_points=np.array([[10.5, 20.2], [30.1, 40.3]], dtype=np.float64),
        scores=np.array([0.9, 0.85], dtype=np.float64),
        inliers=np.array([True, True], dtype=bool),
        cell_ids=[0, 1],
        refine_status=["accepted", "accepted"],
    )

    warped = np.ones((100, 100), dtype=np.uint8) * 128
    return manifest, matches, warped, src_meta, ref_meta


def test_export_registered_products_schema_compliance(sample_manifest_and_matches, tmp_path: Path):
    """Test that exported product artifact schemas match PROJECT_SPEC.md requirements."""
    manifest, matches, warped, src_meta, ref_meta = sample_manifest_and_matches
    out_dir = tmp_path / "products"

    exported = export_registered_products(
        manifest=manifest,
        matches=matches,
        warped_image=warped,
        source_metadata=src_meta,
        reference_metadata=ref_meta,
        output_dir=out_dir,
        overwrite=False,
    )

    assert "manifest" in exported
    assert "metrics" in exported
    assert "transform" in exported
    assert "matches" in exported
    assert "registered" in exported
    assert "geojson" in exported

    # 1. Check run_manifest.json schema
    with exported["manifest"].open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pair_id"] == "source_reference"
    assert data["quality_gate_passed"] is True
    assert "config_hash" in data
    assert "seed" in data

    # 2. Check metrics.json schema
    with exported["metrics"].open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["rmse_px"] == 0.25
    assert data["inlier_count"] == 10

    # 3. Check matches.csv header schema
    with exported["matches"].open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == [
        "source_x",
        "source_y",
        "reference_x",
        "reference_y",
        "score",
        "inlier",
        "cell_id",
        "refine_status",
    ]


def test_overwrite_policy_refuses_when_configured(sample_manifest_and_matches, tmp_path: Path):
    """Test that FileExistsError is raised when file exists and overwrite=False."""
    manifest, matches, warped, src_meta, ref_meta = sample_manifest_and_matches
    out_dir = tmp_path / "overwrite_test"

    # First export succeeds
    export_registered_products(
        manifest=manifest,
        matches=matches,
        warped_image=warped,
        source_metadata=src_meta,
        reference_metadata=ref_meta,
        output_dir=out_dir,
        overwrite=False,
    )

    # Second export with overwrite=False raises FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        export_registered_products(
            manifest=manifest,
            matches=matches,
            warped_image=warped,
            source_metadata=src_meta,
            reference_metadata=ref_meta,
            output_dir=out_dir,
            overwrite=False,
        )

    # Third export with overwrite=True succeeds
    exported = export_registered_products(
        manifest=manifest,
        matches=matches,
        warped_image=warped,
        source_metadata=src_meta,
        reference_metadata=ref_meta,
        output_dir=out_dir,
        overwrite=True,
    )
    assert exported["manifest"].is_file()


def test_geojson_export_conditional(sample_manifest_and_matches, tmp_path: Path):
    """Test GeoJSON export when georeferencing exists, and graceful skip when absent."""
    manifest, matches, warped, src_meta, ref_meta = sample_manifest_and_matches

    # Georeferenced export
    out_geo = tmp_path / "with_geo"
    exp_geo = export_registered_products(
        manifest=manifest,
        matches=matches,
        warped_image=warped,
        source_metadata=src_meta,
        reference_metadata=ref_meta,
        output_dir=out_geo,
        export_geojson=True,
    )
    assert "geojson" in exp_geo
    assert exp_geo["geojson"].is_file()

    # Ungeoreferenced export (CRS = None)
    ref_meta_no_crs = RasterMetadata(
        path=tmp_path / "ref_nocrs.tif",
        width=100,
        height=100,
        bands=1,
        dtype="uint8",
        crs=None,
        transform=None,
    )
    out_nogeo = tmp_path / "no_geo"
    exp_nogeo = export_registered_products(
        manifest=manifest,
        matches=matches,
        warped_image=warped,
        source_metadata=src_meta,
        reference_metadata=ref_meta_no_crs,
        output_dir=out_nogeo,
        export_geojson=True,
    )
    assert "geojson" not in exp_nogeo
