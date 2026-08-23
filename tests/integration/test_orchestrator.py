"""Integration tests for end-to-end baseline pair-registration orchestrator."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from typer.testing import CliRunner

from lunarmatch.cli.app import app
from lunarmatch.orchestrator import register_pair


def generate_synthetic_lunar_terrain(width: int = 512, height: int = 512, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic reference and source terrain pair with known affine transform."""
    # Generate larger canvas to avoid zero-border artifacts after warping
    pad = 50
    w_large, h_large = width + 2 * pad, height + 2 * pad
    rng = np.random.default_rng(seed)

    x = np.linspace(-10, 10, w_large)
    y = np.linspace(-10, 10, h_large)
    xx, yy = np.meshgrid(x, y)

    z = (
        np.sin(xx * 0.8) * np.cos(yy * 0.8) * 50.0
        + np.sin(xx * 2.0 + 0.3) * np.cos(yy * 1.8) * 30.0
        + rng.normal(0, 5, size=(h_large, w_large))
    )

    for _ in range(25):
        cx = rng.uniform(50, w_large - 50)
        cy = rng.uniform(50, h_large - 50)
        crater_r = rng.uniform(15, 45)
        depth = rng.uniform(30, 80)

        grid_y, grid_x = np.ogrid[:h_large, :w_large]
        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        crater_mask = dist_sq <= crater_r**2
        z[crater_mask] -= depth * (1.0 - dist_sq[crater_mask] / crater_r**2)

    dx, dy = np.gradient(z)
    shade = np.sin(np.radians(45.0)) * np.sin(np.pi / 2.0 - np.arctan(np.hypot(dx, dy)))
    large_ref = np.clip((shade - shade.min()) / (shade.max() - shade.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # Known Affine ground truth transformation: rotation = 3 degrees, translation = (10.5, -7.5) px
    angle_rad = np.radians(3.0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    gt_dx, gt_dy = 10.5, -7.5

    affine_gt_2x3 = np.array(
        [[cos_a, -sin_a, gt_dx], [sin_a, cos_a, gt_dy]],
        dtype=np.float32,
    )

    large_src = cv2.warpAffine(large_ref, affine_gt_2x3, (w_large, h_large), flags=cv2.INTER_CUBIC)

    # Crop to central (width x height) region
    ref_img = large_ref[pad : pad + height, pad : pad + width]
    src_img = large_src[pad : pad + height, pad : pad + width]

    return src_img, ref_img, affine_gt_2x3


@pytest.fixture
def synthetic_lunar_pair(tmp_path: Path):
    """Fixture producing synthetic reference and source raster GeoTIFF files with known affine transform."""
    src_img, ref_img, affine_gt_2x3 = generate_synthetic_lunar_terrain(512, 512, seed=100)

    ref_path = tmp_path / "reference.tif"
    src_path = tmp_path / "source.tif"

    profile = {
        "driver": "GTiff",
        "height": 512,
        "width": 512,
        "count": 1,
        "dtype": "uint8",
        "transform": from_origin(0, 0, 1, 1),
    }

    with rasterio.open(ref_path, "w", **profile) as dst:
        dst.write(ref_img, 1)

    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(src_img, 1)

    return src_path, ref_path, affine_gt_2x3


def test_e2e_synthetic_registration_python_api(synthetic_lunar_pair, tmp_path: Path):
    """Integration test: Python API completes registration on synthetic lunar terrain."""
    src_path, ref_path, _gt_mat = synthetic_lunar_pair
    out_dir = tmp_path / "output"

    manifest = register_pair(
        source=src_path,
        reference=ref_path,
        output_dir=out_dir,
        seed=42,
    )

    assert manifest.quality_gate_passed is True
    assert manifest.transform is not None
    assert manifest.transform.inlier_count >= 10
    assert manifest.metrics.rmse_px is not None
    assert manifest.metrics.rmse_px < 1.0

    # Verify all expected output files exist
    assert (out_dir / "run_manifest.json").is_file()
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "transform.json").is_file()
    assert (out_dir / "matches.csv").is_file()
    assert (out_dir / "diagnostics.json").is_file()

    # Validate JSON manifest deserialization
    with (out_dir / "run_manifest.json").open("r", encoding="utf-8") as f:
        manifest_dict = json.load(f)
    assert manifest_dict["pair_id"] == "source_reference"
    assert manifest_dict["quality_gate_passed"] is True


def test_e2e_synthetic_registration_cli(synthetic_lunar_pair, tmp_path: Path):
    """Integration test: CLI command completes registration on synthetic pair."""
    src_path, ref_path, _ = synthetic_lunar_pair
    out_dir = tmp_path / "cli_output"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "register",
            "--source",
            str(src_path),
            "--reference",
            str(ref_path),
            "--output-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert (out_dir / "run_manifest.json").is_file()


def test_deterministic_repeat_runs(synthetic_lunar_pair, tmp_path: Path):
    """Integration test: repeat runs with identical seed yield identical manifests."""
    src_path, ref_path, _ = synthetic_lunar_pair
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    m1 = register_pair(src_path, ref_path, output_dir=out1, seed=42)
    m2 = register_pair(src_path, ref_path, output_dir=out2, seed=42)

    assert m1.config_hash == m2.config_hash
    assert m1.quality_gate_passed == m2.quality_gate_passed
    assert m1.metrics.inlier_count == m2.metrics.inlier_count
    assert m1.transform is not None
    assert m2.transform is not None
    np.testing.assert_allclose(m1.transform.matrix, m2.transform.matrix, atol=1e-6)


def test_ordinary_failure_diagnostics(tmp_path: Path):
    """Integration test: featureless blank rasters trigger explicit diagnostic failure without crashing."""
    blank = np.zeros((200, 200), dtype=np.uint8)
    src_path = tmp_path / "blank_src.tif"
    ref_path = tmp_path / "blank_ref.tif"

    profile = {"driver": "GTiff", "height": 200, "width": 200, "count": 1, "dtype": "uint8"}
    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(blank, 1)
    with rasterio.open(ref_path, "w", **profile) as dst:
        dst.write(blank, 1)

    out_dir = tmp_path / "fail_output"
    manifest = register_pair(src_path, ref_path, output_dir=out_dir, seed=42)

    assert manifest.quality_gate_passed is False
    assert len(manifest.metrics.warnings) > 0
    assert (out_dir / "run_manifest.json").is_file()
    assert (out_dir / "diagnostics.json").is_file()

    with (out_dir / "diagnostics.json").open("r", encoding="utf-8") as f:
        diag = json.load(f)
    assert diag["quality_gate_passed"] is False
    assert len(diag["warnings"]) > 0
