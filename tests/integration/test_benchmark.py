"""Integration tests for batch benchmark command, failure isolation, and restart safety."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from typer.testing import CliRunner

from lunarmatch.benchmark import run_batch_benchmark
from lunarmatch.cli.app import app


@pytest.fixture
def synthetic_benchmark_batch(tmp_path: Path):
    """Fixture generating synthetic multi-pair benchmark dataset."""
    rng = np.random.default_rng(42)

    def create_terrain(h=300, w=300):
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        terrain = np.sin(grid_x / 15.0) * np.cos(grid_y / 15.0) * 80.0 + 128.0
        terrain += rng.normal(0, 3, size=(h, w))
        return np.clip(terrain, 0, 255).astype(np.uint8)

    raw = create_terrain()
    M1 = cv2.getRotationMatrix2D((150, 150), angle=2.0, scale=1.0)
    M1[:, 2] += [5.0, -3.0]
    warped1 = cv2.warpAffine(raw, M1, (300, 300), borderMode=cv2.BORDER_REFLECT)

    M2 = cv2.getRotationMatrix2D((150, 150), angle=-1.5, scale=1.0)
    M2[:, 2] += [-4.0, 2.0]
    warped2 = cv2.warpAffine(raw, M2, (300, 300), borderMode=cv2.BORDER_REFLECT)

    # Crop central 200x200 region
    ref_img = raw[50:250, 50:250]
    src_img1 = warped1[50:250, 50:250]
    src_img2 = warped2[50:250, 50:250]
    blank_img = np.zeros((200, 200), dtype=np.uint8)

    profile = {"driver": "GTiff", "height": 200, "width": 200, "count": 1, "dtype": "uint8"}

    # Pair 1: Good pair
    p1_src = tmp_path / "p1_src.tif"
    p1_ref = tmp_path / "p1_ref.tif"
    with rasterio.open(p1_src, "w", **profile) as dst:
        dst.write(src_img1, 1)
    with rasterio.open(p1_ref, "w", **profile) as dst:
        dst.write(ref_img, 1)

    # Pair 2: Second good pair
    p2_src = tmp_path / "p2_src.tif"
    p2_ref = tmp_path / "p2_ref.tif"
    with rasterio.open(p2_src, "w", **profile) as dst:
        dst.write(src_img2, 1)
    with rasterio.open(p2_ref, "w", **profile) as dst:
        dst.write(ref_img, 1)

    # Pair 3: Failing blank pair
    p3_src = tmp_path / "p3_src.tif"
    p3_ref = tmp_path / "p3_ref.tif"
    with rasterio.open(p3_src, "w", **profile) as dst:
        dst.write(blank_img, 1)
    with rasterio.open(p3_ref, "w", **profile) as dst:
        dst.write(blank_img, 1)

    manifest_data = {
        "pairs": [
            {
                "pair_id": "pair_01_ohrc_lro",
                "source": str(p1_src),
                "reference": str(p1_ref),
                "source_sensor": "OHRC",
                "reference_sensor": "LRO_NAC",
                "pixel_scale_ratio": 1.2,
                "sun_angle_diff_deg": 8.5,
            },
            {
                "pair_id": "pair_02_tmc_lro",
                "source": str(p2_src),
                "reference": str(p2_ref),
                "source_sensor": "TMC2",
                "reference_sensor": "LRO_NAC",
                "pixel_scale_ratio": 2.5,
                "sun_angle_diff_deg": 22.0,
            },
            {
                "pair_id": "pair_03_blank_fail",
                "source": str(p3_src),
                "reference": str(p3_ref),
                "source_sensor": "OHRC",
                "reference_sensor": "LRO_NAC",
                "pixel_scale_ratio": 1.0,
                "sun_angle_diff_deg": 5.0,
            },
        ]
    }

    manifest_path = tmp_path / "pairs_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path


def test_batch_benchmark_execution_and_failure_isolation(synthetic_benchmark_batch, tmp_path: Path):
    """Test batch benchmark execution, failure isolation, and aggregate metrics."""
    out_dir = tmp_path / "bench_out"
    summary = run_batch_benchmark(synthetic_benchmark_batch, output_dir=out_dir, workers=1, seed=42)

    assert summary.total_pairs == 3
    assert summary.passed_pairs == 2
    assert summary.failed_pairs == 1
    assert pytest.approx(summary.pass_rate, abs=0.01) == 2.0 / 3.0

    # Output files exist
    assert (out_dir / "benchmark_summary.json").is_file()
    assert (out_dir / "benchmark_summary.csv").is_file()

    # Binned aggregations exist
    assert len(summary.binned_by_sensor) > 0
    assert len(summary.binned_by_scale) > 0
    assert len(summary.binned_by_sun_angle) > 0


def test_batch_benchmark_restart_resume(synthetic_benchmark_batch, tmp_path: Path):
    """Test restart safety: rerun with resume=True skips already completed pairs."""
    out_dir = tmp_path / "resume_out"

    # First run
    s1 = run_batch_benchmark(synthetic_benchmark_batch, output_dir=out_dir, resume=True, seed=42)
    assert s1.total_pairs == 3

    # Second run with resume=True
    s2 = run_batch_benchmark(synthetic_benchmark_batch, output_dir=out_dir, resume=True, seed=42)
    assert s2.total_pairs == 3

    for res in s2.pair_results:
        assert res.get("resumed") is True


def test_batch_benchmark_cli(synthetic_benchmark_batch, tmp_path: Path):
    """Test lunarmatch benchmark CLI command execution."""
    out_dir = tmp_path / "cli_bench_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--manifest",
            str(synthetic_benchmark_batch),
            "--output",
            str(out_dir),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "Batch Benchmark Complete" in result.output
    assert (out_dir / "benchmark_summary.json").is_file()
