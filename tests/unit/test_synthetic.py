"""Unit and regression tests for procedural synthetic lunar benchmark generator."""
import json
import time
from pathlib import Path

import numpy as np

from lunarmatch.synthetic.generator import (
    generate_synthetic_benchmark_dataset,
    generate_synthetic_pair,
)
from lunarmatch.synthetic.terrain import generate_crater_heightfield, render_hillshade


def test_crater_heightfield_and_hillshade():
    """Test heightfield and Lambertian hillshade rendering shapes and output ranges."""
    hf = generate_crater_heightfield(shape=(100, 100), num_craters=5, seed=42)
    assert hf.shape == (100, 100)
    assert hf.dtype == np.float32

    hs = render_hillshade(hf, azimuth_deg=315.0, elevation_deg=45.0)
    assert hs.shape == (100, 100)
    assert hs.dtype == np.uint8
    assert np.min(hs) >= 0
    assert np.max(hs) <= 255


def test_synthetic_pair_control_point_exactness():
    """Test that ground-truth control points match transform matrix M exactly (< 1e-6 px)."""
    pair = generate_synthetic_pair(
        pair_id="test_exact",
        angle_deg=4.5,
        translation=(12.0, -8.0),
        scale_ratio=1.0,
        shape=(200, 200),
        seed=42,
    )

    M = pair.ground_truth_transform.get_matrix_array()
    assert M.shape == (3, 3)

    for cp in pair.control_points:
        src_p = np.array([cp.source_x, cp.source_y, 1.0], dtype=np.float64)
        ref_p = M @ src_p

        pred_x, pred_y = float(ref_p[0]), float(ref_p[1])
        err = np.sqrt((pred_x - cp.reference_x) ** 2 + (pred_y - cp.reference_y) ** 2)
        assert err < 1e-6, f"Control point error {err} exceeds threshold for {cp.point_id}"


def test_generate_synthetic_benchmark_dataset_export(tmp_path: Path):
    """Test full synthetic dataset generation, files, and disclaimer metadata."""
    t0 = time.perf_counter()
    ds_dir = generate_synthetic_benchmark_dataset(output_dir=tmp_path / "synth_ds", num_pairs=2, seed=42)
    elapsed = time.perf_counter() - t0

    # Speed check for CI resources
    assert elapsed < 3.0, f"Synthetic generation took {elapsed:.2f}s, expected < 3.0s"

    manifest_path = ds_dir / "pairs_manifest.json"
    assert manifest_path.is_file()

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("synthetic") is True
    assert "Synthetic procedural benchmark dataset" in data.get("disclaimer", "")
    assert len(data["pairs"]) == 2

    # Check pair 1 artifacts
    p1_dir = ds_dir / "synth_pair_001"
    assert (p1_dir / "source.tif").is_file()
    assert (p1_dir / "reference.tif").is_file()
    assert (p1_dir / "control_points.csv").is_file()
    assert (p1_dir / "ground_truth_transform.json").is_file()
