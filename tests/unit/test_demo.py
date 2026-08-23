"""Unit tests for offline demo script and fallback synthetic demo execution."""
import subprocess
import sys
from pathlib import Path

import tifffile

from lunarmatch import register_pair
from lunarmatch.synthetic import generate_synthetic_pair


def test_synthetic_fallback_demo_execution(tmp_path: Path):
    """Test executing pair registration using synthetic fallback imagery."""
    synth_data = generate_synthetic_pair(
        pair_id="demo_unit_pair",
        angle_deg=2.0,
        translation=(4.0, -2.0),
        seed=42,
    )
    src_path = tmp_path / "src.tif"
    ref_path = tmp_path / "ref.tif"
    out_dir = tmp_path / "demo_out"

    tifffile.imwrite(str(src_path), synth_data.source_image)
    tifffile.imwrite(str(ref_path), synth_data.reference_image)

    manifest = register_pair(source=src_path, reference=ref_path, output_dir=out_dir, seed=42)

    assert manifest.quality_gate_passed is True
    assert (out_dir / "registered.tif").is_file()
    assert (out_dir / "matches.csv").is_file()
    assert (out_dir / "report.html").is_file()


def test_demo_script_subprocess_invocation(tmp_path: Path):
    """Test invoking scripts/run_demo.py via subprocess."""
    out_dir = tmp_path / "script_demo_out"
    cmd = [
        sys.executable,
        "scripts/run_demo.py",
        "--output-dir",
        str(out_dir),
        "--seed",
        "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert res.returncode == 0
    assert "Registration Status: PASS" in res.stdout
    assert (out_dir / "report.html").is_file()
