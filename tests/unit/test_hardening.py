"""Unit tests for large-raster memory hardening, edge-case failure modes, and atomic writes."""
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lunarmatch.data.raster import RasterReadError, read_raster
from lunarmatch.models.config import PerformanceConfig, PipelineConfig
from lunarmatch.orchestrator import RegistrationOrchestrator


@pytest.fixture
def base_raster_path(tmp_path: Path):
    """Fixture creating a valid base synthetic raster."""
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, size=(100, 100), dtype=np.uint8)
    p = tmp_path / "base.tif"
    profile = {"driver": "GTiff", "height": 100, "width": 100, "count": 1, "dtype": "uint8"}
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(img, 1)
    return p


def test_all_nodata_raster_rejection(base_raster_path, tmp_path: Path):
    """Test explicit actionable failure when source image is 100% nodata."""
    nodata_path = tmp_path / "all_nodata.tif"
    blank_img = np.zeros((100, 100), dtype=np.uint8)
    profile = {"driver": "GTiff", "height": 100, "width": 100, "count": 1, "dtype": "uint8", "nodata": 0}
    with rasterio.open(nodata_path, "w", **profile) as dst:
        dst.write(blank_img, 1)

    orchestrator = RegistrationOrchestrator()
    manifest = orchestrator.register(source=nodata_path, reference=base_raster_path, output_dir=tmp_path / "out1")

    assert manifest.quality_gate_passed is False
    assert any("100% nodata" in w for w in manifest.metrics.warnings)


def test_constant_value_raster_rejection(base_raster_path, tmp_path: Path):
    """Test explicit actionable failure when raster has zero variance."""
    const_path = tmp_path / "constant.tif"
    const_img = np.full((100, 100), 128, dtype=np.uint8)
    profile = {"driver": "GTiff", "height": 100, "width": 100, "count": 1, "dtype": "uint8"}
    with rasterio.open(const_path, "w", **profile) as dst:
        dst.write(const_img, 1)

    orchestrator = RegistrationOrchestrator()
    manifest = orchestrator.register(source=const_path, reference=base_raster_path, output_dir=tmp_path / "out2")

    assert manifest.quality_gate_passed is False
    assert any("zero variance" in w for w in manifest.metrics.warnings)


def test_corrupt_raster_file_handling(base_raster_path, tmp_path: Path):
    """Test that corrupt raster files raise RasterReadError and return a failure manifest."""
    corrupt_path = tmp_path / "corrupt.tif"
    with corrupt_path.open("w", encoding="utf-8") as f:
        f.write("THIS IS NOT A VALID TIFF FILE HEADER DATA")

    with pytest.raises(RasterReadError):
        read_raster(corrupt_path)

    orchestrator = RegistrationOrchestrator()
    manifest = orchestrator.register(source=corrupt_path, reference=base_raster_path, output_dir=tmp_path / "out3")

    assert manifest.quality_gate_passed is False
    assert any("Failed to read raster input files" in w for w in manifest.metrics.warnings)


def test_memory_budget_warning(base_raster_path, tmp_path: Path):
    """Test that exceeding memory budget emits an explicit diagnostic warning."""
    cfg = PipelineConfig(performance=PerformanceConfig(max_memory_mb=0.0001))

    orchestrator = RegistrationOrchestrator(config=cfg)
    manifest = orchestrator.register(source=base_raster_path, reference=base_raster_path, output_dir=tmp_path / "out4")

    assert any("exceeds configured budget" in w for w in manifest.metrics.warnings)
