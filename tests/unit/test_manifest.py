"""Unit tests for pair manifest CSV parsing and path resolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from lunarmatch.data import parse_pair_manifest


def test_parse_example_pair_manifest() -> None:
    """Test parsing repository example pair_manifest CSV file."""
    manifest_path = Path("sample_data/metadata/pair_manifest.example.csv")
    assert manifest_path.is_file()

    entries = parse_pair_manifest(manifest_path)
    assert len(entries) == 1

    entry = entries[0]
    assert entry.pair_id == "example_001"
    assert entry.source_sensor == "OHRC"
    assert entry.reference_sensor == "LRO_NAC"
    assert entry.source_path == (Path.cwd() / "data/raw/ohrc/source.tif").resolve()
    assert entry.reference_path == (Path.cwd() / "data/raw/lro/reference.tif").resolve()

    # Test conversion to ImagePair
    pair = entry.to_image_pair()
    assert pair.pair_id == "example_001"
    assert pair.source_meta.sensor == "OHRC"


def test_parse_custom_manifest(tmp_path: Path) -> None:
    """Test parsing custom manifest with all optional metadata fields."""
    csv_file = tmp_path / "custom_manifest.csv"
    csv_file.write_text(
        "pair_id,source_path,reference_path,source_sensor,reference_sensor,source_pixel_scale_m,reference_pixel_scale_m,source_sun_azimuth_deg,source_sun_elevation_deg,reference_sun_azimuth_deg,reference_sun_elevation_deg,control_points_path\n"
        "test_001,data/src.tif,data/ref.tif,OHRC,TMC2,0.25,5.0,45.0,30.0,60.0,35.0,data/cp.csv\n",
        encoding="utf-8",
    )

    entries = parse_pair_manifest(csv_file, repo_root=tmp_path)
    assert len(entries) == 1

    entry = entries[0]
    assert entry.pair_id == "test_001"
    assert entry.source_path == (tmp_path / "data/src.tif").resolve()
    assert entry.source_pixel_scale_m == 0.25
    assert entry.reference_pixel_scale_m == 5.0
    assert entry.source_sun_azimuth_deg == 45.0
    assert entry.control_points_path == (tmp_path / "data/cp.csv").resolve()


def test_missing_required_columns(tmp_path: Path) -> None:
    """Test manifest parsing failure when required columns are missing."""
    csv_file = tmp_path / "bad_columns.csv"
    csv_file.write_text("pair_id,invalid_column\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        parse_pair_manifest(csv_file)


def test_missing_manifest_file() -> None:
    """Test error handling for non-existent manifest file."""
    with pytest.raises(FileNotFoundError, match="Pair manifest CSV file not found"):
        parse_pair_manifest("non_existent_manifest.csv")
