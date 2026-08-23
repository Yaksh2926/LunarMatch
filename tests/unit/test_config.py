"""Unit tests for LunarMatch configuration models and validation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lunarmatch.cli.app import app
from lunarmatch.models.config import PipelineConfig, load_config

runner = CliRunner()


def test_baseline_config_validation(tmp_path: Path) -> None:
    """Test loading and validating the repository's baseline config."""
    config_path = Path("configs/baseline.yaml")
    assert config_path.is_file()

    cfg = load_config(config_path)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.run.seed == 42
    assert cfg.io.source_band == 1
    assert cfg.preprocess.representation == "gradient"
    assert cfg.features.backend == "sift"
    assert cfg.geometry.model == "affine"
    assert cfg.quality_gates.max_rmse_px is None


def test_config_hash_stability() -> None:
    """Test that config hash is deterministic and changes when config changes."""
    cfg1 = PipelineConfig()
    cfg2 = PipelineConfig()
    assert cfg1.compute_hash() == cfg2.compute_hash()

    # Create modified config
    raw = cfg1.to_dict()
    raw["run"]["seed"] = 999
    cfg_modified = PipelineConfig.model_validate(raw)
    assert cfg_modified.compute_hash() != cfg1.compute_hash()


def test_unknown_key_rejection(tmp_path: Path) -> None:
    """Test that unknown keys at root or section level are strictly rejected."""
    config_path = Path("configs/baseline.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Unknown root key
    data_root = dict(data)
    data_root["unexpected_root_key"] = 123
    root_file = tmp_path / "invalid_root.yaml"
    root_file.write_text(yaml.dump(data_root), encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration validation failed"):
        load_config(root_file)

    # Unknown section key
    data_sec = dict(data)
    data_sec["run"] = dict(data_sec["run"])
    data_sec["run"]["unknown_run_setting"] = "bad"
    sec_file = tmp_path / "invalid_section.yaml"
    sec_file.write_text(yaml.dump(data_sec), encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration validation failed"):
        load_config(sec_file)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (["preprocess", "percentile_clip"], [99.0, 1.0]),  # inverted
        (["preprocess", "percentile_clip"], [-5.0, 99.0]),  # out of range
        (["pyramid", "log2_scale_min"], 5.0),  # > log2_scale_max (4.0)
        (["features", "tile_overlap"], 2048),  # >= tile_size (2048)
        (["features", "backend"], "invalid_backend"),
        (["matching", "ratio_threshold"], 1.5),  # > 1.0
        (["geometry", "reprojection_threshold_px"], -1.0),  # <= 0
        (["geometry", "allowed_scale"], [10.0, 1.0]),  # inverted
        (["coverage", "min_occupied_fraction"], 1.2),  # > 1.0
        (["subpixel", "method"], "nonexistent_method"),
        (["export", "interpolation"], "bicubic_unknown"),
        (["quality_gates", "min_inlier_ratio"], -0.1),
    ],
)
def test_invalid_thresholds_ranges_methods(
    tmp_path: Path, field_path: list[str], invalid_value: object
) -> None:
    """Test cross-field and value validations on invalid input configurations."""
    config_path = Path("configs/baseline.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Mutate field
    target = data
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = invalid_value

    invalid_file = tmp_path / "invalid_config.yaml"
    invalid_file.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration validation failed"):
        load_config(invalid_file)


def test_max_rmse_px_null_and_float(tmp_path: Path) -> None:
    """Test max_rmse_px handles both null and positive float values."""
    config_path = Path("configs/baseline.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Test null (default in baseline)
    cfg_null = PipelineConfig.model_validate(data)
    assert cfg_null.quality_gates.max_rmse_px is None

    # Test float value
    data["quality_gates"]["max_rmse_px"] = 2.5
    float_file = tmp_path / "valid_float_rmse.yaml"
    float_file.write_text(yaml.dump(data), encoding="utf-8")

    cfg_float = load_config(float_file)
    assert cfg_float.quality_gates.max_rmse_px == 2.5


def test_missing_file() -> None:
    """Test load_config error on missing file path."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("non_existent_config_file_path.yaml")


def test_cli_validate_config(tmp_path: Path) -> None:
    """Test CLI subcommand validate-config for valid and invalid inputs."""
    # Valid test
    res_valid = runner.invoke(app, ["validate-config", "--config", "configs/baseline.yaml"])
    assert res_valid.exit_code == 0
    assert "OK" in res_valid.output
    assert "Config hash (SHA-256):" in res_valid.output

    # Invalid test
    invalid_file = tmp_path / "bad.yaml"
    invalid_file.write_text("run:\n  seed: invalid_seed_string\n", encoding="utf-8")
    res_invalid = runner.invoke(app, ["validate-config", "--config", str(invalid_file)])
    assert res_invalid.exit_code != 0
    assert "FAIL" in res_invalid.output
