"""Unit tests for package reproducibility, CLI help screens, and artifact schema versioning."""
from typer.testing import CliRunner

from lunarmatch import __version__
from lunarmatch.cli.app import app
from lunarmatch.models.domain import RegistrationMetrics, RunManifest


def test_cli_version():
    """Test lunarmatch version CLI command output."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output.strip()


def test_cli_root_help():
    """Test lunarmatch root CLI help page."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "register" in result.output
    assert "benchmark" in result.output
    assert "generate-synthetic" in result.output
    assert "validate-config" in result.output


def test_cli_subcommand_help_screens():
    """Test CLI help pages for all subcommands."""
    runner = CliRunner()
    for subcmd in ["register", "benchmark", "generate-synthetic", "validate-config"]:
        result = runner.invoke(app, [subcmd, "--help"])
        assert result.exit_code == 0, f"Help command for '{subcmd}' failed with code {result.exit_code}"
        assert "Usage" in result.output or "Options" in result.output


def test_run_manifest_schema_version():
    """Test that RunManifest captures schema_version field '1.0.0'."""
    metrics = RegistrationMetrics(
        rmse_px=0.5,
        inlier_count=50,
        total_matches=60,
        inlier_ratio=0.83,
        occupied_grid_fraction=0.8,
        convex_hull_coverage_fraction=0.7,
        runtime_seconds=0.1,
    )
    manifest = RunManifest(
        pair_id="pair_test",
        source_path="/path/src.tif",
        reference_path="/path/ref.tif",
        config_hash="hash123",
        timestamp_utc="2026-08-23T12:00:00Z",
        package_version=__version__,
        seed=42,
        metrics=metrics,
        quality_gate_passed=True,
    )

    assert manifest.schema_version == "1.0.0"
    data = manifest.to_dict()
    assert data["schema_version"] == "1.0.0"
