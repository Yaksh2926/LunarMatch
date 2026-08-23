"""Typed configuration models and YAML validation for LunarMatch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BaseConfigModel(BaseModel):
    """Base config model enforcing strict extra key prohibition."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(BaseConfigModel):
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    output_dir: Path = Field(default=Path("outputs/run"), description="Output directory.")
    overwrite: bool = Field(default=False, description="Overwrite output directory if exists.")


class IOConfig(BaseConfigModel):
    source_band: int = Field(default=1, ge=1, description="1-indexed source band.")
    reference_band: int = Field(default=1, ge=1, description="1-indexed reference band.")
    max_megapixels_in_memory: int = Field(default=256, gt=0, description="Max memory budget in megapixels.")
    nodata_policy: Literal["metadata_or_nan", "nan", "ignore", "zero"] = Field(
        default="metadata_or_nan", description="Nodata handling policy."
    )


class PreprocessConfig(BaseConfigModel):
    representation: Literal["gradient", "clahe", "edge", "phase", "raw_contrast"] = Field(
        default="gradient", description="Modality representation method."
    )
    percentile_clip: tuple[float, float] = Field(
        default=(1.0, 99.0), description="Min and max percentiles for clipping."
    )
    clahe_clip_limit: float = Field(default=2.0, gt=0, description="CLAHE clip limit.")
    denoise_sigma: float = Field(default=0.8, ge=0, description="Gaussian denoise sigma (0 to disable).")

    @field_validator("percentile_clip")
    @classmethod
    def validate_percentile_clip(cls, v: tuple[float, float]) -> tuple[float, float]:
        low, high = v
        if not (0.0 <= low <= 100.0 and 0.0 <= high <= 100.0):
            raise ValueError(f"Percentile clip values must be in range [0, 100], got ({low}, {high})")
        if low >= high:
            raise ValueError(f"Min percentile must be strictly less than max percentile, got ({low}, {high})")
        return v


class PyramidConfig(BaseConfigModel):
    metadata_scale_prior: bool = Field(default=True, description="Use raster pixel scales as scale prior.")
    log2_scale_min: float = Field(default=-4.0, description="Min log2 scale search level.")
    log2_scale_max: float = Field(default=4.0, description="Max log2 scale search level.")
    levels_per_octave: int = Field(default=2, ge=1, description="Pyramid levels per octave.")

    @model_validator(mode="after")
    def validate_scale_range(self) -> PyramidConfig:
        if self.log2_scale_min > self.log2_scale_max:
            raise ValueError(
                f"log2_scale_min ({self.log2_scale_min}) cannot be greater than log2_scale_max ({self.log2_scale_max})"
            )
        return self


class FeaturesConfig(BaseConfigModel):
    backend: Literal["sift", "orb", "learned_lightglue"] = Field(default="sift", description="Feature extraction backend.")
    max_keypoints: int = Field(default=30000, gt=0, description="Max keypoints to extract.")
    tile_size: int = Field(default=2048, gt=0, description="Tile dimension in pixels.")
    tile_overlap: int = Field(default=128, ge=0, description="Overlap between tiles in pixels.")
    weights_path: Path | None = Field(default=None, description="Optional local weight file path for learned backends.")

    @model_validator(mode="after")
    def validate_tile_overlap(self) -> FeaturesConfig:
        if self.tile_overlap >= self.tile_size:
            raise ValueError(
                f"tile_overlap ({self.tile_overlap}) must be strictly less than tile_size ({self.tile_size})"
            )
        return self


class MatchingConfig(BaseConfigModel):
    ratio_threshold: float = Field(default=0.80, gt=0.0, le=1.0, description="Lowe's ratio test threshold.")
    mutual: bool = Field(default=True, description="Enforce mutual nearest neighbors.")
    max_descriptor_distance: float | None = Field(
        default=None, ge=0.0, description="Optional max descriptor distance."
    )


class GeometryConfig(BaseConfigModel):
    model: Literal["similarity", "affine", "homography"] = Field(
        default="affine", description="Geometric transformation model."
    )
    robust_method: Literal["usac_magsac", "ransac"] = Field(
        default="usac_magsac", description="Robust estimation method."
    )
    reprojection_threshold_px: float = Field(default=3.0, gt=0.0, description="Inlier threshold in pixels.")
    confidence: float = Field(default=0.999, gt=0.0, lt=1.0, description="RANSAC confidence probability.")
    max_iterations: int = Field(default=10000, gt=0, description="Max RANSAC iterations.")
    allowed_scale: tuple[float, float] = Field(
        default=(0.01, 100.0), description="Allowed scale factor range [min_scale, max_scale]."
    )

    @field_validator("allowed_scale")
    @classmethod
    def validate_allowed_scale(cls, v: tuple[float, float]) -> tuple[float, float]:
        min_s, max_s = v
        if min_s <= 0 or max_s <= 0:
            raise ValueError(f"Allowed scale values must be positive, got ({min_s}, {max_s})")
        if min_s > max_s:
            raise ValueError(f"min scale ({min_s}) cannot be greater than max scale ({max_s})")
        return v


class CoverageConfig(BaseConfigModel):
    grid_rows: int = Field(default=8, gt=0, description="Grid rows for spatial coverage.")
    grid_cols: int = Field(default=8, gt=0, description="Grid columns for spatial coverage.")
    max_matches_per_cell: int = Field(default=20, gt=0, description="Max matches per cell.")
    min_occupied_fraction: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Min fraction of occupied grid cells."
    )


class SubpixelConfig(BaseConfigModel):
    method: Literal["ecc_patch", "phase_correlation", "none"] = Field(
        default="ecc_patch", description="Sub-pixel refinement method."
    )
    patch_radius: int = Field(default=15, gt=0, description="Patch radius for refinement.")
    max_iterations: int = Field(default=50, gt=0, description="Max refinement iterations.")
    epsilon: float = Field(default=0.0001, gt=0.0, description="Convergence threshold.")
    max_shift_px: float = Field(default=3.0, gt=0.0, description="Max allowed sub-pixel shift in pixels.")
    min_quality: float = Field(default=0.2, ge=0.0, le=1.0, description="Min quality threshold to accept refinement.")


class ExportConfig(BaseConfigModel):
    write_registered_raster: bool = Field(default=True, description="Export registered raster.")
    write_geojson: bool = Field(default=True, description="Export matches GeoJSON if georeferenced.")
    write_report: bool = Field(default=True, description="Generate visual HTML report.")
    interpolation: Literal["nearest", "linear", "cubic", "lanczos"] = Field(
        default="cubic", description="Warp interpolation mode."
    )


class QualityGatesConfig(BaseConfigModel):
    min_inliers: int = Field(default=30, ge=0, description="Min inlier match count required.")
    min_inlier_ratio: float = Field(default=0.20, ge=0.0, le=1.0, description="Min inlier ratio required.")
    max_rmse_px: float | None = Field(default=None, ge=0.0, description="Max RMSE threshold in pixels.")


class PerformanceConfig(BaseConfigModel):
    max_memory_mb: float = Field(default=2048.0, gt=0.0, description="Max memory allocation budget limit in MB.")


class PipelineConfig(BaseConfigModel):
    run: RunConfig = Field(default_factory=RunConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    pyramid: PyramidConfig = Field(default_factory=PyramidConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    subpixel: SubpixelConfig = Field(default_factory=SubpixelConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    quality_gates: QualityGatesConfig = Field(default_factory=QualityGatesConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary, handling Path serialization."""
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of configuration."""
        dumped = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If YAML parsing fails or config validation fails.
        TypeError: If top-level YAML data is not a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML file {file_path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise TypeError(f"Configuration file {file_path} must contain a top-level mapping/dictionary")

    try:
        return PipelineConfig.model_validate(raw_data)
    except Exception as exc:
        raise ValueError(f"Configuration validation failed for {file_path}:\n{exc}") from exc
