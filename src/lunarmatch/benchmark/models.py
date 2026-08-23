"""Data models for batch benchmarking inputs, binned statistics, and aggregate summaries."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkPairInput(BaseModel):
    """Pair specification for batch benchmark execution."""

    pair_id: str = Field(description="Unique pair identifier.")
    source: str = Field(description="Path to source raster file.")
    reference: str = Field(description="Path to reference raster file.")
    source_sensor: str | None = Field(default=None, description="Optional source sensor identifier.")
    reference_sensor: str | None = Field(default=None, description="Optional reference sensor identifier.")
    pixel_scale_ratio: float | None = Field(default=None, ge=0.0, description="Optional pixel scale ratio.")
    sun_angle_diff_deg: float | None = Field(default=None, ge=0.0, description="Optional sun angle difference in degrees.")
    control_points_path: str | None = Field(default=None, description="Optional path to control points CSV file.")

    @property
    def sensor_pair(self) -> str | None:
        """Return combined sensor pair string e.g. 'OHRC_LRO_NAC' if both present."""
        if self.source_sensor and self.reference_sensor:
            return f"{self.source_sensor}_{self.reference_sensor}"
        return None

    @property
    def scale_bin(self) -> str | None:
        """Categorize scale ratio into standard evaluation bins."""
        if self.pixel_scale_ratio is None:
            return None
        r = self.pixel_scale_ratio
        if r <= 1.5:
            return "< 1.5x"
        elif r <= 3.0:
            return "1.5x - 3.0x"
        else:
            return "> 3.0x"

    @property
    def sun_angle_bin(self) -> str | None:
        """Categorize sun angle difference in degrees into evaluation bins."""
        if self.sun_angle_diff_deg is None:
            return None
        d = self.sun_angle_diff_deg
        if d <= 10.0:
            return "< 10°"
        elif d <= 30.0:
            return "10° - 30°"
        else:
            return "> 30°"


class BinnedMetrics(BaseModel):
    """Aggregate metric statistics for a specific benchmark bin or subset."""

    bin_name: str = Field(description="Name of category or bin.")
    total_pairs: int = Field(ge=0, description="Total count of pairs in bin.")
    passed_pairs: int = Field(ge=0, description="Count of passed pairs.")
    failed_pairs: int = Field(ge=0, description="Count of failed pairs.")
    pass_rate: float = Field(ge=0.0, le=1.0, description="Passed pairs ratio.")
    mean_rmse_px: float | None = Field(default=None, description="Mean RMSE in pixels across passed pairs.")
    median_rmse_px: float | None = Field(default=None, description="Median RMSE in pixels across passed pairs.")
    mean_inliers: float = Field(default=0.0, description="Mean inlier count.")
    mean_inlier_ratio: float = Field(default=0.0, description="Mean inlier ratio.")
    mean_runtime_seconds: float = Field(default=0.0, description="Mean runtime per pair in seconds.")


class BenchmarkSummary(BaseModel):
    """Overall benchmark summary output capturing aggregate and binned performance."""

    total_pairs: int = Field(ge=0, description="Total pairs processed.")
    passed_pairs: int = Field(ge=0, description="Count of successful registrations.")
    failed_pairs: int = Field(ge=0, description="Count of ordinary failures.")
    pass_rate: float = Field(ge=0.0, le=1.0, description="Overall pass rate ratio.")
    overall_metrics: BinnedMetrics = Field(description="Overall aggregate metrics across all pairs.")
    binned_by_sensor: list[BinnedMetrics] = Field(default_factory=list, description="Aggregations by sensor pair.")
    binned_by_scale: list[BinnedMetrics] = Field(default_factory=list, description="Aggregations by scale bin.")
    binned_by_sun_angle: list[BinnedMetrics] = Field(default_factory=list, description="Aggregations by sun angle bin.")
    pair_results: list[dict[str, Any]] = Field(default_factory=list, description="Per-pair result dictionaries.")

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark summary to dictionary."""
        return self.model_dump()
