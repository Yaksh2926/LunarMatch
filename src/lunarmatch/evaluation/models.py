"""Data models for evaluation metrics, control points, and bootstrap confidence intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from lunarmatch.models.domain import RegistrationMetrics


@dataclass(frozen=True)
class ControlPoint:
    """Independent ground-truth control point correspondence."""

    source_x: float
    source_y: float
    reference_x: float
    reference_y: float
    point_id: str = ""


class ControlPointMetrics(BaseModel):
    """Metrics evaluated against independent ground-truth control points.

    Units & Denominators:
    - cp_rmse_px, cp_median_px, cp_p90_px, cp_p95_px, cp_max_px: pixels (Euclidean distance).
    - count: number of valid evaluated control points.
    - residuals_px: array of per-control-point Euclidean errors.
    """

    cp_rmse_px: float = Field(description="Euclidean RMSE on independent control points (pixels).")
    cp_median_px: float = Field(description="Median residual error on control points (pixels).")
    cp_p90_px: float = Field(description="90th percentile residual error on control points (pixels).")
    cp_p95_px: float = Field(description="95th percentile residual error on control points (pixels).")
    cp_max_px: float = Field(description="Maximum residual error on control points (pixels).")
    count: int = Field(ge=0, description="Total count of evaluated control points.")
    residuals_px: list[float] = Field(default_factory=list, description="Per-point Euclidean residual distances.")
    evaluation_type: str = Field(
        default="independent_control_points",
        description="Explicit identifier confirming ground-truth evaluation.",
    )


class EvaluationSummary(BaseModel):
    """Comprehensive evaluation summary combining self-consistency metrics and control-point validation."""

    pair_id: str
    self_consistency_metrics: RegistrationMetrics
    control_point_metrics: ControlPointMetrics | None = None
    peak_memory_mb: float | None = Field(default=None, ge=0.0, description="Peak RSS process memory in MB.")
    subpixel_accuracy_proven: bool = Field(
        default=False,
        description="True ONLY if independent control-point RMSE < 1.0 px with count >= 5.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation summary to dictionary."""
        return self.model_dump()
