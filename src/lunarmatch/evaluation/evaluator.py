"""Evaluation orchestrator measuring self-consistency, ground-truth control points, and peak memory."""
from __future__ import annotations

from pathlib import Path

from lunarmatch.evaluation.control_points import evaluate_control_points, load_control_points
from lunarmatch.evaluation.models import EvaluationSummary
from lunarmatch.models.domain import RunManifest

try:
    import psutil  # type: ignore[import-untyped]

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_peak_memory_mb() -> float | None:
    """Get peak resident set size (RSS) process memory in Megabytes."""
    if HAS_PSUTIL:
        try:
            process = psutil.Process()
            mem_bytes = process.memory_info().rss
            return float(mem_bytes / (1024.0 * 1024.0))
        except Exception:  # noqa: BLE001
            return None
    return None


def evaluate_registration(
    manifest: RunManifest,
    control_points_path: str | Path | None = None,
    source_bounds: tuple[float, float] | None = None,
    reference_bounds: tuple[float, float] | None = None,
) -> EvaluationSummary:
    """Evaluate image registration results, incorporating optional independent control points.

    Args:
        manifest: RunManifest output from RegistrationOrchestrator.
        control_points_path: Optional CSV filepath containing ground-truth control points.
        source_bounds: Optional (width, height) boundary tuple for source raster.
        reference_bounds: Optional (width, height) boundary tuple for reference raster.

    Returns:
        EvaluationSummary containing self-consistency metrics, ground-truth control point metrics (if supplied),
        peak memory, and sub-pixel accuracy verification status.
    """
    cp_metrics = None
    subpixel_proven = False

    if control_points_path is not None and manifest.transform is not None:
        cpts = load_control_points(
            control_points_path,
            source_bounds=source_bounds,
            reference_bounds=reference_bounds,
        )
        cp_metrics = evaluate_control_points(cpts, manifest.transform)
        # Sub-pixel accuracy is proven ONLY if independent control-point RMSE < 1.0 px with count >= 5
        subpixel_proven = bool(cp_metrics.cp_rmse_px < 1.0 and cp_metrics.count >= 5)

    peak_mem = get_peak_memory_mb()

    return EvaluationSummary(
        pair_id=manifest.pair_id,
        self_consistency_metrics=manifest.metrics,
        control_point_metrics=cp_metrics,
        peak_memory_mb=peak_mem,
        subpixel_accuracy_proven=subpixel_proven,
    )
