"""Evaluation module for self-consistency, ground-truth control points, and bootstrap confidence intervals."""
from lunarmatch.evaluation.bootstrap import compute_bootstrap_ci
from lunarmatch.evaluation.control_points import evaluate_control_points, load_control_points
from lunarmatch.evaluation.evaluator import evaluate_registration, get_peak_memory_mb
from lunarmatch.evaluation.models import ControlPoint, ControlPointMetrics, EvaluationSummary

__all__ = [
    "ControlPoint",
    "ControlPointMetrics",
    "EvaluationSummary",
    "compute_bootstrap_ci",
    "evaluate_control_points",
    "evaluate_registration",
    "get_peak_memory_mb",
    "load_control_points",
]
