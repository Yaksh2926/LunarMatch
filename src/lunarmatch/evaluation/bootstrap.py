"""Non-parametric bootstrap confidence interval estimation for multi-pair metrics."""
from __future__ import annotations

import numpy as np


def compute_bootstrap_ci(
    data: np.ndarray | list[float],
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence intervals across image pair metrics.

    Args:
        data: Array or list of metric values across image pairs (e.g. RMSE values).
        confidence_level: Desired confidence level (e.g., 0.95 for 95% CI).
        n_resamples: Number of bootstrap resamples (default: 1000).
        seed: Random seed for reproducible resampling.

    Returns:
        Tuple of (mean_value, ci_lower, ci_upper).

    Raises:
        ValueError: If input data is empty or confidence_level is invalid.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute bootstrap confidence interval: input data is empty.")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")

    mean_val = float(np.mean(arr))

    # For small sample size (< 3), bootstrap is ill-defined; return sample range
    if len(arr) < 3:
        return mean_val, float(np.min(arr)), float(np.max(arr))

    rng = np.random.default_rng(seed)
    n_samples = len(arr)

    # Resample with replacement: shape (n_resamples, n_samples)
    resample_indices = rng.choice(n_samples, size=(n_resamples, n_samples), replace=True)
    resampled_means = np.mean(arr[resample_indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(resampled_means, lower_pct))
    ci_upper = float(np.percentile(resampled_means, upper_pct))

    return mean_val, ci_lower, ci_upper
