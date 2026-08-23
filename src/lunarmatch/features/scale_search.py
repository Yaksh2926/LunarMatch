"""Scale-hypothesis generation using metadata priors or bounded logarithmic search."""
from __future__ import annotations

import numpy as np

from lunarmatch.models.config import PyramidConfig


def generate_scale_hypotheses(
    source_scale_m: float | None = None,
    reference_scale_m: float | None = None,
    config: PyramidConfig | None = None,
    allowed_scale: tuple[float, float] = (0.01, 100.0),
) -> list[float]:
    """Generate deterministic scale-ratio hypotheses between source and reference rasters.

    Scale ratio s = (reference_pixel_scale) / (source_pixel_scale), mapping source coords to reference.

    Args:
        source_scale_m: Optional source raster pixel scale in meters/pixel.
        reference_scale_m: Optional reference raster pixel scale in meters/pixel.
        config: Optional PyramidConfig instance.
        allowed_scale: Allowed scale ratio bounds (min_scale, max_scale).

    Returns:
        Sorted deterministic list of scale factor hypotheses (floats).
    """
    cfg = config if config is not None else PyramidConfig()
    min_allowed, max_allowed = allowed_scale

    use_metadata_prior = (
        cfg.metadata_scale_prior
        and source_scale_m is not None
        and reference_scale_m is not None
        and source_scale_m > 0.0
        and reference_scale_m > 0.0
    )

    if use_metadata_prior:
        assert source_scale_m is not None and reference_scale_m is not None
        prior_ratio = float(reference_scale_m) / float(source_scale_m)
    else:
        prior_ratio = 1.0

    steps_per_octave = max(1, cfg.levels_per_octave)
    min_exp = float(cfg.log2_scale_min)
    max_exp = float(cfg.log2_scale_max)

    # Number of steps
    num_steps = round((max_exp - min_exp) * steps_per_octave) + 1
    exponents = np.linspace(min_exp, max_exp, num_steps)

    scales_set: set[float] = set()
    if use_metadata_prior and min_allowed <= prior_ratio <= max_allowed:
        scales_set.add(round(prior_ratio, 6))

    for exp in exponents:
        s = prior_ratio * (2.0 ** float(exp))
        if min_allowed <= s <= max_allowed:
            scales_set.add(round(float(s), 6))

    sorted_scales = sorted(scales_set)
    if not sorted_scales:
        sorted_scales = [round(float(prior_ratio), 6)]

    return sorted_scales
