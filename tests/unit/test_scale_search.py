"""Unit tests for scale-hypothesis generation."""
from __future__ import annotations

from lunarmatch.features import generate_scale_hypotheses
from lunarmatch.models.config import PyramidConfig


def test_metadata_scale_prior_hypothesis_generation() -> None:
    """Test scale hypotheses generation when source and reference pixel scales exist."""
    cfg = PyramidConfig(metadata_scale_prior=True, log2_scale_min=-1, log2_scale_max=1, levels_per_octave=1)
    # Source scale 0.5m/px, reference 2.0m/px -> ratio 4.0
    scales = generate_scale_hypotheses(source_scale_m=0.5, reference_scale_m=2.0, config=cfg)

    # Acceptance check: scale ordering is deterministic (sorted)
    assert scales == sorted(scales)
    assert 4.0 in scales
    assert 2.0 in scales  # 4.0 * 2^-1
    assert 8.0 in scales  # 4.0 * 2^1


def test_logarithmic_scale_search_without_metadata() -> None:
    """Test bounded logarithmic scale search when metadata pixel scales are absent."""
    cfg = PyramidConfig(metadata_scale_prior=False, log2_scale_min=-2, log2_scale_max=2, levels_per_octave=1)
    scales = generate_scale_hypotheses(source_scale_m=None, reference_scale_m=None, config=cfg)

    assert scales == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert scales == sorted(scales)


def test_allowed_scale_filtering() -> None:
    """Test scale hypotheses are filtered by allowed_scale range."""
    cfg = PyramidConfig(metadata_scale_prior=False, log2_scale_min=-5, log2_scale_max=5, levels_per_octave=1)
    allowed = (0.5, 8.0)
    scales = generate_scale_hypotheses(config=cfg, allowed_scale=allowed)

    assert all(0.5 <= s <= 8.0 for s in scales)
    assert 0.25 not in scales
    assert 16.0 not in scales
