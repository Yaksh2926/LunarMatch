"""Unit tests for scale-hypothesis generation."""
from __future__ import annotations

from lunarmatch.features import generate_scale_hypotheses
from lunarmatch.models.config import PyramidConfig


def test_metadata_scale_prior_hypothesis_generation() -> None:
    """Test scale hypotheses generation when source and reference pixel scales exist."""
    cfg = PyramidConfig(metadata_scale_prior=True, log2_scale_min=-1, log2_scale_max=1, levels_per_octave=1)
    # Source scale 0.5m/px, reference 2.0m/px -> ratio 0.25
    scales = generate_scale_hypotheses(source_scale_m=0.5, reference_scale_m=2.0, config=cfg)

    # Acceptance check: scale ordering is deterministic (sorted)
    assert scales == sorted(scales)
    assert 0.25 in scales
    assert 0.125 in scales  # 0.25 * 2^-1
    assert 0.5 in scales  # 0.25 * 2^1


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


def test_real_gsd_hypotheses_generation() -> None:
    """Verify that source GSD 0.25 and reference GSD 1.535 generates hypothesis near 0.162866."""
    cfg = PyramidConfig(metadata_scale_prior=True, log2_scale_min=-4, log2_scale_max=4, levels_per_octave=2)
    scales = generate_scale_hypotheses(source_scale_m=0.25, reference_scale_m=1.535, config=cfg)

    # The exact value expected: 0.25 / 1.535 = 0.1628664495...
    # Generates prior_ratio * 2.0**0 = 0.162866
    expected_ratio = 0.25 / 1.535
    assert any(abs(s - expected_ratio) < 1e-5 for s in scales)
    # Checks that 0.162866 is in scales (rounded to 6 decimal places: 0.162866)
    assert 0.162866 in scales

