"""Synthetic lunar benchmark generator package."""
from lunarmatch.synthetic.generator import (
    SyntheticPairData,
    generate_synthetic_benchmark_dataset,
    generate_synthetic_pair,
)
from lunarmatch.synthetic.terrain import generate_crater_heightfield, render_hillshade

__all__ = [
    "SyntheticPairData",
    "generate_crater_heightfield",
    "generate_synthetic_benchmark_dataset",
    "generate_synthetic_pair",
    "render_hillshade",
]
