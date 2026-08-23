"""Matching package for feature correspondences and spatial filtering."""
from lunarmatch.matching.grid import GridPartition
from lunarmatch.matching.knn import compute_descriptor_distances
from lunarmatch.matching.matcher import DescriptorMatcher, MatchResult, match_descriptors
from lunarmatch.matching.spatial_selector import (
    SpatialSelectionResult,
    SpatialSelector,
    select_uniform_matches,
)

__all__ = [
    "DescriptorMatcher",
    "GridPartition",
    "MatchResult",
    "SpatialSelectionResult",
    "SpatialSelector",
    "compute_descriptor_distances",
    "match_descriptors",
    "select_uniform_matches",
]
