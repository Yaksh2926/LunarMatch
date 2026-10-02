"""Abstract BaseMatcher interface and SIFTMatcher adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lunarmatch.matching.matcher import DescriptorMatcher, MatchResult
from lunarmatch.models.config import MatchingConfig
from lunarmatch.models.domain import KeypointSet


class BaseMatcher(ABC):
    """Abstract base class for all feature matchers in LunarMatch."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return unique string name of the matcher."""

    @abstractmethod
    def match(
        self,
        source: KeypointSet | np.ndarray,
        reference: KeypointSet | np.ndarray,
        source_points: np.ndarray | None = None,
        reference_points: np.ndarray | None = None,
    ) -> MatchResult:
        """Find correspondences between source and reference feature representations."""


class SIFTMatcher(BaseMatcher):
    """Classical SIFT descriptor matcher wrapping DescriptorMatcher."""

    def __init__(self, config: MatchingConfig | None = None) -> None:
        self._matcher = DescriptorMatcher(config=config)

    @property
    def name(self) -> str:
        return "sift_classical"

    def match(
        self,
        source: KeypointSet | np.ndarray,
        reference: KeypointSet | np.ndarray,
        source_points: np.ndarray | None = None,
        reference_points: np.ndarray | None = None,
    ) -> MatchResult:
        return self._matcher.match(source, reference, source_points=source_points, reference_points=reference_points)
