"""Protocol definition for feature extraction backends."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lunarmatch.geometry.coordinate_mapping import CoordinateMapping
from lunarmatch.models.domain import KeypointSet


@runtime_checkable
class FeatureBackend(Protocol):
    """Protocol for classical or learned keypoint detection and descriptor extraction backends."""

    @property
    def name(self) -> str:
        """Return backend identifier name (e.g., 'orb', 'sift')."""
        ...

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        keypoint_mapping: CoordinateMapping | None = None,
    ) -> KeypointSet:
        """Detect keypoints and compute descriptors from an image.

        Args:
            image: 2D uint8 or float grayscale image array.
            mask: Optional 2D uint8 mask (nonzero values indicate valid pixels).
            keypoint_mapping: Optional 3x3 CoordinateMapping to transform working keypoint
                coordinates (x, y) back to original raster space.

        Returns:
            KeypointSet containing Nx2 float64 (x, y) coordinates mapped to original space,
            scales, angles, responses, and NxD descriptor array (or empty KeypointSet if none detected).
        """
        ...
