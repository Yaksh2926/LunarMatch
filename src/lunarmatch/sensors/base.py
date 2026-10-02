"""Base classes and data containers for lunar sensor adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SensorProduct:
    """Standardized sensor product container for multi-modal lunar imagery."""

    sensor_name: str  # "OHRC", "TMC", "IIRS", "LROC", "GENERIC"
    image_data: np.ndarray  # 2D float32 or uint8 normalized array
    valid_mask: np.ndarray  # 2D boolean valid pixels mask
    gsd_x: float | None = None  # Pixel scale X in meters
    gsd_y: float | None = None  # Pixel scale Y in meters
    acquisition_time: str | None = None
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    crs_wkt: str | None = None
    transform_affine: tuple[float, ...] | None = None
    metadata_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Width of the image in pixels."""
        return int(self.image_data.shape[1])

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        return int(self.image_data.shape[0])

    @property
    def mean_gsd(self) -> float | None:
        """Mean Ground Sample Distance (GSD) in meters per pixel."""
        if self.gsd_x is not None and self.gsd_y is not None:
            return float((abs(self.gsd_x) + abs(self.gsd_y)) / 2.0)
        return self.gsd_x or self.gsd_y

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata properties to dictionary."""
        return {
            "sensor_name": self.sensor_name,
            "width": self.width,
            "height": self.height,
            "gsd_x": self.gsd_x,
            "gsd_y": self.gsd_y,
            "mean_gsd": self.mean_gsd,
            "acquisition_time": self.acquisition_time,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "sun_elevation_deg": self.sun_elevation_deg,
            "crs_wkt": self.crs_wkt,
            "transform_affine": list(self.transform_affine) if self.transform_affine else None,
            "metadata_provenance": self.metadata_provenance,
        }


class SensorAdapter(ABC):
    """Abstract base class for lunar sensor data adapters."""

    @abstractmethod
    def can_handle(self, path: str | Path) -> bool:
        """Check if this adapter can handle the given file or directory path."""

    @abstractmethod
    def load_product(self, path: str | Path, **kwargs: Any) -> SensorProduct:
        """Load native product and return a standardized SensorProduct container."""
