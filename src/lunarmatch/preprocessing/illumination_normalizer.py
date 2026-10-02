"""Illumination and shadow geometry normalization using solar zenith and azimuth angles."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon, box  # type: ignore[import-untyped]

from lunarmatch.models.config import PreprocessConfig
from lunarmatch.models.domain import RasterMetadata


@dataclass(frozen=True)
class IlluminationAlignmentResult:
    """Result of sun-angle illumination offset calculation between source and reference rasters."""

    offset_dx: float
    offset_dy: float
    footprint_overlap_fraction: float
    uncorrected_overlap_fraction: float
    alignment_applied: bool
    provenance: dict[str, Any]

    @property
    def shadow_offset_x_m(self) -> float:
        """Alias for backward compatibility."""
        return self.offset_dx

    @property
    def shadow_offset_y_m(self) -> float:
        """Alias for backward compatibility."""
        return self.offset_dy

    def to_dict(self) -> dict[str, Any]:
        """Serialize illumination alignment result to dictionary."""
        return {
            "shadow_offset_x_m": self.offset_dx,
            "shadow_offset_y_m": self.offset_dy,
            "footprint_overlap_fraction": self.footprint_overlap_fraction,
            "uncorrected_overlap_fraction": self.uncorrected_overlap_fraction,
            "alignment_applied": self.alignment_applied,
            "provenance": self.provenance,
        }


def compute_footprint_overlap(
    poly1: Polygon | Sequence[tuple[float, float]],
    poly2: Polygon | Sequence[tuple[float, float]],
) -> float:
    """Compute normalized intersection over union (IoU) between two 2D bounding polygons."""
    p1 = Polygon(poly1) if not isinstance(poly1, Polygon) else poly1
    p2 = Polygon(poly2) if not isinstance(poly2, Polygon) else poly2

    if p1.is_empty or p2.is_empty:
        return 0.0
    try:
        intersection = p1.intersection(p2).area
        min_area = min(p1.area, p2.area)
        if min_area <= 0.0:
            return 0.0
        return float(intersection / min_area)
    except (ValueError, RuntimeError, AttributeError):
        return 0.0


def compute_illumination_shadow_offset(
    *args: Any,
    **kwargs: Any,
) -> tuple[float, float]:
    """Compute relative shadow vector shift (dx_meters, dy_meters) from solar azimuth & elevation parameters.

    Can be called with either:
    1. Four float angles: (azimuth_src, elevation_src, azimuth_ref, elevation_ref)
    2. Two RasterMetadata objects: (source_meta, reference_meta)
    """
    if len(args) == 4 and all(isinstance(a, (int, float)) for a in args):
        src_az_deg, src_el_deg, ref_az_deg, ref_el_deg = [float(a) for a in args]
    elif len(args) == 2 and hasattr(args[0], "sun_azimuth_deg") and hasattr(args[1], "sun_azimuth_deg"):
        src_meta, ref_meta = args[0], args[1]
        if src_meta.sun_azimuth_deg is None or ref_meta.sun_azimuth_deg is None:
            return 0.0, 0.0
        src_az_deg = float(src_meta.sun_azimuth_deg)
        ref_az_deg = float(ref_meta.sun_azimuth_deg)
        src_el_deg = float(src_meta.sun_elevation_deg if src_meta.sun_elevation_deg is not None else 45.0)
        ref_el_deg = float(ref_meta.sun_elevation_deg if ref_meta.sun_elevation_deg is not None else 45.0)
    else:
        src_az_deg = float(kwargs.get("az_src", 0.0))
        src_el_deg = float(kwargs.get("el_src", 45.0))
        ref_az_deg = float(kwargs.get("az_ref", 0.0))
        ref_el_deg = float(kwargs.get("el_ref", 45.0))

    src_az = np.radians(src_az_deg)
    ref_az = np.radians(ref_az_deg)

    src_el = np.radians(src_el_deg)
    ref_el = np.radians(ref_el_deg)

    # Relative shadow length and direction vector projected onto local lunar tangent plane (meters per 100m feature height)
    dx = float(np.cos(src_az) / np.tan(src_el) - np.cos(ref_az) / np.tan(ref_el)) * 100.0
    dy = float(np.sin(src_az) / np.tan(src_el) - np.sin(ref_az) / np.tan(ref_el)) * 100.0

    return dx, dy


class IlluminationNormalizer:
    """Sun-angle illumination offset calculator and shadow geometry aligner."""

    def __init__(
        self,
        allow_illumination_correction: bool = False,
        config: PreprocessConfig | None = None,
    ) -> None:
        if config is not None:
            self.allow_illumination_correction = config.allow_illumination_correction
        else:
            self.allow_illumination_correction = allow_illumination_correction

    def align_footprints(
        self,
        source_meta: RasterMetadata,
        reference_meta: RasterMetadata,
    ) -> IlluminationAlignmentResult:
        """Align source and reference footprints accounting for sun-angle shadow displacement."""
        src_w, src_h = float(source_meta.width), float(source_meta.height)
        ref_w, ref_h = float(reference_meta.width), float(reference_meta.height)

        src_poly = box(0.0, 0.0, src_w, src_h)
        ref_poly = box(0.0, 0.0, ref_w, ref_h)

        uncorrected_overlap = compute_footprint_overlap(src_poly, ref_poly)

        if not self.allow_illumination_correction:
            return IlluminationAlignmentResult(
                offset_dx=0.0,
                offset_dy=0.0,
                footprint_overlap_fraction=uncorrected_overlap,
                uncorrected_overlap_fraction=uncorrected_overlap,
                alignment_applied=False,
                provenance={"reason": "illumination_correction_disabled"},
            )

        dx, dy = compute_illumination_shadow_offset(source_meta, reference_meta)
        shifted_src_poly = box(dx, dy, src_w + dx, src_h + dy)
        corrected_overlap = compute_footprint_overlap(shifted_src_poly, ref_poly)

        return IlluminationAlignmentResult(
            offset_dx=dx,
            offset_dy=dy,
            footprint_overlap_fraction=corrected_overlap,
            uncorrected_overlap_fraction=uncorrected_overlap,
            alignment_applied=True,
            provenance={
                "source_sensor": getattr(source_meta, "sensor", "unknown"),
                "reference_sensor": getattr(reference_meta, "sensor", "unknown"),
                "shadow_dx_m": dx,
                "shadow_dy_m": dy,
            },
        )

    def normalize(
        self,
        image: np.ndarray,
        source_meta: RasterMetadata,
        reference_meta: RasterMetadata,
    ) -> tuple[np.ndarray, IlluminationAlignmentResult]:
        """Normalize raster image and compute shadow alignment result."""
        res = self.align_footprints(source_meta, reference_meta)
        return image, res
