"""Chandrayaan-2 Terrain Mapping Camera (TMC) Sensor Adapter."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from lunarmatch.data.raster import read_raster, read_raster_metadata
from lunarmatch.sensors.base import SensorAdapter, SensorProduct


class TMCAdapter(SensorAdapter):
    """Adapter for Chandrayaan-2 TMC (5.0 m/px) stereo optical products."""

    def can_handle(self, path: str | Path) -> bool:
        p = Path(path)
        name_lower = p.name.lower()
        if "tmc" in name_lower or "ch2_tmc" in name_lower:
            return True
        if p.suffix.lower() == ".xml":
            try:
                tree = ET.parse(p)
                root = tree.getroot()
                text = ET.tostring(root, encoding='utf-8').decode('utf-8', errors='ignore').lower()
                return "tmc" in text or "terrain mapping camera" in text
            except (ET.ParseError, OSError, ValueError):
                pass
        return False

    def load_product(self, path: str | Path, **kwargs: Any) -> SensorProduct:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TMC product file not found: {p}")

        ras_meta = read_raster_metadata(p)
        read_res = read_raster(p)

        data = read_res.data
        mask = read_res.valid_mask

        gsd_x = ras_meta.pixel_scale_x if (ras_meta.pixel_scale_x and ras_meta.pixel_scale_x >= 0.1) else 5.0
        gsd_y = ras_meta.pixel_scale_y if (ras_meta.pixel_scale_y and ras_meta.pixel_scale_y >= 0.1) else 5.0

        prov = {
            "source_path": str(p),
            "native_dimensions": (ras_meta.width, ras_meta.height),
            "sensor": "TMC",
            "default_gsd_m": 5.0,
            "original_dtype": ras_meta.dtype,
        }

        return SensorProduct(
            sensor_name="TMC",
            image_data=data,
            valid_mask=mask,
            gsd_x=float(gsd_x),
            gsd_y=float(gsd_y),
            acquisition_time=ras_meta.acquisition_time,
            sun_azimuth_deg=ras_meta.sun_azimuth_deg,
            sun_elevation_deg=ras_meta.sun_elevation_deg,
            crs_wkt=ras_meta.crs,
            transform_affine=ras_meta.transform,
            metadata_provenance=prov,
        )
