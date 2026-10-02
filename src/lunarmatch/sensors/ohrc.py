"""Chandrayaan-2 Optical High Resolution Camera (OHRC) Sensor Adapter."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from lunarmatch.data.raster import read_raster, read_raster_metadata
from lunarmatch.sensors.base import SensorAdapter, SensorProduct


def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag


def parse_ohrc_pds4_xml(xml_path: str | Path) -> dict[str, Any]:
    """Parse Chandrayaan-2 OHRC PDS4 XML label metadata."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta: dict[str, Any] = {}

    corners: dict[str, float] = {}
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in [
            'upper_left_latitude', 'upper_left_longitude',
            'upper_right_latitude', 'upper_right_longitude',
            'lower_left_latitude', 'lower_left_longitude',
            'lower_right_latitude', 'lower_right_longitude'
        ]:
            try:
                corners[tag] = float(elem.text) if elem.text else 0.0
            except ValueError:
                pass
        elif tag == 'start_date_time' and elem.text:
            meta['acquisition_time'] = elem.text.strip()
        elif tag in ['sun_azimuth', 'sun_azimuth_deg'] and elem.text:
            try:
                meta['sun_azimuth_deg'] = float(elem.text.strip())
            except ValueError:
                pass
        elif tag in ['sun_elevation', 'sun_elevation_deg', 'solar_elevation'] and elem.text:
            try:
                meta['sun_elevation_deg'] = float(elem.text.strip())
            except ValueError:
                pass

    meta['corners'] = corners
    return meta


class OHRCAdapter(SensorAdapter):
    """Adapter for Chandrayaan-2 OHRC (0.25 m/px) optical products."""

    def can_handle(self, path: str | Path) -> bool:
        p = Path(path)
        name_lower = p.name.lower()
        if "ohr" in name_lower or "ch2_ohr" in name_lower:
            return True
        if p.suffix.lower() == ".xml":
            try:
                tree = ET.parse(p)
                root = tree.getroot()
                text = ET.tostring(root, encoding='utf-8').decode('utf-8', errors='ignore').lower()
                return "ohrc" in text or "ch2_ohr" in text or "high resolution camera" in text
            except (ET.ParseError, OSError, ValueError):
                pass
        return False

    def load_product(self, path: str | Path, **kwargs: Any) -> SensorProduct:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OHRC product file not found: {p}")

        # Metadata parsing
        xml_meta = {}
        if p.suffix.lower() == ".xml":
            xml_meta = parse_ohrc_pds4_xml(p)

        ras_meta = read_raster_metadata(p)
        read_res = read_raster(p)

        data = read_res.data
        mask = read_res.valid_mask

        gsd_x = ras_meta.pixel_scale_x if (ras_meta.pixel_scale_x and ras_meta.pixel_scale_x != 1.0) else 0.25
        gsd_y = ras_meta.pixel_scale_y if (ras_meta.pixel_scale_y and ras_meta.pixel_scale_y != 1.0) else 0.25
        acq_time = xml_meta.get("acquisition_time") or ras_meta.acquisition_time
        sun_az = xml_meta.get("sun_azimuth_deg") or ras_meta.sun_azimuth_deg
        sun_el = xml_meta.get("sun_elevation_deg") or ras_meta.sun_elevation_deg

        prov = {
            "source_path": str(p),
            "native_dimensions": (ras_meta.width, ras_meta.height),
            "corners": xml_meta.get("corners", {}),
            "original_dtype": ras_meta.dtype,
        }

        return SensorProduct(
            sensor_name="OHRC",
            image_data=data,
            valid_mask=mask,
            gsd_x=float(gsd_x),
            gsd_y=float(gsd_y),
            acquisition_time=acq_time,
            sun_azimuth_deg=sun_az,
            sun_elevation_deg=sun_el,
            crs_wkt=ras_meta.crs,
            transform_affine=ras_meta.transform,
            metadata_provenance=prov,
        )
