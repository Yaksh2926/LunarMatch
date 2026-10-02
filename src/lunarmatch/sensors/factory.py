"""Automatic sensor identification and adapter factory for multi-modal lunar datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from lunarmatch.data.raster import read_raster, read_raster_metadata
from lunarmatch.sensors.base import SensorAdapter, SensorProduct
from lunarmatch.sensors.iirs import IIRSAdapter
from lunarmatch.sensors.ohrc import OHRCAdapter
from lunarmatch.sensors.tmc import TMCAdapter


class GenericRasterAdapter(SensorAdapter):
    """Fallback adapter for generic lunar optical rasters (e.g., LROC NAC/WAC)."""

    def can_handle(self, path: str | Path) -> bool:
        return True

    def load_product(self, path: str | Path, **kwargs: Any) -> SensorProduct:
        p = Path(path)
        ras_meta = read_raster_metadata(p)
        read_res = read_raster(p)

        sensor = ras_meta.sensor or "GENERIC"
        if "nac" in p.name.lower() or "lroc" in p.name.lower():
            sensor = "LROC"

        gsd_x = ras_meta.pixel_scale_x or 1.0
        gsd_y = ras_meta.pixel_scale_y or 1.0

        prov = {
            "source_path": str(p),
            "native_dimensions": (ras_meta.width, ras_meta.height),
            "sensor": sensor,
            "original_dtype": ras_meta.dtype,
        }

        return SensorProduct(
            sensor_name=sensor,
            image_data=read_res.data,
            valid_mask=read_res.valid_mask,
            gsd_x=float(gsd_x),
            gsd_y=float(gsd_y),
            acquisition_time=ras_meta.acquisition_time,
            sun_azimuth_deg=ras_meta.sun_azimuth_deg,
            sun_elevation_deg=ras_meta.sun_elevation_deg,
            crs_wkt=ras_meta.crs,
            transform_affine=ras_meta.transform,
            metadata_provenance=prov,
        )


def detect_sensor_adapter(path: str | Path) -> SensorAdapter:
    """Automatically detect and return the appropriate SensorAdapter for a product file."""
    p = Path(path)
    adapters: list[SensorAdapter] = [
        OHRCAdapter(),
        TMCAdapter(),
        IIRSAdapter(),
    ]
    for adapter in adapters:
        if adapter.can_handle(p):
            return adapter

    return GenericRasterAdapter()


def load_sensor_product(path: str | Path, sensor_name: str | None = None, **kwargs: Any) -> SensorProduct:
    """Load a sensor product either automatically or using an explicitly specified sensor name."""
    if sensor_name:
        s_upper = sensor_name.upper()
        if s_upper == "OHRC":
            return OHRCAdapter().load_product(path, **kwargs)
        elif s_upper == "TMC":
            return TMCAdapter().load_product(path, **kwargs)
        elif s_upper == "IIRS":
            return IIRSAdapter().load_product(path, **kwargs)
        elif s_upper in ["LROC", "GENERIC"]:
            return GenericRasterAdapter().load_product(path, **kwargs)

    adapter = detect_sensor_adapter(path)
    return adapter.load_product(path, **kwargs)
