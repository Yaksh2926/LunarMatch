"""Sensor abstraction layer for Chandrayaan-2 and lunar reference optical imagery."""
from __future__ import annotations

from lunarmatch.sensors.base import SensorAdapter, SensorProduct
from lunarmatch.sensors.factory import (
    GenericRasterAdapter,
    detect_sensor_adapter,
    load_sensor_product,
)
from lunarmatch.sensors.iirs import IIRSAdapter
from lunarmatch.sensors.ohrc import OHRCAdapter
from lunarmatch.sensors.tmc import TMCAdapter

__all__ = [
    "GenericRasterAdapter",
    "IIRSAdapter",
    "OHRCAdapter",
    "SensorAdapter",
    "SensorProduct",
    "TMCAdapter",
    "detect_sensor_adapter",
    "load_sensor_product",
]
