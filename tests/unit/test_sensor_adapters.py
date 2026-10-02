"""Unit test suite for SensorAdapter abstraction and Chandrayaan-2 sensor loaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lunarmatch.sensors.base import SensorProduct
from lunarmatch.sensors.factory import (
    GenericRasterAdapter,
    detect_sensor_adapter,
)
from lunarmatch.sensors.iirs import IIRSAdapter, average_bands, pca_representation, select_band
from lunarmatch.sensors.ohrc import OHRCAdapter
from lunarmatch.sensors.tmc import TMCAdapter


def test_sensor_product_properties() -> None:
    """Test SensorProduct container calculations and serialization."""
    img = np.zeros((100, 200), dtype=np.uint8)
    mask = np.ones((100, 200), dtype=bool)
    prod = SensorProduct(
        sensor_name="OHRC",
        image_data=img,
        valid_mask=mask,
        gsd_x=0.25,
        gsd_y=0.25,
        acquisition_time="2023-03-03T03:50:44Z",
        sun_azimuth_deg=45.0,
        sun_elevation_deg=10.0,
    )

    assert prod.width == 200
    assert prod.height == 100
    assert prod.mean_gsd == 0.25
    d = prod.to_dict()
    assert d["sensor_name"] == "OHRC"
    assert d["mean_gsd"] == 0.25


def test_iirs_spectral_processing() -> None:
    """Test hyperspectral 3D data cube reduction functions."""
    # (B, H, W) cube = (5, 40, 40)
    cube_bhw = np.random.uniform(0, 100, (5, 40, 40)).astype(np.float32)

    s_band = select_band(cube_bhw, 2)
    assert s_band.shape == (40, 40)

    a_band = average_bands(cube_bhw)
    assert a_band.shape == (40, 40)

    pca_img = pca_representation(cube_bhw)
    assert pca_img.shape == (40, 40)


def test_detect_sensor_adapter() -> None:
    """Test automatic sensor adapter detection based on path naming conventions."""
    ohrc_path = Path("sample_data/real/ohrc/ch2_ohr_ncp_20230303T0350447888_d_img_n18.xml")
    adapter = detect_sensor_adapter(ohrc_path)
    assert isinstance(adapter, OHRCAdapter)

    tmc_path = Path("sample_data/ch2_tmc_ncp_20230303.tif")
    adapter_tmc = detect_sensor_adapter(tmc_path)
    assert isinstance(adapter_tmc, TMCAdapter)

    iirs_path = Path("sample_data/ch2_iirs_ncp_20230303.tif")
    adapter_iirs = detect_sensor_adapter(iirs_path)
    assert isinstance(adapter_iirs, IIRSAdapter)

    generic_path = Path("sample_data/reference_moon.tif")
    adapter_gen = detect_sensor_adapter(generic_path)
    assert isinstance(adapter_gen, GenericRasterAdapter)
