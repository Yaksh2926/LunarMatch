"""Chandrayaan-2 Imaging InfraRed Spectrometer (IIRS) Sensor Adapter & Hyperspectral Module."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from lunarmatch.data.raster import read_raster, read_raster_metadata
from lunarmatch.sensors.base import SensorAdapter, SensorProduct


def select_band(cube: np.ndarray, band_idx: int = 0) -> np.ndarray:
    """Extract a single 2D spatial band from a 3D hyperspectral cube (H, W, B) or (B, H, W)."""
    if cube.ndim == 2:
        return cube.astype(np.float32)
    if cube.ndim == 3:
        if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:  # (B, H, W)
            idx = min(max(0, band_idx), cube.shape[0] - 1)
            return cube[idx, :, :].astype(np.float32)
        else:  # (H, W, B)
            idx = min(max(0, band_idx), cube.shape[2] - 1)
            return cube[:, :, idx].astype(np.float32)
    raise ValueError(f"Unsupported cube dimensions: {cube.shape}")


def average_bands(cube: np.ndarray, band_indices: list[int] | None = None) -> np.ndarray:
    """Average specified spectral bands across a 3D hyperspectral cube."""
    if cube.ndim == 2:
        return cube.astype(np.float32)

    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:  # (B, H, W)
        bands = cube[band_indices, :, :] if band_indices else cube
        return np.mean(bands, axis=0).astype(np.float32)
    else:  # (H, W, B)
        bands = cube[:, :, band_indices] if band_indices else cube
        return np.mean(bands, axis=2).astype(np.float32)


def weighted_band_composite(cube: np.ndarray, weights: np.ndarray | list[float]) -> np.ndarray:
    """Compute a weighted composite image across spectral bands."""
    if cube.ndim == 2:
        return cube.astype(np.float32)

    w = np.asarray(weights, dtype=np.float32)
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:  # (B, H, W)
        assert len(w) == cube.shape[0], f"Weights length {len(w)} must match band count {cube.shape[0]}"
        composite = np.tensordot(w, cube, axes=(0, 0))
    else:  # (H, W, B)
        assert len(w) == cube.shape[2], f"Weights length {len(w)} must match band count {cube.shape[2]}"
        composite = np.tensordot(cube, w, axes=(2, 0))

    return composite.astype(np.float32)


def pca_representation(cube: np.ndarray) -> np.ndarray:
    """Extract first principal component image representation from hyperspectral data cube."""
    if cube.ndim == 2:
        return cube.astype(np.float32)

    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:  # (B, H, W)
        b, h, w = cube.shape
        pixels = cube.reshape(b, h * w).T  # (H*W, B)
    else:  # (H, W, B)
        h, w, b = cube.shape
        pixels = cube.reshape(h * w, b)  # (H*W, B)

    # Mean center
    pixels_centered = pixels - np.mean(pixels, axis=0)
    cov = np.cov(pixels_centered, rowvar=False)

    evals, evecs = np.linalg.eigh(cov)
    pc1_vec = evecs[:, np.argmax(evals)]
    pc1_img = np.dot(pixels_centered, pc1_vec).reshape(h, w)

    return cast(np.ndarray, pc1_img.astype(np.float32))


class IIRSAdapter(SensorAdapter):
    """Adapter for Chandrayaan-2 IIRS (~80.0 m/px) hyperspectral products."""

    def can_handle(self, path: str | Path) -> bool:
        p = Path(path)
        name_lower = p.name.lower()
        if "iirs" in name_lower or "ch2_iirs" in name_lower:
            return True
        if p.suffix.lower() == ".xml":
            try:
                tree = ET.parse(p)
                root = tree.getroot()
                text = ET.tostring(root, encoding='utf-8').decode('utf-8', errors='ignore').lower()
                return "iirs" in text or "infrared spectrometer" in text
            except (ET.ParseError, OSError, ValueError):
                pass
        return False

    def load_product(
        self,
        path: str | Path,
        band_strategy: Literal["single", "average", "pca"] = "single",
        band_index: int = 0,
        **kwargs: Any,
    ) -> SensorProduct:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"IIRS product file not found: {p}")

        ras_meta = read_raster_metadata(p)
        read_res = read_raster(p)

        data = read_res.data
        mask = read_res.valid_mask

        # Apply hyperspectral reduction if 3D
        if data.ndim == 3:
            if band_strategy == "single":
                img_2d = select_band(data, band_index)
            elif band_strategy == "average":
                img_2d = average_bands(data)
            elif band_strategy == "pca":
                img_2d = pca_representation(data)
            else:
                img_2d = select_band(data, 0)
        else:
            img_2d = data

        gsd_x = ras_meta.pixel_scale_x or 80.0
        gsd_y = ras_meta.pixel_scale_y or 80.0

        prov = {
            "source_path": str(p),
            "native_dimensions": (ras_meta.width, ras_meta.height),
            "sensor": "IIRS",
            "default_gsd_m": 80.0,
            "band_strategy_applied": band_strategy,
            "band_index_used": band_index,
            "original_dtype": ras_meta.dtype,
        }

        return SensorProduct(
            sensor_name="IIRS",
            image_data=img_2d,
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
