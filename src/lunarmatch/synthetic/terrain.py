"""Procedural lunar height field and analytical Lambertian hillshade generator."""
from __future__ import annotations

import cv2
import numpy as np


def generate_crater_heightfield(
    shape: tuple[int, int] = (256, 256),
    num_craters: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """Generate procedural 2D lunar crater heightfield array.

    Args:
        shape: (height, width) tuple of target heightfield array.
        num_craters: Number of random impact craters to carve into heightfield.
        seed: Random seed integer for reproducible procedural generation.

    Returns:
        2D float32 heightfield array.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Heightfield dimensions must be positive, got ({h}, {w})")

    rng = np.random.default_rng(seed)
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Base elevation with multiscale background topography
    z = np.sin(grid_x / 30.0) * np.cos(grid_y / 30.0) * 15.0 + 100.0

    # Add random craters
    for _ in range(num_craters):
        cx = rng.uniform(0.1 * w, 0.9 * w)
        cy = rng.uniform(0.1 * h, 0.9 * h)
        r = rng.uniform(10.0, max(12.0, min(h, w) / 4.0))
        depth = rng.uniform(15.0, 45.0)

        dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

        # Crater floor depression
        floor_mask = dist < r
        z[floor_mask] -= depth * (1.0 - (dist[floor_mask] / r) ** 2)

        # Raised ejecta rim (r <= dist <= 1.25 * r)
        rim_mask = (dist >= r) & (dist <= 1.25 * r)
        rim_factor = 1.0 - (dist[rim_mask] - r) / (0.25 * r)
        z[rim_mask] += 0.2 * depth * (rim_factor**2)

    # Add regolith surface micro-roughness
    noise = rng.normal(0.0, 1.5, size=(h, w)).astype(np.float32)
    noise_blur = cv2.GaussianBlur(noise, (5, 5), 1.0)
    z += noise_blur

    res_z: np.ndarray = z.astype(np.float32)
    return res_z


def render_hillshade(
    heightfield: np.ndarray,
    azimuth_deg: float = 315.0,
    elevation_deg: float = 45.0,
    altitude_scale: float = 1.0,
) -> np.ndarray:
    """Render Lambertian analytical hillshade from elevation heightfield.

    Args:
        heightfield: 2D numeric heightfield array.
        azimuth_deg: Sun azimuth angle in degrees (0..360, 315 = NW).
        elevation_deg: Sun elevation angle in degrees (0..90, 45 = mid-day).
        altitude_scale: Vertical elevation multiplier for terrain slope contrast.

    Returns:
        2D uint8 hillshade image array [0, 255].
    """
    z = np.asarray(heightfield, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Heightfield must be a 2D array, got shape {z.shape}")

    # Compute surface gradients using Sobel operators
    dz_dx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * altitude_scale
    dz_dy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * altitude_scale

    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)

    # Convert sun angles to radians
    az_rad = np.radians(360.0 - azimuth_deg + 90.0)
    el_rad = np.radians(elevation_deg)

    # Lambertian shading equation
    shaded = np.sin(el_rad) * np.cos(slope) + np.cos(el_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    shaded = np.clip(shaded, 0.0, 1.0)

    res_hs: np.ndarray = (shaded * 255.0).astype(np.uint8)
    return res_hs
