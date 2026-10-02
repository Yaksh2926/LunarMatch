# Implementation Plan: ISRO Problem Statement 26166 (Chandrayaan-2 Multi-Sensor Support)

## Executive Summary

This document establishes the technical implementation plan for expanding **LunarMatch** to support the full scope of **ISRO Problem Statement 26166**:

> **"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"**

The existing core registration pipeline is **100% frozen and verified** (157/157 pytest tests pass, Ruff 0 errors, MyPy 0 type errors). The new sensor-aware architecture is implemented via modular wrapper layers, adapters, and synthetic test fixtures without destabilizing the frozen core.

---

## 1. Current Architecture Map

The core system processes rasters through 10 sequential pipeline stages:

```
[Input Product / File]
          │
          ▼
[Stage 1: Raster I/O] (lunarmatch.data.raster)
          │
          ▼
[Stage 2: Preprocessing] (lunarmatch.preprocessing)
          │
          ▼
[Stage 2.5: Coarse Terrain Overlap Check] (lunarmatch.matching.terrain_check)
          │
          ▼
[Stage 3 & 4: Pyramid Planning & Scale Hypotheses] (lunarmatch.features.pyramid, scale_search)
          │
          ▼
[Stage 5 & 6: Tiled Feature Extraction & Matching] (lunarmatch.features, matching)
          │
          ▼
[Stage 7 & 8: Spatial Selection & Geometric Verification] (lunarmatch.matching.spatial_selector, geometry.verifier)
          │
          ▼
[Stage 9: Sub-Pixel Patch Refinement] (lunarmatch.geometry.subpixel)
          │
          ▼
[Stage 10: Export & Manifest] (lunarmatch.export)
```

---

## 2. Existing Reusable Components

The following production modules in `src/lunarmatch` are fully reusable and form the core registration engine:

1. **`lunarmatch.data.raster`**: Memory-budgeted windowed raster reader using `rasterio`/`tifffile`.
2. **`lunarmatch.preprocessing`**: CLAHE, Sobel gradient, contrast normalization, phase representations.
3. **`lunarmatch.features.tiling`**: Grid-partitioned keypoint extraction preventing spatial clustering.
4. **`lunarmatch.features.pyramid`**: Dynamic multi-resolution anti-aliased Gaussian pyramids.
5. **`lunarmatch.features.scale_search`**: Metadata-centered scale octave search space planner.
6. **`lunarmatch.matching.spatial_selector`**: Convex hull coverage area & grid cell density metrics.
7. **`lunarmatch.geometry.verifier`**: Per-hypothesis RANSAC / USAC_MAGSAC, inlier count, inlier ratio, RMSE.
8. **`lunarmatch.geometry.subpixel`**: ECC patch alignment & parabolic correlation peak interpolation.
9. **`lunarmatch.export`**: GeoTIFF warping, GeoJSON match points, JSON run manifest.

---

## 3. Modules That Need Extension

To satisfy the full ISRO PS 26166 specification without touching frozen core files, we introduce new extensions:

1. **`src/lunarmatch/sensors/` (NEW MODULE)**:
   - `base.py`: Abstract `SensorAdapter` base class and `SensorProduct` data container.
   - `ohrc.py`: `OHRCAdapter` wrapping native Chandrayaan-2 OHRC PDS4 XML metadata and rasters.
   - `tmc.py`: `TMCAdapter` for Chandrayaan-2 Terrain Mapping Camera (TMC) stereo imagery.
   - `iirs.py`: `IIRSAdapter` & hyperspectral band selection module for Imaging InfraRed Spectrometer (IIRS) data.
   - `factory.py`: Automatic sensor detection & adapter instantiation factory.

2. **`src/lunarmatch/matching/spatial_selector.py` & `export/writer.py`**:
   - Expose spatial uniformity metrics (`occupied_grid_fraction`, `count_uniformity_cv`, `convex_hull_coverage_fraction`) explicitly in exported `RunManifest` JSON files.

3. **`src/lunarmatch/evaluation/`**:
   - Add synthetic sub-pixel benchmark evaluator with known fractional pixel displacements ($\text{dx}=3.37\text{ px}, \text{dy}=-1.82\text{ px}$) to verify sub-pixel accuracy.

4. **`configs/`**:
   - `multimodal_default.yaml`, `ohrc_to_tmc.yaml`, `ohrc_to_iirs.yaml`, `tmc_to_reference.yaml`, `iirs_to_reference.yaml`.

---

## 4. Proposed `SensorAdapter` Architecture

```python
@dataclass(frozen=True)
class SensorProduct:
    sensor_name: str  # "OHRC", "TMC", "IIRS", "LROC", "GENERIC"
    image_data: np.ndarray  # 2D float32 or uint8 normalized array
    valid_mask: np.ndarray  # 2D boolean valid pixels mask
    gsd_x: float | None  # Pixel scale X in meters
    gsd_y: float | None  # Pixel scale Y in meters
    acquisition_time: str | None
    sun_azimuth_deg: float | None
    sun_elevation_deg: float | None
    crs_wkt: str | None
    transform_affine: tuple[float, ...] | None
    metadata_provenance: dict[str, Any]

class SensorAdapter(ABC):
    @abstractmethod
    def can_handle(self, path: str | Path) -> bool: ...

    @abstractmethod
    def load_product(self, path: str | Path, **kwargs: Any) -> SensorProduct: ...
```

---

## 5. TMC (Terrain Mapping Camera) Implementation Plan

- **Sensor Characteristics**: Chandrayaan-2 TMC provides 5 m/px GSD stereo imagery (Fore, Aft, Nadir strips).
- **Adapter Responsibilities**:
  - Parse TMC PDS label metadata (GSD 5.0 m/px, solar angles, orbit track).
  - Extract 2D raster channels and normalize dynamic range.
  - Scale Prior: Automatically configure GSD ratio relative to OHRC ($5.0 / 0.25 = 20.0$).
- **Test Strategy**: Unit tests with synthetic 5 m/px TMC-like fixtures and metadata XML mocks.

---

## 6. IIRS (Imaging InfraRed Spectrometer) Implementation Plan

- **Sensor Characteristics**: Chandrayaan-2 IIRS provides hyperspectral imagery across 250+ spectral bands ($0.8 - 5.0\ \mu\text{m}$) at ~80 m/px GSD.
- **Spectral Processing Architecture**:
  - `select_band(cube, band_index)`: Single-band extraction (e.g. 1.0 $\mu\text{m}$ continuum band).
  - `average_bands(cube, band_indices)`: Band range averaging to suppress noise.
  - `weighted_band_composite(cube, weights)`: Multi-spectral weighting.
  - `pca_representation(cube)`: First principal component extraction for maximum spatial variance.
- **Integration**: The spectral module converts 3D hyperspectral cubes into a normalized 2D raster suitable for the LunarMatch core.

---

## 7. Validation Strategy & Hierarchy

1. **Regression Baseline**: Preserved 157 pytest unit/integration tests must pass cleanly at all times.
2. **OHRC Real-Data Benchmark**: Preserve 100% verified OHRC-to-OHRC control registration (135 inliers, 83.3% ratio, 1.47 px RMSE).
3. **Synthetic Sub-Pixel Benchmark**: Test fractional shifts on synthetic fixtures and verify sub-1.0 px RMSE recovery.
4. **Synthetic TMC & IIRS Tests**: Test multi-sensor adapters using synthetic TMC (5 m/px) and IIRS (80 m/px) multi-scale fixtures.

---

## 8. Risks and Assumptions

- **Risk**: Real TMC and IIRS data products are not present locally.
  - **Mitigation**: Build robust adapter interfaces, PDS label parsers, and synthetic test fixtures. Clearly document real-data validation status as `PENDING_REAL_DATA`.
- **Risk**: Multi-scale matching between OHRC (0.25 m/px) and IIRS (80 m/px) spans a 320x resolution ratio.
  - **Mitigation**: Use multi-stage pyramid downsampling and metadata scale priors to center search bounds.
