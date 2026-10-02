# ISRO Problem Statement 26166: Codebase Gap Analysis & Architecture Audit

## Executive Summary

This report performs a comprehensive, evidence-based audit of the **LunarMatch** codebase against the actual **ISRO Chandrayaan-2 Problem Statement 26166**:

> **Problem Title**: Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)  
> **Expected Solution**: A generic software solution for finding correspondence between Chandrayaan-2 acquired optical images and Lunar reference images with sub-pixel accuracy of source image while maintaining uniform distribution across the images.  
> **Required Outputs**: Registered product, Corresponding match points, Evaluation metrics (RMSE, inlier match count, inlier ratio).

---

## 1. Actual ISRO Requirements Matrix

The following table classifies every requirement from the official ISRO problem statement against the current implementation in `src/lunarmatch`:

| ISRO Requirement | Status | Codebase Evidence & Verification Details |
| :--- | :---: | :--- |
| **Generic software solution** | **IMPLEMENTED** | End-to-end Python CLI & API (`lunarmatch` command via `src/lunarmatch/cli/app.py`, `RegistrationOrchestrator` in `src/lunarmatch/orchestrator.py`). |
| **OHRC support** | **IMPLEMENTED** | Explicit PDS4 label parser (`scratch/inspect_ohrc_for_localization.py`), verified real 0.25 m/px OHRC-to-OHRC registration (135 inliers, 83.3% ratio, 1.47 px RMSE). |
| **TMC support** | **PARTIALLY IMPLEMENTED** | Generic raster I/O (`src/lunarmatch/data/raster.py`) can read 2D TIFF/PNG/GEOTIFF files, but lacks explicit Chandrayaan-2 TMC PDS label parsers, metadata extraction, or real TMC test validation. |
| **IIRS support** | **NOT IMPLEMENTED** | Generic raster I/O handles 2D slices only; lacks hyperspectral / multi-band IIRS PDS parsing, spectral band selection, or IIRS test data validation. |
| **Lunar reference image support** | **PARTIALLY IMPLEMENTED** | Catalog search scripts for LROC NAC (`scratch/discover_lroc_candidates.py`) and generic raster I/O exist, but real cross-sensor registration is unverified due to orbit ephemeris shifts. |
| **Multi-modal correspondence** | **IMPLEMENTED** | Modality-invariant representations (`gradient`, `clahe`, `edge`, `phase`, `raw_contrast` in `src/lunarmatch/preprocessing/representations.py`). |
| **Illumination / sun-angle invariance** | **PARTIALLY IMPLEMENTED** | Structural representations (Sobel, CLAHE) minimize lighting bias; however, extreme low sun-angle crater shadow variations at high polar latitudes (~69°S) cause cross-date degradation. |
| **Viewpoint variation robustness** | **IMPLEMENTED** | Per-hypothesis RANSAC affine/similarity model fitting and reflection/handedness toggling (`src/lunarmatch/geometry/verifier.py`, `fitting.py`). |
| **Scale variation robustness** | **IMPLEMENTED** | Metadata-centered GSD search priors (`scale_search.py`), resolution-matched image pyramids (`pyramid.py`), and dynamic scale hypothesis matching. |
| **Correspondence/match point generation** | **IMPLEMENTED** | GeoJSON export of ground-control points (`src/lunarmatch/export/writer.py`) containing source $(x, y)$ and reference $(u, v)$ matched coordinates. |
| **Registered image generation** | **IMPLEMENTED** | Affine/Homography image warping (`src/lunarmatch/export/warper.py`) exporting GeoTIFF or standard raster outputs. |
| **Sub-pixel accuracy** | **PARTIALLY IMPLEMENTED** | **Capability Implemented**: Sub-pixel patch alignment via ECC and phase correlation (`src/lunarmatch/geometry/subpixel.py`). **Experimental Accuracy**: 1.47 px RMSE on real OHRC; sub-1.0 px RMSE proven on synthetic benchmarks, but unproven on real independent GCP sets. |
| **Uniform spatial distribution of matches** | **IMPLEMENTED** | Tiled feature extraction (`src/lunarmatch/features/tiling.py`), grid partitioning & spatial bucketing (`src/lunarmatch/matching/spatial_selector.py`), occupied grid coverage quality gates (`src/lunarmatch/geometry/verifier.py`). |
| **RMSE metric output** | **IMPLEMENTED** | Reprojection RMSE calculated in `src/lunarmatch/geometry/verifier.py` and exported in `RunManifest` (`src/lunarmatch/models/domain.py`). |
| **Inlier match count output** | **IMPLEMENTED** | RANSAC inlier count recorded and reported in `RunManifest.inlier_count`. |
| **Inlier ratio metric output** | **IMPLEMENTED** | Inlier ratio (inliers / total matches) computed and enforced via quality gates in `src/lunarmatch/geometry/verifier.py`. |

---

## 2. Current Architecture & Pipeline Flow

The LunarMatch pipeline is structured into 10 processing stages in `src/lunarmatch/orchestrator.py`:

```
Input Rasters (Source & Reference)
       │
[Stage 1: Raster I/O] (src/lunarmatch/data/raster.py)
       │
[Stage 2: Structural Preprocessing] (src/lunarmatch/preprocessing/normalization.py, representations.py)
       │
[Stage 2.5: Coarse Terrain Overlap Validation] (src/lunarmatch/matching/terrain_check.py)
       │
[Stage 3: Pyramid Planning] (src/lunarmatch/features/pyramid.py)
       │
[Stage 4: Scale Hypotheses Generation] (src/lunarmatch/features/scale_search.py)
       │
[Stage 5: Feature Extraction] (src/lunarmatch/features/sift.py, orb.py, tiling.py)
       │
[Stage 6: Multi-Scale Descriptor Matching] (src/lunarmatch/matching/matcher.py, knn.py)
       │
[Stage 7: Consolidation & Spatial Selection] (src/lunarmatch/matching/spatial_selector.py)
       │
[Stage 8: Robust Geometric Verification] (src/lunarmatch/geometry/verifier.py, fitting.py)
       │
[Stage 9: Sub-Pixel Patch Refinement] (src/lunarmatch/geometry/subpixel.py)
       │
[Stage 10: Warp & Export] (src/lunarmatch/export/warper.py, writer.py)
```

### Primary Classes & Functions:
- **`RegistrationOrchestrator`** (`src/lunarmatch/orchestrator.py`): Main execution loop.
- **`read_raster_metadata` / `read_raster_window`** (`src/lunarmatch/data/raster.py`): Dataset loading.
- **`RepresentationFactory`** (`src/lunarmatch/preprocessing/registry.py`): CLAHE, Sobel gradient, Canny edge transforms.
- **`CoarseTerrainChecker`** (`src/lunarmatch/matching/terrain_check.py`): Fast downsampled early rejection (<1s).
- **`ImagePyramid`** (`src/lunarmatch/features/pyramid.py`): Anti-aliased Gaussian downsampling.
- **`ScaleSearchPlanner`** (`src/lunarmatch/features/scale_search.py`): GSD metadata prior hypothesis generation.
- **`TiledFeatureExtractor`** (`src/lunarmatch/features/tiling.py`): Spatial keypoint bucketing across grid cells.
- **`MultiScaleMatcher`** (`src/lunarmatch/matching/matcher.py`): Mutual nearest-neighbor & ratio test matching.
- **`SpatialSelector`** (`src/lunarmatch/matching/spatial_selector.py`): Convex hull coverage & cell density metrics.
- **`GeometricVerifier`** (`src/lunarmatch/geometry/verifier.py`): RANSAC / USAC_MAGSAC, inlier count, inlier ratio, RMSE.
- **`SubpixelRefiner`** (`src/lunarmatch/geometry/subpixel.py`): ECC patch alignment & parabolic correlation interpolation.
- **`RasterWarper` & `ManifestWriter`** (`src/lunarmatch/export/`): GeoTIFF warping & GeoJSON match point export.

---

## 3. Sensor Support Audit

### A. OHRC (Optical High Resolution Camera)
- **Support Level**: **EXPLICIT & VALIDATED**
- **Capabilities**: Full 0.25 m/px parsing, GeoJSON footprint generation, validated real-data OHRC-to-OHRC registration (135 inliers, 83.3% ratio, 1.47 px RMSE).

### B. TMC (Terrain Mapping Camera)
- **Support Level**: **GENERIC RASTER ONLY**
- **Capabilities**: Reads 2D TIFF/PNG/GeoTIFF arrays through rasterio/GDAL.
- **Gaps**: No explicit Chandrayaan-2 TMC PDS XML label parser, no metadata extraction for TMC stereo angles/GSD, no TMC test data validation in `tests/`.

### C. IIRS (Imaging InfraRed Spectrometer)
- **Support Level**: **NO EXPLICIT SUPPORT**
- **Capabilities**: Generic raster loading handles single 2D bands only.
- **Gaps**: No hyperspectral data cube reader, no band selection algorithm (e.g. continuum removal or absorption band selection), no IIRS PDS parser or test datasets.

### D. LROC & Lunar Reference Imagery
- **Support Level**: **EXPERIMENTAL / UNVERIFIED CROSS-SENSOR**
- **Capabilities**: PDS ODE catalog discovery scripts (`scratch/discover_lroc_candidates.py`) and footprint intersection ranking.
- **Gaps**: Cross-sensor OHRC ↔ LROC registration fails physical terrain verification due to absolute orbit propagation shifts (2–5 km ephemeris error between ISRO and NASA SPICE kernels). Requires ISIS3 SPICE bundle adjustment (`jigsaw`).

---

## 4. Sub-Pixel Accuracy Audit

### Internal Sub-Pixel Estimation Capabilities:
- `src/lunarmatch/geometry/subpixel.py` implements **ECC patch alignment** (`cv2.findTransformECC`) and **sub-pixel parabolic interpolation** on local cross-correlation peaks (`extract_subpixel_patch`).
- Keypoints are localized using OpenCV SIFT/ORB sub-pixel floating-point coordinates.

### Experimentally Demonstrated Sub-Pixel Registration Accuracy:
- **Synthetic Benchmarks**: Achieves sub-pixel RMSE (<0.5 px) on synthetic transformed rasters (`tests/unit/test_scale_aware.py`).
- **Real OHRC Data**: Achieves **1.4721 px RMSE** on native 0.25 m/px OHRC-to-OHRC real lunar imagery.
- **Audit Conclusion**: While internal sub-pixel estimation algorithms are implemented, **experimentally proven sub-pixel accuracy (<1.0 px RMSE) on real independent ground-truth control points is NOT YET DEMONSTRATED**.

---

## 5. Sun-Angle / Illumination Audit

### Implemented Techniques:
- Contrast normalization (`MinMax`, `ZScore` in `src/lunarmatch/preprocessing/normalization.py`).
- CLAHE adaptive histogram equalization (`src/lunarmatch/preprocessing/representations.py`).
- Sobel gradient magnitude & Canny edge representations (`representations.py`).
- Phase correlation representation (`PhaseRepresentation`).

### Experimental Testing Status:
- Illumination invariance is tested on synthetic contrast/brightness variations in unit tests.
- On real polar lunar imagery (~69°S), extreme solar elevation differences (e.g. 8° vs 20° elevation) create severe crater shadow morphing that degrades feature matching across different acquisition dates.

---

## 6. Uniform Match Distribution Audit

| Concept | Status | Codebase Evidence |
| :--- | :---: | :--- |
| **a) Implemented** | **YES** | Tiled feature extraction (`src/lunarmatch/features/tiling.py`), grid partitioning (`src/lunarmatch/matching/grid.py`), spatial bucketing (`src/lunarmatch/matching/spatial_selector.py`). |
| **b) Measured** | **YES** | Occupied grid fraction (`occupied_grid_fraction`), match count coefficient of variation (`count_uniformity_cv`), convex hull coverage fraction (`convex_hull_coverage_fraction` in `spatial_selector.py`). |
| **c) Guaranteed** | **YES** | Quality gates in `src/lunarmatch/geometry/verifier.py` enforce `min_occupied_grid_fraction` (default 0.25) and reject spatially clustered match sets. |

---

## 7. Scale Invariance Audit

### Implemented Scale Handling:
- Metadata GSD search priors (`metadata_scale_uncertainty_octaves`) in `src/lunarmatch/features/scale_search.py`.
- Dynamic Gaussian image pyramids in `src/lunarmatch/features/pyramid.py` using correct scale direction:
  $$s = \frac{\text{GSD}_{\text{source}}}{\text{GSD}_{\text{reference}}}$$
- Per-hypothesis RANSAC matching across scale octaves (`src/lunarmatch/geometry/verifier.py`).

### Experimental Validation:
- Verified on synthetic multi-scale benchmark tests (1.0x, 2.0x, 4.0x resolution ratios) in `tests/unit/test_scale_aware.py`.
- Verified on real OHRC-to-OHRC data (resolving scale ratio $1.0009$).

---

## 8. Validation Audit

| Validation Category | Status | Details & Quantitative Results |
| :--- | :---: | :--- |
| **A. Synthetic Validation** | **PASSED** | 157 unit/integration tests passing in `tests/`. |
| **B. OHRC-to-OHRC Real Validation** | **PASSED** | 135 tile inliers, 83.3% ratio, 1.4721 px RMSE, 618m shift resolved. |
| **C. OHRC-to-Reference Real Validation** | **INCOMPLETE** | 500 catalog candidates evaluated; max RANSAC inliers <= 9 due to ephemeris offset. |
| **D. TMC Validation** | **NOT IMPLEMENTED** | No real TMC datasets tested or evaluated. |
| **E. IIRS Validation** | **NOT IMPLEMENTED** | No real IIRS datasets tested or evaluated. |
| **F. Cross-Modal Validation** | **PARTIALLY IMPLEMENTED** | Synthetic gradient/CLAHE tests pass; real cross-sensor unverified. |

---

## 9. Gap Prioritization

### P0 — Critical (Required for ISRO Problem Statement 26166):
1. **Explicit Sensor Readers/Adapters**: Implement explicit PDS XML/metadata parsers for Chandrayaan-2 TMC and IIRS products.
2. **IIRS Hyperspectral Band Selector**: Add spectral band extraction module to select optimal 2D band for matching against optical reference images.
3. **Spatial Uniformity Metrics in Output Manifest**: Include occupied grid fraction, count uniformity CV, and convex hull coverage explicitly in the exported JSON manifest and CLI summary.
4. **Proven Sub-Pixel Accuracy**: Demonstrate sub-1.0 px RMSE on independent ground-truth control points.

### P1 — Important Improvements:
1. **SPICE / ISIS3 Ephemeris Alignment Adapter**: Add support for USGS ISIS3 `jigsaw` bundle adjustment to resolve orbit propagation shifts before cross-sensor feature matching.
2. **Illumination Invariance Enhancement**: Advanced shadow-masking / phase-congruency representations for low sun-angle polar imagery.

### P2 — Nice-to-Have:
1. **Interactive Visualization GUI**: Web previewer for side-by-side match point inspection.
2. **Learned Feature Backends**: Opt-in PyTorch SuperPoint / LightGlue backends for extreme lighting variations.

---

## 10. Recommended Implementation Roadmap

- **Stage 1**: Formalize current capabilities, sensor support matrix, and exported manifest outputs.
- **Stage 2**: Integrate spatial uniformity metrics (`occupied_grid_fraction`, `count_uniformity_cv`, `convex_hull_coverage`) directly into main `RunManifest` outputs.
- **Stage 3**: Enhance sub-pixel refinement (`SubpixelRefiner`) and validate sub-1.0 px RMSE on benchmark control points.
- **Stage 4**: Improve illumination invariance (phase congruency & shadow-insensitive representations).
- **Stage 5**: Build generic sensor adapter interfaces (`SensorAdapter`) for Chandrayaan-2 metadata.
- **Stage 6**: Implement Chandrayaan-2 TMC sensor adapter and add TMC real-data test fixtures.
- **Stage 7**: Implement Chandrayaan-2 IIRS hyperspectral band selector and IIRS test fixtures.
- **Stage 8**: Perform scientifically valid cross-sensor validation (OHRC ↔ TMC ↔ IIRS ↔ LROC).
- **Stage 9**: Create final demonstration notebook and ISRO project completion report.

---

## 11. Safety Rules for Future Development

1. **Do not modify the frozen registration core without regression tests**: The existing 157-test suite must remain 100% passing at all times.
2. **Preserve current 157-test baseline**: Run `pytest -q`, `ruff check .`, and `mypy src` after every modification.
3. **Real-data validation must distinguish physical overlap from nominal metadata overlap**: Never claim cross-sensor registration success based on metadata bounding boxes alone.
4. **No claim of cross-sensor success without verified terrain correspondence**: Require RANSAC inliers $\ge 20$, inlier ratio $\ge 50\%$, and sub-2.0 px RMSE.
5. **No claim of sub-pixel accuracy without measured sub-pixel validation**: Require independent control point RMSE $< 1.0\text{ px}$.
6. **No claim of OHRC / TMC / IIRS support unless explicitly demonstrated**: Each sensor must have verified test execution evidence.
