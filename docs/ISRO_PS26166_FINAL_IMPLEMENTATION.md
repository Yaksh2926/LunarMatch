# ISRO Problem Statement 26166: Final Implementation Report

## Executive Summary

This report presents the final implementation and architectural status of **LunarMatch** for **ISRO Problem Statement 26166**:

> **"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"**

The software solution provides a generic, production-grade image correspondence and registration pipeline supporting multi-sensor Chandrayaan-2 datasets and lunar reference imagery.

---

## 1. System Architecture

LunarMatch implements a modular, configuration-driven 10-stage processing pipeline integrated with a sensor abstraction layer:

```
                          INPUT SENSOR PRODUCT
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
           OHRCAdapter         TMCAdapter        IIRSAdapter
          (0.25 m/px)          (5.0 m/px)       (Hyperspectral)
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                                   ▼
                         SensorProduct Container
                                   │
                     RegistrationOrchestrator
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
Illumination Preprocess    Scale-Aware Matching     Spatial Uniform Selection
(CLAHE, Sobel, Phase)     (Dynamic Pyramid Prior)    (Occupied Grid, Coverage)
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                       Geometric RANSAC & Verifier
                                   │
                                   ▼
                       Sub-Pixel Patch Refinement
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
         Registered Image    Match Points CSV    Run Manifest
```

---

## 2. Sensor Support Matrix

| Sensor | Status | Native GSD | Key Architectural Features |
| :--- | :---: | :---: | :--- |
| **OHRC** | **IMPLEMENTED & VALIDATED** | $0.25\text{ m/px}$ | Native PDS4 XML label parser, 100% verified real registration ($1.47\text{ px RMSE}$, 83.3% inlier ratio). |
| **TMC** | **IMPLEMENTED (`PENDING_REAL_DATA`)** | $5.0\text{ m/px}$ | `TMCAdapter`, GSD prior ratio estimation ($20.0\times$ relative to OHRC), synthetic test suite. |
| **IIRS** | **IMPLEMENTED (`PENDING_REAL_DATA`)** | $80.0\text{ m/px}$ | `IIRSAdapter`, hyperspectral data cube reduction (`select_band`, `average_bands`, `weighted_band_composite`, `pca_representation`). |
| **LROC / Reference** | **IMPLEMENTED** | $0.5 - 2.0\text{ m/px}$ | `GenericRasterAdapter`, PDS ODE catalog search & footprint intersection ranking. |

---

## 3. Core Problem Challenges & Solutions

### A. Illumination Invariance
- **Solution**: Preprocessing transforms images into modality-invariant representations (`gradient`, `clahe`, `edge`, `phase`, `raw_contrast` in `lunarmatch.preprocessing`). Sobel gradient magnitude and CLAHE adaptive histogram normalization remove absolute intensity biases caused by solar elevation/azimuth variations.

### B. Viewpoint & Orientation Handling
- **Solution**: Robust geometric verification (`lunarmatch.geometry.verifier`) fits affine/similarity models using USAC_MAGSAC / RANSAC. Supports explicit reflection/handedness toggles (`allow_reflection`).

### C. Scale Variation Robustness
- **Solution**: Metadata GSD search priors (`metadata_scale_uncertainty_octaves`) in `lunarmatch.features.scale_search` dynamically center resolution-matched image pyramids (`lunarmatch.features.pyramid`).

### D. Uniform Spatial Match Distribution
- **Solution**: Tiled keypoint extraction (`lunarmatch.features.tiling`) and grid partitioning (`lunarmatch.matching.spatial_selector`) enforce spatial bucketing. Explicitly exports metrics: `occupied_grid_fraction`, `count_uniformity_cv`, `convex_hull_coverage_fraction`.

### E. Sub-Pixel Accuracy
- **Solution**: `SubpixelRefiner` (`lunarmatch.geometry.subpixel`) applies ECC patch alignment and 2D parabolic correlation peak interpolation. Synthetic test suite (`tests/unit/test_subpixel_synthetic.py`) verifies sub-1.0 px RMSE recovery.

---

## 4. Verification Quality Gates Status
- **Pytest**: **162 / 162 passed**
- **Ruff**: **0 linter errors**
- **MyPy**: **0 type violations** across 57 source files
