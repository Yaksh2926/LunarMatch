# LunarMatch — Final Submission Summary

**Automatic Sub-Pixel Registration of High-Resolution Optical and Terrain Mapping Camera Images (OHRC ↔ TMC-2).**

---

## 1. System Functionality Overview

**LunarMatch** is an automated, configuration-driven cross-resolution and cross-sensor image registration framework designed for Chandrayaan-2 OHRC (0.25 m/px), TMC-2 (5.0 m/px), and LROC imagery under extreme lunar lighting and shadow variations.

The core pipeline integrates:
1. **GSD-Ratio Resampling**: Scale normalization bridging the 20× resolution gap between 0.25m OHRC and 5.0m TMC-2.
2. **CLAHE Radiometric Preprocessing**: Contrast-Limited Adaptive Histogram Equalization for shadows and low-contrast surface terrain.
3. **Tiled LoFTR + Confidence Calibrator**: Dense local feature matching across image tiles using a recalibration head (`CrossResolutionAdapter`) on a frozen backbone.
4. **Global Robust Geometric Verification**: Single global 3x3 homography matrix estimation via `USAC_MAGSAC` RANSAC across all pooled tile correspondences.
5. **Spatial Coverage Quality Gate**: Automated classification of registered products into `WELL_CONSTRAINED` vs. `UNDER_CONSTRAINED` based on grid occupancy and convex hull bounds.

---

## 2. Evaluation Metrics on Primary Target Pair

Evaluating the primary Chandrayaan-2 South Pole target pair (`OHRC_0350447888` 0.25m vs `TMC_034500` 5.0m):

* **RANSAC Inlier Count**: **48** geometrically consistent inliers (vs. 4 for classical SIFT).
* **In-Sample Reprojection RMSE**: **0.4204 px** (model fit residual on over-determined homography).
* **Occupied Grid Fraction**: **0.1562** (clears the $\ge 0.15$ quality gate).
* **Convex Hull Coverage Fraction**: **0.5971** (broad spatial distribution across lunar surface).
* **Registration Quality Gate Status**: **`WELL_CONSTRAINED`**.

---

## 3. Deliverables Summary

The submission deliverables are packaged in `outputs/final_deliverable/`:
- `registered_ohrc_to_tmc.tif`: Registered source raster warped to reference grid.
- `match_points.csv`: Match point coordinates, scores, and inlier flags.
- `match_points.geojson`: Georeferenced match point FeatureCollection.
- `registration_manifest.json`: Execution metrics and quality gate status.

---

## 4. Known Limitations & Benchmark Pointer

For detailed multi-pair benchmark results, nominal footprint overlap diagnostics, and documented spatial coverage limitations on secondary tracks, see [docs/LIMITATIONS.md](file:///c:/Users/Asus/OneDrive/Desktop/LUNAR%20AI/LUNAR/docs/LIMITATIONS.md).
