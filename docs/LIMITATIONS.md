# LunarMatch — Known Pipeline Limitations & Multi-Pair Benchmark

## 1. Generalization Limitation on Narrow-Overlap Swath Pairs

> **Known limitation — spatial coverage on narrow-overlap swath pairs:** the GSD-resize + CLAHE + tiled LoFTR + confidence-calibration pipeline achieves `WELL_CONSTRAINED` spatial coverage (grid fraction ≥0.15, convex hull coverage 0.60) on the primary Chandrayaan-2 South Pole OHRC↔TMC-2 target pair, with 48 geometrically consistent RANSAC inliers versus 4 for classical SIFT. On two additional tested pairs (a secondary South Pole track and a cross-mission OHRC↔LROC pair), the pipeline found 64–96 raw inliers but convex hull coverage collapsed to near-zero (0.0057–0.0064), meaning matches re-clustered into a small region despite higher raw counts.

### Nominal Overlap vs. Matched-Region Coverage Diagnostic

A read-only geometric bounding-box footprint overlap analysis was conducted to determine whether clustered coverage on secondary pairs stems from narrow physical ground overlap or matcher clustering:

* **Pair 2 (OHRC_0152168201 vs TMC_034500)**: Nominal footprint overlap 100.0%, matched-region coverage 0.57%.
* **Pair 3 (Preprocessed OHRC vs LROC)**: Nominal footprint overlap 100.0%, matched-region coverage 0.64%.

**Conclusion**: Both secondary pairs possess 100.0% nominal bounding-box footprint overlap by header metadata. The observed spatial clustering (0.57% and 0.64% convex hull coverage) represents a genuine matcher generalization limitation under steep incidence angle variations and feature-sparse lunar terrain, rather than a lack of physical ground footprint overlap.

---

## 2. Multi-Pair Benchmark Results Table (Check D)

The full cross-scale matching pipeline (GSD-ratio downsampling $\rightarrow$ CLAHE $\rightarrow$ Tiled LoFTR $\rightarrow$ Confidence Calibrator $\rightarrow$ Global Homography USAC_MAGSAC) was evaluated across all available real image pairs in `sample_data/real/`:

| Tested Pair | Sensor & Native GSD | Raw Matches | RANSAC Inliers | Occupied Grid Fraction | Convex Hull Coverage | Quality Gate Status | In-Sample RMSE |
|---|---|---|---|---|---|---|---|
| **Pair 1 (Main South Pole Target)** | OHRC 0350447888 (0.25m) vs TMC 034500 (5.0m) | 48 | 48 | **0.1562** | **0.5971** | **`WELL_CONSTRAINED`** | 0.4204 px |
| **Pair 2 (Secondary South Pole Track)** | OHRC 0152168201 (0.25m) vs TMC 034500 (5.0m) | 96 | 96 | 0.1250 | 0.0057 | `UNDER_CONSTRAINED` | 0.2454 px |
| **Pair 3 (Cross-Mission Pair)** | Preprocessed OHRC (0.5m) vs LROC (0.5m) | 64 | 64 | 0.1250 | 0.0064 | `UNDER_CONSTRAINED` | 0.2349 px |

---

## 3. Physical Quantity & Model Scope Summary

* **Illumination Offset Normalizer**: Computes 2D solar illumination translation vectors $(\Delta x, \Delta y)$ from sun azimuth/elevation angles ([illumination_normalizer.py](file:///c:/Users/Asus/OneDrive/Desktop/LUNAR%20AI/LUNAR/src/lunarmatch/preprocessing/illumination_normalizer.py)).
* **LoFTR Confidence Calibrator**: Trains a lightweight 2-layer MLP projection head (`CrossResolutionAdapter`) on a **frozen LoFTR backbone** to recalibrate match confidence thresholds across large resolution gaps ([finetune.py](file:///c:/Users/Asus/OneDrive/Desktop/LUNAR%20AI/LUNAR/src/lunarmatch/features/finetune.py)).
* **Sub-Pixel Patch Refinement**: Computes sub-pixel pixel displacement shifts $(\Delta x, \Delta y)$ via ECC patch alignment and parabolic peak interpolation ([subpixel.py](file:///c:/Users/Asus/OneDrive/Desktop/LUNAR%20AI/LUNAR/src/lunarmatch/geometry/subpixel.py)). In-sample reprojection RMSE measures model optimization fit, not independent ground-truth geodetic error.
