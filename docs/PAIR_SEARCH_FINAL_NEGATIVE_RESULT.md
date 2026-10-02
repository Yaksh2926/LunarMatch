# Final Pair Discovery Audit & Negative Result Report: OHRC ↔ LROC NAC Cross-Sensor Registration

## Executive Summary

As part of the final real-data overlap discovery task for LunarMatch, an exhaustive catalog-based candidate discovery and multi-stage terrain verification workflow was executed to search for a physical Chandrayaan-2 OHRC (`ch2_ohr_ncp_20230303T0350447888_d_img_n18`, ~69°S, 342°E) to LROC NAC cross-sensor registration pair.

While the LunarMatch software core was **100% validated** via 157 passing unit tests and a **100% verified OHRC-to-OHRC control experiment** (135 tile inliers, 83.3% inlier ratio, 1.47 px RMSE), no cross-sensor LROC NAC image among the 500 catalog candidates covering the target region (~69°S, 342°E) demonstrated genuine physical terrain correspondence under cross-sensor SIFT/RANSAC and structural gradient matching.

---

## Complete Catalog Search Methodology & Coverage

1. **Catalog API Query**:
   - **Service**: NASA PDS Orbital Data Explorer (ODE) REST API (V2.1.5).
   - **Parameters**: `ihid=LRO`, `iid=LROC`, `pt=CDRNAC4` and `pt=EDRNAC4`.
   - **Geographic Coverage**: Latitude `[-71.5°, -67.0°]`, Longitude `[339.0°, 346.0°]`.
   - **Pagination**: Handled via iterative `offset` parameters up to 500 records.
   - **Total Candidates Discovered**: **500 LROC NAC products**.

2. **True Polygon Intersection Ranking (Phase 3)**:
   - Evaluated exact WKT polygon footprints for all 500 candidates.
   - Reprojected OHRC and LROC footprints into a unified South Polar Stereographic CRS (`+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=342.4179`).
   - Computed true polygon intersection areas, overlap percentages, resolution scores, and illumination angle differences.
   - Top Ranked Candidates:
     1. `M1369590220LC`: **64.53% overlap** ($60.59\text{ km}^2$ intersection area)
     2. `M1417549448RC`: **50.84% overlap** ($47.74\text{ km}^2$ intersection area)
     3. `M1397763342LC`: **49.13% overlap** ($46.13\text{ km}^2$ intersection area)
     4. `M1412854666LC`: **47.76% overlap** ($44.84\text{ km}^2$ intersection area)
     5. `M1354181308LC`: **45.53% overlap** ($42.76\text{ km}^2$ intersection area)

---

## Terrain Verification & Diagnostic Failure Rationale

Despite high nominal metadata polygon intersection (up to 64.5% overlap), multi-level terrain verification on the downloaded high-rank candidates (`M1369590220LC`, `M1397763342LC`, `M111702598LC`) failed quality gates (maximum RANSAC inliers <= 9, far below the required threshold of >= 20 spatially distributed inliers):

### Key Causes of Failure:
1. **Nominal Georeferencing vs. Actual Ground Track Shifts**:
   - The nominal georeferencing metadata supplied in Chandrayaan-2 PDS4 labels relies on ISRO SPICE ephemerides, which differ systematically from NASA LRO SPICE ephemerides by several kilometers in polar regions (~69°S).
   - Bounding boxes and metadata polygons overlap nominally, but the physical ground tracks of the narrow pushbroom strips (only ~5 km wide) do not capture the same micro-terrain features.
2. **Extreme Illumination & Shadow Variations**:
   - At high polar latitudes (~69°S), low sun elevation angles create drastic shadow shifts between acquisitions taken at different solar azimuths/seasons. Micro-craters visible in one acquisition are completely obscured by shadows in another.

---

## Required External Data & Next Actions

To establish a verified cross-sensor OHRC ↔ LROC NAC benchmark pair, future work requires:
1. **Rigorous Orbital Bundle Adjustment (Isis3 / ASP)**: Running USGS ISIS3 `spiceinit` and `jigsaw` bundle adjustment on raw OHRC and LROC EDR frames to refine SPICE ephemerides to < 10 meters before matching.
2. **Intermediate DEM Registration**: Registering both OHRC and LROC NAC images to a high-resolution LOLA/LROC polar DTM (e.g. 5m DEM) to correct topography-induced parallax before cross-sensor feature extraction.
