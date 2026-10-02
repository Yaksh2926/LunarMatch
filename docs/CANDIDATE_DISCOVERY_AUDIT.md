# Phase 0 Audit: Previous LROC Candidate Discovery Work

## Overview & Executive Summary

This audit evaluates the previous candidate discovery scripts (`search_system_for_file.py`, `test_ode_files_query.py`, `test_ode_url_resolution.py`, `search_all_ranks.py`, `verify_all_candidates.py`, `search_wide.py`, `search_robust.py`) used in LunarMatch to locate corresponding LROC NAC reference images for Chandrayaan-2 OHRC acquisition `ch2_ohr_ncp_20230303T0350447888_d_img_n18` (~69°S, 342°E).

While the LunarMatch core pipeline was successfully verified and frozen with 157/157 passing tests and a 100% successful OHRC-to-OHRC control experiment (83.3% inlier ratio, 1.47 px RMSE), all cross-sensor OHRC-to-LROC candidate registration attempts to date failed genuine terrain verification (maximum RANSAC inliers <= 4, Pearson correlation ~0.20–0.28).

Our audit reveals that the fundamental cause of these failures was **incomplete and non-exhaustive candidate data discovery**, rather than algorithmic deficiencies in the core registration pipeline.

---

## Key Diagnostic Findings

### 1. Hardcoded & Partial Candidate Lists
* **Issue**: Discovery scripts such as `search_all_ranks.py` and `verify_all_candidates.py` operated on hardcoded lists of 3 to 8 pre-selected Product IDs (e.g., `M1397763342LC`, `M1369590220LC`, `M111702598LC`, `M1317902898LC`).
* **Impact**: The scripts never executed a dynamic, automated search across the complete PDS/ODE LROC NAC archive for the region. As a result, hundreds of potential NAC acquisitions covering ~69°S, 342°E were overlooked.

### 2. Over-Reliance on Bounding-Box and Centroid Proximity
* **Issue**: Candidates were selected because their metadata center coordinates or axis-aligned lat/lon bounding boxes overlapped nominal OHRC coordinates.
* **Impact**: LROC NAC images are long, narrow pushbroom strips (typically ~5 km wide by 25–120 km long) inclined relative to latitude lines. Two strips can have overlapping bounding boxes or close center points while sharing **zero physical terrain intersection**.

### 3. Lack of Pagination Handling in Catalog API Queries
* **Issue**: NASA PDS ODE REST API (`https://oderest.rsl.wustl.edu/live2`) caps spatial query responses (typically 50–100 records per request).
* **Impact**: Early exploratory query snippets did not iterate through pagination offsets (`offset` / `start`). Consequently, query results were truncated, excluding valid NAC coverage.

### 4. Unsystematic NAC-L and NAC-R Pairing
* **Issue**: LROC operates as a dual-camera pushbroom system (NAC Left and NAC Right) with a slight cross-track overlap (~100–200 pixels).
* **Impact**: Previous candidate lists frequently included only the `LC` product or only the `RC` product without systematically querying both sides for a given orbit pass.

### 5. Restricted Geographic Search Radius & Missing SPICE Shift Tolerance
* **Issue**: The search region was tightly bounded by nominal OHRC metadata limits without accounting for potential ground track pointing shifts or SPICE ephemeris offsets.
* **Impact**: Ground terrain corresponding to OHRC features could fall slightly outside nominal bounding box queries if orbital geometries differed.

---

## Action Plan for Subsequent Phases

1. **Phase 1**: Construct a mathematically exact, projected GeoJSON polygon footprint for OHRC acquisition `ch2_ohr_ncp_20230303T0350447888` in Polar Stereographic CRS (`+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=342.417291 +a=1737400 +b=1737400`).
2. **Phase 2**: Implement `scratch/discover_lroc_candidates.py` to query the full ODE REST API spatially with complete pagination, capturing ALL NAC-L and NAC-R products covering the latitude range [-71.0°, -67.0°] and longitude range [340.0°, 345.0°].
3. **Phase 3**: Implement `scratch/rank_lroc_candidates.py` using Shapely polygon geometry operations to compute exact physical intersection areas, overlap percentages, and ranking scores in projected meter coordinates.
4. **Phases 4–11**: Systematically verify top candidates visually, download candidates, perform multi-level terrain correlation, and execute LunarMatch registration.
