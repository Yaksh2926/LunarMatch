# Product and Technical Specification

## Problem
Find reliable, uniformly distributed point correspondences between a moving Chandrayaan-2 optical image and a fixed lunar reference image despite illumination, viewpoint, scale, resolution and sensor/modality differences; then estimate and apply an alignment with sub-pixel-refined source coordinates.

## Users
Planetary scientists, image-processing teams and SIH evaluators running pairwise experiments locally/offline.

## Inputs
- Source and reference rasters: TIFF/GeoTIFF initially; PNG/JPEG for debugging; PDS/ISIS through optional conversion adapters.
- Optional metadata: sensor, product ID, pixel scale, acquisition time, sun azimuth/elevation, projection, CRS, affine transform, nodata, approximate footprint.
- YAML configuration.

## Outputs per run
- `matches.csv`: IDs, source/reference floating-point pixels, score, inlier flag, cell ID, refinement status and uncertainty if available.
- `matches.geojson` when georeferencing permits.
- `transform.json`: model, 3x3 matrix, coordinate conventions, provenance.
- `registered.tif`: warped source in reference grid, preserving geospatial metadata when available.
- `metrics.json`: RMSE/median/P95 residual, inliers, ratio, coverage, runtime and warnings.
- `report.html` and quick-look overlays/checkerboards/residual plots.
- `run_manifest.json`: config, versions, hashes, seed and command.

## Functional requirements
FR1 Validate inputs/config before processing.  
FR2 Normalize each modality without destroying geometry.  
FR3 Support scale search and coarse-to-fine matching.  
FR4 Provide at least ORB and SIFT classical feature backends; SIFT may be unavailable in some builds.  
FR5 Provide modality-robust structural representations (gradient magnitude/orientation, CLAHE, phase/edge options).  
FR6 Use kNN matching, mutual checks and robust geometric verification.  
FR7 Support similarity, affine, homography and optional local deformation models with complexity chosen cautiously.  
FR8 Refine inlier positions to sub-pixel precision and retain pre/post-refinement residuals.  
FR9 Select matches uniformly across valid overlap while retaining quality.  
FR10 Warp and export reproducibly.  
FR11 Evaluate against control points when supplied and self-consistency otherwise.  
FR12 Process large images via windows/tiles without loading unnecessary full-resolution arrays.  
FR13 Expose CLI and importable Python API.  
FR14 Learned matchers are optional adapters, not mandatory for baseline operation.

## Non-functional requirements
- Python 3.11+, Linux/Windows where dependencies allow.
- Deterministic runs; structured logging; restart-safe outputs.
- Modular, test-covered, documented; no silent fallbacks.
- CPU baseline; optional CUDA acceleration.
- Tiling and memory budget configuration.

## Coordinate conventions
- Pixel coordinates use `(x=column, y=row)`, origin at top-left, pixel centers, float64 in exported matches.
- Homogeneous transforms map **source pixel centers to reference pixel centers**.
- Every resize/crop/pyramid stage must expose a 3x3 mapping to original pixels.

## Baseline algorithm
1. Read metadata and valid masks.
2. Generate robust structural images independently per sensor.
3. Estimate likely scale from metadata, otherwise search configured log-scale levels.
4. Detect/describe features over pyramids/tiles.
5. Match descriptors with mutual nearest-neighbor and ratio/absolute tests.
6. Robustly fit geometry (USAC/MAGSAC or RANSAC), with degeneracy checks.
7. Partition valid overlap into grid cells; select top spatially diverse inliers.
8. Refine coordinates using local gradient/phase correlation or ECC patches with rejection gates.
9. Refit transform, compute residuals, warp source and export artifacts.

## Success gates
The software must report rather than assume success. Suggested configurable gates:
- `min_inliers >= 30`
- `min_inlier_ratio >= 0.20`
- occupied valid grid cells `>= 40%`
- residual RMSE and P95 under dataset-specific thresholds
- no accepted model with invalid/implausible scale, reflection or singularity

“Sub-pixel accuracy” is demonstrated only on supplied truth/control points or a defensible benchmark—not merely because coordinates are fractional.
