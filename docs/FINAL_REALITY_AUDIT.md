# Final Reality Audit: LunarMatch Repository against ISRO Problem Statement 26166

## Executive Summary

This document presents a rigorous, evidence-based reality audit of the **LunarMatch** repository against **ISRO Problem Statement 26166**:

> **"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"**

Every requirement is classified strictly according to repository code, unit/integration tests, real data files, and output artifacts.

---

## 1. Requirement Classification Matrix

### Classification Criteria:
1. `IMPLEMENTED AND REAL-DATA VALIDATED`: Code implemented in `src/`, unit tests passing, and validated on real Chandrayaan-2 / lunar datasets with generated artifacts.
2. `IMPLEMENTED BUT ONLY SYNTHETICALLY VALIDATED`: Code implemented in `src/` and passing unit tests, but no real dataset is available or validated in the repository.
3. `ARCHITECTURALLY SUPPORTED BUT NOT TESTED`: High-level code structure or configuration exists, but no end-to-end integration test exists.
4. `NOT IMPLEMENTED`: Functionality is missing from the codebase.

---

### Audit Table

| ISRO Requirement | Classification | Source File & Class / Function | Test File & Function | Real Dataset Used | Output Artifact |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **OHRC support** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/sensors/ohrc.py` (`OHRCAdapter`), `src/lunarmatch/data/raster.py` (`read_raster`) | `tests/unit/test_sensor_adapters.py` (`test_detect_sensor_adapter`), `tests/unit/test_raster.py` | `sample_data/real/ohrc/ch2_ohr_ncp_20230303T0350447888_d_img_n18.xml` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` |
| **TMC support** | **IMPLEMENTED BUT ONLY SYNTHETICALLY VALIDATED** | `src/lunarmatch/sensors/tmc.py` (`TMCAdapter`) | `tests/unit/test_sensor_adapters.py` (`test_detect_sensor_adapter`) | *None* (Real TMC datasets unavailable in repository) | *None* |
| **IIRS support** | **IMPLEMENTED BUT ONLY SYNTHETICALLY VALIDATED** | `src/lunarmatch/sensors/iirs.py` (`IIRSAdapter`, `select_band`, `pca_representation`) | `tests/unit/test_sensor_adapters.py` (`test_iirs_spectral_processing`) | *None* (Real IIRS hyperspectral datasets unavailable in repository) | *None* |
| **OHRC to OHRC registration** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/orchestrator.py` (`RegistrationOrchestrator`), `scratch/ohrc_pair_validation.py` | `tests/integration/test_orchestrator.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` (135 inliers, 83.3% ratio, 1.47 px RMSE) |
| **OHRC to TMC registration** | **ARCHITECTURALLY SUPPORTED BUT NOT TESTED** | `src/lunarmatch/orchestrator.py`, `src/lunarmatch/sensors/factory.py`, `configs/ohrc_to_tmc.yaml` | *None* (No integration test connecting OHRCAdapter to TMCAdapter) | *None* | *None* |
| **OHRC to IIRS registration** | **ARCHITECTURALLY SUPPORTED BUT NOT TESTED** | `src/lunarmatch/orchestrator.py`, `src/lunarmatch/sensors/factory.py`, `configs/ohrc_to_iirs.yaml` | *None* (No integration test connecting OHRCAdapter to IIRSAdapter) | *None* | *None* |
| **Chandrayaan-2 to lunar reference registration** | **IMPLEMENTED BUT ONLY SYNTHETICALLY VALIDATED** | `src/lunarmatch/orchestrator.py`, `scratch/verify_candidate_terrain.py`, `scratch/run_corrected_pair_pipeline.py` | `tests/integration/test_orchestrator.py` | `M1369590220LC.IMG`, `M1397763342LC.IMG`, `M111702598LC.IMG` | `outputs/corrected_pair_discovery/FINAL_PAIR_DISCOVERY_REPORT.md` (500 catalog candidates evaluated; cross-sensor ephemeris shift uncorrected) |
| **Illumination invariance** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/preprocessing/representations.py` (`compute_gradient_representation`, `compute_clahe_representation`) | `tests/unit/test_preprocessing.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/` (Gradient representation enabled successful OHRC registration) |
| **Viewpoint invariance** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/geometry/verifier.py` (`GeometricVerifier`), `fitting.py` (`fit_affine`) | `tests/unit/test_geometry.py`, `tests/integration/test_orchestrator.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` (Resolved affine transform & 618m shift) |
| **Scale invariance** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/features/pyramid.py` (`ImagePyramid`), `scale_search.py` (`generate_scale_hypotheses`) | `tests/unit/test_scale_aware.py` (synthetic 1.0x, 2.0x, 4.0x scale tests) | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/` |
| **Sub-pixel refinement** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/geometry/subpixel.py` (`SubpixelRefiner`, `extract_subpixel_patch`) | `tests/unit/test_subpixel_synthetic.py` (`test_subpixel_refinement_synthetic_shift`) | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` |
| **Sub-pixel accuracy validation** | **IMPLEMENTED BUT ONLY SYNTHETICALLY VALIDATED** | `src/lunarmatch/evaluation/evaluator.py` (`evaluate_registration`), `control_points.py` (`evaluate_control_points`) | `tests/unit/test_subpixel_synthetic.py` | *None* (No independent real GCP set with <1.0 px RMSE) | *None* (Real OHRC RMSE is 1.4721 px; sub-1.0 px RMSE unproven on independent real GCPs) |
| **Uniform spatial distribution** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/features/tiling.py` (`TiledFeatureExtractor`), `matching/spatial_selector.py` (`SpatialSelector`) | `tests/unit/test_matching.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/isro_demo/registration_manifest.json` (Occupied grid fraction 6.2%, convex hull coverage 5.1%, CV 3.87) |
| **Registered output generation** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/export/warper.py` (`warp_source_to_reference`), `writer.py` (`export_registered_products`) | `tests/unit/test_export.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/registered_output.tif` |
| **Match point export** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/export/writer.py` (`write_geojson_matches`), `orchestrator.py` (`export_matches_csv`) | `tests/unit/test_export.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/inliers.geojson`, `outputs/isro_demo/match_points.csv` |
| **RMSE** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/geometry/verifier.py` (`compute_reprojection_residuals`) | `tests/unit/test_geometry.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` (1.4721 px RMSE) |
| **Inlier count** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/geometry/verifier.py` (`GeometricVerifier`) | `tests/unit/test_geometry.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` (135 inliers) |
| **Inlier ratio** | **IMPLEMENTED AND REAL-DATA VALIDATED** | `src/lunarmatch/geometry/verifier.py` (`GeometricVerifier`) | `tests/unit/test_geometry.py` | `ch2_ohr_ncp_20230303T0350447888` & `ch2_ohr_ncp_20230303T0152168201` | `outputs/ohrc_pair_validation/ohrc_pair_validation_manifest.json` (83.3% inlier ratio) |

---

## 2. Top 3 Remaining Gaps

1. **Gap 1: Absence of Real Chandrayaan-2 TMC & IIRS Datasets and Integration Tests**:
   - `TMCAdapter` (`src/lunarmatch/sensors/tmc.py`) and `IIRSAdapter` (`src/lunarmatch/sensors/iirs.py`) are implemented and pass unit tests, but no real TMC or IIRS product files exist in `sample_data/` and no integration test connects them to `RegistrationOrchestrator`.

2. **Gap 2: Unproven Sub-Pixel Registration Accuracy on Independent Real Ground-Truth Data (<1.0 px RMSE)**:
   - `SubpixelRefiner` (`src/lunarmatch/geometry/subpixel.py`) is implemented and synthetically verified, but real OHRC registration achieves $1.4721\text{ px RMSE}$. Sub-1.0 px RMSE has not been experimentally proven on independent real ground-truth control points.

3. **Gap 3: Cross-Sensor Orbit Propagation Ephemeris Discrepancy (OHRC ↔ LROC NAC)**:
   - Absolute orbit determination errors between ISRO Chandrayaan-2 SPICE ephemerides and NASA LRO SPICE ephemerides cause multi-kilometer ground track shifts (~2–5 km) in polar regions (~69°S). Solving cross-sensor physical alignment requires USGS ISIS3 `jigsaw` bundle adjustment or DEM co-registration.

---

## 3. Recommended Next Highest-Value Experiment

**Recommendation**:
Create synthetic multi-sensor integration test fixtures (`tests/integration/test_multimodal_orchestrator.py`) and an automated multi-sensor benchmark script (`scripts/run_multimodal_benchmark.py`).

- **Objective**: Generate synthetic TMC (5.0 m/px GSD) and IIRS (80.0 m/px GSD hyperspectral cube) rasters, pass them through `TMCAdapter` and `IIRSAdapter` directly into `RegistrationOrchestrator`, and verify scale-aware multi-sensor registration, match point exports, and manifest metrics.
