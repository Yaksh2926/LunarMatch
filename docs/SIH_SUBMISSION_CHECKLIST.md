# SIH Submission Checklist & Functional Requirements Audit

## Functional Requirements (FR) Audit

| Requirement ID | Requirement Description | Implementation Status | Evidence / Test Module |
| :--- | :--- | :--- | :--- |
| **FR-01** | **Raster I/O & Masking**: Windowed reading, multi-format (TIFF, PNG, JPEG), nodata mask extraction. | **SATISFIED** | `src/lunarmatch/data/raster.py`, `tests/unit/test_raster.py` |
| **FR-02** | **Multi-Scale Pyramid & Scale Search**: Octave scale search, pixel scale ratio priors. | **SATISFIED** | `src/lunarmatch/features/pyramid.py`, `src/lunarmatch/features/scale_search.py`, `tests/unit/test_pyramid.py` |
| **FR-03** | **Classical Feature Extraction**: SIFT and ORB feature backends with tiling support. | **SATISFIED** | `src/lunarmatch/features/sift.py`, `src/lunarmatch/features/orb.py`, `tests/unit/test_features.py` |
| **FR-04** | **Descriptor Matching**: Distance metrics (L2, Hamming), kNN ratio test, mutual consistency, duplicate removal. | **SATISFIED** | `src/lunarmatch/matching/matcher.py`, `tests/unit/test_matching.py` |
| **FR-05** | **Robust Geometric Verification**: USAC/MAGSAC and RANSAC for similarity, affine, and homography models with degeneracy detection. | **SATISFIED** | `src/lunarmatch/geometry/verifier.py`, `tests/unit/test_geometry_verifier.py` |
| **FR-06** | **Uniform Spatial Selection**: Grid-based inlier selection capping match counts per cell. | **SATISFIED** | `src/lunarmatch/geometry/spatial_selector.py`, `tests/unit/test_spatial_selector.py` |
| **FR-07** | **Sub-Pixel Patch Refinement**: ECC sub-pixel patch optimization with convergence gates and maximum shift bounds. | **SATISFIED** | `src/lunarmatch/geometry/subpixel.py`, `tests/unit/test_subpixel.py` |
| **FR-08** | **Configuration-Driven Orchestrator**: YAML config validation, deterministic seed handling, stage timing logs, quality gates. | **SATISFIED** | `src/lunarmatch/orchestrator.py`, `src/lunarmatch/models/config.py`, `tests/integration/test_orchestrator.py` |
| **FR-09** | **Product Export & Schemas**: Standardized output export (`registered.tif`, `matches.csv`, `transform.json`, `metrics.json`, `run_manifest.json`). | **SATISFIED** | `src/lunarmatch/export/writer.py`, `src/lunarmatch/export/warper.py`, `tests/unit/test_export.py` |
| **FR-10** | **Evaluation & Control Points**: Independent control point CSV evaluation, bootstrap 95% CIs, peak RSS memory tracking. | **SATISFIED** | `src/lunarmatch/evaluation/`, `tests/unit/test_evaluation.py` |
| **FR-11** | **Visual Diagnostics & HTML Report**: Self-contained offline dark-mode HTML report (`report.html`) with embedded base64 quick-look cards. | **SATISFIED** | `src/lunarmatch/visualization/`, `tests/unit/test_report.py` |
| **FR-12** | **Batch Benchmark & Synthetic Generator**: Multi-pair batch processor with restart safety, failure isolation, bounded multiprocessing, and synthetic lunar terrain generator. | **SATISFIED** | `src/lunarmatch/benchmark/`, `src/lunarmatch/synthetic/`, `tests/integration/test_benchmark.py` |

---

## Submission Package Inventory

1. **Software Source Package**:
   - `lunarmatch` v0.1.0 clean Python package adhering to PEP 517/518 (`pyproject.toml`).
   - Open source licenses (BSD-3-Clause / MIT / Apache-2.0 compliant).

2. **Registered Product Outputs**:
   - `registered.tif`: GeoTIFF warped source raster matching reference spatial extent.
   - `matches.csv`: Standardized correspondence table.
   - `transform.json`: Typed 3x3 transform matrix $M$.
   - `metrics.json`: Quantitative registration metrics.
   - `run_manifest.json`: Full reproducibility manifest (`schema_version: "1.0.0"`).
   - `report.html`: Self-contained visual diagnostic HTML report.

3. **Licenses & Data Provenance**:
   - Software License: Open permissive license.
   - Deep Learning Model Evaluation: [`docs/MODEL_EVALUATION.md`](file:///d:/LUNAR/docs/MODEL_EVALUATION.md) recording MIT/Apache-2.0 model licenses, weight provenance, and zero auto-download policy.
   - Dataset Attribution: Public mission data from ISRO Chandrayaan-2 (OHRC, TMC-2, IIRS) and NASA LRO (LRO NAC).

4. **Honest Limitations & Fallback Protocol**:
   - Extreme scale differences $> 10.0\times$ trigger explicit diagnostic warnings.
   - Self-consistency metrics are explicitly separated from ground-truth control-point error.
   - Offline fallback demonstration (`scripts/run_demo.py`) generates procedural synthetic lunar terrain when private mission data paths are not supplied.
