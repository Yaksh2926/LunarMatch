# Implementation Plan — LunarMatch

## Overview & Architecture Alignment
`LunarMatch` provides robust multi-modal, illumination-invariant, and scale-invariant image correspondence and registration between Chandrayaan-2 imagery (OHRC, TMC-2, IIRS) and lunar reference datasets (e.g., LRO NAC).

This document maps all Functional Requirements (FR1–FR14) defined in [`docs/PROJECT_SPEC.md`](file:///d:/LUNAR/docs/PROJECT_SPEC.md) to specific implementation modules, data contracts, and unit/integration test specifications across implementation phases.

---

## Audit Findings & Fixes

1. **Import/Environment Contradictions**:
   - **Issue**: `pytest -q` failed out-of-the-box with `ModuleNotFoundError: No module named 'lunarmatch'`.
   - **Resolution**: Updated [`pyproject.toml`](file:///d:/LUNAR/pyproject.toml) to include `pythonpath = ["src"]` under `[tool.pytest.ini_options]`.

2. **Package Structure & Markers**:
   - **Status**: Checked all package subdirectories (`data`, `preprocessing`, `features`, `matching`, `geometry`, `evaluation`, `visualization`, `models`, `cli`, `utils`).
   - **Resolution**: Verified that `__init__.py` markers are present in `src/lunarmatch` and all sub-packages. Added `pythonpath = ["src"]` configuration for clean test discovery.

3. **Dependency Management**:
   - **Status**: Evaluated dependencies in [`pyproject.toml`](file:///d:/LUNAR/pyproject.toml). Core dependencies (`numpy`, `scipy`, `opencv-python`, `scikit-image`, `pydantic`, `PyYAML`, `typer`, `rich`, `tifffile`) handle the baseline matching pipeline.
   - **Resolution**: Optional geospatial dependencies (`rasterio`, `shapely`, `pyproj`) are isolated under `geo` extras. The pipeline safely degrades when `geo` extras are not installed.

4. **Coordinate Conventions**:
   - **Issue**: Standardized coordinate system required to prevent row/column ordering bugs across NumPy (`[row, col]`) and OpenCV (`(x, y)`).
   - **Resolution**: Standardized top-left origin, float64 sub-pixel coordinates `(x, y)` where `x = column` (width) and `y = row` (height). All 3x3 homogeneous transformation matrices map source pixel centers `(x_src, y_src)` to reference pixel centers `(x_ref, y_ref)`.

---

## Requirements-to-Module Matrix

| Requirement | Description | Target Module(s) | Target Test File(s) |
|---|---|---|---|
| **FR1** | Validate inputs/config before processing | `lunarmatch.models.config`, `lunarmatch.cli.app` | `tests/unit/test_config.py`, `tests/unit/test_cli.py` |
| **FR2** | Normalize each modality without destroying geometry | `lunarmatch.preprocessing.normalization`, `lunarmatch.preprocessing.registry` | `tests/unit/test_preprocessing.py` |
| **FR3** | Support scale search and coarse-to-fine matching | `lunarmatch.features.pyramid`, `lunarmatch.matching.scale_search` | `tests/unit/test_pyramid.py`, `tests/unit/test_scale_search.py` |
| **FR4** | Provide ORB & SIFT classical feature backends | `lunarmatch.features.orb`, `lunarmatch.features.sift`, `lunarmatch.features.protocol` | `tests/unit/test_features_orb.py`, `tests/unit/test_features_sift.py` |
| **FR5** | Provide structural representations (gradient, CLAHE, phase, edge) | `lunarmatch.preprocessing.representations`, `lunarmatch.preprocessing.structural` | `tests/unit/test_representations.py` |
| **FR6** | Use kNN matching, mutual checks & robust geometric verification | `lunarmatch.matching.knn`, `lunarmatch.matching.mutual`, `lunarmatch.geometry.verifier` | `tests/unit/test_matching.py`, `tests/unit/test_geometry_verifier.py` |
| **FR7** | Support similarity, affine, homography & deformation models | `lunarmatch.geometry.models`, `lunarmatch.geometry.fitting` | `tests/unit/test_geometry_models.py` |
| **FR8** | Refine inlier positions to sub-pixel precision | `lunarmatch.geometry.subpixel`, `lunarmatch.models.matches` | `tests/unit/test_subpixel.py` |
| **FR9** | Select matches uniformly across valid overlap | `lunarmatch.matching.spatial_selector`, `lunarmatch.matching.grid` | `tests/unit/test_spatial_selector.py` |
| **FR10** | Warp and export reproducibly | `lunarmatch.geometry.warper`, `lunarmatch.data.exporter`, `lunarmatch.utils.reproducibility` | `tests/unit/test_warper.py`, `tests/unit/test_exporter.py` |
| **FR11** | Evaluate against control points & self-consistency | `lunarmatch.evaluation.metrics`, `lunarmatch.evaluation.control_points` | `tests/unit/test_evaluation.py` |
| **FR12** | Process large images via windows/tiles | `lunarmatch.data.raster`, `lunarmatch.features.tiling` | `tests/unit/test_raster_tiling.py` |
| **FR13** | Expose CLI and importable Python API | `lunarmatch.cli.app`, `lunarmatch.orchestrator` | `tests/unit/test_cli.py`, `tests/integration/test_orchestrator.py` |
| **FR14** | Optional learned matcher adapters | `lunarmatch.features.learned_adapters`, `lunarmatch.matching.learned_adapters` | `tests/unit/test_learned_adapters.py` |

---

## Detailed Task Roadmap

### Phase 0 — Foundation
- **Task 00 — Repository audit and execution plan**: Establish baseline verification, `docs/IMPLEMENTATION_PLAN.md`, and `docs/DECISIONS.md`.
- **Task 01 — Typed configuration system**: Implement Pydantic config schemas for baseline parameters, validation, CLI validation subcommand, and config hashing.
- **Task 02 — Core domain models & coordinate mappings**: Implement domain dataclasses (`RasterMetadata`, `MatchPoint`, `TransformEstimate`, `Metrics`), homogeneous 3x3 coordinate transform compositions, and sub-pixel float64 specs.
- **Task 03 — Raster I/O & pair manifests**: Implement windowed raster readers for TIFF/PNG/JPEG, mask handling, optional GeoTIFF/rasterio metadata adapters, and CSV pair manifest parsing.

### Phase 1 — Classical Correspondence Baseline
- **Task 04 — Robust lunar preprocessing**: Modality-invariant representations (gradient magnitude, orientation, CLAHE, high-pass edge filtering) with mask preservation.
- **Task 05 — Pyramid, tiling & scale hypotheses**: Logarithmic scale search, overlap planning, tile coordinate tracking back to original pixel grid.
- **Task 06 — Feature backend protocol, ORB and SIFT**: Protocol definitions, OpenCV ORB and SIFT backends with mask awareness, tiling, and keypoint deduplication.
- **Task 07 — Descriptor matching**: Chunked kNN matching, Lowe's ratio test, mutual nearest-neighbor validation, score propagation.
- **Task 08 — Robust geometric verification**: RANSAC/USAC_MAGSAC fitting for similarity, affine, and homography models with degeneracy checks.
- **Task 09 — Uniform spatial match selection**: Grid partitioning over valid overlap, density capping per cell, convex-hull coverage calculation.
- **Task 10 — Sub-pixel refinement**: ECC patch refinement, gradient correlation, border checking, quality thresholds, and pre/post residual calculation.
- **Task 11 — End-to-end baseline orchestrator**: Full pipeline orchestration via Python API and `lunarmatch register` CLI.

### Phase 2 — Products and Evaluation
- **Task 12 — Warp & product export**: Source warping, `matches.csv`, `transform.json`, `metrics.json`, `run_manifest.json`, `registered.tif`, and optional `matches.geojson`.
- **Task 13 — Metrics & control-point evaluation**: RMSE, P90/P95 residuals, coverage, uniformity metrics, and control point evaluation.
- **Task 14 — Visual diagnostics & HTML report**: Downsampled quick-look checkerboard, match overlay, residual plots, and standalone `report.html`.
- **Task 15 — Batch benchmark command**: Multi-pair execution, failure isolation, and aggregate reporting (`lunarmatch benchmark`).
- **Task 16 — Synthetic lunar benchmark generator**: Procedural crater heightfield generation, sun-angle hillshading, and ground-truth transform evaluation.

### Phase 3 — Hardening & Optional Extensions
- **Task 17 — Large-raster & robustness hardening**: Windowed execution, memory limits, performance profiling, and edge case resilience.
- **Task 18 — Optional learned matcher adapters**: Optional adapters for SuperPoint/LightGlue/LoFTR with local weight requirements and zero automatic downloads.
- **Task 19 — Reproducibility, packaging & CI**: Packaging verification, CI configurations, and environment validation.
- **Task 20 — SIH demo & final audit**: Five-minute offline demo runbook and complete requirements verification checklist.

---

## Verification Strategy

1. **Automated Unit & Integration Testing**:
   - `pytest -q` execution for fast feedback.
   - Comprehensive test suite covering edge cases (nodata, missing optional packages, non-integer scales, degenerate point sets).

2. **Linting & Code Quality**:
   - `ruff check .` for code style and formatting.
   - `mypy src` for strict static type checking.

3. **No Fabricated Data Guardrail**:
   - Synthetic fixtures used for unit testing must be generated programmatically with explicit known transformations.
   - No mock performance metrics or fake lunar calibration parameters will be claimed.
