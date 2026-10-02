# Antigravity One-by-One Tasklist

## How to use
Submit exactly one task block at a time to Antigravity IDE, beginning with Task 00. Do not submit the next task until tests and acceptance checks pass. Keep this file updated by changing `[ ]` to `[x]` only after verification.

---

## Phase 0 — Foundation

### [x] Task 00 — Repository audit and execution plan
**Depends on:** none

**Prompt to paste:**
> Read `ANTIGRAVITY_CONTEXT.md` and every file under `docs/`, plus `configs/baseline.yaml` and `pyproject.toml`. Do not implement the registration pipeline yet. Audit the scaffold for contradictions, missing package markers, invalid dependencies and ambiguous coordinate conventions. Create `docs/IMPLEMENTATION_PLAN.md` mapping every functional requirement in `docs/PROJECT_SPEC.md` to modules and tests. Create `docs/DECISIONS.md` with initial ADR entries for coordinate convention, baseline transform, optional geospatial dependencies and no automatic model downloads. Make only small scaffold fixes needed for imports. Run the smoke test and report changed files, commands, results, assumptions and risks.

**Acceptance checks:** `pytest -q` passes; requirements-to-module matrix exists; no fabricated data or metrics.

### [x] Task 01 — Typed configuration system
**Depends on:** 00

**Prompt to paste:**
> Implement typed configuration models for every section of `configs/baseline.yaml` under `src/lunarmatch/models/config.py`, with cross-field validation and useful error messages. Implement YAML loading, unknown-key rejection and deterministic config serialization/hash. Add CLI command `lunarmatch validate-config --config PATH`. Keep `max_rmse_px: null` valid. Add unit tests for the baseline file and invalid thresholds, ranges, methods and unknown keys. Update documentation. Run pytest, Ruff and mypy.

**Acceptance checks:** baseline validates; invalid configs fail clearly; stable config hash is tested.

### [x] Task 02 — Core domain models and coordinate mappings
**Depends on:** 01

**Prompt to paste:**
> Implement typed domain models for raster metadata, image pair, keypoints, matches, transform estimate, metrics and run manifest. Match arrays must use float64 Nx2 `(x,y)` coordinates and validate length/finite values. Implement composable 3x3 coordinate mappings for crop, resize and pyramid operations, mapping pixel centers from working images back to original images. Add serialization and numerical round-trip tests, including non-integer scale and crop composition. Document conventions.

**Acceptance checks:** mapping round trips meet numerical tolerance; malformed arrays are rejected.

### [x] Task 03 — Raster I/O and pair manifests
**Depends on:** 02

**Prompt to paste:**
> Implement TIFF/GeoTIFF and debug PNG/JPEG raster reading under `data`, supporting windowed reads, band selection, 8/16/32-bit values, nodata/NaN masks and optional rasterio geospatial metadata. Rasterio must remain optional and failures must be actionable. Implement pair-manifest CSV parsing based on the example file and path resolution relative to repository root. Add tiny generated test rasters and tests for dtype, windows, masks, metadata and missing paths. Never add mission imagery.

**Acceptance checks:** window reads equal full-image slices; masks and metadata survive; tests use generated fixtures only.

---

## Phase 1 — Classical correspondence baseline

### [x] Task 04 — Robust lunar preprocessing
**Depends on:** 03

**Prompt to paste:**
> Implement geometry-preserving preprocessing: valid-mask-aware percentile normalization, CLAHE, optional Gaussian denoise, gradient magnitude/orientation representation, edge representation and phase-friendly high-pass representation. Provide a registry selected by config. Do not use raw cross-modal intensity as the only representation. Handle constants, NaNs and sparse valid masks. Add deterministic unit tests and before/after quick-look utility. Record all parameters in provenance.

**Acceptance checks:** outputs are finite on valid pixels; geometry/shape is unchanged; edge cases are tested.

### [x] Task 05 — Pyramid, tiling and scale hypotheses
**Depends on:** 04

**Prompt to paste:**
> Implement image pyramids, overlapping tile planning and scale-hypothesis generation. Use source/reference pixel-scale metadata when both values exist; otherwise create a bounded logarithmic scale search from config. Every level/tile must expose a mapping to original pixel centers. Avoid duplicate border tiles and enforce memory limits. Add tests for extreme aspect ratios, small images, overlap, complete valid-area coverage and coordinate mapping.

**Acceptance checks:** all valid pixels are covered as configured; no tile exceeds bounds; scale ordering is deterministic.

### [x] Task 06 — Feature backend protocol, ORB and SIFT
**Depends on:** 05

**Prompt to paste:**
> Define a `FeatureBackend` protocol and registry. Implement ORB and SIFT OpenCV backends with mask-aware, tiled detection, deterministic keypoint ordering, duplicate suppression in overlaps and descriptors tied to original-coordinate keypoints. Gracefully explain when SIFT is unavailable. Add tests using synthetic geometric patterns and blank images. Expose backend diagnostics without saving huge descriptor dumps.

**Acceptance checks:** ORB works; SIFT works or fails actionably; no-keypoint cases do not crash.

### [x] Task 07 — Descriptor matching
**Depends on:** 06

**Prompt to paste:**
> Implement descriptor matching with correct distance metric by descriptor type, kNN ratio test, optional absolute threshold, mutual nearest-neighbor consistency and duplicate-correspondence removal. Preserve scores and scale-level provenance. Support chunked matching to limit memory. Add deterministic tests with constructed descriptors, including ties, fewer-than-two-neighbor cases and binary/float descriptors.

**Acceptance checks:** expected matches are exact in unit fixtures; edge cases return valid empty results.

### [x] Task 08 — Robust geometric verification
**Depends on:** 07

**Prompt to paste:**
> Implement robust source-to-reference fitting for similarity, affine and homography using configured OpenCV USAC/MAGSAC when available and documented fallback otherwise. Add minimum-point checks, normalization, reprojection residuals, degeneracy/collinearity detection, singularity/reflection/scale plausibility checks and deterministic seed handling. Return a typed result with inlier mask and failure reason rather than throwing for ordinary no-solution cases. Add known-transform and outlier-heavy tests.

**Acceptance checks:** known transforms recover within tolerance; degenerate inputs are rejected; direction is source-to-reference.

### [x] Task 09 — Uniform spatial match selection
**Depends on:** 08

**Prompt to paste:**
> Implement valid-overlap grid selection that ranks verified inliers by quality but caps matches per cell. Compute occupied-cell fraction, convex-hull coverage and per-cell count uniformity. Do not inflate coverage with invalid/nodata cells. Preserve a configurable minimum required set for refitting. Add tests for clustered, uniform and partially masked point sets and deterministic tie-breaking.

**Acceptance checks:** clustered inputs yield honest low coverage; selected points respect cell caps.

### [x] Task 10 — Sub-pixel refinement
**Depends on:** 09

**Prompt to paste:**
> Implement patch-based sub-pixel refinement for verified correspondences, beginning with ECC on structural/gradient patches and allowing a phase-correlation fallback only when quality is measurable. Use floating-point coordinates, validity masks, border checks, convergence gates, maximum-shift rejection and per-match status/quality. Refit geometry after accepted refinements and retain pre/post residuals. Create synthetic fractional-translation tests with blur, noise and contrast change. Clearly state that fractional outputs alone do not prove sub-pixel accuracy.

**Acceptance checks:** held synthetic fixtures improve median localization error; bad/border patches are rejected safely.

### [x] Task 11 — End-to-end baseline orchestrator
**Depends on:** 01–10

**Prompt to paste:**
> Implement the configuration-driven pair-registration orchestrator and Python API. Connect reading, preprocessing, scale/pyramid planning, feature extraction, matching, geometry verification, spatial selection, refinement and final refit. Add structured stage timings, warnings, deterministic seeds and explicit quality-gate status. Ordinary failure must still emit diagnostics and a run manifest. Add `lunarmatch register` with source/reference/output options. Add an integration test using generated lunar-like terrain with a known transform.

**Acceptance checks:** one command completes synthetic registration; run status and failure reasons are explicit; repeat runs are deterministic.

---

## Phase 2 — Products and evaluation

### [x] Task 12 — Warp and product export
**Depends on:** 11

**Prompt to paste:**
> Implement source warping into the reference pixel grid for similarity/affine/homography, with configured interpolation, nodata handling and transform-direction checks. Export `matches.csv`, `transform.json`, `metrics.json`, `run_manifest.json` and `registered.tif`; export GeoJSON only when coordinate conversion is valid. Use atomic writes and refuse overwrite unless configured. Add schema tests and verify geospatial metadata preservation when rasterio is installed.

**Acceptance checks:** required artifact schemas match `PROJECT_SPEC.md`; output dimensions equal reference grid; overwrite policy works.

### [x] Task 13 — Metrics and control-point evaluation
**Depends on:** 12

**Prompt to paste:**
> Implement metrics in `docs/EVALUATION_PROTOCOL.md`: reprojection RMSE/median/P90/P95/max, inlier count/ratio, occupied valid-grid coverage, convex-hull coverage, uniformity, runtime and peak-memory estimate. Implement independent control-point CSV ingestion with coordinate validation and separate control-point error. Never call self-fit residual ground truth. Add bootstrap confidence intervals across pairs where sample count permits. Add numerical unit tests.

**Acceptance checks:** metrics have documented units/denominators; control-point and self-consistency metrics are clearly separated.

### [x] Task 14 — Visual diagnostics and HTML report
**Depends on:** 13

**Prompt to paste:**
> Implement downsampled quick looks: color/edge overlay, checkerboard, match lines, inlier/outlier plot, residual vectors, spatial occupancy heatmap and metric summary. Generate self-contained `report.html` that remains usable when registration fails. Avoid embedding full-resolution images. Escape metadata and paths. Add a smoke test that creates and parses the report and verifies all expected sections.

**Acceptance checks:** report opens offline; failure report explains failed stage; images are bounded in size.

### [x] Task 15 — Batch benchmark command
**Depends on:** 14

**Prompt to paste:**
> Implement `lunarmatch benchmark --manifest ... --config ... --output ...` with restart-safe per-pair directories, aggregate CSV/JSON, failure isolation and deterministic processing. Add optional bounded multiprocessing without oversubscribing OpenCV threads. Aggregate by sensor pair, scale bin and sun-angle-difference bin only when metadata exists. Add a generated multi-pair integration test and document how to freeze config for held-out tests.

**Acceptance checks:** one bad pair does not stop the batch; rerun can resume; aggregate includes failures.

### [x] Task 16 — Synthetic lunar benchmark generator
**Depends on:** 05, 10, 15

**Prompt to paste:**
> Implement a test-only synthetic benchmark using procedurally generated crater-like height fields and hillshade under different sun azimuth/elevation, plus known affine/homography, blur, noise and resolution changes. Export exact correspondence/control points and masks. Label results as synthetic, not mission performance. Use this for regression tests across illumination and scale bins. Keep fixtures small and deterministic.

**Acceptance checks:** ground-truth transforms and points are exact/reproducible; benchmark runs in CI-sized resources.

---

## Phase 3 — Hardening and optional research extensions

### [x] Task 17 — Large-raster and robustness hardening
**Depends on:** 15

**Prompt to paste:**
> Profile the pipeline on generated large rasters, remove accidental full-resolution copies, enforce memory budgets, add cancellation/interrupt-safe artifact states and improve logging. Add tests for empty overlap, all nodata, constant images, extreme scale prior, corrupt files, insufficient matches and unwritable output. Produce `docs/PERFORMANCE.md` with measured environment and honest limits.

**Acceptance checks:** every listed failure is actionable; no silent fallback; memory strategy is documented.

### [x] Task 18 — Optional learned matcher adapters
**Depends on:** stable Task 17 baseline

**Prompt to paste:**
> Add optional learned matcher adapters behind the existing protocols. First create `docs/MODEL_EVALUATION.md` comparing candidate licenses, weight provenance, CPU/GPU memory and expected modality behavior. Implement at most the approved candidate(s), with explicit local weight path, checksum recording, no automatic download and a clean message when extras/weights are absent. Preserve the classical baseline. Add mocked adapter tests and one opt-in real-model test marker.

**Acceptance checks:** base installation remains usable without Torch/weights; licenses and provenance are recorded.

### [x] Task 19 — Reproducibility, packaging and CI
**Depends on:** 17; optionally 18

**Prompt to paste:**
> Finalize package metadata, dependency groups, lock/constraints strategy, pre-commit configuration, CI for supported Python versions, coverage reporting and platform-safe commands. Add a Dockerfile only if it remains small and does not bundle data/weights. Create `docs/REPRODUCIBILITY.md`, example commands and artifact schema versioning. Run the complete test/lint/type suite from a clean environment.

**Acceptance checks:** clean editable install works; CLI help works; CI configuration runs tests without mission data.

### [x] Task 20 — SIH demo and final audit
**Depends on:** 19

**Prompt to paste:**
> Perform a final requirements audit against `docs/PROJECT_SPEC.md`. Create `docs/DEMO_RUNBOOK.md` for a five-minute offline demonstration and `docs/SIH_SUBMISSION_CHECKLIST.md` covering software, registered product, match points, metrics, licenses, data provenance, screenshots, limitations and fallback demo using synthetic data. Add one `scripts/run_demo.py` command that validates inputs and runs a configured pair without hardcoded private paths. Do not claim unmeasured accuracy. Run all checks and list any unmet requirements explicitly.

**Acceptance checks:** each FR has evidence or is marked unmet; demo is reproducible; claims match measured results.

---

## Final command Antigravity should eventually pass
```bash
python scripts/verify_environment.py
pytest -q
ruff check .
mypy src
lunarmatch validate-config --config configs/baseline.yaml
```

## Team reminders
- Use legally obtained datasets and preserve license/citation metadata.
- Keep science products and browse imagery clearly distinguished.
- Do not optimize on the final geographic test regions.
- Save qualitative failures, not only successful examples.
- Sub-pixel performance requires independent control-point evidence.
