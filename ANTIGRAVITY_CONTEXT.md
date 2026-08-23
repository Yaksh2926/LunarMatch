# Instructions for Antigravity IDE

You are implementing SIH 2026 Problem Statement 26166. Treat this repository as the source of truth.

## Required behavior
- Read `docs/PROJECT_SPEC.md`, `docs/ARCHITECTURE.md`, `docs/DATASET_GUIDE.md`, and `docs/EVALUATION_PROTOCOL.md` before coding.
- Execute only the current task copied from `TASKLIST.md`; do not pre-implement later tasks.
- Inspect existing code before edits. Preserve public APIs unless the task explicitly changes them.
- Prefer typed, testable Python modules. Avoid notebooks for production logic.
- Use deterministic seeds and configuration-driven parameters.
- Never fabricate lunar data, calibration values, ground truth, or achieved metrics.
- Keep sensor-specific behavior behind adapters; core matching must remain generic.
- Handle 8/16/32-bit grayscale, NaN/nodata, large rasters, and optional geospatial metadata.
- Fail with actionable messages when optional dependencies or metadata are absent.
- Add/update tests and documentation in every task.
- Run the task's stated checks. Report changed files, commands, results, assumptions, and remaining risks.

## Definition of done for each task
1. Acceptance criteria are met.
2. Tests for changed behavior pass.
3. Formatting/type checks pass when configured.
4. No secrets, datasets, generated model weights, or large outputs are committed.
5. The task checkbox may be marked complete only after verification.

## Technical principles
- Baseline first: OpenCV/SciPy/scikit-image before learned methods.
- Sub-pixel coordinates are floating-point `(x_source, y_source, x_reference, y_reference)`.
- Separate matching error from final image-warp quality.
- Avoid relying on raw intensity correlation across modalities.
- Enforce spatial distribution using grid/quadtree selection and report coverage.
- Preserve transforms through crop, resize, pyramid and map-projection operations.
