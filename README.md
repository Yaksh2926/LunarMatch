# LunarMatch — SIH 2026 PS 26166

Antigravity-ready starter repository for **multi-modal, sun-angle and scale-invariant lunar image correspondence** using Chandrayaan-2 OHRC, TMC-2, IIRS and reference imagery such as LRO NAC.

> This is an engineering scaffold and execution plan—not a claim of completed registration accuracy. Dataset-specific calibration and validation are required.

## Start here
1. Extract this folder into your project directory.
2. Open the folder in Antigravity IDE.
3. Read `ANTIGRAVITY_CONTEXT.md` and `docs/PROJECT_SPEC.md`.
4. Open `TASKLIST.md` and submit **Task 00**, then each task in order.
5. Require Antigravity to satisfy each task's acceptance checks before moving on.

## Intended deliverables
- CLI and Python package for image-pair correspondence and registration.
- Match-point CSV/GeoJSON, registered raster, transform, confidence, and diagnostics.
- Metrics: reprojection RMSE, inlier count/ratio, coverage, repeatability and runtime.
- Classical baseline plus optional learned multi-modal matcher adapters.
- Reproducible experiment configuration and reports.

## Proposed pipeline
Metadata/ISIS/GDAL preparation → robust normalization → image pyramid → candidate overlap → modality-invariant features → coarse-to-fine matching → geometric verification → uniform spatial selection → sub-pixel refinement → transform/warp → metrics and visual report.

## Quick commands (after tasks implement them)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
lunarmatch validate-config --config configs/baseline.yaml
lunarmatch register --config configs/baseline.yaml --source data/source.tif --reference data/reference.tif --output outputs/run_001
pytest -q
```

## Data
Do not commit mission imagery. Place local files under `data/` (gitignored). See `docs/DATASET_GUIDE.md` and `sample_data/metadata/pair_manifest.example.csv`.
