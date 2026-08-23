# 5-Minute Offline Demonstration Runbook

## Overview
This runbook provides a step-by-step presentation guide for evaluating LunarMatch during live code reviews, hackathon presentations, and offline evaluation sessions.

## Demo Flow (Total: 5 Minutes)

### Step 1: Environment & Package Verification (30 Seconds)
Verify that the `lunarmatch` package is installed and operational:
```bash
lunarmatch version
lunarmatch --help
```

### Step 2: Live Registration Demo Run (1.5 Minutes)
Execute the single-command demo script. If custom mission data paths are omitted, the script automatically uses procedural synthetic lunar crater terrain as a self-contained offline fallback:
```bash
python scripts/run_demo.py --output-dir outputs/demo_run
```

**Expected Console Output**:
- Displays source and reference image paths.
- Computes multi-scale pyramid, SIFT/ORB keypoints, RANSAC geometric transform, and sub-pixel ECC refinement.
- Prints `PASS` status, inlier count, inlier ratio, reprojection RMSE (px), and runtime.

### Step 3: Product Artifact Inspection (1 Minute)
Navigate to `outputs/demo_run/` to inspect exported standardized artifacts:
1. `registered.tif`: Source image warped into reference pixel grid.
2. `matches.csv`: Standardized correspondence table (`source_x, source_y, reference_x, reference_y, score, inlier, cell_id, refine_status`).
3. `transform.json`: Typed 3x3 transformation matrix $M$ with direction (`source_to_reference`).
4. `metrics.json`: Complete quantitative evaluation metrics (RMSE, median, P90, grid coverage, uniformity).
5. `run_manifest.json`: Full reproducibility manifest capturing SHA-256 config hash, timestamp, and seed.

### Step 4: Batch Benchmark Demonstration (1 Minute)
Demonstrate multi-pair batch processing with restart safety and failure isolation:
```bash
# Generate synthetic benchmark dataset
lunarmatch generate-synthetic --output-dir outputs/demo_synth_dataset --num-pairs 3 --seed 42

# Execute batch benchmark across manifest
lunarmatch benchmark --manifest outputs/demo_synth_dataset/pairs_manifest.json --output outputs/demo_benchmark --workers 2
```

### Step 5: Visual Diagnostics HTML Report (1 Minute)
Open `outputs/demo_run/report.html` in any web browser.
- Demonstrate **100% offline usability** (zero external HTTP script/style calls).
- Highlight visual quick-look cards:
  - **Color Alignment Overlay**: Red/cyan misalignment overlay.
  - **Checkerboard Continuity**: Alternating tiles to verify edge alignment.
  - **Feature Correspondences**: Inlier (green) vs outlier (red) match lines.
  - **Spatial Occupancy Heatmap**: Cell density grid distribution.
  - **Residual Error Vector Field**: Arrow vectors showing sub-pixel residual displacements.
