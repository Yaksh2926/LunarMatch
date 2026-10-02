# Chandrayaan-2 OHRC ↔ TMC Resolution Normalization Experiment Report

## Executive Summary
This experiment evaluated resolution normalization (anti-aliased Gaussian downsampling) of **OHRC** ($0.25\text{ m/px}$) toward **TMC-2** ($5.0\text{ m/px}$) across 5 effective resolutions ($0.25, 0.5, 1.0, 2.5, 5.0\text{ m/px}$) and 4 image representations (Raw, CLAHE, Gradient, Edge).

### Final Verdict: **PARTIAL_SUCCESS**

---

## 1. Primary Findings
1. **Resolution Normalization Effect**: Downsampling OHRC with Gaussian anti-aliasing reduces high-frequency speckle/regolith texture and improves feature stability at lower scale ratios ($1.0\times$ to $2.0\times$).
2. **Best Performing Resolution**: **0.5\text{ m/px}** effective OHRC resolution.
3. **Best Inlier Count**: **4 inliers** (Inlier Ratio: `36.4\%`).
4. **Classical SIFT Limit**: Classical SIFT feature matching yields 4 inliers under multi-scale normalization, which is insufficient ($\ge 15$ required) for full automatic sub-pixel registration across cross-payload Chandrayaan-2 optical sensors.

---

## 2. Deliverable Output Artifacts
1. `01_footprint_overlap.png`
2. `02_common_ground_region.png`
3. `03_resolution_pyramid.png`
4. `04_matching_results_table.csv`
5. `05_resolution_ablation.png`
6. `06_best_raw_matches.png`
7. `07_best_ransac_inliers.png`
8. `08_best_spatial_distribution.png`
9. `09_registered_output.png`
10. `10_difference_before.png`
11. `11_difference_after.png`
12. `12_match_points.csv`
13. `13_registration_manifest.json`
14. `14_FINAL_REPORT.md`
