# Evaluation Protocol

## Metrics
1. **Control-point error**: Euclidean distance between predicted and known reference pixels after applying the fitted transform; RMSE, median, P90/P95 and max.
2. **Inlier count/ratio** under a declared threshold and robust estimator.
3. **Coverage**: occupied valid-overlap grid cells / valid grid cells; convex-hull area / valid overlap area.
4. **Uniformity**: entropy or coefficient of variation of per-cell accepted match counts.
5. **Repeatability** under illumination, rotation, scale and noise perturbations.
6. **Registration quality**: edge/gradient alignment metrics; intensity metrics only for comparable modalities.
7. Runtime and peak memory.

## Experiments
- Identity and known synthetic transforms.
- Same-sensor/different acquisition.
- OHRC↔LRO NAC, TMC-2↔LRO, IIRS-derived band↔reference.
- Scale-ratio and sun-angle bins.
- Ablations: normalization, representation, scale search, refiner, uniform selector, geometric model.

## Leakage controls
No patches from the same geographic scene across optimization and held-out test sets. Freeze thresholds before final test. Record failures and zero-match runs.

## Config Freezing for Held-Out Benchmarks
To execute an unbiased evaluation on held-out test sets:
1. **Tune Parameters on Development Set**: Adjust preprocessing parameters, feature thresholds, and RANSAC inlier distance thresholds on tuning scene pairs.
2. **Freeze Configuration File**: Export and save the finalized YAML configuration file (`frozen_config.yaml`). Compute and record the SHA-256 config hash (`lunarmatch validate-config --config frozen_config.yaml`).
3. **Execute Benchmark**: Run `lunarmatch benchmark --manifest heldout_pairs.json --config frozen_config.yaml --output outputs/heldout_benchmark --workers 4`.
4. **Audit Manifests**: Verify that all per-pair `run_manifest.json` files record the identical `config_hash` matching `frozen_config.yaml`.

## Accuracy language
Report 95% confidence intervals across image pairs where sample size permits. Do not label a method “sub-pixel accurate” unless held-out control-point RMSE is <1 pixel with uncertainty and coordinate mappings audited.

