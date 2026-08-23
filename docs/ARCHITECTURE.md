# Architecture

```text
CLI/API
  └─ orchestrator
      ├─ config + manifests
      ├─ raster/metadata adapters
      ├─ preprocess + structural representations
      ├─ pyramid/tiling + overlap planning
      ├─ detector/descriptor registry
      ├─ matcher + geometric verifier
      ├─ spatial coverage selector
      ├─ sub-pixel refiner
      ├─ transform estimator + warper
      └─ evaluator + artifact/report writer
```

## Package boundaries
- `data`: raster reading, manifests, sensor metadata, masks and coordinate mappings.
- `preprocessing`: radiometric normalization and modality-invariant representations.
- `features`: backend protocol and classical/learned adapters.
- `matching`: descriptor matching, scale hypotheses and spatial selection.
- `geometry`: model fitting, degeneracy, refinement and warping.
- `evaluation`: metrics, control points, synthetic tests and benchmarking.
- `visualization`: overlays and HTML report.
- `models`: dataclasses/Pydantic-style domain models; avoid dependency cycles.
- `cli`: commands and orchestration.

## Extension interfaces
Define protocols for `RasterReader`, `Preprocessor`, `FeatureBackend`, `Matcher`, `GeometricVerifier`, `SubpixelRefiner`, and `ArtifactWriter`. Registries should reject unknown names clearly.

## Model strategy
Milestone A: classical baseline.  
Milestone B: optional adapters for LightGlue/SuperPoint and/or RoMa/LoFTR-like matchers after license, memory and lunar-domain validation.  
Milestone C: domain adaptation using synthetic illumination/scale transformations and curated pseudo-labels. Never auto-download weights without explicit user action.
