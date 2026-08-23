# Architectural Decision Records (ADRs)

This document records key technical and architectural decisions for `LunarMatch`.

---

## ADR-0001: Floating-Point Coordinate Conventions and Origin Standard

- **Status**: Accepted
- **Date**: 2026-08-23

### Context
Image registration pipelines must maintain strict mathematical consistency across array indexing (NumPy), feature detection (OpenCV/scikit-image), sub-pixel refinement, and geospatial raster exports. Arrays in NumPy use `[row, col]` (where `row` corresponds to `y` and `col` corresponds to `x`), whereas spatial geometry and feature points operate on `(x, y)` pairs. Misalignments between top-left vs center pixel origins lead to Systematic sub-pixel offset errors.

### Decision
1. **Coordinate System**: All pixel coordinates are floating-point 64-bit tuples `(x, y)` where `x = column` (horizontal, `0.0 <= x <= Width`) and `y = row` (vertical, `0.0 <= y <= Height`).
2. **Origin & Center Definition**: The origin `(0.0, 0.0)` is defined at the outer top-left corner of the top-left pixel. The center of pixel `(row, col)` corresponds to floating-point coordinates `(col + 0.5, row + 0.5)`.
3. **Transformation Direction**: All 3x3 homogeneous transformation matrices $H$ map **source pixel centers to reference pixel centers**:
   $$\begin{bmatrix} x_{ref} \\ y_{ref} \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x_{src} \\ y_{src} \\ 1 \end{bmatrix}$$
4. **Pyramid & Crop Composition**: Every crop, resize, or pyramid level exposes a explicit 3x3 affine mapping $M_{level \to orig}$ to translate feature coordinates back to the original full-resolution image coordinate space.

### Consequences
- Eliminates row/column ambiguity across modules.
- Simplifies multi-resolution scale searches and patch extractions.
- Ensures exact numerical round-trip properties in sub-pixel refinement tests.

---

## ADR-0002: Baseline Geometric Model Selection and USAC/MAGSAC Fallback

- **Status**: Accepted
- **Date**: 2026-08-23

### Context
Chandrayaan-2 and reference lunar images present varying spatial resolutions, illumination angles, and viewpoint distortions. While 3x3 homographies or thin-plate splines can model complex surfaces, high-parameter models are prone to overfitting or geometric degeneracy when inlier counts are low or feature distributions are spatially clustered.

### Decision
1. **Default Baseline Model**: The classical baseline standardizes on an **Affine Transformation** (6 degrees of freedom) as configured in `configs/baseline.yaml` (`model: affine`).
2. **Supported Models**: The architecture supports `similarity` (4 DoF), `affine` (6 DoF), and `homography` (8 DoF), with selection controlled via configuration.
3. **Robust Estimation**: Robust fitting uses OpenCV's `USAC_MAGSAC` when available, with explicit fallback to standard `RANSAC`.
4. **Degeneracy & Plausibility Checks**: Estimated transforms must satisfy strict sanity checks:
   - Scale factor bounds: `allowed_scale: [0.01, 100.0]`.
   - Non-zero determinant (no singularity/collinearity).
   - Reflection check (determinant sign preservation).

### Consequences
- Protects registration against extreme warping artifacts on sparse or noisy matches.
- Guarantees numerical stability and predictable convergence during baseline execution.

---

## ADR-0003: Optional Geospatial Dependencies and Graceful Fallback Strategy

- **Status**: Accepted
- **Date**: 2026-08-23

### Context
Geospatial libraries (`rasterio`, `pyproj`, `shapely`) rely on C/C++ system libraries (GDAL, PROJ, GEOS) which may be difficult to compile or install in constrained offline evaluation environments. However, export of GeoTIFF spatial metadata and GeoJSON matches is required when geospatial headers are present.

### Decision
1. **Optional Dependency Classification**: `rasterio`, `shapely`, and `pyproj` are classified as optional dependencies under `[project.optional-dependencies] geo`.
2. **Core I/O Fallback**: Core raster reading and writing defaults to pure Python/OpenCV/`tifffile` implementations without requiring GDAL/`rasterio`.
3. **Graceful Degradation**:
   - If `rasterio` is unavailable, `LunarMatch` processes input rasters purely in pixel space.
   - Spatial metadata export (`matches.geojson`) is omitted with an informative warning when geospatial packages are missing.
   - Primary outputs (`registered.tif`, `matches.csv`, `transform.json`, `metrics.json`) continue to generate reliably.

### Consequences
- Guarantees zero installation friction in lightweight or CPU-only evaluation environments.
- Retains full geospatial capabilities when optional GIS packages are installed.

---

## ADR-0004: Explicit Local Model Weights & Prohibiting Automatic Downloads

- **Status**: Accepted
- **Date**: 2026-08-23

### Context
Deep learning-based feature matchers (e.g. SuperPoint, LightGlue, LoFTR) may be introduced in Phase 3 as optional adapters. Online model hubs (PyTorch Hub, Hugging Face) typically attempt to download pre-trained weights automatically at runtime. In air-gapped, competition, or offline scientific environments, internet access is disabled or restricted.

### Decision
1. **No Automatic Network Downloads**: `LunarMatch` prohibits automatic network downloads of model weights or assets during runtime.
2. **Explicit Weight Path Requirement**: Optional learned matcher adapters must require explicit local filesystem paths to model weight files specified in configuration.
3. **Integrity Verification**: Model weights must be verified using SHA-256 checksums before loading.
4. **Actionable Errors**: If model weight files are missing or checksums fail, the adapter must raise a clear error detailing missing files without crashing the process silently.

### Consequences
- Guarantees 100% offline reproducibility and compliance with competition execution guidelines.
- Prevents unexpected network timeouts or silent failures during batch runs.
