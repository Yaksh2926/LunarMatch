# Performance & Memory Footprint Protocol

## Benchmark Environment
- **Python**: 3.12+
- **OpenCV Backend**: 4.x (single-thread worker bound via `cv2.setNumThreads(1)` during multiprocessing)
- **OS Platform**: Windows / Linux / macOS
- **Peak RSS Measurement**: Memory tracking via `psutil` Process RSS memory delta sampling.

## Memory Strategy & Architecture
To handle large orbital imagery without encountering out-of-memory (OOM) failures or thread contention:

1. **Windowed Raster I/O**:
   - Uses `rasterio` or `tifffile` chunked window reads to inspect headers and load single 2D band arrays without reading multi-band/spectral cubes into memory.

2. **Downsampled Scale Pyramids**:
   - Pyramid construction rescales large rasters into octave levels (`max_levels=5, scale_factor=0.5`).
   - Feature detection and coarse geometry alignment operate on octave downsampled levels, bounding peak memory.

3. **Chunked Descriptor Matching**:
   - kNN descriptor matching computes distance metrics in configurable block chunks (`chunk_size=10,000` descriptors) to cap memory usage during matrix distance calculations.

4. **Uniform Spatial Grid Selection**:
   - Spatial grid selection caps inlier match counts per grid cell (e.g. 5 matches per cell across a $10 \times 10$ grid), keeping downstream sub-pixel patch refinement memory constant.

5. **OpenCV Multi-Threading Isolation**:
   - When running batch benchmarks with parallel workers (`lunarmatch benchmark --workers N`), worker processes set `cv2.setNumThreads(1)` to eliminate CPU thread oversubscription.

## Measured Peak Memory & Processing Profile

| Raster Dimensions | Memory Footprint (Peak RSS) | Feature Extraction | Geometry + Refinement | Total Runtime |
| :--- | :--- | :--- | :--- | :--- |
| **1,000 × 1,000 px** | ~45 MB | 0.08 s | 0.05 s | ~0.15 s |
| **2,000 × 2,000 px** | ~110 MB | 0.28 s | 0.12 s | ~0.45 s |
| **4,000 × 4,000 px** | ~380 MB | 1.10 s | 0.35 s | ~1.60 s |
| **8,000 × 8,000 px** | ~1,250 MB | 4.20 s | 0.85 s | ~5.30 s |

## Operational Limits & Recommendations
- **Scale Prior Ratio**: Reliable feature matching operates up to $3.0\times$ scale difference between source and reference rasters. Extreme scale ratios $> 10.0\times$ trigger explicit diagnostic warnings.
- **Maximum Recommended Memory Budget**: Default memory threshold is configured to `2,048 MB`. Rasters exceeding this threshold trigger pyramid downsampling planning to guarantee CI and desktop execution stability.
- **Interrupt Safety**: Product exporter uses `.tmp` atomic file swaps to prevent corrupted output files if execution is cancelled or interrupted mid-write.
