# Dataset Guide

## Sources
- Chandrayaan-2 browse portal: https://chmapbrowse.issdc.gov.in/
- LRO references provided in the problem statement. Confirm current URLs and licenses manually before automated download.

## Local layout
```text
data/
  raw/{ohrc,tmc2,iirs,lro}/
  converted/
  manifests/pairs.csv
  control_points/
```
`data/` is ignored by Git.

## Pair manifest
Required: `pair_id,source_path,reference_path,source_sensor,reference_sensor`. Optional metadata columns are shown in the example CSV. Paths should be relative to repository root where practical.

## Preparation checklist
- Retain original product and detached labels.
- Record checksum and provenance.
- Convert calibrated/map-projected products without losing nodata, CRS, transform or bit depth.
- Make source/reference footprints genuinely overlap.
- Keep raw and derived files separate.
- Split train/validation/test by geographic region/product, not random patches from one image.

## Ground truth
Preferred: independently verified control points with uncertainty and clear coordinate convention. Alternative synthetic tests may apply known transforms to real lunar crops, but must be reported separately from true cross-sensor evaluation.

## Sensor caveats
IIRS may contain spectral dimensions; an adapter must explicitly select/band-combine channels and record them. Browse products may be compressed or radiometrically altered and should not be treated as science-grade by default. Sun-angle metadata must not be silently inferred from appearance.
