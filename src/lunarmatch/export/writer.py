"""Atomic product exporter for raster, JSON manifests, CSV matches, and GeoJSON outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from lunarmatch.models.domain import MatchSet, RasterMetadata, RunManifest

try:
    import rasterio  # type: ignore[import-untyped]
    from rasterio.transform import Affine  # type: ignore[import-untyped]

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    """Write dictionary to JSON file atomically using temporary file swap."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _atomic_csv_export(matches: MatchSet, path: Path) -> None:
    """Export matches CSV file atomically using temporary file swap."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "score",
                "inlier",
                "cell_id",
                "refine_status",
            ]
        )
        src = matches.source_points
        ref = matches.reference_points
        scores = matches.scores
        inliers = matches.inliers
        cell_ids = matches.cell_ids
        statuses = matches.refine_status

        for i in range(len(src)):
            cid = cell_ids[i] if cell_ids is not None and i < len(cell_ids) else ""
            status = statuses[i] if statuses is not None and i < len(statuses) else ""
            writer.writerow(
                [
                    f"{src[i, 0]:.6f}",
                    f"{src[i, 1]:.6f}",
                    f"{ref[i, 0]:.6f}",
                    f"{ref[i, 1]:.6f}",
                    f"{scores[i]:.6f}",
                    int(inliers[i]),
                    cid,
                    status,
                ]
            )
    tmp_path.replace(path)


def export_geojson_matches(
    matches: MatchSet,
    reference_metadata: RasterMetadata,
    path: Path,
) -> None:
    """Export matches as GeoJSON FeatureCollection using reference geotransform."""
    if reference_metadata.transform is None or reference_metadata.crs is None:
        return

    c, a, b, f, d, e = reference_metadata.transform
    features: list[dict[str, Any]] = []

    src = matches.source_points
    ref = matches.reference_points
    scores = matches.scores
    inliers = matches.inliers

    for i in range(len(ref)):
        xr, yr = ref[i, 0], ref[i, 1]
        easting = c + a * xr + b * yr
        northing = f + d * xr + e * yr

        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [easting, northing],
            },
            "properties": {
                "match_index": i,
                "source_x": float(src[i, 0]),
                "source_y": float(src[i, 1]),
                "reference_x": float(xr),
                "reference_y": float(yr),
                "score": float(scores[i]),
                "inlier": bool(inliers[i]),
            },
        }
        features.append(feat)

    geojson_data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": reference_metadata.crs},
        },
        "features": features,
    }

    _atomic_json_write(path, geojson_data)


def export_registered_products(
    manifest: RunManifest,
    matches: MatchSet,
    warped_image: np.ndarray | None,
    source_metadata: RasterMetadata,
    reference_metadata: RasterMetadata,
    output_dir: Path,
    overwrite: bool = False,
    export_geojson: bool = True,
    stage_timings: dict[str, float] | None = None,
) -> dict[str, Path]:
    """Export all registration artifacts atomically into output_dir.

    Args:
        manifest: RunManifest instance.
        matches: MatchSet of correspondences.
        warped_image: Optional 2D warped source image array.
        source_metadata: RasterMetadata of source image.
        reference_metadata: RasterMetadata of reference image.
        output_dir: Destination output directory.
        overwrite: If True, overwrite existing files; if False, raise FileExistsError.
        export_geojson: If True, export GeoJSON files when CRS/transform permit.
        stage_timings: Optional dict of per-stage execution durations.

    Returns:
        Dict mapping artifact keys ('manifest', 'metrics', 'transform', 'matches', 'registered') to Paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "run_manifest.json"
    metrics_path = out_dir / "metrics.json"
    transform_path = out_dir / "transform.json"
    matches_csv_path = out_dir / "matches.csv"
    registered_path = out_dir / "registered.tif"
    matches_geojson_path = out_dir / "matches.geojson"
    diagnostics_path = out_dir / "diagnostics.json"

    # Check overwrite policy
    check_paths = [manifest_path, metrics_path, matches_csv_path, diagnostics_path]
    if manifest.transform is not None:
        check_paths.append(transform_path)
    if warped_image is not None:
        check_paths.append(registered_path)

    if not overwrite:
        for p in check_paths:
            if p.is_file():
                raise FileExistsError(
                    f"Output file '{p.name}' already exists in '{out_dir}' and overwrite is False."
                )

    exported_files: dict[str, Path] = {}

    # 1. run_manifest.json
    _atomic_json_write(manifest_path, manifest.to_dict())
    exported_files["manifest"] = manifest_path

    # 2. metrics.json
    _atomic_json_write(metrics_path, manifest.metrics.to_dict())
    exported_files["metrics"] = metrics_path

    # 3. transform.json
    if manifest.transform is not None:
        _atomic_json_write(transform_path, manifest.transform.to_dict())
        exported_files["transform"] = transform_path

    # 4. matches.csv
    _atomic_csv_export(matches, matches_csv_path)
    exported_files["matches"] = matches_csv_path

    # 5. diagnostics.json
    diag_data = {
        "pair_id": manifest.pair_id,
        "quality_gate_passed": manifest.quality_gate_passed,
        "stage_timings_seconds": stage_timings or {},
        "warnings": manifest.metrics.warnings,
    }
    _atomic_json_write(diagnostics_path, diag_data)
    exported_files["diagnostics"] = diagnostics_path

    # 6. registered.tif
    if warped_image is not None:
        tmp_reg = registered_path.with_name(f".{registered_path.name}.tmp")
        arr_to_write = np.asarray(warped_image)

        if HAS_RASTERIO and reference_metadata.crs is not None and reference_metadata.transform is not None:
            c, a, b, f, d, e = reference_metadata.transform
            aff_transform = Affine(a, b, c, d, e, f)
            nodata_val = reference_metadata.nodata_value or 0.0

            profile = {
                "driver": "GTiff",
                "height": reference_metadata.height,
                "width": reference_metadata.width,
                "count": 1,
                "dtype": str(arr_to_write.dtype),
                "crs": reference_metadata.crs,
                "transform": aff_transform,
                "nodata": nodata_val,
            }
            with rasterio.open(tmp_reg, "w", **profile) as dst:
                dst.write(arr_to_write, 1)
        else:
            tifffile.imwrite(str(tmp_reg), arr_to_write)

        tmp_reg.replace(registered_path)
        exported_files["registered"] = registered_path

    # 7. matches.geojson
    if export_geojson and reference_metadata.crs is not None and reference_metadata.transform is not None:
        export_geojson_matches(matches, reference_metadata, matches_geojson_path)
        if matches_geojson_path.is_file():
            exported_files["geojson"] = matches_geojson_path

    # 8. report.html
    report_path = out_dir / "report.html"
    try:
        from lunarmatch.visualization.report import generate_html_report

        generate_html_report(
            manifest=manifest,
            matches=matches,
            warped_image=warped_image,
            output_path=report_path,
            stage_timings=stage_timings,
        )
        if report_path.is_file():
            exported_files["report"] = report_path
    except Exception:  # noqa: BLE001, S110
        pass

    return exported_files
