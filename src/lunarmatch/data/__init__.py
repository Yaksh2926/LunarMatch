"""Data access and metadata serialization tools."""
from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

from lunarmatch.data.raster import (
    RasterReadError,
    RasterReadResult,
    RasterWindow,
    is_rasterio_available,
    read_raster,
    read_raster_metadata,
)
from lunarmatch.models.domain import ImagePair, RasterMetadata


class PairManifestEntry(BaseModel):
    """Pair specification entry parsed from CSV."""
    pair_id: str
    source_path: Path
    reference_path: Path
    source_sensor: str | None = None
    reference_sensor: str | None = None
    source_pixel_scale_m: float | None = None
    reference_pixel_scale_m: float | None = None
    source_sun_azimuth_deg: float | None = None
    source_sun_elevation_deg: float | None = None
    reference_sun_azimuth_deg: float | None = None
    reference_sun_elevation_deg: float | None = None
    control_points_path: Path | None = None

    def to_image_pair(self) -> ImagePair:
        """Convert entry to domain ImagePair, loading actual metadata if files exist."""
        try:
            source_meta = read_raster_metadata(self.source_path)
        except Exception:  # noqa: BLE001
            source_meta = RasterMetadata(
                path=self.source_path,
                width=100,
                height=100,
                bands=1,
                dtype="uint8",
                pixel_scale_x=self.source_pixel_scale_m,
                pixel_scale_y=self.source_pixel_scale_m,
                sensor=self.source_sensor,
                sun_azimuth_deg=self.source_sun_azimuth_deg,
                sun_elevation_deg=self.source_sun_elevation_deg,
            )
        try:
            reference_meta = read_raster_metadata(self.reference_path)
        except Exception:  # noqa: BLE001
            reference_meta = RasterMetadata(
                path=self.reference_path,
                width=100,
                height=100,
                bands=1,
                dtype="uint8",
                pixel_scale_x=self.reference_pixel_scale_m,
                pixel_scale_y=self.reference_pixel_scale_m,
                sensor=self.reference_sensor,
                sun_azimuth_deg=self.reference_sun_azimuth_deg,
                sun_elevation_deg=self.reference_sun_elevation_deg,
            )
        return ImagePair(
            pair_id=self.pair_id,
            source_meta=source_meta,
            reference_meta=reference_meta,
        )


def parse_pair_manifest(
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> list[PairManifestEntry]:
    """Parse CSV manifest of image registration pairs.

    Args:
        manifest_path: Path to pair manifest CSV file.
        repo_root: Optional root path for resolving relative image paths.

    Returns:
        List of PairManifestEntry objects.
    """
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Pair manifest CSV file not found: {p}")

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    entries: list[PairManifestEntry] = []

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or lacks headers")

        required_cols = {"pair_id", "source_path", "reference_path"}
        fieldnames_set = set(reader.fieldnames)
        if not required_cols.issubset(fieldnames_set):
            raise ValueError(f"CSV is missing required columns: {required_cols - fieldnames_set}")

        for row in reader:
            pair_id = row.get("pair_id")
            src_p = row.get("source_path")
            ref_p = row.get("reference_path")
            if not pair_id or not src_p or not ref_p:
                continue

            def to_float(val: str | None) -> float | None:
                if val is None or val.strip() == "":
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            def to_path_optional(val: str | None) -> Path | None:
                if val is None or val.strip() == "":
                    return None
                path_val = Path(val)
                if path_val.is_absolute():
                    return path_val.resolve()
                return (root / path_val).resolve()

            def to_path_required(val: str) -> Path:
                path_val = Path(val)
                if path_val.is_absolute():
                    return path_val.resolve()
                return (root / path_val).resolve()

            entry = PairManifestEntry(
                pair_id=pair_id,
                source_path=to_path_required(src_p),
                reference_path=to_path_required(ref_p),
                source_sensor=row.get("source_sensor") or None,
                reference_sensor=row.get("reference_sensor") or None,
                source_pixel_scale_m=to_float(row.get("source_pixel_scale_m")),
                reference_pixel_scale_m=to_float(row.get("reference_pixel_scale_m")),
                source_sun_azimuth_deg=to_float(row.get("source_sun_azimuth_deg")),
                source_sun_elevation_deg=to_float(row.get("source_sun_elevation_deg")),
                reference_sun_azimuth_deg=to_float(row.get("reference_sun_azimuth_deg")),
                reference_sun_elevation_deg=to_float(row.get("reference_sun_elevation_deg")),
                control_points_path=to_path_optional(row.get("control_points_path")),
            )
            entries.append(entry)

    return entries


__all__ = [
    "ImagePair",
    "PairManifestEntry",
    "RasterMetadata",
    "RasterReadError",
    "RasterReadResult",
    "RasterWindow",
    "is_rasterio_available",
    "parse_pair_manifest",
    "read_raster",
    "read_raster_metadata",
]
