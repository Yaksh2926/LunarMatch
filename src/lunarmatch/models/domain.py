"""Typed domain models for lunar image metadata, correspondences, transforms, and metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RasterMetadata(BaseModel):
    """Metadata representation for a single raster image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    width: int = Field(gt=0, description="Raster width in pixels.")
    height: int = Field(gt=0, description="Raster height in pixels.")
    bands: int = Field(default=1, ge=1, description="Number of raster bands.")
    dtype: str = Field(default="uint8", description="Numpy-style data type string.")
    pixel_scale_x: float | None = Field(default=None, gt=0, description="Pixel scale X in meters/pixel.")
    pixel_scale_y: float | None = Field(default=None, gt=0, description="Pixel scale Y in meters/pixel.")
    crs: str | None = Field(default=None, description="Coordinate Reference System string or WKT.")
    transform: tuple[float, ...] | None = Field(default=None, description="6-element affine geotransform tuple.")
    nodata_value: float | None = Field(default=None, description="Nodata value if specified.")
    sensor: str | None = Field(default=None, description="Sensor name (e.g. OHRC, TMC2, IIRS, LRO_NAC).")
    acquisition_time: str | None = Field(default=None, description="Acquisition timestamp ISO 8601 string.")
    sun_azimuth_deg: float | None = Field(default=None, description="Sun azimuth angle in degrees.")
    sun_elevation_deg: float | None = Field(default=None, description="Sun elevation angle in degrees.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RasterMetadata:
        """Deserialize metadata from dictionary."""
        return cls.model_validate(data)


class ImagePair(BaseModel):
    """Pair of source and reference raster metadata for registration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_id: str = Field(description="Unique pair identifier.")
    source_meta: RasterMetadata = Field(description="Source image metadata.")
    reference_meta: RasterMetadata = Field(description="Reference image metadata.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize pair to dictionary."""
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImagePair:
        """Deserialize pair from dictionary."""
        return cls.model_validate(data)


class KeypointSet:
    """Set of detected feature keypoints with Nx2 float64 (x, y) coordinates and descriptors."""

    def __init__(
        self,
        coordinates: np.ndarray,
        scales: np.ndarray | None = None,
        angles: np.ndarray | None = None,
        responses: np.ndarray | None = None,
        descriptors: np.ndarray | None = None,
    ) -> None:
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.size == 0:
            coords = np.empty((0, 2), dtype=np.float64)
        elif coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Keypoint coordinates must have shape (N, 2), got {coords.shape}")
        elif not np.all(np.isfinite(coords)):
            raise ValueError("Keypoint coordinates contain non-finite values (NaN or Inf)")

        n_pts = len(coords)

        if scales is None:
            scales_arr = np.ones(n_pts, dtype=np.float32)
        else:
            scales_arr = np.asarray(scales, dtype=np.float32)
            if scales_arr.shape != (n_pts,):
                raise ValueError(f"Scales must have shape ({n_pts},), got {scales_arr.shape}")
            if not np.all(np.isfinite(scales_arr)):
                raise ValueError("Scales contain non-finite values")

        if angles is None:
            angles_arr = np.zeros(n_pts, dtype=np.float32)
        else:
            angles_arr = np.asarray(angles, dtype=np.float32)
            if angles_arr.shape != (n_pts,):
                raise ValueError(f"Angles must have shape ({n_pts},), got {angles_arr.shape}")
            if not np.all(np.isfinite(angles_arr)):
                raise ValueError("Angles contain non-finite values")

        if responses is None:
            responses_arr = np.zeros(n_pts, dtype=np.float32)
        else:
            responses_arr = np.asarray(responses, dtype=np.float32)
            if responses_arr.shape != (n_pts,):
                raise ValueError(f"Responses must have shape ({n_pts},), got {responses_arr.shape}")

        if descriptors is not None:
            desc_arr: np.ndarray | None = np.asarray(descriptors)
            if n_pts == 0 and desc_arr is not None and desc_arr.size == 0:
                desc_arr = None
            elif desc_arr is not None and (desc_arr.ndim != 2 or len(desc_arr) != n_pts):
                raise ValueError(f"Descriptors must be 2D array with N={n_pts} rows, got shape {desc_arr.shape}")
        else:
            desc_arr = None

        self._coordinates = coords
        self._scales = scales_arr
        self._angles = angles_arr
        self._responses = responses_arr
        self._descriptors = desc_arr

    @property
    def coordinates(self) -> np.ndarray:
        """Nx2 float64 coordinates array."""
        return self._coordinates

    @property
    def scales(self) -> np.ndarray:
        """N float32 keypoint scale levels array."""
        return self._scales

    @property
    def angles(self) -> np.ndarray:
        """N float32 keypoint angles array."""
        return self._angles

    @property
    def responses(self) -> np.ndarray:
        """N float32 keypoint responses array."""
        return self._responses

    @property
    def descriptors(self) -> np.ndarray | None:
        """N x D descriptors array or None."""
        return self._descriptors

    def __len__(self) -> int:
        return len(self._coordinates)

    def to_dict(self) -> dict[str, Any]:
        """Serialize keypoints to dictionary."""
        return {
            "coordinates": self._coordinates.tolist(),
            "scales": self._scales.tolist(),
            "angles": self._angles.tolist(),
            "responses": self._responses.tolist(),
            "descriptors": self._descriptors.tolist() if self._descriptors is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeypointSet:
        """Deserialize keypoints from dictionary."""
        return cls(
            coordinates=np.array(data["coordinates"], dtype=np.float64),
            scales=np.array(data["scales"], dtype=np.float32) if data.get("scales") is not None else None,
            angles=np.array(data["angles"], dtype=np.float32) if data.get("angles") is not None else None,
            responses=np.array(data["responses"], dtype=np.float32) if data.get("responses") is not None else None,
            descriptors=np.array(data["descriptors"]) if data.get("descriptors") is not None else None,
        )


class MatchSet:
    """Set of point correspondences between source and reference rasters in float64 (x, y) coordinates."""

    def __init__(
        self,
        source_points: np.ndarray,
        reference_points: np.ndarray,
        scores: np.ndarray | None = None,
        inliers: np.ndarray | None = None,
        cell_ids: list[int] | np.ndarray | None = None,
        refine_status: list[str] | None = None,
    ) -> None:
        src = np.asarray(source_points, dtype=np.float64)
        ref = np.asarray(reference_points, dtype=np.float64)

        if src.size == 0 and ref.size == 0:
            src = np.empty((0, 2), dtype=np.float64)
            ref = np.empty((0, 2), dtype=np.float64)
        else:
            if src.ndim != 2 or src.shape[1] != 2:
                raise ValueError(f"Source points must have shape (N, 2), got {src.shape}")
            if ref.ndim != 2 or ref.shape[1] != 2:
                raise ValueError(f"Reference points must have shape (N, 2), got {ref.shape}")
            if len(src) != len(ref):
                raise ValueError(f"Source points count ({len(src)}) must match reference points count ({len(ref)})")
            if not np.all(np.isfinite(src)):
                raise ValueError("Source points contain non-finite values (NaN or Inf)")
            if not np.all(np.isfinite(ref)):
                raise ValueError("Reference points contain non-finite values (NaN or Inf)")

        n_pts = len(src)

        if scores is None:
            scores_arr = np.ones(n_pts, dtype=np.float64)
        else:
            scores_arr = np.asarray(scores, dtype=np.float64)
            if scores_arr.shape != (n_pts,):
                raise ValueError(f"Scores must have shape ({n_pts},), got {scores_arr.shape}")
            if not np.all(np.isfinite(scores_arr)):
                raise ValueError("Scores contain non-finite values")

        if inliers is None:
            inliers_arr = np.ones(n_pts, dtype=bool)
        else:
            inliers_arr = np.asarray(inliers, dtype=bool)
            if inliers_arr.shape != (n_pts,):
                raise ValueError(f"Inliers must have shape ({n_pts},), got {inliers_arr.shape}")

        if cell_ids is None:
            cell_ids_list: list[int] | None = None
        else:
            cell_ids_list = [int(c) for c in cell_ids]
            if len(cell_ids_list) != n_pts:
                raise ValueError(f"Cell IDs count ({len(cell_ids_list)}) must match points count ({n_pts})")

        if refine_status is None:
            refine_status_list: list[str] | None = None
        else:
            refine_status_list = [str(s) for s in refine_status]
            if len(refine_status_list) != n_pts:
                raise ValueError(
                    f"Refine status count ({len(refine_status_list)}) must match points count ({n_pts})"
                )

        self._source_points = src
        self._reference_points = ref
        self._scores = scores_arr
        self._inliers = inliers_arr
        self._cell_ids = cell_ids_list
        self._refine_status = refine_status_list

    @property
    def source_points(self) -> np.ndarray:
        """Nx2 float64 source points (x, y)."""
        return self._source_points

    @property
    def reference_points(self) -> np.ndarray:
        """Nx2 float64 reference points (x, y)."""
        return self._reference_points

    @property
    def scores(self) -> np.ndarray:
        """N float64 match scores."""
        return self._scores

    @property
    def inliers(self) -> np.ndarray:
        """N bool inlier mask."""
        return self._inliers

    @property
    def cell_ids(self) -> list[int] | None:
        """Optional cell IDs list."""
        return self._cell_ids

    @property
    def refine_status(self) -> list[str] | None:
        """Optional refine status list."""
        return self._refine_status

    def __len__(self) -> int:
        return len(self._source_points)

    def to_dict(self) -> dict[str, Any]:
        """Serialize match set to dictionary."""
        return {
            "source_points": self._source_points.tolist(),
            "reference_points": self._reference_points.tolist(),
            "scores": self._scores.tolist(),
            "inliers": self._inliers.tolist(),
            "cell_ids": self._cell_ids,
            "refine_status": self._refine_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchSet:
        """Deserialize match set from dictionary."""
        return cls(
            source_points=np.array(data["source_points"], dtype=np.float64),
            reference_points=np.array(data["reference_points"], dtype=np.float64),
            scores=np.array(data["scores"], dtype=np.float64) if data.get("scores") is not None else None,
            inliers=np.array(data["inliers"], dtype=bool) if data.get("inliers") is not None else None,
            cell_ids=data.get("cell_ids"),
            refine_status=data.get("refine_status"),
        )


class TransformEstimate(BaseModel):
    """Transformation model estimate (3x3 homogeneous matrix mapping source to reference)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_type: Literal["similarity", "affine", "homography"]
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = Field(
        description="3x3 matrix mapping source pixel centers to reference pixel centers."
    )
    inlier_count: int = Field(ge=0, description="Number of inlier correspondences.")
    inlier_ratio: float = Field(ge=0.0, le=1.0, description="Inlier ratio (inlier_count / total_matches).")
    rmse_px: float | None = Field(default=None, ge=0.0, description="Reprojection RMSE in pixels.")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Metadata and parameters used during fitting.")

    @field_validator("matrix")
    @classmethod
    def validate_matrix(
        cls, v: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        arr = np.array(v, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"Matrix must have shape (3, 3), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Matrix contains non-finite values")
        return v

    def get_matrix_array(self) -> np.ndarray:
        """Return 3x3 matrix as float64 numpy array."""
        return np.array(self.matrix, dtype=np.float64)

    @classmethod
    def from_matrix(
        cls,
        model_type: Literal["similarity", "affine", "homography"],
        matrix: np.ndarray,
        inlier_count: int,
        inlier_ratio: float,
        rmse_px: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> TransformEstimate:
        """Create TransformEstimate from 3x3 numpy matrix."""
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"Matrix must have shape (3, 3), got {arr.shape}")
        tup = (
            (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
            (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
            (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
        )
        return cls(
            model_type=model_type,
            matrix=tup,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_px=rmse_px,
            provenance=provenance or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize transform estimate to dictionary."""
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformEstimate:
        """Deserialize transform estimate from dictionary."""
        return cls.model_validate(data)


class RegistrationMetrics(BaseModel):
    """Complete evaluation metrics for a pair registration run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rmse_px: float | None = Field(default=None, ge=0.0)
    median_residual_px: float | None = Field(default=None, ge=0.0)
    p90_residual_px: float | None = Field(default=None, ge=0.0)
    p95_residual_px: float | None = Field(default=None, ge=0.0)
    max_residual_px: float | None = Field(default=None, ge=0.0)
    inlier_count: int = Field(ge=0)
    total_matches: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0.0, le=1.0)
    occupied_grid_fraction: float = Field(ge=0.0, le=1.0)
    convex_hull_coverage_fraction: float = Field(ge=0.0, le=1.0)
    count_uniformity_cv: float | None = Field(default=None, ge=0.0)
    spatial_coverage_status: Literal["WELL_CONSTRAINED", "UNDER_CONSTRAINED"] = Field(
        default="UNDER_CONSTRAINED", description="Spatial coverage constraint status."
    )
    spatial_coverage_passed: bool = Field(
        default=False, description="True if spatial coverage clears minimum grid fraction & hull area thresholds."
    )
    runtime_seconds: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistrationMetrics:
        """Deserialize metrics from dictionary."""
        return cls.model_validate(data)


class RunManifest(BaseModel):
    """Run manifest capturing reproducibility information, hashes, and configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="1.0.0", description="Artifact schema version string.")
    pair_id: str
    source_path: str
    reference_path: str
    config_hash: str
    timestamp_utc: str
    package_version: str
    seed: int
    metrics: RegistrationMetrics
    transform: TransformEstimate | None = None
    quality_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary."""
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Deserialize manifest from dictionary."""
        return cls.model_validate(data)
