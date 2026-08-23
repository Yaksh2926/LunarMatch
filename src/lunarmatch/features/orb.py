"""OpenCV ORB feature extraction backend implementation."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from lunarmatch.features.tiling import plan_tiles
from lunarmatch.geometry.coordinate_mapping import CoordinateMapping
from lunarmatch.models.config import FeaturesConfig
from lunarmatch.models.domain import KeypointSet


def suppress_duplicate_keypoints(
    coordinates: np.ndarray,
    scales: np.ndarray,
    angles: np.ndarray,
    responses: np.ndarray,
    descriptors: np.ndarray,
    distance_threshold_px: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Suppress duplicate keypoints in overlapping tile regions.

    Keep keypoints with higher responses when spatial distance is below threshold.
    """
    n_pts = len(coordinates)
    if n_pts <= 1:
        return coordinates, scales, angles, responses, descriptors

    # Sort candidates by response descending
    sort_idx = np.argsort(-responses)
    coords_sorted = coordinates[sort_idx]
    scales_sorted = scales[sort_idx]
    angles_sorted = angles[sort_idx]
    resp_sorted = responses[sort_idx]
    desc_sorted = descriptors[sort_idx]

    keep_mask = np.ones(n_pts, dtype=bool)

    tree = cKDTree(coords_sorted)
    pairs = tree.query_pairs(r=distance_threshold_px)

    for _, j in pairs:
        keep_mask[j] = False

    kept_indices = np.where(keep_mask)[0]

    coords_kept = coords_sorted[kept_indices]
    scales_kept = scales_sorted[kept_indices]
    angles_kept = angles_sorted[kept_indices]
    resp_kept = resp_sorted[kept_indices]
    desc_kept = desc_sorted[kept_indices]

    return coords_kept, scales_kept, angles_kept, resp_kept, desc_kept


def sort_keypoints_deterministically(
    coordinates: np.ndarray,
    scales: np.ndarray,
    angles: np.ndarray,
    responses: np.ndarray,
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort keypoints deterministically by response descending, then x ascending, then y ascending."""
    n_pts = len(coordinates)
    if n_pts <= 1:
        return coordinates, scales, angles, responses, descriptors

    keys = np.empty(n_pts, dtype=[("resp", np.float32), ("x", np.float64), ("y", np.float64)])
    keys["resp"] = -responses
    keys["x"] = coordinates[:, 0]
    keys["y"] = coordinates[:, 1]

    order = np.argsort(keys, order=["resp", "x", "y"])

    return (
        coordinates[order],
        scales[order],
        angles[order],
        responses[order],
        descriptors[order],
    )


class ORBFeatureBackend:
    """ORB classical feature extraction backend using OpenCV."""

    def __init__(self, config: FeaturesConfig | None = None) -> None:
        self._config = config or FeaturesConfig(backend="orb")
        self._max_keypoints = self._config.max_keypoints
        self._tile_size = self._config.tile_size
        self._tile_overlap = self._config.tile_overlap

    @property
    def name(self) -> str:
        return "orb"

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        keypoint_mapping: CoordinateMapping | None = None,
    ) -> KeypointSet:
        """Detect ORB keypoints and compute descriptors."""
        img_arr = np.asarray(image)
        if img_arr.ndim != 2:
            raise ValueError(f"Image must be 2D grayscale, got shape {img_arr.shape}")

        height, width = img_arr.shape
        if height == 0 or width == 0:
            return KeypointSet(
                coordinates=np.empty((0, 2), dtype=np.float64),
                descriptors=np.empty((0, 32), dtype=np.uint8),
            )

        # Convert image to uint8 [0, 255] if necessary
        if img_arr.dtype == np.uint8:
            img_uint8 = img_arr
        else:
            img_float = img_arr.astype(np.float32)
            valid = np.isfinite(img_float)
            if not np.any(valid):
                return KeypointSet(
                    coordinates=np.empty((0, 2), dtype=np.float64),
                    descriptors=np.empty((0, 32), dtype=np.uint8),
                )
            min_v, max_v = img_float[valid].min(), img_float[valid].max()
            if max_v > min_v:
                normed = (img_float - min_v) / (max_v - min_v) * 255.0
            else:
                normed = np.zeros_like(img_float)
            img_uint8 = np.clip(normed, 0, 255).astype(np.uint8)

        # Convert mask to uint8
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.shape != (height, width):
                raise ValueError(f"Mask shape {mask_arr.shape} must match image shape ({height}, {width})")
            mask_uint8 = (mask_arr > 0).astype(np.uint8) * 255
        else:
            mask_uint8 = np.ones((height, width), dtype=np.uint8) * 255

        # Check if tiling is required
        use_tiling = (width > self._tile_size) or (height > self._tile_size)

        all_coords: list[np.ndarray] = []
        all_scales: list[np.ndarray] = []
        all_angles: list[np.ndarray] = []
        all_responses: list[np.ndarray] = []
        all_descriptors: list[np.ndarray] = []

        create_orb = cv2.ORB_create  # type: ignore[attr-defined]

        if use_tiling:
            tiles = plan_tiles(
                image_shape=(height, width),
                mask=mask_uint8 > 0,
                tile_size=self._tile_size,
                tile_overlap=self._tile_overlap,
                max_megapixels_in_memory=None,
            )
            kps_per_tile = max(100, self._max_keypoints // max(1, len(tiles)))
            orb_detector = create_orb(nfeatures=kps_per_tile)

            for tile in tiles:
                w = tile.window
                tile_img = img_uint8[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width]
                tile_mask = mask_uint8[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width]

                if not np.any(tile_mask):
                    continue

                kps, descs = orb_detector.detectAndCompute(tile_img, mask=tile_mask)
                if kps and descs is not None and len(kps) > 0:
                    pts = np.array([kp.pt for kp in kps], dtype=np.float64)
                    pts[:, 0] += w.col_off
                    pts[:, 1] += w.row_off

                    scales = np.array([kp.size for kp in kps], dtype=np.float32)
                    angles = np.array([kp.angle for kp in kps], dtype=np.float32)
                    responses = np.array([kp.response for kp in kps], dtype=np.float32)

                    all_coords.append(pts)
                    all_scales.append(scales)
                    all_angles.append(angles)
                    all_responses.append(responses)
                    all_descriptors.append(descs.astype(np.uint8))
        else:
            orb_detector = create_orb(nfeatures=self._max_keypoints)
            kps, descs = orb_detector.detectAndCompute(img_uint8, mask=mask_uint8)
            if kps and descs is not None and len(kps) > 0:
                pts = np.array([kp.pt for kp in kps], dtype=np.float64)
                scales = np.array([kp.size for kp in kps], dtype=np.float32)
                angles = np.array([kp.angle for kp in kps], dtype=np.float32)
                responses = np.array([kp.response for kp in kps], dtype=np.float32)

                all_coords.append(pts)
                all_scales.append(scales)
                all_angles.append(angles)
                all_responses.append(responses)
                all_descriptors.append(descs.astype(np.uint8))

        if not all_coords:
            return KeypointSet(
                coordinates=np.empty((0, 2), dtype=np.float64),
                descriptors=np.empty((0, 32), dtype=np.uint8),
            )

        coords_arr = np.vstack(all_coords)
        scales_arr = np.concatenate(all_scales)
        angles_arr = np.concatenate(all_angles)
        resp_arr = np.concatenate(all_responses)
        desc_arr = np.vstack(all_descriptors)

        # Deduplicate overlap keypoints if tiling was used
        if use_tiling and len(coords_arr) > 1:
            coords_arr, scales_arr, angles_arr, resp_arr, desc_arr = suppress_duplicate_keypoints(
                coords_arr, scales_arr, angles_arr, resp_arr, desc_arr, distance_threshold_px=1.5
            )

        # Truncate to max_keypoints if needed
        if len(coords_arr) > self._max_keypoints:
            top_idx = np.argsort(-resp_arr)[: self._max_keypoints]
            coords_arr = coords_arr[top_idx]
            scales_arr = scales_arr[top_idx]
            angles_arr = angles_arr[top_idx]
            resp_arr = resp_arr[top_idx]
            desc_arr = desc_arr[top_idx]

        # Sort deterministically
        coords_arr, scales_arr, angles_arr, resp_arr, desc_arr = sort_keypoints_deterministically(
            coords_arr, scales_arr, angles_arr, resp_arr, desc_arr
        )

        # Map working image coordinates back to original space if mapping provided
        if keypoint_mapping is not None:
            mapped_coords = keypoint_mapping.apply(coords_arr)
        else:
            mapped_coords = coords_arr

        return KeypointSet(
            coordinates=mapped_coords,
            scales=scales_arr,
            angles=angles_arr,
            responses=resp_arr,
            descriptors=desc_arr,
        )
