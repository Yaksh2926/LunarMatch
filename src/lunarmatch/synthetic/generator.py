"""Synthetic lunar pair and benchmark dataset generator with exact ground-truth control points."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from lunarmatch.evaluation.models import ControlPoint
from lunarmatch.models.domain import TransformEstimate
from lunarmatch.synthetic.terrain import generate_crater_heightfield, render_hillshade


@dataclass
class SyntheticPairData:
    """Container for synthetic image pair, transform ground-truth, and control points."""

    pair_id: str
    source_image: np.ndarray
    reference_image: np.ndarray
    ground_truth_transform: TransformEstimate
    control_points: list[ControlPoint]
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_synthetic_pair(
    pair_id: str = "synth_pair_001",
    transform_model: str = "affine",
    angle_deg: float = 3.0,
    translation: tuple[float, float] = (8.0, -5.0),
    scale_ratio: float = 1.0,
    sun_azimuth_diff: float = 30.0,
    blur_sigma: float = 0.5,
    noise_std: float = 2.0,
    shape: tuple[int, int] = (256, 256),
    seed: int = 42,
) -> SyntheticPairData:
    """Generate a synthetic lunar image pair with exact known ground-truth transformation.

    Args:
        pair_id: Identifier string for synthetic pair.
        transform_model: Geometric model type ('similarity', 'affine', 'homography').
        angle_deg: Rotation angle in degrees applied to source image.
        translation: (tx, ty) pixel translation offset applied to source image.
        scale_ratio: Scale factor ratio.
        sun_azimuth_diff: Sun azimuth angle difference in degrees between source and reference.
        blur_sigma: Gaussian blur standard deviation applied to source image.
        noise_std: Additive Gaussian noise standard deviation.
        shape: Target (height, width) of exported image arrays.
        seed: Random seed integer for reproducible procedural generation.

    Returns:
        SyntheticPairData container.
    """
    h_target, w_target = shape
    pad = 50
    h_padded, w_padded = h_target + 2 * pad, w_target + 2 * pad
    center = (w_padded / 2.0, h_padded / 2.0)

    # 1. Generate heightfield
    heights = generate_crater_heightfield(shape=(h_padded, w_padded), num_craters=18, seed=seed)

    # 2. Render reference hillshade (Sun Azimuth = 315°)
    ref_hillshade = render_hillshade(heights, azimuth_deg=315.0, elevation_deg=45.0)

    # 3. Render source hillshade with perturbed sun angle
    src_hillshade = render_hillshade(heights, azimuth_deg=315.0 + sun_azimuth_diff, elevation_deg=45.0)

    # 4. Construct ground-truth transformation matrix M (mapping source -> reference)
    # p_ref = M @ p_src
    tx, ty = translation
    M_2x3 = cv2.getRotationMatrix2D(center, angle=angle_deg, scale=scale_ratio)
    M_2x3[0, 2] += tx
    M_2x3[1, 2] += ty

    M_3x3 = np.eye(3, dtype=np.float64)
    M_3x3[:2, :] = M_2x3

    # Warp source image using transformation M
    src_warped_img: np.ndarray = cv2.warpAffine(
        src_hillshade,
        M_2x3,
        (w_padded, h_padded),
        borderMode=cv2.BORDER_REFLECT,
    )

    # 5. Apply sensor perturbations (blur and noise)
    if blur_sigma > 0:
        ksize = int(np.ceil(blur_sigma * 3)) | 1  # Ensure odd kernel size
        src_warped_img = cv2.GaussianBlur(src_warped_img, (ksize, ksize), blur_sigma)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        src_f: np.ndarray = src_warped_img.astype(np.float32)
        noise = rng.normal(0, noise_std, size=src_warped_img.shape).astype(np.float32)
        src_warped_img = np.clip(src_f + noise, 0, 255).astype(np.uint8)

    # Crop central region (h_target, w_target)
    ref_crop = ref_hillshade[pad : pad + h_target, pad : pad + w_target]
    src_crop = src_warped_img[pad : pad + h_target, pad : pad + w_target]

    # Adjust transformation matrix for top-left cropping offset (-pad, -pad)
    # P_cropped = P_padded - pad
    # P_ref_crop + pad = M @ (P_src_crop + pad)
    # P_ref_crop = M @ P_src_crop + (M @ [pad, pad, 1]^T - [pad, pad, 1]^T)
    offset_vec = np.array([pad, pad, 1.0], dtype=np.float64)
    mapped_offset = M_3x3 @ offset_vec
    M_crop = np.copy(M_3x3)
    M_crop[0, 2] = mapped_offset[0] - pad
    M_crop[1, 2] = mapped_offset[1] - pad

    model_type_val: Any = transform_model if transform_model in ("similarity", "affine", "homography") else "affine"
    transform_gt = TransformEstimate.from_matrix(
        model_type=model_type_val,
        matrix=M_crop,
        inlier_count=25,
        inlier_ratio=1.0,
        rmse_px=0.0,
    )

    # 6. Generate exact ground-truth control points on a 5x5 grid
    grid_x = np.linspace(0.15 * w_target, 0.85 * w_target, 5)
    grid_y = np.linspace(0.15 * h_target, 0.85 * h_target, 5)

    control_points: list[ControlPoint] = []
    pt_counter = 1

    for sx in grid_x:
        for sy in grid_y:
            src_p = np.array([sx, sy, 1.0], dtype=np.float64)
            ref_p = M_crop @ src_p
            rx, ry = float(ref_p[0]), float(ref_p[1])

            control_points.append(
                ControlPoint(
                    source_x=float(sx),
                    source_y=float(sy),
                    reference_x=rx,
                    reference_y=ry,
                    point_id=f"synth_cp_{pt_counter:02d}",
                )
            )
            pt_counter += 1

    metadata = {
        "synthetic": True,
        "disclaimer": "Synthetic procedural benchmark dataset for algorithmic regression testing. Not mission flight performance.",
        "angle_deg": angle_deg,
        "translation": list(translation),
        "scale_ratio": scale_ratio,
        "sun_azimuth_diff_deg": sun_azimuth_diff,
        "blur_sigma": blur_sigma,
        "noise_std": noise_std,
    }

    return SyntheticPairData(
        pair_id=pair_id,
        source_image=src_crop,
        reference_image=ref_crop,
        ground_truth_transform=transform_gt,
        control_points=control_points,
        metadata=metadata,
    )


def generate_synthetic_benchmark_dataset(
    output_dir: str | Path,
    num_pairs: int = 3,
    seed: int = 42,
) -> Path:
    """Generate complete synthetic benchmark dataset with manifest, control points, and GeoTIFFs.

    Args:
        output_dir: Destination output directory.
        num_pairs: Number of synthetic image pairs to generate.
        seed: Random seed integer for reproducible generation.

    Returns:
        Path to output directory containing generated dataset.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_manifest_entries: list[dict[str, Any]] = []

    for i in range(1, num_pairs + 1):
        pair_id = f"synth_pair_{i:03d}"
        pair_dir = out_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Vary rotation, scale, and sun angle per pair
        ang = 1.5 * i
        tx = 4.0 * i
        ty = -3.0 * i
        scale = 1.0 + 0.1 * (i - 1)
        sun_diff = 15.0 * i

        pair_data = generate_synthetic_pair(
            pair_id=pair_id,
            angle_deg=ang,
            translation=(tx, ty),
            scale_ratio=scale,
            sun_azimuth_diff=sun_diff,
            seed=seed + i,
        )

        src_path = pair_dir / "source.tif"
        ref_path = pair_dir / "reference.tif"
        cp_path = pair_dir / "control_points.csv"
        gt_trans_path = pair_dir / "ground_truth_transform.json"

        # Write GeoTIFF rasters
        tifffile.imwrite(str(src_path), pair_data.source_image)
        tifffile.imwrite(str(ref_path), pair_data.reference_image)

        # Write ground_truth_transform.json
        with gt_trans_path.open("w", encoding="utf-8") as f:
            json.dump(pair_data.ground_truth_transform.to_dict(), f, indent=2)

        # Write control_points.csv
        with cp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_x", "source_y", "reference_x", "reference_y", "point_id"])
            for cp in pair_data.control_points:
                writer.writerow(
                    [
                        f"{cp.source_x:.6f}",
                        f"{cp.source_y:.6f}",
                        f"{cp.reference_x:.6f}",
                        f"{cp.reference_y:.6f}",
                        cp.point_id,
                    ]
                )

        pairs_manifest_entries.append(
            {
                "pair_id": pair_id,
                "source": str(src_path),
                "reference": str(ref_path),
                "source_sensor": "SYNTH_LUNAR",
                "reference_sensor": "SYNTH_LUNAR",
                "pixel_scale_ratio": scale,
                "sun_angle_diff_deg": sun_diff,
                "control_points_path": str(cp_path),
                "synthetic": True,
            }
        )

    # Export pairs_manifest.json
    manifest_data = {
        "disclaimer": "Synthetic procedural benchmark dataset for algorithmic regression testing. Not mission flight performance.",
        "synthetic": True,
        "pairs": pairs_manifest_entries,
    }
    manifest_path = out_dir / "pairs_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return out_dir
