"""Inexpensive terrain overlap validation and early rejection checking."""
from __future__ import annotations

import cv2
import numpy as np

from lunarmatch.features import get_feature_backend
from lunarmatch.geometry import verify_matches
from lunarmatch.matching.matcher import DescriptorMatcher
from lunarmatch.models.config import FeaturesConfig, GeometryConfig, PipelineConfig


def check_terrain_overlap(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    config: PipelineConfig,
    prior_scale: float | None = None,
) -> tuple[bool, float, int, str | None]:
    """Inexpensive downsampled terrain overlap and structural validation.

    Downsamples both processed images to a common coarse resolution matching overlap footprint
    (max dimension 128px), computes Peak Normalized Cross-Correlation (NCC) and runs a quick
    coarse SIFT/ORB match to verify terrain consensus.

    Args:
        src_img: Preprocessed 2D float32 source image.
        ref_img: Preprocessed 2D float32 reference image.
        config: PipelineConfig configuration model.
        prior_scale: Optional GSD scale ratio (source GSD / reference GSD).

    Returns:
        tuple (is_valid, peak_ncc, inlier_count, failure_reason)
    """
    s = prior_scale if prior_scale is not None else 1.0

    # 1. Scale-match the images if prior_scale is available
    if s < 1.0:
        new_w = max(4, round(src_img.shape[1] * s))
        new_h = max(4, round(src_img.shape[0] * s))
        src_rescaled = cv2.resize(src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ref_rescaled = ref_img
    else:
        src_rescaled = src_img
        new_w = max(4, round(ref_img.shape[1] / s))
        new_h = max(4, round(ref_img.shape[0] / s))
        ref_rescaled = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Downsample both to a common coarse maximum dimension (128px) for extremely fast check
    max_dim = 128

    h_s, w_s = src_rescaled.shape
    scale_s = max_dim / max(h_s, w_s)
    src_coarse = cv2.resize(
        src_rescaled,
        (max(4, int(w_s * scale_s)), max(4, int(h_s * scale_s))),
        interpolation=cv2.INTER_AREA,
    )

    h_r, w_r = ref_rescaled.shape
    scale_r = max_dim / max(h_r, w_r)
    ref_coarse = cv2.resize(
        ref_rescaled,
        (max(4, int(w_r * scale_r)), max(4, int(h_r * scale_r))),
        interpolation=cv2.INTER_AREA,
    )

    src_coarse = src_coarse.astype(np.float32)
    ref_coarse = ref_coarse.astype(np.float32)

    # Normalize to 0 mean and unit standard deviation for robust correlation
    if src_coarse.std() > 1e-6:
        src_coarse = (src_coarse - src_coarse.mean()) / src_coarse.std()
    if ref_coarse.std() > 1e-6:
        ref_coarse = (ref_coarse - ref_coarse.mean()) / ref_coarse.std()

    # 3. Peak NCC calculation using template matching of a central crop
    th, tw = src_coarse.shape
    crop_h, crop_w = max(4, th // 2), max(4, tw // 2)
    y0, x0 = (th - crop_h) // 2, (tw - crop_w) // 2
    template = src_coarse[y0 : y0 + crop_h, x0 : x0 + crop_w]

    peak_ncc = -1.0
    if ref_coarse.shape[0] >= template.shape[0] and ref_coarse.shape[1] >= template.shape[1]:
        try:
            ncc_map = cv2.matchTemplate(ref_coarse, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(ncc_map)
            peak_ncc = float(max_val)
        except cv2.error:
            pass

    # 4. Coarse SIFT/ORB keypoints matching
    inlier_count = 0
    try:
        # Convert coarse normalized images to uint8 in [0, 255] for SIFT
        src_min, src_max = src_coarse.min(), src_coarse.max()
        ref_min, ref_max = ref_coarse.min(), ref_coarse.max()
        src_uint8 = np.clip(
            (src_coarse - src_min) / max(1e-6, src_max - src_min) * 255.0, 0, 255
        ).astype(np.uint8)
        ref_uint8 = np.clip(
            (ref_coarse - ref_min) / max(1e-6, ref_max - ref_min) * 255.0, 0, 255
        ).astype(np.uint8)

        # Set up a small SIFT/ORB config
        backend_name = config.features.backend
        features_cfg = FeaturesConfig(backend=backend_name, max_keypoints=300)
        feature_backend = get_feature_backend(features_cfg)

        src_kps = feature_backend.detect_and_compute(src_uint8)
        ref_kps = feature_backend.detect_and_compute(ref_uint8)

        if len(src_kps) >= 3 and len(ref_kps) >= 3:
            matcher = DescriptorMatcher(config=config.matching)
            match_res = matcher.match(src_kps, ref_kps)
            if len(match_res.match_set) >= 3:
                # Use geometry config from pipeline with allow_reflection override
                geom_cfg = GeometryConfig(
                    model="affine",
                    reprojection_threshold_px=8.0,
                    allow_reflection=config.geometry.allow_reflection,
                )
                geom_res = verify_matches(match_res.match_set, config=geom_cfg)
                if geom_res.success and geom_res.transform is not None:
                    inlier_count = int(geom_res.transform.inlier_count)
    except Exception:  # noqa: BLE001, S110
        pass

    # 5. Combined Rejection Logic
    qg = config.quality_gates
    if qg.verify_terrain_overlap:
        ncc_ok = peak_ncc >= qg.min_coarse_ncc
        inliers_ok = inlier_count >= qg.min_coarse_inliers

        if not ncc_ok and not inliers_ok:
            return False, peak_ncc, inlier_count, "different_terrain"

    return True, peak_ncc, inlier_count, None
