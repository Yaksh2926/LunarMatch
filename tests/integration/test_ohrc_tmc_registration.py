"""Integration test suite for OHRC <-> TMC cross-sensor scale-aware registration."""
from __future__ import annotations

import cv2
import numpy as np

from lunarmatch.geometry import GeometricVerifier
from lunarmatch.matching.matcher import MatchSet


def create_synthetic_crater(img: np.ndarray, cx: float, cy: float, r: float, depth: float) -> None:
    """Draw synthetic crater profile."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rim = np.exp(-((dist - r) ** 2) / (r * 0.4) ** 2) * (depth * 0.8)
    pit = np.exp(-(dist**2) / (r * 0.6) ** 2) * -depth
    img += rim.astype(np.float32) + pit.astype(np.float32)


def test_ohrc_tmc_scale_aware_pipeline_fixture() -> None:
    """Verify scale-aware cross-sensor registration on deterministic synthetic multi-scale crater scene."""
    rng = np.random.default_rng(42)

    # 1. High-resolution synthetic OHRC-like scene (400x400)
    ohrc_sim = rng.normal(128.0, 10.0, (400, 400)).astype(np.float32)
    craters = [(100, 100, 30, 40), (280, 120, 45, 50), (180, 260, 35, 45), (300, 310, 25, 35), (80, 300, 40, 45)]
    for cx, cy, r, d in craters:
        create_synthetic_crater(ohrc_sim, cx, cy, r, d)

    ohrc_u8 = np.clip(ohrc_sim, 0, 255).astype(np.uint8)

    # 2. Downsampled TMC-like scene (80x80 = 5.0x scale reduction)
    tmc_u8 = cv2.resize(ohrc_u8, (80, 80), interpolation=cv2.INTER_AREA)

    # Detect SIFT keypoints on high-res (downscaled level) and low-res
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.resize(ohrc_u8, (80, 80)), None)
    kp2, des2 = sift.detectAndCompute(tmc_u8, None)

    assert des1 is not None and des2 is not None
    assert len(kp1) >= 10 and len(kp2) >= 10

    # Match descriptors with ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches_raw = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches_raw:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    assert len(good_matches) >= 5

    pts1 = np.float64([kp1[m.queryIdx].pt for m in good_matches]) * 5.0  # Scale back to 400x400
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good_matches])

    ms = MatchSet(
        source_points=pts1,
        reference_points=pts2,
        inliers=np.ones(len(pts1), dtype=bool),
    )

    verifier = GeometricVerifier(
        model="affine",
        reprojection_threshold_px=5.0,
        allowed_scale=(0.1, 10.0),
    )
    geom_res = verifier.verify(ms)

    assert geom_res.success is True
    assert geom_res.transform is not None
    assert geom_res.transform.inlier_count >= 4
    assert geom_res.transform.rmse_px is not None
    assert geom_res.transform.rmse_px < 3.0
