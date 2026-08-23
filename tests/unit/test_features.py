"""Unit tests for feature extraction backends (ORB and SIFT)."""
import cv2
import numpy as np
import pytest

from lunarmatch.features import (
    ORBFeatureBackend,
    SIFTFeatureBackend,
    get_feature_backend,
    list_feature_backends,
)
from lunarmatch.geometry import CoordinateMapping
from lunarmatch.models.config import FeaturesConfig
from lunarmatch.models.domain import KeypointSet


def create_synthetic_pattern(width: int = 512, height: int = 512) -> np.ndarray:
    """Create synthetic checkerboard pattern with sharp corners."""
    img = np.zeros((height, width), dtype=np.uint8)
    cell = 64
    for r in range(0, height, cell):
        for c in range(0, width, cell):
            if ((r // cell) + (c // cell)) % 2 == 0:
                img[r : r + cell, c : c + cell] = 255
    return img


def test_registry_and_factory():
    """Test feature backend registry and factory lookup."""
    backends = list_feature_backends()
    assert "orb" in backends
    assert "sift" in backends

    orb_backend = get_feature_backend("orb")
    assert orb_backend.name == "orb"

    cfg = FeaturesConfig(backend="orb", max_keypoints=500)
    orb_from_cfg = get_feature_backend(cfg)
    assert orb_from_cfg.name == "orb"

    with pytest.raises(ValueError, match="Unknown feature backend"):
        get_feature_backend("nonexistent")


def test_orb_feature_backend():
    """Test ORB feature backend on synthetic pattern and blank image."""
    backend = ORBFeatureBackend(FeaturesConfig(max_keypoints=100))
    assert backend.name == "orb"

    pattern = create_synthetic_pattern(256, 256)
    kps = backend.detect_and_compute(pattern)
    assert isinstance(kps, KeypointSet)
    assert len(kps) > 0
    assert kps.descriptors is not None
    assert kps.descriptors.dtype == np.uint8
    assert kps.descriptors.shape[1] == 32

    # Test coordinate mapping
    mapping = CoordinateMapping.from_crop(100.0, 50.0)
    kps_mapped = backend.detect_and_compute(pattern, keypoint_mapping=mapping)
    assert len(kps_mapped) == len(kps)
    np.testing.assert_allclose(kps_mapped.coordinates[:, 0], kps.coordinates[:, 0] + 100.0)
    np.testing.assert_allclose(kps_mapped.coordinates[:, 1], kps.coordinates[:, 1] + 50.0)

    # Test blank image (no keypoints)
    blank = np.zeros((256, 256), dtype=np.uint8)
    kps_blank = backend.detect_and_compute(blank)
    assert len(kps_blank) == 0
    assert kps_blank.coordinates.shape == (0, 2)


def test_sift_feature_backend():
    """Test SIFT feature backend or actionable failure if SIFT missing."""
    if not hasattr(cv2, "SIFT_create"):
        with pytest.raises(RuntimeError, match="SIFT backend is unavailable"):
            SIFTFeatureBackend()
        return

    backend = SIFTFeatureBackend(FeaturesConfig(max_keypoints=100))
    assert backend.name == "sift"

    pattern = create_synthetic_pattern(256, 256)
    kps = backend.detect_and_compute(pattern)
    assert isinstance(kps, KeypointSet)
    assert len(kps) > 0
    assert kps.descriptors is not None
    assert kps.descriptors.dtype == np.float32
    assert kps.descriptors.shape[1] == 128

    # Test blank image
    blank = np.zeros((256, 256), dtype=np.uint8)
    kps_blank = backend.detect_and_compute(blank)
    assert len(kps_blank) == 0
    assert kps_blank.coordinates.shape == (0, 2)


def test_tiled_feature_extraction():
    """Test feature extraction with tiling on image larger than tile_size."""
    cfg = FeaturesConfig(backend="orb", max_keypoints=200, tile_size=128, tile_overlap=32)
    backend = ORBFeatureBackend(cfg)

    pattern = create_synthetic_pattern(256, 256)
    kps = backend.detect_and_compute(pattern)
    assert isinstance(kps, KeypointSet)
    assert len(kps) > 0
    # Coordinates must be within full image bounds
    assert np.all((kps.coordinates[:, 0] >= 0) & (kps.coordinates[:, 0] <= 256))
    assert np.all((kps.coordinates[:, 1] >= 0) & (kps.coordinates[:, 1] <= 256))
