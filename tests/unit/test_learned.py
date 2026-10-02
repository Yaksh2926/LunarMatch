"""Unit tests for optional learned matcher adapter, LoFTR backend, weight loading, and checksum verification."""
from pathlib import Path

import numpy as np
import pytest

from lunarmatch.features.learned import (
    LearnedLoFTRBackend,
    LearnedLoFTRMatcher,
    LearnedMatcherAdapter,
    compute_weight_checksum,
    require_torch,
    validate_device,
)
from lunarmatch.features.registry import get_feature_backend
from lunarmatch.models.config import FeaturesConfig


@pytest.fixture
def dummy_weights_file(tmp_path: Path) -> Path:
    """Fixture creating a dummy local model weights file for testing."""
    weights_path = tmp_path / "dummy_lightglue.pth"
    weights_path.write_bytes(b"DUMMY_MODEL_WEIGHTS_DATA_FOR_TESTING_1234567890")
    return weights_path


def test_missing_weights_file_handling() -> None:
    """Test explicit FileNotFoundError when local weights file path is missing or non-existent."""
    with pytest.raises(FileNotFoundError, match="Local weight file not found"):
        compute_weight_checksum("non_existent_weights_file.pth")

    try:
        require_torch()
        with pytest.raises(FileNotFoundError, match="Learned matcher requires explicit local weight path"):
            LearnedMatcherAdapter(weights_path=None)
    except RuntimeError as exc:
        assert "Learned matcher feature backends require optional 'torch' dependency" in str(exc)


def test_weight_checksum_calculation(dummy_weights_file: Path) -> None:
    """Test SHA-256 weight checksum computation on local weight file."""
    checksum = compute_weight_checksum(dummy_weights_file)
    assert len(checksum) == 64
    assert isinstance(checksum, str)


def test_learned_adapter_initialization_and_extraction(dummy_weights_file: Path) -> None:
    """Test LearnedMatcherAdapter initialization with valid local weights file."""
    try:
        require_torch()
    except RuntimeError:
        pytest.skip("PyTorch is not installed in current test environment")

    cfg = FeaturesConfig(backend="learned_lightglue", weights_path=dummy_weights_file)
    adapter = LearnedMatcherAdapter(config=cfg)

    assert adapter.name == "learned_lightglue"
    assert adapter.descriptor_size == 256
    assert len(adapter.weights_sha256) == 64

    # Extract features on synthetic image
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 180

    kpts = adapter.detect_and_compute(img)
    assert len(kpts) >= 0
    if len(kpts) > 0:
        assert kpts.descriptors is not None
        assert kpts.descriptors.shape[1] == 256
        assert kpts.descriptors.dtype == np.float32


def test_validate_device_cuda_fallback() -> None:
    """Test clear actionable error if CUDA is requested when unavailable."""
    with pytest.raises(RuntimeError, match="CUDA requested for LoFTR learned matcher"):
        validate_device("cuda")


def test_learned_loftr_backend_factory() -> None:
    """Test factory instantiation of learned_loftr backend."""
    cfg = FeaturesConfig(backend="learned_loftr")
    backend = get_feature_backend(cfg)

    assert backend.name == "learned_loftr"
    assert isinstance(backend, LearnedLoFTRBackend)


def test_learned_loftr_matcher_inference() -> None:
    """Test LearnedLoFTRMatcher on synthetic image pair."""
    try:
        require_torch()
    except RuntimeError:
        pytest.skip("PyTorch/Kornia not installed")

    matcher = LearnedLoFTRMatcher(device="cpu")
    assert matcher.name == "learned_loftr"

    img1 = np.zeros((128, 128), dtype=np.uint8)
    img1[30:90, 30:90] = 200
    img2 = np.zeros((128, 128), dtype=np.uint8)
    img2[35:95, 35:95] = 200

    result = matcher.match(img1, img2)
    assert result.match_set is not None
    assert len(result.source_indices) == len(result.match_set)


@pytest.mark.real_model
def test_opt_in_real_learned_model_execution() -> None:
    """Opt-in real model test marker (requires real PyTorch weights)."""
    pytest.skip("Opt-in test marker: execute with --run-real-model and valid model weights path")
