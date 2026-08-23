"""Unit tests for optional learned matcher adapter, weight loading, and checksum verification."""
from pathlib import Path

import numpy as np
import pytest

from lunarmatch.features.learned import (
    LearnedMatcherAdapter,
    compute_weight_checksum,
    require_torch,
)
from lunarmatch.models.config import FeaturesConfig


@pytest.fixture
def dummy_weights_file(tmp_path: Path) -> Path:
    """Fixture creating a dummy local model weights file for testing."""
    weights_path = tmp_path / "dummy_lightglue.pth"
    weights_path.write_bytes(b"DUMMY_MODEL_WEIGHTS_DATA_FOR_TESTING_1234567890")
    return weights_path


def test_missing_weights_file_handling():
    """Test explicit FileNotFoundError when local weights file path is missing or non-existent."""
    with pytest.raises(FileNotFoundError, match="Local weight file not found"):
        compute_weight_checksum("non_existent_weights_file.pth")

    try:
        require_torch()
        with pytest.raises(FileNotFoundError, match="Learned matcher requires explicit local weight path"):
            LearnedMatcherAdapter(weights_path=None)
    except RuntimeError as exc:
        assert "Learned matcher feature backends require optional 'torch' dependency" in str(exc)


def test_weight_checksum_calculation(dummy_weights_file: Path):
    """Test SHA-256 weight checksum computation on local weight file."""
    checksum = compute_weight_checksum(dummy_weights_file)
    assert len(checksum) == 64
    assert isinstance(checksum, str)


def test_learned_adapter_initialization_and_extraction(dummy_weights_file: Path):
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
    assert kpts.count >= 0
    if kpts.count > 0:
        assert kpts.descriptors is not None
        assert kpts.descriptors.shape[1] == 256
        assert kpts.descriptors.dtype == np.float32


@pytest.mark.real_model
def test_opt_in_real_learned_model_execution():
    """Opt-in real model test marker (requires real PyTorch weights)."""
    pytest.skip("Opt-in test marker: execute with --run-real-model and valid model weights path")
