"""Vectorized kNN distance computations for binary and floating-point descriptors."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

# Precomputed lookup table for number of set bits in a uint8 byte (0..255)
_BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.int32)


def compute_descriptor_distances(
    desc1: np.ndarray,
    desc2: np.ndarray,
    metric: str | None = None,
) -> np.ndarray:
    """Compute pairwise distance matrix between desc1 (N1, D) and desc2 (N2, D).

    Args:
        desc1: Array of shape (N1, D).
        desc2: Array of shape (N2, D).
        metric: Optional metric name ('hamming', 'l2', 'euclidean').
            If None, auto-inferred from desc1.dtype ('uint8' -> 'hamming', float -> 'l2').

    Returns:
        Array of shape (N1, N2) containing pairwise distances.
    """
    n1, d1 = desc1.shape
    n2, d2 = desc2.shape

    if d1 != d2:
        raise ValueError(f"Descriptor dimensions must match, got {d1} and {d2}")

    if n1 == 0 or n2 == 0:
        return np.empty((n1, n2), dtype=np.float64)

    if metric is None:
        if desc1.dtype == np.uint8:
            metric = "hamming"
        else:
            metric = "l2"

    metric_clean = metric.lower().strip()

    if metric_clean in ("hamming", "binary"):
        if desc1.dtype != np.uint8 or desc2.dtype != np.uint8:
            # Cast to uint8 if integer or boolean
            d1_u8 = desc1.astype(np.uint8)
            d2_u8 = desc2.astype(np.uint8)
        else:
            d1_u8 = desc1
            d2_u8 = desc2

        # Fast bitwise XOR with precomputed popcount lookup table
        # desc1[:, None, :] ^ desc2[None, :, :] gives (N1, N2, D)
        xor_res = np.bitwise_xor(d1_u8[:, None, :], d2_u8[None, :, :])
        distances = _BIT_COUNTS[xor_res].sum(axis=-1).astype(np.float64)
        return distances

    elif metric_clean in ("l2", "euclidean"):
        d1_flt = desc1.astype(np.float64)
        d2_flt = desc2.astype(np.float64)
        distances = cdist(d1_flt, d2_flt, metric="euclidean")
        return np.asarray(distances, dtype=np.float64)

    else:
        raise ValueError(f"Unsupported distance metric '{metric}'. Supported metrics: 'hamming', 'l2'")
