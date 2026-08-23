"""Composable 3x3 coordinate mapping for pixel center transformations across crop, scale, and pyramid levels."""
from __future__ import annotations

from typing import Any, cast

import numpy as np


class CoordinateMapping:
    """Composable 3x3 homogeneous coordinate mapping between working and original pixel spaces.

    All pixel coordinates are 0-indexed float64 (x, y) where x=column and y=row.
    Transformation matrix M maps working pixel centers to target/original pixel centers:
        [x_target, y_target, 1]^T ~ M @ [x_working, y_working, 1]^T
    """

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        """Initialize coordinate mapping with 3x3 matrix (defaults to identity)."""
        if matrix is None:
            self._matrix = np.eye(3, dtype=np.float64)
        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.shape != (3, 3):
                raise ValueError(f"Matrix must have shape (3, 3), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Matrix contains non-finite values (NaN or Inf)")
            if np.abs(np.linalg.det(arr)) < 1e-12:
                raise ValueError("Matrix is singular or near-singular (determinant near zero)")
            self._matrix = arr

    @property
    def matrix(self) -> np.ndarray:
        """Return 3x3 transformation matrix as float64 array."""
        return self._matrix.copy()

    @property
    def inverse_matrix(self) -> np.ndarray:
        """Return inverse 3x3 transformation matrix."""
        return np.linalg.inv(self._matrix)

    @classmethod
    def identity(cls) -> CoordinateMapping:
        """Create identity mapping (1:1 mapping)."""
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def from_crop(cls, offset_x: float, offset_y: float) -> CoordinateMapping:
        """Create mapping for crop with top-left offset (offset_x, offset_y).

        Maps cropped coordinates (x_crop, y_crop) to original coordinates (x_orig, y_orig):
            x_orig = x_crop + offset_x
            y_orig = y_crop + offset_y
        """
        mat = np.array(
            [
                [1.0, 0.0, float(offset_x)],
                [0.0, 1.0, float(offset_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return cls(mat)

    @classmethod
    def from_scale(cls, scale_x: float, scale_y: float) -> CoordinateMapping:
        """Create mapping for uniform/non-uniform scaling (scale_x, scale_y).

        Maps scaled coordinates (x_scaled, y_scaled) to original coordinates (x_orig, y_orig):
            x_orig = x_scaled * scale_x
            y_orig = y_scaled * scale_y
        """
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError(f"Scale factors must be positive, got scale_x={scale_x}, scale_y={scale_y}")
        mat = np.array(
            [
                [float(scale_x), 0.0, 0.0],
                [0.0, float(scale_y), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return cls(mat)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply forward mapping to Nx2 float64 point coordinates (x, y).

        Returns:
            Nx2 float64 array of mapped coordinates.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Points array must have shape (N, 2), got {pts.shape}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("Points contain non-finite values (NaN or Inf)")

        homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        transformed = (self._matrix @ homo.T).T
        w = transformed[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise ValueError("Transformation resulted in homogeneous coordinate division by near-zero")
        res = transformed[:, :2] / w
        return cast(np.ndarray, res)

    def apply_inverse(self, points: np.ndarray) -> np.ndarray:
        """Apply inverse mapping to Nx2 float64 point coordinates (x, y).

        Returns:
            Nx2 float64 array of inverse-mapped coordinates.
        """
        inv_map = CoordinateMapping(self.inverse_matrix)
        return inv_map.apply(points)

    def compose(self, other: CoordinateMapping) -> CoordinateMapping:
        """Compose this mapping with another mapping.

        If self maps B -> C and other maps A -> B, then self.compose(other) maps A -> C:
            M_composed = M_self @ M_other
        """
        if not isinstance(other, CoordinateMapping):
            raise TypeError(f"Can only compose with another CoordinateMapping, got {type(other)}")
        composed_mat = self._matrix @ other._matrix
        return CoordinateMapping(composed_mat)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mapping to dictionary."""
        return {"matrix": self._matrix.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoordinateMapping:
        """Deserialize mapping from dictionary."""
        if "matrix" not in data:
            raise KeyError("Dictionary must contain 'matrix' key")
        return cls(np.array(data["matrix"], dtype=np.float64))

    def __repr__(self) -> str:
        return f"CoordinateMapping(matrix={self._matrix.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoordinateMapping):
            return False
        return bool(np.allclose(self._matrix, other._matrix, atol=1e-12))
