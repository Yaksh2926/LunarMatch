"""Grid partitioning utilities for spatial match selection and coverage analysis."""
from __future__ import annotations

import numpy as np


class GridPartition:
    """2D spatial grid partition over a bounding box."""

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        grid_rows: int = 8,
        grid_cols: int = 8,
    ) -> None:
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError(f"Grid rows and cols must be positive, got ({grid_rows}, {grid_cols})")

        if max_x <= min_x or max_y <= min_y:
            raise ValueError(
                f"Invalid bounding box: min=({min_x}, {min_y}), max=({max_x}, {max_y})"
            )

        self._min_x = float(min_x)
        self._min_y = float(min_y)
        self._max_x = float(max_x)
        self._max_y = float(max_y)
        self._grid_rows = int(grid_rows)
        self._grid_cols = int(grid_cols)

        self._width = self._max_x - self._min_x
        self._height = self._max_y - self._min_y
        self._cell_width = self._width / self._grid_cols
        self._cell_height = self._height / self._grid_rows

    @property
    def min_x(self) -> float:
        return self._min_x

    @property
    def min_y(self) -> float:
        return self._min_y

    @property
    def max_x(self) -> float:
        return self._max_x

    @property
    def max_y(self) -> float:
        return self._max_y

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def grid_rows(self) -> int:
        return self._grid_rows

    @property
    def grid_cols(self) -> int:
        return self._grid_cols

    @property
    def total_cells(self) -> int:
        return self._grid_rows * self._grid_cols

    @property
    def cell_width(self) -> float:
        return self._cell_width

    @property
    def cell_height(self) -> float:
        return self._cell_height

    def get_cell_row_col(self, x: float, y: float) -> tuple[int, int]:
        """Get (row, col) cell coordinates for a point (x, y)."""
        col = int((x - self._min_x) / self._cell_width)
        row = int((y - self._min_y) / self._cell_height)
        col = max(0, min(col, self._grid_cols - 1))
        row = max(0, min(row, self._grid_rows - 1))
        return row, col

    def get_cell_id(self, x: float, y: float) -> int:
        """Get 1D cell index in range [0, total_cells - 1] for a point (x, y)."""
        r, c = self.get_cell_row_col(x, y)
        return r * self._grid_cols + c

    def get_cell_ids(self, points: np.ndarray) -> np.ndarray:
        """Vectorized cell ID calculation for Nx2 point coordinates (x, y)."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return np.empty((0,), dtype=np.int64)

        x = pts[:, 0]
        y = pts[:, 1]

        cols = np.floor((x - self._min_x) / self._cell_width).astype(np.int64)
        rows = np.floor((y - self._min_y) / self._cell_height).astype(np.int64)

        cols = np.clip(cols, 0, self._grid_cols - 1)
        rows = np.clip(rows, 0, self._grid_rows - 1)

        cell_ids = rows * self._grid_cols + cols
        return np.asarray(cell_ids, dtype=np.int64)

    def compute_valid_cells(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Identify which grid cells overlap valid pixel regions of a mask.

        Args:
            mask: Optional 2D boolean array of shape (height, width).

        Returns:
            Boolean array of length `total_cells` indicating valid cells.
        """
        if mask is None:
            return np.ones(self.total_cells, dtype=bool)

        mask_arr = np.asarray(mask, dtype=bool)
        h, w = mask_arr.shape

        valid_cells = np.zeros(self.total_cells, dtype=bool)

        for r in range(self._grid_rows):
            r_start = int(r * self._cell_height)
            r_end = int(min(h, np.ceil((r + 1) * self._cell_height)))

            for c in range(self._grid_cols):
                c_start = int(c * self._cell_width)
                c_end = int(min(w, np.ceil((c + 1) * self._cell_width)))

                if r_start < r_end and c_start < c_end:
                    sub_mask = mask_arr[r_start:r_end, c_start:c_end]
                    if np.any(sub_mask):
                        cell_id = r * self._grid_cols + c
                        valid_cells[cell_id] = True

        return valid_cells
