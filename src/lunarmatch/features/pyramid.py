"""Image pyramid construction with exact coordinate mappings to original pixel centers."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lunarmatch.geometry import CoordinateMapping
from lunarmatch.models.config import PyramidConfig


@dataclass
class PyramidLevel:
    """Single level of an image pyramid with coordinate mapping to original pixel space."""

    level_idx: int
    scale_factor: float  # Scale factor relative to original image (1.0 = full res)
    image: np.ndarray  # 2D float32 array
    valid_mask: np.ndarray  # 2D bool array
    mapping: CoordinateMapping  # Maps level (x, y) to original (x, y)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class ImagePyramid:
    """Image pyramid containing multi-resolution levels and coordinate mappings."""

    def __init__(self, levels: list[PyramidLevel]) -> None:
        if not levels:
            raise ValueError("ImagePyramid must contain at least one level")
        self.levels = levels

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, idx: int) -> PyramidLevel:
        return self.levels[idx]

    @classmethod
    def build(
        cls,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        config: PyramidConfig | None = None,
        min_dimension: int = 16,
    ) -> ImagePyramid:
        """Build multi-scale image pyramid from input image and valid mask.

        Args:
            image: Input 2D numeric numpy array.
            mask: Optional 2D boolean valid mask.
            config: Optional PyramidConfig instance.
            min_dimension: Minimum allowed dimension (width or height) for pyramid levels.

        Returns:
            ImagePyramid instance containing PyramidLevel items.
        """
        cfg = config if config is not None else PyramidConfig()
        img = np.asarray(image, dtype=np.float32)
        h_orig, w_orig = img.shape[:2]

        if mask is None:
            mask_arr = np.ones((h_orig, w_orig), dtype=bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)

        levels: list[PyramidLevel] = []

        # Level 0 (Full resolution)
        map_0 = CoordinateMapping.identity()
        level_0 = PyramidLevel(
            level_idx=0,
            scale_factor=1.0,
            image=img.copy(),
            valid_mask=mask_arr.copy(),
            mapping=map_0,
        )
        levels.append(level_0)

        # Build downsampled levels
        steps_per_octave = max(1, cfg.levels_per_octave)
        max_octaves = 4  # Practical cap on downsampled pyramid depth

        for octave in range(1, max_octaves + 1):
            for step in range(1, steps_per_octave + 1):
                scale_factor = 1.0 / (2.0 ** (octave - 1 + step / steps_per_octave))
                new_w = round(w_orig * scale_factor)
                new_h = round(h_orig * scale_factor)

                if new_w < min_dimension or new_h < min_dimension:
                    break

                # Downsample image with AREA interpolation
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Downsample mask with NEAREST interpolation
                mask_uint8 = mask_arr.astype(np.uint8)
                resized_mask_uint8 = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                resized_mask = resized_mask_uint8.astype(bool)

                # Coordinate mapping mapping scaled coordinates (x_level, y_level) to original coordinates (x_orig, y_orig).
                # Correct mapping for pixel centers under cv2.resize (INTER_AREA / INTER_NEAREST):
                # x_orig = (x_level + 0.5) * scale_x - 0.5 = scale_x * x_level + 0.5 * (scale_x - 1.0)
                scale_x = float(w_orig) / float(new_w)
                scale_y = float(h_orig) / float(new_h)
                level_mapping = CoordinateMapping(np.array([
                    [scale_x, 0.0, 0.5 * (scale_x - 1.0)],
                    [0.0, scale_y, 0.5 * (scale_y - 1.0)],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64))

                level = PyramidLevel(
                    level_idx=len(levels),
                    scale_factor=float(scale_factor),
                    image=resized_img.astype(np.float32),
                    valid_mask=resized_mask,
                    mapping=level_mapping,
                )
                levels.append(level)

        return cls(levels)
