"""Downsampled quick-look image generators for visual diagnostics."""
from __future__ import annotations

import base64
import io

import cv2
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from lunarmatch.models.domain import MatchSet, TransformEstimate


def downsample_image(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Downsample image so max(height, width) <= max_dim to limit memory and SVG/PNG size.

    Args:
        image: 2D or 3D numeric image array.
        max_dim: Maximum allowed height or width in pixels.

    Returns:
        Downsampled image array.
    """
    img = np.asarray(image)
    if img.ndim < 2:
        return img

    h, w = img.shape[:2]
    if max(h, w) <= max_dim or max(h, w) == 0:
        return img

    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    res: np.ndarray = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return res


def _fig_to_base64_png(fig: matplotlib.figure.Figure) -> str:
    """Convert Matplotlib figure to base64 PNG data URL string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize numeric array to uint8 range [0, 255]."""
    a = np.asarray(arr, dtype=np.float32)
    min_val, max_val = np.min(a), np.max(a)
    if max_val > min_val:
        norm = (a - min_val) / (max_val - min_val) * 255.0
    else:
        norm = np.zeros_like(a)
    res: np.ndarray = norm.astype(np.uint8)
    return res


def generate_overlay_quicklook(
    warped_source: np.ndarray,
    reference: np.ndarray,
    max_dim: int = 800,
) -> str:
    """Generate color overlay quick look (Red = Warped Source, Green/Blue = Reference)."""
    src_ds = _normalize_uint8(downsample_image(warped_source, max_dim=max_dim))
    ref_ds = _normalize_uint8(downsample_image(reference, max_dim=max_dim))

    h = min(src_ds.shape[0], ref_ds.shape[0])
    w = min(src_ds.shape[1], ref_ds.shape[1])

    s_crop = src_ds[:h, :w]
    r_crop = ref_ds[:h, :w]

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = s_crop  # Red channel = Warped Source
    rgb[:, :, 1] = r_crop  # Green channel = Reference
    rgb[:, :, 2] = r_crop  # Blue channel = Reference

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title("Overlay (Red=Source, Cyan=Ref)", fontsize=10)
    ax.axis("off")

    return _fig_to_base64_png(fig)


def generate_checkerboard_quicklook(
    warped_source: np.ndarray,
    reference: np.ndarray,
    grid_size: int = 8,
    max_dim: int = 800,
) -> str:
    """Generate checkerboard quick look alternating tiles of warped source and reference."""
    src_ds = _normalize_uint8(downsample_image(warped_source, max_dim=max_dim))
    ref_ds = _normalize_uint8(downsample_image(reference, max_dim=max_dim))

    h = min(src_ds.shape[0], ref_ds.shape[0])
    w = min(src_ds.shape[1], ref_ds.shape[1])

    s_crop = src_ds[:h, :w]
    r_crop = ref_ds[:h, :w]

    board = np.copy(r_crop)
    tile_h = max(1, h // grid_size)
    tile_w = max(1, w // grid_size)

    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                y0, y1 = i * tile_h, min((i + 1) * tile_h, h)
                x0, x1 = j * tile_w, min((j + 1) * tile_w, w)
                board[y0:y1, x0:x1] = s_crop[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(board, cmap="gray")
    ax.set_title("Checkerboard Alignment", fontsize=10)
    ax.axis("off")

    return _fig_to_base64_png(fig)


def generate_match_lines_quicklook(
    source: np.ndarray,
    reference: np.ndarray,
    matches: MatchSet,
    max_matches: int = 80,
    max_dim: int = 800,
) -> str:
    """Generate side-by-side feature match lines plot (Green = Inliers, Red = Outliers)."""
    src_ds = _normalize_uint8(downsample_image(source, max_dim=max_dim))
    ref_ds = _normalize_uint8(downsample_image(reference, max_dim=max_dim))

    h_src, w_src = src_ds.shape[:2]
    h_ref, w_ref = ref_ds.shape[:2]

    # Compute scale factors for keypoints
    scale_src_x = w_src / max(1, source.shape[1])
    scale_src_y = h_src / max(1, source.shape[0])
    scale_ref_x = w_ref / max(1, reference.shape[1])
    scale_ref_y = h_ref / max(1, reference.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(src_ds, cmap="gray")
    ax1.set_title("Source Image", fontsize=10)
    ax1.axis("off")

    ax2.imshow(ref_ds, cmap="gray")
    ax2.set_title("Reference Image", fontsize=10)
    ax2.axis("off")

    if len(matches) > 0:
        n_show = min(len(matches), max_matches)
        indices = np.linspace(0, len(matches) - 1, n_show, dtype=int)

        for idx in indices:
            sx, sy = matches.source_points[idx] * [scale_src_x, scale_src_y]
            rx, ry = matches.reference_points[idx] * [scale_ref_x, scale_ref_y]
            is_inlier = matches.inliers[idx]

            color = "lime" if is_inlier else "red"
            alpha = 0.7 if is_inlier else 0.4

            ax1.plot(sx, sy, "o", color=color, markersize=3, alpha=alpha)
            ax2.plot(rx, ry, "o", color=color, markersize=3, alpha=alpha)

    fig.suptitle(f"Feature Correspondences (Inliers={np.sum(matches.inliers)}, Outliers={np.sum(~matches.inliers)})", fontsize=11)
    return _fig_to_base64_png(fig)


def generate_inlier_scatter_quicklook(
    reference: np.ndarray,
    matches: MatchSet,
    max_dim: int = 800,
) -> str:
    """Generate spatial scatter plot of match points on reference image."""
    ref_ds = _normalize_uint8(downsample_image(reference, max_dim=max_dim))
    h_ref, w_ref = ref_ds.shape[:2]

    scale_x = w_ref / max(1, reference.shape[1])
    scale_y = h_ref / max(1, reference.shape[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ref_ds, cmap="gray")

    if len(matches) > 0:
        ref_pts = matches.reference_points * [scale_x, scale_y]
        inliers = matches.inliers

        if np.any(~inliers):
            ax.scatter(ref_pts[~inliers, 0], ref_pts[~inliers, 1], c="red", s=15, marker="x", label="Outlier", alpha=0.6)
        if np.any(inliers):
            ax.scatter(ref_pts[inliers, 0], ref_pts[inliers, 1], c="lime", s=20, marker="o", label="Inlier", alpha=0.8)

        ax.legend(loc="upper right", fontsize=8)

    ax.set_title("Spatial Inlier/Outlier Distribution", fontsize=10)
    ax.axis("off")

    return _fig_to_base64_png(fig)


def generate_residual_vector_quicklook(
    reference: np.ndarray,
    matches: MatchSet,
    transform: TransformEstimate | None = None,
    max_dim: int = 800,
) -> str:
    """Generate quiver plot of spatial displacement residual vectors on reference image."""
    ref_ds = _normalize_uint8(downsample_image(reference, max_dim=max_dim))
    h_ref, w_ref = ref_ds.shape[:2]

    scale_x = w_ref / max(1, reference.shape[1])
    scale_y = h_ref / max(1, reference.shape[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ref_ds, cmap="gray")

    if transform is not None and len(matches) > 0 and np.any(matches.inliers):
        M = transform.get_matrix_array()
        inlier_idx = np.where(matches.inliers)[0]

        src_pts = matches.source_points[inlier_idx]
        ref_pts = matches.reference_points[inlier_idx]

        # Predict reference coordinates
        ones = np.ones((len(src_pts), 1), dtype=np.float64)
        src_homo = np.hstack([src_pts, ones])
        pred_homo = (M @ src_homo.T).T
        w = pred_homo[:, 2:3]
        w[w == 0] = 1e-12
        pred_pts = pred_homo[:, :2] / w

        # Displacement vectors (dx, dy)
        dx = (pred_pts[:, 0] - ref_pts[:, 0]) * scale_x
        dy = (pred_pts[:, 1] - ref_pts[:, 1]) * scale_y

        rx = ref_pts[:, 0] * scale_x
        ry = ref_pts[:, 1] * scale_y

        ax.quiver(rx, ry, dx, dy, color="yellow", angles="xy", scale_units="xy", scale=0.1, width=0.005)

    ax.set_title("Residual Error Vector Field (Quiver)", fontsize=10)
    ax.axis("off")

    return _fig_to_base64_png(fig)


def generate_occupancy_heatmap_quicklook(
    reference_shape: tuple[int, int],
    matches: MatchSet,
    grid_shape: tuple[int, int] = (10, 10),
) -> str:
    """Generate spatial cell occupancy heatmap showing match count distribution."""
    h_ref, w_ref = reference_shape
    gh, gw = grid_shape
    grid = np.zeros((gh, gw), dtype=np.int32)

    if len(matches) > 0 and h_ref > 0 and w_ref > 0:
        ref_pts = matches.reference_points
        cell_y = np.clip((ref_pts[:, 1] / h_ref * gh).astype(int), 0, gh - 1)
        cell_x = np.clip((ref_pts[:, 0] / w_ref * gw).astype(int), 0, gw - 1)

        for cy, cx in zip(cell_y, cell_x):
            grid[cy, cx] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Match Count")
    ax.set_title(f"Grid Occupancy Heatmap ({gh}x{gw})", fontsize=10)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    return _fig_to_base64_png(fig)
