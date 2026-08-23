"""Visualization module for downsampled quick looks and self-contained HTML report generation."""
from lunarmatch.visualization.plots import (
    downsample_image,
    generate_checkerboard_quicklook,
    generate_inlier_scatter_quicklook,
    generate_match_lines_quicklook,
    generate_occupancy_heatmap_quicklook,
    generate_overlay_quicklook,
    generate_residual_vector_quicklook,
)
from lunarmatch.visualization.report import generate_html_report

__all__ = [
    "downsample_image",
    "generate_checkerboard_quicklook",
    "generate_html_report",
    "generate_inlier_scatter_quicklook",
    "generate_match_lines_quicklook",
    "generate_occupancy_heatmap_quicklook",
    "generate_overlay_quicklook",
    "generate_residual_vector_quicklook",
]
