"""Self-contained HTML report generator with offline CSS and HTML escaping."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from lunarmatch.models.domain import MatchSet, RunManifest
from lunarmatch.visualization.plots import (
    generate_checkerboard_quicklook,
    generate_inlier_scatter_quicklook,
    generate_match_lines_quicklook,
    generate_occupancy_heatmap_quicklook,
    generate_overlay_quicklook,
    generate_residual_vector_quicklook,
)


def generate_html_report(
    manifest: RunManifest,
    matches: MatchSet | None = None,
    source_image: np.ndarray | None = None,
    reference_image: np.ndarray | None = None,
    warped_image: np.ndarray | None = None,
    output_path: Path | str | None = None,
    stage_timings: dict[str, float] | None = None,
) -> str:
    """Generate self-contained HTML registration report with embedded base64 plots.

    Args:
        manifest: RunManifest from RegistrationOrchestrator.
        matches: Optional MatchSet of correspondences.
        source_image: Optional 2D source raster image array.
        reference_image: Optional 2D reference raster image array.
        warped_image: Optional 2D warped source raster image array.
        output_path: Optional destination file or directory path.
        stage_timings: Optional dict of per-stage execution durations in seconds.

    Returns:
        Generated HTML document content as string.
    """
    pair_id_esc = html.escape(manifest.pair_id)
    src_path_esc = html.escape(manifest.source_path)
    ref_path_esc = html.escape(manifest.reference_path)
    config_hash_esc = html.escape(manifest.config_hash[:12] if manifest.config_hash else "N/A")
    version_esc = html.escape(manifest.package_version)
    timestamp_esc = html.escape(manifest.timestamp_utc)

    status_badge = (
        '<span class="badge badge-pass">PASSED</span>'
        if manifest.quality_gate_passed
        else '<span class="badge badge-fail">FAILED</span>'
    )

    metrics = manifest.metrics
    rmse_str = f"{metrics.rmse_px:.3f} px" if metrics.rmse_px is not None else "N/A"
    inlier_str = f"{metrics.inlier_count} / {metrics.total_matches}"
    ratio_str = f"{metrics.inlier_ratio * 100.0:.1f}%" if metrics.inlier_ratio is not None else "N/A"
    occupancy_str = f"{metrics.occupied_grid_fraction * 100.0:.1f}%" if metrics.occupied_grid_fraction is not None else "N/A"
    runtime_str = f"{metrics.runtime_seconds:.2f} s"

    # Generate quick look images if inputs are provided
    quicklook_images: list[tuple[str, str]] = []

    if warped_image is not None and reference_image is not None:
        try:
            overlay_b64 = generate_overlay_quicklook(warped_image, reference_image)
            quicklook_images.append(("Color Alignment Overlay", overlay_b64))
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            chk_b64 = generate_checkerboard_quicklook(warped_image, reference_image)
            quicklook_images.append(("Checkerboard Continuity", chk_b64))
        except Exception:  # noqa: BLE001, S110
            pass

    if source_image is not None and reference_image is not None and matches is not None:
        try:
            lines_b64 = generate_match_lines_quicklook(source_image, reference_image, matches)
            quicklook_images.append(("Feature Correspondences", lines_b64))
        except Exception:  # noqa: BLE001, S110
            pass

    if reference_image is not None and matches is not None:
        try:
            scatter_b64 = generate_inlier_scatter_quicklook(reference_image, matches)
            quicklook_images.append(("Spatial Match Distribution", scatter_b64))
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            vector_b64 = generate_residual_vector_quicklook(reference_image, matches, manifest.transform)
            quicklook_images.append(("Residual Error Vector Field", vector_b64))
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            ref_shape = (reference_image.shape[0], reference_image.shape[1])
            heat_b64 = generate_occupancy_heatmap_quicklook(ref_shape, matches)
            quicklook_images.append(("Spatial Occupancy Heatmap", heat_b64))
        except Exception:  # noqa: BLE001, S110
            pass

    # Build warning and failure logs HTML
    warnings_html = ""
    if metrics.warnings:
        warn_items = "".join(f"<li>{html.escape(w)}</li>" for w in metrics.warnings)
        warnings_html = f"""
        <div class="card warning-card">
            <h3>Diagnostic Warnings & Failures</h3>
            <ul>{warn_items}</ul>
        </div>
        """

    # Build stage timings HTML
    timings_html = ""
    if stage_timings:
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v:.4f} s</td></tr>" for k, v in stage_timings.items()
        )
        timings_html = f"""
        <h3>Stage Execution Timings</h3>
        <table class="data-table">
            <thead><tr><th>Pipeline Stage</th><th>Duration</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    # Build quick look gallery HTML
    gallery_html = ""
    if quicklook_images:
        items = "".join(
            f"""
            <div class="gallery-item">
                <h4>{html.escape(title)}</h4>
                <img src="{img_b64}" alt="{html.escape(title)}" />
            </div>
            """
            for title, img_b64 in quicklook_images
        )
        gallery_html = f"""
        <div class="section">
            <h2>Visual Diagnostics & Quick Looks</h2>
            <div class="gallery-grid">{items}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LunarMatch Registration Report — {pair_id_esc}</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --pass-color: #22c55e;
            --fail-color: #ef4444;
            --border-color: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{ margin: 0; font-size: 24px; color: var(--accent); }}
        .badge {{
            font-weight: bold;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .badge-pass {{ background-color: rgba(34, 197, 94, 0.2); color: var(--pass-color); border: 1px solid var(--pass-color); }}
        .badge-fail {{ background-color: rgba(239, 68, 68, 0.2); color: var(--fail-color); border: 1px solid var(--fail-color); }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-title {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
        .card-value {{ font-size: 24px; font-weight: bold; color: var(--text-main); }}
        
        .warning-card {{
            border-left: 4px solid var(--fail-color);
            margin-bottom: 24px;
        }}
        .warning-card h3 {{ margin-top: 0; color: var(--fail-color); }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
            background-color: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }}
        .data-table th, .data-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        .data-table th {{ background-color: rgba(255, 255, 255, 0.05); color: var(--accent); }}

        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
        }}
        .gallery-item {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .gallery-item h4 {{ margin: 0 0 8px 0; color: var(--accent); font-size: 14px; }}
        .gallery-item img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto; }}

        .footer {{
            margin-top: 40px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>LunarMatch Registration Report</h1>
                <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">Pair: {pair_id_esc}</div>
            </div>
            <div>{status_badge}</div>
        </div>

        <div class="metrics-grid">
            <div class="card">
                <div class="card-title">Reprojection RMSE</div>
                <div class="card-value">{rmse_str}</div>
            </div>
            <div class="card">
                <div class="card-title">Inliers / Total</div>
                <div class="card-value">{inlier_str}</div>
            </div>
            <div class="card">
                <div class="card-title">Inlier Ratio</div>
                <div class="card-value">{ratio_str}</div>
            </div>
            <div class="card">
                <div class="card-title">Cell Occupancy</div>
                <div class="card-value">{occupancy_str}</div>
            </div>
            <div class="card">
                <div class="card-title">Total Runtime</div>
                <div class="card-value">{runtime_str}</div>
            </div>
        </div>

        {warnings_html}

        <div class="section">
            <h2>Run Provenance & Metadata</h2>
            <table class="data-table">
                <tbody>
                    <tr><td>Source Path</td><td>{src_path_esc}</td></tr>
                    <tr><td>Reference Path</td><td>{ref_path_esc}</td></tr>
                    <tr><td>Config Hash</td><td>{config_hash_esc}</td></tr>
                    <tr><td>Package Version</td><td>{version_esc}</td></tr>
                    <tr><td>Timestamp (UTC)</td><td>{timestamp_esc}</td></tr>
                    <tr><td>Random Seed</td><td>{manifest.seed}</td></tr>
                </tbody>
            </table>
        </div>

        {timings_html}

        {gallery_html}

        <div class="footer">
            <div>LunarMatch v{version_esc}</div>
            <div>Generated: {timestamp_esc}</div>
        </div>
    </div>
</body>
</html>
"""

    if output_path is not None:
        out_p = Path(output_path).resolve()
        if out_p.is_dir():
            out_p = out_p / "report.html"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
