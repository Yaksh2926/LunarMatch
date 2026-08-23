#!/usr/bin/env python3
"""LunarMatch 5-minute offline demonstration script.

Validates input rasters and executes configuration-driven pair registration.
If source/reference paths are not supplied, generates a synthetic lunar pair
in a temporary directory as an offline fallback demonstration.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import tifffile
from rich.console import Console

from lunarmatch import register_pair
from lunarmatch.synthetic import generate_synthetic_pair

console = Console()
err_console = Console(stderr=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LunarMatch offline pair registration demonstration."
    )
    parser.add_argument("--source", "-s", type=str, default=None, help="Path to source raster TIFF.")
    parser.add_argument("--reference", "-r", type=str, default=None, help="Path to reference raster TIFF.")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/demo_run", help="Output directory path.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]=== LunarMatch Registration Demo ===[/bold cyan]")

    if args.source and args.reference:
        src_path = Path(args.source).resolve()
        ref_path = Path(args.reference).resolve()
        if not src_path.is_file():
            err_console.print(f"[bold red]ERROR:[/bold red] Source image file not found: {src_path}")
            sys.exit(1)
        if not ref_path.is_file():
            err_console.print(f"[bold red]ERROR:[/bold red] Reference image file not found: {ref_path}")
            sys.exit(1)
        console.print(f"  Source: [cyan]{src_path.name}[/cyan]")
        console.print(f"  Reference: [cyan]{ref_path.name}[/cyan]")
    else:
        console.print("[yellow]Notice:[/yellow] Source/reference paths omitted. Using synthetic lunar image fallback demo.")
        temp_dir = Path(tempfile.mkdtemp(prefix="lunarmatch_demo_"))
        synth_data = generate_synthetic_pair(
            pair_id="demo_synth_pair",
            angle_deg=2.5,
            translation=(6.0, -4.0),
            sun_azimuth_diff=20.0,
            seed=args.seed,
        )
        src_path = temp_dir / "demo_source.tif"
        ref_path = temp_dir / "demo_reference.tif"

        tifffile.imwrite(str(src_path), synth_data.source_image)
        tifffile.imwrite(str(ref_path), synth_data.reference_image)

        console.print(f"  Generated Synthetic Source: [cyan]{src_path}[/cyan]")
        console.print(f"  Generated Synthetic Reference: [cyan]{ref_path}[/cyan]")

    try:
        manifest = register_pair(
            source=src_path,
            reference=ref_path,
            config=args.config,
            output_dir=out_dir,
            seed=args.seed,
        )

        status_style = "bold green" if manifest.quality_gate_passed else "bold red"
        status_str = "PASS" if manifest.quality_gate_passed else "FAIL"

        console.print(f"\n[{status_style}]Registration Status: {status_str}[/{status_style}]")
        console.print(f"  Pair ID: [bold]{manifest.pair_id}[/bold]")
        console.print(f"  Inlier Matches: [bold]{manifest.metrics.inlier_count}[/bold] / {manifest.metrics.total_matches} ({manifest.metrics.inlier_ratio:.1%})")
        if manifest.metrics.rmse_px is not None:
            console.print(f"  Reprojection RMSE: [bold]{manifest.metrics.rmse_px:.3f}[/bold] px")
        console.print(f"  Runtime: [bold]{manifest.metrics.runtime_seconds:.3f}[/bold] seconds")
        console.print(f"  Output Directory: [cyan]{out_dir}[/cyan]")

        console.print("\n[bold]Exported Artifacts:[/bold]")
        console.print(f"  - Matches CSV: [cyan]{out_dir / 'matches.csv'}[/cyan]")
        console.print(f"  - Transform JSON: [cyan]{out_dir / 'transform.json'}[/cyan]")
        console.print(f"  - Metrics JSON: [cyan]{out_dir / 'metrics.json'}[/cyan]")
        console.print(f"  - Manifest JSON: [cyan]{out_dir / 'run_manifest.json'}[/cyan]")
        console.print(f"  - Registered GeoTIFF: [cyan]{out_dir / 'registered.tif'}[/cyan]")
        console.print(f"  - Offline Report HTML: [cyan]{out_dir / 'report.html'}[/cyan]")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]ERROR:[/bold red] Registration failed with exception: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
