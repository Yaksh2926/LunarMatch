"""CLI commands for LunarMatch."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from lunarmatch import __version__, register_pair
from lunarmatch.models.config import load_config

app = typer.Typer(help="Multi-modal lunar image correspondence and registration.")
console = Console()
err_console = Console(stderr=True)


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command("validate-config")
def validate_config(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Validate YAML configuration file against PipelineConfig schema."""
    try:
        cfg = load_config(config)
        config_hash = cfg.compute_hash()
        console.print(f"[bold green]OK[/bold green] Configuration file [cyan]{config}[/cyan] is valid.")
        console.print(f"[dim]Config hash (SHA-256): {config_hash}[/dim]")
    except Exception as exc:
        err_console.print(f"[bold red]FAIL[/bold red] Invalid configuration: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("register")
def register_cmd(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            "-s",
            help="Path to source raster image file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    reference: Annotated[
        Path,
        typer.Option(
            "--reference",
            "-r",
            help="Path to reference raster image file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Path to output directory.",
            resolve_path=True,
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for reproducibility.",
        ),
    ] = None,
) -> None:
    """Run configuration-driven pair registration."""
    try:
        manifest = register_pair(
            source=source,
            reference=reference,
            config=config,
            output_dir=output_dir,
            seed=seed,
        )
        if manifest.quality_gate_passed:
            console.print(f"[bold green]PASS[/bold green] Pair registration completed for [cyan]{manifest.pair_id}[/cyan].")
            if manifest.transform is not None:
                console.print(f"  Inliers: [bold]{manifest.metrics.inlier_count}[/bold] ({manifest.metrics.inlier_ratio:.1%})")
                if manifest.metrics.rmse_px is not None:
                    console.print(f"  RMSE: [bold]{manifest.metrics.rmse_px:.3f}[/bold] px")
                console.print(f"  Runtime: [bold]{manifest.metrics.runtime_seconds:.2f}[/bold] s")
        else:
            err_console.print(f"[bold red]FAIL[/bold red] Quality gate failed for pair [cyan]{manifest.pair_id}[/cyan].")
            for warn in manifest.metrics.warnings:
                err_console.print(f"  [yellow]Warning:[/yellow] {warn}")
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        err_console.print(f"[bold red]ERROR[/bold red] Registration failed with exception: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("benchmark")
def benchmark_cmd(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            "-m",
            help="Path to JSON or CSV pairs manifest file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to output directory for batch benchmark results.",
            resolve_path=True,
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel worker processes.",
        ),
    ] = 1,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Skip already processed pair directories.",
        ),
    ] = True,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducible processing.",
        ),
    ] = 42,
) -> None:
    """Run batch image pair registration benchmark."""
    try:
        from lunarmatch.benchmark import run_batch_benchmark

        summary = run_batch_benchmark(
            manifest_path=manifest,
            config=config,
            output_dir=output_dir,
            workers=workers,
            resume=resume,
            seed=seed,
        )

        console.print("[bold green]Batch Benchmark Complete[/bold green]")
        console.print(f"  Total Pairs: [bold]{summary.total_pairs}[/bold]")
        console.print(f"  Passed Pairs: [bold green]{summary.passed_pairs}[/bold green]")
        console.print(f"  Failed Pairs: [bold red]{summary.failed_pairs}[/bold red]")
        console.print(f"  Pass Rate: [bold]{summary.pass_rate:.1%}[/bold]")
        if summary.overall_metrics.mean_rmse_px is not None:
            console.print(f"  Mean RMSE: [bold]{summary.overall_metrics.mean_rmse_px:.3f}[/bold] px")

    except Exception as exc:
        err_console.print(f"[bold red]ERROR[/bold red] Batch benchmark failed with exception: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("generate-synthetic")
def generate_synthetic_cmd(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Destination output directory for synthetic dataset.",
            resolve_path=True,
        ),
    ] = Path("outputs/synthetic_dataset"),
    num_pairs: Annotated[
        int,
        typer.Option(
            "--num-pairs",
            "-n",
            help="Number of synthetic image pairs to generate.",
        ),
    ] = 3,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducible procedural generation.",
        ),
    ] = 42,
) -> None:
    """Generate test-only synthetic lunar benchmark dataset."""
    try:
        from lunarmatch.synthetic import generate_synthetic_benchmark_dataset

        ds_path = generate_synthetic_benchmark_dataset(
            output_dir=output_dir,
            num_pairs=num_pairs,
            seed=seed,
        )
        console.print("[bold green]Synthetic Dataset Generated[/bold green]")
        console.print(f"  Location: [cyan]{ds_path}[/cyan]")
        console.print(f"  Pairs: [bold]{num_pairs}[/bold]")
        console.print(f"  Manifest: [cyan]{ds_path / 'pairs_manifest.json'}[/cyan]")
        console.print("[dim]Note: Synthetic benchmark dataset for algorithmic regression testing. Not flight performance.[/dim]")
    except Exception as exc:
        err_console.print(f"[bold red]ERROR[/bold red] Synthetic generation failed with exception: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("serve")
def serve_cmd(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind the LunarMatch web application server.",
        ),
    ] = int(os.getenv("PORT", "8000")),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host interface address to bind.",
        ),
    ] = "0.0.0.0",
) -> None:
    """Launch the interactive LunarMatch web application server."""
    from lunarmatch.web.server import run_server

    run_server(port=port, host=host)


if __name__ == "__main__":
    app()
