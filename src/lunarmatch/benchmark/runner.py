"""Batch benchmark runner with restart safety, failure isolation, and thread control."""
from __future__ import annotations

import concurrent.futures
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lunarmatch.benchmark.models import BenchmarkPairInput, BenchmarkSummary, BinnedMetrics
from lunarmatch.models.config import PipelineConfig
from lunarmatch.models.domain import RunManifest
from lunarmatch.orchestrator import RegistrationOrchestrator


def load_pairs_manifest(manifest_path: str | Path) -> list[BenchmarkPairInput]:
    """Ingest pair specifications from a JSON or CSV manifest file.

    Args:
        manifest_path: Path to JSON or CSV manifest file.

    Returns:
        List of BenchmarkPairInput instances.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If file format is unsupported or pair specifications are invalid.
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pairs manifest file not found: {path}")

    pairs: list[BenchmarkPairInput] = []

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]

        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must contain a list of pair objects, got {type(data)}")

        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            if "pair_id" not in item:
                item["pair_id"] = f"pair_{idx:03d}"
            pairs.append(BenchmarkPairInput(**item))

    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                clean_row: dict[str, Any] = {k.strip(): v.strip() for k, v in row.items() if k and v}
                if "pair_id" not in clean_row:
                    clean_row["pair_id"] = f"pair_{idx:03d}"

                # Parse float fields if present
                for float_field in ["pixel_scale_ratio", "sun_angle_diff_deg"]:
                    if float_field in clean_row:
                        try:
                            clean_row[float_field] = float(clean_row[float_field])
                        except ValueError:
                            del clean_row[float_field]

                pairs.append(BenchmarkPairInput(**clean_row))
    else:
        raise ValueError(f"Unsupported manifest file extension '{path.suffix}'. Expected .json or .csv")

    if not pairs:
        raise ValueError(f"No valid pair entries found in manifest '{path.name}'")

    return pairs


def _execute_single_pair(
    pair_input: BenchmarkPairInput,
    config: PipelineConfig | str | Path | None,
    output_dir: Path,
    seed: int,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute single pair registration with exception isolation and OpenCV thread limiting."""
    # Enforce 1 thread per process for OpenCV to avoid thread contention in multiprocessing
    try:
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001, S110
        pass

    pair_out_dir = output_dir / pair_input.pair_id
    manifest_file = pair_out_dir / "run_manifest.json"

    # Restart safety: skip if manifest already exists
    if resume and manifest_file.is_file():
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = RunManifest.from_dict(data)
            metrics = manifest.metrics
            return {
                "pair_id": pair_input.pair_id,
                "quality_gate_passed": manifest.quality_gate_passed,
                "rmse_px": metrics.rmse_px,
                "inlier_count": metrics.inlier_count,
                "inlier_ratio": metrics.inlier_ratio,
                "runtime_seconds": metrics.runtime_seconds,
                "warnings": metrics.warnings,
                "sensor_pair": pair_input.sensor_pair,
                "scale_bin": pair_input.scale_bin,
                "sun_angle_bin": pair_input.sun_angle_bin,
                "resumed": True,
            }
        except Exception:  # noqa: BLE001, S110
            pass  # Fall through to recompute if existing file is corrupted

    orchestrator = RegistrationOrchestrator(config=config)
    try:
        manifest = orchestrator.register(
            source=pair_input.source,
            reference=pair_input.reference,
            output_dir=pair_out_dir,
            seed=seed,
        )
        metrics = manifest.metrics
        return {
            "pair_id": pair_input.pair_id,
            "quality_gate_passed": manifest.quality_gate_passed,
            "rmse_px": metrics.rmse_px,
            "inlier_count": metrics.inlier_count,
            "inlier_ratio": metrics.inlier_ratio,
            "runtime_seconds": metrics.runtime_seconds,
            "warnings": metrics.warnings,
            "sensor_pair": pair_input.sensor_pair,
            "scale_bin": pair_input.scale_bin,
            "sun_angle_bin": pair_input.sun_angle_bin,
            "resumed": False,
        }
    except Exception as exc:  # noqa: BLE001
        # Exception isolation: emit failure result rather than crashing batch
        return {
            "pair_id": pair_input.pair_id,
            "quality_gate_passed": False,
            "rmse_px": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "runtime_seconds": 0.0,
            "warnings": [f"Worker unhandled exception: {exc}"],
            "sensor_pair": pair_input.sensor_pair,
            "scale_bin": pair_input.scale_bin,
            "sun_angle_bin": pair_input.sun_angle_bin,
            "resumed": False,
        }


def _worker_entrypoint(task_args: tuple[dict[str, Any], Any, str, int, bool]) -> dict[str, Any]:
    """Top-level unpicklable worker function for ProcessPoolExecutor."""
    pair_dict, cfg, out_str, seed, resume = task_args
    pair_input = BenchmarkPairInput(**pair_dict)
    return _execute_single_pair(pair_input, cfg, Path(out_str), seed, resume)


def _compute_binned_stats(bin_name: str, pair_results: list[dict[str, Any]]) -> BinnedMetrics:
    """Compute aggregate metrics for a list of pair results in a specific bin."""
    total = len(pair_results)
    if total == 0:
        return BinnedMetrics(
            bin_name=bin_name,
            total_pairs=0,
            passed_pairs=0,
            failed_pairs=0,
            pass_rate=0.0,
        )

    passed = [r for r in pair_results if r.get("quality_gate_passed", False)]
    failed = [r for r in pair_results if not r.get("quality_gate_passed", False)]

    passed_count = len(passed)
    failed_count = len(failed)
    pass_rate = float(passed_count) / float(total)

    rmses = [r["rmse_px"] for r in passed if r.get("rmse_px") is not None]
    mean_rmse = float(np.mean(rmses)) if len(rmses) > 0 else None
    med_rmse = float(np.median(rmses)) if len(rmses) > 0 else None

    inliers = [r.get("inlier_count", 0) for r in pair_results]
    ratios = [r.get("inlier_ratio", 0.0) for r in pair_results]
    runtimes = [r.get("runtime_seconds", 0.0) for r in pair_results]

    return BinnedMetrics(
        bin_name=bin_name,
        total_pairs=total,
        passed_pairs=passed_count,
        failed_pairs=failed_count,
        pass_rate=pass_rate,
        mean_rmse_px=mean_rmse,
        median_rmse_px=med_rmse,
        mean_inliers=float(np.mean(inliers)) if inliers else 0.0,
        mean_inlier_ratio=float(np.mean(ratios)) if ratios else 0.0,
        mean_runtime_seconds=float(np.mean(runtimes)) if runtimes else 0.0,
    )


def run_batch_benchmark(
    manifest_path: str | Path,
    config: PipelineConfig | str | Path | None = None,
    output_dir: str | Path | None = None,
    workers: int = 1,
    resume: bool = True,
    seed: int = 42,
) -> BenchmarkSummary:
    """Execute batch image pair registration benchmark across manifest pairs.

    Args:
        manifest_path: Path to JSON or CSV pair manifest file.
        config: Optional PipelineConfig instance or YAML config path.
        output_dir: Destination output directory.
        workers: Number of parallel worker processes (default 1).
        resume: If True, skip already processed pair directories.
        seed: Random seed integer for reproducible processing.

    Returns:
        BenchmarkSummary containing aggregate metrics, failure isolation tracking, and binned performance.
    """
    pairs = load_pairs_manifest(manifest_path)
    out_dir = Path(output_dir or "outputs/benchmark_001").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_results: list[dict[str, Any]] = []

    if workers > 1:
        tasks = [
            (p.model_dump(), config, str(out_dir), seed + idx, resume)
            for idx, p in enumerate(pairs)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker_entrypoint, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    pair_results.append(res)
                except Exception as exc:  # noqa: BLE001
                    pair_results.append(
                        {
                            "pair_id": "unknown_future_failure",
                            "quality_gate_passed": False,
                            "rmse_px": None,
                            "inlier_count": 0,
                            "inlier_ratio": 0.0,
                            "runtime_seconds": 0.0,
                            "warnings": [f"Executor future error: {exc}"],
                        }
                    )
    else:
        for idx, p in enumerate(pairs):
            res = _execute_single_pair(p, config, out_dir, seed=seed + idx, resume=resume)
            pair_results.append(res)

    # Sort results by pair_id for deterministic output order
    pair_results.sort(key=lambda r: r.get("pair_id", ""))

    # Compute overall aggregate metrics
    overall_stats = _compute_binned_stats("Overall", pair_results)

    # Compute binned metrics
    binned_sensor: list[BinnedMetrics] = []
    sensor_groups: dict[str, list[dict[str, Any]]] = {}
    for r in pair_results:
        sp = r.get("sensor_pair")
        if sp:
            sensor_groups.setdefault(sp, []).append(r)
    for sp_name, items in sorted(sensor_groups.items()):
        binned_sensor.append(_compute_binned_stats(sp_name, items))

    binned_scale: list[BinnedMetrics] = []
    scale_groups: dict[str, list[dict[str, Any]]] = {}
    for r in pair_results:
        sb = r.get("scale_bin")
        if sb:
            scale_groups.setdefault(sb, []).append(r)
    for sb_name, items in sorted(scale_groups.items()):
        binned_scale.append(_compute_binned_stats(sb_name, items))

    binned_sun: list[BinnedMetrics] = []
    sun_groups: dict[str, list[dict[str, Any]]] = {}
    for r in pair_results:
        sab = r.get("sun_angle_bin")
        if sab:
            sun_groups.setdefault(sab, []).append(r)
    for sab_name, items in sorted(sun_groups.items()):
        binned_sun.append(_compute_binned_stats(sab_name, items))

    summary = BenchmarkSummary(
        total_pairs=len(pair_results),
        passed_pairs=overall_stats.passed_pairs,
        failed_pairs=overall_stats.failed_pairs,
        pass_rate=overall_stats.pass_rate,
        overall_metrics=overall_stats,
        binned_by_sensor=binned_sensor,
        binned_by_scale=binned_scale,
        binned_by_sun_angle=binned_sun,
        pair_results=pair_results,
    )

    # Export benchmark_summary.json
    summary_json_path = out_dir / "benchmark_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Export benchmark_summary.csv
    summary_csv_path = out_dir / "benchmark_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "quality_gate_passed",
                "rmse_px",
                "inlier_count",
                "inlier_ratio",
                "runtime_seconds",
                "sensor_pair",
                "scale_bin",
                "sun_angle_bin",
                "warnings",
            ]
        )
        for r in pair_results:
            writer.writerow(
                [
                    r.get("pair_id", ""),
                    int(r.get("quality_gate_passed", False)),
                    f"{r['rmse_px']:.4f}" if r.get("rmse_px") is not None else "",
                    r.get("inlier_count", 0),
                    f"{r.get('inlier_ratio', 0.0):.4f}",
                    f"{r.get('runtime_seconds', 0.0):.4f}",
                    r.get("sensor_pair") or "",
                    r.get("scale_bin") or "",
                    r.get("sun_angle_bin") or "",
                    "; ".join(r.get("warnings", [])),
                ]
            )

    return summary
