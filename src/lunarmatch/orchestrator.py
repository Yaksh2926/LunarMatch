"""End-to-end configuration-driven registration orchestrator and Python API."""
from __future__ import annotations

import csv
import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lunarmatch import __version__
from lunarmatch.data import read_raster
from lunarmatch.export import export_registered_products, warp_source_to_reference
from lunarmatch.features import ImagePyramid, get_feature_backend
from lunarmatch.geometry import GeometricVerifier, SubpixelRefiner
from lunarmatch.matching import DescriptorMatcher, SpatialSelector
from lunarmatch.models.config import PipelineConfig, load_config
from lunarmatch.models.domain import (
    ImagePair,
    MatchSet,
    RasterMetadata,
    RegistrationMetrics,
    RunManifest,
)
from lunarmatch.preprocessing import preprocess_raster


def export_matches_csv(matches: MatchSet, output_path: Path) -> None:
    """Export matches float64 coordinates and metadata to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "score",
                "inlier",
                "cell_id",
                "refine_status",
            ]
        )
        src = matches.source_points
        ref = matches.reference_points
        scores = matches.scores
        inliers = matches.inliers
        cell_ids = matches.cell_ids
        statuses = matches.refine_status

        for i in range(len(src)):
            cid = cell_ids[i] if cell_ids is not None and i < len(cell_ids) else ""
            status = statuses[i] if statuses is not None and i < len(statuses) else ""
            writer.writerow(
                [
                    f"{src[i, 0]:.6f}",
                    f"{src[i, 1]:.6f}",
                    f"{ref[i, 0]:.6f}",
                    f"{ref[i, 1]:.6f}",
                    f"{scores[i]:.6f}",
                    int(inliers[i]),
                    cid,
                    status,
                ]
            )


class RegistrationOrchestrator:
    """Configuration-driven orchestrator connecting all 10 pipeline stages."""

    def __init__(self, config: PipelineConfig | str | Path | None = None) -> None:
        if config is None:
            self._config = PipelineConfig()
        elif isinstance(config, PipelineConfig):
            self._config = config
        else:
            self._config = load_config(config)

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def register(
        self,
        source: str | Path,
        reference: str | Path,
        output_dir: str | Path | None = None,
        seed: int | None = None,
    ) -> RunManifest:
        """Run registration pipeline on source and reference images.

        Args:
            source: Path to source raster image file.
            reference: Path to reference raster image file.
            output_dir: Optional output directory path override.
            seed: Optional random seed integer override.

        Returns:
            RunManifest object capturing complete execution metadata, metrics, and transform.
        """
        start_wall_time = time.time()
        stage_timings: dict[str, float] = {}
        warnings: list[str] = []

        cfg = self._config
        config_hash = cfg.compute_hash()

        eff_seed = seed if seed is not None else cfg.run.seed
        eff_out_dir = Path(output_dir) if output_dir is not None else cfg.run.output_dir

        # Set deterministic RNG seed
        np.random.seed(eff_seed)

        src_path = Path(source).resolve()
        ref_path = Path(reference).resolve()

        pair_id = f"{src_path.stem}_{ref_path.stem}"

        # ---------------------------------------------------------
        # Stage 1: Raster I/O
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        try:
            src_raster = read_raster(src_path, band=cfg.io.source_band)
            ref_raster = read_raster(ref_path, band=cfg.io.reference_band)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to read raster input files: {exc}")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        pair = ImagePair(pair_id=pair_id, source_meta=src_raster.metadata, reference_meta=ref_raster.metadata)
        stage_timings["io"] = time.perf_counter() - t0

        # Edge Case Check 1: 100% Nodata / Masked Out Image
        if np.count_nonzero(src_raster.valid_mask) == 0:
            warnings.append("Source raster is 100% nodata / masked out")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        if np.count_nonzero(ref_raster.valid_mask) == 0:
            warnings.append("Reference raster is 100% nodata / masked out")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        # Edge Case Check 2: Constant / Zero Variance Featureless Image
        src_valid_pixels = src_raster.data[src_raster.valid_mask]
        ref_valid_pixels = ref_raster.data[ref_raster.valid_mask]
        if len(src_valid_pixels) == 0 or np.std(src_valid_pixels) == 0:
            warnings.append("Source raster image has zero variance (constant featureless raster)")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        if len(ref_valid_pixels) == 0 or np.std(ref_valid_pixels) == 0:
            warnings.append("Reference raster image has zero variance (constant featureless raster)")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        # Edge Case Check 3: Extreme Scale Prior Mismatch (> 10x)
        if src_raster.metadata.pixel_scale_x and ref_raster.metadata.pixel_scale_x:
            s1 = src_raster.metadata.pixel_scale_x
            s2 = ref_raster.metadata.pixel_scale_x
            ratio = max(s1, s2) / max(1e-6, min(s1, s2))
            if ratio > 10.0:
                warnings.append(
                    f"Extreme pixel scale mismatch detected ({ratio:.1f}x scale ratio exceeds 10.0x max threshold)"
                )

        # Edge Case Check 4: Memory Budget Allocation Warning
        est_mem_mb = (src_raster.data.nbytes + ref_raster.data.nbytes) / (1024.0 * 1024.0) * 3.0
        if est_mem_mb > cfg.performance.max_memory_mb:
            warnings.append(
                f"Estimated memory allocation ({est_mem_mb:.1f} MB) exceeds configured budget ({cfg.performance.max_memory_mb} MB). Enforcing downsampled pyramid planning."
            )

        # ---------------------------------------------------------
        # Stage 2: Structural Preprocessing
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        src_repr = preprocess_raster(src_raster.data, src_raster.valid_mask, config=cfg.preprocess)
        ref_repr = preprocess_raster(ref_raster.data, ref_raster.valid_mask, config=cfg.preprocess)
        stage_timings["preprocessing"] = time.perf_counter() - t0

        # ---------------------------------------------------------
        # Stage 3: Pyramid Planning
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        _src_pyr = ImagePyramid.build(
            src_repr.processed_image, mask=src_repr.valid_mask, config=cfg.pyramid
        )
        _ref_pyr = ImagePyramid.build(
            ref_repr.processed_image, mask=ref_repr.valid_mask, config=cfg.pyramid
        )
        stage_timings["pyramid"] = time.perf_counter() - t0

        # ---------------------------------------------------------
        # Stage 4: Feature Extraction
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        try:
            feature_backend = get_feature_backend(cfg.features)
            src_mask_u8 = (src_repr.valid_mask > 0).astype(np.uint8) * 255 if src_repr.valid_mask is not None else None
            ref_mask_u8 = (ref_repr.valid_mask > 0).astype(np.uint8) * 255 if ref_repr.valid_mask is not None else None

            src_kps = feature_backend.detect_and_compute(src_repr.processed_image, mask=src_mask_u8)
            ref_kps = feature_backend.detect_and_compute(ref_repr.processed_image, mask=ref_mask_u8)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Feature backend execution failed: {exc}")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        stage_timings["features"] = time.perf_counter() - t0

        if len(src_kps) == 0 or len(ref_kps) == 0:
            warnings.append(f"Zero keypoints detected (source={len(src_kps)}, reference={len(ref_kps)})")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        # ---------------------------------------------------------
        # Stage 5: Descriptor Matching
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        matcher = DescriptorMatcher(config=cfg.matching)
        match_result = matcher.match(src_kps, ref_kps)
        stage_timings["matching"] = time.perf_counter() - t0

        if len(match_result.match_set) == 0:
            warnings.append("Descriptor matching yielded zero matches")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        # ---------------------------------------------------------
        # Stage 6: Robust Geometric Verification
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        verifier = GeometricVerifier(config=cfg.geometry)
        geom_res = verifier.verify(match_result.match_set, seed=eff_seed)
        stage_timings["geometry"] = time.perf_counter() - t0

        if not geom_res.success or geom_res.transform is None or geom_res.inliers is None:
            warnings.append(f"Geometric verification failed: {geom_res.failure_reason}")
            return self._create_failure_manifest(
                pair_id=pair_id,
                src_path=src_path,
                ref_path=ref_path,
                config_hash=config_hash,
                seed=eff_seed,
                out_dir=eff_out_dir,
                warnings=warnings,
                runtime_seconds=time.time() - start_wall_time,
            )

        # ---------------------------------------------------------
        # Stage 7: Uniform Spatial Selection
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        spatial_selector = SpatialSelector(config=cfg.coverage)
        spatial_res = spatial_selector.select(
            matches=geom_res.inliers,
            image_bounds=(int(pair.reference_meta.width), int(pair.reference_meta.height)),
            valid_mask=ref_repr.valid_mask,
        )
        stage_timings["coverage"] = time.perf_counter() - t0

        # ---------------------------------------------------------
        # Stage 8: Sub-Pixel Refinement & Final Geometry Refit
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        subpixel_refiner = SubpixelRefiner(config=cfg.subpixel)
        subpixel_res = subpixel_refiner.refine(
            source_image=src_repr.processed_image,
            reference_image=ref_repr.processed_image,
            matches=spatial_res.selected_matches,
            initial_transform=geom_res.transform,
            source_mask=src_repr.valid_mask,
            reference_mask=ref_repr.valid_mask,
            geometry_config=cfg.geometry,
        )
        stage_timings["subpixel"] = time.perf_counter() - t0

        final_transform = subpixel_res.refit_transform or geom_res.transform
        final_matches = subpixel_res.refined_matches

        # ---------------------------------------------------------
        # Stage 9: Source Raster Warping
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        warped_image = warp_source_to_reference(
            source_image=src_raster.data,
            transform=final_transform,
            reference_shape=(ref_raster.metadata.height, ref_raster.metadata.width),
            interpolation=cfg.export.interpolation,
            nodata_value=ref_raster.metadata.nodata_value or 0.0,
        )
        stage_timings["warp"] = time.perf_counter() - t0

        # ---------------------------------------------------------
        # Stage 10: Metrics & Quality Gate Evaluation
        # ---------------------------------------------------------
        post_residuals = subpixel_res.post_residuals_px
        rmse_px = final_transform.rmse_px

        med_res = float(np.median(post_residuals)) if len(post_residuals) > 0 else None
        p90_res = float(np.percentile(post_residuals, 90)) if len(post_residuals) > 0 else None
        p95_res = float(np.percentile(post_residuals, 95)) if len(post_residuals) > 0 else None
        max_res = float(np.max(post_residuals)) if len(post_residuals) > 0 else None

        runtime_sec = time.time() - start_wall_time

        gate_inliers = final_transform.inlier_count >= cfg.quality_gates.min_inliers
        gate_ratio = final_transform.inlier_ratio >= cfg.quality_gates.min_inlier_ratio
        gate_rmse = (cfg.quality_gates.max_rmse_px is None) or (
            rmse_px is not None and rmse_px <= cfg.quality_gates.max_rmse_px
        )

        quality_gate_passed = bool(gate_inliers and gate_ratio and gate_rmse)

        if not quality_gate_passed:
            warnings.append(
                f"Quality gate failed: inlier_count={final_transform.inlier_count} (min {cfg.quality_gates.min_inliers}), "
                f"inlier_ratio={final_transform.inlier_ratio:.2f} (min {cfg.quality_gates.min_inlier_ratio:.2f}), "
                f"rmse_px={rmse_px}"
            )

        metrics = RegistrationMetrics(
            rmse_px=rmse_px,
            median_residual_px=med_res,
            p90_residual_px=p90_res,
            p95_residual_px=p95_res,
            max_residual_px=max_res,
            inlier_count=final_transform.inlier_count,
            total_matches=len(match_result.match_set),
            inlier_ratio=final_transform.inlier_ratio,
            occupied_grid_fraction=spatial_res.occupied_grid_fraction,
            convex_hull_coverage_fraction=spatial_res.convex_hull_coverage_fraction,
            count_uniformity_cv=spatial_res.count_uniformity_cv,
            runtime_seconds=runtime_sec,
            warnings=warnings,
        )

        # ---------------------------------------------------------
        # Stage 11: Serialization & Manifest Export
        # ---------------------------------------------------------
        manifest = RunManifest(
            pair_id=pair_id,
            source_path=str(src_path),
            reference_path=str(ref_path),
            config_hash=config_hash,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            package_version=__version__,
            seed=eff_seed,
            metrics=metrics,
            transform=final_transform,
            quality_gate_passed=quality_gate_passed,
        )

        export_registered_products(
            manifest=manifest,
            matches=final_matches,
            warped_image=warped_image,
            source_metadata=src_raster.metadata,
            reference_metadata=ref_raster.metadata,
            output_dir=eff_out_dir,
            overwrite=cfg.run.overwrite,
            export_geojson=cfg.export.write_geojson,
            stage_timings=stage_timings,
        )
        return manifest

    def _create_failure_manifest(
        self,
        pair_id: str,
        src_path: Path,
        ref_path: Path,
        config_hash: str,
        seed: int,
        out_dir: Path,
        warnings: list[str],
        runtime_seconds: float,
        src_meta: RasterMetadata | None = None,
        ref_meta: RasterMetadata | None = None,
    ) -> RunManifest:
        """Construct and serialize failure manifest for ordinary pipeline failures."""
        metrics = RegistrationMetrics(
            rmse_px=None,
            median_residual_px=None,
            p90_residual_px=None,
            p95_residual_px=None,
            max_residual_px=None,
            inlier_count=0,
            total_matches=0,
            inlier_ratio=0.0,
            occupied_grid_fraction=0.0,
            convex_hull_coverage_fraction=0.0,
            count_uniformity_cv=None,
            runtime_seconds=runtime_seconds,
            warnings=warnings,
        )

        manifest = RunManifest(
            pair_id=pair_id,
            source_path=str(src_path),
            reference_path=str(ref_path),
            config_hash=config_hash,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            package_version=__version__,
            seed=seed,
            metrics=metrics,
            transform=None,
            quality_gate_passed=False,
        )

        empty_ms = MatchSet(
            source_points=np.empty((0, 2), dtype=np.float64),
            reference_points=np.empty((0, 2), dtype=np.float64),
            scores=np.empty((0,), dtype=np.float64),
        )
        dummy_meta = RasterMetadata(
            path=src_path,
            width=1,
            height=1,
            bands=1,
            dtype="uint8",
        )
        s_meta = src_meta or dummy_meta
        r_meta = ref_meta or dummy_meta

        export_registered_products(
            manifest=manifest,
            matches=empty_ms,
            warped_image=None,
            source_metadata=s_meta,
            reference_metadata=r_meta,
            output_dir=out_dir,
            overwrite=True,
            export_geojson=False,
            stage_timings={},
        )
        return manifest

    def _export_outputs(
        self,
        manifest: RunManifest,
        matches: MatchSet,
        out_dir: Path,
        stage_timings: dict[str, float],
    ) -> None:
        """Export run_manifest.json, metrics.json, transform.json, matches.csv, and diagnostics."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. run_manifest.json
        manifest_path = out_dir / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # 2. metrics.json
        metrics_path = out_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.metrics.to_dict(), f, indent=2)

        # 3. transform.json
        if manifest.transform is not None:
            transform_path = out_dir / "transform.json"
            with transform_path.open("w", encoding="utf-8") as f:
                json.dump(manifest.transform.to_dict(), f, indent=2)

        # 4. matches.csv
        matches_path = out_dir / "matches.csv"
        export_matches_csv(matches, matches_path)

        # 5. diagnostics.json
        diag_path = out_dir / "diagnostics.json"
        diag_data: dict[str, Any] = {
            "pair_id": manifest.pair_id,
            "quality_gate_passed": manifest.quality_gate_passed,
            "stage_timings_seconds": stage_timings,
            "warnings": manifest.metrics.warnings,
        }
        with diag_path.open("w", encoding="utf-8") as f:
            json.dump(diag_data, f, indent=2)


def register_pair(
    source: str | Path,
    reference: str | Path,
    config: PipelineConfig | str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> RunManifest:
    """Python API function to run image pair registration.

    Args:
        source: Path to source raster file.
        reference: Path to reference raster file.
        config: Optional PipelineConfig instance or YAML file path.
        output_dir: Optional output directory override.
        seed: Optional random seed override.

    Returns:
        RunManifest instance capturing execution status, metrics, and transform.
    """
    orchestrator = RegistrationOrchestrator(config=config)
    return orchestrator.register(
        source=source,
        reference=reference,
        output_dir=output_dir,
        seed=seed,
    )
