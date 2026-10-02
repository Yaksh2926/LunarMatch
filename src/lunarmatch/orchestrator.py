"""End-to-end configuration-driven registration orchestrator and Python API."""
from __future__ import annotations

import csv
import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from lunarmatch import __version__
from lunarmatch.data import read_raster
from lunarmatch.export import export_registered_products, warp_source_to_reference
from lunarmatch.features import ImagePyramid, get_feature_backend
from lunarmatch.features.scale_search import generate_scale_hypotheses
from lunarmatch.geometry import GeometricVerifier, SubpixelRefiner
from lunarmatch.matching import DescriptorMatcher, RegistrationQualityGate, SpatialSelector
from lunarmatch.matching.matcher import MatchResult
from lunarmatch.matching.terrain_check import check_terrain_overlap
from lunarmatch.models.config import PipelineConfig, load_config
from lunarmatch.models.domain import (
    ImagePair,
    KeypointSet,
    MatchSet,
    RasterMetadata,
    RegistrationMetrics,
    RunManifest,
    TransformEstimate,
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
        # Stage 2.5: Coarse Terrain Overlap Validation (Early Rejection)
        # ---------------------------------------------------------
        if cfg.quality_gates.verify_terrain_overlap:
            prior_scale = None
            if src_raster.metadata.pixel_scale_x and ref_raster.metadata.pixel_scale_x:
                prior_scale = float(src_raster.metadata.pixel_scale_x) / float(ref_raster.metadata.pixel_scale_x)
            
            t_valid, peak_ncc, coarse_inliers, fail_reason = check_terrain_overlap(
                src_img=src_repr.processed_image,
                ref_img=ref_repr.processed_image,
                config=cfg,
                prior_scale=prior_scale,
            )
            
            if not t_valid:
                warnings.append(
                    f"Coarse terrain overlap validation failed: peak_ncc={peak_ncc:.4f} (min {cfg.quality_gates.min_coarse_ncc}), "
                    f"coarse_inliers={coarse_inliers} (min {cfg.quality_gates.min_coarse_inliers})."
                )
                return self._create_failure_manifest(
                    pair_id=pair_id,
                    src_path=src_path,
                    ref_path=ref_path,
                    config_hash=config_hash,
                    seed=eff_seed,
                    out_dir=eff_out_dir,
                    warnings=warnings,
                    runtime_seconds=time.time() - start_wall_time,
                    src_meta=src_raster.metadata,
                    ref_meta=ref_raster.metadata,
                )

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
        # Stage 4 & 5 & 6: Feature Extraction & Matching
        # ---------------------------------------------------------
        if cfg.pyramid.scale_aware_matching:
            scales = generate_scale_hypotheses(
                source_scale_m=src_raster.metadata.pixel_scale_x,
                reference_scale_m=ref_raster.metadata.pixel_scale_x,
                config=cfg.pyramid,
                allowed_scale=cfg.geometry.allowed_scale,
            )

            hypotheses_evaluated = len(scales)

            feature_backend = get_feature_backend(cfg.features)
            matcher = DescriptorMatcher(config=cfg.matching)

            cached_src_kps: dict[int, KeypointSet] = {}
            cached_ref_kps: dict[int, KeypointSet] = {}

            features_duration = 0.0
            matching_duration = 0.0
            geometry_duration = 0.0

            total_detected_src_kps = 0
            total_detected_ref_kps = 0
            total_candidate_matches = 0

            # Store provenance/diagnostic for each hypothesis
            hypotheses_diagnostics: list[dict[str, Any]] = []

            best_geom_res = None
            best_scale = None
            best_match_result = None

            for s in scales:
                if s >= 1.0:
                    target_src_scale = 1.0
                    target_ref_scale = 1.0 / s
                else:
                    target_src_scale = s
                    target_ref_scale = 1.0

                # Select closest levels from the pre-built pyramids
                src_level = min(_src_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_src_scale))
                ref_level = min(_ref_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_ref_scale))

                # Extract features (with caching)
                t_f0 = time.perf_counter()
                if src_level.level_idx not in cached_src_kps:
                    src_mask_u8 = (src_level.valid_mask > 0).astype(np.uint8) * 255 if src_level.valid_mask is not None else None
                    cached_src_kps[src_level.level_idx] = feature_backend.detect_and_compute(src_level.image, mask=src_mask_u8)
                    total_detected_src_kps += len(cached_src_kps[src_level.level_idx])
                src_kps_scaled = cached_src_kps[src_level.level_idx]

                if ref_level.level_idx not in cached_ref_kps:
                    ref_mask_u8 = (ref_level.valid_mask > 0).astype(np.uint8) * 255 if ref_level.valid_mask is not None else None
                    cached_ref_kps[ref_level.level_idx] = feature_backend.detect_and_compute(ref_level.image, mask=ref_mask_u8)
                    total_detected_ref_kps += len(cached_ref_kps[ref_level.level_idx])
                ref_kps_scaled = cached_ref_kps[ref_level.level_idx]
                features_duration += time.perf_counter() - t_f0

                if len(src_kps_scaled) == 0 or len(ref_kps_scaled) == 0:
                    hypotheses_diagnostics.append({
                        "scale_hypothesis": float(s),
                        "source_level": src_level.level_idx,
                        "reference_level": ref_level.level_idx,
                        "src_keypoints": len(src_kps_scaled),
                        "ref_keypoints": len(ref_kps_scaled),
                        "raw_matches": 0,
                        "deduplicated_matches": 0,
                        "success": False,
                        "failure_reason": "no_keypoints",
                        "inlier_count": 0,
                        "inlier_ratio": 0.0,
                        "rmse_px": None,
                        "reflection_detected": False,
                        "spatial_spread": 0.0,
                    })
                    continue

                # Match descriptors
                t_m0 = time.perf_counter()
                match_res = matcher.match(src_kps_scaled, ref_kps_scaled)
                matching_duration += time.perf_counter() - t_m0

                if len(match_res.match_set) == 0:
                    hypotheses_diagnostics.append({
                        "scale_hypothesis": float(s),
                        "source_level": src_level.level_idx,
                        "reference_level": ref_level.level_idx,
                        "src_keypoints": len(src_kps_scaled),
                        "ref_keypoints": len(ref_kps_scaled),
                        "raw_matches": 0,
                        "deduplicated_matches": 0,
                        "success": False,
                        "failure_reason": "no_matches",
                        "inlier_count": 0,
                        "inlier_ratio": 0.0,
                        "rmse_px": None,
                        "reflection_detected": False,
                        "spatial_spread": 0.0,
                    })
                    continue

                total_candidate_matches += len(match_res.match_set)

                # Convert matching keypoints back into original-image coordinates
                src_pts_scaled = match_res.match_set.source_points
                ref_pts_scaled = match_res.match_set.reference_points

                src_pts_orig = src_level.mapping.apply(src_pts_scaled)
                ref_pts_orig = ref_level.mapping.apply(ref_pts_scaled)

                # Deduplicate correspondences for this single hypothesis in original coordinates
                tol_px = 1.5
                n_merged = len(src_pts_orig)
                if n_merged > 1:
                    sort_idx = np.lexsort((src_pts_orig[:, 1], src_pts_orig[:, 0], -match_res.match_set.scores))
                    src_sorted = src_pts_orig[sort_idx]
                    ref_sorted = ref_pts_orig[sort_idx]
                    scores_sorted = match_res.match_set.scores[sort_idx]

                    keep_mask = np.ones(n_merged, dtype=bool)

                    src_tree = cKDTree(src_sorted)
                    src_pairs = src_tree.query_pairs(r=tol_px)
                    for i_pair, j_pair in src_pairs:
                        keep_mask[j_pair] = False

                    ref_tree = cKDTree(ref_sorted)
                    ref_pairs = ref_tree.query_pairs(r=tol_px)
                    for i_pair, j_pair in ref_pairs:
                        keep_mask[j_pair] = False

                    kept_idx = np.where(keep_mask)[0]
                    dedup_src_pts = src_sorted[kept_idx]
                    dedup_ref_pts = ref_sorted[kept_idx]
                    dedup_scores = scores_sorted[kept_idx]
                else:
                    dedup_src_pts = src_pts_orig
                    dedup_ref_pts = ref_pts_orig
                    dedup_scores = match_res.match_set.scores

                hyp_match_set = MatchSet(
                    source_points=dedup_src_pts,
                    reference_points=dedup_ref_pts,
                    scores=dedup_scores,
                )

                # Run RANSAC on this hypothesis' matches
                t_g0 = time.perf_counter()
                verifier = GeometricVerifier(config=cfg.geometry)
                geom_res = verifier.verify(hyp_match_set, seed=eff_seed)
                geometry_duration += time.perf_counter() - t_g0

                # Compute spatial spread as sum of std of inliers or candidates
                pts_for_spread = geom_res.inliers.source_points if (geom_res.success and geom_res.inliers is not None) else dedup_src_pts
                spatial_spread = float(np.std(pts_for_spread[:, 0]) + np.std(pts_for_spread[:, 1])) if len(pts_for_spread) > 0 else 0.0

                # Record structured diagnostic info
                src_eff_gsd = src_raster.metadata.pixel_scale_x / src_level.scale_factor if src_raster.metadata.pixel_scale_x is not None else 1.0
                ref_eff_gsd = ref_raster.metadata.pixel_scale_x / ref_level.scale_factor if ref_raster.metadata.pixel_scale_x is not None else 1.0

                diag = {
                    "scale_hypothesis": float(s),
                    "source_level": src_level.level_idx,
                    "reference_level": ref_level.level_idx,
                    "source_effective_gsd": float(src_eff_gsd),
                    "reference_effective_gsd": float(ref_eff_gsd),
                    "src_keypoints": len(src_kps_scaled),
                    "ref_keypoints": len(ref_kps_scaled),
                    "raw_matches": len(match_res.match_set),
                    "deduplicated_matches": len(hyp_match_set),
                    "success": bool(geom_res.success),
                    "failure_reason": geom_res.failure_reason,
                    "inlier_count": int(geom_res.transform.inlier_count) if (geom_res.success and geom_res.transform is not None) else 0,
                    "inlier_ratio": float(geom_res.transform.inlier_ratio) if (geom_res.success and geom_res.transform is not None) else 0.0,
                    "rmse_px": float(geom_res.transform.rmse_px) if (geom_res.success and geom_res.transform is not None and geom_res.transform.rmse_px is not None) else None,
                    "reflection_detected": bool(geom_res.failure_reason == "reflection_detected"),
                    "spatial_spread": spatial_spread,
                }
                hypotheses_diagnostics.append(diag)

                # Keep track of the best geometric result
                if geom_res.success and geom_res.transform is not None:
                    transform_val = geom_res.transform
                    inlier_count = int(transform_val.inlier_count)
                    inlier_ratio = float(transform_val.inlier_ratio)
                    rmse_px = float(transform_val.rmse_px) if transform_val.rmse_px is not None else 1e9
                    
                    if best_geom_res is None:
                        is_better = True
                    else:
                        best_transform = best_geom_res.transform
                        assert best_transform is not None
                        best_inliers = int(best_transform.inlier_count)
                        best_ratio = float(best_transform.inlier_ratio)
                        best_rmse = float(best_transform.rmse_px) if best_transform.rmse_px is not None else 1e9
                        best_diag = next(d for d in hypotheses_diagnostics[:-1] if d["scale_hypothesis"] == best_scale)
                        best_spread = float(best_diag["spatial_spread"])
                        
                        current_key = (inlier_count, inlier_ratio, -rmse_px, spatial_spread, -s)
                        best_key = (best_inliers, best_ratio, -best_rmse, best_spread, -float(best_scale if best_scale is not None else 0.0))
                        is_better = current_key > best_key

                    if is_better:
                        best_geom_res = geom_res
                        best_scale = s
                        best_match_result = MatchResult(
                            match_set=hyp_match_set,
                            source_indices=np.arange(len(hyp_match_set), dtype=np.int64),
                            reference_indices=np.arange(len(hyp_match_set), dtype=np.int64),
                            distances=np.zeros(len(hyp_match_set), dtype=np.float64),
                            ratios=np.zeros(len(hyp_match_set), dtype=np.float64),
                            source_scales=np.ones(len(hyp_match_set), dtype=np.float32) * float(src_level.scale_factor),
                            reference_scales=np.ones(len(hyp_match_set), dtype=np.float32) * float(ref_level.scale_factor),
                        )

            stage_timings["features"] = features_duration
            stage_timings["matching"] = matching_duration
            stage_timings["geometry"] = geometry_duration

            # If no hypothesis succeeded, select the "best" failed hypothesis to report diagnostics
            if best_geom_res is None:
                best_fail_idx = -1
                best_fail_key = (0, 0, 0.0)
                for idx, diag in enumerate(hypotheses_diagnostics):
                    dedup_val = diag["deduplicated_matches"]
                    scale_val = diag["scale_hypothesis"]
                    dedup_int = int(dedup_val) if isinstance(dedup_val, (int, float)) else 0
                    scale_float = float(scale_val) if isinstance(scale_val, (int, float)) else 0.0
                    
                    has_matches = 1 if dedup_int > 0 else 0
                    current_fail_key = (has_matches, dedup_int, -scale_float)
                    if current_fail_key > best_fail_key:
                        best_fail_key = current_fail_key
                        best_fail_idx = idx

                best_failed_diag = hypotheses_diagnostics[best_fail_idx] if best_fail_idx != -1 else hypotheses_diagnostics[0]
                fail_scale = best_failed_diag['scale_hypothesis']
                fail_scale_float = float(fail_scale) if isinstance(fail_scale, (int, float)) else 0.0
                fail_reason = str(best_failed_diag['failure_reason'])
                warnings.append(f"Geometric verification failed across all scale hypotheses. Selected best failed scale: {fail_scale_float:.6f} (reason: {fail_reason})")
                
                self._current_hypotheses_diagnostics = hypotheses_diagnostics
                
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

            # Assign our best hypothesis' outputs to match the downstream pipeline expectations
            assert best_geom_res is not None
            assert best_match_result is not None
            geom_res = best_geom_res
            selected_scale = best_scale
            match_result = best_match_result
            self._current_hypotheses_diagnostics = hypotheses_diagnostics

        else:
            # ---------------------------------------------------------
            # Stage 4: Feature Extraction (Baseline)
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
            # Stage 5: Descriptor Matching (Baseline)
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
            # Stage 6: Robust Geometric Verification (Baseline)
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

        assert geom_res.inliers is not None
        assert geom_res.transform is not None
        assert match_result is not None

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
        assert final_transform is not None
        final_matches = subpixel_res.refined_matches

        # Update provenance with multi-scale matching details
        prov = dict(final_transform.provenance) if final_transform.provenance else {}
        if cfg.pyramid.scale_aware_matching:
            prov["scale_aware"] = True
            prov["hypotheses_evaluated"] = hypotheses_evaluated
            prov["selected_scale"] = selected_scale
            prov["detected_src_kps"] = total_detected_src_kps
            prov["detected_ref_kps"] = total_detected_ref_kps
            prov["candidate_matches"] = total_candidate_matches
            prov["deduplicated_matches"] = len(match_result.match_set)
            prov["hypotheses_diagnostics"] = getattr(self, "_current_hypotheses_diagnostics", None)
        else:
            prov["scale_aware"] = False
            prov["hypotheses_evaluated"] = 1
            prov["selected_scale"] = 1.0
            prov["detected_src_kps"] = len(src_kps)
            prov["detected_ref_kps"] = len(ref_kps)
            prov["candidate_matches"] = len(match_result.match_set)
            prov["deduplicated_matches"] = len(match_result.match_set)

        final_transform = TransformEstimate.from_matrix(
            model_type=final_transform.model_type,
            matrix=final_transform.get_matrix_array(),
            inlier_count=final_transform.inlier_count,
            inlier_ratio=final_transform.inlier_ratio,
            rmse_px=final_transform.rmse_px,
            provenance=prov,
        )

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
        final_rmse_px = final_transform.rmse_px

        med_res = float(np.median(post_residuals)) if len(post_residuals) > 0 else None
        p90_res = float(np.percentile(post_residuals, 90)) if len(post_residuals) > 0 else None
        p95_res = float(np.percentile(post_residuals, 95)) if len(post_residuals) > 0 else None
        max_res = float(np.max(post_residuals)) if len(post_residuals) > 0 else None

        runtime_sec = time.time() - start_wall_time

        gate_inliers = final_transform.inlier_count >= cfg.quality_gates.min_inliers
        gate_ratio = final_transform.inlier_ratio >= cfg.quality_gates.min_inlier_ratio
        gate_rmse = (cfg.quality_gates.max_rmse_px is None) or (
            final_rmse_px is not None and final_rmse_px <= cfg.quality_gates.max_rmse_px
        )

        quality_gate_passed = bool(gate_inliers and gate_ratio and gate_rmse)

        if not quality_gate_passed:
            warnings.append(
                f"Quality gate failed: inlier_count={final_transform.inlier_count} (min {cfg.quality_gates.min_inliers}), "
                f"inlier_ratio={final_transform.inlier_ratio:.2f} (min {cfg.quality_gates.min_inlier_ratio:.2f}), "
                f"rmse_px={final_rmse_px}"
            )

        spatial_gate = RegistrationQualityGate(
            min_occupied_grid_fraction=cfg.quality_gates.min_occupied_grid_fraction,
            min_convex_hull_coverage=cfg.quality_gates.min_convex_hull_coverage,
            min_inliers=cfg.quality_gates.min_inliers,
            max_rmse_px=cfg.quality_gates.max_rmse_px,
        )
        spatial_status, spatial_passed = spatial_gate.evaluate(
            inlier_count=final_transform.inlier_count,
            occupied_grid_fraction=spatial_res.occupied_grid_fraction,
            convex_hull_coverage_fraction=spatial_res.convex_hull_coverage_fraction,
            rmse_px=final_rmse_px,
        )

        if spatial_status == "UNDER_CONSTRAINED":
            warnings.append(
                f"Spatial coverage gate flagged result as UNDER_CONSTRAINED "
                f"(occupied_grid={spatial_res.occupied_grid_fraction:.1%}, "
                f"hull_coverage={spatial_res.convex_hull_coverage_fraction:.1%})"
            )

        metrics = RegistrationMetrics(
            rmse_px=final_rmse_px,
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
            spatial_coverage_status=spatial_status,
            spatial_coverage_passed=spatial_passed,
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
            hypotheses_diagnostics=getattr(self, "_current_hypotheses_diagnostics", None),
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
            hypotheses_diagnostics=getattr(self, "_current_hypotheses_diagnostics", None),
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
