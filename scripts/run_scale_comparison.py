"""Script to run optimized scale comparison benchmarks between baseline and scale-aware pipelines using shared terrain pairs."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import tifffile

from lunarmatch.models.config import PipelineConfig, PyramidConfig
from lunarmatch.orchestrator import RegistrationOrchestrator
from lunarmatch.synthetic.generator import generate_synthetic_pair


def main() -> None:
    # Scale ratios to evaluate
    scale_ratios = [1.0, 1.2, 1.5, 2.0, 4.0, 10.0, 20.0]
    
    # Paths for temporary assets
    temp_dir = Path("outputs/scale_comparison_dataset")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Shared pairs folder to guarantee exact same terrain for both modes
    shared_pairs_dir = temp_dir / "shared_pairs"
    shared_pairs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pre-generate the identical pairs for each scale ratio
    print("====================================================")
    print("Generating identical synthetic pairs (768x768)...")
    print("====================================================")
    
    pair_paths = {}
    for r in scale_ratios:
        pair_id = f"scale_{r:.1f}"
        pair_subdir = shared_pairs_dir / pair_id
        pair_subdir.mkdir(parents=True, exist_ok=True)
        
        src_path = pair_subdir / "source.tif"
        ref_path = pair_subdir / "reference.tif"
        
        # Generate synthetic crater terrain pair
        pair_data = generate_synthetic_pair(
            pair_id=pair_id,
            angle_deg=5.0,
            translation=(10.0, -8.0),
            scale_ratio=r,
            sun_azimuth_diff=15.0,
            shape=(768, 768),
            seed=42,
        )
        
        # Check rasterio availability to set resolution res metadata correctly
        try:
            import rasterio
            from rasterio.transform import Affine
            has_rasterio = True
        except ImportError:
            has_rasterio = False
            
        if has_rasterio:
            profile_src = {
                "driver": "GTiff",
                "height": pair_data.source_image.shape[0],
                "width": pair_data.source_image.shape[1],
                "count": 1,
                "dtype": str(pair_data.source_image.dtype),
                "transform": Affine.scale(0.25, 0.25),
            }
            with rasterio.open(src_path, "w", **profile_src) as dst:
                dst.write(pair_data.source_image, 1)
                
            profile_ref = {
                "driver": "GTiff",
                "height": pair_data.reference_image.shape[0],
                "width": pair_data.reference_image.shape[1],
                "count": 1,
                "dtype": str(pair_data.reference_image.dtype),
                "transform": Affine.scale(0.25 * r, 0.25 * r),
            }
            with rasterio.open(ref_path, "w", **profile_ref) as dst:
                dst.write(pair_data.reference_image, 1)
        else:
            tifffile.imwrite(str(src_path), pair_data.source_image)
            tifffile.imwrite(str(ref_path), pair_data.reference_image)
            
        pair_paths[r] = (src_path, ref_path)
        print(f"  Generated pair for scale {r:.1f}x (Source: {pair_data.source_image.shape}, Reference: {pair_data.reference_image.shape})")

    results = []

    print("\n====================================================")
    # 2. Run Baseline and Scale-Aware on the same generated pairs
    for mode in ["baseline", "scale-aware"]:
        print(f"Running mode: {mode.upper()}")
        print("====================================================")
        for r in scale_ratios:
            src_path, ref_path = pair_paths[r]
            
            # Setup configuration
            scale_aware = (mode == "scale-aware")
            run_output_dir = temp_dir / f"scale_{r:.1f}_{mode}" / "run_output"
            
            cfg = PipelineConfig(
                pyramid=PyramidConfig(
                    scale_aware_matching=scale_aware,
                    metadata_scale_prior=True,
                    log2_scale_min=-2,
                    log2_scale_max=2,
                ),
                run=PipelineConfig().run.model_copy(update={
                    "overwrite": True,
                    "output_dir": run_output_dir
                })
            )
            
            # Run registration
            orchestrator = RegistrationOrchestrator(cfg)
            t_start = time.perf_counter()
            try:
                manifest = orchestrator.register(source=src_path, reference=ref_path)
                elapsed = time.perf_counter() - t_start
                status = "PASS" if manifest.quality_gate_passed else "FAIL"
                
                # Fetch scale-aware details from provenance
                prov = manifest.transform.provenance if manifest.transform else {}
                detected_src = prov.get("detected_src_kps", 0)
                detected_ref = prov.get("detected_ref_kps", 0)
                cand_matches = prov.get("candidate_matches", 0)
                dedup_matches = prov.get("deduplicated_matches", 0)
                selected_scale = prov.get("selected_scale", 1.0)
                eval_hypotheses = prov.get("hypotheses_evaluated", 1)
                
                inlier_cnt = manifest.metrics.inlier_count
                inlier_rat = manifest.metrics.inlier_ratio
                rmse = manifest.metrics.rmse_px
                
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t_start
                status = "CRASH"
                detected_src = 0
                detected_ref = 0
                cand_matches = 0
                dedup_matches = 0
                selected_scale = 0.0
                eval_hypotheses = 0
                inlier_cnt = 0
                inlier_rat = 0.0
                rmse = 999.0
                print(f"  Crashed for scale {r}x: {exc}")

            results.append({
                "scale_ratio": r,
                "mode": mode,
                "detected_source_keypoints": detected_src,
                "detected_reference_keypoints": detected_ref,
                "candidate_matches": cand_matches,
                "deduplicated_matches": dedup_matches,
                "inliers": inlier_cnt,
                "inlier_ratio": inlier_rat,
                "rmse": rmse,
                "runtime": elapsed,
                "registration_status": status,
                "selected_scale_hypothesis": selected_scale,
                "number_of_hypotheses_evaluated": eval_hypotheses
            })
            print(f"  Scale {r:4.1f}x: Status={status:5s} | Inliers={inlier_cnt:4d} | RMSE={f'{rmse:.3f}' if rmse else 'N/A'} px | Runtime={elapsed:.3f} s")

    # Export results to CSV
    csv_path = temp_dir / "scale_comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print("\n====================================================")
    print(f"Benchmark results successfully exported to: {csv_path}")
    print("====================================================")

    # Print final comparison table
    print("\nFinal Comparison Summary:")
    print("Scale | Baseline Status | Baseline Inliers | Baseline RMSE | Baseline Runtime | Scale-Aware Status | Scale-Aware Inliers | Scale-Aware RMSE | Scale-Aware Runtime | Selected Scale")
    print("-" * 150)
    for r in scale_ratios:
        b_res = next(res for res in results if res["scale_ratio"] == r and res["mode"] == "baseline")
        s_res = next(res for res in results if res["scale_ratio"] == r and res["mode"] == "scale-aware")
        
        b_status = b_res["registration_status"]
        b_inliers = b_res["inliers"]
        b_rmse = f"{b_res['rmse']:.3f} px" if b_res["rmse"] is not None else "N/A"
        b_runtime = f"{b_res['runtime']:.3f} s"
        
        s_status = s_res["registration_status"]
        s_inliers = s_res["inliers"]
        s_rmse = f"{s_res['rmse']:.3f} px" if s_res["rmse"] is not None else "N/A"
        s_runtime = f"{s_res['runtime']:.3f} s"
        s_scale = f"{s_res['selected_scale_hypothesis']:.3f}"
        
        print(f"{r:5.1f} | {b_status:15s} | {b_inliers:16d} | {b_rmse:13s} | {b_runtime:16s} | {s_status:18s} | {s_inliers:19d} | {s_rmse:16s} | {s_runtime:19s} | {s_scale}")

if __name__ == "__main__":
    main()
