"""ISRO Problem Statement 26166 Multi-Sensor Lunar Image Registration Demo."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunarmatch.models.config import PipelineConfig
from lunarmatch.orchestrator import RegistrationOrchestrator
from lunarmatch.sensors.factory import detect_sensor_adapter, load_sensor_product


def main() -> None:
    print("=========================================================================")
    print("ISRO Problem Statement 26166: Multi-Sensor Lunar Image Registration Demo")
    print("=========================================================================\n")

    output_dir = Path("outputs/isro_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer preprocessed 4000x4000 overlap crops if full 1GB raster is too large for single-chip RAM
    source_path = Path("sample_data/real/preprocessed/ohrc_preprocessed.tif")
    reference_path = Path("sample_data/real/preprocessed/lroc_preprocessed.tif")

    if not source_path.exists() or not reference_path.exists():
        source_path = Path("sample_data/real/ohrc/ch2_ohr_ncp_20230303T0350447888_d_img_n18.xml")
        reference_path = Path("sample_data/real/ohrc/ch2_ohr_ncp_20230303T0152168201_d_img_n18.xml")

    # Step 1: Detect and load sensor products
    source_adapter = detect_sensor_adapter(source_path)
    reference_adapter = detect_sensor_adapter(reference_path)

    print("1. Sensor Detection:")
    print(f"   Source Path: {source_path.name} -> Adapter: {source_adapter.__class__.__name__}")
    print(f"   Reference Path: {reference_path.name} -> Adapter: {reference_adapter.__class__.__name__}")

    source_product = load_sensor_product(source_path)
    reference_product = load_sensor_product(reference_path)

    print("\n2. Sensor Product Metadata:")
    print(f"   Source Sensor: {source_product.sensor_name} | GSD: {source_product.mean_gsd} m/px | Dimensions: {source_product.width}x{source_product.height}")
    print(f"   Reference Sensor: {reference_product.sensor_name} | GSD: {reference_product.mean_gsd} m/px | Dimensions: {reference_product.width}x{reference_product.height}")

    # Step 2: Configure Registration Pipeline
    config = PipelineConfig()
    config = config.model_copy(update={
        "run": config.run.model_copy(update={"output_dir": output_dir, "overwrite": True}),
        "pyramid": config.pyramid.model_copy(update={"scale_aware_matching": True, "metadata_scale_prior": True}),
        "quality_gates": config.quality_gates.model_copy(update={"min_inliers": 20, "min_inlier_ratio": 0.15})
    })

    # Step 3: Run Orchestration
    print("\n3. Running Registration Orchestrator...")
    orchestrator = RegistrationOrchestrator(config)
    manifest = orchestrator.register(
        source=source_path,
        reference=reference_path,
        output_dir=output_dir,
    )

    # Step 4: Display Output Summary & Metrics
    print("\n=========================================================================")
    print("DEMONSTRATION RESULTS & REGISTRATION METRICS")
    print("=========================================================================")
    print(f"Quality Gate Status: {'PASSED' if manifest.quality_gate_passed else 'FAILED'}")
    print(f"Inlier Match Count:  {manifest.metrics.inlier_count}")
    print(f"Total Correspondences: {manifest.metrics.total_matches}")
    print(f"Inlier Match Ratio:  {manifest.metrics.inlier_ratio * 100.0:.2f}%")
    print(f"Reprojection RMSE:   {manifest.metrics.rmse_px:.4f} pixels" if manifest.metrics.rmse_px else "RMSE: N/A")
    print("\nSpatial Match Distribution Metrics:")
    print(f"Occupied Grid Fraction: {manifest.metrics.occupied_grid_fraction * 100.0:.1f}%")
    print(f"Convex Hull Coverage:   {manifest.metrics.convex_hull_coverage_fraction * 100.0:.1f}%")
    print(f"Count Uniformity CV:    {manifest.metrics.count_uniformity_cv:.4f}" if manifest.metrics.count_uniformity_cv else "CV: N/A")

    if manifest.transform:
        matrix = manifest.transform.get_matrix_array()
        print("\nResolved Transformation Matrix (Affine/Similarity):")
        print(f"  [[{matrix[0,0]:.6f}, {matrix[0,1]:.6f}, {matrix[0,2]:.2f}],")
        print(f"   [{matrix[1,0]:.6f}, {matrix[1,1]:.6f}, {matrix[1,2]:.2f}],")
        print(f"   [{matrix[2,0]:.6f}, {matrix[2,1]:.6f}, {matrix[2,2]:.2f}]]")

    manifest_json = output_dir / "registration_manifest.json"
    print(f"\nManifest saved to: {manifest_json}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
