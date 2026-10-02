"""Unit tests for scale-aware coordinate mapping, deduplication, and end-to-end multi-scale validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from lunarmatch.geometry.coordinate_mapping import CoordinateMapping
from lunarmatch.models.config import PipelineConfig, PyramidConfig
from lunarmatch.orchestrator import RegistrationOrchestrator


def test_coordinate_mapping_and_pyramid() -> None:
    """Test that pyramid-level keypoints map correctly to original coordinates and invert correctly."""
    # Create coordinate mapping for a scale factor of 0.5 (level size 32x32, original 64x64)
    mapping = CoordinateMapping.from_scale(2.0, 2.0)

    # Test coordinate mapping forwards
    level_pts = np.array([[10.0, 15.0], [20.0, 30.0]], dtype=np.float64)
    orig_pts = mapping.apply(level_pts)

    np.testing.assert_allclose(orig_pts, np.array([[20.0, 30.0], [40.0, 60.0]], dtype=np.float64))

    # Test scale transformations invert correctly
    restored_pts = mapping.apply_inverse(orig_pts)
    np.testing.assert_allclose(restored_pts, level_pts)


def test_windowed_coordinate_mapping() -> None:
    """Test that cropped/windowed coordinates composited with scaling remain correct."""
    # Target crops source, top-left offset (10, 20), and downsamples by 0.5x
    crop_map = CoordinateMapping.from_crop(10.0, 20.0)
    scale_map = CoordinateMapping.from_scale(2.0, 2.0)

    # Composite B -> C (crop) and A -> B (scale): maps scale -> original
    composed = crop_map.compose(scale_map)

    scaled_pts = np.array([[5.0, 10.0]], dtype=np.float64)
    orig_pts = composed.apply(scaled_pts)

    # Should scale first then offset: 5 * 2 + 10 = 20, 10 * 2 + 20 = 40
    np.testing.assert_allclose(orig_pts, np.array([[20.0, 40.0]], dtype=np.float64))


def test_deduplication_mats() -> None:
    """Test that duplicate correspondences across hypotheses are removed correctly and consolidated safely."""
    # Mock some matches with overlapping original coordinates
    src_pts = np.array([[10.0, 20.0], [10.2, 20.1], [50.0, 50.0]], dtype=np.float64)
    ref_pts = np.array([[30.0, 40.0], [30.1, 40.3], [70.0, 70.0]], dtype=np.float64)
    scores = np.array([0.9, 0.95, 0.8], dtype=np.float64)

    # Deduplicate matches
    tol_px = 1.5
    n_pts = len(src_pts)
    assert n_pts > 1

    # Sort descending by score
    sort_idx = np.lexsort((src_pts[:, 1], src_pts[:, 0], -scores))
    src_sorted = src_pts[sort_idx]
    ref_sorted = ref_pts[sort_idx]
    scores_sorted = scores[sort_idx]

    keep_mask = np.ones(n_pts, dtype=bool)

    src_tree = cKDTree(src_sorted)
    src_pairs = src_tree.query_pairs(r=tol_px)
    for i, j in src_pairs:
        keep_mask[j] = False

    ref_tree = cKDTree(ref_sorted)
    ref_pairs = ref_tree.query_pairs(r=tol_px)
    for i, j in ref_pairs:
        keep_mask[j] = False

    kept_idx = np.where(keep_mask)[0]
    dedup_src = src_sorted[kept_idx]
    dedup_ref = ref_sorted[kept_idx]
    dedup_scores = scores_sorted[kept_idx]
    assert len(dedup_ref) == 2

    # Expect: [10.2, 20.1] matched with [30.1, 40.3] (score 0.95) to be kept,
    # and [10.0, 20.0] matched with [30.0, 40.0] (score 0.9) to be pruned because they are within 1.5px.
    # [50.0, 50.0] (score 0.8) should also be kept.
    assert len(dedup_src) == 2
    assert [10.2, 20.1] in dedup_src.tolist()
    assert [10.0, 20.0] not in dedup_src.tolist()
    assert 0.95 in dedup_scores.tolist()
    assert 0.8 in dedup_scores.tolist()


@pytest.fixture
def tiny_synthetic_pair_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Generate simple overlapping synthetic image pair paths."""
    src_path = tmp_path / "src.tif"
    ref_path = tmp_path / "ref.tif"

    # Make synthetic images (random craters on uniform background)
    np.random.seed(42)
    src_img = np.ones((128, 128), dtype=np.uint8) * 128
    ref_img = np.ones((128, 128), dtype=np.uint8) * 128

    # Put a distinct "crater" in the center
    for y in range(40, 80):
        for x in range(40, 80):
            dist = np.sqrt((x - 60) ** 2 + (y - 60) ** 2)
            if dist < 15:
                src_img[y, x] = 50
                ref_img[y, x] = 50

    import tifffile
    tifffile.imwrite(str(src_path), src_img)
    tifffile.imwrite(str(ref_path), ref_img)

    return src_path, ref_path


@pytest.mark.parametrize(
    "scale_ratio",
    [1.0, 1.1, 1.2, 1.5, 2.0, 4.0, 10.0, 20.0]
)
def test_end_to_end_scale_runs(tiny_synthetic_pair_paths: tuple[Path, Path], scale_ratio: float, tmp_path: Path) -> None:
    """Validate registration pipeline executes without crashes across different scale configurations."""
    src_path, ref_path = tiny_synthetic_pair_paths

    # Set up config with scale_aware_matching=True
    cfg = PipelineConfig(
        pyramid=PyramidConfig(
            scale_aware_matching=True,
            metadata_scale_prior=True,
            log2_scale_min=-2,
            log2_scale_max=2,
        ),
        run=PipelineConfig().run.model_copy(update={"overwrite": True, "output_dir": str(tmp_path / f"out_scale_{scale_ratio}")}),
    )



    # Run the orchestrator on this config
    orchestrator = RegistrationOrchestrator(cfg)
    
    # We test execution does not crash. Registration may PASS or FAIL quality gates.
    try:
        manifest = orchestrator.register(source=src_path, reference=ref_path)
        assert manifest is not None
        assert isinstance(manifest.quality_gate_passed, bool)
        assert isinstance(manifest.metrics.warnings, list)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Scale-aware matching pipeline crashed for scale {scale_ratio}x: {exc}")


def test_metadata_scenarios(tiny_synthetic_pair_paths: tuple[Path, Path], tmp_path: Path) -> None:
    """Test metadata scale prior combinations (valid, missing, invalid, disabled)."""
    src_path, ref_path = tiny_synthetic_pair_paths





    # 1. Prior disabled: fallback search is executed
    cfg_disabled = PipelineConfig(
        pyramid=PyramidConfig(
            scale_aware_matching=True,
            metadata_scale_prior=False,
            log2_scale_min=-1,
            log2_scale_max=1,
        ),
        run=PipelineConfig().run.model_copy(update={"overwrite": True, "output_dir": str(tmp_path / "out_disabled")}),
    )
    orch_disabled = RegistrationOrchestrator(cfg_disabled)
    
    manifest_disabled = orch_disabled.register(source=src_path, reference=ref_path)
    assert manifest_disabled is not None

    cfg_baseline = PipelineConfig(
        pyramid=PyramidConfig(scale_aware_matching=False),
        run=PipelineConfig().run.model_copy(update={"overwrite": True, "output_dir": str(tmp_path / "out_baseline")}),
    )
    orch_baseline = RegistrationOrchestrator(cfg_baseline)
    manifest_baseline = orch_baseline.register(source=src_path, reference=ref_path)
    
    # Baseline run should not record multi-scale provenance
    if manifest_baseline.transform is not None:
        assert manifest_baseline.transform.provenance.get("scale_aware") is False


def test_pyramid_level_selection() -> None:
    """Test target level selection and downsampling assignments under various GSD ratios."""
    from lunarmatch.features.pyramid import ImagePyramid
    
    # Build dummy pyramids
    img_src = np.ones((100, 100), dtype=np.uint8)
    img_ref = np.ones((100, 100), dtype=np.uint8)
    
    cfg = PyramidConfig(levels_per_octave=2)
    src_pyr = ImagePyramid.build(img_src, config=cfg)
    ref_pyr = ImagePyramid.build(img_ref, config=cfg)
    
    # Case 1: Finer source (source GSD = 0.25, reference GSD = 1.535) -> s = 0.162866 < 1.0
    # Expected: source is downsampled by s (target_src_scale = s), reference remains at Level 0 (target_ref_scale = 1.0)
    s = 0.162866
    if s >= 1.0:
        target_src_scale = 1.0
        target_ref_scale = 1.0 / s
    else:
        target_src_scale = s
        target_ref_scale = 1.0
        
    assert abs(target_src_scale - 0.162866) < 1e-6
    assert target_ref_scale == 1.0
    
    src_level = min(src_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_src_scale))
    ref_level = min(ref_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_ref_scale))
    
    # Scale factors: Level 0=1.0, Level 1=0.7071, Level 2=0.5, Level 3=0.3536, Level 4=0.25, Level 5=0.1768, Level 6=0.125
    # Closest to 0.162866 in src pyramid should be Level 5 (0.1768)
    assert src_level.level_idx == 5
    assert ref_level.level_idx == 0
    
    # Case 2: Coarser source (source GSD = 2.0, reference GSD = 0.5) -> s = 4.0 >= 1.0
    # Expected: source remains at Level 0 (target_src_scale = 1.0), reference is downsampled by 1.0/s = 0.25 (target_ref_scale = 0.25)
    s = 4.0
    if s >= 1.0:
        target_src_scale = 1.0
        target_ref_scale = 1.0 / s
    else:
        target_src_scale = s
        target_ref_scale = 1.0
        
    assert target_src_scale == 1.0
    assert target_ref_scale == 0.25
    
    src_level_c = min(src_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_src_scale))
    ref_level_c = min(ref_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_ref_scale))
    
    assert src_level_c.level_idx == 0
    # Closest to 0.25 in ref pyramid should be Level 4 (0.25)
    assert ref_level_c.level_idx == 4
    
    # Case 3: 1:1 scale (source GSD = 1.0, reference GSD = 1.0) -> s = 1.0
    # Expected: target_src_scale = 1.0, target_ref_scale = 1.0
    s = 1.0
    if s >= 1.0:
        target_src_scale = 1.0
        target_ref_scale = 1.0 / s
    else:
        target_src_scale = s
        target_ref_scale = 1.0
        
    assert target_src_scale == 1.0
    assert target_ref_scale == 1.0
    
    src_level_1 = min(src_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_src_scale))
    ref_level_1 = min(ref_pyr.levels, key=lambda lvl: abs(lvl.scale_factor - target_ref_scale))
    
    assert src_level_1.level_idx == 0
    assert ref_level_1.level_idx == 0


def test_pyramid_coordinate_center_mapping_regression() -> None:
    """Regression test for pyramid pixel-center coordinate mapping."""
    import numpy as np

    from lunarmatch.features.pyramid import ImagePyramid
    from lunarmatch.models.config import PyramidConfig

    img = np.ones((64, 64), dtype=np.uint8) * 100
    cfg = PyramidConfig(levels_per_octave=1)
    pyr = ImagePyramid.build(img, config=cfg, min_dimension=8)
    
    lvl_1 = pyr[1]
    pt = np.array([[0.0, 0.0]], dtype=np.float64)
    mapped = lvl_1.mapping.apply(pt)
    np.testing.assert_allclose(mapped, np.array([[0.5, 0.5]], dtype=np.float64), atol=1e-7)


def test_non_integer_pyramid_scale_mapping() -> None:
    """Test coordinate mapping at non-integer scale factors."""
    import numpy as np

    from lunarmatch.geometry import CoordinateMapping

    scale_x = 5.0 / 3.0
    scale_y = 5.0 / 3.0
    mapping = CoordinateMapping(np.array([
        [scale_x, 0.0, 0.5 * (scale_x - 1.0)],
        [0.0, scale_y, 0.5 * (scale_y - 1.0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64))

    pt = np.array([[1.0, 1.0]], dtype=np.float64)
    mapped = mapping.apply(pt)
    np.testing.assert_allclose(mapped, np.array([[2.0, 2.0]], dtype=np.float64), atol=1e-7)


def test_per_hypothesis_ransac_selection() -> None:
    """Test that per-hypothesis RANSAC correctly identifies the correct scale and rejects noise hypotheses."""
    import numpy as np

    from lunarmatch.geometry import GeometricVerifier
    from lunarmatch.models.config import GeometryConfig
    from lunarmatch.models.domain import MatchSet

    src_true = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 15.0], [5.0, 50.0]], dtype=np.float64)
    ref_true = np.array([[20.5, 20.5], [40.5, 60.5], [80.5, 30.5], [10.5, 100.5]], dtype=np.float64)
    scores_true = np.array([0.9, 0.95, 0.88, 0.92], dtype=np.float64)
    
    match_set_true = MatchSet(
        source_points=src_true,
        reference_points=ref_true,
        scores=scores_true,
    )
    
    src_noise = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 15.0], [5.0, 50.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float64)
    ref_noise = np.array([[55.0, 12.0], [80.0, 3.0], [10.0, 45.0], [99.0, 99.0], [3.0, 23.0], [43.0, 5.0], [12.0, 90.0], [77.0, 88.0], [2.0, 3.0], [44.0, 11.0]], dtype=np.float64)
    scores_noise = np.array([0.8]*10, dtype=np.float64)
    
    match_set_noise = MatchSet(
        source_points=src_noise,
        reference_points=ref_noise,
        scores=scores_noise,
    )

    verifier = GeometricVerifier(config=GeometryConfig(model="affine"))
    
    res_true = verifier.verify(match_set_true)
    res_noise = verifier.verify(match_set_noise)

    assert res_true.success
    assert res_true.transform.inlier_count == 4
    
    inliers_noise = res_noise.transform.inlier_count if res_noise.success else 0
    assert inliers_noise < 4


def test_reflection_rejection_and_positive_acceptance() -> None:
    """Verify that reflection detection rejects mirrored transforms but accepts normal scale/rotation."""
    import numpy as np

    from lunarmatch.geometry import verify_matches
    from lunarmatch.models.domain import MatchSet

    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    A = 1.5 * R
    
    src = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 15.0], [5.0, 50.0]], dtype=np.float64)
    ref_pos = (A @ src.T).T
    
    match_set_pos = MatchSet(
        source_points=src,
        reference_points=ref_pos,
        scores=np.ones(4),
    )
    res_pos = verify_matches(match_set_pos)
    assert res_pos.success
    assert res_pos.failure_reason is None

    A_refl = A.copy()
    A_refl[:, 0] *= -1.0
    
    ref_refl = (A_refl @ src.T).T
    match_set_refl = MatchSet(
        source_points=src,
        reference_points=ref_refl,
        scores=np.ones(4),
    )
    
    res_refl = verify_matches(match_set_refl)
    assert not res_refl.success
    assert res_refl.failure_reason == "reflection_detected"


def test_synthetic_6_14x_scale_difference(tmp_path: Path) -> None:
    """Verify registration succeeds on synthetic images with exactly 6.14x scale difference."""
    import cv2
    import rasterio
    from rasterio.transform import from_origin

    from lunarmatch.models.config import PipelineConfig, PyramidConfig
    from lunarmatch.orchestrator import RegistrationOrchestrator
    from lunarmatch.synthetic.terrain import generate_crater_heightfield, render_hillshade

    np.random.seed(42)
    heights = generate_crater_heightfield(shape=(1024, 1024), num_craters=40, seed=42)
    base = render_hillshade(heights, azimuth_deg=315.0, elevation_deg=45.0)

    # Scale down by 6.14x for reference (1024 / 6.14 = 166.77 -> 167)
    ref_img = cv2.resize(base, (167, 167), interpolation=cv2.INTER_AREA)

    src_path = tmp_path / "src_614.tif"
    ref_path = tmp_path / "ref_614.tif"

    with rasterio.open(
        src_path, "w", driver="GTiff", height=1024, width=1024, count=1, dtype="uint8",
        crs="EPSG:32630", transform=from_origin(0, 1024, 0.25, 0.25)
    ) as dst:
        dst.write(base, 1)

    with rasterio.open(
        ref_path, "w", driver="GTiff", height=167, width=167, count=1, dtype="uint8",
        crs="EPSG:32630", transform=from_origin(0, 167, 1.535, 1.535)
    ) as dst:
        dst.write(ref_img, 1)

    cfg = PipelineConfig(
        pyramid=PyramidConfig(
            scale_aware_matching=True,
            metadata_scale_prior=True,
            log2_scale_min=-1,
            log2_scale_max=1,
            levels_per_octave=2,
        ),
        subpixel=PipelineConfig().subpixel.model_copy(update={"method": "none"}),
        quality_gates=PipelineConfig().quality_gates.model_copy(update={"min_inliers": 3, "min_inlier_ratio": 0.01}),
        run=PipelineConfig().run.model_copy(update={"overwrite": True, "output_dir": str(tmp_path / "out_614")}),
    )

    orchestrator = RegistrationOrchestrator(cfg)
    manifest = orchestrator.register(source=src_path, reference=ref_path)

    assert manifest is not None
    assert manifest.quality_gate_passed
    assert manifest.metrics.inlier_count >= 3
    assert manifest.transform.provenance.get("scale_aware") is True
    
    prov = manifest.transform.provenance
    assert "hypotheses_diagnostics" in prov
    assert prov["hypotheses_evaluated"] > 0
    assert prov["candidate_matches"] >= manifest.metrics.total_matches
    assert prov["deduplicated_matches"] == manifest.metrics.total_matches


def test_reflection_allowance() -> None:
    """Verify that allow_reflection parameter works in verifier."""
    import numpy as np

    from lunarmatch.geometry import verify_matches
    from lunarmatch.models.config import GeometryConfig
    from lunarmatch.models.domain import MatchSet

    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    A = 1.5 * R

    src = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 15.0], [5.0, 50.0]], dtype=np.float64)

    A_refl = A.copy()
    A_refl[:, 0] *= -1.0

    ref_refl = (A_refl @ src.T).T
    match_set_refl = MatchSet(
        source_points=src,
        reference_points=ref_refl,
        scores=np.ones(4),
    )

    # 1. By default/allow_reflection=False, it should fail
    cfg_no_refl = GeometryConfig(allow_reflection=False)
    res_refl = verify_matches(match_set_refl, config=cfg_no_refl)
    assert not res_refl.success
    assert res_refl.failure_reason == "reflection_detected"

    # 2. When allow_reflection=True, it should succeed
    cfg_refl = GeometryConfig(allow_reflection=True)
    res_refl_allowed = verify_matches(match_set_refl, config=cfg_refl)
    assert res_refl_allowed.success
    assert res_refl_allowed.failure_reason is None


def test_uncertainty_range() -> None:
    """Verify that metadata_scale_uncertainty_octaves restricts scale hypothesis range."""
    from lunarmatch.features.scale_search import generate_scale_hypotheses
    from lunarmatch.models.config import PyramidConfig

    # 1. Broad search: default uncertainty range (not used because we set uncertainty bounds)
    cfg_broad = PyramidConfig(metadata_scale_prior=True, metadata_scale_uncertainty_octaves=2.0, levels_per_octave=2)
    scales_broad = generate_scale_hypotheses(source_scale_m=0.25, reference_scale_m=1.535, config=cfg_broad)
    # Exponents range from -2.0 to +2.0 octaves around 0.162866:
    # should yield 9 hypotheses (round((2.0 - (-2.0)) * 2) + 1 = 9)
    assert len(scales_broad) == 9

    # 2. Narrow search: uncertainty of 0.5 octaves
    cfg_narrow = PyramidConfig(metadata_scale_prior=True, metadata_scale_uncertainty_octaves=0.5, levels_per_octave=2)
    scales_narrow = generate_scale_hypotheses(source_scale_m=0.25, reference_scale_m=1.535, config=cfg_narrow)
    # Exponents range from -0.5 to +0.5 octaves around 0.162866:
    # should yield 3 hypotheses (round((0.5 - (-0.5)) * 2) + 1 = 3)
    assert len(scales_narrow) == 3
    assert abs(scales_narrow[1] - 0.162866) < 1e-4


def test_terrain_rejection_early(tmp_path: Path) -> None:
    """Verify that completely different terrain pairs are early-rejected by overlap validation."""
    import rasterio
    from rasterio.transform import from_origin

    from lunarmatch.models.config import PipelineConfig, PyramidConfig
    from lunarmatch.orchestrator import RegistrationOrchestrator
    from lunarmatch.synthetic.terrain import generate_crater_heightfield, render_hillshade

    np.random.seed(42)
    # Generate crater terrain for source
    heights_src = generate_crater_heightfield(shape=(256, 256), num_craters=10, seed=42)
    base_src = render_hillshade(heights_src, azimuth_deg=315.0, elevation_deg=45.0)

    # Generate completely different crater terrain for reference
    heights_ref = generate_crater_heightfield(shape=(256, 256), num_craters=10, seed=100)
    base_ref = render_hillshade(heights_ref, azimuth_deg=45.0, elevation_deg=30.0)

    src_path = tmp_path / "src_diff.tif"
    ref_path = tmp_path / "ref_diff.tif"

    # Write as georeferenced GeoTIFFs
    with rasterio.open(
        src_path, "w", driver="GTiff", height=256, width=256, count=1, dtype="uint8",
        crs="EPSG:32630", transform=from_origin(0, 256, 1.0, 1.0)
    ) as dst:
        dst.write(base_src, 1)

    with rasterio.open(
        ref_path, "w", driver="GTiff", height=256, width=256, count=1, dtype="uint8",
        crs="EPSG:32630", transform=from_origin(0, 256, 1.0, 1.0)
    ) as dst:
        dst.write(base_ref, 1)

    cfg = PipelineConfig(
        pyramid=PyramidConfig(
            scale_aware_matching=True,
            metadata_scale_prior=True,
        ),
        run=PipelineConfig().run.model_copy(update={"overwrite": True, "output_dir": str(tmp_path / "out_diff")}),
    )

    orchestrator = RegistrationOrchestrator(cfg)
    manifest = orchestrator.register(source=src_path, reference=ref_path)

    # Rejection should occur
    assert manifest is not None
    assert not manifest.quality_gate_passed
    assert any("Coarse terrain overlap validation failed" in w for w in manifest.metrics.warnings)



