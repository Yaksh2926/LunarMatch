"""Unit tests for descriptor matching, distance metrics, ratio test, mutual check, and chunking."""
import numpy as np

from lunarmatch.matching import (
    compute_descriptor_distances,
    match_descriptors,
)
from lunarmatch.models.domain import KeypointSet, MatchSet


def test_distance_metrics_uint8_and_float():
    """Test Hamming distance for uint8 and L2 distance for float32."""
    # Binary descriptors (uint8)
    d1 = np.array([[0b11110000, 0b00001111]], dtype=np.uint8)  # 8 bits set
    d2 = np.array([[0b11110000, 0b00000000]], dtype=np.uint8)  # 4 bits set, diff = 4 bits

    dist_hamming = compute_descriptor_distances(d1, d2, metric="hamming")
    assert dist_hamming.shape == (1, 1)
    assert dist_hamming[0, 0] == 4.0

    # Float descriptors (float32)
    f1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    f2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    dist_l2 = compute_descriptor_distances(f1, f2, metric="l2")
    assert dist_l2.shape == (1, 1)
    np.testing.assert_allclose(dist_l2[0, 0], np.sqrt(2.0))


def test_exact_constructed_matches_float():
    """Test exact matching on constructed float descriptors."""
    # 5 source descriptors
    rng = np.random.default_rng(42)
    src_desc = rng.normal(size=(5, 128)).astype(np.float32)
    src_pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float64)

    # Reference descriptors: permute rows and add a disturber
    ref_indices_gt = [3, 0, 4, 1, 2]
    ref_desc = src_desc[ref_indices_gt].copy()
    ref_pts = np.array([[100, 100], [200, 200], [300, 300], [400, 400], [500, 500]], dtype=np.float64)

    # Add 2 distractors to ref
    distractors = rng.normal(size=(2, 128)).astype(np.float32) * 10.0
    ref_desc = np.vstack([ref_desc, distractors])
    ref_pts = np.vstack([ref_pts, np.array([[600, 600], [700, 700]], dtype=np.float64)])

    src_kps = KeypointSet(coordinates=src_pts, descriptors=src_desc)
    ref_kps = KeypointSet(coordinates=ref_pts, descriptors=ref_desc)

    # Ground truth mapping: source index i maps to reference index ref_target_indices[i]
    ref_target_indices = [1, 3, 4, 0, 2]

    res = match_descriptors(src_kps, ref_kps, ratio_threshold=0.80, mutual=True)
    assert len(res.match_set) == 5
    assert np.array_equal(res.source_indices, np.arange(5))
    assert np.array_equal(res.reference_indices, ref_target_indices)
    np.testing.assert_allclose(res.distances, 0.0, atol=1e-5)


def test_exact_constructed_matches_binary():
    """Test exact matching on constructed binary uint8 descriptors."""
    rng = np.random.default_rng(123)
    src_desc = rng.integers(0, 256, size=(4, 32), dtype=np.uint8)
    src_pts = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=np.float64)

    ref_indices_gt = [2, 3, 0, 1]
    ref_desc = src_desc[ref_indices_gt].copy()
    ref_pts = np.array([[5, 5], [15, 15], [25, 25], [35, 35]], dtype=np.float64)

    src_kps = KeypointSet(coordinates=src_pts, descriptors=src_desc)
    ref_kps = KeypointSet(coordinates=ref_pts, descriptors=ref_desc)

    res = match_descriptors(src_kps, ref_kps, ratio_threshold=0.80, mutual=True)
    assert len(res.match_set) == 4
    assert np.array_equal(res.source_indices, np.arange(4))
    assert np.array_equal(res.reference_indices, ref_indices_gt)
    np.testing.assert_allclose(res.distances, 0.0)


def test_knn_ratio_test_and_ties():
    """Test ratio test filtering and distance tie rejection."""
    # Source descriptor
    src_desc = np.array([[0.0, 0.0]], dtype=np.float32)

    # Reference descriptors:
    # Row 0: dist = 1.0
    # Row 1: dist = 2.0 (ratio = 0.5 -> passes ratio 0.8)
    # Row 2: dist = 1.0 (equidistant to Row 0 -> d1=1.0, d2=1.0, ratio=1.0 -> fails ratio 0.8)
    ref_desc_pass = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    ref_desc_tie = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)

    # Test ratio pass
    res_pass = match_descriptors(src_desc, ref_desc_pass, ratio_threshold=0.80, mutual=False)
    assert len(res_pass.match_set) == 1
    assert res_pass.reference_indices[0] == 0
    np.testing.assert_allclose(res_pass.ratios[0], 0.5)

    # Test ratio tie fail
    res_tie = match_descriptors(src_desc, ref_desc_tie, ratio_threshold=0.80, mutual=False)
    assert len(res_tie.match_set) == 0


def test_fewer_than_two_neighbors():
    """Test N1=0, N2=0, and N2=1 edge cases."""
    src = np.ones((3, 32), dtype=np.uint8)
    ref_empty = np.empty((0, 32), dtype=np.uint8)
    ref_one = np.ones((1, 32), dtype=np.uint8)

    # N2 = 0
    res_zero = match_descriptors(src, ref_empty)
    assert isinstance(res_zero.match_set, MatchSet)
    assert len(res_zero.match_set) == 0

    # N2 = 1 with ratio test enabled -> ratio test fails (no 2nd neighbor)
    res_one_ratio = match_descriptors(src, ref_one, ratio_threshold=0.80)
    assert len(res_one_ratio.match_set) == 0

    # N2 = 1 with ratio test disabled -> matches
    res_one_noratio = match_descriptors(src, ref_one, ratio_threshold=None)
    assert len(res_one_noratio.match_set) == 1


def test_mutual_nearest_neighbor_check():
    """Test mutual nearest neighbor consistency check."""
    # src0 is closer to ref0 (dist=1) than ref1 (dist=10)
    # src1 is closer to ref0 (dist=0.1) than ref1 (dist=10)
    # Forward: src0 -> ref0, src1 -> ref0
    # Backward: ref0 -> src1 (dist=0.1 < dist=1.0)
    # Therefore, src0 -> ref0 is NOT mutual!
    src = np.array([[1.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    ref = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

    # With mutual check -> src0 -> ref0 rejected, src1 -> ref0 accepted
    res_mutual = match_descriptors(src, ref, ratio_threshold=0.85, mutual=True)
    assert len(res_mutual.match_set) == 1
    assert res_mutual.source_indices[0] == 1
    assert res_mutual.reference_indices[0] == 0

    # Without mutual check -> both forward pass ratio test, but duplicate removal keeps src1 -> ref0
    res_non_mutual = match_descriptors(src, ref, ratio_threshold=0.85, mutual=False)
    assert len(res_non_mutual.match_set) == 1
    assert res_non_mutual.source_indices[0] == 1


def test_duplicate_correspondence_removal():
    """Test that multiple source keypoints matching same reference keypoint are deduplicated."""
    # src0 dist to ref0 = 2.0 (ratio with ref1 dist 10.0 = 0.2)
    # src1 dist to ref0 = 1.0 (ratio with ref1 dist 10.0 = 0.1)
    # Both src0 and src1 match ref0.
    src = np.array([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    ref = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

    res = match_descriptors(src, ref, ratio_threshold=0.80, mutual=False)
    assert len(res.match_set) == 1
    # src1 is closer (dist 1.0 < dist 2.0), so src1 -> ref0 is retained
    assert res.source_indices[0] == 1
    assert res.reference_indices[0] == 0


def test_chunked_matching_equivalence():
    """Test chunked matching equivalence against single-chunk matching."""
    rng = np.random.default_rng(99)
    src_desc = rng.normal(size=(50, 64)).astype(np.float32)
    ref_desc = rng.normal(size=(60, 64)).astype(np.float32)

    # Force some exact matches
    ref_desc[:10] = src_desc[:10]

    res_large_chunk = match_descriptors(src_desc, ref_desc, ratio_threshold=0.80, chunk_size=100)
    res_small_chunk = match_descriptors(src_desc, ref_desc, ratio_threshold=0.80, chunk_size=7)

    assert len(res_large_chunk.match_set) == len(res_small_chunk.match_set)
    assert np.array_equal(res_large_chunk.source_indices, res_small_chunk.source_indices)
    assert np.array_equal(res_large_chunk.reference_indices, res_small_chunk.reference_indices)
    np.testing.assert_allclose(res_large_chunk.distances, res_small_chunk.distances)


def test_scale_provenance_preservation():
    """Test that keypoint scale levels and scores are preserved in MatchResult."""
    src_kps = KeypointSet(
        coordinates=np.array([[10, 10], [20, 20]], dtype=np.float64),
        scales=np.array([1.5, 2.5], dtype=np.float32),
        descriptors=np.array([[1, 0], [0, 1]], dtype=np.float32),
    )
    ref_kps = KeypointSet(
        coordinates=np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float64),
        scales=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        descriptors=np.array([[1, 0], [0, 1], [5, 5]], dtype=np.float32),
    )

    res = match_descriptors(src_kps, ref_kps, ratio_threshold=0.80)
    assert len(res.match_set) == 2
    np.testing.assert_array_equal(res.source_scales, np.array([1.5, 2.5], dtype=np.float32))
    np.testing.assert_array_equal(res.reference_scales, np.array([10.0, 20.0], dtype=np.float32))
    assert res.match_set.scores.shape == (2,)
