"""Descriptor matcher supporting kNN ratio test, mutual checks, chunking, and provenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lunarmatch.matching.knn import compute_descriptor_distances
from lunarmatch.models.config import MatchingConfig
from lunarmatch.models.domain import KeypointSet, MatchSet


@dataclass
class MatchResult:
    """Detailed result of descriptor matching including indices and scale provenance."""

    match_set: MatchSet
    source_indices: np.ndarray  # (M,) int64 indices into source keypoints
    reference_indices: np.ndarray  # (M,) int64 indices into reference keypoints
    distances: np.ndarray  # (M,) float64 nearest neighbor distances
    ratios: np.ndarray  # (M,) float64 distance ratios (d1 / d2)
    source_scales: np.ndarray  # (M,) float32 source keypoint scales
    reference_scales: np.ndarray  # (M,) float32 reference keypoint scales

    def to_dict(self) -> dict[str, Any]:
        """Serialize match result to dictionary."""
        return {
            "match_set": self.match_set.to_dict(),
            "source_indices": self.source_indices.tolist(),
            "reference_indices": self.reference_indices.tolist(),
            "distances": self.distances.tolist(),
            "ratios": self.ratios.tolist(),
            "source_scales": self.source_scales.tolist(),
            "reference_scales": self.reference_scales.tolist(),
        }


class DescriptorMatcher:
    """Descriptor matching engine with Lowe's ratio test, mutual cross-check, and chunking."""

    def __init__(
        self,
        config: MatchingConfig | None = None,
        ratio_threshold: float | None = 0.80,
        mutual: bool = True,
        max_descriptor_distance: float | None = None,
        chunk_size: int = 2000,
        metric: str | None = None,
    ) -> None:
        if config is not None:
            self._ratio_threshold: float | None = config.ratio_threshold
            self._mutual = config.mutual
            self._max_distance = config.max_descriptor_distance
        else:
            self._ratio_threshold = ratio_threshold
            self._mutual = mutual
            self._max_distance = max_descriptor_distance

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self._chunk_size = chunk_size
        self._metric = metric

    def match(
        self,
        source: KeypointSet | np.ndarray,
        reference: KeypointSet | np.ndarray,
        source_points: np.ndarray | None = None,
        reference_points: np.ndarray | None = None,
    ) -> MatchResult:
        """Match descriptors between source and reference keypoints.

        Args:
            source: Source KeypointSet or 2D descriptor array (N1, D).
            reference: Reference KeypointSet or 2D descriptor array (N2, D).
            source_points: Nx2 float64 coordinates if source is array.
            reference_points: Nx2 float64 coordinates if reference is array.

        Returns:
            MatchResult containing MatchSet and detailed scale/index provenance.
        """
        # Extract descriptors, points, and scales
        if isinstance(source, KeypointSet):
            desc_src = source.descriptors
            pts_src = source.coordinates
            scales_src = source.scales
        else:
            desc_src = source
            pts_src = source_points if source_points is not None else np.zeros((len(source), 2), dtype=np.float64)
            scales_src = np.ones(len(source), dtype=np.float32)

        if isinstance(reference, KeypointSet):
            desc_ref = reference.descriptors
            pts_ref = reference.coordinates
            scales_ref = reference.scales
        else:
            desc_ref = reference
            pts_ref = reference_points if reference_points is not None else np.zeros((len(reference), 2), dtype=np.float64)
            scales_ref = np.ones(len(reference), dtype=np.float32)

        # Handle empty inputs
        if desc_src is None or desc_ref is None or len(desc_src) == 0 or len(desc_ref) == 0:
            empty_ms = MatchSet(
                source_points=np.empty((0, 2), dtype=np.float64),
                reference_points=np.empty((0, 2), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
            )
            return MatchResult(
                match_set=empty_ms,
                source_indices=np.empty((0,), dtype=np.int64),
                reference_indices=np.empty((0,), dtype=np.int64),
                distances=np.empty((0,), dtype=np.float64),
                ratios=np.empty((0,), dtype=np.float64),
                source_scales=np.empty((0,), dtype=np.float32),
                reference_scales=np.empty((0,), dtype=np.float32),
            )

        n1 = len(desc_src)
        n2 = len(desc_ref)

        # 1. Forward kNN matching (source -> reference) in chunks
        candidate_matches: list[tuple[int, int, float, float]] = []  # (src_idx, ref_idx, d1, ratio)

        for i_start in range(0, n1, self._chunk_size):
            i_end = min(i_start + self._chunk_size, n1)
            chunk_src = desc_src[i_start:i_end]

            dists = compute_descriptor_distances(chunk_src, desc_ref, metric=self._metric)  # (B, N2)

            for local_i in range(len(chunk_src)):
                global_i = i_start + local_i
                row_dists = dists[local_i]

                if n2 == 1:
                    d1 = float(row_dists[0])
                    j1 = 0
                    ratio = 1.0
                    # If ratio test is required, cannot compute ratio with 1 neighbor
                    if self._ratio_threshold is not None:
                        continue
                    if self._max_distance is not None and d1 > self._max_distance:
                        continue
                    candidate_matches.append((global_i, j1, d1, ratio))
                else:
                    # Find top 2 smallest distances
                    top2_idx = np.argpartition(row_dists, 1)[:2]
                    # Sort top 2 to ensure d1 <= d2
                    if row_dists[top2_idx[0]] > row_dists[top2_idx[1]]:
                        top2_idx = top2_idx[::-1]

                    j1, j2 = int(top2_idx[0]), int(top2_idx[1])
                    d1, d2 = float(row_dists[j1]), float(row_dists[j2])

                    ratio = d1 / d2 if d2 > 0 else 1.0

                    # Apply ratio test
                    if self._ratio_threshold is not None and d1 >= self._ratio_threshold * d2:
                        continue

                    # Apply max distance threshold
                    if self._max_distance is not None and d1 > self._max_distance:
                        continue

                    candidate_matches.append((global_i, j1, d1, ratio))

        if not candidate_matches:
            empty_ms = MatchSet(
                source_points=np.empty((0, 2), dtype=np.float64),
                reference_points=np.empty((0, 2), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
            )
            return MatchResult(
                match_set=empty_ms,
                source_indices=np.empty((0,), dtype=np.int64),
                reference_indices=np.empty((0,), dtype=np.int64),
                distances=np.empty((0,), dtype=np.float64),
                ratios=np.empty((0,), dtype=np.float64),
                source_scales=np.empty((0,), dtype=np.float32),
                reference_scales=np.empty((0,), dtype=np.float32),
            )

        # 2. Backward nearest-neighbor check if mutual=True
        if self._mutual:
            # Find 1st nearest neighbor of each reference descriptor in source descriptors
            ref_nn_in_src = np.empty(n2, dtype=np.int64)
            for j_start in range(0, n2, self._chunk_size):
                j_end = min(j_start + self._chunk_size, n2)
                chunk_ref = desc_ref[j_start:j_end]
                rev_dists = compute_descriptor_distances(chunk_ref, desc_src, metric=self._metric)  # (B_ref, N1)
                best_src = np.argmin(rev_dists, axis=1)
                ref_nn_in_src[j_start:j_end] = best_src

            # Filter candidates for mutual consistency
            candidate_matches = [
                m for m in candidate_matches if ref_nn_in_src[m[1]] == m[0]
            ]

        if not candidate_matches:
            empty_ms = MatchSet(
                source_points=np.empty((0, 2), dtype=np.float64),
                reference_points=np.empty((0, 2), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
            )
            return MatchResult(
                match_set=empty_ms,
                source_indices=np.empty((0,), dtype=np.int64),
                reference_indices=np.empty((0,), dtype=np.int64),
                distances=np.empty((0,), dtype=np.float64),
                ratios=np.empty((0,), dtype=np.float64),
                source_scales=np.empty((0,), dtype=np.float32),
                reference_scales=np.empty((0,), dtype=np.float32),
            )

        # 3. Duplicate correspondence removal (resolve multiple source keypoints matching same reference keypoint)
        # Sort candidates by distance d1 ascending, then by src_idx ascending for strict determinism
        candidate_matches.sort(key=lambda x: (x[2], x[0]))

        seen_ref: set[int] = set()
        seen_src: set[int] = set()
        filtered_matches: list[tuple[int, int, float, float]] = []

        for src_i, ref_j, d1, ratio in candidate_matches:
            if ref_j in seen_ref or src_i in seen_src:
                continue
            seen_ref.add(ref_j)
            seen_src.add(src_i)
            filtered_matches.append((src_i, ref_j, d1, ratio))

        if not filtered_matches:
            empty_ms = MatchSet(
                source_points=np.empty((0, 2), dtype=np.float64),
                reference_points=np.empty((0, 2), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
            )
            return MatchResult(
                match_set=empty_ms,
                source_indices=np.empty((0,), dtype=np.int64),
                reference_indices=np.empty((0,), dtype=np.int64),
                distances=np.empty((0,), dtype=np.float64),
                ratios=np.empty((0,), dtype=np.float64),
                source_scales=np.empty((0,), dtype=np.float32),
                reference_scales=np.empty((0,), dtype=np.float32),
            )

        src_indices = np.array([m[0] for m in filtered_matches], dtype=np.int64)
        ref_indices = np.array([m[1] for m in filtered_matches], dtype=np.int64)
        dists_arr = np.array([m[2] for m in filtered_matches], dtype=np.float64)
        ratios_arr = np.array([m[3] for m in filtered_matches], dtype=np.float64)

        src_pts_matched = pts_src[src_indices]
        ref_pts_matched = pts_ref[ref_indices]
        src_scales_matched = scales_src[src_indices]
        ref_scales_matched = scales_ref[ref_indices]

        # Compute match scores: 1.0 - ratio if ratio test active, else 1.0 / (1.0 + distance)
        if self._ratio_threshold is not None:
            scores = np.clip(1.0 - ratios_arr, 0.0, 1.0)
        else:
            scores = 1.0 / (1.0 + dists_arr)

        match_set = MatchSet(
            source_points=src_pts_matched,
            reference_points=ref_pts_matched,
            scores=scores,
        )

        return MatchResult(
            match_set=match_set,
            source_indices=src_indices,
            reference_indices=ref_indices,
            distances=dists_arr,
            ratios=ratios_arr,
            source_scales=src_scales_matched,
            reference_scales=ref_scales_matched,
        )


def match_descriptors(
    source: KeypointSet | np.ndarray,
    reference: KeypointSet | np.ndarray,
    config: MatchingConfig | None = None,
    ratio_threshold: float | None = 0.80,
    mutual: bool = True,
    max_descriptor_distance: float | None = None,
    chunk_size: int = 2000,
    metric: str | None = None,
) -> MatchResult:
    """Convenience function to match descriptors between source and reference keypoints.

    Returns MatchResult containing MatchSet and scale provenance.
    """
    matcher = DescriptorMatcher(
        config=config,
        ratio_threshold=ratio_threshold,
        mutual=mutual,
        max_descriptor_distance=max_descriptor_distance,
        chunk_size=chunk_size,
        metric=metric,
    )
    return matcher.match(source=source, reference=reference)
