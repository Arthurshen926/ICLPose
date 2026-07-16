"""Pose-relation features with immutable candidate and missing-mass semantics.

This module deliberately does not define a pose score.  It constructs a fixed
query graph and residual histograms that can subsequently be consumed by a
separately calibrated likelihood-ratio model.  This keeps the historical
Gaussian pair pseudolikelihood an audit baseline rather than silently turning
it into production evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from feature_extract.vfm.localization.candidate_pose_evidence import (
    candidate_pose_evidence,
)


RELATION_FEATURE_VERSION = "candidate_relation_residual_histogram_v4_topology_strength_repeat_geometry"
_REPEAT_ORIENTATIONS = ("horizontal", "nonhorizontal")
_REPEAT_QUERY_SCALES = ("qnear", "qmid", "qfar")
_REPEAT_XYZ_SCALES = ("xnear", "xmid", "xfar")
RELATION_CHANNELS = tuple(
    f"repeat_{orientation}_{query_scale}_{xyz_scale}"
    for orientation in _REPEAT_ORIENTATIONS
    for query_scale in _REPEAT_QUERY_SCALES
    for xyz_scale in _REPEAT_XYZ_SCALES
) + (
    "maplet_only",
    "support_overlap_low",
    "support_overlap_mid",
    "support_overlap_high",
    "directed_neighbor_rank_near",
    "directed_neighbor_rank_far",
    "mutual_neighbor_rank_near",
    "mutual_neighbor_rank_far",
)


def _bucket(values: np.ndarray, first: float, second: float) -> np.ndarray:
    return np.where(values < first, 0, np.where(values < second, 1, 2)).astype(
        np.int64
    )


@dataclass(frozen=True)
class RelationEdgeGraph:
    edges: np.ndarray
    sha256: str

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2).copy()
        if len(edges) and (
            np.any(edges < 0)
            or np.any(edges[:, 0] >= edges[:, 1])
            or len(np.unique(edges, axis=0)) != len(edges)
        ):
            raise ValueError("relation edges must be unique ordered row pairs")
        expected = relation_edge_graph_sha256(edges)
        if str(self.sha256) != expected:
            raise ValueError("relation edge graph hash is stale")
        edges.setflags(write=False)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True)
class CandidateModeMixture:
    xy: np.ndarray
    probabilities: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64).copy()
        probabilities = np.asarray(self.probabilities, dtype=np.float64).copy()
        valid = np.asarray(self.valid_mask, dtype=bool).copy()
        if xy.ndim != 4 or xy.shape[-1] != 2:
            raise ValueError("candidate modes must have shape [N,L,M,2]")
        if probabilities.shape != xy.shape[:-1] or valid.shape != probabilities.shape:
            raise ValueError("candidate mode arrays are not aligned")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("candidate mode probabilities must be finite and non-negative")
        mass = np.sum(np.where(valid, probabilities, 0.0), axis=2)
        candidate_valid = np.any(valid, axis=2)
        if np.any(np.abs(mass[candidate_valid] - 1.0) > 2e-6):
            raise ValueError("candidate mode probability mass must equal one")
        for value in (xy, probabilities, valid):
            value.setflags(write=False)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True)
class RelationResidualHistograms:
    histograms: np.ndarray
    candidate_pair_mass: np.ndarray
    null_touching_mass: np.ndarray
    bin_edges_px: np.ndarray
    edge_graph_sha256: str

    def __post_init__(self) -> None:
        hist = np.asarray(self.histograms, dtype=np.float64).copy()
        candidate = np.asarray(self.candidate_pair_mass, dtype=np.float64).reshape(-1).copy()
        null = np.asarray(self.null_touching_mass, dtype=np.float64).reshape(-1).copy()
        bins = np.asarray(self.bin_edges_px, dtype=np.float64).reshape(-1).copy()
        if hist.ndim != 3 or hist.shape[1] != len(RELATION_CHANNELS):
            raise ValueError("relation histograms must have shape [E,C,B]")
        if hist.shape[0] != len(candidate) or len(null) != len(candidate):
            raise ValueError("relation edge features are not aligned")
        if hist.shape[2] != len(bins) - 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0.0):
            raise ValueError("relation histogram bins are invalid")
        if np.any(~np.isfinite(hist)) or np.any(hist < 0.0):
            raise ValueError("relation histograms must be finite and non-negative")
        if np.any(candidate < 0.0) or np.any(null < 0.0):
            raise ValueError("relation probability mass must be non-negative")
        histogram_mass = np.sum(hist, axis=(1, 2))
        if np.any(np.abs(histogram_mass - candidate) > 2e-5):
            raise ValueError("relation histogram and candidate-pair mass differ")
        if np.any(np.abs(candidate + null - 1.0) > 2e-5):
            raise ValueError("candidate-pair and null-touching mass must equal one")
        for value in (hist, candidate, null, bins):
            value.setflags(write=False)
        object.__setattr__(self, "histograms", hist)
        object.__setattr__(self, "candidate_pair_mass", candidate)
        object.__setattr__(self, "null_touching_mass", null)
        object.__setattr__(self, "bin_edges_px", bins)


def relation_edge_graph_sha256(edges: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(edges, dtype="<i8").reshape(-1, 2))
    digest = hashlib.sha256()
    digest.update(RELATION_FEATURE_VERSION.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()[:16]


def build_query_knn_relation_graph(xy: np.ndarray, *, neighbor_k: int) -> RelationEdgeGraph:
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    requested = int(neighbor_k)
    if requested < 0:
        raise ValueError("relation neighbor count must be non-negative")
    if len(points) < 2 or requested == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    else:
        k = min(requested, len(points) - 1)
        distance = np.sum(np.square(points[:, None] - points[None]), axis=2)
        np.fill_diagonal(distance, np.inf)
        nearest = np.argsort(distance, axis=1, kind="mergesort")[:, :k]
        source = np.repeat(np.arange(len(points), dtype=np.int64), k)
        target = nearest.reshape(-1)
        edges = np.unique(
            np.column_stack([np.minimum(source, target), np.maximum(source, target)]),
            axis=0,
        )
    return RelationEdgeGraph(edges, relation_edge_graph_sha256(edges))


def pose_independent_candidate_modes(pool: Any, *, max_modes: int = 5) -> CandidateModeMixture:
    """Build center plus RGB modes without conditioning them on a tested pose.

    Missing views, dustbin mass, and truncated offset modes all remain on the
    candidate's center mode.  They are never redistributed to another mode or
    another landmark identity.
    """

    limit = int(max_modes)
    if limit < 1:
        raise ValueError("max_modes must be positive")
    candidate_valid = np.asarray(pool.valid_mask, dtype=bool)
    n, l = candidate_valid.shape
    xy = np.full((n, l, limit, 2), np.nan, dtype=np.float64)
    probability = np.zeros((n, l, limit), dtype=np.float64)
    valid = np.zeros((n, l, limit), dtype=bool)
    center = np.asarray(pool.xy, dtype=np.float64).reshape(n, 2)
    xy[:, :, 0] = center[:, None, :]
    probability[:, :, 0] = candidate_valid.astype(np.float64)
    valid[:, :, 0] = candidate_valid
    spatial = pool.spatial_likelihood
    if spatial is None or limit == 1:
        return CandidateModeMixture(xy, probability, valid)

    offsets = np.asarray(spatial.offsets_xy, dtype=np.float64)
    for row, column in zip(*np.nonzero(candidate_valid)):
        views = np.flatnonzero(spatial.valid_mask[row, column])
        if len(views) == 0:
            continue
        offset_mass = np.zeros((len(offsets),), dtype=np.float64)
        accepted_mass = 0.0
        for view in views.tolist():
            view_mass = float(spatial.view_probabilities[row, column, view])
            reliable_mass = view_mass * (
                1.0 - float(spatial.dustbin_probabilities[row, column, view])
            )
            if reliable_mass <= 0.0:
                continue
            logits = np.asarray(
                spatial.local_log_probabilities[row, column, view], dtype=np.float64
            )
            logits = logits - float(np.max(logits))
            local = np.exp(logits)
            local /= max(float(np.sum(local)), 1e-12)
            offset_mass += reliable_mass * local
            accepted_mass += reliable_mass
        accepted_mass = min(max(accepted_mass, 0.0), 1.0)
        keep_count = min(limit - 1, len(offsets))
        order = np.argsort(-offset_mass, kind="mergesort")[:keep_count]
        kept_mass = float(np.sum(offset_mass[order]))
        probability[row, column, 0] = 1.0 - kept_mass
        for local_index, offset_index in enumerate(order.tolist(), start=1):
            mode_mass = float(offset_mass[offset_index])
            if mode_mass <= 0.0:
                continue
            xy[row, column, local_index] = center[row] + offsets[offset_index]
            probability[row, column, local_index] = mode_mass
            valid[row, column, local_index] = True
    return CandidateModeMixture(xy, probability, valid)


def relation_residual_histograms(
    pool: Any,
    pose_w2c: np.ndarray,
    camera: Any,
    graph: RelationEdgeGraph,
    modes: CandidateModeMixture,
    *,
    bin_edges_px: np.ndarray,
    residual_sigma_px: float = 2.0,
    candidate_outlier_likelihood: float = 1e-3,
    null_likelihood: float = 1e-3,
) -> RelationResidualHistograms:
    """Marginalize top-L identities and local modes into fixed residual bins."""

    if not np.isclose(float(pool.identity_prior_temperature), 1.0):
        raise ValueError(
            "relation features currently require identity_prior_temperature=1 so "
            "their candidate simplex exactly matches descriptor_scores"
        )
    if not np.isclose(float(pool.geometry_prior_mix_weight), 0.0):
        raise ValueError(
            "relation features currently require geometry_prior_mix_weight=0 so "
            "their candidate simplex exactly matches descriptor_scores"
        )
    bins = np.asarray(bin_edges_px, dtype=np.float64).reshape(-1)
    if len(bins) < 2 or bins[0] != 0.0 or np.any(np.diff(bins) <= 0.0):
        raise ValueError("relation bins must be increasing and start at zero")
    evidence = candidate_pose_evidence(
        pool,
        pose_w2c,
        camera,
        residual_sigma_px=float(residual_sigma_px),
        outlier_likelihood=float(candidate_outlier_likelihood),
    )
    identity = np.where(pool.valid_mask, pool.descriptor_scores, 0.0).astype(np.float64)
    candidate_weight = identity * evidence.candidate_likelihoods
    null_weight = np.asarray(pool.null_scores, dtype=np.float64) * float(null_likelihood)
    denominator = np.sum(candidate_weight, axis=1) + null_weight
    candidate_posterior = candidate_weight / np.maximum(denominator[:, None], 1e-12)
    null_posterior = null_weight / np.maximum(denominator, 1e-12)

    edge_count = len(graph.edges)
    histogram = np.zeros(
        (edge_count, len(RELATION_CHANNELS), len(bins) - 1), dtype=np.float64
    )
    candidate_mass = np.zeros((edge_count,), dtype=np.float64)
    null_mass = np.ones((edge_count,), dtype=np.float64)
    maplets = np.asarray(pool.maplet_cluster_ids, dtype=np.int64)
    for edge_index, (first, second) in enumerate(graph.edges.tolist()):
        pair_mass = (
            candidate_posterior[first, :, None]
            * candidate_posterior[second, None, :]
        )
        same_maplet = np.zeros(pair_mass.shape, dtype=bool)
        if pool.has_explicit_maplet_clusters:
            same_maplet = maplets[first, :, None] == maplets[second, None, :]
        query_delta = np.asarray(pool.xy[second] - pool.xy[first], dtype=np.float64)
        query_distance = float(np.linalg.norm(query_delta))
        horizontal = int(abs(float(query_delta[0])) < abs(float(query_delta[1])))
        query_scale = int(_bucket(np.asarray(query_distance), 64.0, 160.0))
        xyz_distance = np.linalg.norm(
            np.asarray(pool.xyz[second], dtype=np.float64)[None, :, :]
            - np.asarray(pool.xyz[first], dtype=np.float64)[:, None, :],
            axis=2,
        )
        xyz_scale = _bucket(xyz_distance, 0.5, 2.0)
        relation_channel = horizontal * 9 + query_scale * 3 + xyz_scale
        relation_channel = np.where(same_maplet, 18, relation_channel)
        if pool.has_explicit_topology:
            first_tracks = np.asarray(pool.track_ids[first], dtype=np.int64)
            second_tracks = np.asarray(pool.track_ids[second], dtype=np.int64)
            first_neighbors = np.asarray(
                pool.topology_neighbor_track_ids[first], dtype=np.int64
            )
            second_neighbors = np.asarray(
                pool.topology_neighbor_track_ids[second], dtype=np.int64
            )
            first_matches = (
                first_neighbors[:, None, :] == second_tracks[None, :, None]
            )
            second_matches = (
                second_neighbors[None, :, :] == first_tracks[:, None, None]
            )
            first_direct = np.any(first_matches, axis=2)
            second_direct = np.any(second_matches, axis=2)
            direct_neighbor = first_direct | second_direct
            mutual_neighbor = first_direct & second_direct
            first_rank = np.where(
                first_direct,
                np.argmax(first_matches, axis=2),
                first_neighbors.shape[1],
            )
            second_rank = np.where(
                second_direct,
                np.argmax(second_matches, axis=2),
                second_neighbors.shape[1],
            )
            neighbor_rank = np.minimum(first_rank, second_rank)
            first_supports = np.asarray(
                pool.topology_support_image_indices[first], dtype=np.int64
            )
            second_supports = np.asarray(
                pool.topology_support_image_indices[second], dtype=np.int64
            )
            first_coverage = np.asarray(
                pool.topology_support_coverage_counts[first], dtype=np.float64
            )
            second_coverage = np.asarray(
                pool.topology_support_coverage_counts[second], dtype=np.float64
            )
            support_match = (
                (first_supports[:, None, :, None] >= 0)
                & (
                    first_supports[:, None, :, None]
                    == second_supports[None, :, None, :]
                )
            )
            shared_coverage = np.sum(
                np.where(
                    support_match,
                    np.minimum(
                        first_coverage[:, None, :, None],
                        second_coverage[None, :, None, :],
                    ),
                    0.0,
                ),
                axis=(2, 3),
            )
            coverage_denominator = np.minimum(
                np.sum(first_coverage, axis=1)[:, None],
                np.sum(second_coverage, axis=1)[None, :],
            )
            overlap = shared_coverage / np.maximum(coverage_denominator, 1.0)
            shared_support = overlap > 0.0
            support_channel = 19 + _bucket(overlap, 0.15, 0.40)
            relation_channel = np.where(shared_support, support_channel, relation_channel)
            directed_channel = np.where(neighbor_rank < 8, 22, 23)
            mutual_channel = np.where(neighbor_rank < 8, 24, 25)
            relation_channel = np.where(
                direct_neighbor & ~mutual_neighbor,
                directed_channel,
                relation_channel,
            )
            relation_channel = np.where(
                mutual_neighbor, mutual_channel, relation_channel
            )
        pair_projection_valid = (
            evidence.projection_valid[first, :, None]
            & evidence.projection_valid[second, None, :]
        )
        invalid_projection = (pair_mass > 0.0) & ~pair_projection_valid
        if np.any(invalid_projection):
            np.add.at(
                histogram[edge_index, :, -1],
                relation_channel[invalid_projection],
                pair_mass[invalid_projection],
            )

        predicted_delta = (
            evidence.projected_xy[second, None, :, None, None, :]
            - evidence.projected_xy[first, :, None, None, None, :]
        )
        observed_delta = (
            modes.xy[second, None, :, None, :, :]
            - modes.xy[first, :, None, :, None, :]
        )
        residual = np.linalg.norm(observed_delta - predicted_delta, axis=-1)
        mode_mass = (
            pair_mass[:, :, None, None]
            * modes.probabilities[first, :, None, :, None]
            * modes.probabilities[second, None, :, None, :]
        )
        valid_mode_pair = (
            pair_projection_valid[:, :, None, None]
            & modes.valid_mask[first, :, None, :, None]
            & modes.valid_mask[second, None, :, None, :]
            & (mode_mass > 0.0)
        )
        if np.any(valid_mode_pair):
            bin_index = np.clip(
                np.searchsorted(bins, residual[valid_mode_pair], side="right") - 1,
                0,
                len(bins) - 2,
            )
            channel = np.broadcast_to(
                relation_channel[:, :, None, None], mode_mass.shape
            )[valid_mode_pair]
            np.add.at(
                histogram[edge_index],
                (channel, bin_index),
                mode_mass[valid_mode_pair],
            )
        candidate_mass[edge_index] = float(np.sum(histogram[edge_index]))
        null_mass[edge_index] = max(1.0 - candidate_mass[edge_index], 0.0)
    return RelationResidualHistograms(
        histogram,
        candidate_mass,
        null_mass,
        bins,
        graph.sha256,
    )
