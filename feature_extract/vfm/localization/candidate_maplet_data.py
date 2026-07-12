"""Real-image candidate-maplet episodes for assignment-only training.

Query pose is used exclusively to construct supervision targets.  Every model
input is available at inference time: detector features, candidate metadata,
and real support observations tied to the proposed physical track.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.candidate_maplet_matcher import CandidateMapletBatch
from feature_extract.vfm.localization.candidate_maplet_schema import (
    validate_candidate_maplet_static_feature_names,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationFeatureStore,
    canonical_rows_for_track_candidates,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.radio_intermediate_context import (
    load_radio_intermediate_context_cache,
)
from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


QUERY_DESCRIPTOR_DIM = 64
BASE_QUERY_INPUT_DIM = 70
BASE_SUPPORT_INPUT_DIM = 71


@dataclass(frozen=True)
class CandidateMapletEpisodeArrays:
    query_features: np.ndarray
    query_xy: np.ndarray
    support_features: np.ndarray
    support_track_ids: np.ndarray
    support_xyz: np.ndarray
    static_features: np.ndarray
    target_track_indices: np.ndarray | None
    candidate_label: bool | None
    anchor_residual_px: float | None
    candidate_visible: bool | None
    edge_index: int
    query_id: str
    support_image_id: str

    def __post_init__(self) -> None:
        query_features = np.asarray(self.query_features, dtype=np.float32)
        query_xy = np.asarray(self.query_xy, dtype=np.float32).reshape(-1, 2)
        support_features = np.asarray(self.support_features, dtype=np.float32)
        tracks = np.asarray(self.support_track_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.support_xyz, dtype=np.float32).reshape(-1, 3)
        static = np.asarray(self.static_features, dtype=np.float32).reshape(-1)
        if (self.target_track_indices is None) != (self.candidate_label is None):
            raise ValueError(
                "assignment targets and candidate label must be provided together"
            )
        if (self.anchor_residual_px is None) != (self.candidate_visible is None):
            raise ValueError(
                "anchor residual and candidate visibility must be provided together"
            )
        targets = (
            None
            if self.target_track_indices is None
            else np.asarray(self.target_track_indices, dtype=np.int64).reshape(-1)
        )
        if query_features.ndim != 2 or query_features.shape[0] != len(query_xy):
            raise ValueError("query episode arrays have incompatible shapes")
        if support_features.ndim != 2 or support_features.shape[0] != len(tracks) or len(xyz) != len(tracks):
            raise ValueError("support episode arrays have incompatible shapes")
        if len(query_features) == 0 or len(support_features) == 0:
            raise ValueError("candidate-maplet episodes require non-empty node sets")
        if targets is not None:
            if targets.shape[0] != query_features.shape[0]:
                raise ValueError("assignment targets must match query nodes")
            if np.any((targets < 0) | (targets > len(tracks))):
                raise ValueError("assignment target references an invalid support node")
        residual = (
            None
            if self.anchor_residual_px is None
            else float(self.anchor_residual_px)
        )
        if residual is not None:
            if math.isnan(residual) or residual < 0.0:
                raise ValueError(
                    "anchor residual must be non-negative or positive infinity"
                )
            if bool(self.candidate_visible) != math.isfinite(residual):
                raise ValueError("candidate visibility and anchor residual disagree")
        object.__setattr__(self, "query_features", query_features)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "support_features", support_features)
        object.__setattr__(self, "support_track_ids", tracks)
        object.__setattr__(self, "support_xyz", xyz)
        object.__setattr__(self, "static_features", static)
        object.__setattr__(self, "target_track_indices", targets)
        object.__setattr__(self, "anchor_residual_px", residual)


def build_candidate_maplet_assignment_targets(
    residuals_px: np.ndarray,
    *,
    threshold_px: float,
    candidate_label: bool,
    anchor_threshold_px: float | None = None,
) -> np.ndarray:
    """Build one-to-one query-to-track targets with an explicit query dustbin."""

    residuals = np.asarray(residuals_px, dtype=np.float32)
    if residuals.ndim != 2 or residuals.shape[0] <= 0 or residuals.shape[1] <= 0:
        raise ValueError("residuals must have non-empty shape (Nq, Nt)")
    anchor_threshold = float(threshold_px) if anchor_threshold_px is None else float(anchor_threshold_px)
    if float(threshold_px) <= 0.0 or anchor_threshold <= 0.0:
        raise ValueError("assignment and anchor thresholds must be positive")
    query_count, track_count = residuals.shape
    anchor_is_positive = bool(np.isfinite(residuals[0, 0]) and residuals[0, 0] <= anchor_threshold)
    if anchor_is_positive != bool(candidate_label):
        raise ValueError("candidate label and anchor residual disagree")
    targets = np.full((query_count,), track_count, dtype=np.int64)
    # Query node zero is the proposed 2D anchor.  It is either the proposed
    # physical track or dustbin; neighboring tracks may not steal that node.
    used_queries: set[int] = {0}
    used_tracks: set[int] = set()
    if bool(candidate_label):
        targets[0] = 0
        used_tracks.add(0)
    pairs = np.argwhere(np.isfinite(residuals) & (residuals <= float(threshold_px)))
    if pairs.size == 0:
        return targets
    values = residuals[pairs[:, 0], pairs[:, 1]]
    order = np.lexsort((pairs[:, 1], pairs[:, 0], values))
    for position in order.tolist():
        query_index = int(pairs[position, 0])
        track_index = int(pairs[position, 1])
        if query_index in used_queries or track_index in used_tracks:
            continue
        targets[query_index] = track_index
        used_queries.add(query_index)
        used_tracks.add(track_index)
    return targets


class CandidateMapletEpisodeStore:
    """Lazy, validated episode store backed by image-referenced feature caches."""

    def __init__(
        self,
        *,
        proposals: Path,
        detector_query_cache: Path,
        query_context_detector_cache: Path,
        support_feature_cache: Path,
        support_geometry_index: Path,
        projected_landmark_bank: Path,
        maplet_support_index: Path,
        feature_artifact: Path,
        colmap_model_dir: Path,
        radio_intermediate_cache: Path | None = None,
        query_radius_px: float = 96.0,
        max_query_nodes: int = 48,
        max_support_tracks: int = 33,
        positive_threshold_px: float = 2.0,
        assignment_threshold_px: float = 5.0,
        support_view_count: int = 2,
        static_feature_count: int = 17,
        query_cache_size: int = 16384,
        support_cache_size: int = 8192,
        episode_cache_size: int = 4096,
        load_supervision: bool = True,
    ) -> None:
        if min(
            int(max_query_nodes),
            int(max_support_tracks),
            int(support_view_count),
            int(static_feature_count),
            int(query_cache_size),
            int(support_cache_size),
            int(episode_cache_size),
        ) <= 0:
            raise ValueError("episode dimensions must be positive")
        if (
            float(query_radius_px) <= 0.0
            or float(positive_threshold_px) <= 0.0
            or float(assignment_threshold_px) < float(positive_threshold_px)
        ):
            raise ValueError("episode geometry thresholds must be positive")
        self.query_radius_px = float(query_radius_px)
        self.max_query_nodes = int(max_query_nodes)
        self.max_support_tracks = int(max_support_tracks)
        self.positive_threshold_px = float(positive_threshold_px)
        self.assignment_threshold_px = float(assignment_threshold_px)
        self.support_view_count = int(support_view_count)
        self.static_feature_count = int(static_feature_count)
        self.query_cache_size = int(query_cache_size)
        self.support_cache_size = int(support_cache_size)
        self.episode_cache_size = int(episode_cache_size)
        self.load_supervision = bool(load_supervision)

        with np.load(Path(proposals), allow_pickle=False) as data:
            supervision_keys = {
                "nearest_visible_bank_rows",
                "nearest_visible_track_ids",
                "nearest_visible_residuals_px",
                "candidate_gt_residuals_px",
            }
            self.proposals = {
                key: np.asarray(data[key])
                for key in data.files
                if self.load_supervision or key not in supervision_keys
            }
        with np.load(Path(detector_query_cache), allow_pickle=False) as data:
            self.query_cache = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            self.query_metadata = json.loads(str(data["metadata_json"].item()))
        with np.load(Path(query_context_detector_cache), allow_pickle=False) as data:
            self.context_cache = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            self.context_metadata = json.loads(str(data["metadata_json"].item()))
        with np.load(Path(support_feature_cache), allow_pickle=False) as data:
            support_tracks = np.asarray(data["track_ids"], dtype=np.int64)
            support_descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            support_scores = np.asarray(data["detector_scores"], dtype=np.float32)
            support_metadata = json.loads(str(data["metadata_json"].item()))
        geometry, geometry_metadata = load_support_observation_geometry_index_npz(
            Path(support_geometry_index)
        )
        self.landmark_index, _landmark_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
        self.maplet_index, _maplet_metadata = load_local_maplet_support_index_npz(Path(maplet_support_index))
        with np.load(Path(feature_artifact), allow_pickle=False) as data:
            feature_metadata = json.loads(str(data["metadata_json"].item()))
            feature_names = tuple(str(value) for value in data["feature_names"].tolist())
            self.selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
            self.selected_columns = np.asarray(data["selected_columns"], dtype=np.int64)
            all_features = np.asarray(data["features"], dtype=np.float32)
            self.labels = (
                np.asarray(data["labels"], dtype=bool)
                if self.load_supervision
                else None
            )
            self.valid_edges = np.asarray(data["valid_edges"], dtype=bool)
        expected_hashes = {
            "proposals_sha256": file_sha256_short(Path(proposals)),
            "detector_query_cache_sha256": file_sha256_short(Path(detector_query_cache)),
            "query_context_detector_cache_sha256": file_sha256_short(Path(query_context_detector_cache)),
            "support_feature_cache_sha256": file_sha256_short(Path(support_feature_cache)),
            "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
            "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
            "maplet_support_index_sha256": file_sha256_short(Path(maplet_support_index)),
        }
        mismatches = {
            key: {"expected": value, "actual": feature_metadata.get(key)}
            for key, value in expected_hashes.items()
            if feature_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"stale candidate-maplet feature artifact: {json.dumps(mismatches, sort_keys=True)}")
        if str(feature_metadata.get("query_context_source", "")) != "dense_alike_detector_cache":
            raise ValueError("assignment training requires the dense ALIKE query context artifact")
        if all_features.shape[:2] != self.selected_columns.shape or (
            self.labels is not None
            and self.labels.shape != self.selected_columns.shape
        ):
            raise ValueError("candidate feature artifact arrays have incompatible shapes")
        if int(all_features.shape[2]) < self.static_feature_count:
            raise ValueError("candidate feature artifact lacks requested static features")
        if len(feature_names) != int(all_features.shape[2]):
            raise ValueError("candidate feature names do not match the feature tensor")
        self.static_feature_names = validate_candidate_maplet_static_feature_names(
            feature_names,
            count=self.static_feature_count,
        )
        self.static_features = all_features[:, :, : self.static_feature_count]
        if not np.all(self.valid_edges) or np.any(self.selected_columns < 0):
            raise ValueError("candidate-maplet training currently requires a fixed valid top-K pool")

        self.query_ids = np.asarray(self.proposals["query_ids"]).astype(str)
        self.query_xy = np.asarray(self.proposals["xy"], dtype=np.float32)
        cache_ids = np.repeat(
            self.query_cache["image_ids"].astype(str),
            np.diff(np.asarray(self.query_cache["offsets"], dtype=np.int64)),
        )
        if not np.array_equal(cache_ids, self.query_ids) or not np.array_equal(
            self.query_cache["xy"], self.query_xy
        ):
            raise ValueError("proposal and detector anchor rows differ")
        if self.context_metadata.get("format") != "alike_dense_query_context_cache_v1":
            raise ValueError("unsupported dense query context cache")
        if not np.array_equal(
            self.context_cache["image_ids"].astype(str), self.query_cache["image_ids"].astype(str)
        ):
            raise ValueError("anchor and dense context image lists differ")
        if self.query_metadata.get("image_sha256_by_id") != self.context_metadata.get("image_sha256_by_id"):
            raise ValueError("anchor and dense context image contents differ")
        if str(self.query_metadata.get("alike_checkpoint_sha256", "")) != str(
            self.context_metadata.get("alike_checkpoint_sha256", "")
        ) or str(self.query_metadata.get("alike_checkpoint_sha256", "")) != str(
            support_metadata.get("model_checkpoint_sha256", "")
        ):
            raise ValueError("query and support ALIKE checkpoints differ")
        if str(geometry_metadata.get("support_feature_cache_sha256", "")) != file_sha256_short(
            Path(support_feature_cache)
        ):
            raise ValueError("support geometry index references a different feature cache")
        if not np.array_equal(self.maplet_index.anchor_track_ids, self.landmark_index.track_ids):
            raise ValueError("maplet and canonical landmark rows differ")

        candidate_tracks = np.asarray(self.proposals["candidate_track_ids"], dtype=np.int64)
        self.canonical_candidate_rows = canonical_rows_for_track_candidates(
            candidate_tracks, self.landmark_index.track_ids
        )
        self.compact_candidate_tracks = np.take_along_axis(
            candidate_tracks[self.selected_rows], self.selected_columns, axis=1
        )
        self.compact_canonical_rows = np.take_along_axis(
            self.canonical_candidate_rows[self.selected_rows], self.selected_columns, axis=1
        )
        self.edge_query_ids = np.repeat(self.query_ids[self.selected_rows], self.candidate_top_k)
        if self.load_supervision:
            if self.labels is None:
                raise ValueError("supervised episode loading requires feature labels")
            residuals = np.take_along_axis(
                np.asarray(
                    self.proposals["candidate_gt_residuals_px"], dtype=np.float32
                )[self.selected_rows],
                self.selected_columns,
                axis=1,
            )
            if not np.array_equal(
                self.labels, residuals <= self.positive_threshold_px
            ):
                raise ValueError(
                    "feature-artifact labels and configured positive threshold differ"
                )
            if np.any(np.isnan(residuals)) or np.any(residuals < 0.0):
                raise ValueError(
                    "candidate residuals must be non-negative or positive infinity"
                )
            self.candidate_residuals_px: np.ndarray | None = residuals
            self.candidate_visible: np.ndarray | None = np.isfinite(residuals)
            self.edge_labels: np.ndarray | None = self.labels.reshape(-1)
        else:
            self.candidate_residuals_px = None
            self.candidate_visible = None
            self.edge_labels = None

        self.query_descriptors, valid_query = normalize_rows(
            np.asarray(self.query_cache["local_descriptors"], dtype=np.float32)
        )
        self.context_descriptors, valid_context = normalize_rows(
            np.asarray(self.context_cache["local_descriptors"], dtype=np.float32)
        )
        if not np.all(valid_query) or not np.all(valid_context):
            raise ValueError("query ALIKE caches contain invalid descriptors")
        self.support_store = SupportObservationFeatureStore(
            geometry,
            source_track_ids=support_tracks,
            source_descriptors=support_descriptors,
            source_detector_scores=support_scores,
        )
        self.radio_intermediate_cache_path = (
            None if radio_intermediate_cache is None else Path(radio_intermediate_cache)
        )
        self.radio_descriptor_dim = 0
        self.radio_support_store: SupportObservationFeatureStore | None = None
        self.radio_query_descriptors: np.ndarray | None = None
        self.radio_context_descriptors: np.ndarray | None = None
        if self.radio_intermediate_cache_path is not None:
            radio_cache = load_radio_intermediate_context_cache(
                self.radio_intermediate_cache_path,
                expected_metadata={
                    "support_feature_cache_sha256": file_sha256_short(
                        Path(support_feature_cache)
                    ),
                    "support_geometry_index_sha256": file_sha256_short(
                        Path(support_geometry_index)
                    ),
                    "query_anchor_cache_sha256": file_sha256_short(
                        Path(detector_query_cache)
                    ),
                    "query_context_cache_sha256": file_sha256_short(
                        Path(query_context_detector_cache)
                    ),
                },
            )
            if len(radio_cache.support_descriptors) != len(support_tracks):
                raise ValueError("RADIO support descriptors do not align with ALIKE rows")
            if len(radio_cache.query_anchor_descriptors) != len(self.query_descriptors):
                raise ValueError("RADIO query anchors do not align with the detector cache")
            if len(radio_cache.query_context_descriptors) != len(self.context_descriptors):
                raise ValueError("RADIO query context does not align with the context cache")
            self.radio_descriptor_dim = int(radio_cache.descriptor_dim)
            self.radio_query_descriptors = radio_cache.query_anchor_descriptors
            self.radio_context_descriptors = radio_cache.query_context_descriptors
            self.radio_support_store = SupportObservationFeatureStore(
                geometry,
                source_track_ids=support_tracks,
                source_descriptors=radio_cache.support_descriptors,
                source_detector_scores=support_scores,
            )
        self.context_image_position = {
            str(image_id): int(index)
            for index, image_id in enumerate(self.context_cache["image_ids"].astype(str).tolist())
        }
        self.landmark_track_rows = {
            int(track_id): int(row) for row, track_id in enumerate(self.landmark_index.track_ids.tolist())
        }
        if self.load_supervision:
            self.cameras = read_colmap_cameras_binary(
                Path(colmap_model_dir) / "cameras.bin"
            )
            images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
            self.images_by_name = {
                str(image.image_name): image for image in images.values()
            }
        else:
            self.cameras = {}
            self.images_by_name = {}
        self._query_nodes_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._support_nodes_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, str]] = OrderedDict()
        self._episode_cache: OrderedDict[tuple[int, int], CandidateMapletEpisodeArrays] = OrderedDict()

    @property
    def candidate_top_k(self) -> int:
        return int(self.selected_columns.shape[1])

    @property
    def edge_count(self) -> int:
        return int(self.selected_columns.size)

    @property
    def query_input_dim(self) -> int:
        return int(BASE_QUERY_INPUT_DIM + self.radio_descriptor_dim)

    @property
    def support_input_dim(self) -> int:
        return int(BASE_SUPPORT_INPUT_DIM + self.radio_descriptor_dim)

    @property
    def static_input_dim(self) -> int:
        return int(self.static_feature_count)

    def split_edge_indices(self, query_ids: Sequence[str]) -> np.ndarray:
        return np.flatnonzero(
            np.isin(self.edge_query_ids, np.asarray([str(value) for value in query_ids], dtype=np.str_))
        )

    @staticmethod
    def _bounded_cache_put(cache: OrderedDict, key, value, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > int(limit):
            cache.popitem(last=False)

    def _query_nodes(self, selected_row_index: int) -> tuple[np.ndarray, np.ndarray]:
        key = int(selected_row_index)
        cached = self._query_nodes_cache.get(key)
        if cached is not None:
            self._query_nodes_cache.move_to_end(key)
            return cached
        global_row = int(self.selected_rows[key])
        query_id = str(self.query_ids[global_row])
        context_position = self.context_image_position[query_id]
        start = int(self.context_cache["offsets"][context_position])
        end = int(self.context_cache["offsets"][context_position + 1])
        context_xy = np.asarray(self.context_cache["xy"][start:end], dtype=np.float32)
        context_scores = np.asarray(self.context_cache["detector_scores"][start:end], dtype=np.float32)
        anchor_xy = self.query_xy[global_row]
        distances2 = np.sum((context_xy - anchor_xy[None]) ** 2, axis=1)
        valid = np.all(np.isfinite(context_xy), axis=1) & np.isfinite(context_scores)
        valid &= distances2 <= self.query_radius_px**2
        valid &= distances2 > 4.0
        candidates = np.flatnonzero(valid)
        if candidates.size:
            order = np.lexsort((candidates, -context_scores[candidates]))
            candidates = candidates[order[: max(self.max_query_nodes - 1, 0)]]
        xy = np.concatenate([anchor_xy[None], context_xy[candidates]], axis=0).astype(np.float32)
        descriptors = np.concatenate(
            [self.query_descriptors[global_row : global_row + 1], self.context_descriptors[start:end][candidates]],
            axis=0,
        )
        if self.radio_descriptor_dim:
            assert self.radio_query_descriptors is not None
            assert self.radio_context_descriptors is not None
            radio_descriptors = np.concatenate(
                [
                    self.radio_query_descriptors[global_row : global_row + 1],
                    self.radio_context_descriptors[start:end][candidates],
                ],
                axis=0,
            )
        else:
            radio_descriptors = np.zeros((len(descriptors), 0), dtype=np.float32)
        scores = np.concatenate(
            [
                np.asarray([self.query_cache["detector_scores"][global_row]], dtype=np.float32),
                context_scores[candidates],
            ]
        )
        relative = np.clip((xy - anchor_xy[None]) / self.query_radius_px, -4.0, 4.0)
        radius = np.linalg.norm(relative, axis=1)
        anchor_flag = np.zeros((len(xy),), dtype=np.float32)
        anchor_flag[0] = 1.0
        features = np.concatenate(
            [
                descriptors,
                radio_descriptors,
                relative,
                radius[:, None],
                np.clip(scores, 0.0, 1.0)[:, None],
                np.log10(np.maximum(scores, 1e-8))[:, None],
                anchor_flag[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        if features.shape[1] != self.query_input_dim:
            raise RuntimeError("unexpected query node feature dimension")
        value = (features, xy)
        self._bounded_cache_put(self._query_nodes_cache, key, value, self.query_cache_size)
        return value

    def _support_nodes(
        self,
        canonical_row: int,
        view_rank: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        rank = int(view_rank) % self.support_view_count
        key = (int(canonical_row), rank)
        cached = self._support_nodes_cache.get(key)
        if cached is not None:
            self._support_nodes_cache.move_to_end(key)
            return cached
        anchor_track = int(self.landmark_index.track_ids[int(canonical_row)])
        neighbors = self.maplet_index.neighbor_track_ids[int(canonical_row)]
        maplet_tracks = np.concatenate(
            [np.asarray([anchor_track], dtype=np.int64), neighbors[neighbors >= 0]]
        )[: self.max_support_tracks]
        image_indices = self.maplet_index.support_image_indices[
            int(canonical_row), : self.support_view_count
        ]
        valid_images = image_indices[image_indices >= 0]
        if valid_images.size == 0:
            raise ValueError(f"candidate track has no support view: {anchor_track}")
        image_index = int(valid_images[rank % len(valid_images)])
        image_id = str(self.maplet_index.support_image_ids[image_index])
        view = self.support_store.maplet_view(image_id, maplet_tracks)
        radio_view = (
            None
            if self.radio_support_store is None
            else self.radio_support_store.maplet_view(image_id, maplet_tracks)
        )
        if radio_view is not None and not np.array_equal(
            radio_view.track_ids, view.track_ids
        ):
            raise ValueError("ALIKE and RADIO support-view rows differ")
        anchor_positions = np.flatnonzero(view.track_ids == anchor_track)
        if anchor_positions.size != 1:
            raise ValueError("selected support view does not contain its anchor track")
        if int(anchor_positions[0]) != 0:
            order = np.concatenate(
                [anchor_positions, np.flatnonzero(view.track_ids != anchor_track)]
            )
            tracks = view.track_ids[order]
            xy = view.xy[order]
            descriptors = view.descriptors[order]
            radio_descriptors = (
                np.zeros((len(order), 0), dtype=np.float32)
                if radio_view is None
                else radio_view.descriptors[order]
            )
            detector_scores = view.detector_scores[order]
            reprojection_errors = view.reprojection_errors[order]
        else:
            tracks = view.track_ids
            xy = view.xy
            descriptors = view.descriptors
            radio_descriptors = (
                np.zeros((len(tracks), 0), dtype=np.float32)
                if radio_view is None
                else radio_view.descriptors
            )
            detector_scores = view.detector_scores
            reprojection_errors = view.reprojection_errors
        rows = np.asarray([self.landmark_track_rows[int(track)] for track in tracks], dtype=np.int64)
        xyz = np.asarray(self.landmark_index.xyz[rows], dtype=np.float32)
        relative = np.clip((xy - xy[0:1]) / self.query_radius_px, -4.0, 4.0)
        radius = np.linalg.norm(relative, axis=1)
        anchor_flag = np.zeros((len(tracks),), dtype=np.float32)
        anchor_flag[0] = 1.0
        features = np.concatenate(
            [
                descriptors,
                radio_descriptors,
                relative,
                radius[:, None],
                np.clip(detector_scores, 0.0, 1.0)[:, None],
                np.log10(np.maximum(detector_scores, 1e-8))[:, None],
                np.clip(reprojection_errors / 5.0, 0.0, 2.0)[:, None],
                anchor_flag[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        if features.shape[1] != self.support_input_dim:
            raise RuntimeError("unexpected support node feature dimension")
        value = (features, tracks.astype(np.int64), xyz, image_id)
        self._bounded_cache_put(self._support_nodes_cache, key, value, self.support_cache_size)
        return value

    def episode(self, edge_index: int, *, view_rank: int = 0) -> CandidateMapletEpisodeArrays:
        edge = int(edge_index)
        if edge < 0 or edge >= self.edge_count:
            raise IndexError("candidate edge index out of range")
        cache_key = (edge, int(view_rank) % self.support_view_count)
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            self._episode_cache.move_to_end(cache_key)
            return cached
        selected_row = edge // self.candidate_top_k
        compact_column = edge % self.candidate_top_k
        canonical_row = int(self.compact_canonical_rows[selected_row, compact_column])
        query_features, query_xy = self._query_nodes(selected_row)
        support_features, support_tracks, support_xyz, support_image_id = self._support_nodes(
            canonical_row, int(view_rank)
        )
        global_row = int(self.selected_rows[selected_row])
        query_id = str(self.query_ids[global_row])
        if self.load_supervision:
            image = self.images_by_name.get(query_id)
            if image is None:
                raise KeyError(f"query image missing from COLMAP model: {query_id}")
            camera = self.cameras[int(image.camera_id)]
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = qvec_to_rotmat(image.qvec)
            pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
            projected = project_world_to_image(
                support_xyz.astype(np.float64), pose, camera
            )
            camera_points = (
                support_xyz.astype(np.float64) @ pose[:3, :3].T
                + pose[:3, 3][None]
            )
            visible = np.isfinite(projected).all(axis=1) & (
                camera_points[:, 2] > 1e-6
            )
            visible &= (projected[:, 0] >= 0.0) & (
                projected[:, 0] <= float(camera.width - 1)
            )
            visible &= (projected[:, 1] >= 0.0) & (
                projected[:, 1] <= float(camera.height - 1)
            )
            residuals = np.linalg.norm(
                query_xy[:, None, :] - projected[None, :, :], axis=2
            ).astype(np.float32)
            residuals[:, ~visible] = np.inf
            assert self.labels is not None
            assert self.candidate_residuals_px is not None
            candidate_label: bool | None = bool(
                self.labels[selected_row, compact_column]
            )
            anchor_residual_px: float | None = float(residuals[0, 0])
            candidate_visible: bool | None = bool(visible[0])
            stored_residual_px = float(
                self.candidate_residuals_px[selected_row, compact_column]
            )
            if math.isfinite(stored_residual_px) != math.isfinite(
                anchor_residual_px
            ) or (
                math.isfinite(stored_residual_px)
                and not math.isclose(
                    stored_residual_px,
                    anchor_residual_px,
                    rel_tol=1e-5,
                    abs_tol=1e-3,
                )
            ):
                raise ValueError(
                    "episode anchor projection differs from proposal supervision"
                )
            targets: np.ndarray | None = build_candidate_maplet_assignment_targets(
                residuals,
                threshold_px=self.assignment_threshold_px,
                candidate_label=candidate_label,
                anchor_threshold_px=self.positive_threshold_px,
            )
        else:
            targets = None
            candidate_label = None
            anchor_residual_px = None
            candidate_visible = None
        episode = CandidateMapletEpisodeArrays(
            query_features=query_features,
            query_xy=query_xy,
            support_features=support_features,
            support_track_ids=support_tracks,
            support_xyz=support_xyz,
            static_features=self.static_features[selected_row, compact_column],
            target_track_indices=targets,
            candidate_label=candidate_label,
            anchor_residual_px=anchor_residual_px,
            candidate_visible=candidate_visible,
            edge_index=edge,
            query_id=query_id,
            support_image_id=support_image_id,
        )
        self._bounded_cache_put(self._episode_cache, cache_key, episode, self.episode_cache_size)
        return episode

    def batch(self, edge_indices: Sequence[int], *, view_ranks: Sequence[int] | None = None) -> CandidateMapletBatch:
        edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1)
        if edges.size == 0:
            raise ValueError("cannot build an empty candidate-maplet batch")
        if view_ranks is None:
            ranks = np.zeros(edges.shape, dtype=np.int64)
        else:
            ranks = np.asarray(view_ranks, dtype=np.int64).reshape(-1)
            if ranks.shape != edges.shape:
                raise ValueError("view_ranks must match edge_indices")
        episodes = [self.episode(int(edge), view_rank=int(rank)) for edge, rank in zip(edges, ranks)]
        max_query = max(len(episode.query_features) for episode in episodes)
        max_support = max(len(episode.support_features) for episode in episodes)
        query = np.zeros((len(episodes), max_query, self.query_input_dim), dtype=np.float32)
        query_mask = np.zeros((len(episodes), max_query), dtype=bool)
        support = np.zeros((len(episodes), max_support, self.support_input_dim), dtype=np.float32)
        support_mask = np.zeros((len(episodes), max_support), dtype=bool)
        supervised = episodes[0].target_track_indices is not None
        if any(
            (episode.target_track_indices is not None) != supervised
            for episode in episodes
        ):
            raise ValueError("cannot mix supervised and inference-only episodes")
        targets = (
            np.full((len(episodes), max_query), -1, dtype=np.int64)
            if supervised
            else None
        )
        static = np.zeros((len(episodes), self.static_input_dim), dtype=np.float32)
        labels = (
            np.zeros((len(episodes),), dtype=np.float32) if supervised else None
        )
        anchor_residuals = (
            np.full((len(episodes),), np.inf, dtype=np.float32)
            if supervised
            else None
        )
        candidate_visible = (
            np.zeros((len(episodes),), dtype=bool) if supervised else None
        )
        for batch_index, episode in enumerate(episodes):
            query_count = len(episode.query_features)
            support_count = len(episode.support_features)
            query[batch_index, :query_count] = episode.query_features
            query_mask[batch_index, :query_count] = True
            support[batch_index, :support_count] = episode.support_features
            support_mask[batch_index, :support_count] = True
            if supervised:
                assert targets is not None
                assert labels is not None
                assert anchor_residuals is not None
                assert candidate_visible is not None
                assert episode.target_track_indices is not None
                assert episode.candidate_label is not None
                assert episode.anchor_residual_px is not None
                assert episode.candidate_visible is not None
                targets[batch_index, :query_count] = episode.target_track_indices
                labels[batch_index] = float(episode.candidate_label)
                anchor_residuals[batch_index] = float(episode.anchor_residual_px)
                candidate_visible[batch_index] = bool(episode.candidate_visible)
            static[batch_index] = episode.static_features
        batch = CandidateMapletBatch(
            query_features=torch.from_numpy(query),
            query_mask=torch.from_numpy(query_mask),
            support_features=torch.from_numpy(support),
            support_mask=torch.from_numpy(support_mask),
            static_features=torch.from_numpy(static),
            target_track_indices=(
                None if targets is None else torch.from_numpy(targets)
            ),
            candidate_labels=(None if labels is None else torch.from_numpy(labels)),
            anchor_residuals_px=(
                None
                if anchor_residuals is None
                else torch.from_numpy(anchor_residuals)
            ),
            candidate_visible=(
                None
                if candidate_visible is None
                else torch.from_numpy(candidate_visible)
            ),
            edge_indices=torch.from_numpy(edges),
        )
        batch.validate()
        return batch
