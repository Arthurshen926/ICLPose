"""Episode construction for real-image local landmark assignment training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_matcher import LocalAssignmentEpisode
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.query_to_3d_matching import normalize_rows


def _normalized_mean_by_sorted_track(
    track_ids: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != tracks.shape[0]:
        raise ValueError("features must have shape (N, C) with one track id per row")
    order = np.argsort(tracks, kind="stable")
    sorted_tracks = tracks[order]
    sorted_values, _valid = normalize_rows(values[order])
    unique_tracks, starts, counts = np.unique(sorted_tracks, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_values, starts, axis=0)
    means, _valid_means = normalize_rows(sums / counts[:, None])
    return unique_tracks, means, starts, order


class LocalAssignmentFeatureStore:
    """Validated in-memory feature store that emits one query-image episode."""

    def __init__(
        self,
        *,
        probe_arrays: Path,
        real_feature_cache: Path,
        query_global_cache: Path,
        projected_landmark_bank: Path,
        maplet_support_index: Path,
        max_support_views: int = 8,
    ) -> None:
        if int(max_support_views) <= 0:
            raise ValueError("max_support_views must be positive")
        self.max_support_views = int(max_support_views)
        self.probe_path = Path(probe_arrays)
        with np.load(self.probe_path, allow_pickle=False) as data:
            self.probe = {key: np.asarray(data[key]) for key in data.files}
        self.query_ids = tuple(str(item) for item in self.probe["query_ids"].tolist())
        self.query_xy = np.asarray(self.probe["query_xy"], dtype=np.float32)
        self.correct_track_ids = np.asarray(self.probe["correct_track_ids"], dtype=np.int64)
        self.query_point2d_indices = np.asarray(self.probe["query_point2d_indices"], dtype=np.int64)
        self.candidates = UniqueTrackCandidateSet(
            bank_row_indices=np.asarray(self.probe["bank_row_indices"], dtype=np.int64),
            track_ids=np.asarray(self.probe["candidate_track_ids"], dtype=np.int64),
            prototype_ids=np.asarray(self.probe["candidate_prototype_ids"], dtype=np.int64),
            coarse_scores=np.asarray(self.probe["coarse_scores"], dtype=np.float32),
        )
        if len(self.query_ids) != self.candidates.query_count:
            raise ValueError("probe query ids and candidate rows have different lengths")

        with np.load(Path(real_feature_cache), allow_pickle=False) as data:
            self.query_local_features = np.asarray(data["query_features"], dtype=np.float32)
            self.query_detector_scores = np.asarray(data["query_detector_scores"], dtype=np.float32)
            support_features = np.asarray(data["support_features"], dtype=np.float32)
            support_detector_scores = np.asarray(data["support_detector_scores"], dtype=np.float32)
            support_track_ids = np.asarray(data["support_track_ids"], dtype=np.int64)
            feature_metadata = json.loads(str(data["metadata_json"].item()))
        source_probe = Path(str(feature_metadata.get("base_probe_arrays", "")))
        expected_source_hash = str(feature_metadata.get("base_probe_arrays_sha256", ""))
        if not source_probe.exists() or file_sha256_short(source_probe) != expected_source_hash:
            raise ValueError("real feature cache source probe hash mismatch")
        if self.query_local_features.shape[0] != self.candidates.query_count:
            raise ValueError("real feature cache query rows do not match probe")

        with np.load(Path(query_global_cache), allow_pickle=False) as data:
            self.query_global_features = np.asarray(data["query_global_features"], dtype=np.float32)
            self.query_heatmap_scores = np.asarray(data["query_heatmap_scores"], dtype=np.float32)
            query_metadata = json.loads(str(data["metadata_json"].item()))
        if str(query_metadata.get("probe_arrays_sha256", "")) != file_sha256_short(self.probe_path):
            raise ValueError("query global cache was built from a different probe artifact")
        if self.query_global_features.shape[0] != self.candidates.query_count:
            raise ValueError("query global cache rows do not match probe")

        self.landmark_index, landmark_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
        if str(query_metadata.get("descriptor_space_id", "")) != str(
            landmark_metadata.get("descriptor_space_id", "")
        ):
            raise ValueError("query global and landmark descriptor spaces differ")
        valid = self.candidates.valid_mask
        bank_rows = self.candidates.bank_row_indices[valid]
        if not np.array_equal(self.candidates.track_ids[valid], self.landmark_index.track_ids[bank_rows]):
            raise ValueError("candidate rows are not aligned with the landmark bank")
        if not np.array_equal(self.candidates.prototype_ids[valid], self.landmark_index.prototype_ids[bank_rows]):
            raise ValueError("candidate prototypes are not aligned with the landmark bank")

        self.maplet_index, maplet_metadata = load_local_maplet_support_index_npz(Path(maplet_support_index))
        source_descriptor_id = str(maplet_metadata.get("source_descriptor_space_id", ""))
        if not source_descriptor_id:
            raise ValueError("maplet support index is missing source descriptor identity")
        candidate_unique_tracks = np.unique(self.landmark_index.track_ids)
        if not np.array_equal(candidate_unique_tracks, self.maplet_index.anchor_track_ids):
            raise ValueError("maplet and candidate landmark banks contain different tracks")

        support_order = np.argsort(support_track_ids, kind="stable")
        self.support_track_ids = support_track_ids[support_order]
        self.support_features, _valid_support = normalize_rows(support_features[support_order])
        self.support_detector_scores = support_detector_scores[support_order]
        self.support_unique_tracks, self.support_starts, self.support_counts = np.unique(
            self.support_track_ids,
            return_index=True,
            return_counts=True,
        )
        sums = np.add.reduceat(self.support_features, self.support_starts, axis=0)
        self.support_means, _valid_means = normalize_rows(sums / self.support_counts[:, None])

        bank_order = np.argsort(self.landmark_index.track_ids, kind="stable")
        bank_tracks = self.landmark_index.track_ids[bank_order]
        bank_features, _valid_bank = normalize_rows(self.landmark_index.features[bank_order])
        self.bank_unique_tracks, bank_starts, bank_counts = np.unique(
            bank_tracks,
            return_index=True,
            return_counts=True,
        )
        bank_sums = np.add.reduceat(bank_features, bank_starts, axis=0)
        self.bank_track_features, _valid_track = normalize_rows(bank_sums / bank_counts[:, None])
        self.bank_track_xyz = self.landmark_index.xyz[bank_order[bank_starts]].astype(np.float32)
        self.bank_track_observation_counts = np.maximum.reduceat(
            self.landmark_index.observation_counts[bank_order], bank_starts
        ).astype(np.float32)
        self.bank_track_variances = (
            np.add.reduceat(self.landmark_index.mean_variances[bank_order], bank_starts) / bank_counts
        ).astype(np.float32)
        self.bank_track_reprojection = (
            np.add.reduceat(self.landmark_index.reprojection_errors[bank_order], bank_starts) / bank_counts
        ).astype(np.float32)

        self.rows_by_query: dict[str, np.ndarray] = {}
        for query_id in dict.fromkeys(self.query_ids):
            self.rows_by_query[str(query_id)] = np.flatnonzero(
                np.asarray([value == query_id for value in self.query_ids], dtype=bool)
            )
        self.unique_query_ids = tuple(self.rows_by_query)
        self._episode_cache: dict[str, LocalAssignmentEpisode] = {}
        self._validate_strategy("coarse_prototype")
        self._validate_strategy("all_support_top2_mean")
        self._validate_strategy("alike_support_best")
        self._validate_strategy("alike_support_top2_mean")
        self._validate_strategy("alike_support_top4_mean")

    def _validate_strategy(self, name: str) -> None:
        key = f"strategy__{name}"
        if key not in self.probe:
            raise ValueError(f"probe artifact is missing required strategy evidence: {name}")
        if np.asarray(self.probe[key]).shape != self.candidates.coarse_scores.shape:
            raise ValueError(f"strategy evidence has wrong shape: {name}")

    @property
    def query_input_dim(self) -> int:
        return int(self.query_global_features.shape[1] + self.query_local_features.shape[1] + 6)

    @property
    def track_input_dim(self) -> int:
        return int(self.bank_track_features.shape[1] + self.maplet_index.maplets.context_features.shape[1] + self.support_means.shape[1] + 9)

    @property
    def support_input_dim(self) -> int:
        return int(self.support_features.shape[1] + 2)

    @property
    def edge_input_dim(self) -> int:
        return 9

    def subset_candidates(self, rows: np.ndarray) -> UniqueTrackCandidateSet:
        indices = np.asarray(rows, dtype=np.int64).reshape(-1)
        return UniqueTrackCandidateSet(
            self.candidates.bank_row_indices[indices],
            self.candidates.track_ids[indices],
            self.candidates.prototype_ids[indices],
            self.candidates.coarse_scores[indices],
        )

    def _support_range(self, track_id: int) -> tuple[int, int]:
        position = int(np.searchsorted(self.support_unique_tracks, int(track_id)))
        if position >= len(self.support_unique_tracks) or int(self.support_unique_tracks[position]) != int(track_id):
            raise KeyError(f"support features missing for track {track_id}")
        start = int(self.support_starts[position])
        return start, start + int(self.support_counts[position])

    def _bank_track_position(self, track_ids: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.bank_unique_tracks, np.asarray(track_ids, dtype=np.int64))
        if np.any(positions >= len(self.bank_unique_tracks)) or not np.array_equal(
            self.bank_unique_tracks[positions], np.asarray(track_ids, dtype=np.int64)
        ):
            raise KeyError("episode track missing from landmark bank")
        return positions.astype(np.int64)

    def episode(self, query_id: str) -> LocalAssignmentEpisode:
        cached = self._episode_cache.get(str(query_id))
        if cached is not None:
            return cached
        query_rows = self.rows_by_query[str(query_id)]
        candidate_tracks = self.candidates.track_ids[query_rows]
        valid_candidates = self.candidates.valid_mask[query_rows]
        stable_tracks: list[int] = []
        seen: set[int] = set()
        for track_id in candidate_tracks[valid_candidates].tolist():
            value = int(track_id)
            if value not in seen:
                seen.add(value)
                stable_tracks.append(value)
        episode_tracks = np.asarray(stable_tracks, dtype=np.int64)
        track_position = {int(track_id): int(index) for index, track_id in enumerate(stable_tracks)}
        bank_positions = self._bank_track_position(episode_tracks)
        maplet_positions = np.searchsorted(self.maplet_index.anchor_track_ids, episode_tracks)
        if not np.array_equal(self.maplet_index.anchor_track_ids[maplet_positions], episode_tracks):
            raise KeyError("episode track missing from maplet support index")
        support_positions = np.searchsorted(self.support_unique_tracks, episode_tracks)
        if not np.array_equal(self.support_unique_tracks[support_positions], episode_tracks):
            raise KeyError("episode track missing from local feature support cache")

        xyz = self.bank_track_xyz[bank_positions]
        center = np.median(xyz, axis=0)
        scale = max(float(np.median(np.linalg.norm(xyz - center[None], axis=1))), 1.0)
        xyz_normalized = (xyz - center[None]) / scale
        track_quality = np.stack(
            [
                np.log1p(self.bank_track_observation_counts[bank_positions]) / 5.0,
                np.log1p(np.maximum(self.bank_track_variances[bank_positions], 0.0)),
                np.clip(self.bank_track_reprojection[bank_positions] / 5.0, 0.0, 2.0),
                np.clip(self.maplet_index.maplets.context_radius[maplet_positions] / 5.0, 0.0, 2.0),
                np.clip(self.maplet_index.maplets.covisibility_strength[maplet_positions] / 10.0, 0.0, 2.0),
                np.clip(self.maplet_index.maplets.context_feature_variance[maplet_positions] * 100.0, 0.0, 2.0),
            ],
            axis=1,
        ).astype(np.float32)
        track_features = np.concatenate(
            [
                self.bank_track_features[bank_positions],
                self.maplet_index.maplets.context_features[maplet_positions],
                self.support_means[support_positions],
                xyz_normalized.astype(np.float32),
                track_quality,
            ],
            axis=1,
        )

        local_query = self.query_local_features[query_rows]
        detector = np.clip(self.query_detector_scores[query_rows], 0.0, 1.0)
        detector_log = np.clip((np.log10(np.maximum(detector, 1e-8)) + 8.0) / 8.0, 0.0, 1.0)
        heatmap = np.clip(self.query_heatmap_scores[query_rows], 0.0, 1.0)
        heatmap_log = np.clip((np.log10(np.maximum(heatmap, 1e-12)) + 12.0) / 12.0, 0.0, 1.0)
        xy = self.query_xy[query_rows]
        xy_normalized = np.stack(
            [2.0 * xy[:, 0] / 1023.0 - 1.0, 2.0 * xy[:, 1] / 575.0 - 1.0],
            axis=1,
        ).astype(np.float32)
        query_features = np.concatenate(
            [
                self.query_global_features[query_rows],
                local_query,
                xy_normalized,
                detector[:, None],
                detector_log[:, None],
                heatmap[:, None],
                heatmap_log[:, None],
            ],
            axis=1,
        )

        edge_queries: list[int] = []
        edge_tracks: list[int] = []
        edge_columns: list[int] = []
        edge_features: list[np.ndarray] = []
        edge_support: list[np.ndarray] = []
        edge_support_mask: list[np.ndarray] = []
        coarse = np.asarray(self.probe["strategy__coarse_prototype"], dtype=np.float32)[query_rows]
        radio_top2 = np.asarray(self.probe["strategy__all_support_top2_mean"], dtype=np.float32)[query_rows]
        alike_best = np.asarray(self.probe["strategy__alike_support_best"], dtype=np.float32)[query_rows]
        alike_top2 = np.asarray(self.probe["strategy__alike_support_top2_mean"], dtype=np.float32)[query_rows]
        alike_top4 = np.asarray(self.probe["strategy__alike_support_top4_mean"], dtype=np.float32)[query_rows]
        for query_local_row in range(len(query_rows)):
            top_coarse = float(np.max(coarse[query_local_row, valid_candidates[query_local_row]]))
            for column in np.flatnonzero(valid_candidates[query_local_row]).tolist():
                track_id = int(candidate_tracks[query_local_row, column])
                track_local_row = track_position[track_id]
                start, end = self._support_range(track_id)
                rows = self.support_features[start:end]
                similarities = rows @ local_query[query_local_row]
                order = np.argsort(-similarities, kind="mergesort")[: self.max_support_views]
                selected_features = rows[order]
                selected_scores = np.clip(self.support_detector_scores[start:end][order], 0.0, 1.0)
                selected_score_log = np.clip(
                    (np.log10(np.maximum(selected_scores, 1e-8)) + 8.0) / 8.0,
                    0.0,
                    1.0,
                )
                support_values = np.concatenate(
                    [selected_features, selected_scores[:, None], selected_score_log[:, None]],
                    axis=1,
                ).astype(np.float32)
                padded = np.zeros((self.max_support_views, self.support_input_dim), dtype=np.float32)
                mask = np.zeros((self.max_support_views,), dtype=bool)
                padded[: len(order)] = support_values
                mask[: len(order)] = True
                bank_row = int(self.candidates.bank_row_indices[query_rows[query_local_row], column])
                edge_feature = np.asarray(
                    [
                        coarse[query_local_row, column],
                        radio_top2[query_local_row, column],
                        alike_best[query_local_row, column],
                        alike_top2[query_local_row, column],
                        alike_top4[query_local_row, column],
                        float(column) / max(float(self.candidates.top_l - 1), 1.0),
                        top_coarse - float(coarse[query_local_row, column]),
                        float(self.candidates.prototype_ids[query_rows[query_local_row], column]),
                        np.log1p(float(self.landmark_index.observation_counts[bank_row])) / 5.0,
                    ],
                    dtype=np.float32,
                )
                edge_queries.append(int(query_local_row))
                edge_tracks.append(int(track_local_row))
                edge_columns.append(int(column))
                edge_features.append(edge_feature)
                edge_support.append(padded)
                edge_support_mask.append(mask)

        targets = np.full((len(query_rows),), len(episode_tracks), dtype=np.int64)
        for query_local_row, global_row in enumerate(query_rows.tolist()):
            correct = int(self.correct_track_ids[global_row])
            if correct in track_position and np.any(candidate_tracks[query_local_row] == correct):
                targets[query_local_row] = int(track_position[correct])
        episode = LocalAssignmentEpisode(
            query_features=torch.from_numpy(query_features.astype(np.float32)),
            track_features=torch.from_numpy(track_features.astype(np.float32)),
            edge_query_indices=torch.as_tensor(edge_queries, dtype=torch.long),
            edge_track_indices=torch.as_tensor(edge_tracks, dtype=torch.long),
            edge_features=torch.from_numpy(np.stack(edge_features, axis=0)),
            support_features=torch.from_numpy(np.stack(edge_support, axis=0)),
            support_mask=torch.from_numpy(np.stack(edge_support_mask, axis=0)),
            target_track_indices=torch.from_numpy(targets),
            query_rows=torch.from_numpy(query_rows.astype(np.int64)),
            candidate_columns=torch.as_tensor(edge_columns, dtype=torch.long),
        )
        episode.validate()
        self._episode_cache[str(query_id)] = episode
        return episode
