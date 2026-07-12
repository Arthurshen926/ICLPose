"""Grid-local set-to-set assignment episodes from deployable detector proposals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_matcher import LocalAssignmentEpisode
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.query_to_3d_matching import normalize_rows


EDGE_STRATEGIES = (
    "coarse_prototype",
    "alike_support_best",
    "alike_support_mean",
    "alike_support_top2_mean",
    "alike_support_top4_mean",
    "alike_support_logmeanexp_tau0p05",
)


def greedy_unique_assignment_targets(
    residuals: np.ndarray,
    edge_mask: np.ndarray,
    *,
    threshold_px: float,
) -> np.ndarray:
    """Create deterministic one-to-one targets; N tracks denotes query dustbin."""

    values = np.asarray(residuals, dtype=np.float32)
    allowed = np.asarray(edge_mask, dtype=bool)
    if values.ndim != 2 or allowed.shape != values.shape:
        raise ValueError("residuals and edge_mask must have matching shape (Nq, Nt)")
    query_count, track_count = values.shape
    targets = np.full((query_count,), track_count, dtype=np.int64)
    candidate_pairs = np.argwhere(allowed & np.isfinite(values) & (values <= float(threshold_px)))
    if candidate_pairs.size == 0:
        return targets
    pair_values = values[candidate_pairs[:, 0], candidate_pairs[:, 1]]
    order = np.lexsort((candidate_pairs[:, 1], candidate_pairs[:, 0], pair_values))
    used_queries: set[int] = set()
    used_tracks: set[int] = set()
    for position in order.tolist():
        query_index = int(candidate_pairs[position, 0])
        track_index = int(candidate_pairs[position, 1])
        if query_index in used_queries or track_index in used_tracks:
            continue
        targets[query_index] = track_index
        used_queries.add(query_index)
        used_tracks.add(track_index)
    return targets


class DetectorGridAssignmentStore:
    """Validated feature store yielding one spatial grid-cell assignment episode."""

    def __init__(
        self,
        *,
        proposals: Path,
        detector_query_cache: Path,
        support_feature_cache: Path,
        projected_landmark_bank: Path,
        maplet_support_index: Path,
        candidate_top_k: int = 10,
        grid_rows: int = 4,
        grid_cols: int = 4,
        positive_threshold_px: float = 2.0,
        max_support_views: int = 4,
        image_width: int = 1024,
        image_height: int = 576,
        baseline_strategy: str = "alike_support_top2_mean",
    ) -> None:
        if int(candidate_top_k) <= 0 or int(grid_rows) <= 0 or int(grid_cols) <= 0:
            raise ValueError("candidate_top_k and grid dimensions must be positive")
        if float(positive_threshold_px) <= 0.0 or int(max_support_views) <= 0:
            raise ValueError("positive threshold and max support views must be positive")
        self.candidate_top_k = int(candidate_top_k)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.positive_threshold_px = float(positive_threshold_px)
        self.max_support_views = int(max_support_views)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.baseline_strategy = str(baseline_strategy)

        with np.load(Path(proposals), allow_pickle=False) as data:
            self.probe = {key: np.asarray(data[key]) for key in data.files}
        self.query_ids = np.asarray(self.probe["query_ids"]).astype(str)
        self.query_xy = np.asarray(self.probe["xy"], dtype=np.float32)
        self.candidate_residuals = np.asarray(self.probe["candidate_gt_residuals_px"], dtype=np.float32)
        self.nearest_residuals = np.asarray(self.probe["nearest_visible_residuals_px"], dtype=np.float32)
        self.pose_keep_mask = np.asarray(self.probe["pose_keep_mask"], dtype=bool)
        self.candidates = UniqueTrackCandidateSet(
            self.probe["bank_row_indices"],
            self.probe["candidate_track_ids"],
            self.probe["candidate_prototype_ids"],
            self.probe["coarse_scores"],
        )
        self.candidate_valid_mask = self.candidates.bank_row_indices >= 0
        for strategy in EDGE_STRATEGIES:
            key = f"strategy__{strategy}"
            if key not in self.probe or np.asarray(self.probe[key]).shape != self.candidates.coarse_scores.shape:
                raise ValueError(f"proposal artifact is missing strategy: {strategy}")
        self.baseline_scores = np.asarray(
            self.probe[f"strategy__{self.baseline_strategy}"],
            dtype=np.float32,
        )

        with np.load(Path(detector_query_cache), allow_pickle=False) as data:
            query_cache = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            self.query_cache_metadata = json.loads(str(data["metadata_json"].item()))
        cache_ids = np.repeat(
            np.asarray(query_cache["image_ids"]).astype(str),
            np.diff(np.asarray(query_cache["offsets"], dtype=np.int64)),
        )
        if not np.array_equal(cache_ids, self.query_ids) or not np.array_equal(
            np.asarray(query_cache["xy"], dtype=np.float32), self.query_xy
        ):
            raise ValueError("detector query cache and proposal rows differ")
        self.query_local_features, _valid = normalize_rows(
            np.asarray(query_cache["local_descriptors"], dtype=np.float32)
        )
        self.query_global_features, _valid = normalize_rows(
            np.asarray(query_cache["global_descriptors"], dtype=np.float32)
        )
        self.query_detector_scores = np.asarray(query_cache["detector_scores"], dtype=np.float32)
        self.query_detector_dispersions = np.asarray(query_cache["detector_dispersions"], dtype=np.float32)

        with np.load(Path(support_feature_cache), allow_pickle=False) as data:
            support_features = np.asarray(data["support_features"], dtype=np.float32)
            support_scores = np.asarray(data["support_detector_scores"], dtype=np.float32)
            support_tracks = np.asarray(data["support_track_ids"], dtype=np.int64)
        order = np.argsort(support_tracks, kind="stable")
        self.support_track_ids = support_tracks[order]
        self.support_features, self.support_valid = normalize_rows(support_features[order])
        self.support_detector_scores = support_scores[order]
        self.support_unique_tracks, self.support_starts, self.support_counts = np.unique(
            self.support_track_ids,
            return_index=True,
            return_counts=True,
        )
        support_sums = np.add.reduceat(self.support_features, self.support_starts, axis=0)
        self.support_means, _valid = normalize_rows(support_sums / self.support_counts[:, None])

        self.landmark_index, landmark_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
        if str(self.query_cache_metadata.get("descriptor_space_id", "")) != str(
            landmark_metadata.get("descriptor_space_id", "")
        ):
            raise ValueError("query and landmark descriptor spaces differ")
        valid_candidates = self.candidate_valid_mask
        if not np.array_equal(
            self.candidates.track_ids[valid_candidates],
            self.landmark_index.track_ids[self.candidates.bank_row_indices[valid_candidates]],
        ):
            raise ValueError("proposal and landmark bank rows differ")
        bank_order = np.argsort(self.landmark_index.track_ids, kind="stable")
        bank_tracks = self.landmark_index.track_ids[bank_order]
        bank_features, _valid = normalize_rows(self.landmark_index.features[bank_order])
        self.bank_unique_tracks, starts, counts = np.unique(
            bank_tracks,
            return_index=True,
            return_counts=True,
        )
        feature_sums = np.add.reduceat(bank_features, starts, axis=0)
        self.bank_track_features, _valid = normalize_rows(feature_sums / counts[:, None])
        first_rows = bank_order[starts]
        self.bank_track_xyz = self.landmark_index.xyz[first_rows].astype(np.float32)
        self.bank_track_observation_counts = np.maximum.reduceat(
            self.landmark_index.observation_counts[bank_order], starts
        ).astype(np.float32)
        self.bank_track_variances = (
            np.add.reduceat(self.landmark_index.mean_variances[bank_order], starts) / counts
        ).astype(np.float32)
        self.bank_track_reprojection = (
            np.add.reduceat(self.landmark_index.reprojection_errors[bank_order], starts) / counts
        ).astype(np.float32)
        self.bank_track_ambiguities = (
            np.add.reduceat(self.landmark_index.feature_ambiguities[bank_order], starts) / counts
        ).astype(np.float32)

        self.maplet_index, _maplet_metadata = load_local_maplet_support_index_npz(Path(maplet_support_index))
        if not np.array_equal(self.bank_unique_tracks, self.maplet_index.anchor_track_ids):
            raise ValueError("maplet and landmark banks contain different physical tracks")
        candidate_tracks = np.unique(self.candidates.track_ids[self.candidate_valid_mask])
        support_positions = np.searchsorted(self.support_unique_tracks, candidate_tracks)
        if np.any(support_positions >= len(self.support_unique_tracks)) or not np.array_equal(
            self.support_unique_tracks[support_positions], candidate_tracks
        ):
            raise ValueError("support feature cache does not cover every proposal track")

        self.rows_by_query: dict[str, np.ndarray] = {}
        for query_id in dict.fromkeys(self.query_ids.tolist()):
            self.rows_by_query[str(query_id)] = np.flatnonzero(self.query_ids == str(query_id))
        self.unique_query_ids = tuple(self.rows_by_query)
        self.episode_keys: list[tuple[str, int, int]] = []
        self.episode_keys_by_query: dict[str, list[tuple[str, int, int]]] = {}
        for query_id in self.unique_query_ids:
            rows = self.rows_by_query[query_id]
            cols = np.clip(
                np.floor(self.query_xy[rows, 0] / max(float(self.image_width), 1.0) * self.grid_cols).astype(int),
                0,
                self.grid_cols - 1,
            )
            grid_rows = np.clip(
                np.floor(self.query_xy[rows, 1] / max(float(self.image_height), 1.0) * self.grid_rows).astype(int),
                0,
                self.grid_rows - 1,
            )
            keys = []
            for row_index in range(self.grid_rows):
                for col_index in range(self.grid_cols):
                    if np.any((grid_rows == row_index) & (cols == col_index)):
                        key = (str(query_id), int(row_index), int(col_index))
                        self.episode_keys.append(key)
                        keys.append(key)
            self.episode_keys_by_query[str(query_id)] = keys
        self._episode_cache: dict[tuple[str, int, int], LocalAssignmentEpisode] = {}

    @property
    def query_input_dim(self) -> int:
        return int(self.query_local_features.shape[1] + self.query_global_features.shape[1] + 8)

    @property
    def track_input_dim(self) -> int:
        return int(
            self.bank_track_features.shape[1]
            + self.support_means.shape[1]
            + self.maplet_index.maplets.context_features.shape[1]
            + 9
        )

    @property
    def support_input_dim(self) -> int:
        return int(self.support_features.shape[1] + 2)

    @property
    def edge_input_dim(self) -> int:
        return 12

    def _track_positions(self, tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(tracks, dtype=np.int64)
        bank = np.searchsorted(self.bank_unique_tracks, values)
        support = np.searchsorted(self.support_unique_tracks, values)
        maplet = np.searchsorted(self.maplet_index.anchor_track_ids, values)
        if not np.array_equal(self.bank_unique_tracks[bank], values):
            raise KeyError("episode track missing from landmark bank")
        if not np.array_equal(self.support_unique_tracks[support], values):
            raise KeyError("episode track missing from support cache")
        if not np.array_equal(self.maplet_index.anchor_track_ids[maplet], values):
            raise KeyError("episode track missing from maplet bank")
        return bank, support, maplet

    def _support_range(self, track_id: int) -> tuple[int, int]:
        position = int(np.searchsorted(self.support_unique_tracks, int(track_id)))
        if position >= len(self.support_unique_tracks) or int(self.support_unique_tracks[position]) != int(track_id):
            raise KeyError(f"support track missing: {track_id}")
        start = int(self.support_starts[position])
        return start, start + int(self.support_counts[position])

    def _episode_rows(self, key: tuple[str, int, int]) -> np.ndarray:
        query_id, grid_row, grid_col = key
        rows = self.rows_by_query[str(query_id)]
        cols = np.clip(
            np.floor(self.query_xy[rows, 0] / max(float(self.image_width), 1.0) * self.grid_cols).astype(int),
            0,
            self.grid_cols - 1,
        )
        grid_rows = np.clip(
            np.floor(self.query_xy[rows, 1] / max(float(self.image_height), 1.0) * self.grid_rows).astype(int),
            0,
            self.grid_rows - 1,
        )
        return rows[(grid_rows == int(grid_row)) & (cols == int(grid_col))]

    def episode(self, key: tuple[str, int, int]) -> LocalAssignmentEpisode:
        cached = self._episode_cache.get(key)
        if cached is not None:
            return cached
        query_rows = self._episode_rows(key)
        selected_columns: list[np.ndarray] = []
        stable_tracks: list[int] = []
        seen_tracks: set[int] = set()
        for row in query_rows.tolist():
            valid = np.flatnonzero(self.candidate_valid_mask[row] & np.isfinite(self.baseline_scores[row]))
            order = valid[np.argsort(-self.baseline_scores[row, valid], kind="stable")]
            columns = order[: min(self.candidate_top_k, len(order))].astype(np.int64)
            if len(columns) == 0:
                raise ValueError("every detector query node must retain at least one candidate")
            selected_columns.append(columns)
            for track_id in self.candidates.track_ids[row, columns].tolist():
                value = int(track_id)
                if value not in seen_tracks:
                    seen_tracks.add(value)
                    stable_tracks.append(value)
        episode_tracks = np.asarray(stable_tracks, dtype=np.int64)
        track_position = {int(track_id): index for index, track_id in enumerate(stable_tracks)}
        bank_positions, support_positions, maplet_positions = self._track_positions(episode_tracks)
        xyz = self.bank_track_xyz[bank_positions]
        center = np.median(xyz, axis=0)
        scale = max(float(np.median(np.linalg.norm(xyz - center[None], axis=1))), 1.0)
        xyz_normalized = (xyz - center[None]) / scale
        quality = np.stack(
            [
                np.log1p(self.bank_track_observation_counts[bank_positions]) / 5.0,
                np.log1p(np.maximum(self.bank_track_variances[bank_positions], 0.0)),
                np.clip(self.bank_track_reprojection[bank_positions] / 5.0, 0.0, 2.0),
                np.clip(self.bank_track_ambiguities[bank_positions], 0.0, 2.0),
                np.clip(self.maplet_index.maplets.context_radius[maplet_positions] / 5.0, 0.0, 2.0),
                np.clip(self.maplet_index.maplets.covisibility_strength[maplet_positions] / 10.0, 0.0, 2.0),
            ],
            axis=1,
        ).astype(np.float32)
        track_features = np.concatenate(
            [
                self.bank_track_features[bank_positions],
                self.support_means[support_positions],
                self.maplet_index.maplets.context_features[maplet_positions],
                xyz_normalized.astype(np.float32),
                quality,
            ],
            axis=1,
        )

        xy = self.query_xy[query_rows]
        global_xy = np.stack(
            [2.0 * xy[:, 0] / max(float(self.image_width - 1), 1.0) - 1.0,
             2.0 * xy[:, 1] / max(float(self.image_height - 1), 1.0) - 1.0],
            axis=1,
        )
        cell_width = float(self.image_width) / self.grid_cols
        cell_height = float(self.image_height) / self.grid_rows
        cell_xy = np.stack(
            [
                2.0 * ((xy[:, 0] - key[2] * cell_width) / max(cell_width, 1.0)) - 1.0,
                2.0 * ((xy[:, 1] - key[1] * cell_height) / max(cell_height, 1.0)) - 1.0,
            ],
            axis=1,
        )
        detector = np.clip(self.query_detector_scores[query_rows], 0.0, 1.0)
        detector_log = np.log10(np.maximum(detector, 1e-8))
        dispersion = np.maximum(self.query_detector_dispersions[query_rows], 0.0)
        dispersion_log = np.log1p(dispersion)
        query_features = np.concatenate(
            [
                self.query_local_features[query_rows],
                self.query_global_features[query_rows],
                global_xy.astype(np.float32),
                cell_xy.astype(np.float32),
                detector[:, None],
                detector_log[:, None],
                dispersion[:, None],
                dispersion_log[:, None],
            ],
            axis=1,
        )

        strategy_values = {
            name: np.asarray(self.probe[f"strategy__{name}"], dtype=np.float32)
            for name in EDGE_STRATEGIES
        }
        edge_queries: list[int] = []
        edge_tracks: list[int] = []
        edge_columns: list[int] = []
        edge_features: list[np.ndarray] = []
        edge_support: list[np.ndarray] = []
        edge_support_mask: list[np.ndarray] = []
        target_residuals = np.full((len(query_rows), len(episode_tracks)), np.inf, dtype=np.float32)
        edge_mask = np.zeros_like(target_residuals, dtype=bool)
        for local_query, (global_row, columns) in enumerate(zip(query_rows.tolist(), selected_columns)):
            coarse_max = float(np.max(strategy_values["coarse_prototype"][global_row, columns]))
            local_max = float(np.max(strategy_values["alike_support_top2_mean"][global_row, columns]))
            for column in columns.tolist():
                track_id = int(self.candidates.track_ids[global_row, column])
                local_track = int(track_position[track_id])
                start, end = self._support_range(track_id)
                support_rows = self.support_features[start:end]
                similarities = np.einsum(
                    "nd,d->n",
                    support_rows,
                    self.query_local_features[global_row],
                    optimize=False,
                )
                order = np.argsort(-similarities, kind="stable")[: self.max_support_views]
                selected_features = support_rows[order]
                selected_scores = np.clip(self.support_detector_scores[start:end][order], 0.0, 1.0)
                selected_log = np.log10(np.maximum(selected_scores, 1e-8))
                support_values = np.concatenate(
                    [selected_features, selected_scores[:, None], selected_log[:, None]], axis=1
                ).astype(np.float32)
                padded = np.zeros((self.max_support_views, self.support_input_dim), dtype=np.float32)
                mask = np.zeros((self.max_support_views,), dtype=bool)
                padded[: len(order)] = support_values
                mask[: len(order)] = True
                coarse = float(strategy_values["coarse_prototype"][global_row, column])
                local_top2 = float(strategy_values["alike_support_top2_mean"][global_row, column])
                bank_row = int(self.candidates.bank_row_indices[global_row, column])
                feature = np.asarray(
                    [
                        coarse,
                        strategy_values["alike_support_best"][global_row, column],
                        strategy_values["alike_support_mean"][global_row, column],
                        local_top2,
                        strategy_values["alike_support_top4_mean"][global_row, column],
                        strategy_values["alike_support_logmeanexp_tau0p05"][global_row, column],
                        float(column) / max(float(self.candidates.top_l - 1), 1.0),
                        coarse_max - coarse,
                        local_max - local_top2,
                        np.log1p(float(self.probe["support_observation_counts"][global_row, column])) / 5.0,
                        float(self.candidates.prototype_ids[global_row, column]),
                        float(self.landmark_index.feature_ambiguities[bank_row]),
                    ],
                    dtype=np.float32,
                )
                edge_queries.append(int(local_query))
                edge_tracks.append(local_track)
                edge_columns.append(int(column))
                edge_features.append(feature)
                edge_support.append(padded)
                edge_support_mask.append(mask)
                edge_mask[local_query, local_track] = True
                target_residuals[local_query, local_track] = min(
                    target_residuals[local_query, local_track],
                    float(self.candidate_residuals[global_row, column]),
                )
        targets = greedy_unique_assignment_targets(
            target_residuals,
            edge_mask,
            threshold_px=self.positive_threshold_px,
        )
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
        self._episode_cache[key] = episode
        return episode
