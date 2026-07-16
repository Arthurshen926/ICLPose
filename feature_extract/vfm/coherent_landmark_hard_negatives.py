"""Attach train-only coherent wrong-pose track identities to landmark rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


EXPECTED_FORMAT = "pose_conditioned_system_hard_modes_v2"


def _canonical_query_id(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _csr_row_values(
    offsets: np.ndarray | None,
    values: np.ndarray | None,
    row: int,
) -> np.ndarray:
    if offsets is None or values is None:
        return np.zeros((0,), dtype=np.int64)
    begin = int(np.asarray(offsets, dtype=np.int64)[int(row)])
    end = int(np.asarray(offsets, dtype=np.int64)[int(row) + 1])
    return np.asarray(values, dtype=np.int64)[begin:end]


@dataclass(frozen=True)
class CoherentLandmarkHardNegativeIndex:
    query_xy_by_id: dict[str, np.ndarray]
    track_ids_by_id: dict[str, tuple[np.ndarray, ...]]
    mode_ids_by_id: dict[str, tuple[np.ndarray, ...]]
    metadata: dict[str, object]

    @classmethod
    def from_pose_mode_artifact(
        cls,
        artifact_path: Path,
        proposals_path: Path,
    ) -> "CoherentLandmarkHardNegativeIndex":
        artifact = Path(artifact_path)
        proposals = Path(proposals_path)
        with np.load(artifact, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if str(metadata.get("format", "")) != EXPECTED_FORMAT:
                raise ValueError("coherent hard-negative artifact has an unsupported format")
            if not bool(metadata.get("training_only_target_artifact", False)):
                raise ValueError("coherent hard-negative artifact is not marked training-only")
            if bool(metadata.get("pose_or_ground_truth_used_for_hypothesis_generation", True)):
                raise ValueError("coherent hard-negative hypotheses used pose or ground truth")
            if set(map(str, metadata.get("split_names", []))) != {"train"}:
                raise ValueError("coherent hard-negative artifact must contain only train queries")
            selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
            selected_columns = np.asarray(data["selected_columns"], dtype=np.int64)
            hard_mask = np.asarray(
                data["hard_negative_mask_TARGET_ONLY"], dtype=bool
            )
            query_ids = np.asarray(data["query_ids"]).astype(str)
            mode_ids = (
                np.asarray(data["hard_mode_ids_TARGET_ONLY"], dtype=np.int64)
                if "hard_mode_ids_TARGET_ONLY" in data
                else None
            )
            mode_candidate_mask = (
                np.asarray(
                    data["hard_mode_candidate_mask_TARGET_ONLY"], dtype=bool
                )
                if "hard_mode_candidate_mask_TARGET_ONLY" in data
                else None
            )
        expected_proposals_hash = str(
            dict(metadata.get("inputs", {})).get("proposals_sha256", "")
        )
        actual_proposals_hash = file_sha256_short(proposals)
        if expected_proposals_hash != actual_proposals_hash:
            raise ValueError(
                "coherent hard-negative artifact/proposals lineage mismatch: "
                f"expected={expected_proposals_hash!r}, actual={actual_proposals_hash!r}"
            )
        with np.load(proposals, allow_pickle=False) as data:
            proposal_query_ids = np.asarray(data["query_ids"]).astype(str)
            proposal_xy = np.asarray(data["xy"], dtype=np.float32)
            proposal_track_ids = np.asarray(data["candidate_track_ids"], dtype=np.int64)
        if (
            selected_columns.shape != hard_mask.shape
            or selected_rows.shape[0] != selected_columns.shape[0]
            or query_ids.shape[0] != selected_rows.shape[0]
        ):
            raise ValueError("coherent hard-negative artifact arrays are misaligned")
        if np.any(selected_rows < 0) or np.any(selected_rows >= proposal_xy.shape[0]):
            raise ValueError("coherent hard-negative artifact references invalid proposal rows")
        if not np.array_equal(query_ids, proposal_query_ids[selected_rows]):
            raise ValueError("coherent hard-negative query ids do not match proposals")
        if mode_ids is not None or mode_candidate_mask is not None:
            if mode_ids is None or mode_candidate_mask is None:
                raise ValueError("coherent hard-mode membership arrays are incomplete")
            if mode_candidate_mask.shape[:2] != mode_ids.shape or (
                mode_candidate_mask.shape[0] != selected_rows.shape[0]
                or mode_candidate_mask.shape[2] != selected_columns.shape[1]
            ):
                raise ValueError("coherent hard-mode membership arrays are misaligned")

        xy_by_id: dict[str, list[np.ndarray]] = {}
        tracks_by_id: dict[str, list[np.ndarray]] = {}
        modes_by_id: dict[str, list[np.ndarray]] = {}
        for artifact_row, proposal_row in enumerate(selected_rows.tolist()):
            columns = selected_columns[artifact_row]
            valid_columns = columns >= 0
            if np.any(columns[valid_columns] >= proposal_track_ids.shape[1]):
                raise ValueError("coherent hard-negative artifact references invalid columns")
            if mode_ids is None:
                selected_columns_row = columns[hard_mask[artifact_row] & valid_columns]
                track_ids = proposal_track_ids[int(proposal_row), selected_columns_row]
                track_ids = np.unique(track_ids[track_ids >= 0]).astype(
                    np.int64, copy=False
                )
                track_modes = np.full(track_ids.shape, -1, dtype=np.int64)
            else:
                incidences: list[tuple[int, int]] = []
                for slot, mode_id in enumerate(mode_ids[artifact_row].tolist()):
                    if int(mode_id) < 0:
                        continue
                    member_columns = columns[
                        mode_candidate_mask[artifact_row, slot] & valid_columns
                    ]
                    for track in proposal_track_ids[
                        int(proposal_row), member_columns
                    ].tolist():
                        if int(track) >= 0:
                            incidences.append((int(track), int(mode_id)))
                incidences = list(dict.fromkeys(incidences))
                track_ids = np.asarray(
                    [track for track, _mode in incidences], dtype=np.int64
                )
                track_modes = np.asarray(
                    [mode for _track, mode in incidences], dtype=np.int64
                )
            if track_ids.size == 0:
                continue
            query_id = _canonical_query_id(query_ids[artifact_row])
            xy_by_id.setdefault(query_id, []).append(proposal_xy[int(proposal_row)])
            tracks_by_id.setdefault(query_id, []).append(track_ids)
            modes_by_id.setdefault(query_id, []).append(track_modes)
        return cls(
            query_xy_by_id={
                key: np.asarray(value, dtype=np.float32).reshape(-1, 2)
                for key, value in xy_by_id.items()
            },
            track_ids_by_id={key: tuple(value) for key, value in tracks_by_id.items()},
            mode_ids_by_id={key: tuple(value) for key, value in modes_by_id.items()},
            metadata={
                "format": "coherent_landmark_hard_negative_index_v1",
                "source_artifact": str(artifact),
                "source_artifact_sha256": file_sha256_short(artifact),
                "proposals": str(proposals),
                "proposals_sha256": actual_proposals_hash,
                "query_count": int(len(xy_by_id)),
                "group_count": int(sum(len(value) for value in xy_by_id.values())),
            },
        )

    def nearest_tracks(
        self,
        query_id: str,
        xy: np.ndarray,
        *,
        max_distance_px: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        key = _canonical_query_id(query_id)
        points = self.query_xy_by_id.get(key)
        if points is None or points.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                float("inf"),
            )
        distances = np.linalg.norm(points - np.asarray(xy, dtype=np.float32)[None], axis=1)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if distance > float(max_distance_px):
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                distance,
            )
        return (
            self.track_ids_by_id[key][nearest].copy(),
            self.mode_ids_by_id[key][nearest].copy(),
            distance,
        )


def attach_coherent_landmark_hard_negatives(
    samples,
    index: CoherentLandmarkHardNegativeIndex,
    *,
    max_distance_px: float,
):
    """Return a training set carrying coherent-negative CSR per landmark row."""

    if samples.landmark_track_ids is None:
        raise ValueError("coherent hard negatives require landmark retrieval rows")
    if samples.pair_query_ids is None or samples.landmark_sample_pair_indices is None:
        raise ValueError("coherent hard negatives require pair query ids")
    pair_query_ids = np.asarray(samples.pair_query_ids).astype(str)
    landmark_pairs = np.asarray(samples.landmark_sample_pair_indices, dtype=np.int64)
    landmark_xy = np.asarray(samples.landmark_query_xy, dtype=np.float32)
    targets = np.asarray(samples.landmark_track_ids, dtype=np.int64)
    offsets = np.zeros((targets.size + 1,), dtype=np.int64)
    pieces: list[np.ndarray] = []
    mode_pieces: list[np.ndarray] = []
    matched_rows = 0
    for row, (pair, xy, target) in enumerate(
        zip(landmark_pairs.tolist(), landmark_xy, targets.tolist())
    ):
        tracks, modes, _distance = index.nearest_tracks(
            pair_query_ids[int(pair)], xy, max_distance_px=float(max_distance_px)
        )
        forbidden = {int(target)}
        forbidden.update(
            _csr_row_values(
                samples.landmark_known_positive_offsets,
                samples.landmark_known_positive_track_ids,
                row,
            ).tolist()
        )
        forbidden.update(
            _csr_row_values(
                samples.landmark_strict_positive_offsets,
                samples.landmark_strict_positive_track_ids,
                row,
            ).tolist()
        )
        keep = np.asarray(
            [int(value) not in forbidden for value in tracks.tolist()], dtype=bool
        )
        tracks = tracks[keep]
        modes = modes[keep]
        if tracks.size:
            # A track may support more than one coherent wrong-pose mode.  The
            # structured loss needs every (track, mode) incidence; reducing by
            # track alone silently collapses the configuration supervision back
            # to an unstructured per-row negative set.
            pairs = np.stack([tracks, modes], axis=1)
            _, unique_indices = np.unique(pairs, axis=0, return_index=True)
            unique_indices.sort()
            tracks = tracks[unique_indices]
            modes = modes[unique_indices]
            pieces.append(tracks)
            mode_pieces.append(modes)
            matched_rows += 1
        offsets[row + 1] = offsets[row] + int(tracks.size)
    track_ids = (
        np.concatenate(pieces, axis=0)
        if pieces
        else np.zeros((0,), dtype=np.int64)
    )
    mode_ids = (
        np.concatenate(mode_pieces, axis=0)
        if mode_pieces
        else np.zeros((0,), dtype=np.int64)
    )
    result = replace(
        samples,
        landmark_coherent_hard_negative_offsets=offsets,
        landmark_coherent_hard_negative_track_ids=track_ids,
        landmark_coherent_hard_negative_mode_ids=(
            mode_ids if mode_ids.size and np.all(mode_ids >= 0) else None
        ),
    )
    audit = {
        **dict(index.metadata),
        "max_distance_px": float(max_distance_px),
        "landmark_row_count": int(targets.size),
        "matched_landmark_row_count": int(matched_rows),
        "matched_landmark_row_fraction": float(matched_rows / max(1, targets.size)),
        "coherent_track_incidence_count": int(track_ids.size),
    }
    return result, audit
