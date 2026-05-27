"""COLMAP landmark visibility and coverage-balanced observation sampling."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapTrackObservation
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


_VISIBILITY_FORMAT = "vfm_landmark_visibility_v1"


@dataclass(frozen=True)
class LandmarkVisibilityIndex:
    image_to_track_ids: Mapping[str, frozenset[int]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_to_track_ids",
            {
                str(image_id): frozenset(int(track_id) for track_id in track_ids)
                for image_id, track_ids in self.image_to_track_ids.items()
            },
        )

    @classmethod
    def from_observations(cls, observations: Iterable[ColmapTrackObservation]) -> "LandmarkVisibilityIndex":
        image_to_tracks: dict[str, set[int]] = defaultdict(set)
        for obs in observations:
            image_to_tracks[str(obs.image_id)].add(int(obs.track_id))
        return cls({image_id: frozenset(track_ids) for image_id, track_ids in image_to_tracks.items()})

    def visible_tracks(self, image_ids: Sequence[str] | set[str]) -> set[int]:
        visible: set[int] = set()
        for image_id in image_ids:
            visible.update(self.image_to_track_ids.get(str(image_id), frozenset()))
        return visible

    @property
    def image_count(self) -> int:
        return len(self.image_to_track_ids)

    @property
    def observation_count(self) -> int:
        return int(sum(len(track_ids) for track_ids in self.image_to_track_ids.values()))

    @property
    def track_count(self) -> int:
        all_tracks: set[int] = set()
        for track_ids in self.image_to_track_ids.values():
            all_tracks.update(track_ids)
        return len(all_tracks)

    def save_npz(self, path: Path, metadata: Mapping[str, object] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image_ids = np.asarray(sorted(self.image_to_track_ids), dtype=str)
        offsets = [0]
        flat_track_ids: list[int] = []
        for image_id in image_ids.tolist():
            tracks = sorted(self.image_to_track_ids[str(image_id)])
            flat_track_ids.extend(tracks)
            offsets.append(len(flat_track_ids))
        payload = {
            "format": _VISIBILITY_FORMAT,
            "image_count": int(len(image_ids)),
            "observation_count": int(len(flat_track_ids)),
            "track_count": int(self.track_count),
            "metadata": dict(metadata or {}),
        }
        np.savez_compressed(
            output,
            metadata=np.asarray(json.dumps(payload, sort_keys=True)),
            image_ids=image_ids,
            offsets=np.asarray(offsets, dtype=np.int64),
            track_ids=np.asarray(flat_track_ids, dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "LandmarkVisibilityIndex":
        with np.load(Path(path)) as data:
            payload = json.loads(str(data["metadata"].item()))
            if payload.get("format") != _VISIBILITY_FORMAT:
                raise ValueError(f"unsupported visibility index format in {path}")
            image_ids = [str(item) for item in data["image_ids"].tolist()]
            offsets = data["offsets"].astype(np.int64)
            track_ids = data["track_ids"].astype(np.int64)
        mapping = {}
        for idx, image_id in enumerate(image_ids):
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            mapping[image_id] = frozenset(int(track_id) for track_id in track_ids[start:end])
        return cls(mapping)


def filter_landmarks_by_visibility(
    landmark_index: LandmarkMapIndex,
    visibility_index: LandmarkVisibilityIndex,
    reference_images: Sequence[str] | set[str],
) -> tuple[LandmarkMapIndex, dict[str, float | int]]:
    full_visible = visibility_index.visible_tracks(reference_images)
    if not full_visible:
        subset = landmark_index.subset([])
        return subset, {
            "full_visible_tracks": 0,
            "bank_visible_tracks": 0,
            "bank_visibility_coverage": 0.0,
        }
    mask = np.asarray([int(track_id) in full_visible for track_id in landmark_index.track_ids], dtype=bool)
    subset = landmark_index.subset(mask)
    bank_visible = int(len(subset))
    full_count = int(len(full_visible))
    return subset, {
        "full_visible_tracks": full_count,
        "bank_visible_tracks": bank_visible,
        "bank_visibility_coverage": float(bank_visible / max(full_count, 1)),
    }


def count_projected_landmarks(
    landmark_index: LandmarkMapIndex,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> int:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points = np.asarray(landmark_index.xyz, dtype=np.float64)
    if points.size == 0:
        return 0
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (pose @ homogeneous.T).T
    z = camera_points[:, 2]
    valid_depth = z > 1e-6
    x_norm = camera_points[:, 0] / np.maximum(z, 1e-12)
    y_norm = camera_points[:, 1] / np.maximum(z, 1e-12)
    if camera.model_id == 0:
        f, cx, cy = camera.params[:3]
        x = f * x_norm + cx
        y = f * y_norm + cy
    elif camera.model_id == 1:
        fx, fy, cx, cy = camera.params[:4]
        x = fx * x_norm + cx
        y = fy * y_norm + cy
    elif camera.model_id == 2:
        f, cx, cy, k = camera.params[:4]
        radial = 1.0 + float(k) * (x_norm * x_norm + y_norm * y_norm)
        x = f * x_norm * radial + cx
        y = f * y_norm * radial + cy
    else:
        raise ValueError(f"unsupported camera model id for projection: {camera.model_id}")
    visible = (
        valid_depth
        & (x >= 0.0)
        & (x <= float(camera.width - 1))
        & (y >= 0.0)
        & (y <= float(camera.height - 1))
    )
    return int(np.sum(visible))


def _track_observation_sample(
    track_observations: Sequence[ColmapTrackObservation],
    primary_image_id: str,
    observations_per_track: int,
) -> list[ColmapTrackObservation]:
    ordered = sorted(
        track_observations,
        key=lambda obs: (0 if obs.image_id == primary_image_id else 1, obs.image_id, obs.point2d_idx),
    )
    if len(ordered) <= observations_per_track:
        return list(ordered)
    primary = ordered[0]
    remaining = ordered[1:]
    need = max(int(observations_per_track) - 1, 0)
    if need <= 0:
        return [primary]
    if len(remaining) <= need:
        return [primary] + list(remaining)
    indices = np.linspace(0, len(remaining) - 1, need).round().astype(np.int64)
    return [primary] + [remaining[int(idx)] for idx in indices]


def coverage_balanced_track_observations(
    observations: Iterable[ColmapTrackObservation],
    max_observations: int,
    observations_per_track: int = 3,
    min_track_observations: int = 2,
    seed: int = 0,
) -> list[ColmapTrackObservation]:
    """Select track observations by image-balanced track coverage.

    The sampler chooses track ids in round-robin image order, then includes a
    small, deterministic multi-view sample for each chosen track. This avoids
    prefix-truncation bias while preserving at least two observations per track
    for feature aggregation.
    """

    if max_observations <= 0:
        return list(observations)
    if observations_per_track <= 0:
        raise ValueError("observations_per_track must be positive")
    if min_track_observations <= 0:
        raise ValueError("min_track_observations must be positive")
    observation_list = list(observations)
    by_track: dict[int, list[ColmapTrackObservation]] = defaultdict(list)
    image_to_tracks: dict[str, list[int]] = defaultdict(list)
    for obs in observation_list:
        by_track[int(obs.track_id)].append(obs)
        image_to_tracks[str(obs.image_id)].append(int(obs.track_id))
    eligible_tracks = {
        track_id
        for track_id, track_observations in by_track.items()
        if len(track_observations) >= int(min_track_observations)
    }
    rng = np.random.default_rng(int(seed))
    image_track_lists: dict[str, list[int]] = {}
    for image_id, track_ids in image_to_tracks.items():
        unique = np.asarray(sorted(set(track_ids).intersection(eligible_tracks)), dtype=np.int64)
        if unique.size:
            rng.shuffle(unique)
        image_track_lists[image_id] = [int(track_id) for track_id in unique.tolist()]
    cursors = {image_id: 0 for image_id in image_track_lists}
    image_ids = sorted(image_track_lists)
    selected_tracks: set[int] = set()
    selected_observations: list[ColmapTrackObservation] = []
    while len(selected_observations) < max_observations:
        progressed = False
        for image_id in image_ids:
            tracks = image_track_lists[image_id]
            cursor = cursors[image_id]
            while cursor < len(tracks) and tracks[cursor] in selected_tracks:
                cursor += 1
            cursors[image_id] = cursor
            if cursor >= len(tracks):
                continue
            track_id = int(tracks[cursor])
            cursors[image_id] += 1
            selected_tracks.add(track_id)
            sample = _track_observation_sample(by_track[track_id], image_id, int(observations_per_track))
            remaining_budget = int(max_observations) - len(selected_observations)
            selected_observations.extend(sample[:remaining_budget])
            progressed = True
            if len(selected_observations) >= max_observations:
                break
        if not progressed:
            break
    return selected_observations
