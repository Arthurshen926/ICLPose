"""Lift selected 2D features into explicit 3D track features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    image_id: str
    feature: np.ndarray
    visible: bool
    geometry_valid: bool
    utility: float = 1.0
    camera_center: np.ndarray | None = None
    viewing_ray: np.ndarray | None = None


@dataclass(frozen=True)
class TrackFeature:
    track_id: int
    mean_feature: np.ndarray
    variance: np.ndarray
    observation_count: int
    mean_utility: float
    observation_image_ids: Tuple[str, ...] = ()

    @property
    def mean_variance(self) -> float:
        return float(np.mean(self.variance))


@dataclass(frozen=True)
class SelectedTrackFeatureBank:
    tracks: Mapping[int, TrackFeature]
    feature_dim: int

    def __len__(self) -> int:
        return len(self.tracks)


@dataclass(frozen=True)
class TrackBankMapabilitySummary:
    track_count: int
    feature_dim: int
    mean_observation_count: float
    mean_track_variance: float
    mean_utility: float


def aggregate_selected_tracks(
    observations: Iterable[TrackObservation],
    min_observations: int = 2,
) -> SelectedTrackFeatureBank:
    """Aggregate valid, visible selected-feature observations by 3D track."""

    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    grouped: Dict[int, list[TrackObservation]] = {}
    feature_dim = None
    for obs in observations:
        if not obs.visible or not obs.geometry_valid:
            continue
        feature = np.asarray(obs.feature, dtype=np.float32).reshape(-1)
        if feature_dim is None:
            feature_dim = int(feature.size)
        elif feature.size != feature_dim:
            raise ValueError("all features must have the same dimension")
        grouped.setdefault(obs.track_id, []).append(
            TrackObservation(
                track_id=obs.track_id,
                image_id=obs.image_id,
                feature=feature,
                visible=True,
                geometry_valid=True,
                utility=float(obs.utility),
                camera_center=(
                    None
                    if obs.camera_center is None
                    else np.asarray(obs.camera_center, dtype=np.float64).reshape(3)
                ),
                viewing_ray=(
                    None
                    if obs.viewing_ray is None
                    else np.asarray(obs.viewing_ray, dtype=np.float64).reshape(3)
                ),
            )
        )

    tracks: Dict[int, TrackFeature] = {}
    for track_id, track_obs in grouped.items():
        if len(track_obs) < min_observations:
            continue
        matrix = np.stack([obs.feature for obs in track_obs], axis=0)
        utilities = np.asarray([obs.utility for obs in track_obs], dtype=np.float32)
        tracks[track_id] = TrackFeature(
            track_id=track_id,
            mean_feature=matrix.mean(axis=0),
            variance=matrix.var(axis=0),
            observation_count=int(matrix.shape[0]),
            mean_utility=float(utilities.mean()),
            observation_image_ids=tuple(sorted({obs.image_id for obs in track_obs})),
        )

    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=int(feature_dim or 0))


def save_selected_track_bank_npz(bank: SelectedTrackFeatureBank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    track_ids = np.asarray(sorted(bank.tracks), dtype=np.int64)
    if len(track_ids) == 0:
        mean_features = np.zeros((0, bank.feature_dim), dtype=np.float32)
        variances = np.zeros((0, bank.feature_dim), dtype=np.float32)
        observation_counts = np.zeros((0,), dtype=np.int64)
        mean_utilities = np.zeros((0,), dtype=np.float32)
        observation_image_ids = np.zeros((0,), dtype=str)
    else:
        ordered = [bank.tracks[int(track_id)] for track_id in track_ids]
        mean_features = np.stack([track.mean_feature for track in ordered], axis=0).astype(np.float32)
        variances = np.stack([track.variance for track in ordered], axis=0).astype(np.float32)
        observation_counts = np.asarray([track.observation_count for track in ordered], dtype=np.int64)
        mean_utilities = np.asarray([track.mean_utility for track in ordered], dtype=np.float32)
        observation_image_ids = np.asarray(
            [json.dumps(list(track.observation_image_ids), sort_keys=True) for track in ordered],
            dtype=str,
        )
    np.savez_compressed(
        path,
        track_ids=track_ids,
        mean_features=mean_features,
        variances=variances,
        observation_counts=observation_counts,
        mean_utilities=mean_utilities,
        observation_image_ids=observation_image_ids,
        feature_dim=np.asarray(bank.feature_dim, dtype=np.int64),
    )


def load_selected_track_bank_npz(path: Path) -> SelectedTrackFeatureBank:
    with np.load(path) as data:
        track_ids = data["track_ids"].astype(np.int64)
        mean_features = data["mean_features"].astype(np.float32)
        variances = data["variances"].astype(np.float32)
        observation_counts = data["observation_counts"].astype(np.int64)
        mean_utilities = data["mean_utilities"].astype(np.float32)
        if "observation_image_ids" in data:
            observation_image_ids = tuple(str(item) for item in data["observation_image_ids"].tolist())
        else:
            observation_image_ids = tuple("" for _ in track_ids)
        feature_dim = int(data["feature_dim"])
    tracks: Dict[int, TrackFeature] = {}
    for idx, track_id in enumerate(track_ids):
        if observation_image_ids[idx]:
            track_observation_image_ids = tuple(str(item) for item in json.loads(observation_image_ids[idx]))
        else:
            track_observation_image_ids = ()
        tracks[int(track_id)] = TrackFeature(
            track_id=int(track_id),
            mean_feature=mean_features[idx],
            variance=variances[idx],
            observation_count=int(observation_counts[idx]),
            mean_utility=float(mean_utilities[idx]),
            observation_image_ids=track_observation_image_ids,
        )
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=feature_dim)


def mapability_summary(bank: SelectedTrackFeatureBank) -> TrackBankMapabilitySummary:
    if not bank.tracks:
        return TrackBankMapabilitySummary(
            track_count=0,
            feature_dim=bank.feature_dim,
            mean_observation_count=0.0,
            mean_track_variance=0.0,
            mean_utility=0.0,
        )
    tracks = list(bank.tracks.values())
    return TrackBankMapabilitySummary(
        track_count=len(tracks),
        feature_dim=bank.feature_dim,
        mean_observation_count=float(np.mean([track.observation_count for track in tracks])),
        mean_track_variance=float(np.mean([track.mean_variance for track in tracks])),
        mean_utility=float(np.mean([track.mean_utility for track in tracks])),
    )
