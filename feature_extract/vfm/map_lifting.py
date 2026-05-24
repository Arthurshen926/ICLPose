"""Lift selected 2D features into explicit 3D track features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    image_id: str
    feature: np.ndarray
    visible: bool
    geometry_valid: bool
    utility: float = 1.0


@dataclass(frozen=True)
class TrackFeature:
    track_id: int
    mean_feature: np.ndarray
    variance: np.ndarray
    observation_count: int
    mean_utility: float

    @property
    def mean_variance(self) -> float:
        return float(np.mean(self.variance))


@dataclass(frozen=True)
class SelectedTrackFeatureBank:
    tracks: Mapping[int, TrackFeature]
    feature_dim: int

    def __len__(self) -> int:
        return len(self.tracks)


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
        )

    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=int(feature_dim or 0))
