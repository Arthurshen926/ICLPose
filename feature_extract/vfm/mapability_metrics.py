"""Mapability metrics for selected 3D feature banks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank


@dataclass(frozen=True)
class TrackBankMapabilityReport:
    track_count: int
    expected_track_count: int
    coverage: float
    mean_track_variance: float
    separability_ratio: float
    estimated_storage_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "track_count": self.track_count,
            "expected_track_count": self.expected_track_count,
            "coverage": self.coverage,
            "mean_track_variance": self.mean_track_variance,
            "separability_ratio": self.separability_ratio,
            "estimated_storage_bytes": self.estimated_storage_bytes,
        }


def _pairwise_mean_distance(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return 0.0
    diffs = features[:, None, :] - features[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper = distances[np.triu_indices(features.shape[0], k=1)]
    return float(np.mean(upper))


def estimate_track_bank_storage_bytes(bank: SelectedTrackFeatureBank) -> int:
    per_track = bank.feature_dim * 2 * np.dtype(np.float32).itemsize
    per_track += np.dtype(np.int64).itemsize
    per_track += np.dtype(np.float32).itemsize
    return int(len(bank.tracks) * per_track)


def compare_track_bank_mapability(
    bank: SelectedTrackFeatureBank,
    expected_track_count: int,
    max_pairwise_tracks: int | None = None,
    seed: int = 0,
) -> TrackBankMapabilityReport:
    if expected_track_count <= 0:
        raise ValueError("expected_track_count must be positive")
    tracks = list(bank.tracks.values())
    track_count = len(tracks)
    coverage = min(float(track_count / expected_track_count), 1.0)
    if not tracks:
        return TrackBankMapabilityReport(
            track_count=0,
            expected_track_count=expected_track_count,
            coverage=0.0,
            mean_track_variance=0.0,
            separability_ratio=0.0,
            estimated_storage_bytes=0,
        )

    if max_pairwise_tracks is not None:
        if max_pairwise_tracks <= 1:
            raise ValueError("max_pairwise_tracks must be greater than 1")
        if len(tracks) > max_pairwise_tracks:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(tracks), size=max_pairwise_tracks, replace=False))
            tracks_for_distance = [tracks[int(idx)] for idx in indices]
        else:
            tracks_for_distance = tracks
    else:
        tracks_for_distance = tracks

    means = np.stack([track.mean_feature for track in tracks_for_distance], axis=0).astype(np.float32)
    mean_variance = float(np.mean([track.mean_variance for track in tracks]))
    between_distance = _pairwise_mean_distance(means)
    separability_ratio = float(between_distance / max(mean_variance, 1e-6))
    return TrackBankMapabilityReport(
        track_count=track_count,
        expected_track_count=expected_track_count,
        coverage=coverage,
        mean_track_variance=mean_variance,
        separability_ratio=separability_ratio,
        estimated_storage_bytes=estimate_track_bank_storage_bytes(bank),
    )
