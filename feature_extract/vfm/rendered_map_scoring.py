"""Rendered selected-map feature scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank
from feature_extract.vfm.verifier import HypothesisEvidence, LinearEvidenceVerifier, VerificationScore


@dataclass(frozen=True)
class RenderedSelectedMapFeature:
    candidate_id: str
    track_ids: Tuple[int, ...]
    features: np.ndarray
    visibility: np.ndarray
    geometry_valid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_ids", tuple(self.track_ids))
        object.__setattr__(self, "features", np.asarray(self.features, dtype=np.float32))
        object.__setattr__(self, "visibility", np.asarray(self.visibility, dtype=bool))
        object.__setattr__(self, "geometry_valid", np.asarray(self.geometry_valid, dtype=bool))
        if self.features.ndim != 2:
            raise ValueError("features must have shape (N, D)")
        if self.visibility.shape != (self.features.shape[0],):
            raise ValueError("visibility must have shape (N,)")
        if self.geometry_valid.shape != (self.features.shape[0],):
            raise ValueError("geometry_valid must have shape (N,)")


def render_selected_track_bank(
    bank: SelectedTrackFeatureBank,
    candidate_id: str,
    visible_track_ids: Iterable[int],
) -> RenderedSelectedMapFeature:
    """Project a selected 3D track bank into a candidate view.

    This deterministic representation is enough for protocol tests and offline
    runners. A real renderer can fill the same dataclass from projected tracks.
    """

    track_ids = tuple(int(track_id) for track_id in visible_track_ids)
    features = np.zeros((len(track_ids), bank.feature_dim), dtype=np.float32)
    visibility = np.zeros(len(track_ids), dtype=bool)
    geometry_valid = np.zeros(len(track_ids), dtype=bool)
    for idx, track_id in enumerate(track_ids):
        track = bank.tracks.get(track_id)
        if track is None:
            continue
        features[idx] = np.asarray(track.mean_feature, dtype=np.float32)
        visibility[idx] = True
        geometry_valid[idx] = True
    return RenderedSelectedMapFeature(
        candidate_id=candidate_id,
        track_ids=track_ids,
        features=features,
        visibility=visibility,
        geometry_valid=geometry_valid,
    )


def _cosine_similarity(query: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    rendered_norm = np.linalg.norm(rendered, axis=1, keepdims=True)
    denom = np.maximum(query_norm * rendered_norm, 1e-6)
    return np.sum(query * rendered, axis=1, keepdims=True)[:, 0] / denom[:, 0]


def score_rendered_selected_features(
    query_features: np.ndarray,
    rendered: RenderedSelectedMapFeature,
    query_uncertainty: np.ndarray,
    verifier: LinearEvidenceVerifier,
    candidate_prior: float | None = None,
) -> VerificationScore:
    """Score query selected features against rendered selected map features."""

    query = np.asarray(query_features, dtype=np.float32)
    uncertainty = np.asarray(query_uncertainty, dtype=np.float32).reshape(-1)
    if query.shape != rendered.features.shape:
        raise ValueError("query_features and rendered.features must have identical shape")
    if uncertainty.shape != (query.shape[0],):
        raise ValueError("query_uncertainty must have shape (N,)")

    valid = rendered.visibility & rendered.geometry_valid
    if not np.any(valid):
        evidence = HypothesisEvidence(
            feature_similarity=0.0,
            visibility_fraction=0.0,
            geometry_consistency=0.0,
            uncertainty=1.0,
            candidate_prior=candidate_prior,
        )
        return verifier.score(evidence)

    cosine = _cosine_similarity(query[valid], rendered.features[valid])
    feature_similarity = float(np.clip(cosine.mean(), 0.0, 1.0))
    visibility_fraction = float(np.mean(valid))
    geometry_consistency = float(np.mean(rendered.geometry_valid[rendered.visibility]))
    evidence = HypothesisEvidence(
        feature_similarity=feature_similarity,
        visibility_fraction=visibility_fraction,
        geometry_consistency=geometry_consistency,
        uncertainty=float(np.mean(uncertainty[valid])),
        candidate_prior=candidate_prior,
    )
    return verifier.score(evidence)
