"""Audits for full-map query and projected-observation descriptor parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.landmark_hybrid import sample_projected_track_observations
from feature_extract.vfm.localization.pipeline import _load_feature_map
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import _sample_feature_vector


def audit_full_map_observation_projection_parity(
    observations: Sequence[ColmapTrackObservation],
    token_manifest: TokenBankManifest,
    feature_mapper: Any,
    *,
    feature_key: str = "radio_final",
    sample_mode: str = "bilinear",
    max_observations: int = 64,
    min_cosine_threshold: float = 0.99999,
) -> dict[str, Any]:
    """Compare the bank builder path with an independent eval-style full-map path."""

    records = {str(record.image_id): record for record in token_manifest.records}
    selected = [
        observation
        for observation in observations
        if str(observation.image_id) in records
        and observation.image_width is not None
        and observation.image_height is not None
    ]
    if int(max_observations) > 0:
        selected = selected[: int(max_observations)]
    if not selected:
        raise ValueError("no auditable observations found")

    builder_samples, _metadata = sample_projected_track_observations(
        selected,
        token_manifest,
        feature_mapper,
        feature_key=str(feature_key),
        missing="error",
        sample_mode=str(sample_mode),
    )
    by_image: dict[str, list[ColmapTrackObservation]] = {}
    for observation in selected:
        by_image.setdefault(str(observation.image_id), []).append(observation)
    independent: list[np.ndarray] = []
    for image_id, image_observations in sorted(by_image.items()):
        raw_map = _load_feature_map(Path(records[image_id].token_path), key=str(feature_key))
        projected_map = np.asarray(feature_mapper.project(raw_map).coarse_descriptors, dtype=np.float32)
        for observation in image_observations:
            independent.append(
                _sample_feature_vector(
                    projected_map,
                    observation.xy,
                    int(observation.image_width),
                    int(observation.image_height),
                    str(sample_mode),
                )
            )
    if len(builder_samples) != len(independent):
        raise ValueError(
            f"projection path sample-count mismatch: builder={len(builder_samples)}, eval={len(independent)}"
        )
    builder_matrix = np.stack([np.asarray(item.feature, dtype=np.float64) for item in builder_samples], axis=0)
    independent_matrix = np.stack([np.asarray(item, dtype=np.float64) for item in independent], axis=0)
    builder_norm = np.linalg.norm(builder_matrix, axis=1)
    independent_norm = np.linalg.norm(independent_matrix, axis=1)
    valid = (builder_norm > 1e-12) & (independent_norm > 1e-12)
    if not np.all(valid):
        raise ValueError(f"projection path audit found {int(np.count_nonzero(~valid))} zero descriptors")
    cosine = np.sum(builder_matrix * independent_matrix, axis=1) / (builder_norm * independent_norm)
    max_abs = np.max(np.abs(builder_matrix - independent_matrix), axis=1)
    result = {
        "observation_count": int(len(cosine)),
        "image_count": int(len(by_image)),
        "min_cosine": float(np.min(cosine)),
        "mean_cosine": float(np.mean(cosine)),
        "max_abs_difference": float(np.max(max_abs)),
        "min_cosine_threshold": float(min_cosine_threshold),
        "passed": bool(float(np.min(cosine)) >= float(min_cosine_threshold)),
        "feature_key": str(feature_key),
        "sample_mode": str(sample_mode),
        "builder_path": "full_map_project_then_observation_sample",
        "eval_path": "query_full_map_project_then_observation_sample",
    }
    if not result["passed"]:
        raise ValueError(
            "full-map descriptor path parity failed: "
            f"min_cosine={result['min_cosine']:.8f} < {float(min_cosine_threshold):.8f}"
        )
    return result
