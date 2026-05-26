"""Score candidate hypotheses using token-bank feature similarity."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.score_table import ScoreRow
from feature_extract.vfm.tokens import TokenBankManifest


def _record_index(manifest: TokenBankManifest):
    manifest.validate(verify_checksums=False)
    return {record.image_id: record for record in manifest.records}


@lru_cache(maxsize=4096)
def _pooled_feature(path: str, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        feature = np.asarray(data[layer_name], dtype=np.float32)
    if feature.ndim < 2:
        raise ValueError("token feature must have at least channel and spatial dimensions")
    pooled = feature.reshape(feature.shape[0], -1).mean(axis=1)
    norm = np.linalg.norm(pooled)
    if norm <= 1e-6:
        return pooled
    return pooled / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_candidate_bank_by_token_cosine(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
    layer_name: str,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> list[ScoreRow]:
    query_records = _record_index(query_manifest)
    map_records = _record_index(map_manifest)
    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_records:
            raise ValueError(f"query token not found: {candidate.query_id}")
        if candidate.reference_image not in map_records:
            raise ValueError(f"reference token not found: {candidate.reference_image}")
        query_feature = _pooled_feature(str(query_records[candidate.query_id].token_path), layer_name)
        map_feature = _pooled_feature(str(map_records[candidate.reference_image].token_path), layer_name)
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=_cosine(query_feature, map_feature),
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
            )
        )
    return rows
