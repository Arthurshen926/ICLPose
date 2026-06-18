"""Dynamic rendered-pose VPR utilities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    CambridgePoseRecord,
    camera_center_from_pose_w2c,
    safe_image_id_key,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-6)


def rendered_image_id_for_candidate(candidate: CandidateHypothesis) -> str:
    """Return a stable render image id for a dynamic rendered-pose candidate."""

    if candidate.query_id is None:
        raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
    query_key = safe_image_id_key(candidate.query_id)
    candidate_key = str(candidate.candidate_id).replace("/", "__").replace("\\", "__")
    return f"dynamic_2dgs/{query_key}/{candidate_key}.png"


def candidate_bank_to_render_records(
    bank: CandidateHypothesisBank,
) -> Tuple[Tuple[CambridgePoseRecord, ...], Mapping[str, str]]:
    """Convert pose candidates into renderable Cambridge records.

    The candidate id remains the identity used for scoring. The returned mapping
    links candidate id to the render image id used in token manifests.
    """

    records: list[CambridgePoseRecord] = []
    candidate_to_image: dict[str, str] = {}
    for candidate in bank.candidates:
        if candidate.pose is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose")
        pose = np.asarray(candidate.pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"candidate {candidate.candidate_id} pose must have shape (4, 4)")
        image_id = rendered_image_id_for_candidate(candidate)
        if image_id in candidate_to_image.values():
            raise ValueError(f"duplicate dynamic render image id: {image_id}")
        candidate_to_image[candidate.candidate_id] = image_id
        records.append(
            CambridgePoseRecord(
                image_id=image_id,
                camera_center=camera_center_from_pose_w2c(pose),
                rotation_w2c=pose[:3, :3].astype(np.float64, copy=False),
                pose_w2c=pose,
            )
        )
    if not records:
        raise ValueError("candidate bank contains no renderable candidates")
    return tuple(records), candidate_to_image


def rerank_dynamic_rendered_vpr_candidates(
    *,
    lattice_bank: CandidateHypothesisBank,
    query_descriptors: TokenDescriptorBank,
    rendered_descriptors: TokenDescriptorBank,
    candidate_to_image: Mapping[str, str],
    protocol_name: str,
    top_k: int,
) -> CandidateHypothesisBank:
    """Score each query only against its own dynamic rendered-pose lattice."""

    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if query_descriptors.layer_name != rendered_descriptors.layer_name:
        raise ValueError("query and rendered descriptor layers must match")
    if query_descriptors.descriptors.shape[1] != rendered_descriptors.descriptors.shape[1]:
        raise ValueError("query and rendered descriptor dimensions must match")

    query_index = query_descriptors.index()
    render_index = rendered_descriptors.index()
    query_features = _normalize_rows(query_descriptors.descriptors)
    render_features = _normalize_rows(rendered_descriptors.descriptors)

    grouped: dict[str, list[tuple[float, int, CandidateHypothesis, str]]] = {}
    skipped_missing_render = 0
    for ordinal, candidate in enumerate(lattice_bank.candidates):
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.query_id not in query_index:
            raise ValueError(f"query descriptor not found: {candidate.query_id}")
        image_id = candidate_to_image.get(candidate.candidate_id)
        if image_id is None or image_id not in render_index:
            skipped_missing_render += 1
            continue
        score = float(
            np.dot(
                query_features[query_index[candidate.query_id]],
                render_features[render_index[image_id]],
            )
        )
        grouped.setdefault(candidate.query_id, []).append((score, ordinal, candidate, image_id))

    if not grouped:
        raise ValueError(f"no dynamic rendered candidates could be scored; skipped_missing_render={skipped_missing_render}")

    reranked: list[CandidateHypothesis] = []
    for query_id in sorted(grouped):
        rows = sorted(grouped[query_id], key=lambda item: (-item[0], item[1], item[2].candidate_id))
        for rank, (score, _ordinal, candidate, image_id) in enumerate(rows[: int(top_k)], start=1):
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "candidate_generator": "2dgs_dynamic_init_lattice_vpr",
                    "candidate_uses_gt": False,
                    "gt_used_for_label_only": True,
                    "dynamic_vpr_rank": rank,
                    "dynamic_vpr_score": float(score),
                    "dynamic_render_image_id": image_id,
                    "dynamic_render_descriptor_pooling": rendered_descriptors.pooling,
                    "dynamic_render_layer_name": rendered_descriptors.layer_name,
                    "dynamic_render_top_k": int(top_k),
                    "dynamic_render_skipped_missing_render": int(skipped_missing_render),
                }
            )
            reranked.append(
                replace(
                    candidate,
                    candidate_type="2dgs_dynamic_rendered_pose_vpr",
                    reference_image=image_id,
                    prior_score=float(score),
                    metadata=metadata,
                )
            )
    if not reranked:
        raise ValueError("dynamic rendered VPR reranking produced no candidates")
    return CandidateHypothesisBank.from_candidates(
        protocol_name=str(protocol_name),
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=reranked,
    )
