"""Build fixed candidate banks from descriptor retrieval with pose labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    CambridgePoseRecord,
    parse_cambridge_pose_file,
    rotation_angle_deg,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def _pose_index(path: Path, max_abs_pose_center: float | None = None) -> dict[str, CambridgePoseRecord]:
    records = parse_cambridge_pose_file(Path(path))
    if max_abs_pose_center is None:
        return {record.image_id: record for record in records}
    threshold = float(max_abs_pose_center)
    if threshold <= 0.0:
        raise ValueError("max_abs_pose_center must be positive")
    return {
        record.image_id: record
        for record in records
        if float(np.max(np.abs(record.camera_center))) <= threshold
    }


def _normalize_rows(descriptors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-6)


def build_descriptor_retrieval_reference_pose_bank(
    query_descriptors: TokenDescriptorBank,
    map_descriptors: TokenDescriptorBank,
    query_pose_file: Path,
    reference_pose_file: Path,
    protocol_name: str,
    top_k: int = 10,
    exclude_same_image: bool = False,
    rot_cost_weight: float = 0.05,
    max_abs_pose_center: float | None = None,
) -> CandidateHypothesisBank:
    """Build a reference-pose bank whose candidates come from descriptor retrieval.

    Candidate generation uses only descriptors. Ground-truth poses are used
    afterwards to attach pose-cost labels for supervised selector training.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rot_cost_weight < 0.0:
        raise ValueError("rot_cost_weight must be non-negative")
    if query_descriptors.descriptors.shape[1] != map_descriptors.descriptors.shape[1]:
        raise ValueError("query and map descriptor dimensions must match")
    if query_descriptors.layer_name != map_descriptors.layer_name:
        raise ValueError("query and map descriptor layers must match")

    query_pose_by_id = _pose_index(query_pose_file, max_abs_pose_center=max_abs_pose_center)
    reference_pose_by_id = _pose_index(reference_pose_file, max_abs_pose_center=max_abs_pose_center)
    query_features = _normalize_rows(query_descriptors.descriptors)
    map_features = _normalize_rows(map_descriptors.descriptors)
    scores = query_features @ map_features.T

    candidates: list[CandidateHypothesis] = []
    for query_idx, query_id in enumerate(query_descriptors.image_ids):
        if query_id not in query_pose_by_id:
            continue
        scored_refs = []
        for ref_idx, reference_id in enumerate(map_descriptors.image_ids):
            if exclude_same_image and query_id == reference_id:
                continue
            if reference_id not in reference_pose_by_id:
                continue
            scored_refs.append((float(scores[query_idx, ref_idx]), reference_id))
        if not scored_refs:
            raise ValueError(f"query {query_id} has no descriptor retrieval candidates")
        scored_refs.sort(key=lambda item: (-item[0], item[1]))

        query_pose = query_pose_by_id[query_id]
        for rank, (score, reference_id) in enumerate(scored_refs[:top_k], start=1):
            reference_pose = reference_pose_by_id[reference_id]
            translation_m = float(np.linalg.norm(query_pose.camera_center - reference_pose.camera_center))
            rotation_deg = rotation_angle_deg(query_pose.rotation_w2c, reference_pose.rotation_w2c)
            pose_cost_m = translation_m + rot_cost_weight * rotation_deg
            candidates.append(
                CandidateHypothesis(
                    query_id=query_id,
                    candidate_id=f"{Path(query_id).stem}:descriptor_retrieval:{rank - 1:03d}",
                    candidate_type="descriptor_retrieval_reference_pose",
                    pose=reference_pose.pose_w2c.tolist(),
                    reference_image=reference_id,
                    pose_error=PoseCost(translation_m=translation_m, rotation_deg=rotation_deg),
                    prior_score=score,
                    metadata={
                        "candidate_generator": "descriptor_retrieval",
                        "candidate_uses_gt": False,
                        "pose_label_uses_gt": True,
                        "retrieval_rank": rank,
                        "retrieval_score": score,
                        "pose_cost_m": float(pose_cost_m),
                        "rot_cost_weight": float(rot_cost_weight),
                    },
                )
            )

    if not candidates:
        raise ValueError("descriptor retrieval produced no candidates")
    return CandidateHypothesisBank.from_candidates(
        protocol_name=protocol_name,
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=candidates,
    )
