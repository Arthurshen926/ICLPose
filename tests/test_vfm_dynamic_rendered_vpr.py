from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.cambridge_pose_lattice import pose_w2c_from_center_rotation
from feature_extract.vfm.dynamic_rendered_vpr import (
    candidate_bank_to_render_records,
    rendered_image_id_for_candidate,
    rerank_dynamic_rendered_vpr_candidates,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def _candidate(query_id: str, candidate_id: str, center: tuple[float, float, float]) -> CandidateHypothesis:
    pose = pose_w2c_from_center_rotation(np.asarray(center, dtype=np.float64), np.eye(3, dtype=np.float64))
    return CandidateHypothesis(
        query_id=query_id,
        candidate_id=candidate_id,
        candidate_type="rendered_pose_lattice",
        pose=pose.tolist(),
        pose_error=PoseCost(translation_m=float(np.linalg.norm(center)), rotation_deg=0.0),
        metadata={"candidate_generator": "init_centered_world_translation_lattice", "candidate_uses_gt": False},
    )


def test_candidate_bank_to_render_records_preserves_pose_and_mapping() -> None:
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="lattice",
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=[_candidate("seq/frame0001.png", "seq__frame0001:init_lattice:000:001", (1.0, 2.0, 3.0))],
    )

    records, mapping = candidate_bank_to_render_records(bank)

    assert records[0].image_id == rendered_image_id_for_candidate(bank.candidates[0])
    assert mapping["seq__frame0001:init_lattice:000:001"] == records[0].image_id
    np.testing.assert_allclose(records[0].camera_center, [1.0, 2.0, 3.0], atol=1e-6)


def test_dynamic_rendered_vpr_reranks_candidates_per_query() -> None:
    bad = _candidate("seq/frame0001.png", "bad", (2.0, 0.0, 0.0))
    good = _candidate("seq/frame0001.png", "good", (0.1, 0.0, 0.0))
    lattice = CandidateHypothesisBank.from_candidates(
        protocol_name="lattice",
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=[bad, good],
    )
    records, mapping = candidate_bank_to_render_records(lattice)
    query = TokenDescriptorBank(
        image_ids=("seq/frame0001.png",),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    )
    rendered = TokenDescriptorBank(
        image_ids=(records[0].image_id, records[1].image_id),
        descriptors=np.asarray([[0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    )

    reranked = rerank_dynamic_rendered_vpr_candidates(
        lattice_bank=lattice,
        query_descriptors=query,
        rendered_descriptors=rendered,
        candidate_to_image=mapping,
        protocol_name="dynamic",
        top_k=2,
    )

    assert [candidate.candidate_id for candidate in reranked.candidates] == ["good", "bad"]
    assert reranked.candidates[0].reference_image == records[1].image_id
    assert reranked.candidates[0].metadata["candidate_generator"] == "2dgs_dynamic_init_lattice_vpr"
    assert reranked.candidates[0].metadata["candidate_uses_gt"] is False
    assert reranked.candidates[0].metadata["gt_used_for_label_only"] is True
    assert reranked.candidates[0].metadata["dynamic_vpr_rank"] == 1


def test_dynamic_rendered_vpr_requires_matching_descriptor_dimension() -> None:
    lattice = CandidateHypothesisBank.from_candidates(
        protocol_name="lattice",
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=[_candidate("seq/frame0001.png", "c0", (0.0, 0.0, 0.0))],
    )
    _records, mapping = candidate_bank_to_render_records(lattice)

    with pytest.raises(ValueError, match="dimensions"):
        rerank_dynamic_rendered_vpr_candidates(
            lattice_bank=lattice,
            query_descriptors=TokenDescriptorBank(
                image_ids=("seq/frame0001.png",),
                descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
                layer_name="radio_final",
                pooling="mean",
            ),
            rendered_descriptors=TokenDescriptorBank(
                image_ids=(mapping["c0"],),
                descriptors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                layer_name="radio_final",
                pooling="mean",
            ),
            candidate_to_image=mapping,
            protocol_name="dynamic",
            top_k=1,
        )
