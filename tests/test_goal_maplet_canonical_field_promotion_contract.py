from __future__ import annotations

from feature_extract.tools.vfm.build_goal_maplet_canonical_field_from_contributors import (
    _canonical_promotion_contract,
    _canonical_mapper_route_audit,
    _contributor_filename_identity,
    _surface_mapper_supervision_coordinate_audit,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
)


def test_legacy_mapper_without_coordinate_lineage_is_not_promotable():
    audit = _surface_mapper_supervision_coordinate_audit({
        "trajectory_split": True,
        "strict_holdout_trajectory_ids": ["seq12", "seq14"],
    })
    contract = _canonical_promotion_contract(
        feature_space="retrieval_mapper",
        legacy_pinhole_as_raw_diagnostic=False,
        mapper_audit=audit,
    )
    assert audit["coordinate_correct"] is False
    assert audit["hash_bound"] is False
    assert "missing_supervision_coordinate_lineage" in audit["blockers"]
    assert contract["canonical_fusion_coordinate_correct"] is True
    assert contract["mapper_supervision_coordinate_correct"] is False
    assert contract["promotion_eligible"] is False
    assert contract["promotion_blockers"] == [
        "surface_mapper_supervision_coordinate_lineage_unverified"
    ]


def test_canonical_contributor_route_can_be_filtered_before_archive_open(tmp_path):
    held = tmp_path / "seq12__frame00001.png.npz"
    held.write_bytes(b"not-an-npz-on-purpose")
    assert _contributor_filename_identity(held) == (
        "seq12", "seq12/frame00001.png",
    )


def test_unbound_coordinate_claim_is_rejected():
    audit = _surface_mapper_supervision_coordinate_audit({
        "supervision_coordinate_lineage": {
            "coordinate_correct": True,
            "coordinate_contract": COORDINATE_CONTRACT,
        }
    })
    assert audit["coordinate_correct"] is False
    assert "surface_maplets_supervision_hash_missing_or_invalid" in audit["blockers"]
    assert "radio_manifest_supervision_hash_missing_or_invalid" in audit["blockers"]


def test_hash_claim_without_verified_seal_is_rejected():
    audit = _surface_mapper_supervision_coordinate_audit({
        "supervision_coordinate_lineage": {
            "coordinate_correct": True,
            "coordinate_contract": COORDINATE_CONTRACT,
            "surface_maplets_file_sha256": "a" * 64,
            "radio_final_manifest_file_sha256": "b" * 64,
        }
    })
    assert audit["coordinate_correct"] is False
    assert "coordinate_supervision_seal_missing_or_invalid" in audit["blockers"]


def test_hash_bound_coordinate_correct_mapper_is_promotable():
    audit = _surface_mapper_supervision_coordinate_audit({
        "coordinate_supervision_sealed": True,
        "coordinate_supervision_seal_schema": (
            "goal_maplet_surface_mapper_coordinate_lineage_seal_v1"
        ),
        "supervision_coordinate_lineage": {
            "coordinate_correct": True,
            "coordinate_contract": COORDINATE_CONTRACT,
            "surface_maplets_file_sha256": "a" * 64,
            "radio_final_manifest_file_sha256": "b" * 64,
            "sealed_by": "goal_maplet_surface_mapper_coordinate_lineage_seal_v1",
        }
    })
    contract = _canonical_promotion_contract(
        feature_space="retrieval_mapper",
        legacy_pinhole_as_raw_diagnostic=False,
        mapper_audit=audit,
    )
    assert audit["coordinate_correct"] is True
    assert audit["hash_bound"] is True
    assert audit["blockers"] == []
    assert contract["promotion_eligible"] is True
    assert contract["promotion_blockers"] == []


def test_legacy_fusion_blocks_promotion_even_with_correct_mapper():
    audit = _surface_mapper_supervision_coordinate_audit({
        "coordinate_supervision_sealed": True,
        "coordinate_supervision_seal_schema": (
            "goal_maplet_surface_mapper_coordinate_lineage_seal_v1"
        ),
        "supervision_coordinate_lineage": {
            "coordinate_correct": True,
            "coordinate_contract": COORDINATE_CONTRACT,
            "surface_maplets_file_sha256": "a" * 64,
            "radio_final_manifest_file_sha256": "b" * 64,
            "sealed_by": "goal_maplet_surface_mapper_coordinate_lineage_seal_v1",
        }
    })
    contract = _canonical_promotion_contract(
        feature_space="retrieval_mapper",
        legacy_pinhole_as_raw_diagnostic=True,
        mapper_audit=audit,
    )
    assert contract["canonical_fusion_coordinate_correct"] is False
    assert contract["promotion_eligible"] is False
    assert contract["promotion_blockers"] == [
        "legacy_pinhole_as_raw_coordinate_control"
    ]


def test_canonical_mapper_route_audit_requires_exact_fit_validation_routes():
    mapper = {
        "training_trajectory_ids": ["seq1", "seq2"],
        "validation_trajectory_ids": ["seq9"],
        "strict_holdout_trajectory_ids": ["seq10", "seq12", "seq14"],
    }
    passed = _canonical_mapper_route_audit({"seq1", "seq2", "seq9"}, mapper)
    assert passed["canonical_mapping_routes_match_mapper_fit_validation"] is True
    assert passed["blockers"] == []
    leaked = _canonical_mapper_route_audit({"seq1", "seq2", "seq9", "seq10"}, mapper)
    assert leaked["canonical_mapping_routes_match_mapper_fit_validation"] is False
    assert "canonical_mapping_contains_mapper_strict_holdout" in leaked["blockers"]
    assert "canonical_mapping_routes_differ_from_mapper_fit_validation" in leaked["blockers"]
