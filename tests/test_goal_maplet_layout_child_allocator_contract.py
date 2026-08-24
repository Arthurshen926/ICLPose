import json
from pathlib import Path

import pytest

from feature_extract.vfm.localization_goal_maplet.layout_child_allocator_config import (
    CONFIG_SCHEMA,
    SCORE_RAW_EVIDENCE,
    config_content_sha256,
    load_validate_layout_child_allocator_config,
    retrieval_layout_source_signature,
    validate_layout_source_signature,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    CHILD_PROBABILITY_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.scene_child_evidence import (
    EVIDENCE_LEGACY_MACRO_TOP4,
    SCENE_PARENT_MASK_SEMANTICS,
)
from feature_extract.vfm.tokens import compute_file_sha256
from test_goal_maplet_pure_retrieval import _physical, _result


def _artifacts(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    physical = _physical()
    physical_path = tmp_path / "physical.npz"
    physical.save_npz(physical_path)
    retrieval = _result(physical)
    metadata = dict(retrieval.metadata)
    metadata.update({
        "physical_map_file_sha256": compute_file_sha256(physical_path),
        "canonical_field_file_sha256": "canonical",
        "surface_mapper_file_sha256": "mapper",
        "field_feature_contract_file_sha256": "contract",
        "validity_calibration_file_sha256": "calibration",
        "parent_score_semantics": "parent",
        "parent_scene_ranking_semantics": "parent-rank",
        "parent_mode_temperature": 0.03,
        "anonymous_parent_mode_readout_sha256": "modes",
        "maximum_parent_candidates": 64,
        "maximum_child_candidates": 64,
        "maximum_scene_parents": 64,
        "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
        "pool_sizes": [1, 3, 5, 9],
        "pool_weights": [0.4, 0.3, 0.2, 0.1],
    })
    retrieval = type(retrieval)(**{**retrieval.__dict__, "metadata": metadata})
    retrieval_path = tmp_path / "retrieval.npz"
    retrieval.save_npz(retrieval_path)
    summary_path = tmp_path / "retrieval.json"
    summary_path.write_text(json.dumps({
        "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
        "control_only": True,
        "query_split_audit": {"query_trajectory_ids": ["seq10"]},
        "rows": [{
            "image_id": retrieval.image_id,
            "artifact": str(retrieval_path),
            "artifact_sha256": compute_file_sha256(retrieval_path),
            "content_sha256": retrieval.content_sha256,
        }],
    }, sort_keys=True))
    config_path = tmp_path / "config.json"
    config = {
        "artifact_type": CONFIG_SCHEMA,
        "allocator_semantics": "parent_mass_prefix_one_child_then_global_joint_fill_v1",
        "candidate_parent_mask_semantics": SCENE_PARENT_MASK_SEMANTICS,
        "child_evidence_semantics": EVIDENCE_LEGACY_MACRO_TOP4,
        "child_score_policy": SCORE_RAW_EVIDENCE,
        "child_evidence_local_block_size": 4,
        "parent_mass_fraction": 0.5,
        "maximum_children": 64,
        "maximum_primitive_iou": 0.5,
        "maximum_normal_angle_degrees": 30.0,
        "physical_map_sha256": physical.content_sha256,
        "physical_map_file_sha256": compute_file_sha256(physical_path),
        "tuning_route": "seq10",
        "tuning_retrieval": {
            "retrieval_summary_paths": [str(summary_path)],
            "retrieval_summary_file_sha256": [compute_file_sha256(summary_path)],
        },
        "uses_gt_only_for_offline_route_disjoint_config_selection": True,
        "deployment_uses_gt": False,
        "uses_query_pose": False,
    }
    config["content_sha256"] = config_content_sha256(config)
    config_path.write_text(json.dumps(config, sort_keys=True))
    return physical, physical_path, retrieval, summary_path, config_path


def test_layout_config_binds_tuning_bytes_and_accepts_disjoint_query(tmp_path):
    physical, physical_path, retrieval, _, config_path = _artifacts(tmp_path)
    config, signature = load_validate_layout_child_allocator_config(
        config_path,
        physical,
        physical_path=physical_path,
        query_routes={"seq12"},
    )
    assert config["tuning_route"] == "seq10"
    validate_layout_source_signature(
        signature, retrieval_layout_source_signature(retrieval)
    )


def test_layout_config_rejects_tuning_query_overlap(tmp_path):
    physical, physical_path, _, _, config_path = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="tuning/query routes overlap"):
        load_validate_layout_child_allocator_config(
            config_path,
            physical,
            physical_path=physical_path,
            query_routes={"seq10"},
        )


def test_layout_config_rejects_unsigned_mutation_and_tuning_summary_mutation(tmp_path):
    physical, physical_path, _, summary_path, config_path = _artifacts(tmp_path)
    config = json.loads(config_path.read_text())
    config["parent_mass_fraction"] = 0.75
    config_path.write_text(json.dumps(config, sort_keys=True))
    with pytest.raises(ValueError, match="lineage/claims"):
        load_validate_layout_child_allocator_config(
            config_path, physical, physical_path=physical_path,
        )

    _, _, _, summary_path, config_path = _artifacts(tmp_path / "second")
    summary_path.write_text(summary_path.read_text() + "\n")
    with pytest.raises(ValueError, match="summary bytes differ"):
        load_validate_layout_child_allocator_config(
            config_path, _physical(), physical_path=tmp_path / "second" / "physical.npz",
        )


def test_layout_signature_rejects_representation_distribution_change(tmp_path):
    _, _, retrieval, _, _ = _artifacts(tmp_path)
    tuning = retrieval_layout_source_signature(retrieval)
    deployment = dict(tuning)
    deployment["maximum_child_candidates"] = 32
    with pytest.raises(ValueError, match="maximum_child_candidates"):
        validate_layout_source_signature(tuning, deployment)
