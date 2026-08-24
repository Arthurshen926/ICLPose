from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.materialize_goal_maplet_seq10_layout_tuning_control import (
    ALLOCATOR_BLOCKER,
    CALIBRATION_BLOCKER,
    _materialize_query,
    _validate_source_contract,
)
from feature_extract.vfm.localization_goal_maplet.layout_child_allocator_config import (
    CONFIG_SCHEMA,
    config_content_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from test_goal_maplet_pure_retrieval import _physical, _result


def _config(physical, summary_path, summary_hash):
    value = {
        "artifact_type": CONFIG_SCHEMA,
        "allocator_semantics": "parent_mass_prefix_one_child_then_global_joint_fill_v1",
        "candidate_parent_mask_semantics": "positive_scene_parent_ids_only_v1",
        "child_evidence_semantics": "joint_4x4_block_max_sum_v1",
        "child_evidence_local_block_size": 1,
        "child_score_policy": "raw_child_evidence_v1",
        "parent_mass_fraction": 0.5,
        "maximum_children": 2,
        "maximum_primitive_iou": 1.0,
        "maximum_normal_angle_degrees": 30.0,
        "physical_map_sha256": physical.content_sha256,
        "tuning_route": "seq10",
        "tuning_query_count": 1,
        "tuning_retrieval": {
            "retrieval_summary_paths": [str(summary_path)],
            "retrieval_summary_file_sha256": [summary_hash],
        },
        "deployment_uses_gt": False,
        "uses_query_pose": False,
        "uses_gt_only_for_offline_route_disjoint_config_selection": True,
    }
    value["content_sha256"] = config_content_sha256(value)
    return value


def _summary(physical):
    return {
        "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
        "query_count": 1,
        "physical_map_sha256": physical.content_sha256,
        "promotion_eligible": False,
        "promotion_blockers": [CALIBRATION_BLOCKER],
        "control_only": True,
        "query_split_audit": {
            "query_trajectory_ids": ["seq10"],
            "validity_calibration_fit_trajectory_ids": ["seq10"],
            "blockers": [CALIBRATION_BLOCKER],
            "disjoint": False,
        },
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_mapping_rgb": False,
        "uses_image_retrieval": False,
        "rows": [{"image_id": "seq10/frame.png"}],
    }


def test_source_contract_requires_exact_signed_tuning_retrieval_binding(tmp_path):
    physical = _physical()
    path = (tmp_path / "retrieval.json").resolve()
    config = _config(physical, path, "a" * 64)
    rows = _validate_source_contract(
        [_summary(physical)], [path], ["a" * 64], config,
        physical_map_sha256=physical.content_sha256,
    )
    assert [row["image_id"] for row in rows] == ["seq10/frame.png"]
    with pytest.raises(ValueError, match="tuning binding"):
        _validate_source_contract(
            [_summary(physical)], [path], ["b" * 64], config,
            physical_map_sha256=physical.content_sha256,
        )


def test_materializer_preserves_frozen_fields_and_remains_nonpromotable():
    physical = _physical()
    physical.metadata["child_voxel_size_m"] = 1.0
    base = _result(physical)
    source_split = {
        "query_trajectory_ids": ["seq10"],
        "validity_calibration_fit_trajectory_ids": ["seq10"],
        "blockers": [CALIBRATION_BLOCKER],
        "disjoint": False,
    }
    metadata = {
        **dict(base.metadata),
        "promotion_eligible": False,
        "promotion_blockers": [CALIBRATION_BLOCKER],
        "control_only": True,
        "query_split_audit": source_split,
    }
    retrieval = PureRadioPhysicalRetrieval(**{
        **base.__dict__, "image_id": "seq10/frame.png", "metadata": metadata,
    })
    config = _config(physical, "/tmp/retrieval.json", "a" * 64)
    result, _ = _materialize_query(
        retrieval,
        source_file_sha256="b" * 64,
        config=config,
        config_file_sha256="c" * 64,
        physical=physical,
        child_area_m2=np.ones(
            physical.child_parent_rows.size, dtype=np.float64,
        ),
    )
    for name in (
        "token_xy", "token_parent_ids", "token_parent_probabilities",
        "token_out_of_map_probabilities",
        "token_in_map_tail_probabilities", "token_child_rows",
        "token_child_probabilities", "scene_parent_ids", "scene_parent_scores",
    ):
        np.testing.assert_array_equal(getattr(result, name), getattr(retrieval, name))
    assert result.metadata["promotion_eligible"] is False
    assert result.metadata["control_only"] is True
    assert result.metadata["promotion_blockers"] == [
        CALIBRATION_BLOCKER, ALLOCATOR_BLOCKER,
    ]
    split = result.metadata["query_split_audit"]
    assert split["disjoint"] is False
    assert split["allocator_tuning_query_disjoint"] is False
    assert split["allocator_tuning_same_route_control"] is True
