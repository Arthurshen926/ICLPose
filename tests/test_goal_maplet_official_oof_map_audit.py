import copy
import numpy as np

import pytest

from feature_extract.tools.vfm.audit_goal_maplet_official_oof_maps import audit_fold
from feature_extract.vfm.localization_goal_maplet.mapping_view_graph import MappingViewGraph


def _reports():
    mapping = ["seq1", "seq4"]
    return {
        "surface_mapper": {
            "config": {"checkpoint_protocol": "fixed_epoch_no_selection"},
            "best_validation": None,
            "split": {
                "training_trajectory_ids": mapping,
                "prototype_trajectory_ids": mapping,
                "strict_holdout_trajectory_ids": ["seq2", "seq3", "seq5", "seq13"],
                "training_images": ["seq1/a.png", "seq4/a.png"],
            },
        },
        "geometry_head": {
            "train_count": 2, "eval_count": 1,
            "checkpoint_protocol": "fixed_epoch_no_selection",
            "production_contract": {
                "eval_manifest_used_for_gradient": False,
                "eval_manifest_used_for_checkpoint_selection": False,
                "saved_checkpoint_epoch": 30,
            },
        },
        "canonical_field": {
            "mapping_image_count": 2, "mapping_trajectory_ids": mapping,
            "canonical_field_sha256": "field", "canonical_primitive_count": 4,
            "primitive_coverage_fraction": 0.5,
        },
        "physical_readout": {
            "teacher_supervised_image_count": 2,
            "teacher_supervised_image_ids_sha256": "mapping-hash",
            "partition_counts": {"selection": 0, "validation": 0},
            "selected_validation": None,
        },
        "typed_graph": {
            "canonical_field_sha256": "field", "metadata": {"mapping_view_count": 2},
        },
        "mapping_view_graph": {
            "canonical_field_sha256": "field", "view_node_count": 2,
            "excluded_trajectory_ids": ["seq2", "seq3", "seq5", "seq13"],
        },
        "validity": {
            "image_count": 2,
            "calibration_metadata": {
                "fit_image_ids": ["seq1/a.png", "seq4/a.png"],
                "canonical_field_sha256": "field",
                "validity_target_algorithm": (
                    "exact_owned_contributor_mass_integral_image_v1"
                ),
            },
        },
    }


def _write(tmp_path, reports):
    import json
    from feature_extract.tools.vfm.audit_goal_maplet_official_oof_maps import SUMMARY_PATHS
    for name, relative in SUMMARY_PATHS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports[name]))
    MappingViewGraph(
        poses_w2c=np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
        parent_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        parent_rows=np.asarray([0, 0], dtype=np.int32),
        parent_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        physical_map_sha256="physical",
        canonical_field_sha256="field",
        metadata={"source_contributor_count": 2},
    ).save_npz(tmp_path / "mapping_view_graph.npz")


def _fold():
    return {
        "fold_id": "fold0", "mapping_trajectories": ["seq1", "seq4"],
        "held_query_trajectories": ["seq2"], "mapping_count": 2,
        "query_count": 1, "mapping_image_ids_sha256": "mapping-hash",
    }


def test_audit_accepts_fixed_route_disjoint_fold(tmp_path):
    _write(tmp_path, _reports())
    result = audit_fold(_fold(), tmp_path, official_test_routes={"seq3", "seq5", "seq13"})
    assert result["held_and_official_test_routes_absent_from_map_fit"] is True


def test_audit_rejects_mapping_view_leakage(tmp_path):
    reports = copy.deepcopy(_reports())
    reports["mapping_view_graph"]["excluded_trajectory_ids"].remove("seq2")
    _write(tmp_path, reports)
    with pytest.raises(ValueError, match="mapping-view exclusion"):
        audit_fold(_fold(), tmp_path, official_test_routes={"seq3", "seq5", "seq13"})


def test_audit_accepts_zero_incidence_mapping_view_omission(tmp_path):
    reports = _reports()
    reports["mapping_view_graph"]["view_node_count"] = 1
    _write(tmp_path, reports)
    graph_path = tmp_path / "mapping_view_graph.npz"
    graph = MappingViewGraph.load_npz(graph_path)
    MappingViewGraph(
        poses_w2c=graph.poses_w2c[:1],
        parent_offsets=np.asarray([0, 1], dtype=np.int64),
        parent_rows=graph.parent_rows[:1],
        parent_weights=graph.parent_weights[:1],
        physical_map_sha256=graph.physical_map_sha256,
        canonical_field_sha256=graph.canonical_field_sha256,
        metadata=graph.metadata,
    ).save_npz(graph_path)
    result = audit_fold(
        _fold(), tmp_path, official_test_routes={"seq3", "seq5", "seq13"}
    )
    assert result["mapping_view_zero_incidence_omitted_count"] == 1
