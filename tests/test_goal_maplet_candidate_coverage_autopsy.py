from feature_extract.tools.vfm.audit_goal_maplet_candidate_coverage import (
    _candidate_coverage,
    _classify,
    _grid_graph_diameter,
)

import numpy as np


def _coverage(strict=False, one_m=False):
    return {
        "strict_available": strict,
        "one_m_available": one_m,
    }


def _oracle(translation=None, rotation=None):
    return {
        "translation_m": translation,
        "rotation_deg": rotation,
    }


def _row():
    return {
        "candidate_ladder": {
            "baseline_post_nms": _coverage(),
            "conditioned_child_post_nms": _coverage(),
            "conditioned_child_pre_nms": _coverage(),
            "oracle_parent_actual_child": _coverage(),
            "oracle_parent_oracle_child": _coverage(),
        },
        "pose_sufficient_recall": {
            "parent_structurally_sufficient": True,
            "child_structurally_sufficient": True,
        },
        "pose_oracles": {
            "posterior_supported_child_center": _oracle(0.4, 2.0),
            "truth_child_center": _oracle(0.4, 2.0),
            "truth_exact_surface_bbox_center": _oracle(0.2, 1.0),
            "truth_exact_surface_weighted_xy": _oracle(0.0, 0.0),
        },
    }


def test_autopsy_classifies_configuration_and_retention_failures():
    row = _row()
    assert _classify(row) == "B1_configuration_inference"
    row["candidate_ladder"]["conditioned_child_pre_nms"] = _coverage(one_m=True)
    assert _classify(row) == "B3_proposal_retention"
    row["candidate_ladder"]["conditioned_child_post_nms"] = _coverage(one_m=True)
    assert _classify(row) == "fixed_conditioned_child_enumeration"


def test_grid_graph_diameter_measures_connected_chain_not_bbox_only():
    xy = np.asarray([[0, 0], [1, 0], [2, 0], [2, 1]], dtype=np.int64)
    assert _grid_graph_diameter(xy) == 3


def test_candidate_coverage_uses_all_candidates_not_only_top1():
    rows = {
        "seq/a.png": {
            "mode_details": {
                "actual_parent_actual_child": [
                    {"translation_m": 4.0, "rotation_deg": 2.0},
                    {"translation_m": 0.4, "rotation_deg": 3.0},
                ]
            }
        }
    }
    value = _candidate_coverage(rows, "seq/a.png")
    assert value["candidate_count"] == 2
    assert value["strict_available"] is True
    assert value["one_m_available"] is True
