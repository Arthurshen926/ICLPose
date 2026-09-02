from __future__ import annotations

import json

import pytest

from feature_extract.vfm.localization_goal_maplet.chart_complement_held_plan import (
    build_complement_held_preexecution_plan,
    endpoint_inclusive_complement_rows,
)
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256


def _upstream(tmp_path):
    eligible = list(range(100, 110))
    names = [f"seq1__frame{value:05d}.png" for value in eligible]
    old = [100, 103, 106, 109]
    old_names = [names[eligible.index(value)] for value in old]
    payload = {
        "artifact_type": "goal_maplet_pose_only_chart_densification_plan_v2",
        "held_camera_fields_used_by_source_window_ranker": False,
        "other_route_camera_pose_fields_materialized": False,
        "points2D_or_point3D_fields_decoded": False,
        "query_or_forbidden_route_pose_fields_used": False,
        "uses_initializer_or_surface_geometry": False,
        "uses_query_or_ground_truth": False,
        "uses_rgb_numeric_fields": False,
        "held_camera_diagnostic_executed_only_after_source_freeze": True,
        "source_indices": [200, 201],
        "source_route": "seq4",
        "posed_colmap_root": str(tmp_path / "posed_colmap"),
        "source_ordered_names": ["seq4__a.png", "seq4__b.png"],
        "source_ordered_names_sha256": canonical_json_sha256(
            ["seq4__a.png", "seq4__b.png"]
        ),
        "held_post_selection_camera_neighbor_diagnostic": {
            "seq1": {
                "eligible_camera_count": len(eligible),
                "eligible_global_lexical_indices": eligible,
                "eligible_ordered_names": names,
                "selected_camera_count": len(old),
                "selected_global_lexical_indices": old,
                "selected_ordered_names": old_names,
            }
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "pose_only.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path, payload


def test_endpoint_inclusive_complement_selection_is_deterministic():
    selected, names, unused, unused_names = endpoint_inclusive_complement_rows(
        list(range(505, 535)),
        [f"frame{value}" for value in range(505, 535)],
        [505, 508, 510, 513, 516, 518, 521, 523, 526, 529, 531, 534],
        target_count=12,
    )
    assert selected == [506, 509, 511, 514, 515, 519, 520, 524, 525, 528, 530, 533]
    assert len(unused) == 18
    assert names == [f"frame{value}" for value in selected]
    assert len(unused_names) == 18


def test_complement_plan_is_hash_bound_and_geometry_preexecution_only(tmp_path):
    path, upstream = _upstream(tmp_path)
    plan = build_complement_held_preexecution_plan(
        path,
        expected_pose_only_plan_content_sha256=upstream["content_sha256"],
        intended_source_chart_plan_content_sha256="a" * 64,
        planned_isolation_root=tmp_path / "isolated",
        matcha_repo=tmp_path / "matcha",
        target_count=4,
    )
    assert plan["fresh_held_indices"] == [101, 104, 105, 108]
    assert plan["selection_frozen_before_new_held_geometry"] is True
    assert plan["new_held_geometry_used_for_selection"] is False
    assert plan["new_held_pointmaps_opened"] is False
    assert plan["new_rgb_or_pointmap_may_be_opened_before_contract_freeze"] is False
    assert "--held_indices" in plan["physical_isolation_input_builder_command"]
    assert not set(plan["fresh_held_indices"]) & set(
        plan["superseded_diagnostic_held_indices"]
    )


def test_complement_plan_rejects_stale_hash_and_insufficient_unused(tmp_path):
    path, upstream = _upstream(tmp_path)
    with pytest.raises(ValueError, match="external authority"):
        build_complement_held_preexecution_plan(
            path,
            expected_pose_only_plan_content_sha256="f" * 64,
            intended_source_chart_plan_content_sha256="a" * 64,
            planned_isolation_root=tmp_path / "isolated",
            matcha_repo=tmp_path / "matcha",
            target_count=4,
        )
    with pytest.raises(ValueError, match="smaller than the requested"):
        build_complement_held_preexecution_plan(
            path,
            expected_pose_only_plan_content_sha256=upstream["content_sha256"],
            intended_source_chart_plan_content_sha256="a" * 64,
            planned_isolation_root=tmp_path / "isolated",
            matcha_repo=tmp_path / "matcha",
            target_count=7,
        )
