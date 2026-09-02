"""Freeze a fresh held-camera complement before any new geometry is built."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .lineage import canonical_json_sha256, file_sha256


SCHEMA = "goal_maplet_complement_held_preexecution_plan_v1"
UPSTREAM_SCHEMA = "goal_maplet_pose_only_chart_densification_plan_v2"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def endpoint_inclusive_complement_rows(
    eligible_indices: list[int],
    eligible_names: list[str],
    excluded_indices: list[int],
    *,
    target_count: int,
) -> tuple[list[int], list[str], list[int], list[str]]:
    """Select deterministic endpoint-inclusive rows from the unused inventory."""

    if (
        len(eligible_indices) != len(eligible_names)
        or len(set(eligible_indices)) != len(eligible_indices)
        or len(set(eligible_names)) != len(eligible_names)
        or not eligible_indices
    ):
        raise ValueError("eligible held inventory is empty, duplicate, or ambiguous")
    excluded = set(excluded_indices)
    if len(excluded) != len(excluded_indices) or not excluded.issubset(
        set(eligible_indices)
    ):
        raise ValueError("excluded held inventory is duplicate or outside eligible rows")
    unused = [
        (index, name)
        for index, name in zip(eligible_indices, eligible_names)
        if index not in excluded
    ]
    if target_count < 2 or len(unused) < target_count:
        raise ValueError("unused held complement is smaller than the requested target")
    rows = np.linspace(0, len(unused) - 1, target_count).round().astype(np.int64)
    if len(set(rows.tolist())) != target_count:
        raise ValueError("endpoint-inclusive complement selection produced duplicates")
    selected = [unused[int(row)] for row in rows]
    return (
        [int(index) for index, _ in selected],
        [str(name) for _, name in selected],
        [int(index) for index, _ in unused],
        [str(name) for _, name in unused],
    )


def build_complement_held_preexecution_plan(
    pose_only_plan_path: Path,
    *,
    expected_pose_only_plan_content_sha256: str,
    intended_source_chart_plan_content_sha256: str,
    planned_isolation_root: Path,
    matcha_repo: Path,
    route: str = "seq1",
    target_count: int = 12,
) -> dict[str, object]:
    """Read only a frozen pose-only JSON and select unused held camera names."""

    if not _is_sha256(expected_pose_only_plan_content_sha256):
        raise ValueError("pose-only plan requires an external content SHA-256 pin")
    if not _is_sha256(intended_source_chart_plan_content_sha256):
        raise ValueError("intended source chart plan requires a content SHA-256 pin")
    pose_only_plan_path = Path(pose_only_plan_path)
    planned_isolation_root = Path(planned_isolation_root).resolve()
    matcha_repo = Path(matcha_repo).resolve()
    upstream = json.loads(pose_only_plan_path.read_text())
    claimed = upstream.pop("content_sha256", None)
    if claimed != canonical_json_sha256(upstream):
        raise ValueError("pose-only densification plan content hash differs")
    upstream["content_sha256"] = claimed
    if claimed != expected_pose_only_plan_content_sha256:
        raise ValueError("pose-only densification plan differs from external authority")
    required_false = (
        "held_camera_fields_used_by_source_window_ranker",
        "other_route_camera_pose_fields_materialized",
        "points2D_or_point3D_fields_decoded",
        "query_or_forbidden_route_pose_fields_used",
        "uses_initializer_or_surface_geometry",
        "uses_query_or_ground_truth",
        "uses_rgb_numeric_fields",
    )
    if (
        upstream.get("artifact_type") != UPSTREAM_SCHEMA
        or any(upstream.get(key) is not False for key in required_false)
        or upstream.get("held_camera_diagnostic_executed_only_after_source_freeze")
        is not True
    ):
        raise ValueError("upstream is not a route-clean pose-only selection authority")
    diagnostics = upstream.get("held_post_selection_camera_neighbor_diagnostic")
    if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get(route), dict):
        raise ValueError("requested held route is absent from the frozen diagnostic")
    row = diagnostics[route]
    eligible_indices = row.get("eligible_global_lexical_indices")
    eligible_names = row.get("eligible_ordered_names")
    old_indices = row.get("selected_global_lexical_indices")
    old_names = row.get("selected_ordered_names")
    if not all(
        isinstance(value, list)
        for value in (eligible_indices, eligible_names, old_indices, old_names)
    ):
        raise ValueError("frozen held inventory is incomplete")
    if (
        row.get("eligible_camera_count") != len(eligible_indices)
        or row.get("selected_camera_count") != len(old_indices)
        or len(old_indices) != len(old_names)
    ):
        raise ValueError("frozen held inventory counts differ")
    eligible_by_index = dict(zip(eligible_indices, eligible_names))
    if [eligible_by_index.get(index) for index in old_indices] != old_names:
        raise ValueError("old held names and indices differ from eligible inventory")
    selected_indices, selected_names, unused_indices, unused_names = (
        endpoint_inclusive_complement_rows(
            [int(value) for value in eligible_indices],
            [str(value) for value in eligible_names],
            [int(value) for value in old_indices],
            target_count=target_count,
        )
    )
    if set(selected_indices) & set(old_indices):
        raise AssertionError("fresh held complement overlaps the diagnostic held set")
    source_indices = upstream.get("source_indices")
    source_names = upstream.get("source_ordered_names")
    if (
        not isinstance(source_indices, list)
        or not isinstance(source_names, list)
        or len(source_indices) != len(source_names)
    ):
        raise ValueError("upstream source inventory is incomplete")
    plan: dict[str, object] = {
        "artifact_type": SCHEMA,
        "upstream_pose_only_plan_path": str(pose_only_plan_path.resolve()),
        "upstream_pose_only_plan_file_sha256": file_sha256(pose_only_plan_path),
        "upstream_pose_only_plan_content_sha256": claimed,
        "intended_source_chart_plan_content_sha256": (
            intended_source_chart_plan_content_sha256
        ),
        "source_indices": source_indices,
        "source_ordered_names": source_names,
        "source_ordered_names_sha256": upstream["source_ordered_names_sha256"],
        "held_route": route,
        "eligible_held_indices": eligible_indices,
        "eligible_held_ordered_names": eligible_names,
        "eligible_held_count": len(eligible_indices),
        "superseded_diagnostic_held_indices": old_indices,
        "superseded_diagnostic_held_ordered_names": old_names,
        "unused_complement_indices": unused_indices,
        "unused_complement_ordered_names": unused_names,
        "unused_complement_count": len(unused_indices),
        "fresh_held_indices": selected_indices,
        "fresh_held_ordered_names": selected_names,
        "fresh_held_ordered_names_sha256": canonical_json_sha256(selected_names),
        "fresh_held_count": len(selected_indices),
        "fresh_held_disjoint_from_superseded_diagnostic": True,
        "fresh_held_subset_of_pose_only_eligible_inventory": True,
        "selection_rule": (
            "remove all superseded diagnostic held indices from the frozen eligible "
            "route inventory; select round(linspace(0, unused_count-1, target_count)) "
            "in frozen eligible order, including both endpoints"
        ),
        "selection_row_indices_within_unused_complement": np.linspace(
            0, len(unused_indices) - 1, target_count
        ).round().astype(int).tolist(),
        "selection_frozen_before_new_held_geometry": True,
        "new_held_geometry_used_for_selection": False,
        "new_held_rgb_numeric_fields_used_for_selection": False,
        "new_held_pointmaps_opened": False,
        "held_camera_fields_materialized_by_this_planner": False,
        "source_selection_changed_by_fresh_held_choice": False,
        "recommended_isolated_source_global_indices": source_indices,
        "recommended_isolated_held_global_indices": selected_indices,
        "recommended_source_preexecution_image_idx": list(range(len(source_indices))),
        "recommended_held_preexecution_image_idx": list(range(len(selected_indices))),
        "planned_physical_isolation_root": str(planned_isolation_root),
        "planned_matcha_repo": str(matcha_repo),
        "physical_isolation_input_builder_command": [
            "python",
            "feature_extract/tools/vfm/build_goal_maplet_isolated_chart_sfm_inputs.py",
            "--posed_colmap",
            str(Path(upstream["posed_colmap_root"]).resolve()),
            "--output_root",
            str(planned_isolation_root / "inputs"),
            "--source_indices",
            *[str(value) for value in source_indices],
            "--held_indices",
            *[str(value) for value in selected_indices],
            "--source_routes",
            str(upstream["source_route"]),
            "--held_routes",
            route,
            "--forbidden_routes",
            "seq12",
            "seq14",
        ],
        "sfm_preexecution_contract_builder_command": [
            "python",
            "feature_extract/tools/vfm/build_goal_maplet_chart_sfm_preexecution_contract.py",
            "--isolated_inputs",
            str(planned_isolation_root / "inputs" / "manifest.json"),
            "--matcha_repo",
            str(matcha_repo),
            "--source_output",
            str(planned_isolation_root / "source_seq4_frames271_294_mast3r"),
            "--held_output",
            str(planned_isolation_root / "held_seq1_complement12_mast3r"),
            "--output",
            str(planned_isolation_root / "preexecution_contract.json"),
        ],
        "execution_order_contract": [
            "freeze_this_complement_plan_before_reading_new_RGB_or_pointmaps",
            "run_physical_isolation_input_builder_command",
            "run_sfm_preexecution_contract_builder_command_before_either_MASt3R_role",
            "run_only_the_exact_source_and_held_commands_emitted_by_preexecution_contract",
            "audit_new_disjoint_v2_authority_before_any_full16_initializer_or_alignment",
        ],
        "new_rgb_or_pointmap_may_be_opened_before_contract_freeze": False,
        "uses_query_or_ground_truth": False,
        "decision": "GO_fresh_complement_held_inventory_frozen_before_geometry",
    }
    plan["content_sha256"] = canonical_json_sha256(plan)
    return plan


__all__ = [
    "SCHEMA",
    "build_complement_held_preexecution_plan",
    "endpoint_inclusive_complement_rows",
]
