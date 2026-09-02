from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    CARDINALITY_SCHEMA,
    PROJECT_SUPPORT_SEMANTICS,
    PROJECTION_PRINCIPAL_POINT_CONVENTION,
    ChartSubmapPlan,
    load_model_neutral_alignment_selection,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_seam_chart_selection import (
    LEGACY_AUTHORITY_SEMANTICS,
    LEGACY_EDGE_FORMAL_VALID_DEFINITION,
    PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
    PRODUCTION_AUTHORITY_SEMANTICS,
    PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
    PhysicalSeamSelectionConfig,
    select_physical_seam_chart_submap_plan,
)


def _plan(count: int, edges: list[tuple[int, int]]) -> ChartSubmapPlan:
    names = np.asarray([f"seq4__frame{row:05d}.png" for row in range(count)])
    graph = np.zeros((count, count), bool)
    for first, second in edges:
        graph[first, second] = graph[second, first] = True
    directional = np.full((count, count), 0.30, np.float32)
    np.fill_diagonal(directional, 1.0)
    symmetric = np.minimum(directional, directional.T)
    np.fill_diagonal(symmetric, 0.0)
    zeros = np.zeros((count, count), np.float32)
    ones = np.ones((count, count), np.float32)
    selected_names = names.astype(str).tolist()
    config = {
        "minimum_selected_charts_per_submap": count,
        "maximum_selected_charts_per_submap": count,
        "target_per_view_surface_support": 0.25,
        "sample_stride": 4,
        "maximum_self_reprojection_p90_fraction_of_sample_stride": 0.50,
        "maximum_aggregate_median_reprojection_bias_px": 0.25,
    }
    self_reprojection = {
        str(name): {
            "valid_count": 100,
            "median_dx_px": 0.0,
            "median_dy_px": 0.0,
            "median_px": 0.0,
            "p90_px": 0.0,
            "max_px": 0.0,
        }
        for name in names
    }
    metadata = {
        "artifact_type": CARDINALITY_SCHEMA,
        "chart_count": count,
        "selected_chart_count": count,
        "source_ordered_names_sha256": canonical_json_sha256(selected_names),
        "selected_chart_names_in_order": selected_names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(
            selected_names
        ),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "operational_submap_count": 1,
        "uses_query_or_ground_truth": False,
        "held_geometry_used_for_selection": False,
        "selection_cardinality_frozen_before_held_geometry": True,
        "comparison_inventory_eligible": True,
        "system_control_only": False,
        "uses_mapping_camera_pose": True,
        "project_support_semantics_version": PROJECT_SUPPORT_SEMANTICS,
        "projection_principal_point_convention": (
            PROJECTION_PRINCIPAL_POINT_CONVENTION
        ),
        "project_support_self_reprojection_floor_pass": True,
        "project_support_self_reprojection_shape_pass": True,
        "project_support_self_reprojection_phase_pass": True,
        "project_support_self_reprojection_p90_threshold_px": 2.0,
        "project_support_aggregate_median_dx_px": 0.0,
        "project_support_aggregate_median_dy_px": 0.0,
        "project_support_self_reprojection_metrics": self_reprojection,
        "config": config,
        "components": [
            {
                "operational_coverage_pass": True,
                "selected_chart_count": count,
                "selected_chart_names": selected_names,
            }
        ],
        "lineage": {
            "disjoint_authority_schema": (
                "goal_maplet_disjoint_chart_upstream_authority_v2"
            ),
            "source_tree_sha256": "a" * 64,
            "comparison_inventory_eligible": True,
            "system_control_only": False,
        },
    }
    return ChartSubmapPlan(
        chart_names=names,
        camera_centers_world=np.stack(
            (np.arange(count), np.zeros(count), np.zeros(count)), axis=1
        ).astype(np.float64),
        camera_forward_world=np.tile([0.0, 0.0, 1.0], (count, 1)),
        valid_sample_counts=np.arange(count, dtype=np.int64) + 100,
        directional_frustum_fraction=directional,
        directional_depth_support=directional,
        directional_surface_support=directional,
        symmetric_surface_overlap=symmetric,
        camera_baseline_m=ones,
        median_overlap_depth_m=ones,
        baseline_to_depth_ratio=ones,
        median_triangulation_angle_deg=ones,
        camera_forward_angle_deg=zeros,
        same_surface_side_fraction=ones,
        coverage_edges=graph,
        alignment_edges=graph.copy(),
        component_ids=np.zeros(count, np.int32),
        selected_mask=np.ones(count, bool),
        selection_rank=np.arange(count, dtype=np.int32),
        metadata=metadata,
    ).validated()


def _m0_rows(
    plan: ChartSubmapPlan,
    edges: list[tuple[int, int]],
    formal: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for index, (first, second) in enumerate(edges):
        passed = bool(formal[index])
        rows.append(
            {
                "first": str(plan.chart_names[first]),
                "second": str(plan.chart_names[second]),
                "m0_reachability_pass": passed,
                "geometry_pass": passed,
                "supported_fraction_by_direction": [0.25, 0.20],
            }
        )
    return rows


def test_synthetic_unreachable_edge_is_not_allowed_to_keep_isolated_chart(tmp_path):
    # Coarse coverage says chart 15 is attached to the route.  The sealed
    # projective M0 mask rejects its only edge, so it must disappear rather
    # than making the physical chart graph formally unreachable.
    edges = [(row, row + 1) for row in range(15)]
    plan = _plan(16, edges)
    formal = np.ones(len(edges), bool)
    formal[-1] = False
    directional_formal = np.repeat(formal[:, None], 2, axis=1)
    support = np.tile([0.25, 0.20], (len(edges), 1))
    result = select_physical_seam_chart_submap_plan(
        plan,
        domain_chart_names=plan.chart_names,
        authority_edge_chart_indices=np.asarray(edges, np.int64),
        source_supported_fraction_by_direction=support,
        direction_formal_valid=directional_formal,
        direction_support_valid=directional_formal,
        direction_geometry_valid=np.ones_like(directional_formal),
        edge_formal_valid=formal,
        edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
        authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
        production_authority=True,
        final_model_neutral_map_topology_eligible=True,
        config=PhysicalSeamSelectionConfig(),
    )
    assert result.plan.chart_count == 15
    assert result.plan.selected_chart_names_in_order == tuple(
        plan.chart_names[:15].astype(str).tolist()
    )
    assert result.audit["isolated_candidate_names"] == [str(plan.chart_names[15])]
    assert result.audit["promotion_eligible"] is True
    rejected_edge = result.audit["source_formal_edge_inventory"][-1]
    assert rejected_edge["rejection_reasons"] == [
        "first_to_second_support_gate_failed",
        "second_to_first_support_gate_failed",
    ]
    assert result.plan.metadata["edge_formal_valid_explicitly_sealed"] is True
    assert np.all(result.plan.alignment_edges.sum(1) > 0)

    path = tmp_path / "physical_plan.npz"
    metadata = result.plan.save_npz(path)
    replay = load_model_neutral_alignment_selection(
        path, expected_plan_content_sha256=metadata["content_sha256"]
    )
    assert replay.ordered_names == result.plan.selected_chart_names_in_order


def test_legacy_closest_surface_result_is_structural_dry_run_not_authority(tmp_path):
    edges = [(row, row + 1) for row in range(12)]
    plan = _plan(13, edges)
    formal = np.ones(len(edges), bool)
    directional_formal = np.repeat(formal[:, None], 2, axis=1)
    support = np.tile([0.25, 0.20], (len(edges), 1))
    result = select_physical_seam_chart_submap_plan(
        plan,
        domain_chart_names=plan.chart_names,
        authority_edge_chart_indices=np.asarray(edges, np.int64),
        source_supported_fraction_by_direction=support,
        direction_formal_valid=directional_formal,
        m0_per_edge=_m0_rows(plan, edges, formal),
        edge_formal_valid=formal,
        edge_formal_valid_definition=LEGACY_EDGE_FORMAL_VALID_DEFINITION,
        authority_semantics_version=LEGACY_AUTHORITY_SEMANTICS,
        production_authority=False,
        final_model_neutral_map_topology_eligible=False,
        config=PhysicalSeamSelectionConfig(
            diagnostic_allow_legacy_closest_surface=True
        ),
    )
    assert result.plan.metadata["promotion_eligible"] is False
    assert result.plan.metadata["comparison_inventory_eligible"] is False
    assert result.plan.metadata["system_control_only"] is True
    assert result.plan.metadata["lineage"]["comparison_inventory_eligible"] is False
    assert result.plan.metadata["lineage"]["system_control_only"] is True
    assert (
        result.plan.metadata["lineage"][
            "upstream_comparison_inventory_eligible"
        ]
        is True
    )
    assert result.plan.metadata["lineage"]["upstream_system_control_only"] is False
    assert result.audit["existing_alignment_runner_structurally_compatible"] is True
    assert result.audit["existing_alignment_runner_authority_eligible"] is False
    path = tmp_path / "diagnostic_plan.npz"
    metadata = result.plan.save_npz(path)
    with pytest.raises(ValueError, match="not eligible"):
        load_model_neutral_alignment_selection(
            path, expected_plan_content_sha256=metadata["content_sha256"]
        )


def test_eighteen_candidate_selection_is_deterministic_connected_exact_sixteen():
    count = 18
    edges = [(first, second) for first in range(count) for second in range(first + 1, count)]
    plan = _plan(count, edges)
    formal = np.ones(len(edges), bool)
    directional_formal = np.repeat(formal[:, None], 2, axis=1)
    support = np.tile([0.25, 0.20], (len(edges), 1))
    arguments = dict(
        domain_chart_names=plan.chart_names,
        authority_edge_chart_indices=np.asarray(edges, np.int64),
        source_supported_fraction_by_direction=support,
        direction_formal_valid=directional_formal,
        direction_support_valid=directional_formal,
        direction_geometry_valid=np.ones_like(directional_formal),
        edge_formal_valid=formal,
        edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
        authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
        production_authority=True,
        final_model_neutral_map_topology_eligible=True,
        config=PhysicalSeamSelectionConfig(),
    )
    first = select_physical_seam_chart_submap_plan(plan, **arguments)
    second = select_physical_seam_chart_submap_plan(plan, **arguments)
    assert first.plan.chart_count == 16
    assert first.plan.selected_chart_names_in_order == (
        second.plan.selected_chart_names_in_order
    )
    assert arrays_sha256(first.plan.arrays()) == arrays_sha256(second.plan.arrays())
    assert first.audit["selected_subgraph_metrics"]["connected"] is True


def test_paired_stride2_formal_handoff_is_control_only_and_not_promotable(
    tmp_path,
):
    count = 18
    edges = [
        (first, second)
        for first in range(count)
        for second in range(first + 1, count)
    ]
    plan = _plan(count, edges)
    formal = np.ones(len(edges), bool)
    directional = np.repeat(formal[:, None], 2, axis=1)
    result = select_physical_seam_chart_submap_plan(
        plan,
        domain_chart_names=plan.chart_names,
        authority_edge_chart_indices=np.asarray(edges, np.int64),
        source_supported_fraction_by_direction=np.tile(
            [0.25, 0.20], (len(edges), 1)
        ),
        direction_formal_valid=directional,
        direction_support_valid=directional,
        direction_geometry_valid=directional,
        edge_formal_valid=formal,
        edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
        authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
        production_authority=False,
        formal_selector_handoff_eligible=True,
        topology_stride=2,
        paired_stride2_densification_diagnostic=True,
        topology_caveat=PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
        final_model_neutral_map_topology_eligible=False,
        config=PhysicalSeamSelectionConfig(),
    )
    assert result.plan is not None
    assert result.plan.chart_count == 16
    assert result.plan.metadata["source_seam_authority_production_eligible"] is False
    assert (
        result.plan.metadata[
            "source_seam_authority_formal_selector_handoff_eligible"
        ]
        is True
    )
    assert result.plan.metadata["source_geometry_selection_eligible"] is True
    assert result.plan.metadata["promotion_eligible"] is False
    assert result.plan.metadata["comparison_inventory_eligible"] is False
    assert result.plan.metadata["system_control_only"] is True
    assert result.plan.metadata["lineage"]["comparison_inventory_eligible"] is False
    assert result.plan.metadata["lineage"]["system_control_only"] is True
    assert (
        result.plan.metadata["lineage"][
            "upstream_comparison_inventory_eligible"
        ]
        is True
    )
    assert result.plan.metadata["lineage"]["upstream_system_control_only"] is False
    assert result.plan.metadata["topology_stride"] == 2
    assert result.plan.metadata["selection_geometry_source"] == [
        "source_only_MASt3R_reference_on_exact_stride2_topology"
    ]
    assert result.audit["existing_alignment_runner_authority_eligible"] is False
    assert result.audit["diagnostic_alignment_adapter_required"] is True
    assert result.audit["full_gate_or_exporter_consumption_eligible"] is False
    assert result.audit["full_gate_fail_closed_contract_audit"] == {
        "comparison_inventory_eligible_required": True,
        "comparison_inventory_eligible_observed": False,
        "system_control_only_required": False,
        "system_control_only_observed": True,
        "eligible": False,
    }
    assert result.audit["output_coverage_edges_all_projective_formal"] is True
    assert result.audit["output_alignment_edges_all_projective_formal"] is True
    path = tmp_path / "stride2_control_only_plan.npz"
    metadata = result.plan.save_npz(path)
    with pytest.raises(ValueError, match="not eligible"):
        load_model_neutral_alignment_selection(
            path, expected_plan_content_sha256=metadata["content_sha256"]
        )


def test_sealed_formal_mask_cannot_promote_a_replayed_m0_failure():
    edges = [(row, row + 1) for row in range(12)]
    plan = _plan(13, edges)
    formal = np.ones(len(edges), bool)
    rows = _m0_rows(plan, edges, formal)
    rows[0]["geometry_pass"] = False
    with pytest.raises(ValueError, match="sealed formal edge passes"):
        select_physical_seam_chart_submap_plan(
            plan,
            domain_chart_names=plan.chart_names,
            authority_edge_chart_indices=np.asarray(edges, np.int64),
            source_supported_fraction_by_direction=np.tile(
                [0.25, 0.20], (len(edges), 1)
            ),
            direction_formal_valid=np.repeat(formal[:, None], 2, axis=1),
            direction_support_valid=np.repeat(formal[:, None], 2, axis=1),
            direction_geometry_valid=np.ones((len(edges), 2), bool),
            m0_per_edge=rows,
            edge_formal_valid=formal,
            edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
            authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
            production_authority=True,
            final_model_neutral_map_topology_eligible=True,
            config=PhysicalSeamSelectionConfig(),
        )


def test_production_edge_mask_must_be_and_of_independent_directions():
    edges = [(row, row + 1) for row in range(12)]
    plan = _plan(13, edges)
    edge_formal = np.ones(len(edges), bool)
    direction_formal = np.ones((len(edges), 2), bool)
    direction_formal[0, 1] = False
    with pytest.raises(ValueError, match="AND of two independent directions"):
        select_physical_seam_chart_submap_plan(
            plan,
            domain_chart_names=plan.chart_names,
            authority_edge_chart_indices=np.asarray(edges, np.int64),
            source_supported_fraction_by_direction=np.tile(
                [0.25, 0.20], (len(edges), 1)
            ),
            direction_formal_valid=direction_formal,
            direction_support_valid=direction_formal,
            direction_geometry_valid=np.ones_like(direction_formal),
            edge_formal_valid=edge_formal,
            edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
            authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
            production_authority=True,
            final_model_neutral_map_topology_eligible=True,
            config=PhysicalSeamSelectionConfig(),
        )


def test_production_selector_rejects_legacy_width_over_two_plan():
    edges = [(row, row + 1) for row in range(12)]
    plan = _plan(13, edges)
    changed = dict(plan.metadata)
    changed["projection_principal_point_convention"] = "legacy_width/2_height/2"
    legacy_plan = replace(plan, metadata=changed).validated()
    formal = np.ones(len(edges), bool)
    with pytest.raises(ValueError, match="legacy width/2"):
        select_physical_seam_chart_submap_plan(
            legacy_plan,
            domain_chart_names=legacy_plan.chart_names,
            authority_edge_chart_indices=np.asarray(edges, np.int64),
            source_supported_fraction_by_direction=np.tile(
                [0.25, 0.20], (len(edges), 1)
            ),
            direction_formal_valid=np.repeat(formal[:, None], 2, axis=1),
            direction_support_valid=np.repeat(formal[:, None], 2, axis=1),
            direction_geometry_valid=np.ones((len(edges), 2), bool),
            edge_formal_valid=formal,
            edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
            authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
            production_authority=True,
            final_model_neutral_map_topology_eligible=True,
            config=PhysicalSeamSelectionConfig(),
        )


def test_no_twelve_chart_connected_component_fails_closed():
    edges = [(row, row + 1) for row in range(10)] + [
        (row, row + 1) for row in range(11, 15)
    ]
    plan = _plan(16, edges)
    formal = np.ones(len(edges), bool)
    with pytest.raises(ValueError, match="no connected component"):
        select_physical_seam_chart_submap_plan(
            plan,
            domain_chart_names=plan.chart_names,
            authority_edge_chart_indices=np.asarray(edges, np.int64),
            source_supported_fraction_by_direction=np.tile(
                [0.25, 0.20], (len(edges), 1)
            ),
            direction_formal_valid=np.repeat(formal[:, None], 2, axis=1),
            direction_support_valid=np.repeat(formal[:, None], 2, axis=1),
            direction_geometry_valid=np.ones((len(edges), 2), bool),
            edge_formal_valid=formal,
            edge_formal_valid_definition=PRODUCTION_EDGE_FORMAL_VALID_DEFINITION,
            authority_semantics_version=PRODUCTION_AUTHORITY_SEMANTICS,
            production_authority=True,
            final_model_neutral_map_topology_eligible=True,
            config=PhysicalSeamSelectionConfig(),
        )
