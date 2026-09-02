from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.run_goal_maplet_masked_chart_alignment_gate import (
    _alignment_code_inventory,
    _load_strict_comparison_domain,
    _matcha_internal_lexical_permutations,
    _strict_plan_selection,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    CARDINALITY_SCHEMA,
    PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
    PROJECTIVE_FORMAL_AUTHORITY_SEMANTICS,
    PROJECTIVE_FORMAL_EDGE_DEFINITION,
    ChartSubmapPlan,
    SCHEMA as PLAN_SCHEMA,
    load_model_neutral_alignment_selection,
    load_paired_stride2_diagnostic_alignment_selection,
    paired_stride2_diagnostic_alignment_metadata,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _save_plan(path, names, authority_hash, source_tree, *, diagnostic=False):
    count = len(names)
    zeros = np.zeros((count, count), np.float64)
    edges = np.ones((count, count), bool)
    np.fill_diagonal(edges, False)
    metadata = {
        "artifact_type": CARDINALITY_SCHEMA if diagnostic else PLAN_SCHEMA,
        "uses_query_or_ground_truth": False,
        "chart_count": count,
        "selected_chart_count": count,
        "source_ordered_names_sha256": canonical_json_sha256(names),
        "selected_chart_names_in_order": names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(names),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "components": [
            {
                "operational_coverage_pass": True,
                "selected_chart_names": names,
                "selected_chart_count": count,
            }
        ],
        "operational_submap_count": 1,
        "comparison_inventory_eligible": True,
        "system_control_only": False,
        "lineage": {
            "disjoint_authority_schema": "goal_maplet_disjoint_chart_upstream_authority_v2",
            "disjoint_authority_content_sha256": authority_hash,
            "source_tree_sha256": source_tree,
        },
    }
    if diagnostic:
        metadata.update(
            {
                "representation": "offline_source_physical_seam_filtered_chart_selection",
                "comparison_inventory_eligible": False,
                "system_control_only": True,
                "config": {
                    "minimum_selected_charts_per_submap": 2,
                    "maximum_selected_charts_per_submap": max(2, count),
                },
                "selection_cardinality_frozen_before_held_geometry": True,
                "held_geometry_used_for_selection": False,
                "topology_stride": 2,
                "paired_stride2_densification_diagnostic": True,
                "source_seam_authority_formal_selector_handoff_eligible": True,
                "edge_formal_valid_explicitly_sealed": True,
                "edge_formal_valid_source": "authority_metadata_sealed_mask",
                "edge_formal_valid_definition": PROJECTIVE_FORMAL_EDGE_DEFINITION,
                "source_geometry_selection_eligible": True,
                "source_geometry_selection_scope": (
                    "paired_stride2_diagnostic_selector_handoff_only"
                ),
                "topology_caveat": PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
                "comparison_domain_v3_role": "common_valid_and_lineage_parent_only",
                "source_seam_authority_production_eligible": False,
                "projective_correspondence_production_candidate": False,
                "final_model_neutral_map_topology_eligible": False,
                "promotion_eligible": False,
                "formal_selector_handoff_is_not_production_authority": True,
                "diagnostic_alignment_adapter_required": True,
                "full_gate_or_exporter_consumption_eligible": False,
                "full_gate_fail_closed_by_system_control_only": True,
                "model_neutral_alignment_loader_fail_closed": True,
                "legacy_closest_surface_diagnostic_only": False,
                "output_coverage_edges_all_projective_formal": True,
                "output_alignment_edges_all_projective_formal": True,
                "selection_geometry_source": [
                    "source_only_MASt3R_reference_on_exact_stride2_topology"
                ],
                "official_order_preserved_for_runner": True,
                "uses_mapping_rgb": False,
                "route_clean": True,
                "source_seam_authority_semantics_version": (
                    PROJECTIVE_FORMAL_AUTHORITY_SEMANTICS
                ),
            }
        )
        metadata["lineage"].update(
            {
                "paired_stride2_densification_diagnostic": True,
                "projective_topology_stride": 2,
                "topology_caveat": PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
                "source_seam_authority_formal_selector_handoff_eligible": True,
                "source_seam_authority_production_eligible": False,
                "source_seam_authority_semantics_version": (
                    PROJECTIVE_FORMAL_AUTHORITY_SEMANTICS
                ),
                "query_or_ground_truth_consumed": False,
                "held_root_opened_by_selector": False,
                "source_rgb_numeric_fields_used_by_selector": False,
                "source_tree_bytes_replayed_by_selector": True,
                **{
                    key: "c" * 64
                    for key in (
                        "paired_stride2_topology_arrays_sha256",
                        "paired_stride2_topology_content_sha256",
                        "paired_stride2_topology_file_sha256",
                        "reference_safe_v3_domain_content_sha256",
                        "reference_safe_v3_domain_file_sha256",
                        "source_seam_authority_content_sha256",
                        "source_seam_authority_file_sha256",
                        "edge_formal_valid_sha256",
                        "source_seam_m0_metrics_sha256",
                    )
                },
            }
        )
    plan = ChartSubmapPlan(
        chart_names=np.asarray(names),
        camera_centers_world=np.stack(
            [np.asarray([row, 0.0, 0.0]) for row in range(count)]
        ),
        camera_forward_world=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1)),
        valid_sample_counts=np.full((count,), 100, np.int64),
        directional_frustum_fraction=zeros.copy(),
        directional_depth_support=zeros.copy(),
        directional_surface_support=zeros.copy(),
        symmetric_surface_overlap=zeros.copy(),
        camera_baseline_m=zeros.copy(),
        median_overlap_depth_m=zeros.copy(),
        baseline_to_depth_ratio=zeros.copy(),
        median_triangulation_angle_deg=zeros.copy(),
        camera_forward_angle_deg=zeros.copy(),
        same_surface_side_fraction=zeros.copy(),
        coverage_edges=edges.copy(),
        alignment_edges=edges.copy(),
        component_ids=np.zeros((count,), np.int32),
        selected_mask=np.ones((count,), bool),
        selection_rank=np.arange(count, dtype=np.int32),
        metadata=metadata,
    )
    return plan.save_npz(path)


def _fixture(tmp_path, *, diagnostic=False):
    names = ["seq4__a.png", "seq4__b.png"]
    source_tree = "a" * 64
    authority = {
        "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
        "content_sha256": "b" * 64,
        "source": {"tree_sha256": source_tree},
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))
    cameras_path = tmp_path / "cameras.json"
    cameras_path.write_text(
        json.dumps(
            {
                "filepaths": [str(tmp_path / name) for name in names],
                "focals": [10.0, 10.0],
                "cams2world": np.repeat(np.eye(4)[None], 2, axis=0).tolist(),
            }
        )
    )
    pointmaps = tmp_path / "pointmaps"
    pointmaps.mkdir()
    initializers = tmp_path / "initializers"
    initializers.mkdir()
    for row, name in enumerate(names):
        (pointmaps / f"{name[:-4]}.json").write_text(json.dumps({"row": row}))
        np.savez(initializers / f"{name}.npz", value=np.asarray([row]))
    plan_path = tmp_path / "plan.npz"
    plan_metadata = _save_plan(
        plan_path,
        names,
        authority["content_sha256"],
        source_tree,
        diagnostic=diagnostic,
    )
    arrays = {
        "chart_names": np.asarray(names),
        "valid": np.ones((2, 4, 5), bool),
        "face_valid_stride4": np.ones((2, 1, 1), bool),
        "face_valid_stride8": np.zeros((2, 0, 0), bool),
    }
    metadata = {
        "artifact_type": "goal_maplet_chart_comparison_domain_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority["content_sha256"],
        "source_tree_sha256": source_tree,
        "frozen_submap_plan_file_sha256": file_sha256(plan_path),
        "frozen_submap_plan_content_sha256": plan_metadata["content_sha256"],
        "selected_chart_names_in_order_sha256": canonical_json_sha256(names),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "cameras_file_sha256": file_sha256(cameras_path),
        "pointmap_inventory": {
            name: file_sha256(pointmaps / f"{name[:-4]}.json") for name in names
        },
        "dav2_initializer_file_sha256": {
            name: file_sha256(initializers / f"{name}.npz") for name in names
        },
        "uses_query_or_ground_truth": False,
    }
    if diagnostic:
        plan = ChartSubmapPlan.load_npz(plan_path)
        metadata.update(paired_stride2_diagnostic_alignment_metadata(plan))
        metadata.update(
            {
                "diagnostic_only": True,
                "production_eligible": False,
                "optimizer_pixel_domain_only": True,
                "paired_stride2_topology_embedded": False,
                "full_gate_exact_topology_sealer_eligible": False,
            }
        )
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    domain_path = tmp_path / "domain.npz"
    np.savez_compressed(
        domain_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "names": names,
        "authority": authority,
        "authority_path": authority_path,
        "cameras_path": cameras_path,
        "pointmaps": pointmaps,
        "initializers": initializers,
        "plan_path": plan_path,
        "plan_hash": plan_metadata["content_sha256"],
        "domain_path": domain_path,
        "domain_hash": metadata["content_sha256"],
    }


def test_strict_runner_uses_exact_plan_and_replays_domain_lineage(tmp_path):
    value = _fixture(tmp_path)
    selected, plan = _strict_plan_selection(
        value["plan_path"],
        expected_plan_content_sha256=value["plan_hash"],
        authority=value["authority"],
        cameras=json.loads(value["cameras_path"].read_text()),
    )
    assert selected == value["names"]
    arrays, metadata = _load_strict_comparison_domain(
        value["domain_path"],
        expected_content_sha256=value["domain_hash"],
        selected_names=selected,
        authority_path=value["authority_path"],
        authority=value["authority"],
        plan_path=value["plan_path"],
        plan=plan,
        cameras_path=value["cameras_path"],
        pointmaps_dir=value["pointmaps"],
        initializer="dav2",
        initializer_path=value["initializers"],
    )
    assert arrays["chart_names"].tolist() == value["names"]
    assert metadata["frozen_submap_plan_content_sha256"] == value["plan_hash"]


def test_matcha_internal_sort_is_inverted_back_to_official_plan_order():
    official = [
        "seq4__frame00275.png",
        "seq4__frame00286.png",
        "seq4__frame00282.png",
    ]
    internal, restore = _matcha_internal_lexical_permutations(official)
    assert internal.tolist() == [0, 2, 1]
    lexical_values = np.asarray([official[row] for row in internal])
    assert lexical_values[restore].tolist() == official

    cross_directory_paths = ["/z/seq4__a.png", "/a/seq4__b.png"]
    path_internal, path_restore = _matcha_internal_lexical_permutations(
        ["seq4__a.png", "seq4__b.png"], cross_directory_paths
    )
    assert path_internal.tolist() == [1, 0]
    assert path_internal[path_restore].tolist() == [0, 1]

    four_names = ["c.png", "a.png", "d.png", "b.png"]
    four_internal, four_restore = _matcha_internal_lexical_permutations(four_names)
    assert four_internal.tolist() == [1, 3, 0, 2]
    assert np.asarray(four_names)[four_internal][four_restore].tolist() == four_names
    assert four_restore.tolist() != four_internal.tolist()

    with pytest.raises(ValueError, match="duplicate"):
        _matcha_internal_lexical_permutations([official[0], official[0]])
    with pytest.raises(ValueError, match="path inventory contains duplicates"):
        _matcha_internal_lexical_permutations(
            ["a.png", "b.png"], ["/same/path.png", "/same/path.png"]
        )


def test_strict_runner_rejects_plan_domain_or_initializer_substitution(tmp_path):
    value = _fixture(tmp_path)
    with pytest.raises(ValueError, match="runner authority"):
        _strict_plan_selection(
            value["plan_path"],
            expected_plan_content_sha256="f" * 64,
            authority=value["authority"],
            cameras=json.loads(value["cameras_path"].read_text()),
        )
    selected, plan = _strict_plan_selection(
        value["plan_path"],
        expected_plan_content_sha256=value["plan_hash"],
        authority=value["authority"],
        cameras=json.loads(value["cameras_path"].read_text()),
    )
    with pytest.raises(ValueError, match="domain differs from runner authority"):
        _load_strict_comparison_domain(
            value["domain_path"],
            expected_content_sha256="e" * 64,
            selected_names=selected,
            authority_path=value["authority_path"],
            authority=value["authority"],
            plan_path=value["plan_path"],
            plan=plan,
            cameras_path=value["cameras_path"],
            pointmaps_dir=value["pointmaps"],
            initializer="dav2",
            initializer_path=value["initializers"],
        )
    np.savez(value["initializers"] / f"{selected[0]}.npz", value=np.asarray([99]))
    with pytest.raises(ValueError, match="initializer inventory differs"):
        _load_strict_comparison_domain(
            value["domain_path"],
            expected_content_sha256=value["domain_hash"],
            selected_names=selected,
            authority_path=value["authority_path"],
            authority=value["authority"],
            plan_path=value["plan_path"],
            plan=plan,
            cameras_path=value["cameras_path"],
            pointmaps_dir=value["pointmaps"],
            initializer="dav2",
            initializer_path=value["initializers"],
        )


def test_alignment_code_inventory_detects_python_source_substitution(tmp_path):
    source = tmp_path / "matcha" / "dm_scene"
    source.mkdir(parents=True)
    path = source / "parallel_aligner.py"
    path.write_text("MASK_CONTRACT = True\n")
    rows, first = _alignment_code_inventory(tmp_path)
    assert rows[0]["path"] == "matcha/dm_scene/parallel_aligner.py"
    path.write_text("MASK_CONTRACT = False\n")
    _, second = _alignment_code_inventory(tmp_path)
    assert first != second


def test_stride2_diagnostic_adapter_is_explicit_and_model_neutral_loader_rejects(
    tmp_path,
):
    value = _fixture(tmp_path, diagnostic=True)
    with pytest.raises(ValueError, match="not eligible"):
        load_model_neutral_alignment_selection(
            value["plan_path"],
            expected_plan_content_sha256=value["plan_hash"],
        )
    with pytest.raises(ValueError, match="explicit adapter opt-in"):
        load_paired_stride2_diagnostic_alignment_selection(
            value["plan_path"],
            expected_plan_content_sha256=value["plan_hash"],
        )
    selection = load_paired_stride2_diagnostic_alignment_selection(
        value["plan_path"],
        expected_plan_content_sha256=value["plan_hash"],
        diagnostic_alignment_adapter_opt_in=True,
    )
    assert list(selection.ordered_names) == value["names"]


def test_stride2_diagnostic_runner_and_domain_both_require_adapter_opt_in(tmp_path):
    value = _fixture(tmp_path, diagnostic=True)
    cameras = json.loads(value["cameras_path"].read_text())
    with pytest.raises(ValueError, match="not eligible"):
        _strict_plan_selection(
            value["plan_path"],
            expected_plan_content_sha256=value["plan_hash"],
            authority=value["authority"],
            cameras=cameras,
        )
    selected, plan = _strict_plan_selection(
        value["plan_path"],
        expected_plan_content_sha256=value["plan_hash"],
        authority=value["authority"],
        cameras=cameras,
        allow_paired_stride2_diagnostic_alignment_adapter=True,
    )
    with pytest.raises(ValueError, match="requires explicit adapter opt-in"):
        _load_strict_comparison_domain(
            value["domain_path"],
            expected_content_sha256=value["domain_hash"],
            selected_names=selected,
            authority_path=value["authority_path"],
            authority=value["authority"],
            plan_path=value["plan_path"],
            plan=plan,
            cameras_path=value["cameras_path"],
            pointmaps_dir=value["pointmaps"],
            initializer="dav2",
            initializer_path=value["initializers"],
        )
    arrays, metadata = _load_strict_comparison_domain(
        value["domain_path"],
        expected_content_sha256=value["domain_hash"],
        selected_names=selected,
        authority_path=value["authority_path"],
        authority=value["authority"],
        plan_path=value["plan_path"],
        plan=plan,
        cameras_path=value["cameras_path"],
        pointmaps_dir=value["pointmaps"],
        initializer="dav2",
        initializer_path=value["initializers"],
        allow_paired_stride2_diagnostic_alignment_adapter=True,
    )
    assert arrays["chart_names"].tolist() == value["names"]
    assert metadata["system_control_only"] is True
    assert metadata["promotion_eligible"] is False


def test_stride2_adapter_rejects_rehashed_promotion_or_caveat_tamper(tmp_path):
    value = _fixture(tmp_path, diagnostic=True)
    plan = ChartSubmapPlan.load_npz(value["plan_path"])
    for key, bad_value in (
        ("promotion_eligible", True),
        ("topology_caveat", "model_neutral"),
        ("edge_formal_valid_explicitly_sealed", False),
    ):
        metadata = json.loads(json.dumps(plan.metadata))
        metadata.pop("content_sha256", None)
        metadata.pop("arrays_sha256", None)
        metadata[key] = bad_value
        tampered = ChartSubmapPlan(metadata=metadata, **plan.arrays())
        path = tmp_path / f"tampered_{key}.npz"
        sealed = tampered.save_npz(path)
        with pytest.raises(ValueError, match=key):
            load_paired_stride2_diagnostic_alignment_selection(
                path,
                expected_plan_content_sha256=sealed["content_sha256"],
                diagnostic_alignment_adapter_opt_in=True,
            )
