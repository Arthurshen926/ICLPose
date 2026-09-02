from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    build_exact_topology_arrays,
    exact_topology_inventory,
    seal_exact_comparison_domain,
    source_tree_sha256,
    validate_exact_topology_arrays,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    ChartSubmapPlan,
    SCHEMA as PLAN_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _save_plan(path, names, authority_hash, tree_hash, *, operational=True):
    count = len(names)
    zeros = np.zeros((count, count), np.float64)
    edges = np.ones((count, count), bool)
    np.fill_diagonal(edges, False)
    selected_names = names if operational else []
    metadata = {
        "artifact_type": PLAN_SCHEMA,
        "uses_query_or_ground_truth": False,
        "chart_count": count,
        "selected_chart_count": len(selected_names),
        "source_ordered_names_sha256": canonical_json_sha256(names),
        "selected_chart_names_in_order": selected_names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(
            selected_names
        ),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "components": [
            {
                "operational_coverage_pass": operational,
                "selected_chart_names": selected_names,
            }
        ],
        "operational_submap_count": int(operational),
        "comparison_inventory_eligible": True,
        "system_control_only": False,
        "lineage": {
            "disjoint_authority_schema": (
                "goal_maplet_disjoint_chart_upstream_authority_v2"
            ),
            "disjoint_authority_content_sha256": authority_hash,
            "source_tree_sha256": tree_hash,
        },
    }
    selected_mask = np.full((count,), operational, bool)
    selection_rank = (
        np.arange(count, dtype=np.int32)
        if operational
        else np.full((count,), -1, np.int32)
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
        selected_mask=selected_mask,
        selection_rank=selection_rank,
        metadata=metadata,
    )
    return plan.save_npz(path)


def _fixture(tmp_path, *, operational=True):
    names = ["seq4__b.png", "seq4__a.png"]
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "cameras.json").write_text(json.dumps({"names": names}))
    pointmaps = source_root / "pointmaps"
    pointmaps.mkdir()
    for name in names:
        (pointmaps / f"{name[:-4]}.json").write_text(json.dumps({"name": name}))
    tree_hash = source_tree_sha256(source_root)
    authority = {
        "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
        "source_held_image_disjoint": True,
        "source_held_route_disjoint": True,
        "strict_disjoint_upstream": True,
        "physical_source_held_input_roots_disjoint": True,
        "forbidden_routes_opened": False,
        "uses_query_or_ground_truth": False,
        "source": {
            "root": str(source_root),
            "tree_sha256": tree_hash,
            "ordered_names": names,
        },
    }
    authority["content_sha256"] = canonical_json_sha256(authority)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority, sort_keys=True))
    plan_path = tmp_path / "plan.npz"
    plan_metadata = _save_plan(
        plan_path,
        names,
        authority["content_sha256"],
        tree_hash,
        operational=operational,
    )
    valid = np.ones((2, 40, 40), bool)
    base_arrays = {
        "chart_names": np.asarray(names),
        "valid": valid,
        "face_valid_stride4": np.ones((2, 9, 9), bool),
        "face_valid_stride8": np.ones((2, 4, 4), bool),
    }
    base_metadata = {
        "artifact_type": "goal_maplet_chart_comparison_domain_v1",
        "chart_count": 2,
        "arrays_sha256": arrays_sha256(base_arrays),
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority[
            "content_sha256"
        ],
        "source_tree_sha256": tree_hash,
        "frozen_submap_plan_file_sha256": file_sha256(plan_path),
        "frozen_submap_plan_content_sha256": plan_metadata["content_sha256"],
        "selected_chart_names_in_order_sha256": canonical_json_sha256(
            names if operational else []
        ),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "uses_query_or_ground_truth": False,
    }
    base_metadata["content_sha256"] = canonical_json_sha256(base_metadata)
    domain_path = tmp_path / "domain.npz"
    np.savez_compressed(
        domain_path,
        **base_arrays,
        metadata_json=np.asarray(json.dumps(base_metadata, sort_keys=True)),
    )
    return {
        "names": names,
        "source_root": source_root,
        "tree_hash": tree_hash,
        "authority": authority,
        "authority_path": authority_path,
        "plan_path": plan_path,
        "plan_hash": plan_metadata["content_sha256"],
        "domain_path": domain_path,
        "domain_hash": base_metadata["content_sha256"],
        "base_arrays": base_arrays,
    }


def _seal(value):
    return seal_exact_comparison_domain(
        value["domain_path"],
        expected_upstream_content_sha256=value["domain_hash"],
        plan_path=value["plan_path"],
        expected_plan_content_sha256=value["plan_hash"],
        authority_path=value["authority_path"],
        expected_authority_content_sha256=value["authority"]["content_sha256"],
        source_root=value["source_root"],
        expected_source_tree_sha256=value["tree_hash"],
    )


def test_exact_topology_seal_preserves_plan_order_and_faces(tmp_path):
    value = _fixture(tmp_path)
    arrays, metadata = _seal(value)
    assert arrays["chart_names"].tolist() == value["names"]
    assert arrays["sampled_vertex_offsets_stride4"].tolist() == [0, 100, 200]
    assert arrays["face_offsets_stride4"].tolist() == [0, 162, 324]
    assert arrays["face_offsets_stride8"].tolist() == [0, 32, 64]
    assert metadata["exact_pixel_mask_frozen_for_both_arms"] is True
    assert metadata["exact_face_indices_frozen_for_both_arms"] is True
    assert metadata["full_submap_gate_eligible"] is True
    assert metadata["frozen_submap_plan_content_sha256"] == value["plan_hash"]
    assert metadata["mapping_source_ordered_names_sha256"] == canonical_json_sha256(
        value["names"]
    )


def test_exact_topology_rejects_plan_source_inventory_reorder(tmp_path):
    value = _fixture(tmp_path)
    plan = ChartSubmapPlan.load_npz(value["plan_path"])
    reordered_names = plan.chart_names[::-1]
    metadata = json.loads(json.dumps(plan.metadata))
    metadata.pop("content_sha256", None)
    metadata.pop("arrays_sha256", None)
    metadata["source_ordered_names_sha256"] = canonical_json_sha256(
        reordered_names.astype(str).tolist()
    )
    metadata["selected_chart_names_in_order"] = reordered_names.astype(str).tolist()
    metadata["selected_chart_names_in_order_sha256"] = canonical_json_sha256(
        metadata["selected_chart_names_in_order"]
    )
    metadata["components"][0]["selected_chart_names"] = (
        reordered_names.astype(str).tolist()
    )
    plan = replace(plan, chart_names=reordered_names, metadata=metadata)
    value["plan_hash"] = plan.save_npz(value["plan_path"])["content_sha256"]
    with pytest.raises(ValueError, match="source inventory differs"):
        _seal(value)


def test_exact_topology_full_gate_sealer_rejects_control_only_plan(tmp_path):
    value = _fixture(tmp_path)
    plan = ChartSubmapPlan.load_npz(value["plan_path"])
    metadata = json.loads(json.dumps(plan.metadata))
    metadata.pop("content_sha256", None)
    metadata.pop("arrays_sha256", None)
    metadata["comparison_inventory_eligible"] = False
    metadata["system_control_only"] = True
    control_plan = replace(plan, metadata=metadata)
    value["plan_hash"] = control_plan.save_npz(value["plan_path"])[
        "content_sha256"
    ]
    with pytest.raises(ValueError, match="not eligible"):
        _seal(value)


def test_exact_topology_tamper_fails_even_when_tampered_hash_is_recomputed(tmp_path):
    value = _fixture(tmp_path)
    topology = build_exact_topology_arrays(
        value["base_arrays"]["valid"],
        {
            4: value["base_arrays"]["face_valid_stride4"],
            8: value["base_arrays"]["face_valid_stride8"],
        },
    )
    topology["faces_stride4"] = topology["faces_stride4"].copy()
    topology["faces_stride4"][0, 0] += 1
    with pytest.raises(ValueError, match="does not replay pixel/face masks"):
        validate_exact_topology_arrays(
            value["base_arrays"],
            topology,
            expected_sha256=arrays_sha256(topology),
        )


def test_exact_topology_drops_valid_pixels_not_referenced_by_a_frozen_face():
    valid = np.ones((1, 12, 12), bool)
    stride4_faces = np.zeros((1, 2, 2), bool)
    stride4_faces[0, 0, 0] = True
    stride8_faces = np.zeros((1, 1, 1), bool)
    stride8_faces[0, 0, 0] = True
    topology = build_exact_topology_arrays(
        valid,
        {4: stride4_faces, 8: stride8_faces},
    )
    assert topology["sampled_vertex_offsets_stride4"].tolist() == [0, 4]
    assert topology["sampled_vertex_pixel_indices_stride4"].tolist() == [0, 4, 48, 52]
    assert topology["face_offsets_stride4"].tolist() == [0, 2]
    assert topology["faces_stride4"].tolist() == [[0, 2, 1], [1, 2, 3]]


def test_stride8_may_be_empty_when_only_stride4_is_gate_eligible():
    valid = np.ones((2, 12, 12), bool)
    stride4_faces = np.ones((2, 2, 2), bool)
    stride8_faces = np.ones((2, 1, 1), bool)
    stride8_faces[1] = False
    topology = build_exact_topology_arrays(
        valid,
        {4: stride4_faces, 8: stride8_faces},
    )
    inventory = exact_topology_inventory(topology, np.asarray(["a", "b"]))
    assert inventory["full_submap_gate_eligible_strides"] == [4]
    assert inventory["face_quad_count_per_chart_by_stride"]["stride8"] == [1, 0]
    assert inventory["empty_face_chart_names_by_stride"]["stride8"] == ["b"]


def test_empty_stride4_gate_inventory_still_fails_closed():
    valid = np.ones((2, 12, 12), bool)
    stride4_faces = np.ones((2, 2, 2), bool)
    stride4_faces[1] = False
    topology = build_exact_topology_arrays(
        valid,
        {4: stride4_faces, 8: np.ones((2, 1, 1), bool)},
    )
    with pytest.raises(ValueError, match="stride-4 exact face inventory is empty"):
        exact_topology_inventory(topology, np.asarray(["a", "b"]))


def test_exact_topology_rejects_hash_substitution_and_source_tree_mutation(tmp_path):
    value = _fixture(tmp_path)
    with pytest.raises(ValueError, match="upstream comparison domain differs"):
        seal_exact_comparison_domain(
            value["domain_path"],
            expected_upstream_content_sha256="f" * 64,
            plan_path=value["plan_path"],
            expected_plan_content_sha256=value["plan_hash"],
            authority_path=value["authority_path"],
            expected_authority_content_sha256=value["authority"]["content_sha256"],
            source_root=value["source_root"],
            expected_source_tree_sha256=value["tree_hash"],
        )
    (value["source_root"] / "cameras.json").write_text("mutated")
    with pytest.raises(ValueError, match="source tree bytes differ"):
        _seal(value)


def test_exact_topology_rejects_zero_operational_plan(tmp_path):
    value = _fixture(tmp_path, operational=False)
    with pytest.raises(ValueError, match="no operational submap"):
        _seal(value)


def test_exact_topology_rejects_reordered_domain_even_if_rehashed(tmp_path):
    value = _fixture(tmp_path)
    with np.load(value["domain_path"], allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "chart_names",
                "valid",
                "face_valid_stride4",
                "face_valid_stride8",
            )
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    arrays["chart_names"] = arrays["chart_names"][::-1]
    metadata["arrays_sha256"] = arrays_sha256(arrays)
    metadata.pop("content_sha256")
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(
        value["domain_path"],
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    value["domain_hash"] = metadata["content_sha256"]
    with pytest.raises(ValueError, match="chart order differs"):
        _seal(value)
