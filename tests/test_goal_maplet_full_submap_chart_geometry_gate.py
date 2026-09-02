from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    ChartSubmapPlan,
    SCHEMA as PLAN_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    SCHEMA as V2_COMPARISON_SCHEMA,
    build_exact_topology_arrays,
)
from feature_extract.vfm.localization_goal_maplet.chart_comparison_reference_safe_domain import (
    OPTIMIZER_V1_SCHEMA,
    SCHEMA as COMPARISON_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import (
    ExplicitChartAtlas,
    SCHEMA as ATLAS_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    BOUNDARY_SEMANTICS,
    FullSubmapGeometryGateConfig,
    SourceOnlyBoundedSurfaceBaseline,
    StrictHeldRayInventory,
    _boundary_f1,
    _chart_distortion,
    _seam_audit,
    blocked_input_report,
    bounded_submap_content_sha256,
    evaluate_full_submap_geometry_gate_from_paths,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
    SCHEMA as PLANAR_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


COMMON_HASH = "c" * 64
BOUNDS_MIN = [-20.0, -20.0, 1.0]
BOUNDS_MAX = [20.0, 20.0, 10.0]
BOUNDED_HASH = bounded_submap_content_sha256(BOUNDS_MIN, BOUNDS_MAX)
POINTMAP_HASH = "e" * 64
COORDINATE_CAMERAS_HASH = "1" * 64
COORDINATE_IMAGES_HASH = "2" * 64
HELD_NAMES = ["seq6__held_a.png", "seq6__held_b.png"]
CHART_NAMES = ["seq4__map_a.png", "seq4__map_b.png"]
SOURCE_NAMES_HASH = canonical_json_sha256(CHART_NAMES)
FULL_SOURCE_CAMERAS_HASH = "8" * 64
ALIGNMENT_RUNNER_HASH = "9" * 64
ALIGNMENT_CODE_HASH = "b" * 64


def _write_authority(path: Path) -> dict:
    authority = {
        "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
        "source": {
            "ordered_names": CHART_NAMES,
            "routes": ["seq4"],
            "pointmap_inventory_sha256": "a" * 64,
        },
        "held": {
            "ordered_names": HELD_NAMES,
            "routes": ["seq6"],
            "pointmap_inventory_sha256": POINTMAP_HASH,
        },
        "source_held_image_disjoint": True,
        "source_held_route_disjoint": True,
        "physical_source_held_input_roots_disjoint": True,
        "strict_disjoint_upstream": True,
        "forbidden_routes_opened": False,
        "uses_query_or_ground_truth": False,
        "posed_colmap_cameras_file_sha256": COORDINATE_CAMERAS_HASH,
        "posed_colmap_images_file_sha256": COORDINATE_IMAGES_HASH,
    }
    authority["content_sha256"] = canonical_json_sha256(authority)
    path.write_text(json.dumps(authority, sort_keys=True))
    return authority


def _save_m0(
    path: Path,
    authority_hash: str,
    comparison_content_hash: str,
    comparison_file_hash: str,
    plan_content_hash: str,
) -> None:
    frame = np.eye(3, dtype=np.float64)
    planar = GeometryNativePlanarMap(
        plane_ids=np.asarray([0], np.int64),
        normals_world=np.asarray([[0.0, 0.0, 1.0]]),
        offsets_world=np.asarray([5.0]),
        centers_world=np.asarray([[0.0, 0.0, 5.0]]),
        frames_world=frame[None],
        boundary_offsets=np.asarray([0, 4], np.int64),
        boundary_uv=np.asarray([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]),
        boundary_area_m2=np.asarray([100.0]),
        member_offsets=np.asarray([0, 1], np.int64),
        member_primitive_rows=np.asarray([0], np.int64),
        member_counts=np.asarray([1], np.int64),
        support_area_m2=np.asarray([100.0]),
        residual_rms_m=np.asarray([0.0]),
        residual_p95_m=np.asarray([0.0]),
        normal_cosine_p10=np.asarray([1.0]),
        metadata={
            "artifact_type": PLANAR_SCHEMA,
            "uses_parent_child_partition": False,
            "uses_query_pose_or_ground_truth": False,
            "disjoint_upstream_authority_content_sha256": authority_hash,
            "mapping_source_ordered_names_sha256": SOURCE_NAMES_HASH,
            "held_mapping_images_consumed": False,
            "outside_frozen_source_mapping_images_consumed": False,
            "comparison_budget": "exact_frozen_source_ordered_pool",
            "comparison_role": "equal_budget_main_table_source_only_2dgs_planar",
            "stride": 4,
            "source_2dgs_training_inventory_exact_authority_source": True,
            "comparison_domain_content_sha256": comparison_content_hash,
            "comparison_domain_file_sha256": comparison_file_hash,
            "frozen_submap_plan_content_sha256": plan_content_hash,
            "bounded_submap_content_sha256": BOUNDED_HASH,
            "bounded_submap_min_world": BOUNDS_MIN,
            "bounded_submap_max_world": BOUNDS_MAX,
            "coordinate_cameras_file_sha256": COORDINATE_CAMERAS_HASH,
            "coordinate_images_file_sha256": COORDINATE_IMAGES_HASH,
        },
    ).validated()
    planar.save_npz(path)


def _save_source_surface_m0(
    path: Path,
    authority_hash: str,
    comparison_content_hash: str,
    comparison_file_hash: str,
    plan_content_hash: str,
) -> None:
    vertices = np.asarray(
        [
            [-5.0, -5.0, 5.0], [5.0, -5.0, 5.0],
            [-5.0, 5.0, 5.0], [5.0, 5.0, 5.0],
        ]
        * 2,
        np.float64,
    )
    surface = SourceOnlyBoundedSurfaceBaseline(
        chart_names=np.asarray(CHART_NAMES),
        chart_vertex_offsets=np.asarray([0, 4, 8], np.int64),
        vertices_world=vertices,
        normals_world=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (8, 1)),
        chart_face_offsets=np.asarray([0, 2, 4], np.int64),
        faces=np.asarray([[0, 2, 1], [1, 2, 3], [4, 6, 5], [5, 6, 7]], np.int64),
        metadata={
            "artifact_type": "goal_maplet_source_only_bounded_surface_baseline_v1",
            "uses_query_pose_or_ground_truth": False,
            "comparison_role": "equal_budget_main_table_source_reference_surface_control",
            "comparison_budget": "exact_frozen_source_ordered_pool",
            "held_mapping_images_consumed": False,
            "outside_frozen_source_mapping_images_consumed": False,
            "paired_common_face_inventory": True,
            "stride": 4,
            "disjoint_upstream_authority_content_sha256": authority_hash,
            "mapping_source_ordered_names_sha256": SOURCE_NAMES_HASH,
            "comparison_domain_content_sha256": comparison_content_hash,
            "comparison_domain_file_sha256": comparison_file_hash,
            "frozen_submap_plan_content_sha256": plan_content_hash,
            "bounded_submap_content_sha256": BOUNDED_HASH,
            "bounded_submap_min_world": BOUNDS_MIN,
            "bounded_submap_max_world": BOUNDS_MAX,
            "coordinate_cameras_file_sha256": COORDINATE_CAMERAS_HASH,
            "coordinate_images_file_sha256": COORDINATE_IMAGES_HASH,
            "source_pointmap_inventory_sha256": "a" * 64,
        },
    ).validated()
    surface.save_npz(path)


def _atlas(
    authority_hash: str,
    arm: str,
    *,
    initial: bool,
    comparison_content_hash: str = COMMON_HASH,
    comparison_file_hash: str = "3" * 64,
    plan_content_hash: str = "4" * 64,
    chart_names=None,
    translation=(0.0, 0.0, 0.0),
    x_scale=1.0,
    third_far_chart=False,
    provenance=None,
) -> ExplicitChartAtlas:
    names = list(chart_names if chart_names is not None else CHART_NAMES)
    names += (["seq4__map_c.png"] if third_far_chart else [])
    vertices = []
    normals = []
    uv = []
    faces = []
    vertex_offsets = [0]
    face_offsets = [0]
    for chart, _ in enumerate(names):
        points = np.asarray(
            [[-5.0, -5.0, 5.0], [5.0, -5.0, 5.0], [-5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            np.float64,
        )
        points[:, 0] *= x_scale
        points += np.asarray(translation, np.float64)
        if third_far_chart and chart == 2:
            points += np.asarray([100.0, 0.0, 0.0])
        local_faces = np.asarray([[0, 2, 1], [1, 2, 3]], np.int64) + vertex_offsets[-1]
        vertices.append(points)
        normals.append(np.tile(np.asarray([[0.0, 0.0, 1.0]]), (4, 1)))
        uv.append(np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        faces.append(local_faces)
        vertex_offsets.append(vertex_offsets[-1] + 4)
        face_offsets.append(face_offsets[-1] + 2)
    return ExplicitChartAtlas(
        chart_names=np.asarray(names),
        chart_vertex_offsets=np.asarray(vertex_offsets, np.int64),
        vertices_world=np.concatenate(vertices),
        normals_world=np.concatenate(normals),
        uv=np.concatenate(uv),
        confidence=np.ones(len(names) * 4, np.float32),
        chart_face_offsets=np.asarray(face_offsets, np.int64),
        faces=np.concatenate(faces),
        metadata={
            "artifact_type": ATLAS_SCHEMA,
            "uses_query_or_ground_truth": False,
            "disjoint_upstream_authority_content_sha256": authority_hash,
            "mapping_source_ordered_names_sha256": SOURCE_NAMES_HASH,
            "held_mapping_images_consumed": False,
            "outside_frozen_source_mapping_images_consumed": False,
            "comparison_domain_content_sha256": comparison_content_hash,
            "comparison_domain_file_sha256": comparison_file_hash,
            "frozen_submap_plan_content_sha256": plan_content_hash,
            "bounded_submap_content_sha256": BOUNDED_HASH,
            "bounded_submap_min_world": BOUNDS_MIN,
            "bounded_submap_max_world": BOUNDS_MAX,
            "coordinate_cameras_file_sha256": COORDINATE_CAMERAS_HASH,
            "coordinate_images_file_sha256": COORDINATE_IMAGES_HASH,
            "paired_common_face_inventory": True,
            "is_pre_alignment_geometry": initial,
            "initializer_arm": arm,
            "stride": 4,
            **(provenance or {}),
        },
    ).validated()


def _plan(
    authority_hash: str,
    *,
    third=False,
    only_first_edge=False,
    reverse_selection=False,
) -> ChartSubmapPlan:
    names = CHART_NAMES + (["seq4__map_c.png"] if third else [])
    count = len(names)
    selected_order = names[::-1] if reverse_selection else names
    selection_rank = np.arange(count, dtype=np.int32)
    if reverse_selection:
        selection_rank = selection_rank[::-1]
    zero = np.zeros((count, count), np.float64)
    overlap = zero.copy()
    coverage = np.zeros((count, count), bool)
    coverage[0, 1] = coverage[1, 0] = True
    if third and not only_first_edge:
        coverage[1, 2] = coverage[2, 1] = True
    overlap[coverage] = 0.5
    baseline = zero.copy()
    baseline[coverage] = 1.0
    angle = zero.copy()
    angle[coverage] = 10.0
    return ChartSubmapPlan(
        chart_names=np.asarray(names),
        camera_centers_world=np.stack((np.arange(count), np.zeros(count), np.zeros(count)), axis=1),
        camera_forward_world=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1)),
        valid_sample_counts=np.full(count, 100, np.int64),
        directional_frustum_fraction=overlap.copy(),
        directional_depth_support=overlap.copy(),
        directional_surface_support=overlap.copy(),
        symmetric_surface_overlap=overlap.copy(),
        camera_baseline_m=baseline,
        median_overlap_depth_m=np.where(coverage, 5.0, 0.0),
        baseline_to_depth_ratio=np.where(coverage, 0.2, 0.0),
        median_triangulation_angle_deg=angle,
        camera_forward_angle_deg=angle,
        same_surface_side_fraction=np.where(coverage, 1.0, 0.0),
        coverage_edges=coverage,
        alignment_edges=coverage.copy(),
        component_ids=np.zeros(count, np.int32),
        selected_mask=np.ones(count, bool),
        selection_rank=selection_rank,
        metadata={
            "artifact_type": PLAN_SCHEMA,
            "uses_query_or_ground_truth": False,
            "chart_count": count,
            "selected_chart_count": count,
            "source_ordered_names_sha256": canonical_json_sha256(names),
            "selected_chart_names_in_order": selected_order,
            "selected_chart_names_in_order_sha256": canonical_json_sha256(selected_order),
            "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
            "components": [
                {
                    "component_id": 0,
                    "operational_coverage_pass": True,
                    "selected_chart_names": selected_order,
                }
            ],
            "operational_submap_count": 1,
            "comparison_inventory_eligible": True,
            "system_control_only": False,
            "lineage": {
                "comparison_inventory_eligible": True,
                "system_control_only": False,
                "query_or_ground_truth_consumed": False,
                "held_root_opened_by_selector": False,
                "disjoint_authority_content_sha256": authority_hash,
                "source_ordered_names_sha256": SOURCE_NAMES_HASH,
            },
        },
    ).validated()


def _save_optimizer_v1_domain(
    path: Path,
    authority_path: Path,
    authority: dict,
    plan_content_hash: str,
    chart_names,
):
    arrays = {
        "chart_names": np.asarray(chart_names),
        "valid": np.ones((2, 5, 5), bool),
        "face_valid_stride4": np.ones((2, 1, 1), bool),
        "face_valid_stride8": np.zeros((2, 0, 0), bool),
    }
    metadata = {
        "artifact_type": OPTIMIZER_V1_SCHEMA,
        "chart_count": 2,
        "uses_query_or_ground_truth": False,
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority["content_sha256"],
        "frozen_submap_plan_content_sha256": plan_content_hash,
        "mapping_source_ordered_names_sha256": SOURCE_NAMES_HASH,
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "cameras_file_sha256": FULL_SOURCE_CAMERAS_HASH,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return arrays, metadata


def _save_comparison_domain(
    path: Path,
    authority_path: Path,
    authority: dict,
    plan_content_hash: str,
    optimizer_v1_path: Path,
    optimizer_v1_arrays: dict,
    optimizer_v1_metadata: dict,
    chart_names=CHART_NAMES,
    extra_metadata=None,
) -> dict:
    base_arrays = {name: np.asarray(value) for name, value in optimizer_v1_arrays.items()}
    topology = build_exact_topology_arrays(
        base_arrays["valid"],
        {
            4: base_arrays["face_valid_stride4"],
            8: base_arrays["face_valid_stride8"],
        },
    )
    arrays = {**base_arrays, **topology}
    v2_metadata = {
        **optimizer_v1_metadata,
        "artifact_type": V2_COMPARISON_SCHEMA,
        "chart_count": 2,
        "full_submap_gate_eligible": True,
        "full_submap_gate_eligible_strides": [4],
        "required_nonempty_face_inventory_strides": [4],
        "full_submap_gate_primary_stride": 4,
        "noneligible_stride_empty_inventory_permitted": True,
        "exact_pixel_mask_frozen_for_both_arms": True,
        "exact_face_indices_frozen_for_both_arms": True,
        "orphan_sampled_vertices_present": False,
        "exact_topology_arrays_sha256": arrays_sha256(topology),
        "uses_query_or_ground_truth": False,
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority["content_sha256"],
        "frozen_submap_plan_content_sha256": plan_content_hash,
        "mapping_source_ordered_names_sha256": SOURCE_NAMES_HASH,
        "upstream_comparison_domain_content_sha256": optimizer_v1_metadata[
            "content_sha256"
        ],
        "upstream_comparison_domain_file_sha256": file_sha256(optimizer_v1_path),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "cameras_file_sha256": FULL_SOURCE_CAMERAS_HASH,
        **(extra_metadata or {}),
        "arrays_sha256": arrays_sha256(arrays),
    }
    v2_metadata.pop("content_sha256", None)
    v2_metadata["content_sha256"] = canonical_json_sha256(v2_metadata)
    v2_path = path.with_name("comparison_exact_v2.npz")
    np.savez_compressed(
        v2_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(v2_metadata, sort_keys=True)),
    )
    face_masks = {
        name: base_arrays[name]
        for name in ("face_valid_stride4", "face_valid_stride8")
    }
    metadata = {
        **v2_metadata,
        "artifact_type": COMPARISON_SCHEMA,
        "upstream_exact_topology_v2_file_sha256": file_sha256(v2_path),
        "upstream_exact_topology_v2_content_sha256": v2_metadata["content_sha256"],
        "upstream_exact_topology_v2_arrays_sha256": v2_metadata["arrays_sha256"],
        "upstream_exact_topology_v2_exact_topology_arrays_sha256": v2_metadata[
            "exact_topology_arrays_sha256"
        ],
        "upstream_optimizer_comparison_domain_v1_file_sha256": file_sha256(
            optimizer_v1_path
        ),
        "upstream_optimizer_comparison_domain_v1_content_sha256": (
            optimizer_v1_metadata["content_sha256"]
        ),
        "upstream_optimizer_comparison_domain_v1_arrays_sha256": (
            optimizer_v1_metadata["arrays_sha256"]
        ),
        "source_reference_edge_safe": True,
        "source_reference_edge_safety_replayed": True,
        "physical_face_safety_authority": True,
        "face_valid_v3_subset_of_face_valid_v2": True,
        "exact_topology_repacked_after_reference_safety": True,
        "valid_and_chart_names_byte_equal_upstream_v2": True,
        "source_reference_edge_safe_face_masks_sha256": arrays_sha256(face_masks),
        "final_reference_safe_face_masks_sha256": arrays_sha256(face_masks),
        "parent_v2_face_valid_arrays_sha256": arrays_sha256(face_masks),
    }
    metadata.pop("content_sha256", None)
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def _save_rays(
    path: Path,
    authority_path: Path,
    authority: dict,
    comparison_path: Path,
    comparison_metadata: dict,
    plan_content_hash: str,
) -> None:
    views, height, width = 2, 16, 16
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], views, axis=0)
    depth = np.full((views, height, width), 5.0, np.float64)
    normal = np.zeros((views, height, width, 3), np.float64)
    normal[..., 2] = 1.0
    frozen_builder_config = {
        "topology_stride": 4,
        "bound_margin_m": 1.0,
        "held_confidence_threshold": 0.25,
        "temporal_block_size": 1,
    }
    rays = StrictHeldRayInventory(
        view_names=np.asarray(HELD_NAMES),
        camera_to_world=c2w,
        focal_xy=np.tile(np.asarray([[8.0, 8.0]]), (views, 1)),
        principal_xy=np.tile(np.asarray([[7.5, 7.5]]), (views, 1)),
        reference_depth_m=depth,
        reference_normal_world=normal,
        reference_valid=np.ones((views, height, width), bool),
        reference_boundary=np.zeros((views, height, width), bool),
        block_ids=np.asarray(["block_a", "block_b"]),
        metadata={
            "artifact_type": "goal_maplet_strict_held_ray_inventory_v1",
            "uses_query_or_ground_truth": False,
            "dense_row_major_pixel_centers": True,
            "reference_semantics": "source_disjoint_held_mapping_geometry_not_sensor_depth_gt",
            "boundary_semantics": BOUNDARY_SEMANTICS,
            "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
            "disjoint_upstream_authority_content_sha256": authority["content_sha256"],
            "held_pointmap_inventory_sha256": POINTMAP_HASH,
            "mapping_source_ordered_names_sha256": SOURCE_NAMES_HASH,
            "comparison_domain_content_sha256": comparison_metadata["content_sha256"],
            "comparison_domain_file_sha256": file_sha256(comparison_path),
            "frozen_submap_plan_content_sha256": plan_content_hash,
            "bounded_submap_content_sha256": BOUNDED_HASH,
            "bounded_submap_min_world": BOUNDS_MIN,
            "bounded_submap_max_world": BOUNDS_MAX,
            "coordinate_cameras_file_sha256": COORDINATE_CAMERAS_HASH,
            "coordinate_images_file_sha256": COORDINATE_IMAGES_HASH,
            "held_view_names_sha256": canonical_json_sha256(HELD_NAMES),
            "held_geometry_opened_after_source_bounds_frozen": True,
            "bound_uses_held_geometry": False,
            "held_inventory_exactly_authority_order": True,
            "builder_config_frozen_before_held": True,
            "pre_frozen_builder_config": frozen_builder_config,
            "pre_frozen_builder_config_content_sha256": canonical_json_sha256(
                frozen_builder_config
            ),
        },
    ).validated()
    rays.save_npz(path)


def _sealed_json(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["content_sha256"] = canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def _write_atlas_provenance_fixture(
    tmp_path: Path,
    selected_names: list[str],
    plan_content_hash: str,
    upstream_v1_content_hash: str,
    upstream_v1_file_hash: str,
):
    provenance = {}
    domain_metadata = {}
    runner_files = {}
    for arm, prefix in (("DAV2", "dav2"), ("MoGe3", "moge")):
        initializer_root = tmp_path / f"{prefix}_initializers"
        initializer_root.mkdir()
        rows = []
        file_hashes = {}
        for index, name in enumerate(CHART_NAMES):
            initializer_file = initializer_root / f"{name}.npz"
            initializer_file.write_text(f"{arm}:{name}")
            row = {
                "name": name,
                "file_sha256": file_sha256(initializer_file),
                "content_sha256": canonical_json_sha256(
                    {"arm": arm, "name": name, "row": index}
                ),
            }
            rows.append(row)
            file_hashes[name] = row["file_sha256"]
        initializer_manifest_path = initializer_root / "manifest.json"
        initializer_manifest = _sealed_json(
            initializer_manifest_path,
            {
                "artifact_type": f"synthetic_{prefix}_initializer_manifest",
                "rows": rows,
                "uses_query_or_ground_truth": False,
            },
        )
        alignment_root = tmp_path / f"{prefix}_alignment"
        alignment_root.mkdir()
        charts_path = alignment_root / "charts_data.npz"
        np.savez_compressed(charts_path, scale_factor=np.asarray(1.0))
        cameras_path = alignment_root / "cameras.json"
        cameras_path.write_text(json.dumps({"synthetic": True, "arm": arm}))
        alignment_manifest_path = alignment_root / "manifest.json"
        alignment_manifest = _sealed_json(
            alignment_manifest_path,
            {
                "artifact_type": "goal_maplet_masked_chart_alignment_gate_v1",
                "chart_names": selected_names,
                "comparison_domain_content_sha256": upstream_v1_content_hash,
                "comparison_domain_file_sha256": upstream_v1_file_hash,
                "frozen_submap_plan_content_sha256": plan_content_hash,
                "selection_contract": ALIGNMENT_SELECTION_CONTRACT,
                "alignment_runner_file_sha256": ALIGNMENT_RUNNER_HASH,
                "alignment_code_inventory_sha256": ALIGNMENT_CODE_HASH,
                "subset_cameras_file_sha256": file_sha256(cameras_path),
                "charts_data_file_sha256": file_sha256(charts_path),
                "cameras_source_file_sha256": FULL_SOURCE_CAMERAS_HASH,
                # Real order-fixed runner serializes this as the exact
                # official-plan-order list, not a lexical name dictionary.
                "initializer_file_sha256": [
                    file_hashes[name] for name in selected_names
                ],
                "uses_query_or_ground_truth": False,
            },
        )
        selected_inventory = [
            {
                "name": name,
                "file_sha256": next(row for row in rows if row["name"] == name)[
                    "file_sha256"
                ],
                "content_sha256": next(row for row in rows if row["name"] == name)[
                    "content_sha256"
                ],
            }
            for name in selected_names
        ]
        provenance[arm] = {
            "alignment_manifest_path": str(alignment_manifest_path.resolve()),
            "alignment_manifest_file_sha256": file_sha256(alignment_manifest_path),
            "alignment_manifest_content_sha256": alignment_manifest["content_sha256"],
            "initializer_manifest_path": str(initializer_manifest_path.resolve()),
            "initializer_manifest_file_sha256": file_sha256(initializer_manifest_path),
            "initializer_manifest_content_sha256": initializer_manifest[
                "content_sha256"
            ],
            "initializer_selected_file_inventory": selected_inventory,
            "initializer_selected_file_inventory_sha256": canonical_json_sha256(
                selected_inventory
            ),
            "alignment_upstream_comparison_domain_content_sha256": upstream_v1_content_hash,
            "alignment_upstream_comparison_domain_file_sha256": upstream_v1_file_hash,
            "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
            "alignment_runner_file_sha256": ALIGNMENT_RUNNER_HASH,
            "alignment_code_inventory_sha256": ALIGNMENT_CODE_HASH,
            "source_cameras_file_sha256": file_sha256(cameras_path),
            "source_charts_file_sha256": file_sha256(charts_path),
            "scale_factor": 1.0,
        }
        domain_metadata.update(
            {
                f"{prefix}_initializer_manifest_file_sha256": file_sha256(
                    initializer_manifest_path
                ),
                f"{prefix}_initializer_manifest_content_sha256": initializer_manifest[
                    "content_sha256"
                ],
                f"{prefix}_initializer_file_sha256": file_hashes,
            }
        )
        runner_files[arm] = alignment_manifest_path
    return provenance, domain_metadata, runner_files


def _fixture(
    tmp_path: Path,
    *,
    m2_translation=(0.0, 0.0, 0.0),
    m2_x_scale=1.0,
    reverse_selection=False,
):
    authority_path = tmp_path / "authority.json"
    authority = _write_authority(authority_path)
    paths = {
        "authority_path": authority_path,
        "comparison_domain_path": tmp_path / "comparison.npz",
        "held_ray_inventory_path": tmp_path / "rays.npz",
        "m0_bounded_map_path": tmp_path / "m0.npz",
        "m1_initial_atlas_path": tmp_path / "m1_initial.npz",
        "m1_atlas_path": tmp_path / "m1.npz",
        "m2_initial_atlas_path": tmp_path / "m2_initial.npz",
        "m2_atlas_path": tmp_path / "m2.npz",
        "frozen_submap_plan_path": tmp_path / "plan.npz",
    }
    selected_names = CHART_NAMES[::-1] if reverse_selection else CHART_NAMES
    plan_metadata = _plan(
        authority["content_sha256"], reverse_selection=reverse_selection,
    ).save_npz(
        paths["frozen_submap_plan_path"]
    )
    optimizer_v1_path = tmp_path / "comparison_optimizer_v1.npz"
    optimizer_v1_arrays, optimizer_v1_metadata = _save_optimizer_v1_domain(
        optimizer_v1_path,
        authority_path,
        authority,
        plan_metadata["content_sha256"],
        selected_names,
    )
    provenance, domain_bindings, _ = _write_atlas_provenance_fixture(
        tmp_path,
        selected_names,
        plan_metadata["content_sha256"],
        optimizer_v1_metadata["content_sha256"],
        file_sha256(optimizer_v1_path),
    )
    comparison_metadata = _save_comparison_domain(
        paths["comparison_domain_path"],
        authority_path,
        authority,
        plan_metadata["content_sha256"],
        optimizer_v1_path,
        optimizer_v1_arrays,
        optimizer_v1_metadata,
        chart_names=selected_names,
        extra_metadata=domain_bindings,
    )
    comparison_file_hash = file_sha256(paths["comparison_domain_path"])
    atlas_lineage = {
        "comparison_content_hash": comparison_metadata["content_sha256"],
        "comparison_file_hash": comparison_file_hash,
        "plan_content_hash": plan_metadata["content_sha256"],
        "chart_names": selected_names,
    }
    _save_rays(
        paths["held_ray_inventory_path"],
        authority_path,
        authority,
        paths["comparison_domain_path"],
        comparison_metadata,
        plan_metadata["content_sha256"],
    )
    _save_m0(
        paths["m0_bounded_map_path"],
        authority["content_sha256"],
        comparison_metadata["content_sha256"],
        comparison_file_hash,
        plan_metadata["content_sha256"],
    )
    _atlas(authority["content_sha256"], "DAV2", initial=True, provenance=provenance["DAV2"], **atlas_lineage).save_npz(paths["m1_initial_atlas_path"])
    _atlas(authority["content_sha256"], "DAV2", initial=False, provenance=provenance["DAV2"], **atlas_lineage).save_npz(paths["m1_atlas_path"])
    _atlas(authority["content_sha256"], "MoGe3", initial=True, provenance=provenance["MoGe3"], **atlas_lineage).save_npz(paths["m2_initial_atlas_path"])
    _atlas(
        authority["content_sha256"],
        "MoGe3",
        initial=False,
        translation=m2_translation,
        x_scale=m2_x_scale,
        provenance=provenance["MoGe3"],
        **atlas_lineage,
    ).save_npz(paths["m2_atlas_path"])
    return paths


def _run(paths):
    return evaluate_full_submap_geometry_gate_from_paths(
        **paths,
        config=FullSubmapGeometryGateConfig(bootstrap_resamples=100),
    )


def _rewrite_npz_metadata(path: Path, updates: dict) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    metadata.pop("content_sha256", None)
    metadata.update(updates)
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_perfect_common_domain_is_go(tmp_path):
    report = _run(_fixture(tmp_path))
    assert report["input_contract"] == "PASS"
    assert report["decisions"]["relative_noninferiority"]["M1_DAV2_atlas_vs_M0_source_only_2DGS_planar"]["decision"] == "GO"
    assert report["decisions"]["relative_noninferiority"]["M2_MoGe3_atlas_vs_M0_source_only_2DGS_planar"]["decision"] == "GO"
    assert report["decisions"]["relative_noninferiority"]["M2_MoGe3_vs_M1_DAV2"]["decision"] == "GO"
    assert report["decisions"]["relative_strict_dominance"]["decision"] == "KILL"
    assert report["decisions"]["composite_bounded_gate"]["initializer_independent_chart_atlas_vs_M0"] == "GO"
    for arm in ("M0", "M1", "M2"):
        assert report["representations"][arm]["macro_view"]["good_ray_recall"]["mean"] == 1.0


def test_v3_eligible_stride_authority_tamper_fails_closed(tmp_path):
    paths = _fixture(tmp_path)
    _rewrite_npz_metadata(
        paths["comparison_domain_path"],
        {"full_submap_gate_eligible_strides": [4, 8]},
    )
    with pytest.raises(ValueError, match="formal gate stride authority"):
        _run(paths)


@pytest.mark.parametrize("artifact", ("M0", "M2"))
def test_mixed_or_nonprimary_arm_stride_fails_closed(tmp_path, artifact):
    paths = _fixture(tmp_path)
    if artifact == "M0":
        m0 = GeometryNativePlanarMap.load_npz(paths["m0_bounded_map_path"])
        metadata = dict(m0.metadata)
        metadata.pop("arrays_sha256")
        metadata.pop("content_sha256")
        metadata["stride"] = 8
        GeometryNativePlanarMap(metadata=metadata, **m0.arrays()).save_npz(
            paths["m0_bounded_map_path"]
        )
        expected = "M0 does not use the formal primary topology stride"
    else:
        atlas = ExplicitChartAtlas.load_npz(paths["m2_atlas_path"])
        metadata = dict(atlas.metadata)
        metadata.pop("arrays_sha256")
        metadata.pop("content_sha256")
        metadata["stride"] = 8
        ExplicitChartAtlas(metadata=metadata, **atlas.arrays()).save_npz(
            paths["m2_atlas_path"]
        )
        expected = "M2 does not use the formal primary topology stride"
    with pytest.raises(ValueError, match=expected):
        _run(paths)


def test_selection_rank_order_not_source_inventory_order_is_authoritative(tmp_path):
    report = _run(_fixture(tmp_path, reverse_selection=True))
    assert report["input_contract"] == "PASS"
    assert report["decisions"]["composite_bounded_gate"]["initializer_independent_chart_atlas_vs_M0"] == "GO"


def test_source_reference_surface_control_runs_without_being_called_2dgs(tmp_path):
    paths = _fixture(tmp_path)
    authority = json.loads(paths["authority_path"].read_text())
    with np.load(paths["comparison_domain_path"], allow_pickle=False) as data:
        comparison = json.loads(str(data["metadata_json"].item()))
    plan = ChartSubmapPlan.load_npz(paths["frozen_submap_plan_path"])
    _save_source_surface_m0(
        paths["m0_bounded_map_path"],
        authority["content_sha256"],
        comparison["content_sha256"],
        file_sha256(paths["comparison_domain_path"]),
        plan.metadata["content_sha256"],
    )
    report = _run(paths)
    assert report["M0_is_2DGS"] is False
    assert report["M0_comparison_role"] == (
        "equal_budget_main_table_source_reference_surface_control"
    )
    assert report["decisions"]["composite_bounded_gate"][
        "M1_DAV2_atlas_vs_M0_source_reference_surface_control"
    ]["decision"] == "GO"


def test_missing_rendered_rays_fail_primary_and_do_not_get_conditional_credit(tmp_path):
    paths = _fixture(tmp_path, m2_translation=(12.0, 0.0, 0.0))
    report = _run(paths)
    m2 = report["representations"]["M2"]["macro_view"]
    assert m2["good_ray_recall"]["mean"] == 0.0
    assert m2["absrel_median_conditional"]["mean"] is None
    assert report["decisions"]["relative_noninferiority"]["M2_MoGe3_atlas_vs_M0_source_only_2DGS_planar"]["decision"] == "KILL"
    assert "good_ray_recall" in report["decisions"]["relative_noninferiority"]["M2_MoGe3_atlas_vs_M0_source_only_2DGS_planar"]["failures"]


def test_identically_sparse_arms_are_killed_by_absolute_completeness_floor(tmp_path):
    paths = _fixture(tmp_path)
    m0 = GeometryNativePlanarMap.load_npz(paths["m0_bounded_map_path"])
    m0_metadata = dict(m0.metadata)
    m0_metadata.pop("arrays_sha256")
    m0_metadata.pop("content_sha256")
    m0_arrays = m0.arrays()
    m0_arrays["boundary_uv"] = m0_arrays["boundary_uv"] * 0.1
    GeometryNativePlanarMap(metadata=m0_metadata, **m0_arrays).save_npz(
        paths["m0_bounded_map_path"]
    )
    for key in (
        "m1_initial_atlas_path",
        "m1_atlas_path",
        "m2_initial_atlas_path",
        "m2_atlas_path",
    ):
        atlas = ExplicitChartAtlas.load_npz(paths[key])
        metadata = dict(atlas.metadata)
        metadata.pop("arrays_sha256")
        metadata.pop("content_sha256")
        arrays = atlas.arrays()
        vertices = arrays["vertices_world"].copy()
        vertices[:, :2] *= 0.1
        arrays["vertices_world"] = vertices
        ExplicitChartAtlas(metadata=metadata, **arrays).save_npz(paths[key])
    report = _run(paths)
    relative = report["decisions"]["relative_noninferiority"][
        "M1_DAV2_atlas_vs_M0_source_only_2DGS_planar"
    ]
    absolute = report["decisions"]["absolute_held_coverage"]["M1"]
    composite = report["decisions"]["composite_bounded_gate"][
        "M1_DAV2_atlas_vs_M0_source_only_2DGS_planar"
    ]
    assert relative["decision"] == "GO"
    assert absolute["decision"] == "KILL"
    assert "good_ray_recall" in absolute["failures"]
    assert composite["decision"] == "KILL"
    assert composite["component_decisions"]["relative_noninferiority"] == "GO"
    assert composite["component_decisions"]["absolute_held_coverage"] == "KILL"
    assert report["bounded_gate_eligible"] is False


def test_conditional_absrel_uses_common_finite_views_but_is_not_a_gate(tmp_path):
    paths = _fixture(tmp_path)
    rays = StrictHeldRayInventory.load_npz(paths["held_ray_inventory_path"])
    arrays = rays.arrays()
    arrays["focal_xy"] = arrays["focal_xy"].copy()
    arrays["focal_xy"][1] = 2.0
    arrays["reference_valid"] = arrays["reference_valid"].copy()
    arrays["reference_valid"][1] = False
    arrays["reference_valid"][1, 7, 0] = True
    metadata = dict(rays.metadata)
    metadata.pop("arrays_sha256")
    metadata.pop("content_sha256")
    StrictHeldRayInventory(metadata=metadata, **arrays).save_npz(
        paths["held_ray_inventory_path"]
    )
    report = _run(paths)
    conditional = report["paired_block_bootstrap"]["M1_minus_M0"][
        "absrel_median_conditional"
    ]
    assert conditional["common_finite_view_count"] == 1
    assert report["decisions"]["composite_bounded_gate"][
        "M1_DAV2_atlas_vs_M0_source_only_2DGS_planar"
    ]["decision"] == "GO"


def test_boundary_f1_is_diagnostic_not_a_physical_hard_gate(tmp_path):
    paths = _fixture(tmp_path)
    rays = StrictHeldRayInventory.load_npz(paths["held_ray_inventory_path"])
    arrays = rays.arrays()
    arrays["reference_boundary"] = np.ones_like(arrays["reference_boundary"], bool)
    metadata = dict(rays.metadata)
    metadata.pop("arrays_sha256")
    metadata.pop("content_sha256")
    StrictHeldRayInventory(metadata=metadata, **arrays).save_npz(
        paths["held_ray_inventory_path"]
    )
    report = _run(paths)
    assert report["representations"]["M1"]["macro_view"]["boundary_f1"][
        "mean"
    ] < 0.5
    assert report["decisions"]["composite_bounded_gate"][
        "M1_DAV2_atlas_vs_M0_source_only_2DGS_planar"
    ]["decision"] == "GO"


def test_anisotropic_collapse_is_killed_after_global_scale_is_removed(tmp_path):
    report = _run(_fixture(tmp_path, m2_x_scale=0.1))
    assert report["geometry_integrity"]["M2"]["decision"] == "KILL"
    assert "jacobian_singular_ratio_p05" in report["geometry_integrity"]["M2"]["failures"]
    assert report["distortion"]["M2"]["jacobian_singular_ratio_p05"] < 0.5
    assert report["decisions"]["geometry_integrity"]["M2"]["decision"] == "KILL"
    assert report["decisions"]["composite_bounded_gate"]["M2_MoGe3_atlas_vs_M0_source_only_2DGS_planar"]["decision"] == "KILL"


def test_pure_global_scale_is_reported_but_not_called_local_distortion():
    initial = _atlas("a" * 64, "MoGe3", initial=True)
    aligned = ExplicitChartAtlas(
        metadata=dict(initial.metadata),
        **{
            **initial.arrays(),
            "vertices_world": initial.vertices_world * 0.5,
        },
    ).validated()
    distortion = _chart_distortion(initial, aligned)
    assert distortion["estimated_global_linear_scale"] == pytest.approx(0.5)
    assert distortion["chart_area_ratio_min"] == pytest.approx(1.0)
    assert distortion["jacobian_singular_ratio_p05"] == pytest.approx(1.0)
    assert distortion["face_collapse_fraction"] == 0.0


def test_comparison_domain_mismatch_fails_closed_before_metrics(tmp_path):
    paths = _fixture(tmp_path)
    atlas = ExplicitChartAtlas.load_npz(paths["m2_atlas_path"])
    metadata = dict(atlas.metadata)
    metadata.pop("arrays_sha256")
    metadata.pop("content_sha256")
    metadata["comparison_domain_content_sha256"] = "f" * 64
    ExplicitChartAtlas(metadata=metadata, **atlas.arrays()).save_npz(paths["m2_atlas_path"])
    with pytest.raises(ValueError, match="common comparison domain"):
        _run(paths)
    blocked = blocked_input_report(ValueError("common domain differs"), {"M2": paths["m2_atlas_path"]})
    assert blocked["input_contract"] == "FAIL_CLOSED"
    assert blocked["scientific_performance_conclusion"] == "NOT_EVALUATED"
    assert blocked["decisions"]["composite_bounded_gate"] == "KILL_INPUT_CONTRACT"


def test_initial_and_aligned_must_share_actual_alignment_manifest(tmp_path):
    paths = _fixture(tmp_path)
    atlas = ExplicitChartAtlas.load_npz(paths["m1_atlas_path"])
    metadata = dict(atlas.metadata)
    metadata.pop("arrays_sha256")
    metadata.pop("content_sha256")
    metadata["alignment_manifest_content_sha256"] = "f" * 64
    ExplicitChartAtlas(metadata=metadata, **atlas.arrays()).save_npz(
        paths["m1_atlas_path"]
    )
    with pytest.raises(ValueError, match="initial/aligned provenance differs"):
        _run(paths)


def test_m0_that_consumed_held_mapping_images_is_not_a_fair_baseline(tmp_path):
    paths = _fixture(tmp_path)
    m0 = GeometryNativePlanarMap.load_npz(paths["m0_bounded_map_path"])
    metadata = dict(m0.metadata)
    metadata.pop("arrays_sha256")
    metadata.pop("content_sha256")
    metadata["held_mapping_images_consumed"] = True
    GeometryNativePlanarMap(metadata=metadata, **m0.arrays()).save_npz(
        paths["m0_bounded_map_path"]
    )
    with pytest.raises(ValueError, match="M0 consumed held mapping images"):
        _run(paths)


def test_seam_audit_uses_only_frozen_coverage_edges():
    authority_hash = "a" * 64
    atlas = _atlas(authority_hash, "DAV2", initial=False, third_far_chart=True)
    plan = _plan(authority_hash, third=True, only_first_edge=True)
    seam = _seam_audit(
        atlas,
        plan,
        FullSubmapGeometryGateConfig(bootstrap_resamples=100),
    )
    assert seam["frozen_edge_count"] == 1
    assert seam["per_edge"][0]["first"] == CHART_NAMES[0]
    assert seam["per_edge"][0]["second"] == CHART_NAMES[1]
    assert seam["seam_thickness_p90_m"] == 0.0


def test_one_bad_frozen_edge_cannot_hide_in_pooled_seam_quantiles():
    authority_hash = "a" * 64
    atlas = _atlas(authority_hash, "DAV2", initial=False, third_far_chart=True)
    plan = _plan(authority_hash, third=True, only_first_edge=False)
    seam = _seam_audit(
        atlas,
        plan,
        FullSubmapGeometryGateConfig(bootstrap_resamples=100),
    )
    assert seam["seam_thickness_p90_m"] == 0.0
    assert seam["all_frozen_edges_geometry_pass"] is False
    assert seam["per_edge"][1]["support_pass"] is False


def test_cross_chart_normal_reversal_is_not_hidden_by_unsigned_normal():
    authority_hash = "a" * 64
    atlas = _atlas(authority_hash, "DAV2", initial=False)
    arrays = atlas.arrays()
    normals = arrays["normals_world"].copy()
    normals[4:] *= -1
    arrays["normals_world"] = normals
    reversed_atlas = ExplicitChartAtlas(metadata=atlas.metadata, **arrays).validated()
    seam = _seam_audit(
        reversed_atlas,
        _plan(authority_hash),
        FullSubmapGeometryGateConfig(bootstrap_resamples=100),
    )
    assert seam["unsigned_normal_p90_deg"] == 0.0
    assert seam["oriented_normal_p90_deg"] == 180.0
    assert seam["all_frozen_edges_oriented_normal_pass"] is False


def test_boundary_f1_has_symmetric_tolerance_and_penalises_absence():
    reference = np.zeros((16, 16), bool)
    reference[:, 5] = True
    shifted = np.zeros_like(reference)
    shifted[:, 6] = True
    assert _boundary_f1(reference, shifted, 1.0) == 1.0
    assert _boundary_f1(reference, np.zeros_like(reference), 1.0) == 0.0
