from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    SCHEMA as EXACT_DOMAIN_SCHEMA,
    build_exact_topology_arrays,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    ChartSubmapPlan,
    SCHEMA as PLAN_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import (
    BOUNDED_DOMAIN_SCHEMA,
    PHYSICAL_DOMAIN_SCHEMA,
    build_strict_explicit_chart_atlas,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _seal(metadata):
    metadata = dict(metadata)
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    return metadata


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _plan(path: Path, names: list[str], authority_hash: str, tree_hash: str):
    count = len(names)
    zeros = np.zeros((count, count), np.float64)
    edges = np.ones((count, count), bool)
    np.fill_diagonal(edges, False)
    metadata = {
        "artifact_type": PLAN_SCHEMA,
        "uses_query_or_ground_truth": False,
        "chart_count": count,
        "selected_chart_count": count,
        "source_ordered_names_sha256": canonical_json_sha256(names),
        "selected_chart_names_in_order": names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(names),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "components": [
            {"operational_coverage_pass": True, "selected_chart_names": names}
        ],
        "operational_submap_count": 1,
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
    plan = ChartSubmapPlan(
        chart_names=np.asarray(names),
        camera_centers_world=np.stack(
            [np.asarray([0.2 * row, 0.0, 0.0]) for row in range(count)]
        ),
        camera_forward_world=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1)),
        valid_sample_counts=np.full((count,), 81, np.int64),
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


def _initializer_run(
    root: Path,
    *,
    arm: str,
    names: list[str],
    c2w: np.ndarray,
    image_paths: list[Path],
    authority_hash: str,
    height: int,
    width: int,
):
    root.mkdir()
    rows = []
    depths = 1.5 if arm == "DAV2" else 1.8
    yy, xx = np.mgrid[:height, :width]
    for chart, name in enumerate(names):
        camera_points = np.stack(
            (
                (xx - width / 2) / 20.0 * depths,
                (yy - height / 2) / 20.0 * depths,
                np.full((height, width), depths),
            ),
            axis=2,
        )
        valid = np.ones((height, width), bool)
        if arm == "DAV2":
            points = camera_points @ c2w[chart, :3, :3].T + c2w[chart, :3, 3]
            metadata = {
                "artifact_type": "goal_maplet_dav2_chart_initializer_v1",
                "source_name": name,
                "disjoint_upstream_authority_content_sha256": authority_hash,
                "uses_query_or_ground_truth": False,
            }
            array_name = "points_world"
        else:
            points = camera_points
            metadata = {
                "artifact_type": "goal_maplet_moge3_chart_initializer_v2",
                "source_name": name,
                "source_image_file_sha256": file_sha256(image_paths[chart]),
                "uses_camera_pose": False,
                "uses_query_or_ground_truth": False,
            }
            array_name = "points_camera"
        metadata = _seal(metadata)
        path = root / f"{name}.npz"
        np.savez_compressed(
            path,
            **{array_name: points.astype(np.float32)},
            valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append(
            {
                "name": name,
                "file_sha256": file_sha256(path),
                "content_sha256": metadata["content_sha256"],
            }
        )
    manifest = {
        "artifact_type": (
            "goal_maplet_dav2_chart_initializer_run_v1"
            if arm == "DAV2"
            else "goal_maplet_moge3_chart_initializer_run_v2"
        ),
        "chart_count": len(rows),
        "uses_query_or_ground_truth": False,
        "rows": rows,
    }
    if arm == "DAV2":
        manifest["disjoint_upstream_authority_content_sha256"] = authority_hash
    manifest = _seal(manifest)
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _alignment_arm(
    root: Path,
    *,
    arm: str,
    names: list[str],
    cameras_path: Path,
    c2w: np.ndarray,
    valid: np.ndarray,
    face4: np.ndarray,
    face8: np.ndarray,
    initializer_root: Path,
    authority_hash: str,
    upstream_path: Path,
    upstream_hash: str,
    plan_hash: str,
):
    root.mkdir()
    height, width = valid.shape[1:]
    aligned_depth = 2.0 if arm == "DAV2" else 2.2
    prior_depth = 1.5 if arm == "DAV2" else 1.8
    yy, xx = np.mgrid[:height, :width]
    points = []
    for chart in range(len(names)):
        camera_points = np.stack(
            (
                (xx - width / 2) / 20.0 * aligned_depth,
                (yy - height / 2) / 20.0 * aligned_depth,
                np.full((height, width), aligned_depth),
            ),
            axis=2,
        )
        points.append(camera_points @ c2w[chart, :3, :3].T + c2w[chart, :3, 3])
    charts_path = root / "charts_data.npz"
    np.savez_compressed(
        charts_path,
        pts=np.asarray(points, np.float32),
        depths=np.full(valid.shape, aligned_depth, np.float32),
        prior_depths=np.full(valid.shape, prior_depth, np.float32),
        confs=np.full(valid.shape, 3.0 if arm == "DAV2" else -2.0, np.float32),
        valid=valid,
        chart_names=np.asarray(names),
        scale_factor=np.asarray(1.0),
        comparison_face_valid_stride4=face4,
        comparison_face_valid_stride8=face8,
        training_losses=np.asarray([1.0]),
    )
    initializer_hashes = [
        file_sha256(initializer_root / f"{name}.npz") for name in names
    ]
    manifest = _seal(
        {
            "artifact_type": "goal_maplet_masked_chart_alignment_gate_v1",
            "initializer": "dav2" if arm == "DAV2" else "moge3",
            "chart_names": names,
            "uses_query_or_ground_truth": False,
            "paired_common_pixel_domain": True,
            "common_face_inventory_deferred_to_explicit_atlas_export": True,
            "output_restored_to_frozen_plan_order": True,
            "disjoint_upstream_authority_content_sha256": authority_hash,
            "comparison_domain_content_sha256": upstream_hash,
            "comparison_domain_file_sha256": file_sha256(upstream_path),
            "frozen_submap_plan_content_sha256": plan_hash,
            "selected_chart_names_in_order_sha256": canonical_json_sha256(names),
            "selection_contract": ALIGNMENT_SELECTION_CONTRACT,
            "alignment_runner_file_sha256": "8" * 64,
            "charts_data_file_sha256": file_sha256(charts_path),
            "subset_cameras_file_sha256": file_sha256(cameras_path),
            "initializer_file_sha256": initializer_hashes,
            "common_pixel_mask_sha256": arrays_sha256({"valid": valid}),
            "common_face_mask_sha256": {
                "stride4": arrays_sha256({"face_valid_stride4": face4}),
                "stride8": arrays_sha256({"face_valid_stride8": face8}),
            },
            "alignment_code_inventory_sha256": "9" * 64,
        }
    )
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return charts_path, manifest_path, manifest


def _fixture(tmp_path: Path):
    names = ["seq4__b.png", "seq4__a.png"]
    tree_hash = "1" * 64
    coordinate_camera_hash = "2" * 64
    coordinate_image_hash = "3" * 64
    source_reference_root = tmp_path / "source_reference"
    pointmap_root = source_reference_root / "pointmaps"
    pointmap_root.mkdir(parents=True)
    source_reference_cameras = source_reference_root / "cameras.json"
    _write_json(source_reference_cameras, {"synthetic": True})
    pointmap_inventory = {}
    for row, name in enumerate(names):
        pointmap_path = pointmap_root / Path(name).with_suffix(".json").name
        _write_json(pointmap_path, {"name": name, "row": row})
        pointmap_inventory[name] = file_sha256(pointmap_path)
    authority_path = tmp_path / "authority.json"
    authority = _seal(
        {
            "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
            "strict_disjoint_upstream": True,
            "source_held_image_disjoint": True,
            "source_held_route_disjoint": True,
            "physical_source_held_input_roots_disjoint": True,
            "uses_query_or_ground_truth": False,
            "forbidden_routes_opened": False,
            "posed_colmap_cameras_file_sha256": coordinate_camera_hash,
            "posed_colmap_images_file_sha256": coordinate_image_hash,
            "source": {
                "ordered_names": names,
                "tree_sha256": tree_hash,
                "root": str(source_reference_root.resolve()),
                "cameras_file_sha256": file_sha256(source_reference_cameras),
                "pointmap_inventory_sha256": canonical_json_sha256(
                    pointmap_inventory
                ),
            },
        }
    )
    _write_json(authority_path, authority)
    plan_path = tmp_path / "plan.npz"
    plan_metadata = _plan(plan_path, names, authority["content_sha256"], tree_hash)

    image_paths = [tmp_path / name for name in names]
    for row, path in enumerate(image_paths):
        path.write_bytes(bytes([row + 1]) * 7)
    c2w = np.repeat(np.eye(4)[None], len(names), axis=0)
    c2w[1, 0, 3] = 0.2
    cameras = {
        "filepaths": [str(path) for path in image_paths],
        "focals": [20.0, 20.0],
        "cams2world": c2w.tolist(),
    }
    cameras_path = tmp_path / "cameras.json"
    _write_json(cameras_path, cameras)

    height = width = 9
    valid = np.ones((len(names), height, width), bool)
    face4 = np.ones((len(names), 2, 2), bool)
    face8 = np.ones((len(names), 1, 1), bool)
    base_arrays = {
        "chart_names": np.asarray(names),
        "valid": valid,
        "face_valid_stride4": face4,
        "face_valid_stride8": face8,
    }
    upstream_metadata = _seal(
        {
            "artifact_type": "goal_maplet_chart_comparison_domain_v1",
            "arrays_sha256": arrays_sha256(base_arrays),
            "uses_query_or_ground_truth": False,
        }
    )
    upstream_path = tmp_path / "domain_v1.npz"
    np.savez_compressed(
        upstream_path,
        **base_arrays,
        metadata_json=np.asarray(json.dumps(upstream_metadata, sort_keys=True)),
    )

    initializer = {}
    for arm in ("DAV2", "MoGe3"):
        root = tmp_path / f"{arm.lower()}_initializers"
        manifest_path, manifest = _initializer_run(
            root,
            arm=arm,
            names=names,
            c2w=c2w,
            image_paths=image_paths,
            authority_hash=authority["content_sha256"],
            height=height,
            width=width,
        )
        initializer[arm] = (root, manifest_path, manifest)

    topology = build_exact_topology_arrays(valid, {4: face4, 8: face8})
    domain_arrays = {**base_arrays, **topology}
    domain_metadata = _seal(
        {
            "artifact_type": EXACT_DOMAIN_SCHEMA,
            "chart_count": len(names),
            "arrays_sha256": arrays_sha256(domain_arrays),
            "exact_topology_arrays_sha256": arrays_sha256(topology),
            "exact_pixel_mask_frozen_for_both_arms": True,
            "exact_face_indices_frozen_for_both_arms": True,
            "full_submap_gate_eligible": True,
            "uses_query_or_ground_truth": False,
            "upstream_comparison_domain_content_sha256": upstream_metadata[
                "content_sha256"
            ],
            "upstream_comparison_domain_file_sha256": file_sha256(upstream_path),
            "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
            "disjoint_upstream_authority_content_sha256": authority[
                "content_sha256"
            ],
            "source_tree_sha256": tree_hash,
            "mapping_source_ordered_names_sha256": canonical_json_sha256(names),
            "frozen_submap_plan_file_sha256": file_sha256(plan_path),
            "frozen_submap_plan_content_sha256": plan_metadata["content_sha256"],
            "selected_chart_names_in_order_sha256": canonical_json_sha256(names),
            "dav2_initializer_manifest_file_sha256": file_sha256(
                initializer["DAV2"][1]
            ),
            "dav2_initializer_manifest_content_sha256": initializer["DAV2"][2][
                "content_sha256"
            ],
            "moge_initializer_manifest_file_sha256": file_sha256(
                initializer["MoGe3"][1]
            ),
            "moge_initializer_manifest_content_sha256": initializer["MoGe3"][2][
                "content_sha256"
            ],
        }
    )
    domain_path = tmp_path / "domain_v2.npz"
    np.savez_compressed(
        domain_path,
        **domain_arrays,
        metadata_json=np.asarray(json.dumps(domain_metadata, sort_keys=True)),
    )

    arms = {}
    for arm in ("DAV2", "MoGe3"):
        charts, manifest_path, manifest = _alignment_arm(
            tmp_path / f"{arm.lower()}_alignment",
            arm=arm,
            names=names,
            cameras_path=cameras_path,
            c2w=c2w,
            valid=valid,
            face4=face4,
            face8=face8,
            initializer_root=initializer[arm][0],
            authority_hash=authority["content_sha256"],
            upstream_path=upstream_path,
            upstream_hash=upstream_metadata["content_sha256"],
            plan_hash=plan_metadata["content_sha256"],
        )
        arms[arm] = (charts, manifest_path, manifest)

    minimum = np.asarray([-10.0, -10.0, -1.0])
    maximum = np.asarray([10.0, 10.0, 10.0])
    bounded_hash = canonical_json_sha256(
        {
            "artifact_type": BOUNDED_DOMAIN_SCHEMA,
            "minimum_world": minimum.tolist(),
            "maximum_world": maximum.tolist(),
        }
    )
    bounds = _seal(
        {
            "artifact_type": BOUNDED_DOMAIN_SCHEMA,
            "minimum_world": minimum.tolist(),
            "maximum_world": maximum.tolist(),
            "bounded_submap_content_sha256": bounded_hash,
            "uses_query_or_ground_truth": False,
            "held_geometry_consumed": False,
            "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
            "disjoint_upstream_authority_content_sha256": authority[
                "content_sha256"
            ],
            "mapping_source_ordered_names_sha256": canonical_json_sha256(names),
            "comparison_domain_file_sha256": file_sha256(domain_path),
            "comparison_domain_content_sha256": domain_metadata["content_sha256"],
            "frozen_submap_plan_file_sha256": file_sha256(plan_path),
            "frozen_submap_plan_content_sha256": plan_metadata["content_sha256"],
            "coordinate_cameras_file_sha256": coordinate_camera_hash,
            "coordinate_images_file_sha256": coordinate_image_hash,
            "source_bound_derivation_schema": (
                "goal_maplet_source_only_submap_bound_derivation_v1"
            ),
            "pre_frozen_builder_config_content_sha256": "4" * 64,
            "source_bound_margin_m": 1.0,
        }
    )
    bounds_path = tmp_path / "bounds.json"
    _write_json(bounds_path, bounds)
    return {
        "names": names,
        "authority_path": authority_path,
        "authority": authority,
        "plan_path": plan_path,
        "plan": plan_metadata,
        "cameras_path": cameras_path,
        "domain_path": domain_path,
        "domain": domain_metadata,
        "bounds_path": bounds_path,
        "bounded_hash": bounded_hash,
        "initializer": initializer,
        "arms": arms,
    }


def _build(value, arm, state, *, allow_legacy_v2_diagnostic=True):
    charts, alignment_manifest_path, alignment_manifest = value["arms"][arm]
    initializer_root, initializer_manifest_path, initializer_manifest = value[
        "initializer"
    ][arm]
    return build_strict_explicit_chart_atlas(
        charts,
        value["cameras_path"],
        alignment_manifest_path=alignment_manifest_path,
        expected_alignment_manifest_content_sha256=alignment_manifest[
            "content_sha256"
        ],
        initializer_artifacts_path=initializer_root,
        initializer_manifest_path=initializer_manifest_path,
        expected_initializer_manifest_content_sha256=initializer_manifest[
            "content_sha256"
        ],
        comparison_domain_path=value["domain_path"],
        expected_comparison_domain_content_sha256=value["domain"]["content_sha256"],
        frozen_submap_plan_path=value["plan_path"],
        expected_plan_content_sha256=value["plan"]["content_sha256"],
        authority_path=value["authority_path"],
        expected_authority_content_sha256=value["authority"]["content_sha256"],
        bounded_submap_path=value["bounds_path"],
        expected_bounded_submap_content_sha256=value["bounded_hash"],
        initializer_arm=arm,
        geometry_state=state,
        stride=4,
        allow_legacy_v2_diagnostic=allow_legacy_v2_diagnostic,
    )


def _promote_fixture_to_physical_v3(value):
    with np.load(value["domain_path"], allow_pickle=False) as data:
        v2_arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        v2_metadata = json.loads(str(data["metadata_json"].item()))
    v3_base = {
        name: np.asarray(v2_arrays[name]).copy()
        for name in (
            "chart_names",
            "valid",
            "face_valid_stride4",
            "face_valid_stride8",
        )
    }
    # A source-reference physical-edge audit removes one face that was valid
    # for the two initializer arms.  The optimizer still consumes v1/v2.
    v3_base["face_valid_stride4"][0, 0, 0] = False
    v3_topology = build_exact_topology_arrays(
        v3_base["valid"],
        {
            4: v3_base["face_valid_stride4"],
            8: v3_base["face_valid_stride8"],
        },
    )
    v3_arrays = {**v3_base, **v3_topology}
    final_mask_hash = arrays_sha256(
        {
            "face_valid_stride4": v3_base["face_valid_stride4"],
            "face_valid_stride8": v3_base["face_valid_stride8"],
        }
    )
    source = value["authority"]["source"]
    selected_pointmaps = {}
    for name in value["names"]:
        pointmap_path = (
            Path(source["root"]) / "pointmaps" / Path(name).with_suffix(".json").name
        )
        selected_pointmaps[name] = file_sha256(pointmap_path)
    safety_config = {
        "absolute_edge_threshold_m": 0.5,
        "relative_edge_threshold": 0.05,
        "point_coordinate_frame": "synthetic_source_reference_world",
    }
    v3_metadata = dict(v2_metadata)
    v3_metadata.pop("content_sha256")
    v3_metadata.update(
        {
            "artifact_type": PHYSICAL_DOMAIN_SCHEMA,
            "arrays_sha256": arrays_sha256(v3_arrays),
            "exact_topology_arrays_sha256": arrays_sha256(v3_topology),
            "source_reference_edge_safe_face_masks_sha256": final_mask_hash,
            "final_reference_safe_face_masks_sha256": final_mask_hash,
            "parent_v2_face_valid_arrays_sha256": arrays_sha256(
                {
                    "face_valid_stride4": v2_arrays["face_valid_stride4"],
                    "face_valid_stride8": v2_arrays["face_valid_stride8"],
                }
            ),
            "source_reference_edge_safe": True,
            "physical_face_safety_authority": True,
            "face_valid_v3_subset_of_face_valid_v2": True,
            "exact_topology_repacked_after_reference_safety": True,
            "valid_and_chart_names_byte_equal_upstream_v2": True,
            "source_reference_edge_safety_replayed": True,
            "orphan_sampled_vertices_present": False,
            "source_reference_root": source["root"],
            "source_reference_cameras_file_sha256": source[
                "cameras_file_sha256"
            ],
            "source_reference_selected_pointmap_inventory": selected_pointmaps,
            "source_reference_selected_pointmap_inventory_sha256": (
                canonical_json_sha256(selected_pointmaps)
            ),
            "pointmap_inventory": selected_pointmaps,
            "source_reference_full_pointmap_inventory_sha256": source[
                "pointmap_inventory_sha256"
            ],
            "source_reference_edge_safety_config": safety_config,
            "source_reference_edge_safety_config_sha256": canonical_json_sha256(
                safety_config
            ),
            "upstream_exact_topology_v2_file_sha256": file_sha256(
                value["domain_path"]
            ),
            "upstream_exact_topology_v2_content_sha256": v2_metadata[
                "content_sha256"
            ],
            "upstream_exact_topology_v2_arrays_sha256": v2_metadata[
                "arrays_sha256"
            ],
            "upstream_exact_topology_v2_exact_topology_arrays_sha256": (
                v2_metadata["exact_topology_arrays_sha256"]
            ),
            "upstream_optimizer_comparison_domain_v1_file_sha256": (
                v2_metadata["upstream_comparison_domain_file_sha256"]
            ),
            "upstream_optimizer_comparison_domain_v1_content_sha256": (
                v2_metadata["upstream_comparison_domain_content_sha256"]
            ),
            "upstream_optimizer_comparison_domain_v1_arrays_sha256": (
                v2_metadata["base_domain_arrays_sha256"]
                if "base_domain_arrays_sha256" in v2_metadata
                else arrays_sha256(v3_base)
            ),
        }
    )
    # The synthetic v2 fixture predates the production base-domain hash field.
    if "base_domain_arrays_sha256" not in v2_metadata:
        v2_metadata_without_content = dict(v2_metadata)
        v2_metadata_without_content.pop("content_sha256")
        v2_metadata_without_content["base_domain_arrays_sha256"] = arrays_sha256(
            {
                name: v2_arrays[name]
                for name in (
                    "chart_names",
                    "valid",
                    "face_valid_stride4",
                    "face_valid_stride8",
                )
            }
        )
        v2_metadata_without_content = _seal(v2_metadata_without_content)
        np.savez_compressed(
            value["domain_path"],
            **v2_arrays,
            metadata_json=np.asarray(
                json.dumps(v2_metadata_without_content, sort_keys=True)
            ),
        )
        v2_metadata = v2_metadata_without_content
        v3_metadata["upstream_exact_topology_v2_file_sha256"] = file_sha256(
            value["domain_path"]
        )
        v3_metadata["upstream_exact_topology_v2_content_sha256"] = v2_metadata[
            "content_sha256"
        ]
        v3_metadata["upstream_optimizer_comparison_domain_v1_arrays_sha256"] = (
            v2_metadata["base_domain_arrays_sha256"]
        )
    v3_metadata = _seal(v3_metadata)
    v3_path = value["domain_path"].with_name("domain_v3.npz")
    np.savez_compressed(
        v3_path,
        **v3_arrays,
        metadata_json=np.asarray(json.dumps(v3_metadata, sort_keys=True)),
    )

    bounds = json.loads(value["bounds_path"].read_text())
    bounds.pop("content_sha256")
    bounds["comparison_domain_file_sha256"] = file_sha256(v3_path)
    bounds["comparison_domain_content_sha256"] = v3_metadata["content_sha256"]
    bounds = _seal(bounds)
    _write_json(value["bounds_path"], bounds)
    value.update(
        {
            "v2_domain_path": value["domain_path"],
            "v2_domain": v2_metadata,
            "domain_path": v3_path,
            "domain": v3_metadata,
        }
    )
    return value


def _build_physical(value, arm, state, *, stride=4):
    charts, alignment_manifest_path, alignment_manifest = value["arms"][arm]
    initializer_root, initializer_manifest_path, initializer_manifest = value[
        "initializer"
    ][arm]
    return build_strict_explicit_chart_atlas(
        charts,
        value["cameras_path"],
        alignment_manifest_path=alignment_manifest_path,
        expected_alignment_manifest_content_sha256=alignment_manifest[
            "content_sha256"
        ],
        initializer_artifacts_path=initializer_root,
        initializer_manifest_path=initializer_manifest_path,
        expected_initializer_manifest_content_sha256=initializer_manifest[
            "content_sha256"
        ],
        comparison_domain_path=value["domain_path"],
        expected_comparison_domain_content_sha256=value["domain"]["content_sha256"],
        upstream_exact_topology_v2_path=value["v2_domain_path"],
        expected_upstream_exact_topology_v2_content_sha256=value["v2_domain"][
            "content_sha256"
        ],
        frozen_submap_plan_path=value["plan_path"],
        expected_plan_content_sha256=value["plan"]["content_sha256"],
        authority_path=value["authority_path"],
        expected_authority_content_sha256=value["authority"]["content_sha256"],
        bounded_submap_path=value["bounds_path"],
        expected_bounded_submap_content_sha256=value["bounded_hash"],
        initializer_arm=arm,
        geometry_state=state,
        stride=stride,
    )


def test_v2_requires_explicit_diagnostic_opt_in_and_is_not_gate_eligible(tmp_path):
    value = _fixture(tmp_path)
    with pytest.raises(ValueError, match="legacy diagnostic only"):
        _build(value, "DAV2", "aligned", allow_legacy_v2_diagnostic=False)
    atlas, _ = _build(value, "DAV2", "aligned")
    assert atlas.metadata["legacy_v2_diagnostic"] is True
    assert atlas.metadata["source_reference_edge_safe"] is False
    assert atlas.metadata["full_submap_gate_eligible"] is False


def test_v3_replays_v2_to_v1_and_uses_only_physical_safe_topology(tmp_path):
    value = _promote_fixture_to_physical_v3(_fixture(tmp_path))
    initial, _ = _build_physical(value, "DAV2", "initial")
    aligned, _ = _build_physical(value, "MoGe3", "aligned")
    assert initial.metadata["comparison_domain_artifact_type"] == PHYSICAL_DOMAIN_SCHEMA
    assert initial.metadata["source_reference_edge_safe"] is True
    assert initial.metadata["full_submap_gate_eligible"] is True
    assert initial.metadata["legacy_v2_diagnostic"] is False
    assert np.array_equal(initial.chart_vertex_offsets, aligned.chart_vertex_offsets)
    assert np.array_equal(initial.chart_face_offsets, aligned.chart_face_offsets)
    assert np.array_equal(initial.faces, aligned.faces)
    assert np.array_equal(initial.uv, aligned.uv)
    # One accepted stride-4 quad (two triangles) was removed from v2.
    assert len(initial.faces) == 14


def test_v3_cannot_add_a_face_outside_externally_pinned_v2(tmp_path):
    value = _promote_fixture_to_physical_v3(_fixture(tmp_path))
    with np.load(value["domain_path"], allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    with np.load(value["v2_domain_path"], allow_pickle=False) as data:
        upstream_face4 = np.asarray(data["face_valid_stride4"], bool)
    # First remove a v2 face, then make the rehashed v3 attempt to restore it.
    upstream_face4 = upstream_face4.copy()
    upstream_face4[1, 1, 1] = False
    with np.load(value["v2_domain_path"], allow_pickle=False) as data:
        upstream_arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        upstream_metadata = json.loads(str(data["metadata_json"].item()))
    upstream_arrays["face_valid_stride4"] = upstream_face4
    upstream_topology = build_exact_topology_arrays(
        upstream_arrays["valid"],
        {
            4: upstream_arrays["face_valid_stride4"],
            8: upstream_arrays["face_valid_stride8"],
        },
    )
    upstream_arrays.update(upstream_topology)
    upstream_metadata.pop("content_sha256")
    upstream_metadata["arrays_sha256"] = arrays_sha256(upstream_arrays)
    upstream_metadata["exact_topology_arrays_sha256"] = arrays_sha256(
        upstream_topology
    )
    upstream_metadata = _seal(upstream_metadata)
    np.savez_compressed(
        value["v2_domain_path"],
        **upstream_arrays,
        metadata_json=np.asarray(json.dumps(upstream_metadata, sort_keys=True)),
    )
    metadata.pop("content_sha256")
    metadata.update(
        {
            "upstream_exact_topology_v2_file_sha256": file_sha256(
                value["v2_domain_path"]
            ),
            "upstream_exact_topology_v2_content_sha256": upstream_metadata[
                "content_sha256"
            ],
            "upstream_exact_topology_v2_arrays_sha256": upstream_metadata[
                "arrays_sha256"
            ],
            "upstream_exact_topology_v2_exact_topology_arrays_sha256": (
                upstream_metadata["exact_topology_arrays_sha256"]
            ),
        }
    )
    metadata = _seal(metadata)
    np.savez_compressed(
        value["domain_path"],
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    value["domain"] = metadata
    value["v2_domain"] = upstream_metadata
    bounds = json.loads(value["bounds_path"].read_text())
    bounds.pop("content_sha256")
    bounds["comparison_domain_file_sha256"] = file_sha256(value["domain_path"])
    bounds["comparison_domain_content_sha256"] = metadata["content_sha256"]
    _write_json(value["bounds_path"], _seal(bounds))
    with pytest.raises(ValueError, match="not a subset of v2"):
        _build_physical(value, "DAV2", "aligned")


def test_v3_export_rejects_nonprimary_stride_even_when_stride8_has_faces(tmp_path):
    value = _promote_fixture_to_physical_v3(_fixture(tmp_path))
    with pytest.raises(ValueError, match="restricted to primary stride 4"):
        _build_physical(value, "DAV2", "aligned", stride=8)


def test_strict_initial_and_aligned_use_raw_initializer_and_exact_v2_topology(tmp_path):
    value = _fixture(tmp_path)
    initial, initial_audit = _build(value, "DAV2", "initial")
    aligned, _ = _build(value, "DAV2", "aligned")
    assert np.allclose(initial.vertices_world[:, 2], 1.5)
    assert np.allclose(aligned.vertices_world[:, 2], 2.0)
    assert np.array_equal(initial.chart_vertex_offsets, aligned.chart_vertex_offsets)
    assert np.array_equal(initial.chart_face_offsets, aligned.chart_face_offsets)
    assert np.array_equal(initial.faces, aligned.faces)
    assert np.array_equal(initial.uv, aligned.uv)
    assert initial.metadata["is_pre_alignment_geometry"] is True
    assert aligned.metadata["is_pre_alignment_geometry"] is False
    assert initial.metadata["initial_geometry_contract"].startswith("raw per-chart")
    assert np.allclose(initial_audit["initial_points"], initial.vertices_world)


def test_dav2_and_moge_arms_share_byte_exact_topology_and_uv(tmp_path):
    value = _fixture(tmp_path)
    dav2, _ = _build(value, "DAV2", "aligned")
    moge, _ = _build(value, "MoGe3", "aligned")
    for name in (
        "chart_names",
        "chart_vertex_offsets",
        "chart_face_offsets",
        "faces",
        "uv",
    ):
        assert np.array_equal(getattr(dav2, name), getattr(moge, name))
    assert not np.array_equal(dav2.vertices_world, moge.vertices_world)


def test_tampered_exact_faces_fail_even_with_rehashed_npz(tmp_path):
    value = _fixture(tmp_path)
    with np.load(value["domain_path"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"].item()))
    arrays["faces_stride4"] = arrays["faces_stride4"].copy()
    arrays["faces_stride4"][0] = arrays["faces_stride4"][0, ::-1]
    topology = {name: arrays[name] for name in arrays if name.startswith(("sampled_", "face_offsets", "faces_"))}
    metadata.pop("content_sha256")
    metadata["arrays_sha256"] = arrays_sha256(arrays)
    metadata["exact_topology_arrays_sha256"] = arrays_sha256(topology)
    metadata = _seal(metadata)
    np.savez_compressed(
        value["domain_path"],
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    value["domain"] = metadata
    with pytest.raises(ValueError, match="does not replay pixel/face masks"):
        _build(value, "DAV2", "aligned")


def test_alignment_v1_substitution_cannot_hide_behind_valid_v2_domain(tmp_path):
    value = _fixture(tmp_path)
    _, path, manifest = value["arms"]["DAV2"]
    manifest = dict(manifest)
    manifest.pop("content_sha256")
    manifest["comparison_domain_content_sha256"] = "f" * 64
    manifest = _seal(manifest)
    _write_json(path, manifest)
    value["arms"]["DAV2"] = (value["arms"]["DAV2"][0], path, manifest)
    with pytest.raises(ValueError, match="comparison_domain_content_sha256 differs"):
        _build(value, "DAV2", "aligned")


def test_initializer_substitution_is_rejected_by_alignment_and_run_manifests(tmp_path):
    value = _fixture(tmp_path)
    path = value["initializer"]["DAV2"][0] / f"{value['names'][0]}.npz"
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["points_world"] = payload["points_world"].copy()
    payload["points_world"][0, 0, 2] += 1.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="initializer bytes differ"):
        _build(value, "DAV2", "initial")


def test_missing_or_cross_authority_bounds_fail_closed(tmp_path):
    value = _fixture(tmp_path)
    bounds = json.loads(value["bounds_path"].read_text())
    bounds.pop("content_sha256")
    bounds["disjoint_upstream_authority_content_sha256"] = "e" * 64
    bounds = _seal(bounds)
    _write_json(value["bounds_path"], bounds)
    with pytest.raises(ValueError, match="bounded submap disjoint"):
        _build(value, "DAV2", "aligned")
