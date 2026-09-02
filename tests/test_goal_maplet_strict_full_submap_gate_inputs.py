from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    SCHEMA as V2_COMPARISON_SCHEMA,
    build_exact_topology_arrays,
    source_tree_sha256,
)
from feature_extract.vfm.localization_goal_maplet.chart_comparison_reference_safe_domain import (
    OPTIMIZER_V1_SCHEMA,
    SCHEMA as COMPARISON_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    ChartSubmapPlan,
    SCHEMA as PLAN_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    SourceOnlyBoundedSurfaceBaseline,
    StrictHeldRayInventory,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_gate_inputs import (
    StrictGateInputBuildConfig,
    bounded_submap_authority_from_artifacts,
    build_strict_full_submap_gate_inputs,
    classify_2dgs_comparison_budget,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


SOURCE_NAMES = ["seq4__a.png", "seq4__b.png"]
HELD_NAMES = ["seq1__a.png", "seq1__b.png", "seq1__c.png", "seq1__d.png"]
HEIGHT, WIDTH = 6, 9
SOURCE_HEIGHT, SOURCE_WIDTH = 12, 18


def _seal(value):
    value["content_sha256"] = canonical_json_sha256(value)
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _pointmap_inventory(root: Path, names: list[str]) -> str:
    return canonical_json_sha256(
        [
            {
                "name": name,
                "file_sha256": file_sha256(
                    root / "pointmaps" / f"{Path(name).stem}.json"
                ),
            }
            for name in names
        ]
    )


def _world_grid(z=5.0):
    focal = 8.0
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    return np.stack(
        (
            (xx - (WIDTH - 1) / 2) / focal * z,
            (yy - (HEIGHT - 1) / 2) / focal * z,
            np.full_like(xx, z, np.float64),
        ),
        axis=2,
    )


def _pointmap_payload(points):
    expanded = np.repeat(np.repeat(points, 2, axis=0), 2, axis=1)
    return {
        "points": expanded.reshape(-1, 3).tolist(),
        "confs": np.ones((SOURCE_HEIGHT, SOURCE_WIDTH)).tolist(),
    }


def _write_mapping_root(root: Path, names: list[str], points) -> Path:
    root.mkdir(parents=True)
    cameras = {
        "filepaths": [str(root / "images" / name) for name in names],
        "focals": [16.0] * len(names),
        "cams2world": np.repeat(np.eye(4)[None], len(names), axis=0).tolist(),
    }
    _write_json(root / "cameras.json", cameras)
    for name in names:
        _write_json(
            root / "pointmaps" / f"{Path(name).stem}.json",
            _pointmap_payload(points),
        )
    return root


def _plan(authority_hash: str, source_tree_hash: str) -> ChartSubmapPlan:
    count = len(SOURCE_NAMES)
    overlap = np.asarray([[0.0, 0.5], [0.5, 0.0]])
    coverage = overlap > 0
    return ChartSubmapPlan(
        chart_names=np.asarray(SOURCE_NAMES),
        camera_centers_world=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        camera_forward_world=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        valid_sample_counts=np.full(count, 100, np.int64),
        directional_frustum_fraction=overlap,
        directional_depth_support=overlap,
        directional_surface_support=overlap,
        symmetric_surface_overlap=overlap,
        camera_baseline_m=coverage.astype(np.float64),
        median_overlap_depth_m=coverage.astype(np.float64) * 5.0,
        baseline_to_depth_ratio=coverage.astype(np.float64) * 0.2,
        median_triangulation_angle_deg=coverage.astype(np.float64) * 10.0,
        camera_forward_angle_deg=coverage.astype(np.float64) * 10.0,
        same_surface_side_fraction=coverage.astype(np.float64),
        coverage_edges=coverage,
        alignment_edges=coverage,
        component_ids=np.zeros(count, np.int32),
        selected_mask=np.ones(count, bool),
        selection_rank=np.arange(count, dtype=np.int32),
        metadata={
            "artifact_type": PLAN_SCHEMA,
            "uses_query_or_ground_truth": False,
            "chart_count": count,
            "selected_chart_count": count,
            "source_ordered_names_sha256": canonical_json_sha256(SOURCE_NAMES),
            "selected_chart_names_in_order": SOURCE_NAMES,
            "selected_chart_names_in_order_sha256": canonical_json_sha256(
                SOURCE_NAMES
            ),
            "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
            "components": [
                {
                    "component_id": 0,
                    "operational_coverage_pass": True,
                    "selected_chart_names": SOURCE_NAMES,
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
                "source_ordered_names_sha256": canonical_json_sha256(SOURCE_NAMES),
                "source_tree_sha256": source_tree_hash,
            },
        },
    ).validated()


def _write_initializer_file(root, name, arm, points):
    if arm == "DAV2":
        metadata = _seal(
            {
                "artifact_type": "goal_maplet_dav2_chart_initializer_v1",
                "source_name": name,
                "uses_query_or_ground_truth": False,
            }
        )
        arrays = {"points_world": points}
    else:
        metadata = _seal(
            {
                "artifact_type": "goal_maplet_moge3_chart_initializer_v2",
                "source_name": name,
                "uses_query_or_ground_truth": False,
            }
        )
        arrays = {"points_camera": points}
    path = root / f"{name}.npz"
    np.savez_compressed(
        path,
        **arrays,
        valid=np.ones((HEIGHT, WIDTH), bool),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "name": name,
        "file_sha256": file_sha256(path),
        "content_sha256": metadata["content_sha256"],
    }


def _fixture(tmp_path: Path):
    points = _world_grid()
    source_root = _write_mapping_root(tmp_path / "source", SOURCE_NAMES, points)
    held_root = _write_mapping_root(tmp_path / "held", HELD_NAMES, points)
    source_tree_hash = source_tree_sha256(source_root)
    authority_path = tmp_path / "authority.json"
    authority = _seal(
        {
            "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
            "source": {
                "root": str(source_root.resolve()),
                "ordered_names": SOURCE_NAMES,
                "routes": ["seq4"],
                "tree_sha256": source_tree_hash,
                "cameras_file_sha256": file_sha256(source_root / "cameras.json"),
                "pointmap_inventory_sha256": _pointmap_inventory(
                    source_root, SOURCE_NAMES
                ),
            },
            "held": {
                "root": str(held_root.resolve()),
                "ordered_names": HELD_NAMES,
                "routes": ["seq1"],
                "cameras_file_sha256": file_sha256(held_root / "cameras.json"),
                "pointmap_inventory_sha256": _pointmap_inventory(
                    held_root, HELD_NAMES
                ),
            },
            "source_held_image_disjoint": True,
            "source_held_route_disjoint": True,
            "physical_source_held_input_roots_disjoint": True,
            "strict_disjoint_upstream": True,
            "forbidden_routes_opened": False,
            "uses_query_or_ground_truth": False,
            "posed_colmap_cameras_file_sha256": "1" * 64,
            "posed_colmap_images_file_sha256": "2" * 64,
        }
    )
    _write_json(authority_path, authority)

    plan_path = tmp_path / "plan.npz"
    plan_metadata = _plan(authority["content_sha256"], source_tree_hash).save_npz(
        plan_path
    )

    valid = np.ones((len(SOURCE_NAMES), HEIGHT, WIDTH), bool)
    face4 = np.ones(
        (
            len(SOURCE_NAMES),
            len(np.arange(0, HEIGHT, 4)) - 1,
            len(np.arange(0, WIDTH, 4)) - 1,
        ),
        bool,
    )
    face8 = np.ones(
        (
            len(SOURCE_NAMES),
            len(np.arange(0, HEIGHT, 8)) - 1,
            len(np.arange(0, WIDTH, 8)) - 1,
        ),
        bool,
    )
    base_arrays = {
        "chart_names": np.asarray(SOURCE_NAMES),
        "valid": valid,
        "face_valid_stride4": face4,
        "face_valid_stride8": face8,
    }
    topology = build_exact_topology_arrays(valid, {4: face4, 8: face8})
    domain_arrays = {**base_arrays, **topology}
    optimizer_v1_path = tmp_path / "comparison_optimizer_v1.npz"
    optimizer_v1_metadata = _seal(
        {
            "artifact_type": OPTIMIZER_V1_SCHEMA,
            "chart_count": len(SOURCE_NAMES),
            "uses_query_or_ground_truth": False,
            "disjoint_upstream_authority_file_sha256": file_sha256(
                authority_path
            ),
            "disjoint_upstream_authority_content_sha256": authority[
                "content_sha256"
            ],
            "frozen_submap_plan_file_sha256": file_sha256(plan_path),
            "frozen_submap_plan_content_sha256": plan_metadata[
                "content_sha256"
            ],
            "mapping_source_ordered_names_sha256": canonical_json_sha256(
                SOURCE_NAMES
            ),
            "source_tree_sha256": source_tree_hash,
            "pointmap_inventory": {
                name: file_sha256(
                    source_root / "pointmaps" / f"{Path(name).stem}.json"
                )
                for name in SOURCE_NAMES
            },
            "cameras_file_sha256": file_sha256(source_root / "cameras.json"),
            "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
            "arrays_sha256": arrays_sha256(base_arrays),
        }
    )
    np.savez_compressed(
        optimizer_v1_path,
        **base_arrays,
        metadata_json=np.asarray(
            json.dumps(optimizer_v1_metadata, sort_keys=True)
        ),
    )
    domain_metadata = _seal(
        {
            "artifact_type": V2_COMPARISON_SCHEMA,
            "chart_count": len(SOURCE_NAMES),
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
            "disjoint_upstream_authority_content_sha256": authority[
                "content_sha256"
            ],
            "frozen_submap_plan_file_sha256": file_sha256(plan_path),
            "frozen_submap_plan_content_sha256": plan_metadata["content_sha256"],
            "mapping_source_ordered_names_sha256": canonical_json_sha256(
                SOURCE_NAMES
            ),
            "source_tree_sha256": source_tree_hash,
            "upstream_comparison_domain_file_sha256": file_sha256(
                optimizer_v1_path
            ),
            "upstream_comparison_domain_content_sha256": optimizer_v1_metadata[
                "content_sha256"
            ],
            "arrays_sha256": arrays_sha256(domain_arrays),
        }
    )
    v2_domain_path = tmp_path / "comparison_exact_v2.npz"
    domain_path = tmp_path / "comparison_reference_safe_v3.npz"

    dav2_root = tmp_path / "dav2"
    moge_root = tmp_path / "moge"
    dav2_root.mkdir()
    moge_root.mkdir()
    dav2_rows = [
        _write_initializer_file(dav2_root, name, "DAV2", points)
        for name in SOURCE_NAMES
    ]
    moge_rows = [
        _write_initializer_file(moge_root, name, "MoGe3", points)
        for name in SOURCE_NAMES
    ]
    _write_json(
        dav2_root / "manifest.json",
        _seal(
            {
                "artifact_type": "goal_maplet_dav2_chart_initializer_run_v1",
                "rows": dav2_rows,
                "uses_query_or_ground_truth": False,
                "disjoint_upstream_authority_content_sha256": authority[
                    "content_sha256"
                ],
                "source_only_mast3r_tree_sha256": source_tree_hash,
            }
        ),
    )
    _write_json(
        moge_root / "manifest.json",
        _seal(
            {
                "artifact_type": "goal_maplet_moge3_chart_initializer_run_v2",
                "rows": moge_rows,
                "uses_query_or_ground_truth": False,
                "uses_camera_pose": False,
                "cameras_file_sha256": file_sha256(source_root / "cameras.json"),
            }
        ),
    )
    # The exact v2 parent cross-binds initializer bytes; the formal gate input
    # is the physical-safe v3 child, which in turn replays v2 -> optimizer v1.
    domain_metadata.pop("content_sha256")
    domain_metadata.update(
        {
            "dav2_initializer_manifest_file_sha256": file_sha256(
                dav2_root / "manifest.json"
            ),
            "dav2_initializer_manifest_content_sha256": json.loads(
                (dav2_root / "manifest.json").read_text()
            )["content_sha256"],
            "moge_initializer_manifest_file_sha256": file_sha256(
                moge_root / "manifest.json"
            ),
            "moge_initializer_manifest_content_sha256": json.loads(
                (moge_root / "manifest.json").read_text()
            )["content_sha256"],
            "dav2_initializer_file_sha256": {
                name: file_sha256(dav2_root / f"{name}.npz")
                for name in SOURCE_NAMES
            },
            "moge_initializer_file_sha256": {
                name: file_sha256(moge_root / f"{name}.npz")
                for name in SOURCE_NAMES
            },
        }
    )
    domain_metadata["content_sha256"] = canonical_json_sha256(domain_metadata)
    np.savez_compressed(
        v2_domain_path,
        **domain_arrays,
        metadata_json=np.asarray(json.dumps(domain_metadata, sort_keys=True)),
    )
    face_masks = {
        "face_valid_stride4": face4,
        "face_valid_stride8": face8,
    }
    v3_metadata = dict(domain_metadata)
    v3_metadata.pop("content_sha256")
    v3_metadata.update(
        {
            "artifact_type": COMPARISON_SCHEMA,
            "upstream_exact_topology_v2_file_sha256": file_sha256(
                v2_domain_path
            ),
            "upstream_exact_topology_v2_content_sha256": domain_metadata[
                "content_sha256"
            ],
            "upstream_exact_topology_v2_arrays_sha256": domain_metadata[
                "arrays_sha256"
            ],
            "upstream_exact_topology_v2_exact_topology_arrays_sha256": (
                domain_metadata["exact_topology_arrays_sha256"]
            ),
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
            "parent_v2_face_valid_arrays_sha256": arrays_sha256(face_masks),
            "source_reference_edge_safe_face_masks_sha256": arrays_sha256(
                face_masks
            ),
            "final_reference_safe_face_masks_sha256": arrays_sha256(face_masks),
        }
    )
    v3_metadata["content_sha256"] = canonical_json_sha256(v3_metadata)
    np.savez_compressed(
        domain_path,
        **domain_arrays,
        metadata_json=np.asarray(json.dumps(v3_metadata, sort_keys=True)),
    )
    return {
        "authority_path": authority_path,
        "comparison_domain_path": domain_path,
        "legacy_v2_domain_path": v2_domain_path,
        "optimizer_v1_domain_path": optimizer_v1_path,
        "frozen_submap_plan_path": plan_path,
        "dav2_initializers": dav2_root,
        "moge3_initializers": moge_root,
        "held_root": held_root,
        "authority": authority,
    }


def _build(paths):
    arguments = {key: value for key, value in paths.items() if key in {
        "authority_path",
        "comparison_domain_path",
        "frozen_submap_plan_path",
        "dav2_initializers",
        "moge3_initializers",
    }}
    return build_strict_full_submap_gate_inputs(
        **arguments,
        config=StrictGateInputBuildConfig(
            topology_stride=4,
            bound_margin_m=1.0,
            temporal_block_size=2,
            output_height=HEIGHT,
            output_width=WIDTH,
            source_canvas_height=SOURCE_HEIGHT,
            source_canvas_width=SOURCE_WIDTH,
            normal_minimum_edge_m=1.0,
            enforce_dense_v4_frozen_contract=False,
        ),
    )


def test_builder_freezes_source_bounds_then_builds_exact_held_inventory(tmp_path):
    rays, surface, audit = _build(_fixture(tmp_path))
    assert isinstance(rays, StrictHeldRayInventory)
    assert isinstance(surface, SourceOnlyBoundedSurfaceBaseline)
    assert rays.view_names.tolist() == HELD_NAMES
    assert rays.reference_valid.reshape(len(HELD_NAMES), -1).sum(1).min() > 0
    assert rays.metadata["held_geometry_opened_after_source_bounds_frozen"] is True
    assert rays.metadata["bound_uses_held_geometry"] is False
    assert surface.metadata["comparison_budget"] == "exact_frozen_source_ordered_pool"
    assert surface.metadata["is_2DGS"] is False
    assert audit["scientific_claim_not_yet_available"] == "chart_atlas_vs_source_only_2DGS"
    bounded = bounded_submap_authority_from_artifacts(rays, surface)
    assert bounded["artifact_type"] == "goal_maplet_axis_aligned_bounded_submap_v1"
    assert bounded["held_geometry_consumed"] is False
    assert bounded["bounded_submap_content_sha256"] == rays.metadata[
        "bounded_submap_content_sha256"
    ]


def test_tampered_held_pointmap_fails_closed(tmp_path):
    paths = _fixture(tmp_path)
    target = paths["held_root"] / "pointmaps" / f"{Path(HELD_NAMES[0]).stem}.json"
    payload = json.loads(target.read_text())
    payload["confs"][0][0] = 0.0
    _write_json(target, payload)
    with pytest.raises(ValueError, match="held point-map inventory differs"):
        _build(paths)


def test_initializer_manifest_must_match_sealed_v3_authority(tmp_path):
    paths = _fixture(tmp_path)
    manifest_path = paths["dav2_initializers"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("content_sha256")
    manifest["post_seal_mutation"] = True
    _write_json(manifest_path, _seal(manifest))
    with pytest.raises(ValueError, match="sealed comparison domain dav2_initializer_manifest"):
        _build(paths)


def test_legacy_exact_v2_is_rejected_as_formal_gate_input(tmp_path):
    paths = _fixture(tmp_path)
    paths["comparison_domain_path"] = paths["legacy_v2_domain_path"]
    with pytest.raises(ValueError, match="physical-safe v3"):
        _build(paths)


def test_physical_safe_v3_requires_replayable_optimizer_v1_parent(tmp_path):
    paths = _fixture(tmp_path)
    paths["optimizer_v1_domain_path"].unlink()
    with pytest.raises(ValueError, match="upstream artifact is absent"):
        _build(paths)


def test_dense_v4_parameters_are_frozen_before_held_access():
    with pytest.raises(ValueError, match="pre-frozen"):
        StrictGateInputBuildConfig(bound_margin_m=1.5).validated()


def test_full_train_2dgs_is_unequal_budget_diagnostic():
    assert classify_2dgs_comparison_budget(
        {
            "mapping_source_ordered_names_sha256": "f" * 64,
            "held_mapping_images_consumed": True,
            "outside_frozen_source_mapping_images_consumed": True,
            "source_2dgs_training_inventory_exact_authority_source": False,
        },
        authority_source_names=SOURCE_NAMES,
    ) == "unequal_budget_diagnostic_only"


def test_only_exact_source_trained_2dgs_can_enter_main_table():
    assert classify_2dgs_comparison_budget(
        {
            "mapping_source_ordered_names_sha256": canonical_json_sha256(
                SOURCE_NAMES
            ),
            "held_mapping_images_consumed": False,
            "outside_frozen_source_mapping_images_consumed": False,
            "source_2dgs_training_inventory_exact_authority_source": True,
        },
        authority_source_names=SOURCE_NAMES,
    ) == "equal_budget_main_table_source_only_2DGS"
