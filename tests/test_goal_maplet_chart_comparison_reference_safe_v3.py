from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    BASE_ARRAY_NAMES,
    SCHEMA as V2_SCHEMA,
    build_exact_topology_arrays,
    source_tree_sha256,
    topology_array_names,
    validate_exact_topology_arrays,
)
from feature_extract.vfm.localization_goal_maplet.chart_comparison_reference_safe_domain import (
    SCHEMA as V3_SCHEMA,
    reference_edge_safe_face_domain,
    seal_reference_safe_comparison_domain,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _save_domain(path, arrays, metadata):
    metadata = dict(metadata)
    metadata["arrays_sha256"] = arrays_sha256(arrays)
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def _pointmap_payload(points):
    height, width = points.shape[:2]
    return {
        "points": points.reshape(-1, 3).tolist(),
        "confs": np.ones((height, width), np.float64).tolist(),
    }


def _fixture(tmp_path):
    names = ["seq4__b.png", "seq4__a.png"]
    height = width = 32
    source_root = tmp_path / "source"
    pointmaps = source_root / "pointmaps"
    pointmaps.mkdir(parents=True)
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    poses[1, 0, 3] = 1.0
    cameras = {
        "filepaths": [f"/isolated/source/{name}" for name in names],
        "cams2world": poses.tolist(),
        "focals": [250.0, 250.0],
    }
    (source_root / "cameras.json").write_text(json.dumps(cameras))
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    base_points = np.stack((0.1 * xx, 0.1 * yy, np.full_like(xx, 10.0)), axis=2)
    for row, name in enumerate(names):
        points = base_points.copy()
        if row == 0:
            points[10:, :, 2] += 6.0
        (pointmaps / f"{Path(name).stem}.json").write_text(
            json.dumps(_pointmap_payload(points))
        )
    tree_hash = source_tree_sha256(source_root)
    pointmap_rows = [
        {
            "name": name,
            "file_sha256": file_sha256(pointmaps / f"{Path(name).stem}.json"),
        }
        for name in sorted(names)
    ]
    authority = {
        "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
        "strict_disjoint_upstream": True,
        "source_held_image_disjoint": True,
        "source_held_route_disjoint": True,
        "physical_source_held_input_roots_disjoint": True,
        "forbidden_routes_opened": False,
        "uses_query_or_ground_truth": False,
        "source": {
            "root": str(source_root),
            "ordered_names": names,
            "tree_sha256": tree_hash,
            "cameras_file_sha256": file_sha256(source_root / "cameras.json"),
            "pointmap_inventory_sha256": canonical_json_sha256(pointmap_rows),
        },
    }
    authority["content_sha256"] = canonical_json_sha256(authority)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority, sort_keys=True))

    valid = np.ones((2, height, width), bool)
    v1_arrays = {
        "chart_names": np.asarray(names),
        "valid": valid,
        "face_valid_stride4": np.ones((2, 7, 7), bool),
        "face_valid_stride8": np.ones((2, 3, 3), bool),
    }
    selected_inventory = {
        name: file_sha256(pointmaps / f"{Path(name).stem}.json") for name in names
    }
    v1_metadata = _save_domain(
        tmp_path / "optimizer_v1.npz",
        v1_arrays,
        {
            "artifact_type": "goal_maplet_chart_comparison_domain_v1",
            "uses_query_or_ground_truth": False,
            "pointmap_inventory": selected_inventory,
            "cameras_file_sha256": file_sha256(source_root / "cameras.json"),
            "disjoint_upstream_authority_content_sha256": authority[
                "content_sha256"
            ],
        },
    )
    v1_path = tmp_path / "optimizer_v1.npz"
    topology = build_exact_topology_arrays(
        valid,
        {4: v1_arrays["face_valid_stride4"], 8: v1_arrays["face_valid_stride8"]},
    )
    v2_arrays = {**v1_arrays, **topology}
    v2_metadata = _save_domain(
        tmp_path / "exact_v2.npz",
        v2_arrays,
        {
            "artifact_type": V2_SCHEMA,
            "uses_query_or_ground_truth": False,
            "exact_topology_arrays_sha256": arrays_sha256(topology),
            "upstream_comparison_domain_file_sha256": file_sha256(v1_path),
            "upstream_comparison_domain_content_sha256": v1_metadata[
                "content_sha256"
            ],
            "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
            "disjoint_upstream_authority_content_sha256": authority[
                "content_sha256"
            ],
            "source_tree_sha256": tree_hash,
            "mapping_source_ordered_names_sha256": canonical_json_sha256(names),
            "full_submap_gate_eligible": True,
            "full_submap_gate_eligible_strides": [4],
            "required_nonempty_face_inventory_strides": [4],
            "exact_pixel_mask_frozen_for_both_arms": True,
            "exact_face_indices_frozen_for_both_arms": True,
            "orphan_sampled_vertices_present": False,
        },
    )
    return {
        "names": names,
        "source_root": source_root,
        "tree_hash": tree_hash,
        "authority": authority,
        "authority_path": authority_path,
        "v1_path": v1_path,
        "v1_hash": v1_metadata["content_sha256"],
        "v2_path": tmp_path / "exact_v2.npz",
        "v2_hash": v2_metadata["content_sha256"],
        "v2_arrays": v2_arrays,
    }


def _seal(value):
    return seal_reference_safe_comparison_domain(
        value["v2_path"],
        expected_upstream_v2_content_sha256=value["v2_hash"],
        optimizer_v1_path=value["v1_path"],
        expected_optimizer_v1_content_sha256=value["v1_hash"],
        authority_path=value["authority_path"],
        expected_authority_content_sha256=value["authority"]["content_sha256"],
        source_root=value["source_root"],
        expected_source_tree_sha256=value["tree_hash"],
    )


def test_reference_edge_safety_uses_camera_range_relative_threshold():
    valid = np.ones((1, 12, 12), bool)
    parent = np.ones((1, 2, 2), bool)
    points = np.zeros((1, 12, 12, 3), np.float64)
    points[..., 2] = 20.0
    points[:, :, :, 0] = np.arange(12, dtype=np.float64) * 0.6
    centers = np.zeros((1, 3), np.float64)
    safe, metrics = reference_edge_safe_face_domain(
        parent, valid, points, centers, stride=4
    )
    assert safe.all()  # 0.6m < 5% of the roughly 20m camera range.
    points[:, :, 2:, 0] += 2.0
    safe, metrics = reference_edge_safe_face_domain(
        parent, valid, points, centers, stride=4
    )
    assert not safe[:, :, 0].any()
    assert metrics["unsafe_face_count"] == 2


def test_v3_seal_removes_reference_unsafe_faces_and_repacks(tmp_path):
    value = _fixture(tmp_path)
    arrays, metadata = _seal(value)
    assert metadata["artifact_type"] == V3_SCHEMA
    assert metadata["source_reference_edge_safety_replayed"] is True
    assert metadata["source_reference_edge_safe"] is True
    assert metadata["physical_face_safety_authority"] is True
    assert metadata["face_valid_v3_subset_of_face_valid_v2"] is True
    assert metadata["exact_topology_repacked_after_reference_safety"] is True
    assert metadata["valid_and_chart_names_byte_equal_upstream_v2"] is True
    assert metadata["orphan_sampled_vertices_present"] is False
    assert np.array_equal(arrays["chart_names"], value["v2_arrays"]["chart_names"])
    assert arrays["valid"].tobytes() == value["v2_arrays"]["valid"].tobytes()
    assert np.all(
        arrays["face_valid_stride4"]
        <= value["v2_arrays"]["face_valid_stride4"]
    )
    assert arrays["face_valid_stride4"].sum() < value["v2_arrays"][
        "face_valid_stride4"
    ].sum()
    topology = {name: arrays[name] for name in topology_array_names()}
    validate_exact_topology_arrays(
        {name: arrays[name] for name in BASE_ARRAY_NAMES},
        topology,
        expected_sha256=metadata["exact_topology_arrays_sha256"],
    )
    assert metadata["source_reference_edge_safety_metrics"]["stride4"][
        "unsafe_face_count"
    ] > 0


def test_v3_rejects_source_pointmap_mutation(tmp_path):
    value = _fixture(tmp_path)
    path = value["source_root"] / "pointmaps" / "seq4__b.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source tree bytes differ"):
        _seal(value)


def test_v3_rejects_optimizer_domain_substitution(tmp_path):
    value = _fixture(tmp_path)
    with np.load(value["v1_path"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in BASE_ARRAY_NAMES}
        metadata = json.loads(str(data["metadata_json"].item()))
    metadata.pop("content_sha256")
    metadata["substitution"] = True
    alternate = tmp_path / "alternate_v1.npz"
    alternate_metadata = _save_domain(alternate, arrays, metadata)
    value["v1_path"] = alternate
    value["v1_hash"] = alternate_metadata["content_sha256"]
    with pytest.raises(ValueError, match="v2 and optimizer v1 lineage differ"):
        _seal(value)


def test_v3_rejects_upstream_v2_face_domain_substitution(tmp_path):
    value = _fixture(tmp_path)
    with np.load(value["v2_path"], allow_pickle=False) as data:
        base = {name: np.asarray(data[name]) for name in BASE_ARRAY_NAMES}
        metadata = json.loads(str(data["metadata_json"].item()))
    base["face_valid_stride4"] = base["face_valid_stride4"].copy()
    base["face_valid_stride4"][0, 0, 0] = False
    topology = build_exact_topology_arrays(
        base["valid"],
        {4: base["face_valid_stride4"], 8: base["face_valid_stride8"]},
    )
    arrays = {**base, **topology}
    metadata.pop("content_sha256")
    metadata["exact_topology_arrays_sha256"] = arrays_sha256(topology)
    substituted = tmp_path / "substituted_v2.npz"
    substituted_metadata = _save_domain(substituted, arrays, metadata)
    value["v2_path"] = substituted
    value["v2_hash"] = substituted_metadata["content_sha256"]
    with pytest.raises(ValueError, match="face_valid_stride4 differs bytewise"):
        _seal(value)
