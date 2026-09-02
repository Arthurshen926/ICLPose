"""Seal source-reference edge safety into an exact paired chart topology.

The optimizer keeps using the frozen v1 common pixel domain.  This module is a
downstream physical-topology authority: it intersects the v2 DAV2/MoGe face
domain with edge safety replayed from the source-only MASt3R point maps, then
re-packs only vertices referenced by surviving faces.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .chart_comparison_domain import (
    BASE_ARRAY_NAMES,
    FULL_GATE_ELIGIBLE_STRIDES,
    SCHEMA as UPSTREAM_V2_SCHEMA,
    TOPOLOGY_STRIDES,
    build_exact_topology_arrays,
    exact_topology_inventory,
    source_tree_sha256,
    topology_array_names,
    validate_exact_topology_arrays,
)
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256


SCHEMA = "goal_maplet_chart_comparison_domain_v3"
OPTIMIZER_V1_SCHEMA = "goal_maplet_chart_comparison_domain_v1"
ABSOLUTE_EDGE_THRESHOLD_M = 0.5
RELATIVE_EDGE_THRESHOLD = 0.05


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_metadata(path: Path, *, expected_content_sha256: str) -> dict[str, object]:
    if not _is_sha256(expected_content_sha256):
        raise ValueError("artifact requires an explicit expected content SHA-256")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    claimed = metadata.pop("content_sha256", None)
    if claimed != canonical_json_sha256(metadata):
        raise ValueError("artifact metadata content hash differs")
    metadata["content_sha256"] = claimed
    if claimed != expected_content_sha256:
        raise ValueError("artifact differs from the externally pinned authority")
    return metadata


def _load_domain_arrays(path: Path, *, include_topology: bool) -> dict[str, np.ndarray]:
    names = list(BASE_ARRAY_NAMES)
    if include_topology:
        names.extend(topology_array_names())
    with np.load(path, allow_pickle=False) as data:
        if any(name not in data.files for name in names):
            raise ValueError("comparison-domain array inventory is incomplete")
        return {name: np.asarray(data[name]) for name in names}


def _pointmap_inventory_sha256(root: Path, ordered_names: list[str]) -> str:
    rows = [
        {
            "name": name,
            "file_sha256": file_sha256(root / "pointmaps" / f"{Path(name).stem}.json"),
        }
        for name in sorted(ordered_names)
    ]
    return canonical_json_sha256(rows)


def _selected_pointmap_inventory(
    root: Path,
    selected_names: list[str],
) -> dict[str, str]:
    return {
        name: file_sha256(root / "pointmaps" / f"{Path(name).stem}.json")
        for name in selected_names
    }


def _load_reference_points(
    path: Path,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    payload = json.loads(path.read_text())
    confidence = np.asarray(payload.get("confs"), np.float64)
    points = np.asarray(payload.get("points"), np.float64)
    if confidence.ndim != 2 or points.size != confidence.size * 3:
        raise ValueError("source reference point map has an invalid shape")
    points = points.reshape(*confidence.shape, 3)
    resized = cv2.resize(
        points,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    if resized.shape != (output_height, output_width, 3):
        raise ValueError("source reference point-map resize differs")
    return np.asarray(resized, np.float64)


def source_reference_edge_safety_config() -> dict[str, object]:
    return {
        "absolute_edge_threshold_m": ABSOLUTE_EDGE_THRESHOLD_M,
        "relative_edge_threshold": RELATIVE_EDGE_THRESHOLD,
        "edge_threshold_formula": (
            "max(absolute_edge_threshold_m, relative_edge_threshold * "
            "median_camera_centered_euclidean_point_range_over_inclusive_quad)"
        ),
        "edge_inventory": (
            "all horizontal and vertical unit-pixel edges in the inclusive "
            "stride patch"
        ),
        "point_coordinate_frame": "source_only_MASt3R_world",
        "camera_binding": "source_root/cameras.json cams2world translation",
        "point_range_definition": "euclidean_norm(world_point-camera_center_world)",
        "pointmap_shape_source": "confs_2d_shape",
        "domain_resize": "cv2_INTER_AREA_world_xyz_to_optimizer_pixel_domain",
        "nonfinite_reference_geometry_safe": False,
    }


def reference_edge_safe_face_domain(
    parent_face_valid: np.ndarray,
    common_valid: np.ndarray,
    reference_points_world: np.ndarray,
    camera_centers_world: np.ndarray,
    *,
    stride: int,
    absolute_edge_threshold_m: float = ABSOLUTE_EDGE_THRESHOLD_M,
    relative_edge_threshold: float = RELATIVE_EDGE_THRESHOLD,
) -> tuple[np.ndarray, dict[str, object]]:
    """Intersect parent faces with source-reference unit-edge safety."""

    parent_face_valid = np.asarray(parent_face_valid, bool)
    common_valid = np.asarray(common_valid, bool)
    reference_points_world = np.asarray(reference_points_world, np.float64)
    camera_centers_world = np.asarray(camera_centers_world, np.float64)
    if common_valid.ndim != 3 or stride < 1:
        raise ValueError("invalid reference edge-safety pixel domain")
    chart_count, height, width = common_valid.shape
    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    expected_faces = (chart_count, max(0, len(ys) - 1), max(0, len(xs) - 1))
    if parent_face_valid.shape != expected_faces:
        raise ValueError("parent face domain shape differs from stride")
    if reference_points_world.shape != (chart_count, height, width, 3):
        raise ValueError("reference point maps differ from optimizer pixel domain")
    if camera_centers_world.shape != (chart_count, 3):
        raise ValueError("camera centers differ from chart inventory")
    if absolute_edge_threshold_m <= 0 or relative_edge_threshold <= 0:
        raise ValueError("reference edge thresholds must be positive")

    safe = np.zeros(parent_face_valid.shape, bool)
    maximum_parent_edge = 0.0
    maximum_unsafe_edge = 0.0
    unsafe_by_chart = np.zeros((chart_count,), np.int64)
    nonfinite_by_chart = np.zeros((chart_count,), np.int64)
    parent_by_chart = parent_face_valid.reshape(chart_count, -1).sum(1)
    for chart in range(chart_count):
        for face_row, face_column in np.argwhere(parent_face_valid[chart]):
            y0, y1 = int(ys[face_row]), int(ys[face_row + 1])
            x0, x1 = int(xs[face_column]), int(xs[face_column + 1])
            if not common_valid[chart, y0 : y1 + 1, x0 : x1 + 1].all():
                raise ValueError("parent face references outside the common pixel domain")
            patch = reference_points_world[chart, y0 : y1 + 1, x0 : x1 + 1]
            horizontal = np.linalg.norm(patch[:, 1:] - patch[:, :-1], axis=2)
            vertical = np.linalg.norm(patch[1:] - patch[:-1], axis=2)
            point_range = np.linalg.norm(
                patch - camera_centers_world[chart], axis=2
            )
            finite = (
                np.isfinite(patch).all()
                and np.isfinite(horizontal).all()
                and np.isfinite(vertical).all()
                and np.isfinite(point_range).all()
            )
            maximum_edge = (
                max(float(horizontal.max()), float(vertical.max()))
                if finite
                else float("inf")
            )
            threshold = (
                max(
                    float(absolute_edge_threshold_m),
                    float(relative_edge_threshold) * float(np.median(point_range)),
                )
                if finite
                else float("-inf")
            )
            if finite:
                maximum_parent_edge = max(maximum_parent_edge, maximum_edge)
            accepted = finite and maximum_edge <= threshold
            safe[chart, face_row, face_column] = accepted
            if not accepted:
                unsafe_by_chart[chart] += 1
                if finite:
                    maximum_unsafe_edge = max(maximum_unsafe_edge, maximum_edge)
                else:
                    nonfinite_by_chart[chart] += 1
    if np.any(safe & ~parent_face_valid):
        raise AssertionError("reference safety expanded the parent face domain")
    metrics = {
        "parent_face_count": int(parent_face_valid.sum()),
        "safe_face_count": int(safe.sum()),
        "unsafe_face_count": int(parent_face_valid.sum() - safe.sum()),
        "parent_face_count_per_chart": parent_by_chart.astype(int).tolist(),
        "unsafe_face_count_per_chart": unsafe_by_chart.astype(int).tolist(),
        "nonfinite_face_count_per_chart": nonfinite_by_chart.astype(int).tolist(),
        "maximum_parent_reference_unit_edge_m": float(maximum_parent_edge),
        "maximum_unsafe_reference_unit_edge_m": float(maximum_unsafe_edge),
    }
    return safe, metrics


def seal_reference_safe_comparison_domain(
    upstream_v2_path: Path,
    *,
    expected_upstream_v2_content_sha256: str,
    optimizer_v1_path: Path,
    expected_optimizer_v1_content_sha256: str,
    authority_path: Path,
    expected_authority_content_sha256: str,
    source_root: Path,
    expected_source_tree_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Create a v3 physical topology subset without changing optimizer pixels."""

    expected = (
        expected_upstream_v2_content_sha256,
        expected_optimizer_v1_content_sha256,
        expected_authority_content_sha256,
        expected_source_tree_sha256,
    )
    if any(not _is_sha256(value) for value in expected):
        raise ValueError("v3 comparison domain requires four explicit SHA-256 pins")
    upstream_v2_path = Path(upstream_v2_path)
    optimizer_v1_path = Path(optimizer_v1_path)
    authority_path = Path(authority_path)
    source_root = Path(source_root).resolve()

    v2_metadata = _load_metadata(
        upstream_v2_path,
        expected_content_sha256=expected_upstream_v2_content_sha256,
    )
    v2_arrays = _load_domain_arrays(upstream_v2_path, include_topology=True)
    v2_base = {name: v2_arrays[name] for name in BASE_ARRAY_NAMES}
    v2_topology = {name: v2_arrays[name] for name in topology_array_names()}
    if v2_metadata.get("artifact_type") != UPSTREAM_V2_SCHEMA:
        raise ValueError("reference-safe sealer requires exact-topology v2")
    if v2_metadata.get("arrays_sha256") != arrays_sha256(v2_arrays):
        raise ValueError("upstream v2 arrays differ from metadata")
    validate_exact_topology_arrays(
        v2_base,
        v2_topology,
        expected_sha256=str(v2_metadata.get("exact_topology_arrays_sha256")),
    )
    required_v2_true = (
        "full_submap_gate_eligible",
        "exact_pixel_mask_frozen_for_both_arms",
        "exact_face_indices_frozen_for_both_arms",
    )
    if any(v2_metadata.get(key) is not True for key in required_v2_true):
        raise ValueError("upstream v2 is not an eligible exact paired topology")
    if v2_metadata.get("orphan_sampled_vertices_present") is not False:
        raise ValueError("upstream v2 contains orphan sampled vertices")
    if v2_metadata.get("uses_query_or_ground_truth") is not False:
        raise ValueError("upstream v2 consumed query or ground truth")
    if v2_metadata.get("full_submap_gate_eligible_strides") != list(
        FULL_GATE_ELIGIBLE_STRIDES
    ):
        raise ValueError("upstream v2 gate-eligible stride contract differs")

    v1_metadata = _load_metadata(
        optimizer_v1_path,
        expected_content_sha256=expected_optimizer_v1_content_sha256,
    )
    v1_arrays = _load_domain_arrays(optimizer_v1_path, include_topology=False)
    if v1_metadata.get("artifact_type") != OPTIMIZER_V1_SCHEMA:
        raise ValueError("optimizer pixel domain is not paired comparison v1")
    if v1_metadata.get("arrays_sha256") != arrays_sha256(v1_arrays):
        raise ValueError("optimizer v1 arrays differ from metadata")
    if v1_metadata.get("uses_query_or_ground_truth") is not False:
        raise ValueError("optimizer v1 consumed query or ground truth")
    if (
        v2_metadata.get("upstream_comparison_domain_file_sha256")
        != file_sha256(optimizer_v1_path)
        or v2_metadata.get("upstream_comparison_domain_content_sha256")
        != expected_optimizer_v1_content_sha256
    ):
        raise ValueError("upstream v2 and optimizer v1 lineage differ")
    for name in BASE_ARRAY_NAMES:
        if (
            v2_base[name].dtype != v1_arrays[name].dtype
            or v2_base[name].shape != v1_arrays[name].shape
            or v2_base[name].tobytes() != v1_arrays[name].tobytes()
        ):
            raise ValueError(f"upstream v2 {name} differs bytewise from optimizer v1")

    authority = json.loads(authority_path.read_text())
    authority_claimed = authority.pop("content_sha256", None)
    if authority_claimed != canonical_json_sha256(authority):
        raise ValueError("source authority content hash differs")
    authority["content_sha256"] = authority_claimed
    if authority_claimed != expected_authority_content_sha256:
        raise ValueError("source authority differs from external pin")
    required_authority_true = (
        "strict_disjoint_upstream",
        "source_held_image_disjoint",
        "source_held_route_disjoint",
        "physical_source_held_input_roots_disjoint",
    )
    if (
        authority.get("artifact_type")
        != "goal_maplet_disjoint_chart_upstream_authority_v2"
        or any(authority.get(key) is not True for key in required_authority_true)
        or authority.get("forbidden_routes_opened") is not False
        or authority.get("uses_query_or_ground_truth") is not False
    ):
        raise ValueError("source authority is not strict route-clean v2")
    source = authority.get("source")
    if not isinstance(source, dict) or Path(str(source.get("root"))).resolve() != source_root:
        raise ValueError("source root differs from strict authority")
    source_names = source.get("ordered_names")
    if not isinstance(source_names, list) or len(set(source_names)) != len(source_names):
        raise ValueError("source authority ordered inventory is invalid")
    if source.get("tree_sha256") != expected_source_tree_sha256:
        raise ValueError("source authority tree differs from external pin")
    if source_tree_sha256(source_root) != expected_source_tree_sha256:
        raise ValueError("source tree bytes differ from external pin")
    if _pointmap_inventory_sha256(source_root, source_names) != source.get(
        "pointmap_inventory_sha256"
    ):
        raise ValueError("source point-map inventory differs from authority")
    if (
        v2_metadata.get("disjoint_upstream_authority_file_sha256")
        != file_sha256(authority_path)
        or v2_metadata.get("disjoint_upstream_authority_content_sha256")
        != authority_claimed
        or v2_metadata.get("source_tree_sha256") != expected_source_tree_sha256
    ):
        raise ValueError("upstream v2 and source authority lineage differ")
    if v1_metadata.get("disjoint_upstream_authority_content_sha256") != authority_claimed:
        raise ValueError("optimizer v1 and source authority lineage differ")

    chart_names = v2_base["chart_names"].astype(str).tolist()
    selected_pointmaps = _selected_pointmap_inventory(source_root, chart_names)
    if v1_metadata.get("pointmap_inventory") != selected_pointmaps:
        raise ValueError("optimizer v1 selected point maps differ from source bytes")
    cameras_path = source_root / "cameras.json"
    if (
        file_sha256(cameras_path) != source.get("cameras_file_sha256")
        or v1_metadata.get("cameras_file_sha256") != file_sha256(cameras_path)
    ):
        raise ValueError("source cameras differ across authority and optimizer v1")
    cameras = json.loads(cameras_path.read_text())
    camera_names = [Path(value).name for value in cameras.get("filepaths", [])]
    if camera_names != source_names or len(camera_names) != len(set(camera_names)):
        raise ValueError("source cameras ordered inventory differs from authority")
    if v2_metadata.get("mapping_source_ordered_names_sha256") != canonical_json_sha256(
        source_names
    ):
        raise ValueError("upstream v2 mapping-source order differs from authority")
    camera_row = {name: row for row, name in enumerate(camera_names)}
    if any(name not in camera_row for name in chart_names):
        raise ValueError("selected chart is absent from source cameras")
    camera_poses = np.asarray(cameras.get("cams2world"), np.float64)
    if camera_poses.shape != (len(camera_names), 4, 4):
        raise ValueError("source camera pose array is invalid")
    camera_centers = np.asarray(
        [camera_poses[camera_row[name], :3, 3] for name in chart_names],
        np.float64,
    )

    valid = np.asarray(v2_base["valid"], bool)
    chart_count, height, width = valid.shape
    reference_points = np.stack(
        [
            _load_reference_points(
                source_root / "pointmaps" / f"{Path(name).stem}.json",
                output_height=height,
                output_width=width,
            )
            for name in chart_names
        ]
    )
    final_faces: dict[int, np.ndarray] = {}
    metrics: dict[str, object] = {}
    reference_safe_masks: dict[str, np.ndarray] = {}
    parent_masks: dict[str, np.ndarray] = {}
    for stride in TOPOLOGY_STRIDES:
        parent = np.asarray(v2_base[f"face_valid_stride{stride}"], bool)
        safe, stride_metrics = reference_edge_safe_face_domain(
            parent,
            valid,
            reference_points,
            camera_centers,
            stride=stride,
        )
        if np.any(safe & ~parent):
            raise AssertionError("v3 face domain is not a v2 subset")
        final_faces[stride] = safe
        reference_safe_masks[f"face_valid_stride{stride}"] = safe
        parent_masks[f"face_valid_stride{stride}"] = parent
        metrics[f"stride{stride}"] = stride_metrics

    base_arrays = {
        "chart_names": np.asarray(v2_base["chart_names"]),
        "valid": np.asarray(v2_base["valid"]),
        **{
            f"face_valid_stride{stride}": final_faces[stride]
            for stride in TOPOLOGY_STRIDES
        },
    }
    topology_arrays = build_exact_topology_arrays(valid, final_faces)
    validate_exact_topology_arrays(base_arrays, topology_arrays)
    topology_inventory = exact_topology_inventory(
        topology_arrays,
        base_arrays["chart_names"],
    )
    arrays = {**base_arrays, **topology_arrays}
    safety_config = source_reference_edge_safety_config()
    metadata = dict(v2_metadata)
    metadata.pop("content_sha256", None)
    metadata.update(
        {
            "artifact_type": SCHEMA,
            "upstream_exact_topology_v2_file_sha256": file_sha256(
                upstream_v2_path
            ),
            "upstream_exact_topology_v2_content_sha256": (
                expected_upstream_v2_content_sha256
            ),
            "upstream_exact_topology_v2_arrays_sha256": v2_metadata[
                "arrays_sha256"
            ],
            "upstream_exact_topology_v2_exact_topology_arrays_sha256": (
                v2_metadata["exact_topology_arrays_sha256"]
            ),
            "upstream_optimizer_comparison_domain_v1_file_sha256": file_sha256(
                optimizer_v1_path
            ),
            "upstream_optimizer_comparison_domain_v1_content_sha256": (
                expected_optimizer_v1_content_sha256
            ),
            "upstream_optimizer_comparison_domain_v1_arrays_sha256": v1_metadata[
                "arrays_sha256"
            ],
            "source_reference_root": str(source_root),
            "source_reference_cameras_file_sha256": file_sha256(cameras_path),
            "source_reference_selected_pointmap_inventory": selected_pointmaps,
            "source_reference_selected_pointmap_inventory_sha256": (
                canonical_json_sha256(selected_pointmaps)
            ),
            "source_reference_full_pointmap_inventory_sha256": source[
                "pointmap_inventory_sha256"
            ],
            "source_reference_edge_safety_config": safety_config,
            "source_reference_edge_safety_config_sha256": canonical_json_sha256(
                safety_config
            ),
            "source_reference_edge_safety_metrics": metrics,
            "parent_v2_face_valid_arrays_sha256": arrays_sha256(parent_masks),
            "source_reference_edge_safe_face_masks_sha256": arrays_sha256(
                reference_safe_masks
            ),
            "final_reference_safe_face_masks_sha256": arrays_sha256(
                {
                    f"face_valid_stride{stride}": base_arrays[
                        f"face_valid_stride{stride}"
                    ]
                    for stride in TOPOLOGY_STRIDES
                }
            ),
            "base_domain_arrays_sha256": arrays_sha256(base_arrays),
            "exact_topology_arrays_sha256": arrays_sha256(topology_arrays),
            "arrays_sha256": arrays_sha256(arrays),
            "source_reference_edge_safety_replayed": True,
            "source_reference_edge_safe": True,
            "physical_face_safety_authority": True,
            "face_valid_v3_subset_of_face_valid_v2": True,
            "exact_topology_repacked_after_reference_safety": True,
            "valid_and_chart_names_byte_equal_upstream_v2": True,
            "orphan_sampled_vertices_present": False,
            "full_submap_gate_eligible": True,
            "full_submap_gate_primary_stride": 4,
            "exact_pixel_mask_frozen_for_both_arms": True,
            "exact_face_indices_frozen_for_both_arms": True,
            "uses_query_or_ground_truth": False,
            "topology": (
                "upstream v2 common optimizer pixels; face_valid is v2 DAV2/MoGe "
                "safe faces intersected with source-only MASt3R camera-range edge "
                "safety; packed vertices contain only surviving face corners"
            ),
            **topology_inventory,
        }
    )
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    return arrays, metadata


__all__ = [
    "ABSOLUTE_EDGE_THRESHOLD_M",
    "RELATIVE_EDGE_THRESHOLD",
    "SCHEMA",
    "reference_edge_safe_face_domain",
    "seal_reference_safe_comparison_domain",
    "source_reference_edge_safety_config",
]
