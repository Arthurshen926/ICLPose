"""Exact common pixel/topology contract for paired chart alignment arms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chart_submap_selection import (
    ChartSubmapPlan,
    load_model_neutral_alignment_selection,
)
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256


SCHEMA = "goal_maplet_chart_comparison_domain_v2"
TOPOLOGY_STRIDES = (4, 8)
FULL_GATE_ELIGIBLE_STRIDES = (4,)
BASE_ARRAY_NAMES = (
    "chart_names",
    "valid",
    "face_valid_stride4",
    "face_valid_stride8",
)
UPSTREAM_SCHEMA = "goal_maplet_chart_comparison_domain_v1"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def source_tree_sha256(root: Path) -> str:
    root = Path(root)
    rows: list[dict[str, object]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "sha256": file_sha256(path),
            }
        )
    return canonical_json_sha256(rows)


def topology_array_names(strides: tuple[int, ...] = TOPOLOGY_STRIDES) -> tuple[str, ...]:
    names: list[str] = []
    for stride in strides:
        names.extend(
            (
                f"sampled_vertex_offsets_stride{stride}",
                f"sampled_vertex_pixel_indices_stride{stride}",
                f"face_offsets_stride{stride}",
                f"faces_stride{stride}",
            )
        )
    return tuple(names)


def _one_stride_topology(
    valid: np.ndarray,
    face_valid: np.ndarray,
    *,
    stride: int,
) -> dict[str, np.ndarray]:
    valid = np.asarray(valid, bool)
    face_valid = np.asarray(face_valid, bool)
    if valid.ndim != 3 or stride < 1:
        raise ValueError("invalid common pixel domain or topology stride")
    chart_count, height, width = valid.shape
    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    expected_face_shape = (chart_count, max(0, len(ys) - 1), max(0, len(xs) - 1))
    if face_valid.shape != expected_face_shape:
        raise ValueError(f"stride-{stride} face domain has the wrong shape")
    sampled_pixel_grid = ys[:, None] * width + xs[None, :]
    vertex_offsets = [0]
    face_offsets = [0]
    sampled_pixels: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for chart in range(chart_count):
        sampled_valid = valid[chart][ys[:, None], xs[None, :]]
        used = np.zeros(sampled_valid.shape, bool)
        accepted_quads: list[tuple[int, int]] = []
        for row, column in np.argwhere(face_valid[chart]):
            corners = np.asarray(
                (
                    sampled_valid[row, column],
                    sampled_valid[row, column + 1],
                    sampled_valid[row + 1, column],
                    sampled_valid[row + 1, column + 1],
                )
            )
            if not corners.all():
                raise ValueError("frozen face references a pixel outside the common domain")
            used[row : row + 2, column : column + 2] = True
            accepted_quads.append((int(row), int(column)))
        local = np.full(used.shape, -1, np.int64)
        local[used] = np.arange(int(used.sum()), dtype=np.int64)
        local += np.where(local >= 0, vertex_offsets[-1], 0)
        sampled_pixels.append(sampled_pixel_grid[used])
        chart_faces: list[tuple[int, int, int]] = []
        for row, column in accepted_quads:
            ids = np.asarray(
                (
                    local[row, column],
                    local[row, column + 1],
                    local[row + 1, column],
                    local[row + 1, column + 1],
                ),
                np.int64,
            )
            chart_faces.extend(((ids[0], ids[2], ids[1]), (ids[1], ids[2], ids[3])))
        face_array = np.asarray(chart_faces, np.int64).reshape(-1, 3)
        faces.append(face_array)
        vertex_offsets.append(vertex_offsets[-1] + int(used.sum()))
        face_offsets.append(face_offsets[-1] + len(face_array))
    return {
        f"sampled_vertex_offsets_stride{stride}": np.asarray(vertex_offsets, np.int64),
        f"sampled_vertex_pixel_indices_stride{stride}": (
            np.concatenate(sampled_pixels).astype(np.int64, copy=False)
            if sampled_pixels
            else np.zeros((0,), np.int64)
        ),
        f"face_offsets_stride{stride}": np.asarray(face_offsets, np.int64),
        f"faces_stride{stride}": (
            np.concatenate(faces).astype(np.int64, copy=False)
            if faces
            else np.zeros((0, 3), np.int64)
        ),
    }


def build_exact_topology_arrays(
    valid: np.ndarray,
    face_valid_by_stride: dict[int, np.ndarray],
    *,
    strides: tuple[int, ...] = TOPOLOGY_STRIDES,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for stride in strides:
        if stride not in face_valid_by_stride:
            raise ValueError(f"missing stride-{stride} frozen face domain")
        output.update(
            _one_stride_topology(valid, face_valid_by_stride[stride], stride=stride)
        )
    return output


def validate_exact_topology_arrays(
    base_arrays: dict[str, np.ndarray],
    topology_arrays: dict[str, np.ndarray],
    *,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    expected_names = set(topology_array_names())
    if set(topology_arrays) != expected_names:
        raise ValueError("comparison domain exact-topology inventory differs")
    observed = build_exact_topology_arrays(
        np.asarray(base_arrays["valid"], bool),
        {
            4: np.asarray(base_arrays["face_valid_stride4"], bool),
            8: np.asarray(base_arrays["face_valid_stride8"], bool),
        },
    )
    for name in topology_array_names():
        if not np.array_equal(np.asarray(topology_arrays[name]), observed[name]):
            raise ValueError(f"comparison domain {name} does not replay pixel/face masks")
    if expected_sha256 is not None and arrays_sha256(topology_arrays) != expected_sha256:
        raise ValueError("comparison domain exact-topology hash differs")
    return observed


def exact_topology_inventory(
    topology_arrays: dict[str, np.ndarray],
    chart_names: np.ndarray,
    *,
    eligible_strides: tuple[int, ...] = FULL_GATE_ELIGIBLE_STRIDES,
) -> dict[str, object]:
    """Summarize every stride while enforcing only pre-frozen gate strides."""

    names = np.asarray(chart_names).astype(str).tolist()
    if not names or len(set(names)) != len(names):
        raise ValueError("exact topology chart inventory is empty or duplicate")
    if not set(eligible_strides).issubset(TOPOLOGY_STRIDES):
        raise ValueError("full-gate eligible stride is outside the topology inventory")
    quad_counts: dict[str, list[int]] = {}
    triangle_counts: dict[str, list[int]] = {}
    vertex_counts: dict[str, list[int]] = {}
    empty_names: dict[str, list[str]] = {}
    for stride in TOPOLOGY_STRIDES:
        face_offsets = np.asarray(
            topology_arrays[f"face_offsets_stride{stride}"], np.int64
        )
        vertex_offsets = np.asarray(
            topology_arrays[f"sampled_vertex_offsets_stride{stride}"], np.int64
        )
        if face_offsets.shape != (len(names) + 1,) or vertex_offsets.shape != (
            len(names) + 1,
        ):
            raise ValueError(f"stride-{stride} offsets differ from chart inventory")
        triangles = np.diff(face_offsets)
        vertices = np.diff(vertex_offsets)
        if np.any(triangles < 0) or np.any(vertices < 0) or np.any(triangles % 2):
            raise ValueError(f"stride-{stride} topology offsets are invalid")
        quads = triangles // 2
        key = f"stride{stride}"
        quad_counts[key] = quads.astype(int).tolist()
        triangle_counts[key] = triangles.astype(int).tolist()
        vertex_counts[key] = vertices.astype(int).tolist()
        empty_names[key] = [
            names[row] for row in np.flatnonzero(quads == 0).tolist()
        ]
        if stride in eligible_strides and np.any(quads == 0):
            raise ValueError(
                f"stride-{stride} exact face inventory is empty for a gate chart"
            )
    return {
        "full_submap_gate_eligible_strides": list(eligible_strides),
        "required_nonempty_face_inventory_strides": list(eligible_strides),
        "face_quad_count_per_chart_by_stride": quad_counts,
        "triangle_count_per_chart_by_stride": triangle_counts,
        "packed_vertex_count_per_chart_by_stride": vertex_counts,
        "empty_face_chart_names_by_stride": empty_names,
        "noneligible_stride_empty_inventory_permitted": True,
    }


def seal_exact_comparison_domain(
    upstream_domain_path: Path,
    *,
    expected_upstream_content_sha256: str,
    plan_path: Path,
    expected_plan_content_sha256: str,
    authority_path: Path,
    expected_authority_content_sha256: str,
    source_root: Path,
    expected_source_tree_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Upgrade a paired v1 domain into an explicit, exact-topology v2 seal."""

    pinned_hashes = (
        expected_upstream_content_sha256,
        expected_plan_content_sha256,
        expected_authority_content_sha256,
        expected_source_tree_sha256,
    )
    if any(not _is_sha256(value) for value in pinned_hashes):
        raise ValueError("exact comparison domain requires four explicit SHA-256 pins")
    upstream_domain_path = Path(upstream_domain_path)
    plan_path = Path(plan_path)
    authority_path = Path(authority_path)
    source_root = Path(source_root).resolve()
    with np.load(upstream_domain_path, allow_pickle=False) as data:
        base_arrays = {
            name: np.asarray(data[name])
            for name in BASE_ARRAY_NAMES
        }
        upstream_metadata = json.loads(str(data["metadata_json"].item()))
    claimed_upstream = upstream_metadata.pop("content_sha256", None)
    if claimed_upstream != canonical_json_sha256(upstream_metadata):
        raise ValueError("upstream comparison domain content hash differs")
    upstream_metadata["content_sha256"] = claimed_upstream
    if claimed_upstream != expected_upstream_content_sha256:
        raise ValueError("upstream comparison domain differs from experiment authority")
    if upstream_metadata.get("artifact_type") != UPSTREAM_SCHEMA:
        raise ValueError("exact topology sealer requires the paired v1 comparison domain")
    if upstream_metadata.get("arrays_sha256") != arrays_sha256(base_arrays):
        raise ValueError("upstream comparison domain arrays differ from metadata")
    if upstream_metadata.get("uses_query_or_ground_truth") is not False:
        raise ValueError("upstream comparison domain consumed query or ground truth")

    authority = json.loads(authority_path.read_text())
    claimed_authority = authority.pop("content_sha256", None)
    if claimed_authority != canonical_json_sha256(authority):
        raise ValueError("disjoint upstream authority content hash differs")
    authority["content_sha256"] = claimed_authority
    if claimed_authority != expected_authority_content_sha256:
        raise ValueError("disjoint authority differs from experiment authority")
    required_true = (
        "strict_disjoint_upstream",
        "source_held_image_disjoint",
        "source_held_route_disjoint",
        "physical_source_held_input_roots_disjoint",
    )
    if authority.get("artifact_type") != "goal_maplet_disjoint_chart_upstream_authority_v2":
        raise ValueError("exact comparison domain requires physically isolated v2 authority")
    if any(authority.get(key) is not True for key in required_true):
        raise ValueError("authority does not certify strict physical separation")
    if authority.get("uses_query_or_ground_truth") is not False:
        raise ValueError("authority consumed query or ground truth")
    if authority.get("forbidden_routes_opened") is not False:
        raise ValueError("authority opened a forbidden route")
    source = authority.get("source")
    if not isinstance(source, dict) or Path(str(source.get("root"))).resolve() != source_root:
        raise ValueError("source root differs from v2 authority")
    source_ordered_names = source.get("ordered_names")
    if (
        not isinstance(source_ordered_names, list)
        or not source_ordered_names
        or any(not isinstance(name, str) or not name for name in source_ordered_names)
        or len(set(source_ordered_names)) != len(source_ordered_names)
    ):
        raise ValueError("v2 authority lacks a unique mapping-source ordered inventory")
    mapping_source_ordered_names_sha256 = canonical_json_sha256(
        source_ordered_names
    )
    if source.get("tree_sha256") != expected_source_tree_sha256:
        raise ValueError("authority source tree differs from experiment authority")
    if source_tree_sha256(source_root) != expected_source_tree_sha256:
        raise ValueError("source tree bytes differ from experiment authority")

    plan = ChartSubmapPlan.load_npz(plan_path)
    selection = load_model_neutral_alignment_selection(
        plan_path,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )
    if len(selection.operational_submaps) != 1:
        raise ValueError("paired comparison requires exactly one operational submap")
    if plan.chart_names.astype(str).tolist() != source_ordered_names:
        raise ValueError("frozen plan source inventory differs from v2 authority")
    if (
        plan.metadata.get("source_ordered_names_sha256")
        != mapping_source_ordered_names_sha256
    ):
        raise ValueError("frozen plan source-order hash differs from v2 authority")
    names = list(selection.ordered_names)
    if np.asarray(base_arrays["chart_names"]).astype(str).tolist() != names:
        raise ValueError("comparison domain chart order differs from exact plan selection")
    plan_lineage = plan.metadata.get("lineage")
    if not isinstance(plan_lineage, dict):
        raise ValueError("frozen plan lacks source lineage")
    if plan_lineage.get("disjoint_authority_content_sha256") != claimed_authority:
        raise ValueError("frozen plan and v2 authority differ")
    if plan_lineage.get("source_tree_sha256") != expected_source_tree_sha256:
        raise ValueError("frozen plan and source tree differ")
    required_upstream = {
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": claimed_authority,
        "source_tree_sha256": expected_source_tree_sha256,
        "frozen_submap_plan_file_sha256": file_sha256(plan_path),
        "frozen_submap_plan_content_sha256": expected_plan_content_sha256,
        "selected_chart_names_in_order_sha256": plan.metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "alignment_runner_contract": plan.metadata["alignment_runner_contract"],
    }
    for key, expected in required_upstream.items():
        if upstream_metadata.get(key) != expected:
            raise ValueError(f"upstream comparison domain {key} differs")
    upstream_mapping_order_hash = upstream_metadata.get(
        "mapping_source_ordered_names_sha256"
    )
    if (
        upstream_mapping_order_hash is not None
        and upstream_mapping_order_hash != mapping_source_ordered_names_sha256
    ):
        raise ValueError("upstream mapping-source order differs from v2 authority")

    valid = np.asarray(base_arrays["valid"], bool)
    if valid.ndim != 3 or len(valid) != len(names):
        raise ValueError("comparison domain pixel mask shape differs from selected charts")
    if np.any(valid.reshape(len(valid), -1).sum(1) < 1024):
        raise ValueError("comparison domain has insufficient common pixels for a chart")
    topology_arrays = build_exact_topology_arrays(
        valid,
        {
            4: np.asarray(base_arrays["face_valid_stride4"], bool),
            8: np.asarray(base_arrays["face_valid_stride8"], bool),
        },
    )
    validate_exact_topology_arrays(base_arrays, topology_arrays)
    topology_inventory = exact_topology_inventory(
        topology_arrays,
        base_arrays["chart_names"],
    )
    arrays = {**base_arrays, **topology_arrays}
    metadata = dict(upstream_metadata)
    metadata.pop("content_sha256", None)
    metadata.update(
        {
            "artifact_type": SCHEMA,
            "upstream_comparison_domain_artifact_type": UPSTREAM_SCHEMA,
            "upstream_comparison_domain_file_sha256": file_sha256(
                upstream_domain_path
            ),
            "upstream_comparison_domain_content_sha256": claimed_upstream,
            "base_domain_arrays_sha256": arrays_sha256(base_arrays),
            "exact_topology_arrays_sha256": arrays_sha256(topology_arrays),
            "arrays_sha256": arrays_sha256(arrays),
            "exact_pixel_mask_frozen_for_both_arms": True,
            "exact_face_indices_frozen_for_both_arms": True,
            "orphan_sampled_vertices_present": False,
            "exact_topology_strides": list(TOPOLOGY_STRIDES),
            "full_submap_gate_eligible": True,
            "full_submap_gate_primary_stride": 4,
            "mapping_source_ordered_names_sha256": (
                mapping_source_ordered_names_sha256
            ),
            "mapping_source_ordered_names_binding": (
                "replayed_from_v2_authority_and_exact_plan_source_inventory"
            ),
            "topology": (
                "only sampled pixels referenced by an accepted face are retained; packed "
                "vertices are ordered by chart then raster pixel; every accepted "
                "quad emits fixed triangles (top-left,bottom-left,top-right) and "
                "(top-right,bottom-left,bottom-right)"
            ),
            **topology_inventory,
        }
    )
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    return arrays, metadata


__all__ = [
    "BASE_ARRAY_NAMES",
    "FULL_GATE_ELIGIBLE_STRIDES",
    "SCHEMA",
    "TOPOLOGY_STRIDES",
    "build_exact_topology_arrays",
    "exact_topology_inventory",
    "seal_exact_comparison_domain",
    "source_tree_sha256",
    "topology_array_names",
    "validate_exact_topology_arrays",
]
