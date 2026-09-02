"""Strict paired-comparison stride-2 topology densification diagnostic.

This module deliberately does not claim a model-neutral map topology.  It
starts from the corrected exact-21 v1 common-valid mask, applies the same
unit-pixel edge rule independently to frozen DAV2, MoGe3, and source-only
MASt3R geometry, and repacks only vertices used by surviving faces.

Before a stride-2 topology can be emitted, the generic constructor must
reconstruct the corrected v3 stride-4 face mask and all four packed topology
arrays bit-for-bit.  The resulting stride-2 topology may then be consumed by
the source-ray projective seam implementation with unchanged thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .chart_comparison_domain import (
    BASE_ARRAY_NAMES,
    _one_stride_topology,
    source_tree_sha256,
    topology_array_names,
    validate_exact_topology_arrays,
)
from .chart_comparison_reference_safe_domain import (
    SCHEMA as REFERENCE_SAFE_V3_SCHEMA,
)
from .chart_submap_selection import ChartSubmapPlan
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256
from .projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
    ProjectiveSeamConfig,
    _common_valid_normal_stencil,
    _dense_world_normals,
    _load_source_reference_dense,
    freeze_projective_exact_face_correspondences,
)


SCHEMA = "goal_maplet_paired_stride2_topology_diagnostic_v1"
REPORT_SCHEMA = "goal_maplet_paired_stride2_projective_diagnostic_report_v1"
V1_SCHEMA = "goal_maplet_chart_comparison_domain_v1"
DISJOINT_SCHEMA = "goal_maplet_disjoint_chart_upstream_authority_v2"
GEOMETRY_NAMES = (
    "DAV2_clean_v2",
    "MoGe3",
    "source_MASt3R_reference",
)
ABSOLUTE_EDGE_THRESHOLD_M = 0.5
RELATIVE_EDGE_THRESHOLD = 0.05
FORMAL_COMPONENT_TARGET = 16


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _replay_metadata(metadata: Mapping[str, object], label: str) -> str:
    replay = dict(metadata)
    claimed = replay.pop("content_sha256", None)
    if not _is_sha256(claimed) or claimed != canonical_json_sha256(replay):
        raise ValueError(f"{label} metadata content hash differs")
    return str(claimed)


def _load_npz_arrays_and_metadata(
    path: Path, names: tuple[str, ...] | list[str]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        if any(name not in data.files for name in names):
            raise ValueError("paired stride-2 input array inventory is incomplete")
        arrays = {name: np.asarray(data[name]) for name in names}
        metadata = json.loads(str(data["metadata_json"].item()))
    return arrays, metadata


def unit_edge_safe_face_masks(
    valid: np.ndarray,
    point_sets: Mapping[str, np.ndarray],
    camera_centers: np.ndarray,
    *,
    stride: int,
    absolute_edge_threshold_m: float = ABSOLUTE_EDGE_THRESHOLD_M,
    relative_edge_threshold: float = RELATIVE_EDGE_THRESHOLD,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the strict intersection and independent geometry masks.

    Every face requires the inclusive pixel patch to be common-valid.  For
    each frozen geometry separately, every horizontal and vertical unit-pixel
    edge in that patch must be no longer than
    ``max(0.5 m, 0.05 * median camera-centred range)``.
    """

    valid = np.asarray(valid, bool)
    camera_centers = np.asarray(camera_centers, np.float64)
    if valid.ndim != 3 or stride < 1:
        raise ValueError("paired stride-2 common domain or stride is invalid")
    chart_count, height, width = valid.shape
    if camera_centers.shape != (chart_count, 3):
        raise ValueError("paired stride-2 camera centres differ")
    if tuple(point_sets) != GEOMETRY_NAMES:
        raise ValueError("paired stride-2 requires the three frozen geometries")
    points = {
        name: np.asarray(point_sets[name], np.float64) for name in GEOMETRY_NAMES
    }
    expected_shape = (chart_count, height, width, 3)
    if any(value.shape != expected_shape for value in points.values()):
        raise ValueError("paired stride-2 frozen geometry shape differs")
    if absolute_edge_threshold_m <= 0 or not 0 < relative_edge_threshold < 1:
        raise ValueError("paired stride-2 edge threshold is invalid")

    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    shape = (chart_count, max(0, len(ys) - 1), max(0, len(xs) - 1))
    per_geometry = {name: np.zeros(shape, bool) for name in GEOMETRY_NAMES}
    for chart in range(chart_count):
        center = camera_centers[chart]
        for oy, y in enumerate(ys[:-1]):
            y1 = int(ys[oy + 1])
            for ox, x in enumerate(xs[:-1]):
                x1 = int(xs[ox + 1])
                if not valid[chart, y : y1 + 1, x : x1 + 1].all():
                    continue
                for name in GEOMETRY_NAMES:
                    patch = points[name][chart, y : y1 + 1, x : x1 + 1]
                    if not np.isfinite(patch).all():
                        continue
                    horizontal = np.linalg.norm(
                        patch[:, 1:] - patch[:, :-1], axis=2
                    )
                    vertical = np.linalg.norm(
                        patch[1:] - patch[:-1], axis=2
                    )
                    range_m = float(np.median(np.linalg.norm(patch - center, axis=2)))
                    threshold = max(
                        absolute_edge_threshold_m,
                        relative_edge_threshold * range_m,
                    )
                    maximum_edge = max(
                        float(np.max(horizontal, initial=0.0)),
                        float(np.max(vertical, initial=0.0)),
                    )
                    per_geometry[name][chart, oy, ox] = maximum_edge <= threshold
    strict = np.logical_and.reduce([per_geometry[name] for name in GEOMETRY_NAMES])
    return strict, per_geometry


def _control_topology_arrays(
    valid: np.ndarray, face_valid: np.ndarray
) -> dict[str, np.ndarray]:
    raw = _one_stride_topology(valid, face_valid, stride=4)
    return {f"{name}_control": value for name, value in raw.items()}


def require_stride4_bit_parity(
    valid: np.ndarray,
    face_valid_stride4: np.ndarray,
    corrected_v3_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Fail closed unless generic reconstruction equals corrected v3."""

    observed = _one_stride_topology(valid, face_valid_stride4, stride=4)
    if not np.array_equal(
        np.asarray(face_valid_stride4, bool),
        np.asarray(corrected_v3_arrays["face_valid_stride4"], bool),
    ):
        raise ValueError("generic stride-4 face mask differs from corrected v3")
    for name, value in observed.items():
        if not np.array_equal(value, np.asarray(corrected_v3_arrays[name])):
            raise ValueError(f"generic stride-4 {name} differs from corrected v3")
    return {f"{name}_control": value for name, value in observed.items()}


@dataclass(frozen=True)
class PairedStride2TopologyDiagnostic:
    chart_names: np.ndarray
    valid: np.ndarray
    face_valid_stride4_control: np.ndarray
    sampled_vertex_offsets_stride4_control: np.ndarray
    sampled_vertex_pixel_indices_stride4_control: np.ndarray
    face_offsets_stride4_control: np.ndarray
    faces_stride4_control: np.ndarray
    face_valid_stride2: np.ndarray
    sampled_vertex_offsets_stride2: np.ndarray
    sampled_vertex_pixel_indices_stride2: np.ndarray
    face_offsets_stride2: np.ndarray
    faces_stride2: np.ndarray
    metadata: dict[str, object]

    @staticmethod
    def array_names() -> tuple[str, ...]:
        return (
            "chart_names",
            "valid",
            "face_valid_stride4_control",
            "sampled_vertex_offsets_stride4_control",
            "sampled_vertex_pixel_indices_stride4_control",
            "face_offsets_stride4_control",
            "faces_stride4_control",
            "face_valid_stride2",
            "sampled_vertex_offsets_stride2",
            "sampled_vertex_pixel_indices_stride2",
            "face_offsets_stride2",
            "faces_stride2",
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.array_names()}

    def validated(self) -> "PairedStride2TopologyDiagnostic":
        arrays = self.arrays()
        names = arrays["chart_names"].astype(str)
        valid = arrays["valid"].astype(bool)
        if len(names) < 2 or len(set(names.tolist())) != len(names):
            raise ValueError("paired stride-2 chart inventory is invalid")
        if valid.ndim != 3 or valid.shape[0] != len(names):
            raise ValueError("paired stride-2 common-valid domain is invalid")
        replay4 = _control_topology_arrays(
            valid, arrays["face_valid_stride4_control"]
        )
        replay2 = _one_stride_topology(
            valid, arrays["face_valid_stride2"], stride=2
        )
        for name, value in replay4.items():
            if not np.array_equal(value, arrays[name]):
                raise ValueError(f"paired stride-2 {name} does not replay")
        for name, value in replay2.items():
            if not np.array_equal(value, arrays[name]):
                raise ValueError(f"paired stride-2 {name} does not replay")
        control_hash_input = {
            "face_valid_stride4": arrays["face_valid_stride4_control"],
            "sampled_vertex_offsets_stride4": arrays[
                "sampled_vertex_offsets_stride4_control"
            ],
            "sampled_vertex_pixel_indices_stride4": arrays[
                "sampled_vertex_pixel_indices_stride4_control"
            ],
            "face_offsets_stride4": arrays["face_offsets_stride4_control"],
            "faces_stride4": arrays["faces_stride4_control"],
        }
        control_hash = arrays_sha256(control_hash_input)
        if (
            self.metadata.get("stride4_control_arrays_sha256") != control_hash
            or self.metadata.get("corrected_v3_stride4_arrays_sha256")
            != control_hash
        ):
            raise ValueError("paired stride-2 stride-4 parity hash differs")
        vertex_count = len(arrays["sampled_vertex_pixel_indices_stride2"])
        used = np.unique(arrays["faces_stride2"].astype(np.int64))
        if not np.array_equal(used, np.arange(vertex_count, dtype=np.int64)):
            raise ValueError("paired stride-2 topology contains an orphan vertex")
        offsets = arrays["sampled_vertex_offsets_stride2"].astype(np.int64)
        face_offsets = arrays["face_offsets_stride2"].astype(np.int64)
        if (
            offsets.shape != (len(names) + 1,)
            or face_offsets.shape != (len(names) + 1,)
            or np.any(np.diff(offsets) <= 0)
            or np.any(np.diff(face_offsets) <= 0)
        ):
            raise ValueError("paired stride-2 has an empty chart topology")
        eligible4 = _packed_source_eligible_count(
            valid,
            arrays["sampled_vertex_offsets_stride4_control"],
            arrays["sampled_vertex_pixel_indices_stride4_control"],
        )
        eligible2 = _packed_source_eligible_count(
            valid,
            arrays["sampled_vertex_offsets_stride2"],
            arrays["sampled_vertex_pixel_indices_stride2"],
        )
        replay_counts = {
            "stride2_face_quad_count": int(
                arrays["face_valid_stride2"].sum()
            ),
            "stride2_triangle_count": int(len(arrays["faces_stride2"])),
            "stride2_packed_vertex_count": vertex_count,
            "stride2_face_quad_count_per_chart": arrays[
                "face_valid_stride2"
            ].sum((1, 2)).astype(int).tolist(),
            "stride2_packed_vertex_count_per_chart": np.diff(
                arrays["sampled_vertex_offsets_stride2"]
            ).astype(int).tolist(),
            "source_eligible_count_stride4_control": eligible4,
            "source_eligible_count_stride2": eligible2,
        }
        for key, expected in replay_counts.items():
            if self.metadata.get(key) != expected:
                raise ValueError(f"paired stride-2 metadata {key} does not replay")
        ratio = float(eligible2 / eligible4)
        if not np.isclose(
            float(
                self.metadata.get(
                    "source_eligible_count_stride2_over_stride4_ratio", np.nan
                )
            ),
            ratio,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("paired stride-2 eligible-count ratio does not replay")
        required = {
            "artifact_type": SCHEMA,
            "paired_comparison_topology": True,
            "model_neutral_map_topology_claimed": False,
            "uses_query_or_ground_truth": False,
            "held_geometry_consumed": False,
            "aligned_arm_geometry_consumed": False,
            "stride4_control_bit_equal_corrected_v3": True,
            "stride2_repacked_without_orphans": True,
            "source_eligible_definition_stride_invariant": True,
            "sampled_source_inventory_stride_invariant": False,
            "support_fractions_only_comparable_within_same_stride": True,
        }
        for key, expected in required.items():
            if self.metadata.get(key) != expected:
                raise ValueError(f"paired stride-2 metadata {key} differs")
        if self.metadata.get("geometry_names") != list(GEOMETRY_NAMES):
            raise ValueError("paired stride-2 geometry inventory differs")
        if self.metadata.get("edge_safety") != {
            "absolute_edge_threshold_m": ABSOLUTE_EDGE_THRESHOLD_M,
            "relative_edge_threshold": RELATIVE_EDGE_THRESHOLD,
            "threshold_formula": "max(0.5m,0.05*median_camera_centered_range)",
            "edge_inventory": "all_unit_pixel_edges_in_inclusive_stride_patch",
        }:
            raise ValueError("paired stride-2 edge safety contract differs")
        collapse = self.metadata.get("stride2_triangle_area2_by_geometry")
        if not isinstance(collapse, dict) or tuple(collapse) != GEOMETRY_NAMES:
            raise ValueError("paired stride-2 triangle-collapse inventory differs")
        for name in GEOMETRY_NAMES:
            row = collapse[name]
            if (
                not isinstance(row, dict)
                or row.get("triangle_count") != len(arrays["faces_stride2"])
                or row.get("nonfinite_count") != 0
                or row.get("at_or_below_1e-10_count") != 0
                or not np.isfinite(float(row.get("minimum_area2_m2", np.nan)))
                or float(row["minimum_area2_m2"]) <= 1e-10
            ):
                raise ValueError(f"paired stride-2 {name} triangle collapse differs")
        return self

    def save_npz(self, path: Path) -> dict[str, object]:
        arrays = self.validated().arrays()
        metadata = dict(self.metadata)
        metadata["arrays_sha256"] = arrays_sha256(arrays)
        metadata["content_sha256"] = canonical_json_sha256(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".temporary.npz")
        np.savez_compressed(
            temporary,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        temporary.replace(path)
        return metadata

    @classmethod
    def load_npz(cls, path: Path) -> "PairedStride2TopologyDiagnostic":
        arrays, metadata = _load_npz_arrays_and_metadata(path, cls.array_names())
        if metadata.get("arrays_sha256") != arrays_sha256(arrays):
            raise ValueError("paired stride-2 arrays differ from lineage")
        _replay_metadata(metadata, "paired stride-2 topology")
        return cls(metadata=metadata, **arrays).validated()


def _load_initializer_manifest(
    root: Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
    label: str,
) -> dict[str, object]:
    path = Path(root) / "manifest.json"
    if file_sha256(path) != expected_file_sha256:
        raise ValueError(f"{label} manifest file differs from corrected v1")
    payload = json.loads(path.read_text())
    if _replay_metadata(payload, f"{label} manifest") != expected_content_sha256:
        raise ValueError(f"{label} manifest content differs from corrected v1")
    if payload.get("uses_query_or_ground_truth") is not False:
        raise ValueError(f"{label} manifest consumed query or ground truth")
    return payload


def _packed_source_eligible_count(
    valid: np.ndarray,
    vertex_offsets: np.ndarray,
    sampled_pixel_indices: np.ndarray,
) -> int:
    valid = np.asarray(valid, bool)
    offsets = np.asarray(vertex_offsets, np.int64)
    pixels = np.asarray(sampled_pixel_indices, np.int64)
    count = 0
    for chart in range(len(valid)):
        lo, hi = map(int, offsets[chart : chart + 2])
        stencil = _common_valid_normal_stencil(valid[chart])
        count += int(np.sum(stencil.ravel()[pixels[lo:hi]]))
    return count


def _triangle_area2_stats(
    dense_points: np.ndarray,
    vertex_offsets: np.ndarray,
    sampled_pixel_indices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, object]:
    dense_points = np.asarray(dense_points, np.float64)
    offsets = np.asarray(vertex_offsets, np.int64)
    pixels = np.asarray(sampled_pixel_indices, np.int64)
    faces = np.asarray(faces, np.int64)
    vertices = np.concatenate(
        [
            dense_points[chart].reshape(-1, 3)[
                pixels[offsets[chart] : offsets[chart + 1]]
            ]
            for chart in range(len(dense_points))
        ]
    )
    triangle = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]),
        axis=1,
    )
    finite = np.isfinite(area2)
    stats = {
        "triangle_count": len(area2),
        "nonfinite_count": int(np.sum(~finite)),
        "at_or_below_1e-10_count": int(np.sum(finite & (area2 <= 1e-10))),
        "minimum_area2_m2": float(np.min(area2[finite])) if finite.any() else float("nan"),
        "maximum_area2_m2": float(np.max(area2[finite])) if finite.any() else float("nan"),
    }
    if stats["nonfinite_count"] or stats["at_or_below_1e-10_count"]:
        raise ValueError("paired stride-2 topology has a collapsed geometry triangle")
    return stats


def _load_dav2(
    root: Path,
    names: list[str],
    expected_hashes: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    points, valid = [], []
    for name in names:
        path = Path(root) / f"{name}.npz"
        if file_sha256(path) != expected_hashes.get(name):
            raise ValueError(f"DAV2 initializer file differs for {name}")
        with np.load(path, allow_pickle=False) as data:
            world = np.asarray(data["points_world"], np.float64)
            good = np.asarray(data["valid"], bool)
            metadata = json.loads(str(data["metadata_json"].item()))
        if (
            metadata.get("artifact_type") != "goal_maplet_dav2_chart_initializer_v1"
            or metadata.get("source_name") != name
        ):
            raise ValueError("DAV2 initializer binding differs")
        _replay_metadata(metadata, f"DAV2 initializer {name}")
        points.append(world)
        valid.append(good & np.isfinite(world).all(2))
    return np.stack(points), np.stack(valid)


def _load_moge(
    root: Path,
    names: list[str],
    expected_hashes: Mapping[str, object],
    cameras: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    camera_rows = {
        Path(value).name: row
        for row, value in enumerate(cameras.get("filepaths", []))
    }
    points, valid = [], []
    for name in names:
        path = Path(root) / f"{name}.npz"
        if file_sha256(path) != expected_hashes.get(name):
            raise ValueError(f"MoGe3 initializer file differs for {name}")
        with np.load(path, allow_pickle=False) as data:
            camera_points = np.asarray(data["points_camera"], np.float64)
            good = np.asarray(data["valid"], bool)
            metadata = json.loads(str(data["metadata_json"].item()))
        row = camera_rows[name]
        if (
            metadata.get("artifact_type") != "goal_maplet_moge3_chart_initializer_v2"
            or metadata.get("source_name") != name
            or float(metadata.get("camera_focal_canvas_px"))
            != float(cameras["focals"][row])
        ):
            raise ValueError("MoGe3 initializer binding differs")
        _replay_metadata(metadata, f"MoGe3 initializer {name}")
        c2w = np.asarray(cameras["cams2world"][row], np.float64)
        world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        points.append(world)
        valid.append(good & np.isfinite(world).all(2))
    return np.stack(points), np.stack(valid)


def build_paired_stride2_topology_diagnostic(
    comparison_domain_v1_path: Path,
    *,
    expected_v1_content_sha256: str,
    comparison_domain_v3_path: Path,
    expected_v3_content_sha256: str,
    frozen_submap_plan_path: Path,
    expected_plan_content_sha256: str,
    disjoint_upstream_authority_path: Path,
    expected_disjoint_authority_content_sha256: str,
    source_root: Path,
    expected_source_tree_sha256: str,
    dav2_initializer_root: Path,
    moge3_initializer_root: Path,
) -> PairedStride2TopologyDiagnostic:
    """Build a hash-pinned stride-2 topology after stride-4 parity."""

    pins = (
        expected_v1_content_sha256,
        expected_v3_content_sha256,
        expected_plan_content_sha256,
        expected_disjoint_authority_content_sha256,
        expected_source_tree_sha256,
    )
    if any(not _is_sha256(value) for value in pins):
        raise ValueError("paired stride-2 builder requires five explicit SHA-256 pins")
    comparison_domain_v1_path = Path(comparison_domain_v1_path)
    comparison_domain_v3_path = Path(comparison_domain_v3_path)
    frozen_submap_plan_path = Path(frozen_submap_plan_path)
    disjoint_upstream_authority_path = Path(disjoint_upstream_authority_path)
    source_root = Path(source_root).resolve()
    dav2_initializer_root = Path(dav2_initializer_root).resolve()
    moge3_initializer_root = Path(moge3_initializer_root).resolve()

    v1_arrays, v1_metadata = _load_npz_arrays_and_metadata(
        comparison_domain_v1_path, list(BASE_ARRAY_NAMES)
    )
    if _replay_metadata(v1_metadata, "corrected v1") != expected_v1_content_sha256:
        raise ValueError("paired stride-2 corrected v1 differs from pin")
    if v1_metadata.get("artifact_type") != V1_SCHEMA:
        raise ValueError("paired stride-2 requires corrected optimizer v1")
    if v1_metadata.get("arrays_sha256") != arrays_sha256(v1_arrays):
        raise ValueError("paired stride-2 v1 arrays differ from lineage")

    v3_names = list(BASE_ARRAY_NAMES) + list(topology_array_names())
    v3_arrays, v3_metadata = _load_npz_arrays_and_metadata(
        comparison_domain_v3_path, v3_names
    )
    if _replay_metadata(v3_metadata, "corrected v3") != expected_v3_content_sha256:
        raise ValueError("paired stride-2 corrected v3 differs from pin")
    if v3_metadata.get("artifact_type") != REFERENCE_SAFE_V3_SCHEMA:
        raise ValueError("paired stride-2 requires corrected reference-safe v3")
    if v3_metadata.get("arrays_sha256") != arrays_sha256(v3_arrays):
        raise ValueError("paired stride-2 v3 arrays differ from lineage")
    validate_exact_topology_arrays(
        {name: v3_arrays[name] for name in BASE_ARRAY_NAMES},
        {name: v3_arrays[name] for name in topology_array_names()},
        expected_sha256=v3_metadata.get("exact_topology_arrays_sha256"),
    )
    if (
        v3_metadata.get("upstream_optimizer_comparison_domain_v1_content_sha256")
        != expected_v1_content_sha256
        or v3_metadata.get("frozen_submap_plan_content_sha256")
        != expected_plan_content_sha256
    ):
        raise ValueError("paired stride-2 corrected v1/v3/plan lineage differs")
    for name in ("chart_names", "valid"):
        if not np.array_equal(v1_arrays[name], v3_arrays[name]):
            raise ValueError(f"paired stride-2 corrected v1/v3 {name} differs")

    plan = ChartSubmapPlan.load_npz(frozen_submap_plan_path)
    if plan.metadata.get("content_sha256") != expected_plan_content_sha256:
        raise ValueError("paired stride-2 corrected plan differs from pin")
    names = v1_arrays["chart_names"].astype(str).tolist()
    if names != list(plan.selected_chart_names_in_order):
        raise ValueError("paired stride-2 chart order differs from corrected plan")
    if (
        v1_metadata.get("frozen_submap_plan_file_sha256")
        != file_sha256(frozen_submap_plan_path)
        or v1_metadata.get("frozen_submap_plan_content_sha256")
        != expected_plan_content_sha256
    ):
        raise ValueError("paired stride-2 v1 plan binding differs")

    disjoint = json.loads(disjoint_upstream_authority_path.read_text())
    if (
        _replay_metadata(disjoint, "disjoint authority")
        != expected_disjoint_authority_content_sha256
        or disjoint.get("artifact_type") != DISJOINT_SCHEMA
        or disjoint.get("physical_source_held_input_roots_disjoint") is not True
        or disjoint.get("uses_query_or_ground_truth") is not False
        or disjoint.get("forbidden_routes_opened") is not False
    ):
        raise ValueError("paired stride-2 disjoint authority differs")
    if Path(disjoint.get("source", {}).get("root", "")).resolve() != source_root:
        raise ValueError("paired stride-2 source root differs from authority")
    observed_source_tree = source_tree_sha256(source_root)
    if (
        observed_source_tree != expected_source_tree_sha256
        or disjoint.get("source", {}).get("tree_sha256") != observed_source_tree
        or v1_metadata.get("source_tree_sha256") != observed_source_tree
    ):
        raise ValueError("paired stride-2 source tree differs from pin")

    cameras_path = source_root / "cameras.json"
    cameras_hash = file_sha256(cameras_path)
    cameras = json.loads(cameras_path.read_text())
    if (
        cameras_hash != v1_metadata.get("cameras_file_sha256")
        or cameras_hash != v3_metadata.get("source_reference_cameras_file_sha256")
        or cameras_hash != disjoint.get("source", {}).get("cameras_file_sha256")
    ):
        raise ValueError("paired stride-2 source cameras differ from lineage")
    camera_names = [Path(value).name for value in cameras.get("filepaths", [])]
    camera_rows = {name: row for row, name in enumerate(camera_names)}
    if any(name not in camera_rows for name in names):
        raise ValueError("paired stride-2 selected chart lacks a camera")

    _load_initializer_manifest(
        dav2_initializer_root,
        expected_file_sha256=str(
            v1_metadata.get("dav2_initializer_manifest_file_sha256")
        ),
        expected_content_sha256=str(
            v1_metadata.get("dav2_initializer_manifest_content_sha256")
        ),
        label="DAV2",
    )
    _load_initializer_manifest(
        moge3_initializer_root,
        expected_file_sha256=str(
            v1_metadata.get("moge_initializer_manifest_file_sha256")
        ),
        expected_content_sha256=str(
            v1_metadata.get("moge_initializer_manifest_content_sha256")
        ),
        label="MoGe3",
    )
    dav2_points, dav2_valid = _load_dav2(
        dav2_initializer_root,
        names,
        v1_metadata.get("dav2_initializer_file_sha256", {}),
    )
    moge_points, moge_valid = _load_moge(
        moge3_initializer_root,
        names,
        v1_metadata.get("moge_initializer_file_sha256", {}),
        cameras,
    )
    height, width = v1_arrays["valid"].shape[1:]
    reference_points = []
    pointmap_hashes: dict[str, str] = {}
    for name in names:
        path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        pointmap_hashes[name] = file_sha256(path)
        if (
            pointmap_hashes[name]
            != v3_metadata.get("source_reference_selected_pointmap_inventory", {}).get(
                name
            )
            or pointmap_hashes[name]
            != v1_metadata.get("pointmap_inventory", {}).get(name)
        ):
            raise ValueError(f"paired stride-2 source pointmap differs for {name}")
        points, _ = _load_source_reference_dense(
            path, output_height=height, output_width=width
        )
        reference_points.append(points)
    reference_points_array = np.stack(reference_points)
    common_valid = np.asarray(v1_arrays["valid"], bool)
    # The denominator is inherited exactly from corrected v1; do not derive a
    # new common mask from the three geometry arrays here.
    if np.any(common_valid & ~(dav2_valid & moge_valid)):
        raise ValueError("paired stride-2 v1 common-valid exceeds initializer validity")
    centers = np.stack(
        [
            np.asarray(cameras["cams2world"][camera_rows[name]], np.float64)[:3, 3]
            for name in names
        ]
    )
    point_sets = {
        "DAV2_clean_v2": dav2_points,
        "MoGe3": moge_points,
        "source_MASt3R_reference": reference_points_array,
    }
    face4, individual4 = unit_edge_safe_face_masks(
        common_valid, point_sets, centers, stride=4
    )
    control4 = require_stride4_bit_parity(common_valid, face4, v3_arrays)
    face2, individual2 = unit_edge_safe_face_masks(
        common_valid, point_sets, centers, stride=2
    )
    topology2 = _one_stride_topology(common_valid, face2, stride=2)
    eligible_stride4 = _packed_source_eligible_count(
        common_valid,
        control4["sampled_vertex_offsets_stride4_control"],
        control4["sampled_vertex_pixel_indices_stride4_control"],
    )
    eligible_stride2 = _packed_source_eligible_count(
        common_valid,
        topology2["sampled_vertex_offsets_stride2"],
        topology2["sampled_vertex_pixel_indices_stride2"],
    )
    triangle_area2 = {
        name: _triangle_area2_stats(
            point_sets[name],
            topology2["sampled_vertex_offsets_stride2"],
            topology2["sampled_vertex_pixel_indices_stride2"],
            topology2["faces_stride2"],
        )
        for name in GEOMETRY_NAMES
    }
    stride4_hash_input = {
        "face_valid_stride4": face4,
        **{
            name: control4[f"{name}_control"]
            for name in (
                "sampled_vertex_offsets_stride4",
                "sampled_vertex_pixel_indices_stride4",
                "face_offsets_stride4",
                "faces_stride4",
            )
        },
    }
    metadata = {
        "artifact_type": SCHEMA,
        "paired_comparison_topology": True,
        "model_neutral_map_topology_claimed": False,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "aligned_arm_geometry_consumed": False,
        "geometry_names": list(GEOMETRY_NAMES),
        "edge_safety": {
            "absolute_edge_threshold_m": ABSOLUTE_EDGE_THRESHOLD_M,
            "relative_edge_threshold": RELATIVE_EDGE_THRESHOLD,
            "threshold_formula": "max(0.5m,0.05*median_camera_centered_range)",
            "edge_inventory": "all_unit_pixel_edges_in_inclusive_stride_patch",
        },
        "common_valid_source": "corrected_exact21_optimizer_v1_paired_common_valid",
        "source_eligible_denominator": (
            "five_pixel_central_difference_stencil_on_corrected_v1_common_valid_"
            "at_packed_output_pixels"
        ),
        "source_eligible_definition_stride_invariant": True,
        "sampled_source_inventory_stride_invariant": False,
        "support_fractions_only_comparable_within_same_stride": True,
        "source_eligible_count_stride4_control": eligible_stride4,
        "source_eligible_count_stride2": eligible_stride2,
        "source_eligible_count_stride2_over_stride4_ratio": float(
            eligible_stride2 / eligible_stride4
        ),
        "stride4_control_bit_equal_corrected_v3": True,
        "stride4_control_arrays_sha256": arrays_sha256(stride4_hash_input),
        "corrected_v3_stride4_arrays_sha256": arrays_sha256(
            {
                name: v3_arrays[name]
                for name in (
                    "face_valid_stride4",
                    "sampled_vertex_offsets_stride4",
                    "sampled_vertex_pixel_indices_stride4",
                    "face_offsets_stride4",
                    "faces_stride4",
                )
            }
        ),
        "stride2_repacked_without_orphans": True,
        "stride2_face_quad_count": int(face2.sum()),
        "stride2_triangle_count": int(len(topology2["faces_stride2"])),
        "stride2_packed_vertex_count": int(
            len(topology2["sampled_vertex_pixel_indices_stride2"])
        ),
        "stride2_face_quad_count_per_chart": face2.sum((1, 2)).astype(int).tolist(),
        "stride2_packed_vertex_count_per_chart": np.diff(
            topology2["sampled_vertex_offsets_stride2"]
        ).astype(int).tolist(),
        "stride2_triangle_area2_by_geometry": triangle_area2,
        "safe_face_count_by_geometry_and_stride": {
            "stride4": {
                name: int(individual4[name].sum()) for name in GEOMETRY_NAMES
            },
            "stride2": {
                name: int(individual2[name].sum()) for name in GEOMETRY_NAMES
            },
        },
        "comparison_domain_v1_file_sha256": file_sha256(
            comparison_domain_v1_path
        ),
        "comparison_domain_v1_content_sha256": expected_v1_content_sha256,
        "comparison_domain_v3_file_sha256": file_sha256(
            comparison_domain_v3_path
        ),
        "comparison_domain_v3_content_sha256": expected_v3_content_sha256,
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "frozen_submap_plan_content_sha256": expected_plan_content_sha256,
        "disjoint_upstream_authority_file_sha256": file_sha256(
            disjoint_upstream_authority_path
        ),
        "disjoint_upstream_authority_content_sha256": (
            expected_disjoint_authority_content_sha256
        ),
        "source_root": str(source_root),
        "source_tree_sha256": observed_source_tree,
        "source_cameras_file_sha256": cameras_hash,
        "source_selected_pointmap_inventory": pointmap_hashes,
        "source_selected_pointmap_inventory_sha256": canonical_json_sha256(
            pointmap_hashes
        ),
        "dav2_initializer_root": str(dav2_initializer_root),
        "dav2_initializer_manifest_file_sha256": v1_metadata.get(
            "dav2_initializer_manifest_file_sha256"
        ),
        "dav2_initializer_manifest_content_sha256": v1_metadata.get(
            "dav2_initializer_manifest_content_sha256"
        ),
        "dav2_initializer_file_sha256": v1_metadata.get(
            "dav2_initializer_file_sha256"
        ),
        "moge3_initializer_root": str(moge3_initializer_root),
        "moge3_initializer_manifest_file_sha256": v1_metadata.get(
            "moge_initializer_manifest_file_sha256"
        ),
        "moge3_initializer_manifest_content_sha256": v1_metadata.get(
            "moge_initializer_manifest_content_sha256"
        ),
        "moge3_initializer_file_sha256": v1_metadata.get(
            "moge_initializer_file_sha256"
        ),
        "paired_initializer_geometry_encoded_upstream": True,
        "topology_caveat": (
            "paired_DAV2_MoGe_MASt3R_comparison_topology_not_final_model_neutral_map"
        ),
        "builder_core_file_sha256": file_sha256(Path(__file__)),
    }
    return PairedStride2TopologyDiagnostic(
        chart_names=v1_arrays["chart_names"],
        valid=common_valid,
        face_valid_stride4_control=face4,
        face_valid_stride2=face2,
        metadata=metadata,
        **control4,
        **topology2,
    ).validated()


def _component_summary(
    chart_names: np.ndarray, edges: np.ndarray, keep: np.ndarray
) -> dict[str, object]:
    names = np.asarray(chart_names).astype(str)
    edges = np.asarray(edges, np.int64)
    keep = np.asarray(keep, bool)
    adjacency = [set() for _ in names]
    for first, second in edges[keep]:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    seen: set[int] = set()
    components: list[list[int]] = []
    for seed in range(len(names)):
        if seed in seen:
            continue
        pending = [seed]
        component: list[int] = []
        seen.add(seed)
        while pending:
            node = pending.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    pending.append(neighbour)
        components.append(sorted(component))
    components.sort(key=lambda value: (-len(value), value))
    return {
        "edge_count": int(np.sum(keep)),
        "component_count": len(components),
        "largest_component_chart_count": len(components[0]) if components else 0,
        "largest_component_chart_names": (
            names[components[0]].tolist() if components else []
        ),
        "isolated_chart_names": [
            str(names[value[0]]) for value in components if len(value) == 1
        ],
    }


def projective_graph_summary(
    authority: ProjectiveExactFaceSeamAuthority,
) -> dict[str, object]:
    authority = authority.validated()
    count = authority.source_count_by_direction
    edges = authority.edge_chart_indices
    masks = {
        "any_correspondence": np.sum(count, axis=1) > 0,
        "bidirectional_correspondence": np.all(count > 0, axis=1),
        "minimum_count_both_directions": np.all(
            count
            >= ProjectiveSeamConfig(
                **authority.metadata["config"]
            ).minimum_correspondences_per_direction,
            axis=1,
        ),
        "support_valid_both_directions": np.all(
            authority.direction_support_valid, axis=1
        ),
        "geometry_valid_both_directions": np.all(
            authority.direction_geometry_valid, axis=1
        ),
        "formal_valid_both_directions": authority.edge_formal_valid,
    }
    return {
        name: _component_summary(authority.chart_names, edges, keep)
        for name, keep in masks.items()
    }


def build_stride2_projective_authority(
    topology_path: Path,
    *,
    expected_topology_content_sha256: str,
    frozen_submap_plan_path: Path,
    expected_plan_content_sha256: str,
    source_root: Path,
    expected_source_tree_sha256: str,
    config: ProjectiveSeamConfig = ProjectiveSeamConfig(topology_stride=2),
) -> ProjectiveExactFaceSeamAuthority:
    """Freeze the stride-2 projective diagnostic with stride-4 thresholds."""

    topology_path = Path(topology_path)
    source_root = Path(source_root).resolve()
    topology = PairedStride2TopologyDiagnostic.load_npz(topology_path)
    if topology.metadata.get("content_sha256") != expected_topology_content_sha256:
        raise ValueError("stride-2 projective topology differs from pin")
    config = config.validated()
    if config.topology_stride != 2:
        raise ValueError("stride-2 projective diagnostic requires topology stride 2")
    baseline = ProjectiveSeamConfig().to_dict()
    candidate = config.to_dict()
    baseline.pop("topology_stride")
    candidate.pop("topology_stride")
    if candidate != baseline:
        raise ValueError("stride-2 diagnostic thresholds differ from stride-4 baseline")
    if (
        topology.metadata.get("frozen_submap_plan_content_sha256")
        != expected_plan_content_sha256
        or topology.metadata.get("source_tree_sha256")
        != expected_source_tree_sha256
    ):
        raise ValueError("stride-2 topology and projective lineage differ")
    plan = ChartSubmapPlan.load_npz(frozen_submap_plan_path)
    if plan.metadata.get("content_sha256") != expected_plan_content_sha256:
        raise ValueError("stride-2 projective plan differs from pin")
    if topology.chart_names.astype(str).tolist() != list(
        plan.selected_chart_names_in_order
    ):
        raise ValueError("stride-2 projective chart order differs from plan")
    if source_tree_sha256(source_root) != expected_source_tree_sha256:
        raise ValueError("stride-2 projective source tree differs from pin")

    cameras_path = source_root / "cameras.json"
    if file_sha256(cameras_path) != topology.metadata.get(
        "source_cameras_file_sha256"
    ):
        raise ValueError("stride-2 projective cameras differ from topology")
    cameras = json.loads(cameras_path.read_text())
    camera_names = [Path(value).name for value in cameras.get("filepaths", [])]
    camera_rows = {name: row for row, name in enumerate(camera_names)}
    names = topology.chart_names.astype(str).tolist()
    height, width = topology.valid.shape[1:]
    reference_dense, pointmap_shapes = [], []
    pointmap_hashes: dict[str, str] = {}
    for name in names:
        path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        pointmap_hashes[name] = file_sha256(path)
        if pointmap_hashes[name] != topology.metadata.get(
            "source_selected_pointmap_inventory", {}
        ).get(name):
            raise ValueError(f"stride-2 projective pointmap differs for {name}")
        points, shape = _load_source_reference_dense(
            path, output_height=height, output_width=width
        )
        reference_dense.append(points)
        pointmap_shapes.append(shape)
    reference_dense_array = np.stack(reference_dense)
    eligible_dense = np.stack(
        [_common_valid_normal_stencil(mask) for mask in topology.valid]
    )
    reference_dense_normal = np.stack(
        [
            _dense_world_normals(reference_dense_array[row], eligible_dense[row])
            for row in range(len(names))
        ]
    )
    camera_to_world = np.stack(
        [
            np.asarray(cameras["cams2world"][camera_rows[name]], np.float64)
            for name in names
        ]
    )
    focal_px = np.asarray(
        [
            float(cameras["focals"][camera_rows[name]])
            * width
            / pointmap_shapes[row][1]
            for row, name in enumerate(names)
        ],
        np.float64,
    )
    metadata = {
        "paired_stride2_topology_schema": SCHEMA,
        "paired_stride2_topology_filename": topology_path.name,
        "paired_stride2_topology_file_sha256": file_sha256(topology_path),
        "paired_stride2_topology_content_sha256": expected_topology_content_sha256,
        "paired_stride2_topology_arrays_sha256": topology.metadata.get(
            "arrays_sha256"
        ),
        "paired_stride2_topology_stride4_control_arrays_sha256": (
            topology.metadata.get("stride4_control_arrays_sha256")
        ),
        "comparison_domain_v1_file_sha256": topology.metadata.get(
            "comparison_domain_v1_file_sha256"
        ),
        "comparison_domain_v1_content_sha256": topology.metadata.get(
            "comparison_domain_v1_content_sha256"
        ),
        "comparison_domain_v3_file_sha256": topology.metadata.get(
            "comparison_domain_v3_file_sha256"
        ),
        "comparison_domain_v3_content_sha256": topology.metadata.get(
            "comparison_domain_v3_content_sha256"
        ),
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "frozen_submap_plan_content_sha256": expected_plan_content_sha256,
        "source_root": str(source_root),
        "source_tree_sha256": expected_source_tree_sha256,
        "source_cameras_file_sha256": file_sha256(cameras_path),
        "source_selected_pointmap_inventory": pointmap_hashes,
        "source_selected_pointmap_inventory_sha256": canonical_json_sha256(
            pointmap_hashes
        ),
        "full_submap_gate_primary_stride": 2,
        "source_reference_edge_safe": True,
        "plan_projection_contract_corrected": True,
        "legacy_plan_phase_diagnostic_only": False,
        "projective_correspondence_production_candidate": False,
        "paired_stride2_densification_diagnostic": True,
        "paired_comparison_topology": True,
        "final_model_neutral_map_topology_eligible": False,
        "topology_caveat": (
            "paired_DAV2_MoGe_MASt3R_comparison_topology_not_final_model_neutral_map"
        ),
        "thresholds_identical_to_stride4_projective_v4": True,
        "source_eligible_definition_stride_invariant": True,
        "sampled_source_inventory_stride_invariant": False,
        "support_fractions_only_comparable_within_same_stride": True,
        "source_eligible_definition_note": (
            "five-pixel output-domain stencil is unchanged, but stride-2 packs "
            "many more output pixels than stride-4"
        ),
        "stride2_topology_builder_core_file_sha256": file_sha256(Path(__file__)),
    }
    authority = freeze_projective_exact_face_correspondences(
        chart_names=topology.chart_names,
        common_valid=topology.valid,
        chart_vertex_offsets=topology.sampled_vertex_offsets_stride2,
        sampled_vertex_pixel_indices=topology.sampled_vertex_pixel_indices_stride2,
        chart_face_offsets=topology.face_offsets_stride2,
        faces=topology.faces_stride2,
        reference_points_world=reference_dense_array,
        reference_dense_normals_world=reference_dense_normal,
        camera_to_world=camera_to_world,
        focal_px=focal_px,
        plan_chart_names=plan.chart_names,
        coverage_edges=plan.coverage_edges,
        symmetric_surface_overlap=plan.symmetric_surface_overlap,
        config=config,
        metadata=metadata,
    )
    summary = projective_graph_summary(authority)
    formal_largest = summary["formal_valid_both_directions"][
        "largest_component_chart_count"
    ]
    augmented = dict(authority.metadata)
    augmented.update(
        {
            "stride2_graph_summary": summary,
            "stride2_formal_component_target_minimum_chart_count": (
                FORMAL_COMPONENT_TARGET
            ),
            "stride2_formal_component_target_pass": (
                int(formal_largest) >= FORMAL_COMPONENT_TARGET
            ),
            "formal_selector_handoff_eligible": (
                int(formal_largest) >= FORMAL_COMPONENT_TARGET
            ),
        }
    )
    return replace(authority, metadata=augmented).validated()


def diagnostic_report(
    topology: PairedStride2TopologyDiagnostic,
    authority: ProjectiveExactFaceSeamAuthority,
) -> dict[str, object]:
    topology = topology.validated()
    authority = authority.validated()
    summary = projective_graph_summary(authority)
    formal_largest = int(
        summary["formal_valid_both_directions"][
            "largest_component_chart_count"
        ]
    )
    report = {
        "artifact_type": REPORT_SCHEMA,
        "paired_comparison_topology": True,
        "model_neutral_map_topology_claimed": False,
        "uses_query_or_ground_truth": False,
        "thresholds_identical_to_stride4_projective_v4": True,
        "chart_count": len(topology.chart_names),
        "stride2_face_quad_count": int(topology.face_valid_stride2.sum()),
        "stride2_triangle_count": len(topology.faces_stride2),
        "stride2_packed_vertex_count": len(
            topology.sampled_vertex_pixel_indices_stride2
        ),
        "source_eligible_count_stride4_control": topology.metadata.get(
            "source_eligible_count_stride4_control"
        ),
        "source_eligible_count_stride2": topology.metadata.get(
            "source_eligible_count_stride2"
        ),
        "source_eligible_count_stride2_over_stride4_ratio": topology.metadata.get(
            "source_eligible_count_stride2_over_stride4_ratio"
        ),
        "support_fractions_only_comparable_within_same_stride": True,
        "frozen_correspondence_count": int(
            authority.edge_correspondence_offsets[-1]
        ),
        "graph_summary": summary,
        "formal_component_target_minimum_chart_count": FORMAL_COMPONENT_TARGET,
        "formal_component_target_pass": formal_largest >= FORMAL_COMPONENT_TARGET,
        "decision": "HANDOFF_TO_SELECTOR" if formal_largest >= FORMAL_COMPONENT_TARGET else "KILL",
        "stride1_structurally_justified": False,
        "stride1_justification_rule": (
            "stride1 is justified only if stride2 removes a topology-resolution "
            "bottleneck yet narrowly misses the 16-chart formal component; it is "
            "not automatic after a stride2 KILL"
        ),
        "topology_content_sha256": topology.metadata.get("content_sha256"),
        "authority_content_sha256": authority.metadata.get("content_sha256"),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report
