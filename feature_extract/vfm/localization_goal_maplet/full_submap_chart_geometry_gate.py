"""Fail-closed full-submap geometry gate for M0/M1/M2 surface maps.

The gate is deliberately independent of chart selection and alignment.  It
only consumes frozen artifacts and renders the same dense, held mapping rays
against all three representations:

``M0``
    An equal-source-budget bounded control.  Until a separately certified
    source-only 2DGS exists, this is the source MASt3R reference surface on
    the sealed common topology.  A full-train 2DGS is diagnostic-only.
``M1``
    A DAV2-initialized explicit chart atlas.
``M2``
    A MoGe-3-initialized explicit chart atlas.

The held reference is a source-disjoint mapping-geometry diagnostic, not
sensor depth ground truth and never a Cambridge query pose.  Missing rendered
rays fail the primary recall metrics.  Conditional depth errors are reported
only as secondary diagnostics and therefore cannot reward a sparse atlas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .chart_submap_selection import ChartSubmapPlan
from .chart_comparison_domain import (
    BASE_ARRAY_NAMES as COMPARISON_BASE_ARRAY_NAMES,
    FULL_GATE_ELIGIBLE_STRIDES,
    SCHEMA as LEGACY_EXACT_COMPARISON_DOMAIN_SCHEMA,
    topology_array_names,
    validate_exact_topology_arrays,
)
from .chart_comparison_reference_safe_domain import (
    OPTIMIZER_V1_SCHEMA,
    SCHEMA as PHYSICAL_SAFE_COMPARISON_DOMAIN_SCHEMA,
)
from .explicit_chart_atlas import ExplicitChartAtlas
from .geometry_native_planar_map import GeometryNativePlanarMap
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256


HELD_RAY_SCHEMA = "goal_maplet_strict_held_ray_inventory_v1"
REPORT_SCHEMA = "goal_maplet_full_submap_chart_geometry_gate_v1"
COMPARISON_DOMAIN_SCHEMA = PHYSICAL_SAFE_COMPARISON_DOMAIN_SCHEMA
FORMAL_GATE_PRIMARY_STRIDE = 4
SOURCE_SURFACE_SCHEMA = "goal_maplet_source_only_bounded_surface_baseline_v1"
BOUNDED_DOMAIN_SCHEMA = "goal_maplet_axis_aligned_bounded_submap_v1"
BOUNDARY_SEMANTICS = (
    "geometry_depth_or_unsigned_normal_discontinuity_excluding_confidence_mask_edges_and_parameterization_seams"
)


def _replay_metadata(metadata: Mapping[str, object], label: str) -> None:
    claimed = metadata.get("content_sha256")
    payload = dict(metadata)
    payload.pop("content_sha256", None)
    if claimed != canonical_json_sha256(payload):
        raise ValueError(f"{label} metadata content hash does not replay")


def _require_sha(value: object, label: str) -> str:
    text = str(value) if value is not None else ""
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return text


def bounded_submap_content_sha256(
    minimum_world: np.ndarray | list[float],
    maximum_world: np.ndarray | list[float],
) -> str:
    """Canonical identity of the explicit AABB used by all gate arms."""

    minimum = np.asarray(minimum_world, np.float64)
    maximum = np.asarray(maximum_world, np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
        raise ValueError("invalid bounded-submap AABB")
    return canonical_json_sha256(
        {
            "artifact_type": BOUNDED_DOMAIN_SCHEMA,
            "minimum_world": minimum.tolist(),
            "maximum_world": maximum.tolist(),
        }
    )


def _find_sibling_artifact_by_file_sha256(path: Path, expected: object) -> Path:
    expected_hash = _require_sha(expected, "upstream comparison-domain file hash")
    matches = [
        candidate
        for candidate in Path(path).parent.glob("*.npz")
        if candidate.resolve() != Path(path).resolve()
        and file_sha256(candidate) == expected_hash
    ]
    if not matches:
        raise ValueError("physical-safe comparison domain upstream artifact is absent")
    return sorted(matches)[0]


def _load_domain_payload(
    path: Path,
    *,
    array_names: tuple[str, ...],
    expected_schema: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        if any(name not in data.files for name in array_names):
            raise ValueError("upstream comparison-domain array inventory is incomplete")
        arrays = {name: np.asarray(data[name]) for name in array_names}
        metadata = json.loads(str(data["metadata_json"].item()))
    if arrays_sha256(arrays) != metadata.get("arrays_sha256"):
        raise ValueError("upstream comparison-domain arrays differ from lineage")
    _replay_metadata(metadata, "upstream comparison domain")
    if metadata.get("artifact_type") != expected_schema:
        raise ValueError("upstream comparison-domain schema differs")
    return arrays, metadata


def _replay_physical_safe_domain_chain(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> None:
    """Replay v3 physical topology through exact v2 to optimizer v1."""

    required_true = (
        "source_reference_edge_safe",
        "source_reference_edge_safety_replayed",
        "physical_face_safety_authority",
        "face_valid_v3_subset_of_face_valid_v2",
        "exact_topology_repacked_after_reference_safety",
        "valid_and_chart_names_byte_equal_upstream_v2",
        "full_submap_gate_eligible",
        "exact_pixel_mask_frozen_for_both_arms",
        "exact_face_indices_frozen_for_both_arms",
    )
    if any(metadata.get(key) is not True for key in required_true):
        raise ValueError("comparison domain is not physical-safe v3 authority")
    v2_path = _find_sibling_artifact_by_file_sha256(
        path, metadata.get("upstream_exact_topology_v2_file_sha256")
    )
    all_names = tuple(COMPARISON_BASE_ARRAY_NAMES) + tuple(topology_array_names())
    v2_arrays, v2_metadata = _load_domain_payload(
        v2_path,
        array_names=all_names,
        expected_schema=LEGACY_EXACT_COMPARISON_DOMAIN_SCHEMA,
    )
    v2_expected = {
        "upstream_exact_topology_v2_file_sha256": file_sha256(v2_path),
        "upstream_exact_topology_v2_content_sha256": v2_metadata.get(
            "content_sha256"
        ),
        "upstream_exact_topology_v2_arrays_sha256": v2_metadata.get(
            "arrays_sha256"
        ),
        "upstream_exact_topology_v2_exact_topology_arrays_sha256": v2_metadata.get(
            "exact_topology_arrays_sha256"
        ),
    }
    for key, expected in v2_expected.items():
        if metadata.get(key) != expected:
            raise ValueError(f"physical-safe v3 {key} does not replay v2")
    validate_exact_topology_arrays(
        {name: v2_arrays[name] for name in COMPARISON_BASE_ARRAY_NAMES},
        {name: v2_arrays[name] for name in topology_array_names()},
        expected_sha256=str(v2_metadata.get("exact_topology_arrays_sha256")),
    )
    for name in ("chart_names", "valid"):
        if not np.array_equal(np.asarray(arrays[name]), v2_arrays[name]):
            raise ValueError("physical-safe v3 changed common chart/pixel inventory")
    for stride in (4, 8):
        name = f"face_valid_stride{stride}"
        if np.any(np.asarray(arrays[name], bool) & ~np.asarray(v2_arrays[name], bool)):
            raise ValueError("physical-safe v3 face mask is not a v2 subset")

    v1_path = _find_sibling_artifact_by_file_sha256(
        path, metadata.get("upstream_optimizer_comparison_domain_v1_file_sha256")
    )
    v1_arrays, v1_metadata = _load_domain_payload(
        v1_path,
        array_names=tuple(COMPARISON_BASE_ARRAY_NAMES),
        expected_schema=OPTIMIZER_V1_SCHEMA,
    )
    for name in COMPARISON_BASE_ARRAY_NAMES:
        if not np.array_equal(v2_arrays[name], v1_arrays[name]):
            raise ValueError("exact v2 base domain does not replay optimizer v1")
    v1_expected = {
        "upstream_optimizer_comparison_domain_v1_file_sha256": file_sha256(v1_path),
        "upstream_optimizer_comparison_domain_v1_content_sha256": v1_metadata.get(
            "content_sha256"
        ),
        "upstream_optimizer_comparison_domain_v1_arrays_sha256": v1_metadata.get(
            "arrays_sha256"
        ),
    }
    for key, expected in v1_expected.items():
        if metadata.get(key) != expected:
            raise ValueError(f"physical-safe v3 {key} does not replay optimizer v1")
    if (
        v2_metadata.get("upstream_comparison_domain_file_sha256") != file_sha256(v1_path)
        or v2_metadata.get("upstream_comparison_domain_content_sha256")
        != v1_metadata.get("content_sha256")
    ):
        raise ValueError("exact v2 does not bind the replayed optimizer v1")


@dataclass(frozen=True)
class _ComparisonDomain:
    chart_names: np.ndarray
    valid: np.ndarray
    face_valid_stride4: np.ndarray
    face_valid_stride8: np.ndarray
    exact_topology: dict[str, np.ndarray]
    metadata: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            name: np.asarray(getattr(self, name))
            for name in (
                "chart_names", "valid", "face_valid_stride4", "face_valid_stride8",
            )
        }
        arrays.update(
            {
                name: np.asarray(value)
                for name, value in self.exact_topology.items()
            }
        )
        return arrays

    def validated(self) -> "_ComparisonDomain":
        arrays = self.arrays()
        charts = len(arrays["chart_names"])
        if charts < 2 or len(set(arrays["chart_names"].astype(str))) != charts:
            raise ValueError("comparison domain chart inventory is invalid")
        if arrays["valid"].ndim != 3 or arrays["valid"].shape[0] != charts:
            raise ValueError("comparison domain pixel validity is invalid")
        for stride in (4, 8):
            key = f"face_valid_stride{stride}"
            expected = (
                charts,
                len(np.arange(0, arrays["valid"].shape[1], stride)) - 1,
                len(np.arange(0, arrays["valid"].shape[2], stride)) - 1,
            )
            if arrays[key].shape != expected:
                raise ValueError(f"comparison domain {key} shape differs")
        if self.metadata.get("artifact_type") != COMPARISON_DOMAIN_SCHEMA:
            raise ValueError(
                "full-submap gate requires physical-safe v3 comparison domain; "
                "v2 is legacy diagnostic only"
            )
        if self.metadata.get("uses_query_or_ground_truth") is not False:
            raise ValueError("comparison domain did not explicitly reject query/GT")
        if self.metadata.get("full_submap_gate_eligible") is not True:
            raise ValueError("comparison domain is diagnostic-only")
        if self.metadata.get("exact_pixel_mask_frozen_for_both_arms") is not True:
            raise ValueError("comparison domain did not freeze the exact pixel mask")
        if self.metadata.get("exact_face_indices_frozen_for_both_arms") is not True:
            raise ValueError("comparison domain did not freeze exact face indices")
        if self.metadata.get("orphan_sampled_vertices_present") is not False:
            raise ValueError("comparison domain retains orphan sampled vertices")
        if (
            self.metadata.get("full_submap_gate_eligible_strides")
            != list(FULL_GATE_ELIGIBLE_STRIDES)
            or self.metadata.get("required_nonempty_face_inventory_strides")
            != list(FULL_GATE_ELIGIBLE_STRIDES)
            or self.metadata.get("full_submap_gate_primary_stride")
            != FORMAL_GATE_PRIMARY_STRIDE
            or self.metadata.get("noneligible_stride_empty_inventory_permitted")
            is not True
        ):
            raise ValueError(
                "comparison domain formal gate stride authority differs"
            )
        base_arrays = {name: arrays[name] for name in COMPARISON_BASE_ARRAY_NAMES}
        topology = {name: arrays[name] for name in topology_array_names()}
        primary_face_offsets = np.asarray(
            topology[f"face_offsets_stride{FORMAL_GATE_PRIMARY_STRIDE}"],
            np.int64,
        )
        if np.any(np.diff(primary_face_offsets) <= 0):
            raise ValueError(
                "comparison domain primary stride has an empty chart topology"
            )
        validate_exact_topology_arrays(
            base_arrays,
            topology,
            expected_sha256=str(self.metadata.get("exact_topology_arrays_sha256")),
        )
        _require_sha(
            self.metadata.get("disjoint_upstream_authority_file_sha256"),
            "comparison-domain authority file hash",
        )
        _require_sha(
            self.metadata.get("disjoint_upstream_authority_content_sha256"),
            "comparison-domain authority content hash",
        )
        _require_sha(
            self.metadata.get("frozen_submap_plan_content_sha256"),
            "comparison-domain frozen-plan hash",
        )
        _require_sha(
            self.metadata.get("mapping_source_ordered_names_sha256"),
            "comparison-domain source-name hash",
        )
        return self

    @classmethod
    def load_npz(cls, path: Path) -> "_ComparisonDomain":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {
                name: np.asarray(data[name])
                for name in COMPARISON_BASE_ARRAY_NAMES
            }
            exact_topology = {
                name: np.asarray(data[name])
                for name in topology_array_names()
            }
        if arrays_sha256({**arrays, **exact_topology}) != metadata.get("arrays_sha256"):
            raise ValueError("comparison domain arrays differ from lineage")
        _replay_metadata(metadata, "comparison domain")
        _replay_physical_safe_domain_chain(
            Path(path), {**arrays, **exact_topology}, metadata,
        )
        return cls(
            metadata=metadata,
            exact_topology=exact_topology,
            **arrays,
        ).validated()


def _expected_comparison_topology(
    domain: _ComparisonDomain,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the exact row-major sampled vertices, UVs, and frozen faces."""

    _, height, width = domain.valid.shape
    vertex_offsets = np.asarray(
        domain.exact_topology[f"sampled_vertex_offsets_stride{stride}"],
        np.int64,
    )
    pixel_indices = np.asarray(
        domain.exact_topology[f"sampled_vertex_pixel_indices_stride{stride}"],
        np.int64,
    )
    face_offsets = np.asarray(
        domain.exact_topology[f"face_offsets_stride{stride}"],
        np.int64,
    )
    faces = np.asarray(domain.exact_topology[f"faces_stride{stride}"], np.int64)
    uv = np.stack(
        (
            (pixel_indices % width) / (width - 1),
            (pixel_indices // width) / (height - 1),
        ),
        axis=1,
    )
    return (
        vertex_offsets,
        uv,
        face_offsets,
        faces,
    )


@dataclass(frozen=True)
class StrictHeldRayInventory:
    """Dense held-view ray/reference table with immutable authority binding."""

    view_names: np.ndarray
    camera_to_world: np.ndarray
    focal_xy: np.ndarray
    principal_xy: np.ndarray
    reference_depth_m: np.ndarray
    reference_normal_world: np.ndarray
    reference_valid: np.ndarray
    reference_boundary: np.ndarray
    block_ids: np.ndarray
    metadata: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in (
                "view_names",
                "camera_to_world",
                "focal_xy",
                "principal_xy",
                "reference_depth_m",
                "reference_normal_world",
                "reference_valid",
                "reference_boundary",
                "block_ids",
            )
        }

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.reference_depth_m.shape)

    def validated(self) -> "StrictHeldRayInventory":
        arrays = self.arrays()
        if arrays["reference_depth_m"].ndim != 3:
            raise ValueError("held depth must be a dense VxHxW table")
        views, height, width = arrays["reference_depth_m"].shape
        if views < 2 or height < 2 or width < 2:
            raise ValueError("held ray inventory is too small")
        if arrays["view_names"].shape != (views,) or len(set(arrays["view_names"].astype(str))) != views:
            raise ValueError("held view names are invalid")
        expected = {
            "camera_to_world": (views, 4, 4),
            "focal_xy": (views, 2),
            "principal_xy": (views, 2),
            "reference_normal_world": (views, height, width, 3),
            "reference_valid": (views, height, width),
            "reference_boundary": (views, height, width),
            "block_ids": (views,),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"invalid held {name} shape")
        for name in (
            "camera_to_world",
            "focal_xy",
            "principal_xy",
            "reference_depth_m",
            "reference_normal_world",
        ):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"held {name} contains nonfinite values")
        if np.any(arrays["focal_xy"] <= 0):
            raise ValueError("held focal length must be positive")
        if len(set(arrays["block_ids"].astype(str).tolist())) < 2:
            raise ValueError("paired block bootstrap requires at least two held blocks")
        valid = arrays["reference_valid"].astype(bool)
        if np.any(valid & (arrays["reference_depth_m"] <= 0)) or np.any(valid.reshape(views, -1).sum(1) == 0):
            raise ValueError("held reference-valid rays need positive depth in every view")
        normal_length = np.linalg.norm(arrays["reference_normal_world"], axis=3)
        if np.any(valid & ((normal_length < 0.9) | (normal_length > 1.1))):
            raise ValueError("held reference-valid normals are not unit length")
        if self.metadata.get("artifact_type") != HELD_RAY_SCHEMA:
            raise ValueError("wrong held ray inventory schema")
        if self.metadata.get("uses_query_or_ground_truth") is not False:
            raise ValueError("held ray inventory did not explicitly reject query/GT")
        if self.metadata.get("dense_row_major_pixel_centers") is not True:
            raise ValueError("held ray ordering is not frozen dense row-major")
        if self.metadata.get("reference_semantics") != "source_disjoint_held_mapping_geometry_not_sensor_depth_gt":
            raise ValueError("held reference semantics differ")
        if self.metadata.get("boundary_semantics") != BOUNDARY_SEMANTICS:
            raise ValueError("held boundary semantics differ")
        if self.metadata.get("held_geometry_opened_after_source_bounds_frozen") is not True:
            raise ValueError("held geometry may have influenced submap-bound selection")
        if self.metadata.get("bound_uses_held_geometry") is not False:
            raise ValueError("held geometry was consumed while choosing submap bounds")
        if self.metadata.get("held_inventory_exactly_authority_order") is not True:
            raise ValueError("held ray inventory is not exact authority order")
        if self.metadata.get("builder_config_frozen_before_held") is not True:
            raise ValueError("held ray builder config was not frozen before held access")
        frozen_config = self.metadata.get("pre_frozen_builder_config")
        if not isinstance(frozen_config, dict) or self.metadata.get(
            "pre_frozen_builder_config_content_sha256"
        ) != canonical_json_sha256(frozen_config):
            raise ValueError("held ray pre-frozen builder config does not replay")
        _require_sha(self.metadata.get("disjoint_upstream_authority_file_sha256"), "held authority file hash")
        _require_sha(self.metadata.get("disjoint_upstream_authority_content_sha256"), "held authority content hash")
        _require_sha(self.metadata.get("held_pointmap_inventory_sha256"), "held pointmap inventory hash")
        _require_sha(self.metadata.get("mapping_source_ordered_names_sha256"), "mapping source-name hash")
        _require_sha(self.metadata.get("comparison_domain_content_sha256"), "held comparison domain hash")
        _require_sha(self.metadata.get("comparison_domain_file_sha256"), "held comparison domain file hash")
        _require_sha(self.metadata.get("frozen_submap_plan_content_sha256"), "held frozen-plan hash")
        _require_sha(self.metadata.get("bounded_submap_content_sha256"), "held bounded-submap hash")
        _require_sha(self.metadata.get("coordinate_cameras_file_sha256"), "held coordinate-camera hash")
        _require_sha(self.metadata.get("coordinate_images_file_sha256"), "held coordinate-image hash")
        bounds_min = np.asarray(self.metadata.get("bounded_submap_min_world"), np.float64)
        bounds_max = np.asarray(self.metadata.get("bounded_submap_max_world"), np.float64)
        if (
            bounds_min.shape != (3,)
            or bounds_max.shape != (3,)
            or not np.isfinite(bounds_min).all()
            or not np.isfinite(bounds_max).all()
            or np.any(bounds_min >= bounds_max)
        ):
            raise ValueError("held ray inventory lacks a finite bounded-submap box")
        if self.metadata.get("bounded_submap_content_sha256") != bounded_submap_content_sha256(
            bounds_min, bounds_max,
        ):
            raise ValueError("held bounded-submap hash does not replay its box")
        yy, xx = np.mgrid[:height, :width]
        for view in range(views):
            direction_camera = np.stack(
                (
                    (xx - arrays["principal_xy"][view, 0]) / arrays["focal_xy"][view, 0],
                    (yy - arrays["principal_xy"][view, 1]) / arrays["focal_xy"][view, 1],
                    np.ones_like(xx, dtype=np.float64),
                ),
                axis=2,
            )
            points_camera = direction_camera * arrays["reference_depth_m"][view, ..., None]
            pose = arrays["camera_to_world"][view]
            points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
            selected = points_world[valid[view]]
            if np.any(selected < bounds_min - 1e-6) or np.any(selected > bounds_max + 1e-6):
                raise ValueError("held reference-valid ray escapes the frozen bounded submap")
        names_hash = canonical_json_sha256(arrays["view_names"].astype(str).tolist())
        if self.metadata.get("held_view_names_sha256") != names_hash:
            raise ValueError("held view-name hash differs")
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
    def load_npz(cls, path: Path) -> "StrictHeldRayInventory":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {
                name: np.asarray(data[name])
                for name in (
                    "view_names",
                    "camera_to_world",
                    "focal_xy",
                    "principal_xy",
                    "reference_depth_m",
                    "reference_normal_world",
                    "reference_valid",
                    "reference_boundary",
                    "block_ids",
                )
            }
        if arrays_sha256(arrays) != metadata.get("arrays_sha256"):
            raise ValueError("held ray arrays differ from lineage")
        _replay_metadata(metadata, "held ray inventory")
        return cls(metadata=metadata, **arrays).validated()


@dataclass(frozen=True)
class SourceOnlyBoundedSurfaceBaseline:
    """Finite source-evidence surface control for an equal-budget main table.

    This is intentionally *not* described as a 2DGS baseline.  Its vertices
    are the isolated source MASt3R point maps sampled on the exact sealed
    comparison topology.  It lets the geometry gate run before a separately
    trained source-only 2DGS exists, without laundering a full-train 2DGS map
    into an equal-budget M0 comparison.
    """

    chart_names: np.ndarray
    chart_vertex_offsets: np.ndarray
    vertices_world: np.ndarray
    normals_world: np.ndarray
    chart_face_offsets: np.ndarray
    faces: np.ndarray
    metadata: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in (
                "chart_names",
                "chart_vertex_offsets",
                "vertices_world",
                "normals_world",
                "chart_face_offsets",
                "faces",
            )
        }

    def validated(self) -> "SourceOnlyBoundedSurfaceBaseline":
        arrays = self.arrays()
        charts = len(arrays["chart_names"])
        vertices = len(arrays["vertices_world"])
        faces = len(arrays["faces"])
        if charts < 2 or len(set(arrays["chart_names"].astype(str))) != charts:
            raise ValueError("source surface chart inventory is invalid")
        if arrays["chart_vertex_offsets"].shape != (charts + 1,):
            raise ValueError("source surface vertex offsets are invalid")
        if arrays["chart_face_offsets"].shape != (charts + 1,):
            raise ValueError("source surface face offsets are invalid")
        if (
            arrays["chart_vertex_offsets"][0] != 0
            or arrays["chart_vertex_offsets"][-1] != vertices
            or np.any(np.diff(arrays["chart_vertex_offsets"]) <= 0)
        ):
            raise ValueError("source surface contains an empty/invalid chart")
        if (
            arrays["chart_face_offsets"][0] != 0
            or arrays["chart_face_offsets"][-1] != faces
            or np.any(np.diff(arrays["chart_face_offsets"]) <= 0)
        ):
            raise ValueError("source surface contains an empty/invalid face inventory")
        if arrays["vertices_world"].shape != (vertices, 3):
            raise ValueError("source surface vertices have the wrong shape")
        if arrays["normals_world"].shape != (vertices, 3):
            raise ValueError("source surface normals have the wrong shape")
        if arrays["faces"].shape != (faces, 3):
            raise ValueError("source surface faces have the wrong shape")
        if not np.isfinite(arrays["vertices_world"]).all() or not np.isfinite(
            arrays["normals_world"]
        ).all():
            raise ValueError("source surface contains nonfinite geometry")
        if np.any((arrays["faces"] < 0) | (arrays["faces"] >= vertices)):
            raise ValueError("source surface face leaves the vertex inventory")
        normal_length = np.linalg.norm(arrays["normals_world"], axis=1)
        if np.any((normal_length < 0.9) | (normal_length > 1.1)):
            raise ValueError("source surface normals are not unit length")
        for chart in range(charts):
            vertex_lo, vertex_hi = map(
                int, arrays["chart_vertex_offsets"][chart : chart + 2]
            )
            face_lo, face_hi = map(
                int, arrays["chart_face_offsets"][chart : chart + 2]
            )
            if np.any(
                (arrays["faces"][face_lo:face_hi] < vertex_lo)
                | (arrays["faces"][face_lo:face_hi] >= vertex_hi)
            ):
                raise ValueError("source surface face crosses a chart boundary")
        metadata = self.metadata
        if metadata.get("artifact_type") != SOURCE_SURFACE_SCHEMA:
            raise ValueError("wrong source-only surface baseline schema")
        if metadata.get("uses_query_pose_or_ground_truth") is not False:
            raise ValueError("source surface did not explicitly reject query/GT")
        if metadata.get("comparison_role") != (
            "equal_budget_main_table_source_reference_surface_control"
        ):
            raise ValueError("source surface is not an equal-budget main-table control")
        if metadata.get("comparison_budget") != "exact_frozen_source_ordered_pool":
            raise ValueError("source surface evidence budget differs")
        if metadata.get("held_mapping_images_consumed") is not False:
            raise ValueError("source surface consumed held mapping images")
        if metadata.get("outside_frozen_source_mapping_images_consumed") is not False:
            raise ValueError("source surface consumed images outside frozen source")
        if metadata.get("paired_common_face_inventory") is not True:
            raise ValueError("source surface lacks exact common topology")
        for key, label in (
            ("disjoint_upstream_authority_content_sha256", "source surface authority"),
            ("mapping_source_ordered_names_sha256", "source surface mapping pool"),
            ("comparison_domain_content_sha256", "source surface common domain"),
            ("comparison_domain_file_sha256", "source surface common-domain file"),
            ("frozen_submap_plan_content_sha256", "source surface plan"),
            ("bounded_submap_content_sha256", "source surface bounds"),
            ("coordinate_cameras_file_sha256", "source surface coordinate cameras"),
            ("coordinate_images_file_sha256", "source surface coordinate images"),
            ("source_pointmap_inventory_sha256", "source surface pointmaps"),
        ):
            _require_sha(metadata.get(key), label)
        bounds_min = np.asarray(metadata.get("bounded_submap_min_world"), np.float64)
        bounds_max = np.asarray(metadata.get("bounded_submap_max_world"), np.float64)
        if metadata.get("bounded_submap_content_sha256") != bounded_submap_content_sha256(
            bounds_min, bounds_max
        ):
            raise ValueError("source surface bounded-submap hash does not replay")
        if np.any(arrays["vertices_world"] < bounds_min - 1e-6) or np.any(
            arrays["vertices_world"] > bounds_max + 1e-6
        ):
            raise ValueError("source surface escapes the bounded submap")
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
    def load_npz(cls, path: Path) -> "SourceOnlyBoundedSurfaceBaseline":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {
                name: np.asarray(data[name])
                for name in (
                    "chart_names",
                    "chart_vertex_offsets",
                    "vertices_world",
                    "normals_world",
                    "chart_face_offsets",
                    "faces",
                )
            }
        if arrays_sha256(arrays) != metadata.get("arrays_sha256"):
            raise ValueError("source surface arrays differ from lineage")
        _replay_metadata(metadata, "source-only bounded surface")
        return cls(metadata=metadata, **arrays).validated()


@dataclass(frozen=True)
class FullSubmapGeometryGateConfig:
    depth_absolute_tolerance_m: float = 0.50
    depth_relative_tolerance: float = 0.05
    normal_thresholds_deg: tuple[float, float, float] = (10.0, 20.0, 30.0)
    boundary_tolerance_px: float = 2.0
    boundary_depth_absolute_m: float = 0.25
    boundary_depth_relative: float = 0.04
    boundary_normal_angle_deg: float = 30.0
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 260830
    confidence: float = 0.95
    good_ray_noninferiority: float = 0.02
    normal20_noninferiority: float = 0.02
    boundary_f1_noninferiority: float = 0.02
    absrel_median_noninferiority: float = 0.02
    absrel_p90_noninferiority: float = 0.05
    minimum_absolute_good_ray_recall: float = 0.20
    minimum_absolute_joint_depth_normal_recall_20: float = 0.10
    maximum_seam_correspondence_m: float = 1.0
    minimum_seam_supported_fraction: float = 0.05
    maximum_seam_p50_m: float = 0.10
    maximum_seam_p90_m: float = 0.30
    maximum_seam_normal_p90_deg: float = 30.0
    maximum_seam_oriented_normal_p90_deg: float = 60.0
    maximum_seam_opposed_normal_fraction: float = 0.05
    minimum_chart_area_ratio: float = 0.80
    maximum_chart_area_ratio: float = 1.20
    maximum_face_collapse_fraction: float = 0.01
    maximum_face_flip_fraction: float = 0.001
    maximum_vertex_normal_flip_fraction: float = 0.001
    maximum_face_expansion_fraction: float = 0.01
    minimum_jacobian_singular_ratio_p05: float = 0.50
    maximum_jacobian_singular_ratio_p95: float = 2.00

    def validated(self) -> "FullSubmapGeometryGateConfig":
        if self.depth_absolute_tolerance_m <= 0 or not 0 < self.depth_relative_tolerance < 1:
            raise ValueError("invalid good-ray depth tolerance")
        if tuple(sorted(self.normal_thresholds_deg)) != self.normal_thresholds_deg:
            raise ValueError("normal thresholds must be increasing")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 0 < self.confidence < 1:
            raise ValueError("bootstrap confidence must be in (0, 1)")
        if self.maximum_seam_correspondence_m <= 0:
            raise ValueError("maximum seam correspondence must be positive")
        return self


@dataclass(frozen=True)
class _SurfaceMesh:
    names: np.ndarray
    vertex_offsets: np.ndarray
    vertices: np.ndarray
    normals: np.ndarray
    face_offsets: np.ndarray
    faces: np.ndarray


def _atlas_mesh(atlas: ExplicitChartAtlas) -> _SurfaceMesh:
    return _SurfaceMesh(
        names=np.asarray(atlas.chart_names),
        vertex_offsets=np.asarray(atlas.chart_vertex_offsets),
        vertices=np.asarray(atlas.vertices_world, np.float64),
        normals=np.asarray(atlas.normals_world, np.float64),
        face_offsets=np.asarray(atlas.chart_face_offsets),
        faces=np.asarray(atlas.faces, np.int64),
    )


def _bounded_planar_mesh(planar: GeometryNativePlanarMap) -> _SurfaceMesh:
    if planar.boundary_is_visual_summary_only:
        raise ValueError(
            "convex plane summaries are not finite-support meshes; use the "
            "exact member primitives or a source-view physical topology"
        )
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offsets = [0]
    face_offsets = [0]
    names = []
    for row in range(len(planar.plane_ids)):
        lo, hi = map(int, planar.boundary_offsets[row:row + 2])
        uv = np.asarray(planar.boundary_uv[lo:hi], np.float64)
        if len(uv) < 3:
            continue
        center = np.asarray(planar.centers_world[row], np.float64)
        frame = np.asarray(planar.frames_world[row, :2], np.float64)
        points = center + uv @ frame
        normal = np.asarray(planar.normals_world[row], np.float64)
        local_faces = np.asarray(
            [(0, index, index + 1) for index in range(1, len(points) - 1)],
            np.int64,
        )
        if len(local_faces):
            face_normal = np.cross(
                points[local_faces[0, 1]] - points[local_faces[0, 0]],
                points[local_faces[0, 2]] - points[local_faces[0, 0]],
            )
            if float(face_normal @ normal) < 0:
                local_faces = local_faces[:, [0, 2, 1]]
        local_faces += vertex_offsets[-1]
        vertices.append(points)
        normals.append(np.repeat(normal[None], len(points), axis=0))
        faces.append(local_faces)
        names.append(f"plane_{int(planar.plane_ids[row])}")
        vertex_offsets.append(vertex_offsets[-1] + len(points))
        face_offsets.append(face_offsets[-1] + len(local_faces))
    if not vertices or not faces:
        raise ValueError("bounded M0 has no triangulable finite plane")
    return _SurfaceMesh(
        names=np.asarray(names),
        vertex_offsets=np.asarray(vertex_offsets, np.int64),
        vertices=np.concatenate(vertices),
        normals=np.concatenate(normals),
        face_offsets=np.asarray(face_offsets, np.int64),
        faces=np.concatenate(faces),
    )


def _source_surface_mesh(surface: SourceOnlyBoundedSurfaceBaseline) -> _SurfaceMesh:
    return _SurfaceMesh(
        names=np.asarray(surface.chart_names),
        vertex_offsets=np.asarray(surface.chart_vertex_offsets, np.int64),
        vertices=np.asarray(surface.vertices_world, np.float64),
        normals=np.asarray(surface.normals_world, np.float64),
        face_offsets=np.asarray(surface.chart_face_offsets, np.int64),
        faces=np.asarray(surface.faces, np.int64),
    )


def _m0_mesh(
    m0: GeometryNativePlanarMap | SourceOnlyBoundedSurfaceBaseline,
) -> _SurfaceMesh:
    if isinstance(m0, SourceOnlyBoundedSurfaceBaseline):
        return _source_surface_mesh(m0)
    if isinstance(m0, GeometryNativePlanarMap):
        return _bounded_planar_mesh(m0)
    raise TypeError("unsupported M0 geometry artifact")


def _m0_decision_suffix(
    m0: GeometryNativePlanarMap | SourceOnlyBoundedSurfaceBaseline,
) -> str:
    if isinstance(m0, SourceOnlyBoundedSurfaceBaseline):
        return "M0_source_reference_surface_control"
    return "M0_source_only_2DGS_planar"


def _normalise_rows(value: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(length, 1e-15)


def _project(
    vertices: np.ndarray,
    camera_to_world: np.ndarray,
    focal_xy: np.ndarray,
    principal_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (vertices - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    pixels = np.stack(
        (
            focal_xy[0] * camera[:, 0] / z + principal_xy[0],
            focal_xy[1] * camera[:, 1] / z + principal_xy[1],
        ),
        axis=1,
    )
    valid = np.isfinite(camera).all(1) & np.isfinite(pixels).all(1) & (z > 1e-4)
    return pixels, z, valid


def _render_mesh(
    mesh: _SurfaceMesh,
    camera_to_world: np.ndarray,
    focal_xy: np.ndarray,
    principal_xy: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic CPU z-buffer used identically for M0/M1/M2."""

    pixels, z, valid_vertex = _project(
        mesh.vertices, camera_to_world, focal_xy, principal_xy,
    )
    depth = np.full((height, width), np.inf, np.float64)
    normal = np.zeros((height, width, 3), np.float64)
    vertex_normals = _normalise_rows(mesh.normals)
    for face in mesh.faces:
        if not valid_vertex[face].all():
            continue
        triangle = pixels[face]
        triangle_z = z[face]
        min_x = max(0, int(np.floor(triangle[:, 0].min())))
        max_x = min(width - 1, int(np.ceil(triangle[:, 0].max())))
        min_y = max(0, int(np.floor(triangle[:, 1].min())))
        max_y = min(height - 1, int(np.ceil(triangle[:, 1].max())))
        if min_x > max_x or min_y > max_y:
            continue
        if (max_x - min_x + 1) * (max_y - min_y + 1) > height * width:
            continue
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            continue
        yy, xx = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        inverse_z = w0 / triangle_z[0] + w1 / triangle_z[1] + w2 / triangle_z[2]
        candidate_depth = np.where(inside & (inverse_z > 0), 1.0 / inverse_z, np.inf)
        region_depth = depth[min_y:max_y + 1, min_x:max_x + 1]
        replace = candidate_depth < region_depth
        if not replace.any():
            continue
        perspective = np.stack((w0 / triangle_z[0], w1 / triangle_z[1], w2 / triangle_z[2]), axis=-1)
        perspective /= np.maximum(inverse_z[..., None], 1e-15)
        candidate_normal = np.einsum("...k,kd->...d", perspective, vertex_normals[face])
        candidate_normal = _normalise_rows(candidate_normal)
        region_normal = normal[min_y:max_y + 1, min_x:max_x + 1]
        region_depth[replace] = candidate_depth[replace]
        region_normal[replace] = candidate_normal[replace]
    return depth, normal


def _render_boundary(
    depth: np.ndarray,
    normal: np.ndarray,
    config: FullSubmapGeometryGateConfig,
) -> np.ndarray:
    valid = np.isfinite(depth)
    boundary = np.zeros_like(valid)
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        va, vb = valid[first], valid[second]
        edge = va ^ vb
        both = va & vb
        depth_delta = np.zeros_like(edge, np.float64)
        depth_delta[both] = np.abs(depth[first][both] - depth[second][both])
        depth_limit = np.maximum(
            config.boundary_depth_absolute_m,
            config.boundary_depth_relative * np.minimum(depth[first], depth[second]),
        )
        dot = np.abs(np.sum(normal[first] * normal[second], axis=2))
        normal_edge = both & (dot < np.cos(np.deg2rad(config.boundary_normal_angle_deg)))
        edge |= both & (depth_delta > depth_limit)
        edge |= normal_edge
        boundary[first] |= edge
        boundary[second] |= edge
    return boundary


def _boundary_f1(reference: np.ndarray, estimate: np.ndarray, tolerance_px: float) -> float:
    reference = np.asarray(reference, bool)
    estimate = np.asarray(estimate, bool)
    if not reference.any() and not estimate.any():
        return 1.0
    if not reference.any() or not estimate.any():
        return 0.0
    estimate_distance = distance_transform_edt(~estimate)
    reference_distance = distance_transform_edt(~reference)
    recall = float(np.mean(estimate_distance[reference] <= tolerance_px))
    precision = float(np.mean(reference_distance[estimate] <= tolerance_px))
    return 2.0 * precision * recall / max(precision + recall, 1e-15)


def _view_metrics(
    depth: np.ndarray,
    normal: np.ndarray,
    reference_depth: np.ndarray,
    reference_normal: np.ndarray,
    reference_valid: np.ndarray,
    reference_boundary: np.ndarray,
    config: FullSubmapGeometryGateConfig,
) -> dict[str, float]:
    reference_valid = np.asarray(reference_valid, bool)
    denominator = max(int(reference_valid.sum()), 1)
    rendered = np.isfinite(depth) & (depth > 0)
    paired = reference_valid & rendered
    error = np.zeros_like(reference_depth, np.float64)
    error[paired] = np.abs(depth[paired] - reference_depth[paired])
    tolerance = np.maximum(
        config.depth_absolute_tolerance_m,
        config.depth_relative_tolerance * reference_depth,
    )
    good_depth = paired & (error <= tolerance)
    relative = error[paired] / np.maximum(reference_depth[paired], 1e-8)
    result: dict[str, float] = {
        "reference_ray_count": float(reference_valid.sum()),
        "rendered_ray_recall": float(paired.sum() / denominator),
        "good_ray_recall": float(good_depth.sum() / denominator),
        "absrel_median_conditional": float(np.median(relative)) if len(relative) else float("nan"),
        "absrel_p90_conditional": float(np.quantile(relative, 0.9)) if len(relative) else float("nan"),
    }
    reference_normal_valid = reference_valid & (np.linalg.norm(reference_normal, axis=2) > 0.5)
    normal_denominator = max(int(reference_normal_valid.sum()), 1)
    estimated_normal_valid = rendered & (np.linalg.norm(normal, axis=2) > 0.5)
    normal_pair = reference_normal_valid & estimated_normal_valid
    angle = np.full(reference_valid.shape, 180.0, np.float64)
    dot = np.abs(np.sum(reference_normal[normal_pair] * normal[normal_pair], axis=1))
    angle[normal_pair] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    for threshold in config.normal_thresholds_deg:
        suffix = str(int(threshold))
        good_normal = normal_pair & (angle <= threshold)
        result[f"normal_recall_{suffix}"] = float(good_normal.sum() / normal_denominator)
        result[f"joint_depth_normal_recall_{suffix}"] = float(
            (good_depth & good_normal).sum() / normal_denominator
        )
    estimated_boundary = _render_boundary(depth, normal, config)
    # Evaluate boundary only in/around the frozen reference domain; this keeps
    # out-of-domain map geometry from dominating the metric.
    domain = distance_transform_edt(~reference_valid) <= config.boundary_tolerance_px
    result["boundary_f1"] = _boundary_f1(
        reference_boundary & domain,
        estimated_boundary & domain,
        config.boundary_tolerance_px,
    )
    return result


def _finite_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def _render_and_measure(
    mesh: _SurfaceMesh,
    rays: StrictHeldRayInventory,
    config: FullSubmapGeometryGateConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rows: list[dict[str, float]] = []
    _, height, width = rays.shape
    for view in range(rays.shape[0]):
        depth, normal = _render_mesh(
            mesh,
            rays.camera_to_world[view],
            rays.focal_xy[view],
            rays.principal_xy[view],
            height,
            width,
        )
        rows.append(
            _view_metrics(
                depth,
                normal,
                rays.reference_depth_m[view],
                rays.reference_normal_world[view],
                rays.reference_valid[view],
                rays.reference_boundary[view],
                config,
            )
        )
    keys = tuple(rows[0])
    vectors = {key: np.asarray([row[key] for row in rows], np.float64) for key in keys}
    report = {
        "view_count": len(rows),
        "macro_view": {key: _finite_summary(vectors[key]) for key in keys if key != "reference_ray_count"},
        "per_view": [
            {
                "name": str(rays.view_names[index]),
                "block_id": str(rays.block_ids[index]),
                **{
                    key: (float(value) if np.isfinite(value) else None)
                    for key, value in row.items()
                },
            }
            for index, row in enumerate(rows)
        ],
    }
    return report, vectors


def _paired_block_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    block_ids: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int | None]:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "resamples": 0,
            "common_finite_view_count": 0,
        }
    left = left[finite]
    right = right[finite]
    block_ids = np.asarray(block_ids).astype(str)[finite]
    blocks = np.unique(block_ids)
    members = [np.flatnonzero(block_ids == block) for block in blocks]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, np.float64)
    delta = right - left
    for draw in range(resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([members[index] for index in chosen])
        draws[draw] = float(np.mean(delta[indices]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(delta)),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "resamples": int(resamples),
        "common_finite_view_count": int(finite.sum()),
    }


def _mesh_face_geometry(mesh: _SurfaceMesh) -> tuple[np.ndarray, np.ndarray]:
    triangle = mesh.vertices[mesh.faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return 0.5 * length, cross / np.maximum(length[:, None], 1e-15)


def _chart_distortion(
    initial: ExplicitChartAtlas,
    aligned: ExplicitChartAtlas,
) -> dict[str, object]:
    if not np.array_equal(initial.chart_names, aligned.chart_names):
        raise ValueError("initial/aligned chart names differ")
    if not np.array_equal(initial.chart_vertex_offsets, aligned.chart_vertex_offsets):
        raise ValueError("initial/aligned chart vertex inventory differs")
    if not np.array_equal(initial.chart_face_offsets, aligned.chart_face_offsets):
        raise ValueError("initial/aligned chart face inventory differs")
    if not np.array_equal(initial.faces, aligned.faces) or not np.array_equal(initial.uv, aligned.uv):
        raise ValueError("initial/aligned chart topology or UV differs")
    initial_mesh, aligned_mesh = _atlas_mesh(initial), _atlas_mesh(aligned)
    initial_area, initial_normal = _mesh_face_geometry(initial_mesh)
    aligned_area, aligned_normal = _mesh_face_geometry(aligned_mesh)
    if np.any(initial_area <= 1e-12):
        raise ValueError("initial atlas contains collapsed face")
    raw_ratio = aligned_area / initial_area
    global_area_scale = float(np.median(raw_ratio))
    if not np.isfinite(global_area_scale) or global_area_scale <= 1e-12:
        raise ValueError("aligned atlas has no robust global area scale")
    global_linear_scale = float(np.sqrt(global_area_scale))
    ratio = raw_ratio / global_area_scale
    flip = np.sum(initial_normal * aligned_normal, axis=1) < 0
    chart_ratios = []
    for chart in range(len(aligned.chart_names)):
        lo, hi = map(int, aligned.chart_face_offsets[chart:chart + 2])
        if hi <= lo or float(initial_area[lo:hi].sum()) <= 0:
            raise ValueError("chart has no auditable initial face area")
        chart_ratios.append(
            float(aligned_area[lo:hi].sum() / initial_area[lo:hi].sum())
            / global_area_scale
        )

    # The UV Jacobian is evaluated per corresponding face.  Singular-value
    # ratios distinguish anisotropic stretch from a mere area change.
    singular_ratios = []
    for face in aligned.faces:
        uv_edge = np.stack((aligned.uv[face[1]] - aligned.uv[face[0]], aligned.uv[face[2]] - aligned.uv[face[0]]), axis=1)
        determinant = float(np.linalg.det(uv_edge))
        if abs(determinant) <= 1e-12:
            raise ValueError("atlas contains UV-collapsed face")
        inverse = np.linalg.inv(uv_edge)
        initial_edge = np.stack((initial.vertices_world[face[1]] - initial.vertices_world[face[0]], initial.vertices_world[face[2]] - initial.vertices_world[face[0]]), axis=1)
        aligned_edge = np.stack((aligned.vertices_world[face[1]] - aligned.vertices_world[face[0]], aligned.vertices_world[face[2]] - aligned.vertices_world[face[0]]), axis=1)
        initial_singular = np.linalg.svd(initial_edge @ inverse, compute_uv=False)
        aligned_singular = np.linalg.svd(aligned_edge @ inverse, compute_uv=False)
        if np.any(initial_singular <= 1e-12):
            raise ValueError("initial atlas has singular geometry Jacobian")
        singular_ratios.extend(
            (aligned_singular / initial_singular / global_linear_scale).tolist()
        )
    singular_ratios = np.asarray(singular_ratios, np.float64)
    initial_vertex_normal = _normalise_rows(np.asarray(initial.normals_world, np.float64))
    aligned_vertex_normal = _normalise_rows(np.asarray(aligned.normals_world, np.float64))
    if np.any(np.linalg.norm(initial.normals_world, axis=1) < 0.5) or np.any(
        np.linalg.norm(aligned.normals_world, axis=1) < 0.5
    ):
        raise ValueError("initial/aligned atlas contains an undefined vertex normal")
    vertex_normal_flip = np.sum(
        initial_vertex_normal * aligned_vertex_normal, axis=1
    ) < 0
    chart_vertex_normal_flip = []
    for chart in range(len(aligned.chart_names)):
        lo, hi = map(int, aligned.chart_vertex_offsets[chart : chart + 2])
        chart_vertex_normal_flip.append(float(np.mean(vertex_normal_flip[lo:hi])))
    return {
        "face_count": int(len(ratio)),
        "estimated_global_linear_scale": global_linear_scale,
        "raw_face_area_ratio_p05": float(np.quantile(raw_ratio, 0.05)),
        "raw_face_area_ratio_median": float(np.median(raw_ratio)),
        "raw_face_area_ratio_p95": float(np.quantile(raw_ratio, 0.95)),
        "distortion_ratios_remove_one_global_similarity_scale": True,
        "face_area_ratio_p05": float(np.quantile(ratio, 0.05)),
        "face_area_ratio_median": float(np.median(ratio)),
        "face_area_ratio_p95": float(np.quantile(ratio, 0.95)),
        "face_collapse_fraction": float(np.mean(ratio < 0.25)),
        "face_expansion_fraction": float(np.mean(ratio > 4.0)),
        "face_flip_fraction": float(np.mean(flip)),
        "vertex_normal_orientation_flip_fraction": float(
            np.mean(vertex_normal_flip)
        ),
        "chart_vertex_normal_orientation_flip_fraction_max": float(
            np.max(chart_vertex_normal_flip)
        ),
        "chart_area_ratio_min": float(np.min(chart_ratios)),
        "chart_area_ratio_median": float(np.median(chart_ratios)),
        "chart_area_ratio_max": float(np.max(chart_ratios)),
        "jacobian_singular_ratio_p05": float(np.quantile(singular_ratios, 0.05)),
        "jacobian_singular_ratio_median": float(np.median(singular_ratios)),
        "jacobian_singular_ratio_p95": float(np.quantile(singular_ratios, 0.95)),
    }


def _seam_audit(
    atlas: ExplicitChartAtlas,
    plan: ChartSubmapPlan,
    config: FullSubmapGeometryGateConfig,
) -> dict[str, object]:
    plan_rows = {str(name): row for row, name in enumerate(plan.chart_names)}
    atlas_rows = [plan_rows[str(name)] for name in atlas.chart_names]
    edges = []
    all_distance = []
    all_plane = []
    all_angle = []
    all_oriented_angle = []
    all_opposed = []
    supported_fractions = []
    for first in range(len(atlas.chart_names)):
        for second in range(first + 1, len(atlas.chart_names)):
            if not bool(plan.coverage_edges[atlas_rows[first], atlas_rows[second]]):
                continue
            first_lo, first_hi = map(int, atlas.chart_vertex_offsets[first:first + 2])
            second_lo, second_hi = map(int, atlas.chart_vertex_offsets[second:second + 2])
            first_points = atlas.vertices_world[first_lo:first_hi]
            second_points = atlas.vertices_world[second_lo:second_hi]
            first_normals = _normalise_rows(atlas.normals_world[first_lo:first_hi])
            second_normals = _normalise_rows(atlas.normals_world[second_lo:second_hi])

            def direction(source_points, source_normals, target_points, target_normals):
                distance, nearest = cKDTree(target_points).query(source_points, k=1)
                supported = distance <= config.maximum_seam_correspondence_m
                delta = source_points - target_points[nearest]
                plane = np.abs(np.sum(delta * target_normals[nearest], axis=1))
                signed_dot = np.sum(
                    source_normals * target_normals[nearest], axis=1
                )
                angle = np.degrees(
                    np.arccos(np.clip(np.abs(signed_dot), -1.0, 1.0))
                )
                oriented_angle = np.degrees(
                    np.arccos(np.clip(signed_dot, -1.0, 1.0))
                )
                opposed = signed_dot < 0
                return distance, plane, angle, oriented_angle, opposed, supported

            ab = direction(first_points, first_normals, second_points, second_normals)
            ba = direction(second_points, second_normals, first_points, first_normals)
            distance = np.concatenate((ab[0][ab[5]], ba[0][ba[5]]))
            plane = np.concatenate((ab[1][ab[5]], ba[1][ba[5]]))
            angle = np.concatenate((ab[2][ab[5]], ba[2][ba[5]]))
            oriented_angle = np.concatenate((ab[3][ab[5]], ba[3][ba[5]]))
            opposed = np.concatenate((ab[4][ab[5]], ba[4][ba[5]]))
            supported = float(min(ab[5].mean(), ba[5].mean()))
            frozen_overlap = float(
                plan.symmetric_surface_overlap[atlas_rows[first], atlas_rows[second]]
            )
            support_required = max(
                config.minimum_seam_supported_fraction,
                0.5 * frozen_overlap,
            )
            supported_fractions.append(supported)
            if len(distance):
                all_distance.append(distance)
                all_plane.append(plane)
                all_angle.append(angle)
                all_oriented_angle.append(oriented_angle)
                all_opposed.append(opposed)
            row = {
                    "first": str(atlas.chart_names[first]),
                    "second": str(atlas.chart_names[second]),
                    "frozen_symmetric_surface_overlap": frozen_overlap,
                    "supported_fraction_min_direction": supported,
                    "minimum_supported_fraction_required": support_required,
                    "support_pass": bool(supported >= support_required),
                    "seam_thickness_p50_m": float(np.median(distance)) if len(distance) else None,
                    "seam_thickness_p90_m": float(np.quantile(distance, 0.9)) if len(distance) else None,
                    "point_to_plane_p90_m": float(np.quantile(plane, 0.9)) if len(plane) else None,
                    "unsigned_normal_p90_deg": float(np.quantile(angle, 0.9)) if len(angle) else None,
                    "oriented_normal_p90_deg": (
                        float(np.quantile(oriented_angle, 0.9))
                        if len(oriented_angle) else None
                    ),
                    "opposed_normal_fraction": (
                        float(np.mean(opposed)) if len(opposed) else None
                    ),
                }
            row["seam_p50_pass"] = bool(
                row["seam_thickness_p50_m"] is not None
                and row["seam_thickness_p50_m"] <= config.maximum_seam_p50_m
            )
            row["seam_p90_pass"] = bool(
                row["seam_thickness_p90_m"] is not None
                and row["seam_thickness_p90_m"] <= config.maximum_seam_p90_m
            )
            row["unsigned_normal_pass"] = bool(
                row["unsigned_normal_p90_deg"] is not None
                and row["unsigned_normal_p90_deg"]
                <= config.maximum_seam_normal_p90_deg
            )
            row["oriented_normal_pass"] = bool(
                row["oriented_normal_p90_deg"] is not None
                and row["oriented_normal_p90_deg"]
                <= config.maximum_seam_oriented_normal_p90_deg
                and row["opposed_normal_fraction"] is not None
                and row["opposed_normal_fraction"]
                <= config.maximum_seam_opposed_normal_fraction
            )
            row["all_edge_geometry_pass"] = bool(
                row["support_pass"]
                and row["seam_p50_pass"]
                and row["seam_p90_pass"]
                and row["unsigned_normal_pass"]
                and row["oriented_normal_pass"]
            )
            edges.append(row)
    if not edges:
        raise ValueError("frozen selected submap has no coverage edge for seam audit")
    distance = np.concatenate(all_distance) if all_distance else np.zeros(0)
    plane = np.concatenate(all_plane) if all_plane else np.zeros(0)
    angle = np.concatenate(all_angle) if all_angle else np.zeros(0)
    oriented_angle = (
        np.concatenate(all_oriented_angle) if all_oriented_angle else np.zeros(0)
    )
    opposed = np.concatenate(all_opposed) if all_opposed else np.zeros(0, bool)
    return {
        "edge_source": "frozen_chart_submap_plan_coverage_edges_only",
        "frozen_edge_count": len(edges),
        "minimum_supported_fraction": float(np.min(supported_fractions)),
        "all_frozen_edges_support_pass": all(row["support_pass"] for row in edges),
        "all_frozen_edges_p50_pass": all(row["seam_p50_pass"] for row in edges),
        "all_frozen_edges_p90_pass": all(row["seam_p90_pass"] for row in edges),
        "all_frozen_edges_unsigned_normal_pass": all(
            row["unsigned_normal_pass"] for row in edges
        ),
        "all_frozen_edges_oriented_normal_pass": all(
            row["oriented_normal_pass"] for row in edges
        ),
        "all_frozen_edges_geometry_pass": all(
            row["all_edge_geometry_pass"] for row in edges
        ),
        "gate_uses_per_edge_thresholds_not_pooled_quantiles": True,
        "seam_thickness_p50_m": float(np.median(distance)) if len(distance) else None,
        "seam_thickness_p90_m": float(np.quantile(distance, 0.9)) if len(distance) else None,
        "point_to_plane_p90_m": float(np.quantile(plane, 0.9)) if len(plane) else None,
        "unsigned_normal_p90_deg": float(np.quantile(angle, 0.9)) if len(angle) else None,
        "oriented_normal_p90_deg": (
            float(np.quantile(oriented_angle, 0.9)) if len(oriented_angle) else None
        ),
        "opposed_normal_fraction": float(np.mean(opposed)) if len(opposed) else None,
        "per_edge": edges,
    }


def _load_authority(path: Path) -> dict[str, object]:
    authority = json.loads(Path(path).read_text())
    _replay_metadata(authority, "disjoint upstream authority")
    if authority.get("artifact_type") != "goal_maplet_disjoint_chart_upstream_authority_v2":
        raise ValueError("full-submap gate requires physically isolated v2 authority")
    for name in (
        "source_held_image_disjoint",
        "source_held_route_disjoint",
        "physical_source_held_input_roots_disjoint",
        "strict_disjoint_upstream",
    ):
        if authority.get(name) is not True:
            raise ValueError(f"disjoint authority did not certify {name}")
    if authority.get("forbidden_routes_opened") is not False or authority.get("uses_query_or_ground_truth") is not False:
        raise ValueError("disjoint authority opened forbidden route/query/GT")
    return authority


def _load_sealed_json_file(
    path: Path,
    *,
    expected_file_sha256: object,
    expected_content_sha256: object,
    label: str,
) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected_file_sha256:
        raise ValueError(f"{label} file differs from atlas provenance")
    payload = json.loads(path.read_text())
    _replay_metadata(payload, label)
    if payload.get("content_sha256") != expected_content_sha256:
        raise ValueError(f"{label} content differs from atlas provenance")
    return payload


def _validate_atlas_pair_provenance(
    *,
    label: str,
    arm: str,
    initial: ExplicitChartAtlas,
    aligned: ExplicitChartAtlas,
    comparison_domain: _ComparisonDomain,
    plan: ChartSubmapPlan,
) -> dict[str, object]:
    """Replay actual alignment/initializer manifests for one paired arm."""

    identical_keys = (
        "alignment_manifest_path",
        "alignment_manifest_file_sha256",
        "alignment_manifest_content_sha256",
        "initializer_manifest_path",
        "initializer_manifest_file_sha256",
        "initializer_manifest_content_sha256",
        "initializer_selected_file_inventory",
        "initializer_selected_file_inventory_sha256",
        "alignment_upstream_comparison_domain_content_sha256",
        "alignment_upstream_comparison_domain_file_sha256",
        "alignment_runner_contract",
        "alignment_runner_file_sha256",
        "alignment_code_inventory_sha256",
        "source_cameras_file_sha256",
        "source_charts_file_sha256",
        "scale_factor",
    )
    for key in identical_keys:
        if initial.metadata.get(key) != aligned.metadata.get(key):
            raise ValueError(f"{label} initial/aligned provenance differs for {key}")
    for key in (
        "alignment_manifest_file_sha256",
        "alignment_manifest_content_sha256",
        "initializer_manifest_file_sha256",
        "initializer_manifest_content_sha256",
        "initializer_selected_file_inventory_sha256",
        "alignment_upstream_comparison_domain_content_sha256",
        "alignment_upstream_comparison_domain_file_sha256",
        "alignment_runner_file_sha256",
        "alignment_code_inventory_sha256",
        "source_cameras_file_sha256",
        "source_charts_file_sha256",
    ):
        _require_sha(initial.metadata.get(key), f"{label} {key}")
    upstream_content = comparison_domain.metadata.get(
        "upstream_comparison_domain_content_sha256"
    )
    upstream_file = comparison_domain.metadata.get(
        "upstream_comparison_domain_file_sha256"
    )
    if (
        initial.metadata.get("alignment_upstream_comparison_domain_content_sha256")
        != upstream_content
        or initial.metadata.get("alignment_upstream_comparison_domain_file_sha256")
        != upstream_file
    ):
        raise ValueError(f"{label} alignment did not consume sealed-v2 upstream v1")
    if initial.metadata.get("alignment_runner_contract") != comparison_domain.metadata.get(
        "alignment_runner_contract"
    ):
        raise ValueError(f"{label} alignment runner contract differs")

    alignment_path = Path(str(initial.metadata.get("alignment_manifest_path", "")))
    alignment_manifest = _load_sealed_json_file(
        alignment_path,
        expected_file_sha256=initial.metadata["alignment_manifest_file_sha256"],
        expected_content_sha256=initial.metadata[
            "alignment_manifest_content_sha256"
        ],
        label=f"{label} alignment manifest",
    )
    selected_names = initial.chart_names.astype(str).tolist()
    alignment_expectations = {
        "artifact_type": "goal_maplet_masked_chart_alignment_gate_v1",
        "chart_names": selected_names,
        "comparison_domain_content_sha256": upstream_content,
        "comparison_domain_file_sha256": upstream_file,
        "frozen_submap_plan_content_sha256": plan.metadata.get("content_sha256"),
        "selection_contract": initial.metadata["alignment_runner_contract"],
        "alignment_runner_file_sha256": initial.metadata[
            "alignment_runner_file_sha256"
        ],
        "alignment_code_inventory_sha256": initial.metadata[
            "alignment_code_inventory_sha256"
        ],
        "subset_cameras_file_sha256": initial.metadata[
            "source_cameras_file_sha256"
        ],
        "charts_data_file_sha256": initial.metadata["source_charts_file_sha256"],
    }
    for key, expected in alignment_expectations.items():
        if alignment_manifest.get(key) != expected:
            raise ValueError(f"{label} alignment manifest {key} differs")
    if alignment_manifest.get("cameras_source_file_sha256") != comparison_domain.metadata.get(
        "cameras_file_sha256"
    ):
        raise ValueError(f"{label} alignment source cameras differ from domain")
    if alignment_manifest.get("uses_query_or_ground_truth") is not False:
        raise ValueError(f"{label} alignment manifest consumed query/GT")
    charts_path = alignment_path.parent / "charts_data.npz"
    cameras_path = alignment_path.parent / "cameras.json"
    if (
        not charts_path.is_file()
        or file_sha256(charts_path) != initial.metadata["source_charts_file_sha256"]
        or not cameras_path.is_file()
        or file_sha256(cameras_path) != initial.metadata["source_cameras_file_sha256"]
    ):
        raise ValueError(f"{label} alignment charts/cameras bytes differ")
    with np.load(charts_path, allow_pickle=False) as chart_data:
        scale_factor = float(np.asarray(chart_data["scale_factor"]).item())
    if not np.isfinite(scale_factor) or scale_factor <= 0 or not np.isclose(
        scale_factor, float(initial.metadata.get("scale_factor")), atol=0.0, rtol=0.0
    ):
        raise ValueError(f"{label} scale factor does not replay charts data")

    initializer_path = Path(str(initial.metadata.get("initializer_manifest_path", "")))
    initializer_manifest = _load_sealed_json_file(
        initializer_path,
        expected_file_sha256=initial.metadata["initializer_manifest_file_sha256"],
        expected_content_sha256=initial.metadata[
            "initializer_manifest_content_sha256"
        ],
        label=f"{label} initializer manifest",
    )
    prefix = "dav2" if arm == "DAV2" else "moge"
    if initial.metadata["initializer_manifest_file_sha256"] != comparison_domain.metadata.get(
        f"{prefix}_initializer_manifest_file_sha256"
    ) or initial.metadata[
        "initializer_manifest_content_sha256"
    ] != comparison_domain.metadata.get(f"{prefix}_initializer_manifest_content_sha256"):
        raise ValueError(f"{label} initializer manifest differs from sealed v2")
    rows = initializer_manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} initializer manifest lacks rows")
    row_by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict)
    }
    if len(row_by_name) != len(rows):
        raise ValueError(f"{label} initializer manifest has duplicate/invalid rows")
    selected_inventory = [
        {
            "name": name,
            "file_sha256": row_by_name[name].get("file_sha256"),
            "content_sha256": row_by_name[name].get("content_sha256"),
        }
        for name in selected_names
        if name in row_by_name
    ]
    if len(selected_inventory) != len(selected_names):
        raise ValueError(f"{label} initializer manifest lacks a selected chart")
    if initial.metadata.get("initializer_selected_file_inventory") != selected_inventory:
        raise ValueError(f"{label} selected initializer inventory differs")
    if initial.metadata.get(
        "initializer_selected_file_inventory_sha256"
    ) != canonical_json_sha256(selected_inventory):
        raise ValueError(f"{label} selected initializer inventory hash differs")
    domain_files = comparison_domain.metadata.get(f"{prefix}_initializer_file_sha256")
    alignment_files = alignment_manifest.get("initializer_file_sha256")
    if not isinstance(domain_files, dict):
        raise ValueError(f"{label} lacks per-view initializer file authority")
    if isinstance(alignment_files, list):
        if len(alignment_files) != len(selected_names):
            raise ValueError(
                f"{label} alignment initializer list is not exact official order"
            )
        alignment_files_by_name = dict(zip(selected_names, alignment_files))
    elif isinstance(alignment_files, dict):
        alignment_files_by_name = alignment_files
    else:
        raise ValueError(f"{label} lacks per-view initializer file authority")
    for row in selected_inventory:
        name = row["name"]
        if (
            row["file_sha256"] != domain_files.get(name)
            or row["file_sha256"] != alignment_files_by_name.get(name)
        ):
            raise ValueError(f"{label} initializer file differs for {name}")
    return {
        key: initial.metadata[key]
        for key in identical_keys
        if key != "initializer_selected_file_inventory"
    }


def _validate_lineage(
    *,
    authority_path: Path,
    authority: Mapping[str, object],
    comparison_domain_path: Path,
    comparison_domain: _ComparisonDomain,
    rays: StrictHeldRayInventory,
    m0: GeometryNativePlanarMap | SourceOnlyBoundedSurfaceBaseline,
    m1_initial: ExplicitChartAtlas,
    m1: ExplicitChartAtlas,
    m2_initial: ExplicitChartAtlas,
    m2: ExplicitChartAtlas,
    plan: ChartSubmapPlan,
) -> dict[str, object]:
    authority_file_hash = file_sha256(authority_path)
    authority_content_hash = _require_sha(authority.get("content_sha256"), "authority content hash")
    if rays.metadata["disjoint_upstream_authority_file_sha256"] != authority_file_hash:
        raise ValueError("held rays bind another authority file")
    if rays.metadata["disjoint_upstream_authority_content_sha256"] != authority_content_hash:
        raise ValueError("held rays bind another authority content")
    held = authority.get("held")
    if not isinstance(held, dict):
        raise ValueError("authority lacks held record")
    if rays.view_names.astype(str).tolist() != [str(name) for name in held.get("ordered_names", [])]:
        raise ValueError("held ray inventory is not the exact authority inventory")
    if rays.metadata["held_pointmap_inventory_sha256"] != held.get("pointmap_inventory_sha256"):
        raise ValueError("held ray inventory pointmaps differ from authority")
    source = authority.get("source")
    if not isinstance(source, dict):
        raise ValueError("authority lacks source record")
    source_names_hash = canonical_json_sha256(
        [str(name) for name in source.get("ordered_names", [])]
    )
    if rays.metadata.get("mapping_source_ordered_names_sha256") != source_names_hash:
        raise ValueError("held rays bind another mapping-source inventory")
    coordinate_cameras = _require_sha(
        authority.get("posed_colmap_cameras_file_sha256"),
        "authority coordinate-camera hash",
    )
    coordinate_images = _require_sha(
        authority.get("posed_colmap_images_file_sha256"),
        "authority coordinate-image hash",
    )
    if (
        rays.metadata.get("coordinate_cameras_file_sha256") != coordinate_cameras
        or rays.metadata.get("coordinate_images_file_sha256") != coordinate_images
    ):
        raise ValueError("held rays bind another world-coordinate authority")

    _replay_metadata(m0.metadata, "M0 bounded map")
    if m0.metadata.get("uses_query_pose_or_ground_truth") is not False:
        raise ValueError("M0 did not explicitly reject query pose/GT")
    if m0.metadata.get("comparison_budget") != "exact_frozen_source_ordered_pool":
        raise ValueError("M0 is not an equal source-budget main-table baseline")
    if isinstance(m0, SourceOnlyBoundedSurfaceBaseline):
        if m0.metadata.get("comparison_role") != (
            "equal_budget_main_table_source_reference_surface_control"
        ):
            raise ValueError("source reference M0 has the wrong comparison role")
    elif (
        m0.metadata.get("comparison_role")
        != "equal_budget_main_table_source_only_2dgs_planar"
        or m0.metadata.get("source_2dgs_training_inventory_exact_authority_source")
        is not True
    ):
        raise ValueError(
            "2DGS M0 lacks a certified source-only equal-budget training inventory"
        )
    if m0.metadata.get("disjoint_upstream_authority_content_sha256") != authority_content_hash:
        raise ValueError("M0 binds another disjoint source/held authority")
    if m0.metadata.get("mapping_source_ordered_names_sha256") != source_names_hash:
        raise ValueError("M0 mapping input pool differs from M1/M2")
    if m0.metadata.get("held_mapping_images_consumed") is not False:
        raise ValueError("M0 consumed held mapping images")
    if m0.metadata.get("outside_frozen_source_mapping_images_consumed") is not False:
        raise ValueError("M0 consumed images outside the frozen source pool")
    formal_stride = int(
        comparison_domain.metadata.get("full_submap_gate_primary_stride", 0)
    )
    if formal_stride != FORMAL_GATE_PRIMARY_STRIDE:
        raise ValueError("comparison domain primary topology stride differs")
    if int(m0.metadata.get("stride", 0)) != formal_stride:
        raise ValueError("M0 does not use the formal primary topology stride")
    bounded_hash = _require_sha(m0.metadata.get("bounded_submap_content_sha256"), "M0 bounded-submap hash")
    comparison_hash = _require_sha(
        comparison_domain.metadata.get("content_sha256"),
        "comparison-domain content hash",
    )
    comparison_file_hash = file_sha256(comparison_domain_path)
    if m0.metadata.get("comparison_domain_content_sha256") != comparison_hash:
        raise ValueError("M0 binds another common comparison domain")
    if m0.metadata.get("comparison_domain_file_sha256") != comparison_file_hash:
        raise ValueError("M0 binds another comparison-domain file")
    if rays.metadata.get("comparison_domain_content_sha256") != comparison_hash:
        raise ValueError("held rays bind another comparison-domain content")
    if rays.metadata.get("comparison_domain_file_sha256") != comparison_file_hash:
        raise ValueError("held rays bind another comparison-domain file")
    if comparison_domain.metadata.get("disjoint_upstream_authority_file_sha256") != authority_file_hash:
        raise ValueError("comparison domain binds another authority file")
    if comparison_domain.metadata.get("disjoint_upstream_authority_content_sha256") != authority_content_hash:
        raise ValueError("comparison domain binds another authority content")
    if comparison_domain.metadata.get("mapping_source_ordered_names_sha256") != source_names_hash:
        raise ValueError("comparison domain binds another mapping-source inventory")
    if rays.metadata.get("bounded_submap_content_sha256") != bounded_hash:
        raise ValueError("M0 and held rays use different bounded submaps")
    bounds_min = np.asarray(rays.metadata["bounded_submap_min_world"], np.float64)
    bounds_max = np.asarray(rays.metadata["bounded_submap_max_world"], np.float64)
    if not (
        np.array_equal(np.asarray(m0.metadata.get("bounded_submap_min_world"), np.float64), bounds_min)
        and np.array_equal(np.asarray(m0.metadata.get("bounded_submap_max_world"), np.float64), bounds_max)
    ):
        raise ValueError("M0 and held rays use different bounded-submap boxes")
    if bounded_hash != bounded_submap_content_sha256(bounds_min, bounds_max):
        raise ValueError("M0 bounded-submap hash does not replay its box")
    if (
        m0.metadata.get("coordinate_cameras_file_sha256") != coordinate_cameras
        or m0.metadata.get("coordinate_images_file_sha256") != coordinate_images
    ):
        raise ValueError("M0 binds another world-coordinate authority")

    for label, atlas in (
        ("M1 initial", m1_initial), ("M1", m1),
        ("M2 initial", m2_initial), ("M2", m2),
    ):
        _replay_metadata(atlas.metadata, label)
        if atlas.metadata.get("uses_query_or_ground_truth") is not False:
            raise ValueError(f"{label} did not explicitly reject query/GT")
        if atlas.metadata.get("disjoint_upstream_authority_content_sha256") != authority_content_hash:
            raise ValueError(f"{label} binds another disjoint authority")
        if atlas.metadata.get("mapping_source_ordered_names_sha256") != source_names_hash:
            raise ValueError(f"{label} mapping input pool differs")
        if atlas.metadata.get("held_mapping_images_consumed") is not False:
            raise ValueError(f"{label} consumed held mapping images")
        if atlas.metadata.get("outside_frozen_source_mapping_images_consumed") is not False:
            raise ValueError(f"{label} consumed images outside frozen source")
        if atlas.metadata.get("comparison_domain_content_sha256") != comparison_hash:
            raise ValueError(f"{label} binds another common comparison domain")
        if atlas.metadata.get("comparison_domain_file_sha256") != comparison_file_hash:
            raise ValueError(f"{label} binds another comparison-domain file")
        if atlas.metadata.get("bounded_submap_content_sha256") != bounded_hash:
            raise ValueError(f"{label} binds another bounded submap")
        if not (
            np.array_equal(np.asarray(atlas.metadata.get("bounded_submap_min_world"), np.float64), bounds_min)
            and np.array_equal(np.asarray(atlas.metadata.get("bounded_submap_max_world"), np.float64), bounds_max)
        ):
            raise ValueError(f"{label} binds another bounded-submap box")
        if atlas.metadata.get("paired_common_face_inventory") is not True:
            raise ValueError(f"{label} lacks paired common face inventory")
        if (
            atlas.metadata.get("coordinate_cameras_file_sha256") != coordinate_cameras
            or atlas.metadata.get("coordinate_images_file_sha256") != coordinate_images
        ):
            raise ValueError(f"{label} binds another world-coordinate authority")
    if m1_initial.metadata.get("is_pre_alignment_geometry") is not True or m2_initial.metadata.get("is_pre_alignment_geometry") is not True:
        raise ValueError("initial atlases are not explicitly pre-alignment geometry")
    if m1.metadata.get("is_pre_alignment_geometry") is not False or m2.metadata.get("is_pre_alignment_geometry") is not False:
        raise ValueError("aligned atlases are not explicitly post-alignment geometry")
    if m1.metadata.get("initializer_arm") != "DAV2" or m1_initial.metadata.get("initializer_arm") != "DAV2":
        raise ValueError("M1 is not the frozen DAV2 arm")
    if m2.metadata.get("initializer_arm") != "MoGe3" or m2_initial.metadata.get("initializer_arm") != "MoGe3":
        raise ValueError("M2 is not the frozen MoGe3 arm")
    if not (
        np.array_equal(m1.chart_names, m2.chart_names)
        and np.array_equal(m1.chart_vertex_offsets, m2.chart_vertex_offsets)
        and np.array_equal(m1.chart_face_offsets, m2.chart_face_offsets)
        and np.array_equal(m1.faces, m2.faces)
        and np.array_equal(m1.uv, m2.uv)
    ):
        raise ValueError("M1/M2 do not share the exact chart/face/UV domain")
    margin = 1e-6
    for label, vertices in (
        ("M0", _m0_mesh(m0).vertices),
        ("M1 initial", m1_initial.vertices_world),
        ("M1", m1.vertices_world),
        ("M2 initial", m2_initial.vertices_world),
        ("M2", m2.vertices_world),
    ):
        if np.any(vertices < bounds_min - margin) or np.any(vertices > bounds_max + margin):
            raise ValueError(f"{label} geometry escapes the frozen bounded submap")

    if plan.metadata.get("uses_query_or_ground_truth") is not False:
        raise ValueError("frozen submap plan did not reject query/GT")
    if plan.metadata.get("comparison_inventory_eligible") is not True or plan.metadata.get("system_control_only") is not False:
        raise ValueError("submap plan is not the model-neutral frozen primary inventory")
    plan_lineage = plan.metadata.get("lineage")
    if not isinstance(plan_lineage, dict):
        raise ValueError("submap plan lacks strict source-only lineage")
    if plan_lineage.get("comparison_inventory_eligible") is not True:
        raise ValueError("submap plan lineage is not primary-comparison eligible")
    if plan_lineage.get("system_control_only") is not False:
        raise ValueError("submap plan lineage is a system control")
    if plan_lineage.get("query_or_ground_truth_consumed") is not False:
        raise ValueError("submap selector lineage consumed query/GT")
    if plan_lineage.get("held_root_opened_by_selector") is not False:
        raise ValueError("submap selector opened held geometry")
    if plan_lineage.get("disjoint_authority_content_sha256") != authority_content_hash:
        raise ValueError("submap plan binds another disjoint authority")
    if plan_lineage.get("source_ordered_names_sha256") != source_names_hash:
        raise ValueError("submap plan binds another mapping-source inventory")
    # The selector is deliberately frozen before either initializer and before
    # the held ray table are built.  It must bind the disjoint source authority
    # but must *not* be retroactively coupled to a comparison-domain hash.
    selected_names = [str(name) for name in plan.selected_chart_names_in_order]
    if selected_names != m1.chart_names.astype(str).tolist():
        raise ValueError("atlas inventory is not exactly the frozen selected submap")
    plan_content_hash = _require_sha(
        plan.metadata.get("content_sha256"), "frozen submap plan content hash",
    )
    if comparison_domain.metadata.get("frozen_submap_plan_content_sha256") != plan_content_hash:
        raise ValueError("comparison domain binds another frozen submap plan")
    if rays.metadata.get("frozen_submap_plan_content_sha256") != plan_content_hash:
        raise ValueError("held rays bind another frozen submap plan")
    if m0.metadata.get("frozen_submap_plan_content_sha256") != plan_content_hash:
        raise ValueError("M0 binds another frozen submap plan")
    if comparison_domain.chart_names.astype(str).tolist() != selected_names:
        raise ValueError("comparison domain is not exactly the frozen selected submap")
    for label, atlas in (
        ("M1 initial", m1_initial), ("M1", m1),
        ("M2 initial", m2_initial), ("M2", m2),
    ):
        if atlas.chart_names.astype(str).tolist() != selected_names:
            raise ValueError(f"{label} chart order is not the frozen selection-rank order")
        if atlas.metadata.get("frozen_submap_plan_content_sha256") != plan_content_hash:
            raise ValueError(f"{label} binds another frozen submap plan")
        stride = int(atlas.metadata.get("stride", 0))
        if stride != formal_stride:
            raise ValueError(
                f"{label} does not use the formal primary topology stride"
            )
        expected_vertices, expected_uv, expected_face_offsets, expected_faces = (
            _expected_comparison_topology(comparison_domain, stride)
        )
        if not np.array_equal(atlas.chart_vertex_offsets, expected_vertices):
            raise ValueError(f"{label} vertex inventory does not exhaust the frozen common domain")
        if not np.allclose(atlas.uv, expected_uv, atol=1e-7, rtol=0.0):
            raise ValueError(f"{label} UV does not replay the frozen common domain")
        if not np.array_equal(atlas.chart_face_offsets, expected_face_offsets):
            raise ValueError(f"{label} face offsets do not replay the frozen common domain")
        if not np.array_equal(atlas.faces, expected_faces):
            raise ValueError(f"{label} faces do not replay the frozen common domain")
    if isinstance(m0, SourceOnlyBoundedSurfaceBaseline):
        stride = int(m0.metadata.get("stride", 0))
        if stride != formal_stride:
            raise ValueError(
                "source reference M0 does not use the formal primary topology stride"
            )
        expected_vertices, _, expected_face_offsets, expected_faces = (
            _expected_comparison_topology(comparison_domain, stride)
        )
        if m0.chart_names.astype(str).tolist() != selected_names:
            raise ValueError("source reference M0 chart order differs from frozen plan")
        if not np.array_equal(m0.chart_vertex_offsets, expected_vertices):
            raise ValueError("source reference M0 vertex inventory differs")
        if not np.array_equal(m0.chart_face_offsets, expected_face_offsets):
            raise ValueError("source reference M0 face offsets differ")
        if not np.array_equal(m0.faces, expected_faces):
            raise ValueError("source reference M0 faces differ")
    provenance = {
        "M1": _validate_atlas_pair_provenance(
            label="M1 DAV2",
            arm="DAV2",
            initial=m1_initial,
            aligned=m1,
            comparison_domain=comparison_domain,
            plan=plan,
        ),
        "M2": _validate_atlas_pair_provenance(
            label="M2 MoGe3",
            arm="MoGe3",
            initial=m2_initial,
            aligned=m2,
            comparison_domain=comparison_domain,
            plan=plan,
        ),
    }
    for key in (
        "alignment_runner_contract",
        "alignment_runner_file_sha256",
        "alignment_code_inventory_sha256",
    ):
        if provenance["M1"].get(key) != provenance["M2"].get(key):
            raise ValueError(f"M1/M2 alignment provenance differs for {key}")
    selected = np.flatnonzero(plan.selected_mask)
    if not np.any(plan.coverage_edges[np.ix_(selected, selected)]):
        raise ValueError("selected submap has no frozen coverage edge")
    return {
        "disjoint_upstream_authority_content_sha256": authority_content_hash,
        "comparison_domain_content_sha256": comparison_hash,
        "comparison_domain_file_sha256": comparison_file_hash,
        "frozen_submap_plan_content_sha256": plan_content_hash,
        "bounded_submap_content_sha256": bounded_hash,
        "mapping_source_ordered_names_sha256": source_names_hash,
        "atlas_provenance": provenance,
    }


def _integrity_pass(
    distortion: Mapping[str, object],
    seam: Mapping[str, object],
    config: FullSubmapGeometryGateConfig,
) -> tuple[bool, list[str]]:
    failures = []
    checks = (
        (float(distortion["chart_area_ratio_min"]) >= config.minimum_chart_area_ratio, "chart_area_ratio_min"),
        (float(distortion["chart_area_ratio_max"]) <= config.maximum_chart_area_ratio, "chart_area_ratio_max"),
        (float(distortion["face_collapse_fraction"]) <= config.maximum_face_collapse_fraction, "face_collapse_fraction"),
        (float(distortion["face_expansion_fraction"]) <= config.maximum_face_expansion_fraction, "face_expansion_fraction"),
        (float(distortion["face_flip_fraction"]) <= config.maximum_face_flip_fraction, "face_flip_fraction"),
        (
            float(distortion["chart_vertex_normal_orientation_flip_fraction_max"])
            <= config.maximum_vertex_normal_flip_fraction,
            "vertex_normal_orientation_flip_fraction",
        ),
        (float(distortion["jacobian_singular_ratio_p05"]) >= config.minimum_jacobian_singular_ratio_p05, "jacobian_singular_ratio_p05"),
        (float(distortion["jacobian_singular_ratio_p95"]) <= config.maximum_jacobian_singular_ratio_p95, "jacobian_singular_ratio_p95"),
        (seam["all_frozen_edges_support_pass"] is True, "seam_supported_fraction"),
        (seam["all_frozen_edges_p50_pass"] is True, "per_edge_seam_p50"),
        (seam["all_frozen_edges_p90_pass"] is True, "per_edge_seam_p90"),
        (
            seam["all_frozen_edges_unsigned_normal_pass"] is True,
            "per_edge_seam_unsigned_normal",
        ),
        (
            seam["all_frozen_edges_oriented_normal_pass"] is True,
            "per_edge_seam_oriented_normal",
        ),
    )
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return not failures, failures


def _relative_noninferiority_pass(
    comparison: Mapping[str, Mapping[str, float | int | None]],
    config: FullSubmapGeometryGateConfig,
) -> tuple[bool, list[str]]:
    failures = []
    requirements = (
        ("good_ray_recall", "lower", -config.good_ray_noninferiority),
        ("joint_depth_normal_recall_20", "lower", -config.normal20_noninferiority),
    )
    for metric, bound, threshold in requirements:
        value = comparison[metric][bound]
        if value is None or float(value) < threshold:
            failures.append(metric)
    return not failures, failures


def _absolute_held_coverage_pass(
    absolute_report: Mapping[str, object],
    config: FullSubmapGeometryGateConfig,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    macro = absolute_report.get("macro_view")
    if not isinstance(macro, dict):
        raise ValueError("arm report lacks macro-view absolute metrics")
    absolute_requirements = (
        ("good_ray_recall", config.minimum_absolute_good_ray_recall),
        (
            "joint_depth_normal_recall_20",
            config.minimum_absolute_joint_depth_normal_recall_20,
        ),
    )
    for metric, threshold in absolute_requirements:
        summary = macro.get(metric)
        value = summary.get("mean") if isinstance(summary, dict) else None
        if value is None or float(value) < threshold:
            failures.append(metric)
    return not failures, failures


def evaluate_full_submap_geometry_gate(
    *,
    authority_path: Path,
    authority: Mapping[str, object],
    comparison_domain_path: Path,
    comparison_domain: _ComparisonDomain,
    rays: StrictHeldRayInventory,
    m0: GeometryNativePlanarMap | SourceOnlyBoundedSurfaceBaseline,
    m1_initial: ExplicitChartAtlas,
    m1: ExplicitChartAtlas,
    m2_initial: ExplicitChartAtlas,
    m2: ExplicitChartAtlas,
    plan: ChartSubmapPlan,
    config: FullSubmapGeometryGateConfig,
) -> dict[str, object]:
    """Validate, render, compare, bootstrap, and emit deterministic decisions."""

    config = config.validated()
    rays = rays.validated()
    lineage = _validate_lineage(
        authority_path=authority_path,
        authority=authority,
        comparison_domain_path=comparison_domain_path,
        comparison_domain=comparison_domain,
        rays=rays,
        m0=m0,
        m1_initial=m1_initial,
        m1=m1,
        m2_initial=m2_initial,
        m2=m2,
        plan=plan,
    )
    reports = {}
    vectors = {}
    for name, mesh in (
        ("M0", _m0_mesh(m0)),
        ("M1", _atlas_mesh(m1)),
        ("M2", _atlas_mesh(m2)),
    ):
        reports[name], vectors[name] = _render_and_measure(mesh, rays, config)
    metrics = (
        "good_ray_recall",
        "rendered_ray_recall",
        "absrel_median_conditional",
        "absrel_p90_conditional",
        "normal_recall_10",
        "normal_recall_20",
        "normal_recall_30",
        "joint_depth_normal_recall_10",
        "joint_depth_normal_recall_20",
        "joint_depth_normal_recall_30",
        "boundary_f1",
    )
    comparisons: dict[str, dict[str, object]] = {}
    for left, right in (("M0", "M1"), ("M0", "M2"), ("M1", "M2")):
        key = f"{right}_minus_{left}"
        comparisons[key] = {
            metric: _paired_block_bootstrap(
                vectors[left][metric],
                vectors[right][metric],
                rays.block_ids,
                resamples=config.bootstrap_resamples,
                seed=config.bootstrap_seed + index,
                confidence=config.confidence,
            )
            for index, metric in enumerate(metrics)
        }
    distortion = {
        "M1": _chart_distortion(m1_initial, m1),
        "M2": _chart_distortion(m2_initial, m2),
    }
    seams = {
        "M1": _seam_audit(m1, plan, config),
        "M2": _seam_audit(m2, plan, config),
    }
    integrity = {}
    for arm in ("M1", "M2"):
        passed, failures = _integrity_pass(distortion[arm], seams[arm], config)
        integrity[arm] = {"decision": "GO" if passed else "KILL", "failures": failures}
    relative = {}
    for key, comparison_key in (
        ("M1_vs_M0", "M1_minus_M0"),
        ("M2_vs_M0", "M2_minus_M0"),
        ("M2_vs_M1", "M2_minus_M1"),
    ):
        passed, failures = _relative_noninferiority_pass(
            comparisons[comparison_key], config,
        )
        relative[key] = {
            "decision": "GO" if passed else "KILL",
            "failures": failures,
            "semantics": "paired_block_bootstrap_CI_noninferiority_only",
        }
    absolute = {}
    for arm in ("M0", "M1", "M2"):
        passed, failures = _absolute_held_coverage_pass(reports[arm], config)
        absolute[arm] = {
            "decision": "GO" if passed else "KILL",
            "failures": failures,
            "minimum_good_ray_recall": config.minimum_absolute_good_ray_recall,
            "minimum_joint_depth_normal_recall_20": (
                config.minimum_absolute_joint_depth_normal_recall_20
            ),
        }
    composite = {}
    for arm in ("M1", "M2"):
        component_decisions = {
            "relative_noninferiority": relative[f"{arm}_vs_M0"]["decision"],
            "absolute_held_coverage": absolute[arm]["decision"],
            "geometry_integrity": integrity[arm]["decision"],
        }
        failures = [
            name for name, decision in component_decisions.items()
            if decision != "GO"
        ]
        composite[arm] = {
            "decision": "GO" if not failures else "KILL",
            "failures": failures,
            "component_decisions": component_decisions,
        }
    dominance_failures = []
    dominance_requirements = (
        ("good_ray_recall", "lower", lambda value: value > 0.0),
        ("joint_depth_normal_recall_20", "lower", lambda value: value >= 0.0),
    )
    for metric, bound, predicate in dominance_requirements:
        value = comparisons["M2_minus_M1"][metric][bound]
        if value is None or not predicate(float(value)):
            dominance_failures.append(metric)
    at_least_one = any(row["decision"] == "GO" for row in composite.values())
    both = all(row["decision"] == "GO" for row in composite.values())
    m0_suffix = _m0_decision_suffix(m0)
    result: dict[str, object] = {
        "artifact_type": REPORT_SCHEMA,
        "input_contract": "PASS",
        "scientific_performance_conclusion": "EVALUATED",
        "reference_semantics": rays.metadata["reference_semantics"],
        "primary_metric": "macro_view_good_ray_recall_missing_rendered_ray_is_failure",
        "conditional_error_is_not_primary": True,
        "conditional_absrel_role": "diagnostic_on_common_finite_views_only",
        "boundary_f1_role": "diagnostic_only_geometry_edges_excluding_confidence_mask_edges",
        "absolute_completeness_floor_required": True,
        "M0_comparison_role": str(m0.metadata["comparison_role"]),
        "M0_is_2DGS": isinstance(m0, GeometryNativePlanarMap),
        "full_train_2DGS_may_enter_primary_table": False,
        "lineage": lineage,
        "config": asdict(config),
        "representations": reports,
        "paired_block_bootstrap": comparisons,
        "distortion": distortion,
        "overlap_seams": seams,
        "geometry_integrity": integrity,
        "decisions": {
            "relative_noninferiority": {
                f"M1_DAV2_atlas_vs_{m0_suffix}": relative["M1_vs_M0"],
                f"M2_MoGe3_atlas_vs_{m0_suffix}": relative["M2_vs_M0"],
                "M2_MoGe3_vs_M1_DAV2": relative["M2_vs_M1"],
            },
            "relative_strict_dominance": {
                "decision": "GO" if not dominance_failures else "KILL",
                "failures": dominance_failures,
            },
            "absolute_held_coverage": absolute,
            "geometry_integrity": integrity,
            "composite_bounded_gate": {
                f"M1_DAV2_atlas_vs_{m0_suffix}": composite["M1"],
                f"M2_MoGe3_atlas_vs_{m0_suffix}": composite["M2"],
                "chart_atlas_vs_M0": "GO" if at_least_one else "KILL",
                "initializer_independent_chart_atlas_vs_M0": (
                    "GO" if both else "KILL"
                ),
            },
        },
        "uses_query_or_ground_truth": False,
        "bounded_gate_eligible": bool(both),
        "production_eligible": False,
        "production_blockers": [
            "single_route_four_temporal_blocks_are_not_deployment_coverage",
            "held_MASt3R_reference_is_mapping_diagnostic_not_sensor_depth_ground_truth",
        ],
    }
    result["content_sha256"] = canonical_json_sha256(result)
    return result


def evaluate_full_submap_geometry_gate_from_paths(
    *,
    authority_path: Path,
    comparison_domain_path: Path,
    held_ray_inventory_path: Path,
    m0_bounded_map_path: Path,
    m1_initial_atlas_path: Path,
    m1_atlas_path: Path,
    m2_initial_atlas_path: Path,
    m2_atlas_path: Path,
    frozen_submap_plan_path: Path,
    config: FullSubmapGeometryGateConfig,
) -> dict[str, object]:
    paths = {
        "disjoint_upstream_authority": Path(authority_path),
        "comparison_domain": Path(comparison_domain_path),
        "held_ray_inventory": Path(held_ray_inventory_path),
        "M0_bounded_map": Path(m0_bounded_map_path),
        "M1_initial_atlas": Path(m1_initial_atlas_path),
        "M1_atlas": Path(m1_atlas_path),
        "M2_initial_atlas": Path(m2_initial_atlas_path),
        "M2_atlas": Path(m2_atlas_path),
        "frozen_submap_plan": Path(frozen_submap_plan_path),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("missing strict gate inputs: " + ", ".join(missing))
    authority = _load_authority(paths["disjoint_upstream_authority"])
    with np.load(paths["M0_bounded_map"], allow_pickle=False) as m0_data:
        m0_metadata = json.loads(str(m0_data["metadata_json"].item()))
    m0 = (
        SourceOnlyBoundedSurfaceBaseline.load_npz(paths["M0_bounded_map"])
        if m0_metadata.get("artifact_type") == SOURCE_SURFACE_SCHEMA
        else GeometryNativePlanarMap.load_npz(paths["M0_bounded_map"])
    )
    result = evaluate_full_submap_geometry_gate(
        authority_path=paths["disjoint_upstream_authority"],
        authority=authority,
        comparison_domain_path=paths["comparison_domain"],
        comparison_domain=_ComparisonDomain.load_npz(paths["comparison_domain"]),
        rays=StrictHeldRayInventory.load_npz(paths["held_ray_inventory"]),
        m0=m0,
        m1_initial=ExplicitChartAtlas.load_npz(paths["M1_initial_atlas"]),
        m1=ExplicitChartAtlas.load_npz(paths["M1_atlas"]),
        m2_initial=ExplicitChartAtlas.load_npz(paths["M2_initial_atlas"]),
        m2=ExplicitChartAtlas.load_npz(paths["M2_atlas"]),
        plan=ChartSubmapPlan.load_npz(paths["frozen_submap_plan"]),
        config=config,
    )
    result["input_files"] = {
        name: {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_json_sha256(result)
    return result


def blocked_input_report(error: Exception, input_paths: Mapping[str, Path]) -> dict[str, object]:
    """Machine-readable fail-closed report; it is not a performance result."""

    report: dict[str, object] = {
        "artifact_type": REPORT_SCHEMA,
        "input_contract": "FAIL_CLOSED",
        "scientific_performance_conclusion": "NOT_EVALUATED",
        "decisions": {
            "relative_noninferiority": "KILL_INPUT_CONTRACT",
            "relative_strict_dominance": "KILL_INPUT_CONTRACT",
            "absolute_held_coverage": "KILL_INPUT_CONTRACT",
            "geometry_integrity": "KILL_INPUT_CONTRACT",
            "composite_bounded_gate": "KILL_INPUT_CONTRACT",
        },
        "blocking_error_type": type(error).__name__,
        "blocking_error": str(error),
        "input_presence": {
            name: {"path": str(Path(path)), "exists": Path(path).is_file()}
            for name, path in input_paths.items()
        },
        "uses_query_or_ground_truth": False,
        "production_eligible": False,
        "bounded_gate_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report


__all__ = [
    "BOUNDARY_SEMANTICS",
    "FullSubmapGeometryGateConfig",
    "SOURCE_SURFACE_SCHEMA",
    "SourceOnlyBoundedSurfaceBaseline",
    "StrictHeldRayInventory",
    "blocked_input_report",
    "bounded_submap_content_sha256",
    "evaluate_full_submap_geometry_gate",
    "evaluate_full_submap_geometry_gate_from_paths",
]
