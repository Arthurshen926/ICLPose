"""Source-only, topology-bound seam correspondences for explicit charts.

The full-submap evaluator historically recomputed nearest neighbours on each
candidate atlas.  For partially overlapping charts this mixes true overlap
with unrelated surfaces that merely lie within a broad metric radius.  This
module freezes the *identity* of seam samples from the isolated mapping-source
reference before any held geometry is opened.

Euclidean distance is used only to establish a source-reference association.
Once an association is frozen, surface agreement is measured with symmetric
point-to-plane and normal residuals.  This distinction is important for two
samples of the same plane whose pixel lattices have a tangential phase offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .chart_comparison_domain import (
    BASE_ARRAY_NAMES,
    source_tree_sha256,
    topology_array_names,
    validate_exact_topology_arrays,
)
from .chart_submap_selection import ChartSubmapPlan
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256


AUTHORITY_SCHEMA = "goal_maplet_source_seam_correspondence_authority_v1"
REPORT_SCHEMA = "goal_maplet_source_seam_geometry_gate_v1"
PHYSICAL_DOMAIN_SCHEMA = "goal_maplet_chart_comparison_domain_v3"
DISJOINT_AUTHORITY_SCHEMA = "goal_maplet_disjoint_chart_upstream_authority_v2"


@dataclass(frozen=True)
class SourceSeamConfig:
    """Frozen thresholds inherited from the selector and geometry gate."""

    topology_stride: int = 4
    maximum_reference_association_m: float = 0.30
    maximum_reference_unsigned_normal_angle_deg: float = 35.0
    minimum_correspondences_per_direction: int = 4
    minimum_supported_fraction: float = 0.05
    frozen_overlap_support_fraction_multiplier: float = 0.50
    maximum_point_to_plane_p50_m: float = 0.10
    maximum_point_to_plane_p90_m: float = 0.30
    maximum_unsigned_normal_p90_deg: float = 30.0
    maximum_oriented_normal_p90_deg: float = 60.0
    maximum_opposed_normal_fraction: float = 0.05

    def validated(self) -> "SourceSeamConfig":
        if self.topology_stride != 4:
            raise ValueError("formal source-seam authority requires stride 4")
        if self.maximum_reference_association_m <= 0:
            raise ValueError("source seam association radius must be positive")
        if not 0 < self.maximum_reference_unsigned_normal_angle_deg <= 90:
            raise ValueError("source seam association normal angle is invalid")
        if self.minimum_correspondences_per_direction < 1:
            raise ValueError("source seam needs a positive correspondence count")
        if not 0 < self.minimum_supported_fraction <= 1:
            raise ValueError("source seam minimum support is invalid")
        if not 0 < self.frozen_overlap_support_fraction_multiplier <= 1:
            raise ValueError("source seam overlap multiplier is invalid")
        if not 0 < self.maximum_point_to_plane_p50_m <= self.maximum_point_to_plane_p90_m:
            raise ValueError("source seam point-to-plane thresholds are invalid")
        if not 0 < self.maximum_unsigned_normal_p90_deg <= 90:
            raise ValueError("source seam unsigned normal threshold is invalid")
        if not self.maximum_unsigned_normal_p90_deg <= self.maximum_oriented_normal_p90_deg <= 180:
            raise ValueError("source seam oriented normal threshold is invalid")
        if not 0 <= self.maximum_opposed_normal_fraction <= 1:
            raise ValueError("source seam opposed-normal threshold is invalid")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _replay_metadata(metadata: Mapping[str, object], label: str) -> str:
    content = dict(metadata)
    claimed = content.pop("content_sha256", None)
    if not _is_sha256(claimed) or claimed != canonical_json_sha256(content):
        raise ValueError(f"{label} metadata content hash differs")
    return str(claimed)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("seam vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("seam faces must have shape (F, 3)")
    if np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("seam face leaves the vertex inventory")
    face_normal = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normal = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normal, faces[:, corner], face_normal)
    length = np.linalg.norm(normal, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-10):
        raise ValueError("seam topology contains a vertex without a finite normal")
    return normal / length[:, None]


def _chart_rows(offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, np.int64)
    return np.concatenate(
        [
            np.full(int(offsets[row + 1] - offsets[row]), row, np.int32)
            for row in range(len(offsets) - 1)
        ]
    )


def _closest_points_on_triangle_mesh(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact point-to-triangle closest locations and barycentrics.

    Every point is compared with every target triangle.  Chunking bounds the
    temporary ``points x faces`` arrays without introducing an approximate
    face-candidate search whose misses would become part of the authority.
    """

    points = np.asarray(points, np.float64)
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("continuous seam source points are invalid")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("continuous seam target surface is empty")
    triangle = vertices[faces]
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]
    ab = b - a
    ac = c - a
    bc = c - b
    ab2 = np.sum(ab * ab, axis=1)
    ac2 = np.sum(ac * ac, axis=1)
    bc2 = np.sum(bc * bc, axis=1)
    d00 = ab2
    d01 = np.sum(ab * ac, axis=1)
    d11 = ac2
    denominator = d00 * d11 - d01 * d01
    if np.any(np.minimum(np.minimum(ab2, ac2), bc2) <= 1e-14) or np.any(
        denominator <= 1e-14
    ):
        raise ValueError("continuous seam target contains a degenerate triangle")

    output_face: list[np.ndarray] = []
    output_barycentric: list[np.ndarray] = []
    output_distance: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        point = points[start : start + chunk_size, None, :]
        ap = point - a[None, :, :]
        d20 = np.sum(ap * ab[None, :, :], axis=2)
        d21 = np.sum(ap * ac[None, :, :], axis=2)
        plane_v = (d11[None, :] * d20 - d01[None, :] * d21) / denominator[None, :]
        plane_w = (d00[None, :] * d21 - d01[None, :] * d20) / denominator[None, :]
        plane_u = 1.0 - plane_v - plane_w
        plane_barycentric = np.stack((plane_u, plane_v, plane_w), axis=2)
        plane_point = (
            plane_u[..., None] * a[None, :, :]
            + plane_v[..., None] * b[None, :, :]
            + plane_w[..., None] * c[None, :, :]
        )
        inside = (
            (plane_u >= 0.0) & (plane_v >= 0.0) & (plane_w >= 0.0)
        )
        plane_distance2 = np.sum((point - plane_point) ** 2, axis=2)
        plane_distance2 = np.where(inside, plane_distance2, np.inf)

        t_ab = np.clip(d20 / ab2[None, :], 0.0, 1.0)
        point_ab = a[None, :, :] + t_ab[..., None] * ab[None, :, :]
        bary_ab = np.stack((1.0 - t_ab, t_ab, np.zeros_like(t_ab)), axis=2)
        distance2_ab = np.sum((point - point_ab) ** 2, axis=2)

        bp = point - b[None, :, :]
        t_bc = np.clip(
            np.sum(bp * bc[None, :, :], axis=2) / bc2[None, :], 0.0, 1.0
        )
        point_bc = b[None, :, :] + t_bc[..., None] * bc[None, :, :]
        bary_bc = np.stack((np.zeros_like(t_bc), 1.0 - t_bc, t_bc), axis=2)
        distance2_bc = np.sum((point - point_bc) ** 2, axis=2)

        cp = point - c[None, :, :]
        ca = a - c
        ca2 = ac2
        t_ca = np.clip(
            np.sum(cp * ca[None, :, :], axis=2) / ca2[None, :], 0.0, 1.0
        )
        point_ca = c[None, :, :] + t_ca[..., None] * ca[None, :, :]
        bary_ca = np.stack((t_ca, np.zeros_like(t_ca), 1.0 - t_ca), axis=2)
        distance2_ca = np.sum((point - point_ca) ** 2, axis=2)

        candidate_distance2 = np.stack(
            (plane_distance2, distance2_ab, distance2_bc, distance2_ca), axis=2
        )
        candidate_barycentric = np.stack(
            (plane_barycentric, bary_ab, bary_bc, bary_ca), axis=2
        )
        best_candidate = np.argmin(candidate_distance2, axis=2)
        rows = np.arange(len(point))[:, None]
        columns = np.arange(len(faces))[None, :]
        face_distance2 = candidate_distance2[rows, columns, best_candidate]
        face_barycentric = candidate_barycentric[
            rows, columns, best_candidate
        ]
        best_face = np.argmin(face_distance2, axis=1)
        point_rows = np.arange(len(point))
        output_face.append(best_face.astype(np.int64))
        output_barycentric.append(face_barycentric[point_rows, best_face])
        output_distance.append(np.sqrt(face_distance2[point_rows, best_face]))
    return (
        np.concatenate(output_face),
        np.concatenate(output_barycentric),
        np.concatenate(output_distance),
    )


@dataclass(frozen=True)
class SourceSeamCorrespondenceAuthority:
    chart_names: np.ndarray
    chart_vertex_offsets: np.ndarray
    sampled_vertex_pixel_indices: np.ndarray
    chart_face_offsets: np.ndarray
    faces: np.ndarray
    reference_vertices_world: np.ndarray
    reference_normals_world: np.ndarray
    edge_chart_indices: np.ndarray
    edge_correspondence_offsets: np.ndarray
    source_vertex_indices: np.ndarray
    target_face_indices: np.ndarray
    target_barycentric: np.ndarray
    reference_distance_m: np.ndarray
    reference_unsigned_normal_angle_deg: np.ndarray
    source_count_by_direction: np.ndarray
    source_supported_fraction_by_direction: np.ndarray
    edge_minimum_supported_fraction_required: np.ndarray
    metadata: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in (
                "chart_names",
                "chart_vertex_offsets",
                "sampled_vertex_pixel_indices",
                "chart_face_offsets",
                "faces",
                "reference_vertices_world",
                "reference_normals_world",
                "edge_chart_indices",
                "edge_correspondence_offsets",
                "source_vertex_indices",
                "target_face_indices",
                "target_barycentric",
                "reference_distance_m",
                "reference_unsigned_normal_angle_deg",
                "source_count_by_direction",
                "source_supported_fraction_by_direction",
                "edge_minimum_supported_fraction_required",
            )
        }

    def validated(self) -> "SourceSeamCorrespondenceAuthority":
        arrays = self.arrays()
        names = arrays["chart_names"].astype(str)
        chart_count = len(names)
        if chart_count < 2 or len(set(names.tolist())) != chart_count:
            raise ValueError("source seam chart inventory is invalid")
        vertex_offsets = arrays["chart_vertex_offsets"].astype(np.int64)
        face_offsets = arrays["chart_face_offsets"].astype(np.int64)
        if vertex_offsets.shape != (chart_count + 1,) or face_offsets.shape != (chart_count + 1,):
            raise ValueError("source seam topology offsets differ from chart inventory")
        if vertex_offsets[0] != 0 or face_offsets[0] != 0:
            raise ValueError("source seam topology offsets do not start at zero")
        if np.any(np.diff(vertex_offsets) <= 0) or np.any(np.diff(face_offsets) <= 0):
            raise ValueError("source seam topology contains an empty chart")
        vertex_count = int(vertex_offsets[-1])
        face_count = int(face_offsets[-1])
        if arrays["sampled_vertex_pixel_indices"].shape != (vertex_count,):
            raise ValueError("source seam sampled-pixel inventory differs")
        if arrays["faces"].shape != (face_count, 3):
            raise ValueError("source seam face inventory differs")
        if arrays["reference_vertices_world"].shape != (vertex_count, 3):
            raise ValueError("source seam reference vertices differ")
        if arrays["reference_normals_world"].shape != (vertex_count, 3):
            raise ValueError("source seam reference normals differ")
        if not np.isfinite(arrays["reference_vertices_world"]).all():
            raise ValueError("source seam reference vertices are nonfinite")
        replayed_normals = _vertex_normals(
            arrays["reference_vertices_world"], arrays["faces"]
        )
        if not np.allclose(
            replayed_normals, arrays["reference_normals_world"], atol=1e-10, rtol=1e-10
        ):
            raise ValueError("source seam reference normals do not replay topology")
        for chart in range(chart_count):
            vlo, vhi = map(int, vertex_offsets[chart : chart + 2])
            flo, fhi = map(int, face_offsets[chart : chart + 2])
            if np.any(
                (arrays["faces"][flo:fhi] < vlo)
                | (arrays["faces"][flo:fhi] >= vhi)
            ):
                raise ValueError("source seam face crosses a chart boundary")

        edges = arrays["edge_chart_indices"].astype(np.int64)
        edge_count = len(edges)
        if edges.shape != (edge_count, 2) or np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("source seam edges are not canonical undirected pairs")
        if np.any((edges < 0) | (edges >= chart_count)) or len(set(map(tuple, edges.tolist()))) != edge_count:
            raise ValueError("source seam edge inventory is invalid")
        correspondence_offsets = arrays["edge_correspondence_offsets"].astype(np.int64)
        if correspondence_offsets.shape != (edge_count + 1,) or correspondence_offsets[0] != 0 or np.any(np.diff(correspondence_offsets) < 0):
            raise ValueError("source seam correspondence offsets are invalid")
        correspondence_count = int(correspondence_offsets[-1])
        vector_names = (
            "source_vertex_indices",
            "target_face_indices",
            "reference_distance_m",
            "reference_unsigned_normal_angle_deg",
        )
        if any(arrays[name].shape != (correspondence_count,) for name in vector_names):
            raise ValueError("source seam correspondence arrays differ in length")
        source = arrays["source_vertex_indices"].astype(np.int64)
        target_face = arrays["target_face_indices"].astype(np.int64)
        barycentric = arrays["target_barycentric"].astype(np.float64)
        if barycentric.shape != (correspondence_count, 3):
            raise ValueError("source seam target barycentric inventory differs")
        if np.any((source < 0) | (source >= vertex_count)) or np.any(
            (target_face < 0) | (target_face >= face_count)
        ):
            raise ValueError("source seam correspondence leaves the topology")
        if (
            not np.isfinite(barycentric).all()
            or np.any(barycentric < -1e-10)
            or np.any(barycentric > 1.0 + 1e-10)
            or not np.allclose(barycentric.sum(1), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError("source seam barycentric coordinates are invalid")
        rows = _chart_rows(vertex_offsets)
        face_rows = _chart_rows(face_offsets)
        replayed_counts = np.zeros((edge_count, 2), np.int64)
        for edge, (first, second) in enumerate(edges):
            lo, hi = map(int, correspondence_offsets[edge : edge + 2])
            pair_rows = np.stack(
                (rows[source[lo:hi]], face_rows[target_face[lo:hi]]), axis=1
            )
            allowed = ((pair_rows[:, 0] == first) & (pair_rows[:, 1] == second)) | (
                (pair_rows[:, 0] == second) & (pair_rows[:, 1] == first)
            )
            if not allowed.all():
                raise ValueError("source seam correspondence crosses its frozen edge")
            replayed_counts[edge, 0] = np.sum(
                (pair_rows[:, 0] == first) & (pair_rows[:, 1] == second)
            )
            replayed_counts[edge, 1] = np.sum(
                (pair_rows[:, 0] == second) & (pair_rows[:, 1] == first)
            )
        target_corners = arrays["faces"][target_face]
        replayed_target = np.sum(
            arrays["reference_vertices_world"][target_corners]
            * barycentric[..., None],
            axis=1,
        )
        interpolated_normal = np.sum(
            arrays["reference_normals_world"][target_corners]
            * barycentric[..., None],
            axis=1,
        )
        interpolated_length = np.linalg.norm(interpolated_normal, axis=1)
        if np.any(interpolated_length <= 1e-10):
            raise ValueError("source seam target interpolated normal is degenerate")
        interpolated_normal /= interpolated_length[:, None]
        replayed_distance = np.linalg.norm(
            arrays["reference_vertices_world"][source] - replayed_target, axis=1
        )
        replayed_angle = np.degrees(
            np.arccos(
                np.clip(
                    np.abs(
                        np.sum(
                            arrays["reference_normals_world"][source]
                            * interpolated_normal,
                            axis=1,
                        )
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        if not np.allclose(replayed_distance, arrays["reference_distance_m"], atol=1e-10, rtol=1e-10):
            raise ValueError("source seam reference distances do not replay")
        if not np.allclose(replayed_angle, arrays["reference_unsigned_normal_angle_deg"], atol=1e-9, rtol=1e-9):
            raise ValueError("source seam reference normal angles do not replay")
        if arrays["source_count_by_direction"].shape != (edge_count, 2):
            raise ValueError("source seam directional counts differ")
        if not np.array_equal(arrays["source_count_by_direction"], replayed_counts):
            raise ValueError("source seam directional counts do not replay pairs")
        if arrays["source_supported_fraction_by_direction"].shape != (edge_count, 2):
            raise ValueError("source seam directional support differs")
        chart_vertex_count = np.diff(vertex_offsets)
        replayed_support = np.stack(
            (
                replayed_counts[:, 0] / chart_vertex_count[edges[:, 0]],
                replayed_counts[:, 1] / chart_vertex_count[edges[:, 1]],
            ),
            axis=1,
        )
        if not np.allclose(
            arrays["source_supported_fraction_by_direction"],
            replayed_support,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("source seam directional support does not replay counts")
        if arrays["edge_minimum_supported_fraction_required"].shape != (edge_count,):
            raise ValueError("source seam required support differs")
        if self.metadata.get("artifact_type") != AUTHORITY_SCHEMA:
            raise ValueError("wrong source seam authority schema")
        if self.metadata.get("continuous_surface_correspondence") is not True:
            raise ValueError("source seam authority is not continuous-surface based")
        if self.metadata.get("uses_query_or_ground_truth") is not False or self.metadata.get("held_geometry_consumed") is not False:
            raise ValueError("source seam authority is not source-only")
        if self.metadata.get("euclidean_distance_role") != "association_and_sampling_phase_diagnostic_only_not_surface_thickness_gate":
            raise ValueError("source seam authority confuses association with thickness")
        config_payload = self.metadata.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("source seam authority lacks frozen config")
        config = SourceSeamConfig(**config_payload).validated()
        if self.metadata.get("config_content_sha256") != canonical_json_sha256(
            config.to_dict()
        ):
            raise ValueError("source seam authority config hash differs")
        if np.any(
            arrays["reference_distance_m"]
            > config.maximum_reference_association_m + 1e-10
        ) or np.any(
            arrays["reference_unsigned_normal_angle_deg"]
            > config.maximum_reference_unsigned_normal_angle_deg + 1e-9
        ):
            raise ValueError("source seam correspondence violates association gate")
        reachability = (
            np.min(replayed_counts, axis=1)
            >= config.minimum_correspondences_per_direction
        ) & (
            np.min(replayed_support, axis=1)
            >= arrays["edge_minimum_supported_fraction_required"]
        )
        if self.metadata.get("m0_reachable_edge_count") != int(
            reachability.sum()
        ) or self.metadata.get("m0_unreachable_edge_count") != int(
            edge_count - reachability.sum()
        ):
            raise ValueError("source seam M0 reachability summary does not replay")
        if self.metadata.get("m0_all_frozen_edges_reachable") is not bool(
            reachability.all()
        ):
            raise ValueError("source seam M0 reachability decision does not replay")
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
    def load_npz(cls, path: Path) -> "SourceSeamCorrespondenceAuthority":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            names = (
                "chart_names",
                "chart_vertex_offsets",
                "sampled_vertex_pixel_indices",
                "chart_face_offsets",
                "faces",
                "reference_vertices_world",
                "reference_normals_world",
                "edge_chart_indices",
                "edge_correspondence_offsets",
                "source_vertex_indices",
                "target_face_indices",
                "target_barycentric",
                "reference_distance_m",
                "reference_unsigned_normal_angle_deg",
                "source_count_by_direction",
                "source_supported_fraction_by_direction",
                "edge_minimum_supported_fraction_required",
            )
            arrays = {name: np.asarray(data[name]) for name in names}
        if metadata.get("arrays_sha256") != arrays_sha256(arrays):
            raise ValueError("source seam authority arrays differ from lineage")
        _replay_metadata(metadata, "source seam authority")
        return cls(metadata=metadata, **arrays).validated()


def freeze_source_seam_correspondences(
    *,
    chart_names: np.ndarray,
    chart_vertex_offsets: np.ndarray,
    sampled_vertex_pixel_indices: np.ndarray,
    chart_face_offsets: np.ndarray,
    faces: np.ndarray,
    reference_vertices_world: np.ndarray,
    plan_chart_names: np.ndarray,
    coverage_edges: np.ndarray,
    symmetric_surface_overlap: np.ndarray,
    config: SourceSeamConfig = SourceSeamConfig(),
    metadata: Mapping[str, object] | None = None,
) -> SourceSeamCorrespondenceAuthority:
    """Freeze directed NN pairs on each selected source-only coverage edge."""

    config = config.validated()
    chart_names = np.asarray(chart_names).astype(str)
    plan_chart_names = np.asarray(plan_chart_names).astype(str)
    chart_vertex_offsets = np.asarray(chart_vertex_offsets, np.int64)
    sampled_vertex_pixel_indices = np.asarray(sampled_vertex_pixel_indices, np.int64)
    chart_face_offsets = np.asarray(chart_face_offsets, np.int64)
    faces = np.asarray(faces, np.int64)
    reference_vertices_world = np.asarray(reference_vertices_world, np.float64)
    if len(set(plan_chart_names.tolist())) != len(plan_chart_names):
        raise ValueError("source seam plan contains duplicate chart names")
    plan_rows = {name: row for row, name in enumerate(plan_chart_names.tolist())}
    if any(name not in plan_rows for name in chart_names.tolist()):
        raise ValueError("source seam topology chart is absent from plan")
    selected = [plan_rows[name] for name in chart_names.tolist()]
    coverage_edges = np.asarray(coverage_edges, bool)
    symmetric_surface_overlap = np.asarray(symmetric_surface_overlap, np.float64)
    expected_pairwise = (len(plan_chart_names), len(plan_chart_names))
    if coverage_edges.shape != expected_pairwise or symmetric_surface_overlap.shape != expected_pairwise:
        raise ValueError("source seam plan pairwise arrays differ")
    selected_coverage = coverage_edges[np.ix_(selected, selected)]
    selected_overlap = symmetric_surface_overlap[np.ix_(selected, selected)]
    reference_normals_world = _vertex_normals(reference_vertices_world, faces)
    cosine_threshold = np.cos(
        np.deg2rad(config.maximum_reference_unsigned_normal_angle_deg)
    )

    edges: list[tuple[int, int]] = []
    edge_offsets = [0]
    source_indices: list[np.ndarray] = []
    target_faces: list[np.ndarray] = []
    target_barycentrics: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    direction_counts: list[tuple[int, int]] = []
    direction_support: list[tuple[float, float]] = []
    required_support: list[float] = []
    for first in range(len(chart_names)):
        first_indices = np.arange(
            chart_vertex_offsets[first], chart_vertex_offsets[first + 1], dtype=np.int64
        )
        for second in range(first + 1, len(chart_names)):
            if not bool(selected_coverage[first, second]):
                continue
            second_indices = np.arange(
                chart_vertex_offsets[second], chart_vertex_offsets[second + 1], dtype=np.int64
            )
            edge_sources: list[np.ndarray] = []
            edge_target_faces: list[np.ndarray] = []
            edge_target_barycentrics: list[np.ndarray] = []
            counts: list[int] = []
            support: list[float] = []
            for source_rows, target_chart in (
                (first_indices, second),
                (second_indices, first),
            ):
                target_face_lo, target_face_hi = map(
                    int, chart_face_offsets[target_chart : target_chart + 2]
                )
                target_face_inventory = faces[target_face_lo:target_face_hi]
                local_face, barycentric, distance = _closest_points_on_triangle_mesh(
                    reference_vertices_world[source_rows],
                    reference_vertices_world,
                    target_face_inventory,
                )
                global_face = target_face_lo + local_face
                target_corners = faces[global_face]
                target_normal = np.sum(
                    reference_normals_world[target_corners]
                    * barycentric[..., None],
                    axis=1,
                )
                target_normal /= np.maximum(
                    np.linalg.norm(target_normal, axis=1, keepdims=True), 1e-12
                )
                dot = np.abs(
                    np.sum(
                        reference_normals_world[source_rows]
                        * target_normal,
                        axis=1,
                    )
                )
                accepted = (
                    np.isfinite(distance)
                    & (distance <= config.maximum_reference_association_m)
                    & (dot >= cosine_threshold)
                )
                edge_sources.append(source_rows[accepted])
                edge_target_faces.append(global_face[accepted])
                edge_target_barycentrics.append(barycentric[accepted])
                counts.append(int(accepted.sum()))
                support.append(float(accepted.mean()))
            edge_source = np.concatenate(edge_sources)
            edge_target_face = np.concatenate(edge_target_faces)
            edge_target_barycentric = np.concatenate(edge_target_barycentrics)
            edge_target_corners = faces[edge_target_face]
            edge_target_point = np.sum(
                reference_vertices_world[edge_target_corners]
                * edge_target_barycentric[..., None],
                axis=1,
            )
            edge_target_normal = np.sum(
                reference_normals_world[edge_target_corners]
                * edge_target_barycentric[..., None],
                axis=1,
            )
            edge_target_normal /= np.maximum(
                np.linalg.norm(edge_target_normal, axis=1, keepdims=True), 1e-12
            )
            edge_distance = np.linalg.norm(
                reference_vertices_world[edge_source] - edge_target_point,
                axis=1,
            )
            edge_angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.abs(
                            np.sum(
                                reference_normals_world[edge_source]
                                * edge_target_normal,
                                axis=1,
                            )
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
            edges.append((first, second))
            source_indices.append(edge_source)
            target_faces.append(edge_target_face)
            target_barycentrics.append(edge_target_barycentric)
            distances.append(edge_distance)
            angles.append(edge_angle)
            direction_counts.append((counts[0], counts[1]))
            direction_support.append((support[0], support[1]))
            required_support.append(
                max(
                    config.minimum_supported_fraction,
                    config.frozen_overlap_support_fraction_multiplier
                    * float(selected_overlap[first, second]),
                )
            )
            edge_offsets.append(edge_offsets[-1] + len(edge_source))
    if not edges:
        raise ValueError("source seam selected submap has no frozen coverage edge")

    arrays = {
        "chart_names": chart_names,
        "chart_vertex_offsets": chart_vertex_offsets,
        "sampled_vertex_pixel_indices": sampled_vertex_pixel_indices,
        "chart_face_offsets": chart_face_offsets,
        "faces": faces,
        "reference_vertices_world": reference_vertices_world,
        "reference_normals_world": reference_normals_world,
        "edge_chart_indices": np.asarray(edges, np.int32),
        "edge_correspondence_offsets": np.asarray(edge_offsets, np.int64),
        "source_vertex_indices": np.concatenate(source_indices).astype(np.int64, copy=False),
        "target_face_indices": np.concatenate(target_faces).astype(np.int64, copy=False),
        "target_barycentric": np.concatenate(target_barycentrics).astype(np.float64, copy=False),
        "reference_distance_m": np.concatenate(distances).astype(np.float64, copy=False),
        "reference_unsigned_normal_angle_deg": np.concatenate(angles).astype(np.float64, copy=False),
        "source_count_by_direction": np.asarray(direction_counts, np.int64),
        "source_supported_fraction_by_direction": np.asarray(direction_support, np.float64),
        "edge_minimum_supported_fraction_required": np.asarray(required_support, np.float64),
    }
    count = arrays["source_count_by_direction"]
    support = arrays["source_supported_fraction_by_direction"]
    required = arrays["edge_minimum_supported_fraction_required"]
    reachability = (
        np.min(count, axis=1) >= config.minimum_correspondences_per_direction
    ) & (np.min(support, axis=1) >= required)
    authority_metadata = {
        "artifact_type": AUTHORITY_SCHEMA,
        "uses_mapping_source_geometry": True,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "held_root_opened": False,
        "association_source": "isolated_source_MASt3R_reference_on_v3_exact_stride4_topology",
        "edge_source": "frozen_chart_submap_plan_coverage_edges_only",
        "correspondence_directionality": "both_directed_source_vertex_to_exact_target_triangle_closest_point_per_undirected_edge",
        "continuous_surface_correspondence": True,
        "target_material_location": "frozen_global_face_id_plus_barycentric_weights",
        "correspondence_identity_frozen_before_arm_evaluation": True,
        "euclidean_distance_role": "association_and_sampling_phase_diagnostic_only_not_surface_thickness_gate",
        "formal_surface_residual": "symmetric_fixed-correspondence_point-to-plane_plus_normal",
        "config": config.to_dict(),
        "config_content_sha256": canonical_json_sha256(config.to_dict()),
        "chart_count": len(chart_names),
        "frozen_edge_count": len(edges),
        "frozen_correspondence_count": int(edge_offsets[-1]),
        "m0_reachable_edge_count": int(reachability.sum()),
        "m0_unreachable_edge_count": int(len(edges) - reachability.sum()),
        "m0_all_frozen_edges_reachable": bool(reachability.all()),
        "formal_arm_gate_reachability_eligible": bool(reachability.all()),
        "source_seam_core_file_sha256": file_sha256(Path(__file__)),
        **dict(metadata or {}),
    }
    return SourceSeamCorrespondenceAuthority(
        metadata=authority_metadata, **arrays
    ).validated()


def _load_reference_points(path: Path, height: int, width: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    confidence = np.asarray(payload.get("confs"), np.float64)
    points = np.asarray(payload.get("points"), np.float64)
    if confidence.ndim != 2 or points.size != confidence.size * 3:
        raise ValueError("source seam pointmap has an invalid shape")
    points = points.reshape(*confidence.shape, 3)
    resized = cv2.resize(points, (width, height), interpolation=cv2.INTER_AREA)
    if resized.shape != (height, width, 3) or not np.isfinite(resized).all():
        raise ValueError("source seam pointmap resize is invalid")
    return np.asarray(resized, np.float64)


def build_source_seam_correspondence_authority(
    comparison_domain_v3_path: Path,
    *,
    expected_comparison_domain_content_sha256: str,
    frozen_submap_plan_path: Path,
    expected_plan_content_sha256: str,
    disjoint_upstream_authority_path: Path,
    expected_disjoint_authority_content_sha256: str,
    source_root: Path,
    expected_source_tree_sha256: str,
    config: SourceSeamConfig = SourceSeamConfig(),
) -> SourceSeamCorrespondenceAuthority:
    """Build a fully replayed source-only seam authority from sealed inputs."""

    pins = (
        expected_comparison_domain_content_sha256,
        expected_plan_content_sha256,
        expected_disjoint_authority_content_sha256,
        expected_source_tree_sha256,
    )
    if any(not _is_sha256(value) for value in pins):
        raise ValueError("source seam builder requires four explicit SHA-256 pins")
    comparison_domain_v3_path = Path(comparison_domain_v3_path)
    frozen_submap_plan_path = Path(frozen_submap_plan_path)
    disjoint_upstream_authority_path = Path(disjoint_upstream_authority_path)
    source_root = Path(source_root).resolve()
    base_names = list(BASE_ARRAY_NAMES)
    topology_names = list(topology_array_names())
    with np.load(comparison_domain_v3_path, allow_pickle=False) as data:
        domain_metadata = json.loads(str(data["metadata_json"].item()))
        domain_arrays = {
            name: np.asarray(data[name]) for name in base_names + topology_names
        }
    domain_hash = _replay_metadata(domain_metadata, "source seam comparison domain")
    if domain_hash != expected_comparison_domain_content_sha256:
        raise ValueError("source seam comparison domain differs from experiment pin")
    if domain_metadata.get("artifact_type") != PHYSICAL_DOMAIN_SCHEMA:
        raise ValueError("source seam requires reference-safe v3 topology")
    if domain_metadata.get("full_submap_gate_primary_stride") != 4 or domain_metadata.get("source_reference_edge_safe") is not True:
        raise ValueError("source seam comparison domain is not formal stride-4 physical topology")
    if domain_metadata.get("arrays_sha256") != arrays_sha256(domain_arrays):
        raise ValueError("source seam comparison-domain arrays differ")
    base_arrays = {name: domain_arrays[name] for name in base_names}
    topology_arrays = {name: domain_arrays[name] for name in topology_names}
    validate_exact_topology_arrays(
        base_arrays,
        topology_arrays,
        expected_sha256=domain_metadata.get("exact_topology_arrays_sha256"),
    )

    plan = ChartSubmapPlan.load_npz(frozen_submap_plan_path)
    if plan.metadata.get("content_sha256") != expected_plan_content_sha256:
        raise ValueError("source seam frozen plan differs from experiment pin")
    selected_names = list(plan.selected_chart_names_in_order)
    chart_names = base_arrays["chart_names"].astype(str)
    if chart_names.tolist() != selected_names:
        raise ValueError("source seam domain and frozen selection order differ")
    if domain_metadata.get("frozen_submap_plan_content_sha256") != expected_plan_content_sha256:
        raise ValueError("source seam domain and plan lineage differ")

    authority = json.loads(disjoint_upstream_authority_path.read_text())
    authority_hash = _replay_metadata(authority, "source seam disjoint authority")
    if authority_hash != expected_disjoint_authority_content_sha256:
        raise ValueError("source seam disjoint authority differs from experiment pin")
    if authority.get("artifact_type") != DISJOINT_AUTHORITY_SCHEMA or not authority.get("physical_source_held_input_roots_disjoint"):
        raise ValueError("source seam authority lacks physical source/held isolation")
    source = authority.get("source", {})
    if Path(source.get("root", "")).resolve() != source_root:
        raise ValueError("source seam source root differs from disjoint authority")
    observed_tree = source_tree_sha256(source_root)
    if observed_tree != expected_source_tree_sha256 or source.get("tree_sha256") != observed_tree:
        raise ValueError("source seam source tree differs from experiment pin")
    lineage_equal = {
        "disjoint_upstream_authority_content_sha256": authority_hash,
        "source_tree_sha256": observed_tree,
        "frozen_submap_plan_content_sha256": expected_plan_content_sha256,
    }
    for key, expected in lineage_equal.items():
        if domain_metadata.get(key) != expected:
            raise ValueError(f"source seam comparison-domain {key} differs")

    stride = config.validated().topology_stride
    vertex_offsets = topology_arrays[f"sampled_vertex_offsets_stride{stride}"]
    pixel_indices = topology_arrays[f"sampled_vertex_pixel_indices_stride{stride}"]
    face_offsets = topology_arrays[f"face_offsets_stride{stride}"]
    faces = topology_arrays[f"faces_stride{stride}"]
    height, width = base_arrays["valid"].shape[1:]
    reference_grids = np.stack(
        [
            _load_reference_points(
                source_root / "pointmaps" / f"{Path(name).stem}.json",
                height,
                width,
            )
            for name in chart_names
        ]
    )
    reference_vertices = np.concatenate(
        [
            reference_grids[chart].reshape(-1, 3)[
                pixel_indices[vertex_offsets[chart] : vertex_offsets[chart + 1]]
            ]
            for chart in range(len(chart_names))
        ]
    )
    pointmap_inventory = {
        name: file_sha256(
            source_root / "pointmaps" / f"{Path(name).stem}.json"
        )
        for name in chart_names.tolist()
    }
    metadata = {
        "comparison_domain_v3_file_sha256": file_sha256(comparison_domain_v3_path),
        "comparison_domain_v3_content_sha256": domain_hash,
        "comparison_domain_v3_exact_topology_arrays_sha256": domain_metadata.get("exact_topology_arrays_sha256"),
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "frozen_submap_plan_content_sha256": expected_plan_content_sha256,
        "selected_chart_names_in_order_sha256": plan.metadata.get("selected_chart_names_in_order_sha256"),
        "disjoint_upstream_authority_file_sha256": file_sha256(disjoint_upstream_authority_path),
        "disjoint_upstream_authority_content_sha256": authority_hash,
        "source_root": str(source_root),
        "source_tree_sha256": observed_tree,
        "source_selected_pointmap_inventory": pointmap_inventory,
        "source_selected_pointmap_inventory_sha256": canonical_json_sha256(pointmap_inventory),
        "upstream_optimizer_comparison_domain_content_sha256": domain_metadata.get("upstream_comparison_domain_content_sha256"),
        "full_submap_gate_primary_stride": 4,
        "source_reference_edge_safe": True,
    }
    return freeze_source_seam_correspondences(
        chart_names=chart_names,
        chart_vertex_offsets=vertex_offsets,
        sampled_vertex_pixel_indices=pixel_indices,
        chart_face_offsets=face_offsets,
        faces=faces,
        reference_vertices_world=reference_vertices,
        plan_chart_names=plan.chart_names,
        coverage_edges=plan.coverage_edges,
        symmetric_surface_overlap=plan.symmetric_surface_overlap,
        config=config,
        metadata=metadata,
    )


def _quantile(values: np.ndarray, probability: float) -> float | None:
    return float(np.quantile(values, probability)) if len(values) else None


def _state_metrics(
    authority: SourceSeamCorrespondenceAuthority,
    vertices_world: np.ndarray,
    config: SourceSeamConfig,
) -> dict[str, object]:
    vertices_world = np.asarray(vertices_world, np.float64)
    if vertices_world.shape != authority.reference_vertices_world.shape or not np.isfinite(vertices_world).all():
        raise ValueError("source seam evaluated geometry differs from frozen topology")
    normals = _vertex_normals(vertices_world, authority.faces)
    edge_rows = []
    all_distance: list[np.ndarray] = []
    all_plane: list[np.ndarray] = []
    all_unsigned: list[np.ndarray] = []
    all_oriented: list[np.ndarray] = []
    all_opposed: list[np.ndarray] = []
    for edge, (first, second) in enumerate(authority.edge_chart_indices):
        lo, hi = map(int, authority.edge_correspondence_offsets[edge : edge + 2])
        source = authority.source_vertex_indices[lo:hi]
        target_face = authority.target_face_indices[lo:hi]
        barycentric = authority.target_barycentric[lo:hi]
        target_corners = authority.faces[target_face]
        target_point = np.sum(
            vertices_world[target_corners] * barycentric[..., None], axis=1
        )
        target_normal = np.sum(
            normals[target_corners] * barycentric[..., None], axis=1
        )
        target_normal_length = np.linalg.norm(target_normal, axis=1)
        if np.any(target_normal_length <= 1e-10):
            raise ValueError("source seam evaluated target normal is degenerate")
        target_normal /= target_normal_length[:, None]
        delta = vertices_world[source] - target_point
        distance = np.linalg.norm(delta, axis=1)
        plane = np.abs(np.sum(delta * target_normal, axis=1))
        signed_dot = np.sum(normals[source] * target_normal, axis=1)
        unsigned = np.degrees(np.arccos(np.clip(np.abs(signed_dot), -1.0, 1.0)))
        oriented = np.degrees(np.arccos(np.clip(signed_dot, -1.0, 1.0)))
        opposed = signed_dot < 0
        counts = authority.source_count_by_direction[edge]
        support = authority.source_supported_fraction_by_direction[edge]
        required = float(authority.edge_minimum_supported_fraction_required[edge])
        reachability = bool(
            int(np.min(counts)) >= config.minimum_correspondences_per_direction
            and float(np.min(support)) >= required
        )
        point_p50 = _quantile(plane, 0.5)
        point_p90 = _quantile(plane, 0.9)
        unsigned_p90 = _quantile(unsigned, 0.9)
        oriented_p90 = _quantile(oriented, 0.9)
        opposed_fraction = float(np.mean(opposed)) if len(opposed) else None
        geometry_pass = bool(
            point_p50 is not None
            and point_p50 <= config.maximum_point_to_plane_p50_m
            and point_p90 is not None
            and point_p90 <= config.maximum_point_to_plane_p90_m
            and unsigned_p90 is not None
            and unsigned_p90 <= config.maximum_unsigned_normal_p90_deg
            and oriented_p90 is not None
            and oriented_p90 <= config.maximum_oriented_normal_p90_deg
            and opposed_fraction is not None
            and opposed_fraction <= config.maximum_opposed_normal_fraction
        )
        edge_rows.append(
            {
                "first": str(authority.chart_names[int(first)]),
                "second": str(authority.chart_names[int(second)]),
                "correspondence_count": int(hi - lo),
                "count_by_direction": counts.astype(int).tolist(),
                "supported_fraction_by_direction": support.tolist(),
                "minimum_supported_fraction_required": required,
                "m0_reachability_pass": reachability,
                "euclidean_distance_p50_m_diagnostic_only": _quantile(distance, 0.5),
                "euclidean_distance_p90_m_diagnostic_only": _quantile(distance, 0.9),
                "point_to_plane_p50_m": point_p50,
                "point_to_plane_p90_m": point_p90,
                "unsigned_normal_p90_deg": unsigned_p90,
                "oriented_normal_p90_deg": oriented_p90,
                "opposed_normal_fraction": opposed_fraction,
                "geometry_pass": geometry_pass,
                "reachable_geometry_pass": bool(reachability and geometry_pass),
            }
        )
        if len(distance):
            all_distance.append(distance)
            all_plane.append(plane)
            all_unsigned.append(unsigned)
            all_oriented.append(oriented)
            all_opposed.append(opposed)
    distance = np.concatenate(all_distance) if all_distance else np.zeros(0)
    plane = np.concatenate(all_plane) if all_plane else np.zeros(0)
    unsigned = np.concatenate(all_unsigned) if all_unsigned else np.zeros(0)
    oriented = np.concatenate(all_oriented) if all_oriented else np.zeros(0)
    opposed = np.concatenate(all_opposed) if all_opposed else np.zeros(0, bool)
    reachable = [row for row in edge_rows if row["m0_reachability_pass"]]
    return {
        "frozen_edge_count": len(edge_rows),
        "m0_reachable_edge_count": len(reachable),
        "m0_all_frozen_edges_reachable": bool(len(reachable) == len(edge_rows)),
        "all_frozen_edges_geometry_pass": all(row["geometry_pass"] for row in edge_rows),
        "all_reachable_edges_geometry_pass": bool(reachable) and all(
            row["geometry_pass"] for row in reachable
        ),
        "reachable_geometry_pass_count": sum(
            row["reachable_geometry_pass"] for row in edge_rows
        ),
        "euclidean_distance_p50_m_diagnostic_only": _quantile(distance, 0.5),
        "euclidean_distance_p90_m_diagnostic_only": _quantile(distance, 0.9),
        "point_to_plane_p50_m": _quantile(plane, 0.5),
        "point_to_plane_p90_m": _quantile(plane, 0.9),
        "unsigned_normal_p90_deg": _quantile(unsigned, 0.9),
        "oriented_normal_p90_deg": _quantile(oriented, 0.9),
        "opposed_normal_fraction": float(np.mean(opposed)) if len(opposed) else None,
        "per_edge": edge_rows,
    }


def evaluate_source_seam_geometry(
    authority: SourceSeamCorrespondenceAuthority,
    arm_vertices_world: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Evaluate M0 and aligned arms on exactly the same frozen seam pairs."""

    authority = authority.validated()
    config = SourceSeamConfig(**authority.metadata["config"]).validated()
    m0 = _state_metrics(authority, authority.reference_vertices_world, config)
    m0_eligible = bool(
        m0["m0_all_frozen_edges_reachable"]
        and m0["all_frozen_edges_geometry_pass"]
    )
    arms = {}
    for label, vertices in arm_vertices_world.items():
        metrics = _state_metrics(authority, vertices, config)
        formal_pass = bool(m0_eligible and metrics["all_frozen_edges_geometry_pass"])
        metrics.update(
            {
                "formal_source_seam_gate_eligible": m0_eligible,
                "formal_source_seam_gate_pass": formal_pass,
                "formal_decision": "GO" if formal_pass else "KILL",
                "conditional_reachable_edge_decision": (
                    "GO" if metrics["all_reachable_edges_geometry_pass"] else "KILL"
                ),
            }
        )
        arms[str(label)] = metrics
    return {
        "artifact_type": REPORT_SCHEMA,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "correspondence_identity_source_only_and_frozen": True,
        "euclidean_distance_is_diagnostic_only": True,
        "config": config.to_dict(),
        "m0_source_reference": m0,
        "m0_authority_reachable_and_geometrically_valid": m0_eligible,
        "formal_arm_gate_eligible": m0_eligible,
        "arms": arms,
    }


__all__ = [
    "AUTHORITY_SCHEMA",
    "REPORT_SCHEMA",
    "SourceSeamConfig",
    "SourceSeamCorrespondenceAuthority",
    "build_source_seam_correspondence_authority",
    "evaluate_source_seam_geometry",
    "freeze_source_seam_correspondences",
]
