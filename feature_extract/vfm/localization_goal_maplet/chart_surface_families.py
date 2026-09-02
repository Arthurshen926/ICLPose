"""Canonical surface-family carrier for an aligned explicit chart atlas.

Source charts are view-conditioned UV parameterizations.  They are *not*
runtime localization candidates.  This module first splits each chart at
strong mesh-normal discontinuities, then links compatible patches across
different parameterizations.  The resulting family IDs are the only runtime
candidate IDs; source image names are written exclusively to an offline
lineage sidecar by the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .explicit_chart_atlas import ExplicitChartAtlas
from .lineage import arrays_sha256, canonical_json_sha256


SCHEMA = "goal_maplet_canonical_surface_family_carrier_v1"


@dataclass(frozen=True)
class SurfaceFamilyConfig:
    maximum_intra_chart_face_angle_deg: float = 35.0
    minimum_patch_faces: int = 2
    minimum_patch_area_m2: float = 0.02
    maximum_cross_chart_distance_m: float = 0.75
    maximum_cross_chart_normal_angle_deg: float = 50.0
    minimum_smaller_patch_overlap: float = 0.20
    minimum_larger_patch_overlap: float = 0.03
    maximum_overlap_point_to_plane_median_m: float = 0.40
    minimum_cross_chart_matches: int = 8
    maximum_overlap_samples_per_patch: int = 512
    minimum_online_family_parameterizations: int = 2
    minimum_online_supported_patch_area_fraction: float = 0.50

    def validated(self) -> "SurfaceFamilyConfig":
        if not 0 < self.maximum_intra_chart_face_angle_deg < 180:
            raise ValueError("invalid intra-chart face angle")
        if self.minimum_patch_faces < 1 or self.minimum_patch_area_m2 < 0:
            raise ValueError("invalid patch support threshold")
        if self.maximum_cross_chart_distance_m <= 0:
            raise ValueError("cross-chart distance must be positive")
        if not 0 < self.maximum_cross_chart_normal_angle_deg <= 90:
            raise ValueError("invalid cross-chart normal angle")
        if not 0 < self.minimum_smaller_patch_overlap <= 1:
            raise ValueError("invalid smaller-patch overlap")
        if not 0 <= self.minimum_larger_patch_overlap <= self.minimum_smaller_patch_overlap:
            raise ValueError("invalid larger-patch overlap")
        if self.maximum_overlap_point_to_plane_median_m <= 0:
            raise ValueError("invalid point-to-plane threshold")
        if self.minimum_cross_chart_matches < 1:
            raise ValueError("minimum matches must be positive")
        if self.maximum_overlap_samples_per_patch < self.minimum_cross_chart_matches:
            raise ValueError("sample budget is smaller than minimum matches")
        if self.minimum_online_family_parameterizations < 2:
            raise ValueError("online families must have multi-view support")
        if not 0 < self.minimum_online_supported_patch_area_fraction <= 1:
            raise ValueError("invalid online-supported patch-area fraction")
        return self

    def to_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class CanonicalSurfaceFamilyCarrier:
    patch_parameterization_ids: np.ndarray
    patch_family_ids: np.ndarray
    patch_vertex_offsets: np.ndarray
    patch_vertex_indices: np.ndarray
    patch_face_offsets: np.ndarray
    patch_face_indices: np.ndarray
    patch_area_m2: np.ndarray
    patch_centroids_world: np.ndarray
    patch_normals_world: np.ndarray
    patch_geometry_eligible: np.ndarray
    family_patch_offsets: np.ndarray
    family_patch_ids: np.ndarray
    family_bounds_min_world: np.ndarray
    family_bounds_max_world: np.ndarray
    family_centroids_world: np.ndarray
    family_normals_world: np.ndarray
    family_area_m2: np.ndarray
    family_parameterization_count: np.ndarray
    family_online_eligible: np.ndarray
    overlap_edges: np.ndarray
    overlap_scores: np.ndarray
    overlap_smaller_patch_fraction: np.ndarray
    overlap_larger_patch_fraction: np.ndarray
    overlap_point_to_plane_median_m: np.ndarray
    overlap_unsigned_normal_median_deg: np.ndarray
    metadata: dict[str, object]

    @property
    def patch_count(self) -> int:
        return int(len(self.patch_parameterization_ids))

    @property
    def family_count(self) -> int:
        return int(len(self.family_area_m2))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in (
                "patch_parameterization_ids",
                "patch_family_ids",
                "patch_vertex_offsets",
                "patch_vertex_indices",
                "patch_face_offsets",
                "patch_face_indices",
                "patch_area_m2",
                "patch_centroids_world",
                "patch_normals_world",
                "patch_geometry_eligible",
                "family_patch_offsets",
                "family_patch_ids",
                "family_bounds_min_world",
                "family_bounds_max_world",
                "family_centroids_world",
                "family_normals_world",
                "family_area_m2",
                "family_parameterization_count",
                "family_online_eligible",
                "overlap_edges",
                "overlap_scores",
                "overlap_smaller_patch_fraction",
                "overlap_larger_patch_fraction",
                "overlap_point_to_plane_median_m",
                "overlap_unsigned_normal_median_deg",
            )
        }

    def validated(self) -> "CanonicalSurfaceFamilyCarrier":
        arrays = self.arrays()
        patches = self.patch_count
        families = self.family_count
        edges = len(self.overlap_edges)
        if patches < 1 or families < 1:
            raise ValueError("surface-family carrier is empty")
        if arrays["patch_family_ids"].shape != (patches,):
            raise ValueError("invalid patch family ids")
        if arrays["patch_vertex_offsets"].shape != (patches + 1,):
            raise ValueError("invalid patch vertex offsets")
        if arrays["patch_face_offsets"].shape != (patches + 1,):
            raise ValueError("invalid patch face offsets")
        if arrays["patch_vertex_offsets"][0] != 0 or arrays["patch_vertex_offsets"][-1] != len(arrays["patch_vertex_indices"]):
            raise ValueError("patch vertex offsets do not span inventory")
        if arrays["patch_face_offsets"][0] != 0 or arrays["patch_face_offsets"][-1] != len(arrays["patch_face_indices"]):
            raise ValueError("patch face offsets do not span inventory")
        if np.any(np.diff(arrays["patch_vertex_offsets"]) <= 0):
            raise ValueError("a patch has no vertices")
        if np.any(np.diff(arrays["patch_face_offsets"]) <= 0):
            raise ValueError("a patch has no faces")
        patch_shapes = {
            "patch_area_m2": (patches,),
            "patch_centroids_world": (patches, 3),
            "patch_normals_world": (patches, 3),
            "patch_geometry_eligible": (patches,),
        }
        family_shapes = {
            "family_bounds_min_world": (families, 3),
            "family_bounds_max_world": (families, 3),
            "family_centroids_world": (families, 3),
            "family_normals_world": (families, 3),
            "family_area_m2": (families,),
            "family_parameterization_count": (families,),
            "family_online_eligible": (families,),
        }
        for name, shape in {**patch_shapes, **family_shapes}.items():
            if arrays[name].shape != shape:
                raise ValueError(f"invalid {name} shape")
        if arrays["family_patch_offsets"].shape != (families + 1,):
            raise ValueError("invalid family patch offsets")
        if arrays["family_patch_offsets"][0] != 0 or arrays["family_patch_offsets"][-1] != patches:
            raise ValueError("family patch offsets do not span patches")
        if sorted(arrays["family_patch_ids"].tolist()) != list(range(patches)):
            raise ValueError("family patch inventory is not a permutation")
        if np.any((arrays["patch_family_ids"] < 0) | (arrays["patch_family_ids"] >= families)):
            raise ValueError("patch family id outside family inventory")
        if arrays["overlap_edges"].shape != (edges, 2):
            raise ValueError("invalid overlap edge array")
        for name in (
            "overlap_scores",
            "overlap_smaller_patch_fraction",
            "overlap_larger_patch_fraction",
            "overlap_point_to_plane_median_m",
            "overlap_unsigned_normal_median_deg",
        ):
            if arrays[name].shape != (edges,):
                raise ValueError(f"invalid {name} shape")
        if edges and np.any((arrays["overlap_edges"] < 0) | (arrays["overlap_edges"] >= patches)):
            raise ValueError("overlap edge outside patch inventory")
        finite_names = (
            "patch_area_m2",
            "patch_centroids_world",
            "patch_normals_world",
            "family_bounds_min_world",
            "family_bounds_max_world",
            "family_centroids_world",
            "family_normals_world",
            "family_area_m2",
            "overlap_scores",
            "overlap_smaller_patch_fraction",
            "overlap_larger_patch_fraction",
            "overlap_point_to_plane_median_m",
            "overlap_unsigned_normal_median_deg",
        )
        if any(not np.isfinite(arrays[name]).all() for name in finite_names):
            raise ValueError("surface-family carrier contains nonfinite geometry")
        if self.metadata.get("artifact_type") != SCHEMA:
            raise ValueError("wrong surface-family carrier schema")
        if self.metadata.get("runtime_candidate_unit") != "canonical_surface_family":
            raise ValueError("source parameterizations leaked into runtime candidate identity")
        for forbidden in (
            "stores_mapping_rgb",
            "stores_mapping_image_paths",
            "stores_mapping_image_ids",
            "source_chart_names_in_runtime",
        ):
            if self.metadata.get(forbidden) is not False:
                raise ValueError(f"runtime carrier must explicitly disable {forbidden}")
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
    def load_npz(cls, path: Path) -> "CanonicalSurfaceFamilyCarrier":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {
                name: np.asarray(data[name])
                for name in (
                    "patch_parameterization_ids",
                    "patch_family_ids",
                    "patch_vertex_offsets",
                    "patch_vertex_indices",
                    "patch_face_offsets",
                    "patch_face_indices",
                    "patch_area_m2",
                    "patch_centroids_world",
                    "patch_normals_world",
                    "patch_geometry_eligible",
                    "family_patch_offsets",
                    "family_patch_ids",
                    "family_bounds_min_world",
                    "family_bounds_max_world",
                    "family_centroids_world",
                    "family_normals_world",
                    "family_area_m2",
                    "family_parameterization_count",
                    "family_online_eligible",
                    "overlap_edges",
                    "overlap_scores",
                    "overlap_smaller_patch_fraction",
                    "overlap_larger_patch_fraction",
                    "overlap_point_to_plane_median_m",
                    "overlap_unsigned_normal_median_deg",
                )
            }
        if arrays_sha256(arrays) != metadata.get("arrays_sha256"):
            raise ValueError("surface-family carrier arrays differ from lineage")
        content = dict(metadata)
        expected_content = content.pop("content_sha256", None)
        if expected_content != canonical_json_sha256(content):
            raise ValueError("surface-family carrier metadata differs from lineage")
        return cls(metadata=metadata, **arrays).validated()


class _UnionFind:
    def __init__(self, count: int, parameterizations: np.ndarray):
        self.parent = np.arange(count, dtype=np.int64)
        self.parameterizations = [{int(value)} for value in parameterizations]

    def find(self, row: int) -> int:
        root = row
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[row] != row:
            parent = int(self.parent[row])
            self.parent[row] = root
            row = parent
        return root

    def merge_without_parameterization_collision(self, first: int, second: int) -> bool:
        left, right = self.find(first), self.find(second)
        if left == right:
            return True
        if self.parameterizations[left] & self.parameterizations[right]:
            return False
        if left > right:
            left, right = right, left
        self.parent[right] = left
        self.parameterizations[left] |= self.parameterizations[right]
        return True


def _split_chart_patches(
    atlas: ExplicitChartAtlas,
    config: SurfaceFamilyConfig,
) -> tuple[list[dict[str, object]], int]:
    patches: list[dict[str, object]] = []
    assigned_vertices: set[int] = set()
    threshold = np.cos(np.deg2rad(config.maximum_intra_chart_face_angle_deg))
    for chart in range(len(atlas.chart_names)):
        face_lo, face_hi = map(int, atlas.chart_face_offsets[chart : chart + 2])
        face_rows = np.arange(face_lo, face_hi, dtype=np.int64)
        faces = atlas.faces[face_rows]
        if not len(faces):
            continue
        vertices = atlas.vertices_world
        cross = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        doubled_area = np.linalg.norm(cross, axis=1)
        valid = np.isfinite(doubled_area) & (doubled_area > 1e-10)
        face_rows = face_rows[valid]
        faces = faces[valid]
        cross = cross[valid]
        doubled_area = doubled_area[valid]
        if not len(faces):
            continue
        face_normals = cross / doubled_area[:, None]
        parent = np.arange(len(faces), dtype=np.int64)

        def find(row: int) -> int:
            while parent[row] != row:
                parent[row] = parent[parent[row]]
                row = int(parent[row])
            return row

        def union(first: int, second: int) -> None:
            left, right = find(first), find(second)
            if left != right:
                parent[max(left, right)] = min(left, right)

        edge_faces: dict[tuple[int, int], list[int]] = {}
        for local, face in enumerate(faces):
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (int(min(first, second)), int(max(first, second)))
                edge_faces.setdefault(edge, []).append(local)
        for rows in edge_faces.values():
            if len(rows) < 2:
                continue
            for offset, first in enumerate(rows[:-1]):
                for second in rows[offset + 1 :]:
                    if float(np.dot(face_normals[first], face_normals[second])) >= threshold:
                        union(first, second)
        groups: dict[int, list[int]] = {}
        for row in range(len(faces)):
            groups.setdefault(find(row), []).append(row)
        for local_rows in groups.values():
            local_rows_array = np.asarray(local_rows, np.int64)
            patch_face_rows = face_rows[local_rows_array]
            patch_faces = atlas.faces[patch_face_rows]
            patch_vertices = np.unique(patch_faces.reshape(-1))
            area = float(np.sum(doubled_area[local_rows_array]) * 0.5)
            weighted_normal = np.sum(
                face_normals[local_rows_array] * doubled_area[local_rows_array, None],
                axis=0,
            )
            normal_length = float(np.linalg.norm(weighted_normal))
            normal = weighted_normal / max(normal_length, 1e-12)
            centroid = np.average(
                atlas.vertices_world[patch_faces].mean(1),
                axis=0,
                weights=doubled_area[local_rows_array],
            )
            eligible = (
                len(patch_face_rows) >= config.minimum_patch_faces
                and area >= config.minimum_patch_area_m2
                and normal_length > 1e-8
            )
            patches.append(
                {
                    "parameterization": chart,
                    "vertex_indices": patch_vertices,
                    "face_indices": patch_face_rows,
                    "area": area,
                    "centroid": centroid,
                    "normal": normal,
                    "eligible": bool(eligible),
                }
            )
            assigned_vertices.update(patch_vertices.tolist())
    return patches, len(atlas.vertices_world) - len(assigned_vertices)


def _sample_indices(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    rows = np.linspace(0, len(values) - 1, maximum).round().astype(np.int64)
    return values[rows]


def _patch_overlap(
    atlas: ExplicitChartAtlas,
    first: dict[str, object],
    second: dict[str, object],
    config: SurfaceFamilyConfig,
) -> dict[str, float] | None:
    first_indices = _sample_indices(
        np.asarray(first["vertex_indices"], np.int64),
        config.maximum_overlap_samples_per_patch,
    )
    second_indices = _sample_indices(
        np.asarray(second["vertex_indices"], np.int64),
        config.maximum_overlap_samples_per_patch,
    )
    first_points = atlas.vertices_world[first_indices]
    second_points = atlas.vertices_world[second_indices]
    threshold = config.maximum_cross_chart_distance_m
    lower_gap = np.maximum(
        np.maximum(first_points.min(0) - second_points.max(0), second_points.min(0) - first_points.max(0)),
        0.0,
    )
    if float(np.linalg.norm(lower_gap)) > threshold:
        return None
    first_normals = atlas.normals_world[first_indices]
    second_normals = atlas.normals_world[second_indices]
    minimum_dot = np.cos(np.deg2rad(config.maximum_cross_chart_normal_angle_deg))

    def direction(
        source_points: np.ndarray,
        source_normals: np.ndarray,
        target_points: np.ndarray,
        target_normals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distance, nearest = cKDTree(target_points).query(source_points, k=1, workers=-1)
        chosen_normals = target_normals[nearest]
        source_length = np.linalg.norm(source_normals, axis=1)
        target_length = np.linalg.norm(chosen_normals, axis=1)
        normal_valid = (source_length > 0.5) & (target_length > 0.5)
        dot = np.zeros(len(source_points), np.float64)
        dot[normal_valid] = np.abs(
            np.sum(source_normals[normal_valid] * chosen_normals[normal_valid], axis=1)
            / (source_length[normal_valid] * target_length[normal_valid])
        )
        consistent = (distance <= threshold) & normal_valid & (dot >= minimum_dot)
        delta = source_points - target_points[nearest]
        point_to_plane = np.abs(np.sum(delta * chosen_normals, axis=1)) / np.maximum(target_length, 1e-12)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return consistent, point_to_plane, angle

    first_consistent, first_plane, first_angle = direction(
        first_points, first_normals, second_points, second_normals,
    )
    second_consistent, second_plane, second_angle = direction(
        second_points, second_normals, first_points, first_normals,
    )
    fractions = np.asarray([first_consistent.mean(), second_consistent.mean()])
    smaller_patch_fraction = float(fractions.max())
    larger_patch_fraction = float(fractions.min())
    match_count = int(min(first_consistent.sum(), second_consistent.sum()))
    if match_count < config.minimum_cross_chart_matches:
        return None
    point_to_plane = np.concatenate(
        [first_plane[first_consistent], second_plane[second_consistent]]
    )
    angles = np.concatenate(
        [first_angle[first_consistent], second_angle[second_consistent]]
    )
    p2plane_median = float(np.median(point_to_plane))
    if smaller_patch_fraction < config.minimum_smaller_patch_overlap:
        return None
    if larger_patch_fraction < config.minimum_larger_patch_overlap:
        return None
    if p2plane_median > config.maximum_overlap_point_to_plane_median_m:
        return None
    return {
        "score": float(np.sqrt(smaller_patch_fraction * larger_patch_fraction)),
        "smaller_patch_fraction": smaller_patch_fraction,
        "larger_patch_fraction": larger_patch_fraction,
        "point_to_plane_median_m": p2plane_median,
        "unsigned_normal_median_deg": float(np.median(angles)),
    }


def build_canonical_surface_families(
    atlas: ExplicitChartAtlas,
    *,
    config: SurfaceFamilyConfig,
    source_atlas_content_sha256: str,
) -> tuple[CanonicalSurfaceFamilyCarrier, dict[str, object]]:
    atlas = atlas.validated()
    config = config.validated()
    if atlas.metadata.get("control_only") is not False:
        raise ValueError("control-only atlas cannot produce runtime surface families")
    if atlas.metadata.get("optimization_saw_excluded_routes") is not False:
        raise ValueError("atlas alignment saw excluded routes")
    patches, unassigned_vertex_count = _split_chart_patches(atlas, config)
    if not patches:
        raise ValueError("atlas did not produce any surface patches")
    parameterizations = np.asarray([row["parameterization"] for row in patches], np.int32)
    overlap_rows: list[tuple[int, int, dict[str, float]]] = []
    for first in range(len(patches)):
        if not patches[first]["eligible"]:
            continue
        for second in range(first + 1, len(patches)):
            if not patches[second]["eligible"]:
                continue
            if parameterizations[first] == parameterizations[second]:
                continue
            overlap = _patch_overlap(atlas, patches[first], patches[second], config)
            if overlap is not None:
                overlap_rows.append((first, second, overlap))
    overlap_rows.sort(key=lambda row: (-row[2]["score"], row[0], row[1]))
    union = _UnionFind(len(patches), parameterizations)
    accepted_rows: list[tuple[int, int, dict[str, float]]] = []
    collision_rejected = 0
    for first, second, overlap in overlap_rows:
        if union.merge_without_parameterization_collision(first, second):
            accepted_rows.append((first, second, overlap))
        else:
            collision_rejected += 1
    roots = np.asarray([union.find(row) for row in range(len(patches))], np.int64)
    unique_roots = sorted(set(roots.tolist()))
    root_to_family = {root: family for family, root in enumerate(unique_roots)}
    patch_family_ids = np.asarray([root_to_family[int(root)] for root in roots], np.int32)
    family_patch_ids: list[int] = []
    family_patch_offsets = [0]
    bounds_min = []
    bounds_max = []
    family_centroids = []
    family_normals = []
    family_areas = []
    family_parameterization_count = []
    family_online_eligible = []
    for family in range(len(unique_roots)):
        member_ids = np.flatnonzero(patch_family_ids == family)
        family_patch_ids.extend(member_ids.tolist())
        family_patch_offsets.append(len(family_patch_ids))
        vertex_ids = np.unique(
            np.concatenate([np.asarray(patches[row]["vertex_indices"], np.int64) for row in member_ids])
        )
        points = atlas.vertices_world[vertex_ids]
        bounds_min.append(points.min(0))
        bounds_max.append(points.max(0))
        areas = np.asarray([patches[row]["area"] for row in member_ids], np.float64)
        total_area = float(areas.sum())
        family_areas.append(total_area)
        family_centroids.append(
            np.average(
                np.stack([patches[row]["centroid"] for row in member_ids]),
                axis=0,
                weights=np.maximum(areas, 1e-12),
            )
        )
        normals = np.stack([patches[row]["normal"] for row in member_ids])
        reference = normals[0]
        normals = np.where((normals @ reference)[:, None] < 0, -normals, normals)
        normal = np.sum(normals * np.maximum(areas[:, None], 1e-12), axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        family_normals.append(normal)
        support = len(set(parameterizations[member_ids].tolist()))
        family_parameterization_count.append(support)
        family_online_eligible.append(
            support >= config.minimum_online_family_parameterizations
            and any(bool(patches[row]["eligible"]) for row in member_ids)
        )

    patch_vertex_offsets = [0]
    patch_vertex_indices: list[int] = []
    patch_face_offsets = [0]
    patch_face_indices: list[int] = []
    for patch in patches:
        patch_vertex_indices.extend(np.asarray(patch["vertex_indices"], np.int64).tolist())
        patch_vertex_offsets.append(len(patch_vertex_indices))
        patch_face_indices.extend(np.asarray(patch["face_indices"], np.int64).tolist())
        patch_face_offsets.append(len(patch_face_indices))
    edge_pairs = np.asarray([(row[0], row[1]) for row in accepted_rows], np.int32).reshape(-1, 2)
    values = [row[2] for row in accepted_rows]
    metadata: dict[str, object] = {
        "artifact_type": SCHEMA,
        "representation": "canonical_surface_families_over_anonymous_chart_parameterizations",
        "runtime_candidate_unit": "canonical_surface_family",
        "parameterization_count": int(len(atlas.chart_names)),
        "patch_count": len(patches),
        "family_count": len(unique_roots),
        "online_eligible_family_count": int(np.sum(family_online_eligible)),
        "multi_parameterization_patch_fraction": float(
            np.mean(np.asarray(family_parameterization_count)[patch_family_ids] >= 2)
        ),
        "online_supported_patch_area_fraction": float(
            np.sum(
                np.asarray([row["area"] for row in patches], np.float64)[
                    np.asarray(family_online_eligible, bool)[patch_family_ids]
                ]
            )
            / max(np.sum([row["area"] for row in patches]), 1e-12)
        ),
        "accepted_overlap_edge_count": len(accepted_rows),
        "parameterization_collision_rejected_edge_count": collision_rejected,
        "unassigned_atlas_vertex_count": unassigned_vertex_count,
        "source_atlas_content_sha256": source_atlas_content_sha256,
        "config": config.to_dict(),
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "source_chart_names_in_runtime": False,
        "anonymous_parameterization_ids_are_not_runtime_candidate_ids": True,
        "uses_query_or_ground_truth": False,
        "family_merge_guard": (
            "descending overlap score; components with an existing identical anonymous "
            "parameterization cannot merge"
        ),
        "canonical_geometry_fused": False,
        "family_canonicalization_gate_pass": bool(
            np.any(family_online_eligible)
            and (
                np.sum(
                    np.asarray([row["area"] for row in patches], np.float64)[
                        np.asarray(family_online_eligible, bool)[patch_family_ids]
                    ]
                )
                / max(np.sum([row["area"] for row in patches]), 1e-12)
            )
            >= config.minimum_online_supported_patch_area_fraction
        ),
        "carrier_role": (
            "groups compatible aligned patches and UV parameterizations; a later geometry "
            "fusion stage must create one shared mesh per eligible family"
        ),
    }
    carrier = CanonicalSurfaceFamilyCarrier(
        patch_parameterization_ids=parameterizations,
        patch_family_ids=patch_family_ids,
        patch_vertex_offsets=np.asarray(patch_vertex_offsets, np.int64),
        patch_vertex_indices=np.asarray(patch_vertex_indices, np.int64),
        patch_face_offsets=np.asarray(patch_face_offsets, np.int64),
        patch_face_indices=np.asarray(patch_face_indices, np.int64),
        patch_area_m2=np.asarray([row["area"] for row in patches], np.float32),
        patch_centroids_world=np.asarray([row["centroid"] for row in patches], np.float32),
        patch_normals_world=np.asarray([row["normal"] for row in patches], np.float32),
        patch_geometry_eligible=np.asarray([row["eligible"] for row in patches], bool),
        family_patch_offsets=np.asarray(family_patch_offsets, np.int64),
        family_patch_ids=np.asarray(family_patch_ids, np.int32),
        family_bounds_min_world=np.asarray(bounds_min, np.float32),
        family_bounds_max_world=np.asarray(bounds_max, np.float32),
        family_centroids_world=np.asarray(family_centroids, np.float32),
        family_normals_world=np.asarray(family_normals, np.float32),
        family_area_m2=np.asarray(family_areas, np.float32),
        family_parameterization_count=np.asarray(family_parameterization_count, np.int32),
        family_online_eligible=np.asarray(family_online_eligible, bool),
        overlap_edges=edge_pairs,
        overlap_scores=np.asarray([row["score"] for row in values], np.float32),
        overlap_smaller_patch_fraction=np.asarray(
            [row["smaller_patch_fraction"] for row in values], np.float32,
        ),
        overlap_larger_patch_fraction=np.asarray(
            [row["larger_patch_fraction"] for row in values], np.float32,
        ),
        overlap_point_to_plane_median_m=np.asarray(
            [row["point_to_plane_median_m"] for row in values], np.float32,
        ),
        overlap_unsigned_normal_median_deg=np.asarray(
            [row["unsigned_normal_median_deg"] for row in values], np.float32,
        ),
        metadata=metadata,
    ).validated()
    lineage = {
        "artifact_type": "goal_maplet_canonical_surface_family_lineage_v1",
        "parameterization_id_to_source_chart_name": {
            str(row): str(name) for row, name in enumerate(atlas.chart_names.tolist())
        },
        "source_chart_names_are_offline_lineage_only": True,
        "source_atlas_content_sha256": source_atlas_content_sha256,
        "uses_query_or_ground_truth": False,
    }
    lineage["content_sha256"] = canonical_json_sha256(lineage)
    return carrier, lineage


__all__ = [
    "CanonicalSurfaceFamilyCarrier",
    "SurfaceFamilyConfig",
    "build_canonical_surface_families",
]
