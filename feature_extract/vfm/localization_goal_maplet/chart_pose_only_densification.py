"""Source-only camera-window planning for a denser explicit-chart submap run.

The source window is frozen before any held-route camera is inspected.  Held
mapping cameras are used only afterwards to propose independent validation
rays.  No RGB, point map, mesh, initializer, query image, or query/GT pose is an
input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .lineage import canonical_json_sha256


SCHEMA = "goal_maplet_pose_only_chart_densification_plan_v2"


@dataclass(frozen=True)
class MappingCameraPose:
    name: str
    global_lexical_index: int
    center_world: np.ndarray
    forward_world: np.ndarray

    def validated(self) -> "MappingCameraPose":
        if not self.name or "__" not in self.name:
            raise ValueError("mapping camera name lacks a route prefix")
        if self.global_lexical_index < 0:
            raise ValueError(f"{self.name}: negative lexical index")
        center = np.asarray(self.center_world, np.float64)
        forward = np.asarray(self.forward_world, np.float64)
        if center.shape != (3,) or forward.shape != (3,):
            raise ValueError(f"{self.name}: camera vectors must have shape (3,)")
        if not np.isfinite(center).all() or not np.isfinite(forward).all():
            raise ValueError(f"{self.name}: nonfinite camera pose")
        if not 0.999 < np.linalg.norm(forward) < 1.001:
            raise ValueError(f"{self.name}: forward axis is not normalized")
        return self

    @property
    def route(self) -> str:
        return self.name.split("__", 1)[0]


@dataclass(frozen=True)
class PoseOnlyDensificationConfig:
    source_view_count: int = 24
    maximum_adjacent_baseline_m: float = 1.5
    maximum_adjacent_forward_angle_deg: float = 12.0
    minimum_path_length_m: float = 15.0
    held_neighbor_radius_m: float = 6.0
    held_neighbor_forward_angle_deg: float = 45.0
    held_target_view_count_per_route: int = 12

    def validated(self) -> "PoseOnlyDensificationConfig":
        if not 16 <= self.source_view_count <= 24:
            raise ValueError("source_view_count must be in [16, 24]")
        if self.maximum_adjacent_baseline_m <= 0:
            raise ValueError("maximum adjacent baseline must be positive")
        if not 0 < self.maximum_adjacent_forward_angle_deg < 180:
            raise ValueError("invalid maximum adjacent forward angle")
        if self.minimum_path_length_m <= 0:
            raise ValueError("minimum path length must be positive")
        if self.held_neighbor_radius_m <= 0:
            raise ValueError("held neighbor radius must be positive")
        if not 0 < self.held_neighbor_forward_angle_deg < 180:
            raise ValueError("invalid held neighbor forward angle")
        if self.held_target_view_count_per_route < 1:
            raise ValueError("held target view count must be positive")
        return self

    def to_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def _forward_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.degrees(
        np.arccos(np.clip(np.sum(first * second, axis=-1), -1.0, 1.0))
    )


def _validate_inventory(
    cameras: Iterable[MappingCameraPose],
    *,
    expected_route: str | None = None,
) -> list[MappingCameraPose]:
    rows = [camera.validated() for camera in cameras]
    if not rows or len({row.name for row in rows}) != len(rows):
        raise ValueError("mapping camera inventory is empty or duplicated")
    if len({row.global_lexical_index for row in rows}) != len(rows):
        raise ValueError("mapping camera lexical indices are duplicated")
    rows.sort(key=lambda row: row.name)
    if expected_route is not None and any(row.route != expected_route for row in rows):
        raise ValueError("mapping camera inventory differs from expected route")
    return rows


def select_source_camera_window(
    source_cameras: Iterable[MappingCameraPose],
    *,
    config: PoseOnlyDensificationConfig,
) -> tuple[list[MappingCameraPose], list[dict[str, object]]]:
    """Select a local source window using source-route camera poses only.

    Eligible fixed-length windows are ordered lexicographically by
    ``(endpoint_displacement / path_length, path_length, earliest_start)``.
    The returned candidate table contains every scanned window, including
    rejected windows and their explicit gate reasons.
    """

    config = config.validated()
    rows = _validate_inventory(source_cameras)
    if len({row.route for row in rows}) != 1:
        raise ValueError("source window must come from exactly one route")
    count = config.source_view_count
    if len(rows) < count:
        raise ValueError("source route has fewer cameras than the requested window")
    candidates: list[dict[str, object]] = []
    eligible: list[tuple[tuple[float, float, int], int]] = []
    for start in range(len(rows) - count + 1):
        window = rows[start : start + count]
        centers = np.stack([row.center_world for row in window])
        forwards = np.stack([row.forward_world for row in window])
        adjacent_baseline = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        adjacent_angle = _forward_angles(forwards[:-1], forwards[1:])
        path_length = float(adjacent_baseline.sum())
        displacement = float(np.linalg.norm(centers[-1] - centers[0]))
        straightness = displacement / max(path_length, 1e-12)
        reasons: list[str] = []
        if float(adjacent_baseline.max()) > config.maximum_adjacent_baseline_m:
            reasons.append("adjacent_baseline_above_limit")
        if float(adjacent_angle.max()) > config.maximum_adjacent_forward_angle_deg:
            reasons.append("adjacent_forward_angle_above_limit")
        if path_length < config.minimum_path_length_m:
            reasons.append("path_length_below_minimum")
        passed = not reasons
        candidate = {
            "candidate_rank_after_gate": None,
            "start_name": window[0].name,
            "end_name": window[-1].name,
            "start_global_lexical_index": window[0].global_lexical_index,
            "end_global_lexical_index": window[-1].global_lexical_index,
            "view_count": count,
            "adjacent_baseline_m": {
                "minimum": float(adjacent_baseline.min()),
                "median": float(np.median(adjacent_baseline)),
                "maximum": float(adjacent_baseline.max()),
            },
            "adjacent_forward_angle_deg": {
                "median": float(np.median(adjacent_angle)),
                "maximum": float(adjacent_angle.max()),
            },
            "path_length_m": path_length,
            "endpoint_displacement_m": displacement,
            "endpoint_displacement_over_path": straightness,
            "source_only_gate_pass": passed,
            "gate_reasons": reasons,
        }
        candidates.append(candidate)
        if passed:
            # The negative start index makes an exact tie choose the earliest
            # lexical source window without consulting another route.
            eligible.append(((straightness, path_length, -start), start))
    if not eligible:
        raise ValueError("no source-only camera window passes the densification gate")
    ranked = sorted(eligible, reverse=True)
    for rank, (_, start) in enumerate(ranked):
        candidates[start]["candidate_rank_after_gate"] = rank
    selected_start = ranked[0][1]
    return rows[selected_start : selected_start + count], candidates


def diagnose_held_camera_neighbors(
    selected_source: Iterable[MappingCameraPose],
    held_cameras_by_route: dict[str, Iterable[MappingCameraPose]],
    *,
    config: PoseOnlyDensificationConfig,
) -> tuple[dict[str, dict[str, object]], list[MappingCameraPose]]:
    """Run a post-selection camera-only held-neighbor diagnostic."""

    config = config.validated()
    source = _validate_inventory(selected_source)
    source_centers = np.stack([row.center_world for row in source])
    source_forwards = np.stack([row.forward_world for row in source])
    diagnostics: dict[str, dict[str, object]] = {}
    frozen_held: list[MappingCameraPose] = []
    for route in sorted(held_cameras_by_route):
        held = _validate_inventory(held_cameras_by_route[route], expected_route=route)
        held_centers = np.stack([row.center_world for row in held])
        held_forwards = np.stack([row.forward_world for row in held])
        distances = np.linalg.norm(
            source_centers[:, None, :] - held_centers[None, :, :], axis=2
        )
        angles = np.degrees(
            np.arccos(np.clip(source_forwards @ held_forwards.T, -1.0, 1.0))
        )
        compatible = (
            (distances <= config.held_neighbor_radius_m)
            & (angles <= config.held_neighbor_forward_angle_deg)
        )
        eligible_rows = np.flatnonzero(np.any(compatible, axis=0))
        eligible = [held[row] for row in eligible_rows]
        target_count = min(config.held_target_view_count_per_route, len(eligible))
        if target_count:
            # Uniform-in-ordered-inventory validation rays are selected only
            # after source freeze.  They never influence source ranking.
            selected_local = np.rint(
                np.linspace(0, len(eligible) - 1, target_count)
            ).astype(np.int64)
            selected = [eligible[int(row)] for row in selected_local]
        else:
            selected = []
        held_index_by_name = {row.name: index for index, row in enumerate(held)}
        selected_columns = [held_index_by_name[row.name] for row in selected]
        selected_compatible = (
            compatible[:, selected_columns]
            if selected_columns
            else np.zeros((len(source), 0), dtype=bool)
        )
        selected_distances = (
            distances[:, selected_columns]
            if selected_columns
            else np.zeros((len(source), 0), dtype=np.float64)
        )
        diagnostics[route] = {
            "eligible_camera_count": len(eligible),
            "eligible_ordered_names": [row.name for row in eligible],
            "eligible_global_lexical_indices": [row.global_lexical_index for row in eligible],
            "selected_camera_count": len(selected),
            "selected_ordered_names": [row.name for row in selected],
            "selected_global_lexical_indices": [row.global_lexical_index for row in selected],
            "source_view_count_covered_by_all_eligible": int(
                np.any(compatible[:, eligible_rows], axis=1).sum()
            ) if len(eligible_rows) else 0,
            "source_view_count_covered_by_selected": int(
                np.any(selected_compatible, axis=1).sum()
            ) if selected_columns else 0,
            "selected_nearest_source_distance_m": (
                np.min(selected_distances, axis=0).tolist() if selected_columns else []
            ),
            "selection_used_by_source_window_ranker": False,
        }
        frozen_held.extend(selected)
    frozen_held.sort(key=lambda row: row.global_lexical_index)
    return diagnostics, frozen_held


def ordered_camera_names_sha256(cameras: Iterable[MappingCameraPose]) -> str:
    return canonical_json_sha256([row.name for row in cameras])
