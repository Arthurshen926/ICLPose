"""Pose-free factorized position/orientation support around retrieval seeds.

The artifact stores a bounded position lattice and an orientation bank as two
independent factors.  It never materializes their Cartesian pose product and
never accepts a query target pose.  Ground truth may be opened only by a
separate Phase-2 coverage evaluator after this artifact is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .lineage import arrays_sha256


SCHEMA = "goal_maplet_pose_free_factorized_position_orientation_proposal_v1"
SEMANTICS = (
    "top4_retrieval_position_seeds_camera_local_2m_box_xz10_y4_"
    "independent_retrieval_orientation_bank_no_cartesian_materialization_v1"
)

FACTOR_ARRAY_NAMES = (
    "image_ids",
    "position_seed_candidate_ranks",
    "position_seed_poses_w2c",
    "position_seed_centers_world",
    "position_offsets_camera",
    "position_centers_world",
    "orientation_rotations_w2c",
    "orientation_source_candidate_ranks",
    "orientation_valid",
)


def _validate_proper_rotations(
    rotations: np.ndarray,
    *,
    context: str,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Reject finite matrices that are not members of SO(3)."""

    value = np.asarray(rotations, dtype=np.float64)
    if value.ndim < 2 or value.shape[-2:] != (3, 3) or np.any(~np.isfinite(value)):
        raise ValueError(f"{context} must contain finite 3x3 rotations")
    flat = value.reshape(-1, 3, 3)
    orthogonal_error = np.linalg.norm(
        flat @ np.swapaxes(flat, 1, 2) - np.eye(3, dtype=np.float64),
        axis=(1, 2),
    )
    determinant = np.linalg.det(flat)
    if (
        np.any(orthogonal_error > float(tolerance))
        or np.any(np.abs(determinant - 1.0) > float(tolerance))
    ):
        raise ValueError(f"{context} must contain finite proper SO(3) rotations")
    return value


def _validate_homogeneous_w2c(poses_w2c: np.ndarray, *, context: str) -> np.ndarray:
    pose = np.asarray(poses_w2c, dtype=np.float64)
    if pose.ndim < 2 or pose.shape[-2:] != (4, 4) or np.any(~np.isfinite(pose)):
        raise ValueError(f"{context} must contain finite 4x4 transforms")
    expected_last_row = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if np.any(np.abs(pose[..., 3, :] - expected_last_row) > 1.0e-8):
        raise ValueError(f"{context} must contain homogeneous w2c transforms")
    _validate_proper_rotations(pose[..., :3, :3], context=f"{context} rotation")
    return pose


def camera_centers_from_w2c(poses_w2c: np.ndarray) -> np.ndarray:
    pose = _validate_homogeneous_w2c(
        poses_w2c, context="factorized proposal poses",
    )
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    return (-np.swapaxes(rotation, -1, -2) @ translation[..., None])[..., 0]


def camera_local_position_lattice_offsets(
    *,
    step_m: float = 2.0,
    xz_half_extent_m: float = 10.0,
    y_half_extent_m: float = 4.0,
) -> np.ndarray:
    """Return deterministic x-major/y-middle/z-minor camera-local offsets."""

    step = float(step_m)
    xz = float(xz_half_extent_m)
    y = float(y_half_extent_m)
    if (
        not np.isfinite(step) or step <= 0.0
        or not np.isfinite(xz) or xz < 0.0
        or not np.isfinite(y) or y < 0.0
        or abs(xz / step - round(xz / step)) > 1.0e-12
        or abs(y / step - round(y / step)) > 1.0e-12
    ):
        raise ValueError("camera-local lattice bounds must be nonnegative step multiples")
    x_axis = np.arange(-xz, xz + 0.5 * step, step, dtype=np.float64)
    y_axis = np.arange(-y, y + 0.5 * step, step, dtype=np.float64)
    z_axis = np.arange(-xz, xz + 0.5 * step, step, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    offsets = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    if np.sum(np.all(offsets == 0.0, axis=1)) != 1:
        raise AssertionError("factorized position lattice lost its unique seed offset")
    return offsets


def _rotation_key(rotation: np.ndarray) -> tuple[float, ...]:
    return tuple(np.asarray(rotation, dtype=np.float64).round(10).reshape(-1).tolist())


def build_factorized_pose_free_proposal_arrays(
    pool: Mapping[str, object],
    *,
    position_seed_count: int = 4,
    orientation_budget: int = 64,
    step_m: float = 2.0,
    xz_half_extent_m: float = 10.0,
    y_half_extent_m: float = 4.0,
) -> dict[str, np.ndarray]:
    """Build factors from a validated pose-free pool without a target-pose API."""

    seed_count = int(position_seed_count)
    orientation_limit = int(orientation_budget)
    maximum = int(pool.get("maximum_modes", 0))
    rows = pool.get("rows")
    if (
        not isinstance(rows, list) or not rows or seed_count <= 0
        or orientation_limit <= 0 or seed_count > maximum
        or orientation_limit > maximum
    ):
        raise ValueError("factorized proposal budgets exceed the pose-free pool")
    offsets = camera_local_position_lattice_offsets(
        step_m=step_m,
        xz_half_extent_m=xz_half_extent_m,
        y_half_extent_m=y_half_extent_m,
    )
    image_ids: list[str] = []
    seed_poses: list[np.ndarray] = []
    seed_centers: list[np.ndarray] = []
    position_centers: list[np.ndarray] = []
    orientation_rows: list[np.ndarray] = []
    orientation_ranks: list[np.ndarray] = []
    orientation_valid: list[np.ndarray] = []
    seen_image_ids: set[str] = set()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        details_by_mode = row.get("mode_details")
        details = (
            details_by_mode.get("actual_parent_actual_child")
            if isinstance(details_by_mode, dict) else None
        )
        if (
            not image_id or image_id in seen_image_ids
            or not isinstance(details, list) or len(details) < maximum
        ):
            raise ValueError("factorized proposal pool row is incomplete")
        seen_image_ids.add(image_id)
        poses = np.stack([
            np.asarray(detail.get("pose_w2c"), dtype=np.float64)
            for detail in details[:maximum]
        ])
        if poses.shape != (maximum, 4, 4):
            raise ValueError("factorized proposal pool pose is invalid")
        poses = _validate_homogeneous_w2c(
            poses, context="factorized proposal pool poses",
        )
        for expected, detail in enumerate(details[:maximum], start=1):
            if int(detail.get("rank", -1)) != expected:
                raise ValueError("factorized proposal pool ranks are not a strict prefix")
        seeds = poses[:seed_count]
        centers = camera_centers_from_w2c(seeds)
        # A camera-frame displacement d_c corresponds to world displacement
        # R_w2c^T d_c.  This is not a world-axis lattice.
        world_offsets = np.einsum(
            "sij,li->slj", seeds[:, :3, :3], offsets,
        )
        positions = centers[:, None, :] + world_offsets

        unique_rotations: list[np.ndarray] = []
        source_ranks: list[int] = []
        seen: set[tuple[float, ...]] = set()
        for rank, pose in enumerate(poses[:orientation_limit], start=1):
            key = _rotation_key(pose[:3, :3])
            if key in seen:
                continue
            seen.add(key)
            unique_rotations.append(pose[:3, :3])
            source_ranks.append(rank)
        if not unique_rotations:
            raise ValueError("factorized proposal orientation bank is empty")
        padded = np.repeat(np.eye(3, dtype=np.float64)[None], orientation_limit, axis=0)
        rank_values = np.full((orientation_limit,), -1, dtype=np.int16)
        valid = np.zeros((orientation_limit,), dtype=bool)
        count = len(unique_rotations)
        padded[:count] = np.stack(unique_rotations)
        rank_values[:count] = np.asarray(source_ranks, dtype=np.int16)
        valid[:count] = True
        image_ids.append(image_id)
        seed_poses.append(seeds)
        seed_centers.append(centers)
        position_centers.append(positions)
        orientation_rows.append(padded)
        orientation_ranks.append(rank_values)
        orientation_valid.append(valid)
    return {
        "image_ids": np.asarray(image_ids),
        "position_seed_candidate_ranks": np.broadcast_to(
            np.arange(1, seed_count + 1, dtype=np.int16)[None],
            (len(image_ids), seed_count),
        ).copy(),
        "position_seed_poses_w2c": np.stack(seed_poses),
        "position_seed_centers_world": np.stack(seed_centers),
        "position_offsets_camera": offsets,
        "position_centers_world": np.stack(position_centers),
        "orientation_rotations_w2c": np.stack(orientation_rows),
        "orientation_source_candidate_ranks": np.stack(orientation_ranks),
        "orientation_valid": np.stack(orientation_valid),
    }


def load_factorized_pose_free_proposal(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    artifact = Path(path)
    expected_npz_members = {
        *(f"{name}.npy" for name in FACTOR_ARRAY_NAMES), "metadata_json.npy",
    }
    try:
        with zipfile.ZipFile(artifact, mode="r") as archive:
            members = [value.filename for value in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("factorized proposal is not a valid NPZ archive") from error
    if len(members) != len(set(members)):
        raise ValueError("factorized proposal NPZ contains duplicate ZIP members")
    if set(members) != expected_npz_members or len(members) != len(expected_npz_members):
        raise ValueError("factorized proposal NPZ members differ from the exact schema")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*FACTOR_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("factorized proposal array members differ from the exact schema")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {
            name: np.asarray(data[name]) for name in data.files if name != "metadata_json"
        }
    if metadata.get("artifact_type") != SCHEMA or metadata.get("semantics") != SEMANTICS:
        raise ValueError("not a factorized pose-free proposal")
    if metadata.get("content_sha256") != arrays_sha256(arrays):
        raise ValueError("factorized proposal content hash differs")
    required_false = (
        "uses_query_pose", "uses_query_ground_truth", "uses_alike", "uses_pnp",
        "cartesian_pose_product_materialized",
    )
    if any(metadata.get(flag) is not False for flag in required_false):
        raise ValueError("factorized proposal violates its pose-free boundary")
    if (
        metadata.get("position_collision_free_space_certified") is not False
        or metadata.get("position_lattice_occupancy_checked") is not False
        or metadata.get("raw_coverage_is_implicit_factor_support_upper_bound_only") is not True
    ):
        raise ValueError("factorized proposal lacks its coarse-support safety disclaimer")
    image_ids = np.asarray(arrays["image_ids"]).reshape(-1)
    seed_ranks = np.asarray(arrays["position_seed_candidate_ranks"])
    seeds = np.asarray(arrays["position_seed_poses_w2c"], dtype=np.float64)
    seed_centers = np.asarray(arrays["position_seed_centers_world"], dtype=np.float64)
    offsets = np.asarray(arrays["position_offsets_camera"], dtype=np.float64)
    centers = np.asarray(arrays["position_centers_world"], dtype=np.float64)
    rotations = np.asarray(arrays["orientation_rotations_w2c"], dtype=np.float64)
    source_ranks = np.asarray(arrays["orientation_source_candidate_ranks"])
    valid_raw = np.asarray(arrays["orientation_valid"])
    valid = valid_raw.astype(bool, copy=False)
    query_count = int(image_ids.size)
    seed_count = int(seeds.shape[1]) if seeds.ndim == 4 else 0
    orientation_count = int(rotations.shape[1]) if rotations.ndim == 4 else 0
    if (
        query_count == 0 or image_ids.dtype.kind != "U"
        or any(not str(value) for value in image_ids.tolist())
        or np.unique(image_ids).size != query_count
        or seeds.ndim != 4 or seeds.shape[0] != query_count
        or seeds.shape[2:] != (4, 4) or seed_count <= 0
        or seed_ranks.shape != (query_count, seed_count)
        or seed_ranks.dtype.kind not in "iu"
        or seed_centers.shape != (query_count, seed_count, 3)
        or offsets.ndim != 2 or offsets.shape[1:] != (3,) or offsets.shape[0] <= 0
        or centers.ndim != 4
        or centers.shape[:2] != seeds.shape[:2] or centers.shape[3] != 3
        or centers.shape[2] != offsets.shape[0]
        or rotations.ndim != 4 or rotations.shape[0] != query_count
        or rotations.shape[2:] != (3, 3) or valid.shape != rotations.shape[:2]
        or orientation_count <= 0
        or source_ranks.shape != (query_count, orientation_count)
        or source_ranks.dtype.kind not in "iu" or valid_raw.dtype != np.bool_
        or np.any(~np.isfinite(seeds)) or np.any(~np.isfinite(centers))
        or np.any(~np.isfinite(seed_centers)) or np.any(~np.isfinite(offsets))
        or np.any(~np.isfinite(rotations)) or np.any(np.sum(valid, axis=1) <= 0)
    ):
        raise ValueError("factorized proposal arrays differ from the frozen contract")
    seeds = _validate_homogeneous_w2c(seeds, context="position seed poses")
    _validate_proper_rotations(rotations, context="orientation bank")
    expected_seed_ranks = np.broadcast_to(
        np.arange(1, seed_count + 1, dtype=seed_ranks.dtype)[None],
        seed_ranks.shape,
    )
    if not np.array_equal(seed_ranks, expected_seed_ranks):
        raise ValueError("position seed ranks are not the strict pool prefix")
    recomputed_seed_centers = camera_centers_from_w2c(seeds)
    if not np.allclose(seed_centers, recomputed_seed_centers, atol=1.0e-10, rtol=0.0):
        raise ValueError("stored position seed centers differ from seed poses")
    expected_centers = seed_centers[:, :, None, :] + np.einsum(
        "qsij,li->qslj", seeds[..., :3, :3], offsets,
    )
    if not np.allclose(centers, expected_centers, atol=1.0e-10, rtol=0.0):
        raise ValueError("stored position lattice differs from camera-local seed geometry")
    source_prefix_budget = int(metadata.get("orientation_source_prefix_budget", 0))
    identity = np.eye(3, dtype=np.float64)
    for query in range(query_count):
        count = int(np.sum(valid[query]))
        if not np.array_equal(
            valid[query], np.arange(orientation_count, dtype=np.int64) < count,
        ):
            raise ValueError("orientation validity must be a strict prefix")
        ranks = source_ranks[query, :count]
        if (
            np.any(ranks <= 0) or np.any(ranks > source_prefix_budget)
            or np.unique(ranks).size != count or np.any(np.diff(ranks) <= 0)
        ):
            raise ValueError("valid orientation source ranks must be unique increasing prefix indices")
        if (
            np.any(source_ranks[query, count:] != -1)
            or not np.array_equal(
                rotations[query, count:],
                np.broadcast_to(identity, (orientation_count - count, 3, 3)),
            )
        ):
            raise ValueError("invalid orientation padding is not canonical")
    if (
        int(metadata.get("query_count", -1)) != query_count
        or int(metadata.get("position_seed_count", -1)) != seed_count
        or int(metadata.get("positions_per_seed", -1)) != offsets.shape[0]
        or int(metadata.get("orientation_source_prefix_budget", -1)) != orientation_count
        or int(metadata.get("minimum_stored_orientation_count_per_query", -1))
        != int(np.min(np.sum(valid, axis=1)))
        or int(metadata.get("maximum_stored_orientation_count_per_query", -1))
        != int(np.max(np.sum(valid, axis=1)))
    ):
        raise ValueError("factorized proposal metadata counts differ from arrays")
    return arrays, metadata


def factorized_raw_coverage(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    orientation_valid: np.ndarray,
    target_poses_w2c: np.ndarray,
    *,
    position_seed_budgets: Sequence[int] = (1, 2, 4),
    orientation_budgets: Sequence[int] = (1, 4, 8, 16, 32, 64),
) -> dict[str, object]:
    """Evaluate exact implicit-product support without constructing poses."""

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    valid = np.asarray(orientation_valid, dtype=bool)
    target = np.asarray(target_poses_w2c, dtype=np.float64)
    q = target.shape[0] if target.ndim == 3 else 0
    if (
        q <= 0 or target.shape[1:] != (4, 4) or position.ndim != 4
        or position.shape[0] != q or position.shape[3] != 3
        or rotation.ndim != 4 or rotation.shape[0] != q
        or rotation.shape[2:] != (3, 3) or valid.shape != rotation.shape[:2]
    ):
        raise ValueError("factorized coverage arrays differ")
    target_center = camera_centers_from_w2c(target)
    translation_error = np.linalg.norm(
        position - target_center[:, None, None, :], axis=3,
    )
    relative = rotation @ np.swapaxes(target[:, None, :3, :3], 2, 3)
    cosine = np.clip(
        (np.trace(relative, axis1=2, axis2=3) - 1.0) / 2.0, -1.0, 1.0,
    )
    rotation_error = np.degrees(np.arccos(cosine))
    rotation_error = np.where(valid, rotation_error, np.inf)
    thresholds = {
        "region_2m_45deg": (2.0, 45.0),
        "loose_1m_10deg": (1.0, 10.0),
        "strict_0_5m_5deg": (0.5, 5.0),
    }
    rows: list[dict[str, object]] = []
    for seed_budget_value in position_seed_budgets:
        seed_budget = min(int(seed_budget_value), int(position.shape[1]))
        if seed_budget <= 0:
            raise ValueError("position seed budgets must be positive")
        best_translation = np.min(translation_error[:, :seed_budget], axis=(1, 2))
        for orientation_budget_value in orientation_budgets:
            orientation_budget = min(int(orientation_budget_value), int(rotation.shape[1]))
            if orientation_budget <= 0:
                raise ValueError("orientation budgets must be positive")
            best_rotation = np.min(rotation_error[:, :orientation_budget], axis=1)
            row: dict[str, object] = {
                "position_seed_budget": seed_budget,
                "orientation_budget": orientation_budget,
            }
            for name, (translation_limit, rotation_limit) in thresholds.items():
                hit = (best_translation <= translation_limit) & (
                    best_rotation <= rotation_limit
                )
                row[name] = {
                    "hits": int(np.sum(hit)), "query_count": q,
                    "rate": float(np.mean(hit)),
                }
            row["median_best_translation_m"] = float(np.median(best_translation))
            row["median_best_rotation_deg"] = float(np.median(best_rotation))
            rows.append(row)
    best_translation = np.min(translation_error, axis=(1, 2))
    best_rotation = np.min(rotation_error, axis=1)
    return {
        "query_count": q,
        "rows": rows,
        "full_factor_position_only": {
            "median_m": float(np.median(best_translation)),
            "p90_m": float(np.percentile(best_translation, 90)),
            **{
                f"le_{limit:g}m": {
                    "hits": int(np.sum(best_translation <= limit)),
                    "rate": float(np.mean(best_translation <= limit)),
                }
                for limit in (0.5, 1.0, 2.0)
            },
        },
        "full_factor_orientation_only": {
            "median_deg": float(np.median(best_rotation)),
            "p90_deg": float(np.percentile(best_rotation, 90)),
            **{
                f"le_{limit:g}deg": {
                    "hits": int(np.sum(best_rotation <= limit)),
                    "rate": float(np.mean(best_rotation <= limit)),
                }
                for limit in (5.0, 10.0, 45.0)
            },
        },
    }
