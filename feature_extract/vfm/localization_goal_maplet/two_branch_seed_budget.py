"""Pose-free baseline/layout position-seed sweep with branch-local products.

The represented domain is always ``(P_b x O_b) U (P_l x O_l)``.  Position
prefixes may vary, while each branch keeps its own frozen 64-orientation bank.
No cross-branch product is introduced.
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from .factorized_pose_free_proposal import (
    _validate_homogeneous_w2c,
    _validate_proper_rotations,
    build_factorized_pose_free_proposal_arrays,
    camera_centers_from_w2c,
)
from .lineage import arrays_sha256


SCHEMA = "goal_maplet_pose_free_two_branch_position_seed_budget_sweep_v1"
SEMANTICS = (
    "baseline_and_layout_branch_local_position_orientation_products_union_"
    "position_prefix_4_8_16_orientation64_no_cross_branch_product_v1"
)
BRANCH_NAMES = ("baseline", "layout")
POSITION_SEED_BUDGETS = (4, 8, 16)
ORIENTATION_BUDGET = 64
ARRAY_NAMES = (
    "image_ids", "position_seed_budgets", "position_offsets_camera",
    "branch_position_seed_candidate_ranks", "branch_position_seed_poses_w2c",
    "branch_position_centers_world", "branch_orientation_rotations_w2c",
    "branch_orientation_source_candidate_ranks", "branch_orientation_valid",
    "unique_position_seed_count_by_budget",
    "unique_position_factor_count_by_budget", "unique_orientation_factor_count",
    "implicit_lattice_pose_pair_count_by_budget",
)


def _key(value: np.ndarray) -> tuple[float, ...]:
    return tuple(np.asarray(value, dtype=np.float64).round(10).reshape(-1).tolist())


def _derived_counts(
    seed_pose: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    orientation_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_count = int(seed_pose.shape[0])
    unique_seed = np.zeros((query_count, len(POSITION_SEED_BUDGETS)), dtype=np.int16)
    unique_position = np.zeros(
        (query_count, len(POSITION_SEED_BUDGETS)), dtype=np.int32,
    )
    unique_orientation = np.zeros((query_count,), dtype=np.int16)
    implicit_pose = np.zeros(
        (query_count, len(POSITION_SEED_BUDGETS)), dtype=np.int64,
    )
    for query in range(query_count):
        orientation_sets = []
        for branch in range(2):
            valid = orientation_valid[query, branch]
            orientation_sets.append({
                _key(value) for value in orientation[query, branch, valid]
            })
        unique_orientation[query] = len(orientation_sets[0] | orientation_sets[1])
        for budget_index, budget in enumerate(POSITION_SEED_BUDGETS):
            seed_sets = [
                {_key(value) for value in seed_pose[query, branch, :budget]}
                for branch in range(2)
            ]
            position_sets = [
                {
                    _key(value) for value in
                    position[query, branch, :budget].reshape(-1, 3)
                }
                for branch in range(2)
            ]
            unique_seed[query, budget_index] = len(seed_sets[0] | seed_sets[1])
            unique_position[query, budget_index] = len(
                position_sets[0] | position_sets[1]
            )
            implicit_pose[query, budget_index] = (
                len(position_sets[0]) * len(orientation_sets[0])
                + len(position_sets[1]) * len(orientation_sets[1])
                - len(position_sets[0] & position_sets[1])
                * len(orientation_sets[0] & orientation_sets[1])
            )
    return unique_seed, unique_position, unique_orientation, implicit_pose


def build_two_branch_seed_budget_arrays(
    baseline_pool: dict[str, object], layout_pool: dict[str, object],
) -> dict[str, np.ndarray]:
    branches = [
        build_factorized_pose_free_proposal_arrays(
            pool, position_seed_count=max(POSITION_SEED_BUDGETS),
            orientation_budget=ORIENTATION_BUDGET,
            step_m=2.0, xz_half_extent_m=10.0, y_half_extent_m=4.0,
        )
        for pool in (baseline_pool, layout_pool)
    ]
    if (
        not np.array_equal(branches[0]["image_ids"], branches[1]["image_ids"])
        or not np.array_equal(
            branches[0]["position_offsets_camera"],
            branches[1]["position_offsets_camera"],
        )
    ):
        raise ValueError("two-branch seed sweep query/lattice contracts differ")
    seed_pose = np.stack(
        [value["position_seed_poses_w2c"] for value in branches], axis=1,
    )
    position = np.stack(
        [value["position_centers_world"] for value in branches], axis=1,
    )
    orientation = np.stack(
        [value["orientation_rotations_w2c"] for value in branches], axis=1,
    )
    orientation_valid = np.stack(
        [value["orientation_valid"] for value in branches], axis=1,
    )
    unique_seed, unique_position, unique_orientation, implicit_pose = (
        _derived_counts(seed_pose, position, orientation, orientation_valid)
    )
    return {
        "image_ids": branches[0]["image_ids"].copy(),
        "position_seed_budgets": np.asarray(POSITION_SEED_BUDGETS, dtype=np.int16),
        "position_offsets_camera": branches[0]["position_offsets_camera"].copy(),
        "branch_position_seed_candidate_ranks": np.stack([
            value["position_seed_candidate_ranks"] for value in branches
        ], axis=1),
        "branch_position_seed_poses_w2c": seed_pose,
        "branch_position_centers_world": position,
        "branch_orientation_rotations_w2c": orientation,
        "branch_orientation_source_candidate_ranks": np.stack([
            value["orientation_source_candidate_ranks"] for value in branches
        ], axis=1),
        "branch_orientation_valid": orientation_valid,
        "unique_position_seed_count_by_budget": unique_seed,
        "unique_position_factor_count_by_budget": unique_position,
        "unique_orientation_factor_count": unique_orientation,
        "implicit_lattice_pose_pair_count_by_budget": implicit_pose,
    }


def load_two_branch_seed_budget(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    artifact = Path(path)
    expected = {*(f"{name}.npy" for name in ARRAY_NAMES), "metadata_json.npy"}
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("two-branch seed sweep is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("two-branch seed sweep NPZ members differ")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*ARRAY_NAMES, "metadata_json"}:
            raise ValueError("two-branch seed sweep arrays differ")
        arrays = {name: np.asarray(data[name]) for name in ARRAY_NAMES}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if (
        metadata.get("artifact_type") != SCHEMA
        or metadata.get("semantics") != SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("branch_names") != list(BRANCH_NAMES)
        or metadata.get("cross_branch_cartesian_products_included") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
    ):
        raise ValueError("two-branch seed sweep metadata differs")
    image_ids = arrays["image_ids"]
    budgets = arrays["position_seed_budgets"]
    offsets = arrays["position_offsets_camera"]
    ranks = arrays["branch_position_seed_candidate_ranks"]
    seed_pose = arrays["branch_position_seed_poses_w2c"]
    position = arrays["branch_position_centers_world"]
    orientation = arrays["branch_orientation_rotations_w2c"]
    source_rank = arrays["branch_orientation_source_candidate_ranks"]
    orientation_valid = arrays["branch_orientation_valid"]
    query_count = int(image_ids.size)
    if (
        query_count <= 0 or image_ids.ndim != 1 or image_ids.dtype.kind != "U"
        or np.unique(image_ids).size != query_count
        or not np.array_equal(
            budgets, np.asarray(POSITION_SEED_BUDGETS, dtype=budgets.dtype),
        )
        or offsets.ndim != 2 or offsets.shape[1:] != (3,)
        or offsets.shape[0] != 605 or np.any(~np.isfinite(offsets))
        or ranks.shape != (query_count, 2, 16)
        or ranks.dtype.kind not in "iu"
        or seed_pose.shape != (query_count, 2, 16, 4, 4)
        or position.shape != (query_count, 2, 16, 605, 3)
        or orientation.shape != (query_count, 2, 64, 3, 3)
        or source_rank.shape != (query_count, 2, 64)
        or source_rank.dtype.kind not in "iu"
        or orientation_valid.shape != (query_count, 2, 64)
        or orientation_valid.dtype != np.bool_
        or not np.array_equal(
            ranks,
            np.broadcast_to(
                np.arange(1, 17, dtype=ranks.dtype)[None, None], ranks.shape,
            ),
        )
    ):
        raise ValueError("two-branch seed sweep shapes differ")
    seed_pose = _validate_homogeneous_w2c(
        seed_pose, context="two-branch position seed poses",
    )
    _validate_proper_rotations(
        orientation, context="two-branch orientation banks",
    )
    seed_centers = camera_centers_from_w2c(seed_pose)
    expected_position = seed_centers[..., None, :] + np.einsum(
        "qbsij,li->qbslj", seed_pose[..., :3, :3], offsets,
    )
    if not np.allclose(position, expected_position, atol=1.0e-10, rtol=0.0):
        raise ValueError("two-branch seed sweep lattice geometry differs")
    for query in range(query_count):
        for branch in range(2):
            count = int(np.sum(orientation_valid[query, branch]))
            valid = orientation_valid[query, branch]
            valid_ranks = source_rank[query, branch, :count]
            if (
                count <= 0
                or not np.array_equal(valid, np.arange(64) < count)
                or np.any(valid_ranks <= 0) or np.any(valid_ranks > 64)
                or np.unique(valid_ranks).size != count
                or np.any(np.diff(valid_ranks) <= 0)
                or np.any(source_rank[query, branch, count:] != -1)
            ):
                raise ValueError("two-branch orientation provenance differs")
    derived = _derived_counts(seed_pose, position, orientation, orientation_valid)
    stored = (
        arrays["unique_position_seed_count_by_budget"],
        arrays["unique_position_factor_count_by_budget"],
        arrays["unique_orientation_factor_count"],
        arrays["implicit_lattice_pose_pair_count_by_budget"],
    )
    if any(not np.array_equal(left, right) for left, right in zip(derived, stored)):
        raise ValueError("two-branch seed sweep derived counts differ")
    if (
        int(metadata.get("query_count", -1)) != query_count
        or metadata.get("position_seed_budgets") != list(POSITION_SEED_BUDGETS)
        or int(metadata.get("orientation_budget", -1)) != ORIENTATION_BUDGET
        or int(metadata.get("position_offsets_per_seed", -1)) != 605
    ):
        raise ValueError("two-branch seed sweep metadata counts differ")
    return arrays, metadata
