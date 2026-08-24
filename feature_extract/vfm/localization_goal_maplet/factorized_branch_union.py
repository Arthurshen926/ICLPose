"""Coverage-preserving union of two frozen factorized pose domains.

The union is ``(P_baseline x O_baseline) U (P_layout x O_layout)``.  Factors
are de-duplicated for storage and provenance, but cross-branch products such as
``P_baseline x O_layout`` are deliberately absent.
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from .factorized_pose_free_proposal import (
    FACTOR_ARRAY_NAMES,
    _validate_homogeneous_w2c,
    _validate_proper_rotations,
    camera_centers_from_w2c,
)
from .lineage import arrays_sha256


SCHEMA = "goal_maplet_pose_free_factorized_two_branch_domain_union_v1"
SEMANTICS = "union_of_baseline_domain_and_layout_domain_without_cross_branch_product_v1"
BRANCH_NAMES = ("baseline", "layout")
UNION_ARRAY_NAMES = (
    "image_ids", "position_offsets_camera",
    "unique_position_seed_poses_w2c", "unique_position_seed_valid",
    "unique_position_centers_world", "branch_position_seed_to_unique",
    "unique_position_seed_branch_mask",
    "unique_position_seed_source_candidate_ranks",
    "unique_orientation_rotations_w2c", "unique_orientation_valid",
    "branch_orientation_to_unique", "branch_orientation_valid",
    "unique_orientation_branch_mask",
    "unique_orientation_source_candidate_ranks",
    "implicit_seed_orientation_pair_count_by_query",
    "implicit_lattice_pose_pair_count_by_query",
)


def _key(value: np.ndarray) -> tuple[float, ...]:
    return tuple(np.asarray(value, dtype=np.float64).round(10).reshape(-1).tolist())


def build_factorized_branch_union_arrays(
    baseline: dict[str, np.ndarray], layout: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """De-duplicate factors while retaining both original branch domains."""

    if any(name not in baseline or name not in layout for name in FACTOR_ARRAY_NAMES):
        raise ValueError("factorized branch lacks required arrays")
    if (
        not np.array_equal(baseline["image_ids"], layout["image_ids"])
        or not np.array_equal(
            baseline["position_offsets_camera"], layout["position_offsets_camera"]
        )
    ):
        raise ValueError("factorized branch query/lattice contracts differ")
    image_ids = np.asarray(baseline["image_ids"])
    query_count = int(image_ids.size)
    seed_budget = int(np.asarray(baseline["position_seed_poses_w2c"]).shape[1])
    orientation_budget = int(
        np.asarray(baseline["orientation_rotations_w2c"]).shape[1]
    )
    if (
        seed_budget != 4
        or np.asarray(layout["position_seed_poses_w2c"]).shape[1] != seed_budget
        or orientation_budget != 64
        or np.asarray(layout["orientation_rotations_w2c"]).shape[1]
        != orientation_budget
    ):
        raise ValueError("factorized branch budgets must remain 4x64")
    offset_count = int(np.asarray(baseline["position_offsets_camera"]).shape[0])
    maximum_seed = 2 * seed_budget
    maximum_orientation = 2 * orientation_budget
    seed_poses = np.broadcast_to(
        np.eye(4, dtype=np.float64), (query_count, maximum_seed, 4, 4),
    ).copy()
    seed_valid = np.zeros((query_count, maximum_seed), dtype=bool)
    position_centers = np.zeros(
        (query_count, maximum_seed, offset_count, 3), dtype=np.float64,
    )
    branch_seed_map = np.full((query_count, 2, seed_budget), -1, dtype=np.int16)
    seed_branch_mask = np.zeros((query_count, maximum_seed, 2), dtype=bool)
    seed_source_rank = np.full((query_count, maximum_seed, 2), -1, dtype=np.int16)
    orientations = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (query_count, maximum_orientation, 3, 3),
    ).copy()
    orientation_valid = np.zeros((query_count, maximum_orientation), dtype=bool)
    branch_orientation_map = np.full(
        (query_count, 2, orientation_budget), -1, dtype=np.int16,
    )
    branch_orientation_valid = np.zeros(
        (query_count, 2, orientation_budget), dtype=bool,
    )
    orientation_branch_mask = np.zeros(
        (query_count, maximum_orientation, 2), dtype=bool,
    )
    orientation_source_rank = np.full(
        (query_count, maximum_orientation, 2), -1, dtype=np.int16,
    )
    seed_orientation_pair_count = np.zeros((query_count,), dtype=np.int32)
    lattice_pose_pair_count = np.zeros((query_count,), dtype=np.int64)

    sources = (baseline, layout)
    for query in range(query_count):
        seed_by_key: dict[tuple[float, ...], int] = {}
        for branch, source in enumerate(sources):
            source_pose = np.asarray(source["position_seed_poses_w2c"])[query]
            source_centers = np.asarray(source["position_centers_world"])[query]
            source_ranks = np.asarray(
                source["position_seed_candidate_ranks"], dtype=np.int16,
            )[query]
            for slot in range(seed_budget):
                key = _key(source_pose[slot])
                unique = seed_by_key.get(key)
                if unique is None:
                    unique = len(seed_by_key)
                    seed_by_key[key] = unique
                    seed_poses[query, unique] = source_pose[slot]
                    position_centers[query, unique] = source_centers[slot]
                    seed_valid[query, unique] = True
                elif (
                    not np.allclose(seed_poses[query, unique], source_pose[slot], atol=1e-10, rtol=0)
                    or not np.allclose(
                        position_centers[query, unique], source_centers[slot],
                        atol=1e-10, rtol=0,
                    )
                ):
                    raise ValueError("equal position factor identity has different geometry")
                branch_seed_map[query, branch, slot] = unique
                seed_branch_mask[query, unique, branch] = True
                seed_source_rank[query, unique, branch] = source_ranks[slot]

        orientation_by_key: dict[tuple[float, ...], int] = {}
        for branch, source in enumerate(sources):
            source_rotation = np.asarray(source["orientation_rotations_w2c"])[query]
            source_valid = np.asarray(source["orientation_valid"], dtype=bool)[query]
            source_ranks = np.asarray(
                source["orientation_source_candidate_ranks"], dtype=np.int16,
            )[query]
            branch_orientation_valid[query, branch] = source_valid
            for slot in np.flatnonzero(source_valid).tolist():
                key = _key(source_rotation[slot])
                unique = orientation_by_key.get(key)
                if unique is None:
                    unique = len(orientation_by_key)
                    orientation_by_key[key] = unique
                    orientations[query, unique] = source_rotation[slot]
                    orientation_valid[query, unique] = True
                elif not np.allclose(
                    orientations[query, unique], source_rotation[slot],
                    atol=1e-10, rtol=0,
                ):
                    raise ValueError("equal orientation factor identity differs")
                branch_orientation_map[query, branch, slot] = unique
                orientation_branch_mask[query, unique, branch] = True
                orientation_source_rank[query, unique, branch] = source_ranks[slot]

        seed_sets = [set(branch_seed_map[query, branch].tolist()) for branch in range(2)]
        orientation_sets = [
            set(branch_orientation_map[query, branch][
                branch_orientation_valid[query, branch]
            ].tolist())
            for branch in range(2)
        ]
        branch_pair_count = [
            len(seed_sets[branch]) * len(orientation_sets[branch])
            for branch in range(2)
        ]
        intersection = len(seed_sets[0] & seed_sets[1]) * len(
            orientation_sets[0] & orientation_sets[1]
        )
        seed_orientation_pair_count[query] = sum(branch_pair_count) - intersection
        lattice_pose_pair_count[query] = (
            int(seed_orientation_pair_count[query]) * offset_count
        )
    return {
        "image_ids": image_ids.copy(),
        "position_offsets_camera": np.asarray(
            baseline["position_offsets_camera"], dtype=np.float64,
        ).copy(),
        "unique_position_seed_poses_w2c": seed_poses,
        "unique_position_seed_valid": seed_valid,
        "unique_position_centers_world": position_centers,
        "branch_position_seed_to_unique": branch_seed_map,
        "unique_position_seed_branch_mask": seed_branch_mask,
        "unique_position_seed_source_candidate_ranks": seed_source_rank,
        "unique_orientation_rotations_w2c": orientations,
        "unique_orientation_valid": orientation_valid,
        "branch_orientation_to_unique": branch_orientation_map,
        "branch_orientation_valid": branch_orientation_valid,
        "unique_orientation_branch_mask": orientation_branch_mask,
        "unique_orientation_source_candidate_ranks": orientation_source_rank,
        "implicit_seed_orientation_pair_count_by_query": seed_orientation_pair_count,
        "implicit_lattice_pose_pair_count_by_query": lattice_pose_pair_count,
    }


def load_factorized_branch_union(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    artifact = Path(path)
    expected = {*(f"{name}.npy" for name in UNION_ARRAY_NAMES), "metadata_json.npy"}
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("factorized branch-union is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("factorized branch-union NPZ members differ")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*UNION_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("factorized branch-union arrays differ")
        arrays = {name: np.asarray(data[name]) for name in UNION_ARRAY_NAMES}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if (
        metadata.get("artifact_type") != SCHEMA
        or metadata.get("semantics") != SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("branch_names") != list(BRANCH_NAMES)
        or metadata.get("cross_branch_cartesian_products_included") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("domain_union_is_branch_or_not_cartesian_hull") is not True
        or metadata.get("cartesian_pose_product_materialized") is not False
    ):
        raise ValueError("factorized branch-union metadata differs")
    image_ids = arrays["image_ids"]
    seed_valid = arrays["unique_position_seed_valid"]
    orientation_valid = arrays["unique_orientation_valid"]
    seed_map = arrays["branch_position_seed_to_unique"]
    orientation_map = arrays["branch_orientation_to_unique"]
    branch_orientation_valid = arrays["branch_orientation_valid"]
    seed_poses = arrays["unique_position_seed_poses_w2c"]
    centers = arrays["unique_position_centers_world"]
    offsets = arrays["position_offsets_camera"]
    seed_branch_mask = arrays["unique_position_seed_branch_mask"]
    seed_source_rank = arrays["unique_position_seed_source_candidate_ranks"]
    rotations = arrays["unique_orientation_rotations_w2c"]
    orientation_branch_mask = arrays["unique_orientation_branch_mask"]
    orientation_source_rank = arrays[
        "unique_orientation_source_candidate_ranks"
    ]
    pair_count = arrays["implicit_seed_orientation_pair_count_by_query"]
    lattice_pair_count = arrays["implicit_lattice_pose_pair_count_by_query"]
    q = int(image_ids.size)
    if (
        q <= 0 or image_ids.ndim != 1 or image_ids.dtype.kind != "U"
        or any(not str(value) for value in image_ids.tolist())
        or len(set(image_ids.tolist())) != q
        or offsets.ndim != 2 or offsets.shape[1:] != (3,)
        or offsets.shape[0] <= 0 or np.any(~np.isfinite(offsets))
        or seed_valid.dtype != np.bool_
        or seed_valid.shape != (q, 8)
        or seed_poses.shape != (q, 8, 4, 4)
        or centers.shape != (q, 8, offsets.shape[0], 3)
        or seed_map.shape != (q, 2, 4)
        or seed_map.dtype.kind not in "iu"
        or seed_branch_mask.dtype != np.bool_
        or seed_branch_mask.shape != (q, 8, 2)
        or seed_source_rank.shape != (q, 8, 2)
        or seed_source_rank.dtype.kind not in "iu"
        or orientation_valid.dtype != np.bool_
        or orientation_valid.shape != (q, 128)
        or rotations.shape != (q, 128, 3, 3)
        or orientation_map.shape != (q, 2, 64)
        or orientation_map.dtype.kind not in "iu"
        or branch_orientation_valid.dtype != np.bool_
        or branch_orientation_valid.shape != (q, 2, 64)
        or orientation_branch_mask.dtype != np.bool_
        or orientation_branch_mask.shape != (q, 128, 2)
        or orientation_source_rank.shape != (q, 128, 2)
        or orientation_source_rank.dtype.kind not in "iu"
        or pair_count.shape != (q,) or pair_count.dtype.kind not in "iu"
        or lattice_pair_count.shape != (q,)
        or lattice_pair_count.dtype.kind not in "iu"
        or np.any(~np.isfinite(seed_poses))
        or np.any(~np.isfinite(centers))
        or np.any(~np.isfinite(rotations))
        or np.any(seed_map < 0) or np.any(seed_map >= 8)
        or np.any(orientation_map[branch_orientation_valid] < 0)
        or np.any(orientation_map[branch_orientation_valid] >= 128)
        or np.any(orientation_map[~branch_orientation_valid] != -1)
    ):
        raise ValueError("factorized branch-union shapes/mappings differ")
    seed_poses = _validate_homogeneous_w2c(
        seed_poses, context="factorized branch-union position seeds",
    )
    _validate_proper_rotations(
        rotations, context="factorized branch-union orientations",
    )
    identity_pose = np.eye(4, dtype=np.float64)
    identity_rotation = np.eye(3, dtype=np.float64)
    for query in range(q):
        seed_count = int(np.sum(seed_valid[query]))
        orientation_count = int(np.sum(orientation_valid[query]))
        seed_indices = set(seed_map[query].reshape(-1).tolist())
        mapped_orientation = orientation_map[query][
            branch_orientation_valid[query]
        ]
        orientation_indices = set(mapped_orientation.tolist())
        expected_seed_mask = np.zeros((8, 2), dtype=bool)
        expected_orientation_mask = np.zeros((128, 2), dtype=bool)
        branch_seed_sets = []
        branch_orientation_sets = []
        for branch in range(2):
            branch_seeds = seed_map[query, branch]
            branch_orientations = orientation_map[query, branch][
                branch_orientation_valid[query, branch]
            ]
            branch_seed_sets.append(set(branch_seeds.tolist()))
            branch_orientation_sets.append(set(branch_orientations.tolist()))
            expected_seed_mask[branch_seeds, branch] = True
            expected_orientation_mask[branch_orientations, branch] = True
            count = int(np.sum(branch_orientation_valid[query, branch]))
            ranks = orientation_source_rank[
                query, branch_orientations, branch
            ]
            if (
                count <= 0
                or not np.array_equal(
                    branch_orientation_valid[query, branch],
                    np.arange(64) < count,
                )
                or np.unique(branch_orientations).size != count
                or np.any(ranks <= 0) or np.any(ranks > 64)
                or np.any(np.diff(ranks) <= 0)
            ):
                raise ValueError("factorized branch orientation provenance differs")
        expected_pair_count = sum(
            len(branch_seed_sets[branch]) * len(branch_orientation_sets[branch])
            for branch in range(2)
        ) - len(branch_seed_sets[0] & branch_seed_sets[1]) * len(
            branch_orientation_sets[0] & branch_orientation_sets[1]
        )
        if (
            seed_count <= 0 or orientation_count <= 0
            or not np.array_equal(seed_valid[query], np.arange(8) < seed_count)
            or not np.array_equal(
                orientation_valid[query], np.arange(128) < orientation_count
            )
            or seed_indices != set(range(seed_count))
            or orientation_indices != set(range(orientation_count))
            or np.any(~seed_valid[query, seed_map[query]])
            or np.any(~orientation_valid[query, orientation_map[query][
                branch_orientation_valid[query]
            ]])
            or not np.array_equal(seed_branch_mask[query], expected_seed_mask)
            or not np.array_equal(
                orientation_branch_mask[query], expected_orientation_mask,
            )
            or np.any(seed_source_rank[query][expected_seed_mask] <= 0)
            or np.any(seed_source_rank[query][expected_seed_mask] > 4)
            or np.any(seed_source_rank[query][~expected_seed_mask] != -1)
            or np.any(
                orientation_source_rank[query][expected_orientation_mask] <= 0
            )
            or np.any(
                orientation_source_rank[query][~expected_orientation_mask] != -1
            )
            or int(pair_count[query]) != expected_pair_count
            or int(lattice_pair_count[query])
            != expected_pair_count * int(offsets.shape[0])
        ):
            raise ValueError("factorized branch-union valid prefixes differ")
        valid_poses = seed_poses[query, :seed_count]
        seed_centers = camera_centers_from_w2c(valid_poses)
        expected_centers = seed_centers[:, None, :] + np.einsum(
            "sij,li->slj", valid_poses[:, :3, :3], offsets,
        )
        if (
            not np.allclose(
                centers[query, :seed_count], expected_centers,
                atol=1.0e-10, rtol=0.0,
            )
            or not np.array_equal(
                seed_poses[query, seed_count:],
                np.broadcast_to(identity_pose, (8 - seed_count, 4, 4)),
            )
            or np.any(centers[query, seed_count:] != 0.0)
            or not np.array_equal(
                rotations[query, orientation_count:],
                np.broadcast_to(
                    identity_rotation, (128 - orientation_count, 3, 3),
                ),
            )
        ):
            raise ValueError("factorized branch-union geometry/padding differs")
    seed_counts = np.sum(seed_valid, axis=1)
    orientation_counts = np.sum(orientation_valid, axis=1)
    if (
        int(metadata.get("query_count", -1)) != q
        or int(metadata.get("branch_position_seed_budget", -1)) != 4
        or int(metadata.get("branch_orientation_budget", -1)) != 64
        or int(metadata.get("position_offsets_per_seed", -1))
        != int(offsets.shape[0])
        or metadata.get("unique_position_seed_count_range")
        != [int(np.min(seed_counts)), int(np.max(seed_counts))]
        or metadata.get("unique_orientation_count_range")
        != [int(np.min(orientation_counts)), int(np.max(orientation_counts))]
        or metadata.get("implicit_seed_orientation_pair_count_range")
        != [int(np.min(pair_count)), int(np.max(pair_count))]
        or metadata.get("implicit_lattice_pose_pair_count_range")
        != [int(np.min(lattice_pair_count)), int(np.max(lattice_pair_count))]
    ):
        raise ValueError("factorized branch-union metadata counts differ")
    return arrays, metadata
