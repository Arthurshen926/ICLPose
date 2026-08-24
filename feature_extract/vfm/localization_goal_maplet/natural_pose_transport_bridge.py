"""Strict score-before-label bridge for natural pose-free pose candidates.

The pose-free pool is the deployable input.  A score artifact is built from
that pool without opening query pose or error labels.  Only this module's
post-hoc binding step is allowed to combine the frozen scores with a direct
candidate dataset, whose candidate zero is a diagnostic GT anchor and is
therefore never copied into the score artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fulltoken_pose_ranking import greedy_distinct_pose_basin_order
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256
from .pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)
from .retrieval_surface_metrics import COORDINATE_CONTRACT


POSE_FREE_POOL_SCHEMA = "goal_maplet_pose_free_visibility_candidate_pool_v1"
NATURAL_SCORE_SCHEMA = "goal_maplet_natural_sparse_pose_transport_scores_v1"
NATURAL_EVALUATION_SCHEMA = "goal_maplet_natural_sparse_pose_transport_evaluation_v1"

JOINT_BASINS = {
    "region_2m_45deg": (2.0, 45.0),
    "loose_1m_10deg": (1.0, 10.0),
    "strict_0_5m_5deg": (0.5, 5.0),
}

_POOL_FORBIDDEN_TRUE = (
    "uses_alike",
    "uses_pnp",
    "uses_point_correspondences",
    "uses_query_ground_truth",
    "uses_query_pose",
)


def camera_intrinsics_content_sha256(
    image_id: str,
    camera_model_id: int,
    camera_width: int,
    camera_height: int,
    camera_params: np.ndarray | Sequence[float],
) -> str:
    """Hash the pose-free camera payload and nothing else in a contributor.

    In particular, callers must not hash the contributor file itself: that
    NPZ also stores ``pose_w2c`` and hashing its bytes would make a nominally
    score-before-label phase read GT-dependent content.
    """

    params = np.asarray(camera_params, dtype=np.float64).reshape(-1)
    if (
        not str(image_id)
        or int(camera_width) <= 0
        or int(camera_height) <= 0
        or params.size == 0
        or np.any(~np.isfinite(params))
    ):
        raise ValueError("pose-free camera intrinsics are invalid")
    return arrays_sha256({
        "image_id": np.asarray(str(image_id)),
        "camera_model_id": np.asarray(int(camera_model_id), dtype=np.int32),
        "camera_width": np.asarray(int(camera_width), dtype=np.int32),
        "camera_height": np.asarray(int(camera_height), dtype=np.int32),
        "camera_params": params,
    })


def load_pose_free_camera_binding(
    path: Path, *, image_id: str,
) -> tuple[int, int, int, tuple[float, ...], str]:
    """Read only whitelisted intrinsics members from a contributor NPZ."""

    with np.load(Path(path), allow_pickle=False) as data:
        model_id = int(data["camera_model_id"])
        width = int(data["camera_width"])
        height = int(data["camera_height"])
        params = tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist())
    binding = camera_intrinsics_content_sha256(
        image_id, model_id, width, height, params,
    )
    return model_id, width, height, params, binding


def load_pose_free_candidate_pool(path: Path) -> dict[str, object]:
    """Load and hash-validate a pool that has no query-pose dependency."""

    pool_path = Path(path)
    pool = json.loads(pool_path.read_text())
    content = str(pool.get("content_sha256", ""))
    unhashed = dict(pool)
    unhashed.pop("content_sha256", None)
    if pool.get("artifact_type") != POSE_FREE_POOL_SCHEMA:
        raise ValueError("not a pose-free visibility candidate pool")
    if content != canonical_json_sha256(unhashed):
        raise ValueError("pose-free candidate pool content hash differs")
    if any(pool.get(flag) is not False for flag in _POOL_FORBIDDEN_TRUE):
        raise ValueError("pose-free candidate pool violates the method boundary")
    if pool.get("scores_are_pose_free_and_not_consumed_by_transport_builder") is not True:
        raise ValueError("candidate pool score-consumption contract differs")
    if pool.get("candidate_prefix_stable_across_budgets") is not True:
        raise ValueError("natural candidate pool is not prefix stable")
    route_audit = pool.get("route_disjoint_atlas_audit")
    if (
        not isinstance(route_audit, dict)
        or route_audit.get("route_allowlist_enforced") is not True
        or route_audit.get("query_route_excluded_from_atlas") is not True
        or route_audit.get("coordinate_correct") is not True
        or route_audit.get("coordinate_contract") != COORDINATE_CONTRACT
        or str(pool.get("query_route", ""))
        in {str(value) for value in route_audit.get("allowed_trajectories", ())}
    ):
        raise ValueError("natural candidate pool atlas is not route/coordinate strict")
    maximum = int(pool.get("maximum_modes", 0))
    rows = pool.get("rows")
    if maximum <= 0 or not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("pose-free candidate pool is empty")
    image_ids: list[str] = []
    for row in rows:
        image_id = str(row.get("image_id", ""))
        details_by_mode = row.get("mode_details")
        details = (
            details_by_mode.get("actual_parent_actual_child")
            if isinstance(details_by_mode, dict) else None
        )
        if (
            not image_id
            or not isinstance(details, list)
            or len(details) != maximum
            or not str(row.get("retrieval_artifact", ""))
            or len(str(row.get("retrieval_content_sha256", ""))) != 64
        ):
            raise ValueError("pose-free candidate row is incomplete")
        poses = []
        for expected_rank, detail in enumerate(details, start=1):
            pose = np.asarray(detail.get("pose_w2c"), dtype=np.float64)
            if int(detail.get("rank", -1)) != expected_rank or pose.shape != (4, 4):
                raise ValueError("pose-free candidate rank/matrix differs")
            if np.any(~np.isfinite(pose)) or not np.allclose(
                pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1.0e-10, rtol=0.0
            ):
                raise ValueError("pose-free candidate matrix is invalid")
            poses.append(pose)
        keys = {tuple(value.round(10).reshape(-1).tolist()) for value in poses}
        if len(keys) != len(poses):
            raise ValueError("pose-free candidate row contains duplicate poses")
        image_ids.append(image_id)
    if image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("pose-free candidate queries are not unique sorted IDs")
    if int(pool.get("query_count", -1)) != len(rows):
        raise ValueError("pose-free candidate query count differs")
    return pool


def pose_free_pool_arrays(
    pool: Mapping[str, object],
    *,
    maximum_candidates: int | None = None,
    query_start: int = 0,
    maximum_queries: int = 0,
) -> dict[str, np.ndarray]:
    """Extract only deployable candidate inputs; scores are intentionally ignored."""

    rows = list(pool["rows"])
    start = int(query_start)
    stop = len(rows) if int(maximum_queries) == 0 else min(
        len(rows), start + int(maximum_queries)
    )
    if start < 0 or start >= len(rows) or stop <= start:
        raise ValueError("natural score query slice is empty")
    candidate_count = int(pool["maximum_modes"])
    if maximum_candidates is not None:
        candidate_count = min(candidate_count, int(maximum_candidates))
    if candidate_count <= 0:
        raise ValueError("natural candidate count must be positive")
    selected = rows[start:stop]
    poses = np.stack([
        np.stack([
            np.asarray(detail["pose_w2c"], dtype=np.float64)
            for detail in row["mode_details"]["actual_parent_actual_child"][:candidate_count]
        ])
        for row in selected
    ])
    return {
        "image_ids": np.asarray([str(row["image_id"]) for row in selected]),
        "candidate_poses_w2c": poses,
        "candidate_valid": np.ones(poses.shape[:2], dtype=bool),
        "retrieval_paths": np.asarray([
            str(Path(str(row["retrieval_artifact"])).resolve()) for row in selected
        ]),
        "retrieval_content_sha256": np.asarray([
            str(row["retrieval_content_sha256"]) for row in selected
        ]),
    }


def load_natural_score_artifact(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("natural score artifact lacks metadata")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {
            name: np.asarray(data[name]) for name in data.files if name != "metadata_json"
        }
    if metadata.get("artifact_type") != NATURAL_SCORE_SCHEMA:
        raise ValueError("not a natural sparse pose-transport score artifact")
    if metadata.get("content_sha256") != arrays_sha256(arrays):
        raise ValueError("natural score artifact content hash differs")
    required_false = (
        "uses_alike", "uses_pnp", "uses_point_correspondences",
        "uses_absolute_pose_regression", "uses_query_pose", "uses_query_ground_truth",
    )
    if any(metadata.get(key) is not False for key in required_false):
        raise ValueError("natural score artifact violates the method boundary")
    if (
        metadata.get("candidate_zero_is_present") is not False
        or metadata.get("pose_error_labels_opened_during_scoring") is not False
        or metadata.get("candidate_pool_scores_consumed") is not False
        or metadata.get("contributor_file_bytes_hashed_during_scoring") is not False
        or metadata.get("contributor_pose_member_opened_during_scoring") is not False
        or metadata.get("rendered_training_dataset_opened_during_scoring") is not False
        or metadata.get("gt_derived_model_report_opened_during_scoring") is not False
    ):
        raise ValueError("natural score artifact is not score-before-label")
    required = (
        "image_ids", "candidate_poses_w2c", "candidate_valid", "scores",
        "component_statistics", "retrieval_content_sha256", "radio_file_sha256",
        "camera_intrinsics_content_sha256", "target_mean_rendered_mass",
        "target_feature_valid_mass_fraction", "target_visible_token_fraction",
        "source_retained_mass_fraction",
    )
    if any(name not in arrays for name in required):
        raise ValueError("natural score artifact lacks replay arrays")
    q, c = np.asarray(arrays["candidate_valid"]).shape
    if (
        np.asarray(arrays["image_ids"]).shape != (q,)
        or np.asarray(arrays["candidate_poses_w2c"]).shape != (q, c, 4, 4)
        or np.asarray(arrays["scores"]).shape != (q, c)
        or np.asarray(arrays["component_statistics"]).shape != (q, c, 6)
        or any(np.asarray(arrays[name]).shape != (q, c) for name in (
            "target_mean_rendered_mass", "target_feature_valid_mass_fraction",
            "target_visible_token_fraction",
        ))
        or any(np.asarray(arrays[name]).shape != (q,) for name in (
            "retrieval_content_sha256", "radio_file_sha256",
            "camera_intrinsics_content_sha256", "source_retained_mass_fraction",
        ))
    ):
        raise ValueError("natural score artifact array shapes differ")
    if any(np.any(~np.isfinite(np.asarray(arrays[name]))) for name in (
        "candidate_poses_w2c", "scores", "component_statistics",
        "target_mean_rendered_mass", "target_feature_valid_mass_fraction",
        "target_visible_token_fraction", "source_retained_mass_fraction",
    )):
        raise ValueError("natural score artifact contains nonfinite values")
    if metadata.get("candidate_pose_arrays_sha256") != arrays_sha256({
        "candidate_poses_w2c": np.asarray(arrays["candidate_poses_w2c"]),
        "candidate_valid": np.asarray(arrays["candidate_valid"]),
    }):
        raise ValueError("natural score candidate-pose hash differs")
    return arrays, metadata


def bind_scores_to_direct_labels(
    score_arrays: Mapping[str, np.ndarray],
    score_metadata: Mapping[str, object],
    direct_dataset_path: Path,
    candidate_pool_path: Path,
) -> dict[str, np.ndarray]:
    """Open post-freeze labels and prove exact non-anchor candidate identity."""

    pool_path = Path(candidate_pool_path).resolve()
    pool = load_pose_free_candidate_pool(pool_path)
    if (
        score_metadata.get("candidate_pool_content_sha256") != pool["content_sha256"]
        or score_metadata.get("candidate_pool_file_sha256") != file_sha256(pool_path)
    ):
        raise ValueError("natural score artifact and pose-free pool differ")
    direct_arrays, direct_metadata = load_pose_candidate_dataset(
        Path(direct_dataset_path), require_rendered_targets=False,
    )
    if direct_metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA:
        raise ValueError("natural score labels are not a direct candidate dataset")
    if (
        direct_metadata.get("candidate_pool_content_sha256") != pool["content_sha256"]
        or direct_metadata.get("candidate_pool_file_sha256") != file_sha256(pool_path)
        or direct_metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or direct_metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True
        or direct_metadata.get("pose_errors_computed_only_after_candidate_freeze") is not True
        or direct_metadata.get(
            "nonanchor_candidates_preserve_pose_free_pool_exact_order"
        ) is not True
        or direct_metadata.get(
            "gt_anchor_does_not_change_nonanchor_candidate_membership"
        ) is not True
        or direct_metadata.get(
            "pose_free_pool_internal_duplicates_rejected_before_gt_join"
        ) is not True
    ):
        raise ValueError("direct labels are not bound to the frozen pose-free pool")
    # First bind the score payload back to the original pose-free pool.  This
    # is the authoritative candidate identity: an older direct-label builder
    # de-duplicated pool candidates against the subsequently opened GT anchor,
    # which can shift the prefix and is therefore not safe as a K64 join key.
    expected_pool = pose_free_pool_arrays(
        pool, maximum_candidates=int(np.asarray(score_arrays["candidate_valid"]).shape[1])
    )
    pool_ids = [str(value) for value in np.asarray(expected_pool["image_ids"]).tolist()]
    pool_lookup = {image_id: row for row, image_id in enumerate(pool_ids)}
    score_ids = [str(value) for value in np.asarray(score_arrays["image_ids"]).tolist()]
    if len(set(score_ids)) != len(score_ids) or any(value not in pool_lookup for value in score_ids):
        raise ValueError("natural score queries are absent or duplicated in pose-free pool")
    pool_rows = np.asarray([pool_lookup[value] for value in score_ids], dtype=np.int64)
    if not np.array_equal(
        np.asarray(expected_pool["candidate_poses_w2c"])[pool_rows],
        np.asarray(score_arrays["candidate_poses_w2c"], dtype=np.float64),
    ) or not np.array_equal(
        np.asarray(expected_pool["candidate_valid"])[pool_rows],
        np.asarray(score_arrays["candidate_valid"], dtype=bool),
    ):
        raise ValueError("natural score poses differ from the original pose-free pool")

    direct_ids = [str(value) for value in np.asarray(direct_arrays["image_ids"]).tolist()]
    lookup = {image_id: row for row, image_id in enumerate(direct_ids)}
    if len(set(score_ids)) != len(score_ids) or any(value not in lookup for value in score_ids):
        raise ValueError("natural score queries are absent or duplicated in direct labels")
    rows = np.asarray([lookup[value] for value in score_ids], dtype=np.int64)
    candidate_count = int(np.asarray(score_arrays["candidate_valid"]).shape[1])
    stop = 1 + candidate_count
    if np.asarray(direct_arrays["candidate_valid"]).shape[1] < stop:
        raise ValueError("direct labels have fewer non-anchor candidates than scores")
    direct_pose = np.asarray(direct_arrays["candidate_poses_w2c"], dtype=np.float64)[rows, 1:stop]
    direct_valid = np.asarray(direct_arrays["candidate_valid"], dtype=bool)[rows, 1:stop]
    score_pose = np.asarray(score_arrays["candidate_poses_w2c"], dtype=np.float64)
    score_valid = np.asarray(score_arrays["candidate_valid"], dtype=bool)
    direct_alignment = np.all(direct_pose == score_pose, axis=(1, 2, 3)) & np.all(
        direct_valid == score_valid, axis=1
    )
    if not np.array_equal(
        np.asarray(direct_arrays["radio_file_sha256"])[rows],
        np.asarray(score_arrays["radio_file_sha256"]),
    ):
        raise ValueError("natural score RADIO lineage differs from direct labels")
    direct_camera_bindings: list[str] = []
    target_poses: list[np.ndarray] = []
    for direct_row in rows.tolist():
        contributor_path = Path(str(direct_arrays["contributor_paths"][direct_row]))
        if file_sha256(contributor_path) != str(
            direct_arrays["contributor_file_sha256"][direct_row]
        ):
            raise ValueError("direct-label contributor file hash differs")
        *_, binding = load_pose_free_camera_binding(
            contributor_path, image_id=direct_ids[direct_row],
        )
        direct_camera_bindings.append(binding)
        # This is deliberately Phase 2: scores and candidate identities are
        # already frozen, so opening GT can only label them, never change a
        # candidate or score.
        with np.load(contributor_path, allow_pickle=False) as data:
            target_pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        if target_pose.shape != (4, 4) or np.any(~np.isfinite(target_pose)):
            raise ValueError("direct-label contributor target pose is invalid")
        target_poses.append(target_pose)
    if not np.array_equal(
        np.asarray(direct_camera_bindings),
        np.asarray(score_arrays["camera_intrinsics_content_sha256"]),
    ):
        raise ValueError("natural score camera-intrinsics lineage differs from direct labels")
    target_pose_array = np.stack(target_poses)
    direct_anchor = np.asarray(direct_arrays["candidate_poses_w2c"], dtype=np.float64)[rows, 0]
    if not np.array_equal(direct_anchor, target_pose_array):
        raise ValueError("direct diagnostic anchor and Phase-2 contributor GT differ")
    if np.any(np.asarray(direct_arrays["translation_m"])[rows, 0] > 1.0e-5) or np.any(
        np.asarray(direct_arrays["rotation_deg"])[rows, 0] > 1.0e-4
    ):
        raise ValueError("direct diagnostic anchor error is nonzero")
    rotations = score_pose[:, :, :3, :3]
    translations = score_pose[:, :, :3, 3]
    centers = -np.swapaxes(rotations, 2, 3) @ translations[..., None]
    target_rotation = target_pose_array[:, :3, :3]
    target_center = -np.swapaxes(target_rotation, 1, 2) @ target_pose_array[:, :3, 3, None]
    translation_error = np.linalg.norm(
        centers[..., 0] - target_center[:, None, :, 0], axis=2
    )
    relative = rotations @ np.swapaxes(target_rotation[:, None], 2, 3)
    cosine = np.clip(
        (np.trace(relative, axis1=2, axis2=3) - 1.0) / 2.0, -1.0, 1.0
    )
    rotation_error = np.degrees(np.arccos(cosine))
    exact_target_pose_collision = np.all(
        score_pose == target_pose_array[:, None], axis=(2, 3)
    ) & score_valid
    return {
        "translation_m": translation_error,
        "rotation_deg": rotation_error,
        "direct_nonanchor_exact_pool_alignment": direct_alignment,
        "direct_nonanchor_exact_pool_alignment_query_count": np.asarray(
            int(np.sum(direct_alignment)), dtype=np.int64
        ),
        "direct_nonanchor_gt_conditioned_prefix_drift_query_count": np.asarray(
            int(np.sum(~direct_alignment)), dtype=np.int64
        ),
        "exact_target_pose_collision": exact_target_pose_collision,
        "exact_target_pose_collision_query_count": np.asarray(
            int(np.sum(np.any(exact_target_pose_collision, axis=1))), dtype=np.int64
        ),
        "exact_target_pose_collision_candidate_count": np.asarray(
            int(np.sum(exact_target_pose_collision)), dtype=np.int64
        ),
        "direct_dataset_content_sha256": np.asarray(str(direct_metadata["content_sha256"])),
        "direct_dataset_file_sha256": np.asarray(file_sha256(Path(direct_dataset_path))),
    }


def _rankdata(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    begin = 0
    while begin < order.size:
        end = begin + 1
        while end < order.size and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = 0.5 * float(begin + end - 1)
        begin = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = _rankdata(left), _rankdata(right)
    if a.size < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _pose_separation(poses: np.ndarray, first: int, second: int) -> tuple[float, float]:
    rotation = poses[:, :3, :3]
    center = -np.swapaxes(rotation, 1, 2) @ poses[:, :3, 3, None]
    translation = float(np.linalg.norm(center[first, :, 0] - center[second, :, 0]))
    cosine = float(np.clip((np.sum(rotation[first] * rotation[second]) - 1.0) / 2.0, -1.0, 1.0))
    return translation, float(np.degrees(np.arccos(cosine)))


def ranked_natural_candidate_metrics(
    scores: np.ndarray,
    poses_w2c: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    valid: np.ndarray,
    *,
    image_ids: Sequence[str] | None = None,
    ranks: tuple[int, ...] = (1, 4, 8, 16, 32, 64),
) -> dict[str, object]:
    """Report raw acquisition, q_pose retention, and physical ambiguity."""

    score = np.asarray(scores, dtype=np.float64)
    poses = np.asarray(poses_w2c, dtype=np.float64)
    translation = np.asarray(translation_m, dtype=np.float64)
    rotation = np.asarray(rotation_deg, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if (
        score.ndim != 2 or score.shape[0] == 0
        or poses.shape != score.shape + (4, 4)
        or translation.shape != score.shape
        or rotation.shape != score.shape
        or mask.shape != score.shape
        or np.any(~np.isfinite(score))
        or np.any(~np.isfinite(poses))
        or np.any(~np.isfinite(translation))
        or np.any(~np.isfinite(rotation))
        or np.any(np.sum(mask, axis=1) == 0)
    ):
        raise ValueError("natural ranked metric arrays differ")
    rank_values = tuple(sorted({min(int(value), score.shape[1]) for value in ranks}))
    if not rank_values or rank_values[0] <= 0:
        raise ValueError("natural ranked metric budgets must be positive")
    ids = list(image_ids) if image_ids is not None else [str(row) for row in range(score.shape[0])]
    if len(ids) != score.shape[0]:
        raise ValueError("natural ranked metric image IDs differ")
    raw_hits = {
        name: {value: [] for value in rank_values} for name in JOINT_BASINS
    }
    ranked_hits = {
        name: {value: [] for value in rank_values} for name in JOINT_BASINS
    }
    prefix_top1_hits = {
        name: {value: [] for value in rank_values} for name in JOINT_BASINS
    }
    rows: list[dict[str, object]] = []
    spearman_rows: list[float] = []
    margins: list[float] = []
    runner_translation: list[float] = []
    runner_rotation: list[float] = []
    distinct_counts: list[int] = []
    for query in range(score.shape[0]):
        order = greedy_distinct_pose_basin_order(
            score[query], poses[query], mask[query], excluded_candidate_index=None,
        )
        distinct_counts.append(int(order.size))
        first = int(order[0])
        second = int(order[1]) if order.size > 1 else None
        margin = float(score[query, first] - score[query, second]) if second is not None else None
        separation = _pose_separation(poses[query], first, second) if second is not None else (None, None)
        if margin is not None:
            margins.append(margin)
            runner_translation.append(float(separation[0]))
            runner_rotation.append(float(separation[1]))
        joint_region = np.maximum(translation[query] / 2.0, rotation[query] / 45.0)
        active = np.flatnonzero(mask[query])
        correlation = _spearman(score[query, active], -joint_region[active])
        if correlation is not None:
            spearman_rows.append(correlation)
        first_rank: dict[str, int | None] = {}
        for name, (translation_limit, rotation_limit) in JOINT_BASINS.items():
            qualified = mask[query] & (translation[query] <= translation_limit) & (
                rotation[query] <= rotation_limit
            )
            ranked_positions = np.flatnonzero(qualified[order])
            first_rank[name] = (
                int(ranked_positions[0] + 1) if ranked_positions.size else None
            )
            for value in rank_values:
                prefix = np.arange(min(value, score.shape[1]))
                prefix = prefix[mask[query, prefix]]
                raw_hits[name][value].append(bool(np.any(qualified[prefix])))
                ranked_hits[name][value].append(bool(np.any(qualified[order[:value]])))
                prefix_order = greedy_distinct_pose_basin_order(
                    score[query, :value], poses[query, :value], mask[query, :value],
                    excluded_candidate_index=None,
                )
                prefix_top1_hits[name][value].append(bool(
                    prefix_order.size and qualified[int(prefix_order[0])]
                ))
        rows.append({
            "image_id": str(ids[query]),
            "selected_candidate_index_zero_based_pose_free": first,
            "selected_score": float(score[query, first]),
            "selected_translation_m": float(translation[query, first]),
            "selected_rotation_deg": float(rotation[query, first]),
            "distinct_runner_up_index_zero_based_pose_free": second,
            "distinct_score_margin": margin,
            "runner_up_translation_separation_m": separation[0],
            "runner_up_rotation_separation_deg": separation[1],
            "distinct_basin_count": int(order.size),
            "score_region_error_spearman": correlation,
            "first_qualifying_rank": first_rank,
        })

    def rates(values: Mapping[str, Mapping[int, list[bool]]]) -> dict[str, object]:
        return {
            name: {
                f"at_{rank}": {
                    "hits": int(np.sum(rows_value)),
                    "query_count": len(rows_value),
                    "rate": float(np.mean(rows_value)),
                }
                for rank, rows_value in by_rank.items()
            }
            for name, by_rank in values.items()
        }

    first_rank_summary = {}
    for name in JOINT_BASINS:
        observed = [
            int(row["first_qualifying_rank"][name])
            for row in rows if row["first_qualifying_rank"][name] is not None
        ]
        first_rank_summary[name] = {
            "acquired_query_count": len(observed),
            "missing_query_count": len(rows) - len(observed),
            "conditional_median_rank": float(np.median(observed)) if observed else None,
            "conditional_p90_rank": float(np.percentile(observed, 90)) if observed else None,
        }
    return {
        "query_count": int(score.shape[0]),
        "candidate_count": int(score.shape[1]),
        "candidate_zero_diagnostic_gt_anchor_absent": True,
        "rank_values": list(rank_values),
        "raw_pose_free_prefix_acquisition_upper_bound": rates(raw_hits),
        "q_pose_ranked_full_pool_basin_survival": rates(ranked_hits),
        "q_pose_prefix_top1_selection": rates(prefix_top1_hits),
        "first_qualifying_distinct_basin_rank": first_rank_summary,
        "mean_score_region_error_spearman": (
            float(np.mean(spearman_rows)) if spearman_rows else None
        ),
        "ambiguity": {
            "physical_nms_semantics": (
                "stable_score_order_greedy_0.5m_5deg_nontransitive_evaluation_nms"
            ),
            "mean_distinct_basin_count": float(np.mean(distinct_counts)),
            "minimum_distinct_basin_count": int(np.min(distinct_counts)),
            "distinct_top1_top2_score_margin_median": (
                float(np.median(margins)) if margins else None
            ),
            "distinct_top1_top2_score_margin_p10": (
                float(np.percentile(margins, 10)) if margins else None
            ),
            "runner_up_translation_separation_median_m": (
                float(np.median(runner_translation)) if runner_translation else None
            ),
            "runner_up_rotation_separation_median_deg": (
                float(np.median(runner_rotation)) if runner_rotation else None
            ),
            "single_pose_acceptance_requires_external_disjoint_margin_calibration": True,
        },
        "rows": rows,
    }
