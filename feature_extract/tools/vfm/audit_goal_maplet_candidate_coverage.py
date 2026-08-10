"""Autopsy Goal-Maplet Stage-A/B candidate coverage on frozen held folds.

The report separates physical parent/child recall, query-support geometry,
configuration inference, region-to-surface geometry, raw proposal survival and
post-NMS retention.  It never trains a score and never uses phase to generate
a pose.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_oracle_ladder import _solve
from feature_extract.tools.vfm.evaluate_goal_maplet_pose_modes import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.child_retrieval import (
    retrieve_children_given_parents,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import (
    FieldFeatureContract,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.oracle_pose import (
    token_oracle_evidence,
)
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    contributor_multiscale_child_distribution,
    contributor_multiscale_maplet_distribution,
)
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    ValidityCalibration,
    retrieve_maplet_posterior,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
)


MODE = "actual_parent_actual_child"


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _pool_by_image(pool: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["image_id"]): row for row in pool.get("rows", ())}


def _candidate_coverage(
    rows: dict[str, dict[str, object]], image_id: str, mode: str = MODE,
) -> dict[str, object]:
    details = list(rows.get(image_id, {}).get("mode_details", {}).get(mode, ()))
    translation = np.asarray(
        [float(item["translation_m"]) for item in details], dtype=np.float64
    )
    rotation = np.asarray(
        [float(item["rotation_deg"]) for item in details], dtype=np.float64
    )
    if not len(details):
        return {
            "candidate_count": 0,
            "strict_available": False,
            "one_m_available": False,
            "best_translation_m": None,
            "best_rotation_deg": None,
        }
    joint = translation / 0.5 + rotation / 5.0
    best = int(np.argmin(joint))
    return {
        "candidate_count": len(details),
        "strict_available": bool(
            np.any((translation <= 0.5) & (rotation <= 5.0))
        ),
        "one_m_available": bool(
            np.any((translation <= 1.0) & (rotation <= 10.0))
        ),
        "best_translation_m": float(translation[best]),
        "best_rotation_deg": float(rotation[best]),
    }


def _pose_success(value: dict[str, object], *, strict: bool) -> bool:
    if value.get("translation_m") is None or value.get("rotation_deg") is None:
        return False
    return bool(
        float(value["translation_m"]) <= (0.5 if strict else 1.0)
        and float(value["rotation_deg"]) <= (5.0 if strict else 10.0)
    )


def _project(
    xyz: np.ndarray, pose_w2c: np.ndarray, camera,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation, _ = cv2.Rodrigues(np.asarray(pose_w2c[:3, :3], dtype=np.float64))
    xy, _ = cv2.projectPoints(
        points, rotation, np.asarray(pose_w2c[:3, 3], dtype=np.float64),
        matrix, distortion,
    )
    camera_xyz = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    return xy.reshape(-1, 2), camera_xyz[:, 2]


def _grid_graph_diameter(xy: np.ndarray) -> int:
    coordinates = [tuple(int(item) for item in value) for value in xy.tolist()]
    if len(coordinates) <= 1:
        return 0
    coordinate_set = set(coordinates)
    maximum = 0
    for source in coordinates:
        distance = {source: 0}
        queue = deque([source])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in coordinate_set and neighbor not in distance:
                    distance[neighbor] = distance[(x, y)] + 1
                    queue.append(neighbor)
        maximum = max(maximum, max(distance.values()))
    return int(maximum)


def _conditioning(
    xy_px: np.ndarray,
    child_rows: np.ndarray,
    physical: GoalMapletPhysicalMap,
    pose_w2c: np.ndarray,
    camera,
) -> dict[str, object]:
    rows = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    valid = rows >= 0
    rows = rows[valid]
    xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)[valid]
    if rows.size == 0:
        return {
            "support_count": 0,
            "distinct_parent_count": 0,
            "distinct_child_count": 0,
            "spatial_bin_count": 0,
            "surface_normal_rank": 0,
            "depth_std_m": None,
            "xy_condition": None,
            "xyz_condition": None,
            "structurally_pose_sufficient": False,
        }
    xyz = physical.child_centers[rows]
    parent = physical.child_parent_rows[rows]
    normalized = xy / np.asarray([camera.width, camera.height], dtype=np.float64)
    bins = np.clip(np.floor(3.0 * normalized).astype(np.int64), 0, 2)
    xy_singular = np.linalg.svd(xy - np.mean(xy, axis=0), compute_uv=False)
    xyz_singular = np.linalg.svd(xyz - np.mean(xyz, axis=0), compute_uv=False)
    normal_singular = np.linalg.svd(
        physical.child_normals[rows] - np.mean(physical.child_normals[rows], axis=0),
        compute_uv=False,
    )
    _, depth = _project(xyz, pose_w2c, camera)
    unique_child = int(np.unique(rows).size)
    spatial_bins = int(np.unique(3 * bins[:, 1] + bins[:, 0]).size)
    xy_rank = int(np.sum(xy_singular > 1e-5))
    xyz_rank = int(np.sum(xyz_singular > 1e-5))
    return {
        "support_count": int(rows.size),
        "distinct_parent_count": int(np.unique(parent).size),
        "distinct_child_count": unique_child,
        "spatial_bin_count": spatial_bins,
        "surface_normal_rank": int(np.sum(normal_singular > 1e-5)),
        "depth_std_m": float(np.std(depth)),
        "xy_condition": (
            float(xy_singular[0] / max(xy_singular[1], 1e-8))
            if xy_singular.size >= 2 else None
        ),
        "xyz_condition": (
            float(xyz_singular[0] / max(xyz_singular[1], 1e-8))
            if xyz_singular.size >= 2 else None
        ),
        "structurally_pose_sufficient": bool(
            rows.size >= 6
            and unique_child >= 6
            and spatial_bins >= 4
            and xy_rank >= 2
            and xyz_rank >= 2
        ),
    }


def _group_geometry(
    grouped,
    token_xy: np.ndarray,
    evidence,
    truth_parent: np.ndarray,
    truth_parent_null: np.ndarray,
    truth_child: np.ndarray,
    truth_child_null: np.ndarray,
    parent_ids: np.ndarray,
    parent_probability: np.ndarray,
    child_posterior,
    physical: GoalMapletPhysicalMap,
    camera,
    gt_pose: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    group_xy_px = grouped.xy * np.asarray([camera.width, camera.height])
    group_extent_px = grouped.extent * np.asarray([camera.width, camera.height])
    parent_row = np.argmax(truth_parent, axis=1)
    child_row = np.argmax(truth_child, axis=1)
    valid_parent_truth = (1.0 - truth_parent_null) > 1e-6
    valid_child_truth = (1.0 - truth_child_null) > 1e-6
    truth_parent_id = physical.maplet_ids[parent_row]
    parent_supported = np.asarray([
        bool(valid_parent_truth[index])
        and bool(np.any(
            (parent_ids[index] == truth_parent_id[index])
            & (parent_probability[index] > 0.0)
        ))
        for index in range(group_xy_px.shape[0])
    ])
    child_supported = np.asarray([
        bool(valid_child_truth[index])
        and bool(np.any(
            (child_posterior.candidate_child_rows[index] == child_row[index])
            & (child_posterior.candidate_probabilities[index] > 0.0)
        ))
        for index in range(group_xy_px.shape[0])
    ])
    parent_runtime_rank = np.full((group_xy_px.shape[0],), -1, dtype=np.int64)
    child_runtime_rank = np.full((group_xy_px.shape[0],), -1, dtype=np.int64)
    for index in range(group_xy_px.shape[0]):
        parent_slot = np.flatnonzero(
            (parent_ids[index] == truth_parent_id[index])
            & (parent_probability[index] > 0.0)
        )
        if parent_slot.size:
            parent_runtime_rank[index] = int(parent_slot[0]) + 1
        child_slot = np.flatnonzero(
            (child_posterior.candidate_child_rows[index] == child_row[index])
            & (child_posterior.candidate_probabilities[index] > 0.0)
        )
        if child_slot.size:
            child_runtime_rank[index] = int(child_slot[0]) + 1
    conditioned_best = np.zeros_like(child_supported)
    exact_xyz = np.zeros((group_xy_px.shape[0], 3), dtype=np.float64)
    exact_xy = np.zeros((group_xy_px.shape[0], 2), dtype=np.float64)
    exact_valid = np.zeros((group_xy_px.shape[0],), dtype=bool)
    rows = []
    for group in range(group_xy_px.shape[0]):
        start, end = int(grouped.member_offsets[group]), int(grouped.member_offsets[group + 1])
        members = grouped.member_token_indices[start:end]
        parent_slots = np.flatnonzero(
            (parent_ids[group] == truth_parent_id[group])
            & (parent_probability[group] > 0.0)
        )
        if (
            parent_slots.size
            and child_posterior.best_child_rows_by_parent is not None
            and valid_child_truth[group]
        ):
            conditioned_best[group] = bool(
                child_posterior.best_child_rows_by_parent[
                    group, int(parent_slots[0])
                ] == child_row[group]
            )
        exact_members = members[
            (evidence.child_rows[members] == child_row[group])
            & (evidence.child_mass[members] > 0.0)
        ]
        if exact_members.size:
            weight = evidence.child_mass[exact_members]
            exact_xyz[group] = np.average(
                evidence.child_local_xyz[exact_members], axis=0, weights=weight
            )
            exact_xy[group] = np.average(
                evidence.xy_px[exact_members], axis=0, weights=weight
            )
            exact_valid[group] = True
        projected_child, child_depth = _project(
            physical.child_centers[child_row[group]][None], gt_pose, camera
        )
        if exact_valid[group]:
            projected_exact, exact_depth = _project(
                exact_xyz[group][None], gt_pose, camera
            )
            exact_residual = float(
                np.linalg.norm(projected_exact[0] - group_xy_px[group])
            )
            exact_weighted_residual = float(
                np.linalg.norm(projected_exact[0] - exact_xy[group])
            )
            exact_depth_value = float(exact_depth[0])
        else:
            exact_residual = None
            exact_weighted_residual = None
            exact_depth_value = None
        rows.append({
            "group_index": group,
            "token_count": int(members.size),
            "bbox_diameter_px": float(2.0 * np.linalg.norm(group_extent_px[group])),
            "connected_chain_diameter_tokens": _grid_graph_diameter(token_xy[members]),
            "truth_parent_row": int(parent_row[group]),
            "truth_parent_id": int(truth_parent_id[group]),
            "truth_parent_purity": float(
                np.max(truth_parent[group])
                / max(float(np.sum(truth_parent[group])), 1e-12)
            ),
            "truth_child_row": int(child_row[group]),
            "truth_child_purity": float(
                np.max(truth_child[group])
                / max(float(np.sum(truth_child[group])), 1e-12)
            ),
            "truth_parent_in_runtime_topk": bool(parent_supported[group]),
            "truth_parent_runtime_rank": (
                int(parent_runtime_rank[group]) if parent_runtime_rank[group] > 0 else None
            ),
            "truth_child_in_runtime_topk": bool(child_supported[group]),
            "truth_child_runtime_rank": (
                int(child_runtime_rank[group]) if child_runtime_rank[group] > 0 else None
            ),
            "truth_child_is_conditioned_best": bool(conditioned_best[group]),
            "bbox_center_to_truth_child_projection_px": float(
                np.linalg.norm(projected_child[0] - group_xy_px[group])
            ),
            "truth_child_depth_m": float(child_depth[0]),
            "bbox_center_to_exact_surface_projection_px": exact_residual,
            "exact_surface_weighted_projection_residual_px": exact_weighted_residual,
            "exact_surface_depth_m": exact_depth_value,
        })
    return rows, {
        "group_xy_px": group_xy_px,
        "truth_parent_rows": parent_row,
        "truth_child_rows": child_row,
        "parent_supported": parent_supported,
        "child_supported": child_supported,
        "conditioned_best": conditioned_best,
        "exact_xyz": exact_xyz,
        "exact_xy": exact_xy,
        "exact_valid": exact_valid,
        "valid_child_truth": valid_child_truth,
    }


def _classify(row: dict[str, object]) -> str:
    baseline = row["candidate_ladder"]["baseline_post_nms"]
    updated = row["candidate_ladder"]["conditioned_child_post_nms"]
    raw = row["candidate_ladder"]["conditioned_child_pre_nms"]
    oracle_parent = row["candidate_ladder"]["oracle_parent_actual_child"]
    oracle_child = row["candidate_ladder"]["oracle_parent_oracle_child"]
    oracle = row["pose_oracles"]
    recall = row["pose_sufficient_recall"]
    if baseline["one_m_available"]:
        return "covered_baseline"
    if updated["one_m_available"]:
        return "fixed_conditioned_child_enumeration"
    if raw["one_m_available"]:
        return "B3_proposal_retention"
    if not recall["parent_structurally_sufficient"]:
        return "A1_parent_recall_insufficient"
    if not recall["child_structurally_sufficient"]:
        return "A2_child_recall_insufficient"
    # If fixing only the parent configuration recovers the pose while actual
    # local child retrieval and the same solver remain in place, the loss is
    # the pre-pose hard parent assignment, not child recall or geometry.
    if oracle_parent["one_m_available"]:
        return "B1_configuration_inference"
    if oracle_child["one_m_available"]:
        return "A2_child_retrieval_or_conditioning"
    if _pose_success(oracle["posterior_supported_child_center"], strict=False):
        return "B1_configuration_inference"
    if (
        not _pose_success(oracle["truth_child_center"], strict=False)
        and _pose_success(oracle["truth_exact_surface_bbox_center"], strict=False)
    ):
        return "B2_child_center_geometry"
    if (
        not _pose_success(oracle["truth_exact_surface_bbox_center"], strict=False)
        and _pose_success(oracle["truth_exact_surface_weighted_xy"], strict=False)
    ):
        return "A3_support_center_geometry"
    return "B2_or_map_geometry_unresolved"


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    classifications = Counter(_classify(row) for row in rows)
    result: dict[str, object] = {
        "query_count": len(rows),
        "classification_counts": dict(sorted(classifications.items())),
    }
    for name in (
        "baseline_post_nms", "conditioned_child_post_nms",
        "conditioned_child_pre_nms", "oracle_parent_actual_child",
        "oracle_parent_oracle_child",
    ):
        values = [row["candidate_ladder"][name] for row in rows]
        result[name] = {
            "strict_coverage": float(np.mean([
                bool(value["strict_available"]) for value in values
            ])),
            "one_m_coverage": float(np.mean([
                bool(value["one_m_available"]) for value in values
            ])),
        }
    result["pose_sufficient_physical_recall"] = {
        "parent_structural_fraction": float(np.mean([
            bool(row["pose_sufficient_recall"]["parent_structurally_sufficient"])
            for row in rows
        ])),
        "child_structural_fraction": float(np.mean([
            bool(row["pose_sufficient_recall"]["child_structurally_sufficient"])
            for row in rows
        ])),
        "posterior_supported_oracle_1m_fraction": float(np.mean([
            _pose_success(
                row["pose_oracles"]["posterior_supported_child_center"],
                strict=False,
            )
            for row in rows
        ])),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--crossfit_evaluation", required=True)
    parser.add_argument("--baseline_candidate_pool", required=True)
    parser.add_argument("--conditioned_candidate_pool", required=True)
    parser.add_argument("--pre_nms_candidate_pool", required=True)
    parser.add_argument("--oracle_candidate_pool", required=True)
    parser.add_argument("--include_trajectories", nargs="+", required=True)
    parser.add_argument("--image_id", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--parent_candidates", type=int, default=64)
    parser.add_argument("--child_candidates", type=int, default=64)
    parser.add_argument("--grouping_cosine", type=float, default=0.96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite candidate-coverage autopsy")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    mapper, _ = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device)
    )
    instance, instance_metadata = load_physical_instance_readout(
        Path(args.physical_instance_readout), device=str(args.device)
    )
    if instance_metadata.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("physical readout and map differ")
    if instance_metadata.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("physical readout and field differ")
    readout = readout_canonical_field(field, physical)
    parent_descriptor = instance.project_numpy(
        readout.parent_descriptors, role="context", device=str(args.device)
    )
    child_descriptor = instance.project_numpy(
        readout.child_descriptors, role="local", device=str(args.device)
    )
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    pool_paths = {
        "baseline": Path(args.baseline_candidate_pool),
        "conditioned": Path(args.conditioned_candidate_pool),
        "pre_nms": Path(args.pre_nms_candidate_pool),
        "oracle": Path(args.oracle_candidate_pool),
    }
    pools = {name: json.loads(path.read_text()) for name, path in pool_paths.items()}
    for name, pool in pools.items():
        if pool.get("physical_map_sha256") != physical.content_sha256:
            raise ValueError(f"{name} candidate pool and physical map differ")
        if pool.get("canonical_field_sha256") != field.content_sha256:
            raise ValueError(f"{name} candidate pool and canonical field differ")
    if not bool(pools["conditioned"].get("parent_conditioned_child_enumeration")):
        raise ValueError("conditioned candidate pool predates the enumeration fix")
    if not (
        float(pools["pre_nms"].get("translation_nms_m", -1.0)) == 0.0
        and float(pools["pre_nms"].get("rotation_nms_deg", -1.0)) == 0.0
    ):
        raise ValueError("pre-NMS pool did not disable pose-space NMS")
    pool_rows = {name: _pool_by_image(pool) for name, pool in pools.items()}
    held = {str(value) for value in args.include_trajectories}
    crossfit_path = Path(args.crossfit_evaluation)
    crossfit = json.loads(crossfit_path.read_text())
    if crossfit.get("stage") != (
        "goal_maplet_phase_operator_outer_feature_crossfit_evaluation"
    ) or not bool(crossfit.get("feature_pipeline_outer_crossfit")):
        raise ValueError("candidate autopsy lacks an audited feature-cross-fit fold")
    if {str(value) for value in crossfit.get("heldout_trajectories", ())} != held:
        raise ValueError("candidate autopsy and cross-fit held trajectories differ")
    crossfit_contract = dict(crossfit.get("candidate_generator_contract") or {})
    if crossfit_contract.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("cross-fit evaluation and autopsy physical maps differ")
    if crossfit_contract.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("cross-fit evaluation and autopsy canonical fields differ")
    if crossfit.get("artifacts", {}).get("surface_mapper_sha256") != file_sha256(
        Path(args.surface_mapper)
    ):
        raise ValueError("cross-fit evaluation and autopsy surface mappers differ")
    if crossfit.get("artifacts", {}).get(
        "physical_instance_readout_sha256"
    ) != file_sha256(Path(args.physical_instance_readout)):
        raise ValueError("cross-fit evaluation and autopsy physical readouts differ")
    contributor_paths = []
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if (
            _trajectory(str(metadata["image_id"])) in held
            and (not args.image_id or str(metadata["image_id"]) == str(args.image_id))
        ):
            contributor_paths.append(path)
    rows = []
    for path in contributor_paths:
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata["image_id"])
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        context_descriptor = encode_physical_instance_regions(
            instance, mapped, token_xy, role="context", device=str(args.device)
        )
        local_descriptor = encode_physical_instance_regions(
            instance, mapped, token_xy, role="local", device=str(args.device)
        )
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
            context_descriptor,
            parent_descriptor,
            physical.maplet_ids,
            readout.parent_coverage > 0.0,
            maximum_candidates=int(args.parent_candidates),
            temperature=0.07,
            null_similarity_center=float(calibration.center),
            null_similarity_scale=float(calibration.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy,
            context_descriptor,
            parent_ids[:, 0],
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            image_width=int(camera.width),
            image_height=int(camera.height),
            descriptor_half_size_tokens=2.0,
            minimum_descriptor_cosine=float(args.grouping_cosine),
        )
        grouped_parent_ids, grouped_parent_probability, grouped_parent_null = (
            aggregate_group_posteriors(
                parent_ids,
                parent_probability,
                parent_null,
                grouped.member_offsets,
                grouped.member_token_indices,
                maximum_candidates=int(args.parent_candidates),
            )
        )
        grouped_local = aggregate_group_descriptors(
            local_descriptor,
            grouped.member_offsets,
            grouped.member_token_indices,
        )
        child = retrieve_children_given_parents(
            grouped_local,
            grouped_parent_ids,
            grouped_parent_probability,
            grouped_parent_null,
            child_descriptor,
            readout.child_coverage,
            physical,
            maximum_child_candidates=int(args.child_candidates),
            temperature=0.07,
        )
        truth_parent, truth_parent_null = contributor_multiscale_maplet_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            pool_sizes=(1,),
            pool_weights=(1.0,),
            group_member_offsets=grouped.member_offsets,
            group_member_token_indices=grouped.member_token_indices,
        )
        truth_child, truth_child_null = contributor_multiscale_child_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            group_member_offsets=grouped.member_offsets,
            group_member_token_indices=grouped.member_token_indices,
        )
        evidence = token_oracle_evidence(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            image_height=int(camera.height),
            image_width=int(camera.width),
            camera=camera,
        )
        group_rows, arrays = _group_geometry(
            grouped,
            token_xy,
            evidence,
            truth_parent,
            truth_parent_null,
            truth_child,
            truth_child_null,
            grouped_parent_ids,
            grouped_parent_probability,
            child,
            physical,
            camera,
            labels.pose_w2c,
        )
        group_xy_px = arrays["group_xy_px"]
        valid_truth = arrays["valid_child_truth"]
        supported_parent = valid_truth & arrays["parent_supported"]
        supported_child = supported_parent & arrays["child_supported"]
        truth_child_rows = arrays["truth_child_rows"]
        exact_valid = arrays["exact_valid"]
        pose_oracles = {
            "posterior_supported_child_center": _solve(
                group_xy_px[supported_child],
                physical.child_centers[truth_child_rows[supported_child]],
                camera,
                labels.pose_w2c,
            ),
            "truth_child_center": _solve(
                group_xy_px[valid_truth],
                physical.child_centers[truth_child_rows[valid_truth]],
                camera,
                labels.pose_w2c,
            ),
            "truth_exact_surface_bbox_center": _solve(
                group_xy_px[exact_valid],
                arrays["exact_xyz"][exact_valid],
                camera,
                labels.pose_w2c,
            ),
            "truth_exact_surface_weighted_xy": _solve(
                arrays["exact_xy"][exact_valid],
                arrays["exact_xyz"][exact_valid],
                camera,
                labels.pose_w2c,
            ),
        }
        parent_condition = _conditioning(
            group_xy_px[arrays["parent_supported"] & valid_truth],
            truth_child_rows[arrays["parent_supported"] & valid_truth],
            physical,
            labels.pose_w2c,
            camera,
        )
        child_condition = _conditioning(
            group_xy_px[supported_child],
            truth_child_rows[supported_child],
            physical,
            labels.pose_w2c,
            camera,
        )
        ladder = {
            "baseline_post_nms": _candidate_coverage(
                pool_rows["baseline"], image_id
            ),
            "conditioned_child_post_nms": _candidate_coverage(
                pool_rows["conditioned"], image_id
            ),
            "conditioned_child_pre_nms": _candidate_coverage(
                pool_rows["pre_nms"], image_id
            ),
            "oracle_parent_actual_child": _candidate_coverage(
                pool_rows["oracle"], image_id, "oracle_parent_actual_child"
            ),
            "oracle_parent_oracle_child": _candidate_coverage(
                pool_rows["oracle"], image_id, "oracle_child"
            ),
        }
        row = {
            "image_id": image_id,
            "trajectory_id": _trajectory(image_id),
            "support_count": len(group_rows),
            "candidate_ladder": ladder,
            "pose_oracles": pose_oracles,
            "pose_sufficient_recall": {
                "parent_structurally_sufficient": bool(
                    parent_condition["structurally_pose_sufficient"]
                ),
                "child_structurally_sufficient": bool(
                    child_condition["structurally_pose_sufficient"]
                ),
                "parent_conditioning": parent_condition,
                "child_conditioning": child_condition,
                "parent_supported_group_count": int(
                    np.sum(arrays["parent_supported"] & valid_truth)
                ),
                "child_supported_group_count": int(np.sum(supported_child)),
                "conditioned_best_child_group_count": int(
                    np.sum(arrays["conditioned_best"] & supported_child)
                ),
            },
            "groups": group_rows,
        }
        row["classification"] = _classify(row)
        rows.append(row)
        print(json.dumps({
            "image_id": image_id,
            "classification": row["classification"],
            "candidate_ladder": ladder,
            "pose_sufficient_recall": row["pose_sufficient_recall"],
        }), flush=True)
    result = {
        "stage": "goal_maplet_candidate_coverage_autopsy_g20_3",
        "query_count": len(rows),
        "heldout_trajectories": sorted(held),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_sha256": file_sha256(Path(args.surface_mapper)),
        "physical_instance_readout_sha256": file_sha256(
            Path(args.physical_instance_readout)
        ),
        "feature_pipeline_outer_crossfit": True,
        "geometry_was_outer_crossfit": False,
        "fixed_candidate_budget": 32,
        "phase_used_for_pose_generation": False,
        "pnp_role": "diagnostic_initializer_or_oracle_only",
        "pool_inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in pool_paths.items()
        },
        "crossfit_evaluation": {
            "path": str(crossfit_path),
            "sha256": file_sha256(crossfit_path),
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": result["query_count"],
        "summary": result["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
