"""Build exact round-trip typed-null samples for child-local pose factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.child_eligibility import ChildGeometryEligibility
from feature_extract.vfm.localization_goal_maplet.child_local_factor import (
    NULL_TYPES,
    child_local_factor_runtime_features,
)
from feature_extract.vfm.localization_goal_maplet.child_local_likelihood import predict_child_local_surface_likelihood
from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import child_local_mode_runtime_features
from feature_extract.vfm.localization_goal_maplet.child_retrieval import retrieve_children_given_parents
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.oracle_pose import token_oracle_evidence
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import ValidityCalibration, retrieve_maplet_posterior
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


MODE = "actual_parent_actual_child"


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _factor_features(
    descriptors: np.ndarray,
    children: np.ndarray,
    xy: np.ndarray,
    scale: np.ndarray,
    pose: np.ndarray,
    camera: ColmapCamera,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    parent_probability: np.ndarray,
    child_probability: np.ndarray,
    parent_null: np.ndarray,
    *,
    temperature: float,
    maximum_modes: int,
    likelihood=None,
):
    if likelihood is None:
        likelihood = predict_child_local_surface_likelihood(
            descriptors, children, physical, field,
            temperature=float(temperature), maximum_modes=int(maximum_modes),
        )
    mode_features, mode_valid = child_local_mode_runtime_features(
        likelihood, children, xy, scale, pose, camera, physical, field
    )
    factor = child_local_factor_runtime_features(
        likelihood, mode_features, mode_valid, children, physical,
        parent_probability=parent_probability,
        child_probability=child_probability,
        parent_null_probability=parent_null,
        query_scale_px=scale,
        image_diagonal_px=float(np.hypot(camera.width, camera.height)),
    )
    return factor, likelihood, mode_valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--child_eligibility", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--maximum_modes", type=int, default=8)
    parser.add_argument("--maximum_groups_per_query", type=int, default=128)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite child-local factor samples")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    eligibility = ChildGeometryEligibility.load_npz(Path(args.child_eligibility))
    if eligibility.physical_map_sha256 != physical.content_sha256 or eligibility.canonical_field_sha256 != field.content_sha256:
        raise ValueError("child eligibility lineage differs")
    pool_path = Path(args.candidate_pool)
    pool = json.loads(pool_path.read_text())
    for key, expected in (
        ("physical_map_sha256", physical.content_sha256),
        ("canonical_field_sha256", field.content_sha256),
        ("field_feature_contract_sha256", contract.content_sha256),
    ):
        if pool.get(key) != expected:
            raise ValueError(f"candidate pool lineage differs: {key}")
    pool_rows = {str(row["image_id"]): row for row in pool["rows"]}
    readout = readout_canonical_field(field, physical)
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    sample_feature, sample_target, sample_image, sample_trajectory = [], [], [], []
    sample_source = []
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata["image_id"])
        if image_id not in pool_rows:
            continue
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        context = encode_radio_final_regions(mapped, token_xy, context_config)
        local = encode_radio_final_regions(mapped, token_xy, local_config)
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
            context, readout.parent_descriptors, physical.maplet_ids,
            readout.parent_coverage > 0.0, maximum_candidates=64, temperature=0.07,
            null_similarity_center=float(calibration.center),
            null_similarity_scale=float(calibration.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy, context, parent_ids[:, 0],
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_width=int(camera.width), image_height=int(camera.height),
            descriptor_half_size_tokens=2.0, minimum_descriptor_cosine=0.96,
        )
        group_parent_ids, group_parent_probability, group_parent_null = aggregate_group_posteriors(
            parent_ids, parent_probability, parent_null,
            grouped.member_offsets, grouped.member_token_indices, maximum_candidates=64,
        )
        grouped_local = aggregate_group_descriptors(local, grouped.member_offsets, grouped.member_token_indices)
        child_posterior = retrieve_children_given_parents(
            grouped_local, group_parent_ids, group_parent_probability, group_parent_null,
            readout.child_descriptors, readout.child_coverage, physical,
            maximum_child_candidates=128, temperature=0.07,
        )
        evidence = token_oracle_evidence(
            labels, physical, token_xy,
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_height=int(camera.height), image_width=int(camera.width), camera=camera,
        )
        child_rows, truth_xyz, truth_xy, group_rows = [], [], [], []
        for group in range(grouped.member_offsets.size - 1):
            members = grouped.member_token_indices[
                int(grouped.member_offsets[group]) : int(grouped.member_offsets[group + 1])
            ]
            members = members[evidence.child_rows[members] >= 0]
            if members.size == 0:
                continue
            mass: dict[int, float] = {}
            for token in members.tolist():
                child = int(evidence.child_rows[token])
                mass[child] = mass.get(child, 0.0) + float(evidence.child_mass[token])
            child = min(mass, key=lambda value: (-mass[value], value))
            selected = members[evidence.child_rows[members] == child]
            weight = evidence.child_mass[selected]
            if float(np.sum(weight)) <= 0.0:
                continue
            child_rows.append(child)
            truth_xyz.append(np.average(evidence.child_local_xyz[selected], axis=0, weights=weight))
            truth_xy.append(np.average(evidence.xy_px[selected], axis=0, weights=weight))
            group_rows.append(group)
        if not child_rows:
            continue
        children = np.asarray(child_rows, dtype=np.int64)
        truth = np.asarray(truth_xyz, dtype=np.float64)
        xy = np.asarray(truth_xy, dtype=np.float64)
        groups = np.asarray(group_rows, dtype=np.int64)
        if groups.size > int(args.maximum_groups_per_query):
            selected = np.linspace(
                0, groups.size - 1, int(args.maximum_groups_per_query), dtype=np.int64
            )
            children, truth, xy, groups = (
                value[selected] for value in (children, truth, xy, groups)
            )
        descriptor = grouped_local[groups]
        scale = np.maximum(
            0.5 * np.linalg.norm(
                grouped.extent[groups] * np.asarray([camera.width, camera.height]), axis=1
            ),
            8.0,
        )
        parent_rows = physical.child_parent_rows[children]
        parent_maplet_ids = physical.maplet_ids[parent_rows]
        p_parent = np.asarray([
            np.sum(group_parent_probability[group][group_parent_ids[group] == parent_id])
            for group, parent_id in zip(groups.tolist(), parent_maplet_ids.tolist())
        ], dtype=np.float64)
        p_child = np.asarray([
            np.sum(child_posterior.candidate_probabilities[group][child_posterior.candidate_child_rows[group] == child])
            for group, child in zip(groups.tolist(), children.tolist())
        ], dtype=np.float64)
        p_null = group_parent_null[groups]

        # Exact GT-pose factors teach local validity without a self-generated
        # pose.  A correct child can still be unresolved or field-missing.
        gt_feature, gt_likelihood, gt_valid = _factor_features(
            descriptor, children, xy, scale, labels.pose_w2c, camera, physical, field,
            p_parent, p_child, p_null,
            temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
        )
        gt_points = physical.primitive_centers[np.maximum(gt_likelihood.mode_primitive_rows, 0)]
        gt_error = np.linalg.norm(gt_points - truth[:, None], axis=2)
        topm_error = np.min(np.where(gt_valid, gt_error, np.inf), axis=1)
        gt_target = np.full(children.shape, NULL_TYPES.index("valid"), dtype=np.int64)
        missing = (gt_likelihood.feature_coverage < 0.75) | ~eligibility.retrieval_qualified[children]
        gt_target[missing] = NULL_TYPES.index("field_missing")
        gt_target[~missing & (topm_error > 0.20)] = NULL_TYPES.index("unresolved")

        # The frozen graph Top-1 is an independent runtime pose.  A bad mode
        # must be rejected instead of using its reprojection to validate itself.
        pool_row = pool_rows[image_id]
        diagnostics = pool_row["ranking_diagnostics"][MODE]
        proposal_index = int(np.argmax(np.asarray(diagnostics["proposal_scores"], dtype=np.float64)))
        detail = pool_row["mode_details"][MODE][proposal_index]
        proposal_pose = np.asarray(detail["pose_w2c"], dtype=np.float64)
        proposal_feature, _, _ = _factor_features(
            descriptor, children, xy, scale, proposal_pose, camera, physical, field,
            p_parent, p_child, p_null,
            temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
            likelihood=gt_likelihood,
        )
        proposal_bad = bool(float(detail["translation_m"]) > 0.5 or float(detail["rotation_deg"]) > 5.0)
        proposal_target = gt_target.copy()
        if proposal_bad:
            proposal_target[proposal_target != NULL_TYPES.index("field_missing")] = NULL_TYPES.index("pose_incompatible")

        # Every query also contributes its closest out-of-basin Top-32 mode.
        # This is a hard pose perturbation with the same child and fixed VFM
        # modes, including queries whose graph Top-1 happens to be correct.
        details = pool_row["mode_details"][MODE]
        bad_candidates = [
            (item["translation_m"] / 0.5 + item["rotation_deg"] / 5.0, index)
            for index, item in enumerate(details)
            if item["translation_m"] > 0.5 or item["rotation_deg"] > 5.0
        ]
        near_miss_feature = np.zeros((0, gt_feature.shape[1]), dtype=np.float32)
        near_miss_target = np.zeros((0,), dtype=np.int64)
        if bad_candidates:
            _, bad_index = min(bad_candidates)
            bad_pose = np.asarray(details[bad_index]["pose_w2c"], dtype=np.float64)
            near_miss_feature, _, _ = _factor_features(
                descriptor, children, xy, scale, bad_pose, camera, physical, field,
                p_parent, p_child, p_null,
                temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
                likelihood=gt_likelihood,
            )
            near_miss_target = np.full(
                children.shape, NULL_TYPES.index("pose_incompatible"), dtype=np.int64
            )
            near_miss_target[gt_target == NULL_TYPES.index("field_missing")] = NULL_TYPES.index("field_missing")

        # Hard wrong children come from the actual retrieval posterior.  This
        # is the missing negative branch in v1, not a random easy negative.
        wrong = np.full(children.shape, -1, dtype=np.int64)
        wrong_probability = np.zeros(children.shape, dtype=np.float64)
        for index, (group, truth_child) in enumerate(zip(groups.tolist(), children.tolist())):
            rows = child_posterior.candidate_child_rows[group]
            probabilities = child_posterior.candidate_probabilities[group]
            valid = (rows >= 0) & (rows != truth_child)
            if np.any(valid):
                slots = np.flatnonzero(valid)
                slot = int(slots[np.argmax(probabilities[slots])])
                wrong[index], wrong_probability[index] = int(rows[slot]), float(probabilities[slot])
            else:
                siblings = np.flatnonzero(
                    (physical.child_parent_rows == physical.child_parent_rows[truth_child])
                    & (np.arange(physical.child_parent_rows.size) != truth_child)
                    & eligibility.retrieval_qualified
                )
                if siblings.size:
                    wrong[index] = int(siblings[0])
        keep_wrong = wrong >= 0
        if np.any(keep_wrong):
            wrong_parent_rows = physical.child_parent_rows[wrong[keep_wrong]]
            wrong_parent_ids = physical.maplet_ids[wrong_parent_rows]
            wrong_parent_probability = np.asarray([
                np.sum(group_parent_probability[group][group_parent_ids[group] == parent_id])
                for group, parent_id in zip(groups[keep_wrong].tolist(), wrong_parent_ids.tolist())
            ], dtype=np.float64)
            wrong_feature, _, _ = _factor_features(
                descriptor[keep_wrong], wrong[keep_wrong], xy[keep_wrong], scale[keep_wrong],
                labels.pose_w2c, camera, physical, field,
                wrong_parent_probability, wrong_probability[keep_wrong], p_null[keep_wrong],
                temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
            )
        else:
            wrong_feature = np.zeros((0, gt_feature.shape[1]), dtype=np.float32)

        feature_parts = [gt_feature, proposal_feature, near_miss_feature, wrong_feature]
        target_parts = [
            gt_target,
            proposal_target,
            near_miss_target,
            np.full((wrong_feature.shape[0],), NULL_TYPES.index("wrong_child"), dtype=np.int64),
        ]
        source_parts = [
            np.full(children.shape, "exact_gt_pose", dtype="U32"),
            np.full(children.shape, "frozen_graph_top1", dtype="U32"),
            np.full((near_miss_feature.shape[0],), "top32_near_miss_pose", dtype="U32"),
            np.full((wrong_feature.shape[0],), "retrieved_hard_wrong_child", dtype="U32"),
        ]
        for feature_part, target_part, source_part in zip(feature_parts, target_parts, source_parts):
            sample_feature.append(feature_part)
            sample_target.append(target_part)
            sample_source.append(source_part)
            sample_image.extend([image_id] * feature_part.shape[0])
            sample_trajectory.extend([image_id.split("/", 1)[0]] * feature_part.shape[0])
        print(json.dumps({
            "image_id": image_id, "group_count": int(children.size),
            "sample_count": int(sum(item.shape[0] for item in feature_parts)),
            "proposal_bad": proposal_bad,
        }), flush=True)
    if not sample_feature:
        raise ValueError("no child-local factor samples were generated")
    feature = np.concatenate(sample_feature, axis=0).astype(np.float32)
    target = np.concatenate(sample_target, axis=0).astype(np.int64)
    source = np.concatenate(sample_source, axis=0)
    metadata = {
        "artifact_type": "goal_maplet_child_local_factor_samples_v2",
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "field_feature_contract_sha256": contract.content_sha256,
        "child_eligibility_sha256": eligibility.content_sha256,
        "candidate_pool_sha256": file_sha256(pool_path),
        "null_types": list(NULL_TYPES),
        "temperature": float(args.temperature), "maximum_modes": int(args.maximum_modes),
        "maximum_groups_per_query": int(args.maximum_groups_per_query),
        "uses_gt_only_for_training_target": True, "deployment_artifact": False,
        "stores_mapping_rgb": False, "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": True, "stored_downstream_embedding_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, features=feature, targets=target,
        image_ids=np.asarray(sample_image), trajectories=np.asarray(sample_trajectory),
        source_types=source, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        "output": str(output), "sample_count": int(feature.shape[0]),
        "class_count": {NULL_TYPES[index]: int(np.sum(target == index)) for index in range(len(NULL_TYPES))},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
