"""Build paired sparse mode-relation samples from exact 2DGS round trips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.child_eligibility import ChildGeometryEligibility
from feature_extract.vfm.localization_goal_maplet.child_local_likelihood import predict_child_local_surface_likelihood
from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import child_local_mode_runtime_features
from feature_extract.vfm.localization_goal_maplet.child_retrieval import retrieve_children_given_parents
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.mode_relation import (
    EDGE_FAMILIES,
    FEATURE_NAMES,
    RELATION_NULL_TYPES,
    analytic_relation_score,
    build_sparse_relation_edges,
    mode_relation_runtime_features,
)
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
NEGATIVE_SOURCES = (
    "frozen_graph_top1", "top32_near_miss_pose", "top32_low_rotation_phase_pose",
    "wrong_left_endpoint", "wrong_right_endpoint", "wrong_both_endpoints",
)


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0, model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]), height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _append(
    store: dict[str, list],
    *,
    feature: np.ndarray,
    valid: np.ndarray,
    null_type: np.ndarray,
    source: str,
    image_id: str,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_family: np.ndarray,
    edge_role: np.ndarray,
    candidate_index: int,
    translation_m: float,
    rotation_deg: float,
) -> None:
    keep = np.asarray(valid, dtype=bool)
    if not np.any(keep):
        return
    count = int(np.sum(keep))
    store["features"].append(np.asarray(feature, dtype=np.float32)[keep])
    store["targets"].append(np.full((count,), int(source == "exact_gt_configuration"), dtype=np.int64))
    store["image_ids"].extend([image_id] * count)
    store["trajectories"].extend([image_id.split("/", 1)[0]] * count)
    store["source_types"].extend([source] * count)
    store["edge_left_group_rows"].append(edge_left[keep].astype(np.int64))
    store["edge_right_group_rows"].append(edge_right[keep].astype(np.int64))
    store["edge_families"].append(edge_family[keep].astype(np.int64))
    store["edge_roles"].append(edge_role[keep].astype(np.int64))
    store["candidate_indices"].append(np.full((count,), candidate_index, dtype=np.int64))
    store["pose_translation_m"].append(np.full((count,), translation_m, dtype=np.float32))
    store["pose_rotation_deg"].append(np.full((count,), rotation_deg, dtype=np.float32))
    store["relation_null_types"].append(np.asarray(null_type, dtype=np.int64)[keep])


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
    parser.add_argument("--runtime_maximum_children", type=int, default=16)
    parser.add_argument("--maximum_groups_per_query", type=int, default=64)
    parser.add_argument("--maximum_modes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite mode-relation samples")
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
    store: dict[str, list] = {key: [] for key in (
        "features", "targets", "image_ids", "trajectories", "source_types",
        "edge_left_group_rows", "edge_right_group_rows", "edge_families", "edge_roles",
        "candidate_indices", "pose_translation_m", "pose_rotation_deg", "relation_null_types",
    )}
    analytic_positive, analytic_negative, analytic_source, analytic_family = [], [], [], []
    processed_queries = 0
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
            null_similarity_center=float(calibration.center), null_similarity_scale=float(calibration.scale),
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
        oracle = token_oracle_evidence(
            labels, physical, token_xy,
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_height=int(camera.height), image_width=int(camera.width), camera=camera,
        )
        truth_child, truth_xyz, group_rows = [], [], []
        for group in range(grouped.member_offsets.size - 1):
            members = grouped.member_token_indices[
                int(grouped.member_offsets[group]) : int(grouped.member_offsets[group + 1])
            ]
            members = members[oracle.child_rows[members] >= 0]
            if members.size == 0:
                continue
            mass: dict[int, float] = {}
            for token in members.tolist():
                child = int(oracle.child_rows[token])
                mass[child] = mass.get(child, 0.0) + float(oracle.child_mass[token])
            child = min(mass, key=lambda value: (-mass[value], value))
            selected = members[oracle.child_rows[members] == child]
            weight = oracle.child_mass[selected]
            if float(np.sum(weight)) <= 0.0:
                continue
            runtime = child_posterior.candidate_child_rows[group, : int(args.runtime_maximum_children)]
            if not np.any(runtime == child) or not eligibility.proposal_qualified[child]:
                continue
            truth_child.append(child)
            truth_xyz.append(np.average(oracle.child_local_xyz[selected], axis=0, weights=weight))
            group_rows.append(group)
        if len(group_rows) < 2:
            continue
        groups = np.asarray(group_rows, dtype=np.int64)
        priority = 1.0 - group_parent_null[groups]
        order = np.argsort(-priority, kind="stable")[: int(args.maximum_groups_per_query)]
        groups = groups[order]
        children = np.asarray(truth_child, dtype=np.int64)[order]
        truth = np.asarray(truth_xyz, dtype=np.float64)[order]
        descriptor = grouped_local[groups]
        xy = grouped.xy[groups] * np.asarray([camera.width, camera.height], dtype=np.float64)
        extent = grouped.extent[groups] * np.asarray([camera.width, camera.height], dtype=np.float64)
        scale = np.maximum(0.5 * np.linalg.norm(extent, axis=1), 8.0)
        likelihood = predict_child_local_surface_likelihood(
            descriptor, children, physical, field,
            temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
        )
        mode_feature, mode_valid = child_local_mode_runtime_features(
            likelihood, children, xy, scale, labels.pose_w2c, camera, physical, field,
        )
        primitive = np.asarray(likelihood.mode_primitive_rows, dtype=np.int64)
        point = physical.primitive_centers[np.maximum(primitive, 0)]
        error = np.linalg.norm(point - truth[:, None], axis=2)
        error[~mode_valid] = np.inf
        mode = np.argmin(error, axis=1)
        resolved = (
            np.min(error, axis=1) <= 0.20
        ) & (np.asarray(likelihood.feature_coverage) >= 0.75)
        if np.sum(resolved) < 2:
            continue
        groups, children, descriptor, xy, extent, scale, mode, primitive = (
            value[resolved] for value in (groups, children, descriptor, xy, extent, scale, mode, primitive)
        )
        truth_primitive = primitive[np.arange(mode.size), mode]
        edges = build_sparse_relation_edges(xy, extent, descriptor, scale)
        edge_left = np.concatenate([edges.fit_left, edges.verify_left])
        edge_right = np.concatenate([edges.fit_right, edges.verify_right])
        edge_family = np.concatenate([edges.fit_family, edges.verify_family])
        edge_role = np.concatenate([
            np.zeros(edges.fit_left.size, dtype=np.int64),
            np.ones(edges.verify_left.size, dtype=np.int64),
        ])
        if edge_left.size == 0:
            continue
        positive_feature, positive_valid, positive_null = mode_relation_runtime_features(
            xy, scale, edge_left, edge_right, edge_family,
            truth_primitive[edge_left], truth_primitive[edge_right],
            children[edge_left], children[edge_right], labels.pose_w2c, camera, physical,
        )
        _append(
            store, feature=positive_feature, valid=positive_valid, null_type=positive_null,
            source="exact_gt_configuration", image_id=image_id,
            edge_left=groups[edge_left], edge_right=groups[edge_right], edge_family=edge_family,
            edge_role=edge_role, candidate_index=-1, translation_m=0.0, rotation_deg=0.0,
        )
        positive_score = analytic_relation_score(positive_feature, positive_valid)
        row = pool_rows[image_id]
        details = row["mode_details"][MODE]
        diagnostics = row["ranking_diagnostics"][MODE]
        proposal_index = int(np.argmax(np.asarray(diagnostics["proposal_scores"], dtype=np.float64)))
        negative_pose: list[tuple[str, int]] = []
        if float(details[proposal_index]["translation_m"]) > 0.5 or float(details[proposal_index]["rotation_deg"]) > 5.0:
            negative_pose.append(("frozen_graph_top1", proposal_index))
        bad = [
            (float(item["translation_m"]) / 0.5 + float(item["rotation_deg"]) / 5.0, index)
            for index, item in enumerate(details)
            if float(item["translation_m"]) > 0.5 or float(item["rotation_deg"]) > 5.0
        ]
        bad_index = min(bad)[1] if bad else -1
        if bad_index >= 0 and bad_index != proposal_index:
            negative_pose.append(("top32_near_miss_pose", bad_index))
        phase = [
            (float(item["rotation_deg"]), float(item["translation_m"]), index)
            for index, item in enumerate(details)
            if index not in (proposal_index, bad_index)
            and 0.5 < float(item["translation_m"]) <= 3.0 and float(item["rotation_deg"]) <= 5.0
        ]
        if phase:
            negative_pose.append(("top32_low_rotation_phase_pose", min(phase)[2]))
        for source, candidate in negative_pose:
            detail = details[candidate]
            feature, valid, null_type = mode_relation_runtime_features(
                xy, scale, edge_left, edge_right, edge_family,
                truth_primitive[edge_left], truth_primitive[edge_right],
                children[edge_left], children[edge_right], np.asarray(detail["pose_w2c"]), camera, physical,
            )
            _append(
                store, feature=feature, valid=valid, null_type=null_type, source=source,
                image_id=image_id, edge_left=groups[edge_left], edge_right=groups[edge_right],
                edge_family=edge_family, edge_role=edge_role, candidate_index=candidate,
                translation_m=float(detail["translation_m"]), rotation_deg=float(detail["rotation_deg"]),
            )
            paired = positive_valid & valid
            if np.any(paired):
                analytic_positive.extend(positive_score[paired].tolist())
                analytic_negative.extend(analytic_relation_score(feature, valid)[paired].tolist())
                analytic_source.extend([source] * int(np.sum(paired)))
                analytic_family.extend(edge_family[paired].tolist())

        wrong_child = np.full(children.shape, -1, dtype=np.int64)
        for index, (group, truth_value) in enumerate(zip(groups.tolist(), children.tolist())):
            rows = child_posterior.candidate_child_rows[group, : int(args.runtime_maximum_children)]
            probability = child_posterior.candidate_probabilities[group, : int(args.runtime_maximum_children)]
            valid = (rows >= 0) & (rows != truth_value) & eligibility.proposal_qualified[np.maximum(rows, 0)]
            if np.any(valid):
                slots = np.flatnonzero(valid)
                wrong_child[index] = int(rows[int(slots[np.argmax(probability[slots])])])
        wrong_keep = wrong_child >= 0
        wrong_primitive = np.full(children.shape, -1, dtype=np.int64)
        if np.any(wrong_keep):
            wrong_likelihood = predict_child_local_surface_likelihood(
                descriptor[wrong_keep], wrong_child[wrong_keep], physical, field,
                temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
            )
            wrong_feature, wrong_valid = child_local_mode_runtime_features(
                wrong_likelihood, wrong_child[wrong_keep], xy[wrong_keep], scale[wrong_keep],
                labels.pose_w2c, camera, physical, field,
            )
            score = np.log(np.maximum(wrong_likelihood.mode_probabilities, 1e-12)) + np.clip(
                np.asarray(wrong_feature, dtype=np.float64)[:, :, 5], -12.0, 0.0,
            )
            score[~wrong_valid] = -np.inf
            best = np.argmax(score, axis=1)
            value = wrong_likelihood.mode_primitive_rows[np.arange(best.size), best]
            value[~np.any(wrong_valid, axis=1)] = -1
            wrong_primitive[wrong_keep] = value
        endpoint_specs = (
            ("wrong_left_endpoint", wrong_primitive[edge_left], truth_primitive[edge_right], wrong_child[edge_left], children[edge_right]),
            ("wrong_right_endpoint", truth_primitive[edge_left], wrong_primitive[edge_right], children[edge_left], wrong_child[edge_right]),
            ("wrong_both_endpoints", wrong_primitive[edge_left], wrong_primitive[edge_right], wrong_child[edge_left], wrong_child[edge_right]),
        )
        for source, left_primitive, right_primitive, left_child, right_child in endpoint_specs:
            feature, valid, null_type = mode_relation_runtime_features(
                xy, scale, edge_left, edge_right, edge_family,
                left_primitive, right_primitive, left_child, right_child,
                labels.pose_w2c, camera, physical,
            )
            _append(
                store, feature=feature, valid=valid, null_type=null_type, source=source,
                image_id=image_id, edge_left=groups[edge_left], edge_right=groups[edge_right],
                edge_family=edge_family, edge_role=edge_role, candidate_index=-1,
                translation_m=0.0, rotation_deg=0.0,
            )
            paired = positive_valid & valid
            if np.any(paired):
                analytic_positive.extend(positive_score[paired].tolist())
                analytic_negative.extend(analytic_relation_score(feature, valid)[paired].tolist())
                analytic_source.extend([source] * int(np.sum(paired)))
                analytic_family.extend(edge_family[paired].tolist())
        processed_queries += 1
        print(json.dumps({
            "image_id": image_id, "resolved_groups": int(groups.size),
            "fit_edges": int(edges.fit_left.size), "verify_edges": int(edges.verify_left.size),
        }), flush=True)
    if not store["features"]:
        raise ValueError("no mode-relation samples were generated")
    feature = np.concatenate(store["features"], axis=0)
    target = np.concatenate(store["targets"], axis=0)
    source = np.asarray(store["source_types"])
    analytic_positive_array = np.asarray(analytic_positive, dtype=np.float64)
    analytic_negative_array = np.asarray(analytic_negative, dtype=np.float64)
    analytic_source_array = np.asarray(analytic_source)
    analytic_family_array = np.asarray(analytic_family, dtype=np.int64)
    analytic_report = {
        "pair_count": int(analytic_positive_array.size),
        "concordance": float(np.mean(analytic_positive_array > analytic_negative_array)),
        "by_source": {
            value: float(np.mean(
                analytic_positive_array[analytic_source_array == value]
                > analytic_negative_array[analytic_source_array == value]
            )) for value in sorted(set(analytic_source_array.tolist()))
        },
        "by_edge_family": {
            EDGE_FAMILIES[value]: float(np.mean(
                analytic_positive_array[analytic_family_array == value]
                > analytic_negative_array[analytic_family_array == value]
            )) for value in sorted(set(analytic_family_array.tolist()))
        },
    }
    metadata = {
        "artifact_type": "goal_maplet_mode_relation_samples_v1",
        "feature_names": list(FEATURE_NAMES), "edge_families": list(EDGE_FAMILIES),
        "relation_null_types": list(RELATION_NULL_TYPES),
        "pairing_contract": "same_image_same_query_edge_fixed_options_v1",
        "edge_contract": "query_only_complete_link_fit_tree_disjoint_verify_v2",
        "option_contract": "fixed_vfm_topm_runtime_topc_v1",
        "runtime_maximum_children": int(args.runtime_maximum_children),
        "maximum_modes": int(args.maximum_modes), "temperature": float(args.temperature),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "field_feature_contract_sha256": contract.content_sha256,
        "child_eligibility_sha256": eligibility.content_sha256,
        "candidate_pool_sha256": file_sha256(pool_path),
        "processed_queries": int(processed_queries), "analytic_report": analytic_report,
        "uses_gt_only_for_training_target": True, "deployment_artifact": False,
        "stores_mapping_rgb": False, "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": True, "stored_downstream_embedding_count": 0,
    }
    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in store.items()
        if key not in ("image_ids", "trajectories", "source_types")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, **arrays, image_ids=np.asarray(store["image_ids"]),
        trajectories=np.asarray(store["trajectories"]), source_types=source,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        "output": str(output), "sample_count": int(feature.shape[0]),
        "positive_count": int(np.sum(target == 1)), "negative_count": int(np.sum(target == 0)),
        "analytic_report": analytic_report,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
