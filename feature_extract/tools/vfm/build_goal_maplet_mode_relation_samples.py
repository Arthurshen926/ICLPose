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
from feature_extract.vfm.localization_goal_maplet.child_retrieval import (
    ChildTilePosterior,
    retrieve_children_given_parents,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.endpoint_hierarchy import EndpointHierarchyCalibration
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.mode_relation import (
    EDGE_FAMILIES,
    FEATURE_NAMES,
    RELATION_NULL_TYPES,
    SparseRelationEdges,
    _aggregate_complete_link_observations,
    _aggregate_sparse_probability_rows,
    analytic_relation_score,
    build_sparse_relation_edges,
    mass_adaptive_endpoint_options,
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
    parser.add_argument("--endpoint_hierarchy_calibration")
    parser.add_argument("--child_eligibility", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--runtime_maximum_children", type=int, default=16)
    parser.add_argument("--maximum_groups_per_query", type=int, default=64)
    parser.add_argument("--maximum_modes", type=int, default=8)
    parser.add_argument("--endpoint_state_budget", type=int, default=16)
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
    hierarchy_calibration = (
        EndpointHierarchyCalibration.load_json(Path(args.endpoint_hierarchy_calibration))
        if args.endpoint_hierarchy_calibration else None
    )
    if hierarchy_calibration is not None:
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
            ("field_feature_contract_sha256", contract.content_sha256),
        ):
            if hierarchy_calibration.metadata.get(key) != expected:
                raise ValueError(f"endpoint hierarchy calibration lineage differs: {key}")
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
        parent_ids, parent_probability, parent_null, parent_best_similarity = retrieve_maplet_posterior(
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
        token_support_valid = calibration.predict_valid(parent_best_similarity)
        group_support_valid = np.asarray([
            np.mean(token_support_valid[
                grouped.member_token_indices[
                    int(grouped.member_offsets[group]) : int(grouped.member_offsets[group + 1])
                ]
            ])
            for group in range(grouped.member_offsets.size - 1)
        ], dtype=np.float64)
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
        priority = group_support_valid
        initial_groups = np.argsort(-priority, kind="stable")[: int(args.maximum_groups_per_query)]
        initial_groups = initial_groups[priority[initial_groups] > 0.02]
        if initial_groups.size < 2:
            continue
        descriptor = grouped_local[initial_groups]
        xy = grouped.xy[initial_groups] * np.asarray([camera.width, camera.height], dtype=np.float64)
        extent = grouped.extent[initial_groups] * np.asarray([camera.width, camera.height], dtype=np.float64)
        scale = np.maximum(0.5 * np.linalg.norm(extent, axis=1), 8.0)
        original_edges = build_sparse_relation_edges(
            xy, extent, descriptor, scale, query_priority=priority[initial_groups],
        )
        cluster = np.asarray(original_edges.support_cluster_rows, dtype=np.int64)
        representatives = np.asarray(original_edges.representative_groups, dtype=np.int64)
        if representatives.size < 2:
            continue
        active_parent_ids, active_parent_probability = _aggregate_sparse_probability_rows(
            group_parent_ids[initial_groups], group_parent_probability[initial_groups],
            cluster, representatives, maximum_candidates=group_parent_ids.shape[1],
        )
        active_child_rows, active_child_probability = _aggregate_sparse_probability_rows(
            child_posterior.candidate_child_rows[initial_groups],
            child_posterior.candidate_probabilities[initial_groups],
            cluster, representatives,
            maximum_candidates=child_posterior.candidate_child_rows.shape[1],
        )
        active_support_valid = np.asarray([
            np.mean(group_support_valid[initial_groups[cluster == cluster[row]]])
            for row in representatives.tolist()
        ], dtype=np.float64)
        cluster_member_groups = []
        for row_index in representatives.tolist():
            members = np.flatnonzero(cluster == cluster[row_index])
            cluster_member_groups.append(initial_groups[members])
        descriptor, xy, extent, scale = _aggregate_complete_link_observations(
            descriptor, xy, extent, cluster, representatives,
        )
        groups = initial_groups[representatives]
        remap = {int(source): target for target, source in enumerate(representatives.tolist())}
        edges = SparseRelationEdges(
            fit_left=np.asarray([remap[int(value)] for value in original_edges.fit_left], dtype=np.int64),
            fit_right=np.asarray([remap[int(value)] for value in original_edges.fit_right], dtype=np.int64),
            fit_family=original_edges.fit_family,
            verify_left=np.asarray([remap[int(value)] for value in original_edges.verify_left], dtype=np.int64),
            verify_right=np.asarray([remap[int(value)] for value in original_edges.verify_right], dtype=np.int64),
            verify_family=original_edges.verify_family,
            representative_groups=np.arange(representatives.size, dtype=np.int64),
            support_cluster_rows=np.arange(representatives.size, dtype=np.int64),
            legacy_connected_cluster_rows=np.arange(representatives.size, dtype=np.int64),
        )
        active_child_posterior = ChildTilePosterior(
            active_child_rows, active_child_probability,
            np.clip(1.0 - np.sum(active_child_probability, axis=1), 0.0, 1.0),
        )
        options = mass_adaptive_endpoint_options(
            descriptor, active_support_valid,
            active_parent_ids, active_parent_probability,
            active_child_posterior, physical, field, eligibility,
            state_budget=int(args.endpoint_state_budget), maximum_modes=int(args.maximum_modes),
            temperature=float(args.temperature),
            endpoint_hierarchy_calibration=hierarchy_calibration,
        )
        selected_child = options.factor_child_rows[options.selected_factor_rows]
        selected_primitive = np.asarray(options.likelihood.mode_primitive_rows, dtype=np.int64)[
            options.selected_factor_rows, options.selected_mode_rows,
        ]
        children = np.full((groups.size,), -1, dtype=np.int64)
        truth_primitive = np.full((groups.size,), -1, dtype=np.int64)
        resolved = np.zeros((groups.size,), dtype=bool)
        for active_group, source_groups in enumerate(cluster_member_groups):
            members = np.concatenate([
                grouped.member_token_indices[
                    int(grouped.member_offsets[source]) : int(grouped.member_offsets[source + 1])
                ]
                for source in source_groups.tolist()
            ])
            members = members[oracle.child_rows[members] >= 0]
            if members.size == 0:
                continue
            mass: dict[int, float] = {}
            for token in members.tolist():
                child = int(oracle.child_rows[token])
                mass[child] = mass.get(child, 0.0) + float(oracle.child_mass[token])
            child = min(mass, key=lambda value: (-mass[value], value))
            owned = members[oracle.child_rows[members] == child]
            weight = np.asarray(oracle.child_mass[owned], dtype=np.float64)
            if float(np.sum(weight)) <= 0.0:
                continue
            truth_xyz = np.average(oracle.child_local_xyz[owned], axis=0, weights=weight)
            start, end = int(physical.child_member_offsets[child]), int(physical.child_member_offsets[child + 1])
            primitive_members = physical.child_member_primitive_rows[start:end]
            if primitive_members.size == 0:
                continue
            primitive = int(primitive_members[np.argmin(
                np.linalg.norm(physical.primitive_centers[primitive_members] - truth_xyz, axis=1)
            )])
            state = (
                (options.factor_group_rows[options.selected_factor_rows] == active_group)
                & (selected_child == child) & (selected_primitive == primitive)
            )
            if np.any(state) and float(np.linalg.norm(physical.primitive_centers[primitive] - truth_xyz)) <= 0.20:
                children[active_group] = child
                truth_primitive[active_group] = primitive
                resolved[active_group] = True
        edge_left = np.concatenate([edges.fit_left, edges.verify_left])
        edge_right = np.concatenate([edges.fit_right, edges.verify_right])
        edge_family = np.concatenate([edges.fit_family, edges.verify_family])
        edge_role = np.concatenate([
            np.zeros(edges.fit_left.size, dtype=np.int64),
            np.ones(edges.verify_left.size, dtype=np.int64),
        ])
        supported_edge = resolved[edge_left] & resolved[edge_right]
        edge_left, edge_right, edge_family, edge_role = (
            value[supported_edge] for value in (edge_left, edge_right, edge_family, edge_role)
        )
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
        wrong_primitive = np.full(children.shape, -1, dtype=np.int64)
        selected_group = options.factor_group_rows[options.selected_factor_rows]
        for group in np.flatnonzero(resolved).tolist():
            state = np.flatnonzero(
                (selected_group == group)
                & (
                    (selected_child != children[group])
                    | (selected_primitive != truth_primitive[group])
                )
            )
            if state.size:
                best = int(state[np.argmax(options.selected_base_masses[state])])
                wrong_child[group] = int(selected_child[best])
                wrong_primitive[group] = int(selected_primitive[best])
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
            "image_id": image_id, "resolved_groups": int(np.sum(resolved)),
            "supported_edges": int(edge_left.size),
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
        "artifact_type": "goal_maplet_mode_relation_samples_v2",
        "feature_names": list(FEATURE_NAMES), "edge_families": list(EDGE_FAMILIES),
        "relation_null_types": list(RELATION_NULL_TYPES),
        "pairing_contract": "same_image_same_runtime_query_edge_adaptive_options_v2",
        "edge_contract": "query_only_complete_link_cluster_collapse_fit_verify_v3",
        "option_contract": "mass_adaptive_parent_child_mode_budget_v3",
        "runtime_maximum_children": int(args.runtime_maximum_children),
        "endpoint_state_budget": int(args.endpoint_state_budget),
        "endpoint_hierarchy_calibration_sha256": (
            hierarchy_calibration.content_sha256 if hierarchy_calibration is not None else None
        ),
        "endpoint_hierarchy_fit_trajectories": (
            list(hierarchy_calibration.metadata.get("fit_trajectories", ()))
            if hierarchy_calibration is not None else []
        ),
        "endpoint_hierarchy_validation_trajectories": (
            list(hierarchy_calibration.metadata.get("validation_trajectories", ()))
            if hierarchy_calibration is not None else []
        ),
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
