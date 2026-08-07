"""Calibrate group-level parent->child->mode probabilities with exact 2DGS truth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.child_local_likelihood import predict_child_local_surface_likelihood
from feature_extract.vfm.localization_goal_maplet.child_retrieval import retrieve_children_given_parents
from feature_extract.vfm.localization_goal_maplet.endpoint_hierarchy import (
    EndpointHierarchyCalibration,
    conditional_with_tail,
    fit_endpoint_hierarchy_calibration,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.mode_relation import (
    _aggregate_complete_link_observations,
    _aggregate_sparse_probability_rows,
    build_sparse_relation_edges,
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


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0, model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]), height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _new_store() -> dict[str, list]:
    return defaultdict(list)


def _categorical_nll(distributions: list[np.ndarray], targets: list[int], calibrator, level: str | None) -> float:
    losses = []
    for probability, target in zip(distributions, targets):
        value = (
            np.asarray(probability, dtype=np.float64)
            if level is None else calibrator.calibrate_distribution(probability, level=level)
        )
        losses.append(-np.log(max(float(value[int(target)]), 1e-12)))
    return float(np.mean(losses)) if losses else float("nan")


def _report(store: dict[str, list], calibrator) -> dict[str, object]:
    probability = np.asarray(store["support_probability"], dtype=np.float64)
    target = np.asarray(store["support_target"], dtype=np.float64)
    calibrated = calibrator.calibrate_support(probability)
    binary_before = float(np.mean(
        -target * np.log(np.maximum(probability, 1e-12))
        -(1.0 - target) * np.log(np.maximum(1.0 - probability, 1e-12))
    ))
    binary_after = float(np.mean(
        -target * np.log(np.maximum(calibrated, 1e-12))
        -(1.0 - target) * np.log(np.maximum(1.0 - calibrated, 1e-12))
    ))
    result = {
        "query_count": int(len(set(store["image_ids"]))),
        "group_count": int(target.size),
        "support_positive_fraction": float(np.mean(target)),
        "support_probability_mean_before": float(np.mean(probability)),
        "support_probability_mean_after": float(np.mean(calibrated)),
        "support_nll_before": binary_before,
        "support_nll_after": binary_after,
    }
    for level in ("parent", "child", "mode"):
        distributions = store[f"{level}_distributions"]
        targets = store[f"{level}_targets"]
        result[f"{level}_sample_count"] = int(len(targets))
        result[f"{level}_nll_before"] = _categorical_nll(distributions, targets, calibrator, None)
        result[f"{level}_nll_after"] = _categorical_nll(distributions, targets, calibrator, level)
        result[f"{level}_truth_tail_fraction"] = float(np.mean([
            int(target) == len(distribution) - 1
            for distribution, target in zip(distributions, targets)
        ])) if targets else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--fit_trajectories", nargs="+", default=["seq11"])
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq10"])
    parser.add_argument("--maximum_groups", type=int, default=64)
    parser.add_argument("--maximum_children", type=int, default=128)
    parser.add_argument("--maximum_modes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_json), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite endpoint hierarchy calibration")
    fit_trajectories, validation_trajectories = set(args.fit_trajectories), set(args.validation_trajectories)
    if fit_trajectories & validation_trajectories:
        raise ValueError("endpoint hierarchy fit and validation trajectories overlap")

    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    validity = ValidityCalibration.load_json(Path(args.validity_calibration))
    readout = readout_canonical_field(field, physical)
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    stores = {"fit": _new_store(), "validation": _new_store()}
    requested = fit_trajectories | validation_trajectories

    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata["image_id"])
        trajectory = image_id.split("/", 1)[0]
        if trajectory not in requested:
            continue
        partition = "fit" if trajectory in fit_trajectories else "validation"
        store = stores[partition]
        labels, camera = ContributorLabels.load_npz(path), _camera(path)
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        context = encode_radio_final_regions(mapped, token_xy, context_config)
        local = encode_radio_final_regions(mapped, token_xy, local_config)
        parent_ids, parent_probability, parent_null, best_similarity = retrieve_maplet_posterior(
            context, readout.parent_descriptors, physical.maplet_ids,
            readout.parent_coverage > 0.0, maximum_candidates=64, temperature=float(args.temperature),
            null_similarity_center=float(validity.center), null_similarity_scale=float(validity.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy, context, parent_ids[:, 0], token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]), image_width=int(camera.width), image_height=int(camera.height),
            descriptor_half_size_tokens=2.0, minimum_descriptor_cosine=0.96,
        )
        group_parent_ids, group_parent_probability, group_parent_null = aggregate_group_posteriors(
            parent_ids, parent_probability, parent_null,
            grouped.member_offsets, grouped.member_token_indices, maximum_candidates=64,
        )
        token_support = validity.predict_valid(best_similarity)
        group_support = np.asarray([
            np.mean(token_support[grouped.member_token_indices[
                int(grouped.member_offsets[group]):int(grouped.member_offsets[group + 1])
            ]]) for group in range(grouped.member_offsets.size - 1)
        ], dtype=np.float64)
        grouped_local = aggregate_group_descriptors(local, grouped.member_offsets, grouped.member_token_indices)
        child = retrieve_children_given_parents(
            grouped_local, group_parent_ids, group_parent_probability, group_parent_null,
            readout.child_descriptors, readout.child_coverage, physical,
            maximum_child_candidates=int(args.maximum_children), temperature=float(args.temperature),
        )
        priority = group_support
        groups = np.argsort(-priority, kind="stable")[: int(args.maximum_groups)]
        groups = groups[priority[groups] > 0.02]
        if groups.size == 0:
            continue
        xy = grouped.xy[groups] * np.asarray([camera.width, camera.height], dtype=np.float64)
        extent = grouped.extent[groups] * np.asarray([camera.width, camera.height], dtype=np.float64)
        descriptor = grouped_local[groups]
        scale = np.maximum(0.5 * np.linalg.norm(extent, axis=1), 8.0)
        edges = build_sparse_relation_edges(xy, extent, descriptor, scale, query_priority=priority[groups])
        cluster = np.asarray(edges.support_cluster_rows, dtype=np.int64)
        representatives = np.asarray(edges.representative_groups, dtype=np.int64)
        active_parent_ids, active_parent_probability = _aggregate_sparse_probability_rows(
            group_parent_ids[groups], group_parent_probability[groups], cluster, representatives,
            maximum_candidates=group_parent_ids.shape[1],
        )
        active_child_rows, active_child_probability = _aggregate_sparse_probability_rows(
            child.candidate_child_rows[groups], child.candidate_probabilities[groups], cluster, representatives,
            maximum_candidates=child.candidate_child_rows.shape[1],
        )
        active_descriptor, _, _, _ = _aggregate_complete_link_observations(
            descriptor, xy, extent, cluster, representatives,
        )
        active_support = np.asarray([
            np.mean(group_support[groups][cluster == cluster[row]]) for row in representatives.tolist()
        ], dtype=np.float64)
        oracle = token_oracle_evidence(
            labels, physical, token_xy, token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_height=int(camera.height), image_width=int(camera.width), camera=camera,
        )
        truth_children, truth_primitives, valid_rows = [], [], []
        for active_row, representative in enumerate(representatives.tolist()):
            source_groups = groups[np.flatnonzero(cluster == cluster[representative])]
            members = np.concatenate([
                grouped.member_token_indices[
                    int(grouped.member_offsets[source]):int(grouped.member_offsets[source + 1])
                ] for source in source_groups.tolist()
            ])
            observed = members[oracle.child_rows[members] >= 0]
            store["support_probability"].append(float(active_support[active_row]))
            store["support_target"].append(int(observed.size > 0))
            store["image_ids"].append(image_id)
            if observed.size == 0:
                continue
            mass: dict[int, float] = {}
            for token in observed.tolist():
                current = int(oracle.child_rows[token])
                mass[current] = mass.get(current, 0.0) + float(oracle.child_mass[token])
            truth_child = min(mass, key=lambda value: (-mass[value], value))
            owned = observed[oracle.child_rows[observed] == truth_child]
            weight = np.asarray(oracle.child_mass[owned], dtype=np.float64)
            truth_xyz = np.average(oracle.child_local_xyz[owned], axis=0, weights=weight)
            start, end = int(physical.child_member_offsets[truth_child]), int(physical.child_member_offsets[truth_child + 1])
            primitive_members = physical.child_member_primitive_rows[start:end]
            truth_primitive = int(primitive_members[np.argmin(
                np.linalg.norm(physical.primitive_centers[primitive_members] - truth_xyz, axis=1)
            )])

            parent_valid = (active_parent_ids[active_row] >= 0) & (active_parent_probability[active_row] > 0.0)
            parent_identity = active_parent_ids[active_row, parent_valid]
            parent_distribution = conditional_with_tail(
                active_parent_probability[active_row, parent_valid], active_support[active_row],
            )
            truth_parent = int(physical.maplet_ids[physical.child_parent_rows[truth_child]])
            match = np.flatnonzero(parent_identity == truth_parent)
            store["parent_distributions"].append(parent_distribution)
            store["parent_targets"].append(int(match[0]) if match.size else parent_distribution.size - 1)

            child_parent_ids = physical.maplet_ids[
                physical.child_parent_rows[np.maximum(active_child_rows[active_row], 0)]
            ]
            child_valid = (
                (active_child_rows[active_row] >= 0) & (active_child_probability[active_row] > 0.0)
                & (child_parent_ids == truth_parent)
            )
            child_identity = active_child_rows[active_row, child_valid]
            parent_mass = float(np.sum(
                active_parent_probability[active_row, parent_valid][parent_identity == truth_parent]
            ))
            child_distribution = conditional_with_tail(
                active_child_probability[active_row, child_valid], parent_mass,
            )
            match = np.flatnonzero(child_identity == truth_child)
            store["child_distributions"].append(child_distribution)
            store["child_targets"].append(int(match[0]) if match.size else child_distribution.size - 1)
            truth_children.append(truth_child)
            truth_primitives.append(truth_primitive)
            valid_rows.append(active_row)

        if valid_rows:
            likelihood = predict_child_local_surface_likelihood(
                active_descriptor[np.asarray(valid_rows, dtype=np.int64)],
                np.asarray(truth_children, dtype=np.int64), physical, field,
                temperature=float(args.temperature), maximum_modes=int(args.maximum_modes),
            )
            for modes, probability, truth in zip(
                likelihood.mode_primitive_rows, likelihood.mode_probabilities, truth_primitives,
            ):
                valid = (modes >= 0) & (probability > 0.0)
                identity = modes[valid]
                distribution = conditional_with_tail(probability[valid], 1.0)
                match = np.flatnonzero(identity == int(truth))
                store["mode_distributions"].append(distribution)
                store["mode_targets"].append(int(match[0]) if match.size else distribution.size - 1)
        print(json.dumps({"image_id": image_id, "partition": partition, "clusters": int(representatives.size)}), flush=True)

    fit = stores["fit"]
    fitted_artifact = fit_endpoint_hierarchy_calibration(
        np.asarray(fit["support_probability"]), np.asarray(fit["support_target"]),
        fit["parent_distributions"], np.asarray(fit["parent_targets"]),
        fit["child_distributions"], np.asarray(fit["child_targets"]),
        fit["mode_distributions"], np.asarray(fit["mode_targets"]),
        metadata={
            "fit_trajectories": sorted(fit_trajectories),
            "validation_trajectories": sorted(validation_trajectories),
            "calibration_unit": "complete_link_collapsed_query_group",
            "proper_scoring_rule": "binary_and_categorical_log_score",
            "physical_map_sha256": physical.content_sha256,
            "canonical_field_sha256": field.content_sha256,
            "field_feature_contract_sha256": contract.content_sha256,
            "uses_gt_at_runtime": False, "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False, "stored_downstream_embedding_count": 0,
        },
    )
    fitted_validation = _report(stores["validation"], fitted_artifact)
    accepted_levels = {
        "support": bool(fitted_validation["support_nll_after"] <= fitted_validation["support_nll_before"]),
        "parent": bool(fitted_validation["parent_nll_after"] <= fitted_validation["parent_nll_before"]),
        "child": bool(fitted_validation["child_nll_after"] <= fitted_validation["child_nll_before"]),
        "mode": bool(fitted_validation["mode_nll_after"] <= fitted_validation["mode_nll_before"]),
    }
    selected_metadata = {
        **dict(fitted_artifact.metadata),
        "level_selection": "predefined_validation_proper_score_gate_v1",
        "accepted_levels": accepted_levels,
    }
    artifact = EndpointHierarchyCalibration(
        fitted_artifact.support_logit_scale if accepted_levels["support"] else 1.0,
        fitted_artifact.support_logit_bias if accepted_levels["support"] else 0.0,
        fitted_artifact.parent_temperature if accepted_levels["parent"] else 1.0,
        fitted_artifact.child_temperature if accepted_levels["child"] else 1.0,
        fitted_artifact.mode_temperature if accepted_levels["mode"] else 1.0,
        selected_metadata,
    )
    artifact.save_json(output)
    result = {
        "stage": "train_goal_maplet_endpoint_hierarchy_calibration_v1",
        "artifact": str(output), "content_sha256": artifact.content_sha256,
        "parameters": {
            "support_logit_scale": artifact.support_logit_scale,
            "support_logit_bias": artifact.support_logit_bias,
            "parent_temperature": artifact.parent_temperature,
            "child_temperature": artifact.child_temperature,
            "mode_temperature": artifact.mode_temperature,
        },
        "fitted_parameters": {
            "support_logit_scale": fitted_artifact.support_logit_scale,
            "support_logit_bias": fitted_artifact.support_logit_bias,
            "parent_temperature": fitted_artifact.parent_temperature,
            "child_temperature": fitted_artifact.child_temperature,
            "mode_temperature": fitted_artifact.mode_temperature,
        },
        "accepted_levels": accepted_levels,
        "fitted_validation": fitted_validation,
        "fit": _report(stores["fit"], artifact),
        "validation": _report(stores["validation"], artifact),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
