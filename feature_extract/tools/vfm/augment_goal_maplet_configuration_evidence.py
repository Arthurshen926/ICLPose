"""Augment a frozen Top-32 pool with calibrated configuration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.child_eligibility import ChildGeometryEligibility
from feature_extract.vfm.localization_goal_maplet.child_local_factor import ChildLocalFactorCalibratorArtifact
from feature_extract.vfm.localization_goal_maplet.child_retrieval import retrieve_children_given_parents
from feature_extract.vfm.localization_goal_maplet.configuration_evidence import (
    FEATURE_NAMES,
    LATENT_FEATURE_NAMES,
    LIKELIHOOD_FEATURE_NAMES,
    configuration_candidate_evidence,
    configuration_candidate_latent_evidence,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.endpoint_hierarchy import EndpointHierarchyCalibration
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
    transform_canonical_field_for_role,
)
from feature_extract.vfm.localization_goal_maplet.pose_likelihood_ratio import PoseLikelihoodRatioArtifact
from feature_extract.vfm.localization_goal_maplet.mode_relation import (
    RELATION_EVIDENCE_NAMES,
    ModeRelationLikelihoodRatioArtifact,
    family_preserving_child_shortlist,
    configuration_mode_relation_evidence,
)
from feature_extract.vfm.localization_goal_maplet.child_local_likelihood import (
    predict_child_local_surface_likelihood,
)
from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import (
    child_local_mode_runtime_features,
)
from feature_extract.vfm.localization_goal_maplet.oracle_pose import token_oracle_evidence
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import ValidityCalibration, retrieve_maplet_posterior
from feature_extract.vfm.localization_goal_maplet.typed_graph import TypedParentGraph
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


MODE = "actual_parent_actual_child"


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0, model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]), height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _offline_truth_coverage_ladder(
    *,
    contributor_path: Path,
    camera: ColmapCamera,
    token_xy: np.ndarray,
    token_height: int,
    token_width: int,
    grouped,
    grouped_local: np.ndarray,
    child_posterior,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    eligibility: ChildGeometryEligibility,
    relation,
    runtime_maximum_children: int,
    maximum_modes: int = 8,
    shortlist_modes_per_child: int = 2,
    temperature: float = 0.07,
) -> dict[str, object]:
    """Report the GT endpoint funnel without exposing GT to runtime scores."""

    labels = ContributorLabels.load_npz(contributor_path)
    oracle = token_oracle_evidence(
        labels, physical, token_xy,
        token_height=int(token_height), token_width=int(token_width),
        image_height=int(camera.height), image_width=int(camera.width), camera=camera,
    )
    selected_groups = np.asarray(
        (relation.query_diagnostics or {}).get("selected_group_rows", ()), dtype=np.int64,
    )
    stage_names = (
        "truth_child_full_posterior", "truth_child_adaptive_budget",
        "truth_primitive_top8_given_adaptive_child",
        "truth_primitive_adaptive_budget", "endpoint_pose_valid",
    )
    if selected_groups.size == 0:
        return {
            "offline_gt_only": True, "observable_group_count": 0,
            **{name: {"count": 0, "fraction": 0.0} for name in stage_names},
            "pair_both_endpoints_valid": {"count": 0, "fraction": 0.0},
        }
    truth_child, truth_xyz, truth_local_group = [], [], []
    cluster_members = (relation.query_diagnostics or {}).get("cluster_member_group_rows", ())
    for local_group, group in enumerate(selected_groups.tolist()):
        source_groups = (
            [int(value) for value in cluster_members[local_group]]
            if local_group < len(cluster_members) else [int(group)]
        )
        members = np.concatenate([
            grouped.member_token_indices[
                int(grouped.member_offsets[source]) : int(grouped.member_offsets[source + 1])
            ]
            for source in source_groups
        ])
        members = members[oracle.child_rows[members] >= 0]
        if members.size == 0:
            continue
        mass: dict[int, float] = {}
        for token in members.tolist():
            value = int(oracle.child_rows[token])
            mass[value] = mass.get(value, 0.0) + float(oracle.child_mass[token])
        child = min(mass, key=lambda value: (-mass[value], value))
        owned = members[oracle.child_rows[members] == child]
        weight = np.asarray(oracle.child_mass[owned], dtype=np.float64)
        if float(np.sum(weight)) <= 0.0:
            continue
        truth_child.append(int(child))
        truth_xyz.append(np.average(oracle.child_local_xyz[owned], axis=0, weights=weight))
        truth_local_group.append(int(local_group))
    count = len(truth_child)
    if count == 0:
        return {
            "offline_gt_only": True, "observable_group_count": 0,
            **{name: {"count": 0, "fraction": 0.0} for name in stage_names},
            "pair_both_endpoints_valid": {"count": 0, "fraction": 0.0},
        }
    truth_child = np.asarray(truth_child, dtype=np.int64)
    truth_xyz = np.asarray(truth_xyz, dtype=np.float64)
    truth_local_group = np.asarray(truth_local_group, dtype=np.int64)
    active_candidate_rows = np.asarray(
        (relation.query_diagnostics or {}).get("active_candidate_child_rows", ()), dtype=np.int64,
    )
    adaptive_child_rows = (relation.query_diagnostics or {}).get(
        "adaptive_selected_child_rows", (),
    )
    adaptive_primitive_rows = (relation.query_diagnostics or {}).get(
        "adaptive_selected_primitive_rows", (),
    )
    full_child = np.asarray([
        bool(
            active_candidate_rows.ndim == 2
            and local_group < active_candidate_rows.shape[0]
            and np.any(active_candidate_rows[local_group] == child)
        )
        for local_group, child in zip(truth_local_group.tolist(), truth_child.tolist())
    ], dtype=bool)
    adaptive_child = np.asarray([
        bool(local_group < len(adaptive_child_rows) and child in adaptive_child_rows[local_group])
        for local_group, child in zip(truth_local_group.tolist(), truth_child.tolist())
    ], dtype=bool)

    likelihood = predict_child_local_surface_likelihood(
        grouped_local[selected_groups[truth_local_group]], truth_child, physical, field,
        temperature=float(temperature), maximum_modes=int(maximum_modes),
    )
    mode_rows = np.asarray(likelihood.mode_primitive_rows, dtype=np.int64)
    actual_primitive = np.full((count,), -1, dtype=np.int64)
    for index, child in enumerate(truth_child.tolist()):
        start, end = int(physical.child_member_offsets[child]), int(physical.child_member_offsets[child + 1])
        members = physical.child_member_primitive_rows[start:end]
        if members.size:
            actual_primitive[index] = int(members[np.argmin(
                np.linalg.norm(physical.primitive_centers[members] - truth_xyz[index], axis=1)
            )])
    top8 = np.any(mode_rows == actual_primitive[:, None], axis=1)
    adaptive_primitive = np.asarray([
        bool(
            local_group < len(adaptive_primitive_rows)
            and int(primitive) in adaptive_primitive_rows[local_group]
        )
        for local_group, primitive in zip(
            truth_local_group.tolist(), actual_primitive.tolist(),
        )
    ], dtype=bool)
    scale_px = np.maximum(
        0.5 * np.linalg.norm(
            grouped.extent[selected_groups[truth_local_group]]
            * np.asarray([camera.width, camera.height], dtype=np.float64),
            axis=1,
        ),
        8.0,
    )
    xy_px = grouped.xy[selected_groups[truth_local_group]] * np.asarray(
        [camera.width, camera.height], dtype=np.float64,
    )
    _, mode_valid = child_local_mode_runtime_features(
        likelihood, truth_child, xy_px, scale_px, labels.pose_w2c,
        camera, physical, field,
    )
    endpoint_valid = np.zeros((count,), dtype=bool)
    for index in range(count):
        slots = np.flatnonzero(mode_rows[index] == actual_primitive[index])
        if slots.size:
            endpoint_valid[index] = bool(
                adaptive_primitive[index] and mode_valid[index, slots[0]]
            )
    stages = {
        "truth_child_full_posterior": full_child,
        "truth_child_adaptive_budget": full_child & adaptive_child,
        "truth_primitive_top8_given_adaptive_child": full_child & adaptive_child & top8,
        "truth_primitive_adaptive_budget": full_child & adaptive_child & adaptive_primitive,
        "endpoint_pose_valid": full_child & adaptive_child & adaptive_primitive & endpoint_valid,
    }
    endpoint_by_group = np.zeros((selected_groups.size,), dtype=bool)
    endpoint_by_group[truth_local_group] = stages["endpoint_pose_valid"]
    edge_left = np.concatenate([
        relation.relation_edges.fit_left, relation.relation_edges.verify_left,
    ])
    edge_right = np.concatenate([
        relation.relation_edges.fit_right, relation.relation_edges.verify_right,
    ])
    edge_family = np.concatenate([
        relation.relation_edges.fit_family, relation.relation_edges.verify_family,
    ])
    both = endpoint_by_group[edge_left] & endpoint_by_group[edge_right]
    output: dict[str, object] = {
        "offline_gt_only": True,
        "excluded_from_runtime_score": True,
        "observable_group_count": int(count),
    }
    previous = np.ones((count,), dtype=bool)
    for name in stage_names:
        value = stages[name]
        output[name] = {
            "count": int(np.sum(value)),
            "fraction": float(np.mean(value)),
            "conditional_survival": float(np.sum(value) / max(int(np.sum(previous)), 1)),
        }
        previous = value
    output["pair_both_endpoints_valid"] = {
        "count": int(np.sum(both)), "edge_count": int(both.size),
        "fraction": float(np.mean(both)) if both.size else 0.0,
        "by_family": {
            name: float(np.mean(both[edge_family == family]))
            if np.any(edge_family == family) else 0.0
            for family, name in enumerate(("local", "long_range", "depth_normal"))
        },
    }
    return output


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
    parser.add_argument("--physical_instance_readout", default="")
    parser.add_argument("--typed_graph", required=True)
    parser.add_argument("--child_eligibility", required=True)
    parser.add_argument("--child_local_factor_calibrator", required=True)
    parser.add_argument("--pose_likelihood_ratio")
    parser.add_argument("--mode_relation_likelihood_ratio")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--maximum_groups", type=int, default=64)
    parser.add_argument("--maximum_children", type=int, default=4)
    parser.add_argument("--endpoint_state_budget", type=int, default=16)
    parser.add_argument(
        "--endpoint_state_policy",
        choices=("mass_adaptive_g16", "mass_adaptive_g17_breadth_depth"),
        default="mass_adaptive_g16",
    )
    parser.add_argument(
        "--relation_support_policy",
        choices=("cluster_collapse", "fractional_unary", "g15_llr_only"),
        default="cluster_collapse",
    )
    parser.add_argument("--evidence_version", choices=("v2", "v3", "v4"), default="v2")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--include_trajectories", nargs="+", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic_allow_non_crossfit", action="store_true")
    parser.add_argument("--diagnostic_allow_legacy_relation_model", action="store_true")
    parser.add_argument("--reuse_existing_configuration_evidence", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite augmented configuration pool")
    pool = json.loads(Path(args.candidate_pool).read_text())
    pool_rows = {str(row["image_id"]): row for row in pool["rows"]}
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    graph = TypedParentGraph.load_npz(Path(args.typed_graph))
    eligibility = ChildGeometryEligibility.load_npz(Path(args.child_eligibility))
    calibrator = ChildLocalFactorCalibratorArtifact.load(Path(args.child_local_factor_calibrator))
    if str(args.evidence_version) == "v4" and not args.pose_likelihood_ratio:
        raise ValueError("configuration evidence v4 requires --pose_likelihood_ratio")
    likelihood_ratio = (
        PoseLikelihoodRatioArtifact.load(Path(args.pose_likelihood_ratio))
        if args.pose_likelihood_ratio else None
    )
    relation_ratio = (
        ModeRelationLikelihoodRatioArtifact.load(Path(args.mode_relation_likelihood_ratio))
        if args.mode_relation_likelihood_ratio else None
    )
    hierarchy_calibration = (
        EndpointHierarchyCalibration.load_json(Path(args.endpoint_hierarchy_calibration))
        if args.endpoint_hierarchy_calibration else None
    )
    if relation_ratio is not None and likelihood_ratio is None:
        raise ValueError("mode-relation evidence requires --pose_likelihood_ratio unary factors")
    if relation_ratio is not None and str(args.evidence_version) != "v4":
        raise ValueError("mode-relation evidence requires fixed-denominator evidence v4")
    if likelihood_ratio is not None and int(
        likelihood_ratio.metadata.get("runtime_maximum_children", -1)
    ) != int(args.maximum_children):
        raise ValueError(
            "pose-likelihood training Top-C differs from runtime maximum_children"
        )
    if relation_ratio is not None and int(
        relation_ratio.metadata.get("runtime_maximum_children", -1)
    ) != int(args.maximum_children):
        raise ValueError("mode-relation training Top-C differs from runtime maximum_children")
    expected_option_contract = (
        "mass_adaptive_parent_child_mode_breadth_depth_budget_v4"
        if args.endpoint_state_policy == "mass_adaptive_g17_breadth_depth"
        else "mass_adaptive_parent_child_mode_budget_v3"
    )
    if (
        relation_ratio is not None
        and relation_ratio.metadata.get("option_contract")
        != expected_option_contract
        and not bool(args.diagnostic_allow_legacy_relation_model)
    ):
        raise ValueError(
            "deployment replay requires a relation model trained on the selected endpoint states"
        )
    if (
        relation_ratio is not None
        and relation_ratio.metadata.get("option_contract") == expected_option_contract
        and int(relation_ratio.metadata.get("endpoint_state_budget", -1))
        != int(args.endpoint_state_budget)
    ):
        raise ValueError("mode-relation endpoint-state budget differs from runtime")
    candidate_pool_sha256 = file_sha256(Path(args.candidate_pool))
    training_pool_reuse = calibrator.metadata.get("candidate_pool_sha256") == candidate_pool_sha256
    application_trajectories = set(args.include_trajectories)
    supervised_trajectories = set()
    for artifact in (calibrator, likelihood_ratio, relation_ratio, hierarchy_calibration):
        if artifact is None:
            continue
        supervised_trajectories |= set(artifact.metadata.get("training_trajectories", ()))
        supervised_trajectories |= set(artifact.metadata.get("fit_trajectories", ()))
        supervised_trajectories |= set(artifact.metadata.get("calibration_trajectories", ()))
        supervised_trajectories |= set(artifact.metadata.get("validation_trajectories", ()))
        supervised_trajectories |= set(artifact.metadata.get("endpoint_hierarchy_fit_trajectories", ()))
        supervised_trajectories |= set(artifact.metadata.get("endpoint_hierarchy_validation_trajectories", ()))
    trajectory_cross_fit = bool(application_trajectories) and not bool(
        application_trajectories & supervised_trajectories
    )
    if training_pool_reuse and not trajectory_cross_fit and not bool(args.diagnostic_allow_non_crossfit):
        raise ValueError(
            "factor calibrator was supervised from this candidate pool; "
            "outer cross-fit evidence is required for ranker training"
        )
    for key, expected in (
        ("physical_map_sha256", physical.content_sha256),
        ("canonical_field_sha256", field.content_sha256),
        ("typed_graph_sha256", graph.content_sha256),
        ("field_feature_contract_sha256", contract.content_sha256),
    ):
        if pool.get(key) != expected:
            raise ValueError(f"candidate pool lineage differs: {key}")
    for artifact in (calibrator, likelihood_ratio, relation_ratio):
        if artifact is None:
            continue
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
            ("field_feature_contract_sha256", contract.content_sha256),
            ("child_eligibility_sha256", eligibility.content_sha256),
        ):
            if artifact.metadata.get(key) != expected:
                raise ValueError(f"child-local artifact lineage differs: {key}")
    readout = readout_canonical_field(field, physical)
    instance_readout = None
    local_field = field
    if args.physical_instance_readout:
        instance_readout, instance_metadata = load_physical_instance_readout(
            Path(args.physical_instance_readout), device=str(args.device),
        )
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
        ):
            if instance_metadata.get(key) != expected:
                raise ValueError(f"physical-instance readout lineage differs: {key}")
        readout = type(readout)(
            instance_readout.project_numpy(readout.parent_descriptors, role="context", device=str(args.device)),
            readout.parent_coverage,
            instance_readout.project_numpy(readout.child_descriptors, role="local", device=str(args.device)),
            readout.child_coverage,
        )
        local_field = transform_canonical_field_for_role(
            instance_readout, field, role="local", device=str(args.device),
        )
    actual_instance = (
        file_sha256(Path(args.physical_instance_readout))
        if args.physical_instance_readout else None
    )
    if pool.get("physical_instance_readout_sha256") != actual_instance:
        raise ValueError("candidate pool and runtime physical-instance readout differ")
    graph_instance = graph.metadata.get("physical_instance_readout_sha256")
    if graph_instance != actual_instance:
        raise ValueError("typed graph and runtime physical-instance readout differ")
    if likelihood_ratio is not None:
        unary_instance = likelihood_ratio.metadata.get("physical_instance_readout_sha256")
        if unary_instance != actual_instance and not bool(args.diagnostic_allow_legacy_relation_model):
            raise ValueError("pose-likelihood model and runtime physical-instance readout differ")
    calibrator_instance = calibrator.metadata.get("physical_instance_readout_sha256")
    if calibrator_instance != actual_instance and not bool(args.diagnostic_allow_legacy_relation_model):
        raise ValueError("child-local calibrator and runtime physical-instance readout differ")
    validity = ValidityCalibration.load_json(Path(args.validity_calibration))
    if hierarchy_calibration is not None:
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
            ("field_feature_contract_sha256", contract.content_sha256),
        ):
            if hierarchy_calibration.metadata.get(key) != expected:
                raise ValueError(f"endpoint hierarchy calibration lineage differs: {key}")
    if relation_ratio is not None:
        expected_hierarchy = relation_ratio.metadata.get("endpoint_hierarchy_calibration_sha256")
        actual_hierarchy = hierarchy_calibration.content_sha256 if hierarchy_calibration is not None else None
        if expected_hierarchy != actual_hierarchy:
            raise ValueError("relation model and runtime endpoint hierarchy calibration differ")
        expected_instance = relation_ratio.metadata.get("physical_instance_readout_sha256")
        if expected_instance != actual_instance:
            raise ValueError("relation model and runtime physical-instance readout differ")
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    evidence_names = (
        LIKELIHOOD_FEATURE_NAMES if str(args.evidence_version) == "v4"
        else LATENT_FEATURE_NAMES if str(args.evidence_version) == "v3"
        else FEATURE_NAMES
    )
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    rows = []
    for path in paths:
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata["image_id"])
        if application_trajectories and image_id.split("/", 1)[0] not in application_trajectories:
            continue
        if image_id not in pool_rows:
            continue
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        if instance_readout is None:
            context = encode_radio_final_regions(mapped, token_xy, context_config)
            local = encode_radio_final_regions(mapped, token_xy, local_config)
        else:
            context = encode_physical_instance_regions(
                instance_readout, mapped, token_xy, role="context", device=str(args.device),
            )
            local = encode_physical_instance_regions(
                instance_readout, mapped, token_xy, role="local", device=str(args.device),
            )
        parent_ids, parent_probability, parent_null, parent_best_similarity = retrieve_maplet_posterior(
            context, readout.parent_descriptors, physical.maplet_ids,
            readout.parent_coverage > 0.0, maximum_candidates=64, temperature=0.07,
            null_similarity_center=float(validity.center), null_similarity_scale=float(validity.scale),
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
        token_support_valid = validity.predict_valid(parent_best_similarity)
        group_support_valid = np.asarray([
            np.mean(token_support_valid[
                grouped.member_token_indices[
                    int(grouped.member_offsets[group]) : int(grouped.member_offsets[group + 1])
                ]
            ])
            for group in range(grouped.member_offsets.size - 1)
        ], dtype=np.float64)
        grouped_local = aggregate_group_descriptors(local, grouped.member_offsets, grouped.member_token_indices)
        child = retrieve_children_given_parents(
            grouped_local, group_parent_ids, group_parent_probability, group_parent_null,
            readout.child_descriptors, readout.child_coverage, physical,
            maximum_child_candidates=128, temperature=0.07,
        )
        xy = grouped.xy * np.asarray([camera.width, camera.height], dtype=np.float64)
        scale = np.maximum(
            0.5 * np.linalg.norm(grouped.extent * np.asarray([camera.width, camera.height]), axis=1), 8.0
        )
        row = pool_rows[image_id]
        poses = np.asarray([item["pose_w2c"] for item in row["mode_details"][MODE]], dtype=np.float64)
        updated = json.loads(json.dumps(row))
        extent = grouped.extent * np.asarray([camera.width, camera.height], dtype=np.float64)
        evidence_key = (
            f"configuration_evidence_{args.evidence_version}"
            if str(args.evidence_version) in ("v3", "v4") else "configuration_evidence_v2"
        )
        if bool(args.reuse_existing_configuration_evidence):
            existing = updated["ranking_diagnostics"][MODE].get(evidence_key)
            if existing is None or any(name not in existing for name in evidence_names):
                raise ValueError(
                    f"candidate pool cannot reuse missing {evidence_key}: {image_id}"
                )
        else:
            if str(args.evidence_version) in ("v3", "v4"):
                evidence = configuration_candidate_latent_evidence(
                    poses, grouped_local, xy, extent, scale,
                    group_parent_ids, group_parent_probability, group_parent_null,
                    child, physical, local_field, graph, eligibility, calibrator, camera,
                    pose_likelihood_ratio=likelihood_ratio,
                    maximum_groups=int(args.maximum_groups), maximum_children=int(args.maximum_children),
                )
            else:
                evidence = configuration_candidate_evidence(
                    poses, grouped_local, xy, scale,
                    group_parent_ids, group_parent_probability, group_parent_null,
                    child, physical, local_field, graph, eligibility, calibrator, camera,
                    maximum_groups=int(args.maximum_groups), maximum_children=int(args.maximum_children),
                )
            updated["ranking_diagnostics"][MODE][evidence_key] = {
                name: evidence[:, index].tolist() for index, name in enumerate(evidence_names)
            }
        if relation_ratio is not None:
            relation = configuration_mode_relation_evidence(
                poses, grouped_local, xy, extent, scale,
                group_parent_ids, group_parent_probability, group_parent_null,
                child, physical, local_field, eligibility, likelihood_ratio, relation_ratio, camera,
                maximum_groups=int(args.maximum_groups),
                retrieval_maximum_children=int(args.maximum_children),
                endpoint_state_budget=int(args.endpoint_state_budget),
                endpoint_state_policy=str(args.endpoint_state_policy),
                support_correlation_policy=str(args.relation_support_policy),
                support_valid_probabilities=group_support_valid,
                endpoint_hierarchy_calibration=hierarchy_calibration,
            )
            updated["ranking_diagnostics"][MODE]["mode_relation_evidence_v2"] = {
                name: relation.features[:, index].tolist()
                for index, name in enumerate(RELATION_EVIDENCE_NAMES)
            }
            updated["ranking_diagnostics"][MODE]["mode_relation_assignments_v2"] = {
                "selected_child_rows": relation.selected_child_rows.tolist(),
                "selected_primitive_rows": relation.selected_primitive_rows.tolist(),
                "fit_edges": np.stack([
                    relation.relation_edges.fit_left,
                    relation.relation_edges.fit_right,
                    relation.relation_edges.fit_family,
                ], axis=1).tolist() if relation.relation_edges.fit_left.size else [],
                "verify_edges": np.stack([
                    relation.relation_edges.verify_left,
                    relation.relation_edges.verify_right,
                    relation.relation_edges.verify_family,
                ], axis=1).tolist() if relation.relation_edges.verify_left.size else [],
                "representative_group_rows": relation.relation_edges.representative_groups.tolist(),
                "support_cluster_rows": relation.relation_edges.support_cluster_rows.tolist(),
                "legacy_connected_cluster_rows": relation.relation_edges.legacy_connected_cluster_rows.tolist(),
                "query_diagnostics": dict(relation.query_diagnostics or {}),
            }
            updated["ranking_diagnostics"][MODE]["mode_relation_endpoint_coverage_v2"] = (
                _offline_truth_coverage_ladder(
                    contributor_path=path, camera=camera, token_xy=token_xy,
                    token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
                    grouped=grouped, grouped_local=grouped_local,
                    child_posterior=child, physical=physical, field=local_field,
                    eligibility=eligibility, relation=relation,
                    runtime_maximum_children=int(args.maximum_children),
                )
            )
            # One exact-GT replay is an offline probability-semantics audit;
            # its values are stored separately and never enter candidate
            # ranking or deployment state.
            gt_pose = ContributorLabels.load_npz(path).pose_w2c
            gt_relation = configuration_mode_relation_evidence(
                np.asarray(gt_pose, dtype=np.float64)[None], grouped_local, xy, extent, scale,
                group_parent_ids, group_parent_probability, group_parent_null,
                child, physical, local_field, eligibility, likelihood_ratio, relation_ratio, camera,
                maximum_groups=int(args.maximum_groups),
                retrieval_maximum_children=int(args.maximum_children),
                endpoint_state_budget=int(args.endpoint_state_budget),
                endpoint_state_policy=str(args.endpoint_state_policy),
                support_correlation_policy=str(args.relation_support_policy),
                support_valid_probabilities=group_support_valid,
            )
            updated["ranking_diagnostics"][MODE]["mode_relation_gt_pose_evidence_v2"] = {
                "offline_gt_only": True,
                "excluded_from_runtime_score": True,
                **{
                    name: float(gt_relation.features[0, index])
                    for index, name in enumerate(RELATION_EVIDENCE_NAMES)
                },
            }
        rows.append(updated)
        print(json.dumps({"image_id": image_id, "candidate_count": int(poses.shape[0])}), flush=True)
    result = {
        **{key: value for key, value in pool.items() if key not in ("rows", "summary", "source_shards", "shard_count", "query_count")},
        "stage": f"goal_maplet_configuration_evidence_{args.evidence_version}",
        "query_count": len(rows), "rows": rows,
        "configuration_evidence_contract": {
            "feature_names": list(evidence_names),
            "maximum_groups": int(args.maximum_groups), "maximum_children": int(args.maximum_children),
            "child_local_mode_source": "fixed_vfm_topm",
            "evidence_version": str(args.evidence_version),
            "fixed_group_denominator": bool(str(args.evidence_version) in ("v3", "v4")),
            "one_mode_per_group": bool(str(args.evidence_version) in ("v3", "v4")),
            "typed_null_marginalization": bool(str(args.evidence_version) in ("v3", "v4")),
            "child_capacity": "projected_area_correlated_cluster_cap1to6" if str(args.evidence_version) in ("v3", "v4") else None,
            "primitive_capacity": "one_independent_cluster_per_primitive" if str(args.evidence_version) in ("v3", "v4") else None,
            "pose_likelihood_ratio_sha256": file_sha256(Path(args.pose_likelihood_ratio)) if args.pose_likelihood_ratio else None,
            "pose_likelihood_pairing_contract": likelihood_ratio.metadata.get("pairing_contract") if likelihood_ratio is not None else None,
            "soft_assignment": "entropy_regularized_fixed_topm_with_explicit_null" if str(args.evidence_version) == "v4" else None,
            "soft_capacity": "projected_dual_expected_occupancy" if str(args.evidence_version) == "v4" else None,
            "mode_relation_likelihood_ratio_sha256": file_sha256(Path(args.mode_relation_likelihood_ratio)) if args.mode_relation_likelihood_ratio else None,
            "mode_relation_pairing_contract": relation_ratio.metadata.get("pairing_contract") if relation_ratio is not None else None,
            "mode_relation_edge_contract": relation_ratio.metadata.get("edge_contract") if relation_ratio is not None else None,
            "mode_relation_option_contract": relation_ratio.metadata.get("option_contract") if relation_ratio is not None else None,
            "mode_relation_feature_names": list(RELATION_EVIDENCE_NAMES) if relation_ratio is not None else None,
            "mode_relation_inference": "mass_adaptive_hierarchical_exact_sum_product_fit_tree_v3" if relation_ratio is not None else None,
            "mode_relation_pose_evidence": "layered_endpoint_mass_plus_node_and_fit_llr_v3" if relation_ratio is not None else None,
            "mode_relation_decode": "exact_max_sum" if relation_ratio is not None else None,
            "mode_relation_verification": "exact_fit_tree_pair_marginal_unconditional_predictive_llr_v2" if relation_ratio is not None else None,
            "mode_relation_shortlist": (
                f"breadth_then_depth_parent_child_mode_fixed_budget_{int(args.endpoint_state_budget)}_v4"
                if relation_ratio is not None and args.endpoint_state_policy == "mass_adaptive_g17_breadth_depth"
                else f"best_first_parent_child_mode_fixed_budget_{int(args.endpoint_state_budget)}_v3"
                if relation_ratio is not None else None
            ),
            "mode_relation_support_decorrelation": f"complete_link_{str(args.relation_support_policy)}_v3" if relation_ratio is not None else None,
            "mode_relation_probability_mass": "support_invalid_plus_parent_child_mode_tails_plus_geometry_field_and_nonnull_equals_one_v3" if relation_ratio is not None else None,
            "mode_relation_endpoint_state_budget": int(args.endpoint_state_budget) if relation_ratio is not None else None,
            "mode_relation_endpoint_state_policy": str(args.endpoint_state_policy) if relation_ratio is not None else None,
            "physical_instance_readout_sha256": (
                file_sha256(Path(args.physical_instance_readout))
                if args.physical_instance_readout else None
            ),
            "endpoint_hierarchy_calibration_sha256": (
                hierarchy_calibration.content_sha256 if hierarchy_calibration is not None else None
            ),
            "mode_relation_null_evidence": "fixed_physical_invalid_llr_query_uncertainty_neutral_v2" if relation_ratio is not None else None,
            "child_local_factor_calibrator_sha256": file_sha256(Path(args.child_local_factor_calibrator)),
            "application_trajectories": sorted(application_trajectories),
            "factor_training_pool_disjoint": not training_pool_reuse,
            "outer_cross_fit": bool(trajectory_cross_fit),
            "non_crossfit_diagnostic_only": bool(training_pool_reuse and not trajectory_cross_fit),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
