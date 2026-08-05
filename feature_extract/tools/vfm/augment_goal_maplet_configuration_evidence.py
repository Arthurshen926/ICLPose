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
    configuration_candidate_evidence,
    configuration_candidate_latent_evidence,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--typed_graph", required=True)
    parser.add_argument("--child_eligibility", required=True)
    parser.add_argument("--child_local_factor_calibrator", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--maximum_groups", type=int, default=64)
    parser.add_argument("--maximum_children", type=int, default=4)
    parser.add_argument("--evidence_version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--include_trajectories", nargs="+", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic_allow_non_crossfit", action="store_true")
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
    candidate_pool_sha256 = file_sha256(Path(args.candidate_pool))
    training_pool_reuse = calibrator.metadata.get("candidate_pool_sha256") == candidate_pool_sha256
    application_trajectories = set(args.include_trajectories)
    supervised_trajectories = set(calibrator.metadata.get("training_trajectories", ()))
    supervised_trajectories |= set(calibrator.metadata.get("calibration_trajectories", ()))
    supervised_trajectories |= set(calibrator.metadata.get("validation_trajectories", ()))
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
    for artifact in (calibrator,):
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
            ("field_feature_contract_sha256", contract.content_sha256),
            ("child_eligibility_sha256", eligibility.content_sha256),
        ):
            if artifact.metadata.get(key) != expected:
                raise ValueError(f"child-local artifact lineage differs: {key}")
    readout = readout_canonical_field(field, physical)
    validity = ValidityCalibration.load_json(Path(args.validity_calibration))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    evidence_names = LATENT_FEATURE_NAMES if str(args.evidence_version) == "v3" else FEATURE_NAMES
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
        context = encode_radio_final_regions(mapped, token_xy, context_config)
        local = encode_radio_final_regions(mapped, token_xy, local_config)
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
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
        if str(args.evidence_version) == "v3":
            extent = grouped.extent * np.asarray([camera.width, camera.height], dtype=np.float64)
            evidence = configuration_candidate_latent_evidence(
                poses, grouped_local, xy, extent, scale,
                group_parent_ids, group_parent_probability, group_parent_null,
                child, physical, field, graph, eligibility, calibrator, camera,
                maximum_groups=int(args.maximum_groups), maximum_children=int(args.maximum_children),
            )
            evidence_key = "configuration_evidence_v3"
        else:
            evidence = configuration_candidate_evidence(
                poses, grouped_local, xy, scale,
                group_parent_ids, group_parent_probability, group_parent_null,
                child, physical, field, graph, eligibility, calibrator, camera,
                maximum_groups=int(args.maximum_groups), maximum_children=int(args.maximum_children),
            )
            evidence_key = "configuration_evidence_v2"
        updated = json.loads(json.dumps(row))
        updated["ranking_diagnostics"][MODE][evidence_key] = {
            name: evidence[:, index].tolist() for index, name in enumerate(evidence_names)
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
            "fixed_group_denominator": bool(str(args.evidence_version) == "v3"),
            "one_mode_per_group": bool(str(args.evidence_version) == "v3"),
            "typed_null_marginalization": bool(str(args.evidence_version) == "v3"),
            "child_capacity": "projected_area_correlated_cluster_cap1to6" if str(args.evidence_version) == "v3" else None,
            "primitive_capacity": "one_independent_cluster_per_primitive" if str(args.evidence_version) == "v3" else None,
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
