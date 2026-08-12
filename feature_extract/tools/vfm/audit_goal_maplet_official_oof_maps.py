"""Audit every train-side G23 fold map for route leakage and fit selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.mapping_view_graph import MappingViewGraph
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


SUMMARY_PATHS = {
    "surface_mapper": "surface_mapper.json",
    "geometry_head": "geometry_head/geometry_head_summary.json",
    "canonical_field": "canonical_field.json",
    "physical_readout": "physical_readout.json",
    "typed_graph": "typed_graph.json",
    "mapping_view_graph": "mapping_view_graph.json",
    "validity": "validity.json",
}


def _trajectory_set(image_ids: Sequence[str]) -> set[str]:
    return {str(value).split("/", 1)[0] for value in image_ids}


def audit_fold(
    fold: dict[str, object], fold_dir: Path, *, official_test_routes: set[str]
) -> dict[str, object]:
    reports = {
        name: json.loads((fold_dir / relative).read_text())
        for name, relative in SUMMARY_PATHS.items()
    }
    mapping = set(str(value) for value in fold["mapping_trajectories"])
    held = set(str(value) for value in fold["held_query_trajectories"])
    forbidden = held | official_test_routes
    expected_mapping_count = int(fold["mapping_count"])

    mapper = reports["surface_mapper"]
    mapper_split = mapper["split"]
    mapper_training = set(str(value) for value in mapper_split["training_trajectory_ids"])
    mapper_prototype = set(str(value) for value in mapper_split["prototype_trajectory_ids"])
    mapper_holdout = set(
        str(value) for value in mapper_split["strict_holdout_trajectory_ids"]
    )
    if mapper_training != mapping or mapper_prototype != mapping:
        raise ValueError(f"{fold_dir.name}: mapper route set differs from fold mapping set")
    if not forbidden.issubset(mapper_holdout):
        raise ValueError(f"{fold_dir.name}: mapper strict holdout omits a forbidden route")
    if mapper["config"]["checkpoint_protocol"] != "fixed_epoch_no_selection":
        raise ValueError(f"{fold_dir.name}: mapper uses checkpoint selection")
    if mapper["best_validation"] is not None:
        raise ValueError(f"{fold_dir.name}: mapper has a selected validation score")

    geometry = reports["geometry_head"]
    if int(geometry["train_count"]) != expected_mapping_count:
        raise ValueError(f"{fold_dir.name}: geometry train count mismatch")
    if int(geometry["eval_count"]) != int(fold["query_count"]):
        raise ValueError(f"{fold_dir.name}: geometry OOF diagnostic count mismatch")
    geometry_contract = geometry["production_contract"]
    if (
        geometry["checkpoint_protocol"] != "fixed_epoch_no_selection"
        or bool(geometry_contract["eval_manifest_used_for_gradient"])
        or bool(geometry_contract["eval_manifest_used_for_checkpoint_selection"])
        or int(geometry_contract["saved_checkpoint_epoch"]) != 30
    ):
        raise ValueError(f"{fold_dir.name}: geometry checkpoint protocol is not fixed")

    canonical = reports["canonical_field"]
    if (
        int(canonical["mapping_image_count"]) != expected_mapping_count
        or set(canonical["mapping_trajectory_ids"]) != mapping
    ):
        raise ValueError(f"{fold_dir.name}: canonical field mapping split mismatch")
    if forbidden & set(canonical["mapping_trajectory_ids"]):
        raise ValueError(f"{fold_dir.name}: canonical field contains a forbidden route")

    readout = reports["physical_readout"]
    if (
        int(readout["teacher_supervised_image_count"]) != expected_mapping_count
        or int(readout["partition_counts"]["selection"]) != 0
        or int(readout["partition_counts"]["validation"]) != 0
        or readout["selected_validation"] is not None
    ):
        raise ValueError(f"{fold_dir.name}: physical readout split or selection mismatch")
    if str(readout["teacher_supervised_image_ids_sha256"]) != str(
        fold["mapping_image_ids_sha256"]
    ):
        raise ValueError(f"{fold_dir.name}: teacher supervision image hash mismatch")

    typed = reports["typed_graph"]
    if int(typed["metadata"]["mapping_view_count"]) != expected_mapping_count:
        raise ValueError(f"{fold_dir.name}: typed graph mapping count mismatch")
    mapping_view = reports["mapping_view_graph"]
    mapping_view_artifact = MappingViewGraph.load_npz(
        fold_dir / "mapping_view_graph.npz"
    )
    source_view_count = int(
        mapping_view_artifact.metadata["source_contributor_count"]
    )
    retained_view_count = int(mapping_view["view_node_count"])
    # A source view with zero incidence to the declared physical map has no
    # legal graph edge and is deliberately omitted.  Audit the pre-filter
    # contributor count (the split contract), not equality of retained nodes.
    if (
        source_view_count != expected_mapping_count
        or retained_view_count > source_view_count
        or int(mapping_view_artifact.poses_w2c.shape[0]) != retained_view_count
        or set(mapping_view["excluded_trajectory_ids"]) != forbidden
    ):
        raise ValueError(f"{fold_dir.name}: mapping-view exclusion mismatch")

    validity = reports["validity"]
    validity_metadata = validity["calibration_metadata"]
    validity_ids = list(validity_metadata["fit_image_ids"])
    if (
        int(validity["image_count"]) != expected_mapping_count
        or _trajectory_set(validity_ids) != mapping
        or str(validity_metadata.get("validity_target_algorithm"))
        != "exact_owned_contributor_mass_integral_image_v1"
    ):
        raise ValueError(f"{fold_dir.name}: validity fit split or target mismatch")

    canonical_hash = str(canonical["canonical_field_sha256"])
    if any(str(value) != canonical_hash for value in (
        typed["canonical_field_sha256"], mapping_view["canonical_field_sha256"],
        validity_metadata["canonical_field_sha256"],
    )):
        raise ValueError(f"{fold_dir.name}: canonical field lineage mismatch")
    strict_audit_path = fold_dir / "strict_map_audit.json"
    strict_confirmation = None
    if strict_audit_path.is_file():
        strict_confirmation = json.loads(strict_audit_path.read_text())
        contributor_audit = json.loads(
            (fold_dir / "contributors_official_train_audit.json").read_text()
        )
        physical = GoalMapletPhysicalMap.load_npz(fold_dir / "physical_map.npz")
        if (
            strict_confirmation.get("artifact_type")
            != "goal_maplet_strict_map_fold_audit_v1"
            or str(strict_confirmation["fold_id"]) != str(fold["fold_id"])
            or int(strict_confirmation["mapping_image_count"]) != expected_mapping_count
            or not bool(strict_confirmation["route_clean_end_to_end"])
            or not bool(contributor_audit["pass"])
            or contributor_audit["geometry_source_sha256"]
            != [strict_confirmation["gaussian_ply_sha256"]]
            or str(canonical["physical_map_sha256"]) != physical.content_sha256
        ):
            raise ValueError(f"{fold_dir.name}: strict geometry lineage mismatch")
    return {
        "fold_id": str(fold["fold_id"]),
        "mapping_trajectories": sorted(mapping),
        "held_query_trajectories": sorted(held),
        "mapping_image_count": expected_mapping_count,
        "mapper_supervised_image_count": len(mapper_split["training_images"]),
        "mapper_missing_radio_token_count": (
            expected_mapping_count - len(mapper_split["training_images"])
        ),
        "geometry_train_count": int(geometry["train_count"]),
        "geometry_eval_diagnostic_count": int(geometry["eval_count"]),
        "canonical_primitive_count": int(canonical["canonical_primitive_count"]),
        "canonical_primitive_coverage_fraction": float(
            canonical["primitive_coverage_fraction"]
        ),
        "mapping_view_source_count": source_view_count,
        "mapping_view_count": retained_view_count,
        "mapping_view_zero_incidence_omitted_count": (
            source_view_count - retained_view_count
        ),
        "validity_fit_image_count": int(validity["image_count"]),
        "fixed_checkpoint_without_held_selection": True,
        "held_and_official_test_routes_absent_from_map_fit": True,
        "strict_geometry_confirmation": strict_confirmation is not None,
        "strict_geometry_audit_sha256": (
            file_sha256(strict_audit_path) if strict_confirmation is not None else None
        ),
        "strict_geometry_chart_count": (
            int(strict_confirmation["chart_count"])
            if strict_confirmation is not None else None
        ),
        "strict_geometry_gaussian_iterations": (
            int(strict_confirmation["gaussian_iterations"])
            if strict_confirmation is not None else None
        ),
        "strict_geometry_matcha_commit": (
            str(strict_confirmation["matcha_commit"])
            if strict_confirmation is not None else None
        ),
        "strict_geometry_source_patch_sha256": (
            str(strict_confirmation["matcha_source_patch_sha256"])
            if strict_confirmation is not None else None
        ),
        "summary_sha256": {
            name: file_sha256(fold_dir / relative)
            for name, relative in SUMMARY_PATHS.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite OOF map audit: {output}")
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    fold_root = Path(args.fold_root)
    official_test_routes = set(protocol["official_test"]["trajectory_counts"])
    folds = [
        audit_fold(
            fold, fold_root / str(fold["fold_id"]),
            official_test_routes=official_test_routes,
        )
        for fold in protocol["development"]["folds"]
    ]
    all_strict = all(bool(value["strict_geometry_confirmation"]) for value in folds)
    strict_geometry_configuration = None
    if all_strict:
        chart_counts = {int(value["strict_geometry_chart_count"]) for value in folds}
        iterations = {
            int(value["strict_geometry_gaussian_iterations"]) for value in folds
        }
        commits = {str(value["strict_geometry_matcha_commit"]) for value in folds}
        patches = {
            str(value["strict_geometry_source_patch_sha256"]) for value in folds
        }
        if (
            len(chart_counts) != 1 or len(iterations) != 1
            or len(commits) != 1 or len(patches) != 1
        ):
            raise ValueError("strict geometry configuration differs between folds")
        strict_geometry_configuration = {
            "chart_count": next(iter(chart_counts)),
            "gaussian_iterations": next(iter(iterations)),
            "matcha_commit": next(iter(commits)),
            "source_patch_sha256": next(iter(patches)),
        }
    result = {
        "artifact_type": "goal_maplet_official_oof_map_audit_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "fold_root": str(fold_root),
        "fold_count": len(folds),
        "all_folds_fixed_checkpoint_without_held_selection": True,
        "all_folds_exclude_held_and_official_test_routes": True,
        "all_folds_strict_geometry_confirmation": all_strict,
        "strict_geometry_configuration": strict_geometry_configuration,
        "folds": folds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
