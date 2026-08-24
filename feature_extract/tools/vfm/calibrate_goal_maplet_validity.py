"""Fit Goal-Maplet score-to-validity calibration on an independent trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pfir import contributor_multiscale_in_map_probability
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.multimodal_parent_retrieval import (
    PARENT_SCORE_ANONYMOUS_MODES,
    PARENT_SCORE_SINGLE_MEAN,
    build_anonymous_parent_mode_readout,
    score_anonymous_parent_modes,
)
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.query_support import all_token_coordinates
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    fit_validity_calibration,
    retrieve_maplet_posterior,
    retrieve_maplet_posterior_from_scores,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
    LEGACY_COORDINATE_CONTRACT,
    load_contributors_in_radio_coordinates,
)
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


POOLING = {
    "mapped_1x1": ((1,), (1.0,)),
    "mapped_1x1_3x3": ((1, 3), (0.65, 0.35)),
    "mapped_1x1_3x3_5x5": ((1, 3, 5), (0.50, 0.30, 0.20)),
    "current_1x1_3x3_5x5_9x9": ((1, 3, 5, 9), (0.40, 0.30, 0.20, 0.10)),
}


def _contributor_filename_identity(path: Path) -> tuple[str, str]:
    name = Path(path).name
    if not name.endswith(".npz"):
        raise ValueError(f"contributor is not an NPZ: {path}")
    fields = name[:-4].split("__", 1)
    if (
        len(fields) != 2
        or not fields[0]
        or not fields[1]
        or "/" in fields[0]
        or "\\" in fields[0]
    ):
        raise ValueError(f"contributor filename lacks route identity: {path}")
    return fields[0], f"{fields[0]}/{fields[1]}"


def _calibration_split_audit(
    calibration_routes: set[str],
    field_metadata: Mapping[str, object],
    mapper_metadata: Mapping[str, object],
) -> dict[str, object]:
    mapping_routes = {
        str(value) for value in field_metadata.get("mapping_trajectory_ids", ())
    }
    field_excluded = {
        str(value) for value in field_metadata.get("excluded_trajectory_ids", ())
    }
    mapper_fit = {
        str(value) for value in mapper_metadata.get("training_trajectory_ids", ())
    }
    mapper_validation = {
        str(value) for value in mapper_metadata.get("validation_trajectory_ids", ())
    }
    mapper_holdout = {
        str(value)
        for value in mapper_metadata.get("strict_holdout_trajectory_ids", ())
    }
    blockers: list[str] = []
    if not calibration_routes:
        blockers.append("calibration_routes_not_explicit")
    if calibration_routes & mapping_routes:
        blockers.append("calibration_route_present_in_canonical_fusion")
    if not calibration_routes.issubset(field_excluded):
        blockers.append("canonical_field_does_not_declare_calibration_route_exclusion")
    if field_metadata.get(
        "route_exclusion_applied_before_opening_contributor_archives"
    ) is not True:
        blockers.append("canonical_route_exclusion_not_applied_before_archive_open")
    if calibration_routes & (mapper_fit | mapper_validation):
        blockers.append("calibration_route_used_for_mapper_fit_or_selection")
    if not calibration_routes.issubset(mapper_holdout):
        blockers.append("mapper_does_not_declare_calibration_route_holdout")
    return {
        "disjoint": not blockers,
        "calibration_trajectory_ids": sorted(calibration_routes),
        "canonical_mapping_trajectory_ids": sorted(mapping_routes),
        "canonical_excluded_trajectory_ids": sorted(field_excluded),
        "mapper_training_trajectory_ids": sorted(mapper_fit),
        "mapper_validation_trajectory_ids": sorted(mapper_validation),
        "mapper_strict_holdout_trajectory_ids": sorted(mapper_holdout),
        "blockers": blockers,
    }


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probability >= low) & (probability < high if index + 1 < bins else probability <= high)
        if np.any(selected):
            value += float(np.mean(selected)) * abs(float(np.mean(probability[selected]) - np.mean(target[selected])))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--physical_instance_readout", default="")
    parser.add_argument("--pooling", choices=tuple(POOLING), required=True)
    parser.add_argument(
        "--parent_score_semantics",
        choices=(PARENT_SCORE_SINGLE_MEAN, PARENT_SCORE_ANONYMOUS_MODES),
        default=PARENT_SCORE_SINGLE_MEAN,
    )
    parser.add_argument("--parent_mode_temperature", type=float, default=0.03)
    parser.add_argument("--output_calibration", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include_trajectories", nargs="+", default=[])
    parser.add_argument(
        "--allow_calibration_overlap_control",
        action="store_true",
        help="explicitly allow a non-disjoint calibration as a control-only artifact",
    )
    parser.add_argument(
        "--legacy_pinhole_as_raw_diagnostic",
        action="store_true",
        help="reproduce the old coordinate-misaligned validity target as a diagnostic control",
    )
    parser.add_argument(
        "--allow_unpromoted_mapper_control",
        action="store_true",
        help=(
            "explicitly allow a canonical field whose mapper-supervision "
            "coordinate lineage is unverified; all outputs remain control-only"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_calibration), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite Goal-Maplet calibration")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    field_promotion_eligible = bool(
        field.metadata.get("promotion_eligible", False)
    )
    field_promotion_blockers = list(
        field.metadata.get("promotion_blockers", [])
    )
    if (
        not field_promotion_eligible
        and not bool(args.allow_unpromoted_mapper_control)
    ):
        raise ValueError(
            "validity calibration refuses an unpromoted canonical field; "
            "pass --allow_unpromoted_mapper_control for a diagnostic control"
        )
    readout = readout_canonical_field(field, physical)
    instance_readout = None
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
    valid_maplets = readout.parent_coverage > 0.0
    anonymous_parent_readout = (
        build_anonymous_parent_mode_readout(field, physical)
        if str(args.parent_score_semantics) == PARENT_SCORE_ANONYMOUS_MODES
        else None
    )
    owned_primitive_rows = np.unique(
        np.asarray(physical.membership_primitive_rows, dtype=np.int64)
    )
    sorted_owned_primitive_ids = np.sort(
        np.asarray(physical.primitive_ids, dtype=np.int64)[owned_primitive_rows]
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device)
    )
    pool_sizes, pool_weights = POOLING[str(args.pooling)]
    config = RadioFinalRegionConfig(pool_sizes=pool_sizes, pool_weights=pool_weights)
    all_score, all_target = [], []
    image_ids = []
    coordinate_audits: list[dict[str, object]] = []
    included = set(str(value) for value in args.include_trajectories)
    split_audit = _calibration_split_audit(included, field.metadata, mapper_metadata)
    if not bool(split_audit["disjoint"]) and not bool(
        args.allow_calibration_overlap_control
    ):
        raise ValueError(
            "validity calibration is not disjoint from mapper/canonical fitting: "
            + ",".join(str(value) for value in split_audit["blockers"])
        )
    contributor_paths = [
        path for path in sorted(Path(args.contributors).glob("*.npz"))
        if not included or _contributor_filename_identity(path)[0] in included
    ]
    fit_contributor_inventory: list[dict[str, str]] = []
    for path in contributor_paths:
        filename_trajectory, filename_image_id = _contributor_filename_identity(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if (
            str(metadata.get("trajectory_id", "")) != filename_trajectory
            or str(metadata.get("image_id", "")) != filename_image_id
        ):
            raise ValueError("contributor filename and embedded route identity differ")
        labels, coordinate_audit = load_contributors_in_radio_coordinates(
            path,
            legacy_pinhole_as_raw_diagnostic=bool(args.legacy_pinhole_as_raw_diagnostic),
        )
        coordinate_audits.append(coordinate_audit)
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        descriptor = (
            encode_radio_final_regions(mapped, token_xy, config)
            if instance_readout is None else encode_physical_instance_regions(
                instance_readout, mapped, token_xy, role="context", device=str(args.device),
            )
        )
        if anonymous_parent_readout is None:
            _, _, _, best = retrieve_maplet_posterior(
                descriptor,
                readout.parent_descriptors,
                physical.maplet_ids,
                valid_maplets,
                maximum_candidates=64,
                temperature=0.07,
                null_similarity_center=0.35,
                null_similarity_scale=0.08,
            )
        else:
            best = retrieve_maplet_posterior_from_scores(
                score_anonymous_parent_modes(
                    descriptor,
                    anonymous_parent_readout,
                    mode_temperature=float(args.parent_mode_temperature),
                ),
                physical.maplet_ids,
                anonymous_parent_readout.parent_coverage > 0.0,
                maximum_candidates=64,
                temperature=0.07,
                null_similarity_center=0.35,
                null_similarity_scale=0.08,
            ).best_similarities
        _, truth_null = contributor_multiscale_in_map_probability(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            pool_sizes=pool_sizes,
            pool_weights=pool_weights,
            sorted_owned_primitive_ids=sorted_owned_primitive_ids,
        )
        all_score.append(best.astype(np.float64))
        all_target.append(1.0 - truth_null.astype(np.float64))
        image_ids.append(str(metadata["image_id"]))
        fit_contributor_inventory.append({
            "image_id": str(metadata["image_id"]),
            "resolved_path": str(path.resolve()),
            "file_sha256": file_sha256(path),
        })
        print(json.dumps({"image_id": image_ids[-1], "support_count": int(best.size)}), flush=True)
    if not all_score:
        raise ValueError("no validity-calibration contributors were selected")
    score = np.concatenate(all_score)
    target = np.concatenate(all_target)
    calibration_promotion_eligible = bool(
        field_promotion_eligible
        and not args.legacy_pinhole_as_raw_diagnostic
        and split_audit["disjoint"]
    )
    calibration_control_reasons = []
    if not field_promotion_eligible:
        calibration_control_reasons.append("canonical_field_not_promotion_eligible")
    if args.legacy_pinhole_as_raw_diagnostic:
        calibration_control_reasons.append("legacy_pinhole_as_raw_coordinate_control")
    calibration_control_reasons.extend(
        str(value) for value in split_audit["blockers"]
    )
    calibration = fit_validity_calibration(
        score,
        target,
        metadata={
            "physical_map_sha256": physical.content_sha256,
            "canonical_field_sha256": field.content_sha256,
            "canonical_field_file_sha256": file_sha256(Path(args.canonical_field)),
            "surface_mapper_file_sha256": file_sha256(Path(args.surface_mapper)),
            "physical_instance_readout_sha256": (
                file_sha256(Path(args.physical_instance_readout))
                if args.physical_instance_readout else None
            ),
            "pooling": str(args.pooling),
            "parent_score_semantics": str(args.parent_score_semantics),
            "parent_mode_temperature": float(args.parent_mode_temperature),
            "anonymous_parent_mode_readout_sha256": (
                anonymous_parent_readout.content_sha256
                if anonymous_parent_readout is not None
                else None
            ),
            "fit_support_mode": "all_tokens_exact_multiscale_masks",
            "validity_target_algorithm": (
                "exact_owned_contributor_mass_integral_image_v1"
            ),
            "fit_image_ids": image_ids,
            "fit_trajectory_ids": sorted({value.split("/", 1)[0] for value in image_ids}),
            "fit_contributor_inventory_sha256": canonical_json_sha256(
                fit_contributor_inventory
            ),
            "calibration_split_audit": split_audit,
            "stores_scores_or_query_features": False,
            "contributor_to_radio_coordinate_contract": (
                LEGACY_COORDINATE_CONTRACT
                if args.legacy_pinhole_as_raw_diagnostic
                else COORDINATE_CONTRACT
            ),
            "coordinate_correct": bool(not args.legacy_pinhole_as_raw_diagnostic),
            "canonical_field_promotion_eligible": field_promotion_eligible,
            "canonical_field_promotion_blockers": field_promotion_blockers,
            "promotion_eligible": calibration_promotion_eligible,
            "promotion_blockers": calibration_control_reasons,
            "control_only": bool(not calibration_promotion_eligible),
        },
    )
    calibrated = calibration.predict_valid(score)
    uncalibrated = 1.0 / (1.0 + np.exp(-(score - 0.35) / 0.08))
    report = {
        "stage": "calibrate_goal_maplet_validity",
        "support_count": int(score.size),
        "image_count": len(image_ids),
        "center": float(calibration.center),
        "scale": float(calibration.scale),
        "target_valid_mean": float(np.mean(target)),
        "uncalibrated": {
            "predicted_valid_mean": float(np.mean(uncalibrated)),
            "ece": _ece(uncalibrated, target),
            "brier": float(np.mean(np.square(uncalibrated - target))),
        },
        "calibrated": {
            "predicted_valid_mean": float(np.mean(calibrated)),
            "ece": _ece(calibrated, target),
            "brier": float(np.mean(np.square(calibrated - target))),
        },
        "calibration_sha256": calibration.content_sha256,
        "calibration_metadata": dict(calibration.metadata),
        "promotion_eligible": calibration_promotion_eligible,
        "promotion_blockers": calibration_control_reasons,
        "control_only": bool(not calibration_promotion_eligible),
        "calibration_split_audit": split_audit,
        "fit_contributor_inventory_sha256": canonical_json_sha256(
            fit_contributor_inventory
        ),
        "coordinate_audit": {
            "coordinate_contract": (
                LEGACY_COORDINATE_CONTRACT
                if args.legacy_pinhole_as_raw_diagnostic
                else COORDINATE_CONTRACT
            ),
            "view_count": int(len(coordinate_audits)),
            "camera_model_ids": sorted({int(value["camera_model_id"]) for value in coordinate_audits}),
            "minimum_valid_raw_sample_fraction": (
                float(min(float(value["valid_raw_sample_fraction"]) for value in coordinate_audits))
                if coordinate_audits and not args.legacy_pinhole_as_raw_diagnostic else None
            ),
        },
    }
    calibration.save_json(output)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
