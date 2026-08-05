"""PFIR ablations for one canonical surface field and regenerable readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    QuerySupportPosterior,
    contributor_multiscale_maplet_distribution,
    evaluate_pfir,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    ValidityCalibration,
    retrieve_maplet_posterior,
)
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


POOLING = {
    "mapped_1x1": ((1,), (1.0,)),
    "mapped_1x1_3x3": ((1, 3), (0.65, 0.35)),
    "mapped_1x1_3x3_5x5": ((1, 3, 5), (0.50, 0.30, 0.20)),
    "current_1x1_3x3_5x5_9x9": ((1, 3, 5, 9), (0.40, 0.30, 0.20, 0.10)),
}


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--pooling", choices=tuple(POOLING), required=True)
    parser.add_argument("--support_mode", choices=("balanced128", "all_tokens", "all_grouped"), required=True)
    parser.add_argument("--maximum_candidates", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--null_similarity_center", type=float, default=0.35)
    parser.add_argument("--null_similarity_scale", type=float, default=0.08)
    parser.add_argument("--validity_calibration", default="")
    parser.add_argument("--grouping_cosine", type=float, default=0.90)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite canonical PFIR output")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    feature_contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    if feature_contract.query_readout_type != "surface_maplet_mapper":
        raise ValueError("canonical PFIR evaluator requires the retrieval mapper readout")
    feature_contract.validate(field, query_readout_path=Path(args.surface_mapper))
    readout = readout_canonical_field(field, physical)
    calibration = None
    if str(args.validity_calibration):
        calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
        for name, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
            ("pooling", str(args.pooling)),
        ):
            actual = str(calibration.metadata.get(name, ""))
            if actual != str(expected):
                raise ValueError(f"validity calibration {name} mismatch: {actual} != {expected}")
    null_center = float(calibration.center) if calibration is not None else float(args.null_similarity_center)
    null_scale = float(calibration.scale) if calibration is not None else float(args.null_similarity_scale)
    valid_maplets = readout.parent_coverage > 0.0
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    pool_sizes, pool_weights = POOLING[str(args.pooling)]
    config = RadioFinalRegionConfig(pool_sizes=pool_sizes, pool_weights=pool_weights)
    paths = sorted(Path(args.contributors).glob("*.npz"))
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard")
    paths = paths[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        paths = paths[: int(args.maximum_queries)]
    reports = []
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            image_width, image_height = int(data["camera_width"]), int(data["camera_height"])
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        if str(args.support_mode) == "balanced128":
            _, token_xy = select_spatially_balanced_radio_final_regions(raw)
        else:
            _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        descriptor = encode_radio_final_regions(mapped, token_xy, config)
        candidate_ids, probability, null, best = retrieve_maplet_posterior(
            descriptor,
            readout.parent_descriptors,
            physical.maplet_ids,
            valid_maplets,
            maximum_candidates=int(args.maximum_candidates),
            temperature=float(args.temperature),
            null_similarity_center=null_center,
            null_similarity_scale=null_scale,
        )
        half_size = 0.5 * float(np.average(np.asarray(pool_sizes), weights=np.asarray(pool_weights)))
        if str(args.support_mode) == "all_grouped":
            grouped = group_tokens_after_retrieval(
                token_xy,
                descriptor,
                candidate_ids[:, 0],
                token_height=int(raw.shape[1]),
                token_width=int(raw.shape[2]),
                image_width=image_width,
                image_height=image_height,
                descriptor_half_size_tokens=half_size,
                minimum_descriptor_cosine=float(args.grouping_cosine),
            )
            support_xy, support_extent = grouped.xy, grouped.extent
            support_weight = np.diff(grouped.member_offsets).astype(np.float32)
            candidate_ids, probability, null = aggregate_group_posteriors(
                candidate_ids,
                probability,
                null,
                grouped.member_offsets,
                grouped.member_token_indices,
                maximum_candidates=int(args.maximum_candidates),
            )
            best = np.asarray([
                np.mean(best[grouped.member_token_indices[
                    int(grouped.member_offsets[row]) : int(grouped.member_offsets[row + 1])
                ]])
                for row in range(grouped.member_offsets.size - 1)
            ], dtype=np.float32)
            truth, truth_null = contributor_multiscale_maplet_distribution(
                labels,
                physical,
                token_xy,
                token_height=int(raw.shape[1]),
                token_width=int(raw.shape[2]),
                pool_sizes=pool_sizes,
                pool_weights=pool_weights,
                group_member_offsets=grouped.member_offsets,
                group_member_token_indices=grouped.member_token_indices,
            )
        else:
            support_xy = (token_xy + 0.5) / np.asarray([raw.shape[2], raw.shape[1]], dtype=np.float32)
            support_extent = np.broadcast_to(
                np.asarray([half_size / raw.shape[2], half_size / raw.shape[1]], dtype=np.float32),
                support_xy.shape,
            ).copy()
            support_weight = np.ones((support_xy.shape[0],), dtype=np.float32)
            truth, truth_null = contributor_multiscale_maplet_distribution(
                labels,
                physical,
                token_xy,
                token_height=int(raw.shape[1]),
                token_width=int(raw.shape[2]),
                pool_sizes=pool_sizes,
                pool_weights=pool_weights,
            )
        posterior = QuerySupportPosterior(
            support_xy,
            support_extent,
            candidate_ids,
            probability,
            null,
            support_weight,
        )
        report = evaluate_pfir(posterior, truth, truth_null, physical, labels.pose_w2c)
        report.pop("per_support", None)
        report.update({
            "image_id": str(metadata["image_id"]),
            "mean_best_cosine": float(np.mean(best)),
            "mean_predicted_null": float(np.mean(null)),
        })
        reports.append(report)
        print(json.dumps(report), flush=True)
    metrics = (
        "weighted_recall_at_1", "weighted_recall_at_5", "weighted_recall_at_20", "weighted_recall_at_64",
        "multi_positive_ap", "ndcg", "mrr", "null_ece", "null_brier",
        "top1_expected_maplet_center_distance_m", "candidate_entropy", "mean_best_cosine", "mean_predicted_null",
    )
    result = {
        "stage": "goal_maplet_canonical_field_pfir_ablation",
        "query_count": len(reports),
        "pooling": str(args.pooling),
        "support_mode": str(args.support_mode),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "field_feature_contract_sha256": feature_contract.content_sha256,
        "stored_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "validity_calibration_sha256": calibration.content_sha256 if calibration is not None else None,
        "null_similarity_center": null_center,
        "null_similarity_scale": null_scale,
        **{name: _mean(reports, name) for name in metrics},
        **{f"whole_image_coverage_at_{k}": float(np.mean([row["scene"][f"coverage_at_{k}"] for row in reports])) if reports else 0.0 for k in (1, 5, 20, 64)},
        "pose_sufficient_at_64_fraction": float(np.mean([row["scene"]["pose_sufficient_at_64"] for row in reports])) if reports else 0.0,
        "rows": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
