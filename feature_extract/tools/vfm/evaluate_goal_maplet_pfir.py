"""Evaluate a retrieval bank with independent clean-2DGS PFIR labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_v6_retrieval_pose_basin import _region_geometry
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import SurfaceRetrievalMapletBank
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    QuerySupportPosterior,
    contributor_maplet_distribution,
    evaluate_pfir,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_v6.maplet_retrieval import retrieve_candidate_groups
from feature_extract.vfm.localization_v6.probability_calibration import V6ProbabilityCalibration
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--identity_bank", required=True)
    parser.add_argument("--spatial_bank", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--probability_calibration", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--maximum_maplets", type=int, default=64)
    parser.add_argument("--support_mode", choices=("balanced128",), default="balanced128")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _posterior_from_groups(groups, *, image_width: int, image_height: int) -> QuerySupportPosterior:
    count = len(groups)
    maximum = max((len(group.maplet_ids) for group in groups), default=0)
    ids = np.full((count, maximum), -1, dtype=np.int64)
    probability = np.zeros((count, maximum), dtype=np.float64)
    for row, group in enumerate(groups):
        size = len(group.maplet_ids)
        ids[row, :size] = group.maplet_ids
        probability[row, :size] = group.probabilities
    image_size = np.asarray([image_width, image_height], dtype=np.float64)
    return QuerySupportPosterior(
        xy=np.asarray([group.query_region_xy for group in groups], dtype=np.float64) / image_size,
        extent=np.asarray([group.query_region_extent for group in groups], dtype=np.float64) / image_size,
        candidate_maplet_ids=ids,
        candidate_probabilities=probability,
        null_probabilities=np.asarray([group.null_probability for group in groups]),
    )


def _mean_reports(rows: list[dict[str, object]]) -> dict[str, float | None]:
    names = (
        "weighted_recall_at_1",
        "weighted_recall_at_5",
        "weighted_recall_at_20",
        "weighted_recall_at_64",
        "multi_positive_ap",
        "ndcg",
        "mrr",
        "null_ece",
        "null_brier",
        "top1_expected_maplet_center_distance_m",
        "candidate_entropy",
    )
    output: dict[str, float | None] = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        output[name] = float(np.mean(values)) if values else None
    for k in (1, 5, 20, 64):
        output[f"whole_image_coverage_at_{k}"] = float(np.mean([
            float(row["scene"][f"coverage_at_{k}"]) for row in rows
        ])) if rows else 0.0
    output["pose_sufficient_at_64_fraction"] = float(np.mean([
        bool(row["scene"]["pose_sufficient_at_64"]) for row in rows
    ])) if rows else 0.0
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite PFIR output")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    identity = SurfaceRetrievalMapletBank.load_npz(Path(args.identity_bank))
    spatial = SurfaceRetrievalMapletBank.load_npz(Path(args.spatial_bank))
    mapper, mapper_metadata = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    calibration = V6ProbabilityCalibration.load_json(Path(args.probability_calibration))
    config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))),
        global_context_weight=float(mapper_metadata.get("global_context_weight", 0.0)),
    )
    paths = sorted(Path(args.contributors).glob("*.npz"))
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid PFIR shard")
    paths = paths[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0 and len(paths) > int(args.maximum_queries):
        indices = np.linspace(0, len(paths) - 1, int(args.maximum_queries), dtype=np.int64)
        paths = [paths[int(index)] for index in indices]
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            token_path = Path(str(metadata["token_path"]))
        with np.load(token_path, allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        spatial_mapped = spatial.project_query_feature_map(raw)
        _indices, token_xy = select_spatially_balanced_radio_final_regions(raw)
        descriptors = encode_radio_final_regions(mapped, token_xy, config)
        spatial_descriptors = encode_radio_final_regions(
            spatial_mapped,
            token_xy,
            RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,)),
        )
        labels = ContributorLabels.load_npz(path)
        with np.load(path, allow_pickle=False) as data:
            image_width, image_height = int(data["camera_width"]), int(data["camera_height"])
        region_xy, region_extent = _region_geometry(
            token_xy,
            token_width=int(raw.shape[2]),
            token_height=int(raw.shape[1]),
            image_width=image_width,
            image_height=image_height,
            config=config,
        )
        retrieval = retrieve_candidate_groups(
            descriptors,
            region_xy,
            region_extent,
            identity,
            preliminary_candidates=int(args.maximum_maplets),
            maximum_maplets=int(args.maximum_maplets),
            maximum_components_per_maplet=1,
            spatial_query_descriptors=spatial_descriptors,
            spatial_bank=spatial,
            probability_calibration=calibration,
            compute_spatial_modes=False,
        )
        posterior = _posterior_from_groups(
            retrieval.groups,
            image_width=image_width,
            image_height=image_height,
        )
        truth, truth_null = contributor_maplet_distribution(
            labels,
            physical,
            posterior.xy,
            posterior.extent,
        )
        report = evaluate_pfir(
            posterior,
            truth,
            truth_null,
            physical,
            labels.pose_w2c,
        )
        report["image_id"] = str(metadata["image_id"])
        rows.append(report)
        print(json.dumps({key: value for key, value in report.items() if key != "per_support"}), flush=True)
    result = {
        "stage": "goal_maplet_independent_pfir",
        "query_count": len(rows),
        "support_mode": str(args.support_mode),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "map_stores_one_vfm_feature_type": True,
        "uses_raster_contributor_multi_positive_gt": True,
        "uses_nearest_center_single_label_gt": False,
        "physical_map_sha256": physical.content_sha256,
        **_mean_reports(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
