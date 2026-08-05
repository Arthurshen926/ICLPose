"""Evaluate hierarchical parent-context -> child-local retrieval on exact 2DGS labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.child_retrieval import ChildTilePosterior, retrieve_children_given_parents
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.local_head import load_child_local_head
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    contributor_multiscale_child_distribution,
    contributor_multiscale_maplet_distribution,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import ValidityCalibration, retrieve_maplet_posterior
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def _evaluate_child(
    posterior: ChildTilePosterior,
    truth: np.ndarray,
    truth_null: np.ndarray,
    physical: GoalMapletPhysicalMap,
    support_weight: np.ndarray,
) -> dict[str, float]:
    weight = np.asarray(support_weight, dtype=np.float64).reshape(-1)
    recalls = {k: [] for k in (1, 5, 20, 64)}
    reciprocal_rank, center_distance, parent_correct = [], [], []
    for support in range(truth.shape[0]):
        rows = posterior.candidate_child_rows[support]
        relevance = np.asarray([truth[support, row] if row >= 0 else 0.0 for row in rows], dtype=np.float64)
        positive_mass = max(float(np.sum(truth[support])), 1e-12)
        for k in recalls:
            recalls[k].append(float(np.sum(relevance[:k]) / positive_mass))
        positive = np.flatnonzero(relevance > 1e-8)
        reciprocal_rank.append(0.0 if positive.size == 0 else 1.0 / float(positive[0] + 1))
        gt_rows = np.flatnonzero(truth[support] > 0.0)
        if rows.size and rows[0] >= 0 and gt_rows.size:
            distance = np.linalg.norm(physical.child_centers[gt_rows] - physical.child_centers[int(rows[0])], axis=1)
            center_distance.append(float(np.sum(distance * truth[support, gt_rows]) / positive_mass))
            top_parent = int(physical.child_parent_rows[int(rows[0])])
            parent_mass = np.sum(truth[support, physical.child_parent_rows == top_parent])
            parent_correct.append(float(parent_mass / positive_mass))
        else:
            center_distance.append(np.nan)
            parent_correct.append(0.0)
    confidence = 1.0 - posterior.null_probabilities
    valid_target = 1.0 - np.asarray(truth_null, dtype=np.float64)
    bins = np.minimum((confidence * 10).astype(np.int64), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            selected_weight = weight[selected]
            ece += float(np.sum(selected_weight) / np.sum(weight)) * abs(
                float(np.average(confidence[selected], weights=selected_weight))
                - float(np.average(valid_target[selected], weights=selected_weight))
            )
    valid_distance = np.isfinite(center_distance)
    return {
        **{f"weighted_recall_at_{k}": _weighted_mean(np.asarray(value), weight) for k, value in recalls.items()},
        "mrr": _weighted_mean(np.asarray(reciprocal_rank), weight),
        "top1_expected_child_center_distance_m": (
            _weighted_mean(np.asarray(center_distance)[valid_distance], weight[valid_distance]) if np.any(valid_distance) else None
        ),
        "top1_parent_mass_fraction": _weighted_mean(np.asarray(parent_correct), weight),
        "null_ece": float(ece),
        "null_brier": _weighted_mean(np.square(confidence - valid_target), weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--child_local_head", default="")
    parser.add_argument("--parent_candidates", type=int, default=64)
    parser.add_argument("--child_candidates", type=int, default=64)
    parser.add_argument("--child_temperature", type=float, default=0.07)
    parser.add_argument("--grouping_cosine", type=float, default=0.90)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite child PFIR report")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout = readout_canonical_field(field, physical)
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    head_artifact = None
    child_descriptor_for_retrieval = readout.child_descriptors
    if str(args.child_local_head):
        head_artifact = load_child_local_head(Path(args.child_local_head), device=str(args.device))
        for name, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
        ):
            if str(head_artifact.metadata.get(name, "")) != str(expected):
                raise ValueError(f"child-local head {name} mismatch")
        with torch.no_grad():
            child_descriptor_for_retrieval = head_artifact.model.encode_map(
                torch.as_tensor(readout.child_descriptors, dtype=torch.float32, device=str(args.device))
            ).cpu().numpy()
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        paths = paths[: int(args.maximum_queries)]
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    reports = []
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            image_width, image_height = int(data["camera_width"]), int(data["camera_height"])
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        context_descriptor = encode_radio_final_regions(mapped, token_xy, context_config)
        local_descriptor = encode_radio_final_regions(mapped, token_xy, local_config)
        if head_artifact is not None:
            with torch.no_grad():
                local_descriptor = head_artifact.model.encode_query(
                    torch.as_tensor(local_descriptor, dtype=torch.float32, device=str(args.device))
                ).cpu().numpy()
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
            context_descriptor,
            readout.parent_descriptors,
            physical.maplet_ids,
            readout.parent_coverage > 0.0,
            maximum_candidates=int(args.parent_candidates),
            temperature=0.07,
            null_similarity_center=float(calibration.center),
            null_similarity_scale=float(calibration.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy,
            context_descriptor,
            parent_ids[:, 0],
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            image_width=image_width,
            image_height=image_height,
            descriptor_half_size_tokens=2.0,
            minimum_descriptor_cosine=float(args.grouping_cosine),
        )
        grouped_parent_ids, grouped_parent_probability, grouped_parent_null = aggregate_group_posteriors(
            parent_ids,
            parent_probability,
            parent_null,
            grouped.member_offsets,
            grouped.member_token_indices,
            maximum_candidates=int(args.parent_candidates),
        )
        grouped_local = aggregate_group_descriptors(
            local_descriptor, grouped.member_offsets, grouped.member_token_indices
        )
        support_weight = np.diff(grouped.member_offsets).astype(np.float64)
        actual = retrieve_children_given_parents(
            grouped_local,
            grouped_parent_ids,
            grouped_parent_probability,
            grouped_parent_null,
            child_descriptor_for_retrieval,
            readout.child_coverage,
            physical,
            maximum_child_candidates=int(args.child_candidates),
            temperature=float(args.child_temperature),
        )
        truth_child, truth_child_null = contributor_multiscale_child_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            group_member_offsets=grouped.member_offsets,
            group_member_token_indices=grouped.member_token_indices,
        )
        truth_parent, truth_parent_null = contributor_multiscale_maplet_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            pool_sizes=(1,),
            pool_weights=(1.0,),
            group_member_offsets=grouped.member_offsets,
            group_member_token_indices=grouped.member_token_indices,
        )
        oracle_parent_row = np.argmax(truth_parent, axis=1)
        oracle_parent_ids = physical.maplet_ids[oracle_parent_row, None]
        oracle_parent_probability = (1.0 - truth_parent_null)[:, None]
        oracle = retrieve_children_given_parents(
            grouped_local,
            oracle_parent_ids,
            oracle_parent_probability,
            truth_parent_null,
            child_descriptor_for_retrieval,
            readout.child_coverage,
            physical,
            maximum_child_candidates=int(args.child_candidates),
            temperature=float(args.child_temperature),
        )
        report = {
            "image_id": str(metadata["image_id"]),
            "support_count": int(grouped.member_offsets.size - 1),
            "support_effective_count": float(np.sum(support_weight)),
            "actual_parent_actual_child": _evaluate_child(actual, truth_child, truth_child_null, physical, support_weight),
            "oracle_parent_actual_child": _evaluate_child(oracle, truth_child, truth_child_null, physical, support_weight),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    modes = ("actual_parent_actual_child", "oracle_parent_actual_child")
    metric_names = list(reports[0][modes[0]]) if reports else []
    result = {
        "stage": "goal_maplet_hierarchical_child_pfir",
        "query_count": len(reports),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "validity_calibration_sha256": calibration.content_sha256,
        "one_canonical_stored_feature": True,
        "child_readout_regenerated": True,
        "child_local_head_file_sha256": (
            file_sha256(Path(args.child_local_head)) if head_artifact is not None else None
        ),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "summary": {
            mode: {
                metric: (
                    float(np.mean([row[mode][metric] for row in reports if row[mode].get(metric) is not None]))
                    if any(row[mode].get(metric) is not None for row in reports) else None
                )
                for metric in metric_names
            }
            for mode in modes
        },
        "rows": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
