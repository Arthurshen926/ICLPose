"""Fit query-OOF relation calibration and audit frozen hypothesis ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    concatenate_hypothesis_shard_field,
    load_inference_artifact_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.calibrated_relation_likelihood import (
    CalibratedRelationLikelihood,
    fit_relation_density_ratio,
)
from feature_extract.vfm.localization.candidate_relation_features import (
    RELATION_CHANNELS,
)


STAGE = "query_oof_calibrated_relation_hypothesis_audit_v1"
HISTOGRAM_FIELD = "verification_relation_feature_histograms"
EDGE_HISTOGRAM_FIELD = "verification_relation_feature_edge_histograms"
EDGE_COUNT_FIELD = "verification_relation_feature_edge_counts"
NULL_MASS_FIELD = "verification_relation_feature_null_touching_mass_means"
EDGE_NULL_MASS_FIELD = "verification_relation_feature_edge_null_touching_masses"
UNARY_FIELD = "verification_log_likelihood_means"


def _query_fold(query_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(query_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(folds)


def _relation_family_channels(relation_family: str) -> tuple[int, ...]:
    predicates = {
        "all": lambda name: True,
        "maplet": lambda name: name == "maplet_only",
        "shared_support_view": lambda name: name.startswith("support_overlap_"),
        "sfm_covisible_neighbor": lambda name: "neighbor_rank_" in name,
        "topology": lambda name: (
            name == "maplet_only"
            or name.startswith("support_overlap_")
            or "neighbor_rank_" in name
        ),
        "hard_repeat": lambda name: name.startswith("repeat_"),
    }
    if relation_family not in predicates:
        raise ValueError(f"unsupported relation family: {relation_family}")
    return tuple(
        index
        for index, name in enumerate(RELATION_CHANNELS)
        if predicates[relation_family](name)
    )


def _load_merged_artifacts(
    paths: Sequence[Path], channel_indices: Sequence[int]
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    base_fields = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "shortlisted_for_verification",
        UNARY_FIELD,
        HISTOGRAM_FIELD,
        EDGE_COUNT_FIELD,
        NULL_MASS_FIELD,
    }
    base_shards: list[dict[str, np.ndarray]] = []
    metadata_items: list[dict[str, object]] = []
    row_offsets = [0]
    max_edges = 0
    for path in paths:
        arrays, metadata = load_inference_artifact_fields(path, tuple(base_fields))
        arrays = {key: value for key, value in arrays.items() if key in base_fields}
        arrays[HISTOGRAM_FIELD] = np.asarray(
            arrays[HISTOGRAM_FIELD], dtype=np.float32
        )
        arrays[NULL_MASS_FIELD] = np.asarray(
            arrays[NULL_MASS_FIELD], dtype=np.float32
        )
        count = len(arrays["query_ids"])
        row_offsets.append(row_offsets[-1] + count)
        edge_counts = np.asarray(arrays[EDGE_COUNT_FIELD], dtype=np.int64)
        max_edges = max(max_edges, int(np.max(edge_counts, initial=0)))
        base_shards.append(arrays)
        metadata_items.append(metadata)
    keys = set(base_shards[0])
    if any(set(arrays) != keys for arrays in base_shards):
        raise ValueError("hypothesis artifacts expose different schemas")
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "shortlisted_for_verification",
        UNARY_FIELD,
        HISTOGRAM_FIELD,
        EDGE_COUNT_FIELD,
        NULL_MASS_FIELD,
    }
    if not required.issubset(keys):
        raise ValueError(
            f"hypothesis artifacts lack calibrated relation fields: {sorted(required - keys)}"
        )
    merged = {
        key: np.concatenate([arrays[key] for arrays in base_shards], axis=0)
        for key in sorted(keys)
    }
    del base_shards

    bins = _bin_edges(metadata_items[0])
    selected_width = len(channel_indices) * (len(bins) - 1)
    total_rows = row_offsets[-1]
    edge_histogram = np.full(
        (total_rows, max_edges, selected_width), np.nan, dtype=np.float32
    )
    edge_null = np.full((total_rows, max_edges), np.nan, dtype=np.float32)
    channel_rows = np.asarray(channel_indices, dtype=np.int64)
    for shard_index, path in enumerate(paths):
        start, stop = row_offsets[shard_index : shard_index + 2]
        with np.load(path, allow_pickle=False) as payload:
            source_histogram = np.asarray(
                payload[EDGE_HISTOGRAM_FIELD], dtype=np.float32
            )
            source_null = np.asarray(payload[EDGE_NULL_MASS_FIELD], dtype=np.float32)
        if source_histogram.shape[:2] != source_null.shape or source_histogram.shape[0] != stop - start:
            raise ValueError(f"{path}: per-edge relation arrays are not row-aligned")
        source_histogram = source_histogram.reshape(
            source_histogram.shape[0],
            source_histogram.shape[1],
            len(RELATION_CHANNELS),
            len(bins) - 1,
        )
        width = source_histogram.shape[1]
        edge_histogram[start:stop, :width] = source_histogram[
            :, :, channel_rows, :
        ].reshape(stop - start, width, selected_width)
        edge_null[start:stop, :width] = source_null
    merged[EDGE_HISTOGRAM_FIELD] = edge_histogram
    merged[EDGE_NULL_MASS_FIELD] = edge_null
    return merged, metadata_items


def _validate_targets(
    path: Path,
    *,
    artifact_paths: Sequence[Path],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        target = {key: np.asarray(payload[key]).copy() for key in payload.files}
    metadata = json.loads(str(target["metadata_json"].item()))
    expected_hashes = [file_sha256_short(item) for item in artifact_paths]
    if metadata.get("inference_artifact_sha256") != expected_hashes:
        raise ValueError("target artifact references different hypothesis artifacts")
    for key in ("query_ids", "split_names", "evaluation_labels", "hypothesis_indices"):
        if not np.array_equal(np.asarray(target[key]).astype(str), np.asarray(arrays[key]).astype(str)):
            raise ValueError(f"target and inference rows differ in {key}")
    return target


def _bin_edges(metadata: Mapping[str, object]) -> np.ndarray:
    config = metadata.get("grouped_config")
    if not isinstance(config, Mapping):
        raise ValueError("hypothesis artifact lacks grouped config")
    bins = np.asarray(
        config.get("candidate_relation_feature_bin_edges_px", ()), dtype=np.float64
    )
    if len(bins) < 2:
        raise ValueError("hypothesis artifact lacks relation histogram bins")
    return bins


def _eligible(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    histogram = np.asarray(arrays[HISTOGRAM_FIELD], dtype=np.float64)
    edge_count = np.asarray(arrays[EDGE_COUNT_FIELD], dtype=np.int64)
    unary = np.asarray(arrays[UNARY_FIELD], dtype=np.float64)
    shortlisted = np.asarray(arrays["shortlisted_for_verification"], dtype=bool)
    edge_histogram = np.asarray(arrays[EDGE_HISTOGRAM_FIELD], dtype=np.float64)
    edge_null = np.asarray(arrays[EDGE_NULL_MASS_FIELD], dtype=np.float64)
    if edge_histogram.ndim != 3 or edge_null.shape != edge_histogram.shape[:2]:
        raise ValueError("per-edge relation arrays are not aligned")
    row_valid = shortlisted & (edge_count > 0) & np.isfinite(unary) & np.all(
        np.isfinite(histogram), axis=1
    )
    for row_index in np.flatnonzero(row_valid):
        count = int(edge_count[row_index])
        row_valid[row_index] = count <= edge_histogram.shape[1] and np.all(
            np.isfinite(edge_histogram[row_index, :count])
        ) and np.all(np.isfinite(edge_null[row_index, :count]))
    return row_valid


def _fit_model(
    arrays: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    bins: np.ndarray,
    positive_threshold_m: float,
    negative_threshold_m: float,
    source_manifest: Mapping[str, object],
    relation_family: str = "all",
) -> CalibratedRelationLikelihood:
    histogram = np.asarray(arrays[HISTOGRAM_FIELD], dtype=np.float64)
    histogram = histogram.reshape(len(histogram), len(RELATION_CHANNELS), len(bins) - 1)
    translation = np.asarray(target["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(target["rotation_errors_deg"], dtype=np.float64)
    positive = mask & (translation <= float(positive_threshold_m)) & (rotation <= 5.0)
    negative = mask & (translation >= float(negative_threshold_m))
    if np.count_nonzero(positive) < 4 or np.count_nonzero(negative) < 4:
        raise ValueError("relation calibration lacks positive or hard-negative hypotheses")
    model = fit_relation_density_ratio(
        np.sum(histogram[positive], axis=0),
        np.sum(histogram[negative], axis=0),
        bin_edges_px=bins,
        source_manifest={
            **dict(source_manifest),
            "positive_hypothesis_count": int(np.count_nonzero(positive)),
            "negative_hypothesis_count": int(np.count_nonzero(negative)),
        },
    )
    keep = set(_relation_family_channels(relation_family))
    log_ratio = np.asarray(model.log_density_ratio).copy()
    for channel in range(len(RELATION_CHANNELS)):
        if channel not in keep:
            log_ratio[channel] = 0.0
    return CalibratedRelationLikelihood(
        model.bin_edges_px,
        log_ratio,
        {**dict(model.source_manifest), "relation_family": relation_family},
    )


def _relation_scores(
    arrays: Mapping[str, np.ndarray],
    model: CalibratedRelationLikelihood,
    channel_indices: Sequence[int],
) -> np.ndarray:
    edge_histogram = np.asarray(
        arrays[EDGE_HISTOGRAM_FIELD], dtype=np.float64
    ).reshape(
        len(arrays[EDGE_HISTOGRAM_FIELD]),
        -1,
        len(channel_indices),
        len(model.bin_edges_px) - 1,
    )
    edge_count = np.asarray(arrays[EDGE_COUNT_FIELD], dtype=np.int64)
    edge_null = np.asarray(arrays[EDGE_NULL_MASS_FIELD], dtype=np.float64)
    if edge_null.shape != edge_histogram.shape[:2]:
        raise ValueError("per-edge relation null mass is not aligned")
    score = np.full((len(edge_histogram),), np.nan, dtype=np.float64)
    likelihood_ratio = np.exp(
        model.log_density_ratio[np.asarray(channel_indices, dtype=np.int64)]
    )
    for row_index, count in enumerate(edge_count.tolist()):
        if count <= 0 or count > edge_histogram.shape[1]:
            continue
        histogram = edge_histogram[row_index, :count]
        null_mass = edge_null[row_index, :count]
        if not np.all(np.isfinite(histogram)) or not np.all(np.isfinite(null_mass)):
            continue
        candidate_ratio = np.sum(
            histogram * likelihood_ratio[None, :, :], axis=(1, 2)
        )
        edge_ratio = null_mass + candidate_ratio
        score[row_index] = float(np.mean(np.log(np.maximum(edge_ratio, 1e-12))))
    return score


def summarize_selector(
    arrays: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    score: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    queries = np.asarray(arrays["query_ids"]).astype(str)
    translation = np.asarray(target["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(target["rotation_errors_deg"], dtype=np.float64)
    selected: list[int] = []
    oracle: list[int] = []
    best10_ranks: list[int] = []
    selected_ranks: list[int] = []
    topk_hits = {1: 0, 5: 0, 10: 0}
    query_ids = sorted(set(queries[mask].tolist()))
    for query_id in query_ids:
        indices = np.flatnonzero(mask & (queries == query_id) & np.isfinite(score))
        if len(indices) == 0:
            continue
        order = indices[np.argsort(-score[indices], kind="mergesort")]
        chosen = int(order[0])
        best = int(
            min(indices.tolist(), key=lambda index: (translation[index], rotation[index], index))
        )
        selected.append(chosen)
        oracle.append(best)
        selected_ranks.append(
            1 + int(np.count_nonzero(translation[indices] < translation[chosen] - 1e-12))
        )
        correct10 = (translation[order] <= 0.10) & (rotation[order] <= 5.0)
        if np.any(correct10):
            rank = int(np.flatnonzero(correct10)[0]) + 1
            best10_ranks.append(rank)
            for k in topk_hits:
                topk_hits[k] += int(rank <= k)
    selected_translation = translation[selected]
    selected_rotation = rotation[selected]
    oracle_translation = translation[oracle]
    count = len(selected)
    return {
        "query_count": int(count),
        "median_translation_m": float(np.median(selected_translation)),
        "p90_translation_m": float(np.percentile(selected_translation, 90)),
        "median_rotation_deg": float(np.median(selected_rotation)),
        "p90_rotation_deg": float(np.percentile(selected_rotation, 90)),
        "recall_10cm_5deg": float(
            np.mean((selected_translation <= 0.10) & (selected_rotation <= 5.0))
        ),
        "recall_25cm_2deg": float(
            np.mean((selected_translation <= 0.25) & (selected_rotation <= 2.0))
        ),
        "catastrophic_gt1m_count": int(np.sum(selected_translation > 1.0)),
        "median_selected_true_rank": float(np.median(selected_ranks)),
        "median_best_10cm_score_rank": (
            None if not best10_ranks else float(np.median(best10_ranks))
        ),
        "top1_10cm_hypothesis_recall": float(topk_hits[1] / max(count, 1)),
        "top5_10cm_hypothesis_recall": float(topk_hits[5] / max(count, 1)),
        "top10_10cm_hypothesis_recall": float(topk_hits[10] / max(count, 1)),
        "median_selection_regret_m": float(
            np.median(selected_translation - oracle_translation)
        ),
        "selected_indices": [int(value) for value in selected],
    }


def _without_indices(summary: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "selected_indices"}


def _selective_relation_scores(
    arrays: Mapping[str, np.ndarray],
    unary: np.ndarray,
    relation: np.ndarray,
    mask: np.ndarray,
    *,
    weight: float,
    advantage_threshold: float,
) -> np.ndarray:
    """Keep unary unless relation provides enough independent winner advantage."""
    output = np.asarray(unary, dtype=np.float64).copy()
    queries = np.asarray(arrays["query_ids"]).astype(str)
    combined = unary + float(weight) * relation
    for query_id in sorted(set(queries[mask].tolist())):
        indices = np.flatnonzero(
            mask & (queries == query_id) & np.isfinite(unary) & np.isfinite(relation)
        )
        if len(indices) == 0:
            continue
        unary_winner = int(indices[np.argmax(unary[indices])])
        combined_winner = int(indices[np.argmax(combined[indices])])
        advantage = float(relation[combined_winner] - relation[unary_winner])
        if advantage >= float(advantage_threshold):
            output[indices] = combined[indices]
    return output


def _paired(combined: Mapping[str, object], unary: Mapping[str, object], target: Mapping[str, np.ndarray]) -> dict[str, object]:
    translation = np.asarray(target["translation_errors_m"], dtype=np.float64)
    combined_indices = np.asarray(combined["selected_indices"], dtype=np.int64)
    unary_indices = np.asarray(unary["selected_indices"], dtype=np.int64)
    if len(combined_indices) != len(unary_indices):
        raise ValueError("selector summaries cover different queries")
    delta = translation[combined_indices] - translation[unary_indices]
    return {
        "win_count": int(np.sum(delta < -1e-12)),
        "loss_count": int(np.sum(delta > 1e-12)),
        "tie_count": int(np.sum(np.abs(delta) <= 1e-12)),
        "mean_translation_delta_m": float(np.mean(delta)),
        "median_translation_delta_m": float(np.median(delta)),
        "max_regression_m": float(np.max(delta)),
    }


def _promotion_gate(audit: Mapping[str, object]) -> dict[str, object]:
    failures: list[str] = []
    for split, payload in audit.items():
        unary = payload["unary"]
        combined = payload["unary_plus_frozen_relation"]
        paired = payload["paired_vs_unary"]
        checks = {
            "median_true_rank_improved": float(
                combined["median_selected_true_rank"]
            ) < float(unary["median_selected_true_rank"]),
            "median_translation_improved": float(
                combined["median_translation_m"]
            ) < float(unary["median_translation_m"]),
            "p90_translation_not_worse": float(
                combined["p90_translation_m"]
            ) <= float(unary["p90_translation_m"]) + 1e-12,
            "median_rotation_not_worse": float(
                combined["median_rotation_deg"]
            ) <= float(unary["median_rotation_deg"]) + 1e-12,
            "p90_rotation_not_worse": float(
                combined["p90_rotation_deg"]
            ) <= float(unary["p90_rotation_deg"]) + 1e-12,
            "catastrophic_count_not_worse": int(
                combined["catastrophic_gt1m_count"]
            ) <= int(unary["catastrophic_gt1m_count"]),
            "paired_wins_exceed_losses": int(paired["win_count"])
            > int(paired["loss_count"]),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{split}:{name}")
    return {
        "passed": not failures,
        "failures": failures,
        "deployment_relation_weight": (
            None if not failures else 0.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_split", default="train")
    parser.add_argument("--audit_splits", default="validation,test")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--positive_threshold_m", type=float, default=0.10)
    parser.add_argument("--negative_threshold_m", type=float, default=0.25)
    parser.add_argument("--lambda_grid", default="0,0.05,0.1,0.25,0.5,1,2,4")
    parser.add_argument(
        "--relation_advantage_threshold_grid",
        default="0,0.001,0.002,0.005,0.01,0.02,0.05,0.1",
    )
    parser.add_argument(
        "--relation_family",
        choices=("all", "maplet", "shared_support_view", "sfm_covisible_neighbor", "topology", "hard_repeat"),
        default="all",
    )
    args = parser.parse_args(argv)
    paths = [Path(value) for value in str(args.hypothesis_artifacts).split(",") if value]
    channel_indices = _relation_family_channels(str(args.relation_family))
    arrays, metadata = _load_merged_artifacts(paths, channel_indices)
    target = _validate_targets(Path(args.targets), artifact_paths=paths, arrays=arrays)
    bins = _bin_edges(metadata[0])
    eligible = _eligible(arrays)
    splits = np.asarray(arrays["split_names"]).astype(str)
    queries = np.asarray(arrays["query_ids"]).astype(str)
    calibration_mask = eligible & (splits == str(args.calibration_split))
    folds = int(args.folds)
    if folds < 2:
        raise ValueError("relation calibration requires at least two query folds")
    oof_relation = np.full((len(queries),), np.nan, dtype=np.float64)
    for fold in range(folds):
        heldout_queries = {
            query for query in set(queries[calibration_mask]) if _query_fold(query, folds) == fold
        }
        heldout = calibration_mask & np.isin(queries, list(heldout_queries))
        fit = calibration_mask & ~np.isin(queries, list(heldout_queries))
        if not np.any(heldout):
            continue
        model = _fit_model(
            arrays,
            target,
            fit,
            bins=bins,
            positive_threshold_m=float(args.positive_threshold_m),
            negative_threshold_m=float(args.negative_threshold_m),
            source_manifest={"fold": fold, "query_grouped_oof": True},
            relation_family=str(args.relation_family),
        )
        oof_relation[heldout] = _relation_scores(
            arrays, model, channel_indices
        )[heldout]
    if np.any(~np.isfinite(oof_relation[calibration_mask])):
        missing_queries = sorted(
            set(queries[calibration_mask & ~np.isfinite(oof_relation)].tolist())
        )
        raise RuntimeError(
            "query-grouped OOF relation calibration did not score every eligible "
            f"calibration row; missing queries: {missing_queries}"
        )
    unary = np.asarray(arrays[UNARY_FIELD], dtype=np.float64)
    lambdas = [float(value) for value in str(args.lambda_grid).split(",") if value]
    candidates = []
    unary_train = summarize_selector(arrays, target, unary, calibration_mask)
    advantage_thresholds = [
        float(value)
        for value in str(args.relation_advantage_threshold_grid).split(",")
        if value
    ]
    for weight in lambdas:
        thresholds = (0.0,) if float(weight) == 0.0 else advantage_thresholds
        for threshold in thresholds:
            summary = summarize_selector(
                arrays,
                target,
                _selective_relation_scores(
                    arrays,
                    unary,
                    oof_relation,
                    calibration_mask,
                    weight=weight,
                    advantage_threshold=threshold,
                ),
                calibration_mask,
            )
            candidates.append(
                (weight, threshold, summary, _paired(summary, unary_train, target))
            )

    def train_gate_eligible(
        item: tuple[float, float, dict[str, object], dict[str, object]]
    ) -> bool:
        weight, _threshold, summary, paired = item
        if float(weight) == 0.0:
            return True
        return (
            float(summary["median_selected_true_rank"])
            < float(unary_train["median_selected_true_rank"])
            and float(summary["median_translation_m"])
            < float(unary_train["median_translation_m"])
            and float(summary["p90_translation_m"])
            <= float(unary_train["p90_translation_m"]) + 1e-12
            and float(summary["median_rotation_deg"])
            <= float(unary_train["median_rotation_deg"]) + 1e-12
            and float(summary["p90_rotation_deg"])
            <= float(unary_train["p90_rotation_deg"]) + 1e-12
            and int(summary["catastrophic_gt1m_count"])
            <= int(unary_train["catastrophic_gt1m_count"])
            and int(paired["win_count"]) > int(paired["loss_count"])
        )

    eligible_candidates = [item for item in candidates if train_gate_eligible(item)]
    (
        selected_weight,
        selected_advantage_threshold,
        selected_train,
        selected_train_paired,
    ) = min(
        eligible_candidates,
        key=lambda item: (
            float("inf") if item[2]["median_best_10cm_score_rank"] is None else float(item[2]["median_best_10cm_score_rank"]),
            float(item[2]["median_selected_true_rank"]),
            float(item[2]["p90_translation_m"]),
            float(item[2]["median_translation_m"]),
            abs(float(item[0])),
            -float(item[1]),
        ),
    )
    final_model = _fit_model(
        arrays,
        target,
        calibration_mask,
        bins=bins,
        positive_threshold_m=float(args.positive_threshold_m),
        negative_threshold_m=float(args.negative_threshold_m),
        source_manifest={
            "query_grouped_oof_weight_selection": True,
            "hypothesis_artifact_sha256": [file_sha256_short(path) for path in paths],
            "target_sha256": file_sha256_short(Path(args.targets)),
            "calibration_split": str(args.calibration_split),
            "query_ids_sha256": hashlib.sha256(
                "\n".join(sorted(set(queries[calibration_mask]))).encode("utf-8")
            ).hexdigest()[:16],
        },
        relation_family=str(args.relation_family),
    )
    relation = _relation_scores(arrays, final_model, channel_indices)
    audit: dict[str, object] = {}
    for split in [value for value in str(args.audit_splits).split(",") if value]:
        mask = eligible & (splits == split)
        unary_summary = summarize_selector(arrays, target, unary, mask)
        relation_summary = summarize_selector(arrays, target, relation, mask)
        combined_summary = summarize_selector(
            arrays,
            target,
            _selective_relation_scores(
                arrays,
                unary,
                relation,
                mask,
                weight=selected_weight,
                advantage_threshold=selected_advantage_threshold,
            ),
            mask,
        )
        audit[split] = {
            "unary": _without_indices(unary_summary),
            "relation_only_DIAGNOSTIC": _without_indices(relation_summary),
            "unary_plus_frozen_relation": _without_indices(combined_summary),
            "paired_vs_unary": _paired(combined_summary, unary_summary, target),
        }
    promotion_gate = _promotion_gate(audit)
    if bool(promotion_gate["passed"]):
        promotion_gate["deployment_relation_weight"] = float(selected_weight)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "calibrated_relation_likelihood.json"
    final_model.save(model_path)
    summary = {
        "stage": STAGE,
        "protocol": {
            "pose_generation_target_free": True,
            "targets_joined_only_in_calibration_process": True,
            "query_grouped_oof_weight_selection": True,
            "validation_or_test_used_for_weight_selection": False,
            "production_promoted": bool(promotion_gate["passed"]),
        },
        "inputs": {
            "hypothesis_artifacts": [str(path) for path in paths],
            "hypothesis_artifact_sha256": [file_sha256_short(path) for path in paths],
            "targets": str(args.targets),
            "targets_sha256": file_sha256_short(Path(args.targets)),
        },
        "calibration": {
            "relation_family": str(args.relation_family),
            "split": str(args.calibration_split),
            "folds": folds,
            "positive_threshold_m": float(args.positive_threshold_m),
            "negative_threshold_m": float(args.negative_threshold_m),
            "selected_relation_weight": float(selected_weight),
            "selected_relation_advantage_threshold": float(
                selected_advantage_threshold
            ),
            "selected_train_oof": _without_indices(selected_train),
            "selected_train_oof_paired_vs_unary": selected_train_paired,
            "train_oof_tail_aware_weight_gate": True,
            "weight_candidates": [
                {
                    "weight": weight,
                    "advantage_threshold": threshold,
                    "metrics": _without_indices(metrics),
                    "paired_vs_unary": paired,
                    "train_gate_eligible": train_gate_eligible(
                        (weight, threshold, metrics, paired)
                    ),
                }
                for weight, threshold, metrics, paired in candidates
            ],
        },
        "audit": audit,
        "promotion_gate": promotion_gate,
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
