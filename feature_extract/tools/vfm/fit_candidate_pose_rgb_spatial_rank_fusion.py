"""Fit a tail-safe S0 plus high-resolution RGB rank fusion on train OOF rows.

This tool never generates candidates, re-runs PnP, or changes a pose
hypothesis.  It consumes target-free visual scores produced by models that
explicitly excluded their scored train query, then joins train-only pose errors
after scoring to choose a bounded rank-percentile fusion weight.  Validation
and test artifacts are intentionally not accepted here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    train_query_partition_manifest,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_SCORE_FORMAT,
    validate_target_free_pose_llr_score_metadata,
)
from feature_extract.vfm.localization.candidate_pose_rank_fusion import (
    RANK_PERCENTILE_FUSION_POLICY,
    fuse_rank_percentiles,
    selected_position,
    validate_fusion_alpha,
)


CALIBRATION_FORMAT = "candidate_pose_rgb_spatial_rank_percentile_fusion_calibration_v1"
_TARGET_FORMAT = "grouped_pose_hypothesis_targets_v1"
_EPSILON = 1e-12
_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "baseline_score_top1",
    "baseline_selection_scores",
    "pose_log_likelihood_ratios",
)


@dataclass(frozen=True)
class _QueryGroup:
    split_name: str
    evaluation_label: str
    query_id: str
    fold_index: int
    hypothesis_indices: np.ndarray
    baseline_scores: np.ndarray
    visual_scores: np.ndarray
    source_baseline_top1: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("score artifact paths must be non-empty and unique")
    return paths


def _alpha_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(validate_fusion_alpha(float(item.strip())) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("rank fusion alpha grid is invalid") from error
    if not values or 0.0 not in values or len(set(values)) != len(values):
        raise ValueError("rank fusion alpha grid must be unique and include zero")
    return tuple(sorted(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-visual-score-artifacts", type=_paths, required=True)
    parser.add_argument("--target-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--alpha-grid",
        default="0,0.025,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1,1.5,2,3,4",
        help="fixed non-negative visual rank weights; zero exactly reproduces S0",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, str, str, int], ...]:
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    labels = np.asarray(arrays["evaluation_labels"]).astype(str).reshape(-1)
    hypotheses = np.asarray(arrays["hypothesis_indices"], dtype=np.int64).reshape(-1)
    if len({len(query_ids), len(splits), len(labels), len(hypotheses)}) != 1:
        raise ValueError("score row keys are misaligned")
    return tuple(
        (str(split), str(label), str(query_id), int(hypothesis))
        for split, label, query_id, hypothesis in zip(splits, labels, query_ids, hypotheses)
    )


def _validate_partition(partition: object, *, query_id: str) -> tuple[dict[str, object], int]:
    if not isinstance(partition, Mapping):
        raise ValueError("OOF visual score lacks checkpoint query partition")
    try:
        reconstructed = train_query_partition_manifest(
            all_query_ids=partition["all_train"]["query_ids"],  # type: ignore[index]
            inner_train_query_ids=partition["inner_train"]["query_ids"],  # type: ignore[index]
            inner_validation_query_ids=partition["inner_validation"]["query_ids"],  # type: ignore[index]
            fold_count=int(partition["fold_count"]),
            fold_index=int(partition["fold_index"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OOF visual score checkpoint query partition is malformed") from error
    if dict(partition) != reconstructed:
        raise ValueError("OOF visual score checkpoint query partition is stale")
    inner_train = set(str(value) for value in reconstructed["inner_train"]["query_ids"])  # type: ignore[index]
    inner_validation = set(
        str(value) for value in reconstructed["inner_validation"]["query_ids"]  # type: ignore[index]
    )
    if str(query_id) in inner_train or str(query_id) not in inner_validation:
        raise ValueError("OOF visual score query was not excluded from its checkpoint fit")
    return reconstructed, int(reconstructed["fold_index"])


def _load_oof_visual_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object], int]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(_ROW_FIELDS).difference(payload.files)
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: OOF visual score is incomplete ({sorted(missing)})")
        arrays = {field: np.asarray(payload[field]).copy() for field in _ROW_FIELDS}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: OOF visual score metadata is invalid")
    validate_target_free_pose_llr_score_metadata(metadata)
    if (
        metadata.get("format") != CANDIDATE_POSE_LLR_SCORE_FORMAT
        or metadata.get("evidence_variant") != "visual"
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
    ):
        raise ValueError(f"{path}: score is not an immutable raw visual diagnostic")
    strict = metadata.get("strict_candidate_pose_llr_contract")
    if (
        not isinstance(strict, Mapping)
        or strict.get("fixed_global_topl") is not True
        or strict.get("no_pnp") is not True
        or strict.get("render") is not False
        or strict.get("image_retrieval_or_submap_used") is not False
        or strict.get("oof_train_query_scoring") is not True
    ):
        raise ValueError(f"{path}: OOF visual score strict contract is incomplete")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in _ROW_FIELDS):
        raise ValueError(f"{path}: OOF visual score rows are misaligned")
    if (
        not np.isfinite(np.asarray(arrays["baseline_selection_scores"], dtype=np.float64)).all()
        or not np.isfinite(np.asarray(arrays["pose_log_likelihood_ratios"], dtype=np.float64)).all()
    ):
        raise ValueError(f"{path}: OOF visual score values are non-finite")
    queries = np.unique(np.asarray(arrays["query_ids"]).astype(str))
    splits = np.unique(np.asarray(arrays["split_names"]).astype(str))
    if len(queries) != 1 or len(splits) != 1 or str(splits[0]) != "train":
        raise ValueError(f"{path}: OOF score must contain one train query")
    if len(_row_keys(arrays)) != len(set(_row_keys(arrays))):
        raise ValueError(f"{path}: OOF score repeats frozen hypothesis rows")
    model = metadata.get("model_checkpoint_contract")
    if not isinstance(model, Mapping):
        raise ValueError(f"{path}: OOF score lacks checkpoint contract")
    _partition, fold_index = _validate_partition(
        model.get("oof_train_query_partition"), query_id=str(queries[0])
    )
    return arrays, metadata, fold_index


def _load_targets(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "translation_errors_m",
        "rotation_errors_deg",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(fields).difference(payload.files)
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: target artifact is incomplete ({sorted(missing)})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != _TARGET_FORMAT
        or metadata.get("contains_target_fields") is not True
        or metadata.get("targets_joined_after_inference") is not True
    ):
        raise ValueError("rank fusion requires a post-inference grouped target artifact")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in fields):
        raise ValueError("rank fusion target rows are misaligned")
    translation = np.asarray(arrays["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(arrays["rotation_errors_deg"], dtype=np.float64)
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
        or len(_row_keys(arrays)) != len(set(_row_keys(arrays)))
    ):
        raise ValueError("rank fusion target values are invalid")
    return arrays, metadata


def _target_lineage(
    *, score_metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    expected = set(str(value) for value in target_metadata.get("inference_artifact_sha256", []))
    if not expected:
        raise ValueError("rank fusion target has no frozen hypothesis lineage")
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("OOF visual score lacks frozen input manifest")
        if str(inputs.get("hypothesis_artifact", {}).get("sha256", "")) not in expected:
            raise ValueError("OOF visual score and target use different frozen hypotheses")


def _contract_fingerprint(metadata: Sequence[Mapping[str, object]]) -> str:
    contracts: list[dict[str, object]] = []
    for item in metadata:
        model = item.get("model_checkpoint_contract")
        if not isinstance(model, Mapping):
            raise ValueError("OOF visual score lacks model contract")
        strict = item.get("strict_candidate_pose_llr_contract")
        if not isinstance(strict, Mapping):
            raise ValueError("OOF visual score lacks strict candidate contract")
        normalized_strict = dict(strict)
        # This is intentionally different for OOF train scoring and held-out
        # inference.  It proves provenance but is not an encoder or scoring
        # semantic, so it must not prevent the frozen rank policy from being
        # applied to validation/test artifacts with the same model contract.
        normalized_strict.pop("oof_train_query_scoring", None)
        contracts.append(
            {
                "strict_candidate_pose_llr_contract": normalized_strict,
                "model_architecture": model.get("architecture"),
                "model_config": model.get("config"),
                "raw_score_semantics": item.get("raw_score_semantics"),
            }
        )
    unique = {_canonical_hash(contract) for contract in contracts}
    if len(unique) != 1:
        raise ValueError("OOF visual score contracts differ across folds")
    return next(iter(unique))


def _oof_coverage(
    *, score_metadata: Sequence[Mapping[str, object]], scored_query_ids: Sequence[str]
) -> dict[str, object]:
    """Audit whether the submitted OOF scores cover one complete train partition.

    A rank policy fitted with a conveniently successful subset of folds is not a
    valid global inference policy.  Keep partial results useful for diagnosis,
    but make their non-promotable status explicit and machine-enforced.
    """

    expected_ids: set[str] | None = None
    expected_hash = ""
    for metadata in score_metadata:
        model = metadata.get("model_checkpoint_contract")
        if not isinstance(model, Mapping):
            raise ValueError("OOF visual score lacks model checkpoint contract")
        query_ids = np.unique(np.asarray(metadata.get("query_id", ""), dtype=str))
        if query_ids.shape != (1,) or not str(query_ids[0]):
            raise ValueError("OOF visual score metadata query id is invalid")
        partition, _fold_index = _validate_partition(
            model.get("oof_train_query_partition"), query_id=str(query_ids[0])
        )
        all_train = partition.get("all_train")
        if not isinstance(all_train, Mapping):
            raise ValueError("OOF visual score partition lacks all-train queries")
        current_ids = set(str(value) for value in all_train.get("query_ids", ()))
        current_hash = str(all_train.get("query_ids_sha256", ""))
        if not current_ids or not current_hash:
            raise ValueError("OOF visual score all-train partition is invalid")
        if expected_ids is None:
            expected_ids = current_ids
            expected_hash = current_hash
        elif current_ids != expected_ids or current_hash != expected_hash:
            raise ValueError("OOF visual scores disagree on the complete train partition")

    if expected_ids is None:  # pragma: no cover - parse_args requires score artifacts.
        raise ValueError("rank fusion has no OOF score partition")
    observed_ids = set(str(value) for value in scored_query_ids)
    duplicate_count = int(len(tuple(scored_query_ids)) - len(observed_ids))
    missing = tuple(sorted(expected_ids.difference(observed_ids)))
    unexpected = tuple(sorted(observed_ids.difference(expected_ids)))
    return {
        "expected_query_count": int(len(expected_ids)),
        "expected_query_ids_sha256": expected_hash,
        "scored_query_count": int(len(observed_ids)),
        "scored_query_ids_sha256": _canonical_hash(sorted(observed_ids)),
        "duplicate_scored_query_count": duplicate_count,
        "missing_query_count": int(len(missing)),
        "missing_query_ids": list(missing),
        "unexpected_query_count": int(len(unexpected)),
        "unexpected_query_ids": list(unexpected),
        "complete": bool(
            duplicate_count == 0 and not missing and not unexpected and observed_ids == expected_ids
        ),
    }


def _build_groups(
    *,
    scores: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    fold_by_query: Mapping[str, int],
) -> tuple[_QueryGroup, ...]:
    score_keys = _row_keys(scores)
    target_positions = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        target_rows = np.asarray([target_positions[key] for key in score_keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("OOF visual score row is absent from the train-only target") from error
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row, (split, label, query_id, _hypothesis) in enumerate(score_keys):
        grouped.setdefault((split, label, query_id), []).append(row)
    output: list[_QueryGroup] = []
    for (split, label, query_id), rows in sorted(grouped.items()):
        if split != "train" or query_id not in fold_by_query:
            raise ValueError("OOF score group is not a known train query")
        selected = np.asarray(rows, dtype=np.int64)
        target_selected = target_rows[selected]
        hypotheses = np.asarray(scores["hypothesis_indices"], dtype=np.int64)[selected]
        if len(np.unique(hypotheses)) != len(hypotheses):
            raise ValueError("OOF query group repeats a hypothesis index")
        source_top1 = np.asarray(scores["baseline_score_top1"], dtype=bool)[selected]
        if int(np.count_nonzero(source_top1)) != 1:
            raise ValueError("OOF query group must retain exactly one immutable S0 top-1")
        output.append(
            _QueryGroup(
                split_name=str(split),
                evaluation_label=str(label),
                query_id=str(query_id),
                fold_index=int(fold_by_query[query_id]),
                hypothesis_indices=hypotheses,
                baseline_scores=np.asarray(scores["baseline_selection_scores"], dtype=np.float64)[selected],
                visual_scores=np.asarray(scores["pose_log_likelihood_ratios"], dtype=np.float64)[selected],
                source_baseline_top1=source_top1,
                translation_errors_m=np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_selected],
                rotation_errors_deg=np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_selected],
            )
        )
    if len(output) < 2:
        raise ValueError("rank fusion requires at least two OOF train query groups")
    return tuple(output)


def _rows_for_alpha(groups: Sequence[_QueryGroup], *, alpha: float) -> list[dict[str, object]]:
    weight = validate_fusion_alpha(alpha)
    output: list[dict[str, object]] = []
    for group in groups:
        tie = np.asarray(group.hypothesis_indices, dtype=np.int64)
        baseline_position = selected_position(group.baseline_scores, tie)
        source_position = int(np.flatnonzero(group.source_baseline_top1)[0])
        if baseline_position != source_position:
            raise ValueError("rank-percentile alpha zero does not reproduce immutable S0 top-1")
        fused = fuse_rank_percentiles(
            baseline_scores=group.baseline_scores,
            visual_scores=group.visual_scores,
            tie_break_orders=tie,
            alpha=weight,
        )
        selected = selected_position(fused, tie)
        output.append(
            {
                "split_name": group.split_name,
                "evaluation_label": group.evaluation_label,
                "query_id": group.query_id,
                "fold_index": int(group.fold_index),
                "alpha": float(weight),
                "hypothesis_count": int(len(tie)),
                "selected_hypothesis_index": int(tie[selected]),
                "baseline_hypothesis_index": int(tie[baseline_position]),
                "selected_translation_error_m_TARGET_ONLY": float(group.translation_errors_m[selected]),
                "selected_rotation_error_deg_TARGET_ONLY": float(group.rotation_errors_deg[selected]),
                "baseline_translation_error_m_TARGET_ONLY": float(group.translation_errors_m[baseline_position]),
                "baseline_rotation_error_deg_TARGET_ONLY": float(group.rotation_errors_deg[baseline_position]),
                "selection_changed_from_baseline": bool(selected != baseline_position),
                "new_catastrophic_TARGET_ONLY": bool(
                    group.translation_errors_m[baseline_position] <= 1.0
                    and group.translation_errors_m[selected] > 1.0
                ),
            }
        )
    return output


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("rank fusion cannot summarize an empty query set")
    selected_translation = np.asarray(
        [float(row["selected_translation_error_m_TARGET_ONLY"]) for row in rows], dtype=np.float64
    )
    selected_rotation = np.asarray(
        [float(row["selected_rotation_error_deg_TARGET_ONLY"]) for row in rows], dtype=np.float64
    )
    baseline_translation = np.asarray(
        [float(row["baseline_translation_error_m_TARGET_ONLY"]) for row in rows], dtype=np.float64
    )
    baseline_rotation = np.asarray(
        [float(row["baseline_rotation_error_deg_TARGET_ONLY"]) for row in rows], dtype=np.float64
    )
    delta = selected_translation - baseline_translation
    return {
        "query_count": int(len(rows)),
        "median_selected_translation_cm": 100.0 * float(np.median(selected_translation)),
        "p90_selected_translation_cm": 100.0 * float(np.quantile(selected_translation, 0.9)),
        "median_selected_rotation_deg": float(np.median(selected_rotation)),
        "p90_selected_rotation_deg": float(np.quantile(selected_rotation, 0.9)),
        "recall_3cm_5deg": float(np.mean((selected_translation <= 0.03) & (selected_rotation <= 5.0))),
        "recall_5cm_5deg": float(np.mean((selected_translation <= 0.05) & (selected_rotation <= 5.0))),
        "recall_10cm_5deg": float(np.mean((selected_translation <= 0.10) & (selected_rotation <= 5.0))),
        "recall_25cm_2deg": float(np.mean((selected_translation <= 0.25) & (selected_rotation <= 2.0))),
        "catastrophic_1m_count": int(np.count_nonzero(selected_translation > 1.0)),
        "selection_changed_count": int(
            sum(bool(row["selection_changed_from_baseline"]) for row in rows)
        ),
        "new_catastrophic_count": int(
            sum(bool(row["new_catastrophic_TARGET_ONLY"]) for row in rows)
        ),
        "paired_translation_wins": int(np.count_nonzero(delta < -_EPSILON)),
        "paired_translation_losses": int(np.count_nonzero(delta > _EPSILON)),
        "paired_translation_ties": int(np.count_nonzero(np.abs(delta) <= _EPSILON)),
        "median_translation_delta_cm": 100.0 * float(np.median(delta)),
        "baseline_median_translation_cm": 100.0 * float(np.median(baseline_translation)),
        "baseline_p90_translation_cm": 100.0 * float(np.quantile(baseline_translation, 0.9)),
        "baseline_median_rotation_deg": float(np.median(baseline_rotation)),
        "baseline_p90_rotation_deg": float(np.quantile(baseline_rotation, 0.9)),
        "baseline_catastrophic_1m_count": int(np.count_nonzero(baseline_translation > 1.0)),
    }


def _tail_safe_gate(metrics: Mapping[str, object], *, alpha: float) -> dict[str, object]:
    checks = {
        "positive_alpha": float(alpha) > 0.0,
        "effective_selection_change": int(metrics["selection_changed_count"]) > 0,
        "median_translation_strictly_improved": float(metrics["median_selected_translation_cm"])
        < float(metrics["baseline_median_translation_cm"]) - _EPSILON,
        "p90_translation_not_worse": float(metrics["p90_selected_translation_cm"])
        <= float(metrics["baseline_p90_translation_cm"]) + _EPSILON,
        "median_rotation_not_worse": float(metrics["median_selected_rotation_deg"])
        <= float(metrics["baseline_median_rotation_deg"]) + _EPSILON,
        "p90_rotation_not_worse": float(metrics["p90_selected_rotation_deg"])
        <= float(metrics["baseline_p90_rotation_deg"]) + _EPSILON,
        "catastrophic_tail_not_worse": int(metrics["catastrophic_1m_count"])
        <= int(metrics["baseline_catastrophic_1m_count"]),
        "no_new_catastrophic": int(metrics["new_catastrophic_count"]) == 0,
        "paired_wins_exceed_losses": int(metrics["paired_translation_wins"])
        > int(metrics["paired_translation_losses"]),
    }
    return {"checks": checks, "passes": bool(all(checks.values()))}


def _select_alpha(
    groups: Sequence[_QueryGroup], *, alpha_grid: Sequence[float]
) -> tuple[float, list[dict[str, object]], dict[str, object], dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    selected: tuple[float, list[dict[str, object]], dict[str, object], dict[str, object]] | None = None
    for alpha in alpha_grid:
        rows = _rows_for_alpha(groups, alpha=float(alpha))
        metrics = _metrics(rows)
        gate = _tail_safe_gate(metrics, alpha=float(alpha))
        candidates.append({"alpha": float(alpha), "metrics": metrics, "gate": gate})
        if bool(gate["passes"]):
            item = (float(alpha), rows, metrics, gate)
            if selected is None or (
                float(metrics["recall_10cm_5deg"]),
                float(metrics["recall_25cm_2deg"]),
                -float(metrics["median_selected_translation_cm"]),
                -float(metrics["p90_selected_translation_cm"]),
                int(metrics["paired_translation_wins"])
                - int(metrics["paired_translation_losses"]),
                -float(alpha),
            ) > (
                float(selected[2]["recall_10cm_5deg"]),
                float(selected[2]["recall_25cm_2deg"]),
                -float(selected[2]["median_selected_translation_cm"]),
                -float(selected[2]["p90_selected_translation_cm"]),
                int(selected[2]["paired_translation_wins"])
                - int(selected[2]["paired_translation_losses"]),
                -float(selected[0]),
            ):
                selected = item
    if selected is None:
        rows = _rows_for_alpha(groups, alpha=0.0)
        metrics = _metrics(rows)
        return 0.0, rows, metrics, {
            "checks": {"no_tail_safe_positive_alpha": False},
            "passes": False,
        }, candidates
    return (*selected, candidates)


def _nested_oof(
    groups: Sequence[_QueryGroup], *, alpha_grid: Sequence[float]
) -> dict[str, object]:
    folds = tuple(sorted(set(int(group.fold_index) for group in groups)))
    if len(folds) < 2:
        raise ValueError("rank fusion nested OOF requires at least two folds")
    rows: list[dict[str, object]] = []
    fold_reports: list[dict[str, object]] = []
    for fold in folds:
        fit_groups = tuple(group for group in groups if int(group.fold_index) != fold)
        apply_groups = tuple(group for group in groups if int(group.fold_index) == fold)
        alpha, _fit_rows, fit_metrics, fit_gate, candidates = _select_alpha(
            fit_groups, alpha_grid=alpha_grid
        )
        apply_rows = _rows_for_alpha(apply_groups, alpha=alpha)
        rows.extend(apply_rows)
        fold_reports.append(
            {
                "heldout_fold_index": int(fold),
                "fit_query_count": int(len(fit_groups)),
                "heldout_query_count": int(len(apply_groups)),
                "selected_alpha": float(alpha),
                "fit_metrics_TARGET_ONLY": fit_metrics,
                "fit_gate": fit_gate,
                "fit_candidates_TARGET_ONLY": candidates,
                "heldout_metrics_TARGET_ONLY": _metrics(apply_rows),
            }
        )
    metrics = _metrics(rows)
    gate = _tail_safe_gate(metrics, alpha=1.0)
    return {
        "fold_count": int(len(folds)),
        "fold_reports_TARGET_ONLY": fold_reports,
        "metrics_TARGET_ONLY": metrics,
        "tail_safe_gate": gate,
        "per_query_rows_TARGET_ONLY": rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rank fusion query rows")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    calibration_path = output_dir / "rank_fusion_calibration.json"
    if (summary_path.exists() or calibration_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite rank fusion calibration")
    alpha_grid = _alpha_grid(args.alpha_grid)
    loaded = [_load_oof_visual_score(path) for path in args.oof_visual_score_artifacts]
    score_metadata = [item[1] for item in loaded]
    contract_sha256 = _contract_fingerprint(score_metadata)
    scores = {
        field: np.concatenate([np.asarray(item[0][field]) for item in loaded], axis=0)
        for field in _ROW_FIELDS
    }
    if len(_row_keys(scores)) != len(set(_row_keys(scores))):
        raise ValueError("OOF visual score artifacts repeat frozen hypothesis rows")
    fold_by_query: dict[str, int] = {}
    for arrays, metadata, fold_index in loaded:
        query_id = str(np.unique(np.asarray(arrays["query_ids"]).astype(str))[0])
        if query_id in fold_by_query:
            raise ValueError("OOF visual score artifacts repeat a train query")
        fold_by_query[query_id] = int(fold_index)
        if str(metadata.get("query_id", query_id)) != query_id:
            raise ValueError("OOF score metadata query id is stale")
        model = metadata.get("model_checkpoint_contract")
        if not isinstance(model, Mapping) or model.get("train_only_inner_gate_passed") is not True:
            raise ValueError("OOF visual score derives from a checkpoint that failed its inner gate")
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    _target_lineage(score_metadata=score_metadata, target_metadata=target_metadata)
    groups = _build_groups(scores=scores, targets=targets, fold_by_query=fold_by_query)
    coverage = _oof_coverage(
        score_metadata=score_metadata,
        scored_query_ids=tuple(group.query_id for group in groups),
    )
    alpha, final_rows, final_metrics, final_gate, candidates = _select_alpha(
        groups, alpha_grid=alpha_grid
    )
    nested = _nested_oof(groups, alpha_grid=alpha_grid)
    promotion_allowed = (
        bool(coverage["complete"])
        and bool(final_gate["passes"])
        and bool(nested["tail_safe_gate"]["passes"])  # type: ignore[index]
    )
    frozen_alpha = float(alpha) if promotion_allowed else 0.0
    frozen_rows = _rows_for_alpha(groups, alpha=frozen_alpha)
    frozen_metrics = _metrics(frozen_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query_frozen_train_oof_TARGET_ONLY.csv", frozen_rows)
    calibration: dict[str, object] = {
        "format": CALIBRATION_FORMAT,
        "policy": RANK_PERCENTILE_FUSION_POLICY,
        "promotion_allowed": bool(promotion_allowed),
        "raw_visual_scores_must_not_feed_pnp": True,
        "frozen_alpha": float(frozen_alpha),
        "proposed_alpha_TARGET_ONLY": float(alpha),
        "score_contract_sha256": contract_sha256,
        "protocol": {
            "fit_split": "train_query_grouped_oof_only",
            "query_grouped_nested_oof": True,
            "fixed_global_topl": True,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
            "no_pnp_during_calibration": True,
            "target_join_after_target_free_scoring": True,
            "raw_scale_not_assumed_calibrated": True,
            "complete_legal_oof_coverage_required_for_promotion": True,
        },
        "oof_coverage": coverage,
        "inputs": {
            "oof_visual_score_artifacts": [str(path) for path in args.oof_visual_score_artifacts],
            "oof_visual_score_artifact_sha256": [
                file_sha256_short(path) for path in args.oof_visual_score_artifacts
            ],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
        },
    }
    summary: dict[str, object] = {
        "stage": "fit_candidate_pose_rgb_spatial_rank_fusion",
        "format": "candidate_pose_rgb_spatial_rank_percentile_fusion_fit_v1",
        "calibration": calibration,
        "alpha_grid_TARGET_ONLY": [float(value) for value in alpha_grid],
        "global_candidates_TARGET_ONLY": candidates,
        "global_selected_metrics_TARGET_ONLY": final_metrics,
        "global_selected_gate": final_gate,
        "nested_oof_TARGET_ONLY": {
            key: value for key, value in nested.items() if key != "per_query_rows_TARGET_ONLY"
        },
        "frozen_metrics_TARGET_ONLY": frozen_metrics,
        "outputs": {
            "calibration": str(calibration_path),
            "frozen_train_oof_rows": str(
                output_dir / "per_query_frozen_train_oof_TARGET_ONLY.csv"
            ),
        },
    }
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
