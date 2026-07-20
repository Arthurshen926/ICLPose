"""Target-side audit for frozen S1 candidate-specific multiscale RGB scores.

This program is intentionally the only component in the S1 probe that reads
pose errors.  It cannot generate a score artifact, alter a candidate, or fit a
weight.  The resulting report compares each raw local-evidence profile and a
predeclared rank-percentile fusion grid against the immutable S0 selector.

An alpha in this audit is exploratory only.  Validation/late outcomes never
choose one; a later train-only calibration stage must make that choice before
anything can be promoted into pose selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


SCORE_FORMAT = "frozen_multiscale_candidate_pose_scores_v1"
TARGET_FORMAT = "independent_landmark_hypothesis_score_targets_v1"
_EPSILON = 1e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score_artifacts",
        required=True,
        help="comma-separated target-free S1 score shards",
    )
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_statistic",
        choices=("mean", "median", "worst_quartile_mean", "spatial_median_of_means_2x2"),
        default="mean",
    )
    parser.add_argument(
        "--alpha_grid",
        default="0,0.25,0.5,1",
        help="fixed non-negative rank-percentile fusion alphas; alpha zero is S0",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("score artifact paths must be non-empty and unique")
    return paths


def parse_alpha_grid(value: str) -> tuple[float, ...]:
    try:
        items = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("alpha grid contains a non-numeric value") from error
    if (
        not items
        or any(not np.isfinite(item) or item < 0.0 for item in items)
        or len(set(items)) != len(items)
        or 0.0 not in items
    ):
        raise ValueError("alpha grid must be unique, finite, non-negative, and include zero")
    return tuple(sorted(items))


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _load_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "source_chosen_for_optional_pose",
        "baseline_score_top1",
        "baseline_selection_scores",
        "family_names",
        "family_log_likelihood_means",
        "family_log_likelihood_medians",
        "family_log_likelihood_worst_quartile_means",
        "family_spatial_median_of_means_2x2",
        "family_effective_point_counts",
        "family_effective_view_masses",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(fields).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: incomplete S1 score artifact ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != SCORE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
    ):
        raise ValueError(f"{path}: not a target-free diagnostic S1 score")
    strict = metadata.get("strict_frozen_evidence_contract")
    required = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "query_center_gaussian_fallback": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if not isinstance(strict, Mapping) or any(strict.get(key) is not value for key, value in required.items()):
        raise ValueError(f"{path}: S1 strict frozen contract is incomplete")
    count = int(metadata.get("row_count", -1))
    row_fields = tuple(field for field in fields if field != "family_names")
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in row_fields):
        raise ValueError(f"{path}: S1 row-aligned fields are inconsistent")
    family_names = np.asarray(arrays["family_names"]).astype(str).reshape(-1)
    family_count = len(family_names)
    if family_count == 0 or len(set(family_names.tolist())) != family_count:
        raise ValueError(f"{path}: S1 profile names are invalid")
    for field in (
        "family_log_likelihood_means",
        "family_log_likelihood_medians",
        "family_log_likelihood_worst_quartile_means",
        "family_spatial_median_of_means_2x2",
        "family_effective_point_counts",
        "family_effective_view_masses",
    ):
        values = np.asarray(arrays[field])
        if values.shape != (count, family_count) or not np.isfinite(values).all():
            raise ValueError(f"{path}: {field} is non-finite or profile-misaligned")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate S1 query/hypothesis rows")
    return arrays, metadata


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
        missing = sorted(set(fields).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: incomplete target artifact ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != TARGET_FORMAT
        or metadata.get("contains_target_fields") is not True
    ):
        raise ValueError("target artifact is not the expected post-score GT join")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(values).shape[0] != count for values in arrays.values()):
        raise ValueError("target artifact arrays are not row aligned")
    translation = np.asarray(arrays["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(arrays["rotation_errors_deg"], dtype=np.float64)
    if (
        np.any(~np.isfinite(translation))
        or np.any(~np.isfinite(rotation))
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
    ):
        raise ValueError("target errors are invalid")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError("target artifact repeats query/hypothesis rows")
    return arrays, metadata


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, str, str, int], ...]:
    columns = (
        np.asarray(arrays["split_names"]).astype(str).reshape(-1),
        np.asarray(arrays["evaluation_labels"]).astype(str).reshape(-1),
        np.asarray(arrays["query_ids"]).astype(str).reshape(-1),
        np.asarray(arrays["hypothesis_indices"], dtype=np.int64).reshape(-1),
    )
    if len({len(column) for column in columns}) != 1:
        raise ValueError("query/hypothesis key columns are misaligned")
    return tuple(
        (str(split), str(label), str(query), int(index))
        for split, label, query, index in zip(*columns)
    )


def _merge_scores(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], tuple[tuple[str, str, str, int], ...]]:
    loaded = [_load_score(path) for path in paths]
    family_names = np.asarray(loaded[0][0]["family_names"]).astype(str)
    compatibility = {
        "format": loaded[0][1].get("format"),
        "version": loaded[0][1].get("version"),
        "strict_frozen_evidence_contract": loaded[0][1].get("strict_frozen_evidence_contract"),
        "profiles": loaded[0][1].get("profiles"),
        "context_config": loaded[0][1].get("context_config"),
        "source_cache_lineage": loaded[0][1].get("source_cache_lineage"),
    }
    fingerprint = _canonical_hash(compatibility)
    rows: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, object]] = []
    for arrays, item_metadata in loaded:
        if not np.array_equal(np.asarray(arrays["family_names"]).astype(str), family_names):
            raise ValueError("S1 score shards declare different profile orders")
        item_compatibility = {
            "format": item_metadata.get("format"),
            "version": item_metadata.get("version"),
            "strict_frozen_evidence_contract": item_metadata.get("strict_frozen_evidence_contract"),
            "profiles": item_metadata.get("profiles"),
            "context_config": item_metadata.get("context_config"),
            "source_cache_lineage": item_metadata.get("source_cache_lineage"),
        }
        if _canonical_hash(item_compatibility) != fingerprint:
            raise ValueError("S1 score shards have incompatible frozen evidence contracts")
        rows.append(arrays)
        metadata.append(item_metadata)
    merged = {
        field: (
            family_names.copy()
            if field == "family_names"
            else np.concatenate([np.asarray(row[field]) for row in rows], axis=0)
        )
        for field in rows[0]
    }
    keys = _row_keys(merged)
    if len(keys) != len(set(keys)):
        raise ValueError("merged S1 score shards repeat rows")
    return merged, metadata, keys


def rank_percentiles(values: np.ndarray, tie_break_orders: np.ndarray) -> np.ndarray:
    """Stable within-query score percentiles, preserving equal-score ties."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    tie_orders = np.asarray(tie_break_orders, dtype=np.int64).reshape(-1)
    if (
        len(scores) == 0
        or scores.shape != tie_orders.shape
        or not np.isfinite(scores).all()
        or len(np.unique(tie_orders)) != len(tie_orders)
    ):
        raise ValueError("rank percentile inputs are invalid")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty((len(scores),), dtype=np.float64)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and scores[order[end]] == scores[order[begin]]:
            end += 1
        ranks[order[begin:end]] = 0.5 * float(begin + end - 1)
        begin = end
    return ranks / float(max(len(scores) - 1, 1))


def descending_ranks(values: np.ndarray, tie_break_orders: np.ndarray) -> np.ndarray:
    """One-based, deterministic higher-is-better ranks."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    tie_orders = np.asarray(tie_break_orders, dtype=np.int64).reshape(-1)
    if (
        len(scores) == 0
        or scores.shape != tie_orders.shape
        or not np.isfinite(scores).all()
        or len(np.unique(tie_orders)) != len(tie_orders)
    ):
        raise ValueError("descending-rank inputs are invalid")
    order = np.lexsort((tie_orders, -scores))
    ranks = np.empty((len(scores),), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def _groups(keys: Sequence[tuple[str, str, str, int]]) -> dict[tuple[str, str, str], np.ndarray]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for row, (split, label, query_id, _hypothesis_index) in enumerate(keys):
        groups.setdefault((split, label, query_id), []).append(row)
    return {
        group: np.asarray(rows, dtype=np.int64)
        for group, rows in sorted(groups.items())
    }


def _selected_row(scores: np.ndarray, tie_break_orders: np.ndarray) -> int:
    return int(np.flatnonzero(descending_ranks(scores, tie_break_orders) == 1)[0])


def _oracle_row(
    translation_m: np.ndarray, rotation_deg: np.ndarray, tie_break_orders: np.ndarray
) -> int:
    return int(np.lexsort((tie_break_orders, rotation_deg, translation_m))[0])


def _best_correct_rank(
    scores: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    tie_break_orders: np.ndarray,
    threshold_m: float,
) -> int | None:
    valid = (translation_m <= float(threshold_m)) & (rotation_deg <= 5.0)
    if not np.any(valid):
        return None
    return int(np.min(descending_ranks(scores, tie_break_orders)[valid]))


def _per_query_rows(
    *,
    keys: Sequence[tuple[str, str, str, int]],
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    tie_break_orders: np.ndarray,
    family: str,
    score_mode: str,
    alpha: float | None,
) -> list[dict[str, object]]:
    values = np.asarray(scores, dtype=np.float64)
    base = np.asarray(baseline_scores, dtype=np.float64)
    translation = np.asarray(translation_m, dtype=np.float64)
    rotation = np.asarray(rotation_deg, dtype=np.float64)
    orders = np.asarray(tie_break_orders, dtype=np.int64)
    if any(item.shape != (len(keys),) for item in (values, base, translation, rotation, orders)):
        raise ValueError("per-query score/target rows are not aligned")
    rows: list[dict[str, object]] = []
    for (split, label, query_id), group in _groups(keys).items():
        local_scores = values[group]
        local_base = base[group]
        local_translation = translation[group]
        local_rotation = rotation[group]
        local_order = orders[group]
        selected = _selected_row(local_scores, local_order)
        baseline_selected = _selected_row(local_base, local_order)
        oracle = _oracle_row(local_translation, local_rotation, local_order)
        ranks = descending_ranks(local_scores, local_order)
        rows.append(
            {
                "split_name": split,
                "evaluation_label": label,
                "query_id": query_id,
                "family": str(family),
                "score_mode": str(score_mode),
                "alpha": None if alpha is None else float(alpha),
                "hypothesis_count": int(len(group)),
                "selected_hypothesis_index": int(keys[int(group[selected])][3]),
                "baseline_hypothesis_index": int(keys[int(group[baseline_selected])][3]),
                "selected_translation_error_m": float(local_translation[selected]),
                "selected_rotation_error_deg": float(local_rotation[selected]),
                "baseline_translation_error_m": float(local_translation[baseline_selected]),
                "baseline_rotation_error_deg": float(local_rotation[baseline_selected]),
                "oracle_translation_error_m": float(local_translation[oracle]),
                "oracle_rotation_error_deg": float(local_rotation[oracle]),
                "oracle_score_rank": int(ranks[oracle]),
                "best_3cm_rank": _best_correct_rank(
                    local_scores, local_translation, local_rotation, local_order, 0.03
                ),
                "best_5cm_rank": _best_correct_rank(
                    local_scores, local_translation, local_rotation, local_order, 0.05
                ),
                "best_10cm_rank": _best_correct_rank(
                    local_scores, local_translation, local_rotation, local_order, 0.10
                ),
                "selection_changed_from_baseline": bool(selected != baseline_selected),
            }
        )
    return rows


def _quantile(values: Sequence[float], q: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.quantile(finite, q))


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize an empty pose audit")
    translation = np.asarray(
        [float(row["selected_translation_error_m"]) for row in rows], dtype=np.float64
    )
    rotation = np.asarray(
        [float(row["selected_rotation_error_deg"]) for row in rows], dtype=np.float64
    )
    oracle_ranks = np.asarray([int(row["oracle_score_rank"]) for row in rows], dtype=np.float64)
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "median_selected_translation_cm": 100.0 * float(np.median(translation)),
        "p90_selected_translation_cm": 100.0 * float(np.quantile(translation, 0.9)),
        "median_selected_rotation_deg": float(np.median(rotation)),
        "p90_selected_rotation_deg": float(np.quantile(rotation, 0.9)),
        "recall_3cm_5deg": float(np.mean((translation <= 0.03) & (rotation <= 5.0))),
        "recall_5cm_5deg": float(np.mean((translation <= 0.05) & (rotation <= 5.0))),
        "recall_10cm_5deg": float(np.mean((translation <= 0.10) & (rotation <= 5.0))),
        "recall_25cm_2deg": float(np.mean((translation <= 0.25) & (rotation <= 2.0))),
        "median_oracle_score_rank": float(np.median(oracle_ranks)),
        "p90_oracle_score_rank": float(np.quantile(oracle_ranks, 0.9)),
        "catastrophic_1m_count": int(np.count_nonzero(translation > 1.0)),
        "selection_changed_count": int(
            sum(bool(row["selection_changed_from_baseline"]) for row in rows)
        ),
    }
    for name in ("best_3cm_rank", "best_5cm_rank", "best_10cm_rank"):
        values = [float(row[name]) for row in rows if row[name] is not None]
        output[f"{name}_coverage"] = float(len(values) / len(rows))
        output[f"median_{name}"] = _quantile(values, 0.5)
        output[f"p90_{name}"] = _quantile(values, 0.9)
    return output


def _paired(
    baseline_rows: Sequence[Mapping[str, object]], probe_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    baseline = {key(row): row for row in baseline_rows}
    probe = {key(row): row for row in probe_rows}
    if not baseline or set(baseline) != set(probe):
        raise ValueError("paired pose selections do not cover the same queries")
    delta = np.asarray(
        [
            float(probe[item]["selected_translation_error_m"])
            - float(baseline[item]["selected_translation_error_m"])
            for item in sorted(baseline)
        ],
        dtype=np.float64,
    )
    return {
        "translation_wins": int(np.count_nonzero(delta < -_EPSILON)),
        "translation_losses": int(np.count_nonzero(delta > _EPSILON)),
        "translation_ties": int(np.count_nonzero(np.abs(delta) <= _EPSILON)),
        "median_translation_delta_cm": float(100.0 * np.median(delta)),
        "mean_translation_delta_cm": float(100.0 * np.mean(delta)),
    }


def _split_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["split_name"]), []).append(row)
    for split, split_rows in sorted(grouped.items()):
        result[split] = _summary(split_rows)
    return result


def _baseline_rows(
    *,
    keys: Sequence[tuple[str, str, str, int]],
    baseline_scores: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    tie_break_orders: np.ndarray,
) -> list[dict[str, object]]:
    return _per_query_rows(
        keys=keys,
        scores=baseline_scores,
        baseline_scores=baseline_scores,
        translation_m=translation_m,
        rotation_deg=rotation_deg,
        tie_break_orders=tie_break_orders,
        family="immutable_s0_baseline",
        score_mode="baseline_selection_score",
        alpha=0.0,
    )


def _validate_alpha_zero(
    *,
    source_top1: np.ndarray,
    keys: Sequence[tuple[str, str, str, int]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> None:
    selected_source = {
        key for key, selected in zip(keys, np.asarray(source_top1, dtype=bool)) if bool(selected)
    }
    selected_rows = {
        (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]), int(row["selected_hypothesis_index"]))
        for row in baseline_rows
    }
    if selected_source != selected_rows:
        raise ValueError("baseline score ordering/tie-break does not reproduce immutable S0 top1")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty per-query audit")
    fieldnames = sorted({field for row in rows for field in row})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_paths = _paths(args.score_artifacts)
    alphas = parse_alpha_grid(args.alpha_grid)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing audit: {summary_path}")
    scores, score_metadata, keys = _merge_scores(score_paths)
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    target_keys = _row_keys(targets)
    target_positions = {key: row for row, key in enumerate(target_keys)}
    try:
        target_rows = np.asarray([target_positions[key] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("a frozen S1 score row is absent from the target artifact") from error
    target_hypothesis_hashes = set(str(value) for value in target_metadata.get("hypothesis_artifact_sha256", []))
    target_s0_hashes = set(str(value) for value in target_metadata.get("score_artifact_sha256", []))
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("S1 metadata lacks source input manifest")
        hypothesis_hash = str(inputs.get("hypothesis_artifact", {}).get("sha256", ""))
        baseline_hash = str(inputs.get("baseline_score_artifact", {}).get("sha256", ""))
        if hypothesis_hash not in target_hypothesis_hashes or baseline_hash not in target_s0_hashes:
            raise ValueError("S1 score/target lineage differs from the frozen S0 source")
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline = np.asarray(scores["baseline_selection_scores"], dtype=np.float64)
    source_top1 = np.asarray(scores["baseline_score_top1"], dtype=bool)
    if not np.isfinite(baseline).all() or translation.shape != baseline.shape or rotation.shape != baseline.shape:
        raise ValueError("frozen S1 score/target arrays are misaligned")
    tie_orders = np.arange(len(keys), dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=keys,
        baseline_scores=baseline,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
    )
    _validate_alpha_zero(source_top1=source_top1, keys=keys, baseline_rows=baseline_rows)

    statistic_to_field = {
        "mean": "family_log_likelihood_means",
        "median": "family_log_likelihood_medians",
        "worst_quartile_mean": "family_log_likelihood_worst_quartile_means",
        "spatial_median_of_means_2x2": "family_spatial_median_of_means_2x2",
    }
    family_values = np.asarray(scores[statistic_to_field[str(args.score_statistic)]], dtype=np.float64)
    family_names = np.asarray(scores["family_names"]).astype(str)
    all_rows: list[dict[str, object]] = list(baseline_rows)
    families: dict[str, object] = {}
    for family_index, family in enumerate(family_names.tolist()):
        local = family_values[:, family_index]
        standalone_rows = _per_query_rows(
            keys=keys,
            scores=local,
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=family,
            score_mode=f"standalone_{args.score_statistic}",
            alpha=None,
        )
        all_rows.extend(standalone_rows)
        family_report: dict[str, object] = {
            "standalone": {
                "splits": _split_summary(standalone_rows),
                "paired_vs_s0": _paired(baseline_rows, standalone_rows),
            },
            "rank_percentile_fusion": {},
            "static_evidence": {
                "mean_effective_point_count": float(
                    np.mean(np.asarray(scores["family_effective_point_counts"])[:, family_index])
                ),
                "mean_effective_view_mass": float(
                    np.mean(np.asarray(scores["family_effective_view_masses"])[:, family_index])
                ),
            },
        }
        for alpha in alphas:
            fused = np.empty_like(baseline)
            for _group_key, group in _groups(keys).items():
                fused[group] = rank_percentiles(baseline[group], tie_orders[group]) + float(alpha) * rank_percentiles(
                    local[group], tie_orders[group]
                )
            fused_rows = _per_query_rows(
                keys=keys,
                scores=fused,
                baseline_scores=baseline,
                translation_m=translation,
                rotation_deg=rotation,
                tie_break_orders=tie_orders,
                family=family,
                score_mode="rank_percentile_fusion",
                alpha=float(alpha),
            )
            all_rows.extend(fused_rows)
            family_report["rank_percentile_fusion"][str(alpha)] = {
                "splits": _split_summary(fused_rows),
                "paired_vs_s0": _paired(baseline_rows, fused_rows),
                "exploratory_only": True,
            }
        families[family] = family_report
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", all_rows)
    summary: dict[str, Any] = {
        "stage": "frozen_multiscale_candidate_pose_target_audit",
        "format": "frozen_multiscale_candidate_pose_target_audit_v1",
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "target_free_s1_scores_validated": True,
            "fixed_global_topl": True,
            "no_image_retrieval_or_submap": True,
            "no_render": True,
            "alpha_selection_from_validation_or_late": False,
            "rank_percentile_fusion_is_exploratory_only": True,
            "promotion_allowed": False,
        },
        "score_statistic": str(args.score_statistic),
        "alpha_grid": list(alphas),
        "baseline": {"splits": _split_summary(baseline_rows)},
        "families": families,
        "inputs": {
            "score_artifacts": [str(path) for path in score_paths],
            "score_artifact_sha256": [file_sha256_short(path) for path in score_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "score_contract_hash": _canonical_hash(
                {
                    "strict_frozen_evidence_contract": score_metadata[0].get("strict_frozen_evidence_contract"),
                    "profiles": score_metadata[0].get("profiles"),
                    "context_config": score_metadata[0].get("context_config"),
                    "source_cache_lineage": score_metadata[0].get("source_cache_lineage"),
                }
            ),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
