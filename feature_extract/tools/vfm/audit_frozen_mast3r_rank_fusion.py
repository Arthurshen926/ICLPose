"""Audit a train-calibrated MASt3R/baseline pose-score fusion.

This is deliberately an analysis-only bridge between two target-free score
artifacts.  It never changes a proposal, correspondence, or pose hypothesis.
The MASt3R score is converted to a within-query rank percentile so that its
un-calibrated likelihood units cannot overpower the immutable independent
verifier merely because of a scale mismatch.  A finite, declared alpha grid is
chosen using *train* GT only; validation and test are joined afterwards and are
never used to choose the alpha.

The artifact answers a narrow question: does the new pair-conditioned visual
evidence complement the frozen baseline safely enough to justify a learned
multiscale follow-up?  It is not a production selector and it does not promote
an optional hypothesis into the main localization path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_independent_landmark_pose_scores import (
    _gt_pose_w2c,
    _merge_hypothesis_lookup,
    _merge_score_artifacts,
    _paths,
)
from feature_extract.tools.vfm.eval_pose_conditioned_support_alignment import (
    _merge_scores as _merge_alignment_scores,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


AUDIT_FORMAT = "frozen_mast3r_rank_fusion_audit_v1"
_EPSILON = 1e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_score_artifacts", required=True)
    parser.add_argument("--mast3r_score_artifacts", required=True)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mast3r_score_field",
        default="mast3r_affine_maplet_context9_support_image_prior_mixture",
    )
    parser.add_argument("--fit_split", default="train")
    parser.add_argument("--evaluate_splits", default="validation,test")
    parser.add_argument(
        "--alpha_grid",
        default="0,0.015625,0.03125,0.0625,0.125,0.25,0.5,1",
        help="non-negative MASt3R rank-percentile weights; must include zero",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise ValueError("at least one split is required")
    if len(set(items)) != len(items):
        raise ValueError("split names must be unique")
    return items


def parse_alpha_grid(value: str) -> tuple[float, ...]:
    """Parse a finite predeclared rank-space fusion grid."""

    try:
        values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("alpha grid contains a non-numeric value") from error
    if not values or any(not np.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("alpha grid must contain finite non-negative values")
    if len(set(values)) != len(values) or 0.0 not in values:
        raise ValueError("alpha grid must be unique and include the baseline alpha zero")
    return tuple(sorted(values))


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, str, str, int], ...]:
    required = {"query_ids", "split_names", "evaluation_labels", "hypothesis_indices"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"score artifact lacks row keys: {missing}")
    count = len(np.asarray(arrays["query_ids"]))
    columns = (
        np.asarray(arrays["split_names"]).astype(str),
        np.asarray(arrays["evaluation_labels"]).astype(str),
        np.asarray(arrays["query_ids"]).astype(str),
        np.asarray(arrays["hypothesis_indices"], dtype=np.int64),
    )
    if any(len(column) != count for column in columns):
        raise ValueError("score key columns are not row-aligned")
    keys = tuple(
        (str(split), str(label), str(query_id), int(index))
        for split, label, query_id, index in zip(*columns)
    )
    if len(keys) != len(set(keys)):
        raise ValueError("score artifact repeats a query/hypothesis row")
    return keys


def _aligned_score_arrays(
    *,
    baseline: Mapping[str, np.ndarray],
    mast3r: Mapping[str, np.ndarray],
    mast3r_score_field: str,
) -> tuple[dict[str, np.ndarray], tuple[tuple[str, str, str, int], ...]]:
    """Return both score sources in one deterministic frozen-row order."""

    if "independent_selection_scores" not in baseline:
        raise ValueError("baseline score artifact lacks independent_selection_scores")
    if mast3r_score_field not in mast3r:
        raise ValueError(f"MASt3R score artifact lacks {mast3r_score_field!r}")
    baseline_keys = _row_keys(baseline)
    mast3r_keys = _row_keys(mast3r)
    if set(baseline_keys) != set(mast3r_keys):
        missing_mast3r = len(set(baseline_keys).difference(mast3r_keys))
        missing_baseline = len(set(mast3r_keys).difference(baseline_keys))
        raise ValueError(
            "baseline and MASt3R artifacts do not cover identical frozen rows "
            f"(missing_mast3r={missing_mast3r}, missing_baseline={missing_baseline})"
        )
    canonical = tuple(sorted(baseline_keys))
    baseline_positions = {key: row for row, key in enumerate(baseline_keys)}
    mast3r_positions = {key: row for row, key in enumerate(mast3r_keys)}
    baseline_values = np.asarray(baseline["independent_selection_scores"], dtype=np.float64)
    mast3r_values = np.asarray(mast3r[mast3r_score_field], dtype=np.float64)
    if (
        baseline_values.shape != (len(baseline_keys),)
        or mast3r_values.shape != (len(mast3r_keys),)
        or not np.isfinite(baseline_values).all()
        or not np.isfinite(mast3r_values).all()
    ):
        raise ValueError("baseline or MASt3R fusion score is non-finite or misaligned")
    return (
        {
            "baseline_scores": np.asarray(
                [baseline_values[baseline_positions[key]] for key in canonical],
                dtype=np.float64,
            ),
            "mast3r_scores": np.asarray(
                [mast3r_values[mast3r_positions[key]] for key in canonical],
                dtype=np.float64,
            ),
            # The immutable baseline declares its top-1 tie break as the first
            # frozen hypothesis row.  Preserve that order even though this
            # audit reorders rows canonically for source alignment.
            "baseline_tie_break_orders": np.asarray(
                [baseline_positions[key] for key in canonical], dtype=np.int64
            ),
        },
        canonical,
    )


def _validate_alpha_zero_matches_baseline(
    *,
    baseline: Mapping[str, np.ndarray],
    keys: Sequence[tuple[str, str, str, int]],
    alpha_zero_rows: Sequence[Mapping[str, object]],
) -> None:
    """Prove rank-space alpha zero is exactly the immutable source selector."""

    if "independent_score_top1" not in baseline:
        raise ValueError("baseline score artifact lacks independent_score_top1")
    source_keys = _row_keys(baseline)
    source_selected = {
        key
        for key, selected in zip(source_keys, np.asarray(baseline["independent_score_top1"]))
        if bool(selected)
    }
    expected_group_count = len(_groups(keys))
    if len(source_selected) != expected_group_count:
        raise ValueError("baseline source selector does not select exactly one row per query")
    alpha_zero_selected = {
        (
            str(row["split_name"]),
            str(row["evaluation_label"]),
            str(row["query_id"]),
            int(row["selected_hypothesis_index"]),
        )
        for row in alpha_zero_rows
    }
    if source_selected != alpha_zero_selected:
        raise ValueError(
            "rank-space alpha zero differs from immutable baseline top1; "
            "a tie-break or row-alignment contract is broken"
        )


def rank_percentiles(values: np.ndarray, tie_break_orders: np.ndarray) -> np.ndarray:
    """Map higher-is-better scores to stable [0, 1] percentiles.

    Exact score ties remain exact ties. The source baseline's frozen row order
    is used later only when selecting a top hypothesis; it must not manufacture
    a spurious rank difference at alpha zero.
    """

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    orders = np.asarray(tie_break_orders, dtype=np.int64).reshape(-1)
    if scores.shape != orders.shape or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("rank-percentile inputs are invalid")
    if len(np.unique(orders)) != len(orders):
        raise ValueError("rank-percentile tie-break orders must be unique")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty((len(scores),), dtype=np.float64)
    begin = 0
    while begin < len(order):
        stop = begin + 1
        while stop < len(order) and scores[order[stop]] == scores[order[begin]]:
            stop += 1
        ranks[order[begin:stop]] = 0.5 * float(begin + stop - 1)
        begin = stop
    return ranks / float(max(len(scores) - 1, 1))


def descending_ranks(values: np.ndarray, tie_break_orders: np.ndarray) -> np.ndarray:
    """Return one-based stable ranks, with the largest score ranked first."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    orders = np.asarray(tie_break_orders, dtype=np.int64).reshape(-1)
    if scores.shape != orders.shape or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("descending-rank inputs are invalid")
    if len(np.unique(orders)) != len(orders):
        raise ValueError("descending-rank tie-break orders must be unique")
    order = np.lexsort((orders, -scores))
    ranks = np.empty((len(scores),), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def _groups(keys: Sequence[tuple[str, str, str, int]]) -> dict[tuple[str, str, str], np.ndarray]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for row, (split, label, query_id, _index) in enumerate(keys):
        groups.setdefault((split, label, query_id), []).append(row)
    return {
        group: np.asarray(rows, dtype=np.int64)
        for group, rows in sorted(groups.items())
    }


def _selected_row(values: np.ndarray, tie_break_orders: np.ndarray) -> int:
    ranks = descending_ranks(values, tie_break_orders)
    return int(np.flatnonzero(ranks == 1)[0])


def _oracle_row(
    translation_m: np.ndarray, rotation_deg: np.ndarray, tie_break_orders: np.ndarray
) -> int:
    order = np.lexsort((tie_break_orders, rotation_deg, translation_m))
    return int(order[0])


def _best_correct_rank(
    scores: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    tie_break_orders: np.ndarray,
) -> int | None:
    ranks = descending_ranks(scores, tie_break_orders)
    correct = (translation_m <= 0.10) & (rotation_deg <= 5.0)
    return None if not np.any(correct) else int(np.min(ranks[correct]))


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _p90(values: Sequence[float]) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values, dtype=np.float64), 0.9))


def score_selection(
    *,
    keys: Sequence[tuple[str, str, str, int]],
    baseline_scores: np.ndarray,
    mast3r_scores: np.ndarray,
    baseline_tie_break_orders: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    alpha: float,
) -> list[dict[str, object]]:
    """Select frozen hypotheses using a rank-space MASt3R optional score."""

    count = len(keys)
    arrays = (
        np.asarray(baseline_scores, dtype=np.float64).reshape(-1),
        np.asarray(mast3r_scores, dtype=np.float64).reshape(-1),
        np.asarray(baseline_tie_break_orders, dtype=np.int64).reshape(-1),
        np.asarray(translation_m, dtype=np.float64).reshape(-1),
        np.asarray(rotation_deg, dtype=np.float64).reshape(-1),
    )
    if (
        count == 0
        or any(values.shape != (count,) for values in arrays)
        or not np.isfinite(np.concatenate((arrays[0], arrays[1], arrays[3], arrays[4]))).all()
        or len(np.unique(arrays[2])) != count
        or not np.isfinite(alpha)
        or alpha < 0.0
    ):
        raise ValueError("frozen fusion selection inputs are invalid")
    result: list[dict[str, object]] = []
    for (split, label, query_id), rows in _groups(keys).items():
        local_indices = np.asarray([keys[row][3] for row in rows], dtype=np.int64)
        baseline_local = arrays[0][rows]
        mast3r_local = arrays[1][rows]
        tie_break_local = arrays[2][rows]
        baseline_percentile = rank_percentiles(baseline_local, tie_break_local)
        mast3r_percentile = rank_percentiles(mast3r_local, tie_break_local)
        fused = baseline_percentile + float(alpha) * mast3r_percentile
        selected_local = _selected_row(fused, tie_break_local)
        baseline_local_selected = _selected_row(baseline_local, tie_break_local)
        oracle_local = _oracle_row(arrays[3][rows], arrays[4][rows], tie_break_local)
        ranks = descending_ranks(fused, tie_break_local)
        result.append(
            {
                "split_name": split,
                "evaluation_label": label,
                "query_id": query_id,
                "hypothesis_count": int(len(rows)),
                "alpha": float(alpha),
                "selected_hypothesis_index": int(local_indices[selected_local]),
                "baseline_hypothesis_index": int(local_indices[baseline_local_selected]),
                "selected_translation_error_m": float(arrays[3][rows][selected_local]),
                "selected_rotation_error_deg": float(arrays[4][rows][selected_local]),
                "baseline_translation_error_m": float(
                    arrays[3][rows][baseline_local_selected]
                ),
                "baseline_rotation_error_deg": float(
                    arrays[4][rows][baseline_local_selected]
                ),
                "oracle_translation_error_m": float(arrays[3][rows][oracle_local]),
                "oracle_rotation_error_deg": float(arrays[4][rows][oracle_local]),
                "oracle_score_rank": int(ranks[oracle_local]),
                "best_10cm_rank": _best_correct_rank(
                    fused,
                    arrays[3][rows],
                    arrays[4][rows],
                    tie_break_local,
                ),
                "selection_changed_from_baseline": bool(
                    selected_local != baseline_local_selected
                ),
            }
        )
    return result


def summarize_selection(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize zero selected queries")
    selected_translation = [float(row["selected_translation_error_m"]) for row in rows]
    selected_rotation = [float(row["selected_rotation_error_deg"]) for row in rows]
    oracle_ranks = [float(row["oracle_score_rank"]) for row in rows]
    best_10cm_ranks = [
        float(row["best_10cm_rank"])
        for row in rows
        if row.get("best_10cm_rank") is not None
    ]
    return {
        "query_count": int(len(rows)),
        "median_selected_translation_cm": 100.0 * float(np.median(selected_translation)),
        "p90_selected_translation_cm": 100.0 * float(np.quantile(selected_translation, 0.9)),
        "median_selected_rotation_deg": float(np.median(selected_rotation)),
        "p90_selected_rotation_deg": float(np.quantile(selected_rotation, 0.9)),
        "recall_5cm_5deg": float(
            np.mean(
                [
                    translation <= 0.05 and rotation <= 5.0
                    for translation, rotation in zip(selected_translation, selected_rotation)
                ]
            )
        ),
        "recall_10cm_5deg": float(
            np.mean(
                [
                    translation <= 0.10 and rotation <= 5.0
                    for translation, rotation in zip(selected_translation, selected_rotation)
                ]
            )
        ),
        "median_oracle_score_rank": _median(oracle_ranks),
        "p90_oracle_score_rank": _p90(oracle_ranks),
        "median_best_10cm_rank": _median(best_10cm_ranks),
        "catastrophic_1m_count": int(sum(value > 1.0 for value in selected_translation)),
        "selection_changed_count": int(
            sum(bool(row["selection_changed_from_baseline"]) for row in rows)
        ),
    }


def paired_summary(
    baseline_rows: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compute target-only paired safety metrics after frozen scoring."""

    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    baseline = {key(row): row for row in baseline_rows}
    probe = {key(row): row for row in probe_rows}
    if not baseline or set(baseline) != set(probe):
        raise ValueError("paired fusion rows are not aligned")
    deltas = np.asarray(
        [
            float(probe[item]["selected_translation_error_m"])
            - float(baseline[item]["selected_translation_error_m"])
            for item in sorted(baseline)
        ],
        dtype=np.float64,
    )
    return {
        "translation_wins": int(np.sum(deltas < -_EPSILON)),
        "translation_losses": int(np.sum(deltas > _EPSILON)),
        "translation_ties": int(np.sum(np.abs(deltas) <= _EPSILON)),
        "mean_translation_delta_cm": float(100.0 * np.mean(deltas)),
        "median_translation_delta_cm": float(100.0 * np.median(deltas)),
    }


def _safe_train_candidate(
    metrics: Mapping[str, object], baseline_metrics: Mapping[str, object], paired: Mapping[str, object]
) -> bool:
    return bool(
        float(metrics["median_selected_translation_cm"])
        <= float(baseline_metrics["median_selected_translation_cm"]) + _EPSILON
        and float(metrics["p90_selected_translation_cm"])
        <= float(baseline_metrics["p90_selected_translation_cm"]) + _EPSILON
        and int(metrics["catastrophic_1m_count"])
        <= int(baseline_metrics["catastrophic_1m_count"])
        and int(paired["translation_wins"]) >= int(paired["translation_losses"])
    )


def choose_train_alpha(
    *,
    rows_by_alpha: Mapping[float, Sequence[Mapping[str, object]]],
    fit_split: str,
) -> tuple[float, dict[str, object]]:
    """Choose one alpha with train-only rank improvement and tail safeguards."""

    if 0.0 not in rows_by_alpha:
        raise ValueError("train alpha selection requires the zero baseline")
    base_rows = [row for row in rows_by_alpha[0.0] if row["split_name"] == fit_split]
    if not base_rows:
        raise ValueError(f"fit split {fit_split!r} has no frozen rows")
    baseline_metrics = summarize_selection(base_rows)
    candidates: list[tuple[tuple[float, ...], float, dict[str, object]]] = []
    reports: dict[str, object] = {}
    for alpha in sorted(rows_by_alpha):
        rows = [row for row in rows_by_alpha[alpha] if row["split_name"] == fit_split]
        if len(rows) != len(base_rows):
            raise ValueError("alpha grid changes the train query set")
        metrics = summarize_selection(rows)
        paired = paired_summary(base_rows, rows)
        tail_safe = _safe_train_candidate(metrics, baseline_metrics, paired)
        report = {
            "metrics": metrics,
            "paired_vs_alpha0": paired,
            "tail_safe_train_candidate": tail_safe,
        }
        reports[str(alpha)] = report
        if tail_safe:
            # All terms are train-only.  The explicit alpha tie break keeps a
            # zero-weight baseline preferred when the new evidence is neutral.
            objective = (
                float(metrics["median_oracle_score_rank"]),
                float(metrics["p90_oracle_score_rank"]),
                float(metrics["median_selected_translation_cm"]),
                float(metrics["p90_selected_translation_cm"]),
                float(metrics["catastrophic_1m_count"]),
                float(alpha),
            )
            candidates.append((objective, float(alpha), report))
    if not candidates:
        raise RuntimeError("alpha-zero baseline unexpectedly failed train safeguards")
    _objective, alpha, report = min(candidates, key=lambda item: item[0])
    return alpha, {
        "fit_split": str(fit_split),
        "objective": "train_median_oracle_rank_then_tail_safe_metrics_v1",
        "baseline_alpha": 0.0,
        "chosen_alpha": float(alpha),
        "baseline_metrics": baseline_metrics,
        "chosen_train_report": report,
        "grid": reports,
    }


def _split_gate(
    *,
    baseline_metrics: Mapping[str, object],
    probe_metrics: Mapping[str, object],
    paired: Mapping[str, object],
) -> dict[str, bool]:
    rank_under_20 = float(probe_metrics["median_oracle_score_rank"]) < 20.0
    median_not_worse = float(probe_metrics["median_selected_translation_cm"]) <= float(
        baseline_metrics["median_selected_translation_cm"]
    ) + _EPSILON
    p90_not_worse = float(probe_metrics["p90_selected_translation_cm"]) <= float(
        baseline_metrics["p90_selected_translation_cm"]
    ) + _EPSILON
    catastrophic_not_increased = int(probe_metrics["catastrophic_1m_count"]) <= int(
        baseline_metrics["catastrophic_1m_count"]
    )
    paired_wins_exceed_losses = int(paired["translation_wins"]) > int(
        paired["translation_losses"]
    )
    return {
        "median_oracle_rank_under_20": rank_under_20,
        "median_not_worse": median_not_worse,
        "p90_not_worse": p90_not_worse,
        "catastrophic_not_increased": catastrophic_not_increased,
        "paired_wins_exceed_losses": paired_wins_exceed_losses,
        "pass": bool(
            rank_under_20
            and median_not_worse
            and p90_not_worse
            and catastrophic_not_increased
            and paired_wins_exceed_losses
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty per-query audit")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    alphas = parse_alpha_grid(args.alpha_grid)
    fit_split = str(args.fit_split).strip()
    evaluation_splits = _parse_csv(args.evaluate_splits)
    if not fit_split or fit_split in evaluation_splits:
        raise ValueError("fit split must be non-empty and disjoint from evaluation splits")
    output_dir = Path(args.output_dir)
    output_summary = output_dir / "summary.json"
    if output_summary.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_summary}")

    baseline_paths = _paths(args.baseline_score_artifacts)
    mast3r_paths = _paths(args.mast3r_score_artifacts)
    hypothesis_paths = _paths(args.hypothesis_artifacts)
    baseline, baseline_metadata, baseline_compatibility = _merge_score_artifacts(baseline_paths)
    mast3r, mast3r_metadata, mast3r_compatibility = _merge_alignment_scores(mast3r_paths)
    scores, keys = _aligned_score_arrays(
        baseline=baseline,
        mast3r=mast3r,
        mast3r_score_field=str(args.mast3r_score_field),
    )
    observed_splits = {key[0] for key in keys}
    if fit_split not in observed_splits or not set(evaluation_splits).issubset(observed_splits):
        raise ValueError("requested fit/evaluation split is absent from frozen rows")

    model_dir = Path(args.colmap_model_dir)
    images = {
        str(image.image_name): image
        for image in read_colmap_images_binary(model_dir / "images.bin").values()
    }
    poses = _merge_hypothesis_lookup(hypothesis_paths)
    translation = np.empty((len(keys),), dtype=np.float64)
    rotation = np.empty((len(keys),), dtype=np.float64)
    for row, (_split, label, query_id, hypothesis_index) in enumerate(keys):
        pose = poses.get((str(query_id), str(label), int(hypothesis_index)))
        image = images.get(str(query_id))
        if pose is None or image is None:
            raise ValueError(f"frozen fusion row lacks hypothesis or GT pose: {query_id}")
        error = pnp_pose_error(pose, _gt_pose_w2c(image))
        translation[row] = float(error.translation_m)
        rotation[row] = float(error.rotation_deg)

    rows_by_alpha = {
        alpha: score_selection(
            keys=keys,
            baseline_scores=scores["baseline_scores"],
            mast3r_scores=scores["mast3r_scores"],
            baseline_tie_break_orders=scores["baseline_tie_break_orders"],
            translation_m=translation,
            rotation_deg=rotation,
            alpha=alpha,
        )
        for alpha in alphas
    }
    _validate_alpha_zero_matches_baseline(
        baseline=baseline,
        keys=keys,
        alpha_zero_rows=rows_by_alpha[0.0],
    )
    chosen_alpha, calibration = choose_train_alpha(
        rows_by_alpha=rows_by_alpha, fit_split=fit_split
    )
    baseline_rows = rows_by_alpha[0.0]
    chosen_rows = rows_by_alpha[chosen_alpha]
    evaluation: dict[str, object] = {}
    for split in evaluation_splits:
        base_split = [row for row in baseline_rows if row["split_name"] == split]
        probe_split = [row for row in chosen_rows if row["split_name"] == split]
        baseline_metrics = summarize_selection(base_split)
        probe_metrics = summarize_selection(probe_split)
        paired = paired_summary(base_split, probe_split)
        evaluation[split] = {
            "baseline_alpha0": baseline_metrics,
            "train_calibrated_fusion": probe_metrics,
            "paired": paired,
            "gate": _split_gate(
                baseline_metrics=baseline_metrics,
                probe_metrics=probe_metrics,
                paired=paired,
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_query = [
        {
            **row,
            "role": "baseline_alpha0" if row["alpha"] == 0.0 else "train_calibrated_fusion",
        }
        for row in baseline_rows + chosen_rows
        if row["split_name"] in set((fit_split,) + evaluation_splits)
    ]
    _write_csv(output_dir / "per_query.csv", per_query)
    summary: dict[str, Any] = {
        "stage": AUDIT_FORMAT,
        "diagnostic_only": True,
        "selection_not_promoted": True,
        "targets_joined_only_after_target_free_scores": True,
        "strict_protocol": {
            "frozen_hypotheses": True,
            "baseline_score_rows_target_free": True,
            "mast3r_score_rows_target_free": True,
            "mast3r_pair_conditioned_real_images_only": True,
            "fit_uses_train_only": True,
            "validation_test_not_used_for_alpha_selection": True,
            "no_hypothesis_regeneration": True,
            "no_image_retrieval_or_submap": True,
            "render": False,
        },
        "calibration": calibration,
        "evaluation": evaluation,
        "overall_gate_pass": bool(all(item["gate"]["pass"] for item in evaluation.values())),
        "inputs": {
            "baseline_score_artifacts": [str(path) for path in baseline_paths],
            "baseline_score_artifact_sha256": [file_sha256_short(path) for path in baseline_paths],
            "baseline_score_compatibility_sha256": baseline_compatibility,
            "mast3r_score_artifacts": [str(path) for path in mast3r_paths],
            "mast3r_score_artifact_sha256": [file_sha256_short(path) for path in mast3r_paths],
            "mast3r_score_compatibility_sha256": mast3r_compatibility,
            "mast3r_score_field": str(args.mast3r_score_field),
            "hypothesis_artifacts": [str(path) for path in hypothesis_paths],
            "hypothesis_artifact_sha256": [file_sha256_short(path) for path in hypothesis_paths],
            "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
            "baseline_version": baseline_metadata[0].get("version"),
            "mast3r_version": mast3r_metadata[0].get("version"),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
