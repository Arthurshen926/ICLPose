#!/usr/bin/env python3
"""Train a calibrated render-query pose candidate scorer from eval rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence, confidence_metrics
from feature_extract.vfm.render_pose_scorer import (
    fit_pairwise_pose_ranker,
    pose_candidate_selection_report,
    vectorize_pose_candidate_rows,
    vectorize_pose_rows,
)


def _parse_scalar(value: object) -> object:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        integer = int(text)
    except ValueError:
        integer = None
    if integer is not None and str(integer) == text:
        return integer
    try:
        floating = float(text)
    except ValueError:
        return text
    return floating if math.isfinite(floating) else None


def _load_csv_rows(paths: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path_text in paths:
        with Path(path_text).open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({str(key): _parse_scalar(value) for key, value in row.items() if key is not None})
    return rows


def _query_hash_fraction(query_id: str) -> float:
    import hashlib

    digest = hashlib.sha1(str(query_id).encode("utf8")).hexdigest()
    return float(int(digest[:12], 16)) / float(16**12 - 1)


def _query_count(rows: Sequence[dict[str, object]]) -> int:
    return len({str(row.get("query_id", "")) for row in rows})


def _split_rows(
    rows: Sequence[dict[str, object]],
    *,
    eval_fraction: float,
    eval_on_train: bool,
    split_mode: str = "hash_fraction",
    fold_count: int = 5,
    fold_index: int = 0,
    min_train_queries: int = 0,
    min_eval_queries: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    values = list(rows)
    if bool(eval_on_train):
        split = {
            "mode": "eval_on_train",
            "eval_fraction": float(eval_fraction),
            "fold_count": int(fold_count),
            "fold_index": int(fold_index),
        }
        return values, values, split
    train: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    if str(split_mode) == "kfold":
        if int(fold_count) < 2:
            raise ValueError("--fold_count must be at least 2 for kfold split")
        if int(fold_index) < 0 or int(fold_index) >= int(fold_count):
            raise ValueError("--fold_index must be in [0, fold_count)")
        query_ids = sorted({str(row.get("query_id", "")) for row in values})
        eval_queries = {
            query_id
            for idx, query_id in enumerate(query_ids)
            if int(idx) % int(fold_count) == int(fold_index)
        }
        for row in values:
            if str(row.get("query_id", "")) in eval_queries:
                eval_rows.append(row)
            else:
                train.append(row)
    else:
        for row in values:
            if _query_hash_fraction(str(row.get("query_id", ""))) < float(eval_fraction):
                eval_rows.append(row)
            else:
                train.append(row)
    if not train or not eval_rows:
        midpoint = max(1, len(values) // 2)
        train = values[:midpoint]
        eval_rows = values[midpoint:] or values[:midpoint]
    train_query_count = _query_count(train)
    eval_query_count = _query_count(eval_rows)
    if int(min_train_queries) > 0 and train_query_count < int(min_train_queries):
        raise ValueError(f"train split has {train_query_count} queries, below --min_train_queries={int(min_train_queries)}")
    if int(min_eval_queries) > 0 and eval_query_count < int(min_eval_queries):
        raise ValueError(f"eval split has {eval_query_count} queries, below --min_eval_queries={int(min_eval_queries)}")
    split = {
        "mode": str(split_mode),
        "eval_fraction": float(eval_fraction),
        "fold_count": int(fold_count),
        "fold_index": int(fold_index),
        "min_train_queries": int(min_train_queries),
        "min_eval_queries": int(min_eval_queries),
    }
    return train, eval_rows, split


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--translation_threshold_m", type=float, default=0.10)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--eval_fraction", type=float, default=0.30)
    parser.add_argument("--eval_on_train", action="store_true")
    parser.add_argument("--split_mode", default="hash_fraction", choices=("hash_fraction", "kfold"))
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--min_train_queries", type=int, default=0)
    parser.add_argument("--min_eval_queries", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_iter", type=int, default=800)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--objective", default="logistic", choices=("logistic", "pairwise_rank"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows = _load_csv_rows([str(path) for path in args.rows_csv])
    if not rows:
        raise ValueError("no pose rows loaded")
    train_rows, eval_rows, split_summary = _split_rows(
        rows,
        eval_fraction=float(args.eval_fraction),
        eval_on_train=bool(args.eval_on_train),
        split_mode=str(args.split_mode),
        fold_count=int(args.fold_count),
        fold_index=int(args.fold_index),
        min_train_queries=int(args.min_train_queries),
        min_eval_queries=int(args.min_eval_queries),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pose_scorer_model.json"
    train_x, train_y, names = vectorize_pose_rows(
        train_rows,
        translation_threshold_m=float(args.translation_threshold_m),
        rotation_threshold_deg=float(args.rotation_threshold_deg),
    )
    eval_x, eval_y, _eval_names = vectorize_pose_rows(
        eval_rows,
        translation_threshold_m=float(args.translation_threshold_m),
        rotation_threshold_deg=float(args.rotation_threshold_deg),
    )
    pairwise_train_summary: dict[str, object] = {}
    if str(args.objective) == "pairwise_rank":
        model, pairwise_train_summary = fit_pairwise_pose_ranker(
            train_rows,
            translation_threshold_m=float(args.translation_threshold_m),
            rotation_threshold_deg=float(args.rotation_threshold_deg),
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            l2=float(args.l2),
        )
        model.save_json(model_path, feature_names=names)
        train_scores = model.predict_proba(train_x)
        eval_scores = model.predict_proba(eval_x)
        train_rank_scores = model.predict_scores(train_x)
        eval_rank_scores = model.predict_scores(eval_x)
    else:
        if len(np.unique(train_y)) < 2:
            raise ValueError("pose scorer training rows must contain both positive and negative labels")
        model = CalibratedLogisticConfidence(
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            l2=float(args.l2),
        ).fit(train_x, train_y)
        model.save_json(model_path, feature_names=names)
        train_scores = model.predict_proba(train_x)
        eval_scores = model.predict_proba(eval_x)
        train_rank_scores = train_scores
        eval_rank_scores = eval_scores
    summary = {
        "stage": "render_pose_scorer_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "input_rows_csv": [str(path) for path in args.rows_csv],
        "row_count": int(len(rows)),
        "objective": str(args.objective),
        "train_row_count": int(len(train_rows)),
        "eval_row_count": int(len(eval_rows)),
        "train_query_count": int(_query_count(train_rows)),
        "eval_query_count": int(_query_count(eval_rows)),
        "split": split_summary,
        "thresholds": {
            "translation_m": float(args.translation_threshold_m),
            "rotation_deg": float(args.rotation_threshold_deg),
        },
        "feature_names": names,
        "model_path": str(model_path),
        "train": confidence_metrics(train_y, train_scores, top_fraction=float(args.top_fraction)),
        "eval": confidence_metrics(eval_y, eval_scores, top_fraction=float(args.top_fraction)),
        "train_selection": pose_candidate_selection_report(
            train_rows,
            learned_scores=train_rank_scores,
            translation_threshold_m=float(args.translation_threshold_m),
            rotation_threshold_deg=float(args.rotation_threshold_deg),
        ),
        "eval_selection": pose_candidate_selection_report(
            eval_rows,
            learned_scores=eval_rank_scores,
            translation_threshold_m=float(args.translation_threshold_m),
            rotation_threshold_deg=float(args.rotation_threshold_deg),
        ),
    }
    if pairwise_train_summary:
        summary["pairwise_train"] = pairwise_train_summary
    summary_path = output_dir / "pose_scorer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
