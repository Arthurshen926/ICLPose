#!/usr/bin/env python3
"""Train calibrated rankers for MATCHA coarse render-cell candidates."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.coarse_candidate_ranking import vectorize_coarse_candidate_rows
from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence, confidence_metrics


def _parse_scalar(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null", "nan"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
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


def _load_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="") as handle:
        return [{str(key): _parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _query_hash_fraction(query_id: str) -> float:
    digest = hashlib.sha1(str(query_id).encode("utf8")).hexdigest()
    return float(int(digest[:12], 16)) / float(16**12 - 1)


def _split_rows(
    rows: list[dict[str, object]],
    *,
    eval_fraction: float,
    eval_on_train: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if eval_on_train:
        return rows, rows
    train: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for row in rows:
        if _query_hash_fraction(str(row.get("query_id", ""))) < float(eval_fraction):
            eval_rows.append(row)
        else:
            train.append(row)
    if not train or not eval_rows:
        midpoint = max(1, len(rows) // 2)
        train = rows[:midpoint]
        eval_rows = rows[midpoint:] or rows[:midpoint]
    return train, eval_rows


def _label_count(labels: np.ndarray, keep: np.ndarray) -> dict[str, int]:
    mask = np.asarray(keep, dtype=bool)
    kept = np.asarray(labels, dtype=np.int64)[mask]
    return {
        "kept": int(kept.shape[0]),
        "positive": int(np.sum(kept == 1)),
        "negative": int(np.sum(kept == 0)),
        "ignored": int(np.sum(~mask)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_csv", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_sets", default="descriptor,coarse,coarse_local")
    parser.add_argument("--eval_fraction", type=float, default=0.30)
    parser.add_argument("--eval_on_train", action="store_true")
    parser.add_argument("--stride_positive", type=float, default=1.0)
    parser.add_argument("--weak_positive_stride", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_iter", type=int, default=800)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for path_text in args.match_csv:
        rows.extend(_load_csv(Path(path_text)))
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    if not rows:
        raise ValueError("no match rows loaded")

    train_rows, eval_rows = _split_rows(
        rows,
        eval_fraction=float(args.eval_fraction),
        eval_on_train=bool(args.eval_on_train),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, object] = {}
    for feature_set in [item.strip() for item in str(args.feature_sets).split(",") if item.strip()]:
        train_x, train_y, train_keep, names = vectorize_coarse_candidate_rows(
            train_rows,
            feature_set=feature_set,
            stride_positive=float(args.stride_positive),
            weak_positive_stride=float(args.weak_positive_stride),
        )
        eval_x, eval_y, eval_keep, _eval_names = vectorize_coarse_candidate_rows(
            eval_rows,
            feature_set=feature_set,
            stride_positive=float(args.stride_positive),
            weak_positive_stride=float(args.weak_positive_stride),
        )
        info: dict[str, object] = {
            "feature_set": feature_set,
            "feature_names": names,
            "train_label_counts": _label_count(train_y, train_keep),
            "eval_label_counts": _label_count(eval_y, eval_keep),
        }
        if np.sum(train_keep) == 0 or len(np.unique(train_y[train_keep])) < 2:
            info["status"] = "skipped_missing_train_classes"
            models[feature_set] = info
            continue
        model = CalibratedLogisticConfidence(
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            l2=float(args.l2),
        ).fit(train_x[train_keep], train_y[train_keep])
        model_path = output_dir / f"{feature_set}_model.json"
        model.save_json(model_path, feature_names=names)
        train_scores = model.predict_proba(train_x[train_keep])
        eval_scores = model.predict_proba(eval_x[eval_keep]) if np.any(eval_keep) else np.zeros((0,), dtype=np.float64)
        info.update(
            {
                "status": "trained",
                "model_path": str(model_path),
                "train": confidence_metrics(train_y[train_keep], train_scores, top_fraction=float(args.top_fraction)),
                "eval": confidence_metrics(eval_y[eval_keep], eval_scores, top_fraction=float(args.top_fraction)),
            }
        )
        models[feature_set] = info

    summary = {
        "stage": "coarse_candidate_ranker",
        "elapsed_sec": float(time.perf_counter() - started),
        "input_match_csv": [str(path) for path in args.match_csv],
        "row_count": int(len(rows)),
        "train_row_count": int(len(train_rows)),
        "eval_row_count": int(len(eval_rows)),
        "eval_on_train": bool(args.eval_on_train),
        "label_policy": {
            "stride_positive": float(args.stride_positive),
            "weak_positive_stride": float(args.weak_positive_stride),
        },
        "models": models,
    }
    summary_path = output_dir / "coarse_candidate_ranker_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
