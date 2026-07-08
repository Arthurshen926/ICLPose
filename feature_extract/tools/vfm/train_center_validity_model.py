from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.center_validity_model import (
    RAW_FEATURE_COLUMNS,
    RANK_SOURCE_COLUMNS,
    _read_csv,
    _write_csv,
    build_feature_rows,
    evaluate_scores,
    predict_validity_scores,
    train_validity_model,
)


def _bool_label(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    return float(str(value).strip().lower() in {"1", "true", "yes", "y"})


def _score_column(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        if name == "inverse_local_cost_entropy":
            try:
                values.append(1.0 - float(row.get("local_cost_entropy", "")))
            except (TypeError, ValueError):
                values.append(float("nan"))
        else:
            try:
                values.append(float(row.get(name, "")))
            except (TypeError, ValueError):
                values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _write_prediction_rows(path: Path, rows: Sequence[Mapping[str, Any]], scores: np.ndarray, *, score_name: str) -> None:
    output_rows = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item[score_name] = float(score)
        output_rows.append(item)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    _write_csv(path, output_rows, fieldnames)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center_validity_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--target", default="valid_5px", choices=("valid_5px", "valid_2px"))
    parser.add_argument("--model_types", nargs="+", default=["logistic", "mlp"], choices=("logistic", "mlp"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--calibration_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output_dir)
    rows = _read_csv(Path(args.center_validity_rows_csv))
    feature_rows, feature_columns = build_feature_rows(rows, image_width=int(args.image_width), image_height=int(args.image_height))
    feature_fieldnames = list(feature_rows[0].keys()) if feature_rows else []
    _write_csv(output / "features.csv", feature_rows, feature_fieldnames)
    (output / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2, sort_keys=True) + "\n")
    labels = np.asarray([_bool_label(row.get(args.target, False)) for row in feature_rows], dtype=np.float64)
    heuristic_columns = [
        "inverse_local_cost_entropy",
        "local_cost_peak_prob",
        "local_cost_top2_gap",
        "radio_match_score",
        "measurement_valid_prob",
        *[f"{name}_rank_pct" for name in RANK_SOURCE_COLUMNS],
    ]
    heuristic_metrics: dict[str, Any] = {}
    for column in heuristic_columns:
        scores = _score_column(feature_rows, column)
        if np.isfinite(scores).any():
            heuristic_metrics[column] = evaluate_scores(scores=scores, labels=labels)

    model_summaries: dict[str, Any] = {}
    for model_type in args.model_types:
        result = train_validity_model(
            feature_rows,
            feature_columns,
            target=str(args.target),
            model_type=str(model_type),
            seed=int(args.seed),
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            train_fraction=float(args.train_fraction),
            calibration_fraction=float(args.calibration_fraction),
        )
        scores = predict_validity_scores(result, feature_rows)
        prediction_path = output / f"predictions_{model_type}.csv"
        _write_prediction_rows(prediction_path, feature_rows, scores, score_name=f"p_{args.target}_{model_type}")
        torch.save(
            {
                "model_state": result["model"].state_dict(),
                "model_type": result["model_type"],
                "target": result["target"],
                "feature_columns": result["feature_columns"],
                "mean": result["mean"].tolist(),
                "std": result["std"].tolist(),
                "temperature": result["temperature"],
            },
            output / f"center_validity_{model_type}.pt",
        )
        model_summaries[str(model_type)] = {
            "metrics": result["metrics"],
            "temperature": result["temperature"],
            "predictions_csv": str(prediction_path),
            "checkpoint": str(output / f"center_validity_{model_type}.pt"),
        }

    summary = {
        "stage": "center_validity_model_training",
        "input_rows": int(len(rows)),
        "target": str(args.target),
        "feature_columns": feature_columns,
        "heuristic_metrics": heuristic_metrics,
        "models": model_summaries,
        "outputs": {
            "features_csv": str(output / "features.csv"),
            "feature_columns_json": str(output / "feature_columns.json"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

