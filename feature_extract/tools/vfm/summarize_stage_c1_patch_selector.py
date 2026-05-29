"""Summarize Stage C1 learned selector and random/PCA control runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence


GROUP_FIELDS = (
    "scene",
    "method",
    "output_dim",
    "seed_count",
    "seeds",
    "success_25cm_10deg_mean",
    "success_25cm_10deg_std",
    "success_25cm_10deg_best",
    "success_50cm_10deg_mean",
    "success_50cm_10deg_std",
    "success_50cm_10deg_best",
    "median_translation_error_m_mean",
    "median_translation_error_m_std",
    "median_translation_error_m_best",
    "median_rotation_error_deg_mean",
    "median_rotation_error_deg_std",
    "median_rotation_error_deg_best",
    "mean_pnp_inlier_patch_at_1_mean",
    "mean_pnp_inlier_patch_at_1_std",
    "mean_pnp_inlier_patch_at_1_best",
    "mean_pnp_inlier_patch_at_5_mean",
    "mean_pnp_inlier_patch_at_5_std",
    "mean_pnp_inlier_patch_at_5_best",
    "total_storage_bytes_mean",
    "total_elapsed_sec_mean",
    "train_eval_top1_acc_mean",
    "train_final_loss_mean",
)


def _parse_run(text: str) -> tuple[str, str, int, int, Path, Path, Path]:
    parts = [part.strip() for part in text.split(",", 6)]
    if len(parts) != 7 or not all(parts[:4]) or not all(parts[4:]):
        raise ValueError("--run must be formatted as scene,method,dim,seed,train_summary,compression_summary,evaluation_summary")
    return parts[0], parts[1], int(parts[2]), int(parts[3]), Path(parts[4]), Path(parts[5]), Path(parts[6])


def _load_json(path: Path) -> dict:
    if str(path) == "-":
        return {}
    return json.loads(Path(path).read_text())


def _metric(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values)
    assert mean is not None
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _best(values: list[float], lower_is_better: bool = False) -> float | None:
    if not values:
        return None
    return float(min(values) if lower_is_better else max(values))


def _run_row(
    scene: str,
    method: str,
    output_dim: int,
    seed: int,
    train_path: Path,
    compression_path: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    train = _load_json(train_path)
    compression = _load_json(compression_path)
    evaluation = _load_json(evaluation_path)
    storage = dict(compression.get("storage_bytes") or {})
    query_storage = int(storage.get("query_tokens", 0) or 0)
    landmark_storage = int(storage.get("landmark_bank", 0) or 0)
    transform_storage = int(storage.get("transform", 0) or 0)
    training = dict(train.get("training") or {})
    return {
        "scene": scene,
        "method": method,
        "output_dim": int(output_dim),
        "seed": int(seed),
        "query_count": evaluation.get("query_count"),
        "success_25cm_10deg": evaluation.get("success_25cm_10deg"),
        "success_50cm_10deg": evaluation.get("success_50cm_10deg"),
        "success_1m_10deg": evaluation.get("success_1m_10deg"),
        "median_translation_error_m": evaluation.get("median_translation_error_m"),
        "median_rotation_error_deg": evaluation.get("median_rotation_error_deg"),
        "mean_pnp_inlier_patch_at_1": evaluation.get("mean_pnp_inlier_patch_at_1"),
        "mean_pnp_inlier_patch_at_5": evaluation.get("mean_pnp_inlier_patch_at_5"),
        "mean_pnp_inlier_count": evaluation.get("mean_pnp_inlier_count"),
        "mean_match_count": evaluation.get("mean_match_count"),
        "query_storage_bytes": query_storage,
        "landmark_storage_bytes": landmark_storage,
        "transform_storage_bytes": transform_storage,
        "total_storage_bytes": query_storage + landmark_storage + transform_storage,
        "train_elapsed_sec": train.get("elapsed_sec", 0.0),
        "compression_elapsed_sec": compression.get("elapsed_sec", 0.0),
        "evaluation_elapsed_sec": evaluation.get("elapsed_sec", 0.0),
        "total_elapsed_sec": float(train.get("elapsed_sec", 0.0) or 0.0)
        + float(compression.get("elapsed_sec", 0.0) or 0.0)
        + float(evaluation.get("elapsed_sec", 0.0) or 0.0),
        "train_sample_count": training.get("sample_count", train.get("sample_summary", {}).get("sample_count")),
        "train_eval_top1_acc": training.get("eval_top1_acc"),
        "train_final_loss": training.get("final_loss"),
        "train_summary_path": "" if str(train_path) == "-" else str(train_path),
        "compression_summary_path": str(compression_path),
        "evaluation_summary_path": str(evaluation_path),
    }


def _group_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scene"]), str(row["method"]), int(row["output_dim"]))].append(row)
    summaries = []
    for (scene, method, output_dim), group in sorted(grouped.items()):
        seeds = sorted(int(row["seed"]) for row in group)
        summary: dict[str, object] = {
            "scene": scene,
            "method": method,
            "output_dim": int(output_dim),
            "seed_count": len(group),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for key in (
            "success_25cm_10deg",
            "success_50cm_10deg",
            "median_translation_error_m",
            "median_rotation_error_deg",
            "mean_pnp_inlier_patch_at_1",
            "mean_pnp_inlier_patch_at_5",
            "total_storage_bytes",
            "total_elapsed_sec",
            "train_eval_top1_acc",
            "train_final_loss",
        ):
            values = [_metric(row, key) for row in group]
            numeric = [value for value in values if value is not None]
            summary[f"{key}_mean"] = _mean(numeric)
            if key in {
                "success_25cm_10deg",
                "success_50cm_10deg",
                "median_translation_error_m",
                "median_rotation_error_deg",
                "mean_pnp_inlier_patch_at_1",
                "mean_pnp_inlier_patch_at_5",
            }:
                summary[f"{key}_std"] = _std(numeric)
                summary[f"{key}_best"] = _best(
                    numeric,
                    lower_is_better=key in {"median_translation_error_m", "median_rotation_error_deg"},
                )
        summaries.append(summary)
    return summaries


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(rows: Sequence[dict[str, object]], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_markdown(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scene",
        "method",
        "output_dim",
        "seed_count",
        "success_25cm_10deg_mean",
        "success_25cm_10deg_std",
        "success_25cm_10deg_best",
        "success_50cm_10deg_mean",
        "median_translation_error_m_mean",
        "median_translation_error_m_best",
        "mean_pnp_inlier_patch_at_1_mean",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage C1 patch selector runs")
    parser.add_argument("--run", action="append", required=True, help="scene,method,dim,seed,train_summary,compression_summary,evaluation_summary")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    rows = [_run_row(*_parse_run(item)) for item in args.run]
    groups = _group_rows(rows)
    report = {"stage": "stage_c1_patch_selector_summary", "rows": rows, "groups": groups}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        _write_csv(groups, Path(args.output_csv), GROUP_FIELDS)
    if args.output_md:
        _write_markdown(groups, Path(args.output_md))


if __name__ == "__main__":
    main()
