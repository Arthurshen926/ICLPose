"""Risk-coverage diagnostics for POFD-FS candidate selection tables."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from feature_extract.localizability.metrics import risk_coverage_metrics


def load_candidate_table_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_candidate_rows(rows: Iterable[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["sample_name"]), []).append(row)
    return grouped


def _is_valid(row: dict) -> bool:
    return bool(row.get("valid", True))


def _score(row: dict) -> float:
    return float(row.get("score", 0.0))


def _pose_cost(row: dict) -> float:
    return float(row.get("pose_cost_m", 0.0))


def _in_basin(row: dict, *, trans_basin_m: float | None, rot_basin_deg: float | None) -> bool:
    if trans_basin_m is not None and rot_basin_deg is not None and "trans_err_m" in row and "rot_err_deg" in row:
        return float(row["trans_err_m"]) <= float(trans_basin_m) and float(row["rot_err_deg"]) <= float(rot_basin_deg)
    return bool(row.get("in_basin", False))


def _confidence_from_ranked(ranked: list[dict], mode: str) -> float:
    if not ranked:
        return 0.0
    if mode == "top1_score":
        return _score(ranked[0])
    if mode == "top1_margin":
        if len(ranked) < 2:
            return 0.0
        return _score(ranked[0]) - _score(ranked[1])
    raise ValueError(f"Unsupported confidence mode: {mode}")


def summarize_candidate_table_risk(
    rows: Iterable[dict],
    *,
    topk: Sequence[int] = (1, 5),
    confidence_mode: str = "top1_margin",
    trans_basin_m: float | None = None,
    rot_basin_deg: float | None = None,
) -> dict:
    """Summarize candidate-selection success and failure-prediction risk."""
    grouped = group_candidate_rows(rows)
    confidences: list[float] = []
    selected_success: list[bool] = []
    selected_costs: list[float] = []
    oracle_costs: list[float] = []
    top1_acc: list[bool] = []
    basin_recall = {int(k): [] for k in topk}
    selected_rows: list[dict] = []
    for sample_rows in grouped.values():
        valid_rows = [row for row in sample_rows if _is_valid(row)]
        if not valid_rows:
            continue
        ranked = sorted(valid_rows, key=_score, reverse=True)
        oracle = min(valid_rows, key=_pose_cost)
        selected = ranked[0]
        selected_rows.append(selected)
        success = _in_basin(selected, trans_basin_m=trans_basin_m, rot_basin_deg=rot_basin_deg)
        confidences.append(_confidence_from_ranked(ranked, confidence_mode))
        selected_success.append(success)
        selected_costs.append(_pose_cost(selected))
        oracle_costs.append(_pose_cost(oracle))
        top1_acc.append(int(selected.get("candidate_idx", -1)) == int(oracle.get("candidate_idx", -2)))
        for k in basin_recall:
            top_rows = ranked[: max(1, int(k))]
            basin_recall[k].append(
                any(_in_basin(row, trans_basin_m=trans_basin_m, rot_basin_deg=rot_basin_deg) for row in top_rows)
            )

    if not selected_rows:
        raise ValueError("No valid candidate rows available for risk report")

    confidence_t = torch.as_tensor(confidences, dtype=torch.float32)
    success_t = torch.as_tensor(selected_success, dtype=torch.bool)
    risk = risk_coverage_metrics(confidence_t, success_t, coverages=(0.25, 0.50, 0.75, 1.0))
    selected_cost_t = torch.as_tensor(selected_costs, dtype=torch.float32)
    oracle_cost_t = torch.as_tensor(oracle_costs, dtype=torch.float32)
    summary = {
        "num_samples": int(len(selected_rows)),
        "confidence_mode": str(confidence_mode),
        "selected_success_rate": float(success_t.float().mean().item()),
        "selected_cost_mean_m": float(selected_cost_t.mean().item()),
        "selected_cost_median_m": float(selected_cost_t.median().item()),
        "oracle_cost_mean_m": float(oracle_cost_t.mean().item()),
        "oracle_gap_mean_m": float((selected_cost_t - oracle_cost_t).mean().item()),
        "top1_acc": float(torch.as_tensor(top1_acc, dtype=torch.float32).mean().item()),
    }
    for k, values in basin_recall.items():
        summary[f"basin_recall@{k}"] = float(torch.as_tensor(values, dtype=torch.float32).mean().item())
    for key, value in risk.items():
        summary[key] = float(value.detach().cpu())
    return summary
