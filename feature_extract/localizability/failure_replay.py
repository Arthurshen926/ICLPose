"""Failure-conditioned replay helpers for POFD-FS ranking training."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FailureReplayRow:
    sample_name: str
    positive_idx: int
    negative_idx: int
    oracle_gap_m: float = 0.0
    selected_cost_m: float | None = None
    oracle_cost_m: float | None = None


def _normalize_name(name: str) -> str:
    return str(name).replace("\\", "/")


def _name_aliases(name: str) -> tuple[str, ...]:
    norm = _normalize_name(name)
    aliases = [norm]
    if norm.startswith("images/"):
        aliases.append(norm[len("images/") :])
    path = Path(norm)
    aliases.append(path.name)
    aliases.append(str(path.with_suffix("")))
    aliases.append(path.stem)
    seen = set()
    ordered = []
    for alias in aliases:
        if alias and alias not in seen:
            ordered.append(alias)
            seen.add(alias)
    return tuple(ordered)


def _is_wrong_top1(row: Mapping) -> bool:
    if "top1" in row:
        return not bool(row["top1"])
    if "selected_idx" in row and "oracle_idx" in row:
        return int(row["selected_idx"]) != int(row["oracle_idx"])
    return False


def _from_rows(
    rows: Iterable[Mapping],
    *,
    require_wrong_top1: bool = True,
    min_gap_m: float = 0.0,
) -> "OrderedDict[str, FailureReplayRow]":
    replay: "OrderedDict[str, FailureReplayRow]" = OrderedDict()
    for row in rows:
        if require_wrong_top1 and not _is_wrong_top1(row):
            continue
        gap = float(row.get("oracle_gap_m", 0.0))
        if gap < float(min_gap_m):
            continue
        sample_name = _normalize_name(str(row["sample_name"]))
        replay[sample_name] = FailureReplayRow(
            sample_name=sample_name,
            positive_idx=int(row["oracle_idx"]),
            negative_idx=int(row["selected_idx"]),
            oracle_gap_m=gap,
            selected_cost_m=float(row["selected_cost_m"]) if "selected_cost_m" in row else None,
            oracle_cost_m=float(row["oracle_cost_m"]) if "oracle_cost_m" in row else None,
        )
    return replay


def load_failure_replay_rows(
    path: str | Path,
    *,
    require_wrong_top1: bool = True,
    min_gap_m: float = 0.0,
) -> "OrderedDict[str, FailureReplayRow]":
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return _from_rows(rows, require_wrong_top1=require_wrong_top1, min_gap_m=min_gap_m)


load_failure_replay_rows.from_rows = _from_rows  # type: ignore[attr-defined]


def load_mined_score_hard_pairs(path: str | Path) -> "OrderedDict[str, FailureReplayRow]":
    replay: "OrderedDict[str, FailureReplayRow]" = OrderedDict()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sample_name = _normalize_name(str(row["sample_name"]))
            replay[sample_name] = FailureReplayRow(
                sample_name=sample_name,
                positive_idx=int(row["positive_idx"]),
                negative_idx=int(row["negative_idx"]),
                oracle_gap_m=float(row.get("cost_gap_m", row.get("oracle_gap_m", 0.0))),
                selected_cost_m=float(row["negative_cost_m"]) if "negative_cost_m" in row else None,
                oracle_cost_m=float(row["positive_cost_m"]) if "positive_cost_m" in row else None,
            )
    return replay


def batch_failure_pairs(
    sample_names: Iterable[str],
    replay_rows: Mapping[str, FailureReplayRow],
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lookup: dict[str, FailureReplayRow] = {}
    for key, row in replay_rows.items():
        for alias in _name_aliases(key):
            lookup.setdefault(alias, row)

    active = []
    positive = []
    negative = []
    for name in sample_names:
        row = None
        for alias in _name_aliases(str(name)):
            row = lookup.get(alias)
            if row is not None:
                break
        active.append(row is not None)
        positive.append(int(row.positive_idx) if row is not None else 0)
        negative.append(int(row.negative_idx) if row is not None else 0)

    return (
        torch.as_tensor(active, dtype=torch.bool, device=device),
        torch.as_tensor(positive, dtype=torch.long, device=device),
        torch.as_tensor(negative, dtype=torch.long, device=device),
    )


def cached_failure_pair_margin_loss(
    scores: torch.Tensor,
    *,
    positive_idx: torch.Tensor,
    negative_idx: torch.Tensor,
    active_mask: torch.Tensor,
    margin: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (B,K)")
    bsz, num_candidates = scores.shape
    if positive_idx.shape != (bsz,) or negative_idx.shape != (bsz,) or active_mask.shape != (bsz,):
        raise ValueError("positive_idx, negative_idx, and active_mask must have shape (B,)")
    active = active_mask.bool()
    if not active.any():
        return scores.sum() * 0.0, {"failure_pair_active_frac": active.float().mean()}

    if positive_idx[active].min() < 0 or negative_idx[active].min() < 0:
        raise ValueError("failure replay candidate indices must be non-negative")
    if positive_idx[active].max() >= num_candidates or negative_idx[active].max() >= num_candidates:
        raise ValueError("failure replay candidate index exceeds scores.shape[1]")

    row_idx = torch.arange(bsz, device=scores.device)
    pos_scores = scores[row_idx[active], positive_idx[active]]
    neg_scores = scores[row_idx[active], negative_idx[active]]
    loss = F.softplus(neg_scores.float() - pos_scores.float() + float(margin)).mean()
    return loss, {"failure_pair_active_frac": active.float().mean()}


def summarize_candidate_failure_modes(
    rows: Iterable[Mapping],
    *,
    near_identity_trans_m: float = 0.05,
    high_score_margin: float = 0.5,
    min_oracle_gap_m: float = 0.05,
) -> dict:
    """Summarize wrong-top1 modes directly from a candidate score table."""
    grouped: "OrderedDict[str, list[Mapping]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(_normalize_name(str(row["sample_name"])), []).append(row)

    wrong = 0
    wrong_near_identity = 0
    wrong_outside_basin = 0
    wrong_high_margin = 0
    replay_candidates: list[str] = []
    selected_cost_sum = 0.0
    oracle_cost_sum = 0.0
    for sample_name, sample_rows in grouped.items():
        valid_rows = [r for r in sample_rows if bool(r.get("valid", True))]
        if not valid_rows:
            continue
        selected = max(valid_rows, key=lambda r: float(r.get("score", 0.0)))
        oracle = min(valid_rows, key=lambda r: float(r.get("pose_cost_m", float("inf"))))
        selected_idx = int(selected.get("candidate_idx", -1))
        oracle_idx = int(oracle.get("candidate_idx", -1))
        selected_cost = float(selected.get("pose_cost_m", float("inf")))
        oracle_cost = float(oracle.get("pose_cost_m", float("inf")))
        gap = selected_cost - oracle_cost
        if selected_idx == oracle_idx:
            continue
        wrong += 1
        selected_cost_sum += selected_cost
        oracle_cost_sum += oracle_cost
        if float(selected.get("delta_trans_m", float("inf"))) <= float(near_identity_trans_m):
            wrong_near_identity += 1
        if not bool(selected.get("in_basin", False)):
            wrong_outside_basin += 1
        score_margin = float(selected.get("score", 0.0)) - float(oracle.get("score", 0.0))
        if score_margin >= float(high_score_margin):
            wrong_high_margin += 1
        if gap >= float(min_oracle_gap_m):
            replay_candidates.append(sample_name)

    denom = max(wrong, 1)
    return {
        "num_samples": len(grouped),
        "wrong_top1": wrong,
        "wrong_fraction": wrong / max(len(grouped), 1),
        "wrong_near_identity": wrong_near_identity,
        "wrong_outside_basin": wrong_outside_basin,
        "wrong_high_margin": wrong_high_margin,
        "wrong_selected_mean_cost_m": selected_cost_sum / denom,
        "wrong_oracle_mean_cost_m": oracle_cost_sum / denom,
        "replay_candidates": replay_candidates,
        "num_replay_candidates": len(replay_candidates),
    }


def mine_score_hard_candidate_pairs(
    rows: Iterable[Mapping],
    *,
    cost_gap_m: float = 0.12,
    near_identity_trans_m: float = 0.05,
) -> tuple[list[dict], dict]:
    """Mine oracle-vs-score-high-wrong candidate pairs from score-table rows."""
    grouped: "OrderedDict[str, list[Mapping]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(_normalize_name(str(row["sample_name"])), []).append(row)

    pairs: list[dict] = []
    near_identity_count = 0
    outside_basin_count = 0
    score_margin_positive_count = 0
    cost_gap_sum = 0.0
    score_margin_sum = 0.0
    for sample_name, sample_rows in grouped.items():
        valid_rows = [row for row in sample_rows if bool(row.get("valid", True))]
        if len(valid_rows) < 2:
            continue
        positive = min(valid_rows, key=lambda row: float(row.get("pose_cost_m", float("inf"))))
        positive_cost = float(positive.get("pose_cost_m", float("inf")))
        negative_pool = [
            row
            for row in valid_rows
            if int(row.get("candidate_idx", -1)) != int(positive.get("candidate_idx", -1))
            and float(row.get("pose_cost_m", float("inf"))) >= positive_cost + float(cost_gap_m)
        ]
        if not negative_pool:
            continue
        negative = max(negative_pool, key=lambda row: float(row.get("score", 0.0)))
        negative_cost = float(negative.get("pose_cost_m", float("inf")))
        cost_gap = negative_cost - positive_cost
        score_margin = float(negative.get("score", 0.0)) - float(positive.get("score", 0.0))
        negative_is_near_identity = (
            float(negative.get("delta_trans_m", float("inf"))) <= float(near_identity_trans_m)
        )
        negative_in_basin = bool(negative.get("in_basin", False))
        if negative_is_near_identity:
            near_identity_count += 1
        if not negative_in_basin:
            outside_basin_count += 1
        if score_margin > 0.0:
            score_margin_positive_count += 1
        cost_gap_sum += cost_gap
        score_margin_sum += score_margin
        pairs.append(
            {
                "sample_name": sample_name,
                "positive_idx": int(positive["candidate_idx"]),
                "negative_idx": int(negative["candidate_idx"]),
                "positive_cost_m": positive_cost,
                "negative_cost_m": negative_cost,
                "cost_gap_m": cost_gap,
                "positive_score": float(positive.get("score", 0.0)),
                "negative_score": float(negative.get("score", 0.0)),
                "score_margin": score_margin,
                "negative_is_near_identity": bool(negative_is_near_identity),
                "negative_in_basin": bool(negative_in_basin),
            }
        )

    denom = max(len(pairs), 1)
    summary = {
        "num_samples": len(grouped),
        "num_pairs": len(pairs),
        "pair_fraction": len(pairs) / max(len(grouped), 1),
        "near_identity_negative_pairs": near_identity_count,
        "outside_basin_negative_pairs": outside_basin_count,
        "score_margin_positive_pairs": score_margin_positive_count,
        "mean_cost_gap_m": cost_gap_sum / denom,
        "mean_score_margin": score_margin_sum / denom,
    }
    return pairs, summary
