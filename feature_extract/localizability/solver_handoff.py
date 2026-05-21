"""Solver-conditioned handoff metrics for POFD-FS candidate ranking."""

from __future__ import annotations

from collections import OrderedDict
from statistics import mean, median
from typing import Iterable


def _group_rows(rows: Iterable[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["sample_name"]), []).append(row)
    return grouped


def _valid(row: dict) -> bool:
    return bool(row.get("valid", True))


def _score(row: dict) -> float:
    return float(row.get("score", 0.0))


def _candidate_idx(row: dict) -> int:
    return int(row.get("candidate_idx", -1))


def _select_from_topk(candidates: list[dict], mode: str) -> dict:
    if not candidates:
        raise ValueError("No candidates available for handoff selection")
    if mode == "pofd_score":
        return candidates[0]
    if mode == "oracle_pose":
        return min(candidates, key=lambda row: float(row["pose_cost_m"]))
    if mode == "pnp_inliers":
        return max(
            candidates,
            key=lambda row: (
                float(row.get("retrieval_pnp_success_candidates", 1.0)),
                float(row.get("retrieval_pnp_num_inliers_candidates", 0.0)),
                _score(row),
            ),
        )
    if mode == "pnp_reproj_median":
        return min(
            candidates,
            key=lambda row: (
                -float(row.get("retrieval_pnp_success_candidates", 1.0)),
                float(row.get("retrieval_pnp_reproj_median_candidates", row.get("retrieval_pnp_reproj_rmse_candidates", 1.0e9))),
                -_score(row),
            ),
        )
    raise ValueError(f"Unsupported handoff selection mode: {mode}")


def select_handoff_rows(
    rows: Iterable[dict],
    *,
    topk: int = 1,
    selection_mode: str = "pofd_score",
) -> list[dict]:
    """Select one handoff candidate per sample from a candidate table."""
    grouped = _group_rows(rows)
    selected_rows = []
    for sample_rows in grouped.values():
        valid_rows = [row for row in sample_rows if _valid(row)]
        ranked = sorted(valid_rows, key=_score, reverse=True)
        if not ranked:
            continue
        candidates = ranked[: max(1, int(topk))]
        selected_rows.append(_select_from_topk(candidates, selection_mode))
    return selected_rows


def evaluate_handoff_rows(
    rows: Iterable[dict],
    *,
    topk: int = 1,
    selection_mode: str = "pofd_score",
    success_thresholds: tuple[tuple[float, float], ...] = ((0.10, 5.0), (0.25, 10.0), (0.50, 10.0)),
) -> dict:
    """Evaluate final pose metrics after selecting a candidate from POFD topK.

    The rows are expected to describe already-computed solver candidates.  This
    function does not run a solver; it evaluates a fixed handoff protocol over a
    candidate table.
    """
    selected_rows = select_handoff_rows(rows, topk=topk, selection_mode=selection_mode)
    if not selected_rows:
        raise ValueError("No valid handoff candidates were selected")

    trans = [float(row.get("trans_err_m", row["pose_cost_m"])) for row in selected_rows]
    rot = [float(row.get("rot_err_deg", 0.0)) for row in selected_rows]
    cost = [float(row["pose_cost_m"]) for row in selected_rows]
    out = {
        "num_samples": len(selected_rows),
        "topk": int(topk),
        "selection_mode": str(selection_mode),
        "cost_mean_m": mean(cost),
        "cost_median_m": median(cost),
        "trans_mean_m": mean(trans),
        "trans_median_m": median(trans),
        "rot_mean_deg": mean(rot),
        "rot_median_deg": median(rot),
        "selected_candidate_indices": [_candidate_idx(row) for row in selected_rows],
    }
    if any("retrieval_pnp_success_candidates" in row for row in selected_rows):
        out["pnp_success_frac"] = mean(float(row.get("retrieval_pnp_success_candidates", 0.0)) > 0.0 for row in selected_rows)
    for trans_thresh, rot_thresh in success_thresholds:
        key = f"success_{int(round(trans_thresh * 100))}cm_{int(round(rot_thresh))}deg"
        out[key] = mean((t <= float(trans_thresh)) and (r <= float(rot_thresh)) for t, r in zip(trans, rot))
    return out
