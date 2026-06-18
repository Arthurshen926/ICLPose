"""Pose-cost benchmark summaries for reference-pose retrieval banks."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


def _candidate_rank(candidate: CandidateHypothesis, fallback: int) -> tuple[float, int, str]:
    rank = candidate.metadata.get("retrieval_rank") if candidate.metadata else None
    if rank is not None:
        try:
            return (float(rank), fallback, str(candidate.candidate_id))
        except (TypeError, ValueError):
            pass
    if candidate.prior_score is not None:
        return (-float(candidate.prior_score), fallback, str(candidate.candidate_id))
    return (float(fallback), fallback, str(candidate.candidate_id))


def _format_threshold_cm(value_m: float) -> str:
    cm = float(value_m) * 100.0
    if np.isclose(cm, round(cm)):
        return f"{int(round(cm))}cm"
    return f"{cm:g}cm"


def _format_threshold_deg(value_deg: float) -> str:
    deg = float(value_deg)
    if np.isclose(deg, round(deg)):
        return f"{int(round(deg))}deg"
    return f"{deg:g}deg"


def _parse_top_ks(top_ks: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in top_ks}))
    if not values or values[0] <= 0:
        raise ValueError("top_ks must contain positive integers")
    return values


def summarize_reference_pose_retrieval(
    bank: CandidateHypothesisBank,
    *,
    top_ks: Sequence[int] = (1, 5, 10),
    translation_threshold_m: float = 0.25,
    rotation_threshold_deg: float = 5.0,
    rot_cost_weight: float = 0.1,
) -> dict[str, object]:
    """Summarize Recall@K and best-in-top-K pose-cost for ranked candidates."""

    if float(translation_threshold_m) <= 0.0:
        raise ValueError("translation_threshold_m must be positive")
    if float(rotation_threshold_deg) <= 0.0:
        raise ValueError("rotation_threshold_deg must be positive")
    if float(rot_cost_weight) < 0.0:
        raise ValueError("rot_cost_weight must be non-negative")
    limits = _parse_top_ks(top_ks)
    grouped: dict[str, list[tuple[tuple[float, int, str], CandidateHypothesis]]] = defaultdict(list)
    for idx, candidate in enumerate(bank.candidates):
        if candidate.query_id is None or candidate.pose_error is None:
            continue
        grouped[str(candidate.query_id)].append((_candidate_rank(candidate, idx), candidate))
    if not grouped:
        raise ValueError("candidate bank has no query candidates with pose_error labels")
    ranked = {}
    for query_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        ranked[query_id] = [candidate for _rank, candidate in items]

    threshold_suffix = f"{_format_threshold_cm(translation_threshold_m)}_{_format_threshold_deg(rotation_threshold_deg)}"
    summary: dict[str, object] = {
        "protocol_name": bank.protocol_name,
        "protocol_kind": bank.protocol_kind.value,
        "query_count": int(len(ranked)),
        "candidate_count": int(sum(len(items) for items in ranked.values())),
        "translation_threshold_m": float(translation_threshold_m),
        "rotation_threshold_deg": float(rotation_threshold_deg),
        "rot_cost_weight": float(rot_cost_weight),
        "top_ks": list(limits),
    }
    for top_k in limits:
        recalls = []
        rank_t = []
        rank_r = []
        best_t = []
        best_r = []
        best_cost = []
        available = []
        for candidates in ranked.values():
            subset = candidates[: int(top_k)]
            available.append(len(subset))
            if not subset:
                recalls.append(False)
                continue
            rank_pose = subset[0].pose_error
            assert rank_pose is not None
            rank_t.append(float(rank_pose.translation_m))
            rank_r.append(float(rank_pose.rotation_deg))
            valid = []
            for candidate in subset:
                pose_error = candidate.pose_error
                assert pose_error is not None
                cost = float(pose_error.translation_m) + float(rot_cost_weight) * float(pose_error.rotation_deg)
                valid.append((cost, float(pose_error.translation_m), float(pose_error.rotation_deg), pose_error))
            cost, t_error, r_error, _pose = min(valid, key=lambda item: (item[0], item[1], item[2]))
            best_t.append(t_error)
            best_r.append(r_error)
            best_cost.append(cost)
            recalls.append(t_error <= float(translation_threshold_m) and r_error <= float(rotation_threshold_deg))
        key = f"recall_at_{top_k}_{threshold_suffix}"
        summary[key] = float(mean(1.0 if item else 0.0 for item in recalls))
        summary[f"top{top_k}_available_mean"] = float(mean(available))
        if rank_t:
            summary[f"top{top_k}_median_translation_m"] = float(median(rank_t))
            summary[f"top{top_k}_median_rotation_deg"] = float(median(rank_r))
        if best_t:
            summary[f"top{top_k}_best_median_translation_m"] = float(median(best_t))
            summary[f"top{top_k}_best_median_rotation_deg"] = float(median(best_r))
            summary[f"top{top_k}_best_median_pose_cost_m"] = float(median(best_cost))
    return summary
