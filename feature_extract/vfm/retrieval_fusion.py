"""Rank-fusion utilities for reference-pose VPR candidate banks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind


def _rank_value(candidate: CandidateHypothesis, fallback_rank: int) -> int:
    value = candidate.metadata.get("retrieval_rank")
    if value is None:
        return int(fallback_rank)
    return int(value)


def _source_name(bank: CandidateHypothesisBank, index: int, used: set[str]) -> str:
    base = str(bank.protocol_name) or f"bank_{index}"
    name = base
    suffix = 1
    while name in used:
        suffix += 1
        name = f"{base}_{suffix}"
    used.add(name)
    return name


def _query_key(candidate: CandidateHypothesis) -> str:
    if candidate.query_id is None:
        raise ValueError("reference-pose fusion requires every candidate to have query_id")
    return str(candidate.query_id)


def _reference_key(candidate: CandidateHypothesis) -> str:
    if candidate.reference_image is not None:
        return str(candidate.reference_image)
    return str(candidate.candidate_id)


def _safe_query_stem(query_id: str) -> str:
    return (
        str(query_id)
        .replace("\\", "__")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def fuse_reference_pose_banks(
    banks: Sequence[CandidateHypothesisBank],
    *,
    protocol_name: str,
    top_k: int = 10,
    rrf_k: float = 60.0,
    bank_weights: Sequence[float] | None = None,
) -> CandidateHypothesisBank:
    """Fuse VPR reference-pose banks with reciprocal-rank fusion.

    The fusion score uses only retrieval ranks and optional bank weights. Pose
    error labels are copied through for later benchmark reporting, but they are
    not used to rank candidates.
    """

    if not banks:
        raise ValueError("at least one candidate bank is required")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if float(rrf_k) <= 0.0:
        raise ValueError("rrf_k must be positive")
    weights = [1.0 for _ in banks] if bank_weights is None else [float(value) for value in bank_weights]
    if len(weights) != len(banks):
        raise ValueError("bank_weights must contain one value per bank")
    if any(value < 0.0 for value in weights):
        raise ValueError("bank_weights must be non-negative")
    for bank in banks:
        if bank.protocol_kind != ProtocolKind.REFERENCE_POSE:
            raise ValueError("all banks must use ProtocolKind.REFERENCE_POSE")

    used_sources: set[str] = set()
    source_names = [_source_name(bank, idx, used_sources) for idx, bank in enumerate(banks)]
    by_query: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)

    for bank, source, weight in zip(banks, source_names, weights):
        grouped: dict[str, list[tuple[int, CandidateHypothesis]]] = defaultdict(list)
        for fallback_rank, candidate in enumerate(bank.candidates, start=1):
            grouped[_query_key(candidate)].append((fallback_rank, candidate))
        for query_id, items in grouped.items():
            ordered = sorted(
                items,
                key=lambda item: (
                    _rank_value(item[1], item[0]),
                    -float(item[1].prior_score if item[1].prior_score is not None else 0.0),
                    _reference_key(item[1]),
                ),
            )
            for fallback_rank, candidate in ordered:
                rank = _rank_value(candidate, fallback_rank)
                reference = _reference_key(candidate)
                entry = by_query[query_id].setdefault(
                    reference,
                    {
                        "candidate": candidate,
                        "best_source_rank": int(rank),
                        "best_source_score": float(candidate.prior_score if candidate.prior_score is not None else 0.0),
                        "score": 0.0,
                        "source_ranks": {},
                        "source_scores": {},
                    },
                )
                entry["score"] = float(entry["score"]) + float(weight) / (float(rrf_k) + float(rank))
                entry["source_ranks"][source] = int(rank)
                if candidate.prior_score is not None:
                    entry["source_scores"][source] = float(candidate.prior_score)
                candidate_score = float(candidate.prior_score if candidate.prior_score is not None else 0.0)
                if int(rank) < int(entry["best_source_rank"]) or (
                    int(rank) == int(entry["best_source_rank"]) and candidate_score > float(entry["best_source_score"])
                ):
                    entry["candidate"] = candidate
                    entry["best_source_rank"] = int(rank)
                    entry["best_source_score"] = candidate_score

    fused_candidates: list[CandidateHypothesis] = []
    for query_id in sorted(by_query):
        ranked = sorted(
            by_query[query_id].items(),
            key=lambda item: (
                -float(item[1]["score"]),
                min(int(rank) for rank in item[1]["source_ranks"].values()),
                item[0],
            ),
        )[: int(top_k)]
        stem = _safe_query_stem(query_id)
        for rank, (reference, entry) in enumerate(ranked, start=1):
            base = entry["candidate"]
            score = float(entry["score"])
            metadata = dict(base.metadata)
            metadata.update(
                {
                    "candidate_generator": "reference_pose_rrf_fusion",
                    "candidate_uses_gt": False,
                    "retrieval_rank": rank,
                    "rrf_score": score,
                    "rrf_k": float(rrf_k),
                    "source_protocols": list(source_names),
                    "source_ranks": dict(entry["source_ranks"]),
                    "source_scores": dict(entry["source_scores"]),
                }
            )
            fused_candidates.append(
                replace(
                    base,
                    candidate_id=f"{stem}:rrf_reference_pose:{rank - 1:03d}",
                    candidate_type="rrf_reference_pose",
                    reference_image=reference,
                    prior_score=score,
                    metadata=metadata,
                )
            )

    if not fused_candidates:
        raise ValueError("reference-pose fusion produced no candidates")
    return CandidateHypothesisBank.from_candidates(
        protocol_name=protocol_name,
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=fused_candidates,
    )


def fuse_reference_pose_bank_paths(
    paths: Iterable[Path],
    *,
    protocol_name: str,
    top_k: int = 10,
    rrf_k: float = 60.0,
    bank_weights: Sequence[float] | None = None,
) -> CandidateHypothesisBank:
    banks = [CandidateHypothesisBank.from_jsonl(Path(path)) for path in paths]
    return fuse_reference_pose_banks(
        banks,
        protocol_name=protocol_name,
        top_k=top_k,
        rrf_k=rrf_k,
        bank_weights=bank_weights,
    )
