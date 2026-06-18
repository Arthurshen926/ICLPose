"""Patch-level token MaxSim reranking for VPR candidates."""

from __future__ import annotations

from dataclasses import replace
import zlib

import numpy as np

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


def _manifest_index(manifest: TokenBankManifest) -> dict[str, TokenBankRecord]:
    manifest.validate(verify_checksums=False)
    return {record.image_id: record for record in manifest.records}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-6)


def _record_tokens(
    record: TokenBankRecord,
    layer_name: str,
    *,
    max_tokens_per_image: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    with np.load(record.token_path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
        feature = np.asarray(data[layer_name], dtype=np.float32)
    tokens = feature.reshape(feature.shape[0], -1).T
    max_tokens = int(max_tokens_per_image)
    if max_tokens < 0:
        raise ValueError("max_tokens_per_image must be non-negative")
    if max_tokens > 0 and tokens.shape[0] > max_tokens:
        generator = rng if rng is not None else np.random.default_rng(0)
        choice = generator.choice(tokens.shape[0], size=max_tokens, replace=False)
        tokens = tokens[choice]
    return _normalize_rows(tokens.astype(np.float32, copy=False))


def token_maxsim_score(query_tokens: np.ndarray, reference_tokens: np.ndarray, *, symmetric: bool = True) -> float:
    if query_tokens.ndim != 2 or reference_tokens.ndim != 2:
        raise ValueError("tokens must have shape (N, C)")
    if query_tokens.shape[1] != reference_tokens.shape[1]:
        raise ValueError("query and reference token dimensions must match")
    scores = query_tokens @ reference_tokens.T
    q_to_r = float(np.mean(np.max(scores, axis=1)))
    if not symmetric:
        return q_to_r
    r_to_q = float(np.mean(np.max(scores, axis=0)))
    return 0.5 * (q_to_r + r_to_q)


def rerank_candidate_bank_by_token_maxsim(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
    layer_name: str,
    top_k_per_query: int = 100,
    symmetric: bool = True,
    max_tokens_per_image: int = 0,
    seed: int = 0,
) -> CandidateHypothesisBank:
    if int(top_k_per_query) <= 0:
        raise ValueError("top_k_per_query must be positive")
    if int(max_tokens_per_image) < 0:
        raise ValueError("max_tokens_per_image must be non-negative")
    query_index = _manifest_index(query_manifest)
    map_index = _manifest_index(map_manifest)
    token_cache: dict[tuple[str, str], np.ndarray] = {}

    def get_tokens(image_id: str, source: str) -> np.ndarray:
        key = (source, image_id)
        if key in token_cache:
            return token_cache[key]
        records = query_index if source == "query" else map_index
        if image_id not in records:
            raise ValueError(f"{source} token record not found: {image_id}")
        stable_offset = zlib.adler32(f"{source}:{image_id}".encode("utf-8"))
        tokens = _record_tokens(
            records[image_id],
            layer_name,
            max_tokens_per_image=int(max_tokens_per_image),
            rng=np.random.default_rng(int(seed) + int(stable_offset)),
        )
        token_cache[key] = tokens
        return tokens

    grouped: dict[str, list] = {}
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        grouped.setdefault(candidate.query_id, []).append(candidate)

    reranked = []
    for query_id in sorted(grouped):
        query_tokens = get_tokens(query_id, "query")
        scored = []
        for order, candidate in enumerate(grouped[query_id][: int(top_k_per_query)]):
            reference_tokens = get_tokens(candidate.reference_image, "map")
            score = token_maxsim_score(query_tokens, reference_tokens, symmetric=bool(symmetric))
            scored.append((score, order, candidate))
        scored.sort(key=lambda item: (-item[0], item[1], item[2].candidate_id))
        for rank, (score, _order, candidate) in enumerate(scored[: int(top_k_per_query)], start=1):
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "reranker": "token_maxsim",
                    "rerank_rank": rank,
                    "token_maxsim_score": float(score),
                    "token_maxsim_symmetric": bool(symmetric),
                    "source_protocol_name": bank.protocol_name,
                }
            )
            if int(max_tokens_per_image) > 0:
                metadata["token_maxsim_max_tokens_per_image"] = int(max_tokens_per_image)
            reranked.append(replace(candidate, prior_score=float(score), metadata=metadata))
    return CandidateHypothesisBank.from_candidates(
        protocol_name=f"{bank.protocol_name}_token_maxsim",
        protocol_kind=bank.protocol_kind,
        candidates=reranked,
        protocol_fingerprint=bank.protocol_fingerprint,
    )
