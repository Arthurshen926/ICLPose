"""Rerank VPR candidate banks with token-level MaxSim scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.token_maxsim_reranking import rerank_candidate_bank_by_token_maxsim
from feature_extract.vfm.tokens import TokenBankManifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--map_manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--top_k_per_query", type=int, default=100)
    parser.add_argument("--max_tokens_per_image", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one_way", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reranked = rerank_candidate_bank_by_token_maxsim(
        bank=CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)),
        query_manifest=TokenBankManifest.from_json(Path(args.query_manifest)),
        map_manifest=TokenBankManifest.from_json(Path(args.map_manifest)),
        layer_name=str(args.layer_name),
        top_k_per_query=int(args.top_k_per_query),
        symmetric=not bool(args.one_way),
        max_tokens_per_image=int(args.max_tokens_per_image),
        seed=int(args.seed),
    )
    reranked.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
