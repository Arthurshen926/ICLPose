"""Dataset indexing helpers for selector and verifier training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


@dataclass(frozen=True)
class TokenCandidateExample:
    query_id: str
    query_tokens: TokenBankRecord
    candidates: Tuple[CandidateHypothesis, ...]


@dataclass(frozen=True)
class TokenCandidateIndex:
    examples: Tuple[TokenCandidateExample, ...]
    protocol_name: str

    @property
    def query_count(self) -> int:
        return len(self.examples)


def build_token_candidate_index(
    token_manifest: TokenBankManifest,
    candidate_bank: CandidateHypothesisBank,
    require_pose_labels: bool = False,
) -> TokenCandidateIndex:
    token_manifest.validate(verify_checksums=False)
    token_by_id: Dict[str, TokenBankRecord] = {
        record.image_id: record for record in token_manifest.records
    }
    grouped: Dict[str, list[CandidateHypothesis]] = {}
    for candidate in candidate_bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if require_pose_labels and candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        grouped.setdefault(candidate.query_id, []).append(candidate)

    examples: list[TokenCandidateExample] = []
    for query_id in sorted(grouped):
        if query_id not in token_by_id:
            raise ValueError(f"query_id {query_id} is not present in token manifest")
        examples.append(
            TokenCandidateExample(
                query_id=query_id,
                query_tokens=token_by_id[query_id],
                candidates=tuple(grouped[query_id]),
            )
        )
    return TokenCandidateIndex(
        examples=tuple(examples),
        protocol_name=candidate_bank.protocol_name,
    )
