"""Candidate hypothesis bank serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.protocols import ProtocolKind


def pose_cost_to_dict(cost: Optional[PoseCost]) -> Optional[dict]:
    if cost is None:
        return None
    return {
        "translation_m": cost.translation_m,
        "rotation_deg": cost.rotation_deg,
    }


def pose_cost_from_dict(data: Optional[Mapping[str, object]]) -> Optional[PoseCost]:
    if data is None:
        return None
    return PoseCost(
        translation_m=float(data["translation_m"]),
        rotation_deg=float(data["rotation_deg"]),
    )


def candidate_to_dict(candidate: CandidateHypothesis) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "pose_error": pose_cost_to_dict(candidate.pose_error),
        "prior_score": candidate.prior_score,
        "pose": candidate.pose,
        "reference_image": candidate.reference_image,
        "submap_id": candidate.submap_id,
        "solver_success": candidate.solver_success,
        "hard_case_type": candidate.hard_case_type,
        "query_id": candidate.query_id,
        "metadata": dict(candidate.metadata),
    }


def candidate_from_dict(data: Mapping[str, object]) -> CandidateHypothesis:
    return CandidateHypothesis(
        candidate_id=str(data["candidate_id"]),
        candidate_type=str(data["candidate_type"]),
        pose_error=pose_cost_from_dict(data.get("pose_error")),
        prior_score=None if data.get("prior_score") is None else float(data["prior_score"]),
        pose=data.get("pose"),
        reference_image=None if data.get("reference_image") is None else str(data["reference_image"]),
        submap_id=None if data.get("submap_id") is None else str(data["submap_id"]),
        solver_success=None if data.get("solver_success") is None else bool(data["solver_success"]),
        hard_case_type=None if data.get("hard_case_type") is None else str(data["hard_case_type"]),
        query_id=None if data.get("query_id") is None else str(data["query_id"]),
        metadata=dict(data.get("metadata", {})),
    )


@dataclass(frozen=True)
class CandidateHypothesisBank:
    protocol_name: str
    protocol_kind: ProtocolKind
    candidates: List[CandidateHypothesis]
    protocol_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_kind", ProtocolKind(self.protocol_kind))
        object.__setattr__(self, "candidates", list(self.candidates))

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "record_type": "header",
            "protocol_name": self.protocol_name,
            "protocol_kind": self.protocol_kind.value,
            "protocol_fingerprint": self.protocol_fingerprint,
        }
        lines = [json.dumps(header, sort_keys=True)]
        for candidate in self.candidates:
            record = candidate_to_dict(candidate)
            record["record_type"] = "candidate"
            lines.append(json.dumps(record, sort_keys=True))
        path.write_text("\n".join(lines) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path) -> "CandidateHypothesisBank":
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not lines or lines[0].get("record_type") != "header":
            raise ValueError("candidate bank jsonl must start with a header")
        header = lines[0]
        candidates = [
            candidate_from_dict(line)
            for line in lines[1:]
            if line.get("record_type") == "candidate"
        ]
        return cls(
            protocol_name=str(header["protocol_name"]),
            protocol_kind=ProtocolKind(header["protocol_kind"]),
            candidates=candidates,
            protocol_fingerprint=str(header.get("protocol_fingerprint", "")),
        )

    @classmethod
    def from_candidates(
        cls,
        protocol_name: str,
        protocol_kind: ProtocolKind,
        candidates: Iterable[CandidateHypothesis],
        protocol_fingerprint: str = "",
    ) -> "CandidateHypothesisBank":
        return cls(protocol_name, protocol_kind, list(candidates), protocol_fingerprint)
