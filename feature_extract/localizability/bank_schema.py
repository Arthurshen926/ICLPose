"""Shared schema and leakage guards for POFD-FS candidate banks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from feature_extract.localizability.candidate_bank import (
    CandidateBank,
    CandidateBankMetadata,
    candidate_bank_from_npz,
)


FORBIDDEN_TRAINING_INPUT_FIELDS = frozenset(
    {
        "basin_label",
        "candidate_cost_m",
        "candidate_rot_err_deg",
        "candidate_trans_err_m",
        "cost_m",
        "gt_pose",
        "gt_poses",
        "in_basin",
        "is_oracle",
        "oracle_cost_m",
        "oracle_gap_m",
        "oracle_idx",
        "oracle_score",
        "pose_cost_m",
        "pose_gt",
        "poses_gt",
        "retrieval_original_scores_candidates",
        "retrieval_scores_candidates",
        "rot_err_deg",
        "rot_errors_deg",
        "selected_cost_m",
        "selected_idx",
        "selected_in_basin",
        "selected_score",
        "trans_err_m",
        "trans_errors_m",
    }
)

FORBIDDEN_TRAINING_INPUT_PREFIXES = (
    "retrieval_pnp_",
    "oracle_",
    "selected_",
)


@dataclass(frozen=True)
class CandidateRow:
    """One scored hypothesis row from a candidate table JSONL file."""

    sample_name: str
    candidate_idx: int
    score: float | None = None
    pose_cost_m: float | None = None
    trans_err_m: float | None = None
    rot_err_deg: float | None = None
    valid: bool = True
    in_basin: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CandidateRow":
        known = {
            "sample_name",
            "candidate_idx",
            "score",
            "pose_cost_m",
            "trans_err_m",
            "rot_err_deg",
            "valid",
            "in_basin",
        }
        return cls(
            sample_name=str(row["sample_name"]),
            candidate_idx=int(row["candidate_idx"]),
            score=float(row["score"]) if "score" in row and row["score"] is not None else None,
            pose_cost_m=float(row["pose_cost_m"]) if "pose_cost_m" in row and row["pose_cost_m"] is not None else None,
            trans_err_m=float(row["trans_err_m"]) if "trans_err_m" in row and row["trans_err_m"] is not None else None,
            rot_err_deg=float(row["rot_err_deg"]) if "rot_err_deg" in row and row["rot_err_deg"] is not None else None,
            valid=bool(row.get("valid", True)),
            in_basin=bool(row["in_basin"]) if "in_basin" in row and row["in_basin"] is not None else None,
            extras={str(key): value for key, value in row.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sample_name": self.sample_name,
            "candidate_idx": int(self.candidate_idx),
            "valid": bool(self.valid),
        }
        if self.score is not None:
            row["score"] = float(self.score)
        if self.pose_cost_m is not None:
            row["pose_cost_m"] = float(self.pose_cost_m)
        if self.trans_err_m is not None:
            row["trans_err_m"] = float(self.trans_err_m)
        if self.rot_err_deg is not None:
            row["rot_err_deg"] = float(self.rot_err_deg)
        if self.in_basin is not None:
            row["in_basin"] = bool(self.in_basin)
        row.update(self.extras)
        return row


def candidate_rows_from_jsonl(path: str | Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(CandidateRow.from_dict(json.loads(stripped)))
    return rows


def is_forbidden_training_input_field(field_name: str) -> bool:
    name = str(field_name)
    lowered = name.lower()
    if lowered in FORBIDDEN_TRAINING_INPUT_FIELDS:
        return True
    return any(lowered.startswith(prefix) for prefix in FORBIDDEN_TRAINING_INPUT_PREFIXES)


def forbidden_training_input_fields(fields: Iterable[str]) -> list[str]:
    return [str(field) for field in fields if is_forbidden_training_input_field(str(field))]


def validate_no_forbidden_training_inputs(fields: Iterable[str]) -> None:
    forbidden = forbidden_training_input_fields(fields)
    if forbidden:
        joined = ", ".join(sorted(forbidden))
        raise ValueError(f"Forbidden POFD-FS training input field(s): {joined}")


__all__ = [
    "CandidateBank",
    "CandidateBankMetadata",
    "CandidateRow",
    "FORBIDDEN_TRAINING_INPUT_FIELDS",
    "candidate_bank_from_npz",
    "candidate_rows_from_jsonl",
    "forbidden_training_input_fields",
    "is_forbidden_training_input_field",
    "validate_no_forbidden_training_inputs",
]
