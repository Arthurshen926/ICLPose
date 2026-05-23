"""Protocol metadata for POFD-FS localization evidence reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .bank_schema import forbidden_training_input_fields, validate_no_forbidden_training_inputs


CONTROLLED_LATTICE = "controlled_lattice"
REFERENCE_POSE = "reference_pose"
REAL_RETRIEVAL = "real_retrieval"

VALID_PROTOCOLS = frozenset({CONTROLLED_LATTICE, REFERENCE_POSE, REAL_RETRIEVAL})
VALID_GT_USAGE = frozenset({"none", "eval_only", "candidate_generation_only", "candidate_generation_and_eval"})


@dataclass(frozen=True)
class ArtifactProtocol:
    """Reproducibility metadata for one candidate-ranking or handoff artifact."""

    protocol: str
    candidate_generator: str
    split: str
    input_fields: Sequence[str] = ()
    gt_usage: str = "eval_only"
    solver_conditioned: bool = False
    scene: str | None = None
    artifact_path: str | None = None
    notes: str | None = None
    extras: Mapping[str, object] = field(default_factory=dict)

    @property
    def gt_centered(self) -> bool:
        return self.protocol == CONTROLLED_LATTICE or self.gt_usage.startswith("candidate_generation")

    @property
    def deployment_claim_allowed(self) -> bool:
        return not self.gt_centered and self.protocol in {REFERENCE_POSE, REAL_RETRIEVAL}

    @property
    def claim_group(self) -> str:
        if self.solver_conditioned:
            return f"{self.protocol}:solver_conditioned"
        return f"{self.protocol}:solver_free"

    def to_report_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "protocol": str(self.protocol),
            "candidate_generator": str(self.candidate_generator),
            "split": str(self.split),
            "gt_usage": str(self.gt_usage),
            "gt_centered": bool(self.gt_centered),
            "solver_conditioned": bool(self.solver_conditioned),
            "deployment_claim_allowed": bool(self.deployment_claim_allowed),
            "input_fields": list(self.input_fields),
            "forbidden_training_inputs": forbidden_training_input_fields(self.input_fields),
            "claim_group": self.claim_group,
        }
        if self.scene is not None:
            row["scene"] = str(self.scene)
        if self.artifact_path is not None:
            row["artifact_path"] = str(self.artifact_path)
        if self.notes:
            row["notes"] = str(self.notes)
        row.update({str(key): value for key, value in dict(self.extras).items()})
        return row


def validate_protocol_metadata(protocol: ArtifactProtocol, *, training_input: bool = False) -> None:
    """Validate protocol metadata and optionally enforce training-input leakage rules."""

    if protocol.protocol not in VALID_PROTOCOLS:
        raise ValueError(f"Unsupported POFD-FS protocol: {protocol.protocol}")
    if protocol.gt_usage not in VALID_GT_USAGE:
        raise ValueError(f"Unsupported POFD-FS gt_usage: {protocol.gt_usage}")
    if protocol.protocol == CONTROLLED_LATTICE and protocol.gt_usage == "none":
        raise ValueError("controlled_lattice artifacts must declare GT use for candidate generation/evaluation")
    if protocol.protocol == REAL_RETRIEVAL and protocol.gt_usage.startswith("candidate_generation"):
        raise ValueError("real_retrieval artifacts cannot use GT for candidate generation")
    if training_input:
        validate_no_forbidden_training_inputs(protocol.input_fields)


def assert_protocol_claims_compatible(
    protocols: Iterable[ArtifactProtocol],
    *,
    allow_mixed: bool = False,
) -> None:
    """Prevent accidentally mixing controlled, reference, and real-init rows in one claim."""

    rows = list(protocols)
    for protocol in rows:
        validate_protocol_metadata(protocol, training_input=False)
    groups = {protocol.claim_group for protocol in rows}
    if len(groups) > 1 and not allow_mixed:
        joined = ", ".join(sorted(groups))
        raise ValueError(f"mixed protocol claim groups require explicit opt-in: {joined}")


__all__ = [
    "ArtifactProtocol",
    "CONTROLLED_LATTICE",
    "REAL_RETRIEVAL",
    "REFERENCE_POSE",
    "VALID_GT_USAGE",
    "VALID_PROTOCOLS",
    "assert_protocol_claims_compatible",
    "validate_protocol_metadata",
]
