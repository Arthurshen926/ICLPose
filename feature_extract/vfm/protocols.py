"""Evaluation protocol records and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Tuple


class ProtocolKind(str, Enum):
    CONTROLLED_LATTICE = "controlled_lattice"
    REFERENCE_POSE = "reference_pose"
    REAL_RETRIEVAL = "real_retrieval"
    RENDERED_POSE = "rendered_pose"


FORBIDDEN_TRAINING_INPUTS = {
    "gt_pose",
    "gt_cost",
    "oracle_rank",
    "oracle_cost",
    "pose_error",
    "solver_success_label",
}


@dataclass(frozen=True)
class EvaluationProtocol:
    """A reproducible hypothesis-verification protocol."""

    name: str
    kind: ProtocolKind
    split: str
    candidate_generator: str
    allowed_training_inputs: Tuple[str, ...]
    candidate_uses_gt: bool
    solver_conditioned: bool
    notes: str = ""
    gate: str = ""
    metadata_fields: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        kind = ProtocolKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "allowed_training_inputs", tuple(self.allowed_training_inputs))
        object.__setattr__(self, "metadata_fields", tuple(self.metadata_fields))
        if kind == ProtocolKind.CONTROLLED_LATTICE and not self.candidate_uses_gt:
            raise ValueError("controlled_lattice protocols must disclose GT-centered candidate generation")

    @property
    def deployment_like(self) -> bool:
        return self.kind == ProtocolKind.REAL_RETRIEVAL and not self.candidate_uses_gt


def validate_no_leakage(
    protocol: EvaluationProtocol,
    observed_training_inputs: Iterable[str],
) -> None:
    """Validate that training only uses whitelisted non-oracle inputs."""

    observed = set(observed_training_inputs)
    allowed = set(protocol.allowed_training_inputs)
    extra = observed - allowed
    forbidden = observed & FORBIDDEN_TRAINING_INPUTS
    if extra or forbidden:
        leaked = sorted(extra | forbidden)
        raise ValueError(
            f"training input leakage in protocol '{protocol.name}': {', '.join(leaked)}"
        )
