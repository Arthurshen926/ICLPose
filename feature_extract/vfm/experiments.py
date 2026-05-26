"""Experiment manifests and gate expectation checks."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Tuple

from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind, validate_no_leakage


@dataclass(frozen=True)
class GateExpectation:
    metric: str
    op: str
    value: float


@dataclass(frozen=True)
class VFMExperimentManifest:
    experiment_id: str
    protocol: EvaluationProtocol
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    seeds: Tuple[int, ...]
    output_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))

    def validate_training_inputs(self, observed_training_inputs: Iterable[str]) -> None:
        validate_no_leakage(self.protocol, observed_training_inputs)

    def to_dict(self) -> dict:
        protocol = self.protocol.to_dict()
        return {
            "experiment_id": self.experiment_id,
            "protocol": protocol,
            "protocol_fingerprint": self.protocol.fingerprint(),
            "methods": list(self.methods),
            "metrics": list(self.metrics),
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VFMExperimentManifest":
        protocol_data = dict(data["protocol"])
        protocol_data["kind"] = ProtocolKind(protocol_data["kind"])
        protocol = EvaluationProtocol(**protocol_data)
        return cls(
            experiment_id=str(data["experiment_id"]),
            protocol=protocol,
            methods=tuple(data["methods"]),
            metrics=tuple(data["metrics"]),
            seeds=tuple(data["seeds"]),
            output_dir=str(data["output_dir"]),
        )

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "VFMExperimentManifest":
        return cls.from_dict(json.loads(path.read_text()))


def validate_gate_expectations(
    metrics: Mapping[str, float],
    expectations: Iterable[GateExpectation],
) -> None:
    ops = {
        ">=": operator.ge,
        ">": operator.gt,
        "<=": operator.le,
        "<": operator.lt,
        "==": operator.eq,
    }
    for expectation in expectations:
        if expectation.op not in ops:
            raise ValueError(f"unsupported expectation op: {expectation.op}")
        if expectation.metric not in metrics:
            raise ValueError(f"missing metric: {expectation.metric}")
        actual = float(metrics[expectation.metric])
        target = float(expectation.value)
        if not ops[expectation.op](actual, target):
            raise ValueError(
                f"gate expectation failed: {expectation.metric}={actual} "
                f"does not satisfy {expectation.op} {target}"
            )
