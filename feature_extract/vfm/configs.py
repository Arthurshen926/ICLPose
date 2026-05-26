"""Config loading for VFM-MapLoc protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind


def _read_yaml(path: Path) -> Mapping[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"expected mapping YAML in {path}")
    return data


def load_protocol_config(path: str | Path) -> EvaluationProtocol:
    """Load an `EvaluationProtocol` from a VFM YAML config."""

    config_path = Path(path)
    data = _read_yaml(config_path)
    if "protocol" not in data:
        raise ValueError(f"missing protocol section in {config_path}")
    protocol = dict(data["protocol"])
    return EvaluationProtocol(
        name=str(protocol["name"]),
        kind=ProtocolKind(protocol["kind"]),
        split=str(protocol["split"]),
        candidate_generator=str(protocol["candidate_generator"]),
        allowed_training_inputs=tuple(protocol.get("allowed_training_inputs", ())),
        candidate_uses_gt=bool(protocol["candidate_uses_gt"]),
        solver_conditioned=bool(protocol["solver_conditioned"]),
        notes=str(protocol.get("notes", "")),
        gate=str(protocol.get("gate", "")),
        metadata_fields=tuple(protocol.get("metadata_fields", ())),
    )
