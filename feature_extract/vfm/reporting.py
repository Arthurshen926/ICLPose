"""Reporting helpers for VFM-MapLoc evaluation gates."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from feature_extract.vfm.protocols import EvaluationProtocol


def ensure_protocols_not_mixed(protocols: Iterable[EvaluationProtocol]) -> None:
    kinds = {protocol.kind for protocol in protocols}
    if len(kinds) > 1:
        names = ", ".join(sorted(kind.value for kind in kinds))
        raise ValueError(f"cannot merge different protocol kinds in one claim table: {names}")


def build_gate_table(gate_name: str, rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("rows must be non-empty")
    columns = list(rows[0].keys())
    for row in rows:
        if list(row.keys()) != columns:
            raise ValueError("all rows must share identical columns and order")

    lines = [f"# {gate_name}", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)
