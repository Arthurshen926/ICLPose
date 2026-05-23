"""Protocol-aware reporting helpers for POFD-FS evidence tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .protocol import ArtifactProtocol, assert_protocol_claims_compatible


@dataclass(frozen=True)
class ProtocolResult:
    label: str
    protocol: ArtifactProtocol
    metrics: Mapping[str, float]
    extras: Mapping[str, object] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        row = {
            "label": str(self.label),
            **self.protocol.to_report_row(),
        }
        row.update({str(key): value for key, value in self.metrics.items()})
        row.update({str(key): value for key, value in dict(self.extras).items()})
        return row


def protocol_summary_rows(results: Sequence[ProtocolResult]) -> list[dict[str, object]]:
    return [result.to_row() for result in results]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_protocol_summary_markdown(results: Sequence[ProtocolResult], *, allow_mixed: bool = True) -> str:
    """Format a compact protocol-aware result table."""

    assert_protocol_claims_compatible([result.protocol for result in results], allow_mixed=allow_mixed)
    headers = [
        "label",
        "protocol",
        "candidate generator",
        "split",
        "deployment claim",
        "solver conditioned",
        "pred_cost_m",
        "top1_acc",
        "spearman",
        "basin_recall@5",
    ]
    lines = [
        "# POFD-FS Protocol Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for result in results:
        row = result.to_row()
        values = [
            row.get("label"),
            row.get("protocol"),
            row.get("candidate_generator"),
            row.get("split"),
            row.get("deployment_claim_allowed"),
            row.get("solver_conditioned"),
            row.get("pred_cost_m"),
            row.get("top1_acc"),
            row.get("spearman"),
            row.get("basin_recall@5"),
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["ProtocolResult", "format_protocol_summary_markdown", "protocol_summary_rows"]
