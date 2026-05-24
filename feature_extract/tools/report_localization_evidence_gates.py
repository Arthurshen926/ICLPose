#!/usr/bin/env python3
"""Merge POFD-FS evidence artifacts into Gate 1/2 report tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


METRIC_KEYS = ("pred_cost_m", "top1_acc", "spearman", "basin_recall@5")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _artifact_label(path_value: object) -> str:
    path = Path(str(path_value))
    return path.parent.name if path.name == "metrics.json" else path.name


def _metric_subset(metrics: Mapping[str, object] | None) -> dict[str, float]:
    metrics = metrics or {}
    return {key: float(metrics[key]) for key in METRIC_KEYS if key in metrics}


def _protocol_gate_row(path: str | Path, summary: Mapping[str, object]) -> dict:
    protocol = dict(summary.get("protocol", {}))
    metadata = {
        str(name): _metric_subset(metrics)  # type: ignore[arg-type]
        for name, metrics in dict(summary.get("metadata_baselines", {})).items()
    }
    warnings = [str(value) for value in list(summary.get("control_warnings", []))]
    clean_metadata = {
        name: metrics
        for name, metrics in metadata.items()
        if not (name == "retrieval_score" and any("retrieval_score" in warning and "oracle-like" in warning for warning in warnings))
    }
    best_clean = max(clean_metadata.values(), key=lambda row: float(row.get("top1_acc", 0.0)), default={})
    return {
        "artifact": str(path),
        "label": str(summary.get("label", Path(path).stem)),
        "protocol": protocol,
        "pofd": _metric_subset(summary.get("pofd")),  # type: ignore[arg-type]
        "metadata_baselines": metadata,
        "best_clean_metadata": best_clean,
        "paired_statistics": dict(summary.get("paired_statistics", {})),
        "hard_cases": dict(summary.get("hard_cases", {})),
        "control_warnings": warnings,
    }


def _selector_audit_row(path: str | Path, summary: Mapping[str, object]) -> dict:
    return {
        "artifact": str(path),
        "selector_checkpoint": str(summary.get("selector_checkpoint", "")),
        "pose_candidate_cache": str(summary.get("pose_candidate_cache", "")),
        "feature_shuffle_controls": dict(summary.get("feature_shuffle_controls", {})),
        "spatial_counterfactual": dict(summary.get("spatial_counterfactual", {})),
        "channel_counterfactual": dict(summary.get("channel_counterfactual", {})),
    }


def _mapability_row(path: str | Path, summary: Mapping[str, object]) -> dict:
    return {
        "artifact": str(path),
        "selected_bank": str(summary.get("selected_bank", "")),
        "selected_bank_tracks": int(summary.get("selected_bank_tracks", 0)),
        "selected_bank_feature_dim": int(summary.get("selected_bank_feature_dim", 0)),
        "metrics": _metric_subset(summary.get("metrics")),  # type: ignore[arg-type]
        "coverage": dict(summary.get("coverage", {})),
    }


def _hard_case_final_pose_row(path: str | Path, summary: Mapping[str, object]) -> dict:
    return {
        "artifact": str(path),
        "protocol": str(summary.get("protocol", "")),
        "cases": list(summary.get("cases", [])),
    }


def build_gate_report(
    *,
    protocol_controls: Sequence[str | Path] = (),
    selector_audits: Sequence[str | Path] = (),
    mapability_reports: Sequence[str | Path] = (),
    hard_case_final_pose_reports: Sequence[str | Path] = (),
) -> dict:
    """Build a JSON-serializable POFD-FS evidence gate report."""

    gate1 = [_protocol_gate_row(path, _load_json(path)) for path in protocol_controls]
    gate2_causality = [_selector_audit_row(path, _load_json(path)) for path in selector_audits]
    gate2_mapability = [_mapability_row(path, _load_json(path)) for path in mapability_reports]
    gate3_downstream = [_hard_case_final_pose_row(path, _load_json(path)) for path in hard_case_final_pose_reports]
    return {
        "gate1_predictive_utility": gate1,
        "gate2_causality": gate2_causality,
        "gate2_mapability": gate2_mapability,
        "gate3_downstream": gate3_downstream,
        "coverage": {
            "protocol_controls": len(gate1),
            "selector_audits": len(gate2_causality),
            "mapability_reports": len(gate2_mapability),
            "hard_case_final_pose_reports": len(gate3_downstream),
        },
    }


def _markdown_metric_cells(metrics: Mapping[str, object]) -> list[str]:
    return [_fmt(metrics.get(key)) for key in METRIC_KEYS]


def _append_protocol_controls(lines: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    lines.extend(["## Gate 1: Predictive Utility", "", "| label | protocol | pred | top1 | spearman | basin@5 | best clean metadata top1 |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        protocol = dict(row.get("protocol", {}))
        best_clean = dict(row.get("best_clean_metadata", {}))
        best_meta_top1 = float(best_clean.get("top1_acc", 0.0))
        cells = [
            str(row.get("label", "")),
            str(protocol.get("protocol", "")),
            *_markdown_metric_cells(dict(row.get("pofd", {}))),
            f"{best_meta_top1:.4g}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def _append_selector_audits(lines: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    lines.extend(["## Gate 2: Causality Controls", "", "| artifact | control | pred | top1 | spearman | basin@5 |", "|---|---|---:|---:|---:|---:|"])
    for row in rows:
        artifact = _artifact_label(row.get("artifact", ""))
        shuffle = dict(row.get("feature_shuffle_controls", {}))
        for section in ("base", "candidate_render_shuffle", "query_batch_shuffle", "wrong_scene_render_shuffle"):
            if section in shuffle and not dict(shuffle[section]).get("skipped"):
                cells = [artifact, section, *_markdown_metric_cells(dict(shuffle[section]))]
                lines.append("| " + " | ".join(cells) + " |")
        spatial = dict(row.get("spatial_counterfactual", {}))
        for section in ("drop_high", "drop_low"):
            if section in spatial:
                cells = [artifact, "spatial_" + section, *_markdown_metric_cells(dict(spatial[section]))]
                lines.append("| " + " | ".join(cells) + " |")
        channel = dict(row.get("channel_counterfactual", {}))
        for section in ("drop_high", "drop_low"):
            if section in channel:
                cells = [artifact, "channel_" + section, *_markdown_metric_cells(dict(channel[section]))]
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def _append_mapability(lines: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    lines.extend(["## Gate 2: Mapability", "", "| artifact | selected_bank_tracks | dim | pred | top1 | spearman | basin@5 | valid_px_frac |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        coverage = dict(row.get("coverage", {}))
        cells = [
            _artifact_label(row.get("artifact", "")),
            str(int(row.get("selected_bank_tracks", 0))),
            str(int(row.get("selected_bank_feature_dim", 0))),
            *_markdown_metric_cells(dict(row.get("metrics", {}))),
            _fmt(coverage.get("projected_valid_pixel_frac")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def _append_downstream(lines: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    lines.extend(["## Gate 3: Downstream Hard Cases", "", "| artifact | case | method | n | trans med mm | R@5deg250mm | solver success |", "|---|---|---|---:|---:|---:|---:|"])
    for row in rows:
        artifact = _artifact_label(row.get("artifact", ""))
        for case in list(row.get("cases", [])):
            case_map = dict(case)
            for method_row in list(case_map.get("rows", [])):
                method = dict(method_row)
                metrics = dict(method.get("metrics", {}))
                success = method.get("solver_success_frac")
                success_text = "" if success is None else f"{float(success) * 100.0:.1f}%"
                cells = [
                    artifact,
                    str(case_map.get("case", "")),
                    str(method.get("label", "")),
                    str(int(method.get("num_samples", 0))),
                    _fmt(metrics.get("trans_median")),
                    _fmt(metrics.get("joint_5deg_250mm")),
                    success_text,
                ]
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def format_gate_report_markdown(report: Mapping[str, object]) -> str:
    lines = ["# POFD-FS Evidence Gate Report", ""]
    _append_protocol_controls(lines, list(report.get("gate1_predictive_utility", [])))  # type: ignore[arg-type]
    _append_selector_audits(lines, list(report.get("gate2_causality", [])))  # type: ignore[arg-type]
    _append_mapability(lines, list(report.get("gate2_mapability", [])))  # type: ignore[arg-type]
    _append_downstream(lines, list(report.get("gate3_downstream", [])))  # type: ignore[arg-type]
    lines.append("## Coverage")
    lines.append("")
    coverage = dict(report.get("coverage", {}))
    for key in ("protocol_controls", "selector_audits", "mapability_reports", "hard_case_final_pose_reports"):
        lines.append(f"- {key}: {int(coverage.get(key, 0))}")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-controls", nargs="*", default=[])
    parser.add_argument("--selector-audits", nargs="*", default=[])
    parser.add_argument("--mapability-reports", nargs="*", default=[])
    parser.add_argument("--hard-case-final-pose-reports", nargs="*", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_gate_report(
        protocol_controls=args.protocol_controls,
        selector_audits=args.selector_audits,
        mapability_reports=args.mapability_reports,
        hard_case_final_pose_reports=args.hard_case_final_pose_reports,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(format_gate_report_markdown(report), encoding="utf-8")
    print(json.dumps(report["coverage"], indent=2))


if __name__ == "__main__":
    main()
