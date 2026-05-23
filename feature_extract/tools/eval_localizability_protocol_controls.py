#!/usr/bin/env python3
"""Evaluate POFD-FS protocol controls and metadata-only baselines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.controls import metadata_baseline_scores  # noqa: E402
from feature_extract.localizability.hard_cases import build_hard_case_masks, summarize_hard_case_masks  # noqa: E402
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.protocol import ArtifactProtocol, validate_protocol_metadata  # noqa: E402
from feature_extract.localizability.reporting import ProtocolResult, format_protocol_summary_markdown  # noqa: E402
from feature_extract.localizability.score_calibrator import load_candidate_table_jsonl  # noqa: E402


BASELINE_REQUIRED_FIELDS = {
    "candidate_rank": (("candidate_idx", "candidate_rank", "retrieval_rank"),),
    "retrieval_score": (("retrieval_score", "retrieval_scores", "retrieval_scores_candidates"),),
    "pnp_inliers": (("pnp_inliers", "retrieval_pnp_num_inliers_candidates", "num_inliers"),),
    "reproj_median": (("reproj_median_px", "retrieval_pnp_reproj_median_candidates", "reproj_median"),),
    "delta_pose": (("delta_trans_m", "pose_delta_trans_m"), ("delta_rot_deg", "pose_delta_rot_deg")),
    "score_margin": (("score_margin", "retrieval_score_margin", "metadata_score_margin"),),
}


def _rows_to_grouped(rows: Iterable[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["sample_name"]), []).append(row)
    if not grouped:
        raise ValueError("candidate table is empty")
    return grouped


def _has_any_field(row_keys: set[str], names: Sequence[str]) -> bool:
    return any(name in row_keys for name in names)


def available_metadata_baselines(rows: Sequence[dict]) -> list[str]:
    """Return metadata-only baselines that can be computed from table fields."""

    row_keys = set()
    for row in rows[: min(len(rows), 256)]:
        row_keys.update(str(key) for key in row.keys())
    modes = []
    for mode, required_groups in BASELINE_REQUIRED_FIELDS.items():
        if all(_has_any_field(row_keys, group) for group in required_groups):
            modes.append(mode)
    return modes


def _tensor_table(rows: Sequence[dict], field: str, *, default: float = 0.0) -> torch.Tensor:
    grouped = _rows_to_grouped(rows)
    num_candidates = max(int(row["candidate_idx"]) for row in rows) + 1
    table = torch.full((len(grouped), num_candidates), float(default), dtype=torch.float32)
    for batch_idx, sample_rows in enumerate(grouped.values()):
        for row in sample_rows:
            table[batch_idx, int(row["candidate_idx"])] = float(row.get(field, default))
    return table


def _bool_table(rows: Sequence[dict], field: str, *, default: bool = False) -> torch.Tensor:
    grouped = _rows_to_grouped(rows)
    num_candidates = max(int(row["candidate_idx"]) for row in rows) + 1
    table = torch.full((len(grouped), num_candidates), bool(default), dtype=torch.bool)
    for batch_idx, sample_rows in enumerate(grouped.values()):
        for row in sample_rows:
            table[batch_idx, int(row["candidate_idx"])] = bool(row.get(field, default))
    return table


def _candidate_table_to_tensors(rows: Sequence[dict]) -> tuple[list[str], dict[str, torch.Tensor]]:
    grouped = _rows_to_grouped(rows)
    fields: dict[str, torch.Tensor] = {}
    sample_names = list(grouped)
    for field in (
        "candidate_idx",
        "candidate_rank",
        "retrieval_rank",
        "score",
        "score_rank",
        "score_margin",
        "retrieval_score",
        "retrieval_score_margin",
        "metadata_score_margin",
        "pose_cost_m",
        "trans_err_m",
        "rot_err_deg",
        "delta_trans_m",
        "delta_rot_deg",
        "retrieval_scores_candidates",
        "retrieval_original_scores_candidates",
        "retrieval_pnp_num_inliers_candidates",
        "retrieval_pnp_reproj_median_candidates",
    ):
        if any(field in row for row in rows):
            fields[field] = _tensor_table(rows, field)
    fields["valid"] = _bool_table(rows, "valid", default=True)
    fields["in_basin"] = _bool_table(rows, "in_basin", default=False)
    return sample_names, fields


def _metrics_to_float(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {str(key): float(value.detach().cpu()) for key, value in metrics.items()}


def _control_warnings(
    protocol_meta: ArtifactProtocol,
    metadata_results: Mapping[str, Mapping[str, float]],
    tensors: Mapping[str, torch.Tensor],
) -> list[str]:
    warnings: list[str] = []
    if "score_rank" in tensors:
        warnings.append("score_rank is model-output rank and is ignored for metadata-only candidate_rank controls")
    if protocol_meta.gt_centered:
        retrieval = metadata_results.get("retrieval_score")
        if retrieval is not None:
            top1 = float(retrieval.get("top1_acc", 0.0))
            gap = float(retrieval.get("oracle_gap_m", 1.0))
            if top1 >= 0.99 and gap <= 1.0e-6:
                warnings.append(
                    "retrieval_score is oracle-like under a GT-centered protocol; treat it as a candidate-generator shortcut, not a clean metadata-only control"
                )
    return warnings


def _field_aliases(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    aliases = dict(tensors)
    if "candidate_rank" not in aliases:
        if "retrieval_rank" in tensors:
            aliases["candidate_rank"] = tensors["retrieval_rank"]
        elif "candidate_idx" in tensors:
            aliases["candidate_rank"] = tensors["candidate_idx"]
    if "retrieval_scores_candidates" in tensors:
        aliases["retrieval_score"] = tensors["retrieval_scores_candidates"]
    if "retrieval_pnp_num_inliers_candidates" in tensors:
        aliases["pnp_inliers"] = tensors["retrieval_pnp_num_inliers_candidates"]
    if "retrieval_pnp_reproj_median_candidates" in tensors:
        aliases["reproj_median_px"] = tensors["retrieval_pnp_reproj_median_candidates"]
    if "score_margin" not in aliases:
        if "retrieval_score_margin" in tensors:
            aliases["score_margin"] = tensors["retrieval_score_margin"]
        elif "metadata_score_margin" in tensors:
            aliases["score_margin"] = tensors["metadata_score_margin"]
    return aliases


def summarize_protocol_controls(
    rows: Sequence[dict],
    *,
    label: str,
    protocol: str,
    candidate_generator: str,
    split: str,
    baseline_modes: Sequence[str] | None = None,
    scene: str | None = None,
    solver_conditioned: bool = False,
    gt_usage: str = "eval_only",
    topk: Sequence[int] = (1, 5, 10),
) -> dict:
    """Summarize POFD scores against metadata-only controls and hard cases."""

    sample_names, tensors = _candidate_table_to_tensors(rows)
    valid = tensors["valid"].bool()
    costs = tensors["pose_cost_m"].float()
    basin = tensors["in_basin"].bool()
    protocol_meta = ArtifactProtocol(
        protocol=protocol,
        candidate_generator=candidate_generator,
        split=split,
        input_fields=("score",),
        gt_usage=gt_usage,
        solver_conditioned=solver_conditioned,
        scene=scene,
    )
    validate_protocol_metadata(protocol_meta, training_input=False)
    topk_tuple = tuple(int(k) for k in topk)
    pofd_metrics = _metrics_to_float(
        ranking_metrics(tensors["score"].float(), costs, valid_mask=valid, basin_label=basin, topk=topk_tuple)
    )
    fields = _field_aliases(tensors)
    modes = list(baseline_modes) if baseline_modes is not None else available_metadata_baselines(rows)
    metadata_results: dict[str, dict[str, float]] = {}
    for mode in modes:
        if mode == "pofd_score":
            continue
        try:
            baseline_scores = metadata_baseline_scores(fields, mode=mode, valid_mask=valid)
        except (KeyError, ValueError):
            continue
        metadata_results[mode] = _metrics_to_float(
            ranking_metrics(baseline_scores, costs, valid_mask=valid, basin_label=basin, topk=topk_tuple)
        )
    hard_masks = build_hard_case_masks(
        tensors["score"].float(),
        costs,
        basin_label=basin,
        valid_mask=valid,
        pnp_inliers=fields.get("pnp_inliers"),
        delta_trans_m=fields.get("delta_trans_m"),
        delta_rot_deg=fields.get("delta_rot_deg"),
        retrieval_topk=max(topk_tuple) if topk_tuple else 10,
    )
    return {
        "label": str(label),
        "num_samples": len(sample_names),
        "num_candidate_rows": len(rows),
        "num_candidates": int(valid.shape[1]),
        "protocol": protocol_meta.to_report_row(),
        "available_metadata_baselines": available_metadata_baselines(rows),
        "control_warnings": _control_warnings(protocol_meta, metadata_results, tensors),
        "pofd": pofd_metrics,
        "metadata_baselines": metadata_results,
        "hard_cases": summarize_hard_case_masks(hard_masks),
    }


def format_controls_markdown(summary: Mapping[str, object]) -> str:
    pofd = summary["pofd"]  # type: ignore[index]
    baselines = summary["metadata_baselines"]  # type: ignore[index]
    protocol = summary["protocol"]  # type: ignore[index]
    results = [
        ProtocolResult(
            label=str(summary["label"]) + ":pofd",
            protocol=ArtifactProtocol(
                protocol=str(protocol["protocol"]),  # type: ignore[index]
                candidate_generator=str(protocol["candidate_generator"]),  # type: ignore[index]
                split=str(protocol["split"]),  # type: ignore[index]
                input_fields=("score",),
                gt_usage=str(protocol["gt_usage"]),  # type: ignore[index]
                solver_conditioned=bool(protocol["solver_conditioned"]),  # type: ignore[index]
            ),
            metrics=pofd,  # type: ignore[arg-type]
        )
    ]
    lines = [format_protocol_summary_markdown(results).rstrip(), "", "## Metadata-only baselines", ""]
    lines.append("| mode | pred_cost_m | oracle_gap_m | top1_acc | spearman | basin@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode, metrics in sorted(dict(baselines).items()):  # type: ignore[arg-type]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(mode),
                    f"{float(metrics.get('pred_cost_m', 0.0)):.4g}",
                    f"{float(metrics.get('oracle_gap_m', 0.0)):.4g}",
                    f"{float(metrics.get('top1_acc', 0.0)):.4g}",
                    f"{float(metrics.get('spearman', 0.0)):.4g}",
                    f"{float(metrics.get('basin_recall@5', 0.0)):.4g}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Hard cases", "", "| case | count | fraction |", "|---|---:|---:|"])
    for name, metrics in sorted(dict(summary["hard_cases"]).items()):  # type: ignore[arg-type,index]
        lines.append(f"| {name} | {int(metrics['count'])} | {float(metrics['fraction']):.4g} |")
    warnings = list(summary.get("control_warnings", []))  # type: ignore[arg-type]
    if warnings:
        lines.extend(["", "## Control warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--protocol", choices=("controlled_lattice", "reference_pose", "real_retrieval"), required=True)
    parser.add_argument("--candidate-generator", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--gt-usage", default="eval_only")
    parser.add_argument("--solver-conditioned", action="store_true")
    parser.add_argument("--baseline-modes", default="auto")
    parser.add_argument("--topk", default="1,5,10")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_candidate_table_jsonl(args.candidate_table)
    baseline_modes = None if args.baseline_modes == "auto" else _parse_csv(args.baseline_modes)
    summary = summarize_protocol_controls(
        rows,
        label=args.label,
        protocol=args.protocol,
        candidate_generator=args.candidate_generator,
        split=args.split,
        baseline_modes=baseline_modes,
        scene=args.scene,
        solver_conditioned=bool(args.solver_conditioned),
        gt_usage=args.gt_usage,
        topk=_parse_int_csv(args.topk),
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(format_controls_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
