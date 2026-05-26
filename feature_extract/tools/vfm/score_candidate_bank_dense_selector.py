"""Score fixed candidates using dense selected-token descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import attach_report_inputs
from feature_extract.vfm.dense_selector_scoring import score_candidate_bank_by_dense_selector
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score candidates with dense selector cosine similarity")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--map_manifest", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--layer_name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="dense_selector_cosine")
    parser.add_argument("--utility_weighted_pooling", action="store_true")
    parser.add_argument("--translation_threshold_m", type=float, default=0.25)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args(argv)

    selector = load_selector_from_checkpoint(Path(args.selector_checkpoint), device=args.device)
    bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    rows = score_candidate_bank_by_dense_selector(
        bank=bank,
        query_manifest=TokenBankManifest.from_json(Path(args.query_manifest)),
        map_manifest=TokenBankManifest.from_json(Path(args.map_manifest)),
        selector=selector,
        layer_name=args.layer_name,
        method=args.method,
        translation_threshold_m=args.translation_threshold_m,
        rotation_threshold_deg=args.rotation_threshold_deg,
        device=args.device,
        utility_weighted_pooling=args.utility_weighted_pooling,
    )

    row_payload = []
    for row in rows:
        item = dict(row.__dict__)
        item["protocol_kind"] = row.protocol_kind.value
        row_payload.append(item)
    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    output_rows.write_text(json.dumps(row_payload, indent=2, sort_keys=True) + "\n")

    report = evaluate_score_table(rows).to_dict()
    report_with_inputs = attach_report_inputs(
        report,
        bank,
        Path(args.bank),
        {
            "query_manifest": args.query_manifest,
            "map_manifest": args.map_manifest,
            "selector_checkpoint": args.selector_checkpoint,
        },
    )
    report_with_inputs["inputs"]["utility_weighted_pooling"] = bool(args.utility_weighted_pooling)
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report_with_inputs, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table("Dense Selector Cosine", [report]) + "\n")


if __name__ == "__main__":
    main()
