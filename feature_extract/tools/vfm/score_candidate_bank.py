"""Score a labeled VFM candidate bank with a metadata baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.artifacts import attach_report_inputs
from feature_extract.vfm.candidate_scoring import score_candidate_bank_by_metadata
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_table import evaluate_score_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a candidate bank baseline")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--method", required=True, choices=["retrieval_order", "candidate_prior"])
    parser.add_argument("--translation_threshold_m", type=float, default=0.25)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    rows = score_candidate_bank_by_metadata(
        bank,
        method=args.method,
        translation_threshold_m=args.translation_threshold_m,
        rotation_threshold_deg=args.rotation_threshold_deg,
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
    report_with_inputs = attach_report_inputs(report, bank, Path(args.bank))
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report_with_inputs, indent=2, sort_keys=True) + "\n")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table("Candidate Metadata Baseline", [report]) + "\n")


if __name__ == "__main__":
    main()
