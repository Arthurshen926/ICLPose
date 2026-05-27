"""Report pose-label recall for descriptor-retrieval candidate banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.retrieval_report import (
    DEFAULT_RETRIEVAL_THRESHOLDS,
    summarize_descriptor_retrieval_bank,
)


def _parse_thresholds(items: list[str]) -> tuple[tuple[float, float], ...]:
    if not items:
        return DEFAULT_RETRIEVAL_THRESHOLDS
    thresholds = []
    for item in items:
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError("--threshold values must be 'translation_m,rotation_deg'")
        thresholds.append((float(parts[0]), float(parts[1])))
    return tuple(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize descriptor retrieval candidate recall")
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--threshold", action="append", default=[])
    parser.add_argument("--recall_at", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    report = summarize_descriptor_retrieval_bank(
        CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)),
        thresholds=_parse_thresholds(args.threshold),
        recall_at=tuple(args.recall_at),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
