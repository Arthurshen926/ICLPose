#!/usr/bin/env python3
"""Evaluate a trained Stage C2.8 correspondence confidence model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.correspondence_confidence import (
    CalibratedLogisticConfidence,
    confidence_metrics,
    vectorize_match_rows,
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_jsonl", required=True)
    parser.add_argument("--confidence_model", required=True)
    parser.add_argument("--feature_set", default="descriptor_map")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stride_positive", type=float, default=1.0)
    parser.add_argument("--weak_positive_stride", type=float, default=2.0)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    args = parser.parse_args(argv)

    rows = _load_jsonl(Path(args.match_jsonl))
    features, labels, keep, names = vectorize_match_rows(
        rows,
        feature_set=args.feature_set,
        stride_positive=float(args.stride_positive),
        weak_positive_stride=float(args.weak_positive_stride),
    )
    model = CalibratedLogisticConfidence.load_json(args.confidence_model)
    scores = model.predict_proba(features[keep]) if keep.any() else []
    summary = {
        "stage": "stage_c28_confidence_model_eval",
        "match_jsonl": args.match_jsonl,
        "confidence_model": args.confidence_model,
        "feature_set": args.feature_set,
        "feature_names": names,
        "row_count": int(len(rows)),
        "kept_count": int(keep.sum()),
        "metrics": confidence_metrics(labels[keep], scores, top_fraction=float(args.top_fraction)),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
