#!/usr/bin/env python3
"""Mine score-high wrong candidate pairs from a POFD-FS candidate table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.failure_replay import mine_score_hard_candidate_pairs  # noqa: E402
from feature_extract.localizability.score_calibrator import load_candidate_table_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cost-gap-m", type=float, default=0.12)
    parser.add_argument("--near-identity-trans-m", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_candidate_table_jsonl(args.candidate_table)
    pairs, summary = mine_score_hard_candidate_pairs(
        rows,
        cost_gap_m=float(args.cost_gap_m),
        near_identity_trans_m=float(args.near_identity_trans_m),
    )
    pair_path = out_dir / "pairs.jsonl"
    with pair_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    summary = {
        **summary,
        "candidate_table": str(args.candidate_table),
        "cost_gap_m": float(args.cost_gap_m),
        "near_identity_trans_m": float(args.near_identity_trans_m),
        "pairs_path": str(pair_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
