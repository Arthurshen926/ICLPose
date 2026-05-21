#!/usr/bin/env python3
"""Evaluate solver-conditioned handoff from a POFD-FS candidate table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.score_calibrator import load_candidate_table_jsonl  # noqa: E402
from feature_extract.localizability.solver_handoff import evaluate_handoff_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", default="1,5")
    parser.add_argument("--selection-modes", default="pofd_score,pnp_inliers,pnp_reproj_median,oracle_pose")
    parser.add_argument("--write-selected", action="store_true")
    return parser.parse_args()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def main() -> None:
    args = parse_args()
    rows = load_candidate_table_jsonl(args.candidate_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    summary = {
        "candidate_table": str(args.candidate_table),
        "num_candidate_rows": len(rows),
        "results": [],
    }
    selected_payload = {}
    for topk in _parse_ints(args.topk):
        for mode in _parse_strings(args.selection_modes):
            result = evaluate_handoff_rows(rows, topk=topk, selection_mode=mode)
            selected = result.pop("selected_candidate_indices")
            summary["results"].append(result)
            selected_payload[f"top{topk}_{mode}"] = selected
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if bool(args.write_selected):
        (out_dir / "selected_indices.json").write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
