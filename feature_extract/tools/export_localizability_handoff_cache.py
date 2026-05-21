#!/usr/bin/env python3
"""Export a POFD-FS selected candidate table as a standard init cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.handoff_cache import export_selected_init_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True, help="Standard init/candidate cache .npz")
    parser.add_argument("--candidate-table", required=True, help="POFD-FS candidate_table.jsonl")
    parser.add_argument("--save-path", required=True, help="Output one-candidate init cache .npz")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument(
        "--selection-mode",
        default="pofd_score",
        choices=("pofd_score", "oracle_pose", "pnp_inliers", "pnp_reproj_median"),
    )
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _entries, stats = export_selected_init_cache(
        source_cache_path=args.source_cache,
        candidate_table_path=args.candidate_table,
        save_path=args.save_path,
        topk=args.topk,
        selection_mode=args.selection_mode,
        source_name=args.source_name,
    )
    if args.summary_json:
        output_path = Path(args.summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
