"""Report mapability metrics for a selected 3D track-feature bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.mapability_metrics import compare_track_bank_mapability


def main() -> None:
    parser = argparse.ArgumentParser(description="Report selected track-bank mapability metrics")
    parser.add_argument("--track_bank", required=True)
    parser.add_argument("--expected_track_count", type=int, required=True)
    parser.add_argument("--max_pairwise_tracks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    report = compare_track_bank_mapability(
        load_selected_track_bank_npz(Path(args.track_bank)),
        expected_track_count=args.expected_track_count,
        max_pairwise_tracks=args.max_pairwise_tracks,
        seed=args.seed,
    ).to_dict()
    report["track_bank"] = str(args.track_bank)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
