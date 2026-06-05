"""Combine compatible sparse landmark banks for Stage F side-path evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.map_lifting import load_selected_track_bank_npz, save_selected_track_bank_npz
from feature_extract.vfm.vfm_aware_landmarks import combine_selected_track_banks


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Combine same-dimension selected track banks")
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--track_id_offsets", nargs="+", type=int, default=None)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    banks = [load_selected_track_bank_npz(Path(path)) for path in args.banks]
    combined = combine_selected_track_banks(banks, track_id_offsets=args.track_id_offsets)
    output_bank = Path(args.output_bank)
    save_selected_track_bank_npz(combined, output_bank)
    summary = {
        "stage": "stage_f_bank_combination",
        "inputs": {
            "banks": [str(path) for path in args.banks],
            "track_id_offsets": None if args.track_id_offsets is None else [int(item) for item in args.track_id_offsets],
        },
        "outputs": {
            "bank": str(output_bank),
        },
        "input_track_counts": [int(len(bank.tracks)) for bank in banks],
        "feature_dim": int(combined.feature_dim),
        "combined_track_count": int(len(combined.tracks)),
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
