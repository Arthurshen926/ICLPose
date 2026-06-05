"""Combine track-observation JSONL files for Stage F augmented bank evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Combine track observation JSONL files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_counts = []
    seen_rows = 0
    with output.open("w") as handle:
        for path_text in args.inputs:
            count = 0
            for line in Path(path_text).read_text().splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                handle.write(json.dumps(item, sort_keys=True) + "\n")
                count += 1
                seen_rows += 1
            input_counts.append(int(count))
    summary = {
        "stage": "stage_f_track_observation_combination",
        "inputs": [str(item) for item in args.inputs],
        "outputs": {"track_observations": str(output)},
        "input_row_counts": input_counts,
        "combined_row_count": int(seen_rows),
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
