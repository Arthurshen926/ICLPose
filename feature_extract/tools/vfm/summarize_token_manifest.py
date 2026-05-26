"""Summarize a VFM token-bank manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.token_manifest_summary import summarize_token_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a VFM token manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = summarize_token_manifest(Path(args.manifest)).to_dict()
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
