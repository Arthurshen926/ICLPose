"""Fuse multiple token descriptor banks for multi-scale or multi-crop VPR."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank, combine_token_descriptor_banks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor_banks", nargs="+", required=True)
    parser.add_argument("--mode", choices=("concat", "mean"), default="concat")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    combined = combine_token_descriptor_banks(
        tuple(TokenDescriptorBank.from_npz(Path(path)) for path in args.descriptor_banks),
        mode=str(args.mode),
        normalize=not bool(args.no_normalize),
    )
    combined.to_npz(Path(args.output))


if __name__ == "__main__":
    main()
