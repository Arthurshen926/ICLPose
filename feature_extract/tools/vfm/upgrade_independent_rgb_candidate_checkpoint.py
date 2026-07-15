"""Add an explicit support-view mixture contract to a legacy RGB checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    SUPPORT_VIEW_MIXTURE_CONTRACT,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.input)
    output = Path(args.output)
    payload = torch.load(source, map_location="cpu")
    if payload.get("format") != "independent_rgb_candidate_verifier_v1":
        raise ValueError("checkpoint upgrade requires legacy verifier v1")
    config = dict(payload.get("config", {}))
    existing = config.get("support_view_mixture")
    if existing not in {None, SUPPORT_VIEW_MIXTURE_CONTRACT}:
        raise ValueError("legacy checkpoint declares an incompatible view mixture")
    config["support_view_mixture"] = SUPPORT_VIEW_MIXTURE_CONTRACT
    upgraded = dict(payload)
    upgraded["format"] = "independent_rgb_candidate_verifier_v2"
    upgraded["config"] = config
    upgraded["provenance"] = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256_short(source),
        "state_dict_changed": False,
        "support_view_mixture": SUPPORT_VIEW_MIXTURE_CONTRACT,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(upgraded, output)
    print(
        json.dumps(
            {
                "input": str(source),
                "input_sha256": file_sha256_short(source),
                "output": str(output),
                "output_sha256": file_sha256_short(output),
                "support_view_mixture": SUPPORT_VIEW_MIXTURE_CONTRACT,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
