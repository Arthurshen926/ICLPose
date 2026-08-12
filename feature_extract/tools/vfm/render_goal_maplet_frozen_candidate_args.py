"""Render one-argument-per-line CLI tokens for the frozen candidate recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True)
    args = parser.parse_args(argv)
    frozen = json.loads(Path(args.frozen).read_text())
    values = frozen["candidate_configuration"]["inference_arguments"]
    if set(values) != set(DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS):
        raise ValueError("frozen candidate argument set differs from executable schema")
    tokens = []
    for key in DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS:
        value = values[key]
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
        elif value is not None:
            tokens.extend((flag, str(value)))
    print("\n".join(tokens))


if __name__ == "__main__":
    main()
