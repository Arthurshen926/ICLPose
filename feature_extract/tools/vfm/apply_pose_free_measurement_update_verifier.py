"""Apply a frozen pose-free RGB measurement update verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.pose_free_update_verifier import (
    apply_pose_free_update_verifier,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rows_csv",
        "diagnostics_csv",
        "support_selector_rows_csv",
        "verifier_json",
        "output_dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = apply_pose_free_update_verifier(
        rows_csv=Path(args.rows_csv),
        diagnostics_csv=Path(args.diagnostics_csv),
        support_selector_rows_csv=Path(args.support_selector_rows_csv),
        verifier_json=Path(args.verifier_json),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
