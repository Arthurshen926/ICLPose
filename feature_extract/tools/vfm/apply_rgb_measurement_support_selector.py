"""Apply a frozen RGB measurement support-view selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.support_selection import apply_measurement_support_selector


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics_csv", required=True)
    parser.add_argument("--selector_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    summary = apply_measurement_support_selector(
        diagnostics_csv=Path(args.diagnostics_csv),
        selector_json=Path(args.selector_json),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
