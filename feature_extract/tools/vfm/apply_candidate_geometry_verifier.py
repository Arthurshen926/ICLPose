"""Apply a frozen candidate geometry verifier to RGB diagnostic rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.measurement_v1.candidate_geometry_verifier import (
    apply_candidate_geometry_verifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnostic_rows_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_legacy_missing_checkpoint_hash", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            apply_candidate_geometry_verifier(
                model_path=Path(args.model),
                diagnostic_rows_csv=Path(args.diagnostic_rows_csv),
                output_path=Path(args.output),
                allow_legacy_missing_checkpoint_hash=bool(
                    args.allow_legacy_missing_checkpoint_hash
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
