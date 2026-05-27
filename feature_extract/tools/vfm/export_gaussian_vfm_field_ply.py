"""Export a Gaussian VFM field as a PLY with loc_* feature attributes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    export_gaussian_vfm_field_to_ply,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Gaussian VFM loc_* features to a source Gaussian PLY")
    parser.add_argument("--source_ply", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--nan_unassigned", action="store_true")
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    field = GaussianVFMField.load_npz(Path(args.field))
    export_gaussian_vfm_field_to_ply(
        Path(args.source_ply),
        field,
        Path(args.output_ply),
        zero_unassigned=not args.nan_unassigned,
    )
    summary = {
        "stage": "gaussian_vfm_field_ply_export",
        "source_ply": args.source_ply,
        "field": args.field,
        "output_ply": args.output_ply,
        "feature_bearing_gaussian_count": int(len(field)),
        "feature_dim": int(field.feature_dim),
        "unassigned_value": "nan" if args.nan_unassigned else "zero",
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
