"""Migrate one existing RADIO-final surface code per primitive into Goal-Maplet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    build_canonical_surface_field,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--surface_field", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--minimum_confidence", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_field), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite canonical Goal-Maplet field")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = build_canonical_surface_field(
        SurfaceFeatureField.load_npz(Path(args.surface_field)),
        physical,
        minimum_confidence=float(args.minimum_confidence),
        metadata={
            "source_field_sha256": file_sha256(Path(args.surface_field)),
            "migration": "canonical_code_only_no_view_or_downstream_embedding",
        },
    )
    field.save_npz(output)
    readout = readout_canonical_field(field, physical)
    report = {
        "stage": "build_goal_maplet_canonical_surface_field",
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "canonical_primitive_count": int(field.primitive_rows.size),
        "feature_dim": int(field.feature_dim),
        "maplet_count": int(physical.maplet_ids.size),
        "child_tile_count": int(physical.child_parent_rows.size),
        "parent_feature_coverage": {
            "median": float(np.median(readout.parent_coverage)),
            "p10": float(np.percentile(readout.parent_coverage, 10.0)),
            "nonempty_fraction": float(np.mean(readout.parent_coverage > 0.0)),
        },
        "child_feature_coverage": {
            "median": float(np.median(readout.child_coverage)),
            "p10": float(np.percentile(readout.child_coverage, 10.0)),
            "nonempty_fraction": float(np.mean(readout.child_coverage > 0.0)),
        },
        "storage_contract": dict(field.metadata or {}),
        "output_field": str(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
