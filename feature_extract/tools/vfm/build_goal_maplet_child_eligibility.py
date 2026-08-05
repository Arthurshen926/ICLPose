"""Build retrieval/proposal/refinement child eligibility masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.child_eligibility import build_child_geometry_eligibility
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_npz), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite child eligibility")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    result = build_child_geometry_eligibility(physical, field)
    result.save_npz(output)
    report = {
        "stage": "build_goal_maplet_child_eligibility",
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "eligibility_sha256": result.content_sha256,
        "child_count": int(result.component_count.size),
        "retrieval_qualified_fraction": float(np.mean(result.retrieval_qualified)),
        "proposal_qualified_fraction": float(np.mean(result.proposal_qualified)),
        "refinement_qualified_fraction": float(np.mean(result.refinement_qualified)),
        "thresholds": dict(result.metadata),
        "output_npz": str(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
