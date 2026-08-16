"""Build low-resolution mapping-pose fields from frozen canonical RADIO/3DGS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.visibility_feature_atlas import (
    build_canonical_feature_visibility_pose_atlas,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output_atlas", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_view_count", type=int, default=1487)
    parser.add_argument("--output_rows", type=int, default=9)
    parser.add_argument("--output_cols", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, report_path = Path(args.output_atlas), Path(args.output_json)
    if (output.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite canonical feature atlas")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contributors = sorted(Path(args.contributors).glob("*.npz"))
    if len(contributors) != int(args.expected_view_count):
        raise ValueError("canonical feature atlas contributor count differs")
    atlas = build_canonical_feature_visibility_pose_atlas(
        physical, field, contributors,
        output_rows=int(args.output_rows), output_cols=int(args.output_cols),
        metadata={
            "physical_map_path": str(Path(args.physical_map).resolve()),
            "canonical_field_path": str(Path(args.canonical_field).resolve()),
            "source_contributor_directory": str(Path(args.contributors).resolve()),
        },
    )
    atlas.save_npz(output)
    report = {
        "artifact_type": "goal_maplet_canonical_feature_pose_atlas_build_v1",
        "output_atlas": str(output.resolve()),
        "content_sha256": atlas.content_sha256,
        "view_count": atlas.view_count,
        "feature_dim": int(atlas.features.shape[1]),
        "grid_rows": int(atlas.features.shape[2]),
        "grid_cols": int(atlas.features.shape[3]),
        "mean_valid_grid_fraction": float(atlas.valid.mean()),
        "physical_map_sha256": atlas.physical_map_sha256,
        "canonical_field_sha256": atlas.canonical_field_sha256,
        "claims": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_descriptors": False,
            "uses_reference_image_retrieval": False,
            "uses_alike": False,
            "uses_pnp": False,
            "output_is_final_pose": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
