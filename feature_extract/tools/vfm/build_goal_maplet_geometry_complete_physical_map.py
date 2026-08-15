"""Build a pose-free coverage-complete physical hierarchy from frozen 2DGS geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_complete_physical_map import (
    build_geometry_complete_physical_map,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.tokens import compute_file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_carrier_physical_map", required=True)
    parser.add_argument("--child_voxel_size_m", type=float, default=1.0)
    parser.add_argument("--parent_voxel_size_m", type=float, default=5.0)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, audit_path = Path(args.output_map), Path(args.audit_json)
    if not args.force and (output.exists() or audit_path.exists()):
        raise FileExistsError("refusing to overwrite geometry-complete physical map")
    source_path = Path(args.geometry_carrier_physical_map)
    source = GoalMapletPhysicalMap.load_npz(source_path)
    result, build_audit = build_geometry_complete_physical_map(
        source,
        child_voxel_size_m=float(args.child_voxel_size_m),
        parent_voxel_size_m=float(args.parent_voxel_size_m),
        metadata={
            "source_geometry_carrier_file_sha256": compute_file_sha256(source_path),
        },
    )
    result.save_npz(output)
    parent_member_counts = np.bincount(
        result.membership_primitive_rows, minlength=result.primitive_ids.size
    )
    child_member_counts = np.bincount(
        result.child_member_primitive_rows, minlength=result.primitive_ids.size
    )
    report = {
        "artifact_type": "goal_maplet_geometry_complete_physical_map_build_audit_v1",
        "physical_map_sha256": result.content_sha256,
        "physical_map_file_sha256": compute_file_sha256(output),
        "source_geometry_carrier_physical_map_sha256": source.content_sha256,
        "primitive_count": int(result.primitive_ids.size),
        "parent_count": int(result.maplet_ids.size),
        "child_count": int(result.child_parent_rows.size),
        "minimum_parent_memberships_per_primitive": int(np.min(parent_member_counts)),
        "maximum_parent_memberships_per_primitive": int(np.max(parent_member_counts)),
        "minimum_child_memberships_per_primitive": int(np.min(child_member_counts)),
        "maximum_child_memberships_per_primitive": int(np.max(child_member_counts)),
        "uses_pose_or_image_input": False,
        **build_audit.as_dict(),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
