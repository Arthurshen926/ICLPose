"""Audit the bounded global physical pose-support domain before labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    build_global_physical_support_audit,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--maximum_position_count", type=int, default=131072)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite global support audit")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    report = build_global_physical_support_audit(
        physical, maximum_position_count=int(args.maximum_position_count),
    )
    report["physical_map"] = {
        "path": str(physical_path),
        "file_sha256": file_sha256(physical_path),
        "content_sha256": physical.content_sha256,
        "maplet_count": int(physical.maplet_ids.size),
        "primitive_count": int(physical.primitive_centers.shape[0]),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf8",
    )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "content_sha256": report["content_sha256"],
        "position_count": report["position_count"],
        "implicit_pose_factor_count": report["implicit_pose_factor_count"],
        "decision": report["structural_gate"]["decision"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
