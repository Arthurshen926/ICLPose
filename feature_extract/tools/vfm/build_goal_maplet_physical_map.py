"""Build and audit the exact Goal-Maplet 2DGS physical hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.build_2dgs_surface_feature_field import _clean_source_indices
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.audit import audit_physical_map
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    build_goal_maplet_physical_map,
    load_surface_primitive_geometry,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsAnchorMap


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface_elements", required=True)
    parser.add_argument("--clean_gaussian_ply", required=True)
    parser.add_argument("--legacy_maplets", required=True)
    parser.add_argument("--region_map", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--minimum_orientation_confidence", type=float, default=0.15)
    parser.add_argument("--minimum_child_count", type=int, default=8)
    parser.add_argument("--maximum_child_count", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path, audit_path = Path(args.output_map), Path(args.audit_json)
    if not args.force and (output_path.exists() or audit_path.exists()):
        raise FileExistsError("refusing to overwrite Goal-Maplet physical artifacts")
    inputs = {
        "surface_elements": Path(args.surface_elements),
        "clean_gaussian_ply": Path(args.clean_gaussian_ply),
        "legacy_maplets": Path(args.legacy_maplets),
        "region_map": Path(args.region_map),
        "mapping_pose_file": Path(args.mapping_pose_file),
    }
    poses = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(inputs["mapping_pose_file"])
    }
    physical_map = build_goal_maplet_physical_map(
        VfmSurfaceMapletBank.load_npz(inputs["legacy_maplets"]),
        Vfm2DgsAnchorMap.load_npz(inputs["region_map"]),
        load_surface_primitive_geometry(inputs["surface_elements"]),
        poses,
        clean_primitive_ids=_clean_source_indices(inputs["clean_gaussian_ply"]),
        minimum_orientation_confidence=float(args.minimum_orientation_confidence),
        minimum_child_count=int(args.minimum_child_count),
        maximum_child_count=int(args.maximum_child_count),
        metadata={
            "lineage_sha256": {name: file_sha256(path) for name, path in inputs.items()},
            "mapping_pose_use": "normal_orientation_only_not_stored",
        },
    )
    physical_map.save_npz(output_path)
    report = audit_physical_map(physical_map)
    report["inputs"] = {name: str(path) for name, path in inputs.items()}
    report["output_map"] = str(output_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
