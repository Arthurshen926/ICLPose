"""Extract the stable oriented-disk surface table from a raw 2DGS PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.build_goal_maplet_geometry_native_planar_map import (
    _primitive_table_from_2dgs_ply,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two_dgs_ply", type=Path, required=True)
    parser.add_argument("--minimum_opacity", type=float, default=0.05)
    parser.add_argument("--maximum_scale", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite primitive surface table")
    table = _primitive_table_from_2dgs_ply(
        args.two_dgs_ply,
        minimum_opacity=float(args.minimum_opacity),
        maximum_scale=args.maximum_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.save_npz(args.output)
    loaded = type(table).load_npz(args.output)
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "primitive_count": int(len(loaded.primitive_ids)),
        "source_2dgs_ply_file_sha256": loaded.metadata.get("source_2dgs_ply_file_sha256"),
        "minimum_opacity": loaded.metadata.get("minimum_opacity"),
    }, indent=2))


if __name__ == "__main__":
    main()
