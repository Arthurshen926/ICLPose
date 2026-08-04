"""Build the query-independent V8.1 maplet/cell virtual-pose lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
)
from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    PhysicalMapletGraph,
)
from feature_extract.vfm.localization_v81.virtual_pose_lattice import (
    build_virtual_pose_lattice,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_graph", required=True)
    parser.add_argument("--pose_vote_bank", required=True)
    parser.add_argument("--camera_contributor_dir", required=True)
    parser.add_argument("--output_lattice", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--position_spacing_m", type=float, default=1.0)
    parser.add_argument("--expansion_radius_m", type=float, default=3.0)
    parser.add_argument("--rotation_prototypes", type=int, default=128)
    parser.add_argument("--cell_grid_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _camera_from_contributors(directory: Path) -> ColmapCamera:
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError("camera contributor directory is empty")
    with np.load(paths[0], allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64)),
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_lattice)
    summary = Path(args.summary_json)
    if (output.exists() or summary.exists()) and not args.force:
        raise FileExistsError("refusing to overwrite V8.1 virtual lattice")
    physical_path = Path(args.physical_graph)
    vote_path = Path(args.pose_vote_bank)
    physical = PhysicalMapletGraph.load_npz(physical_path)
    votes = AnonymousMapletPoseVoteBank.load_npz(vote_path)
    if not np.all(np.isin(np.unique(votes.component_maplet_ids), physical.maplet_ids)):
        raise ValueError("pose-vote maplet identity differs from physical graph")
    camera = _camera_from_contributors(Path(args.camera_contributor_dir))
    lattice = build_virtual_pose_lattice(
        votes,
        physical,
        camera,
        physical_graph_path=physical_path,
        pose_vote_bank_path=vote_path,
        position_spacing_m=float(args.position_spacing_m),
        expansion_radius_m=float(args.expansion_radius_m),
        rotation_prototypes=int(args.rotation_prototypes),
        cell_grid_size=int(args.cell_grid_size),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    lattice.save_npz(output)
    report = {
        "stage": "v81_build_query_independent_virtual_pose_lattice",
        "output_lattice": str(output.resolve()),
        "pose_count": lattice.pose_count,
        "position_count": int(lattice.positions.shape[0]),
        "rotation_prototype_count": int(lattice.rotations_w2c.shape[0]),
        "visibility_key_count": int(lattice.visibility_keys.size),
        "compressed_size_bytes": int(output.stat().st_size),
        "metadata": dict(lattice.metadata),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
