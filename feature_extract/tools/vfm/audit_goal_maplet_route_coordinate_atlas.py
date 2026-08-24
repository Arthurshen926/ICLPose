"""Independently replay a route- and coordinate-correct visibility atlas.

This audit deliberately reconstructs the contributor inventory from immutable
filenames and file bytes.  With ``--full_rebuild`` it also recomputes every
atlas array, which catches a stale or incorrectly sampled atlas even when its
own internal content hash is self-consistent.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
    load_contributors_in_radio_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    build_child_visibility_pose_atlas,
)


def _contributor_identity(path: Path) -> tuple[str, str]:
    name = path.name
    if not name.endswith(".npz"):
        raise ValueError(f"contributor is not an NPZ: {path}")
    values = name[:-4].split("__", 1)
    if len(values) != 2 or not all(values):
        raise ValueError(f"contributor filename lacks route identity: {path}")
    route, image = values
    if any(value in route for value in ("/", "\\")):
        raise ValueError(f"invalid contributor route: {route}")
    return route, f"{route}/{image}"


def audit_atlas(
    *,
    atlas_path: Path,
    physical_map_path: Path,
    contributors_directory: Path,
    allowed_trajectories: list[str],
    excluded_trajectories: list[str],
    full_rebuild: bool,
) -> dict[str, object]:
    allowed = set(str(value) for value in allowed_trajectories)
    excluded = set(str(value) for value in excluded_trajectories)
    if (
        not allowed
        or len(allowed) != len(allowed_trajectories)
        or len(excluded) != len(excluded_trajectories)
        or allowed & excluded
    ):
        raise ValueError("route audit requires unique disjoint allow/exclude sets")

    atlas = ChildVisibilityPoseAtlas.load_npz(atlas_path)
    physical = GoalMapletPhysicalMap.load_npz(physical_map_path)
    metadata = dict(atlas.metadata or {})
    if atlas.physical_map_sha256 != physical.content_sha256:
        raise ValueError("atlas and physical map hashes differ")
    if metadata.get("route_allowlist_enforced") is not True:
        raise ValueError("atlas lacks an enforced route allowlist")
    if set(str(value) for value in metadata.get("allowed_trajectories", ())) != allowed:
        raise ValueError("atlas allowed routes differ from the audit allowlist")
    if metadata.get("coordinate_correct") is not True:
        raise ValueError("atlas is not marked coordinate-correct")
    if metadata.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise ValueError("atlas coordinate contract differs")
    if metadata.get("coordinate_transform_applied_before_visibility_aggregation") is not True:
        raise ValueError("atlas did not apply the coordinate transform before aggregation")

    selected: list[tuple[Path, str, str]] = []
    observed_routes: set[str] = set()
    for path in sorted(contributors_directory.glob("*.npz")):
        route, image_id = _contributor_identity(path)
        observed_routes.add(route)
        if route in allowed:
            selected.append((path, route, image_id))
    if not selected or not allowed.issubset(observed_routes):
        raise ValueError("contributor inventory does not cover the allowed routes")
    paths = [path for path, _route, _image_id in selected]
    image_ids = [image_id for _path, _route, image_id in selected]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("selected contributor image IDs are duplicated")
    counts = Counter(route for _path, route, _image_id in selected)
    expected_counts = {
        str(key): int(value)
        for key, value in metadata.get("source_contributor_trajectory_counts", {}).items()
    }
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError("route counts differ from atlas metadata")
    if image_ids != [str(value) for value in metadata.get("source_contributor_image_ids", ())]:
        raise ValueError("ordered contributor image IDs differ from atlas metadata")
    if any(image_id.split("/", 1)[0] in excluded for image_id in image_ids):
        raise ValueError("an excluded route appears in the selected contributor inventory")

    inventory = [
        {
            "image_id": image_id,
            "resolved_path": str(path.resolve()),
            "file_sha256": file_sha256(path),
        }
        for path, image_id in zip(paths, image_ids)
    ]
    image_ids_sha256 = canonical_json_sha256(image_ids)
    inventory_sha256 = canonical_json_sha256(inventory)
    if image_ids_sha256 != metadata.get("source_contributor_image_ids_sha256"):
        raise ValueError("contributor image-ID hash differs")
    if inventory_sha256 != metadata.get("source_contributor_inventory_sha256"):
        raise ValueError("contributor byte inventory hash differs")
    if len(paths) != atlas.view_count:
        raise ValueError("selected contributor count differs from atlas view count")

    poses: list[np.ndarray] = []
    coordinate_audits: list[dict[str, object]] = []
    for path in paths:
        labels, coordinate_audit = load_contributors_in_radio_coordinates(path)
        if coordinate_audit.get("coordinate_contract") != COORDINATE_CONTRACT:
            raise ValueError("replayed contributor coordinate contract differs")
        poses.append(np.asarray(labels.pose_w2c, dtype=np.float64))
        coordinate_audits.append(dict(coordinate_audit))
    coordinate_audits_sha256 = canonical_json_sha256(coordinate_audits)
    if coordinate_audits_sha256 != metadata.get("coordinate_audits_sha256"):
        raise ValueError("replayed coordinate-audit hash differs")
    if int(metadata.get("coordinate_audit_count", -1)) != len(paths):
        raise ValueError("coordinate-audit count differs")
    maximum_pose_delta = float(
        np.max(np.abs(np.stack(poses, axis=0) - atlas.poses_w2c), initial=0.0)
    )
    if maximum_pose_delta != 0.0:
        raise ValueError("atlas poses differ from the selected contributor poses")

    global_matrix, layout_matrix = atlas.sparse_matrices()
    global_norm_squared = np.asarray(
        global_matrix.multiply(global_matrix).sum(axis=1)
    ).reshape(-1)
    layout_norm_squared = np.asarray(
        layout_matrix.multiply(layout_matrix).sum(axis=1)
    ).reshape(-1)
    if not np.allclose(global_norm_squared, 1.0, atol=2e-6, rtol=0.0):
        raise ValueError("atlas global distributions are not L2-normalized")
    if not np.allclose(layout_norm_squared, 1.0, atol=2e-6, rtol=0.0):
        raise ValueError("atlas layout distributions are not L2-normalized")

    rebuild_equal: bool | None = None
    if full_rebuild:
        rebuilt = build_child_visibility_pose_atlas(
            physical,
            paths,
            grid_rows=atlas.grid_rows,
            grid_cols=atlas.grid_cols,
            maximum_global_children=int(metadata["maximum_global_children"]),
            maximum_children_per_cell=int(metadata["maximum_children_per_cell"]),
            metadata={},
        )
        array_names = (
            "poses_w2c",
            "global_offsets",
            "global_child_rows",
            "global_weights",
            "layout_offsets",
            "layout_keys",
            "layout_weights",
        )
        rebuild_equal = bool(
            rebuilt.content_sha256 == atlas.content_sha256
            and all(
                np.array_equal(getattr(rebuilt, name), getattr(atlas, name))
                for name in array_names
            )
        )
        if not rebuild_equal:
            raise ValueError("full deterministic atlas replay differs")

    report: dict[str, object] = {
        "artifact_type": "goal_maplet_route_coordinate_atlas_audit_v1",
        "passed": True,
        "atlas": str(atlas_path.resolve()),
        "atlas_file_sha256": file_sha256(atlas_path),
        "atlas_content_sha256": atlas.content_sha256,
        "physical_map": str(physical_map_path.resolve()),
        "physical_map_sha256": physical.content_sha256,
        "view_count": atlas.view_count,
        "allowed_trajectories": sorted(allowed),
        "excluded_trajectories": sorted(excluded),
        "excluded_route_row_count": 0,
        "source_contributor_trajectory_counts": dict(sorted(counts.items())),
        "source_contributor_image_ids_sha256": image_ids_sha256,
        "source_contributor_inventory_sha256": inventory_sha256,
        "coordinate_correct": True,
        "coordinate_contract": COORDINATE_CONTRACT,
        "coordinate_audits_sha256": coordinate_audits_sha256,
        "maximum_pose_delta_vs_source": maximum_pose_delta,
        "global_l2_norm_squared_range": [
            float(global_norm_squared.min()), float(global_norm_squared.max())
        ],
        "layout_l2_norm_squared_range": [
            float(layout_norm_squared.min()), float(layout_norm_squared.max())
        ],
        "full_deterministic_rebuild_equal": rebuild_equal,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--allowed_trajectory", action="append", required=True)
    parser.add_argument("--excluded_trajectory", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--full_rebuild", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite atlas audit")
    report = audit_atlas(
        atlas_path=Path(args.atlas),
        physical_map_path=Path(args.physical_map),
        contributors_directory=Path(args.contributors),
        allowed_trajectories=list(args.allowed_trajectory),
        excluded_trajectories=list(args.excluded_trajectory),
        full_rebuild=bool(args.full_rebuild),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
