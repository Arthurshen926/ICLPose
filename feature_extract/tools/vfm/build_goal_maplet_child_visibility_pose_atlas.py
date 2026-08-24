"""Build a feature-free child/coarse-layout visibility pose atlas."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    build_child_visibility_pose_atlas,
)


def contributor_route_and_image_id(path: Path) -> tuple[str, str]:
    """Recover the frozen route/image ID without opening pose-bearing bytes."""

    name = Path(path).name
    if not name.endswith(".npz"):
        raise ValueError(f"contributor is not an NPZ: {path}")
    stem = name[:-4]
    fields = stem.split("__", 1)
    if (
        len(fields) != 2
        or not fields[0]
        or not fields[1]
        or "/" in fields[0]
        or "\\" in fields[0]
    ):
        raise ValueError(f"contributor filename lacks route__image identity: {path}")
    return fields[0], f"{fields[0]}/{fields[1]}"


def select_route_disjoint_contributors(
    directory: Path, allowed_trajectories: Sequence[str],
) -> tuple[list[Path], dict[str, object]]:
    """Select an explicit map-route allowlist before pose-bearing NPZs are read."""

    allowed_values = [str(value) for value in allowed_trajectories]
    if not allowed_values or len(set(allowed_values)) != len(allowed_values):
        raise ValueError("allowed trajectories must be a non-empty unique allowlist")
    if any(not value or "/" in value or "\\" in value or "__" in value for value in allowed_values):
        raise ValueError("allowed trajectory contains an invalid route identifier")
    allowed = set(allowed_values)
    inventory: list[tuple[Path, str, str]] = []
    for path in sorted(Path(directory).glob("*.npz")):
        route, image_id = contributor_route_and_image_id(path)
        inventory.append((path, route, image_id))
    if not inventory:
        raise ValueError("contributor directory is empty")
    observed = {route for _path, route, _image_id in inventory}
    absent = sorted(allowed - observed)
    if absent:
        raise ValueError(f"allowed trajectories are absent from contributors: {absent}")
    selected = [(path, route, image_id) for path, route, image_id in inventory if route in allowed]
    if not selected:
        raise ValueError("route allowlist selected no contributors")
    image_ids = [image_id for _path, _route, image_id in selected]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("selected contributor image IDs are duplicated")
    counts = Counter(route for _path, route, _image_id in selected)
    audit: dict[str, object] = {
        "route_allowlist_enforced": True,
        "allowed_trajectories": sorted(allowed),
        "excluded_trajectories": sorted(observed - allowed),
        "source_contributor_trajectories": sorted(counts),
        "source_contributor_trajectory_counts": {
            route: int(counts[route]) for route in sorted(counts)
        },
        "source_contributor_image_ids": image_ids,
        "source_contributor_image_ids_sha256": canonical_json_sha256(image_ids),
        "source_contributor_inventory_count_before_allowlist": len(inventory),
    }
    return [path for path, _route, _image_id in selected], audit


def bind_selected_contributor_bytes(
    paths: Sequence[Path], route_audit: dict[str, object],
) -> dict[str, object]:
    """Bind the ordered route identity to the exact contributor file bytes."""

    image_ids = [str(value) for value in route_audit["source_contributor_image_ids"]]
    if len(paths) != len(image_ids):
        raise ValueError("selected contributor path/identity inventory differs")
    inventory = [
        {
            "image_id": image_id,
            "resolved_path": str(Path(path).resolve()),
            "file_sha256": file_sha256(Path(path)),
        }
        for path, image_id in zip(paths, image_ids)
    ]
    return {
        **route_audit,
        "source_contributor_inventory_count": len(inventory),
        "source_contributor_inventory_sha256": canonical_json_sha256(inventory),
        "source_contributor_inventory_semantics": (
            "ordered_image_id_resolved_path_file_sha256_v1"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output_atlas", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--allowed_trajectory", action="append", required=True,
        help="map route to include; repeat for every allowed route",
    )
    parser.add_argument("--expected_view_count", type=int, required=True)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--maximum_global_children", type=int, default=1024)
    parser.add_argument("--maximum_children_per_cell", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_atlas)
    report_path = Path(args.output_json)
    if (output.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite visibility-atlas outputs")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    contributors, route_audit = select_route_disjoint_contributors(
        Path(args.contributors), args.allowed_trajectory,
    )
    route_audit = bind_selected_contributor_bytes(contributors, route_audit)
    if len(contributors) != int(args.expected_view_count):
        raise ValueError(
            f"expected {args.expected_view_count} contributors, found {len(contributors)}"
        )
    atlas = build_child_visibility_pose_atlas(
        physical,
        contributors,
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
        maximum_global_children=int(args.maximum_global_children),
        maximum_children_per_cell=int(args.maximum_children_per_cell),
        metadata={
            "source_contributor_directory": str(Path(args.contributors).resolve()),
            "physical_map_path": str(Path(args.physical_map).resolve()),
            "maximum_global_children": int(args.maximum_global_children),
            "maximum_children_per_cell": int(args.maximum_children_per_cell),
            **route_audit,
        },
    )
    atlas.save_npz(output)
    report = {
        "artifact_type": "goal_maplet_child_visibility_pose_atlas_build_v3",
        "content_hash_includes_schema_score_grid_and_physical_map": True,
        "output_atlas": str(output.resolve()),
        "content_sha256": atlas.content_sha256,
        "physical_map_sha256": atlas.physical_map_sha256,
        "view_count": atlas.view_count,
        "child_count": atlas.child_count,
        "grid_rows": atlas.grid_rows,
        "grid_cols": atlas.grid_cols,
        "layout_normalization": "joint_cell_child_probability_after_per_cell_truncation",
        "global_relation_count": int(atlas.global_child_rows.size),
        "layout_relation_count": int(atlas.layout_keys.size),
        "mean_global_children_per_view": float(
            atlas.global_child_rows.size / max(atlas.view_count, 1)
        ),
        "mean_layout_children_per_view": float(
            atlas.layout_keys.size / max(atlas.view_count, 1)
        ),
        "route_allowlist": route_audit,
        "coordinate_audit": {
            "coordinate_correct": bool(atlas.metadata.get("coordinate_correct")),
            "coordinate_contract": str(atlas.metadata.get("coordinate_contract", "")),
            "coordinate_transform_applied_before_visibility_aggregation": bool(
                atlas.metadata.get(
                    "coordinate_transform_applied_before_visibility_aggregation"
                )
            ),
            "coordinate_audit_count": int(atlas.metadata.get("coordinate_audit_count", -1)),
            "coordinate_audits_sha256": str(
                atlas.metadata.get("coordinate_audits_sha256", "")
            ),
            "summary": atlas.metadata.get("coordinate_audit_summary"),
        },
        "method_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_descriptors": False,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "uses_alike": False,
            "uses_pnp": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "coordinate_correct": True,
            "coordinate_contract": COORDINATE_CONTRACT,
            "output_is_final_pose": False,
            "output_role": "multi_modal_visibility_chart_centres",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
