"""Evaluate a frozen factorized local pose domain around pose-free seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_basin import (
    CONTINUOUS_HIERARCHICAL_POSE_BASIN_SEMANTICS,
    FACTORIZED_POSE_BASIN_SEMANTICS,
    ContinuousHierarchicalPoseBasin,
    build_default_factorized_pose_basin,
    continuous_factorized_oracle_error,
    factorized_oracle_error,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


SCHEMA_V1 = "goal_maplet_factorized_pose_basin_oracle_v1"
SCHEMA_V2 = "goal_maplet_continuous_hierarchical_pose_basin_oracle_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int)
    parser.add_argument(
        "--domain_semantics",
        choices=("discrete_grid_v1", "continuous_hierarchical_v2"),
        default="discrete_grid_v1",
    )
    parser.add_argument("--translation_half_extent_m", type=float, default=8.0)
    parser.add_argument("--rotation_radius_deg", type=float, default=45.0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite factorized basin oracle")
    dataset_path = Path(args.dataset)
    arrays, metadata = load_pose_candidate_dataset(
        dataset_path, require_rendered_targets=False,
    )
    if args.domain_semantics == "discrete_grid_v1":
        domain = build_default_factorized_pose_basin()
        domain_arrays = {
            "translation_offsets_world": domain.translation_offsets_world,
            "rotation_offsets_left": domain.rotation_offsets_left,
        }
        schema = SCHEMA_V1
        semantics = FACTORIZED_POSE_BASIN_SEMANTICS
    else:
        domain = ContinuousHierarchicalPoseBasin(
            translation_half_extent_m=args.translation_half_extent_m,
            rotation_radius_deg=args.rotation_radius_deg,
        )
        domain_arrays = {
            "translation_half_extent_m": np.asarray(
                [domain.translation_half_extent_m], dtype=np.float64,
            ),
            "rotation_radius_deg": np.asarray(
                [domain.rotation_radius_deg], dtype=np.float64,
            ),
        }
        schema = SCHEMA_V2
        semantics = CONTINUOUS_HIERARCHICAL_POSE_BASIN_SEMANTICS
    begin = int(args.query_start)
    end = int(arrays["image_ids"].size)
    if args.maximum_queries is not None:
        end = min(end, begin + int(args.maximum_queries))
    if not 0 <= begin < end:
        raise ValueError("query range is empty")
    # The fixed domain is completely built and hashed before candidate zero is
    # opened as the diagnostic target below.
    domain_sha256 = arrays_sha256(domain_arrays)
    rows = []
    for query in range(begin, end):
        seed_rows = np.flatnonzero(arrays["candidate_valid"][query])
        seed_rows = seed_rows[seed_rows != 0]
        if args.domain_semantics == "discrete_grid_v1":
            translation, rotation, local_seed, position, orientation = factorized_oracle_error(
                arrays["candidate_poses_w2c"][query, seed_rows],
                arrays["candidate_poses_w2c"][query, 0],
                domain,
            )
        else:
            translation, rotation, local_seed = continuous_factorized_oracle_error(
                arrays["candidate_poses_w2c"][query, seed_rows],
                arrays["candidate_poses_w2c"][query, 0],
                domain,
            )
            position = None
            orientation = None
        rows.append({
            "image_id": str(arrays["image_ids"][query]),
            "seed_candidate_index": int(seed_rows[local_seed]),
            "translation_offset_index": None if position is None else int(position),
            "rotation_offset_index": None if orientation is None else int(orientation),
            "translation_m": translation,
            "rotation_deg": rotation,
            "strict_0_5m_5deg": bool(translation <= 0.5 + 1.0e-6 and rotation <= 5.0 + 1.0e-5),
            "loose_1m_10deg": bool(translation <= 1.0 + 1.0e-6 and rotation <= 10.0 + 1.0e-5),
        })
    per_route_metrics = {}
    for route in sorted({row["image_id"].split("/", 1)[0] for row in rows}):
        route_rows = [row for row in rows if row["image_id"].split("/", 1)[0] == route]
        per_route_metrics[route] = {
            "query_count": len(route_rows),
            "strict_oracle_acquisition_rate": float(np.mean([
                row["strict_0_5m_5deg"] for row in route_rows
            ])),
            "loose_oracle_acquisition_rate": float(np.mean([
                row["loose_1m_10deg"] for row in route_rows
            ])),
            "median_translation_m": float(np.median([
                row["translation_m"] for row in route_rows
            ])),
            "median_rotation_deg": float(np.median([
                row["rotation_deg"] for row in route_rows
            ])),
        }
    report = {
        "artifact_type": schema,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "domain_semantics": semantics,
        "domain_content_sha256": domain_sha256,
        "translation_offset_count": (
            int(domain.translation_offsets_world.shape[0])
            if args.domain_semantics == "discrete_grid_v1" else None
        ),
        "rotation_offset_count": (
            int(domain.rotation_offsets_left.shape[0])
            if args.domain_semantics == "discrete_grid_v1" else None
        ),
        "implicit_pose_count_per_seed": (
            domain.implicit_pose_count_per_seed
            if args.domain_semantics == "discrete_grid_v1" else None
        ),
        "translation_half_extent_m": (
            None if args.domain_semantics == "discrete_grid_v1"
            else domain.translation_half_extent_m
        ),
        "rotation_radius_deg": (
            None if args.domain_semantics == "discrete_grid_v1"
            else domain.rotation_radius_deg
        ),
        "domain_is_continuous_and_not_pose_enumeration": bool(
            args.domain_semantics == "continuous_hierarchical_v2"
        ),
        "domain_built_before_target_pose_opened": True,
        "query_range": [begin, end],
        "query_count": len(rows),
        "strict_oracle_acquisition_rate": float(np.mean([row["strict_0_5m_5deg"] for row in rows])),
        "loose_oracle_acquisition_rate": float(np.mean([row["loose_1m_10deg"] for row in rows])),
        "median_translation_m": float(np.median([row["translation_m"] for row in rows])),
        "median_rotation_deg": float(np.median([row["rotation_deg"] for row in rows])),
        "per_route_metrics": per_route_metrics,
        "rows": rows,
        "uses_query_pose_for_domain_generation": False,
        "uses_query_ground_truth_for_domain_generation": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "selection_or_localization_success_claim": False,
        "production_eligible": False,
        "limitation": (
            "continuous support oracle; no deployable scoring or hierarchical expansion yet"
            if args.domain_semantics == "continuous_hierarchical_v2"
            else "implicit grid support oracle; no deployable scoring or hierarchical expansion yet"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "query_count", "strict_oracle_acquisition_rate", "loose_oracle_acquisition_rate",
        "median_translation_m", "median_rotation_deg", "implicit_pose_count_per_seed",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
