"""Bounded pose-free PnP hypotheses from relative two-plane layout.

The generator associates query-plane regions with retrieved physical map
planes, then compares *unsigned relative normal angles* for pairs of distinct
associations.  Relative angles are invariant to the unknown camera rotation,
so candidate generation needs neither a pose nor a label.  Candidate poses
are frozen before contributors are opened for the diagnostic evaluation.

This is intentionally a bounded tail diagnostic.  It does not tune a fusion
weight and it does not replace the mainline pose selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_continuous_coordinate_pose_oracle import (
    _pose_error,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _choice_key,
    _load,
    _score,
    _solve,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


MINIMUM_TOKENS_PER_ASSOCIATION = 3
MINIMUM_TOKENS_PER_PAIR = 6
MAXIMUM_LAYOUT_PAIRS = 64


def _unsigned_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(3)
    right = np.asarray(right, np.float64).reshape(3)
    left /= max(float(np.linalg.norm(left)), 1e-15)
    right /= max(float(np.linalg.norm(right)), 1e-15)
    return float(np.degrees(np.arccos(np.clip(abs(float(left @ right)), 0.0, 1.0))))


def _layout_pair_groups(
    provenance: np.ndarray,
    tokens: np.ndarray,
    radio_score: np.ndarray,
    query_normals: np.ndarray,
    map_normals: np.ndarray,
    *,
    maximum_pairs: int = MAXIMUM_LAYOUT_PAIRS,
) -> list[dict[str, object]]:
    """Return stable pair groups ranked without a continuous fusion weight."""
    provenance = np.asarray(provenance, np.int64).reshape(-1, 3)
    tokens = np.asarray(tokens, np.int64).reshape(-1)
    radio_score = np.asarray(radio_score, np.float64).reshape(-1)
    query_normals = np.asarray(query_normals, np.float64).reshape(-1, 3)
    map_normals = np.asarray(map_normals, np.float64).reshape(-1, 3)
    if len(provenance) != len(tokens) or len(tokens) != len(radio_score):
        raise ValueError("layout association arrays differ")
    if not np.all(np.isfinite(radio_score)):
        raise ValueError("layout association score is nonfinite")

    associations: list[tuple[int, int, float, int]] = []
    for region, plane in np.unique(provenance[:, :2], axis=0):
        region, plane = int(region), int(plane)
        if not (0 <= region < len(query_normals) and 0 <= plane < len(map_normals)):
            raise ValueError("layout association normal index is out of range")
        rows = np.flatnonzero(
            (provenance[:, 0] == region) & (provenance[:, 1] == plane)
        )
        token_count = int(len(np.unique(tokens[rows])))
        if token_count >= MINIMUM_TOKENS_PER_ASSOCIATION:
            associations.append((region, plane, float(np.max(radio_score[rows])), token_count))

    pairs: list[dict[str, object]] = []
    for index, left in enumerate(associations):
        for right in associations[index + 1:]:
            if left[0] == right[0] or left[1] == right[1]:
                continue
            rows = np.flatnonzero(
                ((provenance[:, 0] == left[0]) & (provenance[:, 1] == left[1]))
                | ((provenance[:, 0] == right[0]) & (provenance[:, 1] == right[1]))
            )
            token_count = int(len(np.unique(tokens[rows])))
            if token_count < MINIMUM_TOKENS_PER_PAIR:
                continue
            query_angle = _unsigned_angle_deg(
                query_normals[left[0]], query_normals[right[0]],
            )
            map_angle = _unsigned_angle_deg(map_normals[left[1]], map_normals[right[1]])
            pairs.append({
                "rows": rows,
                "left_region": left[0], "left_plane": left[1],
                "right_region": right[0], "right_plane": right[1],
                "angle_error_deg": abs(query_angle - map_angle),
                "minimum_radio_score": min(left[2], right[2]),
                "token_count": token_count,
            })
    pairs.sort(key=lambda row: (
        float(row["angle_error_deg"]),
        -float(row["minimum_radio_score"]),
        -int(row["token_count"]),
        int(row["left_region"]), int(row["left_plane"]),
        int(row["right_region"]), int(row["right_plane"]),
    ))
    return pairs[:int(maximum_pairs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--existing_selected_pose", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_candidates", type=Path, required=True)
    parser.add_argument("--output_frozen_selected_pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (
        args.output, args.output_frozen_candidates, args.output_frozen_selected_pose,
    )):
        raise FileExistsError("refusing to overwrite multi-plane layout result")

    arrays, upstream = _load(args.frozen_correspondences)
    existing_selected_arrays, selected_meta = _load_pose_candidate(args.existing_selected_pose)
    names = arrays["names"].astype(str)
    if not np.array_equal(names, existing_selected_arrays["names"].astype(str)):
        raise ValueError("existing selected pose query order differs")
    planar_map = GeometryNativePlanarMap.load_npz(args.planar_map)

    candidate_lists: list[list[dict[str, object]]] = []
    pose_free_rows: list[dict[str, object]] = []
    for query_index, name in enumerate(names.tolist()):
        lo, hi = map(int, arrays["correspondence_offsets"][query_index:query_index + 2])
        world = arrays["world_points"][lo:hi]
        tokens = arrays["query_tokens"][lo:hi]
        provenance = arrays["provenance_region_plane_atlas_row"][lo:hi]
        measurements = arrays.get("query_measurements_xy")
        measurements = None if measurements is None else measurements[lo:hi]
        radio = arrays.get("radio_match_score")
        radio = np.ones(len(world), np.float64) if radio is None else radio[lo:hi]
        query_planes, query_meta = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        groups = _layout_pair_groups(
            provenance, tokens, radio, query_planes.normals_camera,
            planar_map.normals_world,
        )
        candidates: list[dict[str, object]] = []
        for rank, group in enumerate(groups):
            pose = _solve(
                world, tokens, arrays["camera_matrices"][query_index],
                float(arrays["radial_k1"][query_index]), group["rows"], measurements,
            )
            if pose is None:
                continue
            score = _score(
                pose, world, tokens, provenance,
                arrays["camera_matrices"][query_index],
                float(arrays["radial_k1"][query_index]), measurements,
            )
            score.update({key: value for key, value in group.items() if key != "rows"})
            score["layout_rank"] = rank
            candidates.append(score)
        candidate_lists.append(candidates)
        pose_free_rows.append({
            "name": name,
            "query_plane_content_sha256": query_meta.get("content_sha256"),
            "layout_pair_group_count": int(len(groups)),
            "solved_candidate_count": int(len(candidates)),
        })

    offsets = np.zeros(len(candidate_lists) + 1, np.int64)
    for index, candidates in enumerate(candidate_lists):
        offsets[index + 1] = offsets[index] + len(candidates)
    flat = [candidate for candidates in candidate_lists for candidate in candidates]
    frozen_arrays = {
        "names": arrays["names"],
        "candidate_offsets": offsets,
        "candidate_pose_w2c": np.asarray([row["pose"] for row in flat], np.float64).reshape(-1, 4, 4),
        "candidate_layout_rank": np.asarray([row["layout_rank"] for row in flat], np.int64),
        "candidate_angle_error_deg": np.asarray([row["angle_error_deg"] for row in flat], np.float64),
        "candidate_inlier_count": np.asarray([row["inlier_count"] for row in flat], np.int64),
    }
    frozen_meta = {
        "artifact_type": "goal_maplet_relative_multiplane_layout_pnp_candidates_v1",
        "arrays_sha256": arrays_sha256(frozen_arrays),
        "query_count": int(len(names)),
        "maximum_layout_pairs": MAXIMUM_LAYOUT_PAIRS,
        "ranking": "unsigned_relative_normal_angle_error_then_min_RADIO_then_token_support_then_ids",
        "continuous_fusion_weight_count": 0,
        "candidate_generation_reads_pose_or_ground_truth": False,
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": upstream.get("content_sha256"),
        "query_plane_inventory_tree_bound_by_individual_content_hash": True,
        "planar_map_file_sha256": file_sha256(args.planar_map),
    }
    frozen_meta["content_sha256"] = canonical_json_sha256(frozen_meta)
    args.output_frozen_candidates.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_candidates, **frozen_arrays,
        metadata_json=np.asarray(json.dumps(frozen_meta, sort_keys=True)),
    )

    selected_candidates = [
        (max(candidates, key=lambda row: _choice_key(row, "supported_entities"))
         if candidates else None)
        for candidates in candidate_lists
    ]
    layout_selected_arrays = {
        "names": arrays["names"],
        "pose_w2c": np.asarray([
            np.full((4, 4), np.nan) if row is None else row["pose"]
            for row in selected_candidates
        ], np.float64),
        "usable": np.asarray([row is not None for row in selected_candidates], bool),
    }
    selected_frozen_meta = {
        "artifact_type": "goal_maplet_relative_multiplane_layout_selected_pose_v1",
        "arrays_sha256": arrays_sha256(layout_selected_arrays),
        "query_count": int(len(names)),
        "selection_rule": "maximum_supported_region_plane_view_counts_then_inliers_then_reprojection",
        "candidate_inventory_file_sha256": file_sha256(args.output_frozen_candidates),
        "candidate_inventory_content_sha256": frozen_meta["content_sha256"],
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "production_eligible": False,
    }
    selected_frozen_meta["content_sha256"] = canonical_json_sha256(selected_frozen_meta)
    np.savez_compressed(
        args.output_frozen_selected_pose, **layout_selected_arrays,
        metadata_json=np.asarray(json.dumps(selected_frozen_meta, sort_keys=True)),
    )

    postlabel_rows = []
    for query_index, name in enumerate(names.tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        candidates = candidate_lists[query_index]
        errors = [_pose_error(row["pose"], target) for row in candidates]
        oracle_hit = any(t <= 2.0 and r <= 45.0 for t, r in errors)
        chosen = selected_candidates[query_index]
        chosen_error = (float("inf"), float("inf")) if chosen is None else _pose_error(chosen["pose"], target)
        existing_error = _pose_error(
            existing_selected_arrays["pose_w2c"][query_index], target,
        )
        postlabel_rows.append({
            "name": name,
            "existing_selected_is_2m45": bool(existing_error[0] <= 2.0 and existing_error[1] <= 45.0),
            "layout_pool_oracle_is_2m45": bool(oracle_hit),
            "layout_supported_entities_selected_is_2m45": bool(
                chosen_error[0] <= 2.0 and chosen_error[1] <= 45.0
            ),
            "union_existing_plus_layout_pool_oracle_is_2m45": bool(
                oracle_hit or (existing_error[0] <= 2.0 and existing_error[1] <= 45.0)
            ),
            "minimum_layout_candidate_translation_m": (
                None if not errors else float(min(value[0] for value in errors))
            ),
        })

    def count(field: str) -> int:
        return int(sum(bool(row[field]) for row in postlabel_rows))

    report = {
        "artifact_type": "goal_maplet_relative_multiplane_layout_pnp_postlabel_diagnostic_v1",
        "evaluation_role": "HISTORICAL_POSTHOC_DIAGNOSTIC_NOT_PROMOTION",
        "split_name": args.split_name,
        "query_count": int(len(names)),
        "frozen_candidate_file_sha256": file_sha256(args.output_frozen_candidates),
        "frozen_candidate_content_sha256": frozen_meta["content_sha256"],
        "frozen_selected_pose_file_sha256": file_sha256(args.output_frozen_selected_pose),
        "frozen_selected_pose_content_sha256": selected_frozen_meta["content_sha256"],
        "existing_selected_file_sha256": file_sha256(args.existing_selected_pose),
        "existing_selected_content_sha256": selected_meta.get("content_sha256"),
        "existing_selected_2m45_count": count("existing_selected_is_2m45"),
        "layout_pool_oracle_2m45_count": count("layout_pool_oracle_is_2m45"),
        "layout_supported_entities_selected_2m45_count": count(
            "layout_supported_entities_selected_is_2m45"
        ),
        "union_existing_plus_layout_pool_oracle_2m45_count": count(
            "union_existing_plus_layout_pool_oracle_is_2m45"
        ),
        "pose_free_diagnostics": pose_free_rows,
        "postlabel_rows": postlabel_rows,
        "query_pose_or_ground_truth_opened_after_candidate_freeze": True,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("pose_free_diagnostics", "postlabel_rows")}, indent=2))


if __name__ == "__main__":
    main()
