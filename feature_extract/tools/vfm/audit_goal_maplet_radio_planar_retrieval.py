"""Post-label audit of RADIO finite-plane retrieval on historical held views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import _load_aligned_vertices
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_planar_support_pose import (
    _map_regions,
    _normalise,
    _query_regions,
    _radio_records,
    _radio_tokens,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import StrictHeldRayInventory
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import ProjectiveExactFaceSeamAuthority
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import extract_query_plane_regions


SCHEMA = "goal_maplet_radio_planar_region_retrieval_postlabel_audit_v1"
RANKS = (1, 4, 16, 32, 64)
MAXIMUM_NORMAL_ANGLE_DEG = 30.0
MAXIMUM_MEDIAN_DISTANCE_M = 0.75
MAXIMUM_P90_DISTANCE_M = 1.50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected_authority_content_sha256", required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--moge3_query", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite RADIO plane retrieval audit")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("RADIO audit authority differs from pin")
    vertices = _load_aligned_vertices(args.alignment, authority)
    records = _radio_records(args.radio_manifest)
    mapping, map_token_hash = _map_regions(authority, vertices, records)
    map_descriptor = np.asarray([row["descriptor"] for row in mapping], np.float32)
    map_trees = [cKDTree(row["token_points_world"]) for row in mapping]
    query_manifest_path = args.moge3_query / "manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    names = [str(row["name"]) for row in query_manifest.get("rows", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("MoGe3 query inventory is empty or duplicated")

    frozen = []
    for name in names:
        query_feature, query_hash = _radio_tokens(name, records)
        with np.load(args.moge3_query / f"{name}.npz", allow_pickle=False) as data:
            points = np.asarray(data["points_camera"], np.float64)
            normals = np.asarray(data["normal_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
        planes = extract_query_plane_regions(points, normals, valid)
        query = _query_regions(planes, points, query_feature)
        descriptor = np.asarray([row["descriptor"] for row in query], np.float32)
        cosine = descriptor @ map_descriptor.T if len(query) else np.empty((0, len(mapping)))
        order = np.argsort(-cosine, axis=1, kind="stable")
        frozen.append({
            "name": name,
            "query_radio_file_sha256": query_hash,
            "points_camera": points,
            "valid": valid,
            "query_regions": query,
            "cosine": cosine,
            "order": order,
        })

    # Labels and poses become visible only after every full ranking above is frozen.
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    held_row = {name: row for row, name in enumerate(held.view_names.astype(str))}
    if set(held_row) != set(names):
        raise ValueError("held inventory differs from frozen RADIO ranking inventory")
    rows = []
    all_best_rank, all_weight = [], []
    for item in frozen:
        view = held_row[item["name"]]
        common = held.reference_valid[view] & item["valid"]
        scale = float(np.median(
            held.reference_depth_m[view][common] / item["points_camera"][..., 2][common]
        ))
        c2w = held.camera_to_world[view]
        query_rows = []
        for query_row, query in enumerate(item["query_regions"]):
            query_world = scale * (query["token_points_camera"] @ c2w[:3, :3].T) + c2w[:3, 3]
            query_normal_world = query["normal_camera"] @ c2w[:3, :3].T
            compatible = []
            physical_rows = []
            for map_row, mapping_row in enumerate(mapping):
                angle = float(np.degrees(np.arccos(np.clip(
                    abs(float(query_normal_world @ mapping_row["normal_world"])), -1.0, 1.0,
                ))))
                if angle > MAXIMUM_NORMAL_ANGLE_DEG:
                    continue
                distance = map_trees[map_row].query(
                    query_world, k=1,
                )[0]
                median = float(np.median(distance))
                p90 = float(np.quantile(distance, 0.90))
                if median <= MAXIMUM_MEDIAN_DISTANCE_M and p90 <= MAXIMUM_P90_DISTANCE_M:
                    compatible.append(map_row)
                    physical_rows.append({
                        "map_region": map_row,
                        "map_chart_name": mapping_row["chart_name"],
                        "normal_angle_deg": angle,
                        "distance_median_m": median,
                        "distance_p90_m": p90,
                    })
            ranking = item["order"][query_row]
            inverse = np.empty((len(mapping),), np.int64)
            inverse[ranking] = np.arange(1, len(mapping) + 1)
            best_rank = int(np.min(inverse[compatible])) if compatible else None
            weight = int(len(query["token_ids"]))
            if best_rank is not None:
                all_best_rank.append(best_rank)
                all_weight.append(weight)
            query_rows.append({
                "query_region": query_row,
                "query_token_count": weight,
                "physically_compatible_map_region_count": len(compatible),
                "best_physical_rank": best_rank,
                "top1_map_region": int(ranking[0]),
                "top1_cosine": float(item["cosine"][query_row, ranking[0]]),
                "physical_candidates": physical_rows,
            })
        rows.append({
            "name": item["name"],
            "moge_metric_scale_postlabel": scale,
            "query_plane_count": len(item["query_regions"]),
            "physically_matchable_plane_count": int(sum(
                row["best_physical_rank"] is not None for row in query_rows
            )),
            "query_regions": query_rows,
        })
    rank = np.asarray(all_best_rank, np.int64)
    weight = np.asarray(all_weight, np.float64)
    total_plane = sum(row["query_plane_count"] for row in rows)
    summary = {
        "query_count": len(rows),
        "query_plane_count": total_plane,
        "physically_matchable_plane_count": int(len(rank)),
        "physically_matchable_plane_fraction": float(len(rank) / max(total_plane, 1)),
        "conditional_unweighted_recall": {
            f"R@{cutoff}": float(np.mean(rank <= cutoff)) if len(rank) else 0.0
            for cutoff in RANKS
        },
        "conditional_token_weighted_recall": {
            f"R@{cutoff}": float(np.average(rank <= cutoff, weights=weight)) if len(rank) else 0.0
            for cutoff in RANKS
        },
        "best_physical_rank_median": float(np.median(rank)) if len(rank) else None,
        "best_physical_rank_p90": float(np.quantile(rank, 0.90)) if len(rank) else None,
    }
    payload = {
        "artifact_type": SCHEMA,
        "authority_file_sha256": file_sha256(args.authority),
        "authority_content_sha256": args.expected_authority_content_sha256,
        "alignment_manifest_file_sha256": file_sha256(args.alignment / "manifest.json"),
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "moge3_query_manifest_file_sha256": file_sha256(query_manifest_path),
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "map_radio_file_inventory_sha256": canonical_json_sha256(map_token_hash),
        "map_plane_region_count": len(mapping),
        "postlabel_physical_compatibility": {
            "moge_scale_source": "median_held_reference_depth_over_query_MoGe3_depth",
            "maximum_normal_angle_deg": MAXIMUM_NORMAL_ANGLE_DEG,
            "maximum_query_to_map_token_distance_median_m": MAXIMUM_MEDIAN_DISTANCE_M,
            "maximum_query_to_map_token_distance_p90_m": MAXIMUM_P90_DISTANCE_M,
        },
        "phase_separation": {
            "all_RADIO_plane_rankings_frozen_before_held_open": True,
            "held_pose_or_reference_consumed_by_ranking": False,
            "held_used_only_for_physical_label_and_metric_scale_in_phase2": True,
        },
        "held_inventory_previously_opened": True,
        "blind_or_preregistered_claim": False,
        "production_eligible": False,
        "promotion_eligible": False,
        "summary": summary,
        "rows": rows,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"],
        "summary": summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
