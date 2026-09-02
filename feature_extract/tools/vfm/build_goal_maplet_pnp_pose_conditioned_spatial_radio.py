"""Select PnP candidates by pose-nearby full-image spatial RADIO verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.tools.vfm.build_goal_maplet_pnp_pose_conditioned_view_context import (
    _load_global_view_field,
    _pose_center_forward,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _radio, _records
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


PROJECTION_SEED = 260901
PROJECTED_CHANNELS = 128


def _projection() -> np.ndarray:
    generator = np.random.default_rng(PROJECTION_SEED)
    return generator.choice(
        np.asarray([-1.0, 1.0], np.float32), size=(1280, PROJECTED_CHANNELS),
    ) / np.sqrt(float(PROJECTED_CHANNELS))


def _compact(feature: np.ndarray, projection: np.ndarray) -> np.ndarray:
    value = np.asarray(feature, np.float32).reshape(36, 64, 1280)
    value = value.reshape(18, 2, 32, 2, 1280).mean((1, 3)).reshape(-1, 1280)
    value = value @ projection
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def _spatial_match_score(query: np.ndarray, source: np.ndarray) -> tuple[int, float]:
    cosine = np.asarray(query, np.float32) @ np.asarray(source, np.float32).T
    top2 = np.argpartition(cosine, -2, axis=1)[:, -2:]
    values = np.take_along_axis(cosine, top2, axis=1)
    order = np.argsort(values, axis=1)
    best = top2[np.arange(len(query)), order[:, 1]]
    score = cosine[np.arange(len(query)), best]
    second = values[np.arange(len(query)), order[:, 0]]
    reverse = np.argmax(cosine, axis=0)
    keep = (
        (reverse[best] == np.arange(len(query)))
        & (score >= 0.40)
        & ((score - second) >= 0.02)
    )
    query_ids = np.flatnonzero(keep)
    source_ids = best[keep]
    if len(query_ids) < 6:
        return 0, 0.0
    query_xy = np.c_[query_ids % 32, query_ids // 32].astype(np.float64)
    source_xy = np.c_[source_ids % 32, source_ids // 32].astype(np.float64)
    cv2.setRNGSeed(260901)
    _, mask = cv2.findHomography(
        query_xy, source_xy, cv2.RANSAC, 1.5, maxIters=2000, confidence=0.995,
    )
    if mask is None:
        return 0, 0.0
    inlier = mask.reshape(-1).astype(bool)
    return int(np.sum(inlier)), float(np.mean(score[query_ids[inlier]]))


def _nearby_rows(
    pose: np.ndarray,
    centers: np.ndarray,
    forwards: np.ndarray,
    *,
    top_views: int,
) -> np.ndarray:
    center, forward = _pose_center_forward(pose)
    distance = np.linalg.norm(centers - center, axis=1)
    direction = forwards @ forward
    eligible = (distance <= 10.0) & (direction >= np.cos(np.deg2rad(45.0)))
    rows = np.flatnonzero(eligible)
    return rows[np.argsort(distance[rows], kind="stable")[:top_views]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, required=True)
    parser.add_argument("--global_view_field", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--top_views", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite spatial RADIO selection")
    top5, meta5 = _load_frozen_poses(args.top5_pose_inventory)
    top10, meta10 = _load_frozen_poses(args.top10_pose_inventory)
    if not np.array_equal(top5["names"], top10["names"]):
        raise ValueError("paired candidate names differ")
    field, field_meta = _load_global_view_field(args.global_view_field)
    view_names = field["names"].astype(str)
    centers = np.asarray(field["camera_centers_world"], np.float64)
    forwards = np.asarray(field["camera_forwards_world"], np.float64)
    records = _records(args.radio_manifest)
    projection = _projection()
    cache: dict[str, np.ndarray] = {}

    def compact(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = _compact(_radio(name, records), projection)
        return cache[name]

    rows = []
    for index, name in enumerate(top5["names"].astype(str).tolist()):
        query = compact(name)
        payload: dict[str, object] = {"name": name}
        branch_scores = {}
        for branch, candidate in ((5, top5), (10, top10)):
            nearby = (
                _nearby_rows(
                    candidate["pose_w2c"][index], centers, forwards,
                    top_views=int(args.top_views),
                )
                if bool(candidate["usable"][index]) else np.zeros(0, np.int64)
            )
            scores = [
                (*_spatial_match_score(query, compact(str(view_names[row]))), int(row))
                for row in nearby.tolist()
            ]
            best = max(scores, default=(0, 0.0, -1), key=lambda value: (value[0], value[1]))
            branch_scores[branch] = (int(best[0]), float(best[1]))
            payload[f"top{branch}_maximum_homography_inliers"] = int(best[0])
            payload[f"top{branch}_tie_mean_cosine"] = float(best[1])
            payload[f"top{branch}_nearby_mapping_view_count"] = int(len(nearby))
            payload[f"top{branch}_best_mapping_view"] = (
                None if best[2] < 0 else str(view_names[best[2]])
            )
        payload["selected_branch"] = (
            10 if branch_scores[10] > branch_scores[5] else 5
        )
        rows.append(payload)
        if (index + 1) % 10 == 0:
            print(json.dumps({"completed": index + 1, "total": len(top5["names"])}))
    report = {
        "artifact_type": "goal_maplet_pnp_pose_conditioned_spatial_radio_selection_v1",
        "query_count": int(len(rows)),
        "query_pose_or_ground_truth_read": False,
        "selection_rule": "maximum_full_image_pooled_RADIO_mutual_homography_inliers_tie_cosine_then_top5",
        "spatial_pool": "2x2 RADIO tokens to 18x32",
        "projection": "fixed_seed_1280_to_128_Rademacher_JL",
        "projection_seed": PROJECTION_SEED,
        "projected_channels": PROJECTED_CHANNELS,
        "top_views": int(args.top_views),
        "nearby_view_gate": "distance<=10m_and_forward_angle<=45deg_then_closest",
        "top5_pose_inventory_file_sha256": file_sha256(args.top5_pose_inventory),
        "top5_pose_inventory_content_sha256": meta5.get("content_sha256"),
        "top10_pose_inventory_file_sha256": file_sha256(args.top10_pose_inventory),
        "top10_pose_inventory_content_sha256": meta10.get("content_sha256"),
        "global_view_field_file_sha256": file_sha256(args.global_view_field),
        "global_view_field_content_sha256": field_meta.get("content_sha256"),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "cached_image_count": int(len(cache)),
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
