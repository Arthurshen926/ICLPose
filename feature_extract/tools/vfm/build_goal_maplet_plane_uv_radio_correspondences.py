"""Match query RADIO tokens directly to a canonical metric plane UV atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _camera_inventory,
    _mutual_matches,
    _radio,
    _records,
    _region_tokens,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def _metric_homography_filter(
    query_tokens: np.ndarray,
    plane_uv_m: np.ndarray,
    *,
    threshold_m: float,
) -> np.ndarray:
    if len(query_tokens) < 4:
        return np.zeros(len(query_tokens), bool)
    query_xy = np.c_[query_tokens % 64, query_tokens // 64].astype(np.float64)
    target_uv = np.asarray(plane_uv_m, np.float64).reshape(-1, 2)
    cv2.setRNGSeed(260901)
    _, mask = cv2.findHomography(
        query_xy,
        target_uv,
        cv2.RANSAC,
        float(threshold_m),
        maxIters=2000,
        confidence=0.995,
    )
    return np.zeros(len(query_tokens), bool) if mask is None else mask.reshape(-1).astype(bool)


def _top_distinct_hypotheses(
    tokens: np.ndarray,
    scores: np.ndarray,
    plane_rows: np.ndarray,
    texel_rows: np.ndarray,
    *,
    maximum_per_token: int,
) -> np.ndarray:
    chosen: list[int] = []
    for token in np.unique(tokens):
        candidates = np.flatnonzero(tokens == token)
        order = candidates[np.lexsort((texel_rows[candidates], plane_rows[candidates], -scores[candidates]))]
        seen: set[tuple[int, int]] = set()
        for row in order.tolist():
            key = (int(plane_rows[row]), int(texel_rows[row]))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(row)
            if len(seen) == int(maximum_per_token):
                break
    return np.asarray(chosen, np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_uv_atlas", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--plane_ranking", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--query_camera_inventory", type=Path, required=True)
    parser.add_argument("--topk_planes", type=int, default=10)
    parser.add_argument("--hypotheses_per_query_token", type=int, default=3)
    parser.add_argument("--homography_threshold_m", type=float, default=1.0)
    parser.add_argument("--output_correspondences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_correspondences.exists():
        raise FileExistsError("refusing to overwrite plane UV correspondence artifact")
    if int(args.hypotheses_per_query_token) < 1 or float(args.homography_threshold_m) <= 0:
        raise ValueError("invalid plane UV correspondence configuration")

    with np.load(args.plane_uv_atlas, allow_pickle=False) as data:
        atlas_meta = json.loads(str(data["metadata_json"].item()))
        atlas = {name: np.asarray(data[name]) for name in (
            "plane_texel_offsets", "texel_uv_m", "world_points", "radio_features",
            "view_support", "token_support", "texel_identity", "prototype_rank",
        )}
    if (
        atlas_meta.get("artifact_type") != "goal_maplet_metric_plane_uv_radio_atlas_v2"
        or atlas_meta.get("uses_query_pose_depth_or_ground_truth") is not False
        or arrays_sha256(atlas) != atlas_meta.get("arrays_sha256")
    ):
        raise ValueError("metric plane UV atlas contract differs")
    ranking = json.loads(args.plane_ranking.read_text())
    if (
        ranking.get("artifact_type")
        not in (
            "goal_maplet_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v1",
        )
        or ranking.get("uses_pose_or_ground_truth") is not False
        or ranking.get("contains_postlabel_fields") is not False
    ):
        raise ValueError("plane ranking is not pose/label-free")
    radio_records = _records(args.radio_manifest)
    cameras, camera_meta = _camera_inventory(args.query_camera_inventory)

    names, point_rows, token_rows, provenance_rows, matrices, radial = [], [], [], [], [], []
    diagnostic_rows = []
    for query in ranking["rows"]:
        name = str(query["image"])
        if name not in cameras:
            raise ValueError("query camera inventory lacks query")
        model_id, width, height, params = cameras[name]
        # Keep this import local to make the phase boundary explicit: camera
        # inventory contains intrinsics only and has already excluded pose.
        from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics

        K, k1 = _scaled_intrinsics(model_id, params, width, height)
        planes, _ = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        query_feature = _radio(name, radio_records)
        all_points, all_tokens, all_scores, all_planes, all_texels, all_regions = [], [], [], [], [], []
        for region_row in query["regions"]:
            region = int(region_row["region"])
            qtoken = _region_tokens(planes.labels, region)
            if len(qtoken) < 4:
                continue
            qfeature = query_feature[qtoken]
            for rank, plane in enumerate(region_row["top10"][: int(args.topk_planes)]):
                plane = int(plane)
                lo, hi = map(int, atlas["plane_texel_offsets"][plane : plane + 2])
                if hi - lo < 4:
                    continue
                qi, ti, score = _mutual_matches(qfeature, atlas["radio_features"][lo:hi].astype(np.float32))
                if len(qi) < 4:
                    continue
                keep = _metric_homography_filter(
                    qtoken[qi], atlas["texel_uv_m"][lo + ti],
                    threshold_m=float(args.homography_threshold_m),
                )
                selected = np.flatnonzero(keep)
                if not len(selected):
                    continue
                prototype = lo + ti[selected]
                all_points.append(atlas["world_points"][prototype])
                all_tokens.append(qtoken[qi[selected]])
                all_scores.append(score[selected] - 0.02 * rank)
                all_planes.append(np.full(len(selected), plane, np.int64))
                all_texels.append(atlas["texel_identity"][prototype].astype(np.int64))
                all_regions.append(np.full(len(selected), region, np.int64))
        if all_points:
            world = np.concatenate(all_points)
            token = np.concatenate(all_tokens)
            score = np.concatenate(all_scores)
            plane_row = np.concatenate(all_planes)
            texel_row = np.concatenate(all_texels)
            region_row = np.concatenate(all_regions)
            chosen = _top_distinct_hypotheses(
                token, score, plane_row, texel_row,
                maximum_per_token=int(args.hypotheses_per_query_token),
            )
            world, token = world[chosen], token[chosen]
            provenance = np.c_[region_row[chosen], plane_row[chosen], texel_row[chosen]]
        else:
            world = np.zeros((0, 3), np.float64)
            token = np.zeros(0, np.int64)
            provenance = np.zeros((0, 3), np.int64)
        names.append(name); point_rows.append(world); token_rows.append(token)
        provenance_rows.append(provenance); matrices.append(K); radial.append(k1)
        diagnostic_rows.append({
            "name": name,
            "correspondence_count": int(len(token)),
            "unique_query_token_count": int(len(np.unique(token))),
            "physical_plane_count": int(len(np.unique(provenance[:, 1]))) if len(provenance) else 0,
            "metric_texel_count": int(len(np.unique(provenance[:, 2]))) if len(provenance) else 0,
        })

    offsets = np.r_[0, np.cumsum([len(row) for row in token_rows])].astype(np.int64)
    arrays = {
        "names": np.asarray(names),
        "correspondence_offsets": offsets,
        "world_points": np.concatenate(point_rows).astype(np.float64),
        "query_tokens": np.concatenate(token_rows).astype(np.int64),
        "provenance_region_plane_atlas_row": np.concatenate(provenance_rows).astype(np.int64),
        "camera_matrices": np.asarray(matrices, np.float64),
        "radial_k1": np.asarray(radial, np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "correspondence_count": int(offsets[-1]),
        "pose_or_ground_truth_opened": False,
        "query_depth_or_scale_used": False,
        "correspondence_semantics": "query_RADIO_to_view_independent_metric_plane_UV_texels",
        "hypotheses_per_query_token": int(args.hypotheses_per_query_token),
        "topk_planes": int(args.topk_planes),
        "homography_threshold_m": float(args.homography_threshold_m),
        "plane_uv_atlas_file_sha256": file_sha256(args.plane_uv_atlas),
        "plane_uv_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "plane_ranking_file_sha256": file_sha256(args.plane_ranking),
        "query_camera_only_inventory_file_sha256": file_sha256(args.query_camera_inventory),
        "query_camera_only_inventory_content_sha256": camera_meta.get("content_sha256"),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_correspondences.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_correspondences, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        "artifact_type": "goal_maplet_metric_plane_uv_radio_correspondence_build_v1",
        "frozen_correspondence_file_sha256": file_sha256(args.output_correspondences),
        "frozen_correspondence_content_sha256": metadata["content_sha256"],
        "query_count": int(len(names)),
        "median_correspondence_count": float(np.median([row["correspondence_count"] for row in diagnostic_rows])),
        "median_unique_query_token_count": float(np.median([row["unique_query_token_count"] for row in diagnostic_rows])),
        "pose_or_ground_truth_opened": False,
        "rows": diagnostic_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
