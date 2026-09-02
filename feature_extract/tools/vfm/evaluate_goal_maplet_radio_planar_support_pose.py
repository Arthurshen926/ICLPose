"""Evaluate deployable RADIO plane retrieval followed by planar token Sim(3).

The map is the MoGe3/reference-only aligned chart control.  Each chart is
segmented into finite planar regions on its sealed stride-2 lattice and each
region pools only source-view RADIO tokens.  A query uses only pose-free MoGe3
planes and query RADIO tokens.  Plane retrieval and token matching are frozen
before the historical held pose is opened for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_chart_plane_pose_oracle import (
    MINIMUM_POINT_CORRESPONDENCES,
    _robust_point_pose,
    _rotation_error_deg,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_moge_reference_held_render_control import _normals
from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import _load_aligned_vertices
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import StrictHeldRayInventory
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import ProjectiveExactFaceSeamAuthority
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions, extract_query_plane_regions


SCHEMA = "goal_maplet_radio_planar_region_token_sim3_historical_control_v4"
MINIMUM_REGION_TOKENS = 3
MINIMUM_QUERY_TOKEN_PLANE_PIXELS = 8
MINIMUM_PLANE_COSINE = 0.40
MINIMUM_PLANE_MARGIN = 0.01
MINIMUM_TOKEN_COSINE = 0.40
MINIMUM_TOKEN_MARGIN = 0.02
MAXIMUM_PLANE_MATCHES = 16
MAXIMUM_MAP_PLANE_CANDIDATES_PER_QUERY = 16
MINIMUM_HOMOGRAPHY_MATCHES = 6
MINIMUM_HOMOGRAPHY_INLIER_RATIO = 0.50
HOMOGRAPHY_RANSAC_THRESHOLD_TOKEN_CELLS = 2.0


def _normalise(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _radio_records(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text())
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("RADIO manifest lacks records")
    records = {str(row["image_id"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError("RADIO manifest contains duplicate image ids")
    return records


def _radio_tokens(
    name: str,
    records: dict[str, dict[str, object]],
) -> tuple[np.ndarray, str]:
    image_id = name.replace("__", "/", 1)
    if image_id not in records:
        raise ValueError(f"RADIO manifest lacks {image_id}")
    row = records[image_id]
    path = Path(str(row["token_path"]))
    if file_sha256(path) != row.get("checksum"):
        raise ValueError("RADIO token archive differs from manifest")
    with np.load(path, allow_pickle=False) as data:
        feature = np.asarray(data["radio_final"], np.float32)
    if feature.shape != (1280, 36, 64):
        raise ValueError("RADIO-final feature grid differs")
    return _normalise(feature.reshape(1280, -1).T), str(row["checksum"])


def _map_regions(
    authority: ProjectiveExactFaceSeamAuthority,
    vertices: np.ndarray,
    records: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    normals = _normals(vertices, authority.faces)
    face_referenced = np.zeros((len(vertices),), bool)
    face_referenced[np.unique(authority.faces)] = True
    regions: list[dict[str, object]] = []
    token_hash: dict[str, str] = {}
    for chart, name in enumerate(authority.chart_names.astype(str)):
        feature, checksum = _radio_tokens(name, records)
        token_hash[name] = checksum
        lo, hi = map(int, authority.chart_vertex_offsets[chart:chart + 2])
        pixel = authority.sampled_vertex_pixel_indices[lo:hi]
        if np.any(pixel % 2) or np.any((pixel // 256) % 2):
            raise ValueError("RADIO plane map requires sealed stride-2 vertices")
        yy = (pixel // 256) // 2
        xx = (pixel % 256) // 2
        grid_point = np.full((72, 128, 3), np.nan, np.float64)
        grid_normal = np.zeros((72, 128, 3), np.float64)
        grid_vertex = np.full((72, 128), -1, np.int64)
        keep = face_referenced[lo:hi]
        grid_point[yy[keep], xx[keep]] = vertices[lo:hi][keep]
        grid_normal[yy[keep], xx[keep]] = normals[lo:hi][keep]
        grid_vertex[yy[keep], xx[keep]] = np.arange(lo, hi)[keep]
        valid = np.isfinite(grid_point).all(2) & (np.linalg.norm(grid_normal, axis=2) > 0.5)
        segmented = extract_query_plane_regions(
            grid_point,
            grid_normal,
            valid,
            minimum_pixels=20,
            maximum_planes=32,
            maximum_hypotheses=96,
        )
        for local in range(len(segmented.normals_camera)):
            mask = segmented.labels == local
            gy, gx = np.nonzero(mask)
            token_id = (gy // 2) * 64 + (gx // 2)
            unique_token = np.unique(token_id)
            token_points, kept_token = [], []
            for token in unique_token.tolist():
                member = mask & (((np.arange(72)[:, None] // 2) * 64 + (np.arange(128)[None, :] // 2)) == token)
                vertex_index = grid_vertex[member]
                vertex_index = vertex_index[vertex_index >= 0]
                if len(vertex_index):
                    kept_token.append(token)
                    token_points.append(np.mean(vertices[vertex_index], axis=0))
            if len(kept_token) < MINIMUM_REGION_TOKENS:
                continue
            kept_token_array = np.asarray(kept_token, np.int64)
            token_feature = feature[kept_token_array]
            descriptor = _normalise(np.mean(token_feature, axis=0, keepdims=True))[0]
            regions.append({
                "region_id": len(regions),
                "chart": chart,
                "chart_name": name,
                "local_region": local,
                "normal_world": segmented.normals_camera[local],
                "offset_world": float(segmented.offsets_camera[local]),
                "descriptor": descriptor,
                "token_ids": kept_token_array,
                "token_features": token_feature,
                "token_points_world": np.asarray(token_points, np.float64),
            })
    if len(regions) < 4:
        raise ValueError("RADIO map contains fewer than four finite plane regions")
    return regions, token_hash


def _query_regions(
    planes: QueryPlaneRegions,
    points: np.ndarray,
    feature: np.ndarray,
) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    token_grid = np.arange(36 * 64, dtype=np.int64).reshape(36, 64)
    for local in range(len(planes.normals_camera)):
        labels = planes.labels == local
        token_ids, token_points = [], []
        for ty in range(36):
            for tx in range(64):
                block = labels[ty * 4:(ty + 1) * 4, tx * 4:(tx + 1) * 4]
                if int(block.sum()) < MINIMUM_QUERY_TOKEN_PLANE_PIXELS:
                    continue
                sample = points[ty * 4:(ty + 1) * 4, tx * 4:(tx + 1) * 4][block]
                sample = sample[np.isfinite(sample).all(1)]
                if len(sample) < 4:
                    continue
                token_ids.append(int(token_grid[ty, tx]))
                token_points.append(np.median(sample, axis=0))
        if len(token_ids) < MINIMUM_REGION_TOKENS:
            continue
        token_id = np.asarray(token_ids, np.int64)
        token_feature = feature[token_id]
        regions.append({
            "local_region": local,
            "normal_camera": planes.normals_camera[local],
            "offset_camera": float(planes.offsets_camera[local]),
            "descriptor": _normalise(np.mean(token_feature, axis=0, keepdims=True))[0],
            "token_ids": token_id,
            "token_features": token_feature,
            "token_points_camera": np.asarray(token_points, np.float64),
        })
    return regions


def _match_plane_regions(
    query: list[dict[str, object]],
    mapping: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not query:
        return []
    q = np.asarray([row["descriptor"] for row in query], np.float32)
    m = np.asarray([row["descriptor"] for row in mapping], np.float32)
    cosine = q @ m.T
    candidate_count = min(MAXIMUM_MAP_PLANE_CANDIDATES_PER_QUERY + 1, len(mapping))
    top = np.argpartition(cosine, -candidate_count, axis=1)[:, -candidate_count:]
    selected = []
    for query_row in range(len(query)):
        ordered = top[query_row, np.argsort(cosine[query_row, top[query_row]])[::-1]]
        for rank, map_row in enumerate(ordered[:MAXIMUM_MAP_PLANE_CANDIDATES_PER_QUERY]):
            score = float(cosine[query_row, map_row])
            if score < MINIMUM_PLANE_COSINE:
                continue
            selected.append({
                "query_region": query_row,
                "map_region": int(map_row),
                "candidate_rank": rank + 1,
                "plane_cosine": score,
                "plane_margin_to_next": score - float(cosine[query_row, ordered[rank + 1]]),
            })
    selected = sorted(
        selected,
        key=lambda row: (row["candidate_rank"], -row["plane_cosine"], row["query_region"]),
    )[:MAXIMUM_PLANE_MATCHES * MAXIMUM_MAP_PLANE_CANDIDATES_PER_QUERY]
    return selected


def _token_matches(
    query: list[dict[str, object]],
    mapping: list[dict[str, object]],
    plane_matches: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    source, target, rows = [], [], []
    for plane in plane_matches:
        q = query[plane["query_region"]]
        m = mapping[plane["map_region"]]
        cosine = q["token_features"] @ m["token_features"].T
        if cosine.shape[1] < 2:
            continue
        top2 = np.argpartition(cosine, -2, axis=1)[:, -2:]
        value = np.take_along_axis(cosine, top2, axis=1)
        order = np.argsort(value, axis=1)
        best = top2[np.arange(len(cosine)), order[:, 1]]
        score = cosine[np.arange(len(cosine)), best]
        second = value[np.arange(len(cosine)), order[:, 0]]
        reverse = np.argmax(cosine, axis=0)
        keep = (
            (reverse[best] == np.arange(len(cosine)))
            & (score >= MINIMUM_TOKEN_COSINE)
            & ((score - second) >= MINIMUM_TOKEN_MARGIN)
        )
        query_index = np.flatnonzero(keep)
        map_index = best[keep]
        homography_inlier = np.zeros((len(query_index),), bool)
        if len(query_index) >= MINIMUM_HOMOGRAPHY_MATCHES:
            query_token = q["token_ids"][query_index]
            map_token = m["token_ids"][map_index]
            query_xy = np.stack((query_token % 64, query_token // 64), axis=1).astype(np.float64)
            map_xy = np.stack((map_token % 64, map_token // 64), axis=1).astype(np.float64)
            cv2.setRNGSeed(260830)
            _, mask = cv2.findHomography(
                query_xy,
                map_xy,
                method=cv2.RANSAC,
                ransacReprojThreshold=HOMOGRAPHY_RANSAC_THRESHOLD_TOKEN_CELLS,
                maxIters=2000,
                confidence=0.995,
            )
            if mask is not None:
                homography_inlier = mask.reshape(-1).astype(bool)
        accepted = (
            int(homography_inlier.sum()) >= MINIMUM_HOMOGRAPHY_MATCHES
            and float(np.mean(homography_inlier)) >= MINIMUM_HOMOGRAPHY_INLIER_RATIO
        )
        if accepted:
            query_index = query_index[homography_inlier]
            map_index = map_index[homography_inlier]
        else:
            query_index = np.empty((0,), np.int64)
            map_index = np.empty((0,), np.int64)
        source.extend(q["token_points_camera"][query_index])
        target.extend(m["token_points_world"][map_index])
        rows.append({
            **plane,
            "map_chart_name": m["chart_name"],
            "query_token_count": int(len(q["token_ids"])),
            "map_token_count": int(len(m["token_ids"])),
            "mutual_token_match_count": int(len(query_index)),
            "pre_homography_mutual_token_match_count": int(keep.sum()),
            "homography_inlier_count": int(homography_inlier.sum()),
            "homography_inlier_ratio": float(np.mean(homography_inlier)) if len(homography_inlier) else 0.0,
            "homography_accepted": bool(accepted),
            "token_cosine_median": float(np.median(score[keep])) if keep.any() else None,
        })
    return (
        np.asarray(source, np.float64).reshape(-1, 3),
        np.asarray(target, np.float64).reshape(-1, 3),
        rows,
    )


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    usable = [row for row in rows if row["pose"]["usable"]]
    if not usable:
        return {"query_count": len(rows), "usable_count": 0, "recall_2m45": 0.0, "recall_1m10": 0.0}
    translation = np.asarray([row["pose"]["translation_error_m"] for row in usable])
    rotation = np.asarray([row["pose"]["rotation_error_deg"] for row in usable])
    scale = np.asarray([row["pose"]["estimated_scale"] for row in usable])
    return {
        "query_count": len(rows),
        "usable_count": len(usable),
        "translation_error_median_m": float(np.median(translation)),
        "translation_error_p90_m": float(np.quantile(translation, 0.9)),
        "rotation_error_median_deg": float(np.median(rotation)),
        "rotation_error_p90_deg": float(np.quantile(rotation, 0.9)),
        "estimated_scale_median": float(np.median(scale)),
        "full_query_recall_2m45": float(np.sum((translation <= 2) & (rotation <= 45)) / len(rows)),
        "full_query_recall_1m10": float(np.sum((translation <= 1) & (rotation <= 10)) / len(rows)),
    }


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
        raise FileExistsError("refusing to overwrite RADIO planar pose control")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("RADIO planar authority differs from pin")
    vertices = _load_aligned_vertices(args.alignment, authority)
    records = _radio_records(args.radio_manifest)
    mapping, map_token_hash = _map_regions(authority, vertices, records)
    query_manifest_path = args.moge3_query / "manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    query_names = [str(row["name"]) for row in query_manifest.get("rows", [])]
    if len(query_names) != len(set(query_names)) or not query_names:
        raise ValueError("MoGe3 query manifest names are empty or duplicated")
    frozen_rows = []
    for name in query_names:
        query_feature, query_hash = _radio_tokens(name, records)
        path = args.moge3_query / f"{name}.npz"
        with np.load(path, allow_pickle=False) as data:
            points = np.asarray(data["points_camera"], np.float64)
            normal = np.asarray(data["normal_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
        planes = extract_query_plane_regions(points, normal, valid)
        query = _query_regions(planes, points, query_feature)
        plane_match = _match_plane_regions(query, mapping)
        source, target, match_rows = _token_matches(query, mapping, plane_match)
        pose: dict[str, object] = {"usable": False}
        if len(source) >= MINIMUM_POINT_CORRESPONDENCES:
            rotation, translation, scale, inlier, residual = _robust_point_pose(
                source, target, estimate_scale=True,
            )
            pose = {
                "usable": True,
                "rotation_c2w": rotation.tolist(),
                "translation_world": translation.tolist(),
                "estimated_scale": float(scale),
                "inlier_count": int(inlier.sum()),
                "residual_median_m": float(np.median(residual[inlier])) if inlier.any() else None,
                "residual_p90_m": float(np.quantile(residual[inlier], 0.9)) if inlier.any() else None,
            }
        frozen_rows.append({
            "name": name,
            "query_radio_file_sha256": query_hash,
            "query_plane_count": len(query),
            "matched_plane_count": len(match_rows),
            "token_correspondence_count": int(len(source)),
            "plane_matches": match_rows,
            "pose": pose,
        })
    # The held authority, including camera poses, is opened only after every
    # correspondence set and pose hypothesis above has been frozen.
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    held_row = {name: row for row, name in enumerate(held.view_names.astype(str))}
    if set(held_row) != set(query_names):
        raise ValueError("held pose inventory differs from frozen query inventory")
    rows = []
    for frozen in frozen_rows:
        row = dict(frozen)
        pose = dict(row["pose"])
        if pose["usable"]:
            view = held_row[row["name"]]
            rotation = np.asarray(pose.pop("rotation_c2w"), np.float64)
            translation = np.asarray(pose.pop("translation_world"), np.float64)
            pose["rotation_error_deg"] = _rotation_error_deg(
                rotation, held.camera_to_world[view],
            )
            pose["translation_error_m"] = float(np.linalg.norm(
                translation - held.camera_to_world[view, :3, 3]
            ))
        row["pose"] = pose
        rows.append(row)
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
        "matching_contract": {
            "plane_descriptor": "l2_normalized_mean_RADIO_final_tokens_inside_finite_plane",
            "plane_cosine_minimum": MINIMUM_PLANE_COSINE,
            "plane_margin_minimum_top1_control_only": MINIMUM_PLANE_MARGIN,
            "map_plane_candidates_per_query": MAXIMUM_MAP_PLANE_CANDIDATES_PER_QUERY,
            "plane_match_rule": "top16_mean_RADIO_then_within_plane_token_homography_RANSAC",
            "homography_minimum_matches": MINIMUM_HOMOGRAPHY_MATCHES,
            "homography_minimum_inlier_ratio": MINIMUM_HOMOGRAPHY_INLIER_RATIO,
            "homography_ransac_threshold_token_cells": HOMOGRAPHY_RANSAC_THRESHOLD_TOKEN_CELLS,
            "token_match": "within_retrieved_plane_mutual_nearest_RADIO_final",
            "token_cosine_minimum": MINIMUM_TOKEN_COSINE,
            "token_margin_minimum": MINIMUM_TOKEN_MARGIN,
            "target_pose_consumed_before_pose_freeze": False,
        },
        "solver_contract": "robust_planar_support_token_Sim3",
        "held_pose_opened_only_after_correspondences_and_pose_frozen": True,
        "phase_separation": {
            "phase1_inputs": "map_alignment_map_RADIO_query_MoGe3_query_RADIO_only",
            "phase1_output": "all_plane_and_token_correspondences_and_Sim3_hypotheses_frozen_in_memory",
            "phase2_first_action": "open_StrictHeldRayInventory_and_join_by_exact_name",
            "held_pose_available_to_phase1": False,
        },
        "held_inventory_previously_opened": True,
        "blind_or_preregistered_claim": False,
        "system_control_only": True,
        "production_eligible": False,
        "promotion_eligible": False,
        "summary": _summary(rows),
        "rows": rows,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"],
        "summary": payload["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
