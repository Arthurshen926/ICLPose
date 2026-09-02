"""Evaluate large-plane RADIO retrieval, explicit no-match, and Sim(3) pose.

All map families and the no-match operating point are constructed from source
charts only.  Query MoGe3 and RADIO then freeze plane decisions, finite token
correspondences, and Sim(3) poses before the historical held geometry/pose is
opened for diagnosis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.evaluate_goal_maplet_chart_plane_pose_oracle import (
    MINIMUM_POINT_CORRESPONDENCES,
    _robust_point_pose,
    _rotation_error_deg,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_moge_reference_held_render_control import _normals
from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import _load_aligned_vertices
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_planar_support_pose import (
    HOMOGRAPHY_RANSAC_THRESHOLD_TOKEN_CELLS,
    MINIMUM_HOMOGRAPHY_INLIER_RATIO,
    MINIMUM_HOMOGRAPHY_MATCHES,
    MINIMUM_QUERY_TOKEN_PLANE_PIXELS,
    MINIMUM_REGION_TOKENS,
    MINIMUM_TOKEN_COSINE,
    MINIMUM_TOKEN_MARGIN,
    _normalise,
    _radio_records,
    _radio_tokens,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import StrictHeldRayInventory
from feature_extract.vfm.localization_goal_maplet.large_plane_regions import LargePlaneRegions, extract_large_plane_regions
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import ProjectiveExactFaceSeamAuthority


SCHEMA = "goal_maplet_large_plane_radio_sim3_historical_control_v1"
FAMILY_NORMAL_DEGREES = 10.0
FAMILY_OFFSET_M = 0.20
FAMILY_SUPPORT_GAP_M = 0.50
FAMILY_REFIT_P95_M = 0.20
MAXIMUM_QUERY_PLANES = 16
MAXIMUM_FAMILY_CANDIDATES = 4
SOURCE_MAXIMUM_UNMATCHED_FALSE_ACCEPT = 0.10
PHYSICAL_NORMAL_DEGREES = 30.0
PHYSICAL_MEDIAN_M = 0.75
PHYSICAL_P90_M = 1.50


def _fit_plane(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    center = np.mean(points, axis=0)
    covariance = (points - center).T @ (points - center) / max(len(points), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if float(normal @ reference) < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    offset = float(normal @ center)
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0.0:
        normal, offset = -normal, -offset
    return normal, offset, np.abs(points @ normal - offset)


def _regions_from_grid(
    planes: LargePlaneRegions,
    points: np.ndarray,
    feature: np.ndarray,
    *,
    query: bool,
    chart: int | None = None,
    chart_name: str | None = None,
) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    block = 4 if query else 2
    token_height, token_width = 36, 64
    for local in range(len(planes.normals)):
        mask = planes.labels == local
        token_ids, token_points = [], []
        for ty in range(token_height):
            for tx in range(token_width):
                sample_mask = mask[ty * block:(ty + 1) * block, tx * block:(tx + 1) * block]
                required = MINIMUM_QUERY_TOKEN_PLANE_PIXELS if query else 1
                if int(sample_mask.sum()) < required:
                    continue
                sample = points[ty * block:(ty + 1) * block, tx * block:(tx + 1) * block][sample_mask]
                sample = sample[np.isfinite(sample).all(1)]
                if len(sample) < (4 if query else 1):
                    continue
                token_ids.append(ty * token_width + tx)
                token_points.append(np.median(sample, axis=0))
        if len(token_ids) < MINIMUM_REGION_TOKENS:
            continue
        token_id = np.asarray(token_ids, np.int64)
        token_feature = feature[token_id]
        row: dict[str, object] = {
            "local_region": local,
            "normal": planes.normals[local],
            "offset": float(planes.offsets[local]),
            "pixel_count": int(planes.pixel_counts[local]),
            "component_count": int(planes.component_counts[local]),
            "descriptor": _normalise(np.mean(token_feature, axis=0, keepdims=True))[0],
            "token_ids": token_id,
            "token_features": token_feature,
            "token_points": np.asarray(token_points, np.float64),
            # In-memory-only finite support used by downstream local matchers.
            # It is never serialized in the JSON report.
            "grid_mask": mask,
            "grid_points": points,
        }
        if not query:
            row.update(chart=int(chart), chart_name=str(chart_name))
        regions.append(row)
    return regions


def _map_local_regions(
    authority: ProjectiveExactFaceSeamAuthority,
    vertices: np.ndarray,
    records: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    normals = _normals(vertices, authority.faces)
    referenced = np.zeros((len(vertices),), bool)
    referenced[np.unique(authority.faces)] = True
    output: list[dict[str, object]] = []
    token_hash: dict[str, str] = {}
    for chart, name in enumerate(authority.chart_names.astype(str)):
        feature, checksum = _radio_tokens(name, records)
        token_hash[name] = checksum
        lo, hi = map(int, authority.chart_vertex_offsets[chart:chart + 2])
        pixel = authority.sampled_vertex_pixel_indices[lo:hi]
        if np.any(pixel % 2) or np.any((pixel // 256) % 2):
            raise ValueError("large-plane map requires sealed stride-2 vertices")
        yy, xx = (pixel // 256) // 2, (pixel % 256) // 2
        grid_points = np.full((72, 128, 3), np.nan, np.float64)
        grid_normals = np.zeros((72, 128, 3), np.float64)
        keep = referenced[lo:hi]
        grid_points[yy[keep], xx[keep]] = vertices[lo:hi][keep]
        grid_normals[yy[keep], xx[keep]] = normals[lo:hi][keep]
        valid = np.isfinite(grid_points).all(2) & (np.linalg.norm(grid_normals, axis=2) > 0.5)
        planes = extract_large_plane_regions(
            grid_points, grid_normals, valid,
            minimum_pixels=20, maximum_planes=16, maximum_hypotheses=128,
        )
        output.extend(_regions_from_grid(
            planes, grid_points, feature, query=False, chart=chart, chart_name=name,
        ))
    for row, region in enumerate(output):
        region["region_id"] = row
    return output, token_hash


def _family_compatible(region: dict[str, object], members: list[dict[str, object]]) -> tuple[bool, float]:
    if any(int(row["chart"]) == int(region["chart"]) for row in members):
        return False, float("inf")
    points = np.concatenate([np.asarray(row["token_points"]) for row in members], axis=0)
    reference = np.mean([np.asarray(row["normal"]) for row in members], axis=0)
    normal, offset, _ = _fit_plane(points, reference)
    candidate_normal = np.asarray(region["normal"])
    sign = 1.0 if float(candidate_normal @ normal) >= 0.0 else -1.0
    if abs(float(candidate_normal @ normal)) < np.cos(np.deg2rad(FAMILY_NORMAL_DEGREES)):
        return False, float("inf")
    if abs(float(region["offset"]) - sign * offset) > FAMILY_OFFSET_M:
        return False, float("inf")
    distance = cKDTree(points).query(np.asarray(region["token_points"]), k=1)[0]
    gap = float(np.min(distance))
    if gap > FAMILY_SUPPORT_GAP_M:
        return False, gap
    combined = np.concatenate((points, np.asarray(region["token_points"])), axis=0)
    _, _, residual = _fit_plane(combined, reference + sign * candidate_normal)
    return bool(np.quantile(residual, 0.95) <= FAMILY_REFIT_P95_M), gap


def build_plane_families(regions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Greedily fuse overlapping cross-view regions with full-family refit."""
    ordered = sorted(
        regions,
        key=lambda row: (-int(row["pixel_count"]), int(row["chart"]), int(row["local_region"])),
    )
    members: list[list[dict[str, object]]] = []
    for region in ordered:
        candidate = []
        for family, rows in enumerate(members):
            compatible, gap = _family_compatible(region, rows)
            if compatible:
                candidate.append((gap, family))
        if candidate:
            members[min(candidate)[1]].append(region)
        else:
            members.append([region])
    families: list[dict[str, object]] = []
    for family, rows in enumerate(members):
        points = np.concatenate([np.asarray(row["token_points"]) for row in rows])
        reference = np.mean([np.asarray(row["normal"]) for row in rows], axis=0)
        normal, offset, residual = _fit_plane(points, reference)
        for row in rows:
            row["family_id"] = family
        families.append({
            "family_id": family,
            "member_regions": rows,
            "member_region_ids": [int(row["region_id"]) for row in rows],
            "chart_count": len({int(row["chart"]) for row in rows}),
            "normal_world": normal,
            "offset_world": offset,
            "points_world": points,
            "residual_p95_m": float(np.quantile(residual, 0.95)),
        })
    return families


def _family_scores(
    descriptor: np.ndarray,
    families: list[dict[str, object]],
    *,
    exclude_chart: int | None = None,
) -> np.ndarray:
    score = np.full((len(families),), -1.0, np.float64)
    for family, row in enumerate(families):
        member = [
            np.asarray(item["descriptor"])
            for item in row["member_regions"]
            if exclude_chart is None or int(item["chart"]) != exclude_chart
        ]
        if not member:
            continue
        similarity = np.asarray(member) @ descriptor
        take = min(2, len(similarity))
        score[family] = float(np.mean(np.sort(similarity)[-take:]))
    return score


def calibrate_no_match(
    regions: list[dict[str, object]], families: list[dict[str, object]],
) -> dict[str, object]:
    trials = []
    for region in regions:
        score = _family_scores(
            np.asarray(region["descriptor"]), families, exclude_chart=int(region["chart"]),
        )
        order = np.argsort(-score, kind="stable")
        top = int(order[0])
        second = float(score[order[1]]) if len(order) > 1 else -1.0
        own = int(region["family_id"])
        has_other_view = any(
            int(row["chart"]) != int(region["chart"])
            for row in families[own]["member_regions"]
        )
        trials.append({
            "matched": bool(has_other_view),
            "correct": bool(has_other_view and top == own),
            "top_score": float(score[top]),
            "margin": float(score[top] - second),
        })
    best = None
    for threshold in np.linspace(0.40, 0.90, 26):
        for margin in np.linspace(0.0, 0.20, 11):
            accepted = np.asarray([
                row["top_score"] >= threshold and row["margin"] >= margin for row in trials
            ])
            matched = np.asarray([row["matched"] for row in trials])
            correct = np.asarray([row["correct"] for row in trials])
            true_positive = int(np.sum(accepted & correct))
            false_positive = int(np.sum(accepted & ~correct))
            false_accept_unmatched = int(np.sum(accepted & ~matched))
            unmatched = int(np.sum(~matched))
            matched_count = int(np.sum(matched))
            recall = true_positive / max(matched_count, 1)
            precision = true_positive / max(true_positive + false_positive, 1)
            fpr = false_accept_unmatched / max(unmatched, 1)
            if fpr > SOURCE_MAXIMUM_UNMATCHED_FALSE_ACCEPT + 1e-12:
                continue
            key = (recall, precision, -float(threshold), -float(margin))
            if best is None or key > best[0]:
                best = (key, threshold, margin, recall, precision, fpr, accepted)
    if best is None:
        raise ValueError("source-only no-match calibration has no feasible operating point")
    _, threshold, margin, recall, precision, fpr, accepted = best
    return {
        "score_threshold": float(threshold),
        "margin_threshold": float(margin),
        "source_trial_count": len(trials),
        "source_matchable_trial_count": int(sum(row["matched"] for row in trials)),
        "source_accepted_count": int(np.sum(accepted)),
        "source_correct_recall": float(recall),
        "source_precision": float(precision),
        "source_unmatched_false_accept_rate": float(fpr),
        "maximum_unmatched_false_accept_rate": SOURCE_MAXIMUM_UNMATCHED_FALSE_ACCEPT,
        "selection_rule": "max_correct_recall_then_precision_subject_to_source_unmatched_FPR<=0.10",
    }


def _best_member_correspondence(
    query: dict[str, object], family: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    best = None
    for member in family["member_regions"]:
        cosine = np.asarray(query["token_features"]) @ np.asarray(member["token_features"]).T
        if cosine.shape[1] < 2:
            continue
        top2 = np.argpartition(cosine, -2, axis=1)[:, -2:]
        value = np.take_along_axis(cosine, top2, axis=1)
        local_order = np.argsort(value, axis=1)
        target = top2[np.arange(len(cosine)), local_order[:, 1]]
        score = cosine[np.arange(len(cosine)), target]
        second = value[np.arange(len(cosine)), local_order[:, 0]]
        reverse = np.argmax(cosine, axis=0)
        keep = (
            (reverse[target] == np.arange(len(cosine)))
            & (score >= MINIMUM_TOKEN_COSINE)
            & ((score - second) >= MINIMUM_TOKEN_MARGIN)
        )
        source_rows = np.flatnonzero(keep)
        target_rows = target[keep]
        inlier = np.zeros((len(source_rows),), bool)
        if len(source_rows) >= MINIMUM_HOMOGRAPHY_MATCHES:
            source_token = np.asarray(query["token_ids"])[source_rows]
            target_token = np.asarray(member["token_ids"])[target_rows]
            source_xy = np.stack((source_token % 64, source_token // 64), axis=1).astype(np.float64)
            target_xy = np.stack((target_token % 64, target_token // 64), axis=1).astype(np.float64)
            cv2.setRNGSeed(260831)
            _, mask = cv2.findHomography(
                source_xy, target_xy, cv2.RANSAC,
                HOMOGRAPHY_RANSAC_THRESHOLD_TOKEN_CELLS, maxIters=2000, confidence=0.995,
            )
            if mask is not None:
                inlier = mask.reshape(-1).astype(bool)
        accepted = (
            int(inlier.sum()) >= MINIMUM_HOMOGRAPHY_MATCHES
            and float(np.mean(inlier)) >= MINIMUM_HOMOGRAPHY_INLIER_RATIO
        )
        key = (int(inlier.sum()) if accepted else 0, float(np.median(score[keep])) if keep.any() else -1.0)
        if best is None or key > best[0]:
            best = (key, member, source_rows[inlier] if accepted else np.empty(0, np.int64), target_rows[inlier] if accepted else np.empty(0, np.int64), int(keep.sum()), float(np.mean(inlier)) if len(inlier) else 0.0)
    if best is None:
        return np.zeros((0, 3)), np.zeros((0, 3)), {"accepted": False, "inlier_count": 0}
    _, member, source_rows, target_rows, pre_count, ratio = best
    return (
        np.asarray(query["token_points"])[source_rows],
        np.asarray(member["token_points"])[target_rows],
        {
            "accepted": bool(len(source_rows) >= MINIMUM_HOMOGRAPHY_MATCHES),
            "map_chart_name": str(member["chart_name"]),
            "pre_homography_match_count": pre_count,
            "inlier_count": int(len(source_rows)),
            "inlier_ratio": ratio,
        },
    )


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    usable = [row for row in rows if row["pose"]["usable"]]
    translation = np.asarray([row["pose"]["translation_error_m"] for row in usable])
    rotation = np.asarray([row["pose"]["rotation_error_deg"] for row in usable])
    return {
        "query_count": len(rows),
        "usable_count": len(usable),
        "translation_error_median_m": float(np.median(translation)) if len(translation) else None,
        "rotation_error_median_deg": float(np.median(rotation)) if len(rotation) else None,
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
        raise FileExistsError("refusing to overwrite large-plane RADIO control")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("large-plane authority differs from pin")
    vertices = _load_aligned_vertices(args.alignment, authority)
    records = _radio_records(args.radio_manifest)
    local_regions, map_token_hash = _map_local_regions(authority, vertices, records)
    families = build_plane_families(local_regions)
    calibration = calibrate_no_match(local_regions, families)
    query_manifest_path = args.moge3_query / "manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    query_names = [str(row["name"]) for row in query_manifest.get("rows", [])]
    if not query_names or len(query_names) != len(set(query_names)):
        raise ValueError("query MoGe3 inventory is empty or duplicated")
    frozen = []
    for name in query_names:
        feature, query_hash = _radio_tokens(name, records)
        with np.load(args.moge3_query / f"{name}.npz", allow_pickle=False) as data:
            points = np.asarray(data["points_camera"], np.float64)
            normals = np.asarray(data["normal_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
        planes = extract_large_plane_regions(
            points, normals, valid, minimum_pixels=160,
            maximum_planes=MAXIMUM_QUERY_PLANES, maximum_hypotheses=128,
        )
        query_regions = _regions_from_grid(planes, points, feature, query=True)
        source_points, target_points, decisions = [], [], []
        for query_row, query in enumerate(query_regions):
            score = _family_scores(np.asarray(query["descriptor"]), families)
            order = np.argsort(-score, kind="stable")
            margin = float(score[order[0]] - score[order[1]]) if len(order) > 1 else float("inf")
            no_match_accept = bool(
                score[order[0]] >= calibration["score_threshold"]
                and margin >= calibration["margin_threshold"]
            )
            candidates = order[:MAXIMUM_FAMILY_CANDIDATES] if no_match_accept else np.zeros((0,), np.int64)
            candidate_rows = []
            for rank, family_row in enumerate(candidates):
                source, target, match = _best_member_correspondence(query, families[int(family_row)])
                candidate_rows.append({
                    "family_id": int(family_row), "rank": rank + 1,
                    "family_score": float(score[family_row]), **match,
                })
                if match["accepted"]:
                    source_points.extend(source)
                    target_points.extend(target)
            decisions.append({
                "query_region": query_row,
                "query_token_count": int(len(query["token_ids"])),
                "top_family": int(order[0]),
                "top_score": float(score[order[0]]),
                "top_margin": margin,
                "no_match_accepted": no_match_accept,
                "candidates": candidate_rows,
            })
        source = np.asarray(source_points, np.float64).reshape(-1, 3)
        target = np.asarray(target_points, np.float64).reshape(-1, 3)
        pose: dict[str, object] = {"usable": False}
        if len(source) >= MINIMUM_POINT_CORRESPONDENCES:
            rotation, translation, scale, inlier, residual = _robust_point_pose(source, target, estimate_scale=True)
            pose = {
                "usable": True, "rotation_c2w": rotation.tolist(),
                "translation_world": translation.tolist(), "estimated_scale": float(scale),
                "inlier_count": int(inlier.sum()),
                "residual_median_m": float(np.median(residual[inlier])) if inlier.any() else None,
                "residual_p90_m": float(np.quantile(residual[inlier], 0.9)) if inlier.any() else None,
            }
        frozen.append({
            "name": name, "query_radio_file_sha256": query_hash,
            "points_camera": points, "valid": valid, "query_regions": query_regions,
            "query_plane_count": len(query_regions), "token_correspondence_count": int(len(source)),
            "decisions": decisions, "pose": pose,
        })
    # Phase two: only now can reference geometry and target pose be observed.
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    held_row = {name: row for row, name in enumerate(held.view_names.astype(str))}
    if set(held_row) != set(query_names):
        raise ValueError("held inventory differs from frozen large-plane queries")
    family_trees = [cKDTree(np.asarray(row["points_world"])) for row in families]
    rows = []
    matchable_total = accepted_matchable = accepted_unmatchable = accepted_correct = 0
    query_plane_total = 0
    for item in frozen:
        view = held_row[item["name"]]
        common = held.reference_valid[view] & item["valid"]
        scale = float(np.median(held.reference_depth_m[view][common] / item["points_camera"][..., 2][common]))
        c2w = held.camera_to_world[view]
        decisions = []
        for query, decision in zip(item["query_regions"], item["decisions"]):
            world = scale * (np.asarray(query["token_points"]) @ c2w[:3, :3].T) + c2w[:3, 3]
            normal = np.asarray(query["normal"]) @ c2w[:3, :3].T
            compatible = []
            for family, family_row in enumerate(families):
                angle = float(np.degrees(np.arccos(np.clip(abs(float(normal @ family_row["normal_world"])), -1.0, 1.0))))
                if angle > PHYSICAL_NORMAL_DEGREES:
                    continue
                distance = family_trees[family].query(world, k=1)[0]
                if float(np.median(distance)) <= PHYSICAL_MEDIAN_M and float(np.quantile(distance, 0.9)) <= PHYSICAL_P90_M:
                    compatible.append(family)
            accepted = bool(decision["no_match_accepted"])
            matchable = bool(compatible)
            correct = accepted and int(decision["top_family"]) in compatible
            query_plane_total += 1
            matchable_total += int(matchable)
            accepted_matchable += int(accepted and matchable)
            accepted_unmatchable += int(accepted and not matchable)
            accepted_correct += int(correct)
            decisions.append({**decision, "physically_matchable": matchable, "physically_compatible_families": compatible, "accepted_family_physically_correct": correct})
        pose = dict(item["pose"])
        if pose["usable"]:
            rotation = np.asarray(pose.pop("rotation_c2w"), np.float64)
            translation = np.asarray(pose.pop("translation_world"), np.float64)
            pose["rotation_error_deg"] = _rotation_error_deg(rotation, c2w)
            pose["translation_error_m"] = float(np.linalg.norm(translation - c2w[:3, 3]))
        rows.append({
            "name": item["name"], "query_plane_count": item["query_plane_count"],
            "token_correspondence_count": item["token_correspondence_count"],
            "moge_scale_postlabel": scale, "decisions": decisions, "pose": pose,
        })
    payload = {
        "artifact_type": SCHEMA,
        "authority_file_sha256": file_sha256(args.authority),
        "authority_content_sha256": args.expected_authority_content_sha256,
        "alignment_manifest_file_sha256": file_sha256(args.alignment / "manifest.json"),
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "moge3_query_manifest_file_sha256": file_sha256(query_manifest_path),
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "map_radio_file_inventory_sha256": canonical_json_sha256(map_token_hash),
        "map_local_plane_count": len(local_regions),
        "map_large_plane_family_count": len(families),
        "map_multiview_family_count": int(sum(int(row["chart_count"]) >= 2 for row in families)),
        "map_family_chart_counts": [int(row["chart_count"]) for row in families],
        "source_only_no_match_calibration": calibration,
        "held_postlabel_no_match_audit": {
            "query_plane_count": query_plane_total,
            "physically_matchable_count": matchable_total,
            "physically_matchable_fraction": matchable_total / max(query_plane_total, 1),
            "accepted_matchable_recall": accepted_matchable / max(matchable_total, 1),
            "accepted_unmatchable_false_accept_rate": accepted_unmatchable / max(query_plane_total - matchable_total, 1),
            "accepted_family_physical_precision": accepted_correct / max(accepted_matchable + accepted_unmatchable, 1),
        },
        "phase_separation": {
            "source_family_and_no_match_fit_uses_held": False,
            "all_query_plane_decisions_correspondences_and_poses_frozen_before_held_open": True,
            "held_used_only_for_postlabel_physical_audit_and_pose_error": True,
        },
        "summary": _summary(rows),
        "historical_held_control": True,
        "blind_or_preregistered_claim": False,
        "production_eligible": False,
        "promotion_eligible": False,
        "rows": rows,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"], "summary": payload["summary"],
        "no_match": payload["held_postlabel_no_match_audit"],
        "source_calibration": calibration,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
