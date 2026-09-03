"""Pose-free grouped PnP hypotheses with a post-freeze development evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    base_keys = (
        "names", "correspondence_offsets", "world_points", "query_tokens",
        "provenance_region_plane_atlas_row", "camera_matrices", "radial_k1",
    )
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        keys = list(base_keys)
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v2",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
        ):
            keys += ["prototype_atlas_row", "query_plane_visible_fraction", "radio_match_score"]
        if metadata.get("artifact_type") in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
        ):
            keys += [
                "prototype_world_covariance_m2", "prototype_plane_pixel_purity",
                "prototype_plane_depth_dispersion_m",
            ]
        if metadata.get("artifact_type") == "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4":
            keys += ["query_measurements_xy"]
        arrays = {key: np.asarray(data[key]) for key in keys}
    count = len(arrays["names"])
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v1",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v2",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
            "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
        )
        or metadata.get("pose_or_ground_truth_opened") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or arrays["correspondence_offsets"].shape != (count + 1,)
        or arrays["camera_matrices"].shape != (count, 3, 3)
        or arrays["provenance_region_plane_atlas_row"].shape
        != (len(arrays["world_points"]), 3)
        or (
            "query_measurements_xy" in arrays
            and arrays["query_measurements_xy"].shape != (len(arrays["world_points"]), 2)
        )
        or (
            metadata.get("artifact_type") == "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4"
            and (
                metadata.get("query_measurement_semantics")
                != "centroid_of_observed_same_plane_region_pixels_inside_each_4x4_RADIO_token"
                or metadata.get("hidden_or_occluded_pixels_added_to_query_measurement") != 0
            )
        )
    ):
        raise ValueError("frozen PnP correspondence inventory differs")
    return arrays, metadata


def _solve(
    world: np.ndarray,
    tokens: np.ndarray,
    K: np.ndarray,
    k1: float,
    rows: np.ndarray,
    query_measurements_xy: np.ndarray | None = None,
) -> np.ndarray | None:
    if len(rows) < 6:
        return None
    pixel_all = (
        np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        if query_measurements_xy is None
        else np.asarray(query_measurements_xy, np.float64).reshape(-1, 2)
    )
    if len(pixel_all) != len(tokens) or not np.all(np.isfinite(pixel_all)):
        raise ValueError("query measurements differ from tokens")
    pixel = pixel_all[rows]
    distortion = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64)
    cv2.setRNGSeed(260901)
    ok, rvec, tvec, inlier = cv2.solvePnPRansac(
        world[rows].astype(np.float64), pixel.astype(np.float64), K, distortion,
        iterationsCount=1000, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inlier is None or len(inlier) < 6:
        return None
    selected = rows[inlier.reshape(-1)]
    all_pixel = pixel_all[selected]
    rvec, tvec = cv2.solvePnPRefineLM(
        world[selected], all_pixel, K, distortion, rvec, tvec,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = np.asarray(tvec).reshape(3)
    return pose


def _score(
    pose: np.ndarray,
    world: np.ndarray,
    tokens: np.ndarray,
    provenance: np.ndarray,
    K: np.ndarray,
    k1: float,
    query_measurements_xy: np.ndarray | None = None,
) -> dict[str, object]:
    pixel = (
        np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        if query_measurements_xy is None
        else np.asarray(query_measurements_xy, np.float64).reshape(-1, 2)
    )
    if len(pixel) != len(tokens) or not np.all(np.isfinite(pixel)):
        raise ValueError("query measurements differ from tokens")
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], K,
        np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    valid = (camera[:, 2] > 0.0) & (residual <= 4.0)
    # A query token may carry multiple mutually exclusive 3D hypotheses.  A
    # candidate pose can receive support from that image location only once:
    # retain the valid hypothesis with the smallest residual.
    chosen = []
    for token_id in np.unique(tokens):
        rows = np.flatnonzero((tokens == token_id) & valid)
        if len(rows):
            chosen.append(int(rows[np.argmin(residual[rows])]))
    inlier = np.asarray(chosen, np.int64)

    def support(column: int, cap: int) -> tuple[int, int]:
        if not len(inlier):
            return 0, 0
        _, count = np.unique(provenance[inlier, column], return_counts=True)
        return int(np.sum(np.minimum(count, cap))), int(np.sum(count >= 3))

    region_capped, region_supported = support(0, 8)
    plane_capped, plane_supported = support(1, 8)
    view_capped, view_supported = support(2, 8)
    return {
        "pose": pose,
        "inlier_count": int(len(inlier)),
        "inlier_ratio": float(len(inlier) / max(len(np.unique(tokens)), 1)),
        "region_capped_support": region_capped,
        "plane_capped_support": plane_capped,
        "view_capped_support": view_capped,
        "supported_region_count": region_supported,
        "supported_plane_count": plane_supported,
        "supported_view_count": view_supported,
        "reprojection_median_px": None if not len(inlier) else float(np.median(residual[inlier])),
    }


def _top_groups(values: np.ndarray, maximum: int) -> list[np.ndarray]:
    groups = []
    for value in np.unique(values):
        rows = np.flatnonzero(values == value)
        if len(rows) >= 6:
            groups.append(rows)
    groups.sort(key=lambda rows: (-len(rows), int(rows[0])))
    return groups[:maximum]


def _choice_key(candidate: dict[str, object], rule: str) -> tuple[object, ...]:
    if rule == "raw_inliers":
        return (candidate["inlier_count"], -float(candidate["reprojection_median_px"] or 1e9))
    if rule == "balanced_support":
        return (
            candidate["region_capped_support"], candidate["plane_capped_support"],
            candidate["view_capped_support"], candidate["inlier_count"],
            -float(candidate["reprojection_median_px"] or 1e9),
        )
    if rule == "supported_entities":
        return (
            candidate["supported_region_count"], candidate["supported_plane_count"],
            candidate["supported_view_count"], candidate["inlier_count"],
            -float(candidate["reprojection_median_px"] or 1e9),
        )
    raise ValueError("unknown selection rule")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--maximum_groups_per_kind", type=int, default=16)
    parser.add_argument("--output_frozen_candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_candidates.exists():
        raise FileExistsError("refusing to overwrite multi-hypothesis result")
    arrays, upstream = _load(args.frozen_correspondences)
    rules = ("raw_inliers", "balanced_support", "supported_entities")
    selected: dict[str, list[dict[str, object]]] = {rule: [] for rule in rules}
    all_candidates: list[list[dict[str, object]]] = []
    diagnostic_rows = []
    for query_index, name in enumerate(arrays["names"].astype(str).tolist()):
        lo, hi = map(int, arrays["correspondence_offsets"][query_index:query_index + 2])
        world = arrays["world_points"][lo:hi]
        tokens = arrays["query_tokens"][lo:hi]
        measurements = arrays["query_measurements_xy"][lo:hi] if "query_measurements_xy" in arrays else None
        provenance = arrays["provenance_region_plane_atlas_row"][lo:hi]
        K = arrays["camera_matrices"][query_index]
        k1 = float(arrays["radial_k1"][query_index])
        seed_groups: list[tuple[str, np.ndarray]] = [("all", np.arange(len(world)))]
        for label, column in (("plane", 1), ("source_view", 2), ("query_region", 0)):
            seed_groups.extend(
                (label, rows) for rows in _top_groups(provenance[:, column], args.maximum_groups_per_kind)
            )
        candidates = []
        for origin, rows in seed_groups:
            pose = _solve(world, tokens, K, k1, rows, measurements)
            if pose is None:
                continue
            score = _score(pose, world, tokens, provenance, K, k1, measurements)
            score["origin"] = origin
            candidates.append(score)
        if not candidates:
            candidates.append({
                "pose": np.full((4, 4), np.nan), "origin": "none", "inlier_count": 0,
                "inlier_ratio": 0.0, "region_capped_support": 0,
                "plane_capped_support": 0, "view_capped_support": 0,
                "supported_region_count": 0, "supported_plane_count": 0,
                "supported_view_count": 0, "reprojection_median_px": None,
            })
        all_candidates.append(candidates)
        row = {"name": name, "candidate_count": int(len(candidates))}
        for rule in rules:
            best = max(candidates, key=lambda candidate: _choice_key(candidate, rule))
            selected[rule].append(best)
            row[rule] = {key: value for key, value in best.items() if key != "pose"}
        diagnostic_rows.append(row)

    candidate_offsets = np.zeros(len(all_candidates) + 1, np.int64)
    for index, candidates in enumerate(all_candidates):
        candidate_offsets[index + 1] = candidate_offsets[index] + len(candidates)
    frozen_arrays: dict[str, np.ndarray] = {
        "names": arrays["names"],
        "candidate_offsets": candidate_offsets,
        "candidate_pose_w2c": np.asarray([
            candidate["pose"] for candidates in all_candidates for candidate in candidates
        ], np.float64),
        "candidate_origin": np.asarray([
            candidate["origin"] for candidates in all_candidates for candidate in candidates
        ]),
        "candidate_inlier_count": np.asarray([
            candidate["inlier_count"] for candidates in all_candidates for candidate in candidates
        ], np.int64),
    }
    for rule in rules:
        frozen_arrays[f"{rule}_pose_w2c"] = np.asarray([row["pose"] for row in selected[rule]])
        frozen_arrays[f"{rule}_inlier_count"] = np.asarray([row["inlier_count"] for row in selected[rule]])
        frozen_arrays[f"{rule}_inlier_ratio"] = np.asarray(
            [row["inlier_ratio"] for row in selected[rule]], np.float64
        )
    frozen_metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_grouped_multihypothesis_v1",
        "arrays_sha256": arrays_sha256(frozen_arrays),
        "query_count": int(len(arrays["names"])),
        "candidate_generation": "all plus top16 physical-plane/source-view/query-region grouped PnP seeds",
        "selection_rules": list(rules),
        "query_pose_or_ground_truth_opened": False,
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": upstream.get("content_sha256"),
    }
    frozen_metadata["content_sha256"] = canonical_json_sha256(frozen_metadata)
    args.output_frozen_candidates.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_candidates, **frozen_arrays,
        metadata_json=np.asarray(json.dumps(frozen_metadata, sort_keys=True)),
    )

    summaries = {}
    postlabel_rows = []
    for index, name in enumerate(arrays["names"].astype(str).tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        gt_center = -gt[:3, :3].T @ gt[:3, 3]
        row = {"name": name}
        for rule in rules:
            pose = np.asarray(selected[rule][index]["pose"], np.float64)
            if np.all(np.isfinite(pose)):
                center = -pose[:3, :3].T @ pose[:3, 3]
                translation = float(np.linalg.norm(center - gt_center))
                rotation = float(
                    Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude()
                    * 180.0 / np.pi
                )
            else:
                translation = rotation = float("inf")
            row[rule] = {"translation_error_m": translation, "rotation_error_deg": rotation}
        candidate_errors = []
        for candidate in all_candidates[index]:
            pose = np.asarray(candidate["pose"], np.float64)
            if not np.all(np.isfinite(pose)):
                continue
            center = -pose[:3, :3].T @ pose[:3, 3]
            candidate_errors.append((
                float(np.linalg.norm(center - gt_center)),
                float(
                    Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude()
                    * 180.0 / np.pi
                ),
            ))
        row["candidate_oracle_2m45"] = bool(any(
            translation <= 2.0 and rotation <= 45.0
            for translation, rotation in candidate_errors
        ))
        row["candidate_oracle_1m10"] = bool(any(
            translation <= 1.0 and rotation <= 10.0
            for translation, rotation in candidate_errors
        ))
        row["minimum_candidate_translation_m"] = (
            None if not candidate_errors else min(value[0] for value in candidate_errors)
        )
        postlabel_rows.append(row)
    for rule in rules:
        values = [row[rule] for row in postlabel_rows]
        summaries[rule] = {
            "recall_2m45": float(np.mean([
                value["translation_error_m"] <= 2.0 and value["rotation_error_deg"] <= 45.0
                for value in values
            ])),
            "recall_1m10": float(np.mean([
                value["translation_error_m"] <= 1.0 and value["rotation_error_deg"] <= 10.0
                for value in values
            ])),
            "median_translation_m": float(np.median([value["translation_error_m"] for value in values])),
            "median_rotation_deg": float(np.median([value["rotation_error_deg"] for value in values])),
        }
    summaries["candidate_pool_oracle"] = {
        "recall_2m45": float(np.mean([row["candidate_oracle_2m45"] for row in postlabel_rows])),
        "recall_1m10": float(np.mean([row["candidate_oracle_1m10"] for row in postlabel_rows])),
        "note": "post-label upper bound over poses frozen before labels; not deployable selection",
    }
    report = {
        "artifact_type": "goal_maplet_direct_plane_pnp_grouped_multihypothesis_dev_evaluation_v1",
        "candidate_inventory_file_sha256": file_sha256(args.output_frozen_candidates),
        "candidate_inventory_content_sha256": frozen_metadata["content_sha256"],
        "selection_frozen_before_query_pose_opened": True,
        "summaries": summaries,
        "pose_free_diagnostics": diagnostic_rows,
        "postlabel_rows": postlabel_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("pose_free_diagnostics", "postlabel_rows")}, indent=2))


if __name__ == "__main__":
    main()
