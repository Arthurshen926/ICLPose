"""Validate a track-free VFM/2DGS map without reading query/test GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import _load_camera_by_image
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.localization.surface_localization import AnchorLocalDescriptorBank
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap, VfmSurfaceMapletBank


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(np.max(array)),
    }


def _pose_error(estimated_w2c: np.ndarray, target_w2c: np.ndarray) -> tuple[float, float]:
    estimated = np.asarray(estimated_w2c, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_w2c, dtype=np.float64).reshape(4, 4)
    estimated_center = -estimated[:3, :3].T @ estimated[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation_cm = 100.0 * float(np.linalg.norm(estimated_center - target_center))
    relative = estimated[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return translation_cm, float(np.degrees(np.arccos(cosine)))


def _solve_pnp(xyz: np.ndarray, xy: np.ndarray, camera) -> np.ndarray | None:
    import cv2

    if len(xyz) < 6:
        return None
    matrix, distortion = camera_matrix_and_distortion(camera)
    success, rvec, tvec, _inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(xyz.astype(np.float64)),
        np.ascontiguousarray(xy.astype(np.float64)),
        matrix,
        distortion,
        iterationsCount=2000,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not bool(success):
        return None
    rotation, _jacobian = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return pose


def evaluate_mapping_view_geometry(
    anchors: StableSurfaceAnchorMap,
    pose_w2c_by_image: dict[str, np.ndarray],
    camera_by_image: dict[str, object],
    pixel_noise_std: float,
    noise_trials: int,
    seed: int,
) -> dict[str, object]:
    rows_by_image: dict[str, list[tuple[int, int]]] = {}
    for anchor_row in range(len(anchors)):
        start, end = int(anchors.observation_offsets[anchor_row]), int(anchors.observation_offsets[anchor_row + 1])
        for observation_row in range(start, end):
            image_id = anchors.observation_image_ids[observation_row]
            rows_by_image.setdefault(image_id, []).append((anchor_row, observation_row))
    rng = np.random.default_rng(int(seed))
    exact_translation: list[float] = []
    exact_rotation: list[float] = []
    noisy_translation: list[float] = []
    noisy_rotation: list[float] = []
    per_view = []
    for image_id, pairs in sorted(rows_by_image.items()):
        if image_id not in pose_w2c_by_image or image_id not in camera_by_image:
            continue
        anchor_rows = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        observation_rows = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        xyz = anchors.xyz[anchor_rows]
        xy = anchors.observation_xy[observation_rows]
        exact_pose = _solve_pnp(xyz, xy, camera_by_image[image_id])
        exact_error = None
        if exact_pose is not None:
            exact_error = _pose_error(exact_pose, pose_w2c_by_image[image_id])
            exact_translation.append(exact_error[0])
            exact_rotation.append(exact_error[1])
        noisy_errors = []
        for _trial in range(int(noise_trials)):
            noisy_xy = xy + rng.normal(0.0, float(pixel_noise_std), size=xy.shape)
            noisy_pose = _solve_pnp(xyz, noisy_xy, camera_by_image[image_id])
            if noisy_pose is None:
                continue
            error = _pose_error(noisy_pose, pose_w2c_by_image[image_id])
            noisy_translation.append(error[0])
            noisy_rotation.append(error[1])
            noisy_errors.append(error)
        per_view.append(
            {
                "image_id": image_id,
                "anchor_count": int(len(pairs)),
                "exact_translation_cm": None if exact_error is None else float(exact_error[0]),
                "exact_rotation_deg": None if exact_error is None else float(exact_error[1]),
                "noisy_translation_cm_median": (
                    float(np.median([value[0] for value in noisy_errors])) if noisy_errors else None
                ),
                "noisy_rotation_deg_median": (
                    float(np.median([value[1] for value in noisy_errors])) if noisy_errors else None
                ),
            }
        )
    return {
        "view_count": int(len(per_view)),
        "exact": {
            "translation_cm": _stats(exact_translation),
            "rotation_deg": _stats(exact_rotation),
        },
        "pixel_noise": {
            "std_px": float(pixel_noise_std),
            "trials_per_view": int(noise_trials),
            "translation_cm": _stats(noisy_translation),
            "rotation_deg": _stats(noisy_rotation),
        },
        "per_view": per_view,
    }


def evaluate_leave_one_view_out_maplet_retrieval(bank: VfmSurfaceMapletBank) -> dict[str, object]:
    ranks = []
    cosine_positive = []
    for maplet_row in range(len(bank)):
        start, end = int(bank.view_offsets[maplet_row]), int(bank.view_offsets[maplet_row + 1])
        if end - start < 2:
            continue
        for heldout_row in range(start, end):
            other_rows = [row for row in range(start, end) if row != heldout_row]
            weights = np.maximum(bank.view_quality_scores[np.asarray(other_rows)], 1e-8)
            prototype = np.average(bank.view_descriptors[np.asarray(other_rows)], axis=0, weights=weights)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
            candidate = bank.descriptors.copy()
            candidate[maplet_row] = prototype
            query = bank.view_descriptors[heldout_row]
            scores = candidate @ query
            order = np.argsort(-scores, kind="mergesort")
            rank = int(np.flatnonzero(order == maplet_row)[0]) + 1
            ranks.append(rank)
            cosine_positive.append(float(scores[maplet_row]))
    rank_array = np.asarray(ranks, dtype=np.int64)
    return {
        "query_count": int(rank_array.size),
        "recall_at_1": float(np.mean(rank_array <= 1)) if rank_array.size else 0.0,
        "recall_at_5": float(np.mean(rank_array <= 5)) if rank_array.size else 0.0,
        "recall_at_10": float(np.mean(rank_array <= 10)) if rank_array.size else 0.0,
        "rank": _stats(rank_array.astype(np.float64)),
        "positive_cosine": _stats(cosine_positive),
    }


def evaluate_leave_one_view_out_local_anchor_retrieval(
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
) -> dict[str, object]:
    """Resolve anchor identity within its maplet while excluding the query view."""

    anchor_row_by_id = anchors.row_by_id()
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
        if int(anchor_id) in anchor_row_by_id
    }
    bank_rows_by_maplet: dict[int, list[int]] = {}
    for anchor_id, bank_row in bank_row_by_id.items():
        anchor_row = anchor_row_by_id[anchor_id]
        maplet_id = int(anchors.owner_maplet_ids[anchor_row])
        bank_rows_by_maplet.setdefault(maplet_id, []).append(bank_row)
    ranks: list[int] = []
    positive_cosine: list[float] = []
    candidate_counts: list[int] = []
    excluded_no_positive = 0
    for target_anchor_id, target_bank_row in bank_row_by_id.items():
        target_anchor_row = anchor_row_by_id[target_anchor_id]
        maplet_id = int(anchors.owner_maplet_ids[target_anchor_row])
        candidate_bank_rows = bank_rows_by_maplet.get(maplet_id, [])
        target_start = int(descriptor_bank.descriptor_offsets[target_bank_row])
        target_end = int(descriptor_bank.descriptor_offsets[target_bank_row + 1])
        for query_row in range(target_start, target_end):
            query_image = descriptor_bank.support_image_ids[query_row]
            prototypes: list[np.ndarray] = []
            candidate_ids: list[int] = []
            for candidate_bank_row in candidate_bank_rows:
                start = int(descriptor_bank.descriptor_offsets[candidate_bank_row])
                end = int(descriptor_bank.descriptor_offsets[candidate_bank_row + 1])
                rows = np.asarray(
                    [
                        row
                        for row in range(start, end)
                        if descriptor_bank.support_image_ids[row] != query_image
                    ],
                    dtype=np.int64,
                )
                if rows.size == 0:
                    continue
                weights = np.maximum(descriptor_bank.descriptor_quality[rows], 1e-8)
                prototype = np.average(descriptor_bank.descriptors[rows], axis=0, weights=weights)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
                prototypes.append(prototype.astype(np.float32, copy=False))
                candidate_ids.append(int(descriptor_bank.anchor_ids[candidate_bank_row]))
            if target_anchor_id not in candidate_ids:
                excluded_no_positive += 1
                continue
            prototype_matrix = np.stack(prototypes, axis=0)
            scores = prototype_matrix @ descriptor_bank.descriptors[query_row]
            order = np.argsort(-scores, kind="mergesort")
            target_column = candidate_ids.index(target_anchor_id)
            rank = int(np.flatnonzero(order == target_column)[0]) + 1
            ranks.append(rank)
            positive_cosine.append(float(scores[target_column]))
            candidate_counts.append(len(candidate_ids))
    rank_array = np.asarray(ranks, dtype=np.int64)
    return {
        "query_count": int(rank_array.size),
        "excluded_without_cross_view_positive": int(excluded_no_positive),
        "candidate_scope": "same_surface_maplet",
        "same_support_image_excluded": True,
        "recall_at_1": float(np.mean(rank_array <= 1)) if rank_array.size else 0.0,
        "recall_at_3": float(np.mean(rank_array <= 3)) if rank_array.size else 0.0,
        "recall_at_5": float(np.mean(rank_array <= 5)) if rank_array.size else 0.0,
        "rank": _stats(rank_array.astype(np.float64)),
        "candidate_count": _stats(candidate_counts),
        "positive_cosine": _stats(positive_cosine),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--pixel_noise_std", type=float, default=1.0)
    parser.add_argument("--noise_trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference_sfm_map_oracle_median_cm", type=float, default=0.84)
    parser.add_argument("--local_descriptor_bank", default="")
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    camera_by_image = _load_camera_by_image(str(args.camera_model_dir))
    geometry = evaluate_mapping_view_geometry(
        anchors,
        pose_by_image,
        camera_by_image,
        pixel_noise_std=float(args.pixel_noise_std),
        noise_trials=int(args.noise_trials),
        seed=int(args.seed),
    )
    retrieval = evaluate_leave_one_view_out_maplet_retrieval(maplets)
    local_retrieval = None
    if str(args.local_descriptor_bank):
        local_bank = AnchorLocalDescriptorBank.load_npz(Path(args.local_descriptor_bank))
        if bool(local_bank.metadata.get("uses_sfm_tracks", False)):
            raise ValueError("local descriptor bank illegally uses SfM tracks")
        if bool(local_bank.metadata.get("uses_sfm_points", False)):
            raise ValueError("local descriptor bank illegally uses SfM points")
        if bool(local_bank.metadata.get("uses_radio_intermediate", False)):
            raise ValueError("local descriptor bank illegally uses RADIO intermediate")
        local_retrieval = evaluate_leave_one_view_out_local_anchor_retrieval(
            anchors,
            local_bank,
        )
    anchor_view_counts = np.diff(anchors.observation_offsets)
    maplet_anchor_counts = np.diff(maplets.anchor_offsets)
    maplet_view_counts = np.diff(maplets.view_offsets)
    exact_median = float(dict(geometry["exact"])["translation_cm"]["median"])
    summary = {
        "stage": "track_free_vfm_2dgs_surface_map_validation",
        "production_contract": {
            "vfm_layer": maplets.metadata.get("vfm_layer"),
            "uses_radio_intermediate": bool(maplets.metadata.get("uses_radio_intermediate", False)),
            "uses_sfm_points": bool(maplets.metadata.get("uses_sfm_points", False)),
            "uses_sfm_tracks": bool(maplets.metadata.get("uses_sfm_tracks", False)),
            "maplet_centroids_used_for_pnp": False,
        },
        "maplet_count": int(len(maplets)),
        "stable_anchor_count": int(len(anchors)),
        "maplet_anchor_count": _stats(maplet_anchor_counts),
        "maplet_view_count": _stats(maplet_view_counts),
        "stable_anchor_view_count": _stats(anchor_view_counts),
        "anchor_quality": _stats(anchors.quality_scores),
        "anchor_geometry_confidence": _stats(anchors.geometry_confidence),
        "anchor_opacity": _stats(anchors.opacity),
        "anchor_support_radius_m": _stats(anchors.support_radii),
        "mapping_view_geometry": geometry,
        "maplet_retrieval_leave_one_view_out": retrieval,
        "local_anchor_retrieval_leave_one_view_out": local_retrieval,
        "non_regression_reference": {
            "name": "legacy_sfm_map_oracle_median",
            "translation_cm": float(args.reference_sfm_map_oracle_median_cm),
            "coordinate_contract_oracle_pass": bool(
                exact_median <= float(args.reference_sfm_map_oracle_median_cm)
            ),
            "note": (
                "This mapping-view coordinate-contract oracle is necessary but not an end-to-end "
                "localization comparison. Final promotion still requires identical frozen query IDs."
            ),
        },
        "inputs": {
            "maplets": str(args.maplets),
            "anchors": str(args.anchors),
            "mapping_pose_file": str(args.mapping_pose_file),
            "camera_model_dir": str(args.camera_model_dir),
            "local_descriptor_bank": str(args.local_descriptor_bank),
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "maplet_count",
        "stable_anchor_count",
        "mapping_view_geometry",
        "maplet_retrieval_leave_one_view_out",
        "local_anchor_retrieval_leave_one_view_out",
        "non_regression_reference",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
