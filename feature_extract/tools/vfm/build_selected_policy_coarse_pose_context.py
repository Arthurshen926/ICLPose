"""Build no-GT coarse-PnP context for frozen selected-policy matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    resolve_pose_match_conflicts,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_ransac,
)


CONTEXT_FIELDS = (
    "policy_row_index",
    "query_id",
    "track_id",
    "coarse_pose_success",
    "coarse_pose_match_selected",
    "coarse_pose_inlier",
    "coarse_pose_query_match_count",
    "coarse_pose_query_inlier_count",
    "coarse_pose_query_inlier_ratio",
    "coarse_pose_projection_in_front",
    "coarse_pose_projection_x",
    "coarse_pose_projection_y",
    "coarse_pose_offset_dx",
    "coarse_pose_offset_dy",
    "coarse_pose_reprojection_residual_px",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument(
        "--split_names",
        nargs="+",
        default=["train", "validation", "test"],
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _load_policy(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        required = (
            "query_ids",
            "query_xy",
            "selected_track_ids",
            "selected_prototype_ids",
            "selected_canonical_rows",
            "selected_assignment_scores",
            "selected_pose_selection_scores",
        )
        arrays = {key: np.asarray(payload[key]) for key in required}
    if metadata.get("format") != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported selected-policy artifact")
    count = len(arrays["query_ids"])
    if any(len(arrays[key]) != count for key in arrays):
        raise ValueError("selected-policy arrays have inconsistent lengths")
    return arrays, metadata


def _validate_policy_inputs(
    metadata: Mapping[str, object],
    *,
    bank_path: Path,
    split_path: Path,
) -> dict[str, str]:
    hashes = {
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
    }
    mismatches = {
        key: {"policy": metadata.get(key), "input": value}
        for key, value in hashes.items()
        if str(metadata.get(key, "")) != value
    }
    if mismatches:
        raise ValueError(f"stale selected-policy coarse-pose inputs: {mismatches}")
    return hashes


def _project_xyz(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for coarse-pose projection") from exc
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        points,
        rvec,
        pose[:3, 3],
        camera_matrix,
        distortion,
    )
    camera_xyz = points @ pose[:3, :3].T + pose[:3, 3][None, :]
    return projected.reshape(-1, 2), camera_xyz[:, 2] > 0.0


def _pnp_seed(query_id: str) -> None:
    try:
        import cv2

        seed = int.from_bytes(
            hashlib.sha256(str(query_id).encode("utf8")).digest()[:4],
            "little",
        )
        cv2.setRNGSeed(int(seed % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def build_selected_policy_coarse_pose_context(
    *,
    policy_artifact: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    split_json: Path,
    split_names: Sequence[str],
    output_dir: Path,
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 5000,
) -> dict[str, Any]:
    policy_path = Path(policy_artifact)
    bank_path = Path(projected_landmark_bank)
    split_path = Path(split_json)
    arrays, metadata = _load_policy(policy_path)
    bound_hashes = _validate_policy_inputs(
        metadata,
        bank_path=bank_path,
        split_path=split_path,
    )
    split = json.loads(split_path.read_text())
    selected_splits = tuple(dict.fromkeys(str(value) for value in split_names))
    query_to_split: dict[str, str] = {}
    for split_name in selected_splits:
        for query_id in split[split_name]:
            query = str(query_id)
            if query in query_to_split:
                raise ValueError(f"query appears in multiple requested splits: {query}")
            query_to_split[query] = split_name

    landmark_index, _bank_metadata = load_landmark_index_npz(bank_path)
    model_dir = Path(colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    query_xy = np.asarray(arrays["query_xy"], dtype=np.float64)
    track_ids = np.asarray(arrays["selected_track_ids"], dtype=np.int64)
    prototype_ids = np.asarray(arrays["selected_prototype_ids"], dtype=np.int64)
    canonical_rows = np.asarray(arrays["selected_canonical_rows"], dtype=np.int64)
    assignment_scores = np.asarray(
        arrays["selected_assignment_scores"], dtype=np.float64
    )
    selection_scores = np.asarray(
        arrays["selected_pose_selection_scores"], dtype=np.float64
    )
    if np.any(canonical_rows < 0) or np.any(
        canonical_rows >= len(landmark_index.track_ids)
    ):
        raise ValueError("selected policy contains an invalid landmark-bank row")

    rows_by_query: dict[str, list[int]] = {}
    for policy_row, query_id in enumerate(query_ids.tolist()):
        if query_id in query_to_split:
            rows_by_query.setdefault(query_id, []).append(int(policy_row))
    missing_queries = sorted(set(query_to_split) - set(rows_by_query))
    if missing_queries:
        raise ValueError(f"selected policy is missing split queries: {missing_queries[:5]}")

    max_matches_value = metadata.get("max_matches")
    max_matches = None if max_matches_value is None else int(max_matches_value)
    selection_mode = str(metadata.get("selection_mode") or "score_topk")
    output_rows: list[dict[str, object]] = []
    per_split: dict[str, dict[str, int]] = {
        split_name: {"query_count": 0, "success_count": 0, "row_count": 0}
        for split_name in selected_splits
    }
    query_reports: list[dict[str, object]] = []
    for query_id, policy_rows in rows_by_query.items():
        split_name = query_to_split[query_id]
        image = images_by_name.get(query_id)
        if image is None:
            raise ValueError(f"COLMAP model is missing query image: {query_id}")
        camera = cameras[int(image.camera_id)]
        matches = [
            QueryTo3DMatch(
                token_index=int(policy_row),
                xy=query_xy[policy_row],
                track_id=int(track_ids[policy_row]),
                xyz=np.asarray(
                    landmark_index.xyz[int(canonical_rows[policy_row])],
                    dtype=np.float64,
                ),
                similarity=float(selection_scores[policy_row]),
                ratio=0.0,
                landmark_variance=float(
                    landmark_index.mean_variances[
                        int(canonical_rows[policy_row])
                    ]
                ),
                source="selected_policy_coarse_pose_context",
                prototype_id=int(prototype_ids[policy_row]),
            )
            for policy_row in policy_rows
            if np.isfinite(assignment_scores[policy_row])
            and np.isfinite(selection_scores[policy_row])
        ]
        if max_matches is None:
            matches = resolve_pose_match_conflicts(matches)
        else:
            matches = select_pose_safe_matches(
                matches,
                max_matches=int(max_matches),
                image_width=int(camera.width),
                image_height=int(camera.height),
                mode=selection_mode,
            )
        matches = stable_uniform_ransac_order(matches)
        _pnp_seed(query_id)
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=float(pnp_reprojection_error_px),
            iterations=int(pnp_iterations),
            refine_method="LM",
        )
        pose_success = bool(pnp.success and pnp.pose_w2c is not None)
        selected_lookup = {
            int(match.token_index): index for index, match in enumerate(matches)
        }
        projections = None
        in_front = None
        if pose_success:
            xyz = np.asarray(
                [
                    landmark_index.xyz[int(canonical_rows[policy_row])]
                    for policy_row in policy_rows
                ],
                dtype=np.float64,
            )
            projections, in_front = _project_xyz(xyz, pnp.pose_w2c, camera)
        inlier_ratio = (
            0.0
            if int(pnp.match_count) <= 0
            else float(pnp.inlier_count / pnp.match_count)
        )
        for local_index, policy_row in enumerate(policy_rows):
            selected_index = selected_lookup.get(int(policy_row))
            projected_xy = (
                None if projections is None else projections[local_index]
            )
            projection_in_front = bool(
                in_front is not None and in_front[local_index]
            )
            offset = (
                None
                if projected_xy is None
                else projected_xy - query_xy[policy_row]
            )
            output_rows.append(
                {
                    "policy_row_index": int(policy_row),
                    "query_id": query_id,
                    "track_id": int(track_ids[policy_row]),
                    "coarse_pose_success": pose_success,
                    "coarse_pose_match_selected": selected_index is not None,
                    "coarse_pose_inlier": bool(
                        selected_index is not None
                        and pnp.inlier_mask[int(selected_index)]
                    ),
                    "coarse_pose_query_match_count": int(pnp.match_count),
                    "coarse_pose_query_inlier_count": int(pnp.inlier_count),
                    "coarse_pose_query_inlier_ratio": inlier_ratio,
                    "coarse_pose_projection_in_front": projection_in_front,
                    "coarse_pose_projection_x": (
                        "" if projected_xy is None else float(projected_xy[0])
                    ),
                    "coarse_pose_projection_y": (
                        "" if projected_xy is None else float(projected_xy[1])
                    ),
                    "coarse_pose_offset_dx": (
                        "" if offset is None else float(offset[0])
                    ),
                    "coarse_pose_offset_dy": (
                        "" if offset is None else float(offset[1])
                    ),
                    "coarse_pose_reprojection_residual_px": (
                        "" if offset is None else float(np.linalg.norm(offset))
                    ),
                }
            )
        per_split[split_name]["query_count"] += 1
        per_split[split_name]["success_count"] += int(pose_success)
        per_split[split_name]["row_count"] += len(policy_rows)
        query_reports.append(
            {
                "query_id": query_id,
                "split": split_name,
                "success": pose_success,
                "match_count": int(pnp.match_count),
                "inlier_count": int(pnp.inlier_count),
                "inlier_ratio": inlier_ratio,
            }
        )

    output_rows.sort(key=lambda row: int(row["policy_row_index"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "coarse_pose_context.csv"
    with context_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONTEXT_FIELDS))
        writer.writeheader()
        writer.writerows(output_rows)
    reports_path = output / "query_reports.json"
    reports_path.write_text(
        json.dumps(query_reports, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "stage": "selected_policy_no_gt_coarse_pose_context",
        "protocol": {
            "gt_pose_read": False,
            "policy_gt_residual_array_read": False,
            "render": False,
            "image_retrieval": False,
            "submap": False,
            "same_frozen_selected_policy_as_final_pose": True,
            "uniform_ransac_canonical_input_order": True,
        },
        "split_names": list(selected_splits),
        "row_count": int(len(output_rows)),
        "query_count": int(len(query_reports)),
        "success_count": int(sum(bool(row["success"]) for row in query_reports)),
        "per_split": per_split,
        "pnp": {
            "max_matches": max_matches,
            "selection_mode": selection_mode,
            "reprojection_error_px": float(pnp_reprojection_error_px),
            "iterations": int(pnp_iterations),
            "refine_method": "LM",
        },
        "inputs": {
            "policy_artifact": str(policy_path),
            "policy_artifact_sha256": file_sha256_short(policy_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": bound_hashes[
                "projected_landmark_bank_sha256"
            ],
            "split_json": str(split_path),
            "split_json_sha256": bound_hashes["split_json_sha256"],
            "colmap_model_dir": str(model_dir),
            "colmap_cameras_sha256": file_sha256_short(
                model_dir / "cameras.bin"
            ),
            "colmap_images_sha256": file_sha256_short(model_dir / "images.bin"),
        },
        "outputs": {
            "coarse_pose_context": str(context_path),
            "coarse_pose_context_sha256": file_sha256_short(context_path),
            "query_reports": str(reports_path),
            "query_reports_sha256": file_sha256_short(reports_path),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_selected_policy_coarse_pose_context(
        policy_artifact=Path(args.policy_artifact),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        split_json=Path(args.split_json),
        split_names=args.split_names,
        output_dir=Path(args.output_dir),
        pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
