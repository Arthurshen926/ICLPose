"""Build ULF-Loc-style keypoint-consensus Gaussian VFM landmarks.

This is an approximate local implementation of ULF-Loc's landmark sampling:
reference-image keypoints vote for projected Gaussian centers, then the fused
Gaussian VFM field is sampled with the vote count as an additional reliability
term. It does not require changing the canonical SfM sparse pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.semidense_anchor_map import (
    GaussianConsensusAnchorConfig,
    build_gaussian_consensus_anchor_map,
)


def _read_gray(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for keypoint-consensus Gaussian sampling") from exc
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image


def _detect_keypoints(image_gray: np.ndarray, detector: str, max_keypoints: int) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for keypoint-consensus Gaussian sampling") from exc
    mode = str(detector).lower()
    if mode == "sift" and hasattr(cv2, "SIFT_create"):
        extractor = cv2.SIFT_create(nfeatures=int(max_keypoints))
    elif mode == "orb":
        extractor = cv2.ORB_create(nfeatures=int(max_keypoints))
    else:
        raise ValueError(f"unsupported detector or unavailable OpenCV feature extractor: {detector}")
    keypoints = extractor.detect(image_gray, None)
    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32)
    keypoints = sorted(keypoints, key=lambda item: float(item.response), reverse=True)[: int(max_keypoints)]
    return np.asarray([kp.pt for kp in keypoints], dtype=np.float32)


def _project_points(xyz: np.ndarray, pose_w2c: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    depth = camera_points[:, 2]
    valid = depth > 1e-8
    safe_depth = np.where(valid, depth, 1.0)
    x = camera_points[:, 0] / safe_depth
    y = camera_points[:, 1] / safe_depth
    if int(camera.model_id) == 0:
        f, cx, cy = camera.params[:3]
        u = float(f) * x + float(cx)
        v = float(f) * y + float(cy)
    elif int(camera.model_id) == 1:
        fx, fy, cx, cy = camera.params[:4]
        u = float(fx) * x + float(cx)
        v = float(fy) * y + float(cy)
    elif int(camera.model_id) == 2:
        f, cx, cy, k = camera.params[:4]
        radial = 1.0 + float(k) * (x * x + y * y)
        u = float(f) * x * radial + float(cx)
        v = float(f) * y * radial + float(cy)
    else:
        raise ValueError(f"unsupported camera model id: {camera.model_id}")
    xy = np.stack([u, v], axis=1).astype(np.float32)
    valid &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    valid &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy, valid


def _select_pose_records(train_pose_file: Path, image_root: Path, max_views: int) -> list:
    records = [record for record in parse_cambridge_pose_file(Path(train_pose_file)) if (image_root / record.image_id).exists()]
    if int(max_views) <= 0 or len(records) <= int(max_views):
        return records
    indices = np.unique(np.linspace(0, len(records) - 1, num=int(max_views), dtype=np.int64))
    return [records[int(idx)] for idx in indices.tolist()]


def compute_keypoint_votes(
    field: GaussianVFMField,
    *,
    train_pose_file: Path,
    image_root: Path,
    camera,
    detector: str,
    max_keypoints: int,
    vote_radius_px: float,
    max_views: int,
) -> tuple[np.ndarray, dict[str, object]]:
    records = _select_pose_records(train_pose_file, image_root, max_views)
    votes = np.zeros((len(field),), dtype=np.int64)
    view_rows = []
    for record in records:
        image = _read_gray(Path(image_root) / record.image_id)
        if image is None:
            view_rows.append({"image_id": record.image_id, "loaded": False, "keypoints": 0, "projected_gaussians": 0})
            continue
        keypoints = _detect_keypoints(image, detector=detector, max_keypoints=max_keypoints)
        if keypoints.shape[0] == 0:
            view_rows.append({"image_id": record.image_id, "loaded": True, "keypoints": 0, "projected_gaussians": 0})
            continue
        xy, valid = _project_points(field.xyz, record.pose_w2c, camera)
        projected_rows = np.flatnonzero(valid)
        if projected_rows.size == 0:
            view_rows.append(
                {"image_id": record.image_id, "loaded": True, "keypoints": int(keypoints.shape[0]), "projected_gaussians": 0}
            )
            continue
        tree = cKDTree(keypoints.astype(np.float64, copy=False))
        distances, _indices = tree.query(xy[projected_rows].astype(np.float64, copy=False), k=1)
        hit = np.asarray(distances <= float(vote_radius_px), dtype=bool)
        votes[projected_rows[hit]] += 1
        view_rows.append(
            {
                "image_id": record.image_id,
                "loaded": True,
                "keypoints": int(keypoints.shape[0]),
                "projected_gaussians": int(projected_rows.size),
                "vote_hits": int(np.sum(hit)),
            }
        )
    summary = {
        "view_count": int(len(view_rows)),
        "detector": detector,
        "max_keypoints": int(max_keypoints),
        "vote_radius_px": float(vote_radius_px),
        "mean_keypoints": None if not view_rows else float(np.mean([row.get("keypoints", 0) for row in view_rows])),
        "mean_projected_gaussians": None
        if not view_rows
        else float(np.mean([row.get("projected_gaussians", 0) for row in view_rows])),
        "mean_vote_hits": None if not view_rows else float(np.mean([row.get("vote_hits", 0) for row in view_rows])),
        "voted_gaussian_count": int(np.sum(votes > 0)),
        "max_votes": int(np.max(votes)) if votes.size else 0,
        "views": view_rows,
    }
    return votes, summary


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage H keypoint-consensus Gaussian VFM landmark anchors")
    parser.add_argument("--gaussian_field", required=True)
    parser.add_argument("--train_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--votes_npz", default="")
    parser.add_argument("--detector", default="sift", choices=("sift", "orb"))
    parser.add_argument("--max_keypoints", type=int, default=2048)
    parser.add_argument("--vote_radius_px", type=float, default=4.0)
    parser.add_argument("--max_views", type=int, default=80)
    parser.add_argument("--max_anchors", type=int, default=20000)
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--max_gaussian_scale", type=float, default=None)
    parser.add_argument("--max_mean_distance", type=float, default=None)
    parser.add_argument("--nms_voxel_size", type=float, default=0.03)
    parser.add_argument("--min_keypoint_votes", type=int, default=1)
    parser.add_argument("--keypoint_vote_weight", type=float, default=1.0)
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    args = parser.parse_args(argv)

    field = GaussianVFMField.load_npz(Path(args.gaussian_field))
    camera_model_dir = _infer_camera_model_dir(args.train_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    votes, vote_summary = compute_keypoint_votes(
        field,
        train_pose_file=Path(args.train_pose_file),
        image_root=Path(args.image_root),
        camera=camera,
        detector=args.detector,
        max_keypoints=int(args.max_keypoints),
        vote_radius_px=float(args.vote_radius_px),
        max_views=int(args.max_views),
    )
    config = GaussianConsensusAnchorConfig(
        max_anchors=int(args.max_anchors),
        min_support=int(args.min_support),
        min_opacity=float(args.min_opacity),
        max_gaussian_scale=args.max_gaussian_scale,
        max_mean_distance=args.max_mean_distance,
        nms_voxel_size=float(args.nms_voxel_size),
        min_keypoint_votes=int(args.min_keypoint_votes),
        keypoint_vote_weight=float(args.keypoint_vote_weight),
    )
    anchor_map = build_gaussian_consensus_anchor_map(field, config, keypoint_vote_counts=votes)
    anchor_map.save_npz(Path(args.output_npz))
    if args.votes_npz:
        Path(args.votes_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.votes_npz, keypoint_vote_counts=votes.astype(np.int64, copy=False))
    _write_summary(
        Path(args.summary_json),
        {
            "stage": "stage_h_keypoint_consensus_anchor_map",
            "camera_source": camera_source,
            "source_gaussian_count": int(len(field)),
            "anchor_count": int(len(anchor_map)),
            "feature_dim": int(anchor_map.feature_dim),
            "config": config.to_dict(),
            "vote_summary": vote_summary,
            "anchor_metadata": dict(anchor_map.metadata or {}),
        },
    )


if __name__ == "__main__":
    main()
