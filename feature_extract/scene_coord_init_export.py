#!/usr/bin/env python3
"""Export scene-coordinate PnP initial poses as real-init caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import camera_params_to_intrinsics, read_colmap_cameras, read_colmap_images  # noqa: E402
from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    save_retrieval_init_entries,
)
from feature_extract.pose_init_export import _records_by_sample_name  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    TeacherFeatureStore,
    build_all_records,
    build_radio_query_student,
    resolve_query_feature_dims,
    safe_torch_load,
)


def unnormalize_scene_coord(scene_coord, *, center, scale: float) -> torch.Tensor:
    coord = torch.as_tensor(scene_coord).float()
    if coord.ndim == 3:
        coord = coord.unsqueeze(0)
    if coord.ndim != 4:
        raise ValueError(f"scene_coord must have shape [3,H,W] or [B,3,H,W], got {tuple(coord.shape)}")
    if coord.shape[1] != 3 and coord.shape[-1] == 3:
        coord = coord.permute(0, 3, 1, 2).contiguous()
    if coord.shape[1] != 3:
        raise ValueError(f"scene_coord channel dimension must be 3, got {tuple(coord.shape)}")
    center_t = torch.as_tensor(center, dtype=coord.dtype, device=coord.device).view(1, 3, 1, 1)
    return coord * float(scale) + center_t


def _resize_intrinsics(intrinsics: Dict[str, float], from_hw: Tuple[int, int], to_hw: Tuple[int, int]) -> Dict[str, float]:
    sy = float(to_hw[0]) / float(from_hw[0])
    sx = float(to_hw[1]) / float(from_hw[1])
    return {
        "fx": float(intrinsics["fx"]) * sx,
        "fy": float(intrinsics["fy"]) * sy,
        "cx": float(intrinsics["cx"]) * sx,
        "cy": float(intrinsics["cy"]) * sy,
    }


def _select_scene_coord_correspondences(
    world_map: torch.Tensor,
    confidence: torch.Tensor | None,
    *,
    max_points: int,
    stride: int,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    if world_map.ndim != 3 or world_map.shape[0] != 3:
        raise ValueError(f"world_map must have shape [3,H,W], got {tuple(world_map.shape)}")
    H, W = world_map.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    valid = torch.isfinite(world_map).all(dim=0)
    if stride > 1:
        grid_mask = ((yy % int(stride)) == 0) & ((xx % int(stride)) == 0)
        valid = valid & grid_mask
    if confidence is not None:
        conf = confidence.float()
        if conf.ndim == 3:
            conf = conf.squeeze(0)
        if conf.shape != (H, W):
            conf = F.interpolate(conf.view(1, 1, *conf.shape[-2:]), size=(H, W), mode="bilinear", align_corners=False)[
                0,
                0,
            ]
        valid = valid & (conf >= float(min_confidence))
        scores = conf[valid]
    else:
        scores = torch.ones(int(valid.sum().item()), dtype=torch.float32)
    if int(valid.sum().item()) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    ys = yy[valid].float()
    xs = xx[valid].float()
    pts = world_map.permute(1, 2, 0)[valid].float()
    if pts.shape[0] > int(max_points):
        order = torch.argsort(scores, descending=True)[: int(max_points)]
        pts = pts[order]
        xs = xs[order]
        ys = ys[order]
    pts_2d = torch.stack([xs, ys], dim=1)
    return pts.cpu().numpy().astype(np.float32), pts_2d.cpu().numpy().astype(np.float32)


def scene_coord_pnp_from_prediction(
    scene_coord,
    *,
    center,
    scale: float,
    intrinsics: Dict[str, float],
    confidence=None,
    image_hw: tuple[int, int] | None = None,
    reproj_threshold: float = 4.0,
    n_iters: int = 5000,
    min_inliers: int = 32,
    max_points: int = 2048,
    stride: int = 1,
    min_confidence: float = 0.0,
    use_magsac: bool = True,
):
    world = unnormalize_scene_coord(scene_coord, center=center, scale=scale)[0]
    H, W = world.shape[-2:]
    intr = dict(intrinsics)
    if image_hw is not None and tuple(image_hw) != (H, W):
        intr = _resize_intrinsics(intr, tuple(image_hw), (H, W))
    pts_3d, pts_2d = _select_scene_coord_correspondences(
        world,
        None if confidence is None else torch.as_tensor(confidence).float(),
        max_points=max_points,
        stride=max(1, int(stride)),
        min_confidence=float(min_confidence),
    )
    info = {
        "success": False,
        "num_points": int(len(pts_3d)),
        "num_inliers": 0,
        "failure_reason": "",
    }
    if len(pts_3d) < int(min_inliers):
        info["failure_reason"] = "too_few_correspondences"
        return None, info

    camera_matrix = np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    flags = cv2.SOLVEPNP_EPNP
    try:
        if use_magsac and hasattr(cv2, "USAC_MAGSAC"):
            params = cv2.UsacParams()
            params.confidence = 0.999
            params.maxIterations = int(n_iters)
            params.threshold = float(reproj_threshold)
            ret = cv2.solvePnPRansac(pts_3d.astype(np.float64), pts_2d.astype(np.float64), camera_matrix, None, params=params)
            if len(ret) == 5:
                success, _camera_matrix_out, rvec, tvec, inliers = ret
            else:
                success, rvec, tvec, inliers = ret
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d.astype(np.float64),
                pts_2d.astype(np.float64),
                camera_matrix,
                None,
                iterationsCount=int(n_iters),
                reprojectionError=float(reproj_threshold),
                confidence=0.999,
                flags=flags,
            )
    except cv2.error as exc:
        info["failure_reason"] = f"opencv_error:{exc}"
        return None, info
    if not success or inliers is None or len(inliers) < int(min_inliers):
        info["num_inliers"] = 0 if inliers is None else int(len(inliers))
        info["failure_reason"] = "pnp_failed"
        return None, info
    if len(inliers) >= 6:
        try:
            ok, rvec_refined, tvec_refined = cv2.solvePnP(
                pts_3d[inliers.flatten()].astype(np.float64),
                pts_2d[inliers.flatten()].astype(np.float64),
                camera_matrix,
                None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                rvec, tvec = rvec_refined, tvec_refined
        except cv2.error:
            pass
    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.astype(np.float32)
    pose[:3, 3] = tvec.reshape(3).astype(np.float32)
    info["success"] = True
    info["num_inliers"] = int(len(inliers))
    return pose, info


def build_scene_coord_init_entries(
    *,
    query_samples: Sequence[Dict],
    pose_predictions: Sequence[np.ndarray | None],
    scores: Sequence[float],
    inlier_counts: Sequence[int],
    source_name: str,
    save_path: str | None = None,
):
    if not (len(query_samples) == len(pose_predictions) == len(scores) == len(inlier_counts)):
        raise ValueError("query_samples, pose_predictions, scores, and inlier_counts must have the same length")
    entries = []
    success = 0
    for sample, pose, score, inliers in zip(query_samples, pose_predictions, scores, inlier_counts):
        valid = pose is not None
        if valid:
            success += 1
            pose_arr = np.asarray(pose, dtype=np.float32)
        else:
            pose_arr = np.asarray(sample.get("pose_w2c", np.eye(4)), dtype=np.float32)
        entries.append(
            {
                "query_img_id": int(sample["img_id"]),
                "query_image_name": sample["image_name"],
                "query_image_stem": sample["image_stem"],
                "pose_init": pose_arr,
                "init_source": source_name if valid else f"{source_name}_failed_gt_fallback",
                "retrieval_frame_id": -1,
                "retrieval_image_name": "",
                "retrieval_score": float(score),
                "pose_init_candidates": pose_arr.reshape(1, 4, 4).astype(np.float32),
                "candidate_valid_mask": np.array([bool(valid)], dtype=bool),
                "retrieval_frame_ids_candidates": np.full((1,), -1, dtype=np.int64),
                "retrieval_image_names_candidates": np.array([""]),
                "retrieval_scores_candidates": np.array([float(score)], dtype=np.float32),
                "pnp_inliers": int(inliers),
            }
        )
    stats = {
        "method_requested": "scene_coord_pnp",
        "method_used": source_name,
        "retrieval_topk_requested": 1,
        "num_query_samples": int(len(entries)),
        "num_success": int(success),
        "success_rate": float(success / max(1, len(entries))),
        "counts_by_source": {source_name: int(success), f"{source_name}_failed_gt_fallback": int(len(entries) - success)},
    }
    if save_path:
        save_retrieval_init_entries(entries, stats, save_path)
    return entries, stats


def _load_rgb(path: str, input_hw: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if tuple(reversed(input_hw)) != img.size:
            img = img.resize((input_hw[1], input_hw[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _scene_center_scale_from_cfg(cfg):
    map_cfg = cfg.get("map_supervision", {})
    center = map_cfg.get("scene_coord_center", [0.0, 0.0, 0.0])
    scale = float(map_cfg.get("scene_coord_scale", 20.0))
    return torch.tensor(center, dtype=torch.float32), scale


@torch.no_grad()
def export_scene_coord_pnp_init(
    *,
    config_path: str,
    checkpoint_path: str,
    colmap_dir: str,
    query_split: str,
    save_path: str,
    batch_size: int = 1,
    device: str = "cuda",
    source_name: str | None = None,
    reproj_threshold: float = 4.0,
    n_iters: int = 5000,
    min_inliers: int = 32,
    max_points: int = 2048,
    stride: int = 2,
):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not bool(cfg.get("model", {}).get("scene_coord_head", False)):
        raise ValueError("scene-coordinate PnP export requires model.scene_coord_head=true")
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    model = build_radio_query_student(cfg, fine_feature_dim=fine_dim, coarse_feature_dim=coarse_dim).to(device_obj)
    checkpoint = safe_torch_load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    cameras = read_colmap_cameras(str(Path(colmap_dir) / "cameras.bin"))
    colmap_images = read_colmap_images(str(Path(colmap_dir) / "images.bin"))
    image_name_to_camera_id = {
        meta.name.replace("\\", "/"): meta.camera_id
        for meta in colmap_images.values()
    }
    samples = list_colmap_split_samples(colmap_dir, query_split)
    all_records = build_all_records(cfg["dataset"], teacher_store, allow_synthetic=False)
    record_by_name = _records_by_sample_name(all_records)
    input_hw = tuple(cfg["dataset"]["input_hw"])
    center, scale = _scene_center_scale_from_cfg(cfg)
    source = source_name or f"scene_coord_pnp_{Path(checkpoint_path).parent.parent.name}"

    poses = []
    scores = []
    inliers_all = []
    used_samples = []
    for start in range(0, len(samples), int(batch_size)):
        chunk = samples[start : start + int(batch_size)]
        rgbs = []
        kept = []
        intrs = []
        for sample in chunk:
            record = record_by_name.get(sample["image_name"]) or record_by_name.get(Path(sample["image_name"]).name)
            if record is None or record.get("image_path") is None:
                continue
            rgbs.append(_load_rgb(record["image_path"], input_hw))
            kept.append(sample)
            camera_id = image_name_to_camera_id.get(sample["image_name"].replace("\\", "/"))
            cam = cameras[camera_id] if camera_id in cameras else next(iter(cameras.values()))
            intrs.append(camera_params_to_intrinsics(cam, target_hw=input_hw))
        if not rgbs:
            continue
        outputs = model(torch.stack(rgbs, dim=0).to(device_obj))
        if "scene_coord" not in outputs:
            raise RuntimeError("Model did not return scene_coord; set model.scene_coord_head=true")
        scene = outputs["scene_coord"].detach().cpu()
        for idx, sample in enumerate(kept):
            pose, info = scene_coord_pnp_from_prediction(
                scene[idx],
                center=center,
                scale=scale,
                intrinsics=intrs[idx],
                image_hw=input_hw,
                reproj_threshold=reproj_threshold,
                n_iters=n_iters,
                min_inliers=min_inliers,
                max_points=max_points,
                stride=stride,
            )
            poses.append(pose)
            inliers = int(info.get("num_inliers", 0))
            inliers_all.append(inliers)
            scores.append(float(inliers))
            used_samples.append(sample)
    if not used_samples:
        raise RuntimeError("No scene-coordinate PnP predictions were exported; check split/image paths")
    _entries, stats = build_scene_coord_init_entries(
        query_samples=used_samples,
        pose_predictions=poses,
        scores=scores,
        inlier_counts=inliers_all,
        source_name=source,
        save_path=save_path,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export query-student scene-coordinate PnP init cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source_name", default=None)
    parser.add_argument("--reproj_threshold", type=float, default=4.0)
    parser.add_argument("--n_iters", type=int, default=5000)
    parser.add_argument("--min_inliers", type=int, default=32)
    parser.add_argument("--max_points", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_scene_coord_pnp_init(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        save_path=args.save_path,
        batch_size=args.batch_size,
        device=args.device,
        source_name=args.source_name,
        reproj_threshold=args.reproj_threshold,
        n_iters=args.n_iters,
        min_inliers=args.min_inliers,
        max_points=args.max_points,
        stride=args.stride,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
