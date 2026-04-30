#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import (
    add_pose_noise,
    camera_params_to_intrinsics,
    colmap_to_w2c,
    read_colmap_cameras,
    read_colmap_images,
)
from feature_field import build_dcff_runtime
from pose_refine.evaluate_pipeline import _render_rgbd_for_loftr
from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution


def camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    return -(R.T @ t)


def pose_error(pose_pred: np.ndarray, pose_gt: np.ndarray) -> tuple[float, float]:
    R_rel = pose_pred[:3, :3].T @ pose_gt[:3, :3]
    cos_angle = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    rot = math.degrees(math.acos(cos_angle))
    trans = np.linalg.norm(camera_center(pose_pred) - camera_center(pose_gt)) * 1000.0
    return rot, trans


def load_split_names(split_file: str | None) -> set[str] | None:
    if not split_file or not os.path.isfile(split_file):
        return None
    names: set[str] = set()
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            name = line.split()[0]
            stem = os.path.splitext(name)[0]
            names.add(stem + ".png")
            names.add(stem + ".jpg")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Render-at-init LoFTR+PnP local refinement diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--noise_deg", type=float, default=1.0)
    parser.add_argument("--noise_m", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--loftr_long_edge", type=int, default=840)
    parser.add_argument("--loftr_conf", type=float, default=0.3)
    parser.add_argument("--reproj_threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--use_magsac", action="store_true")
    parser.add_argument(
        "--ref_mode",
        choices=["render", "query"],
        default="render",
        help="'render' matches query against RGB rendered at noisy init; "
             "'query' is a same-image PnP sanity check using GT-pose depth.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ds_cfg = config["dataset"]

    runtime = build_dcff_runtime(config, device, printer=print)
    cameras = read_colmap_cameras(str(Path(ds_cfg["colmap_dir"]) / "cameras.bin"))
    images = read_colmap_images(str(Path(ds_cfg["colmap_dir"]) / "images.bin"))
    first_cam = next(iter(cameras.values()))
    intrinsics = camera_params_to_intrinsics(first_cam)
    orig_hw = (int(first_cam.height), int(first_cam.width))
    loftr_hw = _compute_loftr_resolution(orig_hw, args.loftr_long_edge)

    split_names = load_split_names(ds_cfg.get("test_split"))
    source_dir = Path(ds_cfg["source_dir"])
    samples = []
    for img_id in sorted(images.keys()):
        meta = images[img_id]
        if split_names is not None and meta.name not in split_names:
            continue
        query_path = source_dir / meta.name
        if not query_path.is_file():
            continue
        pose_gt = colmap_to_w2c(meta.qvec, meta.tvec).astype(np.float32)
        samples.append((meta.name, query_path, pose_gt))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    loftr = LoFTRInitializer(
        device=device,
        pretrained="outdoor",
        loftr_long_edge=args.loftr_long_edge,
        confidence_threshold=args.loftr_conf,
        reproj_threshold=args.reproj_threshold,
        use_magsac=args.use_magsac,
    )

    init_rot, init_trans = [], []
    final_rot, final_trans = [], []
    inliers, raw_matches, failures = [], [], []

    print(
        f"Render-LoFTR local refine: samples={len(samples)} "
        f"noise={args.noise_deg}deg/{args.noise_m}m "
        f"loftr_hw={loftr_hw} conf={args.loftr_conf} reproj={args.reproj_threshold}px"
    )

    for name, query_path, pose_gt in tqdm(samples, desc="render-loftr", leave=False):
        pose_init = add_pose_noise(pose_gt, args.noise_deg, args.noise_m).astype(np.float32)
        r0, t0 = pose_error(pose_init, pose_gt)
        init_rot.append(r0)
        init_trans.append(t0)

        render_pose = pose_gt if args.ref_mode == "query" else pose_init
        rendered_rgb, rendered_depth = _render_rgbd_for_loftr(
            runtime.gaussians,
            render_pose,
            intrinsics,
            loftr_hw,
            orig_hw,
            device,
        )
        query_bgr = cv2.imread(str(query_path))
        if query_bgr is None:
            failures.append(f"{name}: cannot_read_query")
            continue
        query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
        ref_rgb = query_rgb if args.ref_mode == "query" else rendered_rgb
        ref_pose = pose_gt if args.ref_mode == "query" else pose_init

        result = loftr.estimate_pose(
            query_rgb,
            ref_rgb,
            rendered_depth,
            ref_pose,
            intrinsics,
            orig_hw=orig_hw,
        )
        raw_matches.append(result.num_raw_matches)
        inliers.append(result.num_inliers)
        if not result.success or result.pose_w2c is None:
            failures.append(f"{name}: {result.failure_reason}")
            continue
        r1, t1 = pose_error(result.pose_w2c, pose_gt)
        final_rot.append(r1)
        final_trans.append(t1)

    print(
        f"init med={np.median(init_rot):.3f}deg/{np.median(init_trans):.1f}mm "
        f"mean={np.mean(init_rot):.3f}deg/{np.mean(init_trans):.1f}mm"
    )
    if final_trans:
        print(
            f"final med={np.median(final_rot):.3f}deg/{np.median(final_trans):.1f}mm "
            f"mean={np.mean(final_rot):.3f}deg/{np.mean(final_trans):.1f}mm "
            f"success={len(final_trans)}/{len(samples)} "
            f"inliers_med={np.median(inliers):.1f} raw_med={np.median(raw_matches):.1f}"
        )
        ft = np.asarray(final_trans)
        fr = np.asarray(final_rot)
        print(
            f"joint@0.1deg/10mm={np.mean((fr < 0.1) & (ft < 10.0)) * 100:.1f}% "
            f"joint@1deg/50mm={np.mean((fr < 1.0) & (ft < 50.0)) * 100:.1f}%"
        )
    else:
        print(f"final: no successful poses  failures={len(failures)}/{len(samples)}")
    if failures:
        print("first failures:")
        for msg in failures[:5]:
            print(f"  {msg}")


if __name__ == "__main__":
    main()
