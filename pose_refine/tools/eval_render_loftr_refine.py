#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from feature_extract.tools.export_gt_render_loftr_correspondences import payload_from_loftr_result
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


def resolve_split_file(ds_cfg: dict, split_key: str) -> str | None:
    if split_key not in ds_cfg:
        raise KeyError(f"dataset config does not contain split key: {split_key}")
    return ds_cfg.get(split_key)


def cache_stem_for_image_name(image_name: str) -> str:
    path = Path(str(image_name))
    if path.parent == Path(".") or not path.parent.name:
        return path.stem
    return f"{path.parent.name}_{path.stem}"


def samples_from_init_cache(
    base_samples: list[dict],
    cache_data,
    *,
    max_samples: int = 0,
) -> list[dict]:
    by_stem = {cache_stem_for_image_name(sample["name"]): sample for sample in base_samples}
    stems = list(cache_data["query_image_stems"])
    pose_inits = np.asarray(cache_data["pose_inits"], dtype=np.float32)
    init_sources = cache_data["init_sources"] if "init_sources" in cache_data else None
    samples: list[dict] = []
    limit = int(max_samples)
    for idx, stem in enumerate(stems):
        if limit > 0 and len(samples) >= limit:
            break
        key = str(stem)
        base = by_stem.get(key)
        if base is None:
            continue
        sample = dict(base)
        sample["query_image_stem"] = key
        sample["pose_init"] = pose_inits[idx].astype(np.float32)
        sample["init_source"] = str(init_sources[idx]) if init_sources is not None else "init_cache"
        samples.append(sample)
    return samples


def _is_successful_loftr_result(result) -> bool:
    return bool(getattr(result, "success", False)) and getattr(result, "pose_w2c", None) is not None


def record_from_iterative_refinement_results(
    *,
    query_image_name: str,
    query_image_stem: str,
    pose_init: np.ndarray,
    results: list,
    requested_iterations: int,
) -> dict:
    iteration_success = []
    iteration_num_inliers = []
    iteration_num_raw_matches = []
    last_success_result = None
    last_success_iteration = 0
    failure_reason = ""

    for iter_idx, result in enumerate(results, start=1):
        success = _is_successful_loftr_result(result)
        iteration_success.append(success)
        iteration_num_inliers.append(int(getattr(result, "num_inliers", 0) or 0))
        iteration_num_raw_matches.append(int(getattr(result, "num_raw_matches", 0) or 0))
        if success:
            last_success_result = result
            last_success_iteration = iter_idx
        else:
            failure_reason = str(getattr(result, "failure_reason", "") or "failed")

    if last_success_result is not None:
        pose_refined = np.asarray(last_success_result.pose_w2c, dtype=np.float32)
        num_inliers = int(getattr(last_success_result, "num_inliers", 0) or 0)
        num_raw_matches = int(getattr(last_success_result, "num_raw_matches", 0) or 0)
        success = True
    else:
        last_result = results[-1] if results else None
        pose_refined = None
        num_inliers = int(getattr(last_result, "num_inliers", 0) or 0) if last_result is not None else 0
        num_raw_matches = int(getattr(last_result, "num_raw_matches", 0) or 0) if last_result is not None else 0
        if not failure_reason:
            failure_reason = "no_iterations_attempted" if not results else "failed"
        success = False

    return {
        "query_image_name": query_image_name,
        "query_image_stem": query_image_stem,
        "pose_init": np.asarray(pose_init, dtype=np.float32),
        "pose_refined": pose_refined,
        "success": success,
        "num_inliers": num_inliers,
        "num_raw_matches": num_raw_matches,
        "failure_reason": failure_reason,
        "refine_iterations_requested": int(requested_iterations),
        "refine_attempted_iterations": len(results),
        "refine_successful_iterations": int(np.sum(iteration_success)),
        "refine_last_success_iteration": int(last_success_iteration),
        "iteration_success": iteration_success,
        "iteration_num_inliers": iteration_num_inliers,
        "iteration_num_raw_matches": iteration_num_raw_matches,
    }


def pose_cache_payload_from_records(records: list[dict], *, source: str) -> dict[str, np.ndarray]:
    def _default_attempted_iterations(record: dict) -> int:
        return 1 if int(record.get("num_raw_matches", 0)) > 0 else 0

    def _default_success_iterations(record: dict) -> int:
        return 1 if bool(record.get("success", False)) else 0

    pose_inits = []
    init_sources = []
    for record in records:
        success = bool(record.get("success", False))
        refined = record.get("pose_refined")
        if success and refined is not None:
            pose_inits.append(np.asarray(refined, dtype=np.float32))
            init_sources.append(str(source))
        else:
            pose_inits.append(np.asarray(record["pose_init"], dtype=np.float32))
            init_sources.append(f"{source}_failed")
    pose_inits_arr = np.stack(pose_inits).astype(np.float32)
    query_image_names = np.asarray([record["query_image_name"] for record in records])
    query_image_stems = np.asarray([record["query_image_stem"] for record in records])
    query_img_ids = np.asarray(
        [int(record.get("query_img_id", idx)) for idx, record in enumerate(records)],
        dtype=np.int64,
    )
    retrieval_frame_ids = np.asarray(
        [int(record.get("retrieval_frame_id", -1)) for record in records],
        dtype=np.int64,
    )
    retrieval_image_names = np.asarray(
        [str(record.get("retrieval_image_name", record["query_image_name"])) for record in records]
    )
    retrieval_scores = np.asarray(
        [float(record.get("retrieval_score", 0.0)) for record in records],
        dtype=np.float32,
    )
    return {
        "query_img_ids": query_img_ids,
        "query_image_names": query_image_names,
        "query_image_stems": query_image_stems,
        "pose_inits": pose_inits_arr,
        "init_sources": np.asarray(init_sources),
        "retrieval_frame_ids": retrieval_frame_ids,
        "retrieval_image_names": retrieval_image_names,
        "retrieval_scores": retrieval_scores,
        "pose_init_candidates": pose_inits_arr[:, None],
        "candidate_valid_mask": np.ones((len(records), 1), dtype=bool),
        "retrieval_frame_ids_candidates": retrieval_frame_ids[:, None],
        "retrieval_image_names_candidates": retrieval_image_names[:, None],
        "retrieval_scores_candidates": retrieval_scores[:, None],
        "refine_success": np.asarray([bool(record.get("success", False)) for record in records], dtype=bool),
        "refine_num_inliers": np.asarray([int(record.get("num_inliers", 0)) for record in records], dtype=np.int32),
        "refine_num_raw_matches": np.asarray([int(record.get("num_raw_matches", 0)) for record in records], dtype=np.int32),
        "refine_iterations_requested": np.asarray(
            [int(record.get("refine_iterations_requested", 1)) for record in records],
            dtype=np.int32,
        ),
        "refine_attempted_iterations": np.asarray(
            [int(record.get("refine_attempted_iterations", _default_attempted_iterations(record))) for record in records],
            dtype=np.int32,
        ),
        "refine_successful_iterations": np.asarray(
            [int(record.get("refine_successful_iterations", _default_success_iterations(record))) for record in records],
            dtype=np.int32,
        ),
        "refine_last_success_iteration": np.asarray(
            [int(record.get("refine_last_success_iteration", _default_success_iterations(record))) for record in records],
            dtype=np.int32,
        ),
    }


def maybe_write_teacher_correspondence(
    output_dir: str | Path | None,
    image_name: str,
    result,
    *,
    max_points: int = 512,
    inlier_only: bool = True,
) -> Path | None:
    if output_dir is None:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cache_stem_for_image_name(image_name)}.npz"
    payload = payload_from_loftr_result(
        result,
        max_points=int(max_points),
        inlier_only=bool(inlier_only),
        source="render_init_loftr",
    )
    np.savez_compressed(out_path, **payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render-at-init LoFTR+PnP local refinement diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init_cache", default=None, help="Optional real-init pose cache .npz")
    parser.add_argument("--output_json", default=None, help="Optional path for summary JSON")
    parser.add_argument("--output_pose_cache", default=None, help="Optional .npz cache with refined poses")
    parser.add_argument("--output_corr_dir", default=None, help="Optional directory for TeacherCorrespondenceStore .npz files")
    parser.add_argument(
        "--split_key",
        choices=("test_split", "train_split"),
        default="test_split",
        help="Dataset split key used to build the base sample list before matching an init cache.",
    )
    parser.add_argument("--corr_max_points", type=int, default=512)
    parser.add_argument("--corr_all_matches", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--noise_deg", type=float, default=1.0)
    parser.add_argument("--noise_m", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--loftr_long_edge", type=int, default=840)
    parser.add_argument("--loftr_conf", type=float, default=0.3)
    parser.add_argument("--reproj_threshold", type=float, default=4.0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of render-match-PnP refinement iterations. Default keeps the historical one-pass behavior.",
    )
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
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

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

    split_names = load_split_names(resolve_split_file(ds_cfg, args.split_key))
    source_dir = Path(ds_cfg["source_dir"])
    base_samples = []
    for img_id in sorted(images.keys()):
        meta = images[img_id]
        if split_names is not None and meta.name not in split_names:
            continue
        query_path = source_dir / meta.name
        if not query_path.is_file():
            continue
        pose_gt = colmap_to_w2c(meta.qvec, meta.tvec).astype(np.float32)
        base_samples.append({
            "name": meta.name,
            "query_image_stem": cache_stem_for_image_name(meta.name),
            "query_path": query_path,
            "pose_gt": pose_gt,
        })
    if args.init_cache:
        cache = np.load(args.init_cache, allow_pickle=True)
        samples = samples_from_init_cache(base_samples, cache, max_samples=args.max_samples)
    else:
        samples = base_samples
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
    success_init_trans = []
    records = []
    inliers, raw_matches, failures = [], [], []
    partial_failures = []
    corr_records = 0

    print(
        f"Render-LoFTR local refine: samples={len(samples)} "
        f"mode={'init_cache' if args.init_cache else 'synthetic_noise'} "
        f"noise={args.noise_deg}deg/{args.noise_m}m "
        f"loftr_hw={loftr_hw} conf={args.loftr_conf} reproj={args.reproj_threshold}px "
        f"iterations={args.iterations}"
    )

    for sample in tqdm(samples, desc="render-loftr", leave=False):
        name = sample["name"]
        query_path = sample["query_path"]
        pose_gt = sample["pose_gt"]
        if "pose_init" in sample:
            pose_init = np.asarray(sample["pose_init"], dtype=np.float32)
        else:
            pose_init = add_pose_noise(pose_gt, args.noise_deg, args.noise_m).astype(np.float32)
        r0, t0 = pose_error(pose_init, pose_gt)
        init_rot.append(r0)
        init_trans.append(t0)

        query_image_stem = sample.get("query_image_stem", cache_stem_for_image_name(name))
        query_bgr = cv2.imread(str(query_path))
        if query_bgr is None:
            failures.append(f"{name}: cannot_read_query")
            records.append({
                "query_image_name": name,
                "query_image_stem": query_image_stem,
                "pose_init": pose_init,
                "pose_refined": None,
                "success": False,
                "num_inliers": 0,
                "num_raw_matches": 0,
                "failure_reason": "cannot_read_query",
                "refine_iterations_requested": int(args.iterations),
                "refine_attempted_iterations": 0,
                "refine_successful_iterations": 0,
                "refine_last_success_iteration": 0,
            })
            continue
        query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)

        current_pose = pose_init
        iteration_results = []
        for _iter_idx in range(int(args.iterations)):
            render_pose = pose_gt if args.ref_mode == "query" else current_pose
            rendered_rgb, rendered_depth = _render_rgbd_for_loftr(
                runtime.gaussians,
                render_pose,
                intrinsics,
                loftr_hw,
                orig_hw,
                device,
            )
            ref_rgb = query_rgb if args.ref_mode == "query" else rendered_rgb
            ref_pose = pose_gt if args.ref_mode == "query" else current_pose

            result = loftr.estimate_pose(
                query_rgb,
                ref_rgb,
                rendered_depth,
                ref_pose,
                intrinsics,
                orig_hw=orig_hw,
            )
            iteration_results.append(result)
            if not _is_successful_loftr_result(result):
                break
            current_pose = np.asarray(result.pose_w2c, dtype=np.float32)

        record = record_from_iterative_refinement_results(
            query_image_name=name,
            query_image_stem=query_image_stem,
            pose_init=pose_init,
            results=iteration_results,
            requested_iterations=int(args.iterations),
        )
        raw_matches.append(record["num_raw_matches"])
        inliers.append(record["num_inliers"])
        if args.output_corr_dir and iteration_results:
            successful_results = [result for result in iteration_results if _is_successful_loftr_result(result)]
            corr_result = successful_results[-1] if successful_results else iteration_results[-1]
            maybe_write_teacher_correspondence(
                args.output_corr_dir,
                name,
                corr_result,
                max_points=int(args.corr_max_points),
                inlier_only=not bool(args.corr_all_matches),
            )
            corr_records += 1

        records.append(record)
        if not record["success"]:
            failures.append(f"{name}: {record.get('failure_reason', 'failed')}")
            continue
        if record.get("failure_reason"):
            partial_failures.append(f"{name}: {record['failure_reason']}")
        r1, t1 = pose_error(record["pose_refined"], pose_gt)
        final_rot.append(r1)
        final_trans.append(t1)
        success_init_trans.append(t0)

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
    if partial_failures:
        print("first partial iteration failures:")
        for msg in partial_failures[:5]:
            print(f"  {msg}")
    if args.output_json:
        successful_iterations = [
            int(record.get("refine_successful_iterations", 0))
            for record in records
            if bool(record.get("success", False))
        ]
        summary = {
            "config": args.config,
            "init_cache": args.init_cache,
            "mode": "init_cache" if args.init_cache else "synthetic_noise",
            "samples": len(samples),
            "successes": len(final_trans),
            "failures": len(failures),
            "partial_iteration_failures": len(partial_failures),
            "loftr_long_edge": args.loftr_long_edge,
            "loftr_conf": args.loftr_conf,
            "reproj_threshold": args.reproj_threshold,
            "iterations": int(args.iterations),
            "successful_iterations_mean": float(np.mean(successful_iterations)) if successful_iterations else None,
            "split_key": args.split_key,
            "output_corr_dir": args.output_corr_dir,
            "corr_records": corr_records,
            "init_rot_median_deg": float(np.median(init_rot)) if init_rot else None,
            "init_trans_median_mm": float(np.median(init_trans)) if init_trans else None,
            "final_rot_median_deg": float(np.median(final_rot)) if final_rot else None,
            "final_trans_median_mm": float(np.median(final_trans)) if final_trans else None,
            "final_trans_mean_mm": float(np.mean(final_trans)) if final_trans else None,
            "improved_trans_count": int(np.sum(np.asarray(final_trans) < np.asarray(success_init_trans)))
            if final_trans
            else 0,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.output_pose_cache:
        output_path = Path(args.output_pose_cache)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = pose_cache_payload_from_records(records, source="render_loftr_refine")
        np.savez_compressed(output_path, **payload)


if __name__ == "__main__":
    main()
