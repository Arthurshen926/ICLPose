"""Build MATCHA-style selector samples from query/render keypoint correspondences."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _make_keypoint_detector,
    _parse_default_camera,
    _read_rgb,
    _render_keypoint_detector_image,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    GaussianVFMRenderConfig,
    load_gaussian_rgb_source_from_ply,
    render_gaussian_vfm_feature_map,
    render_gaussian_vfm_feature_map_gsplat,
)
from feature_extract.vfm.patch_selector_training import (
    append_patch_selector_training_set_capped,
    save_patch_selector_training_set_npz,
)
from feature_extract.vfm.rendered_keypoint_matching import bilinear_sample_feature_map
from feature_extract.vfm.rendered_keypoint_selector_samples import (
    RenderedKeypointSelectorSampleConfig,
    build_rendered_keypoint_selector_samples,
    render_keypoint_reprojection_errors,
)
from feature_extract.vfm.tokens import TokenBankManifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=320)
    parser.add_argument("--render_height", type=int, default=180)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--render_depth_epsilon", type=float, default=0.02)
    parser.add_argument("--renderer", default="soft", choices=("soft", "gsplat"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel_chunk", type=int, default=32)
    parser.add_argument("--detector", default="superpoint", choices=("orb", "superpoint", "disk"))
    parser.add_argument("--max_keypoints", type=int, default=1200)
    parser.add_argument("--max_queries", type=int, default=64)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--positive_threshold_px", type=float, default=16.0)
    parser.add_argument("--negative_threshold_px", type=float, default=32.0)
    parser.add_argument("--max_positives_per_keypoint", type=int, default=1)
    parser.add_argument("--hard_negatives_per_keypoint", type=int, default=16)
    parser.add_argument("--hard_negative_pool", type=int, default=128)
    parser.add_argument("--max_samples_per_query", type=int, default=512)
    parser.add_argument("--max_total_samples", type=int, default=20000)
    parser.add_argument("--keypoint_stride_px", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _select_records(records, max_queries: int, mode: str) -> list:
    output = list(records)
    if int(max_queries) <= 0 or len(output) <= int(max_queries):
        return output
    if mode == "uniform":
        indices = np.linspace(0, len(output) - 1, int(max_queries), dtype=np.int64)
        return [output[int(idx)] for idx in indices]
    return output[: int(max_queries)]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = _select_records(manifest.records, int(args.max_queries), str(args.view_selection))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    render_camera = _scale_camera(camera, int(args.render_width), int(args.render_height))
    field = GaussianVFMField.load_npz(Path(args.field))
    rgb_source = load_gaussian_rgb_source_from_ply(Path(args.gaussian_rgb_ply))
    render_config = GaussianVFMRenderConfig(
        width=int(args.render_width),
        height=int(args.render_height),
        radius_px=float(args.render_radius_px),
        depth_epsilon=float(args.render_depth_epsilon),
        l2_normalize_pixels=True,
    )
    detector_fn = _make_keypoint_detector(str(args.detector), int(args.max_keypoints), str(args.device))
    sample_config = RenderedKeypointSelectorSampleConfig(
        positive_threshold_px=float(args.positive_threshold_px),
        negative_threshold_px=float(args.negative_threshold_px),
        max_positives_per_keypoint=int(args.max_positives_per_keypoint),
        hard_negatives_per_keypoint=int(args.hard_negatives_per_keypoint),
        hard_negative_pool=int(args.hard_negative_pool),
        max_samples=int(args.max_samples_per_query),
        keypoint_stride_px=float(args.keypoint_stride_px),
        seed=int(args.seed),
    )
    merged = None
    rows = []
    skipped = {"missing_pose": 0, "dim_mismatch": 0, "empty_samples": 0}
    for record_idx, record in enumerate(records):
        gt = gt_by_query.get(record.image_id)
        if gt is None:
            skipped["missing_pose"] += 1
            continue
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        if int(query_feature.shape[0]) != int(field.feature_dim):
            skipped["dim_mismatch"] += 1
            continue
        pose_w2c = gt.pose_w2c
        rendered = (
            render_gaussian_vfm_feature_map_gsplat(
                field,
                pose_w2c=pose_w2c,
                camera=camera,
                config=render_config,
                device=args.device,
                channel_chunk=int(args.channel_chunk),
            )
            if args.renderer == "gsplat"
            else render_gaussian_vfm_feature_map(field, pose_w2c=pose_w2c, camera=camera, config=render_config)
        )
        query_rgb = _read_rgb(Path(args.image_root) / record.image_id)
        render_rgb = _render_keypoint_detector_image(
            rendered,
            pose_w2c=pose_w2c,
            camera=camera,
            config=render_config,
            rgb_source=rgb_source,
            renderer=args.renderer,
            device=args.device,
        )
        query_xy, _query_scores = detector_fn(query_rgb)
        render_xy, _render_scores = detector_fn(render_rgb)
        query_desc, query_valid = bilinear_sample_feature_map(
            query_feature,
            query_xy,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        render_desc, render_valid = bilinear_sample_feature_map(
            rendered.feature_map,
            render_xy,
            image_width=int(args.render_width),
            image_height=int(args.render_height),
        )
        query_xy_valid = query_xy[query_valid]
        render_xy_valid = render_xy[render_valid]
        query_desc = query_desc[query_valid]
        render_desc = render_desc[render_valid]
        errors, render_depth_valid = render_keypoint_reprojection_errors(
            query_xy_valid,
            render_xy_valid,
            rendered_depth=rendered.depth,
            render_camera=render_camera,
            query_camera=camera,
            render_pose_w2c=pose_w2c,
            query_pose_w2c=pose_w2c,
            render_width=int(args.render_width),
            render_height=int(args.render_height),
        )
        errors[:, ~render_depth_valid] = np.inf
        samples = build_rendered_keypoint_selector_samples(query_desc, render_desc, errors, sample_config)
        if samples.sample_count == 0:
            skipped["empty_samples"] += 1
            continue
        merged = append_patch_selector_training_set_capped(
            merged,
            samples,
            max_samples=int(args.max_total_samples),
            seed=int(args.seed) + int(record_idx),
        )
        rows.append(
            {
                "query_id": record.image_id,
                "query_keypoints": int(query_xy.shape[0]),
                "render_keypoints": int(render_xy.shape[0]),
                "valid_query_keypoints": int(query_desc.shape[0]),
                "valid_render_keypoints": int(render_desc.shape[0]),
                "sample_count": int(samples.sample_count),
                "merged_sample_count": int(merged.sample_count),
                "visible_render_fraction": float(np.mean(rendered.visibility_mask)),
            }
        )
    if merged is None or merged.sample_count == 0:
        raise ValueError(f"no samples built; skipped={skipped}")
    save_patch_selector_training_set_npz(merged, Path(args.output))
    summary = {
        "stage": "rendered_keypoint_selector_samples",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_count": int(merged.sample_count),
        "source_query_count": int(len(rows)),
        "skipped": skipped,
        "camera_source": camera_source,
        "field": {"path": str(args.field), "feature_dim": int(field.feature_dim), "gaussian_count": int(len(field))},
        "config": {
            "renderer": args.renderer,
            "detector": args.detector,
            "max_keypoints": int(args.max_keypoints),
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
            "sample": sample_config.__dict__,
        },
        "rows": rows,
        "outputs": {"sample_cache": str(args.output)},
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
