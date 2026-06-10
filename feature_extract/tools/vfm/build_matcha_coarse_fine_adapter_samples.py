"""Build MATCHA-style coarse-cell and offset-bin adapter samples from rendered RGB."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
    _extract_radio_feature_from_rgb,
    _load_or_extract_render_feature,
    _load_or_render_rgb_depth_cache,
    _render_rgb_and_depth,
    _render_token_cache_path,
    _resolve_render_size,
    _safe_image_stem,
    _select_records,
)
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _parse_default_camera,
    _read_rgb,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMRenderConfig
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    append_matcha_coarse_fine_training_set_capped,
    build_matcha_coarse_fine_training_set,
    save_matcha_coarse_fine_training_set_npz,
)
from feature_extract.vfm.matcha_coarse_supervision import (
    MatchaCoarseSupervisionConfig,
    build_matcha_coarse_supervision,
)
from feature_extract.vfm.matcha_keypoint_distillation import AlikeKeypointExtractor, build_keypoint_label_map
from feature_extract.vfm.matcha_light_fusion import maybe_fuse_feature_map
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.render_pose_protocol import parse_world_offset, translate_pose_world
from feature_extract.vfm.tokens import TokenBankManifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--extract_query_features_from_image", action="store_true")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--feature_fusion_mode", default="none", choices=("none", "local_attention"))
    parser.add_argument("--feature_fusion_radius", type=int, default=1)
    parser.add_argument("--feature_fusion_temperature", type=float, default=5.0)
    parser.add_argument("--feature_fusion_alpha", type=float, default=0.5)
    parser.add_argument("--roundtrip_threshold_px", type=float, default=1.5)
    parser.add_argument("--hard_negatives_per_match", type=int, default=16)
    parser.add_argument("--keypoint_distill_method", default="none", choices=("none", "alike"))
    parser.add_argument("--keypoint_max_rows", type=int, default=4096)
    parser.add_argument("--keypoint_nonkeypoint_divisor", type=int, default=32)
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=4096)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=8000)
    parser.add_argument("--max_queries", type=int, default=32)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--max_total_samples", type=int, default=20000)
    parser.add_argument("--renderer", default="official_2dgs", choices=("official_2dgs",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--query_token_cache_dir", default="")
    parser.add_argument("--skip_existing_query_tokens", action="store_true")
    parser.add_argument("--render_token_cache_dir", default="")
    parser.add_argument("--skip_existing_render_tokens", action="store_true")
    parser.add_argument("--render_rgb_depth_cache_dir", default="")
    parser.add_argument("--skip_existing_render_rgb_depth", action="store_true")
    parser.add_argument("--query_depth_cache_dir", default="")
    parser.add_argument("--skip_existing_query_depth", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = _select_records(manifest.records, int(args.max_queries), str(args.view_selection))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    render_width, render_height = _resolve_render_size(camera, int(args.render_width), int(args.render_height))
    render_camera = _scale_camera(camera, render_width, render_height)
    render_config = GaussianVFMRenderConfig(width=render_width, height=render_height, radius_px=2.0, depth_epsilon=0.02)
    query_depth_config = GaussianVFMRenderConfig(width=int(camera.width), height=int(camera.height), radius_px=2.0, depth_epsilon=0.02)
    rgb_source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
    from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

    radio = RADIOFeatureExtractor(version=args.radio_version, device=args.device, radio_repo=args.radio_repo)
    query_cache_dir = Path(args.query_token_cache_dir) if args.query_token_cache_dir else None
    render_cache_dir = Path(args.render_token_cache_dir) if args.render_token_cache_dir else None
    render_rgb_depth_cache_dir = Path(args.render_rgb_depth_cache_dir) if args.render_rgb_depth_cache_dir else None
    query_depth_cache_dir = Path(args.query_depth_cache_dir) if args.query_depth_cache_dir else None
    offset = parse_world_offset(str(args.render_pose_world_offset))
    supervision_config = MatchaCoarseSupervisionConfig(roundtrip_threshold_px=float(args.roundtrip_threshold_px))
    keypoint_extractor = None
    if str(args.keypoint_distill_method) == "alike":
        keypoint_extractor = AlikeKeypointExtractor(
            matcha_repo=str(args.alike_repo),
            model_name=str(args.alike_model),
            top_k=int(args.alike_top_k),
            scores_th=float(args.alike_scores_th),
            n_limit=int(args.alike_n_limit),
            device=str(args.device),
        )
    merged = None
    rows = []
    skipped = {"missing_pose": 0, "dim_mismatch": 0, "empty_supervision": 0, "empty_samples": 0}
    for record_idx, record in enumerate(records):
        gt = gt_by_query.get(record.image_id)
        if gt is None:
            skipped["missing_pose"] += 1
            continue
        query_rgb = _read_rgb(Path(args.image_root) / record.image_id)
        if bool(args.extract_query_features_from_image):
            query_cache_path = None if query_cache_dir is None else _render_token_cache_path(query_cache_dir, record.image_id, int(camera.width), int(camera.height))
            query_feature = _load_or_extract_render_feature(
                cache_path=query_cache_path,
                render_rgb=query_rgb,
                extractor=radio,
                layer_name=args.layer_name,
                skip_existing=bool(args.skip_existing_query_tokens),
            )
        else:
            query_feature = _load_query_feature(record.token_path, args.layer_name)
        render_pose = translate_pose_world(gt.pose_w2c, offset)
        render_cache_path = None
        if render_rgb_depth_cache_dir is not None:
            render_cache_path = render_rgb_depth_cache_dir / f"{_safe_image_stem(record.image_id)}_matcha_cf_render_{render_width}x{render_height}_{str(args.render_pose_world_offset).replace(',', '_')}.npz"
        render_rgb, render_depth, render_alpha = _load_or_render_rgb_depth_cache(
            cache_path=render_cache_path,
            render_fn=lambda pose=render_pose: _render_rgb_and_depth(
                rgb_source,
                None,
                pose_w2c=pose,
                camera=camera,
                config=render_config,
                renderer="official_2dgs",
                device=args.device,
            ),
            skip_existing=bool(args.skip_existing_render_rgb_depth),
        )
        query_depth_cache_path = None
        if query_depth_cache_dir is not None:
            query_depth_cache_path = query_depth_cache_dir / f"{_safe_image_stem(record.image_id)}_matcha_cf_query_depth_{int(camera.width)}x{int(camera.height)}.npz"
        _query_render_rgb, query_depth, _query_alpha = _load_or_render_rgb_depth_cache(
            cache_path=query_depth_cache_path,
            render_fn=lambda pose=gt.pose_w2c: _render_rgb_and_depth(
                rgb_source,
                None,
                pose_w2c=pose,
                camera=camera,
                config=query_depth_config,
                renderer="official_2dgs",
                device=args.device,
            ),
            skip_existing=bool(args.skip_existing_query_depth),
        )
        render_feature_cache_path = None
        if render_cache_dir is not None:
            render_feature_cache_path = _render_token_cache_path(
                render_cache_dir,
                f"{record.image_id}:matcha_cf_render:{str(args.render_pose_world_offset)}",
                render_width,
                render_height,
            )
        render_feature = _load_or_extract_render_feature(
            cache_path=render_feature_cache_path,
            render_rgb=render_rgb,
            extractor=radio,
            layer_name=args.layer_name,
            skip_existing=bool(args.skip_existing_render_tokens),
        )
        if int(query_feature.shape[0]) != int(render_feature.shape[0]):
            skipped["dim_mismatch"] += 1
            continue
        query_feature = maybe_fuse_feature_map(
            query_feature,
            mode=str(args.feature_fusion_mode),
            radius=int(args.feature_fusion_radius),
            temperature=float(args.feature_fusion_temperature),
            alpha=float(args.feature_fusion_alpha),
            device=str(args.device),
        )
        render_feature = maybe_fuse_feature_map(
            render_feature,
            mode=str(args.feature_fusion_mode),
            radius=int(args.feature_fusion_radius),
            temperature=float(args.feature_fusion_temperature),
            alpha=float(args.feature_fusion_alpha),
            device=str(args.device),
        )
        supervision = build_matcha_coarse_supervision(
            render_depth=render_depth,
            query_depth=query_depth,
            render_camera=render_camera,
            query_camera=camera,
            render_pose_w2c=render_pose,
            query_pose_w2c=gt.pose_w2c,
            render_grid_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
            query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
            config=supervision_config,
        )
        if supervision.count == 0:
            skipped["empty_supervision"] += 1
            continue
        query_keypoint_label_map = None
        render_keypoint_label_map = None
        query_keypoint_stats = {}
        render_keypoint_stats = {}
        if keypoint_extractor is not None:
            query_kpts, query_scores = keypoint_extractor(query_rgb)
            render_kpts, render_scores = keypoint_extractor(render_rgb)
            query_keypoint_label_map, query_keypoint_stats = build_keypoint_label_map(
                query_kpts,
                scores=query_scores,
                image_width=int(camera.width),
                image_height=int(camera.height),
                grid_width=int(query_feature.shape[2]),
                grid_height=int(query_feature.shape[1]),
            )
            render_keypoint_label_map, render_keypoint_stats = build_keypoint_label_map(
                render_kpts,
                scores=render_scores,
                image_width=int(render_width),
                image_height=int(render_height),
                grid_width=int(render_feature.shape[2]),
                grid_height=int(render_feature.shape[1]),
            )
        samples = build_matcha_coarse_fine_training_set(
            query_feature,
            render_feature,
            supervision,
            hard_negatives_per_match=int(args.hard_negatives_per_match),
            query_keypoint_label_map=query_keypoint_label_map,
            render_keypoint_label_map=render_keypoint_label_map,
            max_keypoint_rows=int(args.keypoint_max_rows),
            keypoint_nonkeypoint_divisor=int(args.keypoint_nonkeypoint_divisor),
            seed=int(args.seed) + int(record_idx),
        )
        if samples.sample_count == 0:
            skipped["empty_samples"] += 1
            continue
        merged = append_matcha_coarse_fine_training_set_capped(
            merged,
            samples,
            max_samples=int(args.max_total_samples),
            seed=int(args.seed) + int(record_idx),
        )
        rows.append(
            {
                "query_id": record.image_id,
                "query_feature_shape": list(query_feature.shape),
                "render_feature_shape": list(render_feature.shape),
                "render_alpha_mean": float(np.mean(render_alpha)),
                "render_depth_valid_fraction": float(np.mean(np.isfinite(render_depth) & (render_depth > 0.0))),
                "supervision_count": int(supervision.count),
                "sample_count": int(samples.sample_count),
                "merged_sample_count": int(merged.sample_count),
                "query_keypoint_sample_count": int(samples.query_keypoint_labels.shape[0]),
                "render_keypoint_sample_count": int(samples.render_keypoint_labels.shape[0]),
                "query_keypoint_positive_count": int(query_keypoint_stats.get("positive_count", 0)),
                "render_keypoint_positive_count": int(render_keypoint_stats.get("positive_count", 0)),
                "roundtrip_median_px": float(np.median(supervision.roundtrip_errors_px)) if supervision.count else None,
            }
        )
    if merged is None or merged.sample_count == 0:
        raise ValueError(f"no MATCHA coarse-fine samples built; skipped={skipped}")
    save_matcha_coarse_fine_training_set_npz(merged, Path(args.output))
    summary = {
        "stage": "matcha_coarse_fine_adapter_samples",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_count": int(merged.sample_count),
        "source_query_count": int(len(rows)),
        "skipped": skipped,
        "camera_source": camera_source,
        "config": {
            "renderer": str(args.renderer),
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
            "resolved_render_width": int(render_width),
            "resolved_render_height": int(render_height),
            "render_pose_world_offset": str(args.render_pose_world_offset),
            "feature_fusion_mode": str(args.feature_fusion_mode),
            "feature_fusion_radius": int(args.feature_fusion_radius),
            "feature_fusion_temperature": float(args.feature_fusion_temperature),
            "feature_fusion_alpha": float(args.feature_fusion_alpha),
            "roundtrip_threshold_px": float(args.roundtrip_threshold_px),
            "hard_negatives_per_match": int(args.hard_negatives_per_match),
            "keypoint_distill_method": str(args.keypoint_distill_method),
            "keypoint_max_rows": int(args.keypoint_max_rows),
            "keypoint_nonkeypoint_divisor": int(args.keypoint_nonkeypoint_divisor),
            "alike_repo": str(args.alike_repo),
            "alike_model": str(args.alike_model),
            "alike_top_k": int(args.alike_top_k),
            "alike_scores_th": float(args.alike_scores_th),
            "alike_n_limit": int(args.alike_n_limit),
            "extract_query_features_from_image": bool(args.extract_query_features_from_image),
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
