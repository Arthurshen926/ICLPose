"""Build full-map MATCHA joint training cache from rendered RGB/depth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
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
from feature_extract.vfm.matcha_coarse_supervision import (
    MatchaCoarseSupervisionConfig,
    build_matcha_coarse_supervision,
)
from feature_extract.vfm.matcha_joint_cache import (
    build_matcha_joint_index_training_set_from_maps,
    build_matcha_joint_training_set_from_maps,
)
from feature_extract.vfm.matcha_joint_training import (
    merge_matcha_joint_training_sets,
    save_matcha_joint_training_set_manifest,
    save_matcha_joint_training_set_npz,
)
from feature_extract.vfm.matcha_keypoint_distillation import AlikeKeypointExtractor, build_keypoint_label_map
from feature_extract.vfm.matcha_light_fusion import maybe_fuse_feature_map
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.render_pose_protocol import (
    group_top_reference_poses,
    parse_world_offset,
    render_pose_error_fields,
    sample_se3_perturbation,
    translate_pose_world,
)
from feature_extract.vfm.tokens import TokenBankManifest


PAIR_TYPE_IDS = {
    "A_gt": 0,
    "B_trans025": 1,
    "C_trans050": 2,
    "D_reference": 3,
    "B_trans005": 4,
    "B_trans010": 5,
}


def _parse_pair_types(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("pair_types must contain at least one value")
    unknown = [item for item in values if item not in PAIR_TYPE_IDS]
    if unknown:
        raise ValueError(f"unsupported pair_types: {unknown}")
    return values


def _cache_token(value: object) -> str:
    text = str(value)
    out = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        elif char == ".":
            out.append("p")
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "none"


def _cache_float(value: object) -> str:
    try:
        item = float(value)
    except (TypeError, ValueError):
        return _cache_token(value)
    text = f"{item:.6g}"
    return _cache_token(text.replace("-", "m"))


def _render_pair_cache_key(
    *,
    pair_type: str,
    render_pose_world_offset: str,
    pair_metadata: dict[str, object],
    pair_type_B_translation_m: float,
    pair_type_C_translation_m: float,
    perturb_rotation_deg: float,
    seed: int,
    record_index: int,
    pair_index: int,
) -> str:
    """Build a render/depth cache key that changes with the actual sampled pose."""

    candidate_id = str(pair_metadata.get("candidate_id", ""))
    parts = [
        _cache_token(pair_type),
        "base",
        _cache_token(render_pose_world_offset),
        "pt",
        _cache_float(pair_metadata.get("perturb_translation_m", 0.0)),
        "pr",
        _cache_float(pair_metadata.get("perturb_rotation_deg", 0.0)),
        "bmax",
        _cache_float(pair_type_B_translation_m),
        "cmax",
        _cache_float(pair_type_C_translation_m),
        "rmax",
        _cache_float(perturb_rotation_deg),
        "seed",
        _cache_token(seed),
        "rec",
        _cache_token(record_index),
        "pair",
        _cache_token(pair_index),
    ]
    if candidate_id:
        parts.extend(["cand", _cache_token(candidate_id)])
    return "_".join(parts)


def _pair_render_pose(
    *,
    pair_type: str,
    query_id: str,
    gt_pose_w2c: np.ndarray,
    base_offset: np.ndarray,
    seed: int,
    record_index: int,
    pair_index: int,
    pair_type_B_translation_m: float,
    pair_type_C_translation_m: float,
    perturb_rotation_deg: float,
    reference_top1: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    base_pose = translate_pose_world(gt_pose_w2c, base_offset)
    pair_type = str(pair_type)
    if pair_type == "A_gt":
        t_error, r_error = render_pose_error_fields(base_pose, gt_pose_w2c)
        return base_pose, {
            "pair_type": pair_type,
            "pair_type_id": int(PAIR_TYPE_IDS[pair_type]),
            "perturb_translation_m": float(t_error),
            "perturb_rotation_deg": float(r_error),
            "candidate_id": "",
        }
    if pair_type in {"B_trans005", "B_trans010", "B_trans025", "C_trans050"}:
        max_translation_by_type = {
            "B_trans005": 0.05,
            "B_trans010": 0.10,
            "B_trans025": float(pair_type_B_translation_m),
            "C_trans050": float(pair_type_C_translation_m),
        }
        max_translation = float(max_translation_by_type[pair_type])
        perturb = sample_se3_perturbation(
            base_pose,
            max_translation_m=max_translation,
            max_rotation_deg=float(perturb_rotation_deg),
            seed=int(seed) + 1009 * int(record_index) + 9176 * int(pair_index),
            key=f"{query_id}:{pair_type}",
        )
        t_error, r_error = render_pose_error_fields(perturb.pose_w2c, gt_pose_w2c)
        return perturb.pose_w2c, {
            "pair_type": pair_type,
            "pair_type_id": int(PAIR_TYPE_IDS[pair_type]),
            "perturb_translation_m": float(t_error),
            "perturb_rotation_deg": float(r_error),
            "candidate_id": "",
        }
    if pair_type == "D_reference":
        candidate = reference_top1.get(str(query_id))
        if candidate is None or getattr(candidate, "pose", None) is None:
            raise KeyError(f"reference pose not found for query {query_id!r}")
        pose = np.asarray(candidate.pose, dtype=np.float64).reshape(4, 4)
        t_error, r_error = render_pose_error_fields(pose, gt_pose_w2c)
        return pose, {
            "pair_type": pair_type,
            "pair_type_id": int(PAIR_TYPE_IDS[pair_type]),
            "perturb_translation_m": float(t_error),
            "perturb_rotation_deg": float(r_error),
            "candidate_id": str(candidate.candidate_id),
        }
    raise ValueError(f"unsupported pair_type: {pair_type}")


def _extract_matcha_joint_feature_from_rgb(
    rgb: np.ndarray,
    extractor,
    *,
    feature_mode: str,
    fine_intermediate_index: int,
    coarse_source: str,
    coarse_intermediate_index: int,
) -> np.ndarray:
    """Extract a single-map or RADIO-dual feature map for MATCHA joint cache."""

    if str(feature_mode) == "radio_final":
        return _load_or_extract_render_feature(
            cache_path=None,
            render_rgb=rgb,
            extractor=extractor,
            layer_name="radio_final",
            skip_existing=False,
        )
    if str(feature_mode) != "radio_dual":
        raise ValueError("feature_mode must be 'radio_final' or 'radio_dual'")
    import torch

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    output = extractor.extract_dual(
        tensor,
        fine_intermediate_index=int(fine_intermediate_index),
        coarse_source=str(coarse_source),
        coarse_intermediate_index=int(coarse_intermediate_index),
    )
    return output["dual"].detach().cpu().numpy().astype(np.float32, copy=False)


def _load_or_extract_matcha_joint_feature(
    *,
    cache_path: Path | None,
    rgb: np.ndarray,
    extractor,
    layer_name: str,
    feature_mode: str,
    fine_intermediate_index: int,
    coarse_source: str,
    coarse_intermediate_index: int,
    skip_existing: bool,
) -> np.ndarray:
    if cache_path is not None and bool(skip_existing) and cache_path.exists():
        with np.load(cache_path) as data:
            return np.asarray(data[layer_name], dtype=np.float32)
    feature = _extract_matcha_joint_feature_from_rgb(
        rgb,
        extractor,
        feature_mode=str(feature_mode),
        fine_intermediate_index=int(fine_intermediate_index),
        coarse_source=str(coarse_source),
        coarse_intermediate_index=int(coarse_intermediate_index),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{layer_name: feature})
    return feature


def _resize_rgb_for_grid(rgb: np.ndarray, feature_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    target_h = int(feature_hw[0]) * 8
    target_w = int(feature_hw[1]) * 8
    image = np.asarray(rgb)
    if int(image.shape[0]) == target_h and int(image.shape[1]) == target_w:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > target_h or image.shape[1] > target_w else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def _build_alike_label_map(
    extractor: AlikeKeypointExtractor | None,
    rgb: np.ndarray,
    *,
    feature_hw: tuple[int, int],
) -> tuple[np.ndarray | None, dict[str, object], np.ndarray | None]:
    if extractor is None:
        return None, {}, None
    resized = _resize_rgb_for_grid(rgb, feature_hw)
    keypoints, scores = extractor(resized)
    labels, stats = build_keypoint_label_map(
        keypoints,
        scores=scores,
        image_width=int(resized.shape[1]),
        image_height=int(resized.shape[0]),
        grid_width=int(feature_hw[1]),
        grid_height=int(feature_hw[0]),
    )
    source = np.asarray(rgb)
    if keypoints.shape[0] == 0:
        source_keypoints = np.zeros((0, 2), dtype=np.float64)
    else:
        scale_x = float(source.shape[1]) / max(float(resized.shape[1]), 1.0)
        scale_y = float(source.shape[0]) / max(float(resized.shape[0]), 1.0)
        source_keypoints = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2).copy()
        source_keypoints[:, 0] *= scale_x
        source_keypoints[:, 1] *= scale_y
    return labels, stats, source_keypoints


def _write_matcha_joint_cache_outputs(
    joint_sets: list,
    *,
    output: Path | str,
    output_manifest: Path | str = "",
) -> dict[str, str]:
    """Write either a single merged cache or a sharded manifest cache."""

    manifest_path = Path(output_manifest) if str(output_manifest) else None
    output_path = Path(output) if str(output) else None
    outputs: dict[str, str] = {}
    if manifest_path is not None:
        save_matcha_joint_training_set_manifest(joint_sets, manifest_path)
        outputs["joint_cache_manifest"] = str(manifest_path)
        return outputs
    if output_path is None:
        raise ValueError("either output or output_manifest is required")
    merged = merge_matcha_joint_training_sets(joint_sets)
    save_matcha_joint_training_set_npz(merged, output_path)
    outputs["joint_cache"] = str(output_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--output_manifest", default="")
    parser.add_argument("--cache_format", default="dense", choices=("dense", "index_v2"))
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--split_name", default="", choices=("", "train", "val", "test"))
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--feature_mode", default="radio_final", choices=("radio_final", "radio_dual"))
    parser.add_argument("--radio_fine_intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_coarse_source", default="final", choices=("final", "intermediate"))
    parser.add_argument("--radio_coarse_intermediate_index", type=int, default=-1)
    parser.add_argument("--extract_query_features_from_image", action="store_true")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--pair_types", default="A_gt")
    parser.add_argument("--pair_type_B_translation_m", type=float, default=0.25)
    parser.add_argument("--pair_type_C_translation_m", type=float, default=0.5)
    parser.add_argument("--perturb_rotation_deg", type=float, default=0.0)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--feature_fusion_mode", default="none", choices=("none", "local_attention"))
    parser.add_argument("--feature_fusion_radius", type=int, default=1)
    parser.add_argument("--feature_fusion_temperature", type=float, default=5.0)
    parser.add_argument("--feature_fusion_alpha", type=float, default=0.5)
    parser.add_argument("--roundtrip_threshold_px", type=float, default=1.5)
    parser.add_argument("--roundtrip_heatmap_threshold_px", type=float, default=2.0)
    parser.add_argument("--visibility_alpha_threshold", type=float, default=0.0)
    parser.add_argument("--depth_edge_threshold_m", type=float, default=0.0)
    parser.add_argument("--collect_visibility_no_match", action="store_true")
    parser.add_argument("--max_visibility_no_match", type=int, default=256)
    parser.add_argument("--soft_offset_sigma_bins", type=float, default=0.75)
    parser.add_argument("--pose_confidence_labels", action="store_true")
    parser.add_argument("--pose_confidence_positive_threshold_px", type=float, default=8.0)
    parser.add_argument("--pose_confidence_negative_threshold_px", type=float, default=24.0)
    parser.add_argument("--hard_negatives_per_match", type=int, default=16)
    parser.add_argument("--fine_supervision_source", default="cell_center", choices=("cell_center", "render_alike"))
    parser.add_argument("--keypoint_distill_method", default="alike", choices=("none", "alike"))
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=4096)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=8000)
    parser.add_argument("--max_queries", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
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
    args = parser.parse_args(argv)
    if str(args.feature_mode) == "radio_dual" and str(args.layer_name) == "radio_final":
        args.layer_name = "radio_dual"
    _parse_pair_types(str(args.pair_types))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not str(args.output) and not str(args.output_manifest):
        raise SystemExit("one of --output or --output_manifest is required")
    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = _select_records(
        manifest.records,
        int(args.max_queries),
        str(args.view_selection),
        start_index=int(args.start_index),
    )
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
    pair_types = _parse_pair_types(str(args.pair_types))
    reference_top1 = {}
    if "D_reference" in pair_types:
        if not str(args.candidate_bank):
            raise ValueError("--candidate_bank is required when pair_types includes D_reference")
        reference_top1 = group_top_reference_poses(CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)).candidates)
    supervision_config = MatchaCoarseSupervisionConfig(
        roundtrip_threshold_px=float(args.roundtrip_threshold_px),
        alpha_threshold=float(args.visibility_alpha_threshold),
        depth_edge_threshold_m=float(args.depth_edge_threshold_m),
        collect_no_match=bool(args.collect_visibility_no_match),
        max_no_match=int(args.max_visibility_no_match),
        soft_offset_sigma_bins=float(args.soft_offset_sigma_bins),
        pose_confidence_labels=bool(args.pose_confidence_labels),
        pose_confidence_positive_threshold_px=float(args.pose_confidence_positive_threshold_px),
        pose_confidence_negative_threshold_px=float(args.pose_confidence_negative_threshold_px),
    )
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

    joint_sets = []
    shard_records = []
    manifest_path = Path(args.output_manifest) if str(args.output_manifest) else None
    manifest_shard_dir = manifest_path.parent / "shards" if manifest_path is not None else None
    if manifest_shard_dir is not None:
        manifest_shard_dir.mkdir(parents=True, exist_ok=True)
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
            query_feature = _load_or_extract_matcha_joint_feature(
                cache_path=query_cache_path,
                rgb=query_rgb,
                extractor=radio,
                layer_name=args.layer_name,
                feature_mode=str(args.feature_mode),
                fine_intermediate_index=int(args.radio_fine_intermediate_index),
                coarse_source=str(args.radio_coarse_source),
                coarse_intermediate_index=int(args.radio_coarse_intermediate_index),
                skip_existing=bool(args.skip_existing_query_tokens),
            )
        else:
            query_feature = _load_query_feature(record.token_path, args.layer_name)

        query_depth_cache_path = None
        if query_depth_cache_dir is not None:
            query_depth_cache_path = query_depth_cache_dir / f"{_safe_image_stem(record.image_id)}_matcha_joint_query_depth_{int(camera.width)}x{int(camera.height)}.npz"
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

        query_feature = maybe_fuse_feature_map(
            query_feature,
            mode=str(args.feature_fusion_mode),
            radius=int(args.feature_fusion_radius),
            temperature=float(args.feature_fusion_temperature),
            alpha=float(args.feature_fusion_alpha),
            device=str(args.device),
        )
        query_labels, query_kp_stats, _query_kp_xy = _build_alike_label_map(
            keypoint_extractor,
            query_rgb,
            feature_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
        )
        for pair_idx, pair_type in enumerate(pair_types):
            try:
                render_pose, pair_metadata = _pair_render_pose(
                    pair_type=str(pair_type),
                    query_id=record.image_id,
                    gt_pose_w2c=gt.pose_w2c,
                    base_offset=offset,
                    seed=int(args.seed),
                    record_index=int(record_idx),
                    pair_index=int(pair_idx),
                    pair_type_B_translation_m=float(args.pair_type_B_translation_m),
                    pair_type_C_translation_m=float(args.pair_type_C_translation_m),
                    perturb_rotation_deg=float(args.perturb_rotation_deg),
                    reference_top1=reference_top1,
                )
            except KeyError:
                skipped["missing_pose"] += 1
                continue
            pair_cache_key = _render_pair_cache_key(
                pair_type=str(pair_type),
                render_pose_world_offset=str(args.render_pose_world_offset),
                pair_metadata=pair_metadata,
                pair_type_B_translation_m=float(args.pair_type_B_translation_m),
                pair_type_C_translation_m=float(args.pair_type_C_translation_m),
                perturb_rotation_deg=float(args.perturb_rotation_deg),
                seed=int(args.seed),
                record_index=int(record_idx),
                pair_index=int(pair_idx),
            )
            render_cache_path = None
            if render_rgb_depth_cache_dir is not None:
                render_cache_path = render_rgb_depth_cache_dir / f"{_safe_image_stem(record.image_id)}_matcha_joint_render_{render_width}x{render_height}_{pair_cache_key}.npz"
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
            render_feature_cache_path = None
            if render_cache_dir is not None:
                render_feature_cache_path = _render_token_cache_path(
                    render_cache_dir,
                    f"{record.image_id}:matcha_joint_render:{pair_cache_key}",
                    render_width,
                    render_height,
                )
            render_feature = _load_or_extract_matcha_joint_feature(
                cache_path=render_feature_cache_path,
                rgb=render_rgb,
                extractor=radio,
                layer_name=args.layer_name,
                feature_mode=str(args.feature_mode),
                fine_intermediate_index=int(args.radio_fine_intermediate_index),
                coarse_source=str(args.radio_coarse_source),
                coarse_intermediate_index=int(args.radio_coarse_intermediate_index),
                skip_existing=bool(args.skip_existing_render_tokens),
            )
            if int(query_feature.shape[0]) != int(render_feature.shape[0]):
                skipped["dim_mismatch"] += 1
                continue
            render_feature = maybe_fuse_feature_map(
                render_feature,
                mode=str(args.feature_fusion_mode),
                radius=int(args.feature_fusion_radius),
                temperature=float(args.feature_fusion_temperature),
                alpha=float(args.feature_fusion_alpha),
                device=str(args.device),
            )
            render_labels, render_kp_stats, render_seed_xy = _build_alike_label_map(
                keypoint_extractor,
                render_rgb,
                feature_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
            )
            supervision_seed_xy = (
                render_seed_xy
                if str(args.fine_supervision_source) == "render_alike" and render_seed_xy is not None and render_seed_xy.shape[0] > 0
                else None
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
                render_seed_xy=supervision_seed_xy,
                config=supervision_config,
            )
            if supervision.count == 0:
                skipped["empty_supervision"] += 1
                continue
            builder = (
                build_matcha_joint_index_training_set_from_maps
                if str(args.cache_format) == "index_v2"
                else build_matcha_joint_training_set_from_maps
            )
            joint_kwargs = {
                "query_rgb": query_rgb,
                "render_rgb": render_rgb,
                "query_keypoint_label_map": query_labels,
                "render_keypoint_label_map": render_labels,
                "hard_negatives_per_match": int(args.hard_negatives_per_match),
                "roundtrip_heatmap_threshold_px": float(args.roundtrip_heatmap_threshold_px),
            }
            if str(args.cache_format) != "index_v2":
                joint_kwargs["seed"] = int(args.seed) + int(record_idx) * max(len(pair_types), 1) + int(pair_idx)
            joint = builder(
                query_feature,
                render_feature,
                supervision,
                **joint_kwargs,
            )
            if joint.coarse_fine_samples.sample_count == 0:
                skipped["empty_samples"] += 1
                continue
            object.__setattr__(joint, "pair_type_ids", np.asarray([int(pair_metadata["pair_type_id"])], dtype=np.int64))
            object.__setattr__(joint, "pair_type_names", np.asarray([str(pair_metadata["pair_type"])], dtype=object))
            object.__setattr__(joint, "pair_query_ids", np.asarray([str(record.image_id)], dtype=object))
            object.__setattr__(joint, "pair_split_names", np.asarray([str(args.split_name)], dtype=object))
            object.__setattr__(joint, "pair_candidate_ids", np.asarray([str(pair_metadata.get("candidate_id", ""))], dtype=object))
            object.__setattr__(joint, "pair_translation_errors_m", np.asarray([float(pair_metadata["perturb_translation_m"])], dtype=np.float32))
            object.__setattr__(joint, "pair_rotation_errors_deg", np.asarray([float(pair_metadata["perturb_rotation_deg"])], dtype=np.float32))
            if manifest_path is not None and manifest_shard_dir is not None:
                shard_index = len(shard_records)
                shard_path = manifest_shard_dir / f"shard_{int(shard_index):05d}.npz"
                save_matcha_joint_training_set_npz(joint, shard_path)
                try:
                    relative_path = shard_path.relative_to(manifest_path.parent)
                except ValueError:
                    relative_path = shard_path
                shard_records.append(
                    {
                        "path": str(relative_path),
                        "sample_count": int(joint.coarse_fine_samples.sample_count),
                        "pair_count": int(joint.query_feature_maps.shape[0]) if joint.query_feature_maps is not None else 0,
                        "query_id": str(record.image_id),
                        "split": str(args.split_name),
                        "pair_type": str(pair_metadata["pair_type"]),
                        "pair_type_id": int(pair_metadata["pair_type_id"]),
                        "candidate_id": str(pair_metadata.get("candidate_id", "")),
                        "perturb_translation_m": float(pair_metadata["perturb_translation_m"]),
                        "perturb_rotation_deg": float(pair_metadata["perturb_rotation_deg"]),
                    }
                )
            else:
                joint_sets.append(joint)
            rows.append(
                {
                    "query_id": record.image_id,
                    "split": str(args.split_name),
                    "pair_type": str(pair_metadata["pair_type"]),
                    "pair_type_id": int(pair_metadata["pair_type_id"]),
                    "candidate_id": str(pair_metadata.get("candidate_id", "")),
                    "perturb_translation_m": float(pair_metadata["perturb_translation_m"]),
                    "perturb_rotation_deg": float(pair_metadata["perturb_rotation_deg"]),
                    "query_feature_shape": list(query_feature.shape),
                    "render_feature_shape": list(render_feature.shape),
                    "render_alpha_mean": float(np.mean(render_alpha)),
                    "render_depth_valid_fraction": float(np.mean(np.isfinite(render_depth) & (render_depth > 0.0))),
                    "supervision_count": int(supervision.count),
                    "fine_supervision_source": str(args.fine_supervision_source),
                    "render_seed_count": 0 if supervision_seed_xy is None else int(supervision_seed_xy.shape[0]),
                    "query_keypoint_positive_count": int(query_kp_stats.get("positive_count", 0)),
                    "render_keypoint_positive_count": int(render_kp_stats.get("positive_count", 0)),
                    "roundtrip_median_px": float(np.median(supervision.roundtrip_errors_px)) if supervision.count else None,
                }
            )
    if not joint_sets and not shard_records:
        raise ValueError(f"no MATCHA joint samples built; skipped={skipped}")
    if manifest_path is not None:
        split_counts: dict[str, int] = {}
        pair_type_counts: dict[str, int] = {}
        query_ids: list[str] = []
        for item in shard_records:
            split_name = str(item.get("split", ""))
            pair_type = str(item.get("pair_type", ""))
            query_id = str(item.get("query_id", ""))
            if split_name:
                split_counts[split_name] = int(split_counts.get(split_name, 0) + 1)
            if pair_type:
                pair_type_counts[pair_type] = int(pair_type_counts.get(pair_type, 0) + 1)
            if query_id:
                query_ids.append(query_id)
        manifest_metadata = {
            "format": "vfm_matcha_joint_training_manifest_v1",
            "shard_count": int(len(shard_records)),
            "sample_count": int(sum(int(item["sample_count"]) for item in shard_records)),
            "split_counts": split_counts,
            "pair_type_counts": pair_type_counts,
            "query_count": int(len(set(query_ids))),
            "shards": shard_records,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_metadata, indent=2, sort_keys=True) + "\n")
        outputs = {"joint_cache_manifest": str(manifest_path)}
        sample_count = int(manifest_metadata["sample_count"])
    else:
        outputs = _write_matcha_joint_cache_outputs(
            joint_sets,
            output=Path(args.output) if str(args.output) else "",
            output_manifest="",
        )
        sample_count = int(sum(int(item.coarse_fine_samples.sample_count) for item in joint_sets))
    summary = {
        "stage": "matcha_joint_cache_builder",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_count": sample_count,
        "pair_count": int(len(shard_records) if shard_records else len(joint_sets)),
        "source_query_count": int(len(rows)),
        "skipped": skipped,
        "camera_source": camera_source,
        "config": vars(args),
        "rows": rows,
        "outputs": outputs,
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
