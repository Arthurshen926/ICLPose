"""Train MATCHA joint model from a thin streaming pair manifest.

This script builds one query/render pair on the fly per optimization step. It
does not serialize dense render/query feature-map training shards.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_matcha_joint_cache import (
    _build_alike_label_map,
    _extract_matcha_joint_feature_from_rgb,
    _pair_render_pose,
)
from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
    _load_or_render_rgb_depth_cache,
    _render_rgb_and_depth,
    _resolve_render_size,
    _safe_image_stem,
)
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
    _read_rgb,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMRenderConfig
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.matcha_coarse_fine_adapter import save_matcha_coarse_fine_adapter
from feature_extract.vfm.matcha_coarse_supervision import MatchaCoarseSupervisionConfig, build_matcha_coarse_supervision
from feature_extract.vfm.matcha_joint_cache import build_matcha_joint_index_training_set_from_maps
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingConfig,
    MatchaJointTrainingRun,
    MatchaJointTrainingSet,
    _build_matcha_joint_model_for_samples,
    _evaluate,
    _loss_value_for_samples,
    _model_state_snapshot,
    _sample_indices,
    _total_loss,
    joint_run_as_coarse_fine_adapter_run,
    load_matcha_joint_model,
    save_matcha_joint_model,
)
from feature_extract.vfm.matcha_keypoint_distillation import AlikeKeypointExtractor
from feature_extract.vfm.matcha_light_fusion import maybe_fuse_feature_map
from feature_extract.vfm.matcha_streaming_manifest import MatchaStreamingPairManifest, MatchaStreamingPairRecord
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.render_pose_protocol import group_top_reference_poses, parse_world_offset
from feature_extract.vfm.tokens import TokenBankManifest


def _feature_cache_path(root: Path, query_id: str, *, width: int, height: int, layer_name: str) -> Path:
    return root / f"{_safe_image_stem(query_id)}_{int(width)}x{int(height)}_{str(layer_name)}.npz"


def _load_or_extract_query_feature(
    *,
    cache_path: Path | None,
    rgb: np.ndarray,
    extractor,
    layer_name: str,
    feature_mode: str,
    fine_intermediate_index: int,
    coarse_source: str,
    coarse_intermediate_index: int,
    cache_dtype: str,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
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
        stored = feature.astype(np.float16, copy=False) if str(cache_dtype) == "float16" else feature.astype(np.float32, copy=False)
        np.savez_compressed(cache_path, **{str(layer_name): stored})
    return feature.astype(np.float32, copy=False)


class StreamingPairBuilder:
    def __init__(self, args: argparse.Namespace, manifest: MatchaStreamingPairManifest) -> None:
        self.args = args
        self.manifest = manifest
        source_query_manifest = str(manifest.metadata.get("source_query_manifest", ""))
        if not source_query_manifest:
            raise ValueError("streaming manifest metadata must include source_query_manifest")
        self.query_manifest = TokenBankManifest.from_json(Path(source_query_manifest))
        self.records_by_id = {record.image_id: record for record in self.query_manifest.records}
        self.gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
        camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
        self.camera, self.camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
        self.render_width, self.render_height = _resolve_render_size(self.camera, int(args.render_width), int(args.render_height))
        self.render_camera = _scale_camera(self.camera, self.render_width, self.render_height)
        self.render_config = GaussianVFMRenderConfig(
            width=int(self.render_width),
            height=int(self.render_height),
            radius_px=2.0,
            depth_epsilon=0.02,
        )
        self.query_depth_config = GaussianVFMRenderConfig(
            width=int(self.camera.width),
            height=int(self.camera.height),
            radius_px=2.0,
            depth_epsilon=0.02,
        )
        self.rgb_source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
        from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

        self.radio = RADIOFeatureExtractor(version=args.radio_version, device=args.device, radio_repo=args.radio_repo)
        self.offset = parse_world_offset(str(args.render_pose_world_offset))
        self.reference_top1 = {}
        if any(record.pair_type == "D_reference" for record in manifest.records):
            if not str(args.candidate_bank):
                raise ValueError("--candidate_bank is required when streaming manifest contains D_reference")
            self.reference_top1 = group_top_reference_poses(CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)).candidates)
        self.supervision_config = MatchaCoarseSupervisionConfig(
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
        self.keypoint_extractor = None
        if str(args.keypoint_distill_method) == "alike":
            self.keypoint_extractor = AlikeKeypointExtractor(
                matcha_repo=str(args.alike_repo),
                model_name=str(args.alike_model),
                top_k=int(args.alike_top_k),
                scores_th=float(args.alike_scores_th),
                n_limit=int(args.alike_n_limit),
                device=str(args.device),
            )
        self.query_cache_dir = Path(args.query_feature_cache_dir) if str(args.query_feature_cache_dir) else None

    def build(self, record: MatchaStreamingPairRecord) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        gt = self.gt_by_query.get(record.query_id)
        if gt is None:
            raise KeyError(f"missing GT pose for {record.query_id}")
        _source_record = self.records_by_id.get(record.query_id)
        if _source_record is None:
            raise KeyError(f"query id {record.query_id!r} not present in source manifest")
        query_rgb = _read_rgb(Path(self.args.image_root) / record.query_id)
        query_cache_path = None
        if self.query_cache_dir is not None:
            query_cache_path = _feature_cache_path(
                self.query_cache_dir,
                record.query_id,
                width=int(self.camera.width),
                height=int(self.camera.height),
                layer_name=str(self.args.layer_name),
            )
        query_feature = _load_or_extract_query_feature(
            cache_path=query_cache_path,
            rgb=query_rgb,
            extractor=self.radio,
            layer_name=str(self.args.layer_name),
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
            cache_dtype=str(self.args.query_feature_cache_dtype),
        )
        _query_render_rgb, query_depth, query_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=gt.pose_w2c: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.camera,
                config=self.query_depth_config,
                renderer="official_2dgs",
                device=self.args.device,
            ),
            skip_existing=False,
        )
        query_feature = maybe_fuse_feature_map(
            query_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=str(self.args.device),
        )
        query_labels, query_kp_stats, _query_kp_xy = _build_alike_label_map(
            self.keypoint_extractor,
            query_rgb,
            feature_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
        )
        render_pose, pair_metadata = _pair_render_pose(
            pair_type=str(record.pair_type),
            query_id=str(record.query_id),
            gt_pose_w2c=gt.pose_w2c,
            base_offset=self.offset,
            seed=int(record.seed),
            record_index=int(record.record_index),
            pair_index=int(record.pair_index),
            pair_type_B_translation_m=float(self.args.pair_type_B_translation_m),
            pair_type_C_translation_m=float(self.args.pair_type_C_translation_m),
            perturb_rotation_deg=float(self.args.perturb_rotation_deg),
            reference_top1=self.reference_top1,
        )
        render_rgb, render_depth, render_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=render_pose: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.camera,
                config=self.render_config,
                renderer="official_2dgs",
                device=self.args.device,
            ),
            skip_existing=False,
        )
        render_feature = _extract_matcha_joint_feature_from_rgb(
            render_rgb,
            self.radio,
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
        )
        render_feature = maybe_fuse_feature_map(
            render_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=str(self.args.device),
        )
        render_labels, render_kp_stats, render_seed_xy = _build_alike_label_map(
            self.keypoint_extractor,
            render_rgb,
            feature_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
        )
        supervision_seed_xy = (
            render_seed_xy
            if str(self.args.fine_supervision_source) == "render_alike" and render_seed_xy is not None and render_seed_xy.shape[0] > 0
            else None
        )
        supervision = build_matcha_coarse_supervision(
            render_depth=render_depth,
            query_depth=query_depth,
            render_alpha=render_alpha,
            query_alpha=query_alpha,
            render_camera=self.render_camera,
            query_camera=self.camera,
            render_pose_w2c=render_pose,
            query_pose_w2c=gt.pose_w2c,
            render_grid_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
            query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
            render_seed_xy=supervision_seed_xy,
            config=self.supervision_config,
        )
        if supervision.count == 0:
            raise ValueError("empty coarse supervision")
        joint = build_matcha_joint_index_training_set_from_maps(
            query_feature,
            render_feature,
            supervision,
            query_rgb=query_rgb if query_labels is not None else None,
            render_rgb=render_rgb if render_labels is not None else None,
            query_keypoint_label_map=query_labels,
            render_keypoint_label_map=render_labels,
            hard_negatives_per_match=int(self.args.hard_negatives_per_match),
            roundtrip_heatmap_threshold_px=float(self.args.roundtrip_heatmap_threshold_px),
        )
        object.__setattr__(joint, "pair_type_ids", np.asarray([int(pair_metadata["pair_type_id"])], dtype=np.int64))
        object.__setattr__(joint, "pair_type_names", np.asarray([str(pair_metadata["pair_type"])], dtype=object))
        object.__setattr__(joint, "pair_query_ids", np.asarray([str(record.query_id)], dtype=object))
        object.__setattr__(joint, "pair_split_names", np.asarray([str(record.split)], dtype=object))
        object.__setattr__(joint, "pair_candidate_ids", np.asarray([str(pair_metadata.get("candidate_id", ""))], dtype=object))
        object.__setattr__(joint, "pair_translation_errors_m", np.asarray([float(pair_metadata["perturb_translation_m"])], dtype=np.float32))
        object.__setattr__(joint, "pair_rotation_errors_deg", np.asarray([float(pair_metadata["perturb_rotation_deg"])], dtype=np.float32))
        row = {
            "query_id": str(record.query_id),
            "split": str(record.split),
            "pair_type": str(record.pair_type),
            "candidate_id": str(pair_metadata.get("candidate_id", "")),
            "sample_count": int(joint.coarse_fine_samples.sample_count),
            "query_feature_shape": list(query_feature.shape),
            "render_feature_shape": list(render_feature.shape),
            "render_depth_valid_fraction": float(np.mean(np.isfinite(render_depth) & (render_depth > 0.0))),
            "render_alpha_mean": float(np.mean(render_alpha)),
            "supervision_no_match_count": int(getattr(supervision, "no_match_count", 0)),
            "query_keypoint_positive_count": int(query_kp_stats.get("positive_count", 0)),
            "render_keypoint_positive_count": int(render_kp_stats.get("positive_count", 0)),
            "roundtrip_median_px": float(np.median(supervision.roundtrip_errors_px)) if supervision.count else None,
        }
        return joint, row


def _sample_pair_index(rng: np.random.Generator, records: Sequence[MatchaStreamingPairRecord]) -> int:
    return int(rng.integers(0, len(records)))


def _records_for_curriculum(
    records: Sequence[MatchaStreamingPairRecord],
    *,
    step: int,
    total_steps: int,
    mode: str,
) -> Sequence[MatchaStreamingPairRecord]:
    if str(mode) == "uniform":
        return records
    if str(mode) not in {"robust_default", "conservative_25cm"}:
        raise ValueError(f"unsupported pair_type_curriculum: {mode}")
    denom = max(int(total_steps), 1)
    progress = float(step) / float(denom)
    if str(mode) == "conservative_25cm":
        if progress < 0.40:
            allowed = {"A_gt"}
        elif progress < 0.75:
            allowed = {"A_gt", "B_trans005", "B_trans010"}
        else:
            allowed = {"A_gt", "B_trans005", "B_trans010", "B_trans025"}
    else:
        if progress < 0.20:
            allowed = {"A_gt"}
        elif progress < 0.45:
            allowed = {"A_gt", "B_trans005", "B_trans010"}
        elif progress < 0.75:
            allowed = {"A_gt", "B_trans005", "B_trans010", "B_trans025"}
        else:
            allowed = {"A_gt", "B_trans010", "B_trans025", "C_trans050", "D_reference"}
    selected = [record for record in records if str(record.pair_type) in allowed]
    return selected if selected else records


def _is_head_parameter(name: str) -> bool:
    head_tokens = (
        "offset_head",
        "offset_head_map",
        "pair_confidence_head",
        "pair_fine_head",
        "query_pair_fine_head",
        "original_fine_matcher",
        "query_original_fine_matcher",
        "local_fine",
        "heatmap_head",
        "rgb_keypoint_detector",
    )
    return any(token in str(name) for token in head_tokens)


def _make_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    descriptor_lr_scale: float,
    head_lr_scale: float,
    freeze_descriptor: bool,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    descriptor_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head_parameter(str(name)):
            head_params.append(param)
        else:
            descriptor_params.append(param)
    groups = []
    if descriptor_params:
        groups.append(
            {
                "params": descriptor_params,
                "lr": 0.0 if bool(freeze_descriptor) else float(lr) * float(descriptor_lr_scale),
                "name": "descriptor",
            }
        )
    if head_params:
        groups.append({"params": head_params, "lr": float(lr) * float(head_lr_scale), "name": "heads"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, lr=float(lr), weight_decay=1e-4)
    return optimizer, {"descriptor_param_count": int(sum(p.numel() for p in descriptor_params)), "head_param_count": int(sum(p.numel() for p in head_params))}


def _set_descriptor_group_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    descriptor_lr_scale: float,
    freeze: bool,
) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "descriptor":
            group["lr"] = 0.0 if bool(freeze) else float(base_lr) * float(descriptor_lr_scale)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming_manifest", required=True)
    parser.add_argument("--validation_streaming_manifest", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_joint_model", default="")
    parser.add_argument("--output_best_joint_model", default="")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--warm_start_joint_checkpoint", default="")
    parser.add_argument("--layer_name", default="radio_dual")
    parser.add_argument("--feature_mode", default="radio_dual", choices=("radio_final", "radio_dual"))
    parser.add_argument("--radio_fine_intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_coarse_source", default="final", choices=("final", "intermediate"))
    parser.add_argument("--radio_coarse_intermediate_index", type=int, default=-1)
    parser.add_argument("--query_feature_cache_dir", default="")
    parser.add_argument("--query_feature_cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--pair_type_B_translation_m", type=float, default=0.25)
    parser.add_argument("--pair_type_C_translation_m", type=float, default=0.5)
    parser.add_argument("--perturb_rotation_deg", type=float, default=0.0)
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
    parser.add_argument("--model_type", choices=("residual_adapter", "radio_dual_attention"), default="radio_dual_attention")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--fine_input_dim", type=int, default=1280)
    parser.add_argument("--coarse_input_dim", type=int, default=1280)
    parser.add_argument("--attention_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_depth", type=int, default=1)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_patch_size", type=int, default=4)
    parser.add_argument("--attention_upsample_mode", choices=("bilinear", "pixel_shuffle"), default="pixel_shuffle")
    parser.add_argument("--attention_fusion_mode", choices=("legacy", "matcha_original"), default="legacy")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--offset_loss_weight", type=float, default=0.25)
    parser.add_argument("--pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--render_pair_fine_loss_weight", type=float, default=None)
    parser.add_argument("--query_pair_fine_loss_weight", type=float, default=0.5)
    parser.add_argument("--pair_confidence_loss_weight", type=float, default=0.1)
    parser.add_argument("--dense_heatmap_loss_weight", type=float, default=0.25)
    parser.add_argument("--rgb_keypoint_loss_weight", type=float, default=0.25)
    parser.add_argument("--rgb_keypoint_position_loss_weight", type=float, default=0.0)
    parser.add_argument("--repeatability_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_fine_transformer_loss_weight", type=float, default=0.0)
    parser.add_argument("--patch_correlation_loss_weight", type=float, default=0.1)
    parser.add_argument("--patch_correlation_window_size", type=int, default=3)
    parser.add_argument("--hard_negative_weight", type=float, default=0.1)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--hard_false_match_weight", type=float, default=0.0)
    parser.add_argument("--hard_false_match_margin", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--pair_type_curriculum", default="uniform", choices=("uniform", "robust_default", "conservative_25cm"))
    parser.add_argument("--freeze_descriptor_steps", type=int, default=0)
    parser.add_argument("--descriptor_lr_scale", type=float, default=1.0)
    parser.add_argument("--head_lr_scale", type=float, default=1.0)
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--validation_pair_count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.render_pair_fine_loss_weight is not None:
        args.pair_fine_loss_weight = float(args.render_pair_fine_loss_weight)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    manifest = MatchaStreamingPairManifest.from_json(Path(args.streaming_manifest))
    val_manifest = MatchaStreamingPairManifest.from_json(Path(args.validation_streaming_manifest)) if str(args.validation_streaming_manifest) else None
    cfg = MatchaJointTrainingConfig(
        model_type=str(args.model_type),
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        fine_input_dim=int(args.fine_input_dim),
        coarse_input_dim=int(args.coarse_input_dim),
        attention_hidden_dim=int(args.attention_hidden_dim),
        attention_depth=int(args.attention_depth),
        attention_heads=int(args.attention_heads),
        attention_patch_size=int(args.attention_patch_size),
        attention_upsample_mode=str(args.attention_upsample_mode),
        attention_fusion_mode=str(args.attention_fusion_mode),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        offset_loss_weight=float(args.offset_loss_weight),
        pair_fine_loss_weight=float(args.pair_fine_loss_weight),
        query_pair_fine_loss_weight=float(args.query_pair_fine_loss_weight),
        pair_confidence_loss_weight=float(args.pair_confidence_loss_weight),
        dense_heatmap_loss_weight=float(args.dense_heatmap_loss_weight),
        rgb_keypoint_loss_weight=float(args.rgb_keypoint_loss_weight),
        rgb_keypoint_position_loss_weight=float(args.rgb_keypoint_position_loss_weight),
        repeatability_loss_weight=float(args.repeatability_loss_weight),
        local_fine_transformer_loss_weight=float(args.local_fine_transformer_loss_weight),
        patch_correlation_loss_weight=float(args.patch_correlation_loss_weight),
        patch_correlation_window_size=int(args.patch_correlation_window_size),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        hard_false_match_weight=float(args.hard_false_match_weight),
        hard_false_match_margin=float(args.hard_false_match_margin),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        map_pair_batch_size=int(args.map_pair_batch_size),
        device=str(args.device),
        seed=int(args.seed),
    )
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")
    builder = StreamingPairBuilder(args, manifest)
    val_builder = StreamingPairBuilder(args, val_manifest) if val_manifest is not None else None
    rng = np.random.default_rng(int(cfg.seed))
    skipped = 0

    def build_with_retries(records: Sequence[MatchaStreamingPairRecord], source_builder: StreamingPairBuilder, *, seed_offset: int) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        nonlocal skipped
        last_error: Exception | None = None
        for attempt in range(32):
            idx = _sample_pair_index(np.random.default_rng(int(cfg.seed) + int(seed_offset) + attempt), records)
            try:
                return source_builder.build(records[idx])
            except Exception as exc:  # keep streaming robust to occasional bad render/supervision pairs
                skipped += 1
                last_error = exc
        raise RuntimeError(f"failed to build a streaming training pair after retries: {last_error}") from last_error

    first_records = _records_for_curriculum(
        manifest.records,
        step=0,
        total_steps=int(cfg.steps),
        mode=str(args.pair_type_curriculum),
    )
    first_samples, first_row = build_with_retries(first_records, builder, seed_offset=0)
    model = _build_matcha_joint_model_for_samples(first_samples, cfg, device)
    warm_start_report: dict[str, object] = {}
    if str(args.warm_start_joint_checkpoint):
        warm = load_matcha_joint_model(Path(args.warm_start_joint_checkpoint), device=str(device)).model
        result = model.load_state_dict(warm.state_dict(), strict=False)
        warm_start_report = {
            "warm_start_loaded": True,
            "warm_start_missing_keys": list(result.missing_keys),
            "warm_start_unexpected_keys": list(result.unexpected_keys),
        }
    optimizer, optimizer_report = _make_optimizer(
        model,
        lr=float(cfg.lr),
        descriptor_lr_scale=float(args.descriptor_lr_scale),
        head_lr_scale=float(args.head_lr_scale),
        freeze_descriptor=int(args.freeze_descriptor_steps) > 0,
    )
    initial_loss = _loss_value_for_samples(model, first_samples, cfg, device, seed=int(cfg.seed))
    validation_history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_validation_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    train_rows = [first_row]

    def maybe_validate(step: int) -> None:
        nonlocal best_state, best_validation_loss, best_validation_step
        if val_manifest is None or val_builder is None or int(args.validation_pair_count) <= 0:
            return
        values = []
        for item in range(int(args.validation_pair_count)):
            samples, _row = build_with_retries(val_manifest.records, val_builder, seed_offset=50000 + int(step) * 97 + item)
            values.append(_loss_value_for_samples(model, samples, cfg, device, seed=int(cfg.seed) + 50000 + int(step) + item))
        value = float(np.mean(values)) if values else float("inf")
        validation_history.append({"step": int(step), "loss": value})
        if value < best_validation_loss:
            best_validation_loss = value
            best_validation_step = int(step)
            best_state = _model_state_snapshot(model)

    maybe_validate(0)
    model.train()
    last_samples = first_samples
    last_metrics: dict[str, float] = {}
    for step in range(int(cfg.steps)):
        _set_descriptor_group_lr(
            optimizer,
            base_lr=float(cfg.lr),
            descriptor_lr_scale=float(args.descriptor_lr_scale),
            freeze=int(step) < int(args.freeze_descriptor_steps),
        )
        active_records = _records_for_curriculum(
            manifest.records,
            step=int(step),
            total_steps=int(cfg.steps),
            mode=str(args.pair_type_curriculum),
        )
        idx = _sample_pair_index(rng, active_records)
        samples, row = builder.build(active_records[idx])
        train_rows.append(row)
        last_samples = samples
        batch_idx = _sample_indices(rng, samples.coarse_fine_samples.sample_count, int(cfg.batch_size))
        loss, last_metrics = _total_loss(model, samples, batch_idx, cfg, device, seed=int(cfg.seed) + int(step))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if int(args.validation_interval) > 0 and ((int(step) + 1) % int(args.validation_interval) == 0):
            maybe_validate(int(step) + 1)
    if val_manifest is not None and (not validation_history or int(validation_history[-1]["step"]) != int(cfg.steps)):
        maybe_validate(int(cfg.steps))
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss = _loss_value_for_samples(model, last_samples, cfg, device, seed=int(cfg.seed) + int(cfg.steps) + 1)
    summary = {
        "stage": "matcha_style_streaming_joint_training",
        "streaming_training": True,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "model_type": str(cfg.model_type),
        "input_dim": int(first_samples.coarse_fine_samples.input_dim),
        "output_dim": int(cfg.output_dim),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "manifest_pair_count": int(len(manifest.records)),
        "manifest_query_count": int(manifest.query_count),
        "manifest_pair_type_counts": manifest.pair_type_counts,
        "pair_type_curriculum": str(args.pair_type_curriculum),
        "freeze_descriptor_steps": int(args.freeze_descriptor_steps),
        "descriptor_lr_scale": float(args.descriptor_lr_scale),
        "head_lr_scale": float(args.head_lr_scale),
        "optimizer": optimizer_report,
        "skipped_pair_builds": int(skipped),
        "first_pair": first_row,
        "last_step_metrics": {key: float(value) for key, value in last_metrics.items()},
    }
    summary.update(warm_start_report)
    if validation_history:
        summary.update(
            {
                "best_validation_loss": float(best_validation_loss),
                "best_validation_step": int(best_validation_step),
                "validation_history": validation_history,
                "validation_pair_count": int(args.validation_pair_count),
            }
        )
    summary.update(_evaluate(model, last_samples, cfg, device))
    run = MatchaJointTrainingRun(model=model.cpu().eval(), summary=summary)
    adapter_run = joint_run_as_coarse_fine_adapter_run(run)
    save_matcha_coarse_fine_adapter(adapter_run, Path(args.output_model))
    joint_output = Path(args.output_joint_model) if str(args.output_joint_model) else Path(args.output_model).with_name(Path(args.output_model).stem + "_joint.pt")
    save_matcha_joint_model(run, joint_output)
    best_output = Path(args.output_best_joint_model) if str(args.output_best_joint_model) else None
    if best_output is not None:
        save_matcha_joint_model(run, best_output)
    output_summary = {
        "stage": "matcha_style_streaming_joint_model_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "streaming_manifest": {
            "path": str(args.streaming_manifest),
            "query_count": int(manifest.query_count),
            "pair_count": int(len(manifest.records)),
            "pair_type_counts": manifest.pair_type_counts,
            "metadata": dict(manifest.metadata),
        },
        "validation_streaming_manifest": {
            "path": str(args.validation_streaming_manifest),
            "query_count": int(val_manifest.query_count) if val_manifest is not None else 0,
            "pair_count": int(len(val_manifest.records)) if val_manifest is not None else 0,
        },
        "feature_cache_policy": {
            "query_feature_cache_dir": str(args.query_feature_cache_dir),
            "query_feature_cache_dtype": str(args.query_feature_cache_dtype),
            "render_feature_cache": "disabled",
            "training_sample_cache": "disabled",
        },
        "robust_supervision": {
            "roundtrip_threshold_px": float(args.roundtrip_threshold_px),
            "visibility_alpha_threshold": float(args.visibility_alpha_threshold),
            "depth_edge_threshold_m": float(args.depth_edge_threshold_m),
            "collect_visibility_no_match": bool(args.collect_visibility_no_match),
            "max_visibility_no_match": int(args.max_visibility_no_match),
            "soft_offset_sigma_bins": float(args.soft_offset_sigma_bins),
            "pose_confidence_labels": bool(args.pose_confidence_labels),
            "pose_confidence_positive_threshold_px": float(args.pose_confidence_positive_threshold_px),
            "pose_confidence_negative_threshold_px": float(args.pose_confidence_negative_threshold_px),
        },
        "curriculum": {
            "pair_type_curriculum": str(args.pair_type_curriculum),
            "freeze_descriptor_steps": int(args.freeze_descriptor_steps),
            "descriptor_lr_scale": float(args.descriptor_lr_scale),
            "head_lr_scale": float(args.head_lr_scale),
        },
        "config": asdict(cfg),
        "training": summary,
        "outputs": {
            "adapter_model": str(args.output_model),
            "joint_model": str(joint_output),
            "best_joint_model": str(best_output) if best_output is not None else "",
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
