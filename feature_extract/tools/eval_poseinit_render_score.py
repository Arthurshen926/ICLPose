#!/usr/bin/env python3
"""Evaluate DCFF query-map render scoring over pose-init hypotheses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pose_refine import apply_pose_delta, feature_metric_solve
from feature_extract.students.radio_query_student import AbsolutePoseInitHead
from feature_extract.temporal_protocol import (
    TemporalInitSelector,
    build_temporal_pose_grid,
    summarize_error_pairs as summarize_protocol_error_pairs,
    summarize_temporal_errors,
    temporal_sort_records,
)
from feature_extract.train_impl import (
    JointRADIOQueryDataset,
    MapFeatureRenderer,
    RetrievalTeacherStore,
    TeacherFeatureStore,
    build_all_records,
    build_radio_query_student,
    load_config,
    load_model_warmstart,
    local_render_score_feature_candidates,
    move_batch_to_device,
    pose_anchors_from_dataset,
    pose_error_tensors,
    render_score_feature_candidates,
    resolve_query_feature_dims,
    safe_torch_load,
    select_pose_candidate_indices,
    split_records,
    topk_descriptor_candidate_indices,
)


def _parse_hw(text: str | None):
    if not text:
        return None
    parts = [int(part) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--render-hw/--score-hw must be formatted as H,W")
    return tuple(parts)


def _parse_float_list(text: str | None):
    if text is None or not str(text).strip():
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _resize_feature(feat: torch.Tensor, hw: tuple[int, int] | None) -> torch.Tensor:
    if hw is None or tuple(feat.shape[-2:]) == tuple(hw):
        return feat
    return F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)


def _pose_error_mm_deg(pose_pred: torch.Tensor, pose_gt: torch.Tensor):
    _rot_loss, rot_deg, trans_m = pose_error_tensors(
        pose_pred.float().view(1, 4, 4),
        pose_gt.float().view(1, 4, 4),
    )
    return float(trans_m[0].item() * 1000.0), float(rot_deg[0].item())


def _summarize_error_pairs(pairs):
    summary = summarize_protocol_error_pairs(pairs)
    if not pairs:
        return summary
    trans = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    rot = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    summary.update(
        {
            "recall_2deg_250mm": float(((rot < 2.0) & (trans < 250.0)).float().mean().item() * 100.0),
            "recall_5deg_1000mm": float(((rot < 5.0) & (trans < 1000.0)).float().mean().item() * 100.0),
        }
    )
    return summary


def _records_for_protocol(records, protocol: str):
    protocol_key = str(protocol)
    if protocol_key == "single_frame_real_init":
        return list(records)
    return temporal_sort_records(records)


def _resolve_query_feature(outputs, query_fine_key: str):
    if query_fine_key in outputs:
        return outputs[query_fine_key], query_fine_key
    if "fine" in outputs:
        return outputs["fine"], "fine"
    raise KeyError(f"Query feature '{query_fine_key}' not found and outputs['fine'] is unavailable")


def _render_candidate_features(
    map_renderer: MapFeatureRenderer,
    sample_name: str,
    candidate_poses: torch.Tensor,
    *,
    score_feature: str,
    chunk_size: int,
):
    feat_list = []
    mask_list = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, candidate_poses.shape[0], chunk_size):
            pose_chunk = candidate_poses[start : start + chunk_size]
            for pose in pose_chunk:
                _fine_raw, fine, coarse, mask, _alpha, _rgb, _depth, _position = map_renderer._render_pose(
                    sample_name,
                    pose,
                    require_grad=False,
                )
                feat = coarse if score_feature == "coarse" else fine
                feat_list.append(feat.squeeze(0).detach())
                mask_list.append(mask.squeeze(0).detach())
    return torch.stack(feat_list, dim=0), torch.stack(mask_list, dim=0)


def _feature_metric_refine_pose(
    map_renderer: MapFeatureRenderer,
    sample_name: str,
    pose_init: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    score_feature: str,
    steps: int,
    damping: float,
    update_scale: float,
):
    pose_cur = pose_init.view(1, 4, 4).to(map_renderer.device).float()
    if steps <= 0:
        return pose_cur[0]
    normalized = map_renderer._normalize_name(sample_name)
    intr = map_renderer.name_to_intr[normalized]
    query = query_feat.unsqueeze(0).to(map_renderer.device).float()
    with torch.no_grad():
        for _step in range(int(steps)):
            _fine_raw, fine, coarse, mask, _alpha, _rgb, depth, _position = map_renderer._render_pose(
                sample_name,
                pose_cur[0],
                require_grad=False,
            )
            ref = coarse if score_feature == "coarse" else fine
            if query.shape[-2:] != ref.shape[-2:]:
                query_step = F.interpolate(query, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            else:
                query_step = query
            depth_step = depth[:, 0] if depth.ndim == 4 else depth
            delta, _residual = feature_metric_solve(
                query_step,
                ref.float(),
                depth_step.float(),
                intr,
                damping=float(damping),
                valid_mask=mask.float(),
            )
            pose_cur = apply_pose_delta(pose_cur, delta, scale=float(update_scale))
    return pose_cur[0]


def _prepare_datasets_and_model(cfg, checkpoint_path: str, device: torch.device):
    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)

    retrieval_cfg = cfg.get("retrieval", {})
    retrieval_store = None
    if retrieval_cfg.get("enabled", False):
        retrieval_store = RetrievalTeacherStore(
            retrieval_cfg["feature_dir"],
            subdir=retrieval_cfg.get("teacher_subdir", "cls"),
            cache_in_memory=bool(retrieval_cfg.get("cache_teacher", False)),
        )
        cfg["retrieval"]["student_dim"] = retrieval_store.feature_dim

    all_records = build_all_records(
        cfg["dataset"],
        teacher_store,
        allow_synthetic=bool(cfg["dataset"].get("synthetic_if_missing", False)),
    )
    train_records, val_records = split_records(all_records, cfg["dataset"])
    colmap_dir = (
        cfg["dataset"].get("colmap_dir")
        or cfg.get("map_supervision", {}).get("colmap_dir")
        or str(Path(cfg["dataset"]["source_dir"]) / "sparse" / "0")
    )
    train_dataset = JointRADIOQueryDataset(
        train_records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
        colmap_dir=colmap_dir,
    )
    val_dataset = JointRADIOQueryDataset(
        val_records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
        colmap_dir=colmap_dir,
    )

    model_cfg = cfg["model"]
    if bool(model_cfg.get("pose_init_head", False)) and model_cfg.get("pose_init_anchor_centers") is None:
        anchors, anchor_rotmats = pose_anchors_from_dataset(
            train_dataset,
            num_anchors=int(model_cfg.get("pose_init_num_anchors", 64)),
            sampling=str(model_cfg.get("pose_init_anchor_sampling", "fps")),
        )
        model_cfg["pose_init_anchor_centers"] = anchors.tolist()
        model_cfg["pose_init_anchor_rotmats"] = anchor_rotmats.tolist()

    model = build_radio_query_student(
        cfg,
        fine_feature_dim=fine_dim,
        coarse_feature_dim=coarse_dim,
        retrieval_dim=int(cfg["retrieval"]["student_dim"]) if retrieval_store is not None else None,
        retrieval_hidden_dim=int(cfg["retrieval"].get("hidden_dim", 0)) if retrieval_store is not None else None,
    ).to(device)
    checkpoint = safe_torch_load(checkpoint_path)
    load_model_warmstart(model, checkpoint, strict=False)
    model.eval()
    return train_dataset, val_dataset, model, checkpoint


def _pose_for_record(dataset: JointRADIOQueryDataset, record: dict):
    sample_name = record["sample_name"].replace("\\", "/")
    pose = dataset.name_to_pose.get(sample_name)
    if pose is None:
        pose = dataset.basename_to_pose.get(Path(sample_name).name)
    return pose


def _build_retrieval_pose_bank(train_dataset: JointRADIOQueryDataset, device: torch.device):
    if train_dataset.retrieval_teacher_store is None:
        raise ValueError("candidate-source=retrieval_train requires retrieval.enabled=true")
    descriptors = []
    poses = []
    names = []
    for record in train_dataset.records:
        pose = _pose_for_record(train_dataset, record)
        if pose is None:
            continue
        descriptors.append(train_dataset.retrieval_teacher_store.load_record(record))
        poses.append(pose)
        names.append(record["sample_name"])
    if not descriptors:
        raise RuntimeError("No train retrieval descriptors with COLMAP poses were found.")
    return (
        torch.stack(descriptors, dim=0).to(device),
        torch.stack(poses, dim=0).to(device),
        names,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument(
        "--candidate-source",
        default="pose_init",
        choices=["pose_init", "retrieval_train"],
        help="Use pose-init internal hypotheses or retrieval top-K train poses.",
    )
    parser.add_argument(
        "--candidate-strategy",
        default="score",
        choices=["all", "score", "uniform"],
        help="Which pose-init hypotheses to render-score.",
    )
    parser.add_argument("--render-hw", default=None, help="Optional DCFF render H,W. Defaults to teacher feature HW.")
    parser.add_argument("--score-hw", default=None, help="Optional feature scoring H,W after query/render resize.")
    parser.add_argument("--query-fine-key", default=None)
    parser.add_argument("--score-feature", default="fine", choices=["fine", "coarse"])
    parser.add_argument("--score-mode", default="global", choices=["global", "local"])
    parser.add_argument("--local-radius", type=int, default=6)
    parser.add_argument(
        "--score-preprocess",
        default="none",
        choices=["none", "spatial_center", "spatial_zscore", "highpass"],
    )
    parser.add_argument("--highpass-kernel", type=int, default=5)
    parser.add_argument("--append-gt", action="store_true", help="Append GT pose as a scoring upper-bound diagnostic.")
    parser.add_argument("--render-chunk-size", type=int, default=8)
    parser.add_argument("--feature-metric-steps", type=int, default=0)
    parser.add_argument("--feature-metric-damping", type=float, default=0.01)
    parser.add_argument("--feature-metric-update-scale", type=float, default=0.25)
    parser.add_argument(
        "--protocol",
        default="single_frame_real_init",
        choices=["single_frame_real_init", "temporal_prev_gt_oracle", "temporal_prev_pred_tracking"],
        help="Evaluation protocol. Temporal protocols process records in sequence/frame order.",
    )
    parser.add_argument(
        "--temporal-init-mode",
        default="prev_pose",
        choices=["prev_pose", "constant_velocity"],
        help="Temporal initializer for non-start frames.",
    )
    parser.add_argument(
        "--temporal-grid-trans",
        default=None,
        help="Optional comma-separated world X/Z translation offsets in meters for non-start temporal frames.",
    )
    parser.add_argument(
        "--temporal-grid-yaw-deg",
        default=None,
        help="Optional comma-separated yaw offsets in degrees for non-start temporal frames.",
    )
    parser.add_argument("--map-config", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("eval_poseinit_render_score")
    device = torch.device(args.device if args.device.startswith("cpu") or torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    cfg.setdefault("training", {})["device"] = str(device)
    map_cfg = cfg.setdefault("map_supervision", {})
    map_cfg["enabled"] = True
    map_cfg["cache_rendered"] = False
    map_cfg["trainable"] = False
    if args.map_config:
        map_cfg["config_path"] = args.map_config
    render_hw_arg = _parse_hw(args.render_hw)
    score_hw = _parse_hw(args.score_hw)
    temporal_grid_trans = _parse_float_list(args.temporal_grid_trans)
    temporal_grid_yaw = _parse_float_list(args.temporal_grid_yaw_deg)
    use_temporal_grid = bool(temporal_grid_trans) or bool(temporal_grid_yaw)
    if use_temporal_grid:
        if not temporal_grid_trans:
            temporal_grid_trans = [0.0]
        if not temporal_grid_yaw:
            temporal_grid_yaw = [0.0]

    train_dataset, val_dataset, model, checkpoint = _prepare_datasets_and_model(cfg, args.checkpoint, device)
    val_dataset.records = _records_for_protocol(val_dataset.records, args.protocol)
    train_bank_desc = None
    train_bank_pose = None
    train_bank_names = None
    if args.candidate_source == "retrieval_train":
        train_bank_desc, train_bank_pose, train_bank_names = _build_retrieval_pose_bank(train_dataset, device)
        logger.info("Retrieval train pose bank loaded: %d poses", train_bank_pose.shape[0])
    render_hw = render_hw_arg or tuple(cfg["dataset"].get("teacher_feature_hw") or cfg["dataset"]["feature_hw"])
    map_renderer = MapFeatureRenderer(cfg, feature_hw=render_hw, device=device, logger=logger)
    if checkpoint.get("map_renderer_state_dict"):
        map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
    map_renderer.set_train_mode(False)

    query_fine_key = args.query_fine_key or cfg.get("map_supervision", {}).get("query_fine_key") or "fine"
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    baseline_errors = []
    render_score_errors = []
    refined_errors = []
    oracle_errors = []
    protocol_samples = []
    per_sample = []
    temporal_selector = TemporalInitSelector(protocol=args.protocol, init_mode=args.temporal_init_mode)
    with torch.no_grad():
        for sample_idx, batch in enumerate(loader):
            if sample_idx >= int(args.max_samples):
                break
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["rgb"])
            if "pose_init" not in outputs:
                raise RuntimeError("Model output has no pose_init branch.")
            pose_init = outputs["pose_init"]
            pose_gt = batch["pose_gt"][0].to(device)
            sample_name = batch["sample_name"][0]

            baseline_pose = pose_init["pose_w2c"][0, 0]
            baseline_err = _pose_error_mm_deg(baseline_pose, pose_gt)
            baseline_errors.append(baseline_err)
            protocol_init_pose, protocol_meta = temporal_selector.select(sample_name, baseline_pose.detach())
            use_temporal_init = (
                args.protocol != "single_frame_real_init"
                and not bool(protocol_meta.get("is_sequence_start", False))
            )

            if use_temporal_init:
                if use_temporal_grid:
                    candidate_pose = build_temporal_pose_grid(
                        protocol_init_pose.to(device),
                        trans_offsets_m=temporal_grid_trans,
                        yaw_offsets_deg=temporal_grid_yaw,
                    )
                    candidate_idx = torch.arange(
                        -candidate_pose.shape[0],
                        0,
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    candidate_pose = protocol_init_pose.view(1, 4, 4).to(device)
                    candidate_idx = torch.full((1,), -2, device=device, dtype=torch.long)
            elif args.candidate_source == "retrieval_train":
                if "retrieval" not in outputs:
                    raise RuntimeError("Model output has no retrieval descriptor for retrieval_train candidates.")
                top_idx, top_scores = topk_descriptor_candidate_indices(
                    outputs["retrieval"],
                    train_bank_desc,
                    k=args.num_candidates,
                )
                candidate_idx = top_idx[0].to(device)
                candidate_pose = train_bank_pose.index_select(0, candidate_idx)
                all_scores = top_scores[0].to(device)
            else:
                if "all_center" in pose_init and "all_rotmat" in pose_init:
                    all_pose = AbsolutePoseInitHead._centers_rot_to_w2c(
                        pose_init["all_center"],
                        pose_init["all_rotmat"],
                    )[0]
                    all_scores = pose_init.get("anchor_logits", None)
                    all_scores = all_scores[0] if all_scores is not None else None
                else:
                    all_pose = pose_init["pose_w2c"][0]
                    all_scores = pose_init.get("scores", None)
                    all_scores = all_scores[0] if all_scores is not None else None

                candidate_idx = select_pose_candidate_indices(
                    all_pose.shape[0],
                    limit=args.num_candidates,
                    scores=all_scores,
                    strategy=args.candidate_strategy,
                ).to(device)
                candidate_pose = all_pose.index_select(0, candidate_idx)
            if args.append_gt:
                candidate_pose = torch.cat([candidate_pose, pose_gt.view(1, 4, 4)], dim=0)
                candidate_idx = torch.cat(
                    [candidate_idx, torch.full((1,), -1, device=device, dtype=candidate_idx.dtype)],
                    dim=0,
                )

            candidate_errors = [_pose_error_mm_deg(pose, pose_gt) for pose in candidate_pose]
            oracle_cost = torch.tensor(
                [err[0] / 1000.0 + 0.1 * (err[1] / 180.0) for err in candidate_errors],
                device=device,
            )
            oracle_local_idx = int(oracle_cost.argmin().item())
            oracle_err = candidate_errors[oracle_local_idx]
            oracle_errors.append(oracle_err)

            rendered_feat, rendered_mask = _render_candidate_features(
                map_renderer,
                sample_name,
                candidate_pose,
                score_feature=args.score_feature,
                chunk_size=args.render_chunk_size,
            )
            if args.score_feature == "coarse":
                query_feat = outputs["coarse"][0]
            else:
                query_feat, query_key_used = _resolve_query_feature(outputs, query_fine_key)
                query_feat = query_feat[0]
            query_feat = _resize_feature(query_feat.unsqueeze(0), score_hw).squeeze(0)
            rendered_feat = _resize_feature(rendered_feat, score_hw)
            rendered_mask = _resize_feature(rendered_mask, score_hw)
            if args.score_mode == "local":
                score_result = local_render_score_feature_candidates(
                    query_feat,
                    rendered_feat,
                    mask=rendered_mask,
                    radius=args.local_radius,
                    preprocess=args.score_preprocess,
                    highpass_kernel=args.highpass_kernel,
                )
            else:
                score_result = render_score_feature_candidates(
                    query_feat,
                    rendered_feat,
                    mask=rendered_mask,
                    preprocess=args.score_preprocess,
                    highpass_kernel=args.highpass_kernel,
                )
            best_local_idx = int(score_result["best_idx"].item())
            render_err = candidate_errors[best_local_idx]
            render_score_errors.append(render_err)
            refined_err = None
            final_pose = candidate_pose[best_local_idx]
            final_err = render_err
            if int(args.feature_metric_steps) > 0:
                refined_pose = _feature_metric_refine_pose(
                    map_renderer,
                    sample_name,
                    candidate_pose[best_local_idx],
                    query_feat,
                    score_feature=args.score_feature,
                    steps=int(args.feature_metric_steps),
                    damping=float(args.feature_metric_damping),
                    update_scale=float(args.feature_metric_update_scale),
                )
                refined_err = _pose_error_mm_deg(refined_pose, pose_gt)
                refined_errors.append(refined_err)
                final_pose = refined_pose
                final_err = refined_err
            temporal_selector.update(sample_name, gt_pose=pose_gt, final_pose=final_pose)
            protocol_samples.append(
                {
                    "sample_name": sample_name,
                    "sequence": protocol_meta.get("sequence"),
                    "frame": protocol_meta.get("frame"),
                    "source": protocol_meta.get("source"),
                    "is_sequence_start": bool(protocol_meta.get("is_sequence_start", False)),
                    "init_trans_mm": render_err[0],
                    "init_rot_deg": render_err[1],
                    "final_trans_mm": final_err[0],
                    "final_rot_deg": final_err[1],
                }
            )
            gt_score = None
            gt_score_rank = None
            if args.append_gt:
                scores = score_result["scores"]
                gt_score_t = scores[-1]
                gt_score = float(gt_score_t.item())
                gt_score_rank = int((scores > gt_score_t).sum().item() + 1)

            per_sample.append(
                {
                    "sample_name": sample_name,
                    "protocol": args.protocol,
                    "temporal_sequence": protocol_meta.get("sequence"),
                    "temporal_frame": protocol_meta.get("frame"),
                    "temporal_source": protocol_meta.get("source"),
                    "is_sequence_start": bool(protocol_meta.get("is_sequence_start", False)),
                    "baseline_trans_mm": baseline_err[0],
                    "baseline_rot_deg": baseline_err[1],
                    "render_score_trans_mm": render_err[0],
                    "render_score_rot_deg": render_err[1],
                    "refined_trans_mm": refined_err[0] if refined_err is not None else None,
                    "refined_rot_deg": refined_err[1] if refined_err is not None else None,
                    "oracle_trans_mm": oracle_err[0],
                    "oracle_rot_deg": oracle_err[1],
                    "best_candidate_idx": int(candidate_idx[best_local_idx].item()),
                    "oracle_candidate_idx": int(candidate_idx[oracle_local_idx].item()),
                    "best_candidate_name": (
                        train_bank_names[int(candidate_idx[best_local_idx].item())]
                        if train_bank_names is not None and int(candidate_idx[best_local_idx].item()) >= 0
                        else None
                    ),
                    "oracle_candidate_name": (
                        train_bank_names[int(candidate_idx[oracle_local_idx].item())]
                        if train_bank_names is not None and int(candidate_idx[oracle_local_idx].item()) >= 0
                        else None
                    ),
                    "best_score": float(score_result["best_score"].item()),
                    "score_margin": float(score_result["score_margin"].item()),
                    "mean_valid_fraction": float(score_result["valid_fraction"].mean().item()),
                    "gt_score": gt_score,
                    "gt_score_rank": gt_score_rank,
                }
            )
            logger.info(
                "%03d %s | head %.1fmm/%.2fdeg | render %.1fmm/%.2fdeg | refined %s | oracle %.1fmm/%.2fdeg",
                sample_idx,
                sample_name,
                baseline_err[0],
                baseline_err[1],
                render_err[0],
                render_err[1],
                (
                    f"{refined_err[0]:.1f}mm/{refined_err[1]:.2f}deg"
                    if refined_err is not None
                    else "n/a"
                ),
                oracle_err[0],
                oracle_err[1],
            )

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "score_feature": args.score_feature,
        "score_mode": args.score_mode,
        "local_radius": int(args.local_radius),
        "score_preprocess": args.score_preprocess,
        "append_gt": bool(args.append_gt),
        "query_fine_key": query_fine_key,
        "candidate_source": args.candidate_source,
        "candidate_strategy": args.candidate_strategy,
        "num_candidates": int(args.num_candidates),
        "render_hw": list(render_hw),
        "score_hw": list(score_hw) if score_hw is not None else None,
        "protocol": args.protocol,
        "temporal_init_mode": args.temporal_init_mode,
        "temporal_grid_trans": temporal_grid_trans,
        "temporal_grid_yaw_deg": temporal_grid_yaw,
        "num_samples": len(per_sample),
        "metrics": {
            "pose_head_top1": _summarize_error_pairs(baseline_errors),
            "render_score": _summarize_error_pairs(render_score_errors),
            "feature_metric_refined": _summarize_error_pairs(refined_errors),
            "candidate_oracle": _summarize_error_pairs(oracle_errors),
            "protocol": summarize_temporal_errors(protocol_samples),
        },
        "samples": per_sample,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
