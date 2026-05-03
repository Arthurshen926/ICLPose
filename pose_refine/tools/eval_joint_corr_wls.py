#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import add_pose_noise
from feature_extract.train_impl import (
    JointRADIOQueryDataset,
    MapFeatureRenderer,
    TeacherFeatureStore,
    augment_feature_with_scene_coord,
    build_all_records,
    build_radio_query_student,
    compute_w2c_flow,
    local_correlation_feature_preprocess,
    load_config,
    move_batch_to_device,
    normalize_scene_coord_map,
    pose_error_tensors,
    resolve_query_feature_dims,
    scene_coord_center_scale,
    safe_torch_load,
    shifted_local_correlation,
    split_records,
)
from pose_refine import apply_pose_delta, compute_image_jacobian, diff_pose_solve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    radius = int(radius)
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx.reshape(1, -1, 1, 1), yy.reshape(1, -1, 1, 1)


def decode_local_correlation(
    corr: torch.Tensor,
    *,
    radius: int,
    temperature: float = 0.05,
    flow_decode_mode: str = "softargmax",
    confidence_mode: str = "max",
    confidence_margin_temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Decode a local correlation volume into rendered->query flow and confidence."""
    radius = int(radius)
    flow_decode = str(flow_decode_mode or "softargmax").lower()
    if flow_decode not in {"softargmax", "argmax", "argmax_st"}:
        raise ValueError(f"Unsupported flow_decode_mode={flow_decode_mode!r}")
    conf_mode = str(confidence_mode or "max").lower()
    if conf_mode not in {"max", "margin"}:
        raise ValueError(f"Unsupported confidence_mode={confidence_mode!r}")

    B, C, H, W = corr.shape
    expected = (2 * radius + 1) ** 2
    if C != expected:
        raise ValueError(f"corr has {C} channels, expected {expected} for radius={radius}")

    corr_f = corr.float()
    dx, dy = local_offsets(radius, corr.device, corr_f.dtype)
    probs = torch.softmax(corr_f / max(float(temperature), 1e-6), dim=1)
    soft_flow = torch.cat(
        [
            (probs * dx).sum(dim=1, keepdim=True),
            (probs * dy).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    hard_idx = corr_f.argmax(dim=1, keepdim=True)
    hard_flow = torch.cat(
        [
            dx.expand(B, -1, H, W).gather(1, hard_idx),
            dy.expand(B, -1, H, W).gather(1, hard_idx),
        ],
        dim=1,
    )
    if flow_decode == "argmax":
        flow = hard_flow
    elif flow_decode == "argmax_st":
        flow = hard_flow + (soft_flow - soft_flow.detach())
    else:
        flow = soft_flow

    top2 = torch.topk(corr_f, k=2, dim=1).values
    peak_top1 = top2[:, 0:1]
    peak_top2 = top2[:, 1:2]
    peak_margin = peak_top1 - peak_top2
    if conf_mode == "margin":
        confidence = torch.sigmoid(peak_margin / max(float(confidence_margin_temperature), 1e-6))
    else:
        confidence = probs.max(dim=1, keepdim=True).values

    metrics = {
        "peak_top1": peak_top1.mean().detach(),
        "peak_top2": peak_top2.mean().detach(),
        "peak_margin": peak_margin.mean().detach(),
        "confidence": confidence.mean().detach(),
    }
    return flow.to(dtype=corr.dtype), confidence.to(dtype=corr.dtype), metrics


def apply_confidence_filter(
    confidence: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    top_fraction: float = 0.0,
    power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply WLS confidence power/threshold/top-k filtering per image."""
    conf = confidence.float().clamp(min=0.0)
    valid = torch.ones_like(conf)
    if valid_mask is not None:
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != conf.shape[-2:]:
            valid = F.interpolate(valid, size=conf.shape[-2:], mode="nearest")
        valid = (valid > 0.0).float()

    if abs(float(power) - 1.0) > 1e-6:
        conf = conf.clamp(max=1.0).pow(float(power))
    if float(threshold) > 0.0:
        conf = conf * (conf >= float(threshold)).float()

    frac = float(top_fraction)
    if 0.0 < frac < 1.0:
        filtered = torch.zeros_like(conf)
        for b in range(conf.shape[0]):
            valid_b = valid[b : b + 1] > 0.0
            values = conf[b : b + 1][valid_b]
            if values.numel() == 0:
                continue
            keep_count = max(1, int(math.ceil(values.numel() * frac)))
            kth = torch.topk(values, k=keep_count, largest=True).values[-1]
            keep = valid_b & (conf[b : b + 1] >= kth)
            filtered[b : b + 1] = conf[b : b + 1] * keep.float()
        conf = filtered

    conf = conf * valid
    coverage = ((conf > 0.0).float() * valid).sum() / valid.sum().clamp(min=1.0)
    return conf, coverage.detach()


def compute_query_map_flow(
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    flow_decode_mode: str,
    confidence_mode: str,
    confidence_margin_temperature: float,
    matcher: torch.nn.Module | None = None,
    flow_head: torch.nn.Module | None = None,
    depth: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    feature_preprocess: str = "none",
    highpass_kernel: int = 3,
    highpass_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    query = query_feat.float()
    if query.shape[-2:] != rendered_feat.shape[-2:]:
        query = F.interpolate(query, size=rendered_feat.shape[-2:], mode="bilinear", align_corners=False)
    rendered_corr = local_correlation_feature_preprocess(
        rendered_feat.float(),
        mode=feature_preprocess,
        highpass_kernel=highpass_kernel,
        highpass_scale=highpass_scale,
    )
    query_corr = local_correlation_feature_preprocess(
        query,
        mode=feature_preprocess,
        highpass_kernel=highpass_kernel,
        highpass_scale=highpass_scale,
    )
    rendered_n = F.normalize(rendered_corr, dim=1)
    query_n = F.normalize(query_corr, dim=1)
    corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()
    if matcher is not None:
        corr = matcher(corr, depth=depth, valid_mask=valid_mask).float()
    if flow_head is not None:
        pred = flow_head(corr, depth=depth, valid_mask=valid_mask)
        flow = pred["flow"].float()
        confidence = pred.get("confidence")
        if confidence is None:
            confidence = torch.ones(flow.shape[0], 1, flow.shape[2], flow.shape[3], device=flow.device)
        return flow, confidence.float(), {
            "peak_top1": corr.max(dim=1, keepdim=True).values.mean().detach(),
            "peak_top2": torch.topk(corr, k=2, dim=1).values[:, 1:2].mean().detach(),
            "peak_margin": (
                torch.topk(corr, k=2, dim=1).values[:, 0:1]
                - torch.topk(corr, k=2, dim=1).values[:, 1:2]
            ).mean().detach(),
            "confidence": confidence.float().mean().detach(),
            "explicit_flow": torch.ones((), device=corr.device),
        }
    flow, confidence, metrics = decode_local_correlation(
        corr,
        radius=radius,
        temperature=temperature,
        flow_decode_mode=flow_decode_mode,
        confidence_mode=confidence_mode,
        confidence_margin_temperature=confidence_margin_temperature,
    )
    metrics["explicit_flow"] = torch.zeros((), device=corr.device)
    return flow.float(), confidence.float(), metrics


def maybe_project_local_corr_features(
    model: torch.nn.Module,
    map_cfg: dict,
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the learned local-correlation projector with train/eval parity."""
    if not bool(map_cfg.get("local_corr_projector_enabled", False)):
        return rendered_feat, query_feat
    projector = getattr(model, "local_corr_projector", None)
    if projector is None:
        return rendered_feat, query_feat

    rendered_projected = projector(rendered_feat)
    if bool(map_cfg.get("local_corr_projector_apply_to_query", True)):
        query_feat = projector(query_feat)
    return rendered_projected, query_feat


def tensor_intrinsics_to_dict(intr: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "fx": intr[:, 0],
        "fy": intr[:, 1],
        "cx": intr[:, 2],
        "cy": intr[:, 3],
    }


def stable_sample_seed(base_seed: int, sample_name: str) -> int:
    payload = f"{int(base_seed)}:{sample_name}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def make_initial_pose(
    pose_gt: torch.Tensor,
    noise_deg: float,
    noise_m: float,
    device: torch.device,
    *,
    sample_names: list[str] | None = None,
    base_seed: int = 0,
) -> torch.Tensor:
    poses = []
    np_state = np.random.get_state()
    try:
        for idx, pose in enumerate(pose_gt.detach().cpu()):
            if sample_names is not None:
                np.random.seed(stable_sample_seed(int(base_seed), str(sample_names[idx])))
            noisy = add_pose_noise(pose.numpy(), rot_deg=float(noise_deg), trans_m=float(noise_m))
            poses.append(torch.from_numpy(noisy).float())
    finally:
        np.random.set_state(np_state)
    return torch.stack(poses, dim=0).to(device)


def select_record_window(records, *, sample_offset: int = 0, max_samples: int = 0):
    start = max(0, int(sample_offset))
    if int(max_samples) > 0:
        end = start + int(max_samples)
        return list(records[start:end])
    return list(records[start:])


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    set_seed(int(args.seed))

    cfg = load_config(args.config)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["num_workers"] = int(args.num_workers)
    checkpoint = safe_torch_load(args.checkpoint)
    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    all_records = build_all_records(
        cfg["dataset"],
        teacher_store,
        allow_synthetic=bool(cfg["dataset"].get("synthetic_if_missing", False)),
    )
    train_records, val_records = split_records(all_records, cfg["dataset"])
    records = train_records if args.split == "train" else val_records
    records = select_record_window(
        records,
        sample_offset=int(args.sample_offset),
        max_samples=int(args.max_samples),
    )

    dataset = JointRADIOQueryDataset(
        records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = build_radio_query_student(
        cfg,
        fine_feature_dim=fine_dim,
        coarse_feature_dim=coarse_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    map_renderer = MapFeatureRenderer(
        cfg,
        feature_hw=tuple(cfg["dataset"]["feature_hw"]),
        device=device,
        logger=type("_Logger", (), {"info": staticmethod(lambda *a, **k: None)})(),
    )
    map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
    map_renderer.set_train_mode(False)

    query_fine_key = str(cfg.get("map_supervision", {}).get("query_fine_key", "fine"))
    map_cfg = cfg.get("map_supervision", {})
    query_corr_scene_coord_weight = float(map_cfg.get("query_corr_scene_coord_weight", 0.0))
    scene_center, scene_scale = scene_coord_center_scale(map_cfg, device, dtype=torch.float32)
    radius = int(args.radius if args.radius is not None else map_cfg.get("query_corr_radius", 4))
    temperature = float(args.temperature if args.temperature is not None else map_cfg.get("query_corr_temperature", 0.05))
    flow_decode_mode = str(args.flow_decode_mode or map_cfg.get("query_corr_flow_decode_mode", "softargmax"))
    feature_preprocess = str(map_cfg.get("query_corr_feature_preprocess", "none"))
    highpass_kernel = int(map_cfg.get("query_corr_highpass_kernel", 3))
    highpass_scale = float(map_cfg.get("query_corr_highpass_scale", 1.0))
    damping = float(args.damping)
    update_scale = float(args.update_scale)

    init_rot_all, init_trans_all = [], []
    final_rot_all, final_trans_all = [], []
    flow_epe_all, flow_cos_all, flow_mag_all, gt_flow_mag_all = [], [], [], []
    conf_all, conf_cov_all, peak_margin_all = [], [], []
    per_sample = []

    for batch in tqdm(loader, desc="joint-corr-wls", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["rgb"])
        query_fine = outputs[query_fine_key].float()
        pose_gt = []
        intrinsics = []
        normalized_names = []
        for sample_name in batch["sample_name"]:
            normalized = map_renderer._normalize_name(sample_name)
            normalized_names.append(normalized)
            pose_gt.append(map_renderer.name_to_pose[normalized].to(device))
            intr = map_renderer.name_to_intr[normalized]
            intrinsics.append([intr["fx"], intr["fy"], intr["cx"], intr["cy"]])
        pose_gt = torch.stack(pose_gt, dim=0).float()
        intrinsics = torch.tensor(intrinsics, device=device, dtype=torch.float32)
        pose_cur = make_initial_pose(
            pose_gt,
            args.noise_deg,
            args.noise_m,
            device,
            sample_names=normalized_names,
            base_seed=int(args.seed),
        )

        _init_rot_loss, init_rot, init_trans = pose_error_tensors(pose_cur, pose_gt)
        init_rot_all.extend(init_rot.cpu().tolist())
        init_trans_all.extend((init_trans * 1000.0).cpu().tolist())
        last_sample_metrics = {}

        for _ in range(int(args.outer_iters)):
            rendered_list, depth_list, valid_list, position_list = [], [], [], []
            for sample_name, pose in zip(batch["sample_name"], pose_cur):
                _fine_raw, fine, _coarse, mask, _alpha, _rgb, depth, position = map_renderer._render_pose(
                    sample_name,
                    pose,
                    require_grad=False,
                )
                rendered_list.append(fine.squeeze(0))
                depth_list.append(depth.squeeze(0))
                valid_list.append(mask.squeeze(0))
                position_list.append(position.squeeze(0))
            rendered = torch.stack(rendered_list, dim=0).float()
            depth = torch.stack(depth_list, dim=0).float()
            valid_mask = torch.stack(valid_list, dim=0).float()
            position = torch.stack(position_list, dim=0).float()
            if depth.ndim == 4 and depth.shape[1] == 1:
                depth_s = depth[:, 0]
            elif depth.ndim == 3:
                depth_s = depth
            else:
                depth_s = depth.squeeze(1)

            corr_rendered = rendered
            corr_query = query_fine
            if query_corr_scene_coord_weight > 0.0 and "scene_coord" in outputs:
                rendered_scene = normalize_scene_coord_map(
                    position,
                    scene_center,
                    scene_scale,
                    target_hw=rendered.shape[-2:],
                )
                corr_rendered = augment_feature_with_scene_coord(
                    rendered,
                    rendered_scene,
                    query_corr_scene_coord_weight,
                )
                corr_query = augment_feature_with_scene_coord(
                    query_fine,
                    outputs["scene_coord"],
                    query_corr_scene_coord_weight,
                )
            elif bool(map_cfg.get("local_corr_projector_enabled", False)) and getattr(
                model,
                "local_corr_projector",
                None,
            ) is not None:
                corr_rendered, corr_query = maybe_project_local_corr_features(
                    model,
                    map_cfg,
                    corr_rendered,
                    corr_query,
                )

            flow, confidence, corr_metrics = compute_query_map_flow(
                corr_rendered,
                corr_query,
                radius=radius,
                temperature=temperature,
                flow_decode_mode=flow_decode_mode,
                confidence_mode=args.confidence_mode,
                confidence_margin_temperature=args.confidence_margin_temperature,
                matcher=getattr(model, "local_matcher", None)
                if bool(map_cfg.get("local_matcher_enabled", False))
                else None,
                flow_head=getattr(model, "local_flow_head", None)
                if bool(map_cfg.get("local_flow_head_enabled", False))
                else None,
                depth=depth,
                valid_mask=valid_mask,
                feature_preprocess=feature_preprocess,
                highpass_kernel=highpass_kernel,
                highpass_scale=highpass_scale,
            )
            confidence, conf_cov = apply_confidence_filter(
                confidence,
                valid_mask=valid_mask,
                threshold=float(args.confidence_threshold),
                top_fraction=float(args.confidence_top_fraction),
                power=float(args.confidence_power),
            )

            gt_flow, gt_valid = [], []
            for pose, gt, depth_i, intr in zip(pose_cur, pose_gt, depth_s, intrinsics):
                intr_dict = {
                    "fx": float(intr[0].item()),
                    "fy": float(intr[1].item()),
                    "cx": float(intr[2].item()),
                    "cy": float(intr[3].item()),
                }
                flow_i, valid_i = compute_w2c_flow(
                    pose.unsqueeze(0),
                    gt.unsqueeze(0),
                    depth_i.unsqueeze(0),
                    intr_dict,
                    target_hw=rendered.shape[-2:],
                )
                gt_flow.append(flow_i.squeeze(0))
                gt_valid.append(valid_i.squeeze(0))
            gt_flow = torch.stack(gt_flow, dim=0).float()
            gt_valid = torch.stack(gt_valid, dim=0).float() * valid_mask.float()

            metric_w = gt_valid.clamp(min=0.0)
            denom = metric_w.sum().clamp(min=1.0)
            epe = torch.linalg.norm(flow - gt_flow, dim=1, keepdim=True)
            flow_mag = torch.linalg.norm(flow, dim=1, keepdim=True)
            gt_mag = torch.linalg.norm(gt_flow, dim=1, keepdim=True)
            flow_cos = (flow * gt_flow).sum(dim=1, keepdim=True) / (flow_mag * gt_mag).clamp(min=1e-6)
            flow_epe_all.append(float((epe * metric_w).sum().item() / denom.item()))
            flow_mag_all.append(float((flow_mag * metric_w).sum().item() / denom.item()))
            gt_flow_mag_all.append(float((gt_mag * metric_w).sum().item() / denom.item()))
            flow_cos_all.append(float((flow_cos * metric_w).sum().item() / denom.item()))
            conf_all.append(float((confidence * metric_w).sum().item() / denom.item()))
            conf_cov_all.append(float(conf_cov.item()))
            peak_margin_all.append(float(corr_metrics["peak_margin"].item()))
            if bool(args.save_per_sample):
                denom_b = metric_w.flatten(1).sum(dim=1).clamp(min=1.0)
                last_sample_metrics = {
                    "flow_epe_px": ((epe * metric_w).flatten(1).sum(dim=1) / denom_b).detach().cpu().tolist(),
                    "flow_cos": ((flow_cos * metric_w).flatten(1).sum(dim=1) / denom_b).detach().cpu().tolist(),
                    "flow_mag_px": ((flow_mag * metric_w).flatten(1).sum(dim=1) / denom_b).detach().cpu().tolist(),
                    "gt_flow_mag_px": ((gt_mag * metric_w).flatten(1).sum(dim=1) / denom_b).detach().cpu().tolist(),
                    "confidence_mean": ((confidence * metric_w).flatten(1).sum(dim=1) / denom_b).detach().cpu().tolist(),
                    "confidence_coverage": [float(conf_cov.item())] * int(metric_w.shape[0]),
                    "peak_margin": [float(corr_metrics["peak_margin"].item())] * int(metric_w.shape[0]),
                }

            Ju, Jv, depth_valid = compute_image_jacobian(depth_s.float(), intrinsics)
            delta_xi = diff_pose_solve(
                flow.float(),
                confidence.float(),
                Ju,
                Jv,
                depth_valid,
                damping=damping,
                irls_iters=int(args.irls_iters),
                pixel_stride=int(args.pixel_stride),
                robust_kernel=str(args.robust_kernel),
            )
            pose_cur = apply_pose_delta(pose_cur.float(), delta_xi.float(), scale=update_scale)

        _final_rot_loss, final_rot, final_trans = pose_error_tensors(pose_cur, pose_gt)
        final_rot_all.extend(final_rot.cpu().tolist())
        final_trans_all.extend((final_trans * 1000.0).cpu().tolist())
        if bool(args.save_per_sample):
            for idx, sample_name in enumerate(normalized_names):
                item = {
                    "sample_name": str(sample_name),
                    "init_rot_deg": float(init_rot[idx].detach().cpu().item()),
                    "init_trans_mm": float((init_trans[idx] * 1000.0).detach().cpu().item()),
                    "final_rot_deg": float(final_rot[idx].detach().cpu().item()),
                    "final_trans_mm": float((final_trans[idx] * 1000.0).detach().cpu().item()),
                    "trans_gain_mm": float(((init_trans[idx] - final_trans[idx]) * 1000.0).detach().cpu().item()),
                }
                for key, values in last_sample_metrics.items():
                    item[key] = float(values[idx])
                per_sample.append(item)

    result = {
        "samples": int(len(final_trans_all)),
        "init_rot_med_deg": float(np.median(init_rot_all)),
        "init_trans_med_mm": float(np.median(init_trans_all)),
        "final_rot_med_deg": float(np.median(final_rot_all)),
        "final_trans_med_mm": float(np.median(final_trans_all)),
        "trans_gain_med_mm": float(np.median(init_trans_all) - np.median(final_trans_all)),
        "flow_epe_mean_px": float(np.mean(flow_epe_all)) if flow_epe_all else 0.0,
        "flow_cos_mean": float(np.mean(flow_cos_all)) if flow_cos_all else 0.0,
        "flow_mag_mean_px": float(np.mean(flow_mag_all)) if flow_mag_all else 0.0,
        "gt_flow_mag_mean_px": float(np.mean(gt_flow_mag_all)) if gt_flow_mag_all else 0.0,
        "confidence_mean": float(np.mean(conf_all)) if conf_all else 0.0,
        "confidence_coverage": float(np.mean(conf_cov_all)) if conf_cov_all else 0.0,
        "peak_margin_mean": float(np.mean(peak_margin_all)) if peak_margin_all else 0.0,
    }
    if bool(args.save_per_sample):
        result["per_sample"] = per_sample
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate joint query-student + trainable DCFF map Corr-WLS.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise_deg", type=float, default=1.0)
    parser.add_argument("--noise_m", type=float, default=0.05)
    parser.add_argument("--outer_iters", type=int, default=3)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--flow_decode_mode", default=None)
    parser.add_argument("--confidence_mode", choices=["max", "margin"], default="margin")
    parser.add_argument("--confidence_margin_temperature", type=float, default=0.05)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--confidence_top_fraction", type=float, default=0.0)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--update_scale", type=float, default=0.5)
    parser.add_argument("--irls_iters", type=int, default=2)
    parser.add_argument("--pixel_stride", type=int, default=1)
    parser.add_argument("--robust_kernel", choices=["huber", "gm", "gnc_gm"], default="huber")
    parser.add_argument("--save_per_sample", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate(args)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "args": vars(args),
        "metrics": result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
    else:
        stem = Path(args.checkpoint).parents[1].name
        output = Path("/root/ICLPose/result/pose_refine/eval_joint_corr_wls") / f"{stem}_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
