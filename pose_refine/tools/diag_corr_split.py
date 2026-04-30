#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import collate_fn
from feature_field.utils.project_config import load_mainline_config
from pose_refine.models.concat_pose_net import ConcatPoseNet, local_correlation
from pose_refine.tools.eval_pose_update_scale import load_dcff_runtime_state
from pose_refine.train_impl import (
    ConcatLocTrainer,
    camera_centers_from_w2c,
    local_correlation_soft_flow_loss_from_corr,
    local_correlation_subpixel_ce_loss_from_corr,
)


def parse_split_names(values: Iterable[str]) -> list[str]:
    """Parse split names while preserving user order and aliases."""
    parsed: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        if key == "both":
            names = ["train", "val"]
        elif key == "test":
            names = ["val"]
        elif key in ("train", "val"):
            names = [key]
        else:
            raise ValueError(f"Unknown split '{value}'. Use train, val/test, or both.")
        for name in names:
            if name not in parsed:
                parsed.append(name)
    if not parsed:
        raise ValueError("At least one split is required")
    return parsed


def pose_errors(pose_pred: torch.Tensor, pose_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.amp.autocast(enabled=False):
        pred = pose_pred.float()
        gt = pose_gt.float()
        r_rel = torch.bmm(pred[:, :3, :3].transpose(1, 2), gt[:, :3, :3])
        trace = r_rel[:, 0, 0] + r_rel[:, 1, 1] + r_rel[:, 2, 2]
        rot = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        trans = torch.linalg.norm(
            camera_centers_from_w2c(pred) - camera_centers_from_w2c(gt),
            dim=1,
        )
    return rot * 180.0 / math.pi, trans * 1000.0


def _resize_query_flow_valid(
    query_feat: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    target_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = target_hw
    query = query_feat.float()
    flow = flow_gt.float()
    valid = valid_mask.float()
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if query.shape[-2:] != (h, w):
        src_h, src_w = query.shape[-2:]
        query = F.interpolate(query, (h, w), mode="bilinear", align_corners=False)
        flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / src_w
        flow[:, 1] *= h / src_h
        valid = F.interpolate(valid, (h, w), mode="nearest")
    elif flow.shape[-2:] != (h, w):
        src_h, src_w = flow.shape[-2:]
        flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / src_w
        flow[:, 1] *= h / src_h
        valid = F.interpolate(valid, (h, w), mode="nearest")
    elif valid.shape[-2:] != (h, w):
        valid = F.interpolate(valid, (h, w), mode="nearest")
    return query, flow, valid


def projected_correlation(
    model: torch.nn.Module,
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    rendered = rendered_feat.float()
    query = query_feat.float()
    if getattr(model, "proj_mode", "separate") == "shared":
        q_proj = F.normalize(model.proj_shared(query), dim=1)
        r_proj = F.normalize(model.proj_shared(rendered), dim=1)
    else:
        q_proj = F.normalize(model.proj_query(query), dim=1)
        r_proj = F.normalize(model.proj_render(rendered), dim=1)
    if getattr(model, "use_cross_attention", False):
        q_proj = model.cross_attn(q_proj, r_proj)
        q_proj = F.normalize(q_proj, dim=1)
    return local_correlation(r_proj, q_proj, radius=radius)


def raw_correlation(
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    rendered = F.normalize(rendered_feat.float(), dim=1)
    query = F.normalize(query_feat.float(), dim=1)
    return local_correlation(rendered, query, radius=radius)


def corr_stats(
    corr: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int,
    temperature: float,
    prefix: str,
) -> dict[str, float]:
    with torch.cuda.amp.autocast(enabled=False):
        flow = flow_gt.float()
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        _, soft_metrics = local_correlation_soft_flow_loss_from_corr(
            corr,
            flow,
            valid,
            radius=radius,
            temperature=temperature,
        )
        _, subpx_metrics = local_correlation_subpixel_ce_loss_from_corr(
            corr,
            flow,
            valid,
            radius=radius,
            temperature=temperature,
        )

        b, channels, h, w = corr.shape
        window = 2 * radius + 1
        if channels != window * window:
            raise ValueError(f"Correlation channel count {channels} does not match radius={radius}")
        idx = corr.float().argmax(dim=1, keepdim=True)
        dx = (idx % window).float() - radius
        dy = torch.div(idx, window, rounding_mode="floor").float() - radius
        pred_flow = torch.cat([dx, dy], dim=1)

        valid_pixels = valid > 0.5
        in_window = (
            valid_pixels
            & (flow[:, :1] >= -radius)
            & (flow[:, :1] <= radius)
            & (flow[:, 1:2] >= -radius)
            & (flow[:, 1:2] <= radius)
        )
        denom_valid = valid_pixels.float().sum().clamp(min=1.0)
        denom_window = in_window.float().sum().clamp(min=1.0)
        argmax_epe_valid = (
            torch.linalg.norm(pred_flow - flow, dim=1, keepdim=True) * valid_pixels.float()
        ).sum() / denom_valid
        argmax_epe_window = (
            torch.linalg.norm(pred_flow - flow, dim=1, keepdim=True) * in_window.float()
        ).sum() / denom_window
        nearest_dx = torch.round(flow[:, :1])
        nearest_dy = torch.round(flow[:, 1:2])
        top1_round_acc = (
            (dx == nearest_dx)
            & (dy == nearest_dy)
            & in_window
        ).float().sum() / denom_window
        flow_mag = (
            torch.linalg.norm(flow, dim=1, keepdim=True) * valid_pixels.float()
        ).sum() / denom_valid

    return {
        f"{prefix}_argmax_epe_valid": float(argmax_epe_valid.item()),
        f"{prefix}_argmax_epe_window": float(argmax_epe_window.item()),
        f"{prefix}_soft_epe": float(soft_metrics["corr_flow_epe"]),
        f"{prefix}_subpx_epe": float(subpx_metrics["corr_subpx_flow_epe"]),
        f"{prefix}_top1_round_acc": float(top1_round_acc.item()),
        f"{prefix}_coverage": float(in_window.float().mean().item()),
        f"{prefix}_valid_ratio": float(valid_pixels.float().mean().item()),
        f"{prefix}_flow_mag": float(flow_mag.item()),
    }


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanmedian(arr))


@torch.no_grad()
def evaluate_split(
    trainer: ConcatLocTrainer,
    split_name: str,
    *,
    max_batches: int,
    radius: int,
    temperature: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer.model.eval()
    trainer._set_map_train_mode(False)

    dataset = trainer.train_dataset if split_name == "train" else trainer.val_dataset
    batch_size = int(trainer.config.get("training", {}).get("batch_size", 6))
    if split_name != "train":
        batch_size = min(batch_size, 6)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    render_intr = trainer.model._scale_intrinsics(trainer.render_h, trainer.render_w)
    flow_hw = (trainer.render_h, trainer.render_w)
    accum: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        accum.setdefault(name, []).append(float(value))

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"diag-{split_name}", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        query_fine = batch["query_fine"].to(trainer.device).float()
        pose_gt = batch["pose_gt"].to(trainer.device).float()
        pose_init = batch["pose_init"].to(trainer.device).float()

        init_rot, init_trans = pose_errors(pose_init, pose_gt)
        add("init_rot_deg", float(init_rot.mean().item()))
        add("init_trans_mm", float(init_trans.mean().item()))

        bundle = trainer._render_bundle_batch(
            pose_init,
            differentiable=False,
            render_coarse=False,
        )
        rendered = bundle["fine_features"].float()
        depth = bundle["depth"].float()
        gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
            pose_init,
            pose_gt,
            depth,
            flow_hw,
            render_intr,
        )
        query, gt_flow, gt_valid = _resize_query_flow_valid(
            query_fine,
            gt_flow,
            gt_valid,
            rendered.shape[-2:],
        )

        raw_corr = raw_correlation(rendered, query, radius=radius)
        for key, value in corr_stats(
            raw_corr,
            gt_flow,
            gt_valid,
            radius=radius,
            temperature=temperature,
            prefix="raw",
        ).items():
            add(key, value)

        proj_corr = projected_correlation(trainer.model, rendered, query, radius=radius)
        for key, value in corr_stats(
            proj_corr,
            gt_flow,
            gt_valid,
            radius=radius,
            temperature=temperature,
            prefix="proj",
        ).items():
            add(key, value)

    return {name: summarize(vals) for name, vals in sorted(accum.items())}


def print_summary(split_name: str, metrics: dict[str, tuple[float, float]]) -> None:
    print(f"\n[{split_name}]", flush=True)
    for key, (mean, median) in metrics.items():
        print(f"{key}: mean={mean:.4f} median={median:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DCFF local-correlation train/val split behavior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--warmstart", default=None)
    parser.add_argument("--dcff_state_checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["both"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--exp_name", default=None)
    args = parser.parse_args()

    config = load_mainline_config(args.config)
    config["exp_name"] = args.exp_name or f"{Path(args.config).stem}_diagcorr"
    training_cfg = config.setdefault("training", {})
    training_cfg["num_workers"] = 0
    training_cfg["max_train_batches"] = 0
    training_cfg["max_val_batches"] = 0

    trainer = ConcatLocTrainer(
        config,
        gpu=args.gpu,
        resume_path=args.checkpoint,
        warmstart_path=args.warmstart,
    )
    if args.dcff_state_checkpoint:
        loaded = load_dcff_runtime_state(trainer, args.dcff_state_checkpoint)
        print(f"loaded_dcff_state={','.join(loaded) if loaded else 'none'}", flush=True)

    radius = int(args.radius if args.radius is not None else getattr(trainer.model, "local_radius", 4))
    loss_cfg = config.get("training", {}).get("loss", {})
    temperature = float(
        args.temperature
        if args.temperature is not None
        else loss_cfg.get("corr_flow_temperature", loss_cfg.get("corr_ce_temperature", 0.05))
    )
    print(
        f"diag config={args.config} radius={radius} temp={temperature} "
        f"max_batches={args.max_batches}",
        flush=True,
    )

    for split_name in parse_split_names(args.splits):
        metrics = evaluate_split(
            trainer,
            split_name,
            max_batches=args.max_batches,
            radius=radius,
            temperature=temperature,
            seed=args.seed,
        )
        print_summary(split_name, metrics)


if __name__ == "__main__":
    main()
