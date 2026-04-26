#!/usr/bin/env python3
"""Train a lightweight reranker for real-init top-k candidate selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import camera_params_to_intrinsics, collate_fn, read_colmap_cameras  # noqa: E402
from data.radio_loc_retrieval_dataset import RadioLocRetrievalDataset  # noqa: E402
from pose_refine import load_concat_pose_checkpoint, load_concat_pose_model as load_model  # noqa: E402
from feature_field import build_dcff, intrinsics_to_K  # noqa: E402
from feature_field.runtime import apply_localization_map_state  # noqa: E402
from feature_retrieval.evaluate_impl import (  # noqa: E402
    camera_centers_from_w2c,
    compute_pose_errors,
    infer_retrieval_feature_dir,
    refine_pose_batch,
    render_rgb_alpha_batch,
    summarize_pose_metrics,
)
from feature_field.utils.project_config import load_mainline_config  # noqa: E402
from feature_field.utils.real_init_vis import load_rgb_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned reranker for real-init top-k candidates.")
    parser.add_argument("--config", required=True, help="Concat-localizer config")
    parser.add_argument("--checkpoint", required=True, help="Concat-localizer checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Query batch size for cache building")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--retrieval_method", default="auto", choices=["auto", "cls", "nearest_train_pose_gt"])
    parser.add_argument("--retrieval_feature_dir", default=None)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--outer_iters", type=int, default=10)
    parser.add_argument("--gru_iters", type=int, default=None)
    parser.add_argument(
        "--solver",
        default="default",
        choices=[
            "default",
            "wls_full",
            "pnp",
            "hybrid",
            "irls2",
            "irls3_gnc",
            "direct",
            "flow+direct5",
            "flow+direct10",
            "flow+direct20",
            "direct_neg",
            "flow+direct5_neg",
            "flow+direct5_highdamp",
            "flow+direct5_neg_highdamp",
        ],
    )
    parser.add_argument("--max_train_queries", type=int, default=0)
    parser.add_argument("--max_eval_queries", type=int, default=0)
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--refine_chunk_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--label_temperature", type=float, default=0.5)
    parser.add_argument("--model_type", choices=["mlp", "spatial"], default="spatial")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


class CandidateReranker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class SpatialCandidateVerifier(nn.Module):
    def __init__(self, scalar_dim: int, spatial_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(scalar_dim + spatial_channels + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scalar_features: torch.Tensor, spatial_maps: torch.Tensor) -> torch.Tensor:
        batch_size, topk = scalar_features.shape[:2]
        spatial_shape = spatial_maps.shape[-2:]
        flat_scalar = scalar_features.reshape(batch_size * topk, scalar_features.shape[-1])
        flat_spatial = spatial_maps.reshape(batch_size * topk, spatial_maps.shape[2], *spatial_shape)

        alpha = flat_spatial[:, -1:].clamp(0.0, 1.0)
        encoded = self.spatial_encoder(flat_spatial)
        denom = alpha.sum(dim=(-1, -2), keepdim=True).clamp(min=1.0)
        encoded_pool = (encoded * alpha).sum(dim=(-1, -2)) / denom.squeeze(-1).squeeze(-1)
        raw_pool = (flat_spatial * alpha).sum(dim=(-1, -2)) / denom.squeeze(-1).squeeze(-1)
        fused = torch.cat([flat_scalar, raw_pool, encoded_pool], dim=1)
        logits = self.head(fused).squeeze(-1)
        return logits.view(batch_size, topk)


def pose_cost(rot_deg: torch.Tensor, trans_mm: torch.Tensor) -> torch.Tensor:
    return trans_mm / 1000.0 + 0.1 * rot_deg


def masked_soft_targets(cost: torch.Tensor, valid_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    masked_cost = cost.masked_fill(~valid_mask, float("inf"))
    logits = -masked_cost / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~valid_mask, -1e9)
    target = torch.softmax(logits, dim=1)
    target = target * valid_mask.float()
    target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)
    return target


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_values = values * mask.float()
    denom = torch.clamp(mask.float().sum(dim=(-1, -2)), min=1.0)
    return masked_values.sum(dim=(-1, -2)) / denom


def _robust_limits(arrays: List[np.ndarray], default: Tuple[float, float]) -> Tuple[float, float]:
    flat_parts = [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in arrays if arr is not None]
    if not flat_parts:
        return default
    values = np.concatenate(flat_parts)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return default
    return float(lo), float(hi)


def _safe_image_stem(name: str) -> str:
    stem = Path(str(name)).stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return stem or "query"


def save_reranker_visuals(
    output_dir: Path,
    eval_cache: Dict[str, np.ndarray],
    eval_details: Dict[str, np.ndarray],
    max_queries: int = 8,
) -> Optional[Path]:
    if "query_rgb" not in eval_cache or "render_rgb" not in eval_cache:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_idx = np.asarray(eval_details["pred_idx"], dtype=np.int64)
    oracle_idx = np.asarray(eval_details["oracle_idx"], dtype=np.int64)
    logits = np.asarray(eval_details["logits"], dtype=np.float32)
    image_names = np.asarray(eval_cache["image_names"])
    final_trans = np.asarray(eval_cache["final_trans"], dtype=np.float32)
    final_rot = np.asarray(eval_cache["final_rot"], dtype=np.float32)

    top1_trans = final_trans[:, 0]
    pred_trans = final_trans[np.arange(final_trans.shape[0]), pred_idx]
    oracle_trans = final_trans[np.arange(final_trans.shape[0]), oracle_idx]
    pred_gain = np.maximum(top1_trans - pred_trans, 0.0)
    oracle_gap = np.maximum(pred_trans - oracle_trans, 0.0)
    sort_score = pred_gain + oracle_gap + 0.25 * np.maximum(top1_trans - oracle_trans, 0.0)
    order = np.argsort(-sort_score)
    chosen = order[: min(max_queries, len(order))]

    index_records = []
    for rank, query_idx in enumerate(chosen.tolist()):
        query_rgb = np.asarray(eval_cache["query_rgb"][query_idx], dtype=np.uint8)
        render_rgb = np.asarray(eval_cache["render_rgb"][query_idx], dtype=np.uint8)
        spatial_maps = np.asarray(eval_cache["spatial_maps"][query_idx], dtype=np.float32)
        scalar_features = np.asarray(eval_cache["features"][query_idx], dtype=np.float32)
        name = str(image_names[query_idx])
        roles = [
            ("top1", 0),
            ("selected", int(pred_idx[query_idx])),
            ("oracle", int(oracle_idx[query_idx])),
        ]

        l2_limits = _robust_limits([spatial_maps[idx, 0] for _, idx in roles], default=(0.0, 1.0))
        rgb_err_limits = _robust_limits([spatial_maps[idx, 2] for _, idx in roles], default=(0.0, 1.0))
        fig, axes = plt.subplots(len(roles), 6, figsize=(18, 3.4 * len(roles)))
        if len(roles) == 1:
            axes = axes[None, :]

        for row_idx, (label, cand_idx) in enumerate(roles):
            alpha = spatial_maps[cand_idx, 3]
            axes[row_idx, 0].imshow(query_rgb)
            axes[row_idx, 0].set_title("query_rgb")
            axes[row_idx, 1].imshow(render_rgb[cand_idx])
            axes[row_idx, 1].set_title(f"{label}_render")
            axes[row_idx, 2].imshow(alpha, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 2].set_title("alpha")
            axes[row_idx, 3].imshow(spatial_maps[cand_idx, 1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[row_idx, 3].set_title("feat_cos")
            axes[row_idx, 4].imshow(spatial_maps[cand_idx, 0], cmap="magma", vmin=l2_limits[0], vmax=l2_limits[1])
            axes[row_idx, 4].set_title("feat_l2")
            axes[row_idx, 5].imshow(
                spatial_maps[cand_idx, 2],
                cmap="magma",
                vmin=rgb_err_limits[0],
                vmax=rgb_err_limits[1],
            )
            axes[row_idx, 5].set_title("rgb_err")
            row_text = (
                f"{label} idx={cand_idx} "
                f"trans={final_trans[query_idx, cand_idx]:.1f}mm "
                f"rot={final_rot[query_idx, cand_idx]:.2f}deg "
                f"retr={scalar_features[cand_idx, 0]:.3f} "
                f"logit={logits[query_idx, cand_idx]:.3f}"
            )
            axes[row_idx, 0].set_ylabel(row_text, fontsize=9)
            for col_idx in range(6):
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])

        fig.suptitle(
            f"{name} | top1={top1_trans[query_idx]:.1f}mm "
            f"selected={pred_trans[query_idx]:.1f}mm oracle={oracle_trans[query_idx]:.1f}mm",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        image_path = output_dir / f"{rank:02d}_{_safe_image_stem(name)}.png"
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        index_records.append(
            {
                "rank": rank,
                "image_name": name,
                "top1_idx": 0,
                "selected_idx": int(pred_idx[query_idx]),
                "oracle_idx": int(oracle_idx[query_idx]),
                "top1_trans_mm": float(top1_trans[query_idx]),
                "selected_trans_mm": float(pred_trans[query_idx]),
                "oracle_trans_mm": float(oracle_trans[query_idx]),
                "pred_gain_mm": float(pred_gain[query_idx]),
                "oracle_gap_mm": float(oracle_gap[query_idx]),
                "figure": image_path.name,
            }
        )

    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_records, f, indent=2)
        f.write("\n")
    return output_dir


def build_candidate_cache(
    split_name: str,
    split_file: str,
    output_path: Path,
    config: Dict,
    model,
    gaussians,
    dcff_renderer,
    feat_sharp,
    device: torch.device,
    args: argparse.Namespace,
    max_queries: int = 0,
    store_visuals: bool = False,
) -> Dict[str, np.ndarray]:
    if output_path.is_file() and not args.force_rebuild_cache:
        with np.load(output_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    ds_cfg = config["dataset"]
    dcff_cfg = config["dcff"]
    render_h = dcff_cfg.get("render_height", 68)
    render_w = dcff_cfg.get("render_width", 120)
    fine_hw = tuple(ds_cfg.get("fine_hw", [render_h, render_w]))
    coarse_hw = tuple(ds_cfg.get("coarse_hw", fine_hw))
    use_coarse = config.get("model", {}).get("use_coarse", True)

    retrieval_feature_dir = infer_retrieval_feature_dir(config, args.retrieval_feature_dir)
    # Keep retrieval-init pose caches separate from reranker candidate caches.
    init_cache_name = f"{split_name}_retrieval_init_top{args.retrieval_topk}.npz"
    init_cache_path = output_path.parent / init_cache_name
    exclude_self = os.path.abspath(split_file) == os.path.abspath(ds_cfg["train_split"])

    dataset = RadioLocRetrievalDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        split="train" if split_name == "train" else "test",
        split_file=split_file,
        retrieval_train_split=ds_cfg["train_split"],
        retrieval_feature_dir=retrieval_feature_dir,
        retrieval_method=args.retrieval_method,
        retrieval_topk=args.retrieval_topk,
        exclude_query_from_db=exclude_self,
        source_dir=ds_cfg.get("source_dir"),
        init_poses_path=str(init_cache_path) if init_cache_path.is_file() and not args.force_rebuild_cache else None,
        save_init_poses_path=str(init_cache_path),
        fine_hw=fine_hw,
        coarse_hw=coarse_hw,
        cache_in_memory=True,
        noise_rot_deg=3.0,
        noise_trans_m=0.1,
        normalize_features=ds_cfg.get("normalize_features", False),
    )

    eval_dataset = dataset
    if max_queries > 0:
        eval_dataset = Subset(dataset, list(range(min(max_queries, len(dataset)))))

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    solver_configs = {
        "default": ("wls", 0, "huber", 0),
        "wls_full": ("wls_full", 0, "huber", 0),
        "pnp": ("pnp", 0, "huber", 0),
        "hybrid": ("hybrid", 0, "huber", 0),
        "irls2": ("wls", 2, "huber", 0),
        "irls3_gnc": ("wls", 3, "gnc_gm", 0),
        "direct": ("direct", 0, "huber", 0),
        "direct_neg": ("direct_neg", 0, "huber", 0),
        "flow+direct5": ("wls", 0, "huber", 5),
        "flow+direct10": ("wls", 0, "huber", 10),
        "flow+direct20": ("wls", 0, "huber", 20),
        "flow+direct5_neg": ("flow_neg", 0, "huber", 5),
        "flow+direct5_highdamp": ("flow_highdamp", 0, "huber", 5),
        "flow+direct5_neg_highdamp": ("flow_neg_highdamp", 0, "huber", 5),
    }
    solver_type, irls_iters, robust_kernel, direct_refine_iters = solver_configs[args.solver]
    gru_iters = args.gru_iters if args.gru_iters is not None else config.get("model", {}).get("gru_iters", 6)

    all_features: List[np.ndarray] = []
    all_spatial_maps: List[np.ndarray] = []
    all_valid_masks: List[np.ndarray] = []
    all_costs: List[np.ndarray] = []
    all_final_rot: List[np.ndarray] = []
    all_final_trans: List[np.ndarray] = []
    all_image_names: List[str] = []
    all_query_rgb: List[np.ndarray] = []
    all_render_rgb: List[np.ndarray] = []

    for batch in tqdm(loader, desc=f"build {split_name} cache", leave=False):
        query_fine = batch["query_fine"].to(device)
        query_coarse = batch.get("query_coarse")
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.to(device)
        else:
            query_coarse = None
        pose_gt = batch["pose_gt"].to(device)
        pose_init_candidates = batch["pose_init_candidates"]
        candidate_valid_mask = batch["candidate_valid_mask"]
        retrieval_scores_candidates = batch["retrieval_scores_candidates"]

        batch_size = pose_gt.shape[0]
        for i in range(batch_size):
            valid_mask = candidate_valid_mask[i].bool()
            candidate_poses = pose_init_candidates[i][valid_mask].to(device)
            candidate_scores = retrieval_scores_candidates[i][valid_mask].to(device)
            hyp_count = int(candidate_poses.shape[0])
            pose_gt_rep = pose_gt[i : i + 1].repeat(hyp_count, 1, 1)
            rgb_path = batch["query_rgb_path"][i]
            if rgb_path and os.path.isfile(rgb_path):
                query_rgb = np.array(
                    load_rgb_image(rgb_path, target_hw=(render_h, render_w)),
                    copy=True,
                )
            else:
                query_rgb = np.zeros((render_h, render_w, 3), dtype=np.uint8)
            query_rgb_t = (
                torch.from_numpy(query_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )

            refined_pose_chunks: List[torch.Tensor] = []
            refined_feature_chunks: List[torch.Tensor] = []
            render_rgb_chunks: List[torch.Tensor] = []
            render_alpha_chunks: List[torch.Tensor] = []
            diag_chunk_acc: Dict[str, List[torch.Tensor]] = {}
            chunk_size = max(1, int(args.refine_chunk_size))

            with torch.no_grad():
                for start in range(0, hyp_count, chunk_size):
                    end = min(start + chunk_size, hyp_count)
                    cur_count = end - start
                    query_fine_rep_chunk = query_fine[i : i + 1].repeat(cur_count, 1, 1, 1)
                    query_coarse_rep_chunk = (
                        query_coarse[i : i + 1].repeat(cur_count, 1, 1, 1)
                        if query_coarse is not None
                        else None
                    )
                    refined_poses_chunk, refined_features_chunk, diagnostics_chunk = refine_pose_batch(
                        model=model,
                        gaussians=gaussians,
                        dcff_renderer=dcff_renderer,
                        feat_sharp=feat_sharp,
                        query_fine=query_fine_rep_chunk,
                        query_coarse=query_coarse_rep_chunk,
                        pose_init=candidate_poses[start:end],
                        K=K,
                        render_intr=render_intr,
                        outer_iters=args.outer_iters,
                        solver=solver_type,
                        irls_iters=irls_iters,
                        robust_kernel=robust_kernel,
                        direct_refine_iters=direct_refine_iters,
                        collect_diagnostics=True,
                    )
                    render_rgb_chunk, render_alpha_chunk = render_rgb_alpha_batch(
                        gaussians, dcff_renderer, refined_poses_chunk, K, render_h, render_w
                    )
                    refined_pose_chunks.append(refined_poses_chunk)
                    refined_feature_chunks.append(refined_features_chunk)
                    render_rgb_chunks.append(render_rgb_chunk)
                    render_alpha_chunks.append(render_alpha_chunk)
                    for key, value in diagnostics_chunk.items():
                        diag_chunk_acc.setdefault(key, []).append(value)

            refined_poses = torch.cat(refined_pose_chunks, dim=0)
            refined_features = torch.cat(refined_feature_chunks, dim=0)
            render_rgb = torch.cat(render_rgb_chunks, dim=0)
            render_alpha = torch.cat(render_alpha_chunks, dim=0)
            diagnostics = {key: torch.cat(value, dim=0) for key, value in diag_chunk_acc.items()}
            query_fine_rep = query_fine[i : i + 1].repeat(hyp_count, 1, 1, 1)

            final_rot, final_trans = compute_pose_errors(refined_poses, pose_gt_rep)
            alpha_mask = render_alpha.squeeze(1) > 0.5

            feat_l2_map = (query_fine_rep.float() - refined_features.float()).pow(2).mean(dim=1)
            feat_cos_map = F.cosine_similarity(query_fine_rep.float(), refined_features.float(), dim=1)
            feat_l2 = masked_mean(feat_l2_map, alpha_mask)
            feat_cos = masked_mean(feat_cos_map, alpha_mask)
            rgb_err_map = (render_rgb - query_rgb_t.unsqueeze(0)).pow(2).mean(dim=1)

            rgb_mse = []
            alpha_coverage = alpha_mask.float().mean(dim=(-1, -2))
            for h in range(hyp_count):
                valid = render_alpha[h] > 0.5
                diff = (render_rgb[h] - query_rgb_t).pow(2)
                if valid.any():
                    rgb_mse.append(float(diff[valid.expand_as(diff)].mean().item()))
                else:
                    rgb_mse.append(float(diff.mean().item()))
            rgb_mse = torch.tensor(rgb_mse, device=device, dtype=torch.float32)

            init_centers = camera_centers_from_w2c(candidate_poses)
            refined_centers = camera_centers_from_w2c(refined_poses)
            top1_init_center = init_centers[:1]
            top1_refined_center = refined_centers[:1]
            center_dist_init = torch.norm(init_centers - top1_init_center, dim=1)
            center_dist_refined = torch.norm(refined_centers - top1_refined_center, dim=1)
            refine_delta_trans = torch.norm(refined_centers - init_centers, dim=1)
            refine_delta_rot, _ = compute_pose_errors(refined_poses, candidate_poses)

            rank_norm = torch.linspace(
                0.0,
                1.0 if hyp_count > 1 else 0.0,
                steps=hyp_count,
                device=device,
                dtype=torch.float32,
            )

            feature_matrix = torch.stack(
                [
                    candidate_scores.float(),
                    rank_norm,
                    feat_l2.float(),
                    feat_cos.float(),
                    rgb_mse.float(),
                    alpha_coverage.float(),
                    refine_delta_trans.float(),
                    refine_delta_rot.float(),
                    center_dist_init.float(),
                    center_dist_refined.float(),
                    diagnostics["flow_mag_mean"].float(),
                    diagnostics["flow_mag_std"].float(),
                    diagnostics["flow_mag_max"].float(),
                    diagnostics["confidence_mean"].float(),
                    diagnostics["confidence_std"].float(),
                    diagnostics["confidence_lowfrac"].float(),
                    diagnostics["flow_step_mean"].float(),
                    diagnostics["delta_xi_norm"].float(),
                    diagnostics["depth_valid_ratio"].float(),
                ],
                dim=1,
            )
            spatial_map_tensor = torch.stack(
                [
                    feat_l2_map.float(),
                    feat_cos_map.float(),
                    rgb_err_map.float(),
                    alpha_mask.float(),
                ],
                dim=1,
            )

            padded_features = torch.zeros((args.retrieval_topk, feature_matrix.shape[1]), dtype=torch.float32)
            padded_spatial = torch.zeros(
                (
                    args.retrieval_topk,
                    spatial_map_tensor.shape[1],
                    spatial_map_tensor.shape[2],
                    spatial_map_tensor.shape[3],
                ),
                dtype=torch.float32,
            )
            padded_valid = torch.zeros((args.retrieval_topk,), dtype=torch.bool)
            padded_cost = torch.full((args.retrieval_topk,), fill_value=1e9, dtype=torch.float32)
            padded_final_rot = torch.full((args.retrieval_topk,), fill_value=np.nan, dtype=torch.float32)
            padded_final_trans = torch.full((args.retrieval_topk,), fill_value=np.nan, dtype=torch.float32)

            padded_features[:hyp_count] = feature_matrix.detach().cpu()
            padded_spatial[:hyp_count] = spatial_map_tensor.detach().cpu()
            padded_valid[:hyp_count] = True
            padded_cost[:hyp_count] = pose_cost(final_rot.float(), final_trans.float()).detach().cpu()
            padded_final_rot[:hyp_count] = final_rot.detach().cpu()
            padded_final_trans[:hyp_count] = final_trans.detach().cpu()

            all_features.append(padded_features.numpy())
            all_spatial_maps.append(padded_spatial.numpy())
            all_valid_masks.append(padded_valid.numpy())
            all_costs.append(padded_cost.numpy())
            all_final_rot.append(padded_final_rot.numpy())
            all_final_trans.append(padded_final_trans.numpy())
            all_image_names.append(batch["image_name"][i])
            if store_visuals:
                render_rgb_uint8 = (
                    render_rgb.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu()
                )
                padded_render_rgb = torch.zeros(
                    (args.retrieval_topk, render_h, render_w, 3),
                    dtype=torch.uint8,
                )
                padded_render_rgb[:hyp_count] = render_rgb_uint8
                all_query_rgb.append(query_rgb.astype(np.uint8))
                all_render_rgb.append(padded_render_rgb.numpy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        features=np.stack(all_features).astype(np.float32),
        spatial_maps=np.stack(all_spatial_maps).astype(np.float32),
        valid_mask=np.stack(all_valid_masks).astype(bool),
        target_cost=np.stack(all_costs).astype(np.float32),
        final_rot=np.stack(all_final_rot).astype(np.float32),
        final_trans=np.stack(all_final_trans).astype(np.float32),
        image_names=np.array(all_image_names),
    )
    if store_visuals and all_query_rgb and all_render_rgb:
        save_kwargs["query_rgb"] = np.stack(all_query_rgb).astype(np.uint8)
        save_kwargs["render_rgb"] = np.stack(all_render_rgb).astype(np.uint8)
    np.savez(output_path, **save_kwargs)
    with np.load(output_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def evaluate_reranker(
    model,
    features: torch.Tensor,
    spatial_maps: torch.Tensor,
    valid_mask: torch.Tensor,
    final_rot: torch.Tensor,
    final_trans: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    spatial_mean: torch.Tensor,
    spatial_std: torch.Tensor,
    model_type: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    model.eval()
    with torch.no_grad():
        scalar_x = (features - feat_mean) / feat_std
        if model_type == "spatial":
            spatial_x = (spatial_maps - spatial_mean) / spatial_std
            logits = model(scalar_x, spatial_x).masked_fill(~valid_mask, -1e9)
        else:
            logits = model(scalar_x).masked_fill(~valid_mask, -1e9)
        pred_idx = torch.argmax(logits, dim=1)

        top1_rot = final_rot[:, 0]
        top1_trans = final_trans[:, 0]
        pred_rot = final_rot.gather(1, pred_idx[:, None]).squeeze(1)
        pred_trans = final_trans.gather(1, pred_idx[:, None]).squeeze(1)
        oracle_idx = torch.argmin(final_trans.masked_fill(~valid_mask, 1e9), dim=1)
        oracle_rot = final_rot.gather(1, oracle_idx[:, None]).squeeze(1)
        oracle_trans = final_trans.gather(1, oracle_idx[:, None]).squeeze(1)

    summary = {
        "top1": summarize_pose_metrics(top1_rot.cpu().numpy(), top1_trans.cpu().numpy()),
        "rerank": summarize_pose_metrics(pred_rot.cpu().numpy(), pred_trans.cpu().numpy()),
        "oracle_trans": summarize_pose_metrics(oracle_rot.cpu().numpy(), oracle_trans.cpu().numpy()),
    }
    summary["selection"] = {
        "selected_not_top1_rate": float((pred_idx != 0).float().mean().item() * 100.0),
        "num_queries": int(features.shape[0]),
    }
    details = {
        "pred_idx": pred_idx.cpu().numpy(),
        "oracle_idx": oracle_idx.cpu().numpy(),
        "logits": logits.cpu().numpy(),
    }
    return summary, details


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    config = load_mainline_config(args.config)

    exp_name = config.get("exp_name", Path(args.checkpoint).stem)
    output_dir = Path(args.output_dir or os.path.join(config.get("output_dir", "output"), exp_name, "real_init_reranker"))
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)
    localizer, ckpt_epoch = load_model(config, args.checkpoint, device)
    pose_ckpt = load_concat_pose_checkpoint(args.checkpoint, device)
    restored_map = apply_localization_map_state(
        dcff_renderer,
        feat_sharp,
        pose_ckpt,
        printer=print,
    )
    if restored_map:
        print(f"Restored localization map state: {', '.join(restored_map)}")
    localizer.eval()
    if hasattr(dcff_renderer, "eval"):
        dcff_renderer.eval()
    if feat_sharp is not None and hasattr(feat_sharp, "eval"):
        feat_sharp.eval()

    ds_cfg = config["dataset"]
    colmap_cameras = read_colmap_cameras(os.path.join(ds_cfg["colmap_dir"], "cameras.bin"))
    first_cam = next(iter(colmap_cameras.values()))
    localizer.BASE_INTRINSICS = camera_params_to_intrinsics(first_cam)
    localizer.IMG_HW = (int(first_cam.height), int(first_cam.width))

    train_cache = build_candidate_cache(
        split_name="train",
        split_file=ds_cfg["train_split"],
        output_path=cache_dir / f"train_top{args.retrieval_topk}.npz",
        config=config,
        model=localizer,
        gaussians=gaussians,
        dcff_renderer=dcff_renderer,
        feat_sharp=feat_sharp,
        device=device,
        args=args,
        max_queries=args.max_train_queries,
        store_visuals=False,
    )
    eval_cache = build_candidate_cache(
        split_name="eval",
        split_file=ds_cfg["test_split"],
        output_path=cache_dir / f"eval_top{args.retrieval_topk}.npz",
        config=config,
        model=localizer,
        gaussians=gaussians,
        dcff_renderer=dcff_renderer,
        feat_sharp=feat_sharp,
        device=device,
        args=args,
        max_queries=args.max_eval_queries,
        store_visuals=True,
    )

    train_features = torch.from_numpy(train_cache["features"]).float()
    train_spatial_maps = torch.from_numpy(train_cache["spatial_maps"]).float()
    train_valid = torch.from_numpy(train_cache["valid_mask"]).bool()
    train_cost = torch.from_numpy(train_cache["target_cost"]).float()
    train_final_rot = torch.from_numpy(train_cache["final_rot"]).float()
    train_final_trans = torch.from_numpy(train_cache["final_trans"]).float()

    eval_features = torch.from_numpy(eval_cache["features"]).float()
    eval_spatial_maps = torch.from_numpy(eval_cache["spatial_maps"]).float()
    eval_valid = torch.from_numpy(eval_cache["valid_mask"]).bool()
    eval_cost = torch.from_numpy(eval_cache["target_cost"]).float()
    eval_final_rot = torch.from_numpy(eval_cache["final_rot"]).float()
    eval_final_trans = torch.from_numpy(eval_cache["final_trans"]).float()

    valid_train_rows = train_features[train_valid]
    feat_mean = valid_train_rows.mean(dim=0, keepdim=True)
    feat_std = valid_train_rows.std(dim=0, keepdim=True).clamp(min=1e-6)
    valid_train_spatial = train_spatial_maps[train_valid]
    spatial_mean = valid_train_spatial.mean(dim=(0, 2, 3), keepdim=True)
    spatial_std = valid_train_spatial.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

    if args.model_type == "spatial":
        reranker = SpatialCandidateVerifier(
            scalar_dim=train_features.shape[-1],
            spatial_channels=train_spatial_maps.shape[2],
            hidden_dim=args.hidden_dim,
        ).to(device)
    else:
        reranker = CandidateReranker(train_features.shape[-1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = TensorDataset(train_features, train_spatial_maps, train_valid, train_cost)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    best_eval_trans = float("inf")
    best_summary = None
    best_state = None
    best_details = None
    train_history = []

    for epoch in range(args.epochs):
        reranker.train()
        losses = []
        for feat_batch, spatial_batch, valid_batch, cost_batch in train_loader:
            feat_batch = feat_batch.to(device)
            spatial_batch = spatial_batch.to(device)
            valid_batch = valid_batch.to(device)
            cost_batch = cost_batch.to(device)

            scalar_x = (feat_batch - feat_mean.to(device)) / feat_std.to(device)
            if args.model_type == "spatial":
                spatial_x = (spatial_batch - spatial_mean.to(device)) / spatial_std.to(device)
                logits = reranker(scalar_x, spatial_x).masked_fill(~valid_batch, -1e9)
            else:
                logits = reranker(scalar_x).masked_fill(~valid_batch, -1e9)
            target = masked_soft_targets(cost_batch, valid_batch, args.label_temperature)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(target * log_probs).sum(dim=1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        eval_summary, eval_details = evaluate_reranker(
            reranker,
            eval_features.to(device),
            eval_spatial_maps.to(device),
            eval_valid.to(device),
            eval_final_rot.to(device),
            eval_final_trans.to(device),
            feat_mean.to(device),
            feat_std.to(device),
            spatial_mean.to(device),
            spatial_std.to(device),
            args.model_type,
        )
        train_history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "eval_rerank_trans_median": eval_summary["rerank"]["trans_median"],
                "eval_top1_trans_median": eval_summary["top1"]["trans_median"],
            }
        )
        print(
            f"[epoch {epoch:03d}] loss={train_history[-1]['train_loss']:.4f} "
            f"rerank={eval_summary['rerank']['trans_median']:.1f}mm "
            f"top1={eval_summary['top1']['trans_median']:.1f}mm "
            f"oracle={eval_summary['oracle_trans']['trans_median']:.1f}mm"
        )
        if eval_summary["rerank"]["trans_median"] < best_eval_trans:
            best_eval_trans = eval_summary["rerank"]["trans_median"]
            best_summary = eval_summary
            best_details = eval_details
            best_state = {
                "model_state_dict": reranker.state_dict(),
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "spatial_mean": spatial_mean,
                "spatial_std": spatial_std,
                "train_history": train_history,
                "args": vars(args),
            }

    if best_state is None:
        raise RuntimeError("Reranker training did not produce any checkpoint.")

    ckpt_path = output_dir / "best_reranker.pth"
    torch.save(best_state, ckpt_path)
    qual_dir = save_reranker_visuals(output_dir / "qual", eval_cache, best_details) if best_details is not None else None

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "localizer_epoch": ckpt_epoch,
        "retrieval_topk": args.retrieval_topk,
        "model_type": args.model_type,
        "num_train_queries": int(train_features.shape[0]),
        "num_eval_queries": int(eval_features.shape[0]),
        "best_eval": best_summary,
        "history_tail": train_history[-10:],
        "reranker_checkpoint": str(ckpt_path),
        "qual_dir": str(qual_dir) if qual_dir is not None else None,
    }
    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"reranker_checkpoint: {ckpt_path}\n")
        f.write(f"num_train_queries: {train_features.shape[0]}\n")
        f.write(f"num_eval_queries: {eval_features.shape[0]}\n")
        if best_summary is not None:
            f.write(f"top1: {json.dumps(best_summary['top1'])}\n")
            f.write(f"rerank: {json.dumps(best_summary['rerank'])}\n")
            f.write(f"oracle_trans: {json.dumps(best_summary['oracle_trans'])}\n")
            f.write(f"selection: {json.dumps(best_summary['selection'])}\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
