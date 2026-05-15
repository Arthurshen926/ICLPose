#!/usr/bin/env python3
"""Evaluate CPR fixed-init buckets without running a training loop."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.train_impl import (  # noqa: E402
    JointRADIOQueryDataset,
    MapFeatureRenderer,
    RetrievalTeacherStore,
    TeacherCorrespondenceStore,
    TeacherFeatureStore,
    _candidate_wls_refined_pose_cost,
    apply_pose_delta,
    build_all_records,
    build_local_pose_lattice_candidates,
    build_radio_query_student,
    candidate_score_fusion_listwise_loss,
    fine_candidate_selector_features,
    load_config,
    load_model_warmstart,
    load_pose_candidate_cache_index,
    local_render_score_feature_candidates,
    move_batch_to_device,
    pose_error_tensors,
    project_query_render_for_fine_selector,
    resolve_query_feature_dims,
    resolve_safe_num_workers,
    safe_torch_load,
    set_seed,
    split_records,
)
from feature_extract.students.pose_energy_net import pose_energy_selection_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="FeatureExtract config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing model_state_dict")
    parser.add_argument(
        "--map-checkpoint",
        default=None,
        help="Optional checkpoint whose map_renderer_state_dict is used instead of --checkpoint",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["010cm_2deg:10,2", "025cm_5deg:25,5", "050cm_10deg:50,10"],
        help="Bucket specs as name:trans_cm,rot_deg",
    )
    parser.add_argument("--topk", default="1,4,8", help="Comma-separated topK basin recalls")
    parser.add_argument("--lattice-trans-cm", default=None, help="Override lattice translation radii, comma-separated cm")
    parser.add_argument("--lattice-rot-deg", default=None, help="Override lattice rotation radii, comma-separated deg")
    parser.add_argument(
        "--lattice-direction-mode",
        choices=("axis", "cube"),
        default=None,
        help="Direction set for lattice translation/rotation deltas. cube uses 26 normalized directions.",
    )
    parser.add_argument(
        "--combine-trans-rot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to combine translation and rotation lattice deltas; defaults to config.",
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--limit-strategy", default="uniform")
    parser.add_argument(
        "--candidate-render-batch-size",
        type=int,
        default=None,
        help="Override map_supervision.candidate_render_batch_size for evaluation",
    )
    parser.add_argument("--fine-wls", action="store_true", help="Run one fine WLS update on selected candidates")
    parser.add_argument(
        "--fine-topk",
        type=int,
        default=0,
        help="Number of scorer topK candidates to refine; default reads map_supervision.fine_topk_selector_topk",
    )
    parser.add_argument("--fine-update-scale", type=float, default=None, help="Override WLS pose update scale")
    parser.add_argument(
        "--fine-score-stat",
        choices=("mean", "max", "topk_mean", "peakiness"),
        default="mean",
        help="Local correlation statistic used by fine_score candidate selection",
    )
    parser.add_argument(
        "--fine-select",
        choices=("score", "conf", "fine_score", "fine_score_prior", "fine_selector", "pose_energy", "oracle"),
        default="score",
        help=(
            "Which topK candidate becomes final: scorer top1, max WLS confidence, fine corr score, "
            "trainable selector, PoseEnergy selector, or GT oracle."
        ),
    )
    parser.add_argument(
        "--pose-energy-checkpoint",
        default=None,
        help="Optional Stage2 NVS pose-feature checkpoint used when --fine-select pose_energy.",
    )
    parser.add_argument(
        "--fine-selector-adapter-checkpoint",
        default=None,
        help=(
            "Optional NVS pose-feature adapter checkpoint used to project query/render fine "
            "features before --fine-select fine_selector."
        ),
    )
    parser.add_argument(
        "--fine-selector-score-source",
        choices=("local_corr", "pair_matcher_heatmap"),
        default="local_corr",
        help="Score-map source for --fine-select fine_selector.",
    )
    parser.add_argument(
        "--fine-pool-mode",
        choices=("rank", "rank_uniform", "rank_delta_uniform", "rank_score_uniform", "rank_pose_hard"),
        default="rank",
        help=(
            "How to form the fine selector/WLS candidate pool. rank preserves the original scorer topK; "
            "rank_* modes keep --fine-pool-topm scorer candidates and fill the rest with deterministic "
            "diverse candidates."
        ),
    )
    parser.add_argument(
        "--fine-pool-topm",
        type=int,
        default=16,
        help="Number of scorer-ranked candidates to force-keep before diversity fill in non-rank fine pool modes.",
    )
    parser.add_argument(
        "--fine-prior-weight",
        type=float,
        default=0.0,
        help="For fine_score_prior, subtract this weight times standardized candidate motion from T0.",
    )
    parser.add_argument(
        "--fine-prior-rot-weight",
        type=float,
        default=0.1,
        help="Rotation cost weight for fine_score_prior candidate motion from T0.",
    )
    parser.add_argument(
        "--init-noise-mode",
        choices=("fixed", "random"),
        default="fixed",
        help="Use deterministic axis-aligned T0 noise or continuous random T0 noise.",
    )
    parser.add_argument("--init-jitter-seed", type=int, default=20260510, help="Seed for random T0/candidate jitter")
    parser.add_argument(
        "--candidate-jitter-cm",
        type=float,
        default=0.0,
        help="Apply up to this much random translation jitter to every candidate pose.",
    )
    parser.add_argument(
        "--candidate-jitter-deg",
        type=float,
        default=0.0,
        help="Apply up to this much random rotation jitter to every candidate pose.",
    )
    parser.add_argument(
        "--disable-exact-inverse",
        action="store_true",
        help="Mask candidates that exactly recover GT pose in fixed-lattice diagnostics.",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON")
    return parser.parse_args()


def pose_energy_candidate_selection(
    outputs: Dict[str, torch.Tensor],
    valid: torch.Tensor,
    *,
    confidence_weight: float = 0.0,
    residual_norm_weight: float = 0.0,
    residual_trans_scale_m: float = 1.0,
    residual_rot_scale_rad: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Select a candidate from PoseEnergy outputs with invalid rows masked."""
    scores = pose_energy_selection_scores(
        outputs,
        confidence_weight=float(confidence_weight),
        residual_norm_weight=float(residual_norm_weight),
        residual_trans_scale_m=float(residual_trans_scale_m),
        residual_rot_scale_rad=float(residual_rot_scale_rad),
    )
    if scores.shape != valid.shape:
        raise ValueError(f"score/valid shape mismatch: {tuple(scores.shape)} vs {tuple(valid.shape)}")
    masked = scores.float().masked_fill(~valid.bool(), -1.0e6)
    idx = masked.argmax(dim=1)
    probs = torch.softmax(masked, dim=1)
    entropy = -(probs * torch.log(probs.clamp(min=1.0e-8))).sum(dim=1)
    if masked.shape[1] > 1:
        top2 = masked.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.zeros_like(entropy)
    return {"idx": idx, "scores": masked, "entropy": entropy, "margin": margin}


def parse_bucket_specs(values: Iterable[str]) -> List[Tuple[str, float, float]]:
    buckets = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Bucket spec must be name:trans_cm,rot_deg, got {value!r}")
        name, payload = value.split(":", 1)
        parts = [float(part) for part in payload.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Bucket spec must contain trans_cm,rot_deg, got {value!r}")
        buckets.append((name, float(parts[0]), float(parts[1])))
    return buckets


def parse_float_csv(value: str | None) -> List[float] | None:
    if value is None:
        return None
    values = [float(part) for part in value.split(",") if part.strip()]
    return values if values else None


def apply_eval_overrides(cfg: Dict, args: argparse.Namespace) -> None:
    if getattr(args, "candidate_render_batch_size", None) is not None:
        cfg.setdefault("map_supervision", {})["candidate_render_batch_size"] = max(
            0,
            int(args.candidate_render_batch_size),
        )


def default_lattice(trans_cm: float, rot_deg: float) -> Tuple[List[float], List[float]]:
    if trans_cm <= 10.0 and rot_deg <= 2.0:
        return [0.0, 5.0, 10.0, 25.0], [0.0, 1.0, 2.0, 5.0]
    if trans_cm <= 25.0 and rot_deg <= 5.0:
        return [0.0, 10.0, 25.0, 50.0], [0.0, 2.0, 5.0, 10.0]
    if trans_cm <= 50.0 and rot_deg <= 10.0:
        return [0.0, 25.0, 50.0], [0.0, 5.0, 10.0]
    return [0.0, 25.0, 50.0, 100.0], [0.0, 5.0, 10.0, 20.0]


def make_fixed_init_poses(pose_gt: torch.Tensor, trans_cm: float, rot_deg: float, *, offset: int = 0) -> torch.Tensor:
    bsz = pose_gt.shape[0]
    delta = pose_gt.new_zeros((bsz, 6))
    for idx in range(bsz):
        sample_idx = idx + int(offset)
        trans_axis = sample_idx % 3
        rot_axis = (sample_idx // 3) % 3
        trans_sign = -1.0 if ((sample_idx // 9) % 2) else 1.0
        rot_sign = -1.0 if ((sample_idx // 18) % 2) else 1.0
        delta[idx, trans_axis] = trans_sign * float(trans_cm) / 100.0
        delta[idx, 3 + rot_axis] = rot_sign * math.radians(float(rot_deg))
    return apply_pose_delta(pose_gt.float(), delta.float()).to(dtype=pose_gt.dtype)


def _random_unit_vectors(count: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vectors = torch.randn(count, 3, generator=generator, dtype=torch.float32)
    vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return vectors.to(device=device, dtype=dtype)


def make_random_init_poses(
    pose_gt: torch.Tensor,
    trans_cm: float,
    rot_deg: float,
    *,
    seed: int,
    offset: int = 0,
) -> torch.Tensor:
    bsz = pose_gt.shape[0]
    delta = pose_gt.new_zeros((bsz, 6))
    trans_axes = _random_unit_vectors(
        bsz,
        seed=int(seed) + int(offset) * 2 + 17,
        device=pose_gt.device,
        dtype=pose_gt.dtype,
    )
    rot_axes = _random_unit_vectors(
        bsz,
        seed=int(seed) + int(offset) * 2 + 31,
        device=pose_gt.device,
        dtype=pose_gt.dtype,
    )
    delta[:, :3] = trans_axes * (float(trans_cm) / 100.0)
    delta[:, 3:] = rot_axes * math.radians(float(rot_deg))
    return apply_pose_delta(pose_gt.float(), delta.float()).to(dtype=pose_gt.dtype)


def jitter_candidate_poses(
    candidate_poses: torch.Tensor,
    *,
    trans_cm: float,
    rot_deg: float,
    seed: int,
    offset: int = 0,
) -> torch.Tensor:
    if float(trans_cm) <= 0.0 and float(rot_deg) <= 0.0:
        return candidate_poses
    bsz, num_candidates = candidate_poses.shape[:2]
    count = bsz * num_candidates
    device = candidate_poses.device
    dtype = candidate_poses.dtype
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(offset) * 2 + 101)
    delta = torch.zeros(count, 6, device=device, dtype=dtype)
    if float(trans_cm) > 0.0:
        axes = _random_unit_vectors(count, seed=int(seed) + int(offset) * 2 + 151, device=device, dtype=dtype)
        mag = torch.rand(count, 1, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        delta[:, :3] = axes * mag * (float(trans_cm) / 100.0)
    if float(rot_deg) > 0.0:
        axes = _random_unit_vectors(count, seed=int(seed) + int(offset) * 2 + 181, device=device, dtype=dtype)
        mag = torch.rand(count, 1, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        delta[:, 3:] = axes * mag * math.radians(float(rot_deg))
    jittered = apply_pose_delta(candidate_poses.reshape(count, 4, 4).float(), delta.float())
    return jittered.reshape(bsz, num_candidates, 4, 4).to(dtype=dtype)


def exact_inverse_candidate_mask(candidate_poses: torch.Tensor, pose_gt: torch.Tensor) -> torch.Tensor:
    bsz, num_candidates = candidate_poses.shape[:2]
    pose_gt_bank = pose_gt[:, None].expand(-1, num_candidates, -1, -1)
    _rot_loss, rot_deg, trans_m = pose_error_tensors(
        candidate_poses.reshape(bsz * num_candidates, 4, 4).float(),
        pose_gt_bank.reshape(bsz * num_candidates, 4, 4).float(),
    )
    trans_m = trans_m.reshape(bsz, num_candidates)
    rot_deg = rot_deg.reshape(bsz, num_candidates)
    return (trans_m <= 1.0e-5) & (rot_deg <= 1.0e-4)


def tensor_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
    }


def gather_pose_bank(poses: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view_shape = (indices.shape[0], indices.shape[1], 1, 1)
    return poses.gather(1, indices.view(view_shape).expand(-1, -1, 4, 4))


def _take_unique_from_order(order: torch.Tensor, used: set[int], count: int) -> List[int]:
    if count <= 0 or order.numel() == 0:
        return []
    values = [int(v) for v in order.detach().cpu().tolist() if int(v) not in used]
    if len(values) <= count:
        return values
    if count == 1:
        picks = [len(values) // 2]
    else:
        picks = torch.linspace(0, len(values) - 1, count).round().long().tolist()
    out = []
    seen = set()
    for pos in picks:
        idx = values[int(pos)]
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    if len(out) < count:
        for idx in values:
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
                if len(out) >= count:
                    break
    return out[:count]


def _take_first_unique_from_order(order: torch.Tensor, used: set[int]) -> List[int]:
    for value in order.detach().cpu().tolist():
        idx = int(value)
        if idx not in used:
            return [idx]
    return []


def select_fine_pool_indices(
    logits: torch.Tensor,
    valid: torch.Tensor,
    fine_topk: int,
    *,
    mode: str = "rank",
    rank_topm: int = 16,
    candidate_pose: torch.Tensor | None = None,
    init_pose: torch.Tensor | None = None,
    pose_gt: torch.Tensor | None = None,
    rot_cost_weight: float = 0.1,
) -> torch.Tensor:
    """Select the K candidates passed to fine reranking.

    The default exactly matches the old behavior. GT-free diversity modes keep
    a small scorer prefix, then fill the rest from the lattice.  The
    rank_pose_hard mode is only for supervised cache construction: it also
    forces in GT-near, identity, and opposite-direction examples.
    """
    if logits.ndim != 2 or valid.ndim != 2:
        raise ValueError("logits and valid must have shape (B,K)")
    if logits.shape != valid.shape:
        raise ValueError(f"logits/valid shape mismatch: {tuple(logits.shape)} vs {tuple(valid.shape)}")
    bsz, num_candidates = logits.shape
    keep = max(1, min(int(fine_topk or 1), num_candidates))
    scores = logits.float().masked_fill(~valid.bool(), -1.0e6)
    ranked = torch.argsort(scores, dim=1, descending=True)
    pool_mode = str(mode or "rank").lower()
    if pool_mode == "rank":
        return ranked[:, :keep]

    delta_order = None
    if pool_mode in ("rank_delta_uniform", "rank_pose_hard"):
        if candidate_pose is None or init_pose is None:
            raise ValueError(f"{pool_mode} fine pool requires candidate_pose and init_pose")
        init_bank = init_pose[:, None].expand(-1, num_candidates, -1, -1)
        _rot_loss, delta_rot_deg, delta_trans_m = pose_error_tensors(
            candidate_pose.reshape(bsz * num_candidates, 4, 4).float(),
            init_bank.reshape(bsz * num_candidates, 4, 4).float(),
        )
        delta_cost = delta_trans_m.reshape(bsz, num_candidates) + float(rot_cost_weight) * (
            delta_rot_deg.reshape(bsz, num_candidates) * (math.pi / 180.0)
        )
        delta_cost = delta_cost.masked_fill(~valid.bool(), float("inf"))
        delta_order = torch.argsort(delta_cost, dim=1, descending=False)
    elif pool_mode not in ("rank_uniform", "rank_score_uniform"):
        raise ValueError(f"unsupported fine pool mode: {mode}")

    gt_cost = None
    gt_order = None
    opposite_order = None
    if pool_mode == "rank_pose_hard":
        if pose_gt is None:
            raise ValueError("rank_pose_hard fine pool requires pose_gt")
        gt_bank = pose_gt[:, None].expand(-1, num_candidates, -1, -1)
        _rot_loss, gt_rot_deg, gt_trans_m = pose_error_tensors(
            candidate_pose.reshape(bsz * num_candidates, 4, 4).float(),
            gt_bank.reshape(bsz * num_candidates, 4, 4).float(),
        )
        gt_cost = gt_trans_m.reshape(bsz, num_candidates) + float(rot_cost_weight) * (
            gt_rot_deg.reshape(bsz, num_candidates) * (math.pi / 180.0)
        )
        gt_cost = gt_cost.masked_fill(~valid.bool(), float("inf"))
        gt_order = torch.argsort(gt_cost, dim=1, descending=False)

        def camera_centers(poses: torch.Tensor) -> torch.Tensor:
            rot = poses[..., :3, :3].float()
            trans = poses[..., :3, 3].float()
            return -(rot.transpose(-1, -2) @ trans.unsqueeze(-1)).squeeze(-1)

        init_center = camera_centers(init_pose)[:, None, :]
        gt_center = camera_centers(pose_gt)[:, None, :]
        cand_center = camera_centers(candidate_pose)
        candidate_delta = cand_center - init_center
        target_delta = gt_center - init_center
        denom = torch.linalg.norm(candidate_delta, dim=-1) * torch.linalg.norm(target_delta, dim=-1).clamp_min(1e-6)
        correction_cos = (candidate_delta * target_delta).sum(dim=-1) / denom.clamp_min(1e-6)
        correction_cos = correction_cos.masked_fill((denom <= 1e-8) | ~valid.bool(), float("inf"))
        opposite_order = torch.argsort(correction_cos, dim=1, descending=False)

    rows = []
    prefix = max(0, min(int(rank_topm or 0), keep, num_candidates))
    for row in range(bsz):
        selected: List[int] = []
        used: set[int] = set()
        for idx in ranked[row, :prefix].detach().cpu().tolist():
            idx_i = int(idx)
            if bool(valid[row, idx_i]) and idx_i not in used:
                selected.append(idx_i)
                used.add(idx_i)
        fill = keep - len(selected)
        if pool_mode == "rank_score_uniform":
            order = ranked[row]
        elif pool_mode == "rank_delta_uniform":
            order = delta_order[row]
        elif pool_mode == "rank_pose_hard":
            for order in (gt_order[row], delta_order[row], opposite_order[row]):
                for idx_i in _take_first_unique_from_order(order, used):
                    selected.append(idx_i)
                    used.add(idx_i)
            if gt_cost is not None:
                finite_cost = gt_cost[row][torch.isfinite(gt_cost[row])]
                if finite_cost.numel() > 0:
                    hard_threshold = finite_cost.median()
                    hard_order = torch.tensor(
                        [
                            int(idx)
                            for idx in ranked[row].detach().cpu().tolist()
                            if bool(valid[row, int(idx)]) and float(gt_cost[row, int(idx)].item()) >= float(hard_threshold.item())
                        ],
                        device=logits.device,
                        dtype=torch.long,
                    )
                    for idx_i in _take_first_unique_from_order(hard_order, used):
                        selected.append(idx_i)
                        used.add(idx_i)
            fill = keep - len(selected)
            order = gt_order[row]
        else:
            order = torch.nonzero(valid[row].bool(), as_tuple=False).flatten()
        for idx_i in _take_unique_from_order(order, used, fill):
            selected.append(idx_i)
            used.add(idx_i)
        if len(selected) < keep:
            for idx in ranked[row].detach().cpu().tolist():
                idx_i = int(idx)
                if bool(valid[row, idx_i]) and idx_i not in used:
                    selected.append(idx_i)
                    used.add(idx_i)
                    if len(selected) >= keep:
                        break
        if len(selected) < keep:
            selected.extend([selected[0] if selected else 0] * (keep - len(selected)))
        rows.append(torch.tensor(selected[:keep], device=logits.device, dtype=torch.long))
    return torch.stack(rows, dim=0)


def _row_standardize(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
    return (values - mean) / std


def fine_prior_adjusted_scores(
    fine_scores: torch.Tensor,
    selected_pose: torch.Tensor,
    init_pose: torch.Tensor,
    *,
    prior_weight: float,
    rot_cost_weight: float,
) -> torch.Tensor:
    if float(prior_weight) <= 0.0:
        return fine_scores
    bsz, topk = selected_pose.shape[:2]
    init_bank = init_pose[:, None].expand(-1, topk, -1, -1)
    _rot_loss, delta_rot_deg, delta_trans_m = pose_error_tensors(
        selected_pose.reshape(bsz * topk, 4, 4).float(),
        init_bank.reshape(bsz * topk, 4, 4).float(),
    )
    delta_cost = delta_trans_m.view(bsz, topk) + float(rot_cost_weight) * (
        delta_rot_deg.view(bsz, topk) * (math.pi / 180.0)
    )
    return _row_standardize(fine_scores.float()) - float(prior_weight) * _row_standardize(delta_cost.float())


def selected_pose_error_dict(
    selected_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    *,
    rot_cost_weight: float,
) -> Dict[str, torch.Tensor]:
    bsz, topk = selected_pose.shape[:2]
    pose_gt_bank = pose_gt[:, None].expand(-1, topk, -1, -1)
    _rot_loss, rot_deg, trans_m = pose_error_tensors(
        selected_pose.reshape(bsz * topk, 4, 4).float(),
        pose_gt_bank.reshape(bsz * topk, 4, 4).float(),
    )
    trans_m = trans_m.reshape(bsz, topk)
    rot_deg = rot_deg.reshape(bsz, topk)
    cost = trans_m + float(rot_cost_weight) * (rot_deg * (math.pi / 180.0))
    return {
        "trans_err_m": trans_m,
        "rot_err_deg": rot_deg,
        "init_trans_err_m": trans_m,
        "init_rot_err_deg": rot_deg,
        "cost": cost,
        "conf_mean": torch.zeros_like(trans_m),
    }


def map_pose_gt_for_batch(map_renderer: MapFeatureRenderer, batch: Dict, device: torch.device) -> torch.Tensor:
    poses = []
    for sample_name in batch["sample_name"]:
        normalized = map_renderer._normalize_name(sample_name)
        poses.append(map_renderer.name_to_pose[normalized].to(device=device, dtype=torch.float32))
    return torch.stack(poses, dim=0)


def resolve_map_renderer_state(checkpoint: Dict, args: argparse.Namespace):
    map_checkpoint_path = getattr(args, "map_checkpoint", None)
    if map_checkpoint_path:
        checkpoint = safe_torch_load(map_checkpoint_path)
    return checkpoint.get("map_renderer_state_dict")


def load_pose_energy_selector_bundle(path: str, model, cfg: Dict, device: torch.device) -> Dict:
    """Load the Stage2 NVS adapter, pair matcher, and PoseEnergy head for CPR eval."""
    if not path:
        raise ValueError("--pose-energy-checkpoint is required for --fine-select pose_energy")
    from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher, PoseEnergyNet
    from feature_extract.tools.train_nvs_pose_feature_adapter import (
        load_adapter_checkpoint,
        nvs_pose_energy_vector_dim,
    )
    from feature_extract.tools.train_pose_energy import build_pose_feature_adapter

    checkpoint = safe_torch_load(path)
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise RuntimeError(f"PoseEnergy checkpoint {path} does not contain saved args")
    nvs_args = argparse.Namespace(**saved_args)
    adapter = build_pose_feature_adapter(nvs_args, model, cfg, device)
    if adapter is None:
        raise RuntimeError("PoseEnergy checkpoint requires pose_feature_adapter_enabled=true")
    energy_net = PoseEnergyNet(
        vector_dim=nvs_pose_energy_vector_dim(nvs_args),
        score_map_channels=3,
        map_channels=int(getattr(nvs_args, "pose_energy_map_channels", 16)),
        grid_size=int(getattr(nvs_args, "pose_energy_grid_size", 4)),
        hidden_dim=int(getattr(nvs_args, "pose_energy_hidden_dim", 128)),
        context_layers=int(getattr(nvs_args, "pose_energy_context_layers", 1)),
        context_heads=int(getattr(nvs_args, "pose_energy_context_heads", 1)),
        zero_init_heads=bool(getattr(nvs_args, "pose_energy_zero_init_heads", False)),
        zero_init_residual_head=bool(getattr(nvs_args, "pose_energy_zero_init_residual_head", True)),
        factorized_heads=bool(getattr(nvs_args, "pose_energy_factorized_heads", False)),
    ).to(device)
    pair_matcher = None
    if bool(getattr(nvs_args, "pair_matcher_enabled", False)):
        pair_matcher = PairConditionedLocalMatcher(
            channels=int(adapter.channels),
            hidden_dim=int(getattr(nvs_args, "pair_matcher_hidden_dim", 64)),
            offset_radius=int(getattr(nvs_args, "pair_matcher_radius", 3)),
            zero_init_residual=bool(getattr(nvs_args, "pair_matcher_zero_init_residual", True)),
            base_dot_weight=float(getattr(nvs_args, "pair_matcher_base_dot_weight", 1.0)),
        ).to(device)
    load_adapter_checkpoint(
        path,
        adapter,
        model=model,
        optimizer=None,
        energy_net=energy_net,
        pair_matcher=pair_matcher,
    )
    adapter.eval()
    energy_net.eval()
    if pair_matcher is not None:
        pair_matcher.eval()
    return {"args": nvs_args, "adapter": adapter, "energy_net": energy_net, "pair_matcher": pair_matcher}


def load_pose_feature_adapter_bundle(path: str, model, cfg: Dict, device: torch.device) -> Dict:
    """Load only the NVS pose-feature adapter for CPR fine-selector features."""
    if not path:
        raise ValueError("pose feature adapter checkpoint path is required")
    from feature_extract.tools.train_nvs_pose_feature_adapter import load_adapter_checkpoint
    from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher
    from feature_extract.tools.train_pose_energy import build_pose_feature_adapter

    checkpoint = safe_torch_load(path)
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise RuntimeError(f"Pose-feature adapter checkpoint {path} does not contain saved args")
    nvs_args = argparse.Namespace(**saved_args)
    adapter = build_pose_feature_adapter(nvs_args, model, cfg, device)
    if adapter is None:
        raise RuntimeError("Pose-feature adapter checkpoint requires pose_feature_adapter_enabled=true")
    pair_matcher = None
    if bool(getattr(nvs_args, "pair_matcher_enabled", False)):
        pair_matcher = PairConditionedLocalMatcher(
            channels=int(adapter.channels),
            hidden_dim=int(getattr(nvs_args, "pair_matcher_hidden_dim", 64)),
            offset_radius=int(getattr(nvs_args, "pair_matcher_radius", 3)),
            zero_init_residual=bool(getattr(nvs_args, "pair_matcher_zero_init_residual", True)),
            base_dot_weight=float(getattr(nvs_args, "pair_matcher_base_dot_weight", 1.0)),
        ).to(device)
    load_adapter_checkpoint(path, adapter, model=model, optimizer=None, energy_net=None, pair_matcher=pair_matcher)
    adapter.eval()
    if pair_matcher is not None:
        pair_matcher.eval()
    return {"args": nvs_args, "adapter": adapter, "pair_matcher": pair_matcher}


def project_query_render_with_pose_feature_adapter(
    adapter,
    query_fine: torch.Tensor,
    render_fine: torch.Tensor,
    *,
    query_rgb: torch.Tensor | None = None,
    render_rgb: torch.Tensor | None = None,
    use_uncertainty: bool = False,
    render_chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Project query and candidate render features through an NVS pose adapter."""
    if render_fine.ndim != 5:
        raise ValueError(f"render_fine must have shape B,K,C,H,W, got {tuple(render_fine.shape)}")
    bsz, num_candidates, channels, height, width = render_fine.shape
    flat_render = render_fine.reshape(bsz * num_candidates, channels, height, width)
    flat_rgb = None
    if render_rgb is not None:
        if render_rgb.ndim != 5 or render_rgb.shape[:2] != (bsz, num_candidates):
            raise ValueError(f"render_rgb must have shape B,K,3,H,W, got {tuple(render_rgb.shape)}")
        flat_rgb = render_rgb.reshape(bsz * num_candidates, render_rgb.shape[2], render_rgb.shape[3], render_rgb.shape[4])

    if bool(use_uncertainty):
        query_loc, query_uncertainty = adapter.project_query_with_uncertainty(query_fine, rgb=query_rgb)
    else:
        query_loc = adapter.project_query(query_fine, rgb=query_rgb)
        query_uncertainty = None

    chunk_size = int(render_chunk_size or 0)
    loc_chunks = []
    unc_chunks = []
    for start in range(0, flat_render.shape[0], chunk_size if chunk_size > 0 else flat_render.shape[0]):
        rgb_chunk = flat_rgb[start : start + (chunk_size if chunk_size > 0 else flat_render.shape[0])] if flat_rgb is not None else None
        render_chunk = flat_render[start : start + (chunk_size if chunk_size > 0 else flat_render.shape[0])]
        if bool(use_uncertainty):
            loc, unc = adapter.project_render_with_uncertainty(render_chunk, rgb=rgb_chunk)
            loc_chunks.append(loc)
            unc_chunks.append(unc)
        else:
            loc_chunks.append(adapter.project_render(render_chunk, rgb=rgb_chunk))
    render_loc = torch.cat(loc_chunks, dim=0).reshape(bsz, num_candidates, channels, height, width)
    render_uncertainty = None
    if bool(use_uncertainty):
        render_uncertainty = torch.cat(unc_chunks, dim=0).reshape(bsz, num_candidates, 1, height, width)
    return query_loc, render_loc, query_uncertainty, render_uncertainty


def build_model_and_data(cfg: Dict, args: argparse.Namespace, device: torch.device):
    apply_eval_overrides(cfg, args)
    cfg["training"]["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    cfg.setdefault("map_supervision", {})["perturb_render_negatives"] = False
    set_seed(int(cfg["training"].get("seed", 42)))

    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    retrieval_store = None
    retrieval_cfg = cfg.get("retrieval", {})
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
    records = train_records if args.split == "train" else val_records
    skip_samples = int(getattr(args, "skip_samples", 0) or 0)
    if skip_samples > 0:
        records = records[skip_samples:]
    if args.max_samples is not None and args.max_samples > 0:
        records = records[: int(args.max_samples)]

    colmap_dir = (
        cfg["dataset"].get("colmap_dir")
        or cfg.get("map_supervision", {}).get("colmap_dir")
        or str(Path(cfg["dataset"]["source_dir"]) / "sparse" / "0")
    )
    teacher_corr_path = cfg["dataset"].get("teacher_correspondence_path")
    teacher_corr_train_path = cfg["dataset"].get("teacher_correspondence_train_path") or teacher_corr_path
    teacher_corr_val_path = cfg["dataset"].get("teacher_correspondence_val_path") or teacher_corr_path
    teacher_corr_selected_path = teacher_corr_train_path if args.split == "train" else teacher_corr_val_path
    teacher_corr_store = None
    if teacher_corr_selected_path:
        teacher_corr_store = TeacherCorrespondenceStore(
            teacher_corr_selected_path,
            feature_hw=cfg["dataset"]["feature_hw"],
            max_points=int(cfg["dataset"].get("teacher_correspondence_max_points", 512)),
            coordinate_space=str(cfg["dataset"].get("teacher_correspondence_coordinate_space", "auto")),
        )
    if args.split == "train":
        pose_candidate_cache_paths = cfg["dataset"].get("train_pose_candidate_caches") or cfg["dataset"].get(
            "train_pose_candidate_cache"
        )
        pose_candidate_sampling = cfg["dataset"].get(
            "train_pose_candidate_cache_sampling",
            cfg["dataset"].get("pose_candidate_cache_sampling", "first"),
        )
    else:
        pose_candidate_cache_paths = cfg["dataset"].get("val_pose_candidate_caches") or cfg["dataset"].get(
            "val_pose_candidate_cache"
        )
        pose_candidate_sampling = cfg["dataset"].get(
            "val_pose_candidate_cache_sampling",
            cfg["dataset"].get("pose_candidate_cache_sampling", "first"),
        )
    keep_variants = bool(cfg["dataset"].get("pose_candidate_cache_keep_variants", False)) or str(
        pose_candidate_sampling or "first"
    ).lower() != "first"
    pose_candidate_index = load_pose_candidate_cache_index(pose_candidate_cache_paths, keep_variants=keep_variants)
    pose_candidate_topk = int(
        cfg["dataset"].get(
            "pose_candidate_topk",
            cfg.get("map_supervision", {}).get(
                "candidate_stage1_topk",
                cfg.get("map_supervision", {}).get("candidate_render_score_max_candidates", 0),
            ),
        )
        or 0
    )
    dataset = JointRADIOQueryDataset(
        records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        teacher_correspondence_store=teacher_corr_store,
        colmap_dir=colmap_dir,
        pose_candidate_cache_index=pose_candidate_index,
        pose_candidate_topk=pose_candidate_topk,
        pose_candidate_cache_sampling=pose_candidate_sampling,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=resolve_safe_num_workers(cfg["training"], cfg["dataset"]),
        pin_memory=(
            bool(cfg["training"].get("pin_memory"))
            if cfg["training"].get("pin_memory") is not None
            else device.type == "cuda"
        ),
    )
    model = build_radio_query_student(
        cfg,
        fine_feature_dim=fine_dim,
        coarse_feature_dim=coarse_dim,
        retrieval_dim=int(cfg["retrieval"]["student_dim"]) if retrieval_store is not None else None,
        retrieval_hidden_dim=int(cfg["retrieval"].get("hidden_dim", 0)) if retrieval_store is not None else None,
    ).to(device)
    checkpoint = safe_torch_load(args.checkpoint)
    load_model_warmstart(model, checkpoint, strict=False)
    model.eval()
    logger = logging.getLogger("eval_cpr_buckets")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    map_renderer = MapFeatureRenderer(
        cfg,
        feature_hw=tuple(cfg["dataset"]["feature_hw"]),
        device=device,
        logger=logger,
    )
    map_renderer.load_trainable_state(resolve_map_renderer_state(checkpoint, args))
    map_renderer.set_train_mode(False)
    return model, loader, map_renderer


@torch.no_grad()
def evaluate_bucket(
    model,
    loader,
    map_renderer,
    cfg: Dict,
    args: argparse.Namespace,
    bucket,
    pose_energy_bundle: Dict | None = None,
    fine_selector_adapter_bundle: Dict | None = None,
):
    name, trans_cm, rot_deg = bucket
    map_cfg = cfg.get("map_supervision", {})
    topk_values = [int(v) for v in args.topk.split(",") if v.strip()]
    lattice_trans, lattice_rot = default_lattice(trans_cm, rot_deg)
    lattice_trans = parse_float_csv(args.lattice_trans_cm) or lattice_trans
    lattice_rot = parse_float_csv(args.lattice_rot_deg) or lattice_rot
    if args.max_candidates <= 0:
        max_candidates = int(map_cfg.get("coarse_pose_lattice_max_candidates", 0) or 0)
    else:
        max_candidates = int(args.max_candidates)
    combine_trans_rot = (
        bool(args.combine_trans_rot)
        if args.combine_trans_rot is not None
        else bool(map_cfg.get("coarse_pose_lattice_combine_trans_rot", False))
    )
    direction_mode = str(args.lattice_direction_mode or map_cfg.get("coarse_pose_lattice_direction_mode", "axis"))

    rows = {
        "init_trans_mm": [],
        "init_rot_deg": [],
        "coarse_top1_trans_mm": [],
        "coarse_top1_rot_deg": [],
        "oracle_best_trans_mm": [],
        "oracle_best_rot_deg": [],
        "oracle_gain_mm": [],
        "final_trans_mm": [],
        "final_rot_deg": [],
        "fine_gain_mm": [],
        "fine_topk_oracle_trans_mm": [],
        "fine_topk_oracle_rot_deg": [],
        "fine_topk_conf_trans_mm": [],
        "fine_topk_conf_rot_deg": [],
        "fine_topk_conf_mean": [],
        "fine_topk_score_trans_mm": [],
        "fine_topk_score_rot_deg": [],
        "fine_topk_score_init_trans_mm": [],
        "fine_topk_score_init_rot_deg": [],
        "fine_topk_selector_trans_mm": [],
        "fine_topk_selector_rot_deg": [],
        "fine_topk_selector_oracle_gap_mm": [],
        "fine_topk_selector_entropy": [],
        "fine_topk_selector_margin": [],
        "fine_topk_pose_energy_trans_mm": [],
        "fine_topk_pose_energy_rot_deg": [],
        "fine_topk_pose_energy_oracle_gap_mm": [],
        "fine_topk_pose_energy_entropy": [],
        "fine_topk_pose_energy_margin": [],
        "wls_conf_mean": [],
    }
    topk_hits = {k: [] for k in topk_values}
    oracle_hits = []
    sample_offset = 0
    use_amp = bool(cfg["training"].get("amp", True) and next(model.parameters()).device.type == "cuda")

    for batch in loader:
        batch = move_batch_to_device(batch, next(model.parameters()).device)
        pose_gt = map_pose_gt_for_batch(map_renderer, batch, next(model.parameters()).device)
        current_offset = sample_offset
        if args.init_noise_mode == "random":
            init_pose = make_random_init_poses(
                pose_gt,
                trans_cm,
                rot_deg,
                seed=int(args.init_jitter_seed),
                offset=current_offset,
            )
        else:
            init_pose = make_fixed_init_poses(pose_gt, trans_cm, rot_deg, offset=current_offset)
        sample_offset += pose_gt.shape[0]
        candidate_poses = build_local_pose_lattice_candidates(
            init_pose,
            trans_cm=lattice_trans,
            rot_deg=lattice_rot,
            include_identity=True,
            max_candidates=max_candidates,
            limit_strategy=args.limit_strategy,
            combine_trans_rot=combine_trans_rot,
            direction_mode=direction_mode,
        )
        candidate_poses = jitter_candidate_poses(
            candidate_poses,
            trans_cm=float(args.candidate_jitter_cm),
            rot_deg=float(args.candidate_jitter_deg),
            seed=int(args.init_jitter_seed),
            offset=current_offset,
        )
        valid = torch.ones(candidate_poses.shape[:2], device=pose_gt.device, dtype=torch.bool)
        if args.disable_exact_inverse:
            valid = valid & ~exact_inverse_candidate_mask(candidate_poses, pose_gt)
            empty_rows = ~valid.any(dim=1)
            if empty_rows.any():
                valid[empty_rows, 0] = True

        with torch.autocast(device_type=pose_gt.device.type, enabled=use_amp):
            outputs = model(batch["rgb"])
        eval_batch = dict(batch)
        eval_batch = map_renderer.attach_pose_candidate_renders(
            eval_batch,
            candidate_poses,
            prefix="eval_candidate",
            candidate_valid_mask=valid,
            feature="coarse",
            include_aux=False,
        )
        candidate_coarse = eval_batch["eval_candidate_coarse"].float()
        adapter = getattr(model, "candidate_basin_adapter", None)
        if adapter is not None:
            candidate_coarse = adapter(candidate_coarse).float()
        _loss, _metrics, details = candidate_score_fusion_listwise_loss(
            outputs["coarse"].float(),
            candidate_coarse,
            eval_batch["eval_candidate_pose"].float(),
            pose_gt.float(),
            model.candidate_score_fusion_head,
            batch=eval_batch,
            mask=eval_batch.get("eval_candidate_mask"),
            candidate_valid_mask=eval_batch.get("eval_candidate_valid_mask"),
            mode=map_cfg.get("candidate_score_fusion_mode", "local"),
            temperature=float(map_cfg.get("candidate_score_fusion_temperature", 1.0)),
            radius=int(map_cfg.get("candidate_score_fusion_radius", 2)),
            preprocess=map_cfg.get("candidate_score_fusion_preprocess", "spatial_center"),
            highpass_kernel=int(map_cfg.get("candidate_score_fusion_highpass_kernel", 5)),
            score_map_mode=map_cfg.get("candidate_score_fusion_score_map_mode", "volume_plus_peak_offset"),
            rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
            target_mode=map_cfg.get("candidate_score_fusion_target_mode", "gt_pose_error_soft"),
            target_temperature_m=float(map_cfg.get("candidate_score_fusion_target_temperature_m", 0.25)),
            render_feature_mode=map_cfg.get("candidate_score_fusion_render_feature_mode", "basic"),
            cost_regression_weight=0.0,
            basin_trans_m=float(map_cfg.get("candidate_score_fusion_basin_trans_m", 0.25)),
            basin_rot_deg=float(map_cfg.get("candidate_score_fusion_basin_rot_deg", 5.0)),
            return_details=True,
        )
        logits = details["logits"]
        trans_err_m = details["trans_err_m"]
        rot_err_deg = details["rot_err_deg"]
        basin = (
            (trans_err_m <= float(map_cfg.get("candidate_score_fusion_basin_trans_m", 0.25)))
            & (rot_err_deg <= float(map_cfg.get("candidate_score_fusion_basin_rot_deg", 5.0)))
        )
        pred_idx = logits.argmax(dim=1)
        pose_cost = trans_err_m + float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)) * (
            rot_err_deg * (math.pi / 180.0)
        )
        best_idx = pose_cost.argmin(dim=1)
        for k in topk_values:
            k_eff = min(k, logits.shape[1])
            top_idx = logits.topk(k_eff, dim=1).indices
            topk_hits[k].extend(basin.gather(1, top_idx).any(dim=1).float().cpu().tolist())
        oracle_hits.extend(basin.any(dim=1).float().cpu().tolist())

        _init_rot_loss, init_rot, init_trans = pose_error_tensors(init_pose.float(), pose_gt.float())
        batch_idx = torch.arange(pose_gt.shape[0], device=pose_gt.device)
        coarse_trans = trans_err_m[batch_idx, pred_idx]
        coarse_rot = rot_err_deg[batch_idx, pred_idx]
        oracle_best_trans = trans_err_m[batch_idx, best_idx]
        oracle_best_rot = rot_err_deg[batch_idx, best_idx]
        final_trans = coarse_trans
        final_rot = coarse_rot
        wls_conf = torch.zeros_like(coarse_trans)

        needs_fine_candidates = args.fine_wls or args.fine_select in {
            "conf",
            "fine_score",
            "fine_score_prior",
            "fine_selector",
            "pose_energy",
            "oracle",
        }
        if needs_fine_candidates:
            fine_topk_cfg = int(map_cfg.get("fine_topk_selector_topk", 1) or 1)
            fine_topk = max(1, min(int(args.fine_topk or fine_topk_cfg), logits.shape[1]))
            selected_idx = select_fine_pool_indices(
                logits,
                details["valid"].bool(),
                fine_topk,
                mode=str(args.fine_pool_mode),
                rank_topm=int(args.fine_pool_topm),
                candidate_pose=eval_batch["eval_candidate_pose"].float(),
                init_pose=init_pose.float(),
                pose_gt=pose_gt.float(),
                rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
            )
            selected_pose = gather_pose_bank(eval_batch["eval_candidate_pose"].float(), selected_idx)
            selected_valid = details["valid"].bool().gather(1, selected_idx)
            fine_batch = dict(batch)
            fine_batch = map_renderer.attach_pose_candidate_renders(
                fine_batch,
                selected_pose,
                prefix="eval_selected",
                candidate_valid_mask=selected_valid,
                feature="all",
                include_aux=True,
            )
            query_fine_key = str(map_cfg.get("query_fine_key", "fine"))
            query_fine = outputs.get(query_fine_key, outputs["fine"]).float()
            if args.fine_wls:
                refined = _candidate_wls_refined_pose_cost(
                    query_fine,
                    fine_batch["eval_selected_fine"].float(),
                    fine_batch["eval_selected_pose"].float(),
                    pose_gt.float(),
                    fine_batch["eval_selected_depth"].float(),
                    fine_batch["eval_selected_intrinsics"].float(),
                    fine_batch["eval_selected_valid_mask"].bool(),
                    radius=int(map_cfg.get("query_corr_radius", 4)),
                    temperature=float(map_cfg.get("query_corr_temperature", 0.05)),
                    damping=float(map_cfg.get("query_corr_wls_damping", 1e-3)),
                    update_scale=float(
                        map_cfg.get("query_corr_wls_update_scale", 1.0)
                        if args.fine_update_scale is None
                        else args.fine_update_scale
                    ),
                    rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
                    wls_conf_mode=map_cfg.get("query_corr_wls_conf_mode", "max"),
                    wls_conf_variance_scale=float(map_cfg.get("query_corr_wls_conf_variance_scale", 0.5)),
                    wls_downsample=int(map_cfg.get("query_corr_wls_downsample", 1)),
                )
            else:
                refined = selected_pose_error_dict(
                    fine_batch["eval_selected_pose"].float(),
                    pose_gt.float(),
                    rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
                )
            fine_score_rows = []
            fine_masks = fine_batch.get("eval_selected_mask")
            for row_idx in range(pose_gt.shape[0]):
                score_result = local_render_score_feature_candidates(
                    query_fine[row_idx],
                    fine_batch["eval_selected_fine"][row_idx].float(),
                    mask=None if fine_masks is None else fine_masks[row_idx],
                    radius=int(map_cfg.get("query_corr_radius", 4)),
                    preprocess=map_cfg.get("candidate_score_fusion_preprocess", "spatial_center"),
                    highpass_kernel=int(map_cfg.get("candidate_score_fusion_highpass_kernel", 5)),
                    score_map_mode="peak_offset",
                )
                if args.fine_score_stat == "mean":
                    fine_score_rows.append(score_result["scores"])
                else:
                    fine_score_rows.append(score_result["score_stats"][args.fine_score_stat])
            fine_scores = torch.stack(fine_score_rows, dim=0)
            fine_prior_scores = fine_prior_adjusted_scores(
                fine_scores,
                selected_pose,
                init_pose,
                prior_weight=float(args.fine_prior_weight),
                rot_cost_weight=float(args.fine_prior_rot_weight),
            )
            refined_cost = refined["cost"]
            oracle_refined_idx = refined_cost.argmin(dim=1)
            conf_idx = refined["conf_mean"].argmax(dim=1)
            fine_score_idx = fine_scores.argmax(dim=1)
            fine_prior_idx = fine_prior_scores.argmax(dim=1)
            fine_selector_idx = None
            fine_selector_logits = None
            pose_energy_idx = None
            pose_energy_selection = None
            if args.fine_select == "fine_selector" or getattr(model, "fine_candidate_selector_head", None) is not None:
                selector_head = getattr(model, "fine_candidate_selector_head", None)
                if selector_head is None:
                    raise RuntimeError("--fine-select fine_selector requires model.fine_candidate_selector_head")
                selected_coarse_logits = details.get("raw_logits", details["logits"]).gather(1, selected_idx).detach()
                selector_query = query_fine
                selector_render = fine_batch["eval_selected_fine"].float()
                selector_query_uncertainty = None
                selector_render_uncertainty = None
                if fine_selector_adapter_bundle is not None:
                    nvs_args = fine_selector_adapter_bundle["args"]
                    adapter_needs_rgb = bool(getattr(nvs_args, "pose_feature_adapter_rgb_context_enabled", False)) or bool(
                        getattr(nvs_args, "pose_feature_adapter_texture_branch_enabled", False)
                    )
                    selector_query, selector_render, selector_query_uncertainty, selector_render_uncertainty = (
                        project_query_render_with_pose_feature_adapter(
                            fine_selector_adapter_bundle["adapter"],
                            selector_query,
                            selector_render,
                            query_rgb=batch.get("rgb") if adapter_needs_rgb else None,
                            render_rgb=fine_batch.get("eval_selected_rgb") if adapter_needs_rgb else None,
                            use_uncertainty=bool(getattr(nvs_args, "pose_feature_adapter_uncertainty_enabled", False)),
                            render_chunk_size=int(map_cfg.get("fine_topk_selector_projector_chunk_size", 0) or 0),
                        )
                    )
                elif bool(map_cfg.get("fine_topk_selector_use_projector", True)):
                    selector_query, selector_render, _projector_used = project_query_render_for_fine_selector(
                        getattr(model, "local_corr_projector", None),
                        selector_query,
                        selector_render,
                        require_projector=bool(map_cfg.get("fine_topk_selector_require_projector", False)),
                        render_chunk_size=int(map_cfg.get("fine_topk_selector_projector_chunk_size", 0) or 0),
                    )
                score_map_selector = bool(getattr(selector_head, "expects_score_map", False))
                feature_pack = fine_candidate_selector_features(
                    selector_query.float(),
                    selector_render.float(),
                    fine_batch["eval_selected_pose"].float(),
                    init_pose=init_pose.float(),
                    coarse_logits=selected_coarse_logits.float(),
                    query_rgb=batch.get("rgb"),
                    candidate_rgb=fine_batch.get("eval_selected_rgb"),
                    depth=fine_batch.get("eval_selected_depth"),
                    mask=fine_masks,
                    candidate_valid_mask=fine_batch["eval_selected_valid_mask"].bool(),
                    mode=map_cfg.get("fine_topk_selector_mode", "local"),
                    radius=int(map_cfg.get("fine_topk_selector_radius") or map_cfg.get("query_corr_radius", 4)),
                    preprocess=(
                        map_cfg.get("fine_topk_selector_preprocess")
                        or map_cfg.get(
                            "query_corr_feature_preprocess",
                            map_cfg.get("candidate_score_fusion_preprocess", "spatial_center"),
                        )
                    ),
                    highpass_kernel=int(
                        map_cfg.get("fine_topk_selector_highpass_kernel")
                        or map_cfg.get("query_corr_highpass_kernel", map_cfg.get("candidate_score_fusion_highpass_kernel", 5))
                    ),
                    score_map_mode=map_cfg.get("fine_topk_selector_score_map_mode", "peak_offset"),
                    return_score_maps=score_map_selector,
                    use_coarse_logits=bool(map_cfg.get("fine_topk_selector_use_coarse_logits", True)),
                    use_candidate_delta=bool(map_cfg.get("fine_topk_selector_use_candidate_delta", True)),
                    use_delta_vector=bool(map_cfg.get("fine_topk_selector_use_delta_vector", False)),
                    use_depth=bool(map_cfg.get("fine_topk_selector_use_depth", True)),
                    use_mask=bool(map_cfg.get("fine_topk_selector_use_mask", True)),
                    use_rgb=bool(map_cfg.get("fine_topk_selector_use_rgb", False)),
                    query_uncertainty=selector_query_uncertainty,
                    candidate_uncertainty=selector_render_uncertainty,
                    use_uncertainty=bool(map_cfg.get("fine_topk_selector_use_uncertainty", False)),
                )
                score_source = str(args.fine_selector_score_source or "local_corr").lower()
                if score_source == "pair_matcher_heatmap":
                    if fine_selector_adapter_bundle is None:
                        raise RuntimeError("--fine-selector-score-source pair_matcher_heatmap requires --fine-selector-adapter-checkpoint")
                    pair_matcher = fine_selector_adapter_bundle.get("pair_matcher")
                    if pair_matcher is None:
                        raise RuntimeError("fine selector adapter checkpoint does not contain a pair matcher")
                    from feature_extract.tools.train_nvs_pose_feature_adapter import pair_matcher_local_candidate_score_maps

                    nvs_args = fine_selector_adapter_bundle["args"]
                    pair_score_maps, pair_valid_map = pair_matcher_local_candidate_score_maps(
                        pair_matcher,
                        selector_query.float(),
                        selector_render.float(),
                        mask=fine_masks,
                        radius=int(getattr(nvs_args, "pair_matcher_radius", 3)),
                        stride=int(getattr(nvs_args, "pair_matcher_score_stride", 8)),
                        temperature=float(getattr(nvs_args, "pair_matcher_temperature", 0.05)),
                        chunk_points=int(getattr(nvs_args, "pair_matcher_score_chunk_points", 65536)),
                        candidate_score_mode=str(
                            getattr(nvs_args, "pair_matcher_candidate_score_mode", "center_logprob_margin")
                        ),
                    )
                    feature_pack["score_maps"] = pair_score_maps
                    feature_pack["valid"] = feature_pack["valid"] & pair_valid_map.flatten(2).any(dim=2)
                elif score_source != "local_corr":
                    raise RuntimeError(f"Unknown fine selector score source: {score_source}")
                if score_map_selector:
                    fine_selector_logits = selector_head(feature_pack["score_maps"], feature_pack["features"])
                else:
                    fine_selector_logits = selector_head(feature_pack["features"])
                fine_selector_logits = fine_selector_logits.masked_fill(
                    ~feature_pack["valid"],
                    -1.0e6,
                )
                fine_selector_idx = fine_selector_logits.argmax(dim=1)
            if args.fine_select == "pose_energy" or pose_energy_bundle is not None:
                if pose_energy_bundle is None:
                    raise RuntimeError("--fine-select pose_energy requires --pose-energy-checkpoint")
                from feature_extract.tools.train_nvs_pose_feature_adapter import (
                    pair_matcher_local_candidate_score_maps,
                    project_render_bank,
                    project_render_bank_with_uncertainty,
                )

                nvs_args = pose_energy_bundle["args"]
                pose_adapter = pose_energy_bundle["adapter"]
                energy_net = pose_energy_bundle["energy_net"]
                pair_matcher = pose_energy_bundle["pair_matcher"]
                render_chunk_size = int(map_cfg.get("candidate_render_score_projector_chunk_size", 0) or 0)
                adapter_needs_rgb = bool(getattr(nvs_args, "pose_feature_adapter_rgb_context_enabled", False)) or bool(
                    getattr(nvs_args, "pose_feature_adapter_texture_branch_enabled", False)
                )
                adapter_query_rgb = batch.get("rgb") if adapter_needs_rgb else None
                adapter_render_rgb = fine_batch.get("eval_selected_rgb") if adapter_needs_rgb else None
                energy_use_uncertainty = bool(getattr(nvs_args, "pose_energy_use_uncertainty", False))
                if energy_use_uncertainty:
                    query_loc, query_uncertainty = pose_adapter.project_query_with_uncertainty(
                        query_fine,
                        rgb=adapter_query_rgb,
                    )
                    render_loc, render_uncertainty = project_render_bank_with_uncertainty(
                        pose_adapter,
                        fine_batch["eval_selected_fine"].float(),
                        render_rgb=adapter_render_rgb,
                        render_chunk_size=render_chunk_size,
                    )
                else:
                    query_loc = pose_adapter.project_query(query_fine, rgb=adapter_query_rgb)
                    query_uncertainty = None
                    render_uncertainty = None
                    render_loc = project_render_bank(
                        pose_adapter,
                        fine_batch["eval_selected_fine"].float(),
                        render_rgb=adapter_render_rgb,
                        render_chunk_size=render_chunk_size,
                    )
                energy_feature_pack = fine_candidate_selector_features(
                    query_loc.float(),
                    render_loc.float(),
                    fine_batch["eval_selected_pose"].float(),
                    init_pose=init_pose.float(),
                    query_rgb=batch.get("rgb"),
                    candidate_rgb=fine_batch.get("eval_selected_rgb"),
                    depth=fine_batch.get("eval_selected_depth"),
                    mask=fine_masks,
                    candidate_valid_mask=fine_batch["eval_selected_valid_mask"].bool(),
                    mode="local",
                    radius=int(getattr(nvs_args, "local_corr_radius", map_cfg.get("query_corr_radius", 4))),
                    preprocess=str(getattr(nvs_args, "pose_energy_score_preprocess", "spatial_center")),
                    highpass_kernel=int(getattr(nvs_args, "pose_energy_score_highpass_kernel", 5)),
                    score_map_mode=str(getattr(nvs_args, "pose_energy_score_map_mode", "peak_offset")),
                    return_score_maps=True,
                    use_coarse_logits=False,
                    use_candidate_delta=bool(getattr(nvs_args, "pose_energy_use_candidate_delta", True)),
                    use_delta_vector=bool(getattr(nvs_args, "pose_energy_use_delta_vector", False)),
                    use_center_delta_vector=bool(getattr(nvs_args, "pose_energy_use_center_delta_vector", False)),
                    use_depth=True,
                    use_mask=True,
                    use_rgb=bool(getattr(nvs_args, "pose_energy_use_rgb", False)),
                    query_uncertainty=query_uncertainty,
                    candidate_uncertainty=render_uncertainty,
                    use_uncertainty=energy_use_uncertainty,
                )
                score_source = str(getattr(nvs_args, "pose_energy_score_source", "local_corr") or "local_corr").lower()
                if score_source == "pair_matcher_heatmap":
                    if pair_matcher is None:
                        raise RuntimeError("pose_energy_score_source=pair_matcher_heatmap requires pair matcher state")
                    pair_score_maps, pair_valid_map = pair_matcher_local_candidate_score_maps(
                        pair_matcher,
                        query_loc.float(),
                        render_loc.float(),
                        mask=fine_masks,
                        radius=int(getattr(nvs_args, "pair_matcher_radius", 3)),
                        stride=int(getattr(nvs_args, "pair_matcher_score_stride", 8)),
                        temperature=float(getattr(nvs_args, "pair_matcher_temperature", 0.05)),
                        chunk_points=int(getattr(nvs_args, "pair_matcher_score_chunk_points", 65536)),
                        candidate_score_mode=str(
                            getattr(nvs_args, "pair_matcher_candidate_score_mode", "center_logprob_margin")
                        ),
                    )
                    energy_feature_pack["score_maps"] = pair_score_maps
                    energy_feature_pack["valid"] = energy_feature_pack["valid"] & pair_valid_map.flatten(2).any(dim=2)
                elif score_source != "local_corr":
                    raise RuntimeError(f"Unknown pose_energy_score_source: {score_source}")
                energy_out = energy_net(
                    energy_feature_pack["score_maps"],
                    energy_feature_pack["features"],
                    valid_mask=energy_feature_pack["valid"],
                )
                pose_energy_selection = pose_energy_candidate_selection(
                    energy_out,
                    energy_feature_pack["valid"],
                    confidence_weight=float(getattr(nvs_args, "pose_energy_selection_confidence_weight", 0.0)),
                    residual_norm_weight=float(getattr(nvs_args, "pose_energy_selection_residual_norm_weight", 0.0)),
                    residual_trans_scale_m=float(getattr(nvs_args, "pose_energy_residual_trans_scale_m", 0.25)),
                    residual_rot_scale_rad=math.radians(float(getattr(nvs_args, "pose_energy_residual_rot_scale_deg", 5.0))),
                )
                pose_energy_idx = pose_energy_selection["idx"]
            if args.fine_select == "oracle":
                final_idx = oracle_refined_idx
            elif args.fine_select == "conf":
                final_idx = conf_idx
            elif args.fine_select == "fine_score_prior":
                final_idx = fine_prior_idx
            elif args.fine_select == "fine_score":
                final_idx = fine_score_idx
            elif args.fine_select == "fine_selector":
                final_idx = fine_selector_idx
            elif args.fine_select == "pose_energy":
                final_idx = pose_energy_idx
            else:
                final_idx = torch.zeros_like(pred_idx)
            final_trans = refined["trans_err_m"][batch_idx, final_idx]
            final_rot = refined["rot_err_deg"][batch_idx, final_idx]
            wls_conf = refined["conf_mean"][batch_idx, final_idx]
            rows["fine_topk_oracle_trans_mm"].extend(
                (refined["trans_err_m"][batch_idx, oracle_refined_idx] * 1000.0).detach().cpu().tolist()
            )
            rows["fine_topk_oracle_rot_deg"].extend(
                refined["rot_err_deg"][batch_idx, oracle_refined_idx].detach().cpu().tolist()
            )
            rows["fine_topk_conf_trans_mm"].extend(
                (refined["trans_err_m"][batch_idx, conf_idx] * 1000.0).detach().cpu().tolist()
            )
            rows["fine_topk_conf_rot_deg"].extend(refined["rot_err_deg"][batch_idx, conf_idx].detach().cpu().tolist())
            rows["fine_topk_conf_mean"].extend(refined["conf_mean"][batch_idx, conf_idx].detach().cpu().tolist())
            rows["fine_topk_score_trans_mm"].extend(
                (refined["trans_err_m"][batch_idx, fine_score_idx] * 1000.0).detach().cpu().tolist()
            )
            rows["fine_topk_score_rot_deg"].extend(refined["rot_err_deg"][batch_idx, fine_score_idx].detach().cpu().tolist())
            rows["fine_topk_score_init_trans_mm"].extend(
                (refined["init_trans_err_m"][batch_idx, fine_score_idx] * 1000.0).detach().cpu().tolist()
            )
            rows["fine_topk_score_init_rot_deg"].extend(
                refined["init_rot_err_deg"][batch_idx, fine_score_idx].detach().cpu().tolist()
            )
            if fine_selector_idx is not None and fine_selector_logits is not None:
                rows["fine_topk_selector_trans_mm"].extend(
                    (refined["trans_err_m"][batch_idx, fine_selector_idx] * 1000.0).detach().cpu().tolist()
                )
                rows["fine_topk_selector_rot_deg"].extend(
                    refined["rot_err_deg"][batch_idx, fine_selector_idx].detach().cpu().tolist()
                )
                rows["fine_topk_selector_oracle_gap_mm"].extend(
                    (
                        (refined["trans_err_m"][batch_idx, fine_selector_idx] - refined["trans_err_m"][batch_idx, oracle_refined_idx])
                        * 1000.0
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                selector_probs = torch.softmax(fine_selector_logits, dim=1)
                selector_entropy = -(selector_probs * torch.log(selector_probs.clamp(min=1e-8))).sum(dim=1)
                if fine_selector_logits.shape[1] > 1:
                    selector_top2 = fine_selector_logits.topk(k=2, dim=1).values
                    selector_margin = selector_top2[:, 0] - selector_top2[:, 1]
                else:
                    selector_margin = torch.zeros_like(selector_entropy)
                rows["fine_topk_selector_entropy"].extend(selector_entropy.detach().cpu().tolist())
                rows["fine_topk_selector_margin"].extend(selector_margin.detach().cpu().tolist())
            if pose_energy_idx is not None and pose_energy_selection is not None:
                rows["fine_topk_pose_energy_trans_mm"].extend(
                    (refined["trans_err_m"][batch_idx, pose_energy_idx] * 1000.0).detach().cpu().tolist()
                )
                rows["fine_topk_pose_energy_rot_deg"].extend(
                    refined["rot_err_deg"][batch_idx, pose_energy_idx].detach().cpu().tolist()
                )
                rows["fine_topk_pose_energy_oracle_gap_mm"].extend(
                    (
                        (refined["trans_err_m"][batch_idx, pose_energy_idx] - refined["trans_err_m"][batch_idx, oracle_refined_idx])
                        * 1000.0
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                rows["fine_topk_pose_energy_entropy"].extend(pose_energy_selection["entropy"].detach().cpu().tolist())
                rows["fine_topk_pose_energy_margin"].extend(pose_energy_selection["margin"].detach().cpu().tolist())

        rows["init_trans_mm"].extend((init_trans * 1000.0).cpu().tolist())
        rows["init_rot_deg"].extend(init_rot.cpu().tolist())
        rows["coarse_top1_trans_mm"].extend((coarse_trans * 1000.0).detach().cpu().tolist())
        rows["coarse_top1_rot_deg"].extend(coarse_rot.detach().cpu().tolist())
        rows["oracle_best_trans_mm"].extend((oracle_best_trans * 1000.0).detach().cpu().tolist())
        rows["oracle_best_rot_deg"].extend(oracle_best_rot.detach().cpu().tolist())
        rows["oracle_gain_mm"].extend(((init_trans - oracle_best_trans) * 1000.0).detach().cpu().tolist())
        rows["final_trans_mm"].extend((final_trans * 1000.0).detach().cpu().tolist())
        rows["final_rot_deg"].extend(final_rot.detach().cpu().tolist())
        rows["fine_gain_mm"].extend(((coarse_trans - final_trans) * 1000.0).detach().cpu().tolist())
        rows["wls_conf_mean"].extend(wls_conf.detach().cpu().tolist())

    init_t = torch.tensor(rows["init_trans_mm"])
    final_t = torch.tensor(rows["final_trans_mm"])
    coarse_t = torch.tensor(rows["coarse_top1_trans_mm"])
    summary = {
        "samples": len(rows["init_trans_mm"]),
        "bucket_trans_cm": float(trans_cm),
        "bucket_rot_deg": float(rot_deg),
        "lattice_trans_cm": lattice_trans,
        "lattice_rot_deg": lattice_rot,
        "max_candidates": int(max_candidates),
        "limit_strategy": args.limit_strategy,
        "fine_pool_mode": str(args.fine_pool_mode),
        "fine_pool_topm": int(args.fine_pool_topm),
        "combine_trans_rot": combine_trans_rot,
        "oracle_basin_recall": float(torch.tensor(oracle_hits).mean().item()) if oracle_hits else 0.0,
        "gain_positive_frac": float((final_t < init_t).float().mean().item()) if len(init_t) else 0.0,
        "coarse_gain_positive_frac": float((coarse_t < init_t).float().mean().item()) if len(init_t) else 0.0,
        "final_gain_mm_mean": float((init_t - final_t).mean().item()) if len(init_t) else 0.0,
        "coarse_gain_mm_mean": float((init_t - coarse_t).mean().item()) if len(init_t) else 0.0,
    }
    for key, values in rows.items():
        stats = tensor_stats(values)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_median"] = stats["median"]
    for k, values in topk_hits.items():
        summary[f"top{k}_basin_recall"] = float(torch.tensor(values).mean().item()) if values else 0.0
    return name, summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    requested_device = args.device
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    model, loader, map_renderer = build_model_and_data(cfg, args, device)
    pose_energy_bundle = None
    if args.fine_select == "pose_energy" or args.pose_energy_checkpoint:
        pose_energy_bundle = load_pose_energy_selector_bundle(args.pose_energy_checkpoint, model, cfg, device)
    fine_selector_adapter_bundle = None
    if args.fine_selector_adapter_checkpoint:
        fine_selector_adapter_bundle = load_pose_feature_adapter_bundle(
            args.fine_selector_adapter_checkpoint,
            model,
            cfg,
            device,
        )
    results = {}
    for bucket in parse_bucket_specs(args.buckets):
        name, summary = evaluate_bucket(
            model,
            loader,
            map_renderer,
            cfg,
            args,
            bucket,
            pose_energy_bundle=pose_energy_bundle,
            fine_selector_adapter_bundle=fine_selector_adapter_bundle,
        )
        results[name] = summary
        print(
            f"{name}: init={summary['init_trans_mm_median']:.1f}mm "
            f"coarse={summary['coarse_top1_trans_mm_median']:.1f}mm "
            f"oracle_best={summary['oracle_best_trans_mm_median']:.1f}mm "
            f"final={summary['final_trans_mm_median']:.1f}mm "
            f"top4={summary.get('top4_basin_recall', 0.0):.3f} "
            f"oracle={summary['oracle_basin_recall']:.3f} "
            f"gain+={summary['gain_positive_frac']:.3f}"
        )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
