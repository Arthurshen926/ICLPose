#!/usr/bin/env python3
"""Train a fine candidate selector from exported compact evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.train_impl import (  # noqa: E402
    FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES,
    FINE_CANDIDATE_SELECTOR_RGB_FEATURE_NAMES,
    FINE_CANDIDATE_SELECTOR_VECTOR_FEATURE_NAMES,
    TeacherFeatureStore,
    build_radio_query_student,
    fine_topk_selector_cached_listwise_loss,
    load_config,
    load_model_warmstart,
    resolve_query_feature_dims,
    safe_torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Warmstart checkpoint for the full query model")
    parser.add_argument("--map-checkpoint", default=None, help="Optional checkpoint providing map_renderer_state_dict")
    parser.add_argument("--cache", required=True, nargs="+")
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=973)
    parser.add_argument("--target-mode", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--target-temperature-m", type=float, default=None)
    parser.add_argument("--pairwise-rank-weight", type=float, default=None)
    parser.add_argument("--pairwise-rank-min-gap-m", type=float, default=None)
    parser.add_argument("--pairwise-rank-temperature", type=float, default=None)
    parser.add_argument("--rot-cost-weight", type=float, default=None)
    parser.add_argument(
        "--zero-feature-indices",
        default=None,
        help="Optional comma-separated vector feature column indices to zero in train/val caches.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validation reports without best-metric improvement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--diagnose-score-map-ablation",
        action="store_true",
        help="When score maps are present, also evaluate validation with candidate score maps shuffled.",
    )
    parser.add_argument(
        "--score-map-shuffle-seed",
        type=int,
        default=973,
        help="Seed used by --diagnose-score-map-ablation.",
    )
    parser.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="Randomly permute candidate order within each training row every step.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate cached selector evidence and diagnostics; do not train or save checkpoints.",
    )
    return parser.parse_args()


def _load_cache_tensors(path: str):
    cache = torch.load(path, map_location="cpu")
    features = cache["features"].float()
    valid = cache["valid"].bool()
    trans_err_m = cache["trans_err_m"].float()
    rot_err_deg = cache["rot_err_deg"].float()
    tensors = [features, valid, trans_err_m, rot_err_deg]
    if "score_maps" in cache:
        tensors.append(cache["score_maps"].float())
    return cache, TensorDataset(*tensors)


def _concat_cache_tensors(paths):
    caches = []
    datasets = []
    for path in paths:
        cache, dataset = _load_cache_tensors(str(path))
        caches.append(cache)
        datasets.append(dataset)
    if not datasets:
        raise RuntimeError("at least one cache path is required")
    ref_shapes = [tuple(t.shape[1:]) for t in datasets[0].tensors]
    ref_count = len(datasets[0].tensors)
    for dataset in datasets[1:]:
        if len(dataset.tensors) != ref_count:
            raise RuntimeError("all train caches must have the same tensor fields")
        shapes = [tuple(t.shape[1:]) for t in dataset.tensors]
        if shapes != ref_shapes:
            raise RuntimeError("all train caches must have matching topK/feature/score-map shapes")
    tensors = [torch.cat([dataset.tensors[idx] for dataset in datasets], dim=0) for idx in range(ref_count)]
    merged = dict(caches[0])
    merged["meta"] = {
        "sources": [cache.get("meta", {}) for cache in caches],
        "num_samples": int(tensors[0].shape[0]),
    }
    return merged, TensorDataset(*tensors)


def _parse_int_csv(value: str | None):
    if value is None:
        return []
    return [int(part) for part in str(value).split(",") if part.strip()]


def _zero_feature_columns(dataset: TensorDataset, indices):
    if not indices:
        return dataset
    features = dataset.tensors[0].clone()
    feature_dim = int(features.shape[-1])
    for idx in indices:
        if idx < 0 or idx >= feature_dim:
            raise ValueError(f"zero feature index {idx} is outside [0, {feature_dim - 1}]")
        features[:, :, idx] = 0.0
    return TensorDataset(features, *dataset.tensors[1:])


def _dataset_has_score_maps(dataset) -> bool:
    base = getattr(dataset, "dataset", dataset)
    return bool(hasattr(base, "tensors") and len(base.tensors) > 4)


def _dataset_score_maps_shape(dataset):
    base = getattr(dataset, "dataset", dataset)
    if not hasattr(base, "tensors") or len(base.tensors) <= 4:
        return None
    return list(base.tensors[4].shape)


def _mean_metric(items, key):
    if not items:
        return 0.0
    vals = [float(item[key].detach().cpu().item()) for item in items if key in item]
    return float(sum(vals) / max(len(vals), 1))


def _masked_first_valid(valid):
    if valid.ndim != 2:
        raise ValueError(f"valid must have shape B,K, got {tuple(valid.shape)}")
    first = valid.float().argmax(dim=1)
    if not valid.any(dim=1).all():
        raise ValueError("each row must contain at least one valid candidate")
    return first


def _selected_trans_mm(trans_err_m, index):
    row = torch.arange(trans_err_m.shape[0], device=trans_err_m.device)
    return trans_err_m[row, index] * 1000.0


def _masked_best_index(values, valid, *, largest=True):
    masked_value = -1.0e6 if largest else 1.0e6
    masked = values.masked_fill(~valid, masked_value)
    return masked.argmax(dim=1) if largest else masked.argmin(dim=1)


def _rank_vector(values, *, descending=True):
    order = torch.argsort(values, descending=descending)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _masked_spearman(scores, pose_cost, valid):
    rows = []
    for row_idx in range(scores.shape[0]):
        row_valid = valid[row_idx]
        if int(row_valid.sum().item()) < 2:
            continue
        row_scores = scores[row_idx, row_valid]
        row_targets = -pose_cost[row_idx, row_valid]
        if (
            float((row_scores.max() - row_scores.min()).detach().cpu().item()) <= 1e-8
            or float((row_targets.max() - row_targets.min()).detach().cpu().item()) <= 1e-8
        ):
            rows.append(torch.zeros((), device=scores.device, dtype=torch.float32))
            continue
        score_rank = _rank_vector(row_scores, descending=True)
        target_rank = _rank_vector(row_targets, descending=True)
        score_centered = score_rank - score_rank.mean()
        target_centered = target_rank - target_rank.mean()
        denom = score_centered.norm() * target_centered.norm()
        if float(denom.detach().cpu().item()) <= 0.0:
            continue
        rows.append((score_centered * target_centered).sum() / denom)
    if not rows:
        return torch.zeros((), device=scores.device, dtype=torch.float32)
    return torch.stack(rows).mean()


def _masked_good_bad_auc(scores, pose_cost, valid):
    rows = []
    for row_idx in range(scores.shape[0]):
        row_valid = valid[row_idx]
        valid_count = int(row_valid.sum().item())
        if valid_count < 2:
            continue
        row_scores = scores[row_idx, row_valid]
        row_cost = pose_cost[row_idx, row_valid]
        cost_order = torch.argsort(row_cost, descending=False)
        good_count = max(1, int(math.ceil(valid_count * 0.10)))
        bad_start = min(valid_count - 1, max(good_count, int(math.floor(valid_count * 0.50))))
        good_scores = row_scores[cost_order[:good_count]]
        bad_scores = row_scores[cost_order[bad_start:]]
        if bad_scores.numel() == 0:
            continue
        comparisons = good_scores[:, None] - bad_scores[None, :]
        rows.append((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())
    if not rows:
        return torch.zeros((), device=scores.device, dtype=torch.float32)
    return torch.stack(rows).mean()


def _cached_rank_diagnostics(features, valid, trans_err_m, rot_err_deg, selector_logits, *, rot_cost_weight=0.1):
    pose_cost = trans_err_m + float(rot_cost_weight) * (rot_err_deg * (math.pi / 180.0))
    pose_cost = pose_cost.masked_fill(~valid, 1.0e6)
    oracle_idx = pose_cost.argmin(dim=1)
    candidate0_idx = torch.zeros_like(oracle_idx)
    candidate0_idx = torch.where(valid[:, 0], candidate0_idx, _masked_first_valid(valid))
    diagnostics = {
        "oracle_trans_mm": _selected_trans_mm(trans_err_m, oracle_idx).mean().detach(),
        "candidate0_trans_mm": _selected_trans_mm(trans_err_m, candidate0_idx).mean().detach(),
        "selector_spearman": _masked_spearman(selector_logits, pose_cost, valid).detach(),
        "selector_good_bad_auc": _masked_good_bad_auc(selector_logits, pose_cost, valid).detach(),
    }
    feature_names = list(FINE_CANDIDATE_SELECTOR_VECTOR_FEATURE_NAMES)
    if features.shape[-1] >= len(feature_names) + len(FINE_CANDIDATE_SELECTOR_RGB_FEATURE_NAMES):
        feature_names.extend(FINE_CANDIDATE_SELECTOR_RGB_FEATURE_NAMES)
    if features.shape[-1] >= len(feature_names) + len(FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES):
        feature_names.extend(FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES)
    baseline_columns = {
        "score_mean": ("score_mean", True),
        "score_max": ("score_max", True),
        "score_topk_mean": ("score_topk_mean", True),
        "score_peakiness": ("score_peakiness", True),
        "coarse_logit": ("coarse_logit", True),
        "coarse_rank_norm": ("coarse_rank_norm", True),
        "rgb_l1_mean": ("rgb_l1_mean", False),
        "rgb_l1_zscore": ("rgb_l1_zscore", True),
        "rgb_valid_fraction": ("rgb_valid_fraction", True),
    }
    for metric_prefix, (feature_name, largest) in baseline_columns.items():
        if feature_name not in feature_names:
            continue
        feature_idx = feature_names.index(feature_name)
        if feature_idx >= features.shape[-1]:
            continue
        scores = features[:, :, feature_idx].to(device=trans_err_m.device, dtype=trans_err_m.dtype)
        pred_idx = _masked_best_index(scores, valid, largest=largest)
        rank_scores = scores if largest else -scores
        diagnostics[f"{metric_prefix}_pred_trans_mm"] = _selected_trans_mm(trans_err_m, pred_idx).mean().detach()
        diagnostics[f"{metric_prefix}_spearman"] = _masked_spearman(rank_scores, pose_cost, valid).detach()
        diagnostics[f"{metric_prefix}_good_bad_auc"] = _masked_good_bad_auc(rank_scores, pose_cost, valid).detach()
    if "delta_trans_m" in feature_names:
        feature_idx = feature_names.index("delta_trans_m")
        if feature_idx < features.shape[-1]:
            delta_scores = -features[:, :, feature_idx].to(device=trans_err_m.device, dtype=trans_err_m.dtype)
            pred_idx = _masked_best_index(delta_scores, valid, largest=True)
            diagnostics["delta_min_pred_trans_mm"] = _selected_trans_mm(trans_err_m, pred_idx).mean().detach()
            diagnostics["delta_min_spearman"] = _masked_spearman(delta_scores, pose_cost, valid).detach()
            diagnostics["delta_min_good_bad_auc"] = _masked_good_bad_auc(delta_scores, pose_cost, valid).detach()
    return diagnostics


def _shuffle_score_maps_for_ablation(score_maps, *, seed: int, batch_index: int):
    if score_maps is None:
        return None
    if score_maps.ndim < 2:
        raise ValueError(f"score_maps must have at least B,K dimensions, got {tuple(score_maps.shape)}")
    num_candidates = int(score_maps.shape[1])
    if num_candidates <= 1:
        return score_maps
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(batch_index))
    perm = torch.randperm(num_candidates, generator=generator).to(device=score_maps.device)
    return score_maps[:, perm]


def _shuffle_candidate_batch(features, valid, trans_err_m, rot_err_deg, score_maps=None, *, generator=None):
    if features.ndim != 3:
        raise ValueError(f"features must have shape B,K,D, got {tuple(features.shape)}")
    bsz, num_candidates = features.shape[:2]
    if num_candidates <= 1:
        return features, valid, trans_err_m, rot_err_deg, score_maps
    perm_rows = [
        torch.randperm(num_candidates, generator=generator).to(device=features.device)
        for _ in range(bsz)
    ]
    perm = torch.stack(perm_rows, dim=0)

    def gather_rows(values):
        if values is None:
            return None
        view_shape = (bsz, num_candidates) + (1,) * (values.ndim - 2)
        expand_shape = (bsz, num_candidates) + tuple(values.shape[2:])
        return values.gather(1, perm.reshape(view_shape).expand(expand_shape))

    return (
        gather_rows(features),
        gather_rows(valid),
        gather_rows(trans_err_m),
        gather_rows(rot_err_deg),
        gather_rows(score_maps),
    )


def _evaluate(selector, loader, device, loss_kwargs, *, score_map_shuffle_seed=None):
    selector.eval()
    metrics = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            features, valid, trans_err_m, rot_err_deg = [item.to(device) for item in batch[:4]]
            score_maps = batch[4].to(device) if len(batch) > 4 else None
            if score_map_shuffle_seed is not None:
                score_maps = _shuffle_score_maps_for_ablation(
                    score_maps,
                    seed=int(score_map_shuffle_seed),
                    batch_index=batch_index,
                )
            loss, batch_metrics, details = fine_topk_selector_cached_listwise_loss(
                features,
                valid,
                trans_err_m,
                rot_err_deg,
                selector,
                score_maps=score_maps,
                return_details=True,
                **loss_kwargs,
            )
            batch_metrics.update(
                _cached_rank_diagnostics(
                    features,
                    valid,
                    trans_err_m,
                    rot_err_deg,
                    details["raw_logits"],
                    rot_cost_weight=float(loss_kwargs.get("rot_cost_weight", 0.1)),
                )
            )
            metrics.append(batch_metrics)
            total_loss += float(loss.detach().cpu().item()) * int(features.shape[0])
            total_count += int(features.shape[0])
    selector.train()
    return {
        "loss": total_loss / max(total_count, 1),
        "pred_trans_mm": _mean_metric(metrics, "map_fine_topk_selector_pred_trans_mm"),
        "oracle_gap_mm": _mean_metric(metrics, "map_fine_topk_selector_oracle_gap_mm"),
        "acc": _mean_metric(metrics, "map_fine_topk_selector_acc"),
        "entropy": _mean_metric(metrics, "map_fine_topk_selector_entropy"),
        "oracle_trans_mm": _mean_metric(metrics, "oracle_trans_mm"),
        "candidate0_trans_mm": _mean_metric(metrics, "candidate0_trans_mm"),
        "selector_spearman": _mean_metric(metrics, "selector_spearman"),
        "selector_good_bad_auc": _mean_metric(metrics, "selector_good_bad_auc"),
        "score_mean_pred_trans_mm": _mean_metric(metrics, "score_mean_pred_trans_mm"),
        "score_mean_spearman": _mean_metric(metrics, "score_mean_spearman"),
        "score_mean_good_bad_auc": _mean_metric(metrics, "score_mean_good_bad_auc"),
        "score_max_pred_trans_mm": _mean_metric(metrics, "score_max_pred_trans_mm"),
        "score_max_spearman": _mean_metric(metrics, "score_max_spearman"),
        "score_max_good_bad_auc": _mean_metric(metrics, "score_max_good_bad_auc"),
        "score_topk_mean_pred_trans_mm": _mean_metric(metrics, "score_topk_mean_pred_trans_mm"),
        "score_topk_mean_spearman": _mean_metric(metrics, "score_topk_mean_spearman"),
        "score_topk_mean_good_bad_auc": _mean_metric(metrics, "score_topk_mean_good_bad_auc"),
        "score_peakiness_pred_trans_mm": _mean_metric(metrics, "score_peakiness_pred_trans_mm"),
        "score_peakiness_spearman": _mean_metric(metrics, "score_peakiness_spearman"),
        "score_peakiness_good_bad_auc": _mean_metric(metrics, "score_peakiness_good_bad_auc"),
        "coarse_logit_pred_trans_mm": _mean_metric(metrics, "coarse_logit_pred_trans_mm"),
        "coarse_logit_spearman": _mean_metric(metrics, "coarse_logit_spearman"),
        "coarse_logit_good_bad_auc": _mean_metric(metrics, "coarse_logit_good_bad_auc"),
        "coarse_rank_norm_pred_trans_mm": _mean_metric(metrics, "coarse_rank_norm_pred_trans_mm"),
        "coarse_rank_norm_spearman": _mean_metric(metrics, "coarse_rank_norm_spearman"),
        "coarse_rank_norm_good_bad_auc": _mean_metric(metrics, "coarse_rank_norm_good_bad_auc"),
        "rgb_l1_mean_pred_trans_mm": _mean_metric(metrics, "rgb_l1_mean_pred_trans_mm"),
        "rgb_l1_mean_spearman": _mean_metric(metrics, "rgb_l1_mean_spearman"),
        "rgb_l1_mean_good_bad_auc": _mean_metric(metrics, "rgb_l1_mean_good_bad_auc"),
        "rgb_l1_zscore_pred_trans_mm": _mean_metric(metrics, "rgb_l1_zscore_pred_trans_mm"),
        "rgb_l1_zscore_spearman": _mean_metric(metrics, "rgb_l1_zscore_spearman"),
        "rgb_l1_zscore_good_bad_auc": _mean_metric(metrics, "rgb_l1_zscore_good_bad_auc"),
        "rgb_valid_fraction_pred_trans_mm": _mean_metric(metrics, "rgb_valid_fraction_pred_trans_mm"),
        "rgb_valid_fraction_spearman": _mean_metric(metrics, "rgb_valid_fraction_spearman"),
        "rgb_valid_fraction_good_bad_auc": _mean_metric(metrics, "rgb_valid_fraction_good_bad_auc"),
        "delta_min_pred_trans_mm": _mean_metric(metrics, "delta_min_pred_trans_mm"),
        "delta_min_spearman": _mean_metric(metrics, "delta_min_spearman"),
        "delta_min_good_bad_auc": _mean_metric(metrics, "delta_min_good_bad_auc"),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    cfg = load_config(args.config)
    map_cfg = cfg.get("map_supervision", {})
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cache_paths = args.cache if isinstance(args.cache, (list, tuple)) else [args.cache]
    cache, dataset = _concat_cache_tensors(cache_paths)
    zero_feature_indices = _parse_int_csv(args.zero_feature_indices)
    dataset = _zero_feature_columns(dataset, zero_feature_indices)
    features = dataset.tensors[0]
    val_cache = None
    val_dataset = None
    if args.val_cache:
        val_cache, val_dataset = _load_cache_tensors(args.val_cache)
        val_dataset = _zero_feature_columns(val_dataset, zero_feature_indices)
        if int(val_dataset.tensors[0].shape[1]) != int(features.shape[1]):
            raise RuntimeError("train cache and val cache must use the same topK")
        if int(val_dataset.tensors[0].shape[2]) != int(features.shape[2]):
            raise RuntimeError("train cache and val cache must use the same selector feature dimension")
        train_has_maps = len(dataset.tensors) > 4
        val_has_maps = len(val_dataset.tensors) > 4
        if train_has_maps != val_has_maps:
            raise RuntimeError("train cache and val cache must both include score_maps or both omit them")

    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    model = build_radio_query_student(cfg, fine_feature_dim=fine_dim, coarse_feature_dim=coarse_dim).to(device)
    checkpoint = safe_torch_load(args.checkpoint)
    if args.map_checkpoint:
        map_renderer_state = safe_torch_load(args.map_checkpoint).get("map_renderer_state_dict")
    else:
        map_renderer_state = checkpoint.get("map_renderer_state_dict")
    load_model_warmstart(model, checkpoint, strict=False)
    selector = getattr(model, "fine_candidate_selector_head", None)
    if selector is None:
        raise RuntimeError("config must enable model.fine_candidate_selector_head")
    if bool(getattr(selector, "expects_score_map", False)) and len(dataset.tensors) <= 4:
        raise RuntimeError("score-map fine selector training requires cache['score_maps']")
    selector_info = {
        "selector_class": selector.__class__.__name__,
        "expects_score_map": bool(getattr(selector, "expects_score_map", False)),
        "train_has_score_maps": len(dataset.tensors) > 4,
        "train_score_maps_shape": None if len(dataset.tensors) <= 4 else list(dataset.tensors[4].shape),
        "val_has_score_maps": None if val_dataset is None else len(val_dataset.tensors) > 4,
        "val_score_maps_shape": None
        if val_dataset is None or len(val_dataset.tensors) <= 4
        else list(val_dataset.tensors[4].shape),
    }

    for param in model.parameters():
        param.requires_grad_(False)
    for param in selector.parameters():
        param.requires_grad_(True)
    selector.train()

    generator = torch.Generator().manual_seed(int(args.seed))
    if val_dataset is not None:
        train_set = dataset
        val_set = val_dataset
    else:
        val_count = int(round(len(dataset) * max(0.0, min(float(args.val_fraction), 0.9))))
        train_count = max(1, len(dataset) - val_count)
        val_count = len(dataset) - train_count
        if val_count > 0:
            train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)
        else:
            train_set, val_set = dataset, None
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    val_loader = (
        None
        if val_set is None
        else DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    )
    selector_info["val_has_score_maps"] = None if val_set is None else _dataset_has_score_maps(val_set)
    selector_info["val_score_maps_shape"] = None if val_set is None else _dataset_score_maps_shape(val_set)
    print(json.dumps({"selector_info": selector_info}, sort_keys=True))

    loss_kwargs = {
        "rot_cost_weight": float(
            args.rot_cost_weight
            if args.rot_cost_weight is not None
            else map_cfg.get("fine_topk_selector_rot_cost_weight", 0.1)
        ),
        "temperature": float(
            args.temperature
            if args.temperature is not None
            else map_cfg.get("fine_topk_selector_temperature", 1.0)
        ),
        "target_mode": str(args.target_mode or map_cfg.get("fine_topk_selector_target_mode", "gt_pose_error_soft")),
        "target_temperature_m": float(
            args.target_temperature_m
            if args.target_temperature_m is not None
            else map_cfg.get("fine_topk_selector_target_temperature_m", 0.05)
        ),
        "pairwise_rank_weight": float(
            args.pairwise_rank_weight
            if args.pairwise_rank_weight is not None
            else map_cfg.get("fine_topk_selector_pairwise_rank_weight", 0.0)
        ),
        "pairwise_rank_min_gap_m": float(
            args.pairwise_rank_min_gap_m
            if args.pairwise_rank_min_gap_m is not None
            else map_cfg.get("fine_topk_selector_pairwise_rank_min_gap_m", 0.03)
        ),
        "pairwise_rank_temperature": float(
            args.pairwise_rank_temperature
            if args.pairwise_rank_temperature is not None
            else map_cfg.get("fine_topk_selector_pairwise_rank_temperature", 1.0)
        ),
        "basin_trans_m": float(map_cfg.get("fine_topk_selector_basin_trans_m", 0.25)),
        "basin_rot_deg": float(map_cfg.get("fine_topk_selector_basin_rot_deg", 5.0)),
    }
    if bool(args.eval_only):
        eval_set = val_set if val_set is not None else train_set
        eval_loader = DataLoader(eval_set, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
        eval_report = _evaluate(selector, eval_loader, device, loss_kwargs)
        report = {"eval": eval_report, "selector_info": selector_info}
        if (
            bool(args.diagnose_score_map_ablation)
            and bool(getattr(selector, "expects_score_map", False))
            and _dataset_has_score_maps(eval_set)
        ):
            report["score_map_shuffle_eval"] = _evaluate(
                selector,
                eval_loader,
                device,
                loss_kwargs,
                score_map_shuffle_seed=int(args.score_map_shuffle_seed),
            )
        print(json.dumps(report, sort_keys=True))
        return

    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    out_dir = Path(args.out_dir or Path(cfg.get("output_dir", "result/feature_extract")) / cfg.get("exp_name", "fine_selector_cache"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    best_metric = float("inf")
    best_step = 0
    step = 0
    reports_since_best = 0
    log_path = out_dir / "train_log.jsonl"
    train_iter = iter(train_loader)
    while step < int(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        features_b, valid_b, trans_b, rot_b = [item.to(device) for item in batch[:4]]
        score_maps_b = batch[4].to(device) if len(batch) > 4 else None
        if bool(args.shuffle_candidates):
            features_b, valid_b, trans_b, rot_b, score_maps_b = _shuffle_candidate_batch(
                features_b,
                valid_b,
                trans_b,
                rot_b,
                score_maps_b,
                generator=generator,
            )
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = fine_topk_selector_cached_listwise_loss(
            features_b,
            valid_b,
            trans_b,
            rot_b,
            selector,
            score_maps=score_maps_b,
            **loss_kwargs,
        )
        loss.backward()
        optimizer.step()
        step += 1
        if step % max(int(args.log_every), 1) == 0 or step == 1:
            report = {
                "step": step,
                "train_loss": float(loss.detach().cpu().item()),
                "train_pred_trans_mm": float(metrics["map_fine_topk_selector_pred_trans_mm"].detach().cpu().item()),
                "train_gap_mm": float(metrics["map_fine_topk_selector_oracle_gap_mm"].detach().cpu().item()),
                "train_acc": float(metrics["map_fine_topk_selector_acc"].detach().cpu().item()),
            }
            if val_loader is not None:
                val_report = _evaluate(selector, val_loader, device, loss_kwargs)
                report.update({f"val_{key}": value for key, value in val_report.items()})
                if (
                    bool(args.diagnose_score_map_ablation)
                    and bool(getattr(selector, "expects_score_map", False))
                    and _dataset_has_score_maps(val_set)
                ):
                    shuffled_val_report = _evaluate(
                        selector,
                        val_loader,
                        device,
                        loss_kwargs,
                        score_map_shuffle_seed=int(args.score_map_shuffle_seed),
                    )
                    report.update(
                        {
                            f"val_score_map_shuffle_{key}": value
                            for key, value in shuffled_val_report.items()
                        }
                    )
                metric = float(val_report["pred_trans_mm"])
            else:
                metric = float(report["train_pred_trans_mm"])
            print(json.dumps(report, sort_keys=True))
            with open(log_path, "a") as log_f:
                log_f.write(json.dumps(report, sort_keys=True) + "\n")
            if metric < best_metric:
                best_metric = metric
                best_step = step
                reports_since_best = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "map_renderer_state_dict": map_renderer_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "best_val": best_metric,
                        "cache_meta": cache.get("meta", {}),
                        "val_cache_meta": None if val_cache is None else val_cache.get("meta", {}),
                    },
                    out_dir / "checkpoints" / "best.pth",
                )
            else:
                reports_since_best += 1
            with open(out_dir / "latest_metrics.json", "w") as f:
                json.dump(
                    {
                        "last_report": report,
                        "best_val": best_metric,
                        "best_step": best_step,
                        "reports_since_best": reports_since_best,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            patience = int(args.early_stop_patience or 0)
            if patience > 0 and reports_since_best >= patience:
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "map_renderer_state_dict": map_renderer_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val": best_metric,
            "best_step": best_step,
            "cache_meta": cache.get("meta", {}),
            "val_cache_meta": None if val_cache is None else val_cache.get("meta", {}),
        },
        out_dir / "checkpoints" / "latest.pth",
    )
    with open(out_dir / "cache_train_summary.json", "w") as f:
        json.dump(
            {
                "cache": [str(path) for path in cache_paths],
                "val_cache": None if args.val_cache is None else str(args.val_cache),
                "samples": int(features.shape[0]),
                "val_samples": 0 if val_dataset is None else int(val_dataset.tensors[0].shape[0]),
                "topk": int(features.shape[1]),
                "feature_dim": int(features.shape[2]),
                "best_val": best_metric,
                "best_step": best_step,
                "loss_kwargs": loss_kwargs,
                "selector_info": selector_info,
                "diagnose_score_map_ablation": bool(args.diagnose_score_map_ablation),
                "zero_feature_indices": zero_feature_indices,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps({"done": True, "best_val": best_metric, "best_step": best_step}, sort_keys=True))


if __name__ == "__main__":
    main()
