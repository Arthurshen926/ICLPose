#!/usr/bin/env python3
"""Train a pose-conditioned energy/residual head for learning-based CPR."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.students.pose_energy_net import (  # noqa: E402
    PoseEnergyNet,
    pose_costs_and_residual_targets,
    pose_energy_losses,
)
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES,
    FINE_CANDIDATE_SELECTOR_VECTOR_FEATURE_NAMES,
    build_local_pose_lattice_candidates,
    fine_candidate_selector_features,
    load_config,
    move_batch_to_device,
    project_query_render_for_fine_selector,
    set_seed,
)
from pose_refine import apply_pose_delta  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--resume-pose-energy", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-split", choices=("train", "val"), default="train")
    parser.add_argument("--eval-split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--candidate-render-batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--best-metric", default="pred_cost_m")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-fine-key", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--lattice-trans-cm", default=None)
    parser.add_argument("--lattice-rot-deg", default=None)
    parser.add_argument("--lattice-direction-mode", choices=("axis", "cube"), default=None)
    parser.add_argument("--combine-trans-rot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit-strategy", default=None)
    parser.add_argument("--candidate-center-mode", choices=("gt", "noisy_init"), default=None)
    parser.add_argument("--candidate-center-noise-mode", choices=("uniform", "fixed"), default=None)
    parser.add_argument("--candidate-center-trans-cm", type=float, default=None)
    parser.add_argument("--candidate-center-rot-deg", type=float, default=None)
    parser.add_argument("--target-temperature-m", type=float, default=None)
    parser.add_argument("--rot-cost-weight", type=float, default=None)
    parser.add_argument("--hard-ce-weight", type=float, default=None)
    parser.add_argument("--residual-weight", type=float, default=None)
    parser.add_argument("--improve-weight", type=float, default=None)
    parser.add_argument("--improve-margin-m", type=float, default=None)
    parser.add_argument("--auc-good-m", type=float, default=None)
    parser.add_argument("--auc-bad-m", type=float, default=None)
    parser.add_argument("--score-radius", type=int, default=None)
    parser.add_argument("--score-preprocess", default=None)
    parser.add_argument("--score-highpass-kernel", type=int, default=None)
    parser.add_argument("--score-map-mode", default=None)
    parser.add_argument("--use-delta-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--map-channels", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--context-layers", type=int, default=None)
    parser.add_argument("--context-heads", type=int, default=None)
    parser.add_argument("--synthetic-ratio", type=float, default=None)
    parser.add_argument("--eval-synthetic-ratio", type=float, default=None)
    parser.add_argument("--synthetic-trans-cm", type=float, default=None)
    parser.add_argument("--synthetic-rot-deg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def _csv_or_list(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return str(value)


def parse_float_csv(value) -> List[float]:
    return [float(part) for part in _csv_or_list(value).split(",") if part.strip()]


def apply_config_defaults(args: argparse.Namespace, cfg: Dict) -> argparse.Namespace:
    pose_cfg = dict(cfg.get("pose_energy", {}) or {})
    defaults = {
        "topk": 64,
        "lattice_trans_cm": "0,2,5,10,25,50",
        "lattice_rot_deg": "0,0.5,1,2,5,10",
        "lattice_direction_mode": "cube",
        "combine_trans_rot": False,
        "limit_strategy": "uniform",
        "candidate_center_mode": "gt",
        "candidate_center_noise_mode": "uniform",
        "candidate_center_trans_cm": 25.0,
        "candidate_center_rot_deg": 5.0,
        "target_temperature_m": 0.05,
        "rot_cost_weight": 0.1,
        "hard_ce_weight": 0.0,
        "residual_weight": 0.5,
        "improve_weight": 0.0,
        "improve_margin_m": 0.0,
        "auc_good_m": 0.05,
        "auc_bad_m": 0.25,
        "score_radius": 4,
        "score_preprocess": "spatial_center",
        "score_highpass_kernel": 5,
        "score_map_mode": "peak_offset",
        "use_delta_vector": False,
        "hidden_dim": 128,
        "map_channels": 16,
        "grid_size": 4,
        "context_layers": 1,
        "context_heads": 1,
        "synthetic_ratio": 0.0,
        "synthetic_trans_cm": 25.0,
        "synthetic_rot_deg": 5.0,
    }
    for key, fallback in defaults.items():
        if getattr(args, key, None) is not None:
            continue
        value = pose_cfg.get(key, fallback)
        if key in {"lattice_trans_cm", "lattice_rot_deg"}:
            value = _csv_or_list(value)
        setattr(args, key, value)
    return args


def _set_trainable(model: torch.nn.Module, prefixes: Iterable[str]) -> None:
    allowed = tuple(prefixes)
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in allowed))


def _random_unit_vectors(count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vec = torch.randn(count, 3, device=device, dtype=dtype)
    return vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-6)


def jitter_gt_poses(pose_gt: torch.Tensor, trans_cm: float, rot_deg: float) -> torch.Tensor:
    count = pose_gt.shape[0]
    delta = pose_gt.new_zeros((count, 6))
    if float(trans_cm) > 0.0:
        mag = torch.rand(count, 1, device=pose_gt.device, dtype=pose_gt.dtype)
        delta[:, :3] = _random_unit_vectors(count, device=pose_gt.device, dtype=pose_gt.dtype) * mag * (
            float(trans_cm) / 100.0
        )
    if float(rot_deg) > 0.0:
        mag = torch.rand(count, 1, device=pose_gt.device, dtype=pose_gt.dtype)
        delta[:, 3:] = _random_unit_vectors(count, device=pose_gt.device, dtype=pose_gt.dtype) * mag * math.radians(
            float(rot_deg)
        )
    return apply_pose_delta(pose_gt.float(), delta.float()).to(dtype=pose_gt.dtype)


def maybe_replace_with_synthetic_queries(batch, map_renderer, pose_gt, cfg: Dict, args: argparse.Namespace):
    ratio = float(args.synthetic_ratio)
    if ratio <= 0.0:
        return batch, pose_gt, torch.zeros((pose_gt.shape[0],), device=pose_gt.device, dtype=torch.bool)
    bsz = pose_gt.shape[0]
    synthetic_mask = torch.rand(bsz, device=pose_gt.device) < ratio
    if not bool(synthetic_mask.any()):
        return batch, pose_gt, synthetic_mask
    input_hw = tuple(int(v) for v in cfg["dataset"]["input_hw"])
    pose_out = pose_gt.clone()
    rgb_out = batch["rgb"].clone()
    with torch.no_grad():
        synth_pose = jitter_gt_poses(
            pose_gt[synthetic_mask],
            trans_cm=float(args.synthetic_trans_cm),
            rot_deg=float(args.synthetic_rot_deg),
        )
        synth_indices = torch.nonzero(synthetic_mask, as_tuple=False).flatten().tolist()
        for local_idx, batch_idx in enumerate(synth_indices):
            sample_name = batch["sample_name"][batch_idx]
            _fine_raw, _fine, _coarse, _mask, _alpha, rgb, _depth, _position = map_renderer._render_pose(
                sample_name,
                synth_pose[local_idx],
                require_grad=False,
                feature="coarse",
                include_aux=True,
            )
            rgb = F.interpolate(rgb.float(), size=input_hw, mode="bilinear", align_corners=False).squeeze(0)
            rgb = augment_synthetic_rgb(rgb)
            rgb_out[batch_idx] = rgb.to(device=rgb_out.device, dtype=rgb_out.dtype)
            pose_out[batch_idx] = synth_pose[local_idx]
    batch = dict(batch)
    batch["rgb"] = rgb_out
    return batch, pose_out, synthetic_mask


def augment_synthetic_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Lightweight query-only photometric randomization for rendered RGB."""
    out = rgb.float().clamp(0.0, 1.0)
    brightness = 0.85 + 0.30 * torch.rand((), device=out.device)
    contrast = 0.80 + 0.40 * torch.rand((), device=out.device)
    gamma = 0.80 + 0.40 * torch.rand((), device=out.device)
    out = (out - 0.5) * contrast + 0.5
    out = (out * brightness).clamp(0.0, 1.0)
    out = out.clamp(1e-4, 1.0).pow(gamma)
    if torch.rand((), device=out.device) < 0.5:
        noise_std = 0.005 + 0.025 * torch.rand((), device=out.device)
        out = out + torch.randn_like(out) * noise_std
    if torch.rand((), device=out.device) < 0.35:
        _, height, width = out.shape
        erase_h = max(1, int(height * float(0.05 + 0.10 * torch.rand(()))))
        erase_w = max(1, int(width * float(0.05 + 0.10 * torch.rand(()))))
        y0 = int(torch.randint(0, max(height - erase_h, 1), (1,), device=out.device).item())
        x0 = int(torch.randint(0, max(width - erase_w, 1), (1,), device=out.device).item())
        out[:, y0 : y0 + erase_h, x0 : x0 + erase_w] = out.mean(dim=(1, 2), keepdim=True)
    return out.clamp(0.0, 1.0)


def build_candidate_bank(pose_gt: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return build_local_pose_lattice_candidates(
        pose_gt,
        trans_cm=parse_float_csv(args.lattice_trans_cm),
        rot_deg=parse_float_csv(args.lattice_rot_deg),
        include_identity=True,
        max_candidates=int(args.topk),
        limit_strategy=args.limit_strategy,
        combine_trans_rot=bool(args.combine_trans_rot),
        direction_mode=args.lattice_direction_mode,
    )


def candidate_center_pose(pose_gt: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mode = str(args.candidate_center_mode or "gt").lower()
    if mode == "gt":
        return pose_gt
    if mode == "noisy_init":
        noise_mode = str(args.candidate_center_noise_mode or "uniform").lower()
        if noise_mode == "fixed":
            count = pose_gt.shape[0]
            delta = pose_gt.new_zeros((count, 6))
            if float(args.candidate_center_trans_cm) > 0.0:
                delta[:, :3] = _random_unit_vectors(count, device=pose_gt.device, dtype=pose_gt.dtype) * (
                    float(args.candidate_center_trans_cm) / 100.0
                )
            if float(args.candidate_center_rot_deg) > 0.0:
                delta[:, 3:] = _random_unit_vectors(count, device=pose_gt.device, dtype=pose_gt.dtype) * math.radians(
                    float(args.candidate_center_rot_deg)
                )
            return apply_pose_delta(pose_gt.float(), delta.float()).to(dtype=pose_gt.dtype)
        return jitter_gt_poses(
            pose_gt,
            trans_cm=float(args.candidate_center_trans_cm),
            rot_deg=float(args.candidate_center_rot_deg),
        )
    raise ValueError(f"Unknown candidate_center_mode: {args.candidate_center_mode}")


def pose_energy_vector_dim(args: argparse.Namespace) -> int:
    dim = len(FINE_CANDIDATE_SELECTOR_VECTOR_FEATURE_NAMES)
    if bool(args.use_delta_vector):
        dim += len(FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES)
    return dim


def _spearman_rows(scores: torch.Tensor, costs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = []
    for score, cost, row_valid in zip(scores.detach(), costs.detach(), valid.detach()):
        mask = row_valid.bool() & torch.isfinite(cost)
        if int(mask.sum()) < 2:
            continue
        s = score[mask].float()
        c = (-cost[mask]).float()
        if float(s.std()) < 1e-8 or float(c.std()) < 1e-8:
            continue
        sr = torch.argsort(torch.argsort(s)).float()
        cr = torch.argsort(torch.argsort(c)).float()
        sr = (sr - sr.mean()) / sr.std().clamp(min=1e-6)
        cr = (cr - cr.mean()) / cr.std().clamp(min=1e-6)
        values.append((sr * cr).mean())
    if not values:
        return scores.new_tensor(0.0)
    return torch.stack(values).mean().to(device=scores.device)


def _good_bad_auc_rows(
    scores: torch.Tensor,
    costs: torch.Tensor,
    valid: torch.Tensor,
    *,
    good_m: float,
    bad_m: float,
) -> torch.Tensor:
    values = []
    for score, cost, row_valid in zip(scores.detach(), costs.detach(), valid.detach()):
        mask = row_valid.bool() & torch.isfinite(cost)
        good = mask & (cost <= float(good_m))
        bad = mask & (cost >= float(bad_m))
        if int(good.sum()) == 0 or int(bad.sum()) == 0:
            continue
        good_scores = score[good].float().view(-1, 1)
        bad_scores = score[bad].float().view(1, -1)
        wins = (good_scores > bad_scores).float()
        ties = (good_scores == bad_scores).float() * 0.5
        values.append((wins + ties).mean())
    if not values:
        return scores.new_tensor(0.5)
    return torch.stack(values).mean().to(device=scores.device)


def forward_pose_energy_batch(model, energy_net, map_renderer, batch, cfg: Dict, args: argparse.Namespace, *, train: bool):
    device = next(energy_net.parameters()).device
    batch = move_batch_to_device(batch, device)
    pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
    batch, pose_gt, synthetic_mask = maybe_replace_with_synthetic_queries(batch, map_renderer, pose_gt, cfg, args)
    init_pose = candidate_center_pose(pose_gt, args)
    candidate_poses = build_candidate_bank(init_pose, args)
    cand_batch = dict(batch)
    cand_batch = map_renderer.attach_pose_candidate_renders(
        cand_batch,
        candidate_poses,
        prefix="pose_energy_candidate",
        require_grad=False,
        feature="all",
        include_aux=True,
    )
    query_fine_key = args.query_fine_key or cfg.get("map_supervision", {}).get("query_fine_key", "fine")
    with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
        outputs = model(batch["rgb"])
    query_fine = outputs[query_fine_key].float()
    render_fine = cand_batch["pose_energy_candidate_fine"].float()
    projector = getattr(model, "local_corr_projector", None)
    query_proj, render_proj, used_projector = project_query_render_for_fine_selector(
        projector,
        query_fine,
        render_fine,
        require_projector=bool(args.require_projector),
        render_chunk_size=int(cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0),
    )
    feature_pack = fine_candidate_selector_features(
        query_proj,
        render_proj,
        cand_batch["pose_energy_candidate_pose"],
        init_pose=init_pose,
        depth=cand_batch.get("pose_energy_candidate_depth"),
        mask=cand_batch.get("pose_energy_candidate_mask"),
        mode="local",
        radius=int(args.score_radius),
        preprocess=args.score_preprocess,
        highpass_kernel=int(args.score_highpass_kernel),
        score_map_mode=args.score_map_mode,
        return_score_maps=True,
        use_coarse_logits=False,
        use_candidate_delta=True,
        use_delta_vector=bool(args.use_delta_vector),
        use_depth=True,
        use_mask=True,
        use_rgb=False,
    )
    energy_out = energy_net(
        feature_pack["score_maps"],
        feature_pack["features"],
        valid_mask=feature_pack["valid"],
    )
    losses = pose_energy_losses(
        energy_out,
        cand_batch["pose_energy_candidate_pose"],
        pose_gt,
        valid_mask=feature_pack["valid"],
        target_temperature_m=float(args.target_temperature_m),
        rot_cost_weight=float(args.rot_cost_weight),
        hard_ce_weight=float(args.hard_ce_weight),
        residual_weight=float(args.residual_weight),
        improve_weight=float(args.improve_weight),
        improve_margin_m=float(args.improve_margin_m),
    )
    batch_idx = torch.arange(pose_gt.shape[0], device=device)
    pred_idx = losses["pred_index"].long()
    selected_pose = cand_batch["pose_energy_candidate_pose"].float()[batch_idx, pred_idx]
    selected_delta = energy_out["residual_delta"][batch_idx, pred_idx].float()
    residual_pose = apply_pose_delta(selected_pose, selected_delta)
    residual_cost, _residual_targets, residual_trans, residual_rot = pose_costs_and_residual_targets(
        residual_pose[:, None],
        pose_gt.float(),
        valid_mask=torch.ones((pose_gt.shape[0], 1), device=device, dtype=torch.bool),
        rot_cost_weight=float(args.rot_cost_weight),
    )
    residual_cost = residual_cost[:, 0]
    residual_trans = residual_trans[:, 0]
    residual_rot = residual_rot[:, 0]
    spearman = _spearman_rows(energy_out["energy_logits"], losses["pose_cost"], feature_pack["valid"])
    good_bad_auc = _good_bad_auc_rows(
        energy_out["energy_logits"],
        losses["pose_cost"],
        feature_pack["valid"],
        good_m=float(args.auc_good_m),
        bad_m=float(args.auc_bad_m),
    )
    metrics = {
        "loss": losses["loss"],
        "energy_loss": losses["energy_loss"].detach(),
        "hard_ce_loss": losses["hard_ce_loss"].detach(),
        "residual_loss": losses["residual_loss"].detach(),
        "improve_loss": losses["improve_loss"].detach(),
        "top1_acc": losses["top1_acc"].detach(),
        "spearman": spearman.detach(),
        "good_bad_auc": good_bad_auc.detach(),
        "pred_cost_m": losses["pred_cost"].detach().mean(),
        "residual_pred_cost_m": residual_cost.detach().mean(),
        "residual_cost_gain_m": (losses["pred_cost"].detach() - residual_cost.detach()).mean(),
        "residual_trans_err_m": residual_trans.detach().mean(),
        "residual_rot_deg": (residual_rot.detach().mean() * (180.0 / math.pi)),
        "oracle_cost_m": losses["oracle_cost"].detach().mean(),
        "oracle_gap_m": (losses["pred_cost"].detach() - losses["oracle_cost"].detach()).mean(),
        "center_mode_noisy": torch.tensor(float(str(args.candidate_center_mode) == "noisy_init"), device=device),
        "synthetic_fraction": synthetic_mask.float().mean().detach(),
        "used_projector": torch.tensor(float(used_projector), device=device),
    }
    return losses["loss"], metrics


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


@torch.no_grad()
def evaluate(model, energy_net, loader, map_renderer, cfg: Dict, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    energy_net.eval()
    eval_args = argparse.Namespace(**vars(args))
    if args.eval_synthetic_ratio is not None:
        eval_args.synthetic_ratio = float(args.eval_synthetic_ratio)
    rows = []
    for batch_idx, batch in enumerate(loader):
        _loss, metrics = forward_pose_energy_batch(model, energy_net, map_renderer, batch, cfg, eval_args, train=False)
        rows.append({key: float(value.detach().cpu()) for key, value in metrics.items()})
        if args.eval_max_samples is not None and (batch_idx + 1) * int(args.batch_size) >= int(args.eval_max_samples):
            break
    return mean_metrics(rows)


def save_checkpoint(path: Path, energy_net, model, optimizer, step: int, metrics: Dict[str, float], cfg: Dict, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("local_corr_projector.")
    }
    torch.save(
        {
            "step": int(step),
            "pose_energy_state_dict": energy_net.state_dict(),
            "query_projector_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
            "args": vars(args),
        },
        path,
    )


def load_pose_energy_checkpoint(path: str, energy_net, model, optimizer=None) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    energy_state = checkpoint.get("pose_energy_state_dict", checkpoint)
    energy_net.load_state_dict(energy_state, strict=True)
    projector_state = checkpoint.get("query_projector_state_dict") or {}
    if projector_state:
        model.load_state_dict(projector_state, strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    args = apply_config_defaults(args, cfg)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    train_args = argparse.Namespace(**vars(args))
    train_args.split = args.train_split
    train_args.max_samples = args.max_samples
    model, train_loader, map_renderer = build_model_and_data(cfg, train_args, device)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.split = args.eval_split
    eval_args.max_samples = args.eval_max_samples
    _eval_model, eval_loader, _eval_renderer = build_model_and_data(cfg, eval_args, device)
    # Reuse the training model/renderer for eval to avoid loading a second large map.
    del _eval_model, _eval_renderer

    _set_trainable(model, ["local_corr_projector."] if bool(args.train_projector) else [])
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    energy_net = PoseEnergyNet(
        vector_dim=pose_energy_vector_dim(args),
        score_map_channels=3,
        map_channels=int(args.map_channels),
        grid_size=int(args.grid_size),
        hidden_dim=int(args.hidden_dim),
        context_layers=int(args.context_layers),
        context_heads=int(args.context_heads),
    ).to(device)
    trainable.extend(energy_net.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    if args.resume_pose_energy:
        loaded = load_pose_energy_checkpoint(args.resume_pose_energy, energy_net, model, optimizer=None)
        print(
            "loaded pose-energy checkpoint: {path} step={step}".format(
                path=args.resume_pose_energy,
                step=loaded.get("step", "unknown"),
            ),
            flush=True,
        )

    if args.eval_only:
        eval_metrics = evaluate(model, energy_net, eval_loader, map_renderer, cfg, args)
        summary = {
            "checkpoint": args.resume_pose_energy,
            "eval_split": args.eval_split,
            "eval_synthetic_ratio": args.eval_synthetic_ratio,
            "metrics": eval_metrics,
        }
        (out_dir / "eval_only_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    log_path = out_dir / "train_log.jsonl"
    best_metric = float("inf")
    step = 0
    train_iter = iter(train_loader)
    while step < int(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        model.train(bool(args.train_projector))
        energy_net.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
            loss, metrics = forward_pose_energy_batch(model, energy_net, map_renderer, batch, cfg, args, train=True)
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        step += 1

        row = {"step": step, "split": "train"}
        row.update({key: float(value.detach().cpu()) for key, value in metrics.items()})
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 10 == 0:
            print(
                "step={step} loss={loss:.4f} e={energy_loss:.4f} r={residual_loss:.4f} "
                "pred={pred_cost_m:.3f} oracle={oracle_cost_m:.3f} "
                "gap={oracle_gap_m:.3f} sp={spearman:.3f} auc={good_bad_auc:.3f} syn={synthetic_fraction:.2f}".format(
                    **row
                ),
                flush=True,
            )
        if step % int(args.eval_every) == 0 or step == int(args.max_steps):
            eval_metrics = evaluate(model, energy_net, eval_loader, map_renderer, cfg, args)
            eval_row = {"step": step, "split": "eval"}
            eval_row.update(eval_metrics)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(eval_row) + "\n")
            print("eval step={step} ".format(step=step) + json.dumps(eval_metrics, sort_keys=True), flush=True)
            metric = float(eval_metrics.get(str(args.best_metric), float("inf")))
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(out_dir / "checkpoints" / "best.pth", energy_net, model, optimizer, step, eval_metrics, cfg, args)
        if step % int(args.save_every) == 0:
            save_checkpoint(out_dir / "checkpoints" / f"step_{step:06d}.pth", energy_net, model, optimizer, step, row, cfg, args)

    summary = {"best_pred_cost_m": best_metric, "steps": step, "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
