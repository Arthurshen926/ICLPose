#!/usr/bin/env python3
"""Evaluate PoseEnergyNet on noisy-pose CPR buckets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.students.pose_energy_net import (  # noqa: E402
    PoseEnergyNet,
    pose_energy_factorized_selection,
    pose_costs_and_residual_targets,
    pose_energy_selection_scores,
)
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, gather_pose_bank, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.train_pose_energy import (  # noqa: E402
    _good_bad_auc_rows,
    _spearman_rows,
    apply_pose_feature_adapter,
    apply_pose_feature_adapter_with_uncertainty,
    apply_config_defaults,
    build_pose_feature_adapter,
    load_pose_energy_checkpoint,
    parse_float_csv,
    pose_energy_vector_dim,
)
from feature_extract.tools.train_nvs_pose_feature_adapter import adaptive_lattice_spec_for_error  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    build_local_pose_lattice_candidates,
    camera_centers_from_w2c,
    fine_candidate_selector_features,
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    project_query_render_for_fine_selector,
    set_seed,
)
from pose_refine import apply_pose_delta  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pose-energy-checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-fine-key", default=None)
    parser.add_argument("--require-projector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--lattice-trans-cm", default=None)
    parser.add_argument("--lattice-rot-deg", default=None)
    parser.add_argument("--lattice-direction-mode", choices=("axis", "cube"), default=None)
    parser.add_argument("--combine-trans-rot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit-strategy", default=None)
    parser.add_argument(
        "--candidate-bank-mode",
        choices=("lattice", "balanced", "adaptive", "adaptive_balanced", "bucket_adaptive", "adaptive_direction_balanced"),
        default=None,
    )
    parser.add_argument("--score-radius", type=int, default=None)
    parser.add_argument("--score-preprocess", default=None)
    parser.add_argument("--score-highpass-kernel", type=int, default=None)
    parser.add_argument("--score-map-mode", default=None)
    parser.add_argument("--use-delta-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-center-delta-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-uncertainty", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--map-channels", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--context-layers", type=int, default=None)
    parser.add_argument("--context-heads", type=int, default=None)
    parser.add_argument("--factorized-heads", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rot-cost-weight", type=float, default=None)
    parser.add_argument("--auc-good-m", type=float, default=None)
    parser.add_argument("--auc-bad-m", type=float, default=None)
    parser.add_argument("--buckets", default="10:2,25:5,50:10,100:20")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--selection-mode", choices=("joint", "factorized"), default="joint")
    parser.add_argument("--selection-confidence-weight", type=float, default=None)
    parser.add_argument("--selection-residual-norm-weight", type=float, default=None)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=5.0)
    return parser.parse_args()


def parse_buckets(value: str) -> List[Tuple[str, float, float]]:
    buckets = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        trans, rot = item.split(":")
        trans_cm = float(trans)
        rot_deg = float(rot)
        buckets.append((f"{int(trans_cm):03d}cm_{rot_deg:g}deg", trans_cm, rot_deg))
    return buckets


def random_unit_vectors(count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vec = torch.randn(count, 3, device=device, dtype=dtype)
    return vec / vec.norm(dim=1, keepdim=True).clamp(min=1.0e-6)


def fixed_bucket_init_pose(pose_gt: torch.Tensor, trans_cm: float, rot_deg: float) -> torch.Tensor:
    count = pose_gt.shape[0]
    delta = pose_gt.new_zeros((count, 6))
    if float(trans_cm) > 0.0:
        delta[:, :3] = random_unit_vectors(count, device=pose_gt.device, dtype=pose_gt.dtype) * (float(trans_cm) / 100.0)
    if float(rot_deg) > 0.0:
        delta[:, 3:] = random_unit_vectors(count, device=pose_gt.device, dtype=pose_gt.dtype) * math.radians(float(rot_deg))
    return apply_pose_delta(pose_gt.float(), delta.float()).to(dtype=pose_gt.dtype)


def build_candidate_bank(
    init_pose: torch.Tensor,
    args: argparse.Namespace,
    *,
    trans_cm: float | None = None,
    rot_deg: float | None = None,
) -> torch.Tensor:
    combine_trans_rot = bool(args.combine_trans_rot)
    bank_mode = str(getattr(args, "candidate_bank_mode", "lattice") or "lattice").lower()
    lattice_trans = parse_float_csv(args.lattice_trans_cm)
    lattice_rot = parse_float_csv(args.lattice_rot_deg)
    if (
        bank_mode in {"adaptive", "adaptive_balanced", "bucket_adaptive", "adaptive_direction_balanced"}
        and trans_cm is not None
        and rot_deg is not None
    ):
        lattice_trans, lattice_rot = adaptive_lattice_spec_for_error(
            float(trans_cm) / 100.0,
            math.radians(float(rot_deg)),
        )
        combine_trans_rot = True
    elif bank_mode == "balanced":
        combine_trans_rot = True
    return build_local_pose_lattice_candidates(
        init_pose,
        trans_cm=lattice_trans,
        rot_deg=lattice_rot,
        include_identity=True,
        max_candidates=int(args.topk),
        limit_strategy=args.limit_strategy,
        combine_trans_rot=combine_trans_rot,
        direction_mode=args.lattice_direction_mode,
    )


def stats(values: Iterable[float], *, scale: float = 1.0) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32) * float(scale)
    if tensor.numel() == 0:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": float(tensor.mean().item()), "median": float(tensor.median().item())}


def safe_cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    cosine = (a * b).sum(dim=-1) / denom.clamp(min=1.0e-8)
    return torch.where(denom > 1.0e-8, cosine, torch.zeros_like(cosine))


@torch.no_grad()
def evaluate_bucket(
    model,
    energy_net,
    loader,
    map_renderer,
    cfg: Dict,
    args: argparse.Namespace,
    trans_cm: float,
    rot_deg: float,
    *,
    pose_feature_adapter=None,
):
    device = next(energy_net.parameters()).device
    rows: Dict[str, List[float]] = {
        "init_trans_m": [],
        "init_rot_deg": [],
        "selected_trans_m": [],
        "selected_rot_deg": [],
        "residual_trans_m": [],
        "residual_rot_deg": [],
        "oracle_trans_m": [],
        "oracle_rot_deg": [],
        "selected_gain": [],
        "residual_gain": [],
        "selected_basin": [],
        "residual_basin": [],
        "oracle_basin": [],
        "spearman": [],
        "good_bad_auc": [],
        "pred_cost_m": [],
        "oracle_cost_m": [],
        "selected_delta_trans_m": [],
        "selected_delta_rot_deg": [],
        "oracle_delta_trans_m": [],
        "oracle_delta_rot_deg": [],
        "selected_delta_over_init": [],
        "oracle_delta_over_init": [],
        "selected_correction_cos": [],
        "oracle_correction_cos": [],
        "selected_identity": [],
        "selected_near_center": [],
        "oracle_identity": [],
        "oracle_near_center": [],
        "selected_trans_improve_m": [],
        "residual_trans_improve_m": [],
        "oracle_trans_improve_m": [],
    }
    model.eval()
    energy_net.eval()
    if pose_feature_adapter is not None:
        pose_feature_adapter.eval()
    processed = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
        init_pose = fixed_bucket_init_pose(pose_gt, trans_cm, rot_deg)
        candidate_poses = build_candidate_bank(init_pose, args, trans_cm=trans_cm, rot_deg=rot_deg)
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
            if bool(cfg.get("model", {}).get("teacher_fine_condition", False)):
                outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
            else:
                outputs = model(batch["rgb"])
        query_fine = outputs[query_fine_key].float()
        render_fine = cand_batch["pose_energy_candidate_fine"].float()
        projector = getattr(model, "local_corr_projector", None)
        query_uncertainty = None
        render_uncertainty = None
        if pose_feature_adapter is not None:
            query_proj = query_fine
            render_proj = render_fine
            if bool(args.pose_feature_adapter_after_projector):
                query_proj, render_proj, _used_projector = project_query_render_for_fine_selector(
                    projector,
                    query_fine,
                    render_fine,
                    require_projector=bool(args.require_projector),
                    render_chunk_size=int(
                        cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0
                    ),
                )
            render_chunk_size = int(
                cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0
            )
            if bool(args.use_uncertainty):
                (
                    query_proj,
                    render_proj,
                    query_uncertainty,
                    render_uncertainty,
                    _used_pose_feature_adapter,
                ) = apply_pose_feature_adapter_with_uncertainty(
                    pose_feature_adapter,
                    query_proj,
                    render_proj,
                    render_chunk_size=render_chunk_size,
                )
            else:
                query_proj, render_proj, _used_pose_feature_adapter = apply_pose_feature_adapter(
                    pose_feature_adapter,
                    query_proj,
                    render_proj,
                    render_chunk_size=render_chunk_size,
                )
        else:
            if bool(args.use_uncertainty):
                raise ValueError("use_uncertainty requires pose_feature_adapter_enabled=true")
            query_proj, render_proj, _used_projector = project_query_render_for_fine_selector(
                projector,
                query_fine,
                render_fine,
                require_projector=bool(args.require_projector),
                render_chunk_size=int(
                    cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0
                ),
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
            use_center_delta_vector=bool(args.use_center_delta_vector),
            use_depth=True,
            use_mask=True,
            use_rgb=False,
            query_uncertainty=query_uncertainty,
            candidate_uncertainty=render_uncertainty,
            use_uncertainty=bool(args.use_uncertainty),
        )
        energy_out = energy_net(
            feature_pack["score_maps"],
            feature_pack["features"],
            valid_mask=feature_pack["valid"],
        )
        costs, _residual_targets, trans_err, rot_err = pose_costs_and_residual_targets(
            cand_batch["pose_energy_candidate_pose"].float(),
            pose_gt.float(),
            valid_mask=feature_pack["valid"],
            rot_cost_weight=float(args.rot_cost_weight),
        )
        logits = pose_energy_selection_scores(
            energy_out,
            confidence_weight=float(args.selection_confidence_weight or 0.0),
            residual_norm_weight=float(args.selection_residual_norm_weight or 0.0),
            residual_trans_scale_m=float(getattr(args, "residual_trans_scale_m", 0.25)),
            residual_rot_scale_rad=math.radians(float(getattr(args, "residual_rot_scale_deg", 5.0))),
        )
        selection_mode = str(args.selection_mode or "joint").lower()
        best_idx = costs.argmin(dim=1)
        batch_idx = torch.arange(pose_gt.shape[0], device=device)
        if selection_mode == "factorized":
            selected_pose, trans_idx, rot_idx = pose_energy_factorized_selection(
                energy_out,
                cand_batch["pose_energy_candidate_pose"].float(),
                valid_mask=feature_pack["valid"],
            )
            pred_idx = trans_idx
            residual_pose = selected_pose.float()
            selected_cost, _selected_residual, _selected_trans_err, _selected_rot_err = pose_costs_and_residual_targets(
                selected_pose.float()[:, None],
                pose_gt.float(),
                rot_cost_weight=float(args.rot_cost_weight),
            )
            selected_cost = selected_cost[:, 0]
        else:
            pred_idx = logits.argmax(dim=1)
            selected_pose = gather_pose_bank(cand_batch["pose_energy_candidate_pose"].float(), pred_idx[:, None])[:, 0]
            residual_delta = energy_out["residual_delta"][batch_idx, pred_idx].float() * float(args.residual_scale)
            residual_pose = apply_pose_delta(selected_pose.float(), residual_delta)
            selected_cost = costs[batch_idx, pred_idx]
        _init_loss, init_rot, init_trans = pose_error_tensors(init_pose.float(), pose_gt.float())
        _sel_loss, sel_rot, sel_trans = pose_error_tensors(selected_pose.float(), pose_gt.float())
        _res_loss, res_rot, res_trans = pose_error_tensors(residual_pose.float(), pose_gt.float())
        oracle_trans = trans_err[batch_idx, best_idx]
        oracle_rot_deg = rot_err[batch_idx, best_idx] * (180.0 / math.pi)
        init_centers = camera_centers_from_w2c(init_pose.float())
        gt_centers = camera_centers_from_w2c(pose_gt.float())
        cand_centers = camera_centers_from_w2c(
            cand_batch["pose_energy_candidate_pose"].float().reshape(-1, 4, 4)
        ).reshape(pose_gt.shape[0], -1, 3)
        correction_target = gt_centers - init_centers
        selected_centers = camera_centers_from_w2c(selected_pose.float())
        selected_correction = selected_centers - init_centers
        oracle_correction = cand_centers[batch_idx, best_idx] - init_centers
        selected_delta_trans = torch.linalg.norm(selected_correction, dim=1)
        oracle_delta_trans = torch.linalg.norm(oracle_correction, dim=1)
        flat_init = init_pose[:, None].expand_as(cand_batch["pose_energy_candidate_pose"]).reshape(-1, 4, 4)
        _delta_rot_loss, delta_rot_deg_flat, delta_trans_flat = pose_error_tensors(
            cand_batch["pose_energy_candidate_pose"].float().reshape(-1, 4, 4),
            flat_init.float(),
        )
        delta_rot_deg = delta_rot_deg_flat.reshape(pose_gt.shape[0], -1)
        delta_trans_m = delta_trans_flat.reshape(pose_gt.shape[0], -1)
        identity_idx = delta_trans_m.argmin(dim=1)
        _selected_delta_loss, selected_delta_rot, _selected_delta_trans_pose = pose_error_tensors(
            selected_pose.float(),
            init_pose.float(),
        )
        oracle_delta_rot = delta_rot_deg[batch_idx, best_idx]
        init_trans_safe = init_trans.clamp(min=1.0e-6)
        selected_correction_cos = safe_cosine_rows(selected_correction, correction_target)
        oracle_correction_cos = safe_cosine_rows(oracle_correction, correction_target)
        basin_trans = float(args.basin_trans_m)
        basin_rot = float(args.basin_rot_deg)
        rows["init_trans_m"].extend(init_trans.cpu().tolist())
        rows["init_rot_deg"].extend(init_rot.cpu().tolist())
        rows["selected_trans_m"].extend(sel_trans.cpu().tolist())
        rows["selected_rot_deg"].extend(sel_rot.cpu().tolist())
        rows["residual_trans_m"].extend(res_trans.cpu().tolist())
        rows["residual_rot_deg"].extend(res_rot.cpu().tolist())
        rows["oracle_trans_m"].extend(oracle_trans.cpu().tolist())
        rows["oracle_rot_deg"].extend(oracle_rot_deg.cpu().tolist())
        rows["selected_gain"].extend((sel_trans < init_trans).float().cpu().tolist())
        rows["residual_gain"].extend((res_trans < init_trans).float().cpu().tolist())
        rows["selected_basin"].extend(((sel_trans <= basin_trans) & (sel_rot <= basin_rot)).float().cpu().tolist())
        rows["residual_basin"].extend(((res_trans <= basin_trans) & (res_rot <= basin_rot)).float().cpu().tolist())
        rows["oracle_basin"].extend(
            ((oracle_trans <= basin_trans) & (oracle_rot_deg <= basin_rot)).float().cpu().tolist()
        )
        rows["spearman"].append(float(_spearman_rows(logits, costs, feature_pack["valid"]).cpu()))
        rows["good_bad_auc"].append(
            float(
                _good_bad_auc_rows(
                    logits,
                    costs,
                    feature_pack["valid"],
                    good_m=float(args.auc_good_m),
                    bad_m=float(args.auc_bad_m),
                ).cpu()
            )
        )
        rows["pred_cost_m"].extend(selected_cost.cpu().tolist())
        rows["oracle_cost_m"].extend(costs[batch_idx, best_idx].cpu().tolist())
        rows["selected_delta_trans_m"].extend(selected_delta_trans.cpu().tolist())
        rows["selected_delta_rot_deg"].extend(selected_delta_rot.cpu().tolist())
        rows["oracle_delta_trans_m"].extend(oracle_delta_trans.cpu().tolist())
        rows["oracle_delta_rot_deg"].extend(oracle_delta_rot.cpu().tolist())
        rows["selected_delta_over_init"].extend((selected_delta_trans / init_trans_safe).cpu().tolist())
        rows["oracle_delta_over_init"].extend((oracle_delta_trans / init_trans_safe).cpu().tolist())
        rows["selected_correction_cos"].extend(selected_correction_cos.cpu().tolist())
        rows["oracle_correction_cos"].extend(oracle_correction_cos.cpu().tolist())
        rows["selected_identity"].extend(
            ((selected_delta_trans < 0.01) & (selected_delta_rot < 0.01)).float().cpu().tolist()
        )
        rows["selected_near_center"].extend((selected_delta_trans < 0.01).float().cpu().tolist())
        rows["oracle_identity"].extend((best_idx == identity_idx).float().cpu().tolist())
        rows["oracle_near_center"].extend((oracle_delta_trans < 0.01).float().cpu().tolist())
        rows["selected_trans_improve_m"].extend((init_trans - sel_trans).cpu().tolist())
        rows["residual_trans_improve_m"].extend((init_trans - res_trans).cpu().tolist())
        rows["oracle_trans_improve_m"].extend((init_trans - oracle_trans).cpu().tolist())
        processed += int(pose_gt.shape[0])
        if args.max_samples is not None and processed >= int(args.max_samples):
            break
    summary = {
        "samples": int(min(processed, int(args.max_samples or processed))),
        "init_trans_mm": stats(rows["init_trans_m"], scale=1000.0),
        "init_rot_deg": stats(rows["init_rot_deg"]),
        "selected_trans_mm": stats(rows["selected_trans_m"], scale=1000.0),
        "selected_rot_deg": stats(rows["selected_rot_deg"]),
        "residual_trans_mm": stats(rows["residual_trans_m"], scale=1000.0),
        "residual_rot_deg": stats(rows["residual_rot_deg"]),
        "oracle_trans_mm": stats(rows["oracle_trans_m"], scale=1000.0),
        "oracle_rot_deg": stats(rows["oracle_rot_deg"]),
        "selected_gain_frac": float(torch.tensor(rows["selected_gain"]).mean().item()) if rows["selected_gain"] else 0.0,
        "residual_gain_frac": float(torch.tensor(rows["residual_gain"]).mean().item()) if rows["residual_gain"] else 0.0,
        "selected_basin_recall": float(torch.tensor(rows["selected_basin"]).mean().item()) if rows["selected_basin"] else 0.0,
        "residual_basin_recall": float(torch.tensor(rows["residual_basin"]).mean().item()) if rows["residual_basin"] else 0.0,
        "oracle_basin_recall": float(torch.tensor(rows["oracle_basin"]).mean().item()) if rows["oracle_basin"] else 0.0,
        "spearman": float(torch.tensor(rows["spearman"]).mean().item()) if rows["spearman"] else 0.0,
        "good_bad_auc": float(torch.tensor(rows["good_bad_auc"]).mean().item()) if rows["good_bad_auc"] else 0.5,
        "pred_cost_m": stats(rows["pred_cost_m"]),
        "oracle_cost_m": stats(rows["oracle_cost_m"]),
        "selected_delta_trans_mm": stats(rows["selected_delta_trans_m"], scale=1000.0),
        "selected_delta_rot_deg": stats(rows["selected_delta_rot_deg"]),
        "oracle_delta_trans_mm": stats(rows["oracle_delta_trans_m"], scale=1000.0),
        "oracle_delta_rot_deg": stats(rows["oracle_delta_rot_deg"]),
        "selected_delta_over_init": stats(rows["selected_delta_over_init"]),
        "oracle_delta_over_init": stats(rows["oracle_delta_over_init"]),
        "selected_correction_cos": stats(rows["selected_correction_cos"]),
        "oracle_correction_cos": stats(rows["oracle_correction_cos"]),
        "selected_identity_frac": float(torch.tensor(rows["selected_identity"]).mean().item())
        if rows["selected_identity"]
        else 0.0,
        "selected_near_center_frac": float(torch.tensor(rows["selected_near_center"]).mean().item())
        if rows["selected_near_center"]
        else 0.0,
        "oracle_identity_frac": float(torch.tensor(rows["oracle_identity"]).mean().item()) if rows["oracle_identity"] else 0.0,
        "oracle_near_center_frac": float(torch.tensor(rows["oracle_near_center"]).mean().item())
        if rows["oracle_near_center"]
        else 0.0,
        "selected_trans_improve_mm": stats(rows["selected_trans_improve_m"], scale=1000.0),
        "residual_trans_improve_mm": stats(rows["residual_trans_improve_m"], scale=1000.0),
        "oracle_trans_improve_mm": stats(rows["oracle_trans_improve_m"], scale=1000.0),
    }
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    args = apply_config_defaults(args, cfg)
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    data_args = argparse.Namespace(**vars(args))
    data_args.max_samples = args.max_samples
    model, loader, map_renderer = build_model_and_data(cfg, data_args, device)
    for param in model.parameters():
        param.requires_grad_(False)
    energy_net = PoseEnergyNet(
        vector_dim=pose_energy_vector_dim(args),
        score_map_channels=3,
        map_channels=int(args.map_channels),
        grid_size=int(args.grid_size),
        hidden_dim=int(args.hidden_dim),
        context_layers=int(args.context_layers),
        context_heads=int(args.context_heads),
        factorized_heads=bool(args.factorized_heads),
    ).to(device)
    pose_feature_adapter = build_pose_feature_adapter(args, model, cfg, device)
    load_pose_energy_checkpoint(
        args.pose_energy_checkpoint,
        energy_net,
        model,
        optimizer=None,
        pose_feature_adapter=pose_feature_adapter,
    )
    results = {}
    for name, trans_cm, rot_deg in parse_buckets(args.buckets):
        print(f"evaluating bucket {name}", flush=True)
        results[name] = evaluate_bucket(
            model,
            energy_net,
            loader,
            map_renderer,
            cfg,
            args,
            trans_cm,
            rot_deg,
            pose_feature_adapter=pose_feature_adapter,
        )
        print(json.dumps({name: results[name]}, sort_keys=True), flush=True)
    summary = {
        "checkpoint": args.pose_energy_checkpoint,
        "split": args.split,
        "topk": int(args.topk),
        "lattice_trans_cm": parse_float_csv(args.lattice_trans_cm),
        "lattice_rot_deg": parse_float_csv(args.lattice_rot_deg),
        "buckets": results,
    }
    (out_dir / "bucket_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
