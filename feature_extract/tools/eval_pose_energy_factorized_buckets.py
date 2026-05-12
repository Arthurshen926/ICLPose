#!/usr/bin/env python3
"""Evaluate a two-stage factorized PoseEnergy CPR pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.students.pose_energy_net import PoseEnergyNet, pose_costs_and_residual_targets  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, gather_pose_bank, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.eval_pose_energy_buckets import (  # noqa: E402
    fixed_bucket_init_pose,
    parse_buckets,
    safe_cosine_rows,
    stats,
)
from feature_extract.tools.train_pose_energy import (  # noqa: E402
    _good_bad_auc_rows,
    _spearman_rows,
    apply_config_defaults,
    parse_float_csv,
    pose_energy_vector_dim,
)
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
    parser.add_argument("--pose-energy-checkpoint", default=None)
    parser.add_argument("--rot-pose-energy-checkpoint", default=None)
    parser.add_argument("--trans-pose-energy-checkpoint", default=None)
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
    parser.add_argument("--require-projector", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--rot-cost-weight", type=float, default=None)
    parser.add_argument("--auc-good-m", type=float, default=None)
    parser.add_argument("--auc-bad-m", type=float, default=None)
    parser.add_argument("--buckets", default="10:2,25:5,50:10,100:20")
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=5.0)
    parser.add_argument("--stage-order", choices=("rot_trans", "trans_rot"), default="rot_trans")
    parser.add_argument("--stage-transition", choices=("selected", "residual"), default="selected")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--rot-stage-topk", type=int, default=64)
    parser.add_argument("--rot-stage-lattice-rot-deg", default="0,0.5,1,2,5,10,20")
    parser.add_argument("--rot-stage-direction-mode", choices=("axis", "cube"), default="cube")
    parser.add_argument("--rot-stage-limit-strategy", default="uniform")
    parser.add_argument("--trans-stage-topk", type=int, default=64)
    parser.add_argument("--trans-stage-lattice-trans-cm", default="0,2,5,10,25,50,100")
    parser.add_argument("--trans-stage-direction-mode", choices=("axis", "cube"), default="cube")
    parser.add_argument("--trans-stage-limit-strategy", default="uniform")
    return parser.parse_args()


def load_stage_checkpoint(path: str, energy_net: PoseEnergyNet) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    energy_state = checkpoint.get("pose_energy_state_dict", checkpoint)
    energy_net.load_state_dict(energy_state, strict=True)
    return checkpoint.get("query_projector_state_dict") or {}


def make_energy_net(args: argparse.Namespace, device: torch.device) -> PoseEnergyNet:
    return PoseEnergyNet(
        vector_dim=pose_energy_vector_dim(args),
        score_map_channels=3,
        map_channels=int(args.map_channels),
        grid_size=int(args.grid_size),
        hidden_dim=int(args.hidden_dim),
        context_layers=int(args.context_layers),
        context_heads=int(args.context_heads),
    ).to(device)


def build_stage_candidates(base_pose: torch.Tensor, stage: str, args: argparse.Namespace) -> torch.Tensor:
    if stage == "rot":
        return build_local_pose_lattice_candidates(
            base_pose,
            trans_cm=[0.0],
            rot_deg=parse_float_csv(args.rot_stage_lattice_rot_deg),
            include_identity=True,
            max_candidates=int(args.rot_stage_topk),
            limit_strategy=args.rot_stage_limit_strategy,
            combine_trans_rot=False,
            direction_mode=args.rot_stage_direction_mode,
        )
    if stage == "trans":
        return build_local_pose_lattice_candidates(
            base_pose,
            trans_cm=parse_float_csv(args.trans_stage_lattice_trans_cm),
            rot_deg=[0.0],
            include_identity=True,
            max_candidates=int(args.trans_stage_topk),
            limit_strategy=args.trans_stage_limit_strategy,
            combine_trans_rot=False,
            direction_mode=args.trans_stage_direction_mode,
        )
    raise ValueError(f"unknown stage: {stage}")


def append_metric(rows: Dict[str, List[float]], key: str, tensor: torch.Tensor) -> None:
    rows.setdefault(key, []).extend(tensor.detach().float().cpu().tolist())


@torch.no_grad()
def run_stage(
    *,
    stage: str,
    model,
    energy_net: PoseEnergyNet,
    projector_state: Dict,
    raw_query_fine: torch.Tensor,
    batch: Dict,
    map_renderer,
    pose_gt: torch.Tensor,
    base_pose: torch.Tensor,
    cfg: Dict,
    args: argparse.Namespace,
) -> Dict:
    if projector_state:
        model.load_state_dict(projector_state, strict=False)
    candidate_poses = build_stage_candidates(base_pose, stage, args)
    cand_batch = dict(batch)
    prefix = f"factor_{stage}"
    cand_batch = map_renderer.attach_pose_candidate_renders(
        cand_batch,
        candidate_poses,
        prefix=prefix,
        require_grad=False,
        feature="all",
        include_aux=True,
    )
    render_fine = cand_batch[f"{prefix}_fine"].float()
    projector = getattr(model, "local_corr_projector", None)
    query_proj, render_proj, _used_projector = project_query_render_for_fine_selector(
        projector,
        raw_query_fine.float(),
        render_fine,
        require_projector=bool(args.require_projector),
        render_chunk_size=int(cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0),
    )
    feature_pack = fine_candidate_selector_features(
        query_proj,
        render_proj,
        cand_batch[f"{prefix}_pose"],
        init_pose=base_pose,
        depth=cand_batch.get(f"{prefix}_depth"),
        mask=cand_batch.get(f"{prefix}_mask"),
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
    costs, _residual_targets, trans_err, rot_err = pose_costs_and_residual_targets(
        cand_batch[f"{prefix}_pose"].float(),
        pose_gt.float(),
        valid_mask=feature_pack["valid"],
        rot_cost_weight=float(args.rot_cost_weight),
    )
    logits = energy_out["energy_logits"]
    pred_idx = logits.argmax(dim=1)
    best_idx = costs.argmin(dim=1)
    batch_idx = torch.arange(pose_gt.shape[0], device=pose_gt.device)
    selected_pose = gather_pose_bank(cand_batch[f"{prefix}_pose"].float(), pred_idx[:, None])[:, 0]
    oracle_pose = gather_pose_bank(cand_batch[f"{prefix}_pose"].float(), best_idx[:, None])[:, 0]
    residual_delta = energy_out["residual_delta"][batch_idx, pred_idx].float() * float(args.residual_scale)
    residual_pose = apply_pose_delta(selected_pose.float(), residual_delta)
    base_centers = camera_centers_from_w2c(base_pose.float())
    gt_centers = camera_centers_from_w2c(pose_gt.float())
    candidate_centers = camera_centers_from_w2c(
        cand_batch[f"{prefix}_pose"].float().reshape(-1, 4, 4)
    ).reshape(pose_gt.shape[0], -1, 3)
    selected_correction = candidate_centers[batch_idx, pred_idx] - base_centers
    oracle_correction = candidate_centers[batch_idx, best_idx] - base_centers
    needed_correction = gt_centers - base_centers
    flat_base = base_pose[:, None].expand_as(cand_batch[f"{prefix}_pose"]).reshape(-1, 4, 4)
    _delta_loss, delta_rot_deg_flat, delta_trans_flat = pose_error_tensors(
        cand_batch[f"{prefix}_pose"].float().reshape(-1, 4, 4),
        flat_base.float(),
    )
    delta_rot_deg = delta_rot_deg_flat.reshape(pose_gt.shape[0], -1)
    delta_trans_m = delta_trans_flat.reshape(pose_gt.shape[0], -1)
    identity_idx = delta_trans_m.add(delta_rot_deg * 0.01).argmin(dim=1)
    return {
        "stage": stage,
        "selected_pose": selected_pose,
        "oracle_pose": oracle_pose,
        "residual_pose": residual_pose,
        "pred_idx": pred_idx,
        "best_idx": best_idx,
        "costs": costs,
        "logits": logits,
        "valid": feature_pack["valid"],
        "selected_delta_trans_m": torch.linalg.norm(selected_correction, dim=1),
        "oracle_delta_trans_m": torch.linalg.norm(oracle_correction, dim=1),
        "selected_delta_rot_deg": delta_rot_deg[batch_idx, pred_idx],
        "oracle_delta_rot_deg": delta_rot_deg[batch_idx, best_idx],
        "selected_correction_cos": safe_cosine_rows(selected_correction, needed_correction),
        "oracle_correction_cos": safe_cosine_rows(oracle_correction, needed_correction),
        "selected_identity": (pred_idx == identity_idx).float(),
        "oracle_identity": (best_idx == identity_idx).float(),
        "spearman": _spearman_rows(logits, costs, feature_pack["valid"]),
        "good_bad_auc": _good_bad_auc_rows(
            logits,
            costs,
            feature_pack["valid"],
            good_m=float(args.auc_good_m),
            bad_m=float(args.auc_bad_m),
        ),
        "pred_cost_m": costs[batch_idx, pred_idx],
        "oracle_cost_m": costs[batch_idx, best_idx],
        "used_projector": float(_used_projector),
    }


def summarize_pose(rows: Dict[str, List[float]], prefix: str, pose: torch.Tensor, pose_gt: torch.Tensor) -> None:
    _loss, rot_deg, trans_m = pose_error_tensors(pose.float(), pose_gt.float())
    append_metric(rows, f"{prefix}_trans_m", trans_m)
    append_metric(rows, f"{prefix}_rot_deg", rot_deg)


@torch.no_grad()
def evaluate_bucket(model, rot_energy, trans_energy, loader, map_renderer, cfg, args, trans_cm: float, rot_deg: float):
    device = next(rot_energy.parameters()).device
    rows: Dict[str, List[float]] = {}
    order = args.stage_order.split("_")
    projector_states = {
        "rot": getattr(args, "_rot_projector_state", {}),
        "trans": getattr(args, "_trans_projector_state", {}),
    }
    energies = {"rot": rot_energy, "trans": trans_energy}
    model.eval()
    rot_energy.eval()
    trans_energy.eval()
    processed = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
        init_pose = fixed_bucket_init_pose(pose_gt, trans_cm, rot_deg)
        query_fine_key = args.query_fine_key or cfg.get("map_supervision", {}).get("query_fine_key", "fine")
        with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
            outputs = model(batch["rgb"])
        raw_query_fine = outputs[query_fine_key].float()
        summarize_pose(rows, "init", init_pose, pose_gt)
        base_pose = init_pose
        first_result = None
        second_result = None
        for idx, stage in enumerate(order):
            result = run_stage(
                stage=stage,
                model=model,
                energy_net=energies[stage],
                projector_state=projector_states[stage],
                raw_query_fine=raw_query_fine,
                batch=batch,
                map_renderer=map_renderer,
                pose_gt=pose_gt,
                base_pose=base_pose,
                cfg=cfg,
                args=args,
            )
            stage_name = f"stage{idx + 1}_{stage}"
            summarize_pose(rows, f"{stage_name}_selected", result["selected_pose"], pose_gt)
            summarize_pose(rows, f"{stage_name}_residual", result["residual_pose"], pose_gt)
            summarize_pose(rows, f"{stage_name}_oracle", result["oracle_pose"], pose_gt)
            for key in (
                "selected_delta_trans_m",
                "oracle_delta_trans_m",
                "selected_delta_rot_deg",
                "oracle_delta_rot_deg",
                "selected_correction_cos",
                "oracle_correction_cos",
                "selected_identity",
                "oracle_identity",
                "pred_cost_m",
                "oracle_cost_m",
            ):
                append_metric(rows, f"{stage_name}_{key}", result[key])
            rows.setdefault(f"{stage_name}_spearman", []).append(float(result["spearman"].cpu()))
            rows.setdefault(f"{stage_name}_good_bad_auc", []).append(float(result["good_bad_auc"].cpu()))
            if idx == 0:
                first_result = result
            else:
                second_result = result
            base_pose = result["residual_pose"] if args.stage_transition == "residual" else result["selected_pose"]
        final_selected = second_result["selected_pose"] if second_result is not None else first_result["selected_pose"]
        final_residual = second_result["residual_pose"] if second_result is not None else first_result["residual_pose"]
        final_oracle = second_result["oracle_pose"] if second_result is not None else first_result["oracle_pose"]
        summarize_pose(rows, "final_selected", final_selected, pose_gt)
        summarize_pose(rows, "final_residual", final_residual, pose_gt)
        summarize_pose(rows, "final_oracle", final_oracle, pose_gt)
        _init_loss, _init_rot, init_trans = pose_error_tensors(init_pose.float(), pose_gt.float())
        _sel_loss, sel_rot, sel_trans = pose_error_tensors(final_selected.float(), pose_gt.float())
        _res_loss, res_rot, res_trans = pose_error_tensors(final_residual.float(), pose_gt.float())
        basin_trans = float(args.basin_trans_m)
        basin_rot = float(args.basin_rot_deg)
        append_metric(rows, "final_selected_gain", (sel_trans < init_trans).float())
        append_metric(rows, "final_residual_gain", (res_trans < init_trans).float())
        append_metric(rows, "final_selected_basin", ((sel_trans <= basin_trans) & (sel_rot <= basin_rot)).float())
        append_metric(rows, "final_residual_basin", ((res_trans <= basin_trans) & (res_rot <= basin_rot)).float())
        append_metric(rows, "final_selected_trans_improve_m", init_trans - sel_trans)
        append_metric(rows, "final_residual_trans_improve_m", init_trans - res_trans)
        processed += int(pose_gt.shape[0])
        if args.max_samples is not None and processed >= int(args.max_samples):
            break
    summary = {
        "samples": int(min(processed, int(args.max_samples or processed))),
        "stage_order": args.stage_order,
        "stage_transition": args.stage_transition,
    }
    for key, values in sorted(rows.items()):
        scale = 1000.0 if key.endswith("_trans_m") or key.endswith("_improve_m") or key.endswith("_delta_trans_m") else 1.0
        out_key = key
        if scale == 1000.0:
            out_key = key[:-2] + "mm" if key.endswith("_m") else key
        if key.endswith("_gain") or key.endswith("_basin") or key.endswith("_identity"):
            summary[out_key + "_frac"] = float(torch.tensor(values, dtype=torch.float32).mean().item()) if values else 0.0
        elif key.endswith("_spearman") or key.endswith("_good_bad_auc"):
            summary[out_key] = float(torch.tensor(values, dtype=torch.float32).mean().item()) if values else 0.0
        else:
            summary[out_key] = stats(values, scale=scale)
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
    shared_checkpoint = args.pose_energy_checkpoint
    rot_checkpoint = args.rot_pose_energy_checkpoint or shared_checkpoint
    trans_checkpoint = args.trans_pose_energy_checkpoint or shared_checkpoint
    if not rot_checkpoint or not trans_checkpoint:
        raise ValueError("provide --pose-energy-checkpoint or both stage-specific checkpoints")
    rot_energy = make_energy_net(args, device)
    trans_energy = make_energy_net(args, device)
    args._rot_projector_state = load_stage_checkpoint(rot_checkpoint, rot_energy)
    args._trans_projector_state = load_stage_checkpoint(trans_checkpoint, trans_energy)
    results = {}
    for name, bucket_trans_cm, bucket_rot_deg in parse_buckets(args.buckets):
        print(f"evaluating factorized bucket {name}", flush=True)
        results[name] = evaluate_bucket(
            model,
            rot_energy,
            trans_energy,
            loader,
            map_renderer,
            cfg,
            args,
            bucket_trans_cm,
            bucket_rot_deg,
        )
        print(json.dumps({name: results[name]}, sort_keys=True), flush=True)
    summary = {
        "rot_checkpoint": rot_checkpoint,
        "trans_checkpoint": trans_checkpoint,
        "split": args.split,
        "stage_order": args.stage_order,
        "stage_transition": args.stage_transition,
        "rot_stage_lattice_rot_deg": parse_float_csv(args.rot_stage_lattice_rot_deg),
        "trans_stage_lattice_trans_cm": parse_float_csv(args.trans_stage_lattice_trans_cm),
        "buckets": results,
    }
    (out_dir / "bucket_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
