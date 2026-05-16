#!/usr/bin/env python3
"""Audit whether existing query/map feature spaces can rank local pose candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.students.pose_energy_net import pose_costs_and_residual_targets  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import (  # noqa: E402
    build_model_and_data,
    default_lattice,
    make_fixed_init_poses,
    make_random_init_poses,
    map_pose_gt_for_batch,
)
from feature_extract.tools.train_pose_energy import _good_bad_auc_rows, _spearman_rows  # noqa: E402
from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher  # noqa: E402
from feature_extract.tools.train_nvs_pose_feature_adapter import (  # noqa: E402
    apply_config_defaults as apply_nvs_config_defaults,
    build_pose_feature_adapter,
    load_adapter_checkpoint,
    pair_matcher_local_candidate_scores,
    project_render_bank,
)
from feature_extract.train_impl import (  # noqa: E402
    build_local_pose_lattice_candidates,
    fine_candidate_selector_features,
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    project_query_render_for_fine_selector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help="Optional POFD/NVS adapter checkpoint; enables adapted query/render audit.",
    )
    parser.add_argument("--adapter-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-render-chunk-size", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--candidate-render-batch-size", type=int, default=None)
    parser.add_argument("--buckets", default="25:5,50:10")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--candidate-bank-mode", choices=("lattice", "cache"), default="lattice")
    parser.add_argument("--lattice-trans-cm", default=None)
    parser.add_argument("--lattice-rot-deg", default=None)
    parser.add_argument("--lattice-direction-mode", choices=("axis", "cube"), default="cube")
    parser.add_argument("--combine-trans-rot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit-strategy", default="uniform")
    parser.add_argument("--init-noise-mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--init-jitter-seed", type=int, default=20260513)
    parser.add_argument("--score-mode", choices=("global", "local", "pair_matcher_local"), default="global")
    parser.add_argument("--score-radius", type=int, default=4)
    parser.add_argument("--score-preprocess", default="spatial_center")
    parser.add_argument("--score-highpass-kernel", type=int, default=5)
    parser.add_argument("--score-map-mode", default="peak_offset")
    parser.add_argument("--pair-matcher-radius", type=int, default=None)
    parser.add_argument("--pair-matcher-score-stride", type=int, default=8)
    parser.add_argument("--pair-matcher-temperature", type=float, default=0.05)
    parser.add_argument("--pair-matcher-score-chunk-points", type=int, default=65536)
    parser.add_argument("--pair-matcher-score-candidate-chunk-size", type=int, default=0)
    parser.add_argument("--pair-matcher-candidate-score-mode", default="center_logprob_margin")
    parser.add_argument("--feature-hw", default="68,120", help="Optional H,W downsample for audit scoring; empty keeps native.")
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--auc-good-m", type=float, default=0.05)
    parser.add_argument("--auc-bad-m", type=float, default=0.25)
    parser.add_argument("--require-projector", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_buckets(value: str) -> List[Tuple[str, float, float]]:
    buckets: List[Tuple[str, float, float]] = []
    for item in str(value).split(","):
        if not item.strip():
            continue
        trans, rot = item.split(":")
        trans_cm = float(trans)
        rot_deg = float(rot)
        buckets.append((f"{int(trans_cm):03d}cm_{rot_deg:g}deg", trans_cm, rot_deg))
    return buckets


def parse_float_csv(value: str | None) -> List[float]:
    if value is None or str(value).strip() == "":
        return []
    return [float(part) for part in str(value).split(",") if part.strip()]


def parse_hw(value: str | None) -> Tuple[int, int] | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [int(part) for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--feature-hw must be formatted as H,W")
    return int(parts[0]), int(parts[1])


def resize_feature(feature: torch.Tensor, hw: Tuple[int, int] | None) -> torch.Tensor:
    if hw is None or tuple(feature.shape[-2:]) == tuple(hw):
        return feature.float()
    if feature.ndim == 4:
        return F.interpolate(feature.float(), size=hw, mode="bilinear", align_corners=False)
    if feature.ndim == 5:
        bsz, num, channels, _height, _width = feature.shape
        flat = feature.reshape(bsz * num, channels, *feature.shape[-2:])
        flat = F.interpolate(flat.float(), size=hw, mode="bilinear", align_corners=False)
        return flat.reshape(bsz, num, channels, *hw)
    raise ValueError(f"unsupported feature shape for resize: {tuple(feature.shape)}")


def stats(values: Iterable[float], *, scale: float = 1.0) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32) * float(scale)
    if tensor.numel() == 0:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": float(tensor.mean().item()), "median": float(tensor.median().item())}


def append_combo_metrics(
    rows: Dict[str, Dict[str, List[float]]],
    name: str,
    scores: torch.Tensor,
    costs: torch.Tensor,
    valid: torch.Tensor,
    trans_err: torch.Tensor,
    rot_err: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    combo = rows.setdefault(
        name,
        {
            "spearman": [],
            "good_bad_auc": [],
            "selected_cost_m": [],
            "oracle_cost_m": [],
            "selected_trans_m": [],
            "selected_rot_deg": [],
            "oracle_trans_m": [],
            "oracle_rot_deg": [],
            "top1_acc": [],
            "identity_frac": [],
        },
    )
    scores = scores.float().masked_fill(~valid, -1.0e6)
    costs = costs.float().masked_fill(~valid, float("inf"))
    pred_idx = scores.argmax(dim=1)
    best_idx = costs.argmin(dim=1)
    batch_idx = torch.arange(scores.shape[0], device=scores.device)
    combo["spearman"].append(float(_spearman_rows(scores, costs, valid).detach().cpu()))
    combo["good_bad_auc"].append(
        float(
            _good_bad_auc_rows(
                scores,
                costs,
                valid,
                good_m=float(args.auc_good_m),
                bad_m=float(args.auc_bad_m),
            )
            .detach()
            .cpu()
        )
    )
    combo["selected_cost_m"].extend(costs[batch_idx, pred_idx].detach().cpu().tolist())
    combo["oracle_cost_m"].extend(costs[batch_idx, best_idx].detach().cpu().tolist())
    combo["selected_trans_m"].extend(trans_err[batch_idx, pred_idx].detach().cpu().tolist())
    combo["selected_rot_deg"].extend((rot_err[batch_idx, pred_idx] * (180.0 / math.pi)).detach().cpu().tolist())
    combo["oracle_trans_m"].extend(trans_err[batch_idx, best_idx].detach().cpu().tolist())
    combo["oracle_rot_deg"].extend((rot_err[batch_idx, best_idx] * (180.0 / math.pi)).detach().cpu().tolist())
    combo["top1_acc"].extend((pred_idx == best_idx).float().detach().cpu().tolist())
    identity_idx = torch.zeros_like(pred_idx)
    combo["identity_frac"].extend((pred_idx == identity_idx).float().detach().cpu().tolist())


def score_combo(
    query_feat: torch.Tensor,
    candidate_feat: torch.Tensor,
    candidate_pose: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
    *,
    pair_matcher=None,
) -> torch.Tensor:
    if str(args.score_mode or "").lower() == "pair_matcher_local":
        if pair_matcher is None:
            raise ValueError("score-mode=pair_matcher_local requires a loaded pair matcher")
        scores, _stats = pair_matcher_local_candidate_scores(
            pair_matcher,
            query_feat,
            candidate_feat,
            mask=None,
            radius=int(args.pair_matcher_radius if args.pair_matcher_radius is not None else args.score_radius),
            stride=int(args.pair_matcher_score_stride),
            temperature=float(args.pair_matcher_temperature),
            chunk_points=int(args.pair_matcher_score_chunk_points),
            candidate_chunk_size=int(args.pair_matcher_score_candidate_chunk_size),
            candidate_score_mode=str(args.pair_matcher_candidate_score_mode),
        )
        return scores, valid.bool()
    pack = fine_candidate_selector_features(
        query_feat,
        candidate_feat,
        candidate_pose,
        candidate_valid_mask=valid,
        mode=args.score_mode,
        radius=int(args.score_radius),
        preprocess=args.score_preprocess,
        highpass_kernel=int(args.score_highpass_kernel),
        score_map_mode=args.score_map_mode,
        return_score_maps=False,
        use_coarse_logits=False,
        use_candidate_delta=False,
        use_delta_vector=False,
        use_depth=False,
        use_mask=False,
    )
    return pack["render_scores"], pack["valid"]


def build_adapter_bundle(model, cfg: Dict, args: argparse.Namespace, device: torch.device):
    if not args.adapter_checkpoint:
        return None
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    checkpoint_args = dict(checkpoint.get("args") or {})
    if not checkpoint_args:
        checkpoint_args = dict(cfg.get("nvs_pose_feature_adapter", {}))
    adapter_args = argparse.Namespace(**checkpoint_args)
    adapter_args = apply_nvs_config_defaults(adapter_args, cfg)
    adapter = build_pose_feature_adapter(adapter_args, model, cfg, device)
    if adapter is None:
        raise ValueError("adapter checkpoint requires pose_feature_adapter_enabled=true")
    pair_matcher = None
    if checkpoint.get("pair_matcher_state_dict") is not None or str(args.score_mode) == "pair_matcher_local":
        pair_matcher = PairConditionedLocalMatcher(
            channels=int(adapter.channels),
            hidden_dim=int(getattr(adapter_args, "pair_matcher_hidden_dim", 64)),
            offset_radius=int(getattr(adapter_args, "pair_matcher_radius", 3)),
            zero_init_residual=bool(getattr(adapter_args, "pair_matcher_zero_init_residual", True)),
            base_dot_weight=float(getattr(adapter_args, "pair_matcher_base_dot_weight", 1.0)),
        ).to(device)
    loaded = load_adapter_checkpoint(
        args.adapter_checkpoint,
        adapter,
        model=model,
        pair_matcher=pair_matcher,
        strict_adapter=bool(args.adapter_strict),
    )
    adapter.eval()
    if pair_matcher is not None:
        pair_matcher.eval()
    return {
        "adapter": adapter,
        "pair_matcher": pair_matcher,
        "args": adapter_args,
        "checkpoint_step": loaded.get("step"),
    }


@torch.no_grad()
def evaluate_bucket(model, loader, map_renderer, cfg: Dict, args: argparse.Namespace, bucket, adapter_bundle=None) -> Dict:
    name, trans_cm, rot_deg = bucket
    device = next(model.parameters()).device
    feature_hw = parse_hw(args.feature_hw)
    lattice_trans = parse_float_csv(args.lattice_trans_cm) or default_lattice(trans_cm, rot_deg)[0]
    lattice_rot = parse_float_csv(args.lattice_rot_deg) or default_lattice(trans_cm, rot_deg)[1]
    rows: Dict[str, Dict[str, List[float]]] = {}
    init_trans_rows: List[float] = []
    init_rot_rows: List[float] = []
    processed = 0
    sample_offset = int(args.skip_samples or 0)
    model.eval()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
        if str(args.candidate_bank_mode).lower() == "cache":
            if "pose_init_candidates" not in batch:
                raise KeyError("candidate-bank-mode=cache requires pose_init_candidates in the dataset batch")
            candidate_poses = batch["pose_init_candidates"].to(device=device, dtype=torch.float32)
            topk = max(1, min(int(args.topk), int(candidate_poses.shape[1])))
            candidate_poses = candidate_poses[:, :topk]
            valid = batch.get("candidate_valid_mask")
            if valid is None:
                valid = torch.ones(candidate_poses.shape[:2], device=device, dtype=torch.bool)
            else:
                valid = valid.to(device=device).bool()[:, :topk]
            if "pose_init" in batch:
                init_pose = batch["pose_init"].to(device=device, dtype=torch.float32)
            else:
                init_pose = candidate_poses[:, 0].detach().clone()
            sample_offset += int(pose_gt.shape[0])
        else:
            if str(args.init_noise_mode) == "random":
                init_pose = make_random_init_poses(
                    pose_gt,
                    trans_cm,
                    rot_deg,
                    seed=int(args.init_jitter_seed),
                    offset=sample_offset,
                )
            else:
                init_pose = make_fixed_init_poses(pose_gt, trans_cm, rot_deg, offset=sample_offset)
            sample_offset += int(pose_gt.shape[0])
            candidate_poses = build_local_pose_lattice_candidates(
                init_pose,
                trans_cm=lattice_trans,
                rot_deg=lattice_rot,
                include_identity=True,
                max_candidates=int(args.topk),
                limit_strategy=args.limit_strategy,
                combine_trans_rot=bool(args.combine_trans_rot),
                direction_mode=args.lattice_direction_mode,
            )
            valid = torch.ones(candidate_poses.shape[:2], device=device, dtype=torch.bool)
        adapter_args = adapter_bundle.get("args") if adapter_bundle is not None else None
        adapter_needs_rgb = bool(adapter_args is not None and (
            getattr(adapter_args, "pose_feature_adapter_rgb_context_enabled", False)
            or getattr(adapter_args, "pose_feature_adapter_texture_branch_enabled", False)
        ))
        cand_batch = map_renderer.attach_pose_candidate_renders(
            dict(batch),
            candidate_poses,
            prefix="audit_candidate",
            candidate_valid_mask=valid,
            require_grad=False,
            feature="all",
            include_aux=adapter_needs_rgb,
        )
        if bool(cfg.get("model", {}).get("teacher_fine_condition", False)):
            outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
        else:
            outputs = model(batch["rgb"])
        query_key = cfg.get("map_supervision", {}).get("query_fine_key", "fine")
        teacher_query = resize_feature(batch["teacher_fine"].float(), feature_hw)
        student_query = resize_feature(outputs.get(query_key, outputs["fine"]).float(), feature_hw)
        map_render = resize_feature(cand_batch["audit_candidate_fine"].float(), feature_hw)
        candidate_pose = cand_batch["audit_candidate_pose"].float()
        pose_cost, _residual, trans_err, rot_err = pose_costs_and_residual_targets(
            candidate_pose,
            pose_gt.float(),
            valid_mask=valid,
            rot_cost_weight=float(args.rot_cost_weight),
        )
        _init_loss, init_rot, init_trans = pose_error_tensors(init_pose.float(), pose_gt.float())
        init_trans_rows.extend(init_trans.detach().cpu().tolist())
        init_rot_rows.extend(init_rot.detach().cpu().tolist())

        pair_matcher = adapter_bundle.get("pair_matcher") if adapter_bundle is not None else None
        scores, combo_valid = score_combo(teacher_query, map_render, candidate_pose, valid, args, pair_matcher=pair_matcher)
        append_combo_metrics(rows, "teacher_query_vs_map_base", scores, pose_cost, combo_valid, trans_err, rot_err, args)

        scores, combo_valid = score_combo(student_query, map_render, candidate_pose, valid, args, pair_matcher=pair_matcher)
        append_combo_metrics(rows, "student_query_vs_map_base", scores, pose_cost, combo_valid, trans_err, rot_err, args)

        if adapter_bundle is not None:
            adapter = adapter_bundle["adapter"]
            query_rgb = batch.get("rgb") if adapter_needs_rgb else None
            render_rgb = cand_batch.get("audit_candidate_rgb") if adapter_needs_rgb else None
            query_adapt = adapter.project_query(outputs.get(query_key, outputs["fine"]).float(), rgb=query_rgb)
            render_adapt = project_render_bank(
                adapter,
                cand_batch["audit_candidate_fine"].float(),
                render_rgb=render_rgb,
                render_chunk_size=int(args.adapter_render_chunk_size),
            )
            query_adapt = resize_feature(query_adapt, feature_hw)
            render_adapt = resize_feature(render_adapt, feature_hw)
            scores, combo_valid = score_combo(
                query_adapt,
                render_adapt,
                candidate_pose,
                valid,
                args,
                pair_matcher=pair_matcher,
            )
            append_combo_metrics(rows, "student_adapter_vs_map_adapter", scores, pose_cost, combo_valid, trans_err, rot_err, args)

        projector = getattr(model, "local_corr_projector", None)
        if projector is not None:
            query_proj, render_proj, used_projector = project_query_render_for_fine_selector(
                projector,
                outputs.get(query_key, outputs["fine"]).float(),
                cand_batch["audit_candidate_fine"].float(),
                require_projector=bool(args.require_projector),
                render_chunk_size=int(cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0),
            )
            if used_projector:
                query_proj = resize_feature(query_proj, feature_hw)
                render_proj = resize_feature(render_proj, feature_hw)
                scores, combo_valid = score_combo(
                    query_proj,
                    render_proj,
                    candidate_pose,
                    valid,
                    args,
                    pair_matcher=pair_matcher,
                )
                append_combo_metrics(rows, "student_projected_vs_map_projected", scores, pose_cost, combo_valid, trans_err, rot_err, args)

        processed += int(pose_gt.shape[0])
        if args.max_samples is not None and processed >= int(args.max_samples):
            break

    summary = {
        "bucket": name,
        "samples": int(min(processed, int(args.max_samples or processed))),
        "init_trans_mm": stats(init_trans_rows, scale=1000.0),
        "init_rot_deg": stats(init_rot_rows),
        "feature_hw": list(feature_hw) if feature_hw is not None else None,
        "score_mode": args.score_mode,
        "combos": {},
    }
    for combo_name, combo_rows in rows.items():
        summary["combos"][combo_name] = {
            "spearman": float(torch.tensor(combo_rows["spearman"]).mean().item()) if combo_rows["spearman"] else 0.0,
            "good_bad_auc": float(torch.tensor(combo_rows["good_bad_auc"]).mean().item()) if combo_rows["good_bad_auc"] else 0.5,
            "selected_cost_m": stats(combo_rows["selected_cost_m"]),
            "oracle_cost_m": stats(combo_rows["oracle_cost_m"]),
            "selected_trans_mm": stats(combo_rows["selected_trans_m"], scale=1000.0),
            "selected_rot_deg": stats(combo_rows["selected_rot_deg"]),
            "oracle_trans_mm": stats(combo_rows["oracle_trans_m"], scale=1000.0),
            "oracle_rot_deg": stats(combo_rows["oracle_rot_deg"]),
            "top1_acc": float(torch.tensor(combo_rows["top1_acc"]).mean().item()) if combo_rows["top1_acc"] else 0.0,
            "identity_frac": float(torch.tensor(combo_rows["identity_frac"]).mean().item()) if combo_rows["identity_frac"] else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg["training"]["num_workers"] = int(args.num_workers)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, loader, map_renderer = build_model_and_data(cfg, args, device)
    for param in model.parameters():
        param.requires_grad_(False)
    adapter_bundle = build_adapter_bundle(model, cfg, args, device)
    results = {}
    for bucket in parse_buckets(args.buckets):
        print(f"auditing bucket {bucket[0]}", flush=True)
        results[bucket[0]] = evaluate_bucket(model, loader, map_renderer, cfg, args, bucket, adapter_bundle=adapter_bundle)
        print(json.dumps({bucket[0]: results[bucket[0]]}, sort_keys=True), flush=True)
    summary = {
        "checkpoint": args.checkpoint,
        "map_checkpoint": args.map_checkpoint,
        "adapter_checkpoint": args.adapter_checkpoint,
        "split": args.split,
        "topk": int(args.topk),
        "candidate_bank_mode": args.candidate_bank_mode,
        "buckets": results,
    }
    (out_dir / "feature_pose_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
