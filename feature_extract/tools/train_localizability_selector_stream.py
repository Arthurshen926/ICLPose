#!/usr/bin/env python3
"""Streaming POFD-FS selector training from frozen query/map features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.losses import (  # noqa: E402
    basin_bce_loss,
    channel_sparsity_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
    spatial_utility_entropy_loss,
)
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.scorer import PoseHypothesisScorer  # noqa: E402
from feature_extract.localizability.selector import LocalizationFeatureSelector  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.eval_localizability_feature_score import (  # noqa: E402
    _load_pose_adapter_bundle,
    _parse_hw,
    _resize_feature,
    _resize_mask,
)
from feature_extract.train_impl import (  # noqa: E402
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    project_query_render_for_fine_selector,
    set_seed,
)


if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            options = []
            for option in option_strings:
                options.append(option)
                if option.startswith("--"):
                    options.append("--no-" + option[2:])
            super().__init__(option_strings=options, dest=dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

    argparse.BooleanOptionalAction = _BooleanOptionalAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--pose-adapter-checkpoint", default=None)
    parser.add_argument("--pose-adapter-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-pose-candidate-cache", required=True)
    parser.add_argument("--eval-pose-candidate-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-split", choices=("train", "val"), default="train")
    parser.add_argument("--eval-split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--candidate-render-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument(
        "--query-feature-source",
        choices=("raw_radio", "query_student", "projected_query_student"),
        default="query_student",
    )
    parser.add_argument("--query-feature-key", default="fine")
    parser.add_argument("--selector-out-dim", type=int, default=64)
    parser.add_argument("--selector-group-size", type=int, default=8)
    parser.add_argument("--train-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-channel-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-utility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-uncertainty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-mode", choices=("same_pixel", "local_corr", "pair_matcher_local"), default="local_corr")
    parser.add_argument("--score-radius", type=int, default=4)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--score-feature-hw", default="34,60")
    parser.add_argument("--pair-matcher-stride", type=int, default=4)
    parser.add_argument("--pair-matcher-chunk-points", type=int, default=1024)
    parser.add_argument("--pair-matcher-candidate-chunk-size", type=int, default=4)
    parser.add_argument("--pair-matcher-offset-chunk-size", type=int, default=0)
    parser.add_argument("--pair-matcher-candidate-score-mode", default="center_logprob_margin")
    parser.add_argument("--pair-matcher-score-channel", type=int, default=0)
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--basin-weight", type=float, default=0.5)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--utility-entropy-weight", type=float, default=0.01)
    return parser.parse_args()


def _set_selector_trainable(selector: LocalizationFeatureSelector, args: argparse.Namespace) -> list[torch.nn.Parameter]:
    for param in selector.parameters():
        param.requires_grad_(False)
    if bool(args.train_projection):
        for param in selector.proj.parameters():
            param.requires_grad_(True)
    if bool(args.train_channel_gate):
        for param in selector.channel_gate.parameters():
            param.requires_grad_(True)
    if bool(args.train_utility) and selector.utility_head is not None:
        for param in selector.utility_head.parameters():
            param.requires_grad_(True)
    if bool(args.train_uncertainty) and selector.uncertainty_head is not None:
        for param in selector.uncertainty_head.parameters():
            param.requires_grad_(True)
    params = [param for param in selector.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No selector parameters are trainable")
    return params


def _build_selector_stream_scorer(
    args: argparse.Namespace,
    *,
    pair_matcher: torch.nn.Module | None,
) -> PoseHypothesisScorer:
    if str(args.score_mode) == "pair_matcher_local" and pair_matcher is None:
        raise ValueError("--score-mode=pair_matcher_local requires --pose-adapter-checkpoint with pair_matcher_state_dict")
    return PoseHypothesisScorer(
        mode=args.score_mode,
        radius=int(args.score_radius),
        temperature=float(args.score_temperature),
        pair_matcher=pair_matcher,
        pair_matcher_stride=int(args.pair_matcher_stride),
        pair_matcher_chunk_points=int(args.pair_matcher_chunk_points),
        pair_matcher_candidate_chunk_size=int(args.pair_matcher_candidate_chunk_size),
        pair_matcher_offset_chunk_size=int(args.pair_matcher_offset_chunk_size),
        pair_matcher_candidate_score_mode=str(args.pair_matcher_candidate_score_mode),
        pair_matcher_score_channel=int(args.pair_matcher_score_channel),
    )


def _loader_args(args: argparse.Namespace, *, split: str, max_samples: int | None, cache: str) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.split = split
    ns.max_samples = max_samples
    ns.skip_samples = 0
    ns.pose_candidate_cache = cache
    return ns


def _apply_cache_override(cfg: dict, *, split: str, cache: str) -> dict:
    cloned = json.loads(json.dumps(cfg))
    cloned.setdefault("dataset", {})[f"{split}_pose_candidate_cache"] = str(cache)
    return cloned


def _query_feature(args: argparse.Namespace, model, batch: dict) -> torch.Tensor:
    if args.query_feature_source == "raw_radio":
        return batch["teacher_fine"].float()
    outputs = model(batch["rgb"])
    if args.query_feature_key not in outputs:
        raise KeyError(f"query feature key {args.query_feature_key!r} not found in model outputs")
    return outputs[args.query_feature_key].float()


def _pose_costs(args: argparse.Namespace, candidates: torch.Tensor, pose_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_candidates = candidates.shape[:2]
    pose_gt_flat = pose_gt[:, None].expand(-1, num_candidates, -1, -1).reshape(-1, 4, 4)
    _, rot_err_deg, trans_err_m = pose_error_tensors(candidates.reshape(-1, 4, 4), pose_gt_flat)
    trans_err_m = trans_err_m.reshape(bsz, num_candidates)
    rot_err_deg = rot_err_deg.reshape(bsz, num_candidates)
    pose_cost = trans_err_m + float(args.rot_cost_weight) * torch.deg2rad(rot_err_deg)
    return pose_cost, trans_err_m, rot_err_deg


def _forward_scores(
    args: argparse.Namespace,
    *,
    model,
    map_renderer,
    selector: LocalizationFeatureSelector,
    scorer: PoseHypothesisScorer,
    batch: dict,
    device: torch.device,
    score_hw: tuple[int, int] | None,
    return_selected_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
    candidates = batch["pose_init_candidates"].float()
    cand_batch = map_renderer.attach_pose_candidate_renders(
        batch,
        candidates,
        require_grad=False,
        prefix="localizability_candidate",
        candidate_valid_mask=batch.get("candidate_valid_mask"),
        feature="fine",
        include_aux=True,
    )
    if args.query_feature_source == "projected_query_student":
        outputs = model(batch["rgb"])
        if args.query_feature_key not in outputs:
            raise KeyError(f"query feature key {args.query_feature_key!r} not found in model outputs")
        query_base, render_base, used_projector = project_query_render_for_fine_selector(
            getattr(model, "local_corr_projector", None),
            outputs[args.query_feature_key].float(),
            cand_batch["localizability_candidate_fine"].float(),
            require_projector=True,
            render_chunk_size=16,
        )
        if not used_projector:
            raise RuntimeError("projected_query_student requested but no projector was used")
    else:
        query_base = _query_feature(args, model, batch)
        render_base = cand_batch["localizability_candidate_fine"].float()
    query_base = _resize_feature(query_base, score_hw)
    render_base = _resize_feature(render_base, score_hw)
    render_mask = _resize_mask(cand_batch.get("localizability_candidate_mask"), score_hw)
    q_out = selector(query_base)
    bsz, num_candidates, channels, height, width = render_base.shape
    r_out = selector(render_base.reshape(bsz * num_candidates, channels, height, width))
    render_z = r_out["z"].reshape(bsz, num_candidates, args.selector_out_dim, height, width)
    scores, aux = scorer(q_out["z"], render_z, query_utility=q_out["utility"], render_valid_mask=render_mask)
    pose_cost, trans_err_m, rot_err_deg = _pose_costs(args, candidates, pose_gt)
    aux_out = {
        "query_channel_gate": q_out["channel_gate"],
        "query_utility": q_out["utility"],
        **aux,
    }
    if return_selected_features:
        aux_out.update(
            {
                "query_z": q_out["z"],
                "render_z": render_z,
                "render_mask": render_mask,
            }
        )
    return scores, pose_cost, trans_err_m, rot_err_deg, batch.get("candidate_valid_mask"), aux_out


def _evaluate(
    args: argparse.Namespace,
    *,
    model,
    map_renderer,
    loader,
    selector,
    scorer,
    device,
    score_hw,
    step: int,
) -> dict:
    selector.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            scores, pose_cost, trans_err, rot_err, valid, _aux = _forward_scores(
                args,
                model=model,
                map_renderer=map_renderer,
                selector=selector,
                scorer=scorer,
                batch=batch,
                device=device,
                score_hw=score_hw,
            )
            basin = (trans_err <= float(args.basin_trans_m)) & (rot_err <= float(args.basin_rot_deg))
            rows.append(ranking_metrics(scores, pose_cost, valid_mask=valid, basin_label=basin, topk=(1, 5)))
    selector.train()
    out = {"split": "eval", "step": int(step)}
    for key in rows[0]:
        out[key] = float(torch.stack([row[key].detach().cpu() for row in rows]).mean())
    return out


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    base_cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    train_cfg = _apply_cache_override(base_cfg, split=args.train_split, cache=args.train_pose_candidate_cache)
    train_args = _loader_args(args, split=args.train_split, max_samples=args.max_samples, cache=args.train_pose_candidate_cache)
    model, train_loader, map_renderer = build_model_and_data(train_cfg, train_args, device)
    eval_cfg = _apply_cache_override(base_cfg, split=args.eval_split, cache=args.eval_pose_candidate_cache)
    eval_args = _loader_args(args, split=args.eval_split, max_samples=args.eval_max_samples, cache=args.eval_pose_candidate_cache)
    _eval_model, eval_loader, _eval_renderer = build_model_and_data(eval_cfg, eval_args, device)
    del _eval_model, _eval_renderer
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    in_channels = int(getattr(model, "fine_feature_dim", 64))
    if args.query_feature_source == "raw_radio":
        in_channels = int(train_cfg.get("model", {}).get("fine_feature_dim", in_channels))
    selector = LocalizationFeatureSelector(
        in_channels=in_channels,
        out_channels=int(args.selector_out_dim),
        group_size=int(args.selector_group_size),
    ).to(device)
    pair_matcher = None
    if args.pose_adapter_checkpoint:
        _adapter, pair_matcher, _adapter_args = _load_pose_adapter_bundle(
            args.pose_adapter_checkpoint,
            channels=int(args.selector_out_dim),
            device=device,
            strict_adapter=bool(args.pose_adapter_strict),
        )
        del _adapter, _adapter_args
    scorer = _build_selector_stream_scorer(args, pair_matcher=pair_matcher).to(device)
    trainable_params = _set_selector_trainable(selector, args)
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    score_hw = _parse_hw(args.score_feature_hw)
    best = float("inf")
    train_iter = iter(train_loader)
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, int(args.max_steps) + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = move_batch_to_device(batch, device)
            selector.train()
            scores, pose_cost, trans_err, rot_err, valid, aux = _forward_scores(
                args,
                model=model,
                map_renderer=map_renderer,
                selector=selector,
                scorer=scorer,
                batch=batch,
                device=device,
                score_hw=score_hw,
            )
            basin = (trans_err <= float(args.basin_trans_m)) & (rot_err <= float(args.basin_rot_deg))
            rank_loss, _rank = pose_distance_soft_rank_loss(scores, pose_cost, valid_mask=valid)
            hard_loss, _hard = online_score_hard_negative_loss(scores, pose_cost, valid_mask=valid)
            loss = (
                float(args.rank_weight) * rank_loss
                + float(args.hard_weight) * hard_loss
                + float(args.basin_weight) * basin_bce_loss(scores, basin, valid_mask=valid)
                + float(args.sparsity_weight) * channel_sparsity_loss(aux["query_channel_gate"])
                + float(args.utility_entropy_weight) * spatial_utility_entropy_loss(aux["query_utility"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), float(args.grad_clip))
            optimizer.step()
            if step % int(args.eval_every) == 0 or step == int(args.max_steps):
                row = _evaluate(
                    args,
                    model=model,
                    map_renderer=map_renderer,
                    loader=eval_loader,
                    selector=selector,
                    scorer=scorer,
                    device=device,
                    score_hw=score_hw,
                    step=step,
                )
                row["loss"] = float(loss.detach().cpu())
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if row["pred_cost_m"] < best:
                    best = row["pred_cost_m"]
                    torch.save(
                        {
                            "selector_state_dict": selector.state_dict(),
                            "step": int(step),
                            "metrics": row,
                            "args": vars(args),
                        },
                        out_dir / "best.pth",
                    )


if __name__ == "__main__":
    main()
