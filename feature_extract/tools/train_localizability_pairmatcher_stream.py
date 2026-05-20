#!/usr/bin/env python3
"""Geometry-supervised POFD-FS pair-matcher training.

This entry point deliberately avoids external candidate-quality labels.  It
uses only GT pose distance within a candidate bank to shape the localization
feature score surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.adapter_bundle import collect_pose_adapter_trainable_parameters  # noqa: E402
from feature_extract.localizability.losses import (  # noqa: E402
    basin_bce_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
)
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.scorer import PoseHypothesisScorer  # noqa: E402
from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher, PoseFeatureDomainAdapter  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.eval_localizability_feature_score import (  # noqa: E402
    _load_pose_adapter_bundle,
    _parse_hw,
    _resize_mask,
)
from feature_extract.tools.train_localizability_selector_stream import _apply_cache_override, _loader_args  # noqa: E402
from feature_extract.tools.train_nvs_pose_feature_adapter import project_render_bank  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    set_seed,
)


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
    parser.add_argument("--eval-max-samples", type=int, default=128)
    parser.add_argument("--candidate-render-batch-size", type=int, default=4)
    parser.add_argument("--render-project-chunk-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--query-feature-key", default="fine")
    parser.add_argument("--score-radius", type=int, default=16)
    parser.add_argument("--score-temperature", type=float, default=0.04)
    parser.add_argument("--score-feature-hw", default="34,60")
    parser.add_argument("--pair-matcher-stride", type=int, default=4)
    parser.add_argument("--pair-matcher-chunk-points", type=int, default=1024)
    parser.add_argument("--pair-matcher-candidate-chunk-size", type=int, default=4)
    parser.add_argument("--pair-matcher-offset-chunk-size", type=int, default=0)
    parser.add_argument("--pair-matcher-score-channel", type=int, default=0)
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-temperature-m", type=float, default=0.05)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--hard-cost-gap-m", type=float, default=0.12)
    parser.add_argument("--hard-margin", type=float, default=0.08)
    parser.add_argument("--basin-weight", type=float, default=0.5)
    parser.add_argument("--train-query-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-render-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-rgb-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-texture-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-uncertainty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-pair-matcher", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _fresh_bundle(channels: int, device: torch.device) -> tuple[PoseFeatureDomainAdapter, PairConditionedLocalMatcher, dict]:
    adapter = PoseFeatureDomainAdapter(
        channels=int(channels),
        hidden_dim=128,
        residual_scale=0.2,
        zero_init=True,
        l2_normalize=True,
    ).to(device)
    pair_matcher = PairConditionedLocalMatcher(
        channels=int(channels),
        hidden_dim=128,
        offset_radius=16,
        zero_init_residual=True,
        base_dot_weight=1.0,
    ).to(device)
    return adapter, pair_matcher, {}


def _pose_costs(
    args: argparse.Namespace,
    candidates: torch.Tensor,
    pose_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_candidates = candidates.shape[:2]
    pose_gt_flat = pose_gt[:, None].expand(-1, num_candidates, -1, -1).reshape(-1, 4, 4)
    _, rot_err_deg, trans_err_m = pose_error_tensors(candidates.reshape(-1, 4, 4), pose_gt_flat)
    trans_err_m = trans_err_m.reshape(bsz, num_candidates)
    rot_err_deg = rot_err_deg.reshape(bsz, num_candidates)
    pose_cost = trans_err_m + float(args.rot_cost_weight) * torch.deg2rad(rot_err_deg)
    return pose_cost, trans_err_m, rot_err_deg


def _adapter_needs_rgb(adapter_args: dict) -> bool:
    return bool(adapter_args.get("pose_feature_adapter_rgb_context_enabled", False)) or bool(
        adapter_args.get("pose_feature_adapter_texture_branch_enabled", False)
    )


def _forward_scores(
    args: argparse.Namespace,
    *,
    model,
    map_renderer,
    adapter: PoseFeatureDomainAdapter,
    scorer: PoseHypothesisScorer,
    batch: dict,
    device: torch.device,
    score_hw: tuple[int, int] | None,
    adapter_args: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    with torch.no_grad():
        outputs = model(batch["rgb"])
        if args.query_feature_key not in outputs:
            raise KeyError(f"query feature key {args.query_feature_key!r} not found in model outputs")
        query_base = outputs[args.query_feature_key].float()
    needs_rgb = _adapter_needs_rgb(adapter_args)
    query_rgb = batch.get("rgb") if needs_rgb else None
    render_rgb = cand_batch.get("localizability_candidate_rgb") if needs_rgb else None
    query_loc = adapter.project_query(query_base, rgb=query_rgb)
    render_loc = project_render_bank(
        adapter,
        cand_batch["localizability_candidate_fine"].float(),
        render_rgb=render_rgb,
        render_chunk_size=int(args.render_project_chunk_size),
    )
    if score_hw is not None and tuple(query_loc.shape[-2:]) != tuple(score_hw):
        query_loc = torch.nn.functional.interpolate(query_loc.float(), size=score_hw, mode="bilinear", align_corners=False)
    if score_hw is not None and tuple(render_loc.shape[-2:]) != tuple(score_hw):
        bsz, num_candidates, channels, height, width = render_loc.shape
        render_loc = torch.nn.functional.interpolate(
            render_loc.reshape(bsz * num_candidates, channels, height, width).float(),
            size=score_hw,
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, num_candidates, channels, score_hw[0], score_hw[1])
    render_mask = _resize_mask(cand_batch.get("localizability_candidate_mask"), score_hw)
    scores, _aux = scorer(query_loc, render_loc, render_valid_mask=render_mask)
    pose_cost, trans_err_m, rot_err_deg = _pose_costs(args, candidates, pose_gt)
    return scores, pose_cost, trans_err_m, rot_err_deg, batch.get("candidate_valid_mask")


def _mean_rows(rows: list[dict[str, torch.Tensor]], *, step: int) -> dict:
    out = {"split": "eval", "step": int(step)}
    for key in rows[0]:
        out[key] = float(torch.stack([row[key].detach().cpu() for row in rows]).mean())
    return out


def _evaluate(args, *, model, map_renderer, loader, adapter, scorer, device, score_hw, adapter_args, step: int) -> dict:
    adapter.eval()
    scorer.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            scores, pose_cost, trans_err, rot_err, valid = _forward_scores(
                args,
                model=model,
                map_renderer=map_renderer,
                adapter=adapter,
                scorer=scorer,
                batch=batch,
                device=device,
                score_hw=score_hw,
                adapter_args=adapter_args,
            )
            basin = (trans_err <= float(args.basin_trans_m)) & (rot_err <= float(args.basin_rot_deg))
            rows.append(ranking_metrics(scores, pose_cost, valid_mask=valid, basin_label=basin, topk=(1, 5)))
    adapter.train()
    scorer.train()
    return _mean_rows(rows, step=step)


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
    _eval_model, eval_loader, eval_renderer = build_model_and_data(eval_cfg, eval_args, device)
    del _eval_model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    channels = int(getattr(model, "fine_feature_dim", 64))
    if args.pose_adapter_checkpoint:
        adapter, pair_matcher, adapter_args = _load_pose_adapter_bundle(
            args.pose_adapter_checkpoint,
            channels=channels,
            device=device,
            strict_adapter=bool(args.pose_adapter_strict),
        )
    else:
        adapter, pair_matcher, adapter_args = _fresh_bundle(channels, device)

    trainable = collect_pose_adapter_trainable_parameters(
        adapter,
        pair_matcher,
        train_query_adapter=bool(args.train_query_adapter),
        train_render_adapter=bool(args.train_render_adapter),
        train_rgb_context=bool(args.train_rgb_context),
        train_texture_branch=bool(args.train_texture_branch),
        train_uncertainty=bool(args.train_uncertainty),
        train_pair_matcher=bool(args.train_pair_matcher),
    )
    if not trainable:
        raise ValueError("No trainable pose-adapter or pair-matcher parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scorer = PoseHypothesisScorer(
        mode="pair_matcher_local",
        radius=int(args.score_radius),
        temperature=float(args.score_temperature),
        pair_matcher=pair_matcher,
        pair_matcher_stride=int(args.pair_matcher_stride),
        pair_matcher_chunk_points=int(args.pair_matcher_chunk_points),
        pair_matcher_candidate_chunk_size=int(args.pair_matcher_candidate_chunk_size),
        pair_matcher_offset_chunk_size=int(args.pair_matcher_offset_chunk_size),
        pair_matcher_score_channel=int(args.pair_matcher_score_channel),
    ).to(device)
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
            adapter.train()
            scorer.train()
            scores, pose_cost, trans_err, rot_err, valid = _forward_scores(
                args,
                model=model,
                map_renderer=map_renderer,
                adapter=adapter,
                scorer=scorer,
                batch=batch,
                device=device,
                score_hw=score_hw,
                adapter_args=adapter_args,
            )
            basin = (trans_err <= float(args.basin_trans_m)) & (rot_err <= float(args.basin_rot_deg))
            rank_loss, _rank = pose_distance_soft_rank_loss(
                scores,
                pose_cost,
                valid_mask=valid,
                temperature_m=float(args.rank_temperature_m),
            )
            hard_loss, _hard = online_score_hard_negative_loss(
                scores,
                pose_cost,
                valid_mask=valid,
                cost_gap_m=float(args.hard_cost_gap_m),
                margin=float(args.hard_margin),
            )
            loss = (
                float(args.rank_weight) * rank_loss
                + float(args.hard_weight) * hard_loss
                + float(args.basin_weight) * basin_bce_loss(scores, basin, valid_mask=valid)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            optimizer.step()
            if step % int(args.eval_every) == 0 or step == int(args.max_steps):
                row = _evaluate(
                    args,
                    model=model,
                    map_renderer=eval_renderer,
                    loader=eval_loader,
                    adapter=adapter,
                    scorer=scorer,
                    device=device,
                    score_hw=score_hw,
                    adapter_args=adapter_args,
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
                            "step": int(step),
                            "pose_feature_adapter_state_dict": adapter.state_dict(),
                            "pair_matcher_state_dict": pair_matcher.state_dict(),
                            "metrics": row,
                            "args": vars(args),
                            "config": base_cfg,
                        },
                        out_dir / "best.pth",
                    )


if __name__ == "__main__":
    main()
