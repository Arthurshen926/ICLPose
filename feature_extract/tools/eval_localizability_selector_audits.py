#!/usr/bin/env python3
"""Counterfactual audits for POFD-FS selector hypothesis scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.controls import (  # noqa: E402
    feature_batch_shuffle_control,
    wrong_scene_feature_control,
)
from feature_extract.localizability.interpretability import (  # noqa: E402
    channel_group_counterfactual_drop,
    spatial_utility_counterfactual_drop,
)
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.selector import LocalizationFeatureSelector  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data  # noqa: E402
from feature_extract.tools.eval_feature_track_mapability import _extract_selector_state, _safe_torch_load  # noqa: E402
from feature_extract.tools.eval_localizability_feature_score import _load_pose_adapter_bundle, _parse_hw  # noqa: E402
from feature_extract.tools.train_localizability_selector_stream import (  # noqa: E402
    _apply_cache_override,
    _build_selector_stream_scorer,
    _forward_scores,
    _loader_args,
)
from feature_extract.train_impl import load_config, move_batch_to_device, set_seed  # noqa: E402


METRIC_KEYS = (
    "pred_cost_m",
    "oracle_gap_m",
    "top1_acc",
    "spearman",
    "ndcg",
    "basin_recall@1",
    "basin_recall@5",
)


def _metrics_to_float(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {str(key): float(value.detach().cpu()) for key, value in metrics.items()}


def spatial_counterfactual_report(
    score_maps: torch.Tensor,
    utility: torch.Tensor,
    costs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    drop_fraction: float = 0.2,
    base_weight: torch.Tensor | None = None,
) -> dict[str, object]:
    """Summarize ranking damage after high-/low-utility spatial removal."""

    if utility.shape[-2:] != score_maps.shape[-2:]:
        utility = F.interpolate(utility.float(), size=score_maps.shape[-2:], mode="bilinear", align_corners=False)
    if base_weight is not None and base_weight.shape[-2:] != score_maps.shape[-2:]:
        base_weight = F.interpolate(base_weight.float(), size=score_maps.shape[-2:], mode="bilinear", align_corners=False)
    counterfactual = spatial_utility_counterfactual_drop(
        score_maps,
        utility,
        costs,
        drop_fraction=float(drop_fraction),
        base_weight=base_weight,
        valid_mask=valid_mask,
    )
    topk = (1, 5) if costs.shape[1] >= 5 else (1,)
    base = _metrics_to_float(
        ranking_metrics(
            counterfactual["base_scores"],
            costs,
            valid_mask=valid_mask,
            basin_label=basin_label,
            topk=topk,
        )
    )
    drop_high = _metrics_to_float(
        ranking_metrics(
            counterfactual["drop_high_scores"],
            costs,
            valid_mask=valid_mask,
            basin_label=basin_label,
            topk=topk,
        )
    )
    drop_low = _metrics_to_float(
        ranking_metrics(
            counterfactual["drop_low_scores"],
            costs,
            valid_mask=valid_mask,
            basin_label=basin_label,
            topk=topk,
        )
    )
    return {
        "base": base,
        "drop_high": drop_high,
        "drop_low": drop_low,
        "drop_high_cost_delta_m": float(counterfactual["drop_high_cost_delta_m"].detach().cpu()),
        "drop_low_cost_delta_m": float(counterfactual["drop_low_cost_delta_m"].detach().cpu()),
    }


def channel_counterfactual_report(
    base_scores: torch.Tensor,
    ablated_group_scores: torch.Tensor,
    costs: torch.Tensor,
    group_importance: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    importance_mode: str = "provided",
) -> dict[str, object]:
    """Summarize ranking damage after high-/low-importance channel removal."""

    if group_importance.ndim == 2:
        group_importance = group_importance.float().mean(dim=0)
    if group_importance.ndim != 1:
        raise ValueError("group_importance must have shape (G,) or (B,G)")
    if ablated_group_scores.ndim != 3 or ablated_group_scores.shape[0] != group_importance.shape[0]:
        raise ValueError("ablated_group_scores must have shape (G,B,K) matching group_importance")
    if importance_mode not in {"provided", "leave_one_group_out"}:
        raise ValueError("importance_mode must be 'provided' or 'leave_one_group_out'")
    group_report = channel_group_counterfactual_drop(
        base_scores,
        ablated_group_scores,
        costs,
        valid_mask=valid_mask,
    )
    if importance_mode == "leave_one_group_out":
        group_importance = group_report["group_pred_cost_drop_m"].detach()
    topk = (1, 5) if costs.shape[1] >= 5 else (1,)
    high_idx = int(group_importance.argmax().detach().cpu())
    low_idx = int(group_importance.argmin().detach().cpu())
    base = _metrics_to_float(
        ranking_metrics(
            base_scores,
            costs,
            valid_mask=valid_mask,
            basin_label=basin_label,
            topk=topk,
        )
    )
    drop_high = _metrics_to_float(
        ranking_metrics(
            ablated_group_scores[high_idx],
            costs,
            valid_mask=valid_mask,
            basin_label=basin_label,
            topk=topk,
        )
    )
    drop_low = _metrics_to_float(
        ranking_metrics(
            ablated_group_scores[low_idx],
            costs,
            valid_mask=valid_mask,
            basin_label=basin_label,
            topk=topk,
        )
    )
    return {
        "importance_mode": importance_mode,
        "base": base,
        "drop_high": drop_high,
        "drop_low": drop_low,
        "drop_high_group_idx": high_idx,
        "drop_low_group_idx": low_idx,
        "drop_high_cost_delta_m": float(drop_high["pred_cost_m"] - base["pred_cost_m"]),
        "drop_low_cost_delta_m": float(drop_low["pred_cost_m"] - base["pred_cost_m"]),
        "group_importance": [float(value) for value in group_importance.detach().cpu()],
        "group_pred_cost_drop_m": [
            float(value) for value in group_report["group_pred_cost_drop_m"].detach().cpu()
        ],
    }


def _ranking_report(
    scores: torch.Tensor,
    costs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    basin_label: torch.Tensor | None,
) -> dict[str, float]:
    topk = (1, 5) if costs.shape[1] >= 5 else (1,)
    return _metrics_to_float(ranking_metrics(scores, costs, valid_mask=valid_mask, basin_label=basin_label, topk=topk))


def _skip_report(reason: str) -> dict[str, object]:
    return {"skipped": True, "reason": str(reason)}


def feature_shuffle_control_report(
    scorer,
    query_z: torch.Tensor,
    render_z: torch.Tensor,
    costs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    query_utility: torch.Tensor | None = None,
    render_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Run query/candidate/wrong-scene feature-shuffle negative controls."""

    if query_z.ndim != 4 or render_z.ndim != 5:
        raise ValueError("query_z must be (B,C,H,W) and render_z must be (B,K,C,H,W)")
    if query_z.shape[0] != render_z.shape[0] or costs.shape != render_z.shape[:2]:
        raise ValueError("query/render/cost batch and candidate dimensions must match")
    with torch.no_grad():
        base_scores, _base_aux = scorer(query_z, render_z, query_utility=query_utility, render_valid_mask=render_mask)
        out: dict[str, object] = {
            "base": _ranking_report(base_scores, costs, valid_mask=valid_mask, basin_label=basin_label)
        }
        if render_z.shape[1] < 2:
            out["candidate_render_shuffle"] = _skip_report("need at least two candidates")
        else:
            shuffled_render = torch.roll(render_z, shifts=1, dims=1)
            shuffled_mask = torch.roll(render_mask, shifts=1, dims=1) if render_mask is not None else None
            candidate_scores, _candidate_aux = scorer(
                query_z,
                shuffled_render,
                query_utility=query_utility,
                render_valid_mask=shuffled_mask,
            )
            out["candidate_render_shuffle"] = _ranking_report(
                candidate_scores,
                costs,
                valid_mask=valid_mask,
                basin_label=basin_label,
            )
        if query_z.shape[0] < 2:
            out["query_batch_shuffle"] = _skip_report("need batch size at least two")
            out["wrong_scene_render_shuffle"] = _skip_report("need batch size at least two")
        else:
            permutation = torch.roll(torch.arange(query_z.shape[0], device=query_z.device), shifts=1)
            shuffled_query, permutation = feature_batch_shuffle_control(query_z, permutation=permutation)
            shuffled_utility = None
            if query_utility is not None:
                shuffled_utility = query_utility.index_select(0, permutation.to(device=query_utility.device))
            query_scores, _query_aux = scorer(
                shuffled_query,
                render_z,
                query_utility=shuffled_utility,
                render_valid_mask=render_mask,
            )
            out["query_batch_shuffle"] = _ranking_report(
                query_scores,
                costs,
                valid_mask=valid_mask,
                basin_label=basin_label,
            )
            wrong_render = wrong_scene_feature_control(render_z, render_z.index_select(0, permutation))
            wrong_mask = render_mask.index_select(0, permutation.to(device=render_mask.device)) if render_mask is not None else None
            wrong_scores, _wrong_aux = scorer(
                query_z,
                wrong_render,
                query_utility=query_utility,
                render_valid_mask=wrong_mask,
            )
            out["wrong_scene_render_shuffle"] = _ranking_report(
                wrong_scores,
                costs,
                valid_mask=valid_mask,
                basin_label=basin_label,
            )
        return out


def _channel_group_importance(
    query_z: torch.Tensor,
    channel_gate: torch.Tensor | None,
    *,
    group_size: int,
) -> torch.Tensor:
    if query_z.ndim != 4:
        raise ValueError("query_z must have shape (B,C,H,W)")
    channels = int(query_z.shape[1])
    if channels % int(group_size) != 0:
        raise ValueError("query_z channels must be divisible by channel group size")
    num_groups = channels // int(group_size)
    if channel_gate is not None and channel_gate.ndim == 2 and int(channel_gate.shape[1]) == num_groups:
        return channel_gate.float().mean(dim=0)
    grouped = query_z.float().abs().reshape(query_z.shape[0], num_groups, int(group_size), *query_z.shape[-2:])
    return grouped.mean(dim=(0, 2, 3, 4))


def _channel_group_ablated_scores(
    scorer,
    query_z: torch.Tensor,
    render_z: torch.Tensor,
    *,
    group_size: int,
    query_utility: torch.Tensor | None,
    render_mask: torch.Tensor | None,
) -> torch.Tensor:
    channels = int(query_z.shape[1])
    if channels % int(group_size) != 0:
        raise ValueError("query_z channels must be divisible by channel group size")
    num_groups = channels // int(group_size)
    rows = []
    for group_idx in range(num_groups):
        mask = query_z.new_ones((1, channels, 1, 1))
        start = group_idx * int(group_size)
        mask[:, start : start + int(group_size)] = 0.0
        masked_query = query_z * mask
        masked_render = render_z * mask[:, None]
        scores, _aux = scorer(
            masked_query,
            masked_render,
            query_utility=query_utility,
            render_valid_mask=render_mask,
        )
        rows.append(scores)
    return torch.stack(rows, dim=0)


def _mean_nested_reports(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No audit rows to summarize")
    out: dict[str, object] = {"num_batches": int(len(rows))}
    for section in ("base", "drop_high", "drop_low"):
        section_rows = [dict(row[section]) for row in rows]  # type: ignore[index]
        out[section] = {
            key: float(sum(float(row[key]) for row in section_rows if key in row) / max(sum(1 for row in section_rows if key in row), 1))
            for key in METRIC_KEYS
            if any(key in row for row in section_rows)
        }
    for key in ("drop_high_cost_delta_m", "drop_low_cost_delta_m"):
        out[key] = float(sum(float(row[key]) for row in rows) / len(rows))  # type: ignore[index]
    return out


def _mean_feature_shuffle_reports(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No feature-shuffle rows to summarize")
    out: dict[str, object] = {"num_batches": int(len(rows))}
    sections = ("base", "candidate_render_shuffle", "query_batch_shuffle", "wrong_scene_render_shuffle")
    for section in sections:
        section_rows = [dict(row[section]) for row in rows if section in row]  # type: ignore[index]
        if not section_rows:
            continue
        if all(row.get("skipped") for row in section_rows):
            out[section] = dict(section_rows[0])
            continue
        metric_rows = [row for row in section_rows if not row.get("skipped")]
        out[section] = {
            key: float(sum(float(row[key]) for row in metric_rows if key in row) / max(sum(1 for row in metric_rows if key in row), 1))
            for key in METRIC_KEYS
            if any(key in row for row in metric_rows)
        }
    return out


def _load_selector(path: str | Path, *, in_channels: int, out_channels: int, group_size: int, device: torch.device) -> LocalizationFeatureSelector:
    selector = LocalizationFeatureSelector(
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        group_size=int(group_size),
    ).to(device)
    selector.load_state_dict(_extract_selector_state(_safe_torch_load(path)))
    selector.eval()
    for param in selector.parameters():
        param.requires_grad_(False)
    return selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--pose-adapter-checkpoint", default=None)
    parser.add_argument("--pose-adapter-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--pose-candidate-cache", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--candidate-render-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--query-feature-source", choices=("raw_radio", "query_student", "projected_query_student"), default="query_student")
    parser.add_argument("--query-feature-key", default="fine")
    parser.add_argument("--selector-out-dim", type=int, default=64)
    parser.add_argument("--selector-group-size", type=int, default=8)
    parser.add_argument("--score-mode", choices=("same_pixel", "local_corr", "pair_matcher_local"), default="pair_matcher_local")
    parser.add_argument("--score-radius", type=int, default=16)
    parser.add_argument("--score-temperature", type=float, default=0.04)
    parser.add_argument("--score-feature-hw", default="34,60")
    parser.add_argument("--pair-matcher-stride", type=int, default=4)
    parser.add_argument("--pair-matcher-chunk-points", type=int, default=256)
    parser.add_argument("--pair-matcher-candidate-chunk-size", type=int, default=1)
    parser.add_argument("--pair-matcher-offset-chunk-size", type=int, default=32)
    parser.add_argument("--pair-matcher-candidate-score-mode", default="center_logprob_margin")
    parser.add_argument("--pair-matcher-score-channel", type=int, default=0)
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--drop-fraction", type=float, default=0.2)
    parser.add_argument("--channel-counterfactual", action="store_true")
    parser.add_argument("--channel-ablation-group-size", type=int, default=8)
    parser.add_argument(
        "--channel-importance-mode",
        choices=("provided", "leave_one_group_out"),
        default="leave_one_group_out",
        help="How to rank high/low channel groups for the channel counterfactual table.",
    )
    parser.add_argument("--feature-shuffle-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    cfg = load_config(args.config)
    cfg = _apply_cache_override(cfg, split=args.split, cache=args.pose_candidate_cache)
    loader_args = _loader_args(args, split=args.split, max_samples=args.max_samples, cache=args.pose_candidate_cache)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, loader, map_renderer = build_model_and_data(cfg, loader_args, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    in_channels = int(getattr(model, "fine_feature_dim", 64))
    selector = _load_selector(
        args.selector_checkpoint,
        in_channels=in_channels,
        out_channels=int(args.selector_out_dim),
        group_size=int(args.selector_group_size),
        device=device,
    )
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
    scorer.eval()
    score_hw = _parse_hw(args.score_feature_hw)
    spatial_reports = []
    channel_reports = []
    feature_shuffle_reports = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            scores, pose_cost, trans_err, rot_err, valid, aux = _forward_scores(
                args,
                model=model,
                map_renderer=map_renderer,
                selector=selector,
                scorer=scorer,
                batch=batch,
                device=device,
                score_hw=score_hw,
                return_selected_features=bool(args.channel_counterfactual or args.feature_shuffle_controls),
            )
            score_maps = aux["score_maps"]
            if score_maps.ndim == 5:
                score_maps = score_maps[:, :, int(args.pair_matcher_score_channel)]
            basin = (trans_err <= float(args.basin_trans_m)) & (rot_err <= float(args.basin_rot_deg))
            spatial_reports.append(
                spatial_counterfactual_report(
                    score_maps,
                    aux["query_utility"],
                    pose_cost,
                    valid_mask=valid,
                    basin_label=basin,
                    drop_fraction=float(args.drop_fraction),
                    base_weight=aux["query_utility"],
                )
            )
            if args.feature_shuffle_controls:
                feature_shuffle_reports.append(
                    feature_shuffle_control_report(
                        scorer,
                        aux["query_z"],
                        aux["render_z"],
                        pose_cost,
                        valid_mask=valid,
                        basin_label=basin,
                        query_utility=aux.get("query_utility"),
                        render_mask=aux.get("render_mask"),
                    )
                )
            if args.channel_counterfactual:
                group_importance = _channel_group_importance(
                    aux["query_z"],
                    aux.get("query_channel_gate"),
                    group_size=int(args.channel_ablation_group_size),
                )
                ablated_group_scores = _channel_group_ablated_scores(
                    scorer,
                    aux["query_z"],
                    aux["render_z"],
                    group_size=int(args.channel_ablation_group_size),
                    query_utility=aux.get("query_utility"),
                    render_mask=aux.get("render_mask"),
                )
                channel_reports.append(
                    channel_counterfactual_report(
                        scores,
                        ablated_group_scores,
                        pose_cost,
                        group_importance,
                        valid_mask=valid,
                        basin_label=basin,
                        importance_mode=str(args.channel_importance_mode),
                    )
                )
    summary = {
        "selector_checkpoint": str(args.selector_checkpoint),
        "pose_candidate_cache": str(args.pose_candidate_cache),
        "drop_fraction": float(args.drop_fraction),
        "spatial_counterfactual": _mean_nested_reports(spatial_reports),
    }
    if channel_reports:
        summary["channel_counterfactual"] = _mean_nested_reports(channel_reports)
    if feature_shuffle_reports:
        summary["feature_shuffle_controls"] = _mean_feature_shuffle_reports(feature_shuffle_reports)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
