#!/usr/bin/env python3
"""Evaluate projected selected 3D track-bank hypothesis scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.rendered_map_scoring import (  # noqa: E402
    densify_projected_feature_maps,
    render_selected_track_feature_maps,
)
from feature_extract.localizability.selected_feature_map import (  # noqa: E402
    SelectedTrackFeatureBank,
    load_selected_track_feature_bank,
)
from feature_extract.localizability.selector import LocalizationFeatureSelector  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.eval_feature_track_mapability import _extract_selector_state, _safe_torch_load  # noqa: E402
from feature_extract.tools.eval_localizability_feature_score import _load_pose_adapter_bundle, _parse_hw  # noqa: E402
from feature_extract.tools.train_localizability_selector_stream import (  # noqa: E402
    _apply_cache_override,
    _build_selector_stream_scorer,
    _loader_args,
    _pose_costs,
    _query_feature,
)
from feature_extract.train_impl import load_config, move_batch_to_device, set_seed  # noqa: E402


def intrinsics_dicts_to_scaled_k(
    intrinsics: Sequence[Mapping[str, float]],
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Convert `{fx,fy,cx,cy}` intrinsics to target-grid 3x3 matrices."""

    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("source_hw and target_hw must be positive")
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    rows = []
    for intr in intrinsics:
        rows.append(
            torch.tensor(
                [
                    [float(intr["fx"]) * sx, 0.0, float(intr["cx"]) * sx],
                    [0.0, float(intr["fy"]) * sy, float(intr["cy"]) * sy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    return torch.stack(rows, dim=0)


def intrinsics_for_batch(map_renderer, batch: Mapping[str, object], *, target_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Fetch per-query intrinsics from the map renderer and scale to target grid."""

    names = [str(name) for name in batch["sample_name"]]  # type: ignore[index]
    intrinsics = []
    for name in names:
        normalized = map_renderer._normalize_name(name)
        intrinsics.append(map_renderer.name_to_intr[normalized])
    rgb = batch.get("rgb")
    if not torch.is_tensor(rgb):
        raise KeyError("batch must contain rgb tensor to infer source_hw")
    return intrinsics_dicts_to_scaled_k(
        intrinsics,
        source_hw=tuple(int(v) for v in rgb.shape[-2:]),
        target_hw=target_hw,
        device=device,
    )


def _resize_feature(feature: torch.Tensor, hw: tuple[int, int] | None) -> torch.Tensor:
    if hw is None or tuple(feature.shape[-2:]) == tuple(hw):
        return feature
    return F.interpolate(feature.float(), size=hw, mode="bilinear", align_corners=False)


def score_selected_query_against_projected_bank(
    query_feature: torch.Tensor,
    bank: SelectedTrackFeatureBank,
    candidate_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    selector: torch.nn.Module,
    scorer: torch.nn.Module,
    splat_radius: int = 0,
    densify_radius: int = 0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Score query selected features against an already-selected projected bank."""

    rendered, render_valid = render_selected_track_feature_maps(
        bank,
        candidate_w2c,
        intrinsics,
        image_hw=image_hw,
        splat_radius=int(splat_radius),
    )
    rendered, render_valid = densify_projected_feature_maps(rendered, render_valid, radius=int(densify_radius))
    rendered = rendered.to(device=query_feature.device, dtype=query_feature.dtype)
    render_valid = render_valid.to(device=query_feature.device)
    q_out = selector(query_feature)
    scores, scorer_aux = scorer(
        q_out["z"],
        rendered,
        query_utility=q_out.get("utility"),
        render_valid_mask=render_valid,
    )
    return scores, {
        "query_selected": q_out,
        "scorer": scorer_aux,
        "render_valid_mask": render_valid,
        "rendered_selected_map_feature": rendered,
    }


def _metrics_to_float(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {str(key): float(value.detach().cpu()) for key, value in metrics.items()}


def _mean_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No rows to summarize")
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(float(row[key]) for row in rows if key in row) / max(sum(1 for row in rows if key in row), 1))
        for key in keys
    }


def projected_valid_coverage(render_valid: torch.Tensor) -> dict[str, float]:
    """Summarize projected map coverage from a boolean render-valid mask."""

    valid_f = render_valid.float()
    return {
        "projected_valid_pixel_frac": float(valid_f.mean().detach().cpu()),
        "projected_valid_pixels_per_candidate": float(valid_f.flatten(2).sum(dim=-1).mean().detach().cpu()),
    }


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
    parser.add_argument("--pose-candidate-cache", required=True)
    parser.add_argument("--selected-bank", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--pose-adapter-checkpoint", default=None)
    parser.add_argument("--pose-adapter-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--candidate-render-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--query-feature-source", choices=("raw_radio", "query_student", "projected_query_student"), default="query_student")
    parser.add_argument("--query-feature-key", default="fine")
    parser.add_argument("--selector-out-dim", type=int, default=64)
    parser.add_argument("--selector-group-size", type=int, default=8)
    parser.add_argument("--score-mode", choices=("same_pixel", "local_corr", "pair_matcher_local"), default="local_corr")
    parser.add_argument("--score-radius", type=int, default=4)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--score-feature-hw", default="34,60")
    parser.add_argument("--splat-radius", type=int, default=0)
    parser.add_argument("--densify-radius", type=int, default=0)
    parser.add_argument("--pair-matcher-stride", type=int, default=4)
    parser.add_argument("--pair-matcher-chunk-points", type=int, default=256)
    parser.add_argument("--pair-matcher-candidate-chunk-size", type=int, default=1)
    parser.add_argument("--pair-matcher-offset-chunk-size", type=int, default=32)
    parser.add_argument("--pair-matcher-candidate-score-mode", default="center_logprob_margin")
    parser.add_argument("--pair-matcher-score-channel", type=int, default=0)
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    cfg = load_config(args.config)
    cfg = _apply_cache_override(cfg, split=args.split, cache=args.pose_candidate_cache)
    loader_args = _loader_args(args, split=args.split, max_samples=args.max_samples, cache=args.pose_candidate_cache)
    model, loader, map_renderer = build_model_and_data(cfg, loader_args, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    bank, bank_metadata = load_selected_track_feature_bank(args.selected_bank)
    if bank.xyz is None:
        raise ValueError("--selected-bank must contain xyz for projected-map retention")
    bank = SelectedTrackFeatureBank(
        track_ids=bank.track_ids.to(device),
        features=bank.features.to(device),
        visibility_count=bank.visibility_count.to(device),
        feature_variance=bank.feature_variance.to(device),
        utility_mean=bank.utility_mean.to(device),
        xyz=bank.xyz.to(device),
    )
    selector = _load_selector(
        args.selector_checkpoint,
        in_channels=int(bank.features.shape[1]),
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
    if score_hw is None:
        raise ValueError("--score-feature-hw is required for projected bank scoring")

    metric_rows = []
    coverage_rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            query = _resize_feature(_query_feature(args, model, batch), score_hw)
            intrinsics = intrinsics_for_batch(map_renderer, batch, target_hw=score_hw, device=device)
            candidates = batch["pose_init_candidates"].float()
            scores, aux = score_selected_query_against_projected_bank(
                query,
                bank,
                candidates,
                intrinsics,
                image_hw=score_hw,
                selector=selector,
                scorer=scorer,
                splat_radius=int(args.splat_radius),
                densify_radius=int(args.densify_radius),
            )
            pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
            pose_cost, trans_err_m, rot_err_deg = _pose_costs(args, candidates, pose_gt)
            valid = batch.get("candidate_valid_mask")
            basin = (trans_err_m <= float(args.basin_trans_m)) & (rot_err_deg <= float(args.basin_rot_deg))
            topk = (1, 5) if scores.shape[1] >= 5 else (1,)
            metric_rows.append(
                _metrics_to_float(
                    ranking_metrics(
                        scores,
                        pose_cost,
                        valid_mask=valid,
                        basin_label=basin,
                        topk=topk,
                    )
                )
            )
            render_valid = aux["render_valid_mask"]
            coverage_rows.append(projected_valid_coverage(render_valid))
    summary = {
        "selected_bank": str(args.selected_bank),
        "selected_bank_metadata": bank_metadata,
        "selected_bank_tracks": int(bank.features.shape[0]),
        "selected_bank_feature_dim": int(bank.features.shape[1]),
        "pose_candidate_cache": str(args.pose_candidate_cache),
        "score_feature_hw": [int(score_hw[0]), int(score_hw[1])],
        "splat_radius": int(args.splat_radius),
        "densify_radius": int(args.densify_radius),
        "metrics": _mean_metric_rows(metric_rows),
        "coverage": _mean_metric_rows(coverage_rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
