#!/usr/bin/env python3
"""Streaming POFD-FS feature-score audit without saving candidate feature tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.metrics import (  # noqa: E402
    candidate_score_table_rows,
    ranking_metrics,
    ranking_row_diagnostics,
)
from feature_extract.localizability.scorer import PoseHypothesisScorer  # noqa: E402
from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher, PoseFeatureDomainAdapter  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.train_nvs_pose_feature_adapter import load_adapter_checkpoint, project_render_bank  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    project_query_render_for_fine_selector,
)

OPTIONAL_CANDIDATE_TABLE_FIELDS = (
    "retrieval_scores_candidates",
    "retrieval_original_scores_candidates",
    "retrieval_pnp_success_candidates",
    "retrieval_pnp_num_inliers_candidates",
    "retrieval_pnp_num_matches_candidates",
    "retrieval_pnp_reproj_rmse_candidates",
    "retrieval_pnp_reproj_median_candidates",
    "retrieval_pnp_inlier_ratio_candidates",
    "retrieval_pnp_inlier_conf_mean_candidates",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--pose-candidate-cache", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--candidate-render-batch-size", type=int, default=4)
    parser.add_argument(
        "--query-feature-source",
        choices=("raw_radio", "query_student", "projected_query_student", "pose_adapter"),
        default="raw_radio",
    )
    parser.add_argument("--query-feature-key", default="fine")
    parser.add_argument("--pose-adapter-checkpoint", default=None)
    parser.add_argument("--pose-adapter-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-mode", choices=("same_pixel", "local_corr", "pair_matcher_local"), default="local_corr")
    parser.add_argument("--score-radius", type=int, default=12)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--pair-matcher-stride", type=int, default=4)
    parser.add_argument("--pair-matcher-chunk-points", type=int, default=1024)
    parser.add_argument("--pair-matcher-candidate-chunk-size", type=int, default=4)
    parser.add_argument("--pair-matcher-offset-chunk-size", type=int, default=0)
    parser.add_argument("--pair-matcher-candidate-score-mode", default="center_logprob_margin")
    parser.add_argument("--pair-matcher-score-channel", type=int, default=0)
    parser.add_argument("--score-feature-hw", default="34,60", help="H,W score resolution; empty keeps render feature size")
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--topk", default="1,5")
    parser.add_argument("--dump-rows", action="store_true")
    parser.add_argument("--dump-candidate-table", action="store_true")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _parse_hw(value: str | None) -> tuple[int, int] | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [int(part) for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2 or min(parts) <= 0:
        raise ValueError("--score-feature-hw must be H,W")
    return int(parts[0]), int(parts[1])


def _resize_feature(feature: torch.Tensor, hw: tuple[int, int] | None) -> torch.Tensor:
    if hw is None or tuple(feature.shape[-2:]) == tuple(hw):
        return feature
    mode = "bilinear"
    if feature.ndim == 5:
        bsz, num, channels, height, width = feature.shape
        flat = feature.reshape(bsz * num, channels, height, width)
        resized = F.interpolate(flat.float(), size=hw, mode=mode, align_corners=False)
        return resized.reshape(bsz, num, channels, hw[0], hw[1])
    return F.interpolate(feature.float(), size=hw, mode=mode, align_corners=False)


def _resize_mask(mask: torch.Tensor | None, hw: tuple[int, int] | None) -> torch.Tensor | None:
    if mask is None or hw is None or tuple(mask.shape[-2:]) == tuple(hw):
        return mask
    if mask.ndim == 5:
        bsz, num = mask.shape[:2]
        flat = mask.reshape(bsz * num, *mask.shape[2:])
        resized = F.interpolate(flat.float(), size=hw, mode="nearest")
        return resized.reshape(bsz, num, *resized.shape[1:])
    return F.interpolate(mask.float(), size=hw, mode="nearest")


def _mean_metric_dict(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        out[key] = float(torch.stack([row[key].detach().cpu() for row in rows]).mean())
    return out


def _checkpoint_args(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    return dict(checkpoint.get("args") or {})


def _load_pose_adapter_bundle(
    path: str | Path,
    *,
    channels: int,
    device: torch.device,
    strict_adapter: bool,
) -> tuple[PoseFeatureDomainAdapter, PairConditionedLocalMatcher, dict]:
    ck_args = _checkpoint_args(path)
    adapter = PoseFeatureDomainAdapter(
        channels=int(channels),
        hidden_dim=int(ck_args.get("pose_feature_adapter_hidden_dim", 128)),
        residual_scale=float(ck_args.get("pose_feature_adapter_residual_scale", 0.2)),
        zero_init=bool(ck_args.get("pose_feature_adapter_zero_init", True)),
        l2_normalize=bool(ck_args.get("pose_feature_adapter_l2_normalize", True)),
        uncertainty_enabled=bool(ck_args.get("pose_feature_adapter_uncertainty_enabled", False)),
        rgb_context_enabled=bool(ck_args.get("pose_feature_adapter_rgb_context_enabled", False)),
        rgb_context_channels=int(ck_args.get("pose_feature_adapter_rgb_context_channels", 16)),
        texture_branch_enabled=bool(ck_args.get("pose_feature_adapter_texture_branch_enabled", False)),
        texture_branch_hidden_dim=int(ck_args.get("pose_feature_adapter_texture_branch_hidden_dim", 96)),
        texture_branch_scale=float(ck_args.get("pose_feature_adapter_texture_branch_scale", 1.0)),
        texture_branch_zero_init=bool(ck_args.get("pose_feature_adapter_texture_branch_zero_init", False)),
        texture_fusion_mode=str(ck_args.get("pose_feature_adapter_texture_fusion_mode", "replace")),
        base_anchor_weight=float(ck_args.get("pose_feature_adapter_base_anchor_weight", 0.25)),
    ).to(device)
    pair_matcher = PairConditionedLocalMatcher(
        channels=int(channels),
        hidden_dim=int(ck_args.get("pair_matcher_hidden_dim", 128)),
        offset_radius=int(ck_args.get("pair_matcher_radius", 12)),
        zero_init_residual=bool(ck_args.get("pair_matcher_zero_init_residual", True)),
        base_dot_weight=float(ck_args.get("pair_matcher_base_dot_weight", 1.0)),
    ).to(device)
    load_adapter_checkpoint(
        str(path),
        adapter,
        pair_matcher=pair_matcher,
        strict_adapter=bool(strict_adapter),
    )
    adapter.eval()
    pair_matcher.eval()
    for module in (adapter, pair_matcher):
        for param in module.parameters():
            param.requires_grad_(False)
    return adapter, pair_matcher, ck_args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.pose_candidate_cache:
        cfg.setdefault("dataset", {})[f"{args.split}_pose_candidate_cache"] = str(args.pose_candidate_cache)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, loader, map_renderer = build_model_and_data(cfg, args, device)
    model.eval()
    adapter = None
    pair_matcher = None
    adapter_args = {}
    if args.pose_adapter_checkpoint:
        adapter, pair_matcher, adapter_args = _load_pose_adapter_bundle(
            args.pose_adapter_checkpoint,
            channels=int(getattr(model, "fine_feature_dim", 64)),
            device=device,
            strict_adapter=bool(args.pose_adapter_strict),
        )
        if int(args.score_radius) == 12 and int(adapter_args.get("pair_matcher_radius", 12)) != 12:
            pass
    if args.query_feature_source == "pose_adapter" and adapter is None:
        raise ValueError("--query-feature-source=pose_adapter requires --pose-adapter-checkpoint")
    if args.score_mode == "pair_matcher_local" and pair_matcher is None:
        raise ValueError("--score-mode=pair_matcher_local requires --pose-adapter-checkpoint with pair_matcher_state_dict")
    scorer = PoseHypothesisScorer(
        mode=args.score_mode,
        radius=args.score_radius,
        temperature=args.score_temperature,
        pair_matcher=pair_matcher,
        pair_matcher_stride=args.pair_matcher_stride,
        pair_matcher_chunk_points=args.pair_matcher_chunk_points,
        pair_matcher_candidate_chunk_size=args.pair_matcher_candidate_chunk_size,
        pair_matcher_offset_chunk_size=args.pair_matcher_offset_chunk_size,
        pair_matcher_candidate_score_mode=args.pair_matcher_candidate_score_mode,
        pair_matcher_score_channel=args.pair_matcher_score_channel,
    ).to(device)
    score_hw = _parse_hw(args.score_feature_hw)
    topk = tuple(int(part) for part in str(args.topk).split(",") if part.strip())
    rows = []
    row_diagnostics = []
    candidate_table = []
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if "pose_init_candidates" not in batch:
                raise KeyError("Batch does not include pose_init_candidates; pass --pose-candidate-cache or set config cache")
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
            if args.query_feature_source == "raw_radio":
                query_feature = batch["teacher_fine"].float()
                render_feature = cand_batch["localizability_candidate_fine"].float()
            elif args.query_feature_source == "query_student":
                outputs = model(batch["rgb"])
                if args.query_feature_key not in outputs:
                    raise KeyError(f"query feature key {args.query_feature_key!r} not found in model outputs")
                query_feature = outputs[args.query_feature_key].float()
                render_feature = cand_batch["localizability_candidate_fine"].float()
            elif args.query_feature_source == "pose_adapter":
                outputs = model(batch["rgb"])
                if args.query_feature_key not in outputs:
                    raise KeyError(f"query feature key {args.query_feature_key!r} not found in model outputs")
                adapter_needs_rgb = bool(adapter_args.get("pose_feature_adapter_rgb_context_enabled", False)) or bool(
                    adapter_args.get("pose_feature_adapter_texture_branch_enabled", False)
                )
                query_rgb = batch.get("rgb") if adapter_needs_rgb else None
                render_rgb = cand_batch.get("localizability_candidate_rgb") if adapter_needs_rgb else None
                query_feature = adapter.project_query(outputs[args.query_feature_key].float(), rgb=query_rgb)
                render_feature = project_render_bank(
                    adapter,
                    cand_batch["localizability_candidate_fine"].float(),
                    render_rgb=render_rgb,
                    render_chunk_size=16,
                )
            else:
                outputs = model(batch["rgb"])
                if args.query_feature_key not in outputs:
                    raise KeyError(f"query feature key {args.query_feature_key!r} not found in model outputs")
                query_feature, render_feature, used_projector = project_query_render_for_fine_selector(
                    getattr(model, "local_corr_projector", None),
                    outputs[args.query_feature_key].float(),
                    cand_batch["localizability_candidate_fine"].float(),
                    require_projector=True,
                    render_chunk_size=16,
                )
                if not used_projector:
                    raise RuntimeError("projected_query_student requested but no projector was used")
            render_mask = cand_batch.get("localizability_candidate_mask")
            query_feature = _resize_feature(query_feature, score_hw)
            render_feature = _resize_feature(render_feature, score_hw)
            render_mask = _resize_mask(render_mask, score_hw)
            scores, _aux = scorer(query_feature, render_feature, render_valid_mask=render_mask)

            bsz, num_candidates = candidates.shape[:2]
            pose_gt_flat = pose_gt[:, None].expand(-1, num_candidates, -1, -1).reshape(-1, 4, 4)
            _, rot_err_deg, trans_err_m = pose_error_tensors(candidates.reshape(-1, 4, 4), pose_gt_flat)
            trans_err_m = trans_err_m.reshape(bsz, num_candidates)
            rot_err_deg = rot_err_deg.reshape(bsz, num_candidates)
            pose_cost = trans_err_m + float(args.rot_cost_weight) * torch.deg2rad(rot_err_deg)
            extra_candidate_fields = {}
            if "pose_init" in batch:
                pose_init = batch["pose_init"].float()
                pose_init_flat = pose_init[:, None].expand(-1, num_candidates, -1, -1).reshape(-1, 4, 4)
                _, delta_rot_deg, delta_trans_m = pose_error_tensors(candidates.reshape(-1, 4, 4), pose_init_flat)
                extra_candidate_fields["delta_trans_m"] = delta_trans_m.reshape(bsz, num_candidates)
                extra_candidate_fields["delta_rot_deg"] = delta_rot_deg.reshape(bsz, num_candidates)
            for key in OPTIONAL_CANDIDATE_TABLE_FIELDS:
                if key in batch and tuple(batch[key].shape[:2]) == (bsz, num_candidates):
                    extra_candidate_fields[key] = batch[key].float()
            basin = (trans_err_m <= float(args.basin_trans_m)) & (rot_err_deg <= float(args.basin_rot_deg))
            valid = batch.get("candidate_valid_mask")
            metrics = ranking_metrics(scores, pose_cost, valid_mask=valid, basin_label=basin, topk=topk)
            rows.append(metrics)
            names = [str(name) for name in batch.get("sample_name", [str(i) for i in range(bsz)])]
            if bool(args.dump_candidate_table):
                for item in candidate_score_table_rows(
                    scores,
                    pose_cost,
                    trans_err_m=trans_err_m,
                    rot_err_deg=rot_err_deg,
                    valid_mask=valid,
                    basin_label=basin,
                    sample_names=names,
                    extra_fields=extra_candidate_fields,
                ):
                    item["global_row"] = int(sample_count + int(item["row"]))
                    candidate_table.append(item)
            if bool(args.dump_rows):
                diag = ranking_row_diagnostics(scores, pose_cost, valid_mask=valid, basin_label=basin, sample_names=names)
                for item in diag:
                    local_row = int(item["row"])
                    selected = int(item["selected_idx"])
                    oracle = int(item["oracle_idx"])
                    item["global_row"] = int(sample_count + local_row)
                    item["selected_trans_m"] = float(trans_err_m[local_row, selected].detach().cpu())
                    item["oracle_trans_m"] = float(trans_err_m[local_row, oracle].detach().cpu())
                    item["selected_rot_deg"] = float(rot_err_deg[local_row, selected].detach().cpu())
                    item["oracle_rot_deg"] = float(rot_err_deg[local_row, oracle].detach().cpu())
                    row_diagnostics.append(item)
            sample_count += bsz

    row = {
        "split": "eval",
        "num_samples": int(sample_count),
        "query_feature_source": args.query_feature_source,
        "score_mode": args.score_mode,
        "score_radius": int(args.score_radius),
        "score_feature_hw": list(score_hw) if score_hw is not None else None,
        "pair_matcher_score_channel": int(args.pair_matcher_score_channel),
    }
    row.update(_mean_metric_dict(rows))
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    if bool(args.dump_rows):
        with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for item in row_diagnostics:
                handle.write(json.dumps(item) + "\n")
    if bool(args.dump_candidate_table):
        with (out_dir / "candidate_table.jsonl").open("w", encoding="utf-8") as handle:
            for item in candidate_table:
                handle.write(json.dumps(item) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
