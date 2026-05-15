#!/usr/bin/env python3
"""Export compact fine top-K selector evidence for offline selector training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.tools.eval_cpr_buckets import (  # noqa: E402
    build_model_and_data,
    default_lattice,
    exact_inverse_candidate_mask,
    fine_candidate_selector_features,
    gather_pose_bank,
    jitter_candidate_poses,
    load_pose_feature_adapter_bundle,
    make_fixed_init_poses,
    make_random_init_poses,
    parse_bucket_specs,
    parse_float_csv,
    pose_error_tensors,
    project_query_render_with_pose_feature_adapter,
    project_query_render_for_fine_selector,
    select_fine_pool_indices,
    selected_pose_error_dict,
)
from feature_extract.train_impl import (  # noqa: E402
    build_local_pose_lattice_candidates,
    candidate_score_fusion_listwise_loss,
    load_config,
    move_batch_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument(
        "--fine-selector-adapter-checkpoint",
        default=None,
        help="Optional NVS pose-feature adapter checkpoint used before cache feature extraction.",
    )
    parser.add_argument(
        "--fine-selector-score-source",
        choices=("local_corr", "pair_matcher_heatmap"),
        default="local_corr",
        help="Score-map source stored in the cache.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--buckets", nargs="+", default=["025cm_5deg:25,5"])
    parser.add_argument("--fine-topk", type=int, default=8)
    parser.add_argument(
        "--fine-pool-mode",
        choices=("rank", "rank_uniform", "rank_delta_uniform", "rank_score_uniform", "rank_pose_hard"),
        default="rank",
        help="How to form the fine selector candidate pool before export.",
    )
    parser.add_argument(
        "--fine-pool-topm",
        type=int,
        default=16,
        help="Number of scorer-ranked candidates to force-keep in non-rank fine pool modes.",
    )
    parser.add_argument("--topk", default="1,4,8", help="Unused compatibility option for build_model_and_data")
    parser.add_argument("--lattice-trans-cm", default=None)
    parser.add_argument("--lattice-rot-deg", default=None)
    parser.add_argument("--lattice-direction-mode", choices=("axis", "cube"), default=None)
    parser.add_argument("--combine-trans-rot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--limit-strategy", default="uniform")
    parser.add_argument("--candidate-render-batch-size", type=int, default=32)
    parser.add_argument("--init-noise-mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--init-jitter-seed", type=int, default=13)
    parser.add_argument("--candidate-jitter-cm", type=float, default=0.0)
    parser.add_argument("--candidate-jitter-deg", type=float, default=0.0)
    parser.add_argument("--disable-exact-inverse", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--include-score-maps", action="store_true")
    return parser.parse_args()


def _pose_gt_for_batch(map_renderer, batch, device):
    poses = []
    for sample_name in batch["sample_name"]:
        normalized = map_renderer._normalize_name(sample_name)
        poses.append(map_renderer.name_to_pose[normalized].to(device=device, dtype=torch.float32))
    return torch.stack(poses, dim=0)


@torch.no_grad()
def export_cache(args: argparse.Namespace):
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, loader, map_renderer = build_model_and_data(cfg, args, device)
    map_cfg = cfg.get("map_supervision", {})
    fine_selector_adapter_bundle = None
    if args.fine_selector_adapter_checkpoint:
        fine_selector_adapter_bundle = load_pose_feature_adapter_bundle(
            args.fine_selector_adapter_checkpoint,
            model,
            cfg,
            device,
        )
    rows = {
        "features": [],
        "valid": [],
        "trans_err_m": [],
        "rot_err_deg": [],
        "selected_pose": [],
        "init_pose": [],
        "pose_gt": [],
        "selected_coarse_logits": [],
    }
    if args.include_score_maps:
        rows["score_maps"] = []
    sample_names = []
    bucket_names = []

    for bucket_name, trans_cm, rot_deg in parse_bucket_specs(args.buckets):
        lattice_trans, lattice_rot = default_lattice(trans_cm, rot_deg)
        lattice_trans = parse_float_csv(args.lattice_trans_cm) or lattice_trans
        lattice_rot = parse_float_csv(args.lattice_rot_deg) or lattice_rot
        if args.max_candidates <= 0:
            max_candidates = int(map_cfg.get("coarse_pose_lattice_max_candidates", 0) or 0)
        else:
            max_candidates = int(args.max_candidates)
        combine_trans_rot = (
            bool(args.combine_trans_rot)
            if args.combine_trans_rot is not None
            else bool(map_cfg.get("coarse_pose_lattice_combine_trans_rot", True))
        )
        direction_mode = args.lattice_direction_mode or map_cfg.get("coarse_pose_lattice_direction_mode", "axis")

        for batch_offset, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            pose_gt = _pose_gt_for_batch(map_renderer, batch, device)
            current_offset = batch_offset * int(loader.batch_size or 1)
            if args.init_noise_mode == "random":
                init_pose = make_random_init_poses(
                    pose_gt,
                    trans_cm,
                    rot_deg,
                    seed=int(args.init_jitter_seed),
                    offset=current_offset,
                )
            else:
                init_pose = make_fixed_init_poses(pose_gt, trans_cm, rot_deg, offset=current_offset)
            candidate_poses = build_local_pose_lattice_candidates(
                init_pose,
                trans_cm=lattice_trans,
                rot_deg=lattice_rot,
                include_identity=True,
                max_candidates=max_candidates,
                limit_strategy=args.limit_strategy,
                combine_trans_rot=combine_trans_rot,
                direction_mode=direction_mode,
            )
            candidate_poses = jitter_candidate_poses(
                candidate_poses,
                trans_cm=float(args.candidate_jitter_cm),
                rot_deg=float(args.candidate_jitter_deg),
                seed=int(args.init_jitter_seed),
                offset=current_offset,
            )
            valid = torch.ones(candidate_poses.shape[:2], device=device, dtype=torch.bool)
            if args.disable_exact_inverse:
                valid = valid & ~exact_inverse_candidate_mask(candidate_poses, pose_gt)
                empty_rows = ~valid.any(dim=1)
                if empty_rows.any():
                    valid[empty_rows, 0] = True

            outputs = model(batch["rgb"])
            coarse_batch = map_renderer.attach_pose_candidate_renders(
                dict(batch),
                candidate_poses,
                prefix="cache_candidate",
                candidate_valid_mask=valid,
                feature="coarse",
                include_aux=False,
            )
            candidate_coarse = coarse_batch["cache_candidate_coarse"].float()
            adapter = getattr(model, "candidate_basin_adapter", None)
            if adapter is not None:
                candidate_coarse = adapter(candidate_coarse).float()
            _loss, _metrics, details = candidate_score_fusion_listwise_loss(
                outputs["coarse"].float(),
                candidate_coarse,
                coarse_batch["cache_candidate_pose"].float(),
                pose_gt.float(),
                model.candidate_score_fusion_head,
                batch=coarse_batch,
                mask=coarse_batch.get("cache_candidate_mask"),
                candidate_valid_mask=coarse_batch.get("cache_candidate_valid_mask"),
                mode=map_cfg.get("candidate_score_fusion_mode", "local"),
                temperature=float(map_cfg.get("candidate_score_fusion_temperature", 1.0)),
                radius=int(map_cfg.get("candidate_score_fusion_radius", 2)),
                preprocess=map_cfg.get("candidate_score_fusion_preprocess", "spatial_center"),
                highpass_kernel=int(map_cfg.get("candidate_score_fusion_highpass_kernel", 5)),
                score_map_mode=map_cfg.get("candidate_score_fusion_score_map_mode", "volume_plus_peak_offset"),
                rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
                target_mode=map_cfg.get("candidate_score_fusion_target_mode", "gt_pose_error_soft"),
                target_temperature_m=float(map_cfg.get("candidate_score_fusion_target_temperature_m", 0.25)),
                render_feature_mode=map_cfg.get("candidate_score_fusion_render_feature_mode", "basic"),
                cost_regression_weight=0.0,
                basin_trans_m=float(map_cfg.get("candidate_score_fusion_basin_trans_m", 0.25)),
                basin_rot_deg=float(map_cfg.get("candidate_score_fusion_basin_rot_deg", 5.0)),
                return_details=True,
            )
            logits = details.get("raw_logits", details["logits"]).masked_fill(~details["valid"].bool(), -1.0e6)
            fine_topk = max(1, min(int(args.fine_topk), logits.shape[1]))
            selected_idx = select_fine_pool_indices(
                logits,
                details["valid"].bool(),
                fine_topk,
                mode=str(args.fine_pool_mode),
                rank_topm=int(args.fine_pool_topm),
                candidate_pose=coarse_batch["cache_candidate_pose"].float(),
                init_pose=init_pose.float(),
                pose_gt=pose_gt.float(),
                rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
            )
            selected_pose = gather_pose_bank(coarse_batch["cache_candidate_pose"].float(), selected_idx)
            selected_valid = details["valid"].bool().gather(1, selected_idx)
            selected_coarse_logits = logits.gather(1, selected_idx)

            fine_batch = map_renderer.attach_pose_candidate_renders(
                dict(batch),
                selected_pose,
                prefix="cache_selected",
                candidate_valid_mask=selected_valid,
                feature="all",
                include_aux=True,
            )
            query_key = str(map_cfg.get("query_fine_key", "fine"))
            query_fine = outputs.get(query_key, outputs["fine"]).float()
            render_fine = fine_batch["cache_selected_fine"].float()
            query_uncertainty = None
            render_uncertainty = None
            if fine_selector_adapter_bundle is not None:
                nvs_args = fine_selector_adapter_bundle["args"]
                adapter_needs_rgb = bool(getattr(nvs_args, "pose_feature_adapter_rgb_context_enabled", False)) or bool(
                    getattr(nvs_args, "pose_feature_adapter_texture_branch_enabled", False)
                )
                query_fine, render_fine, query_uncertainty, render_uncertainty = (
                    project_query_render_with_pose_feature_adapter(
                        fine_selector_adapter_bundle["adapter"],
                        query_fine,
                        render_fine,
                        query_rgb=batch.get("rgb") if adapter_needs_rgb else None,
                        render_rgb=fine_batch.get("cache_selected_rgb") if adapter_needs_rgb else None,
                        use_uncertainty=bool(getattr(nvs_args, "pose_feature_adapter_uncertainty_enabled", False)),
                        render_chunk_size=int(map_cfg.get("fine_topk_selector_projector_chunk_size", 0) or 0),
                    )
                )
            elif bool(map_cfg.get("fine_topk_selector_use_projector", True)):
                query_fine, render_fine, _used = project_query_render_for_fine_selector(
                    getattr(model, "local_corr_projector", None),
                    query_fine,
                    render_fine,
                    require_projector=bool(map_cfg.get("fine_topk_selector_require_projector", False)),
                    render_chunk_size=int(map_cfg.get("fine_topk_selector_projector_chunk_size", 0) or 0),
                )
            feature_pack = fine_candidate_selector_features(
                query_fine.float(),
                render_fine.float(),
                fine_batch["cache_selected_pose"].float(),
                init_pose=init_pose.float(),
                coarse_logits=selected_coarse_logits.float(),
                query_rgb=batch.get("rgb"),
                candidate_rgb=fine_batch.get("cache_selected_rgb"),
                depth=fine_batch.get("cache_selected_depth"),
                mask=fine_batch.get("cache_selected_mask"),
                candidate_valid_mask=selected_valid,
                mode=map_cfg.get("fine_topk_selector_mode", "local"),
                radius=int(map_cfg.get("fine_topk_selector_radius") or map_cfg.get("query_corr_radius", 4)),
                preprocess=(
                    map_cfg.get("fine_topk_selector_preprocess")
                    or map_cfg.get(
                        "query_corr_feature_preprocess",
                        map_cfg.get("candidate_score_fusion_preprocess", "spatial_center"),
                    )
                ),
                highpass_kernel=int(
                    map_cfg.get("fine_topk_selector_highpass_kernel")
                    or map_cfg.get("query_corr_highpass_kernel", map_cfg.get("candidate_score_fusion_highpass_kernel", 5))
                ),
                score_map_mode=map_cfg.get("fine_topk_selector_score_map_mode", "peak_offset"),
                return_score_maps=bool(args.include_score_maps),
                use_coarse_logits=bool(map_cfg.get("fine_topk_selector_use_coarse_logits", True)),
                use_candidate_delta=bool(map_cfg.get("fine_topk_selector_use_candidate_delta", True)),
                use_delta_vector=bool(map_cfg.get("fine_topk_selector_use_delta_vector", False)),
                use_depth=bool(map_cfg.get("fine_topk_selector_use_depth", True)),
                use_mask=bool(map_cfg.get("fine_topk_selector_use_mask", True)),
                use_rgb=bool(map_cfg.get("fine_topk_selector_use_rgb", False)),
                query_uncertainty=query_uncertainty,
                candidate_uncertainty=render_uncertainty,
                use_uncertainty=bool(map_cfg.get("fine_topk_selector_use_uncertainty", False)),
            )
            score_source = str(args.fine_selector_score_source or "local_corr").lower()
            if score_source == "pair_matcher_heatmap":
                if fine_selector_adapter_bundle is None:
                    raise RuntimeError("--fine-selector-score-source pair_matcher_heatmap requires --fine-selector-adapter-checkpoint")
                pair_matcher = fine_selector_adapter_bundle.get("pair_matcher")
                if pair_matcher is None:
                    raise RuntimeError("fine selector adapter checkpoint does not contain a pair matcher")
                from feature_extract.tools.train_nvs_pose_feature_adapter import pair_matcher_local_candidate_score_maps

                nvs_args = fine_selector_adapter_bundle["args"]
                pair_score_maps, pair_valid_map = pair_matcher_local_candidate_score_maps(
                    pair_matcher,
                    query_fine.float(),
                    render_fine.float(),
                    mask=fine_batch.get("cache_selected_mask"),
                    radius=int(getattr(nvs_args, "pair_matcher_radius", 3)),
                    stride=int(getattr(nvs_args, "pair_matcher_score_stride", 8)),
                    temperature=float(getattr(nvs_args, "pair_matcher_temperature", 0.05)),
                    chunk_points=int(getattr(nvs_args, "pair_matcher_score_chunk_points", 65536)),
                    candidate_score_mode=str(
                        getattr(nvs_args, "pair_matcher_candidate_score_mode", "center_logprob_margin")
                    ),
                )
                feature_pack["score_maps"] = pair_score_maps
                feature_pack["valid"] = feature_pack["valid"] & pair_valid_map.flatten(2).any(dim=2)
            elif score_source != "local_corr":
                raise RuntimeError(f"Unknown fine selector score source: {score_source}")
            pose_errors = selected_pose_error_dict(
                fine_batch["cache_selected_pose"].float(),
                pose_gt.float(),
                rot_cost_weight=float(map_cfg.get("candidate_score_fusion_rot_cost_weight", 0.1)),
            )
            rows["features"].append(feature_pack["features"].detach().cpu())
            if args.include_score_maps:
                rows["score_maps"].append(feature_pack["score_maps"].detach().to(dtype=torch.float16).cpu())
            rows["valid"].append(feature_pack["valid"].detach().cpu())
            rows["trans_err_m"].append(pose_errors["trans_err_m"].detach().cpu())
            rows["rot_err_deg"].append(pose_errors["rot_err_deg"].detach().cpu())
            rows["selected_pose"].append(fine_batch["cache_selected_pose"].detach().cpu())
            rows["init_pose"].append(init_pose.detach().cpu())
            rows["pose_gt"].append(pose_gt.detach().cpu())
            rows["selected_coarse_logits"].append(selected_coarse_logits.detach().cpu())
            sample_names.extend([str(name) for name in batch["sample_name"]])
            bucket_names.extend([bucket_name] * pose_gt.shape[0])
            if int(args.progress_every) > 0 and (batch_offset + 1) % int(args.progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "bucket": bucket_name,
                            "batches": batch_offset + 1,
                            "rows": len(bucket_names),
                            "feature_dim": int(feature_pack["features"].shape[-1]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload = {key: torch.cat(value, dim=0) for key, value in rows.items()}
    payload["sample_names"] = sample_names
    payload["bucket_names"] = bucket_names
    payload["meta"] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "map_checkpoint": None if args.map_checkpoint is None else str(args.map_checkpoint),
        "fine_selector_adapter_checkpoint": (
            None if args.fine_selector_adapter_checkpoint is None else str(args.fine_selector_adapter_checkpoint)
        ),
        "fine_selector_score_source": str(args.fine_selector_score_source),
        "split": args.split,
        "fine_topk": int(args.fine_topk),
        "fine_pool_mode": str(args.fine_pool_mode),
        "fine_pool_topm": int(args.fine_pool_topm),
        "buckets": list(args.buckets),
        "init_noise_mode": args.init_noise_mode,
        "candidate_render_batch_size": int(args.candidate_render_batch_size),
        "feature_dim": int(payload["features"].shape[-1]),
        "score_map_shape": (
            None if "score_maps" not in payload else [int(v) for v in payload["score_maps"].shape[2:]]
        ),
        "num_samples": int(payload["features"].shape[0]),
        "skip_samples": int(args.skip_samples),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(json.dumps(payload["meta"], indent=2, sort_keys=True))


def main() -> None:
    export_cache(parse_args())


if __name__ == "__main__":
    main()
