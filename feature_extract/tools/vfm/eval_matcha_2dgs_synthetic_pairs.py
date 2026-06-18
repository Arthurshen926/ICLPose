"""Evaluate a MATCHA joint model on 2DGS synthetic source/target pairs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    match_reprojection_errors,
    reprojection_error_stats,
)


_PNP_REPROJECTION_THRESHOLDS_PX = (5.0, 10.0, 16.0, 32.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument(
        "--matcha_eval_preset",
        default="radio_matcha_local_window",
        choices=("radio_matcha_local_window", "radio_matcha_patch_corr"),
    )
    parser.add_argument("--layer_name", default="radio_dual")
    parser.add_argument("--feature_mode", default="radio_dual", choices=("radio_final", "radio_dual"))
    parser.add_argument("--radio_fine_intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_coarse_source", default="final", choices=("final", "intermediate"))
    parser.add_argument("--radio_coarse_intermediate_index", type=int, default=-1)
    parser.add_argument("--query_feature_cache_dir", default="")
    parser.add_argument("--query_feature_cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--pair_type_B_translation_m", type=float, default=0.25)
    parser.add_argument("--pair_type_C_translation_m", type=float, default=0.5)
    parser.add_argument("--perturb_rotation_deg", type=float, default=0.0)
    parser.add_argument("--feature_fusion_mode", default="none", choices=("none", "local_attention"))
    parser.add_argument("--feature_fusion_radius", type=int, default=1)
    parser.add_argument("--feature_fusion_temperature", type=float, default=5.0)
    parser.add_argument("--feature_fusion_alpha", type=float, default=0.5)
    parser.add_argument("--roundtrip_threshold_px", type=float, default=1.5)
    parser.add_argument("--roundtrip_heatmap_threshold_px", type=float, default=2.0)
    parser.add_argument("--visibility_alpha_threshold", type=float, default=0.0)
    parser.add_argument("--depth_edge_threshold_m", type=float, default=0.0)
    parser.add_argument("--multiview_supervision_support_views", type=int, default=0)
    parser.add_argument("--multiview_supervision_min_support_views", type=int, default=0)
    parser.add_argument("--multiview_supervision_depth_tolerance_m", type=float, default=0.05)
    parser.add_argument("--collect_visibility_no_match", action="store_true")
    parser.add_argument("--max_visibility_no_match", type=int, default=256)
    parser.add_argument("--soft_offset_sigma_bins", type=float, default=0.75)
    parser.add_argument("--pose_confidence_label_source", default="supervision", choices=("supervision", "gt_reprojection"))
    parser.add_argument("--pose_confidence_labels", action="store_true")
    parser.add_argument("--pose_confidence_positive_threshold_px", type=float, default=8.0)
    parser.add_argument("--pose_confidence_negative_threshold_px", type=float, default=24.0)
    parser.add_argument("--hard_negatives_per_match", type=int, default=16)
    parser.add_argument(
        "--fine_supervision_source",
        default="render_subcell_stratified",
        choices=("cell_center", "render_subcell", "render_subcell_stratified", "render_alike"),
    )
    parser.add_argument("--merge_fine_labels_into_coarse", action="store_true")
    parser.add_argument("--keypoint_distill_method", default="alike", choices=("none", "alike"))
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=4096)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--fine_confidence_blend", type=float, default=0.0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=10.0)
    parser.add_argument("--pnp_refine_method", type=str.upper, default="LM", choices=("NONE", "LM", "VVS"))
    parser.add_argument(
        "--pnp_soft_order_mode",
        default="none",
        choices=("none", "similarity", "margin", "reliability", "confidence", "uncertainty", "composite"),
    )
    parser.add_argument("--pnp_soft_order_top_n", type=int, default=0)
    parser.add_argument("--pnp_pair_confidence", action="store_true")
    parser.add_argument("--pnp_pair_confidence_blend", type=float, default=1.0)
    parser.add_argument("--pnp_spatial_nms_radius_px", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--builder_device", default="")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _numeric_values(values: Sequence[object]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        try:
            item = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(item):
            numeric.append(float(item))
    return numeric


def _mean_or_none(values: Sequence[object]) -> float | None:
    numeric = _numeric_values(values)
    return None if not numeric else float(np.mean(numeric))


def _median_or_none(values: Sequence[object]) -> float | None:
    numeric = _numeric_values(values)
    return None if not numeric else float(np.median(numeric))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _aggregate_rows_by_bin(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        label = str(row.get("synthetic_pose_bin", "unknown") or "unknown")
        grouped.setdefault(label, []).append(row)

    summary: dict[str, dict[str, object]] = {}
    for label in sorted(grouped):
        items = grouped[label]
        summary[label] = {
            "pair_count": int(len(items)),
            "pnp_solve_rate": _mean_or_none([1.0 if _truthy(item.get("pnp_success")) else 0.0 for item in items]),
            "median_translation_error_m": _median_or_none([item.get("translation_error_m") for item in items]),
            "median_rotation_error_deg": _median_or_none([item.get("rotation_error_deg") for item in items]),
            "mean_gt_precision_5px": _mean_or_none([item.get("gt_precision_5px") for item in items]),
            "mean_gt_precision_10px": _mean_or_none([item.get("gt_precision_10px") for item in items]),
            "mean_gt_precision_16px": _mean_or_none([item.get("gt_precision_16px") for item in items]),
            "mean_gt_precision_32px": _mean_or_none([item.get("gt_precision_32px") for item in items]),
            "mean_pnp_inlier_gt_precision_5px": _mean_or_none(
                [item.get("pnp_inlier_gt_precision_5px") for item in items]
            ),
            "mean_pnp_inlier_gt_precision_10px": _mean_or_none(
                [item.get("pnp_inlier_gt_precision_10px") for item in items]
            ),
            "mean_pnp_inlier_gt_precision_16px": _mean_or_none(
                [item.get("pnp_inlier_gt_precision_16px") for item in items]
            ),
            "mean_pnp_inlier_gt_precision_32px": _mean_or_none(
                [item.get("pnp_inlier_gt_precision_32px") for item in items]
            ),
            "mean_pnp_inlier_count": _mean_or_none([item.get("pnp_inlier_count") for item in items]),
            "mean_fine_offset_before_median_px": _mean_or_none(
                [item.get("fine_offset_before_median_px") for item in items]
            ),
            "mean_fine_offset_after_median_px": _mean_or_none(
                [item.get("fine_offset_after_median_px") for item in items]
            ),
            "mean_fine_offset_improvement_px": _mean_or_none(
                [item.get("fine_offset_improvement_mean_px") for item in items]
            ),
            "mean_fine_offset_improved_ratio": _mean_or_none(
                [item.get("fine_offset_improved_ratio") for item in items]
            ),
        }
    return summary


def _aggregate_overall_row_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    return _aggregate_rows_by_bin([{**dict(row), "synthetic_pose_bin": "all"} for row in rows])["all"]


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _resolved_device(requested: str):
    import torch

    requested_name = str(requested)
    if requested_name.startswith("cuda") and not torch.cuda.is_available():
        requested_name = "cpu"
    return torch.device(requested_name)


def _model_config_from_run(run, args: argparse.Namespace, device) -> object:
    from feature_extract.vfm.matcha_joint_training import MatchaJointTrainingConfig

    model = run.model
    summary_config = run.summary.get("config", {}) if isinstance(run.summary, Mapping) else {}
    summary_config = dict(summary_config) if isinstance(summary_config, Mapping) else {}
    is_radio_dual = hasattr(model, "fine_input_dim") and hasattr(model, "coarse_input_dim")
    model_type = str(summary_config.get("model_type", "radio_dual_attention" if is_radio_dual else "residual_adapter"))
    adapter = getattr(model, "adapter", None)

    config_kwargs = {
        "model_type": model_type,
        "output_dim": int(summary_config.get("output_dim", getattr(model, "output_dim", 128))),
        "residual_hidden_dim": int(
            summary_config.get(
                "residual_hidden_dim",
                getattr(model, "residual_hidden_dim", getattr(adapter, "residual_hidden_dim", 256)),
            )
        ),
        "fine_input_dim": int(summary_config.get("fine_input_dim", getattr(model, "fine_input_dim", 0))),
        "coarse_input_dim": int(summary_config.get("coarse_input_dim", getattr(model, "coarse_input_dim", 0))),
        "attention_hidden_dim": int(summary_config.get("attention_hidden_dim", getattr(model, "attention_hidden_dim", 256))),
        "attention_depth": int(summary_config.get("attention_depth", getattr(model, "attention_depth", 2))),
        "attention_heads": int(summary_config.get("attention_heads", getattr(model, "attention_heads", 4))),
        "attention_patch_size": int(summary_config.get("attention_patch_size", getattr(model, "attention_patch_size", 2))),
        "attention_upsample_mode": str(
            summary_config.get("attention_upsample_mode", getattr(model, "attention_upsample_mode", "bilinear"))
        ),
        "attention_fusion_mode": str(
            summary_config.get("attention_fusion_mode", getattr(model, "attention_fusion_mode", "matcha_original"))
        ),
        "batch_size": int(args.batch_size),
        "map_pair_batch_size": int(args.map_pair_batch_size),
        "local_window_fine_loss_weight": 0.5,
        "patch_corr_fine_loss_weight": 0.0,
        "patch_corr_fine_epe_weight": float(summary_config.get("patch_corr_fine_epe_weight", 0.1)),
        "patch_corr_fine_batch_size": int(summary_config.get("patch_corr_fine_batch_size", 256)),
        "patch_corr_fine_max_samples_per_pair": int(summary_config.get("patch_corr_fine_max_samples_per_pair", 512)),
        "patch_corr_fine_detach_context": bool(summary_config.get("patch_corr_fine_detach_context", True)),
        "group_size": int(summary_config.get("group_size", getattr(adapter, "group_size", 64))),
        "input_norm_mode": str(summary_config.get("input_norm_mode", getattr(adapter, "input_norm_mode", "identity"))),
        "gate_mode": str(summary_config.get("gate_mode", getattr(adapter, "gate_mode", "residual"))),
        "residual_gate_scale": float(
            summary_config.get("residual_gate_scale", getattr(adapter, "residual_gate_scale", 0.1))
        ),
        "device": str(device),
        "seed": int(args.seed),
    }
    if str(args.matcha_eval_preset) == "radio_matcha_patch_corr":
        config_kwargs["local_window_fine_loss_weight"] = 0.0
        config_kwargs["patch_corr_fine_loss_weight"] = float(summary_config.get("patch_corr_fine_loss_weight", 1.0))
    elif "local_window_fine_loss_weight" in summary_config:
        config_kwargs["local_window_fine_loss_weight"] = float(summary_config["local_window_fine_loss_weight"])
    return MatchaJointTrainingConfig(**config_kwargs)


def _match_confidence(match: object) -> float:
    for name in ("dual_softmax_confidence", "coarse_score", "similarity"):
        value = getattr(match, name, None)
        if value is None:
            continue
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(confidence):
            return confidence
    return 0.0


def _target_depth_matches_to_pnp(
    matches: Sequence[object],
    *,
    render_depth: np.ndarray,
    render_pose_w2c: np.ndarray,
    render_camera,
) -> list[QueryTo3DMatch]:
    from feature_extract.vfm.rendered_keypoint_matching import keypoint_feature_matches_to_pnp_matches

    depth = np.asarray(render_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("render_depth must have shape (H, W)")
    if depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise ValueError("render_depth must be non-empty")
    return keypoint_feature_matches_to_pnp_matches(
        matches,
        depth,
        render_camera,
        np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4),
        image_width=int(render_camera.width),
        image_height=int(render_camera.height),
    )


def _replace_match_confidence_scores(
    matches: Sequence[object],
    confidences: np.ndarray,
    *,
    blend: float = 1.0,
) -> list[object]:
    values = list(matches)
    scores = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if scores.shape[0] != len(values):
        raise ValueError("confidences must contain one score per match")
    amount = float(np.clip(float(blend), 0.0, 1.0))
    blended = []
    for match, score in zip(values, scores):
        pose_score = float(np.clip(score, 0.0, 1.0))
        existing = getattr(match, "dual_softmax_confidence", None)
        if existing is None or not np.isfinite(float(existing)):
            base_score = pose_score
        else:
            base_score = float(np.clip(float(existing), 0.0, 1.0))
        blended.append((1.0 - amount) * base_score + amount * pose_score)
    updated = [
        replace(match, dual_softmax_confidence=float(np.clip(score, 0.0, 1.0)))
        for match, score in zip(values, blended)
    ]
    updated.sort(
        key=lambda item: (
            float(getattr(item, "dual_softmax_confidence", 0.0) or 0.0),
            float(getattr(item, "similarity", 0.0)),
        ),
        reverse=True,
    )
    return updated


def _apply_pair_confidence_head_to_matches(
    matches: Sequence[object],
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    device: str,
    blend: float = 1.0,
) -> list[object]:
    from feature_extract.vfm.matcha_joint_training import predict_matcha_joint_pair_heads_for_matches

    values = list(matches)
    if not values:
        return []
    confidences, _fine_logits = predict_matcha_joint_pair_heads_for_matches(
        joint_model,
        np.asarray(query_feature_map, dtype=np.float32),
        np.asarray(render_feature_map, dtype=np.float32),
        values,
        device=str(device),
        fine_target_side="render",
    )
    return _replace_match_confidence_scores(values, confidences, blend=float(blend))


def _spatially_diverse_pnp_matches(
    pnp_matches: Sequence[QueryTo3DMatch],
    *,
    radius_px: float,
    max_matches: int,
) -> list[QueryTo3DMatch]:
    radius = float(radius_px)
    limit = int(max_matches)
    if radius <= 0.0 or limit <= 0:
        return list(pnp_matches) if limit <= 0 else list(pnp_matches)[:limit]
    selected: list[QueryTo3DMatch] = []
    for match in pnp_matches:
        query_xy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        render_xy = None if match.render_xy is None else np.asarray(match.render_xy, dtype=np.float64).reshape(2)
        too_close = False
        for kept in selected:
            kept_query_xy = np.asarray(kept.xy, dtype=np.float64).reshape(2)
            if float(np.linalg.norm(query_xy - kept_query_xy)) < radius:
                too_close = True
                break
            if render_xy is not None and kept.render_xy is not None:
                kept_render_xy = np.asarray(kept.render_xy, dtype=np.float64).reshape(2)
                if float(np.linalg.norm(render_xy - kept_render_xy)) < radius:
                    too_close = True
                    break
        if too_close:
            continue
        selected.append(match)
        if len(selected) >= limit:
            break
    return selected


def _order_pnp_matches_for_eval(
    pnp_matches: Sequence[QueryTo3DMatch],
    *,
    mode: str,
    top_n: int = 0,
    spatial_nms_radius_px: float = 0.0,
) -> list[QueryTo3DMatch]:
    from feature_extract.vfm.query_to_3d_matching import soft_order_pnp_matches

    values = list(pnp_matches)
    limit = int(top_n)
    if limit < 0:
        raise ValueError("top_n must be non-negative")
    spatial_radius = float(spatial_nms_radius_px)
    if spatial_radius < 0.0:
        raise ValueError("spatial_nms_radius_px must be non-negative")
    max_matches = None if limit == 0 or spatial_radius > 0.0 else limit
    if str(mode) == "none":
        ordered = values if max_matches is None else values[:max_matches]
    else:
        ordered = soft_order_pnp_matches(values, mode=str(mode), max_matches=max_matches)
    if spatial_radius > 0.0 and limit > 0:
        return _spatially_diverse_pnp_matches(ordered, radius_px=spatial_radius, max_matches=limit)
    return ordered


def _pnp_reprojection_stats_row(
    pnp_matches: Sequence[QueryTo3DMatch],
    source_pose_w2c: np.ndarray,
    camera,
    *,
    pnp_inlier_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    return dict(
        reprojection_error_stats(
            pnp_matches,
            np.asarray(source_pose_w2c, dtype=np.float64).reshape(4, 4),
            camera,
            thresholds_px=_PNP_REPROJECTION_THRESHOLDS_PX,
            pnp_inlier_mask=pnp_inlier_mask if pnp_matches else None,
        )
    )


def _empty_fine_offset_stats_row() -> dict[str, float | int | None]:
    return {
        "fine_offset_eval_count": 0,
        "fine_offset_before_median_px": None,
        "fine_offset_after_median_px": None,
        "fine_offset_improvement_mean_px": None,
        "fine_offset_improved_ratio": None,
        "fine_offset_gt16_before": None,
        "fine_offset_gt16_after": None,
    }


def _coarse_cell_center_xy(token_index: int, *, query_grid_hw: Sequence[int], camera) -> np.ndarray | None:
    grid_h = int(query_grid_hw[0])
    grid_w = int(query_grid_hw[1])
    if grid_h <= 0 or grid_w <= 0:
        return None
    token = int(token_index)
    if token < 0 or token >= int(grid_h * grid_w):
        return None
    row = token // grid_w
    col = token % grid_w
    x = (float(col) + 0.5) * float(camera.width) / float(grid_w)
    y = (float(row) + 0.5) * float(camera.height) / float(grid_h)
    return np.asarray([x, y], dtype=np.float64)


def _fine_offset_stats_row(
    pnp_matches: Sequence[QueryTo3DMatch],
    source_pose_w2c: np.ndarray,
    camera,
    *,
    query_grid_hw: Sequence[int],
) -> dict[str, float | int | None]:
    before_matches: list[QueryTo3DMatch] = []
    after_matches: list[QueryTo3DMatch] = []
    for match in pnp_matches:
        before_xy = _coarse_cell_center_xy(match.token_index, query_grid_hw=query_grid_hw, camera=camera)
        if before_xy is None:
            continue
        after_xy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        xyz = np.asarray(match.xyz, dtype=np.float64).reshape(3)
        if not (np.isfinite(before_xy).all() and np.isfinite(after_xy).all() and np.isfinite(xyz).all()):
            continue
        after_matches.append(match)
        before_matches.append(replace(match, xy=before_xy))

    if not after_matches:
        return _empty_fine_offset_stats_row()

    pose = np.asarray(source_pose_w2c, dtype=np.float64).reshape(4, 4)
    before_errors = match_reprojection_errors(before_matches, pose, camera)
    after_errors = match_reprojection_errors(after_matches, pose, camera)
    valid = np.isfinite(before_errors) & np.isfinite(after_errors)
    if not np.any(valid):
        return _empty_fine_offset_stats_row()

    before = before_errors[valid]
    after = after_errors[valid]
    improvement = before - after
    return {
        "fine_offset_eval_count": int(before.shape[0]),
        "fine_offset_before_median_px": float(np.median(before)),
        "fine_offset_after_median_px": float(np.median(after)),
        "fine_offset_improvement_mean_px": float(np.mean(improvement)),
        "fine_offset_improved_ratio": float(np.mean(improvement > 0.0)),
        "fine_offset_gt16_before": float(np.mean(before <= 16.0)),
        "fine_offset_gt16_after": float(np.mean(after <= 16.0)),
    }


def _descriptor_maps_for_eval_pnp(
    model,
    state: Mapping[str, object],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str, str | None]:
    query_feature_map = np.asarray(state["query_feature_map"], dtype=np.float32)
    render_feature_map = np.asarray(state["render_feature_map"], dtype=np.float32)
    if not hasattr(model, "forward_feature_map"):
        return query_feature_map, render_feature_map, None, None, "raw_feature_map_fallback", "model lacks forward_feature_map"
    try:
        from feature_extract.vfm.matcha_joint_training import project_feature_map_with_matcha_joint_model

        query_desc, query_offsets, _query_heatmap = project_feature_map_with_matcha_joint_model(
            model,
            query_feature_map,
            device=str(device),
        )
        render_desc, render_offsets, _render_heatmap = project_feature_map_with_matcha_joint_model(
            model,
            render_feature_map,
            device=str(device),
        )
        return query_desc, render_desc, query_offsets, render_offsets, "model_forward_feature_map", None
    except (AttributeError, NotImplementedError, TypeError) as exc:
        return query_feature_map, render_feature_map, None, None, "raw_feature_map_fallback", str(exc)


def _predict_pair_conditioned_local_window_logits_for_matches(
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    matches: Sequence[object],
    *,
    target_side: str,
    device: str,
) -> np.ndarray:
    import torch

    values = list(matches)
    if not values:
        return np.zeros((0, 64), dtype=np.float32)
    side = str(target_side)
    if side not in {"query", "render"}:
        raise ValueError("target_side must be 'query' or 'render'")
    if side == "render":
        source_feature_map = query_feature_map
        target_feature_map = render_feature_map
        source_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
        target_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
    else:
        source_feature_map = render_feature_map
        target_feature_map = query_feature_map
        source_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
        target_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
    pairs = np.zeros((len(values),), dtype=np.int64)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    with torch.no_grad():
        source_maps = torch.as_tensor(np.asarray(source_feature_map, dtype=np.float32)[None], device=torch_device)
        target_maps = torch.as_tensor(np.asarray(target_feature_map, dtype=np.float32)[None], device=torch_device)
        logits = joint_model.to(torch_device).eval().local_window_fine_logits_from_maps(
            source_maps,
            target_maps,
            torch.as_tensor(pairs, dtype=torch.long, device=torch_device),
            torch.as_tensor(source_indices, dtype=torch.long, device=torch_device),
            torch.as_tensor(target_indices, dtype=torch.long, device=torch_device),
        )
    return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _refine_matches_with_pair_conditioned_local_window(
    matches: Sequence[object],
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    device: str,
    coordinate_mode: str = "argmax",
    render_coordinate_mode: str | None = None,
    query_coordinate_mode: str | None = None,
) -> list[object]:
    from feature_extract.vfm.matcha_coarse_to_fine import apply_pair_fine_logits_to_matches

    values = list(matches)
    if not values:
        return []
    query_map = np.asarray(query_feature_map, dtype=np.float32)
    render_map = np.asarray(render_feature_map, dtype=np.float32)
    if query_map.ndim != 3 or render_map.ndim != 3:
        raise ValueError("query/render feature maps must have shape (C, H, W)")
    if not hasattr(joint_model, "local_window_fine_logits_from_maps"):
        raise AttributeError("joint model lacks local_window_fine_logits_from_maps")
    render_logits = _predict_pair_conditioned_local_window_logits_for_matches(
        joint_model,
        query_map,
        render_map,
        values,
        target_side="render",
        device=str(device),
    )
    values = apply_pair_fine_logits_to_matches(
        values,
        render_logits,
        target_side="render",
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        render_grid_width=int(render_map.shape[2]),
        render_grid_height=int(render_map.shape[1]),
        coordinate_mode=str(coordinate_mode if render_coordinate_mode is None else render_coordinate_mode),
    )
    query_logits = _predict_pair_conditioned_local_window_logits_for_matches(
        joint_model,
        query_map,
        render_map,
        values,
        target_side="query",
        device=str(device),
    )
    return apply_pair_fine_logits_to_matches(
        values,
        query_logits,
        target_side="query",
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        query_grid_width=int(query_map.shape[2]),
        query_grid_height=int(query_map.shape[1]),
        coordinate_mode=str(coordinate_mode if query_coordinate_mode is None else query_coordinate_mode),
    )


def _predict_pair_conditioned_patch_corr_logits_for_matches(
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    matches: Sequence[object],
    *,
    target_side: str,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    device: str,
) -> np.ndarray:
    import torch

    from feature_extract.vfm.matcha_joint_cache import _rgb_to_bchw_float

    values = list(matches)
    if not values:
        return np.zeros((0, 64), dtype=np.float32)
    side = str(target_side)
    if side not in {"query", "render"}:
        raise ValueError("target_side must be 'query' or 'render'")
    if side == "render":
        source_feature = np.asarray(query_feature_map, dtype=np.float32)
        target_feature = np.asarray(render_feature_map, dtype=np.float32)
        source_rgb = _rgb_to_bchw_float(
            query_rgb,
            grid_hw=(int(source_feature.shape[1]), int(source_feature.shape[2])),
        )
        target_rgb = _rgb_to_bchw_float(
            render_rgb,
            grid_hw=(int(target_feature.shape[1]), int(target_feature.shape[2])),
        )
        source_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
        target_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
        source_xy = np.stack([np.asarray(match.query_xy, dtype=np.float64).reshape(2) for match in values], axis=0)
        source_width = int(query_image_width)
        source_height = int(query_image_height)
    else:
        source_feature = np.asarray(render_feature_map, dtype=np.float32)
        target_feature = np.asarray(query_feature_map, dtype=np.float32)
        source_rgb = _rgb_to_bchw_float(
            render_rgb,
            grid_hw=(int(source_feature.shape[1]), int(source_feature.shape[2])),
        )
        target_rgb = _rgb_to_bchw_float(
            query_rgb,
            grid_hw=(int(target_feature.shape[1]), int(target_feature.shape[2])),
        )
        source_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
        target_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
        source_xy = np.stack([np.asarray(match.render_xy, dtype=np.float64).reshape(2) for match in values], axis=0)
        source_width = int(render_image_width)
        source_height = int(render_image_height)
    source_xy = source_xy.astype(np.float32, copy=False)
    source_xy[:, 0] *= float(source_rgb.shape[3]) / max(float(source_width), 1.0)
    source_xy[:, 1] *= float(source_rgb.shape[2]) / max(float(source_height), 1.0)
    pairs = np.zeros((len(values),), dtype=np.int64)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    with torch.no_grad():
        logits = joint_model.to(torch_device).eval().patch_corr_fine_logits_from_maps_and_rgb(
            torch.as_tensor(source_feature[None], dtype=torch.float32, device=torch_device),
            torch.as_tensor(target_feature[None], dtype=torch.float32, device=torch_device),
            torch.as_tensor(source_rgb, dtype=torch.float32, device=torch_device),
            torch.as_tensor(target_rgb, dtype=torch.float32, device=torch_device),
            torch.as_tensor(pairs, dtype=torch.long, device=torch_device),
            torch.as_tensor(source_indices, dtype=torch.long, device=torch_device),
            torch.as_tensor(target_indices, dtype=torch.long, device=torch_device),
            query_xy=torch.as_tensor(source_xy, dtype=torch.float32, device=torch_device),
        )
    return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _refine_matches_with_pair_conditioned_patch_corr(
    matches: Sequence[object],
    joint_model,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    device: str,
    coordinate_mode: str = "argmax",
    render_coordinate_mode: str | None = None,
    query_coordinate_mode: str | None = None,
    fine_confidence_blend: float = 0.0,
) -> list[object]:
    from feature_extract.vfm.matcha_coarse_to_fine import apply_fine_logit_confidence_to_matches, apply_pair_fine_logits_to_matches

    values = list(matches)
    if not values:
        return []
    query_map = np.asarray(query_feature_map, dtype=np.float32)
    render_map = np.asarray(render_feature_map, dtype=np.float32)
    if query_map.ndim != 3 or render_map.ndim != 3:
        raise ValueError("query/render feature maps must have shape (C, H, W)")
    if not hasattr(joint_model, "patch_corr_fine_logits_from_maps_and_rgb"):
        raise AttributeError("joint model lacks patch_corr_fine_logits_from_maps_and_rgb")
    render_logits = _predict_pair_conditioned_patch_corr_logits_for_matches(
        joint_model,
        query_map,
        render_map,
        np.asarray(query_rgb),
        np.asarray(render_rgb),
        values,
        target_side="render",
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        device=str(device),
    )
    values = apply_pair_fine_logits_to_matches(
        values,
        render_logits,
        target_side="render",
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        render_grid_width=int(render_map.shape[2]),
        render_grid_height=int(render_map.shape[1]),
        coordinate_mode=str(coordinate_mode if render_coordinate_mode is None else render_coordinate_mode),
    )
    blend = float(np.clip(float(fine_confidence_blend), 0.0, 1.0))
    if blend > 0.0 and len(values) == int(render_logits.shape[0]):
        values = apply_fine_logit_confidence_to_matches(values, render_logits, blend=blend)
    query_logits = _predict_pair_conditioned_patch_corr_logits_for_matches(
        joint_model,
        query_map,
        render_map,
        np.asarray(query_rgb),
        np.asarray(render_rgb),
        values,
        target_side="query",
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        device=str(device),
    )
    values = apply_pair_fine_logits_to_matches(
        values,
        query_logits,
        target_side="query",
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        query_grid_width=int(query_map.shape[2]),
        query_grid_height=int(query_map.shape[1]),
        coordinate_mode=str(coordinate_mode if query_coordinate_mode is None else query_coordinate_mode),
    )
    if blend > 0.0 and len(values) == int(query_logits.shape[0]):
        values = apply_fine_logit_confidence_to_matches(values, query_logits, blend=blend)
    return values


def _refinement_source_name(base: str, coordinate_mode: str, *, render_mode: str | None = None, query_mode: str | None = None) -> str:
    mode = str(coordinate_mode)
    if render_mode is not None or query_mode is not None:
        rmode = mode if render_mode is None else str(render_mode)
        qmode = mode if query_mode is None else str(query_mode)
        if rmode != qmode:
            return f"{base}_render_{rmode}_query_{qmode}"
        mode = rmode
    return str(base) if mode == "argmax" else f"{base}_{mode}"


def _matcha_eval_matches_for_pnp(
    query_match_map: np.ndarray,
    render_match_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    query_offset_logits: np.ndarray | None = None,
    render_offset_logits: np.ndarray | None = None,
    joint_model=None,
    raw_query_feature_map: np.ndarray | None = None,
    raw_render_feature_map: np.ndarray | None = None,
    query_rgb: np.ndarray | None = None,
    render_rgb: np.ndarray | None = None,
    fine_refinement_mode: str = "local_window",
    fine_coordinate_mode: str = "argmax",
    render_fine_coordinate_mode: str | None = None,
    query_fine_coordinate_mode: str | None = None,
    fine_confidence_blend: float = 0.0,
    device: str = "cpu",
) -> tuple[list[object], str, str | None]:
    from feature_extract.vfm.matcha_coarse_to_fine import (
        apply_offset_logits_to_matches,
        matcha_coarse_to_fine_keypoint_matches,
    )

    matches = matcha_coarse_to_fine_keypoint_matches(
        query_match_map,
        render_match_map,
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        logit_scale=10.0,
        min_confidence=0.0,
        min_similarity=-1.0,
        max_matches=1000,
        fine_search_radius_px=0.0,
        mutual=False,
        coarse_top_k_per_query=1,
        coarse_mutual_mode="annotate",
    )
    mode = str(fine_refinement_mode)
    if mode not in {"local_window", "patch_corr"}:
        raise ValueError("fine_refinement_mode must be 'local_window' or 'patch_corr'")
    if (
        mode == "patch_corr"
        and joint_model is not None
        and raw_query_feature_map is not None
        and raw_render_feature_map is not None
        and query_rgb is not None
        and render_rgb is not None
        and hasattr(joint_model, "patch_corr_fine_logits_from_maps_and_rgb")
    ):
        refined = _refine_matches_with_pair_conditioned_patch_corr(
            matches,
            joint_model,
            np.asarray(raw_query_feature_map, dtype=np.float32),
            np.asarray(raw_render_feature_map, dtype=np.float32),
            np.asarray(query_rgb),
            np.asarray(render_rgb),
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            coordinate_mode=str(fine_coordinate_mode),
            render_coordinate_mode=render_fine_coordinate_mode,
            query_coordinate_mode=query_fine_coordinate_mode,
            fine_confidence_blend=float(fine_confidence_blend),
            device=str(device),
        )
        source_name = _refinement_source_name(
            "pair_conditioned_patch_corr",
            str(fine_coordinate_mode),
            render_mode=render_fine_coordinate_mode,
            query_mode=query_fine_coordinate_mode,
        )
        blend = float(np.clip(float(fine_confidence_blend), 0.0, 1.0))
        if blend > 0.0:
            source_name = f"{source_name}_fineconf{blend:g}"
        return (
            refined,
            source_name,
            None,
        )
    if (
        mode == "local_window"
        and joint_model is not None
        and raw_query_feature_map is not None
        and raw_render_feature_map is not None
        and hasattr(joint_model, "local_window_fine_logits_from_maps")
    ):
        refined = _refine_matches_with_pair_conditioned_local_window(
            matches,
            joint_model,
            np.asarray(raw_query_feature_map, dtype=np.float32),
            np.asarray(raw_render_feature_map, dtype=np.float32),
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            coordinate_mode=str(fine_coordinate_mode),
            render_coordinate_mode=render_fine_coordinate_mode,
            query_coordinate_mode=query_fine_coordinate_mode,
            device=str(device),
        )
        return (
            refined,
            _refinement_source_name(
                "pair_conditioned_local_window",
                str(fine_coordinate_mode),
                render_mode=render_fine_coordinate_mode,
                query_mode=query_fine_coordinate_mode,
            ),
            None,
        )
    if query_offset_logits is not None or render_offset_logits is not None:
        return (
            apply_offset_logits_to_matches(
                matches,
                query_offset_logits,
                render_offset_logits,
                query_image_width=int(query_image_width),
                query_image_height=int(query_image_height),
                render_image_width=int(render_image_width),
                render_image_height=int(render_image_height),
            ),
            "dense_offset_fallback",
            None if joint_model is None else f"joint model lacks {mode} refinement inputs",
        )
    return matches, "coarse_descriptor", None


def main(argv: Sequence[str] | None = None) -> None:
    import torch

    from feature_extract.tools.vfm.train_matcha_joint_streaming_model import (
        SyntheticStreamingPairBuilder,
        _builder_class_for_manifest,
    )
    from feature_extract.vfm.matcha_joint_training import _evaluate, load_matcha_joint_model
    from feature_extract.vfm.matcha_streaming_manifest import MatchaStreamingPairManifest
    from feature_extract.vfm.query_to_3d_matching import estimate_pose_pnp_ransac, pnp_pose_error

    args = parse_args(argv)
    manifest = MatchaStreamingPairManifest.from_json(Path(args.streaming_manifest))
    builder_cls = _builder_class_for_manifest(manifest)
    if builder_cls is not SyntheticStreamingPairBuilder:
        raise ValueError("eval_matcha_2dgs_synthetic_pairs requires a 2dgs_synthetic manifest")

    device = _resolved_device(str(args.device))
    args.device = str(device)
    if not str(args.builder_device):
        args.builder_device = str(device)
    builder = builder_cls(args, manifest)
    run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=str(device))
    model = run.model.to(device).eval()
    cfg = _model_config_from_run(run, args, device)

    records = list(manifest.records)
    if int(args.max_pairs) > 0:
        records = records[: int(args.max_pairs)]
    fine_refinement_mode = "patch_corr" if str(args.matcha_eval_preset) == "radio_matcha_patch_corr" else "local_window"
    fine_coordinate_mode = "argmax"
    render_fine_coordinate_mode = "argmax"
    query_fine_coordinate_mode = "softargmax" if fine_refinement_mode == "patch_corr" else "argmax"

    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []
    for index, record in enumerate(records):
        samples, row, state = builder.build_with_eval_state(record)
        with torch.no_grad():
            metrics = _evaluate(model, samples, cfg, device)
        metric_rows.append(dict(metrics))
        (
            query_match_map,
            render_match_map,
            query_offset_logits,
            render_offset_logits,
            pnp_feature_source,
            fallback_reason,
        ) = _descriptor_maps_for_eval_pnp(
            model,
            state,
            device=str(device),
        )
        matches, refinement_source, refinement_fallback = _matcha_eval_matches_for_pnp(
            query_match_map,
            render_match_map,
            query_image_width=int(builder.camera.width),
            query_image_height=int(builder.camera.height),
            render_image_width=int(builder.render_camera.width),
            render_image_height=int(builder.render_camera.height),
            query_offset_logits=query_offset_logits,
            render_offset_logits=render_offset_logits,
            joint_model=model,
            raw_query_feature_map=np.asarray(state["query_feature_map"]),
            raw_render_feature_map=np.asarray(state["render_feature_map"]),
            query_rgb=None if "query_rgb" not in state else np.asarray(state["query_rgb"]),
            render_rgb=None if "render_rgb" not in state else np.asarray(state["render_rgb"]),
            fine_refinement_mode=fine_refinement_mode,
            fine_coordinate_mode=fine_coordinate_mode,
            render_fine_coordinate_mode=render_fine_coordinate_mode,
            query_fine_coordinate_mode=query_fine_coordinate_mode,
            fine_confidence_blend=float(args.fine_confidence_blend),
            device=str(device),
        )
        pnp_feature_source = f"{pnp_feature_source}+{refinement_source}"
        fallback_reason = fallback_reason or refinement_fallback
        if bool(args.pnp_pair_confidence):
            matches = _apply_pair_confidence_head_to_matches(
                matches,
                model,
                query_match_map,
                render_match_map,
                device=str(device),
                blend=float(args.pnp_pair_confidence_blend),
            )
            pnp_feature_source = f"{pnp_feature_source}+pair_confidence_rank"
        pnp_matches = _target_depth_matches_to_pnp(
            matches,
            render_depth=np.asarray(state["target_depth"]),
            render_pose_w2c=np.asarray(state["target_pose_w2c"]),
            render_camera=builder.render_camera,
        )
        pnp_matches = _order_pnp_matches_for_eval(
            pnp_matches,
            mode=str(args.pnp_soft_order_mode),
            top_n=int(args.pnp_soft_order_top_n),
            spatial_nms_radius_px=float(args.pnp_spatial_nms_radius_px),
        )
        pnp = estimate_pose_pnp_ransac(
            pnp_matches,
            builder.camera,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=1000,
            refine_method=str(args.pnp_refine_method),
        )
        pose_error = pnp_pose_error(pnp.pose_w2c, np.asarray(state["source_pose_w2c"])) if bool(pnp.success) else None
        pnp_geometry_fields = _pnp_reprojection_stats_row(
            pnp_matches,
            np.asarray(state["source_pose_w2c"]),
            builder.camera,
            pnp_inlier_mask=pnp.inlier_mask,
        )
        fine_offset_fields = _fine_offset_stats_row(
            pnp_matches,
            np.asarray(state["source_pose_w2c"]),
            builder.camera,
            query_grid_hw=(int(query_match_map.shape[-2]), int(query_match_map.shape[-1])),
        )
        pnp_fields = {
            "pnp_attempted": True,
            "pnp_success": bool(pnp.success),
            "pnp_match_count": int(pnp.match_count),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": None if int(pnp.match_count) <= 0 else float(pnp.inlier_ratio),
            "translation_error_m": None if pose_error is None else float(pose_error.translation_m),
            "rotation_error_deg": None if pose_error is None else float(pose_error.rotation_deg),
            "pnp_match_feature_source": str(pnp_feature_source),
            "pnp_match_feature_fallback_reason": fallback_reason,
            "pnp_reprojection_error_px": float(args.pnp_reprojection_error_px),
            "pnp_refine_method": str(args.pnp_refine_method),
            "fine_confidence_blend": float(args.fine_confidence_blend),
            "pnp_soft_order_mode": str(args.pnp_soft_order_mode),
            "pnp_soft_order_top_n": int(args.pnp_soft_order_top_n),
            "pnp_pair_confidence": bool(args.pnp_pair_confidence),
            "pnp_pair_confidence_blend": float(args.pnp_pair_confidence_blend),
            "pnp_spatial_nms_radius_px": float(args.pnp_spatial_nms_radius_px),
        }
        rows.append(
            {
                **dict(row),
                "pair_index_eval": int(index),
                "train_top1_acc": metrics.get("train_top1_acc"),
                "map_descriptor_top1_acc": metrics.get("map_descriptor_top1_acc"),
                "local_window_fine_acc": metrics.get("local_window_fine_acc"),
                **pnp_geometry_fields,
                **fine_offset_fields,
                **pnp_fields,
            }
        )

    output_dir = Path(args.output_dir)
    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.json"
    _write_rows(rows_path, rows)
    metric_names = sorted({key for item in metric_rows for key in item.keys()})
    output_metrics = {
        key: _mean_or_none([item.get(key) for item in metric_rows])
        for key in metric_names
    }
    output_metrics.update(_aggregate_overall_row_metrics(rows))
    summary = {
        "stage": "matcha_2dgs_synthetic_pair_eval",
        "query_count": int(len({row.get("query_id") for row in rows})),
        "pair_count": int(len(rows)),
        "pnp_status": "predicted_matcha_depth_pnp",
        "metrics": output_metrics,
        "by_pose_bin": _aggregate_rows_by_bin(rows),
        "inputs": {
            "streaming_manifest": str(args.streaming_manifest),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
        },
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
