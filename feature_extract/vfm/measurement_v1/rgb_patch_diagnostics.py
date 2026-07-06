from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementBranch,
    continuous_offset_nll_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    _prior_scale_batch,
    _read_csv,
    _render_cache_by_query,
    _support_patch_source_audit,
    _stack_patch_batch,
)


DIAGNOSTIC_FIELDNAMES = [
    "row_index",
    "query_id",
    "anchor_id",
    "track_id",
    "support_track_id",
    "target_is_dustbin",
    "baseline_epe_px",
    "likelihood_epe_px",
    "mode_epe_px",
    "direct_epe_px",
    "improved",
    "mode_improved",
    "direct_improved",
    "center_x",
    "center_y",
    "query_pred_x",
    "query_pred_y",
    "query_mode_x",
    "query_mode_y",
    "query_direct_x",
    "query_direct_y",
    "query_gt_x",
    "query_gt_y",
    "target_dx",
    "target_dy",
    "pred_dx",
    "pred_dy",
    "peak_dx",
    "peak_dy",
    "direct_dx",
    "direct_dy",
    "dustbin_probability",
    "visualization",
]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in DIAGNOSTIC_FIELDNAMES})


def _patch_to_uint8(patch_chw: torch.Tensor) -> np.ndarray:
    arr = patch_chw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = np.moveaxis(arr[:3], 0, -1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _heatmap_to_uint8(probs: torch.Tensor) -> np.ndarray:
    values = probs.detach().cpu().float().numpy()
    values = values - float(values.min(initial=0.0))
    denom = max(float(values.max(initial=0.0)), 1e-12)
    gray = (values / denom * 255.0 + 0.5).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _save_visualization(
    *,
    path: Path,
    query_patch: torch.Tensor,
    render_patch: torch.Tensor,
    spatial_probs: torch.Tensor,
) -> None:
    query_img = Image.fromarray(_patch_to_uint8(query_patch))
    render_img = Image.fromarray(_patch_to_uint8(render_patch))
    side = int(round(float(spatial_probs.numel()) ** 0.5))
    heatmap = Image.fromarray(_heatmap_to_uint8(spatial_probs.reshape(side, side))).resize(query_img.size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (query_img.width * 3, query_img.height))
    canvas.paste(query_img, (0, 0))
    canvas.paste(render_img, (query_img.width, 0))
    canvas.paste(heatmap, (query_img.width * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _load_model(checkpoint: Path, *, device: torch.device) -> RGBPatchMeasurementBranch:
    payload = torch.load(Path(checkpoint), map_location=device)
    config = dict(payload.get("config", {}) if isinstance(payload, dict) else {})
    model = RGBPatchMeasurementBranch(
        search_radius_px=float(config.get("search_radius_px", 2.0)),
        context_radius_px=float(config.get("context_radius_px", 8.0)),
        step_px=float(config.get("step_px", 0.5)),
        feature_dim=int(config.get("feature_dim", 32)),
        hidden_dim=None if config.get("hidden_dim") is None else int(config.get("hidden_dim")),
        input_mode=str(config.get("input_mode", "rgb")),
        template_scale_factors=tuple(float(value) for value in config.get("template_scale_factors", [1.0])),
        condition_on_prior_scale=bool(config.get("condition_on_prior_scale", False)),
        prior_scale_expert_centers_px=tuple(float(value) for value in config.get("prior_scale_expert_centers_px", [])),
        prior_scale_expert_projection=bool(config.get("prior_scale_expert_projection", False)),
        prior_scale_expert_gate=str(config.get("prior_scale_expert_gate", "soft")),
    ).to(device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = set()
    if int(model.prior_scale_expert_centers.numel()) == 0:
        allowed_missing.add("prior_scale_expert_centers")
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible with RGBPatchMeasurementBranch: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def export_rgb_patch_diagnostics(
    *,
    rows_csv: Path,
    render_cache_manifest_csv: Path | None,
    image_root: Path,
    checkpoint: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    query_source: str = "real",
    support_patch_warp: str = "none",
    max_rows: int | None = None,
    visualize_limit: int = 16,
    batch_size: int = 1,
    prior_scale_key: str = "",
    device: str = "cuda",
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    render_map = _render_cache_by_query(None if render_cache_manifest_csv is None else Path(render_cache_manifest_csv), base_dir=base)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model = _load_model(Path(checkpoint), device=torch_device)
    query_cache: dict[str, torch.Tensor] = {}
    render_cache: dict[str, torch.Tensor] = {}
    output = Path(output_dir)
    diagnostic_rows: list[dict[str, object]] = []
    visual_candidates: list[tuple[float, int, Path, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    batch = max(1, int(batch_size))
    with torch.no_grad():
        for batch_start in range(0, len(rows), batch):
            batch_rows = rows[batch_start : batch_start + batch]
            query_patch, render_patch, target, baseline, _target_is_dustbin = _stack_patch_batch(
                batch_rows,
                image_root=Path(image_root),
                render_cache_by_query=render_map,
                image_width=int(image_width),
                image_height=int(image_height),
                crop_radius_px=model.crop_radius_px,
                step_px=model.step_px,
                query_cache=query_cache,
                render_cache=render_cache,
                query_source=str(query_source),
                render_patch_augmentation="none",
                support_patch_warp=str(support_patch_warp),
            )
            prior_scale = _prior_scale_batch(batch_rows, prior_scale_key=str(prior_scale_key))
            pred0 = model.forward_from_patches(
                query_patch.to(torch_device),
                render_patch.to(torch_device),
                prior_scale_px=None if prior_scale is None else prior_scale.to(torch_device),
            )
            _loss, likelihood = continuous_offset_nll_with_dustbin(
                pred0.logits,
                pred0.offsets_xy,
                target.to(torch_device),
                dustbin_logit=pred0.dustbin_logit,
                search_radius_px=model.search_radius_px,
                target_is_dustbin=None if _target_is_dustbin is None else _target_is_dustbin.to(torch_device),
            )
            spatial_probs_batch = torch.exp(likelihood.local_log_probs).detach().cpu()
            mean_xy_batch = likelihood.mean_offset_xy.detach().cpu()
            epe_batch = likelihood.epe_px.detach().cpu()
            dustbin_batch = likelihood.dustbin_probability.detach().cpu()
            target_cpu = target.detach().cpu()
            direct_mean_batch = None if pred0.direct_mean_offset_xy is None else pred0.direct_mean_offset_xy.detach().cpu()
            for local_index, row in enumerate(batch_rows):
                row_index = int(batch_start + local_index)
                spatial_probs = spatial_probs_batch[local_index]
                peak_idx = int(torch.argmax(spatial_probs).item())
                peak_xy = likelihood.offsets_xy[peak_idx].detach().cpu()
                mean_xy = mean_xy_batch[local_index]
                epe = float(epe_batch[local_index].item())
                mode_epe = float(torch.linalg.norm(peak_xy - target_cpu[local_index]).item())
                direct_xy = None if direct_mean_batch is None else direct_mean_batch[local_index]
                direct_epe = None if direct_xy is None else float(torch.linalg.norm(direct_xy - target_cpu[local_index]).item())
                baseline_value = float(baseline[local_index].item())
                center_x = float(str(row.get("center_x", "0")).strip())
                center_y = float(str(row.get("center_y", "0")).strip())
                vis_path = ""
                if int(visualize_limit) > 0:
                    candidate_path = output / "visualizations" / f"rank_pending_row{row_index:06d}.png"
                    visual_candidates.append((epe, row_index, candidate_path, query_patch[local_index], render_patch[local_index], spatial_probs))
                diagnostic_rows.append(
                    {
                        "row_index": int(row_index),
                        "query_id": str(row.get("query_id", "")),
                        "anchor_id": str(row.get("anchor_id", "")),
                        "track_id": str(row.get("track_id", "")),
                        "support_track_id": str(row.get("support_track_id", "")),
                        "target_is_dustbin": str(row.get("target_is_dustbin", "")),
                        "baseline_epe_px": baseline_value,
                        "likelihood_epe_px": epe,
                        "mode_epe_px": mode_epe,
                        "direct_epe_px": "" if direct_epe is None else direct_epe,
                        "improved": bool(epe < baseline_value),
                        "mode_improved": bool(mode_epe < baseline_value),
                        "direct_improved": "" if direct_epe is None else bool(direct_epe < baseline_value),
                        "center_x": center_x,
                        "center_y": center_y,
                        "query_pred_x": center_x + float(mean_xy[0].item()),
                        "query_pred_y": center_y + float(mean_xy[1].item()),
                        "query_mode_x": center_x + float(peak_xy[0].item()),
                        "query_mode_y": center_y + float(peak_xy[1].item()),
                        "query_direct_x": "" if direct_xy is None else center_x + float(direct_xy[0].item()),
                        "query_direct_y": "" if direct_xy is None else center_y + float(direct_xy[1].item()),
                        "query_gt_x": str(row.get("query_gt_x", "")),
                        "query_gt_y": str(row.get("query_gt_y", "")),
                        "target_dx": float(target_cpu[local_index, 0].item()),
                        "target_dy": float(target_cpu[local_index, 1].item()),
                        "pred_dx": float(mean_xy[0].item()),
                        "pred_dy": float(mean_xy[1].item()),
                        "peak_dx": float(peak_xy[0].item()),
                        "peak_dy": float(peak_xy[1].item()),
                        "direct_dx": "" if direct_xy is None else float(direct_xy[0].item()),
                        "direct_dy": "" if direct_xy is None else float(direct_xy[1].item()),
                        "dustbin_probability": float(dustbin_batch[local_index].item()),
                        "visualization": vis_path,
                    }
                )
    visual_candidates.sort(key=lambda item: item[0], reverse=True)
    row_to_vis: dict[int, str] = {}
    for rank, (_epe, row_index, _path, query_patch, render_patch, spatial_probs) in enumerate(visual_candidates[: int(visualize_limit)]):
        path = output / "visualizations" / f"worst_{rank:03d}_row{row_index:06d}.png"
        _save_visualization(path=path, query_patch=query_patch, render_patch=render_patch, spatial_probs=spatial_probs)
        row_to_vis[row_index] = str(path)
    for row in diagnostic_rows:
        row["visualization"] = row_to_vis.get(int(row["row_index"]), "")
    _write_csv(output / "diagnostic_rows.csv", diagnostic_rows)
    epe_values = np.asarray([float(row["likelihood_epe_px"]) for row in diagnostic_rows], dtype=np.float64)
    mode_epe_values = np.asarray([float(row["mode_epe_px"]) for row in diagnostic_rows], dtype=np.float64)
    direct_pairs = [
        (float(row["direct_epe_px"]), float(row["baseline_epe_px"]))
        for row in diagnostic_rows
        if str(row["direct_epe_px"]).strip()
    ]
    direct_epe_values = np.asarray([item[0] for item in direct_pairs], dtype=np.float64)
    direct_baseline_values = np.asarray([item[1] for item in direct_pairs], dtype=np.float64)
    baseline_values = np.asarray([float(row["baseline_epe_px"]) for row in diagnostic_rows], dtype=np.float64)
    summary = {
        "stage": "measurement_v1_rgb_patch_diagnostics",
        "row_count": int(len(diagnostic_rows)),
        "checkpoint": str(checkpoint),
        "rows_csv": str(rows_csv),
        "query_source": str(query_source),
        "support_patch_warp": str(support_patch_warp),
        "support_patch_source_audit": _support_patch_source_audit(rows, query_source=str(query_source)),
        "batch_size": int(batch),
        "prior_scale_key": str(prior_scale_key),
        "metrics": {
            "baseline_median_px": float(np.median(baseline_values)) if baseline_values.size else None,
            "likelihood_median_px": float(np.median(epe_values)) if epe_values.size else None,
            "likelihood_p90_px": float(np.percentile(epe_values, 90.0)) if epe_values.size else None,
            "mode_median_px": float(np.median(mode_epe_values)) if mode_epe_values.size else None,
            "mode_p90_px": float(np.percentile(mode_epe_values, 90.0)) if mode_epe_values.size else None,
            "direct_median_px": float(np.median(direct_epe_values)) if direct_epe_values.size else None,
            "direct_p90_px": float(np.percentile(direct_epe_values, 90.0)) if direct_epe_values.size else None,
            "improve_ratio": float(np.mean(epe_values < baseline_values)) if epe_values.size else None,
            "mode_improve_ratio": float(np.mean(mode_epe_values < baseline_values)) if mode_epe_values.size else None,
            "direct_improve_ratio": float(np.mean(direct_epe_values < direct_baseline_values)) if direct_epe_values.size else None,
            "recall_0p5px": float(np.mean(epe_values <= 0.5)) if epe_values.size else None,
            "recall_1px": float(np.mean(epe_values <= 1.0)) if epe_values.size else None,
            "mode_recall_0p5px": float(np.mean(mode_epe_values <= 0.5)) if mode_epe_values.size else None,
            "mode_recall_1px": float(np.mean(mode_epe_values <= 1.0)) if mode_epe_values.size else None,
        },
        "outputs": {
            "diagnostic_rows": str(output / "diagnostic_rows.csv"),
            "visualizations": str(output / "visualizations"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
