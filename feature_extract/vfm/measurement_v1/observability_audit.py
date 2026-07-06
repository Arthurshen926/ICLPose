from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import template_search_cost_volume_logits
from feature_extract.vfm.measurement_v1.rgb_patch_training import _read_csv, _render_cache_by_query, _stack_patch_batch


OBSERVABILITY_CLASSES = (
    "observable_subpixel",
    "observable_coarse_only",
    "ambiguous",
    "domain_mismatch",
    "textureless",
    "window_out",
)


AUDIT_FIELDNAMES = [
    "row_index",
    "query_id",
    "match_index",
    "requested_residual_px",
    "center_residual_px",
    "target_is_dustbin",
    "gt_in_window",
    "observability_class",
    "train_weight",
    "center_x",
    "center_y",
    "query_gt_x",
    "query_gt_y",
    "render_x",
    "render_y",
    "target_dx",
    "target_dy",
    "mode_dx",
    "mode_dy",
    "mode_epe_px",
    "gt_rank",
    "gt_probability",
    "mode_probability",
    "entropy_norm",
    "peak_gap_z",
    "logit_std",
    "render_texture",
    "radio_match_score",
]


def classify_observability(
    *,
    gt_in_window: bool,
    render_texture: float,
    mode_epe_px: float,
    gt_rank: int,
    entropy_norm: float,
    peak_gap_z: float,
    texture_threshold: float = 0.03,
    subpixel_epe_threshold_px: float = 0.5,
    coarse_epe_threshold_px: float = 4.0,
    coarse_rank_threshold: int = 8,
    ambiguous_entropy_threshold: float = 0.90,
    ambiguous_peak_gap_z_threshold: float = 0.25,
) -> str:
    if not bool(gt_in_window):
        return "window_out"
    if float(render_texture) < float(texture_threshold):
        return "textureless"
    if float(mode_epe_px) <= float(subpixel_epe_threshold_px) and int(gt_rank) <= 1:
        return "observable_subpixel"
    if float(mode_epe_px) <= float(coarse_epe_threshold_px) and int(gt_rank) <= int(coarse_rank_threshold):
        return "observable_coarse_only"
    if float(entropy_norm) >= float(ambiguous_entropy_threshold) or float(peak_gap_z) <= float(ambiguous_peak_gap_z_threshold):
        return "ambiguous"
    return "domain_mismatch"


def _train_weight(label: str) -> float:
    if label == "observable_subpixel":
        return 1.0
    if label == "observable_coarse_only":
        return 0.5
    if label == "ambiguous":
        return 0.1
    if label == "domain_mismatch":
        return 0.05
    return 0.0


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    text = str(row.get(key, "")).strip()
    return float(text) if text else float(default)


def _bool_text(row: Mapping[str, object], key: str) -> bool:
    text = str(row.get(key, "")).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _prepare_features(patch: torch.Tensor, *, input_mode: str) -> torch.Tensor:
    mode = str(input_mode)
    values = patch.float()
    if mode == "rgb":
        return values
    gray = 0.2989 * values[:, 0:1] + 0.5870 * values[:, 1:2] + 0.1140 * values[:, 2:3]
    if mode == "norm_graygrad":
        mean = torch.mean(gray, dim=(2, 3), keepdim=True)
        std = torch.std(gray, dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        gray = (gray - mean) / std
    dx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
    dy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
    if mode in {"graygrad", "norm_graygrad"}:
        return torch.cat([gray, dx, dy], dim=1)
    if mode == "rgb_graygrad":
        return torch.cat([values, gray, dx, dy], dim=1)
    raise ValueError("input_mode must be 'rgb', 'graygrad', 'norm_graygrad', or 'rgb_graygrad'")


def _render_template_texture(render_patch: torch.Tensor, *, search_radius_px: float, context_radius_px: float, step_px: float) -> torch.Tensor:
    if render_patch.ndim != 4 or int(render_patch.shape[1]) != 3:
        raise ValueError("render_patch must have shape (B,3,H,W)")
    search_steps = int(round(float(search_radius_px) / float(step_px)))
    context_steps = int(round(float(context_radius_px) / float(step_px)))
    crop_steps = search_steps + context_steps
    center = crop_steps
    template = render_patch[:, :, center - context_steps : center + context_steps + 1, center - context_steps : center + context_steps + 1].float()
    gray = 0.2989 * template[:, 0:1] + 0.5870 * template[:, 1:2] + 0.1140 * template[:, 2:3]
    dx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
    dy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
    return torch.mean(torch.sqrt(dx * dx + dy * dy + 1e-12), dim=(1, 2, 3))


def _summarise(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    counts = {name: 0 for name in OBSERVABILITY_CLASSES}
    for row in rows:
        label = str(row.get("observability_class", ""))
        if label in counts:
            counts[label] += 1
    row_count = int(len(rows))
    mode_epes = np.asarray([float(row["mode_epe_px"]) for row in rows], dtype=np.float64) if rows else np.asarray([], dtype=np.float64)
    trainable_count = counts["observable_subpixel"] + counts["observable_coarse_only"]
    return {
        "row_count": row_count,
        "class_counts": counts,
        "class_fractions": {key: (float(value) / float(row_count) if row_count else 0.0) for key, value in counts.items()},
        "trainable_count": int(trainable_count),
        "trainable_fraction": float(trainable_count) / float(row_count) if row_count else 0.0,
        "mode_epe_median_px": float(np.median(mode_epes)) if mode_epes.size else None,
        "mode_epe_p90_px": float(np.percentile(mode_epes, 90.0)) if mode_epes.size else None,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _merge_original_and_audit_rows(
    original_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    if len(original_rows) != len(audit_rows):
        raise ValueError("original_rows and audit_rows must have the same length")
    original_fieldnames: list[str] = []
    for row in original_rows:
        for key in row.keys():
            if key not in original_fieldnames:
                original_fieldnames.append(str(key))
    appended = [name for name in AUDIT_FIELDNAMES if name not in original_fieldnames]
    fieldnames = original_fieldnames + appended
    merged: list[dict[str, object]] = []
    for original, audit in zip(original_rows, audit_rows):
        row = dict(original)
        for key in appended:
            row[key] = audit.get(key, "")
        merged.append(row)
    return merged, fieldnames


def export_measurement_observability_audit(
    *,
    rows_csv: Path,
    render_cache_manifest_csv: Path | None,
    image_root: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    input_mode: str = "norm_graygrad",
    temperature: float = 10.0,
    template_scale_factors: Sequence[float] = (1.0,),
    query_source: str = "real",
    support_patch_warp: str = "none",
    max_rows: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    base_dir: Path | None = None,
    cache_images_on_device: bool = True,
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    render_map = _render_cache_by_query(None if render_cache_manifest_csv is None else Path(render_cache_manifest_csv), base_dir=base)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    image_cache_device = torch_device if bool(cache_images_on_device) else None
    q_width = int(query_image_width if query_image_width is not None else image_width)
    q_height = int(query_image_height if query_image_height is not None else image_height)
    r_width = int(render_image_width if render_image_width is not None else image_width)
    r_height = int(render_image_height if render_image_height is not None else image_height)
    query_cache: dict[str, torch.Tensor] = {}
    render_cache: dict[str, torch.Tensor] = {}
    out_rows: list[dict[str, object]] = []
    batch = max(1, int(batch_size))
    with torch.no_grad():
        for batch_start in range(0, len(rows), batch):
            batch_rows = rows[batch_start : batch_start + batch]
            query_patch, render_patch, target, baseline, target_is_dustbin = _stack_patch_batch(
                batch_rows,
                image_root=Path(image_root),
                render_cache_by_query=render_map,
                image_width=int(image_width),
                image_height=int(image_height),
                query_image_width=int(q_width),
                query_image_height=int(q_height),
                render_image_width=int(r_width),
                render_image_height=int(r_height),
                crop_radius_px=float(search_radius_px) + float(context_radius_px),
                step_px=float(step_px),
                query_cache=query_cache,
                render_cache=render_cache,
                query_source=str(query_source),
                support_patch_warp=str(support_patch_warp),
                image_cache_device=image_cache_device,
            )
            query_features = _prepare_features(query_patch.to(torch_device), input_mode=str(input_mode))
            render_features = _prepare_features(render_patch.to(torch_device), input_mode=str(input_mode))
            logits, offsets = template_search_cost_volume_logits(
                query_features,
                render_features,
                search_radius_px=float(search_radius_px),
                context_radius_px=float(context_radius_px),
                step_px=float(step_px),
                temperature=float(temperature),
                template_scale_factors=tuple(float(value) for value in template_scale_factors),
            )
            probs = torch.softmax(logits, dim=1)
            logit_std = torch.std(logits, dim=1, unbiased=False).clamp_min(1e-6)
            top2 = torch.topk(logits, k=2, dim=1).values
            peak_gap_z = (top2[:, 0] - top2[:, 1]) / logit_std
            entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)
            entropy_norm = entropy / max(math.log(float(logits.shape[1])), 1e-12)
            mode_idx = torch.argmax(probs, dim=1)
            offsets_device = offsets.to(device=torch_device, dtype=target.dtype)
            mode_offsets = offsets_device[mode_idx]
            target_device = target.to(torch_device)
            distances_to_gt = torch.linalg.norm(offsets_device.reshape(1, -1, 2) - target_device.reshape(-1, 1, 2), dim=2)
            nearest_gt_idx = torch.argmin(distances_to_gt, dim=1)
            nearest_gt_distance = distances_to_gt[torch.arange(int(distances_to_gt.shape[0]), device=torch_device), nearest_gt_idx]
            order = torch.argsort(logits, dim=1, descending=True)
            inverse_rank = torch.empty_like(order)
            rank_values = torch.arange(1, int(order.shape[1]) + 1, device=torch_device).reshape(1, -1).expand_as(order)
            inverse_rank.scatter_(1, order, rank_values)
            gt_rank = inverse_rank[torch.arange(int(order.shape[0]), device=torch_device), nearest_gt_idx]
            gt_probability = probs[torch.arange(int(order.shape[0]), device=torch_device), nearest_gt_idx]
            mode_probability = probs[torch.arange(int(order.shape[0]), device=torch_device), mode_idx]
            mode_epe = torch.linalg.norm(mode_offsets - target_device, dim=1)
            texture = _render_template_texture(
                render_patch.to(torch_device),
                search_radius_px=float(search_radius_px),
                context_radius_px=float(context_radius_px),
                step_px=float(step_px),
            )
            explicit_dustbin = (
                torch.zeros((len(batch_rows),), dtype=torch.bool)
                if target_is_dustbin is None
                else target_is_dustbin.detach().cpu().bool()
            )
            for local_idx, row in enumerate(batch_rows):
                gt_in_window = bool((nearest_gt_distance[local_idx].detach().cpu().item() <= 0.5 * float(step_px) + 1e-6) and not bool(explicit_dustbin[local_idx]))
                label = classify_observability(
                    gt_in_window=gt_in_window,
                    render_texture=float(texture[local_idx].detach().cpu().item()),
                    mode_epe_px=float(mode_epe[local_idx].detach().cpu().item()),
                    gt_rank=int(gt_rank[local_idx].detach().cpu().item()),
                    entropy_norm=float(entropy_norm[local_idx].detach().cpu().item()),
                    peak_gap_z=float(peak_gap_z[local_idx].detach().cpu().item()),
                )
                out_rows.append(
                    {
                        "row_index": int(batch_start + local_idx),
                        "query_id": str(row.get("query_id", "")),
                        "match_index": str(row.get("match_index", "")),
                        "requested_residual_px": str(row.get("requested_residual_px", "")),
                        "center_residual_px": float(baseline[local_idx].detach().cpu().item()),
                        "target_is_dustbin": bool(explicit_dustbin[local_idx]),
                        "gt_in_window": bool(gt_in_window),
                        "observability_class": label,
                        "train_weight": _train_weight(label),
                        "center_x": _float(row, "center_x"),
                        "center_y": _float(row, "center_y"),
                        "query_gt_x": _float(row, "query_gt_x"),
                        "query_gt_y": _float(row, "query_gt_y"),
                        "render_x": _float(row, "render_x"),
                        "render_y": _float(row, "render_y"),
                        "target_dx": float(target[local_idx, 0].detach().cpu().item()),
                        "target_dy": float(target[local_idx, 1].detach().cpu().item()),
                        "mode_dx": float(mode_offsets[local_idx, 0].detach().cpu().item()),
                        "mode_dy": float(mode_offsets[local_idx, 1].detach().cpu().item()),
                        "mode_epe_px": float(mode_epe[local_idx].detach().cpu().item()),
                        "gt_rank": int(gt_rank[local_idx].detach().cpu().item()),
                        "gt_probability": float(gt_probability[local_idx].detach().cpu().item()),
                        "mode_probability": float(mode_probability[local_idx].detach().cpu().item()),
                        "entropy_norm": float(entropy_norm[local_idx].detach().cpu().item()),
                        "peak_gap_z": float(peak_gap_z[local_idx].detach().cpu().item()),
                        "logit_std": float(logit_std[local_idx].detach().cpu().item()),
                        "render_texture": float(texture[local_idx].detach().cpu().item()),
                        "radio_match_score": str(row.get("radio_match_score", "")),
                    }
                )
    output = Path(output_dir)
    rows_out = output / "observability_rows.csv"
    trainable_out = output / "trainable_rows.csv"
    weighted_out = output / "weighted_rows.csv"
    trainable_weighted_out = output / "trainable_weighted_rows.csv"
    weighted_rows, weighted_fieldnames = _merge_original_and_audit_rows(rows, out_rows)
    _write_csv(rows_out, out_rows, fieldnames=AUDIT_FIELDNAMES)
    _write_csv(
        trainable_out,
        [row for row in out_rows if str(row.get("observability_class", "")) in {"observable_subpixel", "observable_coarse_only"}],
        fieldnames=AUDIT_FIELDNAMES,
    )
    trainable_labels = {"observable_subpixel", "observable_coarse_only"}
    _write_csv(weighted_out, weighted_rows, fieldnames=weighted_fieldnames)
    _write_csv(
        trainable_weighted_out,
        [row for row in weighted_rows if str(row.get("observability_class", "")) in trainable_labels],
        fieldnames=weighted_fieldnames,
    )
    summary = {
        "stage": "measurement_v1_observability_audit",
        "rows_csv": str(rows_csv),
        "render_cache_manifest_csv": "" if render_cache_manifest_csv is None else str(render_cache_manifest_csv),
        "image_root": str(image_root),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "query_image_width": int(q_width),
        "query_image_height": int(q_height),
        "render_image_width": int(r_width),
        "render_image_height": int(r_height),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "step_px": float(step_px),
        "input_mode": str(input_mode),
        "temperature": float(temperature),
        "template_scale_factors": [float(value) for value in template_scale_factors],
        "query_source": str(query_source),
        "batch_size": int(batch),
        "device": str(torch_device),
        "cache_images_on_device": bool(cache_images_on_device),
        **_summarise(out_rows),
        "outputs": {
            "rows_csv": str(rows_out),
            "trainable_rows_csv": str(trainable_out),
            "weighted_rows_csv": str(weighted_out),
            "trainable_weighted_rows_csv": str(trainable_weighted_out),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
