from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from feature_extract.vfm.dense_depth_measurement_fusion import dense_depth_measurement_summary
from feature_extract.vfm.measurement_v1.rgb_patch_coarse_to_fine import (
    CoarseToFineRGBPatchMeasurementBranch,
    load_coarse_to_fine_rgb_patch_measurement_branch,
    likelihood_mode,
    likelihood_moments,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import crop_rgb_window
from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import (
    RGB_PATCH_FUSION_FIELDNAMES,
    _fieldnames_for_rows,
    _optional_float,
    _query_center,
    _read_csv_rows,
    _write_jsonl,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    _load_query_rgb,
    _load_render_rgb,
    _load_tensor_cached,
    _prior_scale_batch,
    _render_cache_by_query,
)


COARSE_TO_FINE_FIELDNAMES = [
    *RGB_PATCH_FUSION_FIELDNAMES,
    "coarse_measurement_dx",
    "coarse_measurement_dy",
    "coarse_recenter_dx",
    "coarse_recenter_dy",
    "fine_measurement_dx",
    "fine_measurement_dy",
    "coarse_measurement_sigma_px",
    "fine_measurement_sigma_px",
    "coarse_recenter_head",
    "coarse_to_fine_enabled",
]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _load_query_image(row: Mapping[str, object], *, image_root: Path, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        raise ValueError("match row missing query_id")
    path = Path(image_root) / query_id
    return _load_tensor_cached(cache, str(path), lambda p=path: _load_query_rgb(p))


def _load_render_image(
    row: Mapping[str, object],
    *,
    render_cache_by_query: Mapping[str, Path],
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        raise ValueError("match row missing query_id")
    path = render_cache_by_query.get(query_id)
    if path is None:
        raise ValueError(f"missing render cache for query_id={query_id}")
    return _load_tensor_cached(cache, str(path), lambda p=path: _load_render_rgb(p))


def _render_anchor(row: Mapping[str, object]) -> tuple[float, float]:
    x = _optional_float(row, "render_x")
    y = _optional_float(row, "render_y")
    if x is None or y is None:
        raise ValueError("match row missing render_x/render_y")
    return float(x), float(y)


def _sigma(cov: torch.Tensor) -> float:
    cov_cpu = cov.detach().cpu().reshape(2, 2)
    return float(math.sqrt(max(0.5 * (float(cov_cpu[0, 0]) + float(cov_cpu[1, 1])), 1e-12)))


def apply_coarse_to_fine_rgb_patch_measurements_to_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    image_root: Path,
    render_cache_by_query: Mapping[str, Path],
    model: CoarseToFineRGBPatchMeasurementBranch,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    batch_size: int = 8,
    device: torch.device | str = "cuda",
    coarse_recenter_head: str | None = None,
    coarse_prior_scale_key: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model = model.to(torch_device).eval()
    if coarse_recenter_head is not None:
        recenter = str(coarse_recenter_head)
        if recenter not in {"likelihood", "mode"}:
            raise ValueError("coarse_recenter_head must be 'likelihood' or 'mode'")
        model.coarse_recenter_head = recenter
    output = [dict(row) for row in rows]
    query_cache: dict[str, torch.Tensor] = {}
    render_cache: dict[str, torch.Tensor] = {}
    batch = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(output), batch):
            batch_rows = output[start : start + batch]
            coarse_query_patches: list[torch.Tensor] = []
            coarse_render_patches: list[torch.Tensor] = []
            centers: list[tuple[float, float]] = []
            anchors: list[tuple[float, float]] = []
            for row in batch_rows:
                query_image = _load_query_image(row, image_root=Path(image_root), cache=query_cache).unsqueeze(0)
                render_image = _load_render_image(row, render_cache_by_query=render_cache_by_query, cache=render_cache).unsqueeze(0)
                center = _query_center(row)
                anchor = _render_anchor(row)
                centers.append(center)
                anchors.append(anchor)
                coarse_query_patch, _ = crop_rgb_window(
                    query_image,
                    torch.tensor([center], dtype=torch.float32),
                    radius_px=float(model.coarse.crop_radius_px),
                    step_px=float(model.coarse.step_px),
                    image_width=int(query_image_width),
                    image_height=int(query_image_height),
                )
                coarse_render_patch, _ = crop_rgb_window(
                    render_image,
                    torch.tensor([anchor], dtype=torch.float32),
                    radius_px=float(model.coarse.crop_radius_px),
                    step_px=float(model.coarse.step_px),
                    image_width=int(render_image_width),
                    image_height=int(render_image_height),
                )
                coarse_query_patches.append(coarse_query_patch[0])
                coarse_render_patches.append(coarse_render_patch[0])
            coarse_prior_scale = _prior_scale_batch(batch_rows, prior_scale_key=str(coarse_prior_scale_key))
            coarse_pred = model.coarse.forward_from_patches(
                torch.stack(coarse_query_patches, dim=0).to(torch_device),
                torch.stack(coarse_render_patches, dim=0).to(torch_device),
                prior_scale_px=None if coarse_prior_scale is None else coarse_prior_scale.to(torch_device),
            )
            coarse_mean, coarse_cov_gpu = likelihood_moments(coarse_pred.logits, coarse_pred.offsets_xy)
            if model.coarse_recenter_head == "mode":
                recenter_gpu = likelihood_mode(coarse_pred.logits, coarse_pred.offsets_xy)
            else:
                recenter_gpu = coarse_mean
            recenter_cpu = recenter_gpu.detach().cpu()
            fine_query_patches: list[torch.Tensor] = []
            fine_render_patches: list[torch.Tensor] = []
            for local_index, row in enumerate(batch_rows):
                query_image = _load_query_image(row, image_root=Path(image_root), cache=query_cache).unsqueeze(0)
                render_image = _load_render_image(row, render_cache_by_query=render_cache_by_query, cache=render_cache).unsqueeze(0)
                cx, cy = centers[local_index]
                anchor = anchors[local_index]
                fine_center = (
                    float(cx + float(recenter_cpu[local_index, 0])),
                    float(cy + float(recenter_cpu[local_index, 1])),
                )
                fine_query_patch, _ = crop_rgb_window(
                    query_image,
                    torch.tensor([fine_center], dtype=torch.float32),
                    radius_px=float(model.fine.crop_radius_px),
                    step_px=float(model.fine.step_px),
                    image_width=int(query_image_width),
                    image_height=int(query_image_height),
                )
                fine_render_patch, _ = crop_rgb_window(
                    render_image,
                    torch.tensor([anchor], dtype=torch.float32),
                    radius_px=float(model.fine.crop_radius_px),
                    step_px=float(model.fine.step_px),
                    image_width=int(render_image_width),
                    image_height=int(render_image_height),
                )
                fine_query_patches.append(fine_query_patch[0])
                fine_render_patches.append(fine_render_patch[0])
            fine_pred = model.fine.forward_from_patches(
                torch.stack(fine_query_patches, dim=0).to(torch_device),
                torch.stack(fine_render_patches, dim=0).to(torch_device),
            )
            fine_mean, fine_cov_gpu = likelihood_moments(fine_pred.logits, fine_pred.offsets_xy)
            final_gpu = recenter_gpu + fine_mean
            final_cov_gpu = coarse_cov_gpu + fine_cov_gpu
            coarse_dustbin = torch.sigmoid(coarse_pred.dustbin_logit.reshape(-1))
            fine_dustbin = torch.sigmoid(fine_pred.dustbin_logit.reshape(-1))
            valid_gpu = (1.0 - coarse_dustbin) * (1.0 - fine_dustbin)
            final = final_gpu.detach().cpu()
            coarse = coarse_mean.detach().cpu()
            recenter = recenter_gpu.detach().cpu()
            fine = fine_mean.detach().cpu()
            valid = valid_gpu.detach().cpu()
            final_cov = final_cov_gpu.detach().cpu()
            coarse_cov = coarse_cov_gpu.detach().cpu()
            fine_cov = fine_cov_gpu.detach().cpu()
            for local_index, row in enumerate(batch_rows):
                cx, cy = centers[local_index]
                dx = float(final[local_index, 0])
                dy = float(final[local_index, 1])
                cov = final_cov[local_index]
                row["query_center_x"] = row.get("query_center_x", row.get("center_x", row.get("query_x", cx)))
                row["query_center_y"] = row.get("query_center_y", row.get("center_y", row.get("query_y", cy)))
                row["query_refined_x"] = float(cx + dx)
                row["query_refined_y"] = float(cy + dy)
                row["measurement_dx"] = dx
                row["measurement_dy"] = dy
                row["measurement_cov_xx"] = float(cov[0, 0])
                row["measurement_cov_xy"] = float(cov[0, 1])
                row["measurement_cov_yy"] = float(cov[1, 1])
                row["measurement_sigma_px"] = _sigma(cov)
                row["measurement_valid_prob"] = float(valid[local_index])
                row["coarse_measurement_dx"] = float(coarse[local_index, 0])
                row["coarse_measurement_dy"] = float(coarse[local_index, 1])
                row["coarse_recenter_dx"] = float(recenter[local_index, 0])
                row["coarse_recenter_dy"] = float(recenter[local_index, 1])
                row["fine_measurement_dx"] = float(fine[local_index, 0])
                row["fine_measurement_dy"] = float(fine[local_index, 1])
                row["coarse_measurement_sigma_px"] = _sigma(coarse_cov[local_index])
                row["fine_measurement_sigma_px"] = _sigma(fine_cov[local_index])
                row["measurement_search_radius_px"] = float(model.coarse.search_radius_px)
                row["measurement_context_radius_px"] = float(model.coarse.context_radius_px)
                row["measurement_step_px"] = float(model.coarse.step_px)
                row["rgb_patch_prediction_head"] = "coarse_to_fine_likelihood"
                row["coarse_recenter_head"] = str(model.coarse_recenter_head)
                row["coarse_to_fine_enabled"] = True
    summary = dense_depth_measurement_summary(output)
    summary.update(
        {
            "stage": "measurement_v1_rgb_patch_coarse_to_fine_match_table_fusion",
            "row_count": int(len(output)),
            "batch_size": int(batch),
            "coarse_search_radius_px": float(model.coarse.search_radius_px),
            "coarse_step_px": float(model.coarse.step_px),
            "fine_search_radius_px": float(model.fine.search_radius_px),
            "fine_step_px": float(model.fine.step_px),
            "coarse_recenter_head": str(model.coarse_recenter_head),
            "coarse_prior_scale_key": str(coarse_prior_scale_key),
        }
    )
    return output, summary


def apply_coarse_to_fine_rgb_patch_measurements_to_match_table(
    *,
    match_table_csv: Path,
    render_cache_manifest_csv: Path,
    image_root: Path,
    coarse_checkpoint: Path,
    fine_checkpoint: Path,
    output_dir: Path,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    batch_size: int = 8,
    device: str = "cuda",
    base_dir: Path | None = None,
    max_rows: int | None = None,
    coarse_recenter_head: str | None = None,
    coarse_prior_scale_key: str = "",
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    rows = _read_csv_rows(Path(match_table_csv), max_rows=max_rows)
    render_map = _render_cache_by_query(Path(render_cache_manifest_csv), base_dir=base)
    model = load_coarse_to_fine_rgb_patch_measurement_branch(
        coarse_checkpoint=Path(coarse_checkpoint),
        fine_checkpoint=Path(fine_checkpoint),
        device=torch_device,
    )
    fused_rows, summary = apply_coarse_to_fine_rgb_patch_measurements_to_rows(
        rows,
        image_root=Path(image_root),
        render_cache_by_query=render_map,
        model=model,
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        batch_size=int(batch_size),
        device=torch_device,
        coarse_recenter_head=coarse_recenter_head,
        coarse_prior_scale_key=str(coarse_prior_scale_key),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames_for_rows(fused_rows)
    for name in COARSE_TO_FINE_FIELDNAMES:
        if name not in fieldnames:
            fieldnames.append(name)
    _write_csv(output / "match_table.csv", fused_rows, fieldnames)
    _write_jsonl(output / "match_table.jsonl", fused_rows)
    summary = {
        **summary,
        "match_table_csv": str(match_table_csv),
        "render_cache_manifest_csv": str(render_cache_manifest_csv),
        "image_root": str(image_root),
        "coarse_checkpoint": str(coarse_checkpoint),
        "fine_checkpoint": str(fine_checkpoint),
        "coarse_prior_scale_key": str(coarse_prior_scale_key),
        "outputs": {
            "match_table_csv": str(output / "match_table.csv"),
            "match_table_jsonl": str(output / "match_table.jsonl"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
