#!/usr/bin/env python3
"""Visualize current query/map features against cached RADIO teacher features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.tools.eval_cpr_buckets import build_model_and_data  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    load_config,
    move_batch_to_device,
    project_query_render_for_fine_selector,
)
from feature_field.utils.feature_track_vis import (  # noqa: E402
    error_to_heatmap_image,
    feature_group_to_rgb_images,
    save_feature_track_visual,
    tensor_to_display_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--query-fine-key", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--panel-scale", type=int, default=4)
    parser.add_argument(
        "--uniform-panel-size",
        default=None,
        help="Optional WIDTHxHEIGHT used to resize every feature/RGB panel before adding labels.",
    )
    parser.add_argument("--save-pairwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheet-cols", type=int, default=2)
    parser.add_argument(
        "--candidate-render-batch-size",
        type=int,
        default=None,
        help="Kept for compatibility with eval_cpr_buckets build_model_and_data.",
    )
    return parser.parse_args()


def _parse_panel_size(value: str | None) -> tuple[int, int] | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).lower().replace(",", "x")
    parts = [part for part in text.split("x") if part]
    if len(parts) != 2:
        raise ValueError("--uniform-panel-size must be formatted as WIDTHxHEIGHT")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--uniform-panel-size values must be positive")
    return width, height


def _resize_feature(feature: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(hw):
        return feature
    return F.interpolate(feature.float(), size=tuple(hw), mode="bilinear", align_corners=False)


def _resize_mask(mask: torch.Tensor | None, hw: tuple[int, int]) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask[:, None]
    if tuple(mask.shape[-2:]) != tuple(hw):
        mask = F.interpolate(mask.float(), size=tuple(hw), mode="nearest")
    return mask.float()


def _masked_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target_hw = tuple(second.shape[-2:])
    first = _resize_feature(first.float(), target_hw)
    second = second.float()
    channels = min(first.shape[1], second.shape[1])
    first = F.normalize(first[:, :channels], dim=1, eps=1e-6)
    second = F.normalize(second[:, :channels], dim=1, eps=1e-6)
    cosine = (first * second).sum(dim=1, keepdim=True)
    valid = _resize_mask(mask, target_hw)
    if valid is None:
        return cosine.flatten(1).mean(dim=1)
    valid = valid.to(device=cosine.device, dtype=cosine.dtype)
    denom = valid.flatten(1).sum(dim=1).clamp(min=1.0)
    return (cosine * valid).flatten(1).sum(dim=1) / denom


def _sample_stem(sample_name: str) -> str:
    return Path(sample_name).with_suffix("").as_posix().replace("/", "_")


def _tensor_or_none(batch: Dict, key: str, idx: int):
    if key not in batch:
        return None
    return batch[key][idx].detach().cpu()


def _feature_stats(feature: torch.Tensor) -> Dict:
    feature = feature.detach().float().cpu()
    if feature.dim() == 4:
        feature = feature[0]
    flat_chw = feature.reshape(feature.shape[0], -1)
    channel_std = flat_chw.std(dim=1)
    pixel_norm = flat_chw.norm(dim=0)
    return {
        "shape": [int(v) for v in feature.shape],
        "mean": float(feature.mean()),
        "std": float(feature.std()),
        "min": float(feature.min()),
        "max": float(feature.max()),
        "abs_mean": float(feature.abs().mean()),
        "channel_std_mean": float(channel_std.mean()),
        "channel_std_min": float(channel_std.min()),
        "channel_std_max": float(channel_std.max()),
        "pixel_norm_mean": float(pixel_norm.mean()),
        "pixel_norm_std": float(pixel_norm.std()),
    }


def _make_contact_sheet(image_paths: List[Path], output_path: Path, cols: int) -> None:
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w = min(900, max(image.width for image in images))
    thumbs = []
    for image, path in zip(images, image_paths):
        scale = thumb_w / float(image.width)
        thumb_h = max(1, int(round(image.height * scale)))
        thumb = image.resize((thumb_w, thumb_h), Image.BILINEAR)
        header_h = 24
        canvas = Image.new("RGB", (thumb_w, thumb_h + header_h), color=(0, 0, 0))
        canvas.paste(thumb, (0, header_h))
        ImageDraw.Draw(canvas).text((8, 5), path.name, fill=(255, 255, 255))
        thumbs.append(canvas)

    cols = max(1, int(cols))
    rows = (len(thumbs) + cols - 1) // cols
    cell_w = max(thumb.width for thumb in thumbs)
    cell_h = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        sheet.paste(thumb, (col * cell_w, row * cell_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _annotate_and_scale(
    image: Image.Image,
    title: str,
    scale: int,
    panel_size: tuple[int, int] | None = None,
) -> Image.Image:
    if panel_size is not None:
        image = image.resize(panel_size, Image.BILINEAR)
    scale = max(1, int(scale))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    header_h = 28
    canvas = Image.new("RGB", (image.width, image.height + header_h), color=(16, 16, 16))
    canvas.paste(image, (0, header_h))
    ImageDraw.Draw(canvas).text((8, 7), title, fill=(255, 255, 255))
    return canvas


def _panel_grid(panels: List[Image.Image], output_path: Path, *, cols: int = 4) -> None:
    cols = max(1, int(cols))
    rows = (len(panels) + cols - 1) // cols
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(0, 0, 0))
    for idx, panel in enumerate(panels):
        row, col = divmod(idx, cols)
        grid.paste(panel, (col * cell_w, row * cell_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def save_pairwise_comparison(
    output_path: Path,
    *,
    query_rgb: torch.Tensor,
    teacher_fine: torch.Tensor,
    student_fine: torch.Tensor,
    rendered_map_fine: torch.Tensor,
    rendered_map_fine_raw: torch.Tensor | None,
    teacher_coarse: torch.Tensor,
    student_coarse: torch.Tensor,
    rendered_map_coarse: torch.Tensor,
    rendered_map_mask: torch.Tensor | None,
    rendered_map_alpha: torch.Tensor | None,
    scale: int,
    panel_size: tuple[int, int] | None = None,
    projected_query_fine: torch.Tensor | None = None,
    projected_map_fine: torch.Tensor | None = None,
) -> None:
    feature_hw = tuple(teacher_fine.shape[-2:])
    query_rgb_small = query_rgb.detach().float()
    if query_rgb_small.dim() == 3:
        query_rgb_small = query_rgb_small.unsqueeze(0)
    if tuple(query_rgb_small.shape[-2:]) != feature_hw:
        query_rgb_small = F.interpolate(
            query_rgb_small,
            size=feature_hw,
            mode="bilinear",
            align_corners=False,
        )
    query_rgb_small = query_rgb_small.squeeze(0)

    student_teacher = feature_group_to_rgb_images([student_fine, teacher_fine], [None, None])
    map_internal = feature_group_to_rgb_images(
        [rendered_map_fine_raw, rendered_map_fine],
        [rendered_map_mask, rendered_map_mask],
    )
    map_teacher_cross = feature_group_to_rgb_images(
        [rendered_map_fine, teacher_fine],
        [rendered_map_mask, None],
    )
    coarse_student_teacher = feature_group_to_rgb_images([student_coarse, teacher_coarse], [None, None])
    coarse_map_teacher = feature_group_to_rgb_images(
        [rendered_map_coarse, teacher_coarse],
        [rendered_map_mask, None],
    )
    projected_pair = None
    if projected_query_fine is not None and projected_map_fine is not None:
        projected_pair = feature_group_to_rgb_images(
            [projected_query_fine, projected_map_fine],
            [None, rendered_map_mask],
        )

    panels = [
        _annotate_and_scale(tensor_to_display_rgb(query_rgb_small), "query_rgb", scale, panel_size),
        _annotate_and_scale(student_teacher[0], "student_fine_pairPCA", scale, panel_size),
        _annotate_and_scale(student_teacher[1], "teacher_fine_pairPCA", scale, panel_size),
        _annotate_and_scale(
            tensor_to_display_rgb(rendered_map_alpha if rendered_map_alpha is not None else rendered_map_mask),
            "map_alpha_or_mask",
            scale,
            panel_size,
        ),
        _annotate_and_scale(
            map_internal[0] if map_internal[0] is not None else map_internal[1],
            "map_fine_raw_internalPCA",
            scale,
            panel_size,
        ),
        _annotate_and_scale(map_internal[1], "map_fine_internalPCA", scale, panel_size),
        _annotate_and_scale(map_teacher_cross[0], "map_fine_vs_teacherPCA", scale, panel_size),
        _annotate_and_scale(map_teacher_cross[1], "teacher_for_mapPCA", scale, panel_size),
        _annotate_and_scale(coarse_student_teacher[0], "student_coarse_pairPCA", scale, panel_size),
        _annotate_and_scale(coarse_student_teacher[1], "teacher_coarse_pairPCA", scale, panel_size),
        _annotate_and_scale(coarse_map_teacher[0], "map_coarse_vs_teacherPCA", scale, panel_size),
        _annotate_and_scale(coarse_map_teacher[1], "teacher_coarse_for_mapPCA", scale, panel_size),
    ]
    if projected_pair is not None:
        panels.extend(
            [
                _annotate_and_scale(projected_pair[0], "projected_query_fine", scale, panel_size),
                _annotate_and_scale(projected_pair[1], "projected_map_fine", scale, panel_size),
                _annotate_and_scale(
                    error_to_heatmap_image(projected_map_fine, projected_query_fine, mask=rendered_map_mask),
                    "projected_map_vs_query_err",
                    scale,
                    panel_size,
                ),
            ]
        )
    _panel_grid(panels, output_path, cols=4)


def save_highres_comparison(
    output_path: Path,
    *,
    query_rgb: torch.Tensor,
    teacher_fine: torch.Tensor,
    student_fine: torch.Tensor,
    rendered_map_fine: torch.Tensor,
    rendered_map_fine_raw: torch.Tensor | None,
    teacher_coarse: torch.Tensor,
    student_coarse: torch.Tensor,
    rendered_map_coarse: torch.Tensor,
    rendered_map_mask: torch.Tensor | None,
    rendered_map_alpha: torch.Tensor | None,
    scale: int,
    panel_size: tuple[int, int] | None = None,
) -> None:
    feature_hw = tuple(teacher_fine.shape[-2:])
    query_rgb_small = query_rgb.detach().float()
    if query_rgb_small.dim() == 3:
        query_rgb_small = query_rgb_small.unsqueeze(0)
    if tuple(query_rgb_small.shape[-2:]) != feature_hw:
        query_rgb_small = F.interpolate(
            query_rgb_small,
            size=feature_hw,
            mode="bilinear",
            align_corners=False,
        )
    query_rgb_small = query_rgb_small.squeeze(0)

    fine_group = [teacher_fine, student_fine, rendered_map_fine_raw, rendered_map_fine]
    fine_masks = [None, None, rendered_map_mask, rendered_map_mask]
    fine_rgb = feature_group_to_rgb_images(fine_group, fine_masks)
    coarse_rgb = feature_group_to_rgb_images(
        [teacher_coarse, student_coarse, rendered_map_coarse],
        [None, None, rendered_map_mask],
    )

    fine_panels = [
        _annotate_and_scale(tensor_to_display_rgb(query_rgb_small), "query_rgb", scale, panel_size),
        _annotate_and_scale(fine_rgb[0], "RADIO_teacher_fine", scale, panel_size),
        _annotate_and_scale(fine_rgb[1], "query_student_fine", scale, panel_size),
        _annotate_and_scale(fine_rgb[2] if fine_rgb[2] is not None else fine_rgb[3], "map_fine_raw", scale, panel_size),
        _annotate_and_scale(fine_rgb[3], "map_fine", scale, panel_size),
        _annotate_and_scale(error_to_heatmap_image(student_fine, teacher_fine), "student_vs_RADIO_err", scale, panel_size),
        _annotate_and_scale(
            error_to_heatmap_image(rendered_map_fine, teacher_fine, mask=rendered_map_mask),
            "map_vs_RADIO_err",
            scale,
            panel_size,
        ),
        _annotate_and_scale(
            tensor_to_display_rgb(rendered_map_alpha if rendered_map_alpha is not None else rendered_map_mask),
            "map_alpha_or_mask",
            scale,
            panel_size,
        ),
    ]
    coarse_panels = [
        _annotate_and_scale(coarse_rgb[0], "RADIO_teacher_coarse", scale, panel_size),
        _annotate_and_scale(coarse_rgb[1], "query_student_coarse", scale, panel_size),
        _annotate_and_scale(coarse_rgb[2], "map_coarse", scale, panel_size),
        _annotate_and_scale(
            error_to_heatmap_image(rendered_map_coarse, teacher_coarse, mask=rendered_map_mask),
            "coarse_map_vs_RADIO_err",
            scale,
            panel_size,
        ),
    ]
    _panel_grid(fine_panels + coarse_panels, output_path, cols=4)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    uniform_panel_size = _parse_panel_size(args.uniform_panel_size)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, loader, map_renderer = build_model_and_data(cfg, args, device)
    query_fine_key = args.query_fine_key or cfg.get("map_supervision", {}).get("query_fine_key", "fine")
    use_amp = bool(cfg.get("training", {}).get("amp", True) and device.type == "cuda" and not args.no_amp)

    rows = []
    image_paths: List[Path] = []
    saved = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batch = map_renderer.attach_to_batch(batch, require_grad=False)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if bool(cfg.get("model", {}).get("teacher_fine_condition", False)):
                    outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
                else:
                    outputs = model(batch["rgb"])

            if query_fine_key not in outputs:
                available = ", ".join(sorted(outputs.keys()))
                raise KeyError(f"query_fine_key={query_fine_key!r} not found in outputs: {available}")

            student_radio_fine = _masked_cosine(outputs[query_fine_key], batch["teacher_fine"])
            map_radio_fine = _masked_cosine(
                batch["rendered_map_fine"],
                batch["teacher_fine"],
                mask=batch.get("rendered_map_mask"),
            )
            map_student_fine = _masked_cosine(
                batch["rendered_map_fine"],
                outputs[query_fine_key],
                mask=batch.get("rendered_map_mask"),
            )
            student_radio_coarse = _masked_cosine(outputs["coarse"], batch["teacher_coarse"])
            map_radio_coarse = _masked_cosine(
                batch["rendered_map_coarse"],
                batch["teacher_coarse"],
                mask=batch.get("rendered_map_mask"),
            )
            map_student_coarse = _masked_cosine(
                batch["rendered_map_coarse"],
                outputs["coarse"],
                mask=batch.get("rendered_map_mask"),
            )
            projected_query_fine = None
            projected_map_fine = None
            projected_map_student_fine = None
            projector = getattr(model, "local_corr_projector", None)
            if projector is not None:
                projected_query_fine, projected_map_fine, _used_projector = project_query_render_for_fine_selector(
                    projector,
                    outputs[query_fine_key].float(),
                    batch["rendered_map_fine"].float(),
                    render_chunk_size=int(cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0),
                )
                projected_map_student_fine = _masked_cosine(
                    projected_map_fine,
                    projected_query_fine,
                    mask=batch.get("rendered_map_mask"),
                )

            batch_size = batch["rgb"].shape[0]
            for idx in range(batch_size):
                if saved >= int(args.max_samples):
                    break
                stem = _sample_stem(batch["sample_name"][idx])
                image_path = out_dir / f"{saved:03d}_{stem}.png"
                save_feature_track_visual(
                    image_path,
                    query_rgb=batch["rgb"][idx].detach().cpu(),
                    teacher_fine=batch["teacher_fine"][idx].detach().cpu(),
                    student_fine=outputs[query_fine_key][idx].detach().cpu(),
                    teacher_coarse=batch["teacher_coarse"][idx].detach().cpu(),
                    student_coarse=outputs["coarse"][idx].detach().cpu(),
                    rendered_map_fine_raw=_tensor_or_none(batch, "rendered_map_fine_raw", idx),
                    rendered_map_fine=_tensor_or_none(batch, "rendered_map_fine", idx),
                    rendered_map_coarse=_tensor_or_none(batch, "rendered_map_coarse", idx),
                    rendered_map_mask=_tensor_or_none(batch, "rendered_map_mask", idx),
                    rendered_map_alpha=_tensor_or_none(batch, "rendered_map_alpha", idx),
                    prior_mask=_tensor_or_none(batch, "prior_mask", idx),
                    sample_name=batch["sample_name"][idx],
                )
                highres_path = out_dir / f"{saved:03d}_{stem}_highres.png"
                save_highres_comparison(
                    highres_path,
                    query_rgb=batch["rgb"][idx].detach().cpu(),
                    teacher_fine=batch["teacher_fine"][idx].detach().cpu(),
                    student_fine=outputs[query_fine_key][idx].detach().cpu(),
                    rendered_map_fine=batch["rendered_map_fine"][idx].detach().cpu(),
                    rendered_map_fine_raw=_tensor_or_none(batch, "rendered_map_fine_raw", idx),
                    teacher_coarse=batch["teacher_coarse"][idx].detach().cpu(),
                    student_coarse=outputs["coarse"][idx].detach().cpu(),
                    rendered_map_coarse=batch["rendered_map_coarse"][idx].detach().cpu(),
                    rendered_map_mask=_tensor_or_none(batch, "rendered_map_mask", idx),
                    rendered_map_alpha=_tensor_or_none(batch, "rendered_map_alpha", idx),
                    scale=int(args.panel_scale),
                    panel_size=uniform_panel_size,
                )
                pairwise_path = None
                if bool(args.save_pairwise):
                    pairwise_path = out_dir / f"{saved:03d}_{stem}_pairwise.png"
                    save_pairwise_comparison(
                        pairwise_path,
                        query_rgb=batch["rgb"][idx].detach().cpu(),
                        teacher_fine=batch["teacher_fine"][idx].detach().cpu(),
                        student_fine=outputs[query_fine_key][idx].detach().cpu(),
                        rendered_map_fine=batch["rendered_map_fine"][idx].detach().cpu(),
                        rendered_map_fine_raw=_tensor_or_none(batch, "rendered_map_fine_raw", idx),
                        teacher_coarse=batch["teacher_coarse"][idx].detach().cpu(),
                        student_coarse=outputs["coarse"][idx].detach().cpu(),
                        rendered_map_coarse=batch["rendered_map_coarse"][idx].detach().cpu(),
                        rendered_map_mask=_tensor_or_none(batch, "rendered_map_mask", idx),
                        rendered_map_alpha=_tensor_or_none(batch, "rendered_map_alpha", idx),
                        projected_query_fine=projected_query_fine[idx].detach().cpu()
                        if projected_query_fine is not None
                        else None,
                        projected_map_fine=projected_map_fine[idx].detach().cpu()
                        if projected_map_fine is not None
                        else None,
                        scale=int(args.panel_scale),
                        panel_size=uniform_panel_size,
                    )
                image_paths.append(pairwise_path if pairwise_path is not None else highres_path)
                rows.append(
                    {
                        "sample": batch["sample_name"][idx],
                        "image": str(image_path),
                        "highres_image": str(highres_path),
                        "pairwise_image": str(pairwise_path) if pairwise_path is not None else None,
                        "student_radio_fine_cosine": float(student_radio_fine[idx].detach().cpu()),
                        "map_radio_fine_cosine": float(map_radio_fine[idx].detach().cpu()),
                        "map_student_fine_cosine": float(map_student_fine[idx].detach().cpu()),
                        "projected_map_student_fine_cosine": float(projected_map_student_fine[idx].detach().cpu())
                        if projected_map_student_fine is not None
                        else None,
                        "student_radio_coarse_cosine": float(student_radio_coarse[idx].detach().cpu()),
                        "map_radio_coarse_cosine": float(map_radio_coarse[idx].detach().cpu()),
                        "map_student_coarse_cosine": float(map_student_coarse[idx].detach().cpu()),
                        "feature_stats": {
                            "teacher_fine": _feature_stats(batch["teacher_fine"][idx]),
                            "student_fine": _feature_stats(outputs[query_fine_key][idx]),
                            "map_fine_raw": _feature_stats(batch["rendered_map_fine_raw"][idx]),
                            "map_fine": _feature_stats(batch["rendered_map_fine"][idx]),
                            "teacher_coarse": _feature_stats(batch["teacher_coarse"][idx]),
                            "student_coarse": _feature_stats(outputs["coarse"][idx]),
                            "map_coarse": _feature_stats(batch["rendered_map_coarse"][idx]),
                            "projected_query_fine": _feature_stats(projected_query_fine[idx])
                            if projected_query_fine is not None
                            else None,
                            "projected_map_fine": _feature_stats(projected_map_fine[idx])
                            if projected_map_fine is not None
                            else None,
                        },
                    }
                )
                saved += 1
            if saved >= int(args.max_samples):
                break

    if not rows:
        raise RuntimeError("No samples were visualized")

    metric_keys = [key for key in rows[0].keys() if key.endswith("_cosine")]
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "map_checkpoint": str(Path(args.map_checkpoint).resolve()) if args.map_checkpoint else None,
        "dataset_feature_dir": str(cfg.get("dataset", {}).get("feature_dir")),
        "dataset_feature_hw": cfg.get("dataset", {}).get("feature_hw"),
        "dataset_coarse_feature_hw": cfg.get("dataset", {}).get("coarse_feature_hw"),
        "student_feature_hw": cfg.get("dataset", {}).get("student_feature_hw"),
        "student_coarse_feature_hw": cfg.get("dataset", {}).get("student_coarse_feature_hw"),
        "map_renderer_feature_hw": [int(v) for v in getattr(map_renderer, "feature_hw", [])],
        "split": args.split,
        "query_fine_key": query_fine_key,
        "num_samples": len(rows),
        "mean": {key: sum(float(row[key]) for row in rows) / len(rows) for key in metric_keys},
        "samples": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _make_contact_sheet(image_paths, out_dir / "contact_sheet.png", cols=args.contact_sheet_cols)
    print(json.dumps(summary["mean"], indent=2))
    print(f"wrote {len(rows)} feature visualizations to {out_dir}")


if __name__ == "__main__":
    main()
