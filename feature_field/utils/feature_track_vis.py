from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _mask_to_numpy(mask, hw):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu()
        if mask.dim() == 4:
            mask = mask[0]
        if mask.dim() == 3:
            mask = mask[0]
        if tuple(mask.shape[-2:]) != tuple(hw):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=tuple(hw),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        mask = mask.numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape != tuple(hw):
        raise ValueError(f"Expected mask shape {tuple(hw)}, got {mask.shape}")
    return mask > 0.5


def _normalize_map(array: np.ndarray, percentile: float = 2.0, mask: np.ndarray = None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if array.ndim == 2:
            valid = array[mask]
        elif array.ndim == 3:
            valid = array[mask]
        else:
            raise ValueError(f"Unsupported array ndim for mask-aware normalization: {array.ndim}")
        if valid.size == 0:
            return np.zeros_like(array, dtype=np.float32)
    else:
        valid = array

    if percentile <= 0.0:
        if array.ndim >= 3 and array.shape[-1] in (3, 4):
            lo = valid.min(axis=0, keepdims=True)
            hi = valid.max(axis=0, keepdims=True)
        else:
            lo = float(valid.min())
            hi = float(valid.max())
    elif array.ndim >= 3 and array.shape[-1] in (3, 4):
        lo = np.percentile(valid, percentile, axis=0, keepdims=True)
        hi = np.percentile(valid, 100.0 - percentile, axis=0, keepdims=True)
    else:
        lo = float(np.percentile(valid, percentile))
        hi = float(np.percentile(valid, 100.0 - percentile))

    scale = np.maximum(hi - lo, 1e-8)
    normalized = np.clip((array - lo) / scale, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        if normalized.ndim == 2:
            normalized[~mask] = 0.0
        else:
            normalized[~mask] = 0.0
    return normalized


def tensor_to_display_rgb(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[0] == 1:
        arr = _normalize_map(tensor.squeeze(0).numpy())
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        arr = tensor[:3].permute(1, 2, 0).numpy()
        if arr.max() > 1.5 or arr.min() < -0.5:
            arr = _normalize_map(arr)
        rgb = np.clip(arr, 0.0, 1.0)

    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def feature_to_rgb_image(features: torch.Tensor, mask: torch.Tensor = None) -> Image.Image:
    features = features.detach().float().cpu()
    if features.dim() == 4:
        features = features[0]

    c, h, w = features.shape
    flat = features.reshape(c, -1).transpose(0, 1)
    valid_mask = _mask_to_numpy(mask, (h, w))
    if valid_mask is not None:
        valid_flat = torch.from_numpy(valid_mask.reshape(-1))
        if bool(valid_flat.any()):
            valid_values = flat[valid_flat]
        else:
            valid_values = flat
    else:
        valid_flat = None
        valid_values = flat

    mean = valid_values.mean(dim=0, keepdim=True)
    flat_centered = flat - mean
    valid_centered = valid_values - mean

    q = min(3, valid_centered.shape[0], valid_centered.shape[1])
    if q == 0:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
    else:
        try:
            _, _, v = torch.pca_lowrank(valid_centered, q=q)
            proj = flat_centered @ v[:, :q]
        except RuntimeError:
            _, _, v = torch.linalg.svd(valid_centered, full_matrices=False)
            proj = flat_centered @ v[:q].transpose(0, 1)

        proj = proj.numpy()
        if q < 3:
            proj = np.pad(proj, ((0, 0), (0, 3 - q)))
        proj = proj[:, :3]
        proj = _normalize_map(proj, mask=valid_mask.reshape(-1) if valid_mask is not None else None)
        rgb = proj.reshape(h, w, 3)

    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def feature_group_to_rgb_images(
    features_list: list[torch.Tensor | None],
    masks: list[torch.Tensor | None] | None = None,
) -> list[Image.Image | None]:
    if masks is None:
        masks = [None] * len(features_list)
    if len(masks) != len(features_list):
        raise ValueError("masks must match features_list length")

    prepared = []
    valid_chunks = []
    for features, mask in zip(features_list, masks):
        if features is None:
            prepared.append(None)
            continue
        feat = features.detach().float().cpu()
        if feat.dim() == 4:
            feat = feat[0]
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        c, h, w = feat.shape
        flat = feat.reshape(c, -1).transpose(0, 1)
        valid_mask = _mask_to_numpy(mask, (h, w))
        if valid_mask is not None and bool(valid_mask.any()):
            valid_values = flat[torch.from_numpy(valid_mask.reshape(-1))]
        else:
            valid_values = flat
            valid_mask = None if valid_mask is None else valid_mask
        prepared.append((feat, flat, valid_mask))
        if valid_values.numel() > 0:
            valid_chunks.append(valid_values)

    if not valid_chunks:
        return [None if feat is None else feature_to_rgb_image(feat) for feat in features_list]

    stacked = torch.cat(valid_chunks, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    centered = stacked - mean
    q = min(3, centered.shape[0], centered.shape[1])

    if q == 0:
        return [None if feat is None else Image.fromarray(np.zeros((feat.shape[-2], feat.shape[-1], 3), dtype=np.uint8))
                for feat in features_list]

    try:
        _, _, v = torch.pca_lowrank(centered, q=q)
        basis = v[:, :q]
    except RuntimeError:
        _, _, v = torch.linalg.svd(centered, full_matrices=False)
        basis = v[:q].transpose(0, 1)

    stacked_proj = centered @ basis
    if q < 3:
        stacked_proj = torch.cat(
            [stacked_proj, torch.zeros(stacked_proj.shape[0], 3 - q, dtype=stacked_proj.dtype)],
            dim=1,
        )
    value_min = stacked_proj[:, :3].min(dim=0).values
    value_max = stacked_proj[:, :3].max(dim=0).values
    value_scale = torch.clamp(value_max - value_min, min=1e-8)

    images: list[Image.Image | None] = []
    for item in prepared:
        if item is None:
            images.append(None)
            continue
        feat, flat, valid_mask = item
        proj = (flat - mean) @ basis
        if q < 3:
            proj = torch.cat([proj, torch.zeros(proj.shape[0], 3 - q, dtype=proj.dtype)], dim=1)
        proj = ((proj[:, :3] - value_min) / value_scale).clamp(0.0, 1.0).numpy()
        rgb = proj.reshape(feat.shape[-2], feat.shape[-1], 3)
        if valid_mask is not None:
            rgb[~valid_mask] = 0.0
        images.append(Image.fromarray((rgb * 255.0).astype(np.uint8)))
    return images


def error_to_heatmap_image(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Image.Image:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    if pred.dim() == 4:
        pred = pred[0]
    if target.dim() == 4:
        target = target[0]
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(
            pred.unsqueeze(0),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    h, w = pred.shape[-2:]
    valid_mask = _mask_to_numpy(mask, (h, w))
    err = (pred - target).abs().mean(dim=0).numpy()
    err = _normalize_map(err, mask=valid_mask)
    rgb = np.stack(
        [
            err,
            np.sqrt(np.clip(err, 0.0, 1.0)),
            1.0 - err,
        ],
        axis=-1,
    )
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))


def _annotate(image: Image.Image, title: str) -> Image.Image:
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), color=(16, 16, 16))
    canvas.paste(image, (0, pad))
    drawer = ImageDraw.Draw(canvas)
    drawer.text((8, 4), title, fill=(255, 255, 255))
    return canvas


def save_feature_track_visual(
    output_path,
    query_rgb: torch.Tensor,
    teacher_fine: torch.Tensor,
    student_fine: torch.Tensor,
    teacher_coarse: torch.Tensor,
    student_coarse: torch.Tensor,
    rendered_map_fine: torch.Tensor = None,
    rendered_map_coarse: torch.Tensor = None,
    rendered_map_fine_raw: torch.Tensor = None,
    rendered_map_mask: torch.Tensor = None,
    rendered_map_alpha: torch.Tensor = None,
    prior_mask: torch.Tensor = None,
    sample_name: str = "",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fine_group = [teacher_fine, student_fine, rendered_map_fine_raw, rendered_map_fine]
    fine_masks = [None, None, rendered_map_mask, rendered_map_mask]
    fine_rgb = feature_group_to_rgb_images(fine_group, fine_masks)
    coarse_group = [teacher_coarse, student_coarse, rendered_map_coarse]
    coarse_masks = [None, None, rendered_map_mask]
    coarse_rgb = feature_group_to_rgb_images(coarse_group, coarse_masks)

    panels = []
    if rendered_map_fine is None or rendered_map_coarse is None:
        panels.extend(
            [
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(fine_rgb[0], "teacher_fine"),
                _annotate(fine_rgb[1], "student_fine"),
                _annotate(error_to_heatmap_image(student_fine, teacher_fine), "fine_error"),
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(coarse_rgb[0], "teacher_coarse"),
                _annotate(coarse_rgb[1], "student_coarse"),
                _annotate(error_to_heatmap_image(student_coarse, teacher_coarse), "coarse_error"),
            ]
        )
    else:
        panels.extend(
            [
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(fine_rgb[0], "teacher_fine"),
                _annotate(fine_rgb[1], "student_fine"),
                _annotate(
                    fine_rgb[2],
                    "map_fine_raw",
                )
                if rendered_map_fine_raw is not None
                else _annotate(fine_rgb[3], "map_fine"),
                _annotate(fine_rgb[3], "map_fine"),
                _annotate(error_to_heatmap_image(student_fine, teacher_fine), "fine_student_err"),
                _annotate(
                    error_to_heatmap_image(rendered_map_fine_raw, teacher_fine, mask=rendered_map_mask),
                    "fine_map_raw_err",
                )
                if rendered_map_fine_raw is not None
                else _annotate(error_to_heatmap_image(rendered_map_fine, teacher_fine, mask=rendered_map_mask), "fine_map_err"),
                _annotate(error_to_heatmap_image(rendered_map_fine, teacher_fine, mask=rendered_map_mask), "fine_map_err"),
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(coarse_rgb[0], "teacher_coarse"),
                _annotate(coarse_rgb[1], "student_coarse"),
                _annotate(coarse_rgb[2], "map_coarse"),
                _annotate(tensor_to_display_rgb(rendered_map_alpha), "map_alpha")
                if rendered_map_alpha is not None
                else _annotate(tensor_to_display_rgb(rendered_map_mask), "map_mask"),
                _annotate(tensor_to_display_rgb(rendered_map_mask), "map_mask"),
                _annotate(tensor_to_display_rgb(prior_mask), "prior_mask")
                if prior_mask is not None
                else _annotate(tensor_to_display_rgb(rendered_map_mask), "prior_mask(n/a)"),
                _annotate(error_to_heatmap_image(student_coarse, teacher_coarse), "coarse_student_err"),
                _annotate(error_to_heatmap_image(rendered_map_coarse, teacher_coarse, mask=rendered_map_mask), "coarse_map_err"),
            ]
        )

    cell_w = min(panel.width for panel in panels)
    cell_h = min(panel.height for panel in panels)
    resized = [panel.resize((cell_w, cell_h), Image.BILINEAR) for panel in panels]

    n_cols = 4
    n_rows = (len(resized) + n_cols - 1) // n_cols
    grid = Image.new("RGB", (cell_w * n_cols, cell_h * n_rows), color=(0, 0, 0))
    for idx, panel in enumerate(resized):
        row, col = divmod(idx, n_cols)
        grid.paste(panel, (col * cell_w, row * cell_h))

    if sample_name:
        header_h = 28
        canvas = Image.new("RGB", (grid.width, grid.height + header_h), color=(0, 0, 0))
        canvas.paste(grid, (0, header_h))
        drawer = ImageDraw.Draw(canvas)
        drawer.text((8, 6), sample_name, fill=(255, 255, 255))
        grid = canvas

    grid.save(output_path)
