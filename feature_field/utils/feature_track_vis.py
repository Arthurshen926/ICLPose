from pathlib import Path

import numpy as np
import torch
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


def error_to_heatmap_image(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Image.Image:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    if pred.dim() == 4:
        pred = pred[0]
    if target.dim() == 4:
        target = target[0]

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
    sample_name: str = "",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = []
    if rendered_map_fine is None or rendered_map_coarse is None:
        panels.extend(
            [
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(feature_to_rgb_image(teacher_fine), "teacher_fine"),
                _annotate(feature_to_rgb_image(student_fine), "student_fine"),
                _annotate(error_to_heatmap_image(student_fine, teacher_fine), "fine_error"),
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(feature_to_rgb_image(teacher_coarse), "teacher_coarse"),
                _annotate(feature_to_rgb_image(student_coarse), "student_coarse"),
                _annotate(error_to_heatmap_image(student_coarse, teacher_coarse), "coarse_error"),
            ]
        )
    else:
        panels.extend(
            [
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(feature_to_rgb_image(teacher_fine), "teacher_fine"),
                _annotate(feature_to_rgb_image(student_fine), "student_fine"),
                _annotate(
                    feature_to_rgb_image(rendered_map_fine_raw, mask=rendered_map_mask),
                    "map_fine_raw",
                )
                if rendered_map_fine_raw is not None
                else _annotate(feature_to_rgb_image(rendered_map_fine, mask=rendered_map_mask), "map_fine"),
                _annotate(feature_to_rgb_image(rendered_map_fine, mask=rendered_map_mask), "map_fine"),
                _annotate(error_to_heatmap_image(student_fine, teacher_fine), "fine_student_err"),
                _annotate(
                    error_to_heatmap_image(rendered_map_fine_raw, teacher_fine, mask=rendered_map_mask),
                    "fine_map_raw_err",
                )
                if rendered_map_fine_raw is not None
                else _annotate(error_to_heatmap_image(rendered_map_fine, teacher_fine, mask=rendered_map_mask), "fine_map_err"),
                _annotate(error_to_heatmap_image(rendered_map_fine, teacher_fine, mask=rendered_map_mask), "fine_map_err"),
                _annotate(tensor_to_display_rgb(query_rgb), "query_rgb"),
                _annotate(feature_to_rgb_image(teacher_coarse), "teacher_coarse"),
                _annotate(feature_to_rgb_image(student_coarse), "student_coarse"),
                _annotate(feature_to_rgb_image(rendered_map_coarse, mask=rendered_map_mask), "map_coarse"),
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
