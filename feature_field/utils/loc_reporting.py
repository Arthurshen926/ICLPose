"""
Reusable helpers for compact localization experiment reporting.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional here
    torch = None


PathLike = Union[str, Path]
ImageInput = Union[PathLike, np.ndarray, "torch.Tensor", Image.Image]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, full_key))
        else:
            flat[full_key] = value
    return flat


def format_metric_value(value: Any) -> str:
    value = _to_serializable(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_metric_value(v) for v in value)
    return str(value)


def save_metrics(
    metrics: Mapping[str, Any],
    output_dir: PathLike,
    json_name: str = "metrics.json",
    text_name: str = "metrics.txt",
) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    serializable = _to_serializable(metrics)
    json_path = output_dir / json_name
    text_path = output_dir / text_name

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    flat_metrics = flatten_metrics(serializable)
    with text_path.open("w", encoding="utf-8") as f:
        for key in sorted(flat_metrics):
            f.write(f"{key}: {format_metric_value(flat_metrics[key])}\n")

    return {"json": json_path, "text": text_path}


def _normalize_image_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.ndim != 3:
        raise ValueError(f"Unsupported image array shape: {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 4:
        array = array[..., :3]

    array = array.astype(np.float32)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        array = np.zeros_like(array, dtype=np.uint8)
        return array

    if array.max() <= 1.0 and array.min() >= 0.0:
        array = array * 255.0
    else:
        lo = float(array[finite_mask].min())
        hi = float(array[finite_mask].max())
        if hi > lo:
            array = (array - lo) / (hi - lo) * 255.0
        else:
            array = np.zeros_like(array)

    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        return array
    return array[..., :3]


def image_input_to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if torch is not None and isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray):
        return Image.fromarray(_normalize_image_array(image))

    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def save_side_by_side_panel(
    images: Sequence[ImageInput],
    output_path: PathLike,
    captions: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    max_panel_height: int = 360,
    padding: int = 16,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    if not images:
        raise ValueError("At least one image is required to build a qualitative panel.")

    pil_images = [image_input_to_pil(image) for image in images]
    captions = list(captions) if captions is not None else [""] * len(pil_images)
    if len(captions) != len(pil_images):
        raise ValueError("Number of captions must match number of images.")

    font = ImageFont.load_default()
    title_height = 28 if title else 0
    caption_height = 24 if any(captions) else 0

    resized: List[Image.Image] = []
    for image in pil_images:
        width, height = image.size
        if height > max_panel_height:
            scale = max_panel_height / float(height)
            image = image.resize((max(1, int(width * scale)), max_panel_height), Image.BILINEAR)
        resized.append(image)

    panel_width = padding + sum(img.width for img in resized) + padding * (len(resized))
    panel_height = (
        padding
        + title_height
        + (padding if title else 0)
        + max(img.height for img in resized)
        + (padding if caption_height else 0)
        + caption_height
        + padding
    )

    canvas = Image.new("RGB", (panel_width, panel_height), background)
    draw = ImageDraw.Draw(canvas)
    y_offset = padding

    if title:
        draw.text((padding, y_offset), title, fill=(0, 0, 0), font=font)
        y_offset += title_height + padding

    max_height = max(img.height for img in resized)
    x_offset = padding
    for image, caption in zip(resized, captions):
        top = y_offset + (max_height - image.height) // 2
        canvas.paste(image, (x_offset, top))
        if caption_height and caption:
            text_y = y_offset + max_height + padding
            draw.text((x_offset, text_y), caption, fill=(0, 0, 0), font=font)
        x_offset += image.width + padding

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    canvas.save(output_path)
    return output_path


def build_progress_report(
    exp_name: str,
    metrics: Mapping[str, Any],
    summary_lines: Optional[Sequence[str]] = None,
    notes: Optional[Sequence[str]] = None,
    artifact_paths: Optional[Sequence[PathLike]] = None,
    panel_paths: Optional[Sequence[PathLike]] = None,
) -> Dict[str, str]:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    serializable_metrics = _to_serializable(metrics)
    flat_metrics = flatten_metrics(serializable_metrics)

    summary_lines = list(summary_lines or [])
    notes = list(notes or [])
    artifact_paths = [str(Path(p)) for p in (artifact_paths or [])]
    panel_paths = [str(Path(p)) for p in (panel_paths or [])]

    if not summary_lines:
        ordered_keys = sorted(flat_metrics)[: min(8, len(flat_metrics))]
        summary_lines = [f"{key}: {format_metric_value(flat_metrics[key])}" for key in ordered_keys]

    metric_lines = [f"- `{key}`: {format_metric_value(flat_metrics[key])}" for key in sorted(flat_metrics)]

    markdown_lines = [
        f"# Localization Report: {exp_name}",
        "",
        f"- Generated: {timestamp}",
        f"- Experiment: `{exp_name}`",
        "",
        "## Progress Summary",
    ]
    markdown_lines.extend(f"- {line}" for line in summary_lines)

    markdown_lines.extend(["", "## Metrics", *metric_lines])

    if panel_paths:
        markdown_lines.extend(["", "## Qualitative Visualizations"])
        markdown_lines.extend(f"- `{panel}`" for panel in panel_paths)

    if artifact_paths:
        markdown_lines.extend(["", "## Artifacts"])
        markdown_lines.extend(f"- `{artifact}`" for artifact in artifact_paths)

    if notes:
        markdown_lines.extend(["", "## Notes"])
        markdown_lines.extend(f"- {note}" for note in notes)

    text_lines = [
        f"Localization Report: {exp_name}",
        f"Generated: {timestamp}",
        "",
        "Progress Summary:",
        *[f"  - {line}" for line in summary_lines],
        "",
        "Metrics:",
        *[f"  - {key}: {format_metric_value(flat_metrics[key])}" for key in sorted(flat_metrics)],
    ]

    if panel_paths:
        text_lines.extend(["", "Qualitative Visualizations:"])
        text_lines.extend(f"  - {panel}" for panel in panel_paths)

    if artifact_paths:
        text_lines.extend(["", "Artifacts:"])
        text_lines.extend(f"  - {artifact}" for artifact in artifact_paths)

    if notes:
        text_lines.extend(["", "Notes:"])
        text_lines.extend(f"  - {note}" for note in notes)

    return {
        "markdown": "\n".join(markdown_lines).rstrip() + "\n",
        "text": "\n".join(text_lines).rstrip() + "\n",
    }


def save_progress_report(
    report: Mapping[str, str],
    output_dir: PathLike,
    markdown_name: str = "progress_report.md",
    text_name: str = "progress_report.txt",
) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    markdown_path = output_dir / markdown_name
    text_path = output_dir / text_name

    with markdown_path.open("w", encoding="utf-8") as f:
        f.write(report["markdown"])

    with text_path.open("w", encoding="utf-8") as f:
        f.write(report["text"])

    return {"markdown": markdown_path, "text": text_path}


def create_report_bundle(
    exp_name: str,
    output_dir: PathLike,
    metrics: Mapping[str, Any],
    summary_lines: Optional[Sequence[str]] = None,
    notes: Optional[Sequence[str]] = None,
    panel_images: Optional[Sequence[ImageInput]] = None,
    panel_captions: Optional[Sequence[str]] = None,
    panel_title: Optional[str] = None,
    panel_name: str = "qualitative_panel.png",
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    metrics_paths = save_metrics(metrics, output_dir)

    panel_paths: List[Path] = []
    if panel_images:
        panel_path = save_side_by_side_panel(
            images=panel_images,
            output_path=output_dir / panel_name,
            captions=panel_captions,
            title=panel_title,
        )
        panel_paths.append(panel_path)

    artifact_paths = list(metrics_paths.values()) + panel_paths
    report = build_progress_report(
        exp_name=exp_name,
        metrics=metrics,
        summary_lines=summary_lines,
        notes=notes,
        artifact_paths=artifact_paths,
        panel_paths=panel_paths,
    )
    report_paths = save_progress_report(report, output_dir)

    return {
        "output_dir": output_dir,
        "metrics": metrics_paths,
        "report": report_paths,
        "panels": panel_paths,
    }


def save_experiment_bundle(
    exp_name: str,
    output_dir: PathLike,
    metrics: Mapping[str, Any],
    summary_lines: Optional[Sequence[str]] = None,
    notes: Optional[Sequence[str]] = None,
    artifact_paths: Optional[Sequence[PathLike]] = None,
    panel_images: Optional[Sequence[ImageInput]] = None,
    panel_captions: Optional[Sequence[str]] = None,
    panel_title: Optional[str] = None,
    panel_name: str = "qualitative_panel.png",
    results_json_name: str = "results.json",
    results_text_name: str = "results.txt",
    report_markdown_name: str = "report.md",
    report_text_name: str = "report.txt",
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    results_paths = save_metrics(
        metrics,
        output_dir,
        json_name=results_json_name,
        text_name=results_text_name,
    )

    panel_paths: List[Path] = []
    if panel_images:
        panel_path = save_side_by_side_panel(
            images=panel_images,
            output_path=output_dir / panel_name,
            captions=panel_captions,
            title=panel_title,
        )
        panel_paths.append(panel_path)

    bundle_artifacts = list(results_paths.values())
    if artifact_paths:
        bundle_artifacts.extend(Path(p) for p in artifact_paths)
    bundle_artifacts.extend(panel_paths)

    report = build_progress_report(
        exp_name=exp_name,
        metrics=metrics,
        summary_lines=summary_lines,
        notes=notes,
        artifact_paths=bundle_artifacts,
        panel_paths=panel_paths,
    )
    report_paths = save_progress_report(
        report,
        output_dir,
        markdown_name=report_markdown_name,
        text_name=report_text_name,
    )

    return {
        "output_dir": output_dir,
        "results": results_paths,
        "report": report_paths,
        "panels": panel_paths,
        "artifacts": [Path(p) for p in bundle_artifacts],
    }
