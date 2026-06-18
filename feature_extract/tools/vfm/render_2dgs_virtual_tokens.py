"""Render virtual Cambridge poses with 2DGS and extract VFM token manifests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _extract_radio_feature_from_rgb
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord, parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.official_2dgs_renderer import (
    Official2DGSSource,
    load_official_2dgs_source_from_ply,
    render_official_2dgs_rgb_depth,
)
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


RenderRgbDepthFn = Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _safe_token_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__")


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("rendered rgb must have shape (H, W, 3)")
    if value.dtype == np.uint8:
        return value
    scaled = value.astype(np.float32, copy=False)
    if float(np.nanmax(scaled, initial=0.0)) <= 1.0:
        scaled = scaled * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgb_to_uint8(rgb), mode="RGB").save(output)


def _alpha_stats(alpha: np.ndarray) -> tuple[float, float]:
    values = np.asarray(alpha, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("rendered alpha must have shape (H, W)")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite)), float(np.mean(finite > 1e-4))


def _depth_stats(depth: np.ndarray) -> tuple[float, float]:
    values = np.asarray(depth, dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.median(valid)), float(np.mean(valid > 0.0))


def _select_pose_records(
    records: Sequence[CambridgePoseRecord],
    *,
    start_index: int,
    max_poses: int,
    view_selection: str,
) -> tuple[CambridgePoseRecord, ...]:
    values = tuple(records)[max(0, int(start_index)) :]
    if int(max_poses) <= 0 or len(values) <= int(max_poses):
        return values
    if str(view_selection) == "prefix":
        return values[: int(max_poses)]
    if str(view_selection) == "uniform":
        indices = np.linspace(0, len(values) - 1, int(max_poses), dtype=np.int64)
        return tuple(values[int(index)] for index in indices)
    raise ValueError("view_selection must be one of: prefix, uniform")


def render_virtual_pose_token_manifest(
    *,
    records: Sequence[CambridgePoseRecord],
    source: Official2DGSSource | object,
    camera: ColmapCamera,
    width: int,
    height: int,
    extractor,
    render_rgb_depth_fn: RenderRgbDepthFn,
    image_output_root: Path | None,
    token_output_root: Path,
    scene: str,
    split: str,
    layer_name: str,
    model_name: str,
    storage_dtype: str,
    renderer_name: str,
    source_path: str,
    min_alpha_coverage: float = 0.0,
    max_poses: int = 0,
    start_index: int = 0,
    view_selection: str = "prefix",
) -> TokenBankManifest:
    """Render virtual poses and write a token manifest aligned to the pose file."""

    selected = _select_pose_records(
        records,
        start_index=int(start_index),
        max_poses=int(max_poses),
        view_selection=str(view_selection),
    )
    if not selected:
        raise ValueError("at least one pose record is required")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    if float(min_alpha_coverage) < 0.0 or float(min_alpha_coverage) > 1.0:
        raise ValueError("min_alpha_coverage must be in [0, 1]")
    dtype = np.dtype(storage_dtype)
    layer_specs = (
        TokenLayerSpec(
            name=str(layer_name),
            model=str(model_name),
            layer="final",
            channels=1280,
            stride=16,
        ),
    )
    token_root = Path(token_output_root)
    image_root = None if image_output_root is None else Path(image_output_root)
    output_records: list[TokenBankRecord] = []
    skipped_low_alpha = 0
    for ordinal, record in enumerate(selected):
        rgb, depth, alpha = render_rgb_depth_fn(
            source=source,
            record=record,
            pose_w2c=record.pose_w2c,
            camera=camera,
            width=int(width),
            height=int(height),
        )
        rgb_u8 = _rgb_to_uint8(rgb)
        alpha_mean, alpha_coverage = _alpha_stats(alpha)
        if alpha_coverage < float(min_alpha_coverage):
            skipped_low_alpha += 1
            continue
        if image_root is not None:
            _write_rgb(image_root / record.image_id, rgb_u8)
        feature = _extract_radio_feature_from_rgb(rgb_u8, extractor)
        token_path = token_root / f"{_safe_token_name(record.image_id)}.npz"
        write_npz_token_record(token_path, {str(layer_name): np.asarray(feature, dtype=dtype)})
        depth_median, depth_valid_ratio = _depth_stats(depth)
        output_records.append(
            TokenBankRecord(
                image_id=record.image_id,
                token_path=token_path,
                layers=layer_specs,
                split=str(split),
                scene=str(scene),
                checksum=compute_file_sha256(token_path),
                metadata={
                    "renderer": str(renderer_name),
                    "source_path": str(source_path),
                    "render_width": int(width),
                    "render_height": int(height),
                    "pose_ordinal": int(max(0, int(start_index)) + ordinal),
                    "view_selection": str(view_selection),
                    "camera_center": np.asarray(record.camera_center, dtype=np.float64).tolist(),
                    "alpha_mean": float(alpha_mean),
                    "alpha_coverage": float(alpha_coverage),
                    "depth_median": float(depth_median),
                    "depth_valid_ratio": float(depth_valid_ratio),
                },
            )
        )
    if not output_records:
        raise ValueError(f"all rendered poses were filtered; skipped_low_alpha={skipped_low_alpha}")
    manifest = TokenBankManifest(records=tuple(output_records))
    manifest.validate(verify_checksums=True)
    return manifest


def _render_official_wrapper(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return render_official_2dgs_rgb_depth(
        kwargs["source"],
        pose_w2c=kwargs["pose_w2c"],
        camera=kwargs["camera"],
        width=int(kwargs["width"]),
        height=int(kwargs["height"]),
        device=str(kwargs.get("device", "cuda")),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--token_output_root", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--image_output_root", default="")
    parser.add_argument("--scene", default="OldHospital")
    parser.add_argument("--split", default="virtual")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1920,1080,1400.0,960.0,540.0,0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--storage_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--min_alpha_coverage", type=float, default=0.0)
    parser.add_argument("--max_poses", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = parse_cambridge_pose_file(Path(args.pose_file))
    camera_model_dir = _infer_camera_model_dir(args.pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    render_camera = _scale_camera(camera, int(args.render_width), int(args.render_height))
    source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
    from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

    radio = RADIOFeatureExtractor(version=args.radio_version, device=args.device, radio_repo=args.radio_repo)

    def render_fn(**kwargs):
        return render_official_2dgs_rgb_depth(
            kwargs["source"],
            pose_w2c=kwargs["pose_w2c"],
            camera=kwargs["camera"],
            width=int(kwargs["width"]),
            height=int(kwargs["height"]),
            device=str(args.device),
        )

    manifest = render_virtual_pose_token_manifest(
        records=records,
        source=source,
        camera=render_camera,
        width=int(args.render_width),
        height=int(args.render_height),
        extractor=radio,
        render_rgb_depth_fn=render_fn,
        image_output_root=Path(args.image_output_root) if args.image_output_root else None,
        token_output_root=Path(args.token_output_root),
        scene=str(args.scene),
        split=str(args.split),
        layer_name=str(args.layer_name),
        model_name=str(args.radio_version),
        storage_dtype=str(args.storage_dtype),
        renderer_name="official_2dgs",
        source_path=str(args.gaussian_rgb_ply),
        min_alpha_coverage=float(args.min_alpha_coverage),
        max_poses=int(args.max_poses),
        start_index=int(args.start_index),
        view_selection=str(args.view_selection),
    )
    manifest_path = Path(args.output_manifest)
    manifest.to_json(manifest_path)
    summary = {
        "pose_file": str(args.pose_file),
        "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
        "output_manifest": str(manifest_path),
        "token_output_root": str(args.token_output_root),
        "image_output_root": str(args.image_output_root),
        "scene": str(args.scene),
        "split": str(args.split),
        "renderer": "official_2dgs",
        "camera_source": str(camera_source),
        "render_width": int(args.render_width),
        "render_height": int(args.render_height),
        "input_pose_count": len(records),
        "rendered_pose_count": len(manifest.records),
        "min_alpha_coverage": float(args.min_alpha_coverage),
        "view_selection": str(args.view_selection),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    output_summary = Path(args.summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
