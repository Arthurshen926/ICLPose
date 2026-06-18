"""Build render-RGB RADIO keypoint samples for geometry-aware descriptor adaptation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence
import zipfile

import numpy as np

from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _make_keypoint_detector,
    _parse_default_camera,
    _read_rgb,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianRGBSource,
    GaussianVFMField,
    GaussianVFMRenderConfig,
    load_gaussian_rgb_source_from_ply,
    render_gaussian_rgb_image_gsplat,
    render_gaussian_rgb_image_soft,
    render_gaussian_vfm_feature_map,
)
from feature_extract.vfm.official_2dgs_renderer import (
    Official2DGSSource,
    load_official_2dgs_source_from_ply,
    render_official_2dgs_rgb_depth,
)
from feature_extract.vfm.patch_selector_training import (
    append_patch_selector_training_set_capped,
    save_patch_selector_training_set_npz,
)
from feature_extract.vfm.render_rgb_keypoint_samples import (
    build_render_rgb_keypoint_adapter_samples,
    render_rgb_keypoint_pair_diagnostics,
)
from feature_extract.vfm.rendered_keypoint_matching import bilinear_sample_feature_map
from feature_extract.vfm.rendered_keypoint_selector_samples import (
    RenderedKeypointSelectorSampleConfig,
    render_keypoint_reprojection_errors,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _select_records(records, max_queries: int, mode: str, start_index: int = 0) -> list:
    values = list(records)[max(0, int(start_index)) :]
    if int(max_queries) <= 0 or len(values) <= int(max_queries):
        return values
    if mode == "uniform":
        indices = np.linspace(0, len(values) - 1, int(max_queries), dtype=np.int64)
        return [values[int(idx)] for idx in indices]
    return values[: int(max_queries)]


def _safe_image_stem(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__")


def _resolve_render_size(camera, render_width: int, render_height: int) -> tuple[int, int]:
    """Resolve render size; zero or negative values inherit the COLMAP camera size."""

    width = int(render_width) if int(render_width) > 0 else int(camera.width)
    height = int(render_height) if int(render_height) > 0 else int(camera.height)
    if width <= 0 or height <= 0:
        raise ValueError("resolved render width and height must be positive")
    return width, height


def _load_or_render_rgb_depth_cache(
    *,
    cache_path: Path | None,
    render_fn,
    skip_existing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached rendered RGB/depth/alpha or render and cache them."""

    if cache_path is not None and bool(skip_existing) and cache_path.exists():
        try:
            with np.load(cache_path) as data:
                return (
                    np.asarray(data["rgb"], dtype=np.uint8),
                    np.asarray(data["depth"], dtype=np.float32),
                    np.asarray(data["alpha"], dtype=np.float32),
                )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            Path(cache_path).unlink(missing_ok=True)
    rgb, depth, alpha = render_fn()
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_npz_compressed_atomic(cache_path, rgb=rgb, depth=depth, alpha=alpha)
    return rgb, depth, alpha


def _render_token_cache_path(cache_dir: Path, image_id: str, render_width: int, render_height: int) -> Path:
    return Path(cache_dir) / f"{_safe_image_stem(image_id)}_{int(render_width)}x{int(render_height)}.npz"


def _extract_radio_feature_from_rgb(rgb: np.ndarray, extractor) -> np.ndarray:
    import torch

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    output = extractor.extract(tensor)
    return output["local"].detach().cpu().numpy().astype(np.float32, copy=False)


def _save_npz_compressed_atomic(path: Path, **arrays: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.tmp")
    np.savez_compressed(tmp_path, **arrays)
    generated = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    generated.replace(output)


def _depth_field_from_rgb_source(source: GaussianRGBSource) -> GaussianVFMField:
    count = int(source.xyz.shape[0])
    return GaussianVFMField(
        xyz=source.xyz,
        features=np.ones((count, 1), dtype=np.float32),
        opacity=source.opacity,
        scale=source.scale,
        gaussian_indices=source.gaussian_indices,
        nearest_track_ids=np.full((count,), -1, dtype=np.int64),
        support_counts=np.ones((count,), dtype=np.int64),
        mean_distances=np.zeros((count,), dtype=np.float32),
        metadata={"source": "rgb_gaussian_depth_proxy"},
    )


def _render_rgb_and_depth(
    source: GaussianRGBSource | Official2DGSSource,
    depth_field: GaussianVFMField | None,
    *,
    pose_w2c: np.ndarray,
    camera,
    config: GaussianVFMRenderConfig,
    renderer: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if renderer == "official_2dgs":
        if not isinstance(source, Official2DGSSource):
            raise TypeError("renderer=official_2dgs requires an Official2DGSSource")
        rgb, depth, alpha = render_official_2dgs_rgb_depth(
            source,
            pose_w2c=pose_w2c,
            camera=camera,
            width=int(config.width),
            height=int(config.height),
            device=str(device),
        )
        rgb_u8 = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
        return rgb_u8, np.asarray(depth, dtype=np.float32), np.asarray(alpha, dtype=np.float32)
    if depth_field is None:
        raise ValueError("diagnostic Gaussian renderers require a depth_field")
    if renderer == "gsplat":
        rgb, alpha = render_gaussian_rgb_image_gsplat(source, pose_w2c, camera, config, device=device)
    else:
        rgb, alpha = render_gaussian_rgb_image_soft(source, pose_w2c, camera, config)
    depth_render = render_gaussian_vfm_feature_map(depth_field, pose_w2c=pose_w2c, camera=camera, config=config)
    rgb_u8 = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    return rgb_u8, depth_render.depth, np.asarray(alpha, dtype=np.float32)


def _load_or_extract_render_feature(
    *,
    cache_path: Path | None,
    render_rgb: np.ndarray,
    extractor,
    layer_name: str,
    skip_existing: bool,
) -> np.ndarray:
    if cache_path is not None and skip_existing and cache_path.exists():
        try:
            with np.load(cache_path) as data:
                return np.asarray(data[layer_name], dtype=np.float32)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            Path(cache_path).unlink(missing_ok=True)
    feature = _extract_radio_feature_from_rgb(render_rgb, extractor)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_npz_compressed_atomic(cache_path, **{layer_name: feature})
    return feature


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--extract_query_features_from_image", action="store_true")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=0)
    parser.add_argument("--render_height", type=int, default=0)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--render_depth_epsilon", type=float, default=0.02)
    parser.add_argument("--renderer", default="soft", choices=("soft", "gsplat", "official_2dgs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--render_token_cache_dir", default="")
    parser.add_argument("--skip_existing_render_tokens", action="store_true")
    parser.add_argument("--render_rgb_depth_cache_dir", default="")
    parser.add_argument("--skip_existing_render_rgb_depth", action="store_true")
    parser.add_argument("--detector", default="superpoint", choices=("orb", "superpoint", "disk"))
    parser.add_argument("--max_keypoints", type=int, default=1200)
    parser.add_argument("--max_queries", type=int, default=32)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--positive_threshold_px", type=float, default=16.0)
    parser.add_argument("--negative_threshold_px", type=float, default=32.0)
    parser.add_argument("--max_positives_per_keypoint", type=int, default=1)
    parser.add_argument("--hard_negatives_per_keypoint", type=int, default=16)
    parser.add_argument("--hard_negative_pool", type=int, default=128)
    parser.add_argument("--max_samples_per_query", type=int, default=512)
    parser.add_argument("--max_total_samples", type=int, default=20000)
    parser.add_argument("--keypoint_stride_px", type=float, default=16.0)
    parser.add_argument("--visualize_render_dir", default="")
    parser.add_argument("--visualize_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = _select_records(manifest.records, int(args.max_queries), str(args.view_selection))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    render_width, render_height = _resolve_render_size(camera, int(args.render_width), int(args.render_height))
    render_camera = _scale_camera(camera, render_width, render_height)
    render_config = GaussianVFMRenderConfig(
        width=render_width,
        height=render_height,
        radius_px=float(args.render_radius_px),
        depth_epsilon=float(args.render_depth_epsilon),
        l2_normalize_pixels=True,
    )
    if str(args.renderer) == "official_2dgs":
        rgb_source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
        depth_field = None
    else:
        rgb_source = load_gaussian_rgb_source_from_ply(Path(args.gaussian_rgb_ply))
        depth_field = _depth_field_from_rgb_source(rgb_source)
    detector_fn = _make_keypoint_detector(str(args.detector), int(args.max_keypoints), str(args.device))
    from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

    radio = RADIOFeatureExtractor(version=args.radio_version, device=args.device, radio_repo=args.radio_repo)
    sample_config = RenderedKeypointSelectorSampleConfig(
        positive_threshold_px=float(args.positive_threshold_px),
        negative_threshold_px=float(args.negative_threshold_px),
        max_positives_per_keypoint=int(args.max_positives_per_keypoint),
        hard_negatives_per_keypoint=int(args.hard_negatives_per_keypoint),
        hard_negative_pool=int(args.hard_negative_pool),
        max_samples=int(args.max_samples_per_query),
        keypoint_stride_px=float(args.keypoint_stride_px),
        seed=int(args.seed),
    )
    render_cache_dir = Path(args.render_token_cache_dir) if args.render_token_cache_dir else None
    render_rgb_depth_cache_dir = Path(args.render_rgb_depth_cache_dir) if args.render_rgb_depth_cache_dir else None
    render_vis_dir = Path(args.visualize_render_dir) if args.visualize_render_dir else None
    merged = None
    rows = []
    skipped = {"missing_pose": 0, "dim_mismatch": 0, "empty_samples": 0}
    for record_idx, record in enumerate(records):
        gt = gt_by_query.get(record.image_id)
        if gt is None:
            skipped["missing_pose"] += 1
            continue
        query_rgb = _read_rgb(Path(args.image_root) / record.image_id)
        query_feature = (
            _extract_radio_feature_from_rgb(query_rgb, radio)
            if bool(args.extract_query_features_from_image)
            else _load_query_feature(record.token_path, args.layer_name)
        )
        rgb_depth_cache_path = None
        if render_rgb_depth_cache_dir is not None:
            rgb_depth_cache_path = (
                render_rgb_depth_cache_dir / f"{_safe_image_stem(record.image_id)}_{render_width}x{render_height}.npz"
            )
        render_rgb, render_depth, render_alpha = _load_or_render_rgb_depth_cache(
            cache_path=rgb_depth_cache_path,
            render_fn=lambda gt_pose=gt.pose_w2c: _render_rgb_and_depth(
                rgb_source,
                depth_field,
                pose_w2c=gt_pose,
                camera=camera,
                config=render_config,
                renderer=args.renderer,
                device=args.device,
            ),
            skip_existing=bool(args.skip_existing_render_rgb_depth),
        )
        cache_path = None
        if render_cache_dir is not None:
            cache_path = _render_token_cache_path(render_cache_dir, record.image_id, render_width, render_height)
        render_feature = _load_or_extract_render_feature(
            cache_path=cache_path,
            render_rgb=render_rgb,
            extractor=radio,
            layer_name=args.layer_name,
            skip_existing=bool(args.skip_existing_render_tokens),
        )
        if int(query_feature.shape[0]) != int(render_feature.shape[0]):
            skipped["dim_mismatch"] += 1
            continue
        query_xy, _query_scores = detector_fn(query_rgb)
        render_xy, _render_scores = detector_fn(render_rgb)
        query_desc, query_valid = bilinear_sample_feature_map(
            query_feature,
            query_xy,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        render_desc, render_valid = bilinear_sample_feature_map(
            render_feature,
            render_xy,
            image_width=render_width,
            image_height=render_height,
        )
        query_xy_valid = query_xy[query_valid]
        render_xy_valid = render_xy[render_valid]
        query_desc = query_desc[query_valid]
        render_desc = render_desc[render_valid]
        errors, render_depth_valid = render_keypoint_reprojection_errors(
            query_xy_valid,
            render_xy_valid,
            rendered_depth=render_depth,
            render_camera=render_camera,
            query_camera=camera,
            render_pose_w2c=gt.pose_w2c,
            query_pose_w2c=gt.pose_w2c,
            render_width=render_width,
            render_height=render_height,
        )
        errors[:, ~render_depth_valid] = np.inf
        diagnostics = render_rgb_keypoint_pair_diagnostics(
            query_desc,
            render_desc,
            errors,
            positive_threshold_px=float(args.positive_threshold_px),
            negative_threshold_px=float(args.negative_threshold_px),
        )
        samples = build_render_rgb_keypoint_adapter_samples(query_desc, render_desc, errors, sample_config)
        if samples.sample_count == 0:
            skipped["empty_samples"] += 1
            continue
        merged = append_patch_selector_training_set_capped(
            merged,
            samples,
            max_samples=int(args.max_total_samples),
            seed=int(args.seed) + int(record_idx),
        )
        if render_vis_dir is not None and len(rows) < int(args.visualize_limit):
            try:
                import cv2

                out_path = render_vis_dir / f"{_safe_image_stem(record.image_id)}_render.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
            except Exception:
                pass
        rows.append(
            {
                "query_id": record.image_id,
                "query_keypoints": int(query_xy.shape[0]),
                "render_keypoints": int(render_xy.shape[0]),
                "valid_query_keypoints": int(query_desc.shape[0]),
                "valid_render_keypoints": int(render_desc.shape[0]),
                "render_width": int(render_width),
                "render_height": int(render_height),
                "query_feature_shape": list(query_feature.shape),
                "render_feature_shape": list(render_feature.shape),
                "render_alpha_mean": float(np.mean(render_alpha)),
                "render_depth_valid_fraction": float(np.mean(np.isfinite(render_depth) & (render_depth > 0.0))),
                "sample_count": int(samples.sample_count),
                "merged_sample_count": int(merged.sample_count),
                **{f"diag_{key}": value for key, value in diagnostics.to_dict().items()},
            }
        )
    if merged is None or merged.sample_count == 0:
        raise ValueError(f"no samples built; skipped={skipped}")
    save_patch_selector_training_set_npz(merged, Path(args.output))
    diag_keys = [key for key in rows[0] if key.startswith("diag_")] if rows else []
    summary_diag = {}
    for key in diag_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        if values:
            summary_diag[key] = float(np.mean(values))
    summary = {
        "stage": "render_rgb_keypoint_adapter_samples",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_count": int(merged.sample_count),
        "source_query_count": int(len(rows)),
        "skipped": skipped,
        "camera_source": camera_source,
        "config": {
            "renderer": args.renderer,
            "detector": args.detector,
            "max_keypoints": int(args.max_keypoints),
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
            "resolved_render_width": int(render_width),
            "resolved_render_height": int(render_height),
            "radio_version": args.radio_version,
            "extract_query_features_from_image": bool(args.extract_query_features_from_image),
            "sample": sample_config.__dict__,
        },
        "diagnostics_mean": summary_diag,
        "rows": rows,
        "outputs": {
            "sample_cache": str(args.output),
            "render_token_cache_dir": str(render_cache_dir or ""),
            "render_rgb_depth_cache_dir": str(render_rgb_depth_cache_dir or ""),
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
