"""Build compact RADIO-intermediate descriptors at real support/query nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.radio_intermediate_context import (
    RadioIntermediateContextCache,
    fit_normalized_pca,
    load_radio_intermediate_context_cache,
    project_and_sample_radio_map,
    save_radio_intermediate_context_cache,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--query_anchor_cache", required=True)
    parser.add_argument("--query_context_cache", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--dual_feature_root", required=True)
    parser.add_argument(
        "--coordinate_model_dir",
        default=None,
        help=(
            "COLMAP model defining the pixel grid of every support/query xy; "
            "required for resolution-safe caches"
        ),
    )
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--projection_source_cache",
        default=None,
        help=(
            "reuse the exact PCA projection and support descriptors from a compatible "
            "training cache; only new query images are sampled"
        ),
    )
    parser.add_argument(
        "--projection_source_support_feature_cache",
        default=None,
        help="source ALIKE support cache used to validate/reindex reused RADIO rows",
    )
    parser.add_argument(
        "--projection_source_support_geometry_index",
        default=None,
        help="source support geometry used to reindex reused RADIO rows by observation identity",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    parser.add_argument("--intermediate_index", type=int, default=-6)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--pca_source_image_count", type=int, default=64)
    parser.add_argument("--pca_samples_per_image", type=int, default=256)
    parser.add_argument(
        "--pca_source",
        choices=("dual_cache", "fresh_rgb"),
        default="dual_cache",
    )
    parser.add_argument(
        "--disable_dual_feature_reuse",
        action="store_true",
        help="extract every intermediate map from RGB with the explicit checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dtype", default="float16", choices=("float16", "float32"))
    return parser.parse_args(argv)


def _load_point_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"].item()))
    image_ids = payload.get("image_ids")
    offsets = payload.get("offsets")
    xy = payload.get("xy")
    if image_ids is None or offsets is None or xy is None:
        raise ValueError(f"point cache is missing required arrays: {path}")
    if offsets.shape != (len(image_ids) + 1,) or int(offsets[-1]) != len(xy):
        raise ValueError(f"point cache offsets are invalid: {path}")
    return payload, metadata


def _dual_feature_paths(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    suffix = "_1920x1080_radio_dual.npz"
    for path in sorted(Path(root).glob(f"*{suffix}")):
        stem = path.name[: -len(suffix)]
        parts = stem.split("__", 1)
        if len(parts) != 2:
            continue
        image_id = f"{parts[0]}/{parts[1]}"
        if image_id in output:
            raise ValueError(f"duplicate RADIO dual map for {image_id}")
        output[image_id] = path
    return output


def _fit_pca_from_dual_maps(
    paths: Sequence[Path],
    *,
    source_image_count: int,
    samples_per_image: int,
    output_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(paths) < int(source_image_count):
        raise ValueError("not enough RADIO dual maps for PCA fitting")
    rng = np.random.default_rng(int(seed))
    chosen_indices = np.sort(
        rng.choice(len(paths), size=int(source_image_count), replace=False)
    )
    chosen = [Path(paths[int(index)]) for index in chosen_indices.tolist()]
    sample_blocks = []
    source_manifest = []
    for path in chosen:
        with np.load(path, allow_pickle=False) as data:
            feature_map = np.asarray(data["radio_dual"], dtype=np.float32)
        if feature_map.ndim != 3 or feature_map.shape[0] < 1280:
            raise ValueError(f"invalid RADIO dual map: {path}")
        fine = feature_map[:1280].reshape(1280, -1).T
        count = min(int(samples_per_image), len(fine))
        rows = rng.choice(len(fine), size=count, replace=False)
        sample_blocks.append(fine[rows])
        stat = path.stat()
        source_manifest.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    samples = np.concatenate(sample_blocks, axis=0).astype(np.float32)
    mean, components, pca_metrics = fit_normalized_pca(
        samples, output_dim=int(output_dim), seed=int(seed)
    )
    return mean, components, {
        "source": "legacy_dual_cache_without_embedded_model_manifest",
        "source_image_count": int(len(chosen)),
        "samples_per_image": int(samples_per_image),
        "sample_count": int(len(samples)),
        "source_manifest_sha256": hashlib.sha256(
            "\n".join(source_manifest).encode("utf8")
        ).hexdigest()[:16],
        **pca_metrics,
    }


def _fit_pca_from_rgb(
    image_ids: Sequence[str],
    *,
    image_root: Path,
    source_image_count: int,
    samples_per_image: int,
    output_dim: int,
    seed: int,
    device: str,
    radio_repo: str,
    checkpoint_path: Path,
    intermediate_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(image_ids) < int(source_image_count):
        raise ValueError("not enough RGB images for fresh PCA fitting")
    rng = np.random.default_rng(int(seed))
    chosen_indices = np.sort(
        rng.choice(len(image_ids), size=int(source_image_count), replace=False)
    )
    chosen = [str(image_ids[int(index)]) for index in chosen_indices.tolist()]
    extractor = RADIOFeatureExtractor(
        version=str(checkpoint_path.resolve()),
        device=str(device),
        radio_repo=str(radio_repo),
    )
    sample_blocks = []
    source_manifest = []
    for image_id in chosen:
        path = Path(image_root) / image_id
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        feature_map = extractor.extract_intermediate_batch(
            tensor,
            intermediate_index=int(intermediate_index),
            norm_intermediates=True,
            aggregation="sparse",
        )[0]
        fine = feature_map.permute(1, 2, 0).reshape(-1, int(feature_map.shape[0]))
        count = min(int(samples_per_image), len(fine))
        rows = rng.choice(len(fine), size=count, replace=False)
        sample_blocks.append(fine[rows].float().cpu().numpy())
        source_manifest.append(f"{image_id}:{file_sha256_short(path)}")
        del feature_map
    samples = np.concatenate(sample_blocks, axis=0).astype(np.float32)
    mean, components, pca_metrics = fit_normalized_pca(
        samples, output_dim=int(output_dim), seed=int(seed)
    )
    return mean, components, {
        "source": "fresh_rgb_explicit_radio_checkpoint_v1",
        "source_image_count": int(len(chosen)),
        "samples_per_image": int(samples_per_image),
        "sample_count": int(len(samples)),
        "source_manifest_sha256": hashlib.sha256(
            "\n".join(source_manifest).encode("utf8")
        ).hexdigest()[:16],
        **pca_metrics,
    }


def _image_entries(
    image_ids: np.ndarray, offsets: np.ndarray
) -> dict[str, tuple[int, int]]:
    return {
        str(image_id): (int(offsets[index]), int(offsets[index + 1]))
        for index, image_id in enumerate(image_ids.astype(str).tolist())
    }


def _stat_manifest_hash(paths: Sequence[Path]) -> str:
    rows = []
    for path in sorted(Path(value) for value in paths):
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        rows.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(rows).encode("utf8")).hexdigest()[:16]


def _reindex_support_descriptors(
    *,
    source_descriptors: np.ndarray,
    source_feature_track_ids: np.ndarray,
    source_geometry,
    target_feature_track_ids: np.ndarray,
    target_geometry,
    allow_missing: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reuse descriptors only for identical image/track/xy observations."""

    source_values = np.asarray(source_descriptors, dtype=np.float32)
    source_tracks = np.asarray(source_feature_track_ids, dtype=np.int64)
    target_tracks = np.asarray(target_feature_track_ids, dtype=np.int64)
    if len(source_values) != len(source_tracks):
        raise ValueError("source RADIO and source feature rows differ")
    if not np.array_equal(
        source_tracks[source_geometry.source_row_indices], source_geometry.track_ids
    ):
        raise ValueError("source support geometry and feature rows differ")
    if not np.array_equal(
        target_tracks[target_geometry.source_row_indices], target_geometry.track_ids
    ):
        raise ValueError("target support geometry and feature rows differ")
    source_image_positions = {
        str(image_id): int(index)
        for index, image_id in enumerate(
            np.asarray(source_geometry.image_ids).astype(str).tolist()
        )
    }
    output = np.full(
        (len(target_tracks), int(source_values.shape[1])), np.nan, dtype=np.float32
    )
    matched_observations = 0
    missing_observations = 0
    missing_images: list[str] = []
    for target_image_index, image_id in enumerate(
        np.asarray(target_geometry.image_ids).astype(str).tolist()
    ):
        target_start = int(target_geometry.image_offsets[target_image_index])
        target_end = int(target_geometry.image_offsets[target_image_index + 1])
        source_image_index = source_image_positions.get(str(image_id))
        if source_image_index is None:
            missing_images.append(str(image_id))
            missing_observations += int(target_end - target_start)
            continue
        source_start = int(source_geometry.image_offsets[source_image_index])
        source_end = int(source_geometry.image_offsets[source_image_index + 1])
        source_by_track: dict[int, int] = {}
        for source_row in range(source_start, source_end):
            track_id = int(source_geometry.track_ids[source_row])
            if track_id in source_by_track:
                raise ValueError(
                    f"projection source has duplicate image/track observation: {image_id}/{track_id}"
                )
            source_by_track[track_id] = source_row
        for target_row in range(target_start, target_end):
            track_id = int(target_geometry.track_ids[target_row])
            source_row = source_by_track.get(track_id)
            if source_row is None:
                missing_observations += 1
                continue
            if not np.allclose(
                source_geometry.xy[source_row],
                target_geometry.xy[target_row],
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(
                    f"projection source observation coordinates changed: {image_id}/{track_id}"
                )
            source_feature_row = int(source_geometry.source_row_indices[source_row])
            target_feature_row = int(target_geometry.source_row_indices[target_row])
            output[target_feature_row] = source_values[source_feature_row]
            matched_observations += 1
    missing = int(np.sum(~np.isfinite(output).all(axis=1)))
    if missing != int(missing_observations):
        raise RuntimeError("support reindex missing-row accounting changed")
    if missing and not bool(allow_missing):
        raise ValueError(f"support reindex left {missing} target feature rows missing")
    return output, {
        "method": "exact_image_track_xy_reindex_v1",
        "matched_observation_count": int(matched_observations),
        "missing_observation_count": int(missing),
        "missing_image_count": int(len(missing_images)),
        "missing_images": missing_images,
        "source_feature_row_count": int(len(source_tracks)),
        "target_feature_row_count": int(len(target_tracks)),
        "source_image_count": int(len(source_geometry.image_ids)),
        "target_image_count": int(len(target_geometry.image_ids)),
    }


def _coordinate_space(
    model_dir: Path | None,
    image_ids: Sequence[str],
    *,
    image_root: Path,
) -> tuple[dict[str, tuple[int, int]], dict[str, object]]:
    ids = tuple(str(value) for value in image_ids)
    if model_dir is None:
        sizes: dict[str, tuple[int, int]] = {}
        for image_id in ids:
            with Image.open(Path(image_root) / image_id) as image:
                sizes[image_id] = (int(image.width), int(image.height))
        source = "source_rgb_dimensions_LEGACY"
        model_hashes = None
    else:
        cameras_path = Path(model_dir) / "cameras.bin"
        images_path = Path(model_dir) / "images.bin"
        cameras = read_colmap_cameras_binary(cameras_path)
        images = read_colmap_images_binary(images_path)
        images_by_name = {str(image.image_name): image for image in images.values()}
        sizes = {}
        for image_id in ids:
            image = images_by_name.get(image_id)
            if image is None:
                raise KeyError(f"coordinate model is missing image: {image_id}")
            camera = cameras[int(image.camera_id)]
            sizes[image_id] = (int(camera.width), int(camera.height))
        source = "colmap_per_image_v1"
        model_hashes = {
            "cameras_sha256": file_sha256_short(cameras_path),
            "images_sha256": file_sha256_short(images_path),
        }
    unique_dimensions = sorted(set(sizes.values()))
    identity_payload = {
        "convention": "sfm_pixel_endpoint_to_feature_endpoint_align_corners_true",
        "dimension_source": source,
        "unique_dimensions": [list(value) for value in unique_dimensions],
    }
    coordinate_space_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf8")
    ).hexdigest()[:16]
    return sizes, {
        **identity_payload,
        "coordinate_space_id": coordinate_space_id,
        "coordinate_model_dir": None if model_dir is None else str(model_dir),
        "coordinate_model_hashes": model_hashes,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = time.time()
    devices = tuple(value.strip() for value in str(args.devices).split(",") if value.strip())
    if not devices:
        raise ValueError("at least one device is required")
    support_path = Path(args.support_feature_cache)
    geometry_path = Path(args.support_geometry_index)
    anchor_path = Path(args.query_anchor_cache)
    context_path = Path(args.query_context_cache)
    with np.load(support_path, allow_pickle=False) as data:
        support_tracks = np.asarray(data["track_ids"], dtype=np.int64)
    geometry, _geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if len(geometry) != len(support_tracks) or not np.array_equal(
        support_tracks[geometry.source_row_indices], geometry.track_ids
    ):
        raise ValueError("support geometry and feature cache rows differ")
    anchor, _anchor_metadata = _load_point_cache(anchor_path)
    context, _context_metadata = _load_point_cache(context_path)
    if not np.array_equal(anchor["image_ids"].astype(str), context["image_ids"].astype(str)):
        raise ValueError("query anchor and context image lists differ")

    image_root = Path(args.image_root)
    checkpoint_path = Path(args.radio_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    dual_paths = _dual_feature_paths(Path(args.dual_feature_root))
    source_cache = None
    source_cache_path = (
        None
        if args.projection_source_cache is None
        else Path(args.projection_source_cache)
    )
    source_support_feature_path = (
        None
        if args.projection_source_support_feature_cache is None
        else Path(args.projection_source_support_feature_cache)
    )
    source_support_geometry_path = (
        None
        if args.projection_source_support_geometry_index is None
        else Path(args.projection_source_support_geometry_index)
    )
    if (source_support_feature_path is None) != (source_support_geometry_path is None):
        raise ValueError(
            "projection source support feature and geometry paths must be provided together"
        )
    if source_cache_path is None and source_support_feature_path is not None:
        raise ValueError("support reindex requires projection_source_cache")
    if source_cache_path is None:
        if str(args.pca_source) == "fresh_rgb":
            pca_mean, pca_components, pca_metadata = _fit_pca_from_rgb(
                sorted(set(str(value) for value in geometry.image_ids)),
                image_root=image_root,
                source_image_count=int(args.pca_source_image_count),
                samples_per_image=int(args.pca_samples_per_image),
                output_dim=int(args.projection_dim),
                seed=int(args.seed),
                device=str(devices[0]),
                radio_repo=str(args.radio_repo),
                checkpoint_path=checkpoint_path,
                intermediate_index=int(args.intermediate_index),
            )
        else:
            pca_mean, pca_components, pca_metadata = _fit_pca_from_dual_maps(
                sorted(dual_paths.values()),
                source_image_count=int(args.pca_source_image_count),
                samples_per_image=int(args.pca_samples_per_image),
                output_dim=int(args.projection_dim),
                seed=int(args.seed),
            )
        support_output = np.full(
            (len(support_tracks), int(args.projection_dim)), np.nan, dtype=np.float32
        )
    else:
        expected_source_support_feature_hash = file_sha256_short(
            support_path
            if source_support_feature_path is None
            else source_support_feature_path
        )
        expected_source_support_geometry_hash = file_sha256_short(
            geometry_path
            if source_support_geometry_path is None
            else source_support_geometry_path
        )
        source_cache = load_radio_intermediate_context_cache(
            source_cache_path,
            expected_metadata={
                "support_feature_cache_sha256": expected_source_support_feature_hash,
                "support_geometry_index_sha256": expected_source_support_geometry_hash,
                "radio_version": str(args.radio_version),
                "radio_checkpoint_sha256": file_sha256_short(checkpoint_path),
                "radio_model_load_spec": "explicit_checkpoint_path_v1",
                "intermediate_index": int(args.intermediate_index),
                "norm_intermediates": True,
                "aggregation": "sparse",
                "projection": "normalized_pca_then_l2_v1",
                "sampling": "bilinear_endpoint_align_corners_true",
                "descriptor_dim": int(args.projection_dim),
            },
        )
        pca_mean = source_cache.pca_mean.copy()
        pca_components = source_cache.pca_components.copy()
        pca_metadata = dict(source_cache.metadata.get("pca") or {})
        if source_support_feature_path is None:
            if len(source_cache.support_descriptors) != len(support_tracks):
                raise ValueError("projection source support descriptors do not align")
            support_output = source_cache.support_descriptors.copy()
            support_reindex_metadata = None
        else:
            assert source_support_geometry_path is not None
            with np.load(source_support_feature_path, allow_pickle=False) as data:
                source_support_tracks = np.asarray(data["track_ids"], dtype=np.int64)
            source_geometry, _source_geometry_metadata = (
                load_support_observation_geometry_index_npz(
                    source_support_geometry_path
                )
            )
            support_output, support_reindex_metadata = _reindex_support_descriptors(
                source_descriptors=source_cache.support_descriptors,
                source_feature_track_ids=source_support_tracks,
                source_geometry=source_geometry,
                target_feature_track_ids=support_tracks,
                target_geometry=geometry,
                allow_missing=True,
            )
    anchor_output = np.full(
        (len(anchor["xy"]), int(args.projection_dim)), np.nan, dtype=np.float32
    )
    context_output = np.full(
        (len(context["xy"]), int(args.projection_dim)), np.nan, dtype=np.float32
    )
    anchor_entries = _image_entries(anchor["image_ids"], anchor["offsets"])
    context_entries = _image_entries(context["image_ids"], context["offsets"])
    if source_cache is None:
        support_image_ids = set(geometry.image_ids)
    elif source_support_feature_path is None:
        support_image_ids = set()
    else:
        support_image_ids = set()
        for image_index, image_id in enumerate(geometry.image_ids):
            begin = int(geometry.image_offsets[image_index])
            end = int(geometry.image_offsets[image_index + 1])
            feature_rows = geometry.source_row_indices[begin:end]
            if np.any(~np.isfinite(support_output[feature_rows]).all(axis=1)):
                support_image_ids.add(str(image_id))
    image_ids = tuple(sorted(support_image_ids | set(anchor_entries) | set(context_entries)))
    source_image_stat_manifest_sha256 = _stat_manifest_hash(
        [image_root / image_id for image_id in image_ids]
    )
    reused_dual_paths = [
        dual_paths[image_id]
        for image_id in image_ids
        if not bool(args.disable_dual_feature_reuse) and image_id in dual_paths
    ]
    dual_feature_stat_manifest_sha256 = (
        None
        if not reused_dual_paths
        else _stat_manifest_hash(reused_dual_paths)
    )
    partitions = [list(image_ids[index:: len(devices)]) for index in range(len(devices))]
    coordinate_sizes, coordinate_metadata = _coordinate_space(
        None
        if args.coordinate_model_dir is None
        else Path(args.coordinate_model_dir),
        image_ids,
        image_root=image_root,
    )
    if source_cache is not None:
        source_coordinate_space_id = str(
            source_cache.metadata.get("coordinate_space_id", "")
        )
        if source_coordinate_space_id != str(
            coordinate_metadata["coordinate_space_id"]
        ):
            raise ValueError(
                "projection source and query RADIO caches use different coordinate spaces"
            )

    def worker(worker_index: int) -> dict[str, int]:
        device = devices[worker_index]
        extractor = RADIOFeatureExtractor(
            version=str(checkpoint_path.resolve()),
            device=device,
            radio_repo=str(args.radio_repo),
        )
        reused = 0
        extracted = 0
        point_count = 0
        for image_id in partitions[worker_index]:
            image_path = image_root / str(image_id)
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            width, height = coordinate_sizes[str(image_id)]
            dual_path = (
                None
                if bool(args.disable_dual_feature_reuse)
                else dual_paths.get(str(image_id))
            )
            if dual_path is not None:
                with np.load(dual_path, allow_pickle=False) as data:
                    feature_map = np.asarray(data["radio_dual"][:1280], dtype=np.float32)
                reused += 1
            else:
                with Image.open(image_path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None]
                feature_map = extractor.extract_intermediate_batch(
                    tensor,
                    intermediate_index=int(args.intermediate_index),
                    norm_intermediates=True,
                    aggregation="sparse",
                )[0]
                extracted += 1
            blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
            geometry_slice = geometry.image_slice(str(image_id))
            if int(geometry_slice.stop) > int(geometry_slice.start) and (
                source_cache is None or source_support_feature_path is not None
            ):
                rows = np.arange(int(geometry_slice.start), int(geometry_slice.stop), dtype=np.int64)
                if source_cache is not None:
                    missing = ~np.isfinite(
                        support_output[geometry.source_row_indices[rows]]
                    ).all(axis=1)
                    rows = rows[missing]
                if len(rows):
                    blocks.append(
                        (
                            "support",
                            geometry.source_row_indices[rows],
                            geometry.xy[rows],
                        )
                    )
            if str(image_id) in anchor_entries:
                begin, end = anchor_entries[str(image_id)]
                rows = np.arange(begin, end, dtype=np.int64)
                blocks.append(("anchor", rows, anchor["xy"][rows]))
            if str(image_id) in context_entries:
                begin, end = context_entries[str(image_id)]
                rows = np.arange(begin, end, dtype=np.int64)
                blocks.append(("context", rows, context["xy"][rows]))
            coordinates = np.concatenate([block[2] for block in blocks], axis=0)
            descriptors = project_and_sample_radio_map(
                feature_map,
                coordinates,
                image_width=int(width),
                image_height=int(height),
                pca_mean=pca_mean,
                pca_components=pca_components,
                device=device,
            )
            offset = 0
            for name, rows, xy in blocks:
                values = descriptors[offset : offset + len(xy)]
                if name == "support":
                    support_output[rows] = values
                elif name == "anchor":
                    anchor_output[rows] = values
                else:
                    context_output[rows] = values
                offset += len(xy)
            point_count += int(len(coordinates))
            del feature_map
        return {
            "reused_dual_image_count": int(reused),
            "extracted_image_count": int(extracted),
            "point_count": int(point_count),
        }

    if len(devices) == 1:
        worker_metadata = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            worker_metadata = list(executor.map(worker, range(len(devices))))
    for name, values in (
        ("support", support_output),
        ("query anchor", anchor_output),
        ("query context", context_output),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"RADIO extraction left missing {name} descriptors")
    metadata = {
        "support_feature_cache_sha256": file_sha256_short(support_path),
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "query_anchor_cache_sha256": file_sha256_short(anchor_path),
        "query_context_cache_sha256": file_sha256_short(context_path),
        "radio_version": str(args.radio_version),
        "radio_checkpoint_sha256": file_sha256_short(checkpoint_path),
        "radio_model_load_spec": "explicit_checkpoint_path_v1",
        "intermediate_index": int(args.intermediate_index),
        "norm_intermediates": True,
        "aggregation": "sparse",
        "projection": "normalized_pca_then_l2_v1",
        "sampling": "bilinear_endpoint_align_corners_true",
        "pca_source": (
            str(args.pca_source)
            if source_cache is None
            else str(source_cache.metadata.get("pca_source", ""))
        ),
        "dual_feature_reuse_enabled": not bool(args.disable_dual_feature_reuse),
        **coordinate_metadata,
        "image_count": int(len(image_ids)),
        "source_image_stat_manifest_sha256": source_image_stat_manifest_sha256,
        "dual_feature_stat_manifest_sha256": dual_feature_stat_manifest_sha256,
        "support_descriptor_count": int(len(support_output)),
        "query_anchor_descriptor_count": int(len(anchor_output)),
        "query_context_descriptor_count": int(len(context_output)),
        "devices": list(devices),
        "pca": pca_metadata,
        "projection_source_cache_sha256": (
            None
            if source_cache_path is None
            else file_sha256_short(source_cache_path)
        ),
        "support_descriptor_source": (
            "extracted"
            if source_cache is None
            else (
                "reindexed_projection_source_cache"
                if source_support_feature_path is not None
                else "reused_projection_source_cache"
            )
        ),
        "support_reindex": (
            None if source_cache is None else support_reindex_metadata
        ),
        "projection_source_support_feature_cache_sha256": (
            None
            if source_support_feature_path is None
            else file_sha256_short(source_support_feature_path)
        ),
        "projection_source_support_geometry_index_sha256": (
            None
            if source_support_geometry_path is None
            else file_sha256_short(source_support_geometry_path)
        ),
        "workers": worker_metadata,
    }
    cache = RadioIntermediateContextCache(
        support_descriptors=support_output,
        query_anchor_descriptors=anchor_output,
        query_context_descriptors=context_output,
        pca_mean=pca_mean,
        pca_components=pca_components,
        metadata=metadata,
    )
    output_path = Path(args.output_cache)
    save_radio_intermediate_context_cache(
        cache, output_path, cache_dtype=str(args.cache_dtype)
    )
    summary = {
        "stage": "radio_intermediate_observation_context_cache",
        "metadata": metadata,
        "runtime_seconds": float(time.time() - start_time),
        "outputs": {
            "cache": str(output_path),
            "cache_sha256": file_sha256_short(output_path),
            "summary": str(args.summary_json),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
