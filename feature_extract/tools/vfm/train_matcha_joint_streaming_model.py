"""Train MATCHA joint model from a thin streaming pair manifest.

This script builds one query/render pair on the fly per optimization step. It
does not serialize dense render/query feature-map training shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import MutableMapping, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_matcha_joint_cache import (
    _build_alike_label_map,
    _extract_matcha_joint_feature_from_rgb,
    _pair_render_pose,
)
from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
    _load_or_render_rgb_depth_cache,
    _render_rgb_and_depth,
    _resolve_render_size,
    _safe_image_stem,
)
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
    _read_rgb,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord, camera_center_from_pose_w2c, parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMRenderConfig
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.matcha_coarse_fine_adapter import save_matcha_coarse_fine_adapter
from feature_extract.vfm.matcha_coarse_supervision import (
    MatchaCoarseSupervisionConfig,
    _sample_depth,
    build_matcha_coarse_supervision,
    merge_fine_labels_by_cell_pair,
)
from feature_extract.vfm.matcha_joint_cache import build_matcha_joint_index_training_set_from_maps
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingConfig,
    MatchaJointTrainingRun,
    MatchaJointTrainingSet,
    _build_matcha_joint_model_for_samples,
    _evaluate,
    _loss_value_for_samples,
    _model_state_snapshot,
    _sample_indices,
    _total_loss,
    joint_run_as_coarse_fine_adapter_run,
    load_matcha_joint_training_set_npz,
    load_matcha_joint_model,
    save_matcha_joint_training_set_npz,
    save_matcha_joint_model,
)
from feature_extract.vfm.matcha_keypoint_distillation import AlikeKeypointExtractor
from feature_extract.vfm.matcha_light_fusion import maybe_fuse_feature_map
from feature_extract.vfm.matcha_multiview_supervision import (
    MatchaGeometryView,
    MatchaMultiviewSupervisionConfig,
    build_matcha_3dgs_multiview_coarse_supervision,
)
from feature_extract.vfm.matcha_streaming_manifest import MatchaStreamingPairManifest, MatchaStreamingPairRecord
from feature_extract.vfm.matcha_synthetic_pairs import (
    SYNTHETIC_PAIR_SOURCE,
    sample_synthetic_pair_poses,
    synthetic_config_from_metadata,
)
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.render_pose_protocol import group_top_reference_poses, parse_world_offset
from feature_extract.vfm.tokens import TokenBankManifest


DEFAULT_REAL_PAIR_TYPE_SAMPLING_WEIGHTS = "A_gt:0.30,B_trans005:0.25,B_trans010:0.15,B_trans025:0.10,C_trans050:0.10,D_reference:0.10"


def _feature_cache_path(root: Path, query_id: str, *, width: int, height: int, layer_name: str) -> Path:
    return root / f"{_safe_image_stem(query_id)}_{int(width)}x{int(height)}_{str(layer_name)}.npz"


def _load_or_extract_query_feature(
    *,
    cache_path: Path | None,
    rgb: np.ndarray,
    extractor,
    layer_name: str,
    feature_mode: str,
    fine_intermediate_index: int,
    coarse_source: str,
    coarse_intermediate_index: int,
    cache_dtype: str,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as data:
            return np.asarray(data[layer_name], dtype=np.float32)
    feature = _extract_matcha_joint_feature_from_rgb(
        rgb,
        extractor,
        feature_mode=str(feature_mode),
        fine_intermediate_index=int(fine_intermediate_index),
        coarse_source=str(coarse_source),
        coarse_intermediate_index=int(coarse_intermediate_index),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stored = feature.astype(np.float16, copy=False) if str(cache_dtype) == "float16" else feature.astype(np.float32, copy=False)
        np.savez_compressed(cache_path, **{str(layer_name): stored})
    return feature.astype(np.float32, copy=False)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _array_digest(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _camera_cache_payload(camera) -> dict[str, object]:
    return {
        "camera_id": int(camera.camera_id),
        "model_id": int(camera.model_id),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(item) for item in camera.params],
    }


def _synthetic_training_sample_cache_key(
    record: MatchaStreamingPairRecord,
    *,
    manifest_metadata: Mapping[str, object],
    args: argparse.Namespace,
    query_camera,
    render_camera,
    source_pose_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
) -> str:
    arg_names = (
        "gaussian_rgb_ply",
        "layer_name",
        "feature_mode",
        "radio_version",
        "radio_repo",
        "radio_fine_intermediate_index",
        "radio_coarse_source",
        "radio_coarse_intermediate_index",
        "feature_fusion_mode",
        "feature_fusion_radius",
        "feature_fusion_temperature",
        "feature_fusion_alpha",
        "roundtrip_threshold_px",
        "roundtrip_heatmap_threshold_px",
        "visibility_alpha_threshold",
        "depth_edge_threshold_m",
        "soft_offset_sigma_bins",
        "pose_confidence_label_source",
        "pose_confidence_labels",
        "pose_confidence_positive_threshold_px",
        "pose_confidence_negative_threshold_px",
        "collect_visibility_no_match",
        "max_visibility_no_match",
        "hard_negatives_per_match",
        "fine_supervision_source",
        "merge_fine_labels_into_coarse",
        "keypoint_distill_method",
        "alike_repo",
        "alike_model",
        "alike_top_k",
        "alike_scores_th",
        "alike_n_limit",
        "render_width",
        "render_height",
        "default_camera",
        "camera_model_dir",
        "synthetic_pair_cache_format",
    )
    payload = {
        "version": 1,
        "record": record.to_dict(),
        "manifest_metadata": _jsonable(dict(manifest_metadata)),
        "args": {name: _jsonable(getattr(args, name, None)) for name in arg_names},
        "query_camera": _camera_cache_payload(query_camera),
        "render_camera": _camera_cache_payload(render_camera),
        "source_pose_sha256": _array_digest(source_pose_w2c),
        "target_pose_sha256": _array_digest(target_pose_w2c),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_training_sample_cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    root = Path(cache_dir) / str(key)[:2]
    return root / f"{key}.npz", root / f"{key}.row.json"


def _load_synthetic_training_sample_cache(cache_dir: Path, key: str) -> tuple[MatchaJointTrainingSet, dict[str, object]] | None:
    sample_path, row_path = _synthetic_training_sample_cache_paths(Path(cache_dir), str(key))
    if not sample_path.exists() or not row_path.exists():
        return None
    sample, _metadata = load_matcha_joint_training_set_npz(sample_path)
    row = json.loads(row_path.read_text())
    return sample, dict(row)


def _save_synthetic_training_sample_cache(
    cache_dir: Path,
    key: str,
    sample: MatchaJointTrainingSet,
    row: Mapping[str, object],
    *,
    compressed: bool,
) -> None:
    sample_path, row_path = _synthetic_training_sample_cache_paths(Path(cache_dir), str(key))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    save_matcha_joint_training_set_npz(sample, sample_path, compressed=compressed)
    row_path.write_text(json.dumps(_jsonable(dict(row)), indent=2, sort_keys=True) + "\n")


def _synthetic_training_sample_memory_cache_get(
    cache: MutableMapping[str, tuple[MatchaJointTrainingSet, dict[str, object]]],
    key: str,
) -> tuple[MatchaJointTrainingSet, dict[str, object]] | None:
    if str(key) not in cache:
        return None
    value = cache[str(key)]
    if hasattr(cache, "move_to_end"):
        cache.move_to_end(str(key))  # type: ignore[attr-defined]
    sample, row = value
    return sample, dict(row)


def _remember_synthetic_training_sample_memory_cache(
    cache: MutableMapping[str, tuple[MatchaJointTrainingSet, dict[str, object]]],
    key: str,
    sample: MatchaJointTrainingSet,
    row: Mapping[str, object],
    *,
    max_items: int,
) -> None:
    if int(max_items) <= 0:
        return
    cache[str(key)] = (sample, dict(row))
    if hasattr(cache, "move_to_end"):
        cache.move_to_end(str(key))  # type: ignore[attr-defined]
    while len(cache) > int(max_items):
        if hasattr(cache, "popitem"):
            try:
                cache.popitem(last=False)  # type: ignore[call-arg]
                continue
            except TypeError:
                pass
        oldest = next(iter(cache))
        del cache[oldest]


def _select_multiview_support_pose_records(
    pose_records_by_id: dict[str, CambridgePoseRecord],
    *,
    query_id: str,
    target_pose_w2c: np.ndarray,
    max_count: int,
) -> tuple[CambridgePoseRecord, ...]:
    if int(max_count) <= 0:
        return ()
    target_center = camera_center_from_pose_w2c(target_pose_w2c)
    candidates = []
    for image_id, record in pose_records_by_id.items():
        if str(image_id) == str(query_id):
            continue
        center = np.asarray(record.camera_center, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(center - target_center))
        candidates.append((distance, str(record.image_id), record))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in candidates[: int(max_count)])


def _render_subcell_seed_xy(
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    seed: int,
) -> np.ndarray:
    grid_w = int(grid_width)
    grid_h = int(grid_height)
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("grid dimensions must be positive")
    width = float(image_width)
    height = float(image_height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("image dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    frac = rng.uniform(0.125, 0.875, size=(grid_h, grid_w, 2))
    cell_w = width / float(grid_w)
    cell_h = height / float(grid_h)
    cols = np.arange(grid_w, dtype=np.float64)[None, :, None]
    rows = np.arange(grid_h, dtype=np.float64)[:, None, None]
    x = (cols + frac[..., 0:1]) * cell_w
    y = (rows + frac[..., 1:2]) * cell_h
    xy = np.concatenate([x, y], axis=2).reshape(-1, 2)
    xy[:, 0] = np.clip(xy[:, 0], 0.0, np.nextafter(width, 0.0))
    xy[:, 1] = np.clip(xy[:, 1], 0.0, np.nextafter(height, 0.0))
    return xy.astype(np.float64, copy=False)


def _render_subcell_stratified_seed_xy(
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    seed: int,
    offset_bins: int = 8,
    jitter: bool = True,
) -> np.ndarray:
    grid_w = int(grid_width)
    grid_h = int(grid_height)
    bins = int(offset_bins)
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("grid dimensions must be positive")
    if bins <= 0:
        raise ValueError("offset_bins must be positive")
    width = float(image_width)
    height = float(image_height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("image dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    count = int(grid_w * grid_h)
    labels = np.arange(count, dtype=np.int64) % int(bins * bins)
    rng.shuffle(labels)
    bin_x = labels % bins
    bin_y = labels // bins
    if bool(jitter):
        frac_x = (bin_x.astype(np.float64) + rng.uniform(0.0, 1.0, size=count)) / float(bins)
        frac_y = (bin_y.astype(np.float64) + rng.uniform(0.0, 1.0, size=count)) / float(bins)
    else:
        frac_x = (bin_x.astype(np.float64) + 0.5) / float(bins)
        frac_y = (bin_y.astype(np.float64) + 0.5) / float(bins)
    cols = np.arange(grid_w, dtype=np.float64)[None, :].repeat(grid_h, axis=0).reshape(-1)
    rows = np.arange(grid_h, dtype=np.float64)[:, None].repeat(grid_w, axis=1).reshape(-1)
    cell_w = width / float(grid_w)
    cell_h = height / float(grid_h)
    xy = np.stack([(cols + frac_x) * cell_w, (rows + frac_y) * cell_h], axis=1)
    xy[:, 0] = np.clip(xy[:, 0], 0.0, np.nextafter(width, 0.0))
    xy[:, 1] = np.clip(xy[:, 1], 0.0, np.nextafter(height, 0.0))
    return xy.astype(np.float64, copy=False)


def _label_entropy_bits(labels: np.ndarray, *, valid_label_count: int = 64) -> float:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = values[(values >= 0) & (values < int(valid_label_count))]
    if valid.size == 0:
        return 0.0
    counts = np.bincount(valid, minlength=int(valid_label_count)).astype(np.float64)
    probs = counts[counts > 0.0] / float(valid.size)
    return float(-np.sum(probs * np.log2(probs)))


def _label_fraction(labels: np.ndarray, label: int) -> float:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = values[values >= 0]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid == int(label)))


def _synthetic_supervision_overlap_fraction(
    supervision_count: int,
    query_grid_hw: Sequence[int],
    render_grid_hw: Sequence[int],
) -> float:
    query_h, query_w = int(query_grid_hw[0]), int(query_grid_hw[1])
    render_h, render_w = int(render_grid_hw[0]), int(render_grid_hw[1])
    query_area = int(query_h * query_w)
    render_area = int(render_h * render_w)
    return float(supervision_count) / float(max(min(query_area, render_area), 1))


class StreamingPairBuilder:
    def __init__(self, args: argparse.Namespace, manifest: MatchaStreamingPairManifest) -> None:
        self.args = args
        self.manifest = manifest
        source_query_manifest = str(manifest.metadata.get("source_query_manifest", ""))
        if not source_query_manifest:
            raise ValueError("streaming manifest metadata must include source_query_manifest")
        self.query_manifest = TokenBankManifest.from_json(Path(source_query_manifest))
        self.records_by_id = {record.image_id: record for record in self.query_manifest.records}
        self.gt_records = tuple(parse_cambridge_pose_file(Path(args.query_pose_file)))
        self.gt_by_query = {record.image_id: record for record in self.gt_records}
        camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
        self.camera, self.camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
        self.render_width, self.render_height = _resolve_render_size(self.camera, int(args.render_width), int(args.render_height))
        self.render_camera = _scale_camera(self.camera, self.render_width, self.render_height)
        self.render_config = GaussianVFMRenderConfig(
            width=int(self.render_width),
            height=int(self.render_height),
            radius_px=2.0,
            depth_epsilon=0.02,
        )
        self.query_depth_config = GaussianVFMRenderConfig(
            width=int(self.camera.width),
            height=int(self.camera.height),
            radius_px=2.0,
            depth_epsilon=0.02,
        )
        self.rgb_source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
        from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

        self.builder_device = str(args.builder_device or args.device)
        self.radio = RADIOFeatureExtractor(version=args.radio_version, device=self.builder_device, radio_repo=args.radio_repo)
        self.offset = parse_world_offset(str(args.render_pose_world_offset))
        self.reference_top1 = {}
        if any(record.pair_type == "D_reference" for record in manifest.records):
            if not str(args.candidate_bank):
                raise ValueError("--candidate_bank is required when streaming manifest contains D_reference")
            self.reference_top1 = group_top_reference_poses(CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)).candidates)
        self.supervision_config = MatchaCoarseSupervisionConfig(
            roundtrip_threshold_px=float(args.roundtrip_threshold_px),
            alpha_threshold=float(args.visibility_alpha_threshold),
            depth_edge_threshold_m=float(args.depth_edge_threshold_m),
            collect_no_match=bool(args.collect_visibility_no_match),
            max_no_match=int(args.max_visibility_no_match),
            soft_offset_sigma_bins=float(args.soft_offset_sigma_bins),
            pose_confidence_labels=bool(args.pose_confidence_labels),
            pose_confidence_positive_threshold_px=float(args.pose_confidence_positive_threshold_px),
            pose_confidence_negative_threshold_px=float(args.pose_confidence_negative_threshold_px),
        )
        self.keypoint_extractor = None
        if str(args.keypoint_distill_method) == "alike":
            self.keypoint_extractor = AlikeKeypointExtractor(
                matcha_repo=str(args.alike_repo),
                model_name=str(args.alike_model),
                top_k=int(args.alike_top_k),
                scores_th=float(args.alike_scores_th),
                n_limit=int(args.alike_n_limit),
                device=self.builder_device,
            )
        self.query_cache_dir = Path(args.query_feature_cache_dir) if str(args.query_feature_cache_dir) else None

    def _support_geometry_views(self, *, query_id: str, target_pose_w2c: np.ndarray) -> tuple[MatchaGeometryView, ...]:
        support_records = _select_multiview_support_pose_records(
            self.gt_by_query,
            query_id=str(query_id),
            target_pose_w2c=target_pose_w2c,
            max_count=int(self.args.multiview_supervision_support_views),
        )
        views: list[MatchaGeometryView] = []
        for support in support_records:
            _support_rgb, support_depth, support_alpha = _load_or_render_rgb_depth_cache(
                cache_path=None,
                render_fn=lambda pose=support.pose_w2c: _render_rgb_and_depth(
                    self.rgb_source,
                    None,
                    pose_w2c=pose,
                    camera=self.camera,
                    config=self.query_depth_config,
                    renderer="official_2dgs",
                    device=self.builder_device,
                ),
                skip_existing=False,
            )
            views.append(
                MatchaGeometryView(
                    camera=self.camera,
                    pose_w2c=support.pose_w2c,
                    depth=support_depth,
                    alpha=support_alpha,
                    view_id=str(support.image_id),
                )
            )
        return tuple(views)

    def build(self, record: MatchaStreamingPairRecord) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        gt = self.gt_by_query.get(record.query_id)
        if gt is None:
            raise KeyError(f"missing GT pose for {record.query_id}")
        _source_record = self.records_by_id.get(record.query_id)
        if _source_record is None:
            raise KeyError(f"query id {record.query_id!r} not present in source manifest")
        query_rgb = _read_rgb(Path(self.args.image_root) / record.query_id)
        query_cache_path = None
        if self.query_cache_dir is not None:
            query_cache_path = _feature_cache_path(
                self.query_cache_dir,
                record.query_id,
                width=int(self.camera.width),
                height=int(self.camera.height),
                layer_name=str(self.args.layer_name),
            )
        query_feature = _load_or_extract_query_feature(
            cache_path=query_cache_path,
            rgb=query_rgb,
            extractor=self.radio,
            layer_name=str(self.args.layer_name),
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
            cache_dtype=str(self.args.query_feature_cache_dtype),
        )
        _query_render_rgb, query_depth, query_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=gt.pose_w2c: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.camera,
                config=self.query_depth_config,
                renderer="official_2dgs",
                device=self.builder_device,
            ),
            skip_existing=False,
        )
        query_feature = maybe_fuse_feature_map(
            query_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=self.builder_device,
        )
        query_labels, query_kp_stats, _query_kp_xy = _build_alike_label_map(
            self.keypoint_extractor,
            query_rgb,
            feature_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
        )
        render_pose, pair_metadata = _pair_render_pose(
            pair_type=str(record.pair_type),
            query_id=str(record.query_id),
            gt_pose_w2c=gt.pose_w2c,
            base_offset=self.offset,
            seed=int(record.seed),
            record_index=int(record.record_index),
            pair_index=int(record.pair_index),
            pair_type_B_translation_m=float(self.args.pair_type_B_translation_m),
            pair_type_C_translation_m=float(self.args.pair_type_C_translation_m),
            perturb_rotation_deg=float(self.args.perturb_rotation_deg),
            reference_top1=self.reference_top1,
        )
        render_rgb, render_depth, render_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=render_pose: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.camera,
                config=self.render_config,
                renderer="official_2dgs",
                device=self.builder_device,
            ),
            skip_existing=False,
        )
        render_feature = _extract_matcha_joint_feature_from_rgb(
            render_rgb,
            self.radio,
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
        )
        render_feature = maybe_fuse_feature_map(
            render_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=self.builder_device,
        )
        render_labels, render_kp_stats, render_seed_xy = _build_alike_label_map(
            self.keypoint_extractor,
            render_rgb,
            feature_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
        )
        if str(self.args.fine_supervision_source) == "render_alike" and render_seed_xy is not None and render_seed_xy.shape[0] > 0:
            fine_seed_xy = render_seed_xy
        elif str(self.args.fine_supervision_source) == "render_subcell":
            fine_seed_xy = _render_subcell_seed_xy(
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
                grid_width=int(render_feature.shape[2]),
                grid_height=int(render_feature.shape[1]),
                seed=int(record.seed),
            )
        elif str(self.args.fine_supervision_source) == "render_subcell_stratified":
            fine_seed_xy = _render_subcell_stratified_seed_xy(
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
                grid_width=int(render_feature.shape[2]),
                grid_height=int(render_feature.shape[1]),
                seed=int(record.seed),
                offset_bins=int(self.supervision_config.offset_bins),
            )
        else:
            fine_seed_xy = None
        support_views = self._support_geometry_views(query_id=str(record.query_id), target_pose_w2c=render_pose)
        fine_seed_transfer_count = 0
        if int(self.args.multiview_supervision_support_views) > 0:
            query_geometry_view = MatchaGeometryView(
                camera=self.camera,
                pose_w2c=gt.pose_w2c,
                depth=query_depth,
                alpha=query_alpha,
                view_id=str(record.query_id),
            )
            render_geometry_view = MatchaGeometryView(
                camera=self.render_camera,
                pose_w2c=render_pose,
                depth=render_depth,
                alpha=render_alpha,
                view_id=str(pair_metadata.get("candidate_id", "")),
            )
            multiview_config = MatchaMultiviewSupervisionConfig(
                min_support_views=int(self.args.multiview_supervision_min_support_views),
                support_depth_tolerance_m=float(self.args.multiview_supervision_depth_tolerance_m),
                coarse_config=self.supervision_config,
            )

            def build_supervision(render_seed_xy):
                return build_matcha_3dgs_multiview_coarse_supervision(
                    query_view=query_geometry_view,
                    render_view=render_geometry_view,
                    support_views=support_views,
                    render_grid_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
                    query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
                    render_seed_xy=render_seed_xy,
                    config=multiview_config,
                )

        else:
            def build_supervision(render_seed_xy):
                return build_matcha_coarse_supervision(
                    render_depth=render_depth,
                    query_depth=query_depth,
                    render_alpha=render_alpha,
                    query_alpha=query_alpha,
                    render_camera=self.render_camera,
                    query_camera=self.camera,
                    render_pose_w2c=render_pose,
                    query_pose_w2c=gt.pose_w2c,
                    render_grid_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
                    query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
                    render_seed_xy=render_seed_xy,
                    config=self.supervision_config,
                )

        supervision = build_supervision(None)
        fine_seed_supervision = None
        fine_seed_transfer_count = 0
        if fine_seed_xy is not None:
            fine_seed_supervision = build_supervision(fine_seed_xy)
            if bool(self.args.merge_fine_labels_into_coarse):
                supervision, fine_seed_transfer_count = merge_fine_labels_by_cell_pair(supervision, fine_seed_supervision)
        if supervision.count == 0:
            raise ValueError("empty coarse supervision")
        fine_render_depth = None
        fine_validity_weight = None
        if fine_seed_supervision is not None:
            depth_values, depth_valid = _sample_depth(
                render_depth,
                fine_seed_supervision.render_xy,
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
            )
            fine_render_depth = np.where(depth_valid, depth_values, np.nan).astype(np.float32, copy=False)
            fine_validity_weight = (
                np.asarray(fine_seed_supervision.confidence_targets, dtype=np.float32).reshape(-1)
                * depth_valid.astype(np.float32, copy=False)
            )
        pose_confidence_kwargs = {}
        if str(self.args.pose_confidence_label_source) == "gt_reprojection":
            pose_confidence_kwargs = {
                "pose_confidence_render_depth": render_depth,
                "pose_confidence_query_camera": self.camera,
                "pose_confidence_render_camera": self.render_camera,
                "pose_confidence_query_pose_w2c": gt.pose_w2c,
                "pose_confidence_render_pose_w2c": render_pose,
                "pose_confidence_positive_threshold_px": float(self.args.pose_confidence_positive_threshold_px),
                "pose_confidence_negative_threshold_px": float(self.args.pose_confidence_negative_threshold_px),
            }
        joint = build_matcha_joint_index_training_set_from_maps(
            query_feature,
            render_feature,
            supervision,
            fine_supervision=fine_seed_supervision,
            fine_render_depth=fine_render_depth,
            fine_validity_weight=fine_validity_weight,
            query_rgb=query_rgb,
            render_rgb=render_rgb,
            query_keypoint_label_map=query_labels,
            render_keypoint_label_map=render_labels,
            hard_negatives_per_match=int(self.args.hard_negatives_per_match),
            roundtrip_heatmap_threshold_px=float(self.args.roundtrip_heatmap_threshold_px),
            **pose_confidence_kwargs,
        )
        object.__setattr__(joint, "pair_type_ids", np.asarray([int(pair_metadata["pair_type_id"])], dtype=np.int64))
        object.__setattr__(joint, "pair_type_names", np.asarray([str(pair_metadata["pair_type"])], dtype=object))
        object.__setattr__(joint, "pair_query_ids", np.asarray([str(record.query_id)], dtype=object))
        object.__setattr__(joint, "pair_split_names", np.asarray([str(record.split)], dtype=object))
        object.__setattr__(joint, "pair_candidate_ids", np.asarray([str(pair_metadata.get("candidate_id", ""))], dtype=object))
        object.__setattr__(joint, "pair_translation_errors_m", np.asarray([float(pair_metadata["perturb_translation_m"])], dtype=np.float32))
        object.__setattr__(joint, "pair_rotation_errors_deg", np.asarray([float(pair_metadata["perturb_rotation_deg"])], dtype=np.float32))
        row = {
            "query_id": str(record.query_id),
            "split": str(record.split),
            "pair_type": str(record.pair_type),
            "candidate_id": str(pair_metadata.get("candidate_id", "")),
            "sample_count": int(joint.coarse_fine_samples.sample_count),
            "confidence_label_source": str(joint.coarse_fine_samples.metadata.get("confidence_label_source", "supervision")),
            "confidence_target_mean": (
                None
                if joint.coarse_fine_samples.sample_confidence_targets is None
                else float(np.mean(joint.coarse_fine_samples.sample_confidence_targets))
            ),
            "confidence_target_gt05_fraction": (
                None
                if joint.coarse_fine_samples.sample_confidence_targets is None
                else float(np.mean(np.asarray(joint.coarse_fine_samples.sample_confidence_targets) > 0.5))
            ),
            "query_feature_shape": list(query_feature.shape),
            "render_feature_shape": list(render_feature.shape),
            "render_depth_valid_fraction": float(np.mean(np.isfinite(render_depth) & (render_depth > 0.0))),
            "render_alpha_mean": float(np.mean(render_alpha)),
            "supervision_source": str(supervision.source),
            "fine_supervision_source": str(self.args.fine_supervision_source),
            "render_seed_count": 0 if fine_seed_xy is None else int(fine_seed_xy.shape[0]),
            "fine_seed_transfer_count": int(fine_seed_transfer_count),
            "fine_seed_transfer_fraction": float(fine_seed_transfer_count) / max(float(supervision.count), 1.0),
            "fine_aux_supervision_count": int(0 if fine_seed_supervision is None else fine_seed_supervision.count),
            "query_offset_label_entropy_bits": _label_entropy_bits(supervision.query_offset_labels),
            "render_offset_label_entropy_bits": _label_entropy_bits(supervision.render_offset_labels),
            "query_offset_center_fraction": _label_fraction(
                supervision.query_offset_labels,
                (int(self.supervision_config.offset_bins) // 2)
                + int(self.supervision_config.offset_bins) * (int(self.supervision_config.offset_bins) // 2),
            ),
            "render_offset_center_fraction": _label_fraction(
                supervision.render_offset_labels,
                (int(self.supervision_config.offset_bins) // 2)
                + int(self.supervision_config.offset_bins) * (int(self.supervision_config.offset_bins) // 2),
            ),
            "multiview_support_view_count": int(len(support_views)),
            "supervision_no_match_count": int(getattr(supervision, "no_match_count", 0)),
            "query_keypoint_positive_count": int(query_kp_stats.get("positive_count", 0)),
            "render_keypoint_positive_count": int(render_kp_stats.get("positive_count", 0)),
            "roundtrip_median_px": float(np.median(supervision.roundtrip_errors_px)) if supervision.count else None,
        }
        return joint, row


class SyntheticStreamingPairBuilder(StreamingPairBuilder):
    def __init__(self, args: argparse.Namespace, manifest: MatchaStreamingPairManifest) -> None:
        super().__init__(args, manifest)
        self.synthetic_config = synthetic_config_from_metadata(manifest.metadata)
        self.query_cache_dir = None
        cache_dir = str(getattr(args, "synthetic_pair_cache_dir", "") or "")
        self.synthetic_pair_cache_dir = Path(cache_dir) if cache_dir else None
        self.synthetic_pair_cache_hits = 0
        self.synthetic_pair_cache_misses = 0
        self.synthetic_pair_memory_cache_size = int(getattr(args, "synthetic_pair_memory_cache_size", 1))
        self.synthetic_pair_cache_format = str(getattr(args, "synthetic_pair_cache_format", "npz"))
        self._synthetic_pair_memory_cache: OrderedDict[str, tuple[MatchaJointTrainingSet, dict[str, object]]] = OrderedDict()

    def _support_geometry_views(self, *, query_id: str, target_pose_w2c: np.ndarray) -> tuple[MatchaGeometryView, ...]:
        return ()

    def build(self, record: MatchaStreamingPairRecord) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        joint, row, _state = self._build_synthetic(record, include_eval_state=False)
        return joint, row

    def build_with_eval_state(
        self,
        record: MatchaStreamingPairRecord,
    ) -> tuple[MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
        return self._build_synthetic(record, include_eval_state=True)

    def _build_synthetic(
        self,
        record: MatchaStreamingPairRecord,
        *,
        include_eval_state: bool,
    ) -> tuple[MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
        anchor = self.gt_by_query.get(record.query_id)
        if anchor is None:
            raise KeyError(f"missing GT pose for {record.query_id}")
        synthetic_pair = sample_synthetic_pair_poses(
            anchor.pose_w2c,
            config=self.synthetic_config,
            seed=int(record.seed),
            key=f"{record.query_id}:{record.record_index}:{record.pair_index}",
        )
        cache_key = None
        if self.synthetic_pair_cache_dir is not None and not bool(include_eval_state):
            cache_key = _synthetic_training_sample_cache_key(
                record,
                manifest_metadata=self.manifest.metadata,
                args=self.args,
                query_camera=self.camera,
                render_camera=self.render_camera,
                source_pose_w2c=synthetic_pair.source_pose_w2c,
                target_pose_w2c=synthetic_pair.target_pose_w2c,
            )
            cached = _synthetic_training_sample_memory_cache_get(self._synthetic_pair_memory_cache, cache_key)
            if cached is None:
                cached = _load_synthetic_training_sample_cache(self.synthetic_pair_cache_dir, cache_key)
                if cached is not None:
                    _remember_synthetic_training_sample_memory_cache(
                        self._synthetic_pair_memory_cache,
                        cache_key,
                        cached[0],
                        cached[1],
                        max_items=self.synthetic_pair_memory_cache_size,
                    )
            if cached is not None:
                self.synthetic_pair_cache_hits += 1
                cached_joint, cached_row = cached
                row = dict(cached_row)
                row["training_sample_cache_hit"] = True
                row["training_sample_cache_key"] = str(cache_key)
                return cached_joint, row, {}
            self.synthetic_pair_cache_misses += 1
        source_rgb, source_depth, source_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=synthetic_pair.source_pose_w2c: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.camera,
                config=self.query_depth_config,
                renderer="official_2dgs",
                device=self.builder_device,
            ),
            skip_existing=False,
        )
        target_rgb, target_depth, target_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=synthetic_pair.target_pose_w2c: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.render_camera,
                config=self.render_config,
                renderer="official_2dgs",
                device=self.builder_device,
            ),
            skip_existing=False,
        )
        source_feature = _extract_matcha_joint_feature_from_rgb(
            source_rgb,
            self.radio,
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
        )
        source_feature = maybe_fuse_feature_map(
            source_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=self.builder_device,
        )
        target_feature = _extract_matcha_joint_feature_from_rgb(
            target_rgb,
            self.radio,
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
        )
        target_feature = maybe_fuse_feature_map(
            target_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=self.builder_device,
        )
        source_labels, source_kp_stats, _source_kp_xy = _build_alike_label_map(
            self.keypoint_extractor,
            source_rgb,
            feature_hw=(int(source_feature.shape[1]), int(source_feature.shape[2])),
        )
        target_labels, target_kp_stats, target_seed_xy = _build_alike_label_map(
            self.keypoint_extractor,
            target_rgb,
            feature_hw=(int(target_feature.shape[1]), int(target_feature.shape[2])),
        )
        if str(self.args.fine_supervision_source) == "render_alike" and target_seed_xy is not None and target_seed_xy.shape[0] > 0:
            fine_seed_xy = target_seed_xy
        elif str(self.args.fine_supervision_source) == "render_subcell":
            fine_seed_xy = _render_subcell_seed_xy(
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
                grid_width=int(target_feature.shape[2]),
                grid_height=int(target_feature.shape[1]),
                seed=int(record.seed),
            )
        elif str(self.args.fine_supervision_source) == "render_subcell_stratified":
            fine_seed_xy = _render_subcell_stratified_seed_xy(
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
                grid_width=int(target_feature.shape[2]),
                grid_height=int(target_feature.shape[1]),
                seed=int(record.seed),
                offset_bins=int(self.supervision_config.offset_bins),
            )
        else:
            fine_seed_xy = None
        source_grid_hw = (int(source_feature.shape[1]), int(source_feature.shape[2]))
        target_grid_hw = (int(target_feature.shape[1]), int(target_feature.shape[2]))

        def build_supervision(render_seed_xy):
            return build_matcha_coarse_supervision(
                render_depth=target_depth,
                query_depth=source_depth,
                render_alpha=target_alpha,
                query_alpha=source_alpha,
                render_camera=self.render_camera,
                query_camera=self.camera,
                render_pose_w2c=synthetic_pair.target_pose_w2c,
                query_pose_w2c=synthetic_pair.source_pose_w2c,
                render_grid_hw=target_grid_hw,
                query_grid_hw=source_grid_hw,
                render_seed_xy=render_seed_xy,
                config=self.supervision_config,
            )

        supervision = build_supervision(None)
        synthetic_overlap_fraction = _synthetic_supervision_overlap_fraction(
            int(supervision.count),
            source_grid_hw,
            target_grid_hw,
        )
        min_overlap = self.synthetic_config.min_overlap
        if min_overlap is not None and synthetic_overlap_fraction < float(min_overlap):
            raise ValueError(
                "synthetic pair overlap fraction "
                f"{synthetic_overlap_fraction:.6f} below minimum {float(min_overlap):.6f}"
            )
        min_count = self.synthetic_config.min_supervision_count
        if min_count is not None and int(supervision.count) < int(min_count):
            raise ValueError(f"synthetic pair supervision count {int(supervision.count)} below minimum {int(min_count)}")
        fine_seed_supervision = None
        fine_seed_transfer_count = 0
        if fine_seed_xy is not None:
            fine_seed_supervision = build_supervision(fine_seed_xy)
            if bool(self.args.merge_fine_labels_into_coarse):
                supervision, fine_seed_transfer_count = merge_fine_labels_by_cell_pair(supervision, fine_seed_supervision)
        fine_render_depth = None
        fine_validity_weight = None
        if fine_seed_supervision is not None:
            depth_values, depth_valid = _sample_depth(
                target_depth,
                fine_seed_supervision.render_xy,
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
            )
            fine_render_depth = np.where(depth_valid, depth_values, np.nan).astype(np.float32, copy=False)
            fine_validity_weight = (
                np.asarray(fine_seed_supervision.confidence_targets, dtype=np.float32).reshape(-1)
                * depth_valid.astype(np.float32, copy=False)
            )
        pose_confidence_kwargs = {}
        if str(self.args.pose_confidence_label_source) == "gt_reprojection":
            pose_confidence_kwargs = {
                "pose_confidence_render_depth": target_depth,
                "pose_confidence_query_camera": self.camera,
                "pose_confidence_render_camera": self.render_camera,
                "pose_confidence_query_pose_w2c": synthetic_pair.source_pose_w2c,
                "pose_confidence_render_pose_w2c": synthetic_pair.target_pose_w2c,
                "pose_confidence_positive_threshold_px": float(self.args.pose_confidence_positive_threshold_px),
                "pose_confidence_negative_threshold_px": float(self.args.pose_confidence_negative_threshold_px),
            }
        joint = build_matcha_joint_index_training_set_from_maps(
            source_feature,
            target_feature,
            supervision,
            fine_supervision=fine_seed_supervision,
            fine_render_depth=fine_render_depth,
            fine_validity_weight=fine_validity_weight,
            query_rgb=source_rgb,
            render_rgb=target_rgb,
            query_keypoint_label_map=source_labels,
            render_keypoint_label_map=target_labels,
            hard_negatives_per_match=int(self.args.hard_negatives_per_match),
            roundtrip_heatmap_threshold_px=float(self.args.roundtrip_heatmap_threshold_px),
            **pose_confidence_kwargs,
        )
        object.__setattr__(joint, "pair_type_ids", np.asarray([int(record.pair_type_id)], dtype=np.int64))
        object.__setattr__(joint, "pair_type_names", np.asarray([str(record.pair_type)], dtype=object))
        object.__setattr__(joint, "pair_query_ids", np.asarray([str(record.query_id)], dtype=object))
        object.__setattr__(joint, "pair_split_names", np.asarray([str(record.split)], dtype=object))
        object.__setattr__(joint, "pair_candidate_ids", np.asarray([""], dtype=object))
        object.__setattr__(
            joint,
            "pair_translation_errors_m",
            np.asarray([float(synthetic_pair.target_source_translation_m)], dtype=np.float32),
        )
        object.__setattr__(
            joint,
            "pair_rotation_errors_deg",
            np.asarray([float(synthetic_pair.target_source_rotation_deg)], dtype=np.float32),
        )
        offset_bins = int(self.supervision_config.offset_bins)
        offset_center_label = (offset_bins // 2) + offset_bins * (offset_bins // 2)
        row = {
            "query_id": str(record.query_id),
            "split": str(record.split),
            "pair_type": str(record.pair_type),
            "candidate_id": "",
            "sample_count": int(joint.coarse_fine_samples.sample_count),
            "confidence_label_source": str(joint.coarse_fine_samples.metadata.get("confidence_label_source", "supervision")),
            "confidence_target_mean": (
                None
                if joint.coarse_fine_samples.sample_confidence_targets is None
                else float(np.mean(joint.coarse_fine_samples.sample_confidence_targets))
            ),
            "confidence_target_gt05_fraction": (
                None
                if joint.coarse_fine_samples.sample_confidence_targets is None
                else float(np.mean(np.asarray(joint.coarse_fine_samples.sample_confidence_targets) > 0.5))
            ),
            "query_feature_shape": list(source_feature.shape),
            "render_feature_shape": list(target_feature.shape),
            "render_depth_valid_fraction": float(np.mean(np.isfinite(target_depth) & (target_depth > 0.0))),
            "render_alpha_mean": float(np.mean(target_alpha)),
            "supervision_source": str(supervision.source),
            "fine_supervision_source": str(self.args.fine_supervision_source),
            "render_seed_count": 0 if fine_seed_xy is None else int(fine_seed_xy.shape[0]),
            "fine_seed_transfer_count": int(fine_seed_transfer_count),
            "fine_seed_transfer_fraction": float(fine_seed_transfer_count) / max(float(supervision.count), 1.0),
            "fine_aux_supervision_count": int(0 if fine_seed_supervision is None else fine_seed_supervision.count),
            "query_offset_label_entropy_bits": _label_entropy_bits(supervision.query_offset_labels),
            "render_offset_label_entropy_bits": _label_entropy_bits(supervision.render_offset_labels),
            "query_offset_center_fraction": _label_fraction(supervision.query_offset_labels, offset_center_label),
            "render_offset_center_fraction": _label_fraction(supervision.render_offset_labels, offset_center_label),
            "multiview_support_view_count": 0,
            "supervision_no_match_count": int(getattr(supervision, "no_match_count", 0)),
            "query_keypoint_positive_count": int(source_kp_stats.get("positive_count", 0)),
            "render_keypoint_positive_count": int(target_kp_stats.get("positive_count", 0)),
            "roundtrip_median_px": float(np.median(supervision.roundtrip_errors_px)) if supervision.count else None,
            "synthetic_pose_bin": str(synthetic_pair.pose_bin),
            "synthetic_overlap_fraction": float(synthetic_overlap_fraction),
            "synthetic_source_anchor_translation_m": float(synthetic_pair.source_anchor_translation_m),
            "synthetic_source_anchor_rotation_deg": float(synthetic_pair.source_anchor_rotation_deg),
            "synthetic_target_source_translation_m": float(synthetic_pair.target_source_translation_m),
            "synthetic_target_source_rotation_deg": float(synthetic_pair.target_source_rotation_deg),
        }
        state: dict[str, object] = {}
        if bool(include_eval_state):
            state = {
                "source_pose_w2c": synthetic_pair.source_pose_w2c,
                "target_pose_w2c": synthetic_pair.target_pose_w2c,
                "target_depth": target_depth,
                "query_rgb": source_rgb,
                "render_rgb": target_rgb,
                "query_feature_map": source_feature,
                "render_feature_map": target_feature,
            }
        if self.synthetic_pair_cache_dir is not None and cache_key is not None and not bool(include_eval_state):
            row = {
                **row,
                "training_sample_cache_hit": False,
                "training_sample_cache_key": str(cache_key),
            }
            _save_synthetic_training_sample_cache(
                self.synthetic_pair_cache_dir,
                cache_key,
                joint,
                row,
                compressed=self.synthetic_pair_cache_format == "compressed_npz",
            )
            _remember_synthetic_training_sample_memory_cache(
                self._synthetic_pair_memory_cache,
                cache_key,
                joint,
                row,
                max_items=self.synthetic_pair_memory_cache_size,
            )
        return joint, row, state


def _builder_class_for_manifest(manifest: MatchaStreamingPairManifest) -> type[StreamingPairBuilder]:
    if manifest.metadata.get("pair_source") == SYNTHETIC_PAIR_SOURCE:
        return SyntheticStreamingPairBuilder
    return StreamingPairBuilder


def _synthetic_pose_bin_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        pose_bin = row.get("synthetic_pose_bin")
        if pose_bin is None:
            continue
        pose_bin_name = str(pose_bin)
        if not pose_bin_name:
            continue
        counts[pose_bin_name] = counts.get(pose_bin_name, 0) + 1
    return counts


def _supervision_balance_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    row_values = list(rows)
    positive_count = 0
    no_match_count = 0
    rows_with_no_match = 0
    for row in row_values:
        positive = int(row.get("supervision_positive_count", row.get("sample_count", 0)) or 0)
        no_match = int(row.get("supervision_no_match_count", 0) or 0)
        positive_count += max(positive, 0)
        no_match_count += max(no_match, 0)
        if no_match > 0:
            rows_with_no_match += 1
    rows_without_no_match = int(len(row_values) - rows_with_no_match)
    return {
        "row_count": int(len(row_values)),
        "positive_match_count": int(positive_count),
        "no_match_count": int(no_match_count),
        "rows_with_no_match": int(rows_with_no_match),
        "rows_without_no_match": int(rows_without_no_match),
        "no_match_to_positive_ratio": float(no_match_count) / max(float(positive_count), 1.0),
    }


def _synthetic_pair_cache_stats(*builders: object) -> dict[str, int]:
    hits = 0
    misses = 0
    for builder in builders:
        if builder is None:
            continue
        hits += int(getattr(builder, "synthetic_pair_cache_hits", 0))
        misses += int(getattr(builder, "synthetic_pair_cache_misses", 0))
    return {"hits": int(hits), "misses": int(misses)}


def _split_optional_manifest_paths(value: object) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _aggregate_manifest_pair_type_counts(manifests: Sequence[MatchaStreamingPairManifest]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for manifest in manifests:
        for key, value in manifest.pair_type_counts.items():
            counts[str(key)] = int(counts.get(str(key), 0) + int(value))
    return counts


def _synthetic_source_name(path: str, manifest: MatchaStreamingPairManifest, index: int) -> str:
    preset = str(manifest.metadata.get("synthetic_pose_bin_preset", "") or "")
    suffix = preset if preset else Path(path).stem
    return f"synthetic:{int(index)}:{suffix}"


def _synthetic_manifest_report(
    paths: Sequence[str],
    manifests: Sequence[MatchaStreamingPairManifest],
    *,
    sampling_weight: float,
) -> dict[str, object]:
    per_source_weight = float(sampling_weight) / float(len(manifests)) if manifests else 0.0
    return {
        "path": ",".join(str(path) for path in paths),
        "paths": [str(path) for path in paths],
        "source_count": int(len(manifests)),
        "query_count": int(sum(int(manifest.query_count) for manifest in manifests)),
        "pair_count": int(sum(int(len(manifest.records)) for manifest in manifests)),
        "pair_type_counts": _aggregate_manifest_pair_type_counts(manifests),
        "sampling_weight": float(sampling_weight),
        "per_source_sampling_weight": float(per_source_weight),
        "metadata": dict(manifests[0].metadata) if len(manifests) == 1 else {},
        "sources": [
            {
                "name": _synthetic_source_name(str(path), manifest, index),
                "path": str(path),
                "query_count": int(manifest.query_count),
                "pair_count": int(len(manifest.records)),
                "pair_type_counts": manifest.pair_type_counts,
                "sampling_weight": float(per_source_weight),
                "metadata": dict(manifest.metadata),
            }
            for index, (path, manifest) in enumerate(zip(paths, manifests))
        ],
    }


def _share_builder_runtime_state(primary_builder: object, secondary_builder: object | None) -> None:
    if secondary_builder is None:
        return
    for attr in ("rgb_source", "radio", "keypoint_extractor"):
        if hasattr(primary_builder, attr):
            setattr(secondary_builder, attr, getattr(primary_builder, attr))


def _sample_pair_index(rng: np.random.Generator, records: Sequence[MatchaStreamingPairRecord]) -> int:
    return int(rng.integers(0, len(records)))


def _build_streaming_pair_with_retries(
    records: Sequence[MatchaStreamingPairRecord],
    builder,
    *,
    seed: int,
    max_attempts: int = 32,
) -> tuple[MatchaJointTrainingSet, dict[str, object], int]:
    if not records:
        raise ValueError("records must be non-empty")
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be positive")
    last_error: Exception | None = None
    skipped = 0
    for attempt in range(int(max_attempts)):
        idx = _sample_pair_index(np.random.default_rng(int(seed) + int(attempt)), records)
        try:
            samples, row = builder.build(records[idx])
            return samples, row, skipped
        except Exception as exc:  # keep streaming robust to occasional bad render/supervision pairs
            skipped += 1
            last_error = exc
    raise RuntimeError(f"failed to build a streaming training pair after retries: {last_error}") from last_error


def _records_for_curriculum(
    records: Sequence[MatchaStreamingPairRecord],
    *,
    step: int,
    total_steps: int,
    mode: str,
) -> Sequence[MatchaStreamingPairRecord]:
    if str(mode) == "uniform":
        return records
    if str(mode) not in {"robust_default", "conservative_25cm"}:
        raise ValueError(f"unsupported pair_type_curriculum: {mode}")
    denom = max(int(total_steps), 1)
    progress = float(step) / float(denom)
    if str(mode) == "conservative_25cm":
        if progress < 0.40:
            allowed = {"A_gt"}
        elif progress < 0.75:
            allowed = {"A_gt", "B_trans005", "B_trans010"}
        else:
            allowed = {"A_gt", "B_trans005", "B_trans010", "B_trans025"}
    else:
        if progress < 0.20:
            allowed = {"A_gt"}
        elif progress < 0.45:
            allowed = {"A_gt", "B_trans005", "B_trans010"}
        elif progress < 0.75:
            allowed = {"A_gt", "B_trans005", "B_trans010", "B_trans025"}
        else:
            allowed = {"A_gt", "B_trans010", "B_trans025", "C_trans050", "D_reference"}
    selected = [record for record in records if str(record.pair_type) in allowed]
    return selected if selected else records


def _parse_pair_type_sampling_weights(spec: str | Mapping[str, float] | None) -> dict[str, float]:
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        return {str(key): max(float(value), 0.0) for key, value in spec.items()}
    text = str(spec).strip()
    if not text:
        return {}
    weights: dict[str, float] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"pair type sampling weight must use name:weight syntax: {item}")
        name, value = item.split(":", 1)
        key = name.strip()
        if not key:
            raise ValueError("pair type sampling weight name cannot be empty")
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"pair type sampling weight must be non-negative for {key}")
        weights[key] = float(weight)
    return weights


def _select_records_by_pair_type_weights(
    records: Sequence[MatchaStreamingPairRecord],
    *,
    weights: Mapping[str, float] | None,
    rng: np.random.Generator | None = None,
) -> Sequence[MatchaStreamingPairRecord]:
    values = list(records)
    parsed = _parse_pair_type_sampling_weights(weights)
    if not values or not parsed:
        return records
    available: dict[str, list[MatchaStreamingPairRecord]] = {}
    for record in values:
        available.setdefault(str(record.pair_type), []).append(record)
    labels = [label for label in available if float(parsed.get(label, 0.0)) > 0.0]
    if not labels:
        return records
    label_weights = np.asarray([float(parsed[label]) for label in labels], dtype=np.float64)
    if float(np.sum(label_weights)) <= 0.0:
        return records
    if rng is None:
        selected_label = labels[int(np.argmax(label_weights))]
    else:
        probs = label_weights / float(np.sum(label_weights))
        selected_label = labels[int(rng.choice(len(labels), p=probs))]
    return available[selected_label]


@dataclass(frozen=True)
class TrainingSource:
    name: str
    manifest: MatchaStreamingPairManifest
    builder: object
    sampling_weight: float = 1.0
    pair_type_weights: Mapping[str, float] | None = None


@dataclass(frozen=True)
class TrainingSourceSelection:
    name: str
    manifest: MatchaStreamingPairManifest
    builder: object
    records: Sequence[MatchaStreamingPairRecord]


def _select_training_source_for_step(
    sources: Sequence[TrainingSource],
    *,
    rng: np.random.Generator,
    step: int,
    total_steps: int,
    curriculum: str,
) -> TrainingSourceSelection:
    if not sources:
        raise ValueError("at least one training source is required")
    candidates: list[tuple[TrainingSource, Sequence[MatchaStreamingPairRecord]]] = []
    weights: list[float] = []
    for source in sources:
        records = _records_for_curriculum(
            source.manifest.records,
            step=int(step),
            total_steps=int(total_steps),
            mode=str(curriculum),
        )
        records = _select_records_by_pair_type_weights(records, weights=source.pair_type_weights, rng=rng)
        if not records:
            continue
        candidates.append((source, records))
        weights.append(max(float(source.sampling_weight), 0.0))
    if not candidates:
        raise ValueError("no active training records are available")
    if float(sum(weights)) <= 0.0:
        weights = [1.0 for _source, _records in candidates]
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / float(np.sum(probs))
    selected_idx = int(rng.choice(len(candidates), p=probs))
    source, records = candidates[selected_idx]
    return TrainingSourceSelection(
        name=str(source.name),
        manifest=source.manifest,
        builder=source.builder,
        records=records,
    )


def _is_head_parameter(name: str) -> bool:
    head_tokens = (
        "offset_head",
        "offset_head_map",
        "pair_confidence_head",
        "pair_fine_head",
        "query_pair_fine_head",
        "original_fine_matcher",
        "query_original_fine_matcher",
        "local_fine",
        "patch_corr_fine_head",
        "heatmap_head",
        "rgb_keypoint_detector",
    )
    return any(token in str(name) for token in head_tokens)


def _set_frozen_descriptor_batchnorm_eval(model: torch.nn.Module, *, freeze: bool) -> None:
    for name, module in model.named_modules():
        if not name or _is_head_parameter(str(name)):
            continue
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train(not bool(freeze))


def _make_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    descriptor_lr_scale: float,
    head_lr_scale: float,
    freeze_descriptor: bool,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    descriptor_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head_parameter(str(name)):
            head_params.append(param)
        else:
            descriptor_params.append(param)
    groups = []
    if descriptor_params:
        groups.append(
            {
                "params": descriptor_params,
                "lr": 0.0 if bool(freeze_descriptor) else float(lr) * float(descriptor_lr_scale),
                "name": "descriptor",
            }
        )
    if head_params:
        groups.append({"params": head_params, "lr": float(lr) * float(head_lr_scale), "name": "heads"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, lr=float(lr), weight_decay=1e-4)
    return optimizer, {"descriptor_param_count": int(sum(p.numel() for p in descriptor_params)), "head_param_count": int(sum(p.numel() for p in head_params))}


def _set_descriptor_group_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    descriptor_lr_scale: float,
    freeze: bool,
) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "descriptor":
            group["lr"] = 0.0 if bool(freeze) else float(base_lr) * float(descriptor_lr_scale)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming_manifest", required=True)
    parser.add_argument("--synthetic_streaming_manifest", default="")
    parser.add_argument("--real_sampling_weight", type=float, default=0.70)
    parser.add_argument("--synthetic_sampling_weight", type=float, default=0.30)
    parser.add_argument("--real_pair_type_sampling_weights", default=DEFAULT_REAL_PAIR_TYPE_SAMPLING_WEIGHTS)
    parser.add_argument("--validation_streaming_manifest", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_joint_model", default="")
    parser.add_argument("--output_best_joint_model", default="")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--warm_start_joint_checkpoint", default="")
    parser.add_argument(
        "--matcha_train_preset",
        default="none",
        choices=("none", "radio_matcha_patch_corr", "radio_matcha_2dgs_synthetic"),
    )
    parser.add_argument("--layer_name", default="radio_dual")
    parser.add_argument("--feature_mode", default="radio_dual", choices=("radio_final", "radio_dual"))
    parser.add_argument("--radio_fine_intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_coarse_source", default="final", choices=("final", "intermediate"))
    parser.add_argument("--radio_coarse_intermediate_index", type=int, default=-1)
    parser.add_argument("--query_feature_cache_dir", default="")
    parser.add_argument("--query_feature_cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--synthetic_pair_cache_dir", default="")
    parser.add_argument("--synthetic_pair_cache_format", default="npz", choices=("npz", "compressed_npz"))
    parser.add_argument("--synthetic_pair_memory_cache_size", type=int, default=1)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--pair_type_B_translation_m", type=float, default=0.25)
    parser.add_argument("--pair_type_C_translation_m", type=float, default=0.5)
    parser.add_argument("--perturb_rotation_deg", type=float, default=0.0)
    parser.add_argument("--feature_fusion_mode", default="none", choices=("none", "local_attention"))
    parser.add_argument("--feature_fusion_radius", type=int, default=1)
    parser.add_argument("--feature_fusion_temperature", type=float, default=5.0)
    parser.add_argument("--feature_fusion_alpha", type=float, default=0.5)
    parser.add_argument("--roundtrip_threshold_px", type=float, default=1.5)
    parser.add_argument("--roundtrip_heatmap_threshold_px", type=float, default=2.0)
    parser.add_argument("--visibility_alpha_threshold", type=float, default=0.0)
    parser.add_argument("--depth_edge_threshold_m", type=float, default=0.0)
    parser.add_argument("--multiview_supervision_support_views", type=int, default=3)
    parser.add_argument("--multiview_supervision_min_support_views", type=int, default=1)
    parser.add_argument("--multiview_supervision_depth_tolerance_m", type=float, default=0.05)
    parser.add_argument("--collect_visibility_no_match", action="store_true")
    parser.add_argument("--max_visibility_no_match", type=int, default=256)
    parser.add_argument("--soft_offset_sigma_bins", type=float, default=0.75)
    parser.add_argument("--pose_confidence_label_source", default="supervision", choices=("supervision", "gt_reprojection"))
    parser.add_argument("--pose_confidence_labels", action="store_true")
    parser.add_argument("--pose_confidence_positive_threshold_px", type=float, default=8.0)
    parser.add_argument("--pose_confidence_negative_threshold_px", type=float, default=24.0)
    parser.add_argument("--hard_negatives_per_match", type=int, default=16)
    parser.add_argument("--fine_supervision_source", default="render_subcell_stratified", choices=("cell_center", "render_subcell", "render_subcell_stratified", "render_alike"))
    parser.add_argument(
        "--merge_fine_labels_into_coarse",
        action="store_true",
        help="Legacy path: copy fine seed labels onto one-to-one coarse rows. Disabled by default because random sub-cell labels are not learnable from a single coarse token.",
    )
    parser.add_argument("--keypoint_distill_method", default="alike", choices=("none", "alike"))
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=4096)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=8000)
    parser.add_argument("--model_type", choices=("residual_adapter", "radio_dual_attention"), default="radio_dual_attention")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--fine_input_dim", type=int, default=1280)
    parser.add_argument("--coarse_input_dim", type=int, default=1280)
    parser.add_argument("--attention_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_depth", type=int, default=1)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_patch_size", type=int, default=4)
    parser.add_argument("--attention_upsample_mode", choices=("bilinear", "pixel_shuffle"), default="pixel_shuffle")
    parser.add_argument("--attention_fusion_mode", choices=("legacy", "matcha_original"), default="matcha_original")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--offset_loss_weight", type=float, default=0.25)
    parser.add_argument("--pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--render_pair_fine_loss_weight", type=float, default=None)
    parser.add_argument("--query_pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--fine_continuous_loss_weight", type=float, default=0.25)
    parser.add_argument("--fine_uncertainty_loss_weight", type=float, default=0.05)
    parser.add_argument("--pair_confidence_loss_weight", type=float, default=0.1)
    parser.add_argument("--dense_heatmap_loss_weight", type=float, default=0.25)
    parser.add_argument("--rgb_keypoint_loss_weight", type=float, default=0.25)
    parser.add_argument("--rgb_keypoint_position_loss_weight", type=float, default=0.0)
    parser.add_argument("--repeatability_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_fine_transformer_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_window_fine_loss_weight", type=float, default=0.5)
    parser.add_argument("--patch_corr_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--patch_corr_fine_epe_weight", type=float, default=0.1)
    parser.add_argument("--patch_corr_fine_batch_size", type=int, default=256)
    parser.add_argument("--patch_corr_fine_max_samples_per_pair", type=int, default=512)
    parser.add_argument("--patch_corr_fine_backprop_context", action="store_true")
    parser.add_argument("--patch_correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--patch_correlation_window_size", type=int, default=3)
    parser.add_argument("--hard_negative_weight", type=float, default=0.1)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--coarse_candidate_rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--coarse_candidate_rank_margin", type=float, default=0.2)
    parser.add_argument("--hard_false_match_weight", type=float, default=0.0)
    parser.add_argument("--hard_false_match_margin", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--pair_type_curriculum", default="uniform", choices=("uniform", "robust_default", "conservative_25cm"))
    parser.add_argument("--freeze_descriptor_steps", type=int, default=0)
    parser.add_argument("--descriptor_lr_scale", type=float, default=1.0)
    parser.add_argument("--head_lr_scale", type=float, default=1.0)
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--validation_pair_count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--builder_device", default="")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if str(args.matcha_train_preset) == "radio_matcha_patch_corr":
        args.fine_supervision_source = "render_subcell_stratified"
        args.multiview_supervision_support_views = 3
        args.multiview_supervision_min_support_views = 1
        args.offset_loss_weight = 0.0
        args.pair_fine_loss_weight = 0.0
        args.query_pair_fine_loss_weight = 0.0
        args.local_window_fine_loss_weight = 0.0
        args.patch_correlation_loss_weight = 0.0
        args.patch_corr_fine_loss_weight = 1.0
    if str(args.matcha_train_preset) == "radio_matcha_2dgs_synthetic":
        args.fine_supervision_source = "render_subcell_stratified"
        args.multiview_supervision_support_views = 0
        args.multiview_supervision_min_support_views = 0
        args.offset_loss_weight = 0.25
        args.pair_confidence_loss_weight = 0.1
        args.dense_heatmap_loss_weight = 0.25
        args.rgb_keypoint_loss_weight = 0.25
        args.local_window_fine_loss_weight = 1.0
        args.patch_correlation_loss_weight = 0.0
        args.patch_corr_fine_loss_weight = 0.0
    if args.render_pair_fine_loss_weight is not None:
        args.pair_fine_loss_weight = float(args.render_pair_fine_loss_weight)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    manifest = MatchaStreamingPairManifest.from_json(Path(args.streaming_manifest))
    synthetic_manifest_paths = _split_optional_manifest_paths(args.synthetic_streaming_manifest)
    synthetic_manifests = [MatchaStreamingPairManifest.from_json(Path(path)) for path in synthetic_manifest_paths]
    val_manifest = MatchaStreamingPairManifest.from_json(Path(args.validation_streaming_manifest)) if str(args.validation_streaming_manifest) else None
    cfg = MatchaJointTrainingConfig(
        model_type=str(args.model_type),
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        fine_input_dim=int(args.fine_input_dim),
        coarse_input_dim=int(args.coarse_input_dim),
        attention_hidden_dim=int(args.attention_hidden_dim),
        attention_depth=int(args.attention_depth),
        attention_heads=int(args.attention_heads),
        attention_patch_size=int(args.attention_patch_size),
        attention_upsample_mode=str(args.attention_upsample_mode),
        attention_fusion_mode=str(args.attention_fusion_mode),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        offset_loss_weight=float(args.offset_loss_weight),
        pair_fine_loss_weight=float(args.pair_fine_loss_weight),
        query_pair_fine_loss_weight=float(args.query_pair_fine_loss_weight),
        fine_continuous_loss_weight=float(args.fine_continuous_loss_weight),
        fine_uncertainty_loss_weight=float(args.fine_uncertainty_loss_weight),
        pair_confidence_loss_weight=float(args.pair_confidence_loss_weight),
        dense_heatmap_loss_weight=float(args.dense_heatmap_loss_weight),
        rgb_keypoint_loss_weight=float(args.rgb_keypoint_loss_weight),
        rgb_keypoint_position_loss_weight=float(args.rgb_keypoint_position_loss_weight),
        repeatability_loss_weight=float(args.repeatability_loss_weight),
        local_fine_transformer_loss_weight=float(args.local_fine_transformer_loss_weight),
        local_window_fine_loss_weight=float(args.local_window_fine_loss_weight),
        patch_corr_fine_loss_weight=float(args.patch_corr_fine_loss_weight),
        patch_corr_fine_epe_weight=float(args.patch_corr_fine_epe_weight),
        patch_corr_fine_batch_size=int(args.patch_corr_fine_batch_size),
        patch_corr_fine_max_samples_per_pair=int(args.patch_corr_fine_max_samples_per_pair),
        patch_corr_fine_detach_context=not bool(args.patch_corr_fine_backprop_context),
        patch_correlation_loss_weight=float(args.patch_correlation_loss_weight),
        patch_correlation_window_size=int(args.patch_correlation_window_size),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        coarse_candidate_rank_loss_weight=float(args.coarse_candidate_rank_loss_weight),
        coarse_candidate_rank_margin=float(args.coarse_candidate_rank_margin),
        hard_false_match_weight=float(args.hard_false_match_weight),
        hard_false_match_margin=float(args.hard_false_match_margin),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        map_pair_batch_size=int(args.map_pair_batch_size),
        device=str(args.device),
        seed=int(args.seed),
    )
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")
    builder = _builder_class_for_manifest(manifest)(args, manifest)
    synthetic_builders = [_builder_class_for_manifest(item)(args, item) for item in synthetic_manifests]
    for synthetic_builder in synthetic_builders:
        _share_builder_runtime_state(builder, synthetic_builder)
    if synthetic_builders and device.type == "cuda":
        torch.cuda.empty_cache()
    val_builder = _builder_class_for_manifest(val_manifest)(args, val_manifest) if val_manifest is not None else None
    _share_builder_runtime_state(builder, val_builder)
    if val_builder is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    real_pair_type_weights = _parse_pair_type_sampling_weights(args.real_pair_type_sampling_weights)
    training_sources = [
        TrainingSource(
            name="primary",
            manifest=manifest,
            builder=builder,
            sampling_weight=float(args.real_sampling_weight),
            pair_type_weights=real_pair_type_weights,
        ),
    ]
    synthetic_source_weight = float(args.synthetic_sampling_weight) / float(len(synthetic_builders)) if synthetic_builders else 0.0
    for synthetic_index, (synthetic_path, synthetic_manifest, synthetic_builder) in enumerate(
        zip(synthetic_manifest_paths, synthetic_manifests, synthetic_builders)
    ):
        training_sources.append(
            TrainingSource(
                name=_synthetic_source_name(str(synthetic_path), synthetic_manifest, synthetic_index),
                manifest=synthetic_manifest,
                builder=synthetic_builder,
                sampling_weight=float(synthetic_source_weight),
            )
        )
    rng = np.random.default_rng(int(cfg.seed))
    skipped = 0

    def build_with_retries(
        records: Sequence[MatchaStreamingPairRecord],
        source_builder: StreamingPairBuilder,
        *,
        seed_offset: int,
        source_name: str = "",
    ) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        nonlocal skipped
        samples, row, skipped_count = _build_streaming_pair_with_retries(
            records,
            source_builder,
            seed=int(cfg.seed) + int(seed_offset),
            max_attempts=32,
        )
        skipped += int(skipped_count)
        if str(source_name):
            row = dict(row)
            row["training_source"] = str(source_name)
        return samples, row

    first_source = _select_training_source_for_step(
        training_sources,
        rng=np.random.default_rng(int(cfg.seed) + 17),
        step=0,
        total_steps=int(cfg.steps),
        curriculum=str(args.pair_type_curriculum),
    )
    first_samples, first_row = build_with_retries(
        first_source.records,
        first_source.builder,
        seed_offset=0,
        source_name=first_source.name,
    )
    model = _build_matcha_joint_model_for_samples(first_samples, cfg, device)
    warm_start_report: dict[str, object] = {}
    if str(args.warm_start_joint_checkpoint):
        warm = load_matcha_joint_model(Path(args.warm_start_joint_checkpoint), device="cpu").model
        result = model.load_state_dict(warm.state_dict(), strict=False)
        del warm
        if device.type == "cuda":
            torch.cuda.empty_cache()
        warm_start_report = {
            "warm_start_loaded": True,
            "warm_start_missing_keys": list(result.missing_keys),
            "warm_start_unexpected_keys": list(result.unexpected_keys),
        }
    optimizer, optimizer_report = _make_optimizer(
        model,
        lr=float(cfg.lr),
        descriptor_lr_scale=float(args.descriptor_lr_scale),
        head_lr_scale=float(args.head_lr_scale),
        freeze_descriptor=int(args.freeze_descriptor_steps) > 0,
    )
    initial_loss = _loss_value_for_samples(model, first_samples, cfg, device, seed=int(cfg.seed))
    validation_history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_validation_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    train_rows = [first_row]

    def maybe_validate(step: int) -> None:
        nonlocal best_state, best_validation_loss, best_validation_step
        if val_manifest is None or val_builder is None or int(args.validation_pair_count) <= 0:
            return
        values = []
        for item in range(int(args.validation_pair_count)):
            samples, _row = build_with_retries(
                val_manifest.records,
                val_builder,
                seed_offset=50000 + int(step) * 97 + item,
                source_name="validation",
            )
            values.append(_loss_value_for_samples(model, samples, cfg, device, seed=int(cfg.seed) + 50000 + int(step) + item))
        value = float(np.mean(values)) if values else float("inf")
        validation_history.append({"step": int(step), "loss": value})
        if value < best_validation_loss:
            best_validation_loss = value
            best_validation_step = int(step)
            best_state = _model_state_snapshot(model)

    maybe_validate(0)
    model.train()
    last_samples = first_samples
    last_metrics: dict[str, float] = {}
    for step in range(int(cfg.steps)):
        descriptor_frozen = int(step) < int(args.freeze_descriptor_steps)
        _set_descriptor_group_lr(
            optimizer,
            base_lr=float(cfg.lr),
            descriptor_lr_scale=float(args.descriptor_lr_scale),
            freeze=descriptor_frozen,
        )
        _set_frozen_descriptor_batchnorm_eval(model, freeze=descriptor_frozen)
        active_source = _select_training_source_for_step(
            training_sources,
            rng=rng,
            step=int(step),
            total_steps=int(cfg.steps),
            curriculum=str(args.pair_type_curriculum),
        )
        samples, row = build_with_retries(
            active_source.records,
            active_source.builder,
            seed_offset=10000 + int(step) * 97,
            source_name=active_source.name,
        )
        train_rows.append(row)
        last_samples = samples
        batch_idx = _sample_indices(rng, samples.coarse_fine_samples.sample_count, int(cfg.batch_size))
        loss, last_metrics = _total_loss(model, samples, batch_idx, cfg, device, seed=int(cfg.seed) + int(step))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if int(args.validation_interval) > 0 and ((int(step) + 1) % int(args.validation_interval) == 0):
            maybe_validate(int(step) + 1)
    if val_manifest is not None and (not validation_history or int(validation_history[-1]["step"]) != int(cfg.steps)):
        maybe_validate(int(cfg.steps))
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss = _loss_value_for_samples(model, last_samples, cfg, device, seed=int(cfg.seed) + int(cfg.steps) + 1)
    all_train_records = [record for source in training_sources for record in source.manifest.records]
    manifest_pair_type_counts: dict[str, int] = {}
    for record in all_train_records:
        manifest_pair_type_counts[str(record.pair_type)] = int(manifest_pair_type_counts.get(str(record.pair_type), 0) + 1)
    training_source_counts: dict[str, int] = {}
    for row in train_rows:
        source_name = str(row.get("training_source", "primary"))
        training_source_counts[source_name] = int(training_source_counts.get(source_name, 0) + 1)
    summary = {
        "stage": "matcha_style_streaming_joint_training",
        "streaming_training": True,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "model_type": str(cfg.model_type),
        "input_dim": int(first_samples.coarse_fine_samples.input_dim),
        "output_dim": int(cfg.output_dim),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "manifest_pair_count": int(len(all_train_records)),
        "manifest_query_count": int(len({record.query_id for record in all_train_records})),
        "manifest_pair_type_counts": manifest_pair_type_counts,
        "training_source_counts": training_source_counts,
        "pair_type_curriculum": str(args.pair_type_curriculum),
        "real_sampling_weight": float(args.real_sampling_weight),
        "synthetic_sampling_weight": float(args.synthetic_sampling_weight),
        "real_pair_type_sampling_weights": real_pair_type_weights,
        "freeze_descriptor_steps": int(args.freeze_descriptor_steps),
        "descriptor_lr_scale": float(args.descriptor_lr_scale),
        "head_lr_scale": float(args.head_lr_scale),
        "optimizer": optimizer_report,
        "skipped_pair_builds": int(skipped),
        "first_pair": first_row,
        "last_step_metrics": {key: float(value) for key, value in last_metrics.items()},
    }
    synthetic_counts = _synthetic_pose_bin_counts(train_rows)
    if synthetic_counts:
        summary["synthetic_pose_bin_counts"] = synthetic_counts
    summary["supervision_balance"] = _supervision_balance_summary(train_rows)
    cache_stats = _synthetic_pair_cache_stats(builder, *synthetic_builders, val_builder)
    if str(args.synthetic_pair_cache_dir):
        summary["synthetic_pair_cache"] = {
            "dir": str(args.synthetic_pair_cache_dir),
            "format": str(args.synthetic_pair_cache_format),
            **cache_stats,
        }
    summary.update(warm_start_report)
    if validation_history:
        summary.update(
            {
                "best_validation_loss": float(best_validation_loss),
                "best_validation_step": int(best_validation_step),
                "validation_history": validation_history,
                "validation_pair_count": int(args.validation_pair_count),
            }
        )
    summary.update(_evaluate(model, last_samples, cfg, device))
    run = MatchaJointTrainingRun(model=model.cpu().eval(), summary=summary)
    adapter_run = joint_run_as_coarse_fine_adapter_run(run)
    save_matcha_coarse_fine_adapter(adapter_run, Path(args.output_model))
    joint_output = Path(args.output_joint_model) if str(args.output_joint_model) else Path(args.output_model).with_name(Path(args.output_model).stem + "_joint.pt")
    save_matcha_joint_model(run, joint_output)
    best_output = Path(args.output_best_joint_model) if str(args.output_best_joint_model) else None
    if best_output is not None:
        save_matcha_joint_model(run, best_output)
    output_summary = {
        "stage": "matcha_style_streaming_joint_model_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "streaming_manifest": {
            "path": str(args.streaming_manifest),
            "query_count": int(manifest.query_count),
            "pair_count": int(len(manifest.records)),
            "pair_type_counts": manifest.pair_type_counts,
            "metadata": dict(manifest.metadata),
        },
        "synthetic_streaming_manifest": _synthetic_manifest_report(
            synthetic_manifest_paths,
            synthetic_manifests,
            sampling_weight=float(args.synthetic_sampling_weight),
        ),
        "validation_streaming_manifest": {
            "path": str(args.validation_streaming_manifest),
            "query_count": int(val_manifest.query_count) if val_manifest is not None else 0,
            "pair_count": int(len(val_manifest.records)) if val_manifest is not None else 0,
        },
        "feature_cache_policy": {
            "query_feature_cache_dir": str(args.query_feature_cache_dir),
            "query_feature_cache_dtype": str(args.query_feature_cache_dtype),
            "render_feature_cache": "disabled",
            "training_sample_cache": "enabled" if str(args.synthetic_pair_cache_dir) else "disabled",
            "synthetic_pair_cache_dir": str(args.synthetic_pair_cache_dir),
            "synthetic_pair_cache_format": str(args.synthetic_pair_cache_format),
            "synthetic_pair_memory_cache_size": int(args.synthetic_pair_memory_cache_size),
            "synthetic_pair_cache_hits": int(cache_stats["hits"]),
            "synthetic_pair_cache_misses": int(cache_stats["misses"]),
            "builder_device": str(args.builder_device or args.device),
            "training_device": str(args.device),
        },
        "robust_supervision": {
            "roundtrip_threshold_px": float(args.roundtrip_threshold_px),
            "visibility_alpha_threshold": float(args.visibility_alpha_threshold),
            "depth_edge_threshold_m": float(args.depth_edge_threshold_m),
            "multiview_supervision_support_views": int(args.multiview_supervision_support_views),
            "multiview_supervision_min_support_views": int(args.multiview_supervision_min_support_views),
            "multiview_supervision_depth_tolerance_m": float(args.multiview_supervision_depth_tolerance_m),
            "collect_visibility_no_match": bool(args.collect_visibility_no_match),
            "max_visibility_no_match": int(args.max_visibility_no_match),
            "soft_offset_sigma_bins": float(args.soft_offset_sigma_bins),
            "pose_confidence_label_source": str(args.pose_confidence_label_source),
            "pose_confidence_labels": bool(args.pose_confidence_labels),
            "pose_confidence_positive_threshold_px": float(args.pose_confidence_positive_threshold_px),
            "pose_confidence_negative_threshold_px": float(args.pose_confidence_negative_threshold_px),
        },
        "curriculum": {
            "pair_type_curriculum": str(args.pair_type_curriculum),
            "real_sampling_weight": float(args.real_sampling_weight),
            "synthetic_sampling_weight": float(args.synthetic_sampling_weight),
            "real_pair_type_sampling_weights": real_pair_type_weights,
            "freeze_descriptor_steps": int(args.freeze_descriptor_steps),
            "descriptor_lr_scale": float(args.descriptor_lr_scale),
            "head_lr_scale": float(args.head_lr_scale),
        },
        "config": asdict(cfg),
        "training": summary,
        "outputs": {
            "adapter_model": str(args.output_model),
            "joint_model": str(joint_output),
            "best_joint_model": str(best_output) if best_output is not None else "",
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
