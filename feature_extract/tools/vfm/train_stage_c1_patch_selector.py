"""Train a Stage C1 supervised linear patch selector and export compressed banks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_c0_compressed_features import (
    _dir_bytes,
    _transform_track_bank_npz,
    _write_compressed_query_manifest,
)
from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _limit_submap,
    _load_camera_with_source,
    _load_query_feature,
    _load_reference_submaps,
    _load_track_stats,
    _parse_default_camera,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_selector_training import (
    PatchSelectorSampleConfig,
    PatchSelectorTrainingConfig,
    append_patch_selector_training_set_capped,
    build_patch_selector_samples_for_query,
    load_safe_patch_selector_checkpoint,
    load_patch_selector_training_set_npz,
    merge_patch_selector_training_sets,
    save_patch_selector_training_set_npz,
    train_linear_patch_selector,
)
from feature_extract.vfm.patch_to_3d_matching import build_patch_positive_sets, filter_landmarks_by_projected_visibility
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, filter_landmarks_by_reference_images
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def _stable_query_seed(seed: int, index: int, query_id: str) -> int:
    value = int(seed) * 1000003 + int(index)
    for char in str(query_id):
        value = (value * 131 + ord(char)) % (2**32)
    return int(value)


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _path_sha_or_empty(path: str) -> str:
    if not str(path):
        return ""
    value = Path(path)
    return file_sha256_short(value) if value.exists() and value.is_file() else ""


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Stage C1 supervised linear patch selector")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", default="")
    parser.add_argument("--semidense_anchor_npz", default="")
    parser.add_argument("--track_observations", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "gt_visible", "none"))
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--max_tokens_per_query", type=int, default=256)
    parser.add_argument("--max_positives_per_token", type=int, default=4)
    parser.add_argument("--hard_negatives_per_token", type=int, default=32)
    parser.add_argument("--hard_negative_pool", type=int, default=256)
    parser.add_argument("--negative_curriculum", default="hard", choices=("hard", "medium", "semi_hard", "mixed"))
    parser.add_argument("--medium_min_stride", type=float, default=4.0)
    parser.add_argument("--semi_hard_min_stride", type=float, default=2.0)
    parser.add_argument("--semi_hard_max_stride", type=float, default=4.0)
    parser.add_argument("--mixed_hard_fraction", type=float, default=0.1)
    parser.add_argument("--min_positive_count", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=20000)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--sample_cache", default="")
    parser.add_argument("--write_sample_cache", default="")
    parser.add_argument("--build_sample_cache_only", action="store_true")
    parser.add_argument("--negative_mining_safe_checkpoint", default="")
    parser.add_argument("--negative_mining_device", default="")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
    parser.add_argument("--center_inputs", action="store_true")
    parser.add_argument("--group_size", type=int, default=0)
    parser.add_argument("--group_lasso_weight", type=float, default=0.0)
    parser.add_argument("--hard_gate_keep_fraction", type=float, default=1.0)
    parser.add_argument("--hard_gate_min_groups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_transform", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_query_dir", default="")
    parser.add_argument("--output_query_manifest", default="")
    parser.add_argument("--output_landmark_bank", default="")
    parser.add_argument("--export_device", default="")
    parser.add_argument("--batch_rows", type=int, default=65536)
    parser.add_argument("--batch_tokens", type=int, default=65536)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = None
    records = []
    camera_source = "not_loaded_sample_cache"
    merged_samples = None
    sampled_query_count = 0
    query_sample_counts = []
    query_positive_counts = []
    skipped_missing_pose = 0
    skipped_empty_submap = 0
    if args.sample_cache:
        merged_samples, _cache_metadata = load_patch_selector_training_set_npz(Path(args.sample_cache))
        if args.max_train_samples > 0 and merged_samples.sample_count > int(args.max_train_samples):
            merged_samples = merge_patch_selector_training_sets(
                [merged_samples],
                max_samples=int(args.max_train_samples),
                seed=int(args.seed),
            )
    else:
        manifest = TokenBankManifest.from_json(Path(args.query_manifest))
        manifest.validate(verify_checksums=False)
        if args.semidense_anchor_npz:
            landmark_index = SemiDenseAnchorMap.load_npz(Path(args.semidense_anchor_npz)).to_landmark_index()
        else:
            if not args.landmark_bank or not args.track_observations:
                raise ValueError("provide --landmark_bank and --track_observations, or --semidense_anchor_npz")
            bank = load_selected_track_bank_npz(Path(args.landmark_bank))
            xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
            landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
        visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index)) if args.visibility_index else None
        reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n)
        gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
        camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
        camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
        negative_mining_run = (
            load_safe_patch_selector_checkpoint(Path(args.negative_mining_safe_checkpoint), device=args.negative_mining_device or args.device)
            if args.negative_mining_safe_checkpoint
            else None
        )

        records = list(manifest.records)
        if args.max_queries > 0:
            records = records[: int(args.max_queries)]

        for record_idx, record in enumerate(records):
            gt_pose = gt_by_query.get(record.image_id)
            if gt_pose is None:
                skipped_missing_pose += 1
                continue
            if args.submap_mode == "gt_visible":
                submap = filter_landmarks_by_projected_visibility(landmark_index, gt_pose.pose_w2c, camera)
            elif args.submap_mode == "reference_visibility":
                references = reference_submaps.get(record.image_id, [])
                if visibility_index is not None:
                    submap, _gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
                else:
                    submap = filter_landmarks_by_reference_images(landmark_index, references)
            else:
                submap = landmark_index
            submap = _limit_submap(submap, int(args.max_submap_landmarks))
            if len(submap) == 0:
                skipped_empty_submap += 1
                continue
            query_feature = _load_query_feature(record.token_path, args.layer_name)
            _channels, token_height, token_width = query_feature.shape
            positives = build_patch_positive_sets(
                submap,
                gt_pose.pose_w2c,
                camera,
                token_width=token_width,
                token_height=token_height,
                patch_scale=float(args.patch_scale),
            )
            negative_mining_query_feature = None
            negative_mining_landmark_features = None
            if negative_mining_run is not None:
                channels, height, width = query_feature.shape
                query_rows = query_feature.reshape(channels, height * width).T
                encoded_query = negative_mining_run.encode_rows(
                    query_rows,
                    device=args.negative_mining_device or args.device,
                    batch_size=int(args.batch_tokens),
                )
                negative_mining_query_feature = encoded_query.T.reshape(negative_mining_run.summary.output_dim, height, width)
                negative_mining_landmark_features = negative_mining_run.encode_rows(
                    submap.features,
                    device=args.negative_mining_device or args.device,
                    batch_size=int(args.batch_rows),
                )
            sample_config = PatchSelectorSampleConfig(
                max_tokens_per_query=int(args.max_tokens_per_query),
                max_positives_per_token=int(args.max_positives_per_token),
                hard_negatives_per_token=int(args.hard_negatives_per_token),
                hard_negative_pool=int(args.hard_negative_pool),
                query_token_step=int(args.query_token_step),
                min_positive_count=int(args.min_positive_count),
                seed=_stable_query_seed(args.seed, record_idx, record.image_id),
                negative_curriculum=str(args.negative_curriculum),
                medium_min_stride=float(args.medium_min_stride),
                semi_hard_min_stride=float(args.semi_hard_min_stride),
                semi_hard_max_stride=float(args.semi_hard_max_stride),
                mixed_hard_fraction=float(args.mixed_hard_fraction),
            )
            samples = build_patch_selector_samples_for_query(
                query_feature,
                submap,
                positives,
                sample_config,
                negative_mining_query_feature_map=negative_mining_query_feature,
                negative_mining_landmark_features=negative_mining_landmark_features,
            )
            if samples.sample_count > 0:
                sampled_query_count += 1
                query_sample_counts.append(float(samples.sample_count))
                query_positive_counts.append(float(sum(item.count for item in positives.by_token.values())))
                merged_samples = append_patch_selector_training_set_capped(
                    merged_samples,
                    samples,
                    max_samples=int(args.max_train_samples),
                    seed=_stable_query_seed(args.seed, record_idx, record.image_id),
                )
        if merged_samples is None:
            raise ValueError("no patch selector training samples were built")
    if args.write_sample_cache:
        save_patch_selector_training_set_npz(
            merged_samples,
            Path(args.write_sample_cache),
        )
    camera_summary = (
        {"source": camera_source, "model_id": None, "width": None, "height": None, "params": []}
        if args.sample_cache
        else {
            "source": camera_source,
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        }
    )
    if args.build_sample_cache_only:
        if not args.write_sample_cache:
            raise ValueError("--build_sample_cache_only requires --write_sample_cache")
        summary = {
            "stage": "stage_c1_supervised_linear_patch_selector",
            "cache_only": True,
            "elapsed_sec": float(time.perf_counter() - started),
            "query_count": len(records),
            "sampled_query_count": int(sampled_query_count),
            "skipped_missing_pose": int(skipped_missing_pose),
            "skipped_empty_submap": int(skipped_empty_submap),
            "camera": camera_summary,
            "sample_config": {
                "submap_mode": args.submap_mode,
                "submap_top_n": int(args.submap_top_n),
                "max_submap_landmarks": int(args.max_submap_landmarks),
                "patch_scale": float(args.patch_scale),
                "query_token_step": int(args.query_token_step),
                "max_tokens_per_query": int(args.max_tokens_per_query),
                "max_positives_per_token": int(args.max_positives_per_token),
                "hard_negatives_per_token": int(args.hard_negatives_per_token),
                "hard_negative_pool": int(args.hard_negative_pool),
                "negative_curriculum": str(args.negative_curriculum),
                "medium_min_stride": float(args.medium_min_stride),
                "semi_hard_min_stride": float(args.semi_hard_min_stride),
                "semi_hard_max_stride": float(args.semi_hard_max_stride),
                "mixed_hard_fraction": float(args.mixed_hard_fraction),
                "min_positive_count": int(args.min_positive_count),
                "max_train_samples": int(args.max_train_samples),
            },
            "sample_summary": {
                "sample_count": int(merged_samples.sample_count),
                "sample_cache": str(args.sample_cache) if args.sample_cache else "",
                "written_sample_cache": str(args.write_sample_cache),
                "per_query_samples": _summary_stats(query_sample_counts),
                "per_query_positive_assignments": _summary_stats(query_positive_counts),
            },
            "inputs": {
                "query_manifest": {
                    "path": str(args.query_manifest),
                    "sha256": _path_sha_or_empty(args.query_manifest),
                },
                "landmark_bank": {
                    "path": str(args.landmark_bank),
                    "sha256": _path_sha_or_empty(args.landmark_bank),
                },
                "semidense_anchor_npz": {
                    "path": str(args.semidense_anchor_npz),
                    "sha256": _path_sha_or_empty(args.semidense_anchor_npz),
                },
                "track_observations": str(args.track_observations),
                "query_pose_file": str(args.query_pose_file),
                "candidate_bank": str(args.candidate_bank),
                "visibility_index": str(args.visibility_index),
            },
        }
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return
    run = train_linear_patch_selector(
        merged_samples,
        PatchSelectorTrainingConfig(
            output_dim=int(args.output_dim),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            temperature=float(args.temperature),
            seed=int(args.seed),
            device=args.device,
            eval_split_fraction=float(args.eval_split_fraction),
            center_inputs=bool(args.center_inputs),
            group_size=int(args.group_size),
            group_lasso_weight=float(args.group_lasso_weight),
            hard_gate_keep_fraction=float(args.hard_gate_keep_fraction),
            hard_gate_min_groups=int(args.hard_gate_min_groups),
        ),
    )
    run.transform.to_npz(
        Path(args.output_transform),
        metadata={
            "stage": "stage_c1_supervised_linear_patch_selector",
            "source_landmark_bank": str(args.landmark_bank),
            "source_semidense_anchor_npz": str(args.semidense_anchor_npz),
            "source_query_manifest": str(args.query_manifest),
            "query_pose_file": str(args.query_pose_file),
            "candidate_bank": str(args.candidate_bank),
            "submap_mode": str(args.submap_mode),
            "submap_top_n": int(args.submap_top_n),
            "seed": int(args.seed),
            "negative_curriculum": str(args.negative_curriculum),
        },
    )

    export_summary: dict[str, object] = {}
    export_args = [args.output_query_dir, args.output_query_manifest, args.output_landmark_bank]
    if any(export_args):
        if not all(export_args):
            raise ValueError("--output_query_dir, --output_query_manifest and --output_landmark_bank must be provided together")
        if args.semidense_anchor_npz:
            raise ValueError("export from --semidense_anchor_npz is not supported by this script; use project_stage_h2_gaussian_anchor_map.py")
        if manifest is None:
            manifest = TokenBankManifest.from_json(Path(args.query_manifest))
            manifest.validate(verify_checksums=False)
        export_device = args.export_device or args.device
        mapability = _transform_track_bank_npz(
            Path(args.landmark_bank),
            Path(args.output_landmark_bank),
            run.transform,
            device=export_device,
            batch_rows=int(args.batch_rows),
        )
        output_layer_name = args.output_layer_name or args.layer_name
        compressed_manifest, query_count = _write_compressed_query_manifest(
            manifest,
            run.transform,
            layer_name=args.layer_name,
            output_query_dir=Path(args.output_query_dir),
            output_layer_name=output_layer_name,
            device=export_device,
            batch_tokens=int(args.batch_tokens),
        )
        compressed_manifest.to_json(Path(args.output_query_manifest))
        export_summary = {
            "query_record_count": int(query_count),
            "mapability": mapability,
            "device": export_device,
            "storage_bytes": {
                "query_tokens": _dir_bytes(Path(args.output_query_dir)),
                "landmark_bank": _dir_bytes(Path(args.output_landmark_bank)),
                "transform": _dir_bytes(Path(args.output_transform)),
            },
            "outputs": {
                "query_manifest": str(args.output_query_manifest),
                "query_dir": str(args.output_query_dir),
                "landmark_bank": str(args.output_landmark_bank),
                "transform": str(args.output_transform),
            },
        }

    summary = {
        "stage": "stage_c1_supervised_linear_patch_selector",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": len(records),
        "sampled_query_count": int(sampled_query_count),
        "skipped_missing_pose": int(skipped_missing_pose),
        "skipped_empty_submap": int(skipped_empty_submap),
        "camera": camera_summary,
        "sample_config": {
            "submap_mode": args.submap_mode,
            "submap_top_n": int(args.submap_top_n),
            "max_submap_landmarks": int(args.max_submap_landmarks),
            "patch_scale": float(args.patch_scale),
            "query_token_step": int(args.query_token_step),
            "max_tokens_per_query": int(args.max_tokens_per_query),
            "max_positives_per_token": int(args.max_positives_per_token),
            "hard_negatives_per_token": int(args.hard_negatives_per_token),
            "hard_negative_pool": int(args.hard_negative_pool),
            "negative_curriculum": str(args.negative_curriculum),
            "medium_min_stride": float(args.medium_min_stride),
            "semi_hard_min_stride": float(args.semi_hard_min_stride),
            "semi_hard_max_stride": float(args.semi_hard_max_stride),
            "mixed_hard_fraction": float(args.mixed_hard_fraction),
            "min_positive_count": int(args.min_positive_count),
            "max_train_samples": int(args.max_train_samples),
        },
        "sample_summary": {
            "sample_count": int(merged_samples.sample_count),
            "sample_cache": str(args.sample_cache) if args.sample_cache else "",
            "written_sample_cache": str(args.write_sample_cache) if args.write_sample_cache else "",
            "per_query_samples": _summary_stats(query_sample_counts),
            "per_query_positive_assignments": _summary_stats(query_positive_counts),
        },
        "training": asdict(run.summary),
        "inputs": {
            "query_manifest": {
                "path": str(args.query_manifest),
                "sha256": _path_sha_or_empty(args.query_manifest),
            },
            "landmark_bank": {
                "path": str(args.landmark_bank),
                "sha256": _path_sha_or_empty(args.landmark_bank),
            },
            "track_observations": str(args.track_observations),
            "query_pose_file": str(args.query_pose_file),
            "candidate_bank": str(args.candidate_bank),
            "visibility_index": str(args.visibility_index),
        },
        "export": export_summary,
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
