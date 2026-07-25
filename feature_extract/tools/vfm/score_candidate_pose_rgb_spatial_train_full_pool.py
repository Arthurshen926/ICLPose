"""Score every frozen *train* pose with a target-free RGB spatial likelihood.

This is deliberately a two-stage protocol.  The scorer sees the fixed P1
query/candidate/support layout and frozen pose hypotheses, but never target
poses, residuals, registered tracks, or a target artifact.  Its output is a
row-keyed score overlay.  A separate train-only builder may later join those
row keys to target errors and projections to materialize coherent hard modes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood import (
    _crop_runtime_rgb_patches,
    _load_bank_xyz,
    _load_checkpoint_model,
    _query_geometry,
    _score_hypotheses,
    _slice_layout,
    _source_table,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    grouped_hypothesis_semantic_manifest,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_full_pool_scores import (
    CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
    CandidatePoseRGBSpatialTrainFullPoolScores,
    save_candidate_pose_rgb_spatial_train_full_pool_scores,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    candidate_pose_rgb_spatial_score_component_prediction,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


_HYPOTHESIS_FORMAT = "grouped_pose_hypotheses_inference_only_v1"
_SCORE_COMPONENTS = (
    "combined",
    "rgb_cost_volume",
    "rgb_cost_volume_with_dustbin",
    "learned_spatial_no_context",
    "spatial_residual",
    "context_only",
)


@dataclass(frozen=True)
class _DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


@dataclass(frozen=True)
class _FrozenTrainQuery:
    query_id: str
    source_artifact_index: int
    source_row_indices: np.ndarray
    hypothesis_indices: np.ndarray
    evaluation_labels: np.ndarray
    poses_w2c: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        hypotheses = np.asarray(self.hypothesis_indices, dtype=np.int64).reshape(-1)
        labels = np.asarray(self.evaluation_labels).astype(str).reshape(-1)
        poses = np.asarray(self.poses_w2c, dtype=np.float64)
        count = len(rows)
        if (
            not str(self.query_id)
            or int(self.source_artifact_index) < 0
            or count == 0
            or len(np.unique(rows)) != count
            or np.any(rows < 0)
            or np.any(hypotheses < 0)
            or labels.shape != (count,)
            or np.any(labels == "")
            or poses.shape != (count, 4, 4)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("frozen train full-pool query is invalid")
        object.__setattr__(self, "source_row_indices", rows)
        object.__setattr__(self, "hypothesis_indices", hypotheses)
        object.__setattr__(self, "evaluation_labels", labels)
        object.__setattr__(self, "poses_w2c", poses)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument(
        "--hypothesis-artifact",
        required=True,
        action="append",
        help="Repeat in the immutable source order used by the later target join.",
    )
    parser.add_argument("--evaluation-label", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hypothesis-batch-size", type=int, default=256)
    parser.add_argument("--edge-chunk-size", type=int, default=0)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument(
        "--rgb-cache-dtype", choices=("uint8", "float16", "float32"), default="uint8"
    )
    parser.add_argument("--score-component", choices=_SCORE_COMPONENTS, default="combined")
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help=(
            "Development-only cap on lexicographically ordered train queries. "
            "A nonzero value produces an explicitly incomplete artifact which the "
            "hard-target builder will reject."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _initialize_distributed(device_name: str) -> _DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("full-pool train scoring under torchrun requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("full-pool train scorer requested CUDA but CUDA is unavailable")
    return _DistributedState(rank, world_size, local_rank, device, enabled)


def _finalize_distributed(state: _DistributedState) -> None:
    if state.enabled and distributed.is_initialized():
        # A rank-local exception must not make every peer wait forever here.
        distributed.destroy_process_group()


def _load_metadata(payload: Mapping[str, np.ndarray], *, name: str) -> dict[str, object]:
    try:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} metadata is invalid")
    return metadata


def load_frozen_train_full_pool_queries(
    *,
    hypothesis_artifacts: Sequence[Path],
    evaluation_label: str,
    expected_train_query_ids: set[str],
) -> tuple[dict[str, _FrozenTrainQuery], dict[str, object]]:
    """Load only train rows from target-free frozen hypothesis shards."""

    paths = tuple(Path(path) for path in hypothesis_artifacts)
    label = str(evaluation_label)
    if not paths or len(paths) != len(set(paths)) or not label or not expected_train_query_ids:
        raise ValueError("train full-pool hypothesis sources are invalid")
    queries: dict[str, _FrozenTrainQuery] = {}
    source_manifest: list[dict[str, object]] = []
    semantic_hash = ""
    for artifact_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"frozen train full-pool hypothesis is absent: {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "query_ids",
                "split_names",
                "evaluation_labels",
                "hypothesis_indices",
                "poses_w2c",
                "metadata_json",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"frozen train full-pool hypothesis lacks {sorted(missing)}")
            arrays = {name: np.asarray(payload[name]).copy() for name in required if name != "metadata_json"}
            metadata = _load_metadata(payload, name="frozen train full-pool hypothesis")
        if (
            metadata.get("format") != _HYPOTHESIS_FORMAT
            or metadata.get("contains_target_fields") is not False
            or metadata.get("pose_or_ground_truth_used_for_generation") is not False
        ):
            raise ValueError("train full-pool source is not a target-free frozen hypothesis")
        manifest = grouped_hypothesis_semantic_manifest(metadata)
        current_hash = str(manifest["semantic_hash"])
        if semantic_hash and current_hash != semantic_hash:
            raise ValueError("train full-pool hypothesis semantic lineage differs across shards")
        semantic_hash = current_hash
        query_ids = np.asarray(arrays["query_ids"]).astype(str)
        splits = np.asarray(arrays["split_names"]).astype(str)
        labels = np.asarray(arrays["evaluation_labels"]).astype(str)
        hypotheses = np.asarray(arrays["hypothesis_indices"], dtype=np.int64)
        poses = np.asarray(arrays["poses_w2c"], dtype=np.float64)
        count = len(query_ids)
        if (
            count == 0
            or any(value.shape != (count,) for value in (splits, labels, hypotheses))
            or poses.shape != (count, 4, 4)
            or np.any(query_ids == "")
            or np.any(hypotheses < 0)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("train full-pool frozen hypothesis arrays are invalid")
        train_mask = (splits == "train") & (labels == label)
        for query_id in np.unique(query_ids[train_mask]).tolist():
            rows = np.flatnonzero(train_mask & (query_ids == str(query_id))).astype(np.int64)
            if str(query_id) in queries:
                raise ValueError("frozen train query appears in more than one hypothesis shard")
            queries[str(query_id)] = _FrozenTrainQuery(
                query_id=str(query_id),
                source_artifact_index=int(artifact_index),
                source_row_indices=rows,
                hypothesis_indices=hypotheses[rows],
                evaluation_labels=labels[rows],
                poses_w2c=poses[rows],
            )
        source_manifest.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
                "semantic_hash": current_hash,
            }
        )
    if set(queries) != set(expected_train_query_ids):
        missing = sorted(expected_train_query_ids.difference(queries))
        extra = sorted(set(queries).difference(expected_train_query_ids))
        raise ValueError(
            "train full-pool query coverage differs from layout: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return queries, {
        "hypothesis_artifacts": source_manifest,
        "hypothesis_semantic_hash": semantic_hash,
        "evaluation_label": label,
    }


def _validate_args(args: argparse.Namespace) -> None:
    values = (float(args.rgb_cache_gb),)
    if (
        int(args.hypothesis_batch_size) <= 0
        or int(args.edge_chunk_size) < 0
        or int(args.query_limit) < 0
        or not all(math.isfinite(value) and value > 0.0 for value in values)
        or not str(args.evaluation_label)
    ):
        raise ValueError("train full-pool RGB scorer arguments are invalid")


_SCORE_ROW_DTYPES: Mapping[str, np.dtype] = {
    "source_artifact_indices": np.dtype(np.int64),
    "source_row_indices": np.dtype(np.int64),
    "query_ids": np.dtype(np.str_),
    "split_names": np.dtype(np.str_),
    "evaluation_labels": np.dtype(np.str_),
    "hypothesis_indices": np.dtype(np.int64),
    "pose_log_likelihood_ratios": np.dtype(np.float32),
}


def _empty_score_rows() -> dict[str, np.ndarray]:
    """Return an empty shard with the same schemas as a nonempty rank."""

    return {name: np.empty((0,), dtype=dtype) for name, dtype in _SCORE_ROW_DTYPES.items()}


def _select_scored_train_query_ids(
    query_ids: Sequence[str], *, query_limit: int
) -> tuple[str, ...]:
    """Use a deterministic target-free subset only for scorer smoke tests."""

    ordered = tuple(sorted(str(query_id) for query_id in query_ids))
    limit = int(query_limit)
    if limit < 0:
        raise ValueError("train full-pool query limit is invalid")
    return ordered if limit == 0 else ordered[:limit]


def _validate_merged_score_rows(
    *,
    merged: Mapping[str, np.ndarray],
    frozen_queries: Mapping[str, _FrozenTrainQuery],
    scored_query_ids: Sequence[str],
) -> None:
    """Prove every scheduled frozen row was scored exactly once across ranks."""

    expected = {
        (int(frozen_queries[query_id].source_artifact_index), int(row_index))
        for query_id in scored_query_ids
        for row_index in frozen_queries[query_id].source_row_indices.tolist()
    }
    actual = set(
        zip(
            np.asarray(merged["source_artifact_indices"], dtype=np.int64).tolist(),
            np.asarray(merged["source_row_indices"], dtype=np.int64).tolist(),
        )
    )
    if not expected or actual != expected or len(actual) != len(merged["source_row_indices"]):
        raise ValueError("train full-pool RGB scorer rows do not exactly cover the scheduled sources")


def _merge_rank_rows(rows_by_rank: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows_by_rank:
        raise ValueError("train full-pool RGB scorer has no rank outputs")
    names = (
        "source_artifact_indices",
        "source_row_indices",
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "pose_log_likelihood_ratios",
    )
    if any(set(rows) != set(names) for rows in rows_by_rank):
        raise ValueError("train full-pool RGB rank outputs differ")
    merged = {name: np.concatenate([rows[name] for rows in rows_by_rank], axis=0) for name in names}
    order = np.lexsort(
        (
            np.asarray(merged["source_row_indices"], dtype=np.int64),
            np.asarray(merged["source_artifact_indices"], dtype=np.int64),
        )
    )
    return {name: np.asarray(value)[order] for name, value in merged.items()}


@torch.no_grad()
def score_candidate_pose_rgb_spatial_train_full_pool(args: argparse.Namespace) -> dict[str, object]:
    """Produce an inference-only score overlay for current train-pool errors."""

    _validate_args(args)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    state = _initialize_distributed(str(args.device))
    try:
        output_available = bool(args.force) or not (output_path.exists() or summary_path.exists())
        if state.enabled:
            availability = torch.tensor(
                [int(output_available) if state.rank == 0 else 0],
                dtype=torch.int64,
                device=state.device,
            )
            distributed.broadcast(availability, src=0)
            output_available = bool(int(availability.item()))
        if not output_available:
            raise FileExistsError("refusing to overwrite train full-pool RGB scores")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(173 + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - older torch
            pass

        layout_path = Path(args.rgb_spatial_layout)
        bank_path = Path(args.projected_landmark_bank)
        checkpoint_path = Path(args.checkpoint)
        paths = {
            "radio_final_context_cache": Path(args.radio_final_context_cache),
            "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
            "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        }
        for name, path in {
            **paths,
            "rgb_spatial_layout": layout_path,
            "projected_landmark_bank": bank_path,
            "checkpoint": checkpoint_path,
            "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
            "colmap_images_bin": Path(args.colmap_model_dir) / "images.bin",
        }.items():
            if not path.is_file():
                raise FileNotFoundError(f"train full-pool RGB scorer input is missing: {name} ({path})")
        if not Path(args.image_root).is_dir():
            raise FileNotFoundError("train full-pool RGB scorer image root is absent")

        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        layout_sha256 = file_sha256_short(layout_path)
        train_query_ids = set(
            np.unique(np.asarray(layout.query_ids)[np.asarray(layout.split_names) == "train"]).astype(str).tolist()
        )
        frozen_queries, frozen_lineage = load_frozen_train_full_pool_queries(
            hypothesis_artifacts=[Path(path) for path in args.hypothesis_artifact],
            evaluation_label=str(args.evaluation_label),
            expected_train_query_ids=train_query_ids,
        )

        sources = load_context_attention_sources(
            radio_final_context_cache=paths["radio_final_context_cache"],
            radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
            alike_spatial_context_cache=paths["alike_spatial_context_cache"],
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("train full-pool RGB scorer requires a common coordinate image size")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        source_metadata = sources[0].metadata
        cache_hashes = {name: file_sha256_short(path) for name, path in paths.items()}
        model, checkpoint_metadata = _load_checkpoint_model(
            path=checkpoint_path,
            layout=layout,
            source_tensors=source_tensors,
            image_sizes=image_sizes,
            cache_hashes=cache_hashes,
            source_image_manifest_sha256=str(
                source_metadata.get("source_image_manifest_sha256", "")
            ),
            device=state.device,
        )
        if int(args.edge_chunk_size) > 0:
            model.edge_chunk_size = int(args.edge_chunk_size)
        rgb_size = None
        from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
            _discover_rgb_image_size,
            validate_rgb_coordinate_bridge,
        )

        rgb_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=source_metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_size,
        )
        bank_track_ids, bank_xyz, bank_metadata = _load_bank_xyz(bank_path)
        if str(bank_metadata.get("descriptor_space_manifest", {}).get("projection_space_id", "")) != str(
            layout.metadata.get("projection_space_id", "")
        ):
            raise ValueError("train full-pool RGB scorer bank projection space differs from layout")
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        local: dict[str, list[np.ndarray]] = {
            "source_artifact_indices": [],
            "source_row_indices": [],
            "query_ids": [],
            "split_names": [],
            "evaluation_labels": [],
            "hypothesis_indices": [],
            "pose_log_likelihood_ratios": [],
        }
        all_train_query_ids = tuple(sorted(frozen_queries))
        ordered_queries = _select_scored_train_query_ids(
            all_train_query_ids,
            query_limit=int(args.query_limit),
        )
        if not ordered_queries:
            raise ValueError("train full-pool RGB scorer selected no train queries")
        complete_train_coverage = len(ordered_queries) == len(all_train_query_ids)
        for position, query_id in enumerate(ordered_queries):
            if position % state.world_size != state.rank:
                continue
            rows = np.flatnonzero(
                (np.asarray(layout.query_ids) == query_id)
                & (np.asarray(layout.split_names) == "train")
            ).astype(np.int64)
            query_layout = _slice_layout(layout, rows)
            runtime = runtime_from_target_free_layout(query_layout, image_ids=image_ids)
            query_patches, support_patches = _crop_runtime_rgb_patches(
                runtime=runtime,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_size,
                radius_px=float(model.search_radius_px + model.context_radius_px),
                step_px=float(model.step_px),
                cache=cache,
                device=state.device,
            )
            with torch.autocast(device_type=state.device.type, enabled=state.device.type == "cuda"):
                prediction = model(
                    runtime=runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=support_patches,
                )
            prediction = candidate_pose_rgb_spatial_score_component_prediction(
                prediction=prediction, component=str(args.score_component)
            )
            geometry = _query_geometry(
                query_id=query_id,
                query_layout=query_layout,
                bank_track_ids=bank_track_ids,
                bank_xyz=bank_xyz,
                colmap_model_dir=Path(args.colmap_model_dir),
            )
            frozen = frozen_queries[query_id]
            statistics = _score_hypotheses(
                model=model,
                runtime=runtime,
                prediction=prediction,
                geometry=geometry,
                poses_w2c=frozen.poses_w2c,
                point_sources=query_layout.point_sources,
                hypothesis_batch_size=int(args.hypothesis_batch_size),
                edge_chunk_size=int(model.edge_chunk_size),
                max_abs_pose_log_ratio=float(checkpoint_metadata["config"]["max_abs_pose_log_ratio"]),
            )
            count = len(frozen.source_row_indices)
            local["source_artifact_indices"].append(
                np.full((count,), frozen.source_artifact_index, dtype=np.int64)
            )
            local["source_row_indices"].append(frozen.source_row_indices)
            local["query_ids"].append(np.full((count,), query_id))
            local["split_names"].append(np.full((count,), "train"))
            local["evaluation_labels"].append(frozen.evaluation_labels)
            local["hypothesis_indices"].append(frozen.hypothesis_indices)
            local["pose_log_likelihood_ratios"].append(
                np.asarray(statistics["pose_log_likelihood_ratios"], dtype=np.float32)
            )
        empty_rows = _empty_score_rows()
        local_arrays = {
            name: np.concatenate(values, axis=0) if values else empty_rows[name]
            for name, values in local.items()
        }
        if state.enabled:
            gathered: list[dict[str, np.ndarray] | None] = [None] * state.world_size
            distributed.all_gather_object(gathered, local_arrays)
            rank_rows = [value for value in gathered if value is not None]
        else:
            rank_rows = [local_arrays]
        if state.rank == 0:
            merged = _merge_rank_rows(rank_rows)
            _validate_merged_score_rows(
                merged=merged,
                frozen_queries=frozen_queries,
                scored_query_ids=ordered_queries,
            )
            metadata = {
                "format": CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
                "contains_target_fields": False,
                "pose_or_ground_truth_used_for_scoring": False,
                "supervision_arrays_loaded": False,
                "runtime_layout_is_target_free": True,
                "projection_after_network_only": True,
                "train_rows_only": True,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "raw_scores_must_not_feed_pnp": True,
                "complete_train_coverage": bool(complete_train_coverage),
                "source_train_query_count": int(len(all_train_query_ids)),
                "scored_train_query_count": int(len(ordered_queries)),
                "query_limit": int(args.query_limit),
                "score_component": str(args.score_component),
                "hypothesis_batch_size": int(args.hypothesis_batch_size),
                "edge_chunk_size": int(model.edge_chunk_size),
                "hypothesis_artifacts": frozen_lineage["hypothesis_artifacts"],
                "hypothesis_semantic_hash": frozen_lineage["hypothesis_semantic_hash"],
                "evaluation_label": str(args.evaluation_label),
                "rgb_spatial_layout_sha256": layout_sha256,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": file_sha256_short(checkpoint_path),
                },
                "source_image_manifest_sha256": str(
                    source_metadata.get("source_image_manifest_sha256", "")
                ),
                "rgb_coordinate_bridge": rgb_bridge,
                "candidate_count": int(layout.candidate_count),
                "support_view_count": int(layout.support_view_count),
                "fixed_global_topl": True,
                "explicit_null": True,
            }
            scores = CandidatePoseRGBSpatialTrainFullPoolScores(metadata=metadata, **merged)
            save_candidate_pose_rgb_spatial_train_full_pool_scores(scores, output_path)
            summary = {
                "stage": "score_candidate_pose_rgb_spatial_train_full_pool",
                "output": str(output_path),
                "output_sha256": file_sha256_short(output_path),
                "row_count": scores.row_count,
                "train_query_count": int(len(ordered_queries)),
                "source_train_query_count": int(len(all_train_query_ids)),
                "complete_train_coverage": bool(complete_train_coverage),
                "score_quantiles": [
                    float(value)
                    for value in np.quantile(
                        scores.pose_log_likelihood_ratios, [0.0, 0.1, 0.5, 0.9, 1.0]
                    ).tolist()
                ],
                "protocol": {
                    "target_free_runtime_scoring": True,
                    "target_join_not_loaded": True,
                    "pose_matrix_not_serialized": True,
                    "validation_or_test_rows_not_serialized": True,
                    "render": False,
                    "image_retrieval_or_submap_used": False,
                },
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return {
            "output": str(output_path),
            "rank": int(state.rank),
        }
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = score_candidate_pose_rgb_spatial_train_full_pool(args)
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
