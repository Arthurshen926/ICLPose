"""Freeze a target-free RGB-peakiness selector into a real P1 subset layout.

The source layout remains the ordinary detector-anchor P1 layout.  A frozen
candidate-specific RGB likelihood scores every point before any train target is
opened, then a spatially diverse target-free selector retains a fixed budget
per query.  The resulting layout contains no pose, residual, identity, or
target fields and can therefore be used by both train-time and runtime paths.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_candidate_pose_rgb_spatial_hard_pose_pretrain_p1 import (
    _load_cross_layout_curriculum_checkpoint,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _slice_runtime,
    _source_table,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
    save_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialLikelihood,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    TARGET_FREE_SELECTOR_POLICIES,
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    summarize_target_free_point_selection,
    target_free_rgb_point_quality,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


_RGB_POLICIES = frozenset(
    policy for policy in TARGET_FREE_SELECTOR_POLICIES if "rgb" in str(policy)
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cross-layout-component-audit", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--selector-policy",
        choices=tuple(sorted(_RGB_POLICIES)),
        default="rgb_peakiness",
    )
    parser.add_argument("--points-per-query", type=int, default=32)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--search-radius-px", type=float, default=8.0)
    parser.add_argument("--context-radius-px", type=float, default=12.0)
    parser.add_argument("--step-px", type=float, default=0.5)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-context-log-ratio", type=float, default=3.0)
    parser.add_argument("--edge-chunk-size", type=int, default=64)
    parser.add_argument("--radio-final-context-window", type=int, default=15)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=15)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument("--context-encoder-arch", default="absolute_cross_attention_v3")
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _context_windows(args: argparse.Namespace) -> dict[str, int]:
    windows = {
        "radio_final": int(args.radio_final_context_window),
        "radio_intermediate": int(args.radio_intermediate_context_window),
        "alike": int(args.alike_context_window),
    }
    if any(value < 3 or value % 2 != 1 for value in windows.values()):
        raise ValueError("frozen RGB selector context windows are invalid")
    return windows


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.search_radius_px),
        float(args.context_radius_px),
        float(args.step_px),
        float(args.max_abs_context_log_ratio),
        float(args.rgb_cache_gb),
    )
    if (
        str(args.selector_policy) not in _RGB_POLICIES
        or int(args.points_per_query) < 4
        or int(args.grid_rows) <= 0
        or int(args.grid_columns) <= 0
        or int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or not all(np.isfinite(value) and value > 0.0 for value in values)
    ):
        raise ValueError("frozen RGB selector builder arguments are invalid")
    return _context_windows(args)


def _subset_layout(
    *, layout: CandidatePoseRGBSpatialLayout, rows: np.ndarray, metadata: Mapping[str, object]
) -> CandidatePoseRGBSpatialLayout:
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    if (
        len(selected) == 0
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= layout.row_count)
    ):
        raise ValueError("frozen RGB selector layout rows are invalid")
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=layout.source_point_ids[selected],
        query_ids=layout.query_ids[selected],
        split_names=layout.split_names[selected],
        xy=layout.xy[selected],
        point_sources=layout.point_sources[selected],
        candidate_track_ids=layout.candidate_track_ids[selected],
        candidate_bank_rows=layout.candidate_bank_rows[selected],
        candidate_coarse_similarities=layout.candidate_coarse_similarities[selected],
        candidate_prior_probabilities=layout.candidate_prior_probabilities[selected],
        null_probabilities=layout.null_probabilities[selected],
        support_image_ids=layout.support_image_ids[selected],
        support_xy=layout.support_xy[selected],
        support_view_valid=layout.support_view_valid[selected],
        support_view_weights=layout.support_view_weights[selected],
        support_coverage_counts=layout.support_coverage_counts[selected],
        metadata=dict(metadata),
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def build_frozen_rgb_peakiness_p1_layout(args: argparse.Namespace) -> dict[str, object]:
    """Score each frozen P1 point once and save a target-free selected layout."""

    windows = _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_path = Path(args.output)
        summary_path = Path(args.summary_json)
        if state.rank == 0 and (output_path.exists() or summary_path.exists()) and not bool(args.force):
            raise FileExistsError("refusing to overwrite frozen RGB selector output")
        if state.enabled:
            distributed.barrier()
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - torch compatibility
            pass

        layout_path = Path(args.rgb_spatial_layout)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        if layout.metadata.get("training_only_anchor_layout") is True:
            raise ValueError("frozen RGB selector must start from a real detector-anchor P1 layout")
        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        if np.unique(image_sizes, axis=0).shape != (1, 2):
            raise ValueError("frozen RGB selector requires common image dimensions")
        coordinate_image_size = (int(image_sizes[0, 0]), int(image_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        source_metadata = sources[0].metadata
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=source_metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        cache_paths = {
            "radio_final": Path(args.radio_final_context_cache),
            "radio_intermediate": Path(args.radio_intermediate_context_cache),
            "alike": Path(args.alike_spatial_context_cache),
        }
        model = CandidatePoseRGBSpatialLikelihood(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            search_radius_px=float(args.search_radius_px),
            context_radius_px=float(args.context_radius_px),
            step_px=float(args.step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=False,
            context_windows=windows,
            context_encoder_arch=str(args.context_encoder_arch),
        )
        initialization = _load_cross_layout_curriculum_checkpoint(
            path=Path(args.checkpoint),
            model=model,
            destination_layout=layout,
            source_cache_paths=cache_paths,
            source_image_manifest_sha256=str(
                source_metadata.get("source_image_manifest_sha256", "")
            ),
            rgb_coordinate_bridge=rgb_bridge,
            search_radius_px=float(args.search_radius_px),
            context_radius_px=float(args.context_radius_px),
            step_px=float(args.step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
            context_windows=windows,
            context_encoder_arch=str(args.context_encoder_arch),
            component_audit_path=Path(args.cross_layout_component_audit),
        )
        model = model.to(state.device).eval()
        if state.enabled:
            model_for_scoring: torch.nn.Module = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_scoring = model
        runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        rows_by_query: dict[str, np.ndarray] = {}
        for query_id in np.unique(layout.query_ids).tolist():
            rows_by_query[str(query_id)] = np.flatnonzero(layout.query_ids == str(query_id))
        local_records: list[dict[str, object]] = []
        with torch.no_grad():
            for query_position, query_id in enumerate(sorted(rows_by_query)):
                if query_position % state.world_size != state.rank:
                    continue
                rows = rows_by_query[query_id]
                selector_input = selector_input_from_target_free_layout(
                    layout=layout,
                    rows=rows,
                    rgb_context_radius_px=float(args.search_radius_px + args.context_radius_px),
                    coordinate_image_size=coordinate_image_size,
                )
                if int(args.points_per_query) > selector_input.point_count:
                    raise ValueError("frozen RGB selector budget exceeds a query point pool")
                local_runtime = _slice_runtime(runtime, rows).to(state.device)
                query_patches, support_patches = _crop_runtime_rgb_patches(
                    runtime=local_runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=float(args.search_radius_px + args.context_radius_px),
                    step_px=float(args.step_px),
                    cache=cache,
                    device=state.device,
                )
                with torch.cuda.amp.autocast(
                    enabled=state.device.type == "cuda" and not bool(args.no_amp)
                ):
                    prediction = model_for_scoring(
                        runtime=local_runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                    )
                rgb_quality = target_free_rgb_point_quality(
                    runtime=local_runtime, prediction=prediction
                )
                quality = target_free_selector_scores(
                    selector_input=selector_input,
                    policy=str(args.selector_policy),
                    rgb_quality=rgb_quality,
                )
                selected_positions = select_target_free_spatial_quota(
                    selector_input=selector_input,
                    quality_scores=quality,
                    point_budget=int(args.points_per_query),
                    grid_rows=int(args.grid_rows),
                    grid_columns=int(args.grid_columns),
                    image_size=coordinate_image_size,
                )
                selected_rows = rows[selected_positions]
                local_records.append(
                    {
                        "query_id": str(query_id),
                        "selected_rows": selected_rows.astype(np.int64).tolist(),
                        "selected_source_point_ids": layout.source_point_ids[selected_rows]
                        .astype(np.int64)
                        .tolist(),
                        "selection": summarize_target_free_point_selection(
                            selector_input=selector_input,
                            selected_positions=selected_positions,
                            quality_scores=quality,
                            grid_rows=int(args.grid_rows),
                            grid_columns=int(args.grid_columns),
                            image_size=coordinate_image_size,
                        ),
                    }
                )
        records_by_rank: list[object] = [None] * state.world_size
        if state.enabled:
            distributed.all_gather_object(records_by_rank, local_records)
        else:
            records_by_rank[0] = local_records
        if state.rank != 0:
            return {"rank": int(state.rank)}
        records = [
            record
            for rank_records in records_by_rank
            for record in (rank_records if isinstance(rank_records, list) else [])
        ]
        if len(records) != len(rows_by_query):
            raise RuntimeError("frozen RGB selector did not score every query exactly once")
        records.sort(key=lambda value: str(value["query_id"]))
        selected_rows = np.asarray(
            [row for record in records for row in record["selected_rows"]], dtype=np.int64
        )
        if len(np.unique(selected_rows)) != len(selected_rows):
            raise RuntimeError("frozen RGB selector emitted duplicate layout rows")
        selected_rows.sort()
        split_counts = Counter(layout.split_names[selected_rows].tolist())
        per_query_counts = Counter(layout.query_ids[selected_rows].tolist())
        if set(per_query_counts) != set(rows_by_query) or set(per_query_counts.values()) != {
            int(args.points_per_query)
        }:
            raise RuntimeError("frozen RGB selector point budgets differ by query")
        selection_metadata = {
            "format": "frozen_rgb_peakiness_p1_subset_layout_v1",
            "source_layout_sha256": file_sha256_short(layout_path),
            "source_checkpoint_sha256": str(initialization["sha256"]),
            "source_component_audit_sha256": str(initialization["component_audit"]["sha256"]),
            "selector_policy": str(args.selector_policy),
            "points_per_query": int(args.points_per_query),
            "grid_rows": int(args.grid_rows),
            "grid_columns": int(args.grid_columns),
            "selection_before_train_target_join": True,
            "selection_inputs": [
                "frozen_query_anchor_xy",
                "fixed_global_topl_candidate_prior",
                "fixed_support_observation_xy",
                "real_rgb_query_and_support_patches",
            ],
            "selection_excludes": [
                "pose_matrix",
                "projection_offset",
                "reprojection_residual",
                "ground_truth_label",
                "track_id",
                "candidate_rank",
                "coarse_score",
            ],
            "runtime_layout_target_free": True,
        }
        output_metadata = {**dict(layout.metadata), "frozen_rgb_selector": selection_metadata}
        subset = _subset_layout(layout=layout, rows=selected_rows, metadata=output_metadata)
        save_candidate_pose_rgb_spatial_layout(subset, output_path)
        summary = {
            "stage": "build_frozen_rgb_peakiness_p1_layout",
            "layout": str(output_path),
            "layout_sha256": file_sha256_short(output_path),
            "source_layout": str(layout_path),
            "source_layout_sha256": file_sha256_short(layout_path),
            "checkpoint": initialization,
            "selector": selection_metadata,
            "source_row_count": int(layout.row_count),
            "selected_row_count": int(subset.row_count),
            "split_counts": dict(sorted(split_counts.items())),
            "per_query_selected_count": int(args.points_per_query),
            "query_count": int(len(rows_by_query)),
            "per_query": records,
            "rgb_cache_rank0": cache.summary(),
            "protocol": {
                "target_free_selection_before_train_target_join": True,
                "no_render": True,
                "no_image_retrieval_or_submap": True,
                "pose_or_ground_truth_not_available_to_selector": True,
                "source_component_audit_required": True,
            },
        }
        _write_json(summary_path, summary)
        return {"layout": str(output_path), "summary": str(summary_path), "rank": 0}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_frozen_rgb_peakiness_p1_layout(args)
    if int(result.get("rank", 0)) == 0:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
