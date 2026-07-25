"""Mine inner-train hard-repeat targets from a frozen current RGB scorer.

For each training query, this tool runs the frozen target-free RGB likelihood
on its complete P1 pool, applies a fixed target-free point selector, and ranks
a fixed top-H set of coherent-wrong pose modes. Only after those decisions are
made does it join train-only exact-track/projection targets to materialize
distinct positive/negative edge pairs. Held-out inner-validation queries are
never scored or serialized into the output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _DistributedState,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _slice_runtime,
    _source_table,
    build_train_query_groups,
    current_inner_gate_evaluator_manifest,
    require_current_inner_gate_evaluator_manifest,
    train_query_partition_manifest,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood import (
    _checkpoint_validation_selector_positions,
    _load_checkpoint_model,
    _slice_layout,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
    CandidatePoseRGBSpatialHardRepeatTargets,
    save_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    runtime_from_target_free_layout,
    score_candidate_pose_rgb_spatial_batch,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    slice_target_free_edge_prediction,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_system_hard import (
    CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
    CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT,
    rank_current_system_hard_wrong_modes,
    select_current_system_hard_repeat_edges_for_modes,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument(
        "--registered-identity-targets",
        default="",
        help=(
            "Train-only registered-identity target artifact. It supplies exact positive "
            "track labels while --training-targets supplies the full coherent-wrong pose pool."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--search-radius-px",
        type=float,
        default=None,
        help="Optional assertion; when supplied it must match the frozen checkpoint.",
    )
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--positive-radius-px", type=float, default=None)
    parser.add_argument("--negative-radius-px", type=float, default=None)
    parser.add_argument(
        "--max-wrong-modes-per-query",
        type=int,
        default=1,
        help=(
            "Target-free top-H coherent-wrong pose modes to retain per train query. "
            "Modes are ranked before the train-only exact-track join."
        ),
    )
    parser.add_argument("--minimum-negative-candidate-posterior", type=float, default=0.02)
    parser.add_argument("--missing-edge-log-likelihood-ratio", type=float, default=0.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _query_id_hash(query_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(str(value) for value in query_ids):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _repeat_selected_query_ids(rows: Sequence[dict[str, object]]) -> np.ndarray:
    """Preserve full query paths when serializing variable-length target rows."""

    if not rows:
        raise ValueError("current-system hard target query rows are empty")
    try:
        lengths = [len(str(row["query_id"])) for row in rows]
        counts = [int(row["selected_hard_edge_count"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("current-system hard target query rows are invalid") from error
    if any(length <= 0 for length in lengths) or any(count <= 0 for count in counts):
        raise ValueError("current-system hard target query rows are invalid")
    dtype = f"<U{max(lengths)}"
    return np.concatenate(
        [
            np.full(count, str(row["query_id"]), dtype=dtype)
            for row, count in zip(rows, counts)
        ]
    )


def _validate_args(args: argparse.Namespace) -> None:
    values = (
        float(args.minimum_negative_candidate_posterior),
        float(args.missing_edge_log_likelihood_ratio),
        float(args.rgb_cache_gb),
    )
    if (
        int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
        or int(args.inner_validation_fold_index) >= int(args.inner_validation_fold_count)
        or int(args.max_wrong_modes_per_query) <= 0
        or not all(math.isfinite(value) for value in values)
        or not 0.0 <= float(args.minimum_negative_candidate_posterior) < 1.0
        or float(args.rgb_cache_gb) <= 0.0
    ):
        raise ValueError("current-system hard target mining arguments are invalid")


def _json_sha256(value: Mapping[str, object]) -> str:
    """Hash a JSON-normalized frozen runtime configuration."""

    try:
        payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("current-system hard checkpoint config is not serializable") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_checkpoint_partition_for_mining(
    *,
    checkpoint_metadata: Mapping[str, object],
    expected_partition: Mapping[str, object],
) -> dict[str, object]:
    """Require mining to use exactly the checkpoint's P1 inner-train fold.

    A passed checkpoint can only mine rows it actually fitted.  This rejects a
    seemingly harmless fold-index mismatch before any RGB forward or target
    join occurs, and makes the resulting artifact safe to consume solely as an
    extra inner-train loss.
    """

    if checkpoint_metadata.get("train_only_inner_gate_passed") is not True:
        raise ValueError("current-system hard checkpoint did not pass its train-only gate")
    require_current_inner_gate_evaluator_manifest(
        checkpoint_metadata.get("inner_gate_evaluator_manifest"),
        subject="current-system hard checkpoint",
    )
    training = checkpoint_metadata.get("training")
    inner_validation = training.get("inner_validation") if isinstance(training, Mapping) else None
    serialized = (
        inner_validation.get("query_partition")
        if isinstance(inner_validation, Mapping)
        else None
    )
    if not isinstance(serialized, Mapping):
        raise ValueError("current-system hard checkpoint lacks a query partition")
    try:
        reconstructed = train_query_partition_manifest(
            all_query_ids=serialized["all_train"]["query_ids"],  # type: ignore[index]
            inner_train_query_ids=serialized["inner_train"]["query_ids"],  # type: ignore[index]
            inner_validation_query_ids=serialized["inner_validation"]["query_ids"],  # type: ignore[index]
            fold_count=int(serialized["fold_count"]),
            fold_index=int(serialized["fold_index"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("current-system hard checkpoint query partition is malformed") from error
    if dict(serialized) != reconstructed:
        raise ValueError("current-system hard checkpoint query partition digest is stale")
    if dict(reconstructed) != dict(expected_partition):
        raise ValueError("current-system hard checkpoint uses a different inner validation fold")
    return reconstructed


def _registered_identity_masks_by_query(
    *,
    groups: Mapping[str, object],
    identity_targets: CandidatePoseRGBSpatialTrainingTargets,
) -> dict[str, np.ndarray]:
    """Align sparse registered track labels to full-pose query group order.

    Full-pose targets deliberately carry dense correct-pose geometry labels:
    many top-L landmarks may project locally under the correct pose.  They are
    not identity labels.  The current-system hard miner therefore accepts a
    second train-only artifact whose ``observed`` bit means the registered
    query observation's exact physical track, and joins it only after the
    target-free wrong mode has been selected.
    """

    from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
        TrainQueryGroup,
    )

    # Keep the public helper narrow while validating the structural type at
    # the boundary.  This avoids leaking the identity target into runtime code.
    if not isinstance(identity_targets, CandidatePoseRGBSpatialTrainingTargets):
        raise ValueError("registered identity targets are invalid")
    metadata = identity_targets.metadata
    if (
        str(metadata.get("format", ""))
        != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        or str(metadata.get("spatial_supervision_mode", ""))
        != "registered_exact_identity"
        or str(metadata.get("spatial_target_semantics", ""))
        != "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
    ):
        raise ValueError("registered identity targets do not carry exact-track labels")
    target_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(
            np.asarray(identity_targets.source_point_ids, dtype=np.int64).tolist()
        )
    }
    target_query_ids = np.asarray(identity_targets.query_ids).astype(str)
    observed = np.asarray(identity_targets.spatial_target_observed, dtype=bool)
    output: dict[str, np.ndarray] = {}
    for query_id, raw_group in groups.items():
        if not isinstance(raw_group, TrainQueryGroup):
            raise ValueError("registered identity query group is invalid")
        try:
            target_rows = np.asarray(
                [target_row_by_source[int(source_id)] for source_id in raw_group.source_point_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "registered identity targets do not cover the full-pose query group"
            ) from error
        if (
            np.any(target_query_ids[target_rows] != str(query_id))
            or observed.shape[1] != raw_group.correct_projection_offsets_xy.shape[1]
        ):
            raise ValueError("registered identity target/query candidate alignment differs")
        mask = np.asarray(observed[target_rows], dtype=bool)
        if np.any(mask.sum(axis=1) > 1):
            raise ValueError("registered identity targets assign multiple positive tracks")
        output[str(query_id)] = mask
    if not output or not any(bool(mask.any()) for mask in output.values()):
        raise ValueError("registered identity targets contain no positive P1 track")
    return output


def _output_conflict(*, state: _DistributedState, output: Path, summary: Path) -> bool:
    del state
    return output.exists() or summary.exists()


@torch.no_grad()
def mine_candidate_pose_rgb_spatial_system_hard_targets(args: argparse.Namespace) -> dict[str, object]:
    """Build a train-only hard-repeat artifact from frozen current failures."""

    _validate_args(args)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    state = _initialize_distributed(str(args.device))
    try:
        if not bool(args.force) and _output_conflict(
            state=state, output=output_path, summary=summary_path
        ):
            raise FileExistsError("refusing to overwrite current-system hard target output")
        if state.enabled:
            distributed.barrier()
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - compatibility
            pass

        layout_path = Path(args.rgb_spatial_layout)
        targets_path = Path(args.training_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        layout_sha256 = file_sha256_short(layout_path)
        targets_sha256 = file_sha256_short(targets_path)
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=layout_sha256
        )
        identity_targets_path = (
            Path(str(args.registered_identity_targets))
            if str(args.registered_identity_targets).strip()
            else targets_path
        )
        identity_targets = (
            targets
            if identity_targets_path == targets_path
            else load_candidate_pose_rgb_spatial_training_targets(identity_targets_path)
        )
        validate_training_layout_and_targets(
            layout=layout,
            targets=identity_targets,
            layout_sha256=layout_sha256,
        )
        identity_targets_sha256 = file_sha256_short(identity_targets_path)
        target_radius = float(targets.metadata["spatial_search_radius_px"])
        search_radius = target_radius if args.search_radius_px is None else float(args.search_radius_px)
        if not math.isclose(search_radius, target_radius, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("current-system hard miner search radius differs from targets")
        positive_radius = search_radius if args.positive_radius_px is None else float(args.positive_radius_px)
        negative_radius = search_radius if args.negative_radius_px is None else float(args.negative_radius_px)
        if (
            not math.isfinite(positive_radius)
            or not math.isfinite(negative_radius)
            or positive_radius <= 0.0
            or negative_radius <= 0.0
            or positive_radius > search_radius
            or negative_radius > search_radius
        ):
            raise ValueError("current-system hard miner local radii are invalid")
        groups = build_train_query_groups(layout=layout, targets=targets)
        identity_masks_by_query = _registered_identity_masks_by_query(
            groups=groups,
            identity_targets=identity_targets,
        )
        inner_train, inner_validation = _partition_train_queries_for_inner_validation(
            query_ids=tuple(sorted(groups)),
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        expected_partition = train_query_partition_manifest(
            all_query_ids=tuple(sorted(groups)),
            inner_train_query_ids=inner_train,
            inner_validation_query_ids=inner_validation,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("current-system hard miner requires common context image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        source_metadata = sources[0].metadata
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=source_metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        cache_hashes = {
            "radio_final_context_cache": file_sha256_short(Path(args.radio_final_context_cache)),
            "radio_intermediate_context_cache": file_sha256_short(
                Path(args.radio_intermediate_context_cache)
            ),
            "alike_spatial_context_cache": file_sha256_short(
                Path(args.alike_spatial_context_cache)
            ),
        }
        model, checkpoint_metadata = _load_checkpoint_model(
            path=Path(args.checkpoint),
            layout=layout,
            source_tensors=source_tensors,
            image_sizes=image_sizes,
            cache_hashes=cache_hashes,
            source_image_manifest_sha256=str(
                source_metadata.get("source_image_manifest_sha256", "")
            ),
            device=state.device,
        )
        checkpoint_config = checkpoint_metadata.get("config")
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("current-system hard checkpoint lacks runtime configuration")
        checkpoint_config = dict(checkpoint_config)
        checkpoint_partition = _validate_checkpoint_partition_for_mining(
            checkpoint_metadata=checkpoint_metadata,
            expected_partition=expected_partition,
        )
        search_radius = float(checkpoint_config["search_radius_px"])
        context_radius = float(checkpoint_config["context_radius_px"])
        step_px = float(checkpoint_config["step_px"])
        max_abs_pose_log_ratio = float(checkpoint_config["max_abs_pose_log_ratio"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (search_radius, context_radius, step_px, max_abs_pose_log_ratio)
        ):
            raise ValueError("current-system hard checkpoint geometry is invalid")
        if not math.isclose(search_radius, target_radius, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("current-system hard checkpoint search radius differs from targets")
        if args.search_radius_px is not None and not math.isclose(
            float(args.search_radius_px), search_radius, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("current-system hard miner search radius differs from checkpoint")
        model_for_mining: torch.nn.Module
        if state.enabled:
            model_for_mining = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_mining = model
        model_for_mining.eval()
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        local_rows: list[dict[str, object]] = []
        for query_position, query_id in enumerate(inner_train):
            if query_position % state.world_size != state.rank:
                continue
            group = groups[str(query_id)]
            query_layout = _slice_layout(layout, group.layout_rows)
            positions, point_selection = _checkpoint_validation_selector_positions(
                layout=query_layout,
                config=checkpoint_config,
                coordinate_image_size=coordinate_image_size,
            )
            full_runtime = _slice_runtime(complete_runtime, group.layout_rows)
            query_patches, support_patches = _crop_runtime_rgb_patches(
                runtime=full_runtime,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=search_radius + context_radius,
                step_px=step_px,
                cache=cache,
                device=state.device,
            )
            with torch.cuda.amp.autocast(enabled=state.device.type == "cuda" and not bool(args.no_amp)):
                prediction = model_for_mining(
                    runtime=full_runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=support_patches,
                    rgb_cost_volume_only=False,
                )
            selected_runtime = _slice_runtime(full_runtime, positions)
            selected_prediction = slice_target_free_edge_prediction(
                prediction=prediction, positions=positions
            )
            correct_score = score_candidate_pose_rgb_spatial_batch(
                runtime=selected_runtime,
                prediction=selected_prediction,
                candidate_projection_offsets_xy=torch.from_numpy(
                    group.correct_projection_offsets_xy[positions]
                ).to(device=state.device).unsqueeze(0),
                candidate_projection_valid=torch.from_numpy(
                    group.correct_projection_valid[positions]
                ).to(device=state.device).unsqueeze(0),
                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=max_abs_pose_log_ratio,
            )
            wrong_score = score_candidate_pose_rgb_spatial_batch(
                runtime=selected_runtime,
                prediction=selected_prediction,
                candidate_projection_offsets_xy=torch.from_numpy(
                    group.wrong_projection_offsets_xy[:, positions]
                ).to(device=state.device),
                candidate_projection_valid=torch.from_numpy(
                    group.wrong_projection_valid[:, positions]
                ).to(device=state.device),
                missing_edge_log_likelihood_ratio=float(args.missing_edge_log_likelihood_ratio),
                max_abs_log_likelihood_ratio=max_abs_pose_log_ratio,
            )
            wrong_pose_scores = (
                wrong_score.pose_log_likelihood_ratios.detach().cpu().numpy()
            )
            # This ranking sees only frozen runtime scores.  Do it before the
            # train-only identity/projection join, and preserve every selected
            # rank even when a mode later has no eligible edge.
            ranked_modes = rank_current_system_hard_wrong_modes(
                pair_ids=group.wrong_pair_ids,
                wrong_pose_log_likelihood_ratios=wrong_pose_scores,
                max_wrong_modes_per_query=int(args.max_wrong_modes_per_query),
            )
            # The scorer/selector above are target-free. The following call is
            # the sole train-only join that turns the fixed top-H wrong modes
            # into registered identity targets.
            selections = select_current_system_hard_repeat_edges_for_modes(
                source_point_ids=group.source_point_ids[positions],
                pair_ids=group.wrong_pair_ids,
                wrong_pose_log_likelihood_ratios=wrong_pose_scores,
                candidate_track_ids=np.asarray(
                    layout.candidate_track_ids[group.layout_rows[positions]], dtype=np.int64
                ),
                candidate_probabilities=selected_runtime.candidate_probabilities.detach().cpu().numpy(),
                null_probabilities=selected_runtime.null_probabilities.detach().cpu().numpy(),
                observed_candidate_mask=identity_masks_by_query[str(query_id)][positions],
                correct_offsets_xy=group.correct_projection_offsets_xy[positions],
                correct_valid=group.correct_projection_valid[positions],
                correct_edge_usable=correct_score.edge_usable[0].detach().cpu().numpy(),
                wrong_offsets_xy=group.wrong_projection_offsets_xy[:, positions],
                wrong_valid=group.wrong_projection_valid[:, positions],
                wrong_candidate_log_likelihood_ratios=wrong_score.candidate_log_likelihood_ratios
                .detach()
                .cpu()
                .numpy(),
                wrong_edge_usable=wrong_score.edge_usable.detach().cpu().numpy(),
                positive_radius_px=positive_radius,
                negative_radius_px=negative_radius,
                minimum_negative_candidate_posterior=float(args.minimum_negative_candidate_posterior),
                max_wrong_modes_per_query=int(args.max_wrong_modes_per_query),
            )
            if len(selections) != len(ranked_modes):
                raise RuntimeError("current-system hard miner lost a selected wrong mode")
            materialized_selections = [selection for selection in selections if selection is not None]
            selection_by_mode = {
                int(selection.hardest_mode_index): selection
                for selection in materialized_selections
            }
            if len(selection_by_mode) != len(materialized_selections):
                raise RuntimeError("current-system hard miner selected a wrong mode twice")
            row: dict[str, object] = {
                "query_id": str(query_id),
                "selection": dict(point_selection),
                "selected_target_free_wrong_mode_count": int(len(ranked_modes)),
                "materialized_wrong_mode_count": int(len(materialized_selections)),
                "selected_hard_edge_count": int(
                    sum(int(selection.count) for selection in materialized_selections)
                ),
                "ranked_wrong_modes": [
                    {
                        "mode_rank": int(mode_rank),
                        "mode_index": int(mode_index),
                        "pair_id": int(group.wrong_pair_ids[int(mode_index)]),
                        "wrong_pose_log_likelihood_ratio": float(
                            wrong_pose_scores[int(mode_index)]
                        ),
                        "materialized_hard_edge_count": int(
                            selection_by_mode[int(mode_index)].count
                            if int(mode_index) in selection_by_mode
                            else 0
                        ),
                    }
                    for mode_rank, mode_index in enumerate(ranked_modes.tolist())
                ],
                "mode_selections": materialized_selections,
            }
            local_rows.append(row)
        if state.enabled:
            gathered: list[list[dict[str, object]] | None] = [None] * state.world_size
            distributed.all_gather_object(gathered, local_rows)
            rows = [row for part in gathered if part is not None for row in part]
        else:
            rows = local_rows
        rows = sorted(rows, key=lambda row: str(row["query_id"]))
        if len(rows) != len(inner_train):
            raise RuntimeError("current-system hard miner did not cover every inner-train query")
        nonempty = [row for row in rows if int(row["selected_hard_edge_count"]) > 0]
        if not nonempty:
            raise ValueError("current-system hard miner found no eligible exact-track hard edges")
        if state.rank == 0:
            materialized = [
                selection
                for row in nonempty
                for selection in row["mode_selections"]  # type: ignore[index]
            ]
            if not materialized:
                raise RuntimeError("current-system hard miner lost all materialized selections")
            output_query_ids = _repeat_selected_query_ids(nonempty)
            ranked_modes_by_query = {
                str(row["query_id"]): [
                    {
                        "mode_rank": int(mode["mode_rank"]),
                        "mode_index": int(mode["mode_index"]),
                        "pair_id": int(mode["pair_id"]),
                    }
                    for mode in row["ranked_wrong_modes"]  # type: ignore[index]
                ]
                for row in rows
            }
            output_pairs = np.concatenate(
                [
                    np.full(
                        int(selection.count), int(selection.hardest_pair_id), dtype=np.int64
                    )
                    for selection in materialized
                ]
            )
            output_sources = np.concatenate(
                [np.asarray(selection.source_point_ids, dtype=np.int64) for selection in materialized]
            )
            if set(output_query_ids.tolist()) - set(inner_train):
                raise RuntimeError("current-system hard target includes a held-out query")
            artifact = CandidatePoseRGBSpatialHardRepeatTargets(
                source_point_ids=output_sources,
                query_ids=output_query_ids,
                pair_ids=output_pairs,
                positive_candidate_indices=np.concatenate(
                    [
                        np.asarray(selection.positive_candidate_indices, dtype=np.int64)
                        for selection in materialized
                    ]
                ),
                negative_candidate_indices=np.concatenate(
                    [
                        np.asarray(selection.negative_candidate_indices, dtype=np.int64)
                        for selection in materialized
                    ]
                ),
                positive_offsets_xy=np.concatenate(
                    [
                        np.asarray(selection.positive_offsets_xy, dtype=np.float32)
                        for selection in materialized
                    ]
                ),
                negative_offsets_xy=np.concatenate(
                    [
                        np.asarray(selection.negative_offsets_xy, dtype=np.float32)
                        for selection in materialized
                    ]
                ),
                metadata={
                    "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
                    "training_only_target_artifact": True,
                    "contains_ground_truth": True,
                    "contains_validation_or_test_targets": False,
                    "runtime_layout_is_target_free": True,
                    "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
                    "render": False,
                    "image_retrieval_or_submap_used": False,
                    "rgb_spatial_layout_sha256": layout_sha256,
                    "rgb_spatial_targets_sha256": targets_sha256,
                    "registered_identity_targets_sha256": identity_targets_sha256,
                    "projection_space_id": str(layout.metadata["projection_space_id"]),
                    "descriptor_space_id": str(layout.metadata["descriptor_space_id"]),
                    "candidate_count": int(layout.candidate_count),
                    "positive_radius_px": float(positive_radius),
                    "negative_radius_px": float(negative_radius),
                    "selection": (
                        "gate_approved_current_full_rgb_radio_alike_static_selector_"
                        "target_free_ranked_top_h_coherent_wrong_modes_then_"
                        "registered_exact_track_vs_distinct_wrong_candidate_posterior_v3"
                    ),
                    "positive_requires_registered_exact_track": True,
                    "positive_identity_targets": {
                        "path": str(identity_targets_path),
                        "sha256": identity_targets_sha256,
                        "semantics": str(
                            identity_targets.metadata.get("spatial_target_semantics", "")
                        ),
                    },
                    "negative_requires_distinct_track_and_wrong_mode_posterior": True,
                    "mining_format": (
                        CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT
                        if int(args.max_wrong_modes_per_query) > 1
                        else CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMAT
                    ),
                    "wrong_mode_selection": {
                        "policy": (
                            "target_free_combined_pose_llr_descending_pair_id_tiebreak_top_h_v1"
                        ),
                        "max_wrong_modes_per_query": int(args.max_wrong_modes_per_query),
                        "target_join_after_mode_ranking": True,
                        "label_based_mode_backfill": False,
                    },
                    "target_free_ranked_wrong_modes": {
                        "format": CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT,
                        "query_count": int(len(ranked_modes_by_query)),
                        "ranked_modes_by_query": ranked_modes_by_query,
                        "selection_before_train_only_target_join": True,
                    },
                    "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                    "frozen_checkpoint": {
                        "path": str(Path(args.checkpoint)),
                        "sha256": file_sha256_short(Path(args.checkpoint)),
                        "train_only_inner_gate_passed": True,
                        "runtime_config_sha256": _json_sha256(checkpoint_config),
                        "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                    },
                    "mining_checkpoint_sha256": file_sha256_short(Path(args.checkpoint)),
                    "mining_checkpoint_config": checkpoint_config,
                    "mining_checkpoint_config_sha256": _json_sha256(checkpoint_config),
                    "checkpoint_sha256": file_sha256_short(Path(args.checkpoint)),
                    "inner_validation_fold_count": int(args.inner_validation_fold_count),
                    "inner_validation_fold_index": int(args.inner_validation_fold_index),
                    "inner_train_query_ids_sha256": _query_id_hash(inner_train),
                    "inner_validation_query_ids_sha256": _query_id_hash(inner_validation),
                    "train_query_partition": checkpoint_partition,
                    "target_free_selector": {
                        "mode": "checkpoint_validation_static_target_free_selector_v1",
                        "config": dict(checkpoint_config["validation_selector"]),
                    },
                    "minimum_negative_candidate_posterior": float(
                        args.minimum_negative_candidate_posterior
                    ),
                    "score_component": "combined",
                    "inputs": {
                        "rgb_spatial_layout": str(layout_path),
                        "training_targets": str(targets_path),
                        "registered_identity_targets": str(identity_targets_path),
                        "checkpoint": str(Path(args.checkpoint)),
                    },
                },
            )
            save_candidate_pose_rgb_spatial_hard_repeat_targets(artifact, output_path)
            posterior_values = np.concatenate(
                [
                    np.asarray(selection.negative_candidate_posteriors, dtype=np.float32)
                    for selection in materialized
                ]
            )
            wrong_scores = np.asarray(
                [
                    float(selection.wrong_pose_log_likelihood_ratio)
                    for selection in materialized
                ],
                dtype=np.float32,
            )
            summary = {
                "stage": "mine_candidate_pose_rgb_spatial_system_hard_targets",
                "output": str(output_path),
                "output_sha256": file_sha256_short(output_path),
                "target_count": int(artifact.count),
                "inner_train_query_count": int(len(inner_train)),
                "inner_validation_query_count": int(len(inner_validation)),
                "nonempty_query_count": int(len(nonempty)),
                "max_wrong_modes_per_query": int(args.max_wrong_modes_per_query),
                "target_free_wrong_mode_count_per_query": {
                    "min": int(
                        min(int(row["selected_target_free_wrong_mode_count"]) for row in rows)
                    ),
                    "median": float(
                        np.median(
                            [int(row["selected_target_free_wrong_mode_count"]) for row in rows]
                        )
                    ),
                    "max": int(
                        max(int(row["selected_target_free_wrong_mode_count"]) for row in rows)
                    ),
                },
                "materialized_wrong_mode_count_per_query": {
                    "min": int(min(int(row["materialized_wrong_mode_count"]) for row in rows)),
                    "median": float(
                        np.median([int(row["materialized_wrong_mode_count"]) for row in rows])
                    ),
                    "max": int(max(int(row["materialized_wrong_mode_count"]) for row in rows)),
                },
                "hard_edge_count_per_nonempty_query": {
                    "min": int(min(int(row["selected_hard_edge_count"]) for row in nonempty)),
                    "median": float(
                        np.median([int(row["selected_hard_edge_count"]) for row in nonempty])
                    ),
                    "max": int(max(int(row["selected_hard_edge_count"]) for row in nonempty)),
                },
                "negative_candidate_posterior_quantiles": np.quantile(
                    posterior_values, [0.0, 0.1, 0.5, 0.9, 1.0]
                ).tolist(),
                "selected_wrong_pose_score_quantiles": np.quantile(
                    wrong_scores, [0.0, 0.1, 0.5, 0.9, 1.0]
                ).tolist(),
                "protocol": {
                    "frozen_checkpoint": True,
                    "model_weights_updated": False,
                    "scored_queries_are_inner_train_only": True,
                    "inner_validation_queries_serialized": False,
                    "target_free_selector_runs_before_target_join": True,
                    "target_free_top_h_wrong_modes_ranked_before_target_join": True,
                    "label_based_wrong_mode_backfill": False,
                    "runtime_scorer_must_not_load_output": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                },
                "per_query": [
                    {
                        "query_id": str(row["query_id"]),
                        "selection": dict(row["selection"]),
                        "selected_target_free_wrong_mode_count": int(
                            row["selected_target_free_wrong_mode_count"]
                        ),
                        "materialized_wrong_mode_count": int(
                            row["materialized_wrong_mode_count"]
                        ),
                        "selected_hard_edge_count": int(row["selected_hard_edge_count"]),
                        "ranked_wrong_modes": list(row["ranked_wrong_modes"]),
                    }
                    for row in rows
                ],
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return {"output": str(output_path), "rank": int(state.rank)}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = mine_candidate_pose_rgb_spatial_system_hard_targets(args)
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
