"""Audit static target-free token selection for the phase/RGB hybrid.

This is a train-only diagnostic.  It first runs the frozen phase/RGB visual
forwards on a query and produces static token weights from the observed query
appearance only.  Only afterwards does it join correct/coherent-wrong pose
targets to compare uniform aggregation with spatial-diverse phase, RGB, and
joint token selectors.

The audit cannot promote a checkpoint to PnP.  Its sole question is whether
the already validated local RGB evidence is being diluted by uniform token
averaging under the same fixed global top-L and explicit-null contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_candidate_multiscale_phase_identity_llr import (
    _checkpoint_partition,
    _load_checkpoint as _load_phase_checkpoint,
    _load_model as _load_phase_model,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    _source_table,
)
from feature_extract.tools.vfm.train_candidate_phase_identity_spatial_likelihood import (
    CHECKPOINT_FORMAT,
    _hybrid_pose_scores,
    _phase_control_common_availability,
)
from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    pose_margin_terms,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _query_batch_from_group,
    _slice_runtime,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
    CandidateHighresRGBMultiscaleLikelihood,
)
from feature_extract.vfm.localization.candidate_phase_identity_spatial_likelihood import (
    permute_support_patches_with_phase_identity_point_blocks,
)
from feature_extract.vfm.localization.candidate_pose_evidence_selector import (
    CANDIDATE_POSE_EVIDENCE_SELECTOR_FORMAT,
    aggregate_static_point_log_likelihood_ratios,
    blend_selector_with_uniform_mass,
    combine_selector_confidences,
    phase_identity_confidence,
    rgb_spatial_mode_quality,
    runtime_visual_edge_availability,
    score_reweighted_selector_weights,
    spatial_diverse_topk_weights,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


AUDIT_FORMAT = "candidate_phase_identity_spatial_static_selector_audit_v2"


def parse_selector_topk(value: str) -> tuple[int, ...]:
    """Parse a unique positive top-K audit list without implicit defaults."""

    try:
        values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("selector top-K values must be integers") from error
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise ValueError("selector top-K values must be unique positive integers")
    return values


def parse_uniform_mixture_fractions(value: str) -> tuple[float, ...]:
    """Parse strictly interior, unique uniform-fallback masses."""

    text = str(value).strip()
    if not text:
        return ()
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("uniform mixture fractions must be floats") from error
    if (
        not values
        or any(not math.isfinite(item) or item <= 0.0 or item >= 1.0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("uniform mixture fractions must be unique values in (0, 1)")
    return values


def _uniform_mixture_name(value: float) -> str:
    """Stable artifact key for a predeclared uniform fallback mass."""

    fraction = float(value)
    scaled = int(round(fraction * 1000.0))
    if not 0 < scaled < 1000 or abs(fraction - scaled / 1000.0) > 1e-8:
        raise ValueError("uniform mixture fraction cannot be represented in selector output")
    return f"u{scaled:03d}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selector-top-k", default="8,16,32,64")
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument(
        "--score-weight-floor",
        type=float,
        default=0.10,
        help="Positive floor for target-free RGB-quality score reweighting.",
    )
    parser.add_argument(
        "--score-weight-power",
        type=float,
        default=1.0,
        help="Exponent for target-free RGB-quality score reweighting.",
    )
    parser.add_argument(
        "--uniform-mixture-top-k",
        type=int,
        default=12,
        help="Predeclared RGB top-K whose score weights receive a uniform fallback.",
    )
    parser.add_argument(
        "--uniform-mixture-fractions",
        default="0.25,0.50,0.75",
        help="Comma-separated all-token mass for the RGB top-K score mixture; empty disables it.",
    )
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _read_hybrid_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != CHECKPOINT_FORMAT
        or not isinstance(payload.get("state_dict"), Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
    ):
        raise ValueError("hybrid selector audit checkpoint is invalid")
    metadata = dict(payload["metadata"])
    if (
        metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("contains_target_fields") is not False
        or metadata.get("fixed_global_topl") is not True
        or metadata.get("explicit_null") is not True
        or metadata.get("projection_after_network_only") is not True
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
    ):
        raise ValueError("hybrid selector audit checkpoint violates the runtime protocol")
    return (
        {str(name): torch.as_tensor(value).clone() for name, value in payload["state_dict"].items()},
        metadata,
    )


def _load_rgb_model(
    *,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
    image_sizes: np.ndarray,
    device: torch.device,
) -> CandidateHighresRGBMultiscaleLikelihood:
    if metadata.get("model_format") != CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT:
        raise ValueError("hybrid selector audit RGB model format is invalid")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("hybrid selector audit RGB checkpoint has no config")
    try:
        model = CandidateHighresRGBMultiscaleLikelihood(
            image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
            fine_search_radius_px=float(config["fine_search_radius_px"]),
            fine_context_radius_px=float(config["fine_context_radius_px"]),
            broad_search_radius_px=float(config["fine_search_radius_px"]),
            broad_context_radius_px=float(config["fine_context_radius_px"]),
            texture_feature_dim=int(config["texture_feature_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            edge_chunk_size=int(config["edge_chunk_size"]),
            rgb_temperature=float(config["rgb_temperature"]),
            max_abs_edge_log_ratio=float(config["max_abs_edge_log_ratio"]),
        ).to(device)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("hybrid selector audit RGB checkpoint config is invalid") from error
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    return model


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_spatial_context_cache),
    }


def _selector_weights(
    *,
    runtime,
    phase_prediction,
    rgb_prediction,
    availability: torch.Tensor,
    image_size: tuple[int, int],
    top_k: int,
    grid_rows: int,
    grid_columns: int,
    score_weight_floor: float,
    score_weight_power: float,
    uniform_mixture_top_k: int,
    uniform_mixture_fractions: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Build every candidate selector before correct/wrong targets are read."""

    phase = phase_identity_confidence(
        runtime=runtime,
        prediction=phase_prediction,
        source_name="radio_final",
        edge_availability_override=availability,
        include_fixed_candidate_prior=False,
    )
    rgb = rgb_spatial_mode_quality(
        runtime=runtime,
        prediction=rgb_prediction,
        source_name="fine",
        edge_availability_override=availability,
    )
    joint = combine_selector_confidences(phase_confidence=phase, rgb_quality=rgb)
    xy = runtime.query_xy
    uniform = torch.ones((runtime.point_count,), device=xy.device)
    result = {
        "uniform_all": uniform,
        # This is still an all-token aggregation.  It tests whether the
        # target-free RGB mode quality is useful as a *soft* evidence weight,
        # without silently changing the candidate or pose contract.
        "rgb_soft_all_score": score_reweighted_selector_weights(
            selected_weights=uniform,
            scores=rgb,
            floor=float(score_weight_floor),
            power=float(score_weight_power),
        ),
    }
    if int(top_k) == 0:
        return result
    for name, scores in (("phase", phase), ("rgb", rgb), ("joint", joint)):
        selected = spatial_diverse_topk_weights(
            xy=xy,
            scores=scores,
            image_size=image_size,
            top_k=int(top_k),
            grid_rows=int(grid_rows),
            grid_columns=int(grid_columns),
        )
        result[f"{name}_top{int(top_k)}_grid"] = selected
        # Phase confidence was already falsified as an independent selector
        # signal.  Only evaluate a soft reweighting for the empirical RGB
        # spatial-mode signal, and keep its static weights shared by every
        # pose hypothesis and support-image counterfactual.
        if name == "rgb":
            score_weights = score_reweighted_selector_weights(
                selected_weights=selected,
                scores=rgb,
                floor=float(score_weight_floor),
                power=float(score_weight_power),
            )
            result[f"rgb_top{int(top_k)}_grid_score"] = score_weights
            if int(top_k) == int(uniform_mixture_top_k):
                for uniform_mass in uniform_mixture_fractions:
                    result[
                        f"rgb_top{int(top_k)}_grid_score_uniform{_uniform_mixture_name(uniform_mass)}"
                    ] = blend_selector_with_uniform_mass(
                        selector_weights=score_weights, uniform_mass=float(uniform_mass)
                    )
    return result


def _accumulate_variant(
    *,
    accumulator: dict[str, float],
    normal_correct_points: torch.Tensor,
    normal_wrong_points: torch.Tensor,
    rgb_deranged_correct_points: torch.Tensor,
    rgb_deranged_wrong_points: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
    temperature: float,
) -> None:
    normal_correct = aggregate_static_point_log_likelihood_ratios(
        point_log_likelihood_ratios=normal_correct_points, selector_weights=weights
    )
    normal_wrong = aggregate_static_point_log_likelihood_ratios(
        point_log_likelihood_ratios=normal_wrong_points, selector_weights=weights
    )
    rgb_correct = aggregate_static_point_log_likelihood_ratios(
        point_log_likelihood_ratios=rgb_deranged_correct_points, selector_weights=weights
    )
    rgb_wrong = aggregate_static_point_log_likelihood_ratios(
        point_log_likelihood_ratios=rgb_deranged_wrong_points, selector_weights=weights
    )
    _loss, _soft, normal_gap = pose_margin_terms(
        correct_scores=normal_correct,
        wrong_scores=normal_wrong,
        margin=float(margin),
        temperature=float(temperature),
    )
    _loss, _soft, rgb_gap = pose_margin_terms(
        correct_scores=rgb_correct,
        wrong_scores=rgb_wrong,
        margin=float(margin),
        temperature=float(temperature),
    )
    accumulator["query_count"] += 1.0
    accumulator["normal_gap_sum"] += float(normal_gap.item())
    accumulator["normal_win_sum"] += float((normal_gap > 0.0).item())
    accumulator["rgb_deranged_gap_sum"] += float(rgb_gap.item())
    accumulator["normal_minus_rgb_deranged_sum"] += float((normal_gap - rgb_gap).item())
    weight_sum = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    normalized = weights / weight_sum
    accumulator["active_token_sum"] += float((weights > 0.0).sum().item())
    accumulator["weight_sum"] += float(weight_sum.item())
    accumulator["effective_sample_size_sum"] += float(
        (1.0 / normalized.square().sum().clamp_min(torch.finfo(weights.dtype).tiny)).item()
    )


def summarize_pose_gap_distribution(
    *,
    normal_pose_gaps: np.ndarray,
    normal_minus_rgb_deranged_pose_gaps: np.ndarray,
    uniform_normal_pose_gaps: np.ndarray,
) -> dict[str, float | int]:
    """Report per-query tails and a paired comparison against uniform pooling.

    The aggregate mean alone can hide a selector that improves a few easy
    queries while turning previously positive queries into losses.  All three
    vectors are computed only after the target-free visual selector has been
    frozen, but this helper itself is deliberately a pure statistics routine.
    """

    normal = np.asarray(normal_pose_gaps, dtype=np.float64).reshape(-1)
    visual = np.asarray(normal_minus_rgb_deranged_pose_gaps, dtype=np.float64).reshape(-1)
    uniform = np.asarray(uniform_normal_pose_gaps, dtype=np.float64).reshape(-1)
    if (
        len(normal) == 0
        or visual.shape != normal.shape
        or uniform.shape != normal.shape
        or not np.isfinite(normal).all()
        or not np.isfinite(visual).all()
        or not np.isfinite(uniform).all()
    ):
        raise ValueError("selector pose-gap distribution is invalid")
    paired = normal - uniform
    tolerance = 1e-9
    uniform_wins = uniform > tolerance
    selector_losses = normal <= tolerance
    return {
        "median_normal_pose_gap": float(np.median(normal)),
        "p10_normal_pose_gap": float(np.quantile(normal, 0.10)),
        "minimum_normal_pose_gap": float(np.min(normal)),
        "negative_normal_pose_gap_count": int(np.sum(normal <= tolerance)),
        "median_normal_minus_rgb_deranged_pose_gap": float(np.median(visual)),
        "p10_normal_minus_rgb_deranged_pose_gap": float(np.quantile(visual, 0.10)),
        "paired_vs_uniform_mean_gap_delta": float(np.mean(paired)),
        "paired_vs_uniform_median_gap_delta": float(np.median(paired)),
        "paired_vs_uniform_p10_gap_delta": float(np.quantile(paired, 0.10)),
        "paired_vs_uniform_minimum_gap_delta": float(np.min(paired)),
        "paired_vs_uniform_win_count": int(np.sum(paired > tolerance)),
        "paired_vs_uniform_loss_count": int(np.sum(paired < -tolerance)),
        "paired_vs_uniform_tie_count": int(np.sum(np.abs(paired) <= tolerance)),
        "uniform_win_to_selector_loss_count": int(np.sum(uniform_wins & selector_losses)),
    }


def _assert_target_free_runtime_alignment(*, visual_runtime, target_runtime) -> None:
    """Ensure delayed targets are joined to precisely the visualized layout."""

    names = (
        "query_image_indices",
        "query_xy",
        "support_image_indices",
        "support_xy",
        "support_view_valid",
        "candidate_view_weights",
        "candidate_probabilities",
        "null_probabilities",
    )
    if any(
        not torch.equal(getattr(visual_runtime, name), getattr(target_runtime, name))
        for name in names
    ):
        raise RuntimeError("delayed pose targets do not align to the target-free visual runtime")


@torch.no_grad()
def run_audit(args: argparse.Namespace) -> dict[str, object]:
    if (
        int(args.grid_rows) <= 0
        or int(args.grid_columns) <= 0
        or int(args.support_permutation_shift) == 0
        or float(args.rgb_cache_gb) <= 0.0
        or not math.isfinite(float(args.score_weight_floor))
        or not math.isfinite(float(args.score_weight_power))
        or float(args.score_weight_floor) <= 0.0
        or float(args.score_weight_floor) > 1.0
        or float(args.score_weight_power) <= 0.0
    ):
        raise ValueError("selector audit arguments are invalid")
    topk_values = parse_selector_topk(args.selector_top_k)
    uniform_mixture_fractions = parse_uniform_mixture_fractions(args.uniform_mixture_fractions)
    if (
        int(args.uniform_mixture_top_k) <= 0
        or (uniform_mixture_fractions and int(args.uniform_mixture_top_k) not in topk_values)
    ):
        raise ValueError("uniform selector mixture top-K must be listed in --selector-top-k")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"selector audit output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("selector audit requested CUDA but CUDA is unavailable")

    layout_path = Path(args.rgb_spatial_layout)
    targets_path = Path(args.geometry_training_targets)
    phase_path = Path(args.phase_checkpoint)
    hybrid_path = Path(args.hybrid_checkpoint)
    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
    layout_sha = file_sha256_short(layout_path)
    targets_sha = file_sha256_short(targets_path)
    # ``build_train_query_groups`` validates that all pose targets remain
    # train-only and are row-aligned to the frozen target-free layout.
    groups = build_train_query_groups(layout=layout, targets=targets)
    phase_checkpoint = _load_phase_checkpoint(phase_path)
    phase_partition, heldout_query_ids = _checkpoint_partition(
        phase_checkpoint, all_train_query_ids=tuple(sorted(groups))
    )
    state_dict, hybrid_metadata = _read_hybrid_checkpoint(hybrid_path)
    lineage = hybrid_metadata.get("lineage")
    if not isinstance(lineage, Mapping) or str(lineage.get("layout_sha256", "")) != layout_sha:
        raise ValueError("selector audit hybrid checkpoint layout lineage is stale")
    if str(lineage.get("geometry_training_targets_sha256", "")) != targets_sha:
        raise ValueError("selector audit hybrid checkpoint geometry target lineage is stale")
    training = hybrid_metadata.get("training")
    if not isinstance(training, Mapping) or training.get("phase_train_query_partition") != phase_partition:
        raise ValueError("selector audit hybrid checkpoint phase partition differs from phase expert")
    source_paths = _source_paths(args)
    sources = load_context_attention_sources(
        radio_final_context_cache=source_paths["radio_final"],
        radio_intermediate_context_cache=source_paths["radio_intermediate"],
        alike_spatial_context_cache=source_paths["alike"],
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    image_ids, image_sizes, source_grids = _source_table(sources)
    phase_model = _load_phase_model(
        checkpoint=phase_checkpoint,
        source_grids=source_grids,
        image_sizes=image_sizes,
        device=device,
    ).eval()
    rgb_model = _load_rgb_model(
        state_dict=state_dict, metadata=hybrid_metadata, image_sizes=image_sizes, device=device
    )
    unique_sizes = np.unique(image_sizes, axis=0)
    if unique_sizes.shape != (1, 2):
        raise ValueError("selector audit requires one processed coordinate image size")
    coordinate_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
    rgb_size = _discover_rgb_image_size(image_root=Path(args.image_root), image_id=str(image_ids[0]))
    validate_rgb_coordinate_bridge(
        source_metadata=sources[0].metadata,
        coordinate_image_size=coordinate_size,
        rgb_image_size=rgb_size,
    )
    complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
    config = hybrid_metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("selector audit hybrid checkpoint config is missing")
    score_args = SimpleNamespace(
        identity_weight=float(config["identity_weight"]),
        spatial_weight=float(config["spatial_weight"]),
        max_abs_pose_log_ratio=float(config["max_abs_pose_log_ratio"]),
        pose_margin=0.20,
        soft_hard_temperature=0.35,
    )
    cache = TensorImageLRUCache(
        max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
        storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
    )
    cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else device
    accumulators: dict[str, dict[str, float]] = {}
    per_query: list[dict[str, object]] = []
    for query_id in heldout_query_ids:
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("selector audit held-out query is absent from train groups")
        points = np.arange(group.point_count, dtype=np.int64)
        # Do not materialize a target-bearing batch before all visual forwards
        # and static token weights are complete.  The runtime comes only from
        # the validated target-free layout.
        visual_runtime = _slice_runtime(complete_runtime, group.layout_rows[points])
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=visual_runtime,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_size,
            rgb_image_size=rgb_size,
            radius_px=float(rgb_model.full_patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=device,
            cache_device=cache_device,
        )
        rgb_deranged_patches = permute_support_patches_with_phase_identity_point_blocks(
            runtime=visual_runtime,
            support_patches=support_patches,
            shift=int(args.support_permutation_shift),
        )
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            rgb_normal = rgb_model(
                runtime=visual_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                active_sources=("fine",),
            )
            rgb_deranged = rgb_model(
                runtime=visual_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=rgb_deranged_patches,
                active_sources=("fine",),
            )
        phase_normal = phase_model(runtime=visual_runtime)
        phase_deranged = phase_model(
            runtime=visual_runtime, support_permutation_shift=int(args.support_permutation_shift)
        )
        paired_availability = _phase_control_common_availability(
            phase_normal=phase_normal,
            phase_deranged=phase_deranged,
            rgb_normal=rgb_normal,
            rgb_deranged=rgb_deranged,
        )
        runtime_availability = runtime_visual_edge_availability(
            runtime=visual_runtime,
            phase_prediction=phase_normal,
            rgb_prediction=rgb_normal,
            phase_source_name="radio_final",
            rgb_source_name="fine",
        )
        selector_sets = _selector_weights(
            runtime=visual_runtime,
            phase_prediction=phase_normal,
            rgb_prediction=rgb_normal,
            availability=runtime_availability,
            image_size=coordinate_size,
            top_k=0,
            grid_rows=int(args.grid_rows),
            grid_columns=int(args.grid_columns),
            score_weight_floor=float(args.score_weight_floor),
            score_weight_power=float(args.score_weight_power),
            uniform_mixture_top_k=int(args.uniform_mixture_top_k),
            uniform_mixture_fractions=uniform_mixture_fractions,
        )
        for top_k in topk_values:
            selector_sets.update(
                _selector_weights(
                    runtime=visual_runtime,
                    phase_prediction=phase_normal,
                    rgb_prediction=rgb_normal,
                    availability=runtime_availability,
                    image_size=coordinate_size,
                    top_k=int(top_k),
                    grid_rows=int(args.grid_rows),
                    grid_columns=int(args.grid_columns),
                    score_weight_floor=float(args.score_weight_floor),
                    score_weight_power=float(args.score_weight_power),
                    uniform_mixture_top_k=int(args.uniform_mixture_top_k),
                    uniform_mixture_fractions=uniform_mixture_fractions,
                )
            )
        # Only now join train-only correct/coherent-wrong projections.  The
        # equality check makes it impossible to accidentally score targets
        # against a different visual point order or geometry layout.
        batch = _query_batch_from_group(
            group=group, complete_runtime=complete_runtime, point_positions=points, device=device
        )
        _assert_target_free_runtime_alignment(
            visual_runtime=visual_runtime, target_runtime=batch.runtime
        )
        normal_correct, normal_wrong = _hybrid_pose_scores(
            runtime=batch.runtime,
            phase_prediction=phase_normal,
            rgb_prediction=rgb_normal,
            availability=paired_availability,
            correct_offsets_xy=batch.correct_projection_offsets_xy,
            correct_valid=batch.correct_projection_valid,
            wrong_offsets_xy=batch.wrong_projection_offsets_xy,
            wrong_valid=batch.wrong_projection_valid,
            args=score_args,
        )
        rgb_correct, rgb_wrong = _hybrid_pose_scores(
            runtime=batch.runtime,
            phase_prediction=phase_normal,
            rgb_prediction=rgb_deranged,
            availability=paired_availability,
            correct_offsets_xy=batch.correct_projection_offsets_xy,
            correct_valid=batch.correct_projection_valid,
            wrong_offsets_xy=batch.wrong_projection_offsets_xy,
            wrong_valid=batch.wrong_projection_valid,
            args=score_args,
        )
        uniform = selector_sets["uniform_all"]
        uniform_correct = aggregate_static_point_log_likelihood_ratios(
            point_log_likelihood_ratios=normal_correct.point_log_likelihood_ratios,
            selector_weights=uniform,
        )
        uniform_wrong = aggregate_static_point_log_likelihood_ratios(
            point_log_likelihood_ratios=normal_wrong.point_log_likelihood_ratios,
            selector_weights=uniform,
        )
        if not torch.allclose(
            uniform_correct, normal_correct.pose_log_likelihood_ratios, atol=1e-5, rtol=1e-5
        ) or not torch.allclose(
            uniform_wrong, normal_wrong.pose_log_likelihood_ratios, atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError("selector audit uniform aggregation does not reproduce hybrid scoring")
        query_record: dict[str, object] = {"query_id": str(query_id), "point_count": int(group.point_count)}
        for name, weights in selector_sets.items():
            stats = accumulators.setdefault(
                name,
                {
                    "query_count": 0.0,
                    "normal_gap_sum": 0.0,
                    "normal_win_sum": 0.0,
                    "rgb_deranged_gap_sum": 0.0,
                    "normal_minus_rgb_deranged_sum": 0.0,
                    "active_token_sum": 0.0,
                    "weight_sum": 0.0,
                    "effective_sample_size_sum": 0.0,
                },
            )
            before = dict(stats)
            _accumulate_variant(
                accumulator=stats,
                normal_correct_points=normal_correct.point_log_likelihood_ratios,
                normal_wrong_points=normal_wrong.point_log_likelihood_ratios,
                rgb_deranged_correct_points=rgb_correct.point_log_likelihood_ratios,
                rgb_deranged_wrong_points=rgb_wrong.point_log_likelihood_ratios,
                weights=weights,
                margin=float(score_args.pose_margin),
                temperature=float(score_args.soft_hard_temperature),
            )
            query_record[name] = {
                "normal_pose_gap": stats["normal_gap_sum"] - before["normal_gap_sum"],
                "rgb_deranged_pose_gap": stats["rgb_deranged_gap_sum"]
                - before["rgb_deranged_gap_sum"],
                "normal_minus_rgb_deranged_pose_gap": stats[
                    "normal_minus_rgb_deranged_sum"
                ]
                - before["normal_minus_rgb_deranged_sum"],
                "active_token_count": int((weights > 0.0).sum().item()),
                "weight_sum": float(weights.sum().item()),
                "effective_sample_size": float(
                    (
                        1.0
                        / (
                            (weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny))
                            .square()
                            .sum()
                            .clamp_min(torch.finfo(weights.dtype).tiny)
                        )
                    ).item()
                ),
            }
        per_query.append(query_record)
    summary_selectors: dict[str, object] = {}
    uniform_records = [record["uniform_all"] for record in per_query]
    if len(uniform_records) != len(per_query):
        raise RuntimeError("selector audit is missing the uniform paired baseline")
    uniform_normal_gaps = np.asarray(
        [float(record["normal_pose_gap"]) for record in uniform_records], dtype=np.float64
    )
    for name, values in sorted(accumulators.items()):
        count = float(values["query_count"])
        if count <= 0.0:
            raise RuntimeError("selector audit produced an empty selector summary")
        selector_records = [record[name] for record in per_query]
        if len(selector_records) != len(per_query):
            raise RuntimeError("selector audit lost a per-query selector record")
        normal_gaps = np.asarray(
            [float(record["normal_pose_gap"]) for record in selector_records], dtype=np.float64
        )
        visual_gaps = np.asarray(
            [float(record["normal_minus_rgb_deranged_pose_gap"]) for record in selector_records],
            dtype=np.float64,
        )
        summary_selectors[name] = {
            "query_count": int(count),
            "mean_normal_pose_gap": values["normal_gap_sum"] / count,
            "normal_pose_win_fraction": values["normal_win_sum"] / count,
            "mean_rgb_deranged_pose_gap": values["rgb_deranged_gap_sum"] / count,
            "mean_normal_minus_rgb_deranged_pose_gap": values[
                "normal_minus_rgb_deranged_sum"
            ]
            / count,
            "mean_active_token_count": values["active_token_sum"] / count,
            "mean_weight_sum": values["weight_sum"] / count,
            "mean_effective_sample_size": values["effective_sample_size_sum"] / count,
            **summarize_pose_gap_distribution(
                normal_pose_gaps=normal_gaps,
                normal_minus_rgb_deranged_pose_gaps=visual_gaps,
                uniform_normal_pose_gaps=uniform_normal_gaps,
            ),
        }
    summary = {
        "format": AUDIT_FORMAT,
        "selector_format": CANDIDATE_POSE_EVIDENCE_SELECTOR_FORMAT,
        "stage": "frozen_train_only_static_selector_diagnostic",
        "target_free_selector": True,
        "targets_joined_only_after_visual_forward": True,
        "targets_materialized_only_after_static_selector": True,
        "diagnostic_only": True,
        "pnp_integration_allowed": False,
        "heldout_scope": "phase_inner_validation_train_queries_only_v1",
        "checkpoint_failed_parent_gate": hybrid_metadata.get("train_only_inner_gate_passed") is not True,
        "config": {
            "selector_top_k": list(topk_values),
            "grid_rows": int(args.grid_rows),
            "grid_columns": int(args.grid_columns),
            "phase_source": "radio_final",
            "rgb_source": "fine",
            "phase_selector_excludes_fixed_candidate_prior": True,
            "selector_fixed_before_pose_projection": True,
            "same_weights_used_for_rgb_derangement_control": True,
            "selector_availability": "normal_runtime_phase_rgb_intersection_only",
            "paired_score_availability": "normal_deranged_common_intersection_only",
            "rgb_score_reweighting": {
                "floor": float(args.score_weight_floor),
                "power": float(args.score_weight_power),
                "phase_score_reweighting_evaluated": False,
            },
            "uniform_fallback_mixture": {
                "rgb_top_k": int(args.uniform_mixture_top_k),
                "uniform_mass_fractions": list(uniform_mixture_fractions),
                "each_component_normalized_before_mixing": True,
            },
        },
        "lineage": {
            "layout": str(layout_path.resolve()),
            "layout_sha256": layout_sha,
            "geometry_training_targets": str(targets_path.resolve()),
            "geometry_training_targets_sha256": targets_sha,
            "phase_checkpoint": str(phase_path.resolve()),
            "phase_checkpoint_sha256": file_sha256_short(phase_path),
            "hybrid_checkpoint": str(hybrid_path.resolve()),
            "hybrid_checkpoint_sha256": file_sha256_short(hybrid_path),
            "source_cache_sha256": {
                name: file_sha256_short(path) for name, path in source_paths.items()
            },
        },
        "phase_partition": phase_partition,
        "selectors": summary_selectors,
        "per_query": per_query,
        "rgb_cache": cache.summary(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selector_audit": summary_selectors, "output": str(output_dir)}, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    run_audit(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
