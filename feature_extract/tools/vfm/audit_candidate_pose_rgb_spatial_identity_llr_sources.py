"""Audit source-specific identity evidence on the train-only inner fold.

This command is a diagnostic, not a localization evaluation.  It loads a
fixed P1 or gate-approved broad-observation identity checkpoint and evaluates
the same frozen P1 candidate set with every appearance source enabled, one
source enabled, and one source removed.  Query selection remains target-free;
exact-track and hard-repeat labels are joined only after the visual scores
exist.  It never reads a pose, validation/test image, render, image retrieval
result, or submap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT,
    _evaluate_inner_validation_target_free,
    _validate_identity_target_contract,
    _write_json_atomically,
)
from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT as OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _source_table,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    runtime_from_target_free_layout,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES,
    CandidatePoseRGBSpatialIdentityLLR,
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


_ACCEPTED_CHECKPOINT_FORMATS = frozenset(
    {CHECKPOINT_FORMAT, OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT}
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--validation-selector-policy", default="support_coverage")
    parser.add_argument("--validation-selector-point-budget", type=int, default=64)
    parser.add_argument("--validation-selector-grid-rows", type=int, default=4)
    parser.add_argument("--validation-selector-grid-columns", type=int, default=4)
    parser.add_argument("--candidate-prior-logit-weight", type=float, default=1.0)
    parser.add_argument("--identity-margin", type=float, default=0.25)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.25)
    parser.add_argument("--hard-pose-margin", type=float, default=0.25)
    parser.add_argument("--minimum-hard-pose-points", type=int, default=4)
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=64)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=1)
    parser.add_argument("--permutation-control-shift", type=int, default=1)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _source_scale_configurations() -> dict[str, Mapping[str, float] | None]:
    """Return fixed target-free all/only/leave-one-out appearance ablations."""

    names = tuple(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES)
    output: dict[str, Mapping[str, float] | None] = {"all": None}
    for selected in names:
        output[f"only_{selected}"] = {
            name: float(name == selected) for name in names
        }
    for omitted in names:
        output[f"without_{omitted}"] = {name: float(name != omitted) for name in names}
    return output


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], Mapping[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("identity LLR source audit checkpoint is malformed")
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata")
    checkpoint_format = payload.get("format")
    if (
        checkpoint_format not in _ACCEPTED_CHECKPOINT_FORMATS
        or not isinstance(state_dict, Mapping)
        or not isinstance(metadata, Mapping)
        or metadata.get("format") != checkpoint_format
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
    ):
        raise ValueError("identity LLR source audit checkpoint format is incompatible")
    return dict(state_dict), metadata


def _load_model_state(
    *, model: CandidatePoseRGBSpatialIdentityLLR, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Accept only the documented neutral heads missing from an older L0 init."""

    try:
        result = model.load_state_dict(dict(state_dict), strict=False)
    except RuntimeError as error:
        raise ValueError("identity LLR source audit checkpoint state is incompatible") from error
    allowed_missing = {
        name
        for prefix in ("support_view_head", "null_head")
        for name in model.state_dict()
        if name.startswith(prefix + ".")
    }
    if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
        raise ValueError("identity LLR source audit checkpoint state is incompatible")


def _lineage(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    source_metadata: Mapping[str, object],
    rgb_coordinate_bridge: Mapping[str, object],
) -> dict[str, object]:
    return {
        "radio_final_context_cache_sha256": file_sha256_short(radio_final_context_cache),
        "radio_intermediate_context_cache_sha256": file_sha256_short(
            radio_intermediate_context_cache
        ),
        "alike_spatial_context_cache_sha256": file_sha256_short(alike_spatial_context_cache),
        "source_image_manifest_sha256": str(
            source_metadata.get("source_image_manifest_sha256", "")
        ),
        "rgb_coordinate_bridge": dict(rgb_coordinate_bridge),
    }


def _validate_checkpoint_source_lineage(
    *, checkpoint_lineage: Mapping[str, object], source_lineage: Mapping[str, object]
) -> str:
    """Require the source cache contract that actually bound the checkpoint.

    New identity-LLR checkpoints carry the source hashes directly.  The first
    numerically valid frozen-head smoke checkpoint predates that persistence
    fix, but still contains exact hashes for the three source artifacts under
    ``inputs``.  That fallback remains strict: it cannot accept a different
    cache, and it is reported as legacy rather than being treated as canonical.
    """

    required = (
        "radio_final_context_cache_sha256",
        "radio_intermediate_context_cache_sha256",
        "alike_spatial_context_cache_sha256",
        "source_image_manifest_sha256",
        "rgb_coordinate_bridge",
    )
    if all(name in checkpoint_lineage for name in required):
        if any(checkpoint_lineage.get(name) != source_lineage.get(name) for name in required):
            raise ValueError("identity LLR source audit checkpoint lineage differs from inputs")
        return "canonical_source_lineage_v1"

    inputs = checkpoint_lineage.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("identity LLR source audit checkpoint lacks source cache lineage")
    artifact_names = {
        "radio_final_context_cache": "radio_final_context_cache_sha256",
        "radio_intermediate_context_cache": "radio_intermediate_context_cache_sha256",
        "alike_spatial_context_cache": "alike_spatial_context_cache_sha256",
    }
    for artifact_name, source_name in artifact_names.items():
        record = inputs.get(artifact_name)
        if not isinstance(record, Mapping) or record.get("sha256") != source_lineage[source_name]:
            raise ValueError("identity LLR source audit checkpoint cache lineage differs from inputs")
    if checkpoint_lineage.get("rgb_coordinate_bridge") != source_lineage["rgb_coordinate_bridge"]:
        raise ValueError("identity LLR source audit checkpoint RGB coordinate bridge differs")
    # Old checkpoints did not preserve this field.  A nonempty value must
    # agree; an empty value is accepted only because the exact cache hashes
    # above bind the source metadata transitively.
    stored_manifest = str(checkpoint_lineage.get("source_image_manifest_sha256", ""))
    if stored_manifest and stored_manifest != source_lineage["source_image_manifest_sha256"]:
        raise ValueError("identity LLR source audit checkpoint image manifest differs")
    return "legacy_exact_input_cache_hash_fallback_v1"


def _derived_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    normal = float(metrics["normal_mean_correct_minus_hardest_wrong"])
    hard_repeat = float(metrics["hard_repeat_mean_correct_minus_coherent_wrong"])
    return {
        "normal_minus_permuted_gap": normal
        - float(metrics["permuted_mean_correct_minus_hardest_wrong"]),
        "normal_minus_position_only_gap": normal
        - float(metrics["position_only_mean_correct_minus_hardest_wrong"]),
        "hard_repeat_minus_position_only_gap": hard_repeat
        - float(metrics["hard_repeat_position_only_mean_correct_minus_coherent_wrong"]),
    }


def _write_audit_output(*, path: Path, value: Mapping[str, object]) -> None:
    """Atomically persist the rank-zero result even for a new output directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(path, value)


def audit_candidate_pose_rgb_spatial_identity_llr_sources(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run all target-free source ablations on the train-only inner fold."""

    checkpoint_path = Path(args.checkpoint)
    layout_path = Path(args.rgb_spatial_layout)
    targets_path = Path(args.training_targets)
    hard_path = Path(args.hard_repeat_targets)
    output_path = Path(args.output_json)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite source audit: {output_path}")
    if (
        not checkpoint_path.is_file()
        or not layout_path.is_file()
        or not targets_path.is_file()
        or not hard_path.is_file()
        or float(args.rgb_cache_gb) <= 0.0
        or int(args.validation_selector_point_budget) <= 0
        or int(args.max_hard_repeat_edges_per_query) <= 0
        or int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
    ):
        raise ValueError("identity LLR source audit arguments are invalid")
    state = _initialize_distributed(str(args.device))
    try:
        state_dict, metadata = _load_checkpoint(checkpoint_path)
        config = metadata.get("config")
        checkpoint_lineage = metadata.get("lineage")
        if not isinstance(config, Mapping) or not isinstance(checkpoint_lineage, Mapping):
            raise ValueError("identity LLR source audit checkpoint lacks config or lineage")

        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        layout_sha = file_sha256_short(layout_path)
        targets_sha = file_sha256_short(targets_path)
        from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
            validate_training_layout_and_targets,
        )

        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=layout_sha
        )
        hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(hard_path)
        _validate_identity_target_contract(targets=targets, hard_targets=hard_targets)
        groups = build_train_query_groups(layout=layout, targets=targets)
        hard_by_query = build_hard_repeat_query_targets(
            layout=layout,
            targets=targets,
            hard_repeat_targets=hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=targets_sha,
        )
        _, inner_query_ids = _partition_train_queries_for_inner_validation(
            query_ids=tuple(sorted(groups)),
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
            raise ValueError("identity LLR source audit requires common context image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        bridge = validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        source_lineage = _lineage(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            source_metadata=sources[0].metadata,
            rgb_coordinate_bridge=bridge,
        )
        checkpoint_lineage_contract = _validate_checkpoint_source_lineage(
            checkpoint_lineage=checkpoint_lineage,
            source_lineage=source_lineage,
        )

        model = CandidatePoseRGBSpatialIdentityLLR(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            rgb_context_radius_px=float(config["rgb_context_radius_px"]),
            rgb_step_px=float(config["rgb_step_px"]),
            texture_feature_dim=int(config["texture_feature_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            max_abs_log_ratio=float(config["max_abs_log_ratio"]),
            edge_chunk_size=int(config["edge_chunk_size"]),
            activation_checkpointing=False,
            context_windows=dict(config["context_windows"]),
        )
        _load_model_state(model=model, state_dict=state_dict)
        model = model.to(state.device).eval()
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)

        results: dict[str, object] = {}
        for name, scales in _source_scale_configurations().items():
            metrics = _evaluate_inner_validation_target_free(
                model=model,
                layout=layout,
                groups=groups,
                hard_by_query=hard_by_query,
                complete_runtime=complete_runtime,
                query_ids=inner_query_ids,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                rgb_context_radius_px=float(config["rgb_context_radius_px"]),
                rgb_step_px=float(config["rgb_step_px"]),
                cache=cache,
                state=state,
                selector_policy=str(args.validation_selector_policy),
                selector_point_budget=int(args.validation_selector_point_budget),
                selector_grid_rows=int(args.validation_selector_grid_rows),
                selector_grid_columns=int(args.validation_selector_grid_columns),
                identity_margin=float(args.identity_margin),
                hard_repeat_margin=float(args.hard_repeat_margin),
                hard_pose_margin=float(args.hard_pose_margin),
                minimum_hard_pose_points=int(args.minimum_hard_pose_points),
                candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
                seed=int(args.seed),
                permutation_shift=int(args.permutation_control_shift),
                amp_enabled=amp_enabled,
                visual_source_scales=scales,
            )
            results[name] = {
                "visual_source_scales": None if scales is None else dict(scales),
                "metrics": metrics,
                "derived": _derived_metrics(metrics),
            }

        output: dict[str, object] = {
            "stage": "audit_candidate_pose_rgb_spatial_identity_llr_sources",
            "output_json": str(output_path),
            "rank": int(state.rank),
            "world_size": int(state.world_size),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "checkpoint_source_lineage_contract": checkpoint_lineage_contract,
            "inner_validation": {
                "fold_count": int(args.inner_validation_fold_count),
                "fold_index": int(args.inner_validation_fold_index),
                "query_count": int(len(inner_query_ids)),
                "query_ids": list(inner_query_ids),
                "target_free_selector": {
                    "policy": str(args.validation_selector_policy),
                    "point_budget": int(args.validation_selector_point_budget),
                    "grid_rows": int(args.validation_selector_grid_rows),
                    "grid_columns": int(args.validation_selector_grid_columns),
                },
            },
            "source_ablations": results,
            "rgb_cache": cache.summary(),
            "protocol": {
                "diagnostic_only": True,
                "train_only_targets_joined_after_target_free_inference": True,
                "heldout_validation_or_test_not_run": True,
                "pose_or_ground_truth_not_available_to_runtime_encoder": True,
                "no_render": True,
                "no_image_retrieval_or_submap": True,
            },
        }
        if state.rank == 0:
            _write_audit_output(path=output_path, value=output)
        return output
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    result = audit_candidate_pose_rgb_spatial_identity_llr_sources(parse_args(argv))
    if int(result["rank"]) == 0:
        print(
            json.dumps(
                {
                    "output_json": str(result.get("output_json", "")),
                    "checkpoint": result["checkpoint"],
                    "source_count": len(result["source_ablations"]),
                    "world_size": int(result["world_size"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
