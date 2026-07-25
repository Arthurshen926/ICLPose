"""Re-audit a frozen multiscale phase-identity checkpoint on exact hard edges.

The original training checkpoint serializes its final inner-fold gate.  This
separate audit exists because early checkpoints predate two protocol fixes:

* static geometric repeat edges must be filtered to registered exact-track
  identity before they can audit an identity likelihood; and
* a visual control must derange complete support appearances across distant
  query-point blocks, not merely roll adjacent flattened edges.

The model forward is target-free.  Registered identity and hard-repeat labels
are joined only after every fixed P1 point for the checkpoint's held-out
train-query fold has been scored.  A passing result permits only the next
diagnostic stage (frozen pose-rank audit), never PnP integration.
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

from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    CHECKPOINT_FORMAT,
    GATE_FORMAT,
    _source_table,
    build_exact_identity_query_targets,
    evaluate_static_hard_gate,
    filter_static_hard_repeat_groups_to_registered_exact_identity,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _DistributedState,
    _partition_train_queries_for_inner_validation,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    train_query_partition_manifest,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT,
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityLLR,
    PhaseIdentitySourceConfig,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
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


AUDIT_FORMAT = "candidate_multiscale_phase_identity_static_hard_reaudit_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--registered-identity-targets", required=True)
    parser.add_argument("--static-hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("phase identity checkpoint is not a mapping")
    return dict(payload)


def _source_configs_from_checkpoint(config: Mapping[str, object]) -> dict[str, PhaseIdentitySourceConfig]:
    raw = config.get("source_configs")
    if not isinstance(raw, Mapping) or set(raw) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES):
        raise ValueError("phase identity checkpoint source configuration is incomplete")
    resolved: dict[str, PhaseIdentitySourceConfig] = {}
    for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        item = raw.get(name)
        if not isinstance(item, Mapping):
            raise ValueError("phase identity checkpoint source configuration is malformed")
        try:
            resolved[name] = PhaseIdentitySourceConfig(
                name=str(item["name"]),
                window_size=int(item["window_size"]),
                shift_radius=int(item["shift_radius"]),
                region_bins=int(item["region_bins"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("phase identity checkpoint source configuration is malformed") from error
        if resolved[name].name != name:
            raise ValueError("phase identity checkpoint source configuration name differs")
    return resolved


def promotable_phase_identity_sources(gate: Mapping[str, object]) -> tuple[str, ...]:
    """Return source-specific promotions from a validated re-audit gate.

    A combined convex blend is not treated as a product of independent
    likelihoods.  Therefore, if it does not itself pass, only explicitly
    passing source names can move to the frozen pose-rank diagnostic.
    """

    if not isinstance(gate, Mapping) or gate.get("format") != GATE_FORMAT:
        raise ValueError("phase identity gate has an invalid format")
    sources = gate.get("sources")
    declared = gate.get("promotable_source_names")
    source_passes = gate.get("independent_source_passes")
    if (
        not isinstance(sources, Mapping)
        or not isinstance(declared, list)
        or not isinstance(source_passes, Mapping)
        or set(source_passes) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
    ):
        raise ValueError("phase identity gate lacks source-specific promotion evidence")
    values = tuple(str(name) for name in declared)
    if len(values) != len(set(values)) or set(values) - set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES):
        raise ValueError("phase identity gate declares invalid promotable sources")
    expected = tuple(
        name
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        if source_passes.get(name) is True
        and isinstance(sources.get(name), Mapping)
        and isinstance(sources[name].get("gate"), Mapping)
        and sources[name]["gate"].get("passed") is True
    )
    if values != expected:
        raise ValueError("phase identity gate source promotion disagrees with source audits")
    combined = sources.get("combined")
    combined_passed = bool(
        isinstance(combined, Mapping)
        and isinstance(combined.get("gate"), Mapping)
        and combined["gate"].get("passed") is True
    )
    if bool(gate.get("combined_passed")) != combined_passed:
        raise ValueError("phase identity gate combined result is inconsistent")
    if bool(gate.get("passed")) != bool(combined_passed or values):
        raise ValueError("phase identity gate aggregate result is inconsistent")
    return values


def _validate_checkpoint(
    *,
    checkpoint: Mapping[str, object],
    checkpoint_path: Path,
    layout_path: Path,
    layout_metadata: Mapping[str, object],
    source_paths: Mapping[str, Path],
    source_manifest_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    runtime_contract = checkpoint.get("runtime_contract")
    lineage = checkpoint.get("lineage")
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("model_format") != CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT
        or not isinstance(model_config, Mapping)
        or not isinstance(state_dict, Mapping)
        or not isinstance(runtime_contract, Mapping)
        or not isinstance(lineage, Mapping)
    ):
        raise ValueError("phase identity checkpoint is incomplete")
    forbidden = set(str(value) for value in runtime_contract.get("forbidden_encoder_inputs", ()))
    required_forbidden = {
        "pose",
        "projection_offset",
        "residual",
        "track_id",
        "candidate_rank",
        "coarse_score",
        "training_label",
    }
    if (
        runtime_contract.get("target_free_runtime") is not True
        or not required_forbidden.issubset(forbidden)
        or runtime_contract.get("render") is not False
        or runtime_contract.get("image_retrieval_or_submap") is not False
        or tuple(runtime_contract.get("visual_sources", ()))
        != CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        or runtime_contract.get("support_view_marginalization")
        != "fixed_mass_neutral_missing_view_v1"
    ):
        raise ValueError("phase identity checkpoint violates the target-free runtime contract")
    if (
        str(lineage.get("layout_sha256", "")) != file_sha256_short(layout_path)
        or str(lineage.get("descriptor_space_id", ""))
        != str(layout_metadata.get("descriptor_space_id", ""))
        or str(lineage.get("projection_space_id", ""))
        != str(layout_metadata.get("projection_space_id", ""))
        or str(lineage.get("source_image_manifest_sha256", "")) != str(source_manifest_sha256)
    ):
        raise ValueError("phase identity checkpoint lineage differs from this target-free layout")
    source_hashes = lineage.get("source_cache_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(source_paths):
        raise ValueError("phase identity checkpoint source-cache lineage is incomplete")
    for name, path in source_paths.items():
        if str(source_hashes.get(name, "")) != file_sha256_short(path):
            raise ValueError("phase identity checkpoint source-cache lineage is stale")
    storage = str(model_config.get("source_storage_dtype", ""))
    if storage not in {"float16", "float32"}:
        raise ValueError("phase identity checkpoint source storage dtype is invalid")
    _source_configs_from_checkpoint(model_config)
    weights = model_config.get("source_weights")
    if not isinstance(weights, Mapping) or set(weights) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES):
        raise ValueError("phase identity checkpoint source weights are invalid")
    if not Path(checkpoint_path).is_file():
        raise ValueError("phase identity checkpoint path is invalid")
    return dict(model_config), dict(lineage)


def _checkpoint_partition(
    checkpoint: Mapping[str, object], *, all_train_query_ids: Sequence[str]
) -> tuple[dict[str, object], tuple[str, ...]]:
    partition = checkpoint.get("train_query_partition")
    if not isinstance(partition, Mapping):
        raise ValueError("phase identity checkpoint lacks its train-query partition")
    try:
        all_train = tuple(str(value) for value in partition["all_train"]["query_ids"])
        inner_train = tuple(str(value) for value in partition["inner_train"]["query_ids"])
        inner_validation = tuple(str(value) for value in partition["inner_validation"]["query_ids"])
        fold_count = int(partition["fold_count"])
        fold_index = int(partition["fold_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("phase identity checkpoint train-query partition is malformed") from error
    rebuilt = train_query_partition_manifest(
        all_query_ids=tuple(sorted(str(value) for value in all_train_query_ids)),
        inner_train_query_ids=inner_train,
        inner_validation_query_ids=inner_validation,
        fold_count=fold_count,
        fold_index=fold_index,
    )
    if dict(partition) != rebuilt or tuple(sorted(all_train)) != tuple(sorted(all_train_query_ids)):
        raise ValueError("phase identity checkpoint train-query partition no longer matches inputs")
    expected_train, expected_validation = _partition_train_queries_for_inner_validation(
        query_ids=all_train_query_ids, fold_count=fold_count, fold_index=fold_index
    )
    if tuple(inner_train) != tuple(expected_train) or tuple(inner_validation) != tuple(expected_validation):
        raise ValueError("phase identity checkpoint train-query split is not deterministic")
    return rebuilt, tuple(inner_validation)


def _load_model(
    *,
    checkpoint: Mapping[str, object],
    source_grids: Mapping[str, torch.Tensor],
    image_sizes: np.ndarray,
    device: torch.device,
) -> CandidateMultiscalePhaseIdentityLLR:
    config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("phase identity checkpoint model state is invalid")
    storage = torch.float16 if str(config.get("source_storage_dtype")) == "float16" else torch.float32
    model = CandidateMultiscalePhaseIdentityLLR(
        sources=source_grids,
        image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        source_configs=_source_configs_from_checkpoint(config),
        source_weights={name: float(value) for name, value in dict(config["source_weights"]).items()},
        hidden_dim=int(config["hidden_dim"]),
        max_abs_log_ratio=float(config["max_abs_log_ratio"]),
        source_storage_dtype=storage,
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    return model.eval()


def _atomic_json_write(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_candidate_multiscale_phase_identity_llr(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite phase identity re-audit")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("phase identity re-audit requested CUDA but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    paths = {
        "layout": Path(args.rgb_spatial_layout),
        "geometry": Path(args.geometry_training_targets),
        "identity": Path(args.registered_identity_targets),
        "static_hard": Path(args.static_hard_repeat_targets),
        "checkpoint": Path(args.checkpoint),
    }
    source_paths = {
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_spatial_context_cache),
    }
    if not all(path.is_file() for path in (*paths.values(), *source_paths.values())):
        raise FileNotFoundError("phase identity re-audit input is absent")

    layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
    geometry_targets = load_candidate_pose_rgb_spatial_training_targets(paths["geometry"])
    identity_targets = load_candidate_pose_rgb_spatial_training_targets(paths["identity"])
    layout_sha = file_sha256_short(paths["layout"])
    geometry_sha = file_sha256_short(paths["geometry"])
    validate_training_layout_and_targets(
        layout=layout, targets=geometry_targets, layout_sha256=layout_sha
    )
    validate_training_layout_and_targets(
        layout=layout, targets=identity_targets, layout_sha256=layout_sha
    )
    groups = build_train_query_groups(layout=layout, targets=geometry_targets)
    exact_by_query = build_exact_identity_query_targets(groups=groups, identity_targets=identity_targets)
    static_geometry_groups = build_hard_repeat_query_targets(
        layout=layout,
        targets=geometry_targets,
        hard_repeat_targets=load_candidate_pose_rgb_spatial_hard_repeat_targets(paths["static_hard"]),
        layout_sha256=layout_sha,
        targets_sha256=geometry_sha,
    )
    static_groups, static_filter = filter_static_hard_repeat_groups_to_registered_exact_identity(
        static_groups=static_geometry_groups, registered_identity_targets=identity_targets
    )

    sources = load_context_attention_sources(
        radio_final_context_cache=source_paths["radio_final"],
        radio_intermediate_context_cache=source_paths["radio_intermediate"],
        alike_spatial_context_cache=source_paths["alike"],
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    image_ids, image_sizes, source_grids = _source_table(sources)
    source_manifest = str(sources[0].metadata.get("source_image_manifest_sha256", ""))
    if not source_manifest:
        raise ValueError("phase identity source image manifest is absent")
    checkpoint = _load_checkpoint(paths["checkpoint"])
    model_config, lineage = _validate_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=paths["checkpoint"],
        layout_path=paths["layout"],
        layout_metadata=layout.metadata,
        source_paths=source_paths,
        source_manifest_sha256=source_manifest,
    )
    partition, heldout_query_ids = _checkpoint_partition(
        checkpoint, all_train_query_ids=tuple(sorted(groups))
    )
    if not set(heldout_query_ids).issubset(static_groups):
        raise ValueError("phase identity re-audit static targets miss a held-out query")
    runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
    model = _load_model(
        checkpoint=checkpoint,
        source_grids=source_grids,
        image_sizes=image_sizes,
        device=device,
    )
    training_args = checkpoint.get("training_args")
    if not isinstance(training_args, Mapping):
        raise ValueError("phase identity checkpoint training arguments are invalid")
    gate_thresholds = {
        "minimum_eligible_query_fraction": float(
            training_args.get("gate_minimum_eligible_query_fraction", 0.90)
        ),
        "minimum_win_fraction": float(training_args.get("gate_minimum_win_fraction", 0.55)),
        "minimum_gap": float(training_args.get("gate_minimum_gap", 0.05)),
        "minimum_visual_gap_delta": float(
            training_args.get("gate_minimum_visual_gap_delta", 0.05)
        ),
    }
    support_shift = int(training_args.get("support_permutation_shift", 1))
    gate = evaluate_static_hard_gate(
        model=model,
        groups=groups,
        exact_by_query=exact_by_query,
        static_hard_by_query=static_groups,
        complete_runtime=runtime,
        query_ids=heldout_query_ids,
        state=_DistributedState(
            rank=0, world_size=1, local_rank=0, device=device, enabled=False
        ),
        support_permutation_shift=support_shift,
        gate_thresholds=gate_thresholds,
        static_hard_semantics=static_filter,
    )
    promotable = promotable_phase_identity_sources(gate)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "format": AUDIT_FORMAT,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256_short(paths["checkpoint"]),
        "model_config": model_config,
        "checkpoint_lineage": lineage,
        "checkpoint_train_query_partition": partition,
        "heldout_train_query_count": len(heldout_query_ids),
        "heldout_train_query_ids": list(heldout_query_ids),
        "gate": gate,
        "promotable_source_names": list(promotable),
        "promotion": {
            "allowed": bool(promotable or bool(gate["combined_passed"])),
            "allowed_next_stage": "frozen_topl_pose_rank_audit_only",
            "pnp_integration_allowed": False,
            "requires_explicit_source_name_when_combined_fails": True,
        },
        "protocol": {
            "target_free_visual_forward": True,
            "registered_identity_and_static_hard_labels_joined_after_forward": True,
            "registered_exact_identity_filter": static_filter,
            "support_appearance_control": "distant_point_block_derangement_common_availability_v2",
            "render": False,
            "image_retrieval_or_submap": False,
        },
        "inputs": {
            **{name: {"path": str(path), "sha256": file_sha256_short(path)} for name, path in paths.items()},
            "source_caches": {
                name: {"path": str(path), "sha256": file_sha256_short(path)}
                for name, path in source_paths.items()
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(summary_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = audit_candidate_multiscale_phase_identity_llr(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
