"""Audit a broad identity-LLR checkpoint with the current strict visual gate.

The observation-pair checkpoint is trained only with train-query SfM labels,
but its runtime scorer is target-free.  This command keeps that boundary: it
uses train-only labels only after inference to measure whether a checkpoint's
same-track candidate score itself falls under support derangement and
position-only controls.  It never evaluates pose, validation/test images, or
PnP.
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

from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT,
    OBSERVATION_IDENTITY_GATE_VERSION,
    _evaluate_inner_validation,
    _hard_pose_identity_mask_lookup,
    _identity_pretrain_gate,
    _limited_rows,
)
from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_likelihood import (
    _discover_rgb_image_size,
    _source_table,
    _validate_rgb_coordinate_bridge,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_GATE_VERSION,
    _write_json_atomically,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CandidatePoseRGBSpatialIdentityLLR,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_identity import (
    load_candidate_pose_rgb_spatial_hard_pose_identity_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    load_candidate_pose_rgb_spatial_observation_pairs,
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observation-pairs", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--hard-pose-identity-targets",
        default="",
        help=(
            "Optional train-only coherent-wrong sidecar. It is joined only after "
            "target-free scoring to report conditional source-gain hard-pose metrics."
        ),
    )
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-margin", type=float, default=0.05)
    parser.add_argument("--minimum-support-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-position-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-support-correct-score-gap", type=float, default=0.05)
    parser.add_argument("--minimum-position-correct-score-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-eligible-query-fraction", type=float, default=0.9)
    parser.add_argument("--minimum-conditional-visual-ablation-margin", type=float, default=0.05)
    parser.add_argument(
        "--minimum-conditional-visual-ablation-win-fraction", type=float, default=0.55
    )
    parser.add_argument(
        "--minimum-conditional-visual-ablation-top1-delta", type=float, default=0.0
    )
    parser.add_argument(
        "--minimum-conditional-visual-ablation-hard-pose-gap", type=float, default=0.05
    )
    parser.add_argument(
        "--minimum-conditional-visual-ablation-hard-pose-win-fraction",
        type=float,
        default=0.55,
    )
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], Mapping[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("identity observation checkpoint is malformed")
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or not isinstance(state_dict, Mapping)
        or not isinstance(metadata, Mapping)
        or metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
    ):
        raise ValueError("identity observation checkpoint format is incompatible")
    return dict(state_dict), metadata


def audit_candidate_pose_rgb_spatial_identity_llr_observation_pretrain(
    args: argparse.Namespace,
) -> dict[str, object]:
    checkpoint = Path(args.checkpoint)
    pair_path = Path(args.observation_pairs)
    output = Path(args.output_json)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"identity observation checkpoint is absent: {checkpoint}")
    if int(args.max_validation_rows) < 0 or float(args.rgb_cache_gb) <= 0.0:
        raise ValueError("identity observation audit arguments are invalid")
    state_dict, metadata = _load_checkpoint(checkpoint)
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("identity observation checkpoint lacks model configuration")
    pairs = load_candidate_pose_rgb_spatial_observation_pairs(pair_path)
    hard_pose_target_path = (
        Path(str(args.hard_pose_identity_targets))
        if str(args.hard_pose_identity_targets).strip()
        else None
    )
    hard_pose_lookup = None
    if hard_pose_target_path is not None:
        hard_pose_lookup = _hard_pose_identity_mask_lookup(
            pairs=pairs,
            targets=load_candidate_pose_rgb_spatial_hard_pose_identity_targets(
                hard_pose_target_path
            ),
            pair_path=pair_path,
        )
    if int(metadata.get("fixed_candidate_count", -1)) != int(pairs.negative_count + 1):
        raise ValueError("identity observation checkpoint candidate count differs from audit pairs")
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
        raise ValueError("identity observation audit requires common context image dimensions")
    coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
    rgb_image_size = _discover_rgb_image_size(
        image_root=Path(args.image_root), image_id=str(image_ids[0])
    )
    bridge = _validate_rgb_coordinate_bridge(
        source_metadata=sources[0].metadata,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    checkpoint_lineage = metadata.get("lineage")
    source_lineage = {
        "radio_final_context_cache_sha256": file_sha256_short(
            Path(args.radio_final_context_cache)
        ),
        "radio_intermediate_context_cache_sha256": file_sha256_short(
            Path(args.radio_intermediate_context_cache)
        ),
        "alike_spatial_context_cache_sha256": file_sha256_short(
            Path(args.alike_spatial_context_cache)
        ),
        "source_image_manifest_sha256": str(
            sources[0].metadata.get("source_image_manifest_sha256", "")
        ),
        "rgb_coordinate_bridge": bridge,
    }
    if (
        not isinstance(checkpoint_lineage, Mapping)
        or any(checkpoint_lineage.get(name) != value for name, value in source_lineage.items())
    ):
        raise ValueError("identity observation checkpoint lineage differs from audit sources")
    image_index_by_id = {str(image_id): index for index, image_id in enumerate(image_ids.tolist())}
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
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("identity observation checkpoint state is incompatible") from error
    device = torch.device(str(args.device))
    model = model.to(device).eval()
    rows = _limited_rows(
        np.flatnonzero(pairs.split_names == "inner_validation"),
        limit=int(args.max_validation_rows),
        seed=int(args.seed) + 29,
    )
    cache = TensorImageLRUCache(
        max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
        storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
    )
    metrics = _evaluate_inner_validation(
        model=model,
        pairs=pairs,
        rows=rows,
        image_index_by_id=image_index_by_id,
        image_ids=image_ids,
        image_root=Path(args.image_root),
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        radius_px=float(config["rgb_context_radius_px"]),
        step_px=float(config["rgb_step_px"]),
        batch_size=64,
        seed=int(args.seed),
        cache=cache,
        device=device,
        amp_enabled=device.type == "cuda" and not bool(args.no_amp),
        hard_pose_lookup=hard_pose_lookup,
    )
    gate = _identity_pretrain_gate(
        metrics=metrics,
        args=args,
        hard_pose_enabled=hard_pose_lookup is not None,
    )
    eligible = bool(
        gate["passed"]
        and metadata.get("visual_evidence_gate_version")
        == REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_GATE_VERSION
        and OBSERVATION_IDENTITY_GATE_VERSION
        == REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_GATE_VERSION
    )
    result: dict[str, object] = {
        "stage": "audit_candidate_pose_rgb_spatial_identity_llr_observation_pretrain",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256_short(checkpoint),
        "checkpoint_visual_evidence_gate_version": metadata.get("visual_evidence_gate_version"),
        "current_visual_evidence_gate_version": OBSERVATION_IDENTITY_GATE_VERSION,
        "inner_validation": {"row_count": int(len(rows)), "metrics": metrics, "gate": gate},
        "p1_initialization_eligible": eligible,
        "rgb_cache": cache.summary(),
        "protocol": {
            "train_only_observation_targets": True,
            "runtime_scorer_target_free": True,
            "model_weights_updated": False,
            "pose_or_pnp_not_run": True,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
            "raw_scores_must_not_feed_pnp": True,
        },
        "lineage": {
            "observation_pairs_sha256": file_sha256_short(pair_path),
            "hard_pose_identity_targets_sha256": (
                ""
                if hard_pose_target_path is None
                else file_sha256_short(hard_pose_target_path)
            ),
            **source_lineage,
        },
    }
    _write_json_atomically(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = audit_candidate_pose_rgb_spatial_identity_llr_observation_pretrain(parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
