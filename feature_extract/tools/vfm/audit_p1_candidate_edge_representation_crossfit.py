"""Audit frozen current-P1 candidate-edge representations with OOF probes.

This is deliberately a train-only *representation* diagnostic, not a pose
evaluation or a checkpoint trainer.  For each immutable current-P1 query it
first exports target-free candidate/support-view features from the frozen V3
identity encoder:

* full 2-D RADIO final context;
* full 2-D RADIO intermediate context;
* ALIKE spatial-context factors; and
* high-resolution RGB FPN pair features.

Only after those visual forwards are stored does rank zero join the existing
coherent-repeat training targets.  It then fits a fixed-capacity, query-grouped
OOF pairwise linear probe per predeclared source family.  The support-view
permutation and all-visual-content-zero controls use the exact same held-query
probe and target rows.  The output is runtime-ineligible by construction and
must not be used for PnP or validation/test scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_candidate_pose_rgb_spatial_identity_llr_sources import (
    _validate_checkpoint_source_lineage,
)
from feature_extract.tools.vfm.audit_p1_rgb_radio_bridge_crossfit import (
    _load_radio_final_identity_checkpoint,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    _write_json_atomically,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _source_table,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES,
    CandidatePoseRGBSpatialIdentityLLR,
    CandidatePoseRGBSpatialIdentityLLREdgeRepresentation,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    permute_runtime_support_appearance,
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


AUDIT_FORMAT = "p1_candidate_edge_representation_crossfit_audit_v1"
TRAINING_FEATURE_FORMAT = "p1_candidate_edge_representation_training_feature_v1"
FEATURE_SOURCES = (
    "radio_final",
    "radio_intermediate",
    "alike",
    "rgb_fpn",
    "candidate_relative",
)
PROFILE_SOURCES: Mapping[str, tuple[str, ...]] = {
    "radio_final": ("radio_final",),
    "radio_intermediate": ("radio_intermediate",),
    "alike": ("alike",),
    "rgb_fpn": ("rgb_fpn",),
    "radio_final_intermediate": ("radio_final", "radio_intermediate"),
    "all_raw_multiscale": ("radio_final", "radio_intermediate", "alike", "rgb_fpn"),
    # This is not a new source.  It exposes the frozen encoder's existing
    # candidate-set-relative representation to determine whether a simple
    # linear separator can recover information that its scalar MLP missed.
    "candidate_relative": ("candidate_relative",),
}
_ZERO_VISUAL_SCALES = {
    name: 0.0 for name in CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES
}


@dataclass(frozen=True)
class CandidateEdgeProbeQueryFeatures:
    """One query's target-free frozen candidate-edge features.

    The target-free feature artifact contains neither hard-repeat labels nor
    poses/projection offsets.  ``source_point_ids`` and fixed support weights
    are retained only to make a later train-only target join unambiguous.
    """

    query_id: str
    source_point_ids: np.ndarray
    candidate_view_weights: np.ndarray
    normal_features: Mapping[str, np.ndarray]
    permuted_features: Mapping[str, np.ndarray]
    position_features: Mapping[str, np.ndarray]
    normal_context_usable: np.ndarray
    normal_rgb_usable: np.ndarray
    normal_edge_usable: np.ndarray
    permuted_context_usable: np.ndarray
    permuted_rgb_usable: np.ndarray
    permuted_edge_usable: np.ndarray
    position_context_usable: np.ndarray
    position_rgb_usable: np.ndarray
    position_edge_usable: np.ndarray

    def __post_init__(self) -> None:
        query_id = str(self.query_id)
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        weights = np.asarray(self.candidate_view_weights, dtype=np.float32)
        normal = {str(name): np.asarray(value, dtype=np.float32) for name, value in self.normal_features.items()}
        permuted = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in self.permuted_features.items()
        }
        position = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in self.position_features.items()
        }
        masks = {
            "normal_context": np.asarray(self.normal_context_usable, dtype=bool),
            "normal_rgb": np.asarray(self.normal_rgb_usable, dtype=bool),
            "normal_edge": np.asarray(self.normal_edge_usable, dtype=bool),
            "permuted_context": np.asarray(self.permuted_context_usable, dtype=bool),
            "permuted_rgb": np.asarray(self.permuted_rgb_usable, dtype=bool),
            "permuted_edge": np.asarray(self.permuted_edge_usable, dtype=bool),
            "position_context": np.asarray(self.position_context_usable, dtype=bool),
            "position_rgb": np.asarray(self.position_rgb_usable, dtype=bool),
            "position_edge": np.asarray(self.position_edge_usable, dtype=bool),
        }
        base_shape = weights.shape
        valid_features = (
            set(normal) == set(FEATURE_SOURCES)
            and set(permuted) == set(FEATURE_SOURCES)
            and set(position) == set(FEATURE_SOURCES)
            and all(
                values.shape[:3] == base_shape
                and values.ndim == 4
                and values.shape[3] > 0
                and np.isfinite(values).all()
                for collection in (normal, permuted, position)
                for values in collection.values()
            )
        )
        if (
            not query_id
            or weights.ndim != 3
            or weights.shape[0] != len(source_ids)
            or len(source_ids) == 0
            or len(np.unique(source_ids)) != len(source_ids)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not valid_features
            or any(mask.shape != base_shape for mask in masks.values())
            or any(np.any(rgb & ~edge) for rgb, edge in (
                (masks["normal_rgb"], masks["normal_edge"]),
                (masks["permuted_rgb"], masks["permuted_edge"]),
                (masks["position_rgb"], masks["position_edge"]),
            ))
            or any(np.any(context & ~edge) for context, edge in (
                (masks["normal_context"], masks["normal_edge"]),
                (masks["permuted_context"], masks["permuted_edge"]),
                (masks["position_context"], masks["position_edge"]),
            ))
            or any(
                not np.array_equal(edge, context | rgb)
                for context, rgb, edge in (
                    (masks["normal_context"], masks["normal_rgb"], masks["normal_edge"]),
                    (masks["permuted_context"], masks["permuted_rgb"], masks["permuted_edge"]),
                    (masks["position_context"], masks["position_rgb"], masks["position_edge"]),
                )
            )
        ):
            raise ValueError("candidate-edge probe query features are invalid")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "candidate_view_weights", weights)
        object.__setattr__(self, "normal_features", normal)
        object.__setattr__(self, "permuted_features", permuted)
        object.__setattr__(self, "position_features", position)
        for name, value in masks.items():
            object.__setattr__(self, name + "_usable", value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-checkpoint", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crossfit-fold-count", type=int, default=5)
    parser.add_argument("--permutation-control-shift", type=int, default=1)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--probe-diagonal-ridge-lambda", type=float, default=0.1)
    parser.add_argument("--minimum-points-per-pose", type=int, default=4)
    parser.add_argument("--minimum-eligible-query-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--catastrophic-gap-threshold", type=float, default=-0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--reuse-training-features",
        action="store_true",
        help="Reuse complete, lineage-matched target-free feature shards instead of rerunning visual forwards.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _safe_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _feature_filename(query_id: str) -> str:
    value = str(query_id)
    if not value:
        raise ValueError("candidate-edge probe query ID is empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20] + ".npz"


def _save_npz_atomically(path: Path, **arrays: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)


def _load_json_scalar(data: Mapping[str, np.ndarray], field: str) -> Mapping[str, object]:
    try:
        value = json.loads(str(np.asarray(data[field]).item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("candidate-edge probe feature metadata is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError("candidate-edge probe feature metadata is not an object")
    return dict(value)


def _feature_metadata(*, query_id: str, lineage: Mapping[str, object]) -> dict[str, object]:
    return {
        "format": TRAINING_FEATURE_FORMAT,
        "query_id": str(query_id),
        "contains_target_fields": False,
        "runtime_scorer_must_not_load_this_artifact": True,
        "target_join_after_visual_inference": True,
        "pose_or_residual_serialized": False,
        "no_render": True,
        "no_image_retrieval_or_submap": True,
        "lineage": dict(lineage),
    }


def _representation_feature_mapping(
    representation: CandidatePoseRGBSpatialIdentityLLREdgeRepresentation,
) -> dict[str, np.ndarray]:
    return {
        "radio_final": representation.context_source_features["radio_final"].detach().cpu().numpy(),
        "radio_intermediate": representation.context_source_features["radio_intermediate"]
        .detach()
        .cpu()
        .numpy(),
        "alike": representation.context_source_features["alike"].detach().cpu().numpy(),
        "rgb_fpn": representation.rgb_pair_features.detach().cpu().numpy(),
        "candidate_relative": representation.relative_features.detach().cpu().numpy(),
    }


def _write_training_feature(
    *,
    path: Path,
    features: CandidateEdgeProbeQueryFeatures,
    lineage: Mapping[str, object],
) -> None:
    arrays: dict[str, object] = {
        "source_point_ids": np.asarray(features.source_point_ids, dtype=np.int64),
        "candidate_view_weights": np.asarray(features.candidate_view_weights, dtype=np.float32),
        "normal_context_usable": np.asarray(features.normal_context_usable, dtype=bool),
        "normal_rgb_usable": np.asarray(features.normal_rgb_usable, dtype=bool),
        "normal_edge_usable": np.asarray(features.normal_edge_usable, dtype=bool),
        "permuted_context_usable": np.asarray(features.permuted_context_usable, dtype=bool),
        "permuted_rgb_usable": np.asarray(features.permuted_rgb_usable, dtype=bool),
        "permuted_edge_usable": np.asarray(features.permuted_edge_usable, dtype=bool),
        "position_context_usable": np.asarray(features.position_context_usable, dtype=bool),
        "position_rgb_usable": np.asarray(features.position_rgb_usable, dtype=bool),
        "position_edge_usable": np.asarray(features.position_edge_usable, dtype=bool),
        "metadata_json": np.asarray(_safe_json(_feature_metadata(
            query_id=features.query_id, lineage=lineage
        ))),
    }
    for branch, values in (
        ("normal", features.normal_features),
        ("permuted", features.permuted_features),
        ("position", features.position_features),
    ):
        for name in FEATURE_SOURCES:
            arrays[f"{branch}_{name}"] = np.asarray(values[name], dtype=np.float16)
    _save_npz_atomically(path, **arrays)


def _load_training_feature(
    *, path: Path, expected_query_id: str, expected_lineage: Mapping[str, object]
) -> CandidateEdgeProbeQueryFeatures:
    required = {
        "source_point_ids",
        "candidate_view_weights",
        "normal_context_usable",
        "normal_rgb_usable",
        "normal_edge_usable",
        "permuted_context_usable",
        "permuted_rgb_usable",
        "permuted_edge_usable",
        "position_context_usable",
        "position_rgb_usable",
        "position_edge_usable",
        "metadata_json",
        *(f"{branch}_{name}" for branch in ("normal", "permuted", "position") for name in FEATURE_SOURCES),
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"candidate-edge probe feature lacks {sorted(missing)}")
        metadata = _load_json_scalar(data, "metadata_json")
        if (
            metadata.get("format") != TRAINING_FEATURE_FORMAT
            or metadata.get("query_id") != str(expected_query_id)
            or metadata.get("contains_target_fields") is not False
            or metadata.get("runtime_scorer_must_not_load_this_artifact") is not True
            or metadata.get("target_join_after_visual_inference") is not True
            or metadata.get("pose_or_residual_serialized") is not False
            or metadata.get("no_render") is not True
            or metadata.get("no_image_retrieval_or_submap") is not True
            or metadata.get("lineage") != dict(expected_lineage)
        ):
            raise ValueError("candidate-edge probe feature contract differs from this audit")
        return CandidateEdgeProbeQueryFeatures(
            query_id=str(expected_query_id),
            source_point_ids=np.asarray(data["source_point_ids"], dtype=np.int64),
            candidate_view_weights=np.asarray(data["candidate_view_weights"], dtype=np.float32),
            normal_features={
                name: np.asarray(data[f"normal_{name}"], dtype=np.float32)
                for name in FEATURE_SOURCES
            },
            permuted_features={
                name: np.asarray(data[f"permuted_{name}"], dtype=np.float32)
                for name in FEATURE_SOURCES
            },
            position_features={
                name: np.asarray(data[f"position_{name}"], dtype=np.float32)
                for name in FEATURE_SOURCES
            },
            normal_context_usable=np.asarray(data["normal_context_usable"], dtype=bool),
            normal_rgb_usable=np.asarray(data["normal_rgb_usable"], dtype=bool),
            normal_edge_usable=np.asarray(data["normal_edge_usable"], dtype=bool),
            permuted_context_usable=np.asarray(data["permuted_context_usable"], dtype=bool),
            permuted_rgb_usable=np.asarray(data["permuted_rgb_usable"], dtype=bool),
            permuted_edge_usable=np.asarray(data["permuted_edge_usable"], dtype=bool),
            position_context_usable=np.asarray(data["position_context_usable"], dtype=bool),
            position_rgb_usable=np.asarray(data["position_rgb_usable"], dtype=bool),
            position_edge_usable=np.asarray(data["position_edge_usable"], dtype=bool),
        )


def _profile_feature_and_usable(
    *, query: CandidateEdgeProbeQueryFeatures, profile: str, branch: str
) -> tuple[np.ndarray, np.ndarray]:
    names = PROFILE_SOURCES.get(str(profile))
    if names is None:
        raise ValueError("candidate-edge probe profile is invalid")
    values = {
        "normal": query.normal_features,
        "permuted": query.permuted_features,
        "position": query.position_features,
    }.get(str(branch))
    if values is None:
        raise ValueError("candidate-edge probe branch is invalid")
    prefix = str(branch)
    context = getattr(query, f"{prefix}_context_usable")
    rgb = getattr(query, f"{prefix}_rgb_usable")
    edge = getattr(query, f"{prefix}_edge_usable")
    if all(name in {"radio_final", "radio_intermediate", "alike"} for name in names):
        usable = context
    elif names == ("rgb_fpn",):
        usable = rgb
    elif names == ("candidate_relative",):
        usable = edge
    else:
        usable = edge
    return np.concatenate([values[name] for name in names], axis=3), np.asarray(usable, dtype=bool)


def aggregate_fixed_support_view_features(
    *, features: np.ndarray, candidate_view_weights: np.ndarray, usable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate fixed support slots without moving missing-view mass.

    A missing support observation contributes a neutral zero vector rather
    than causing the remaining view to be renormalized.  This matches the
    runtime marginalization contract and prevents crop availability from
    becoming an implicit confidence bonus.
    """

    values = np.asarray(features, dtype=np.float32)
    weights = np.asarray(candidate_view_weights, dtype=np.float32)
    valid = np.asarray(usable, dtype=bool)
    if (
        values.ndim != 4
        or values.shape[:3] != weights.shape
        or valid.shape != weights.shape
        or values.shape[3] == 0
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("candidate-edge probe support aggregation inputs are invalid")
    weighted = weights[..., None] * valid[..., None].astype(np.float32)
    # Keep availability at [point, candidate].  Retaining the singleton
    # feature axis here would silently broadcast into target-edge indexing and
    # make a candidate appear active for every feature dimension.
    available_mass = np.sum(weighted, axis=2, dtype=np.float32)[..., 0]
    return np.sum(weighted * values, axis=2, dtype=np.float32), available_mass > 0.0


def _pair_feature_differences(
    *,
    query: CandidateEdgeProbeQueryFeatures,
    targets: HardRepeatQueryTargets,
    profile: str,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    if str(targets.query_id) != str(query.query_id):
        raise ValueError("candidate-edge probe query and target IDs differ")
    features, usable = _profile_feature_and_usable(query=query, profile=profile, branch=branch)
    candidate, available = aggregate_fixed_support_view_features(
        features=features,
        candidate_view_weights=query.candidate_view_weights,
        usable=usable,
    )
    source_to_row = {int(source_id): index for index, source_id in enumerate(query.source_point_ids)}
    try:
        point_indices = np.asarray(
            [source_to_row[int(source_id)] for source_id in targets.source_point_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("hard-repeat target source point is absent from frozen query features") from error
    positive = np.asarray(targets.positive_candidate_indices, dtype=np.int64)
    negative = np.asarray(targets.negative_candidate_indices, dtype=np.int64)
    if (
        np.any(positive < 0)
        or np.any(negative < 0)
        or np.any(positive >= candidate.shape[1])
        or np.any(negative >= candidate.shape[1])
    ):
        raise ValueError("hard-repeat target candidate index exceeds frozen features")
    diff = candidate[point_indices, positive] - candidate[point_indices, negative]
    active = available[point_indices, positive] & available[point_indices, negative]
    if diff.shape != (len(point_indices), candidate.shape[2]) or not np.isfinite(diff).all():
        raise RuntimeError("candidate-edge pair features are invalid")
    return diff, active


@dataclass(frozen=True)
class DiagonalPairwiseLinearProbe:
    """A fixed-capacity diagonal ridge separator for pairwise edge features.

    It is the diagonal approximation to an L2 linear discriminant trained on
    the symmetric ``[x_pos - x_neg, x_neg - x_pos]`` dataset.  Unlike a dual
    logistic solver, it is O(ND), so the diagnostic scales to every mined
    coherent-repeat edge without changing the held-query protocol.
    """

    coefficients: np.ndarray
    second_moment: np.ndarray
    ridge_lambda: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).reshape(-1)
        second = np.asarray(self.second_moment, dtype=np.float64).reshape(-1)
        ridge = float(self.ridge_lambda)
        if (
            len(coefficients) == 0
            or coefficients.shape != second.shape
            or not np.isfinite(coefficients).all()
            or not np.isfinite(second).all()
            or np.any(second < 0.0)
            or not math.isfinite(ridge)
            or ridge <= 0.0
            or float(np.linalg.norm(coefficients)) <= 0.0
        ):
            raise ValueError("candidate-edge diagonal probe is invalid")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "second_moment", second)
        object.__setattr__(self, "ridge_lambda", ridge)


def _fit_pairwise_linear_probe(
    *, features: np.ndarray, ridge_lambda: float
) -> DiagonalPairwiseLinearProbe:
    values = np.asarray(features, dtype=np.float64)
    ridge = float(ridge_lambda)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] == 0
        or not np.isfinite(values).all()
        or not math.isfinite(ridge)
        or ridge <= 0.0
    ):
        raise ValueError("candidate-edge pairwise probe fitting inputs are invalid")
    # Pairwise orientation is the only label: x_pos - x_neg is positive and
    # its exact negation is negative.  The diagonal normal equation is
    # therefore proportional to mean(x) / (mean(x^2) + lambda).  No candidate
    # index, rank, coarse score, pose, or target field enters the feature.
    mean = np.mean(values, axis=0)
    second = np.mean(np.square(values), axis=0)
    coefficients = mean / (second + ridge)
    coefficient_norm = float(np.linalg.norm(coefficients))
    if not math.isfinite(coefficient_norm) or coefficient_norm <= 1e-12:
        raise RuntimeError("candidate-edge diagonal probe found no train-fold direction")
    # Unit-norm weights make OOF margins comparable across folds while leaving
    # their ordering unchanged within each held query.
    coefficients = coefficients / coefficient_norm
    return DiagonalPairwiseLinearProbe(
        coefficients=coefficients, second_moment=second, ridge_lambda=ridge
    )


def _probe_margin(
    *, probe: DiagonalPairwiseLinearProbe, features: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(probe.coefficients)
        or not np.isfinite(values).all()
    ):
        raise ValueError("candidate-edge probe score inputs are invalid")
    margin = np.asarray(values @ probe.coefficients, dtype=np.float64).reshape(-1)
    if margin.shape != (len(values),) or not np.isfinite(margin).all():
        raise RuntimeError("candidate-edge probe score is invalid")
    return margin


def aggregate_hard_pose_group_gaps(
    *,
    normal_margins: np.ndarray,
    permuted_margins: np.ndarray,
    position_margins: np.ndarray,
    common_active: np.ndarray,
    targets: HardRepeatQueryTargets,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collapse multi-negative edges into coherent wrong-pose group gaps."""

    normal = np.asarray(normal_margins, dtype=np.float64).reshape(-1)
    permuted = np.asarray(permuted_margins, dtype=np.float64).reshape(-1)
    position = np.asarray(position_margins, dtype=np.float64).reshape(-1)
    active = np.asarray(common_active, dtype=bool).reshape(-1)
    required = int(minimum_points)
    count = len(targets.source_point_ids)
    if (
        normal.shape != (count,)
        or permuted.shape != (count,)
        or position.shape != (count,)
        or active.shape != (count,)
        or not np.isfinite(normal).all()
        or not np.isfinite(permuted).all()
        or not np.isfinite(position).all()
        or required < 2
    ):
        raise ValueError("candidate-edge coherent pose group inputs are invalid")
    normal_groups: list[float] = []
    permuted_groups: list[float] = []
    position_groups: list[float] = []
    active_points = 0
    pair_ids = np.asarray(targets.pair_ids, dtype=np.int64)
    source_ids = np.asarray(targets.source_point_ids, dtype=np.int64)
    for pair_id in np.unique(pair_ids):
        pair_rows = np.flatnonzero((pair_ids == int(pair_id)) & active)
        if not len(pair_rows):
            continue
        per_point_normal: list[float] = []
        per_point_permuted: list[float] = []
        per_point_position: list[float] = []
        for source_id in np.unique(source_ids[pair_rows]):
            rows = pair_rows[source_ids[pair_rows] == int(source_id)]
            # One exact positive can have several coherent wrong candidate
            # slots.  The group must defeat the strongest wrong identity, so
            # retain the smallest positive-minus-negative margin per point.
            selected = int(rows[np.argmin(normal[rows])])
            per_point_normal.append(float(normal[selected]))
            per_point_permuted.append(float(permuted[selected]))
            per_point_position.append(float(position[selected]))
        if len(per_point_normal) < required:
            continue
        normal_groups.append(float(np.mean(per_point_normal)))
        permuted_groups.append(float(np.mean(per_point_permuted)))
        position_groups.append(float(np.mean(per_point_position)))
        active_points += len(per_point_normal)
    return (
        np.asarray(normal_groups, dtype=np.float64),
        np.asarray(permuted_groups, dtype=np.float64),
        np.asarray(position_groups, dtype=np.float64),
        int(active_points),
    )


def _profile_summary(
    *, rows: Sequence[Mapping[str, object]], catastrophic_threshold: float
) -> dict[str, object]:
    eligible = [row for row in rows if bool(row.get("eligible", False))]
    if not eligible:
        return {
            "eligible_query_count": 0,
            "eligible_query_fraction": 0.0,
            "normal": None,
            "support_permuted_control": None,
            "position_only_control": None,
            "normal_minus_permuted_mean_gap": None,
            "normal_minus_position_mean_gap": None,
        }
    normal = np.asarray([float(row["normal_gap"]) for row in eligible], dtype=np.float64)
    permuted = np.asarray([float(row["permuted_gap"]) for row in eligible], dtype=np.float64)
    position = np.asarray([float(row["position_gap"]) for row in eligible], dtype=np.float64)

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_gap": float(values.mean()),
            "median_gap": float(np.median(values)),
            "p10_gap": float(np.quantile(values, 0.1)),
            "win_fraction": float(np.mean(values > 0.0)),
            "catastrophic_count": float(np.sum(values <= float(catastrophic_threshold))),
        }

    normal_minus_permuted = normal - permuted
    normal_minus_position = normal - position
    return {
        "eligible_query_count": int(len(eligible)),
        "normal": summarize(normal),
        "support_permuted_control": summarize(permuted),
        "position_only_control": summarize(position),
        "normal_minus_permuted_mean_gap": float(normal_minus_permuted.mean()),
        "normal_minus_permuted_median_gap": float(np.median(normal_minus_permuted)),
        "normal_minus_permuted_win_fraction": float(np.mean(normal_minus_permuted > 0.0)),
        "normal_minus_position_mean_gap": float(normal_minus_position.mean()),
        "normal_minus_position_median_gap": float(np.median(normal_minus_position)),
        "normal_minus_position_win_fraction": float(np.mean(normal_minus_position > 0.0)),
        "mean_coherent_pose_groups": float(np.mean([float(row["pose_group_count"]) for row in eligible])),
        "mean_active_points": float(np.mean([float(row["active_point_count"]) for row in eligible])),
    }


def _probe_gate(
    *, summary: Mapping[str, object], total_query_count: int, args: argparse.Namespace
) -> dict[str, object]:
    normal = summary.get("normal")
    permuted = summary.get("support_permuted_control")
    position = summary.get("position_only_control")
    if not isinstance(normal, Mapping) or not isinstance(permuted, Mapping) or not isinstance(position, Mapping):
        return {
            "passed": False,
            "checks": {"has_eligible_queries": False},
            "policy": "train-only raw representation probe; never a PnP promotion gate",
        }
    coverage = int(summary["eligible_query_count"]) / float(total_query_count)
    checks = {
        "eligible_query_coverage": coverage >= float(args.minimum_eligible_query_fraction),
        "normal_mean_gap": float(normal["mean_gap"]) >= float(args.minimum_normal_gap),
        "normal_win_fraction": float(normal["win_fraction"]) >= float(args.minimum_win_fraction),
        "support_permutation_visual_gap": float(summary["normal_minus_permuted_mean_gap"])
        >= float(args.minimum_visual_gap_delta),
        "position_only_visual_gap": float(summary["normal_minus_position_mean_gap"])
        >= float(args.minimum_visual_gap_delta),
        "normal_catastrophic_tail_not_worse_than_controls": float(normal["catastrophic_count"])
        <= min(float(permuted["catastrophic_count"]), float(position["catastrophic_count"])),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "eligible_query_fraction": float(coverage),
        "minimum_eligible_query_fraction": float(args.minimum_eligible_query_fraction),
        "minimum_normal_gap": float(args.minimum_normal_gap),
        "minimum_win_fraction": float(args.minimum_win_fraction),
        "minimum_visual_gap_delta": float(args.minimum_visual_gap_delta),
        "policy": "train-only raw representation probe; never a PnP promotion gate",
    }


def crossfit_profile(
    *,
    queries: Mapping[str, CandidateEdgeProbeQueryFeatures],
    hard_targets: Mapping[str, HardRepeatQueryTargets],
    profile: str,
    fold_count: int,
    ridge_lambda: float,
    minimum_points: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit one fixed linear pairwise probe per train fold and score held queries."""

    query_ids = tuple(sorted(queries))
    if (
        set(query_ids) != set(hard_targets)
        or int(fold_count) < 2
        or int(fold_count) > len(query_ids)
        or str(profile) not in PROFILE_SOURCES
    ):
        raise ValueError("candidate-edge probe cross-fit contract is invalid")
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    for fold in range(int(fold_count)):
        train_ids = [query_id for index, query_id in enumerate(query_ids) if index % fold_count != fold]
        held_ids = [query_id for index, query_id in enumerate(query_ids) if index % fold_count == fold]
        train_chunks: list[np.ndarray] = []
        for query_id in train_ids:
            normal, normal_active = _pair_feature_differences(
                query=queries[query_id], targets=hard_targets[query_id], profile=profile, branch="normal"
            )
            if bool(np.any(normal_active)):
                train_chunks.append(normal[normal_active])
        if not train_chunks:
            raise RuntimeError("candidate-edge probe train fold has no usable pair features")
        train_features = np.concatenate(train_chunks, axis=0)
        probe = _fit_pairwise_linear_probe(
            features=train_features, ridge_lambda=float(ridge_lambda)
        )
        fits.append(
            {
                "fold": int(fold),
                "train_query_count": int(len(train_ids)),
                "held_query_count": int(len(held_ids)),
                "train_pair_count": int(len(train_features)),
                "feature_dimension": int(train_features.shape[1]),
                "coefficient_l2": float(np.linalg.norm(probe.coefficients)),
                "diagonal_ridge_lambda": float(probe.ridge_lambda),
            }
        )
        for query_id in held_ids:
            query = queries[query_id]
            targets = hard_targets[query_id]
            normal, normal_active = _pair_feature_differences(
                query=query, targets=targets, profile=profile, branch="normal"
            )
            permuted, permuted_active = _pair_feature_differences(
                query=query, targets=targets, profile=profile, branch="permuted"
            )
            position, position_active = _pair_feature_differences(
                query=query, targets=targets, profile=profile, branch="position"
            )
            common_active = normal_active & permuted_active & position_active
            normal_margin = _probe_margin(probe=probe, features=normal)
            permuted_margin = _probe_margin(probe=probe, features=permuted)
            position_margin = _probe_margin(probe=probe, features=position)
            normal_groups, permuted_groups, position_groups, active_points = aggregate_hard_pose_group_gaps(
                normal_margins=normal_margin,
                permuted_margins=permuted_margin,
                position_margins=position_margin,
                common_active=common_active,
                targets=targets,
                minimum_points=int(minimum_points),
            )
            if len(normal_groups) == 0:
                rows.append(
                    {
                        "query_id": query_id,
                        "fold": int(fold),
                        "eligible": False,
                        "common_active_edge_count": int(np.sum(common_active)),
                        "pose_group_count": 0,
                        "active_point_count": 0,
                    }
                )
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "fold": int(fold),
                    "eligible": True,
                    "common_active_edge_count": int(np.sum(common_active)),
                    "pose_group_count": int(len(normal_groups)),
                    "active_point_count": int(active_points),
                    "normal_gap": float(np.mean(normal_groups)),
                    "permuted_gap": float(np.mean(permuted_groups)),
                    "position_gap": float(np.mean(position_groups)),
                }
            )
    rows.sort(key=lambda row: str(row["query_id"]))
    return rows, fits


def _source_lineage(
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
        "source_image_manifest_sha256": str(source_metadata.get("source_image_manifest_sha256", "")),
        "rgb_coordinate_bridge": dict(rgb_coordinate_bridge),
    }


def _feature_lineage(
    *,
    layout_path: Path,
    identity_checkpoint: Path,
    source_lineage: Mapping[str, object],
    layout: CandidatePoseRGBSpatialLayout,
) -> dict[str, object]:
    return {
        "layout_sha256": file_sha256_short(layout_path),
        "identity_checkpoint_sha256": file_sha256_short(identity_checkpoint),
        "source_lineage": dict(source_lineage),
        "candidate_count": int(layout.candidate_count),
        "support_view_count": int(layout.support_view_count),
        "feature_sources": list(FEATURE_SOURCES),
        "visual_content_controls": {
            "normal": "all_real_image_sources_v1",
            "support_permuted": "fixed_support_slot_derangement_v1",
            "position_only": "all_visual_content_scales_zero_v1",
        },
    }


@torch.inference_mode()
def _forward_query_target_free(
    *,
    group: object,
    complete_runtime: object,
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    model: CandidatePoseRGBSpatialIdentityLLR,
    cache: TensorImageLRUCache,
    device: torch.device,
    amp_enabled: bool,
    permutation_shift: int,
) -> CandidateEdgeProbeQueryFeatures:
    """Run the three visual controls before joining any train-only target."""

    from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import _slice_runtime

    query_id = str(getattr(group, "query_id"))
    layout_rows = np.asarray(getattr(group, "layout_rows"), dtype=np.int64).reshape(-1)
    source_point_ids = np.asarray(getattr(group, "source_point_ids"), dtype=np.int64).reshape(-1)
    if len(layout_rows) == 0 or source_point_ids.shape != layout_rows.shape:
        raise ValueError("candidate-edge probe frozen P1 query rows are invalid")
    runtime = _slice_runtime(complete_runtime, layout_rows)
    permuted_runtime = permute_runtime_support_appearance(
        runtime, shift=int(permutation_shift)
    )
    if (
        torch.equal(runtime.support_image_indices, permuted_runtime.support_image_indices)
        and torch.equal(runtime.support_xy, permuted_runtime.support_xy)
    ):
        raise ValueError("candidate-edge probe support permutation did not change ownership")
    query_rgb, support_rgb = _crop_runtime_rgb_patches(
        runtime=runtime,
        image_ids=image_ids,
        image_root=image_root,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        radius_px=float(model.rgb_context_radius_px),
        step_px=float(model.rgb_step_px),
        cache=cache,
        device=device,
    )
    permuted_support_rgb = permute_support_patch_appearance(
        runtime=permuted_runtime,
        support_patches=support_rgb,
        shift=int(permutation_shift),
    )
    with torch.cuda.amp.autocast(enabled=bool(amp_enabled)):
        normal = model.forward_edge_representation(
            runtime=runtime, query_rgb_patches=query_rgb, support_rgb_patches=support_rgb
        )
        permuted = model.forward_edge_representation(
            runtime=permuted_runtime,
            query_rgb_patches=query_rgb,
            support_rgb_patches=permuted_support_rgb,
        )
        position = model.forward_edge_representation(
            runtime=runtime,
            query_rgb_patches=query_rgb,
            support_rgb_patches=support_rgb,
            visual_source_scales=_ZERO_VISUAL_SCALES,
        )
    return CandidateEdgeProbeQueryFeatures(
        query_id=query_id,
        source_point_ids=source_point_ids,
        candidate_view_weights=runtime.candidate_view_weights.detach().cpu().numpy(),
        normal_features=_representation_feature_mapping(normal),
        permuted_features=_representation_feature_mapping(permuted),
        position_features=_representation_feature_mapping(position),
        normal_context_usable=normal.context_edge_usable.detach().cpu().numpy(),
        normal_rgb_usable=normal.rgb_edge_usable.detach().cpu().numpy(),
        normal_edge_usable=normal.edge_usable.detach().cpu().numpy(),
        permuted_context_usable=permuted.context_edge_usable.detach().cpu().numpy(),
        permuted_rgb_usable=permuted.rgb_edge_usable.detach().cpu().numpy(),
        permuted_edge_usable=permuted.edge_usable.detach().cpu().numpy(),
        position_context_usable=position.context_edge_usable.detach().cpu().numpy(),
        position_rgb_usable=position.rgb_edge_usable.detach().cpu().numpy(),
        position_edge_usable=position.edge_usable.detach().cpu().numpy(),
    )


def audit_p1_candidate_edge_representation_crossfit(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "identity_checkpoint": Path(args.identity_checkpoint),
        "layout": Path(args.rgb_spatial_layout),
        "targets": Path(args.training_targets),
        "hard_repeat": Path(args.hard_repeat_targets),
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_spatial_context_cache),
        "image_root": Path(args.image_root),
    }
    output_dir = Path(args.output_dir)
    values = (
        float(args.rgb_cache_gb),
        float(args.probe_diagonal_ridge_lambda),
        float(args.minimum_eligible_query_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_win_fraction),
        float(args.minimum_visual_gap_delta),
        float(args.catastrophic_gap_threshold),
    )
    if (
        any(not path.exists() for path in paths.values())
        or int(args.crossfit_fold_count) < 2
        or int(args.permutation_control_shift) == 0
        or int(args.minimum_points_per_pose) < 2
        or not all(math.isfinite(value) for value in values)
        or float(args.rgb_cache_gb) <= 0.0
        or float(args.probe_diagonal_ridge_lambda) <= 0.0
        or not 0.0 < float(args.minimum_eligible_query_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or not 0.0 < float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_visual_gap_delta) < 0.0
    ):
        raise ValueError("candidate-edge probe arguments are invalid")
    if output_dir.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite candidate-edge probe output: {output_dir}")
    state = _initialize_distributed(str(args.device))
    try:
        if state.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "training_features").mkdir(parents=True, exist_ok=True)
        if state.enabled:
            distributed.barrier()
        layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
        targets = load_candidate_pose_rgb_spatial_training_targets(paths["targets"])
        layout_sha256 = file_sha256_short(paths["layout"])
        targets_sha256 = file_sha256_short(paths["targets"])
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=layout_sha256
        )
        hard_repeat = load_candidate_pose_rgb_spatial_hard_repeat_targets(paths["hard_repeat"])
        hard_targets = build_hard_repeat_query_targets(
            layout=layout,
            targets=targets,
            hard_repeat_targets=hard_repeat,
            layout_sha256=layout_sha256,
            targets_sha256=targets_sha256,
        )
        groups = build_train_query_groups(layout=layout, targets=targets)
        query_ids = tuple(sorted(groups))
        if (
            set(query_ids) != set(hard_targets)
            or int(args.crossfit_fold_count) > len(query_ids)
            or any(int(group.point_count) != 32 for group in groups.values())
        ):
            raise ValueError("candidate-edge probe requires complete frozen 32-point hard-repeat queries")
        headers = load_context_attention_source_headers(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            expected_radio_checkpoint="",
        )
        sources = load_context_attention_sources(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        if not (
            np.array_equal(np.asarray(headers.image_ids).astype(str), image_ids)
            and np.array_equal(np.asarray(headers.image_sizes, dtype=np.int64), image_sizes)
        ):
            raise ValueError("candidate-edge probe source headers differ from feature maps")
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("candidate-edge probe requires common aligned source dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=paths["image_root"], image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        source_lineage = _source_lineage(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            source_metadata=sources[0].metadata,
            rgb_coordinate_bridge=rgb_bridge,
        )
        if not str(source_lineage["source_image_manifest_sha256"]):
            raise ValueError("candidate-edge probe source caches lack an image manifest hash")
        model, checkpoint_metadata, checkpoint_lineage_contract = _load_radio_final_identity_checkpoint(
            path=paths["identity_checkpoint"],
            source_tensors=source_tensors,
            image_sizes=image_sizes,
            layout=layout,
            source_lineage=source_lineage,
            device=state.device,
        )
        # Keep the strict lineage check explicit in this script as well.  The
        # helper above performs it, but this guards against future helper
        # relaxation silently accepting a stale projected context cache.
        checkpoint_lineage = checkpoint_metadata.get("lineage")
        if not isinstance(checkpoint_lineage, Mapping):
            raise ValueError("candidate-edge probe identity checkpoint lacks lineage")
        if _validate_checkpoint_source_lineage(
            checkpoint_lineage=checkpoint_lineage, source_lineage=source_lineage
        ) != checkpoint_lineage_contract:
            raise RuntimeError("candidate-edge probe identity lineage contract drifted")
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        feature_lineage = _feature_lineage(
            layout_path=paths["layout"],
            identity_checkpoint=paths["identity_checkpoint"],
            source_lineage=source_lineage,
            layout=layout,
        )
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        local_query_count = sum(
            index % state.world_size == state.rank for index in range(len(query_ids))
        )
        local_completed = 0
        if not bool(args.reuse_training_features):
            for index, query_id in enumerate(query_ids):
                if index % state.world_size != state.rank:
                    continue
                features = _forward_query_target_free(
                    group=groups[query_id],
                    complete_runtime=complete_runtime,
                    image_ids=image_ids,
                    image_root=paths["image_root"],
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    model=model,
                    cache=cache,
                    device=state.device,
                    amp_enabled=amp_enabled,
                    permutation_shift=int(args.permutation_control_shift),
                )
                _write_training_feature(
                    path=output_dir / "training_features" / _feature_filename(query_id),
                    features=features,
                    lineage=feature_lineage,
                )
                local_completed += 1
                if local_completed == 1 or local_completed % 4 == 0 or local_completed == local_query_count:
                    print(
                        json.dumps(
                            {
                                "stage": "frozen_candidate_edge_feature_shard",
                                "rank": int(state.rank),
                                "completed_queries": int(local_completed),
                                "local_query_count": int(local_query_count),
                                "query_id": str(query_id),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        if state.enabled:
            distributed.barrier()
        output: dict[str, object] = {
            "rank": int(state.rank),
            "output_dir": str(output_dir),
            "world_size": int(state.world_size),
        }
        if state.rank == 0:
            queries = {
                query_id: _load_training_feature(
                    path=output_dir / "training_features" / _feature_filename(query_id),
                    expected_query_id=query_id,
                    expected_lineage=feature_lineage,
                )
                for query_id in query_ids
            }
            profiles: dict[str, object] = {}
            for profile in PROFILE_SOURCES:
                rows, fits = crossfit_profile(
                    queries=queries,
                    hard_targets=hard_targets,
                    profile=profile,
                    fold_count=int(args.crossfit_fold_count),
                    ridge_lambda=float(args.probe_diagonal_ridge_lambda),
                    minimum_points=int(args.minimum_points_per_pose),
                )
                summary = _profile_summary(
                    rows=rows, catastrophic_threshold=float(args.catastrophic_gap_threshold)
                )
                profiles[profile] = {
                    "sources": list(PROFILE_SOURCES[profile]),
                    "fold_fits": fits,
                    "oof_rows": rows,
                    "summary": summary,
                    "gate": _probe_gate(
                        summary=summary, total_query_count=len(query_ids), args=args
                    ),
                }
            passed = [
                profile for profile, value in profiles.items() if bool(value["gate"]["passed"])
            ]
            result: dict[str, object] = {
                "format": AUDIT_FORMAT,
                "stage": "frozen_current_p1_candidate_edge_representation_query_grouped_crossfit",
                "output_dir": str(output_dir),
                "world_size": int(state.world_size),
                "query_count": int(len(query_ids)),
                "query_ids": list(query_ids),
                "feature_lineage": feature_lineage,
                "target_lineage": {
                    "training_targets_sha256": targets_sha256,
                    "hard_repeat_targets_sha256": file_sha256_short(paths["hard_repeat"]),
                    "target_join_after_visual_inference": True,
                },
                "checkpoint": {
                    "path": str(paths["identity_checkpoint"]),
                    "sha256": file_sha256_short(paths["identity_checkpoint"]),
                    "source_lineage_contract": checkpoint_lineage_contract,
                    "broad_pretrain_gate_passed": True,
                    "current_p1_scalar_head_promoted": False,
                },
                "probe_configuration": {
                    "crossfit_fold_count": int(args.crossfit_fold_count),
                    "predeclared_profiles": {name: list(values) for name, values in PROFILE_SOURCES.items()},
                    "pairwise_linear_probe": "symmetric_pos_minus_neg_diagonal_l2_ridge_v1",
                    "diagonal_ridge_lambda": float(args.probe_diagonal_ridge_lambda),
                    "minimum_points_per_coherent_pose": int(args.minimum_points_per_pose),
                    "selection_uses_train_folds_only": True,
                },
                "profiles": profiles,
                "passed_representation_profiles": passed,
                "rgb_cache_rank0": cache.summary(),
                "protocol": {
                    "diagnostic_only": True,
                    "model_weights_updated": False,
                    "runtime_layout_target_free": True,
                    "raw_feature_artifacts_contain_targets": False,
                    "target_join_after_visual_inference": True,
                    "train_query_only": True,
                    "heldout_validation_or_test_not_run": True,
                    "pnp_or_pose_estimation_run": False,
                    "runtime_scorer_must_not_load_feature_artifacts": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                },
            }
            _write_json_atomically(output_dir / "audit.json", result)
            output = {
                "rank": int(state.rank),
                "output_dir": str(output_dir),
                "passed_representation_profiles": passed,
            }
        return output
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    result = audit_p1_candidate_edge_representation_crossfit(parse_args(argv))
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
