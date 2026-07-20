"""Fixed-prior residual probes over frozen full-track appearance summaries.

The full-track S1 artifacts contain one all-observation aggregate per frozen
query/candidate edge.  They are not per-view tensors, so they must not be fed
through the maplet per-view residual model.  This module implements a separate
diagnostic model with deliberately narrow semantics:

* the global top-20 tracks are immutable;
* the input null probability and total candidate probability mass are exact
  invariants;
* a zero residual reproduces the frozen posterior exactly; and
* the linear residual may only redistribute the existing candidate mass.

It therefore tests conditional identity information in the real-image
all-observation summaries without recalibrating null, selecting views, or
scoring a pose.  It is explicitly not the final per-view S2 architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.full_track_support_view_probe import (
    FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    MonotonicTop1RelativeTop4Residual,
)


FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_MODEL_FORMAT = (
    "frozen_fulltrack_candidate_appearance_conditional_residual_model_v1"
)
FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT = (
    "frozen_fulltrack_candidate_appearance_conditional_residual_prediction_v1"
)
FROZEN_FULLTRACK_APPEARANCE_ARTIFACT_FORMAT = (
    FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT
)
SUMMARY_TOP4_BALANCED_ARCHITECTURE = (
    "monotonic_top1_relative_positive_uplift_summary_top4_tanh_bounded_traincal_balanced"
)
SUMMARY_TOP4_FEATURE_GRANULARITY = (
    "candidate_summary_uniform_top4_real_observation_ncc_v1"
)


@dataclass(frozen=True)
class FulltrackResidualFamily:
    """A predeclared low-capacity residual and its train-only loss policy."""

    feature_prefixes: tuple[str, ...]
    rank2_hard_pair_weight: float
    input_mode: str = "absolute_with_availability_flags"
    summary_kind: str = "local_context_summary"

    def __post_init__(self) -> None:
        prefixes = tuple(str(value) for value in self.feature_prefixes)
        if (
            not prefixes
            or len(set(prefixes)) != len(prefixes)
            or any(not value for value in prefixes)
            or float(self.rank2_hard_pair_weight) < 0.0
            or self.input_mode
            not in {
                "absolute_with_availability_flags",
                "top1_relative_common_evidence",
            }
            or self.summary_kind
            not in {"local_context_summary", "global_context_summary"}
        ):
            raise ValueError("full-track residual family is invalid")
        object.__setattr__(self, "feature_prefixes", prefixes)


# These are fixed before validation is read.  The first three establish whether
# one source family carries conditional identity information.  The final probe
# is the predeclared hard-pair experiment motivated by rank-2-to-top-20 errors.
FROZEN_FULLTRACK_RESIDUAL_FAMILIES: Mapping[str, FulltrackResidualFamily] = {
    "fixedprior_fulltrack_alike_nll": FulltrackResidualFamily(
        feature_prefixes=("alike_",), rank2_hard_pair_weight=0.0
    ),
    "fixedprior_fulltrack_radio_nll": FulltrackResidualFamily(
        feature_prefixes=("radio_final_", "radio_intermediate_"),
        rank2_hard_pair_weight=0.0,
    ),
    "fixedprior_fulltrack_multiscale_nll": FulltrackResidualFamily(
        feature_prefixes=("radio_final_", "radio_intermediate_", "alike_"),
        rank2_hard_pair_weight=0.0,
    ),
    "fixedprior_fulltrack_multiscale_rank2hard": FulltrackResidualFamily(
        feature_prefixes=("radio_final_", "radio_intermediate_", "alike_"),
        rank2_hard_pair_weight=0.5,
    ),
    # The availability flags above are retained as a historical diagnostic.
    # These families enforce the correct missing-evidence semantics instead:
    # only a feature jointly visible in a challenger and the frozen rank-one
    # candidate may contribute a top1-relative identity residual.
    "fixedprior_fulltrack_top1relative_alike_nll": FulltrackResidualFamily(
        feature_prefixes=("alike_",),
        rank2_hard_pair_weight=0.0,
        input_mode="top1_relative_common_evidence",
    ),
    "fixedprior_fulltrack_top1relative_radio_nll": FulltrackResidualFamily(
        feature_prefixes=("radio_final_", "radio_intermediate_"),
        rank2_hard_pair_weight=0.0,
        input_mode="top1_relative_common_evidence",
    ),
    "fixedprior_fulltrack_top1relative_multiscale_nll": FulltrackResidualFamily(
        feature_prefixes=("radio_final_", "radio_intermediate_", "alike_"),
        rank2_hard_pair_weight=0.0,
        input_mode="top1_relative_common_evidence",
    ),
    "fixedprior_fulltrack_top1relative_multiscale_rank2hard": FulltrackResidualFamily(
        feature_prefixes=("radio_final_", "radio_intermediate_", "alike_"),
        rank2_hard_pair_weight=0.5,
        input_mode="top1_relative_common_evidence",
    ),
    # This larger absolute-context diagnostic is only enabled after local
    # layout evidence fails its held-out gate.  It compares a challenger with
    # the fixed rank-one candidate through their already-fixed real support
    # images; it cannot retrieve an image, change a support set, or alter the
    # input null mass.
    "fixedprior_fulltrack_globalcontext_top1relative_nll": FulltrackResidualFamily(
        feature_prefixes=("radio_final_global_", "radio_final_summary_"),
        rank2_hard_pair_weight=0.0,
        input_mode="top1_relative_common_evidence",
        summary_kind="global_context_summary",
    ),
}


@dataclass(frozen=True)
class FulltrackSummaryTop4Family:
    """A constrained all-observation top-four summary probe.

    This is deliberately separate from the historical linear summary residual:
    it consumes only an already-materialized ``uniform_top4_mean_ncc`` value
    for each named profile.  The model can boost a challenger only when that
    exact common observation summary exceeds the frozen top-one candidate;
    it never learns availability, candidate IDs, or null mass.
    """

    profile_names: tuple[str, ...]
    rank2_hard_pair_weight: float
    coarse_top1_stability_weight: float
    residual_scale: float
    residual_cap: float
    training_seed_key: str

    def __post_init__(self) -> None:
        profiles = tuple(str(value) for value in self.profile_names)
        if (
            not profiles
            or len(set(profiles)) != len(profiles)
            or any(not value for value in profiles)
            or float(self.rank2_hard_pair_weight) < 0.0
            or float(self.coarse_top1_stability_weight) <= 0.0
            or not np.isfinite(float(self.residual_scale))
            or float(self.residual_scale) <= 0.0
            or not np.isfinite(float(self.residual_cap))
            or float(self.residual_cap) <= 0.0
            or not str(self.training_seed_key).strip()
        ):
            raise ValueError("full-track summary top-four family is invalid")
        object.__setattr__(self, "profile_names", profiles)
        object.__setattr__(self, "training_seed_key", str(self.training_seed_key))


# This compares an aggregate candidate summary with the successful per-view
# raw-top4 objective while keeping the exported RADIO-intermediate PCA lineage
# fixed.  It is a candidate-summary diagnostic only: all profiles were exported
# from real SfM support observations before either train or validation
# identities are read, and no per-view S2 claim is made from this path.
FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES: Mapping[str, FulltrackSummaryTop4Family] = {
    "fixedprior_fulltrack_summarytop4_positive_uplift_multiscale_tanh_cap_traincal_balanced": FulltrackSummaryTop4Family(
        profile_names=(
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
        ),
        rank2_hard_pair_weight=8.0,
        coarse_top1_stability_weight=8.0,
        residual_scale=0.60,
        residual_cap=3.0,
        training_seed_key="fixedprior_fulltrack_rawtop4_positive_uplift_multiscale",
    ),
}


@dataclass(frozen=True)
class FrozenFulltrackAppearanceFeatures:
    """Aligned target-free full-track candidate summary rows."""

    paths: tuple[Path, ...]
    query_ids: np.ndarray
    split_names: np.ndarray
    source_row_indices: np.ndarray
    xy: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    candidate_summary_features: np.ndarray
    candidate_summary_feature_valid: np.ndarray
    feature_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    artifact_metadata: tuple[Mapping[str, Any], ...]
    compatibility: Mapping[str, Any]

    def __post_init__(self) -> None:
        paths = tuple(Path(path) for path in self.paths)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        split_names = np.asarray(self.split_names).astype(str).reshape(-1)
        source_rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        candidate = np.asarray(self.candidate_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        values = np.asarray(self.candidate_summary_features, dtype=np.float32)
        valid = np.asarray(self.candidate_summary_feature_valid, dtype=bool)
        feature_names = tuple(str(value) for value in self.feature_names)
        profile_names = tuple(str(value) for value in self.profile_names)
        metadata = tuple(dict(item) for item in self.artifact_metadata)
        count = len(query_ids)
        if (
            not paths
            or len(set(paths)) != len(paths)
            or len(metadata) != len(paths)
            or split_names.shape != (count,)
            or source_rows.shape != (count,)
            or xy.shape != (count, 2)
            or tracks.ndim != 2
            or tracks.shape[0] != count
            or candidate.shape != tracks.shape
            or null.shape != (count,)
            or values.shape[:2] != tracks.shape
            or valid.shape != values.shape
            or values.shape[2] != len(feature_names)
            or not feature_names
            or len(set(feature_names)) != len(feature_names)
            or not profile_names
            or len(set(profile_names)) != len(profile_names)
            or np.any(~np.isfinite(xy))
            or np.any(~np.isfinite(candidate))
            or np.any(~np.isfinite(null))
            or np.any(candidate < 0.0)
            or np.any(null < 0.0)
            or np.any(~np.isfinite(values[valid]))
            or np.any(np.isfinite(values[~valid]))
        ):
            raise ValueError("full-track appearance feature arrays are invalid")
        mass = candidate.sum(axis=1, dtype=np.float64) + null.astype(np.float64)
        if np.max(np.abs(mass - 1.0)) > 2e-5:
            raise ValueError("full-track candidate posterior does not conserve mass")
        active = candidate > 0.0
        if np.any(active & (tracks < 0)):
            raise ValueError("positive full-track candidate mass lacks a track id")
        keys = tuple((str(query_id), int(row)) for query_id, row in zip(query_ids, source_rows))
        if len(keys) != len(set(keys)):
            raise ValueError("full-track rows repeat query/source identities")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "split_names", split_names)
        object.__setattr__(self, "source_row_indices", source_rows)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_probabilities", candidate)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "candidate_summary_features", values)
        object.__setattr__(self, "candidate_summary_feature_valid", valid)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "profile_names", profile_names)
        object.__setattr__(self, "artifact_metadata", metadata)
        object.__setattr__(self, "compatibility", dict(self.compatibility))


def _metadata(payload: Mapping[str, np.ndarray], *, path: Path) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{path}: full-track appearance artifact lacks metadata")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: full-track appearance metadata is not an object")
    return metadata


def _radio_intermediate_projection_contract(
    metadata: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    """Return the exact RADIO-intermediate PCA lineage used by an artifact.

    A context-cache checksum alone is sufficient for machine verification, but
    it is opaque in experiment reports.  Carry the projection dimensions and
    source/PCA-fit lineage alongside that checksum so a summary-vs-per-view
    comparison cannot be mislabeled as a PCA-dimension ablation.
    """

    raw = metadata.get("radio_intermediate_projection_override")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("enabled"), bool):
        raise ValueError(
            f"{path}: full-track appearance metadata lacks an intermediate PCA contract"
        )
    enabled = bool(raw["enabled"])
    if not enabled:
        if set(raw).difference({"enabled"}):
            raise ValueError(
                f"{path}: disabled intermediate PCA contract has unexpected fields"
            )
        return {"enabled": False}

    required_text = (
        "source_cache",
        "source_cache_sha256",
        "override_cache",
        "override_cache_sha256",
        "same_source_context_sha256",
        "same_pca_training_manifest_sha256",
    )
    required_dimensions = ("source_projection_dim", "override_projection_dim")
    if (
        any(not isinstance(raw.get(key), str) or not raw[key] for key in required_text)
        or any(
            not isinstance(raw.get(key), int) or int(raw[key]) <= 0
            for key in required_dimensions
        )
        or int(raw["override_projection_dim"])
        <= int(raw["source_projection_dim"])
    ):
        raise ValueError(f"{path}: intermediate PCA contract is invalid")
    return {
        "enabled": True,
        **{key: str(raw[key]) for key in required_text},
        **{key: int(raw[key]) for key in required_dimensions},
    }


def _descriptor_projection_contract(
    metadata: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    """Bind a full-track summary to each descriptor projection it actually uses.

    An all-observation summary can contain mapped RADIO-intermediate context or
    only a RADIO-final full-image context factor.  The latter has no
    intermediate-PCA projection to validate.  Treating that absence as a
    relaxed intermediate contract would let descriptor spaces be mixed without
    an error, so the non-applicability is an explicit, hashable state.
    """

    profiles = metadata.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"{path}: full-track appearance metadata lacks profiles")
    profile_names: list[str] = []
    for profile in profiles:
        if not isinstance(profile, Mapping) or not isinstance(profile.get("name"), str):
            raise ValueError(f"{path}: full-track appearance profile is invalid")
        name = str(profile["name"])
        if not name or name in profile_names:
            raise ValueError(f"{path}: full-track appearance profile names are invalid")
        profile_names.append(name)

    uses_radio_intermediate = any(
        name.startswith("radio_intermediate_") for name in profile_names
    )
    if uses_radio_intermediate:
        return {
            "profile_names": profile_names,
            "radio_intermediate_projection": _radio_intermediate_projection_contract(
                metadata, path=path
            ),
        }
    if "radio_intermediate_projection_override" in metadata:
        raise ValueError(
            f"{path}: non-intermediate full-track summary carries an unexpected PCA contract"
        )
    return {
        "profile_names": profile_names,
        "radio_intermediate_projection": {
            "applicable": False,
            "reason": "no_radio_intermediate_profile_v1",
        },
    }


def _load_fulltrack_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one target-free summary and reject relaxed lineage contracts."""

    required = {
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_support_observation_counts",
        "source_maplet_support_view_counts",
        "candidate_summary_features",
        "candidate_summary_feature_valid",
        "feature_names",
        "profile_names",
        "candidate_profile_usable_counts",
        "candidate_profile_usable_fractions",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path}: full-track appearance artifact lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required if name != "metadata_json"}
        metadata = _metadata(payload, path=Path(path))
    strict = metadata.get("strict_fulltrack_appearance_contract")
    if (
        metadata.get("format") != FROZEN_FULLTRACK_APPEARANCE_ARTIFACT_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or not isinstance(strict, Mapping)
        or strict.get("candidate_identity_fixed") is not True
        or strict.get("candidate_posterior_preserved") is not True
        or strict.get("candidate_reselection") is not False
        or strict.get("support_reselection") is not False
        or strict.get("all_real_sfm_track_observations_enumerated") is not True
        or strict.get("support_view_count_cap") is not None
        or strict.get("candidate_3d_projection_or_pose_used") is not False
        or strict.get("image_retrieval_or_submap_used") is not False
        or strict.get("render") is not False
        or strict.get("heldout_s0_verification_rows") is not True
        or strict.get("raw_summary_not_calibrated_likelihood") is not True
    ):
        raise ValueError(f"{path}: full-track appearance artifact violates strict contract")
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
    xy = np.asarray(arrays["verification_xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    full_counts = np.asarray(arrays["candidate_support_observation_counts"], dtype=np.int64)
    maplet_counts = np.asarray(arrays["source_maplet_support_view_counts"], dtype=np.int64)
    values = np.asarray(arrays["candidate_summary_features"], dtype=np.float32)
    valid = np.asarray(arrays["candidate_summary_feature_valid"], dtype=bool)
    names = np.asarray(arrays["feature_names"]).astype(str).reshape(-1)
    profiles = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
    descriptor_contract = _descriptor_projection_contract(metadata, path=Path(path))
    profile_counts = np.asarray(arrays["candidate_profile_usable_counts"], dtype=np.int64)
    profile_fractions = np.asarray(arrays["candidate_profile_usable_fractions"], dtype=np.float32)
    count = len(query_ids)
    if (
        count != 192
        or len(set(query_ids.tolist())) != 1
        or len(set(split_names.tolist())) != 1
        or split_names[0] not in {"train", "validation", "test"}
        or len(np.unique(rows)) != count
        or xy.shape != (count, 2)
        or tracks.shape != (count, 20)
        or candidate.shape != tracks.shape
        or null.shape != (count,)
        or full_counts.shape != tracks.shape
        or maplet_counts.shape != tracks.shape
        or values.shape[:2] != tracks.shape
        or valid.shape != values.shape
        or values.shape[2] != len(names)
        or not len(names)
        or len(set(names.tolist())) != len(names)
        or not len(profiles)
        or len(set(profiles.tolist())) != len(profiles)
        or tuple(descriptor_contract["profile_names"]) != tuple(profiles.tolist())
        or profile_counts.shape != (*tracks.shape, len(profiles))
        or profile_fractions.shape != profile_counts.shape
        or np.any(~np.isfinite(xy))
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any(candidate < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
        or np.any(full_counts < 0)
        or np.any(maplet_counts < 0)
        or np.any(profile_counts < 0)
        or np.any((profile_fractions < 0.0) | (profile_fractions > 1.0))
        or np.any(~np.isfinite(values[valid]))
        or np.any(np.isfinite(values[~valid]))
        or np.any((candidate > 0.0) & (full_counts <= 0))
    ):
        raise ValueError(f"{path}: full-track appearance arrays are invalid")
    return arrays, metadata


def load_frozen_fulltrack_appearance_features(
    paths: Sequence[Path],
) -> FrozenFulltrackAppearanceFeatures:
    """Merge complete full-track artifacts while retaining their exact contract."""

    artifact_paths = tuple(Path(path) for path in paths)
    if not artifact_paths or len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("full-track appearance paths must be non-empty and unique")
    loaded = [_load_fulltrack_artifact(path) for path in artifact_paths]
    reference_arrays, reference_metadata = loaded[0]
    compatibility = {
        "format": reference_metadata.get("format"),
        "version": reference_metadata.get("version"),
        "profiles": reference_metadata.get("profiles"),
        "summary_statistics": reference_metadata.get("summary_statistics"),
        "appearance_config": reference_metadata.get("appearance_config"),
        "support_geometry_index_sha256": reference_metadata.get(
            "support_geometry_index_sha256"
        ),
        "context_cache_sha256": reference_metadata.get("context_cache_sha256"),
        "descriptor_projection_contract": _descriptor_projection_contract(
            reference_metadata, path=artifact_paths[0]
        ),
        "implementation": reference_metadata.get("implementation"),
    }
    constant_fields = {"feature_names", "profile_names"}
    merge_fields = tuple(name for name in reference_arrays if name not in constant_fields)
    merged: dict[str, list[np.ndarray]] = {name: [] for name in merge_fields}
    keys: list[tuple[str, int]] = []
    all_metadata: list[dict[str, Any]] = []
    for path, (arrays, metadata) in zip(artifact_paths, loaded):
        item_compatibility = {
            "format": metadata.get("format"),
            "version": metadata.get("version"),
            "profiles": metadata.get("profiles"),
            "summary_statistics": metadata.get("summary_statistics"),
            "appearance_config": metadata.get("appearance_config"),
            "support_geometry_index_sha256": metadata.get(
                "support_geometry_index_sha256"
            ),
            "context_cache_sha256": metadata.get("context_cache_sha256"),
            "descriptor_projection_contract": _descriptor_projection_contract(
                metadata, path=path
            ),
            "implementation": metadata.get("implementation"),
        }
        if (
            item_compatibility != compatibility
            or not np.array_equal(arrays["feature_names"], reference_arrays["feature_names"])
            or not np.array_equal(arrays["profile_names"], reference_arrays["profile_names"])
        ):
            raise ValueError(f"{path}: full-track appearance configuration differs")
        ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
        rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
        keys.extend((str(query_id), int(row)) for query_id, row in zip(ids, rows))
        for name in merge_fields:
            merged[name].append(np.asarray(arrays[name]))
        all_metadata.append(metadata)
    if len(keys) != len(set(keys)):
        raise ValueError("full-track appearance artifacts overlap query/source rows")
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in merged.items()}
    return FrozenFulltrackAppearanceFeatures(
        paths=artifact_paths,
        query_ids=arrays["verification_query_ids"],
        split_names=arrays["split_names"],
        source_row_indices=arrays["verification_source_row_indices"],
        xy=arrays["verification_xy"],
        candidate_track_ids=arrays["candidate_track_ids"],
        candidate_probabilities=arrays["candidate_probabilities"],
        null_probabilities=arrays["null_probabilities"],
        candidate_summary_features=arrays["candidate_summary_features"],
        candidate_summary_feature_valid=arrays["candidate_summary_feature_valid"],
        feature_names=tuple(np.asarray(reference_arrays["feature_names"]).astype(str).tolist()),
        profile_names=tuple(np.asarray(reference_arrays["profile_names"]).astype(str).tolist()),
        artifact_metadata=tuple(all_metadata),
        compatibility=compatibility,
    )


def fulltrack_feature_indices_for_family(
    family: str, *, feature_names: Sequence[str], profile_names: Sequence[str]
) -> np.ndarray:
    """Resolve one predeclared source family without a target-side feature sweep."""

    spec = FROZEN_FULLTRACK_RESIDUAL_FAMILIES.get(str(family))
    names = tuple(str(value) for value in feature_names)
    profiles = tuple(str(value) for value in profile_names)
    if spec is None:
        raise ValueError(f"unsupported full-track residual family: {family!r}")
    global_profiles = {"radio_final_global", "radio_final_summary"}
    if set(profiles).intersection(global_profiles) and set(profiles) != global_profiles:
        raise ValueError("full-track summary mixes global and local descriptor profiles")
    actual_kind = (
        "global_context_summary"
        if set(profiles) == global_profiles
        else "local_context_summary"
    )
    if actual_kind != spec.summary_kind:
        raise ValueError(
            f"full-track residual family {family!r} is incompatible with "
            f"{actual_kind} features"
        )
    indices = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if any(name.startswith(prefix) for prefix in spec.feature_prefixes)
        ],
        dtype=np.int64,
    )
    if len(indices) == 0:
        raise ValueError(f"full-track schema has no fields for residual family {family!r}")
    return indices


@dataclass(frozen=True)
class FulltrackAppearanceFeatureNormalizer:
    """Target-free train-split standardization of finite candidate summaries."""

    mean: np.ndarray
    scale: np.ndarray
    feature_indices: np.ndarray
    input_mode: str = "absolute_with_availability_flags"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        indices = np.asarray(self.feature_indices, dtype=np.int64).reshape(-1)
        if (
            len(mean) == 0
            or mean.shape != scale.shape
            or indices.shape != mean.shape
            or len(set(indices.tolist())) != len(indices)
            or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(scale))
            or np.any(scale <= 0.0)
            or np.any(indices < 0)
            or self.input_mode
            not in {
                "absolute_with_availability_flags",
                "top1_relative_common_evidence",
            }
        ):
            raise ValueError("full-track appearance normalizer is invalid")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_indices", indices)


def fit_fulltrack_appearance_feature_normalizer(
    features: FrozenFulltrackAppearanceFeatures,
    *,
    feature_indices: np.ndarray,
    train_rows: np.ndarray,
    input_mode: str = "absolute_with_availability_flags",
) -> FulltrackAppearanceFeatureNormalizer:
    """Fit only from frozen train-split appearance values, never identities."""

    indices = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    if (
        len(indices) == 0
        or np.any((indices < 0) | (indices >= len(features.feature_names)))
        or len(rows) == 0
        or np.any((rows < 0) | (rows >= len(features.query_ids)))
        or len(set(rows.tolist())) != len(rows)
        or input_mode
        not in {
            "absolute_with_availability_flags",
            "top1_relative_common_evidence",
        }
    ):
        raise ValueError("full-track normalizer rows or feature indices are invalid")
    raw = np.asarray(
        features.candidate_summary_features[rows][..., indices], dtype=np.float32
    )
    active = features.candidate_probabilities[rows][..., None] > 0.0
    finite = (
        active
        & features.candidate_summary_feature_valid[rows][..., indices]
        & np.isfinite(raw)
    )
    counts = finite.sum(axis=(0, 1), dtype=np.int64)
    if np.any(counts == 0):
        missing = [
            features.feature_names[index]
            for index, count in zip(indices.tolist(), counts.tolist())
            if count == 0
        ]
        raise ValueError(f"full-track normalizer has no finite train values for {missing}")
    masked = np.where(finite, raw, 0.0).astype(np.float64, copy=False)
    mean = masked.sum(axis=(0, 1), dtype=np.float64) / counts
    variance = np.maximum(
        np.square(masked).sum(axis=(0, 1), dtype=np.float64) / counts - mean**2,
        0.0,
    )
    return FulltrackAppearanceFeatureNormalizer(
        mean=mean.astype(np.float32),
        scale=np.maximum(np.sqrt(variance), 1e-3).astype(np.float32),
        feature_indices=indices,
        input_mode=str(input_mode),
    )


def normalized_fulltrack_appearance_model_input(
    features: FrozenFulltrackAppearanceFeatures,
    normalizer: FulltrackAppearanceFeatureNormalizer,
    *,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Build one predeclared residual input without treating missingness as ID."""

    row_indices = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    if np.any((row_indices < 0) | (row_indices >= len(features.query_ids))):
        raise ValueError("full-track model input rows are out of range")
    raw = np.asarray(
        features.candidate_summary_features[row_indices][..., normalizer.feature_indices],
        dtype=np.float32,
    )
    active = features.candidate_probabilities[row_indices][..., None] > 0.0
    finite = (
        active
        & features.candidate_summary_feature_valid[row_indices][
            ..., normalizer.feature_indices
        ]
        & np.isfinite(raw)
    )
    standardized = np.where(
        finite,
        (raw - normalizer.mean) / normalizer.scale,
        0.0,
    )
    if normalizer.input_mode == "absolute_with_availability_flags":
        return np.concatenate(
            [standardized.astype(np.float32), finite.astype(np.float32)], axis=-1
        )
    if normalizer.input_mode == "top1_relative_common_evidence":
        top_columns = np.argmax(
            features.candidate_probabilities[row_indices], axis=1
        ).astype(np.int64)
        top_raw = raw[np.arange(len(row_indices)), top_columns]
        top_finite = finite[np.arange(len(row_indices)), top_columns]
        jointly_observed = finite & top_finite[:, None, :]
        relative = np.where(
            jointly_observed,
            (raw - top_raw[:, None, :]) / normalizer.scale,
            0.0,
        )
        # The frozen rank-one candidate is the explicit comparison reference;
        # assigning it zero makes a no-evidence challenger bit-exact fallback.
        relative[np.arange(len(row_indices)), top_columns] = 0.0
        return relative.astype(np.float32)
    raise RuntimeError("unsupported full-track residual input mode")


@dataclass(frozen=True)
class FulltrackSummaryTop4RelativeNormalizer:
    """Train-only scales for common frozen top-one-relative summaries."""

    scale: np.ndarray
    feature_indices: np.ndarray

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        indices = np.asarray(self.feature_indices, dtype=np.int64).reshape(-1)
        if (
            len(scale) == 0
            or scale.shape != indices.shape
            or len(set(indices.tolist())) != len(indices)
            or np.any(~np.isfinite(scale))
            or np.any(scale <= 0.0)
            or np.any(indices < 0)
        ):
            raise ValueError("full-track summary top-four normalizer is invalid")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_indices", indices)


def fulltrack_summary_top4_feature_indices_for_family(
    family: str, *, feature_names: Sequence[str]
) -> np.ndarray:
    """Resolve only fixed all-observation top-four NCC summary fields."""

    spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES.get(str(family))
    names = tuple(str(value) for value in feature_names)
    if spec is None:
        raise ValueError(f"unsupported full-track summary top-four family: {family!r}")
    lookup = {name: index for index, name in enumerate(names)}
    required = tuple(
        f"{profile}__uniform_top4_mean_ncc" for profile in spec.profile_names
    )
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(
            "full-track summary lacks required top-four NCC fields: "
            + ", ".join(missing)
        )
    return np.asarray([lookup[name] for name in required], dtype=np.int64)


def _summary_top4_values(
    features: FrozenFulltrackAppearanceFeatures,
    *,
    feature_indices: np.ndarray,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load explicit top-four summary values without availability imputation."""

    indices = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
    row_indices = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    if (
        len(indices) == 0
        or np.any((indices < 0) | (indices >= len(features.feature_names)))
        or len(set(indices.tolist())) != len(indices)
        or np.any((row_indices < 0) | (row_indices >= len(features.query_ids)))
    ):
        raise ValueError("full-track summary top-four rows or fields are invalid")
    values = np.asarray(
        features.candidate_summary_features[row_indices][..., indices],
        dtype=np.float32,
    )
    valid = np.asarray(
        features.candidate_summary_feature_valid[row_indices][..., indices],
        dtype=bool,
    )
    if np.any(~np.isfinite(values[valid])) or np.any(np.isfinite(values[~valid])):
        raise ValueError("full-track summary top-four missing-value semantics differ")
    return values, valid


def fit_fulltrack_summary_top4_relative_normalizer(
    features: FrozenFulltrackAppearanceFeatures,
    *,
    feature_indices: np.ndarray,
    train_rows: np.ndarray,
) -> FulltrackSummaryTop4RelativeNormalizer:
    """Fit scales from unlabelled train rows of common top-one evidence."""

    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    if (
        len(rows) == 0
        or len(set(rows.tolist())) != len(rows)
        or np.any((rows < 0) | (rows >= len(features.query_ids)))
    ):
        raise ValueError("full-track summary top-four normalizer rows are invalid")
    values, valid = _summary_top4_values(features, feature_indices=feature_indices)
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    top_columns = np.argmax(candidate, axis=1).astype(np.int64)
    top_values = values[np.arange(len(values)), top_columns]
    top_valid = valid[np.arange(len(valid)), top_columns]
    common = valid & top_valid[:, None, :] & (candidate > 0.0)[..., None]
    deltas = values - top_values[:, None, :]
    selected = common[rows]
    selected_deltas = deltas[rows]
    count = selected.sum(axis=(0, 1), dtype=np.int64)
    if np.any(count == 0):
        raise ValueError("full-track summary top-four normalizer lacks common train support")
    masked = np.where(selected, selected_deltas, 0.0).astype(
        np.float64, copy=False
    )
    variance = np.maximum(
        np.square(masked).sum(axis=(0, 1), dtype=np.float64) / count,
        0.0,
    )
    return FulltrackSummaryTop4RelativeNormalizer(
        scale=np.maximum(np.sqrt(variance), 1e-3).astype(np.float32),
        feature_indices=np.asarray(feature_indices, dtype=np.int64),
    )


def normalized_fulltrack_summary_top4_relative_features(
    features: FrozenFulltrackAppearanceFeatures,
    normalizer: FulltrackSummaryTop4RelativeNormalizer,
    *,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact-zero unknowns for non-common frozen top-one evidence."""

    values, valid = _summary_top4_values(
        features,
        feature_indices=normalizer.feature_indices,
        rows=rows,
    )
    row_indices = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    candidate = np.asarray(features.candidate_probabilities[row_indices], dtype=np.float32)
    top_columns = np.argmax(candidate, axis=1).astype(np.int64)
    top_values = values[np.arange(len(values)), top_columns]
    top_valid = valid[np.arange(len(valid)), top_columns]
    common = valid & top_valid[:, None, :] & (candidate > 0.0)[..., None]
    output = np.zeros_like(values, dtype=np.float32)
    expanded_top = np.broadcast_to(top_values[:, None, :], values.shape)
    expanded_scale = np.broadcast_to(normalizer.scale[None, None, :], values.shape)
    output[common] = (values[common] - expanded_top[common]) / expanded_scale[common]
    output[np.arange(len(output)), top_columns] = 0.0
    if np.any(~np.isfinite(output)):
        raise RuntimeError("full-track summary top-four relative features are non-finite")
    return output, common


def fixed_candidate_conditional_log_priors(
    features: FrozenFulltrackAppearanceFeatures,
    *,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable conditional candidate priors and candidate mass.

    The null probability is intentionally excluded from this re-ranking
    parameterization.  Candidate residuals work only within the original
    non-null mass, so no identity experiment can masquerade as null calibration.
    """

    row_indices = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    if np.any((row_indices < 0) | (row_indices >= len(features.query_ids))):
        raise ValueError("full-track prior rows are out of range")
    candidate = np.asarray(features.candidate_probabilities[row_indices], dtype=np.float32)
    mass = candidate.sum(axis=1, dtype=np.float32)
    expected = 1.0 - np.asarray(features.null_probabilities[row_indices], dtype=np.float32)
    if np.max(np.abs(mass - expected)) > 2e-5:
        raise ValueError("full-track candidate and null mass disagree")
    conditional_log = np.full(candidate.shape, -np.inf, dtype=np.float32)
    active = candidate > 0.0
    positive_rows = mass > 0.0
    normalized = np.divide(
        candidate,
        mass[:, None],
        out=np.zeros_like(candidate),
        where=positive_rows[:, None],
    )
    conditional_log[active] = np.log(normalized[active])
    if np.any((~positive_rows) & active.any(axis=1)):
        raise RuntimeError("zero candidate mass has an active candidate")
    return conditional_log, mass.astype(np.float32)


@dataclass(frozen=True)
class Rank2HardPairSelection:
    """Train-only highest-prior correct-vs-top-wrong candidate comparisons."""

    row_positions: np.ndarray
    positive_columns: np.ndarray
    negative_columns: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.row_positions, dtype=np.int64).reshape(-1)
        positive = np.asarray(self.positive_columns, dtype=np.int64).reshape(-1)
        negative = np.asarray(self.negative_columns, dtype=np.int64).reshape(-1)
        if (
            rows.shape != positive.shape
            or rows.shape != negative.shape
            or np.any(rows < 0)
            or np.any(positive < 0)
            or np.any(negative < 0)
            or np.any(positive == negative)
            or len(np.unique(rows)) != len(rows)
        ):
            raise ValueError("rank-2 hard-pair selection is invalid")
        object.__setattr__(self, "row_positions", rows)
        object.__setattr__(self, "positive_columns", positive)
        object.__setattr__(self, "negative_columns", negative)

    @property
    def pair_count(self) -> int:
        return int(len(self.row_positions))


def select_rank2_to_top1_wrong_training_pairs(
    *, candidate_probabilities: np.ndarray, candidate_labels: np.ndarray
) -> Rank2HardPairSelection:
    """Select hard train pairs using only fixed prior and train identities."""

    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    labels = np.asarray(candidate_labels, dtype=bool)
    if (
        probabilities.ndim != 2
        or labels.shape != probabilities.shape
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(labels.sum(axis=1) > 1)
    ):
        raise ValueError("rank-2 hard-pair inputs are invalid")
    rows: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for row in range(len(probabilities)):
        valid = probabilities[row] > 0.0
        positive = valid & labels[row]
        negative = valid & ~labels[row]
        if not np.any(positive) or not np.any(negative):
            continue
        positive_column = int(np.flatnonzero(positive)[0])
        ordered = np.flatnonzero(valid)[
            np.argsort(-probabilities[row, valid], kind="stable")
        ]
        positive_rank = np.flatnonzero(ordered == positive_column)
        if len(positive_rank) != 1:
            raise RuntimeError("hard-pair positive is absent from frozen ordering")
        if int(positive_rank[0]) == 0:
            continue
        negative_column = int(next(column for column in ordered if negative[column]))
        rows.append(row)
        positives.append(positive_column)
        negatives.append(negative_column)
    return Rank2HardPairSelection(
        row_positions=np.asarray(rows, dtype=np.int64),
        positive_columns=np.asarray(positives, dtype=np.int64),
        negative_columns=np.asarray(negatives, dtype=np.int64),
    )


class FixedCandidateMassLinearResidual(nn.Module):
    """A zero-preserving linear re-ranker that cannot alter null mass."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("full-track residual input dimension must be positive")
        self.linear = nn.Linear(int(input_dim), 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(
        self,
        model_input: torch.Tensor,
        candidate_conditional_log_prior: torch.Tensor,
        candidate_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            model_input.ndim != 3
            or candidate_conditional_log_prior.shape != model_input.shape[:2]
            or candidate_mass.shape != (model_input.shape[0],)
            or torch.any(candidate_mass < 0.0)
        ):
            raise ValueError("full-track residual tensors are incompatible")
        residual = self.linear(model_input).squeeze(-1)
        active = torch.isfinite(candidate_conditional_log_prior)
        candidate_logits = candidate_conditional_log_prior + residual
        # Softmax needs a finite row even for the degenerate all-null case.  The
        # output is then explicitly zeroed, retaining its input candidate mass.
        safe_logits = torch.where(active, candidate_logits, torch.full_like(candidate_logits, -torch.inf))
        zero_mass = candidate_mass <= 0.0
        safe_logits = torch.where(zero_mass[:, None], torch.zeros_like(safe_logits), safe_logits)
        conditional = torch.softmax(safe_logits, dim=1)
        conditional = torch.where(active, conditional, torch.zeros_like(conditional))
        candidate = conditional * candidate_mass[:, None]
        return candidate, residual, safe_logits


def candidate_membership_nll(
    conditional_logits: torch.Tensor, membership: torch.Tensor
) -> torch.Tensor:
    """NLL over exactly one retrieved train identity, never the null class."""

    if conditional_logits.shape != membership.shape or conditional_logits.ndim != 2:
        raise ValueError("full-track residual identity membership is incompatible")
    targets = membership.to(dtype=torch.bool)
    if torch.any(targets.sum(dim=1) != 1):
        raise ValueError("full-track residual needs exactly one retrieved target per row")
    log_probabilities = F.log_softmax(conditional_logits, dim=1)
    selected = torch.where(targets, log_probabilities, torch.full_like(log_probabilities, -torch.inf))
    return -torch.logsumexp(selected, dim=1).mean()


def rank2_hard_pair_loss(
    candidate_logits: torch.Tensor,
    row_positions: torch.Tensor,
    positive_columns: torch.Tensor,
    negative_columns: torch.Tensor,
) -> torch.Tensor:
    """Encourage a train correct candidate to exceed its frozen top wrong track."""

    if (
        row_positions.ndim != 1
        or positive_columns.shape != row_positions.shape
        or negative_columns.shape != row_positions.shape
    ):
        raise ValueError("rank-2 hard-pair loss indices are incompatible")
    if len(row_positions) == 0:
        return candidate_logits.new_zeros(())
    differences = (
        candidate_logits[row_positions, positive_columns]
        - candidate_logits[row_positions, negative_columns]
    )
    return F.softplus(-differences).mean()


def coarse_top1_stability_loss(
    candidate_logits: torch.Tensor,
    membership: torch.Tensor,
    coarse_top1_columns: torch.Tensor,
) -> torch.Tensor:
    """Prevent a summary uplift from overturning a correct frozen top-one."""

    if (
        candidate_logits.ndim != 2
        or membership.shape != candidate_logits.shape
        or membership.dtype != torch.bool
        or coarse_top1_columns.shape != (candidate_logits.shape[0],)
        or coarse_top1_columns.dtype != torch.long
        or bool(torch.any(membership.sum(dim=1) != 1))
        or bool(
            torch.any(
                (coarse_top1_columns < 0)
                | (coarse_top1_columns >= candidate_logits.shape[1])
            )
        )
    ):
        raise ValueError("full-track summary top-one stability inputs are invalid")
    correct_columns = torch.argmax(membership.to(dtype=torch.int64), dim=1)
    stable_rows = coarse_top1_columns == correct_columns
    if not bool(torch.any(stable_rows)):
        return candidate_logits.new_zeros(())
    logits = candidate_logits[stable_rows]
    labels = membership[stable_rows]
    correct = logits[
        torch.arange(len(logits), device=logits.device), correct_columns[stable_rows]
    ]
    strongest_wrong = torch.where(
        labels,
        torch.full_like(logits, -torch.inf),
        logits,
    ).max(dim=1).values
    if not bool(torch.isfinite(correct).all() & torch.isfinite(strongest_wrong).all()):
        raise ValueError("full-track summary top-one stability logits are non-finite")
    return F.softplus(strongest_wrong - correct).mean()


def fit_fixedprior_fulltrack_summary_top4_probe(
    *,
    features: FrozenFulltrackAppearanceFeatures,
    family: str,
    train_normalizer_rows: np.ndarray,
    target_train_rows: np.ndarray,
    target_candidate_membership: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[
    MonotonicTop1RelativeTop4Residual,
    FulltrackSummaryTop4RelativeNormalizer,
    dict[str, Any],
]:
    """Fit the bounded balanced objective on immutable top-four summaries."""

    spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES.get(str(family))
    target_rows = np.asarray(target_train_rows, dtype=np.int64).reshape(-1)
    membership = np.asarray(target_candidate_membership, dtype=bool)
    if (
        spec is None
        or len(target_rows) == 0
        or len(set(target_rows.tolist())) != len(target_rows)
        or np.any((target_rows < 0) | (target_rows >= len(features.query_ids)))
        or membership.shape
        != (len(target_rows), features.candidate_track_ids.shape[1])
        or np.any(membership.sum(axis=1) != 1)
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("full-track summary top-four fit arguments are invalid")
    feature_indices = fulltrack_summary_top4_feature_indices_for_family(
        str(family), feature_names=features.feature_names
    )
    normalizer = fit_fulltrack_summary_top4_relative_normalizer(
        features,
        feature_indices=feature_indices,
        train_rows=np.asarray(train_normalizer_rows, dtype=np.int64),
    )
    relative, common = normalized_fulltrack_summary_top4_relative_features(
        features, normalizer
    )
    profile_common_coverage = np.mean(common, axis=(0, 1), dtype=np.float64)
    candidate_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features
    )
    pair_selection = select_rank2_to_top1_wrong_training_pairs(
        candidate_probabilities=features.candidate_probabilities[target_rows],
        candidate_labels=membership,
    )
    pair_positive = np.full((len(target_rows),), -1, dtype=np.int64)
    pair_negative = np.full((len(target_rows),), -1, dtype=np.int64)
    pair_positive[pair_selection.row_positions] = pair_selection.positive_columns
    pair_negative[pair_selection.row_positions] = pair_selection.negative_columns
    coarse_top1 = np.argmax(
        features.candidate_probabilities[target_rows], axis=1
    ).astype(np.int64)
    positive_columns = np.argmax(membership, axis=1).astype(np.int64)
    coarse_top1_correct = coarse_top1 == positive_columns
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = MonotonicTop1RelativeTop4Residual(
        int(relative.shape[2]),
        positive_uplift=True,
        residual_scale=float(spec.residual_scale),
        residual_cap=float(spec.residual_cap),
    ).to(device)
    # These constrained positive weights must be able to suppress a profile;
    # decoupled decay instead pulls softplus logits toward a nonzero weight.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    rng = np.random.default_rng(int(seed))
    final_identity = float("nan")
    final_hard = float("nan")
    final_stability = float("nan")
    final_total = float("nan")
    model.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(target_rows))
        for begin in range(0, len(order), int(batch_size)):
            selected = order[begin : begin + int(batch_size)]
            rows = target_rows[selected]
            _candidate, _residual, logits = model(
                relative_features=torch.from_numpy(relative[rows]).to(device),
                candidate_conditional_log_prior=torch.from_numpy(
                    candidate_log_prior[rows]
                ).to(device),
                candidate_mass=torch.from_numpy(candidate_mass[rows]).to(device),
            )
            labels = torch.from_numpy(membership[selected]).to(
                device=device, dtype=torch.bool
            )
            identity_loss = candidate_membership_nll(logits, labels)
            batch_positive = pair_positive[selected]
            pair_mask = batch_positive >= 0
            hard_loss = rank2_hard_pair_loss(
                logits,
                torch.from_numpy(np.flatnonzero(pair_mask)).to(device),
                torch.from_numpy(batch_positive[pair_mask]).to(device),
                torch.from_numpy(pair_negative[selected][pair_mask]).to(device),
            )
            stability_loss = coarse_top1_stability_loss(
                logits,
                labels,
                torch.from_numpy(coarse_top1[selected]).to(
                    device=device, dtype=torch.long
                ),
            )
            total_loss = (
                identity_loss
                + float(spec.rank2_hard_pair_weight) * hard_loss
                + float(spec.coarse_top1_stability_weight) * stability_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            final_identity = float(identity_loss.detach().cpu())
            final_hard = float(hard_loss.detach().cpu())
            final_stability = float(stability_loss.detach().cpu())
            final_total = float(total_loss.detach().cpu())
    model.eval()
    raw_weights = model.weights().detach().cpu().numpy().astype(np.float32, copy=True)
    effective_weights = (
        raw_weights * float(spec.residual_scale)
    ).astype(np.float32, copy=False)
    return model, normalizer, {
        "family": str(family),
        "architecture": SUMMARY_TOP4_BALANCED_ARCHITECTURE,
        "candidate_evidence_transform": (
            "relu_positive_relative_summary_top4_uplift_tanh_bounded_traincal_v1"
        ),
        "summary_statistic": "uniform_top4_mean_ncc",
        "profile_names": list(spec.profile_names),
        "profile_feature_names": [
            features.feature_names[index] for index in feature_indices.tolist()
        ],
        "profile_count": int(len(feature_indices)),
        "profile_common_coverage": {
            profile: float(coverage)
            for profile, coverage in zip(
                spec.profile_names, profile_common_coverage.tolist()
            )
        },
        "train_identity_row_count": int(len(target_rows)),
        "train_normalizer_row_count": int(
            len(np.asarray(train_normalizer_rows, dtype=np.int64).reshape(-1))
        ),
        "rank2_to_top1_wrong_train_pair_count": pair_selection.pair_count,
        "rank2_hard_pair_weight": float(spec.rank2_hard_pair_weight),
        "coarse_top1_correct_train_pair_count": int(np.sum(coarse_top1_correct)),
        "coarse_top1_stability_weight": float(spec.coarse_top1_stability_weight),
        "training_objective": (
            "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "effective_weight_decay": 0.0,
        "seed": int(seed),
        "last_train_identity_nll": final_identity,
        "last_train_rank2_hard_pair_loss": final_hard,
        "last_train_coarse_top1_stability_loss": final_stability,
        "last_train_total_loss": final_total,
        "raw_fit_monotonic_nonnegative_weights": raw_weights.tolist(),
        "monotonic_nonnegative_weights": effective_weights.tolist(),
        "residual_calibration_applied_during_train": True,
        "residual_scale": float(spec.residual_scale),
        "residual_cap": float(spec.residual_cap),
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "all_observation_aggregation": (
            "deterministic_uniform_top4_real_observation_summary_v1"
        ),
        "missing_evidence_semantics": (
            "common_top1_profile_missing_zero_residual_v1"
        ),
        "per_view_model": False,
        "zero_residual_reproduces_fixed_posterior": False,
    }


@torch.inference_mode()
def predict_fixedprior_fulltrack_summary_top4_probe(
    *,
    model: MonotonicTop1RelativeTop4Residual,
    features: FrozenFulltrackAppearanceFeatures,
    normalizer: FulltrackSummaryTop4RelativeNormalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frozen summary-top-four overlay without changing null mass."""

    if int(batch_size) <= 0:
        raise ValueError("full-track summary top-four prediction batch size is invalid")
    relative, _common = normalized_fulltrack_summary_top4_relative_features(
        features, normalizer
    )
    candidate_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features
    )
    candidate_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(features.query_ids), int(batch_size)):
        end = min(start + int(batch_size), len(features.query_ids))
        candidate, residual, _logits = model(
            relative_features=torch.from_numpy(relative[start:end]).to(device),
            candidate_conditional_log_prior=torch.from_numpy(
                candidate_log_prior[start:end]
            ).to(device),
            candidate_mass=torch.from_numpy(candidate_mass[start:end]).to(device),
        )
        candidate_blocks.append(candidate.cpu().numpy().astype(np.float32))
        residual_blocks.append(residual.cpu().numpy().astype(np.float32))
    candidate = np.concatenate(candidate_blocks, axis=0)
    null = np.asarray(features.null_probabilities, dtype=np.float32).copy()
    if (
        np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
        or np.max(np.abs(null - features.null_probabilities)) > 0.0
    ):
        raise RuntimeError("full-track summary top-four prediction changed fixed mass")
    return candidate, null, np.concatenate(residual_blocks, axis=0)


def fit_fixedprior_fulltrack_linear_residual(
    *,
    features: FrozenFulltrackAppearanceFeatures,
    family: str,
    train_normalizer_rows: np.ndarray,
    target_train_rows: np.ndarray,
    target_candidate_membership: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[FixedCandidateMassLinearResidual, FulltrackAppearanceFeatureNormalizer, dict[str, Any]]:
    """Fit one train-only conditional identity residual with fixed hard pairs."""

    spec = FROZEN_FULLTRACK_RESIDUAL_FAMILIES.get(str(family))
    rows = np.asarray(target_train_rows, dtype=np.int64).reshape(-1)
    membership = np.asarray(target_candidate_membership, dtype=bool)
    if (
        spec is None
        or len(rows) == 0
        or membership.shape != (len(rows), features.candidate_track_ids.shape[1])
        or np.any(membership.sum(axis=1) != 1)
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
    ):
        raise ValueError("full-track residual fit arguments are invalid")
    indices = fulltrack_feature_indices_for_family(
        family,
        feature_names=features.feature_names,
        profile_names=features.profile_names,
    )
    normalizer = fit_fulltrack_appearance_feature_normalizer(
        features,
        feature_indices=indices,
        train_rows=train_normalizer_rows,
        input_mode=spec.input_mode,
    )
    inputs = normalized_fulltrack_appearance_model_input(features, normalizer, rows=rows)
    candidate_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features, rows=rows
    )
    pair_selection = select_rank2_to_top1_wrong_training_pairs(
        candidate_probabilities=features.candidate_probabilities[rows],
        candidate_labels=membership,
    )
    pair_positive_by_row = np.full((len(rows),), -1, dtype=np.int64)
    pair_negative_by_row = np.full((len(rows),), -1, dtype=np.int64)
    pair_positive_by_row[pair_selection.row_positions] = pair_selection.positive_columns
    pair_negative_by_row[pair_selection.row_positions] = pair_selection.negative_columns
    input_tensor = torch.from_numpy(inputs).to(device)
    prior_tensor = torch.from_numpy(candidate_log_prior).to(device)
    mass_tensor = torch.from_numpy(candidate_mass).to(device)
    membership_tensor = torch.from_numpy(membership).to(device=device, dtype=torch.bool)
    positive_tensor = torch.from_numpy(pair_positive_by_row).to(device)
    negative_tensor = torch.from_numpy(pair_negative_by_row).to(device)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = FixedCandidateMassLinearResidual(input_tensor.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed))
    final_identity_loss = float("nan")
    final_hard_pair_loss = float("nan")
    final_total_loss = float("nan")
    model.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(rows))
        for begin in range(0, len(order), int(batch_size)):
            batch = torch.as_tensor(order[begin : begin + int(batch_size)], device=device)
            _candidate, _residual, candidate_logits = model(
                input_tensor[batch], prior_tensor[batch], mass_tensor[batch]
            )
            identity_loss = candidate_membership_nll(
                candidate_logits, membership_tensor[batch]
            )
            batch_positive = positive_tensor[batch]
            pair_mask = batch_positive >= 0
            hard_loss = rank2_hard_pair_loss(
                candidate_logits,
                torch.nonzero(pair_mask, as_tuple=False).reshape(-1),
                batch_positive[pair_mask],
                negative_tensor[batch][pair_mask],
            )
            total_loss = identity_loss + float(spec.rank2_hard_pair_weight) * hard_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            final_identity_loss = float(identity_loss.detach().cpu())
            final_hard_pair_loss = float(hard_loss.detach().cpu())
            final_total_loss = float(total_loss.detach().cpu())
    model.eval()
    return model, normalizer, {
        "family": str(family),
        "architecture": "fixed_candidate_mass_linear_summary_residual_no_bias_v1",
        "feature_names": [features.feature_names[index] for index in indices],
        "feature_count": int(len(indices)),
        "train_identity_row_count": int(len(rows)),
        "train_normalizer_row_count": int(len(np.asarray(train_normalizer_rows).reshape(-1))),
        "rank2_to_top1_wrong_train_pair_count": pair_selection.pair_count,
        "rank2_hard_pair_weight": float(spec.rank2_hard_pair_weight),
        "input_mode": str(spec.input_mode),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "last_train_identity_nll": final_identity_loss,
        "last_train_rank2_hard_pair_loss": final_hard_pair_loss,
        "last_train_total_loss": final_total_loss,
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "all_observation_aggregation": "frozen_fulltrack_summary_input_v1",
        "missing_evidence_semantics": (
            "feature_available_flag_learned_v1"
            if spec.input_mode == "absolute_with_availability_flags"
            else "joint_top1_challenger_availability_is_zero_residual_unknown_v1"
        ),
        "per_view_model": False,
        "zero_residual_reproduces_fixed_posterior": True,
    }


@torch.inference_mode()
def predict_fixedprior_fulltrack_linear_residual(
    *,
    model: FixedCandidateMassLinearResidual,
    features: FrozenFulltrackAppearanceFeatures,
    normalizer: FulltrackAppearanceFeatureNormalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one frozen aggregate residual while preserving null exactly."""

    if int(batch_size) <= 0:
        raise ValueError("full-track residual prediction batch size must be positive")
    candidate_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(features)
    candidate_output: list[np.ndarray] = []
    residual_output: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(features.query_ids), int(batch_size)):
        end = min(start + int(batch_size), len(features.query_ids))
        inputs = normalized_fulltrack_appearance_model_input(
            features, normalizer, rows=np.arange(start, end, dtype=np.int64)
        )
        candidate, residual, _logits = model(
            torch.from_numpy(inputs).to(device),
            torch.from_numpy(candidate_log_prior[start:end]).to(device),
            torch.from_numpy(candidate_mass[start:end]).to(device),
        )
        candidate_output.append(candidate.cpu().numpy().astype(np.float32))
        residual_output.append(residual.cpu().numpy().astype(np.float32))
    candidate = np.concatenate(candidate_output, axis=0)
    null = np.asarray(features.null_probabilities, dtype=np.float32).copy()
    if np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5:
        raise RuntimeError("full-track residual prediction does not conserve posterior mass")
    if np.max(np.abs(null - features.null_probabilities)) > 0.0:
        raise RuntimeError("full-track residual changed its input null probability")
    return candidate, null, np.concatenate(residual_output, axis=0)


def zero_residual_fulltrack_posterior(
    features: FrozenFulltrackAppearanceFeatures,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact invariant baseline posterior for audit provenance."""

    return (
        np.asarray(features.candidate_probabilities, dtype=np.float32).copy(),
        np.asarray(features.null_probabilities, dtype=np.float32).copy(),
    )
