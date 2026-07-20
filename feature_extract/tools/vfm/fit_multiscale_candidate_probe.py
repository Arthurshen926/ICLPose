"""Fit fixed S1 appearance probes on train images and export frozen overlays.

Only the train split is allowed to construct supervision targets.  The
resulting prediction and overlay artifacts contain no targets; validation/test
labels are intentionally joined by a separate audit command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT,
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
    COST_VOLUME_CONTEXT_SCALE_NAMES,
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT,
    DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES,
    GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT,
    GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FEATURE_NAMES,
    GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    PerViewFeatureNormalizer,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    predict_per_view_candidate_probe,
    train_per_view_candidate_probe,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_candidate_identity_target_membership,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


FEATURE_ARTIFACT_FORMAT = "multiscale_candidate_probe_features_v1"
STRUCTURED_FEATURE_ARTIFACT_FORMAT = "structured_multiscale_candidate_probe_features_v1"
STRUCTURED_FEATURE_ARTIFACT_FORMAT_V2 = "structured_multiscale_candidate_probe_features_v2"
STRUCTURED_FEATURE_ARTIFACT_FORMATS = frozenset(
    {STRUCTURED_FEATURE_ARTIFACT_FORMAT, STRUCTURED_FEATURE_ARTIFACT_FORMAT_V2}
)
COST_VOLUME_FEATURE_ARTIFACT_FORMAT = "cost_volume_multiscale_candidate_probe_features_v1"
WIDE_FULL_CORRELATION_FEATURE_ARTIFACT_FORMAT = (
    "wide_full_correlation_multiscale_candidate_probe_features_v1"
)
GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT, GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT}
)
LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMATS = frozenset(
    {
        LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
        HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
        MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    }
)
DENSE_LOCAL_MODE_FEATURE_ARTIFACT_FORMATS = frozenset(
    {DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT}
)
ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT}
)
ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT}
)
SPARSE_COST_VOLUME_FEATURE_ARTIFACT_FORMATS = frozenset(
    {COST_VOLUME_FEATURE_ARTIFACT_FORMAT, WIDE_FULL_CORRELATION_FEATURE_ARTIFACT_FORMAT}
)
COMPLETE_FROZEN_LAYOUT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {
        *STRUCTURED_FEATURE_ARTIFACT_FORMATS,
        *SPARSE_COST_VOLUME_FEATURE_ARTIFACT_FORMATS,
        *GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMATS,
        *LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMATS,
        *DENSE_LOCAL_MODE_FEATURE_ARTIFACT_FORMATS,
        *ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMATS,
        *ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMATS,
    }
)
PREDICTION_ARTIFACT_FORMAT = "multiscale_candidate_probe_predictions_v2"
OVERLAY_ARTIFACT_FORMAT = "multiscale_candidate_probe_prior_overlay_v2"
MODEL_FORMAT = "multiscale_per_view_candidate_probe_v3"
SET_MEMBERSHIP_OBJECTIVE = "set_log_mass_nll_over_target_membership_v1"
GEOMETRIC_SET_SUPERVISION_MODE = "geometric_set"
REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE = "registered_track_identity"
SUPPORTED_SUPERVISION_MODES = frozenset(
    {GEOMETRIC_SET_SUPERVISION_MODE, REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE}
)
GEOMETRIC_PROBABILITY_SEMANTICS = (
    "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one"
)
EXACT_IDENTITY_PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
REGISTERED_TRACK_IDENTITY_OBJECTIVE = (
    "registered_query_observation_exact_track_or_explicit_null_nll_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--base_prior_overlay", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--families",
        default="all",
        help="all or a comma-separated fixed family list",
    )
    parser.add_argument("--geometric_positive_threshold_px", type=float, default=2.0)
    parser.add_argument(
        "--supervision_mode",
        choices=tuple(sorted(SUPPORTED_SUPERVISION_MODES)),
        default=GEOMETRIC_SET_SUPERVISION_MODE,
        help=(
            "geometric_set uses projected-location membership; "
            "registered_track_identity uses only train-anchor SfM track identities"
        ),
    )
    parser.add_argument(
        "--colmap_model_dir",
        default="",
        help="required only for --supervision_mode=registered_track_identity",
    )
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--architecture",
        choices=("linear", "mlp"),
        default="linear",
        help="frozen-feature probe capacity; never changes the candidate inputs",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="hidden width for --architecture mlp",
    )
    parser.add_argument(
        "--prior_residual",
        action="store_true",
        help=(
            "add visual candidate offsets to the fixed target-free base prior; "
            "zero offsets exactly reproduce the base candidate/null distribution"
        ),
    )
    parser.add_argument(
        "--allow_diagnostic_feature_artifact",
        action="store_true",
        help="allow a --max_queries export only for code diagnostics",
    )
    return parser.parse_args(argv)


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, object]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} has no metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _cost_volume_materialized_feature_names(
    metadata: Mapping[str, object], *, feature_names: Sequence[str]
) -> frozenset[str]:
    """Resolve the explicitly exported subset of a sparse cost-volume schema."""

    payload = metadata.get("cost_volume")
    if not isinstance(payload, Mapping):
        raise ValueError("cost-volume S1 feature artifact lacks cost-volume metadata")
    materialized = payload.get("materialized_scale_names")
    if not isinstance(materialized, list) or not materialized:
        raise ValueError("cost-volume S1 feature artifact lacks materialized scales")
    scales = tuple(str(value) for value in materialized)
    artifact_format = str(metadata.get("format"))
    if artifact_format == COST_VOLUME_FEATURE_ARTIFACT_FORMAT:
        declared_scales = COST_VOLUME_CONTEXT_SCALE_NAMES
        anchors = COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    elif artifact_format == WIDE_FULL_CORRELATION_FEATURE_ARTIFACT_FORMAT:
        declared_scales = WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES
        anchors = WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    else:
        raise ValueError("unsupported sparse cost-volume S1 feature artifact")
    if len(set(scales)) != len(scales) or set(scales) - set(declared_scales):
        raise ValueError("cost-volume S1 feature artifact has invalid materialized scales")
    names = tuple(str(value) for value in feature_names)
    available = set(anchors)
    for scale in scales:
        available.update(name for name in names if name.startswith(f"{scale}_"))
    return frozenset(available)


def _validate_cost_volume_family_materialization(
    metadata: Mapping[str, object], *, feature_names: Sequence[str], families: Sequence[str]
) -> None:
    if metadata.get("format") not in SPARSE_COST_VOLUME_FEATURE_ARTIFACT_FORMATS:
        return
    available = _cost_volume_materialized_feature_names(
        metadata, feature_names=feature_names
    )
    for family in families:
        requested = ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES.get(str(family))
        if requested is None:
            raise ValueError(f"unsupported cost-volume family {family!r}")
        missing = set(requested) - available
        if missing:
            raise ValueError(
                f"cost-volume family {family!r} references unmaterialized scales: "
                f"{sorted(missing)[:3]}"
            )


def _allowed_soft_global_context(metadata: Mapping[str, object]) -> bool:
    """Allow only the explicit fixed-support soft context protocol.

    Whole-image features are normally prohibited in S1.  This narrow exception
    is the post-S1e diagnostic: a query image may compare only with the already
    fixed support image of each candidate/view.  It cannot retrieve or reseat
    candidates, so any broader global descriptor use remains rejected.
    """

    artifact_format = str(metadata.get("format"))
    expected_usage = GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT.get(artifact_format)
    return bool(
        expected_usage is not None
        and metadata.get("whole_image_summary_or_global_used") is True
        and metadata.get("soft_global_context_factor") is True
        and metadata.get("global_context_usage") == expected_usage
        and metadata.get("global_context_hard_retrieval_or_candidate_reselection") is False
    )


def _load_features(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "xy",
        "candidate_track_ids",
        "candidate_canonical_rows",
        "candidate_features",
        "candidate_view_valid",
        "feature_names",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError("S1 feature artifact unexpectedly contains labels")
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"S1 feature artifact lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
        metadata = _metadata(data, context="S1 feature artifact")
    if metadata.get("format") not in {
        FEATURE_ARTIFACT_FORMAT,
        *COMPLETE_FROZEN_LAYOUT_FEATURE_ARTIFACT_FORMATS,
    }:
        raise ValueError("unsupported S1 feature artifact format")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "pose_or_ground_truth_used"
    ) is not False:
        raise ValueError("S1 feature artifact is not target-free")
    if bool(metadata.get("image_retrieval_or_submap_used", True)):
        raise ValueError("S1 feature artifact violates the no-retrieval protocol")
    if bool(metadata.get("whole_image_summary_or_global_used", True)) and not _allowed_soft_global_context(
        metadata
    ):
        raise ValueError("S1 feature artifact has an unapproved whole-image context path")
    if metadata.get("format") in GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMATS and not _allowed_soft_global_context(
        metadata
    ):
        raise ValueError("global-context S1 feature artifact lacks its strict soft-factor manifest")
    if metadata.get("format") in COMPLETE_FROZEN_LAYOUT_FEATURE_ARTIFACT_FORMATS and metadata.get(
        "is_complete_frozen_layout"
    ) is not True:
        raise ValueError("structured S1 feature artifact is not a fully merged frozen layout")
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32).reshape(-1, 2)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    canonical = np.asarray(arrays["candidate_canonical_rows"], dtype=np.int64)
    features = np.asarray(arrays["candidate_features"], dtype=np.float32)
    views = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    names = tuple(str(value) for value in arrays["feature_names"].tolist())
    if (
        len(rows) == 0
        or np.unique(rows).size != len(rows)
        or query_ids.shape != split_names.shape != (len(rows),)
        or xy.shape != (len(rows), 2)
        or tracks.ndim != 2
        or canonical.shape != tracks.shape
        or features.ndim != 4
        or views.shape != features.shape[:3]
        or features.shape[:2] != tracks.shape
        or features.shape[0] != len(rows)
        or features.shape[3] != len(names)
    ):
        raise ValueError("S1 feature artifact arrays are not aligned")
    if names not in {
        MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES,
        GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FEATURE_NAMES,
        LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES,
        ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
        ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    }:
        raise ValueError("S1 feature names differ from a supported frozen probe schema")
    if set(split_names.tolist()) - {"train", "validation", "test"}:
        raise ValueError("S1 feature rows have an unknown split name")
    valid_candidates = tracks >= 0
    if np.any(canonical[valid_candidates] < 0) or np.any(
        canonical[~valid_candidates] >= 0
    ):
        raise ValueError("candidate canonical-row identities are invalid")
    if np.any(valid_candidates & ~np.any(views, axis=2)):
        raise ValueError("a valid candidate has no real support view")
    valid_feature_rows = features[views]
    if np.any(np.isinf(valid_feature_rows)):
        raise ValueError("valid S1 feature entries contain infinity")
    if names == MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES and np.any(
        ~np.isfinite(valid_feature_rows)
    ):
        raise ValueError("legacy valid S1 feature entries are non-finite")
    if names == STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES and np.any(
        ~np.isfinite(valid_feature_rows[:, :3])
    ):
        raise ValueError("structured S1 anchor features are non-finite")
    if names in {
        LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES,
        ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
        ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    } and np.any(
        ~np.isfinite(valid_feature_rows[:, :1])
    ):
        raise ValueError("landmark-region prototype S1 anchor feature is non-finite")
    if names in {
        COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    }:
        materialized = _cost_volume_materialized_feature_names(
            metadata, feature_names=names
        )
        columns = np.asarray(
            [index for index, name in enumerate(names) if name in materialized],
            dtype=np.int64,
        )
        if columns.size == 0 or np.any(~np.isfinite(valid_feature_rows[:, columns])):
            raise ValueError("materialized cost-volume S1 features are non-finite")
    return {
        "source_row_indices": rows,
        "query_ids": query_ids,
        "split_names": split_names,
        "xy": xy,
        "candidate_track_ids": tracks,
        "candidate_canonical_rows": canonical,
        "candidate_features": features,
        "candidate_view_valid": views,
        "feature_names": np.asarray(names, dtype=np.str_),
    }, metadata


def _load_proposal_tracks(path: Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as data:
        if "candidate_track_ids" not in data.files:
            raise ValueError("proposals lack candidate_track_ids")
        tracks = np.asarray(data["candidate_track_ids"], dtype=np.int64)
    if tracks.ndim != 2 or tracks.shape[0] == 0:
        raise ValueError("proposal candidate tracks are invalid")
    return tracks


def _load_base_overlay(
    path: Path, *, proposal_tracks: np.ndarray, proposals_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    required = {"candidate_track_ids", "candidate_probabilities", "null_probabilities"}
    with np.load(Path(path), allow_pickle=False) as data:
        if set(key for key in data.files if key != "metadata_json") != required:
            raise ValueError("base overlay fields differ from the strict contract")
        arrays = {key: np.asarray(data[key]) for key in required}
        metadata = _metadata(data, context="base prior overlay")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "contains_target_errors"
    ) is not False:
        raise ValueError("base overlay is not target-free")
    if str(metadata.get("probability_semantics")) not in {
        "candidate_identity_probability_plus_explicit_null_equals_one",
        GEOMETRIC_PROBABILITY_SEMANTICS,
    }:
        raise ValueError("base overlay has incompatible probability semantics")
    if str(metadata.get("proposals_sha256")) != str(file_sha256_short(proposals_path)):
        raise ValueError("base overlay references different proposals")
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probability = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    if not np.array_equal(tracks, proposal_tracks) or probability.shape != tracks.shape or null.shape != (
        len(tracks),
    ):
        raise ValueError("base overlay rows do not align with proposals")
    valid = tracks >= 0
    if (
        np.any(~np.isfinite(probability))
        or np.any(~np.isfinite(null))
        or np.any(probability < 0.0)
        or np.any(null < 0.0)
        or np.any(np.abs(probability[~valid]) > 1e-6)
    ):
        raise ValueError("base overlay probabilities are invalid")
    mass = probability.sum(axis=1, dtype=np.float64) + null.astype(np.float64)
    if np.max(np.abs(mass - 1.0)) > 1e-4:
        raise ValueError("base overlay does not conserve candidate plus null mass")
    return {
        "candidate_track_ids": tracks,
        "candidate_probabilities": probability,
        "null_probabilities": null,
    }, metadata


def _resolve_families(value: str) -> tuple[str, ...]:
    if str(value).strip() == "all":
        return tuple(ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES)
    requested = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("families must be unique non-empty fixed names")
    unsupported = set(requested) - set(ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES)
    if unsupported:
        raise ValueError(f"unsupported S1 probe families: {sorted(unsupported)}")
    return requested


def _stable_family_seed(seed: int, family: str) -> int:
    """Derive a family seed without making result depend on CLI ordering."""

    digest = hashlib.sha256(str(family).encode("utf8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(seed) + offset) % (2**31 - 1))


def _train_geometric_target_membership(
    *,
    proposals_path: Path,
    source_rows: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    positive_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build train-only valid-candidate sets plus an explicit-null fallback."""

    train_rows = np.flatnonzero(np.asarray(split_names).astype(str) == "train")
    if train_rows.size == 0:
        raise ValueError("S1 feature artifact has no train rows")
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    train_tracks = np.asarray(candidate_tracks, dtype=np.int64)[train_rows]
    with np.load(Path(proposals_path), allow_pickle=False) as data:
        if "candidate_gt_residuals_px" not in data.files:
            raise ValueError("train supervision proposals lack candidate_gt_residuals_px")
        residuals = np.asarray(data["candidate_gt_residuals_px"], dtype=np.float32)[
            rows[train_rows]
        ]
    if residuals.shape != train_tracks.shape or np.any(np.isnan(residuals)) or np.any(
        residuals < 0.0
    ):
        raise ValueError("train geometric residuals do not align with candidate rows")
    positive = (
        (train_tracks >= 0)
        & np.isfinite(residuals)
        & (residuals <= float(positive_threshold_px))
    )
    positive_count = np.sum(positive, axis=1)
    membership = np.zeros((len(train_rows), train_tracks.shape[1] + 1), dtype=bool)
    membership[:, :-1] = positive
    membership[positive_count == 0, -1] = True
    membership_count = np.sum(membership, axis=1)
    if np.any(membership_count[positive_count == 0] != 1) or np.any(
        membership_count[positive_count > 0] != positive_count[positive_count > 0]
    ):
        raise RuntimeError("train geometric target membership is invalid")
    return train_rows, membership, {
        "supervision": "query_to_projected_landmark_set_membership_or_explicit_null_v1",
        "training_objective": SET_MEMBERSHIP_OBJECTIVE,
        "geometric_positive_threshold_px": float(positive_threshold_px),
        "train_row_count": int(len(train_rows)),
        "geometry_positive_train_row_count": int(np.sum(positive_count > 0)),
        "geometry_positive_train_row_rate": float(np.mean(positive_count > 0)),
        "multi_positive_train_row_count": int(np.sum(positive_count > 1)),
        "positive_candidate_train_count": int(np.sum(positive)),
        "explicit_null_train_row_count": int(np.sum(positive_count == 0)),
    }


def _train_registered_track_identity_target_membership(
    *,
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    images_by_name: Mapping[str, object],
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build train-only exact-track targets at registered detector anchors.

    A detector token without a nearby registered SfM observation is unlabeled,
    not null.  A registered token whose exact track is absent from frozen top-L
    is a supervised explicit-null example.  This distinction prevents sparse
    SfM coverage from becoming an accidental reject label.
    """

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    tracks = np.asarray(candidate_tracks, dtype=np.int64)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    if (
        tracks.ndim != 2
        or len(ids) != len(xy)
        or len(ids) != len(splits)
        or tracks.shape[0] != len(ids)
        or float(identity_radius_px) <= 0.0
    ):
        raise ValueError("registered-track supervision arrays are incompatible")
    train_rows = np.flatnonzero(splits == "train")
    if train_rows.size == 0:
        raise ValueError("S1 feature artifact has no train rows")
    targets = registered_query_observation_targets(
        query_ids=ids[train_rows],
        query_xy=xy[train_rows],
        images_by_name=images_by_name,
        max_distance_px=float(identity_radius_px),
    )
    labels = registered_candidate_identity_labels(tracks[train_rows], targets)
    membership = registered_candidate_identity_target_membership(
        tracks[train_rows], targets
    )
    supervised_local = np.asarray(targets.supervised, dtype=bool)
    supervised_rows = train_rows[supervised_local]
    supervised_membership = membership[supervised_local]
    if supervised_rows.size == 0:
        raise ValueError("registered-track supervision found no train anchors")
    if np.any(np.sum(supervised_membership, axis=1) != 1):
        raise RuntimeError("registered-track train membership is not singleton-or-null")
    identity_summary = summarize_registered_candidate_identity(labels, targets)
    exact_retrieved = np.any(labels, axis=1)
    return supervised_rows, supervised_membership, {
        "supervision": "registered_query_observation_exact_track_or_explicit_null_v1",
        "training_objective": REGISTERED_TRACK_IDENTITY_OBJECTIVE,
        "registered_identity_radius_px": float(identity_radius_px),
        "train_split_row_count": int(len(train_rows)),
        "registered_supervised_train_row_count": int(np.sum(supervised_local)),
        "registered_supervised_train_row_rate": float(np.mean(supervised_local)),
        "unsupervised_train_row_count": int(np.sum(~supervised_local)),
        "exact_track_retrieved_train_row_count": int(
            np.sum(supervised_local & exact_retrieved)
        ),
        "exact_track_retrieved_given_registered_train_rate": (
            float(np.mean(exact_retrieved[supervised_local]))
            if np.any(supervised_local)
            else None
        ),
        "explicit_null_registered_train_row_count": int(
            np.sum(supervised_local & ~exact_retrieved)
        ),
        "positive_candidate_train_count": int(np.sum(labels)),
        "registered_identity_target_coverage": identity_summary,
    }


def _save_model(
    *,
    path: Path,
    model: torch.nn.Module,
    normalizer: PerViewFeatureNormalizer,
    family: str,
    metadata: Mapping[str, object],
) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "format": MODEL_FORMAT,
            "family": str(family),
            "state_dict": state,
            "normalizer": {
                "mean": normalizer.mean,
                "scale": normalizer.scale,
                "feature_indices": normalizer.feature_indices,
            },
            "metadata": dict(metadata),
        },
        path,
    )


def _replace_overlay_rows(
    *,
    base: Mapping[str, np.ndarray],
    source_rows: np.ndarray,
    source_tracks: np.ndarray,
    probabilities: np.ndarray,
    null_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    tracks = np.asarray(source_tracks, dtype=np.int64)
    candidate = np.asarray(probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    base_tracks = np.asarray(base["candidate_track_ids"], dtype=np.int64)
    if (
        tracks.shape != candidate.shape
        or candidate.shape != (len(rows), base_tracks.shape[1])
        or null.shape != (len(rows),)
        or np.unique(rows).size != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= len(base_tracks))
        or not np.array_equal(base_tracks[rows], tracks)
    ):
        raise ValueError("probe prediction rows do not align with the base overlay")
    valid = tracks >= 0
    if np.any(candidate[~valid] != 0.0):
        raise ValueError("invalid candidates received learned probability mass")
    mass = candidate.sum(axis=1, dtype=np.float64) + null.astype(np.float64)
    maximum_mass_error = float(np.max(np.abs(mass - 1.0)))
    if maximum_mass_error > 1e-5:
        raise ValueError("learned candidate and null probabilities do not conserve mass")
    output_candidate = np.asarray(base["candidate_probabilities"], dtype=np.float32).copy()
    output_null = np.asarray(base["null_probabilities"], dtype=np.float32).copy()
    output_candidate[rows] = candidate
    output_null[rows] = null
    output_mass = output_candidate.sum(axis=1, dtype=np.float64) + output_null.astype(
        np.float64
    )
    return output_candidate, output_null, {
        "maximum_replaced_row_mass_error": maximum_mass_error,
        "maximum_full_overlay_mass_error": float(np.max(np.abs(output_mass - 1.0))),
    }


def fit_multiscale_candidate_probe(
    *,
    features_path: Path,
    proposals_path: Path,
    base_overlay_path: Path,
    output_dir: Path,
    families: Sequence[str],
    geometric_positive_threshold_px: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    allow_diagnostic_feature_artifact: bool,
    prior_residual: bool,
    architecture: str,
    hidden_dim: int,
    supervision_mode: str = GEOMETRIC_SET_SUPERVISION_MODE,
    colmap_model_dir: Path | None = None,
    registered_identity_radius_px: float = 2.0,
) -> dict[str, object]:
    mode = str(supervision_mode)
    if (
        float(geometric_positive_threshold_px) <= 0.0
        or float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or int(hidden_dim) <= 0
        or mode not in SUPPORTED_SUPERVISION_MODES
    ):
        raise ValueError("S1 probe optimization arguments are invalid")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    arrays, feature_metadata = _load_features(features_path)
    if (
        int(feature_metadata.get("diagnostic_max_queries", 0)) > 0
        or int(feature_metadata.get("diagnostic_max_rows", 0)) > 0
    ) and not bool(allow_diagnostic_feature_artifact):
        raise ValueError("refusing to fit from a diagnostic feature artifact")
    _validate_cost_volume_family_materialization(
        feature_metadata,
        feature_names=tuple(str(value) for value in arrays["feature_names"].tolist()),
        families=tuple(str(family) for family in families),
    )
    proposal_tracks = _load_proposal_tracks(proposals_path)
    rows = arrays["source_row_indices"]
    if np.any(rows < 0) or np.any(rows >= len(proposal_tracks)) or not np.array_equal(
        proposal_tracks[rows], arrays["candidate_track_ids"]
    ):
        raise ValueError("S1 source rows do not align with proposal tracks")
    if str(feature_metadata.get("proposals_sha256")) != str(
        file_sha256_short(proposals_path)
    ):
        raise ValueError("S1 feature artifact references different proposals")
    base, base_metadata = _load_base_overlay(
        base_overlay_path,
        proposal_tracks=proposal_tracks,
        proposals_path=proposals_path,
    )
    if mode == GEOMETRIC_SET_SUPERVISION_MODE:
        train_rows, train_target_membership, target_audit = _train_geometric_target_membership(
            proposals_path=proposals_path,
            source_rows=rows,
            candidate_tracks=arrays["candidate_track_ids"],
            split_names=arrays["split_names"],
            positive_threshold_px=float(geometric_positive_threshold_px),
        )
        train_groups = arrays["split_names"] == "train"
        if not np.array_equal(train_rows, np.flatnonzero(train_groups)):
            raise RuntimeError("train target rows differ from the frozen feature split")
        probability_semantics = GEOMETRIC_PROBABILITY_SEMANTICS
        probability_role = "geometric_correspondence_not_exact_track_identity"
        training_objective = SET_MEMBERSHIP_OBJECTIVE
    else:
        if colmap_model_dir is None or not str(colmap_model_dir).strip():
            raise ValueError("registered-track supervision requires --colmap_model_dir")
        model_dir = Path(colmap_model_dir)
        images_path = model_dir / "images.bin"
        if not images_path.is_file():
            raise FileNotFoundError(
                f"registered-track supervision is missing {images_path}"
            )
        images = read_colmap_images_binary(images_path)
        images_by_name = {str(image.image_name): image for image in images.values()}
        train_rows, train_target_membership, target_audit = (
            _train_registered_track_identity_target_membership(
                query_ids=arrays["query_ids"],
                query_xy=arrays["xy"],
                candidate_tracks=arrays["candidate_track_ids"],
                split_names=arrays["split_names"],
                images_by_name=images_by_name,
                identity_radius_px=float(registered_identity_radius_px),
            )
        )
        train_groups = np.zeros((len(rows),), dtype=bool)
        train_groups[train_rows] = True
        if np.any(arrays["split_names"][train_rows] != "train"):
            raise RuntimeError("registered-track targets include a held-out feature row")
        probability_semantics = EXACT_IDENTITY_PROBABILITY_SEMANTICS
        probability_role = "exact_registered_track_identity_or_explicit_null"
        training_objective = REGISTERED_TRACK_IDENTITY_OBJECTIVE
        target_audit["colmap_images_sha256"] = file_sha256_short(images_path)
    target_audit["supervision_mode"] = mode
    all_target_membership = np.zeros(
        (len(rows), arrays["candidate_track_ids"].shape[1] + 1), dtype=bool
    )
    all_target_membership[:, -1] = True
    all_target_membership[train_rows] = train_target_membership
    target_device = torch.device(str(device))
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {target_device}")
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "models"
    overlay_dir = output_dir / "overlays"
    model_dir.mkdir()
    overlay_dir.mkdir()
    predictions: list[np.ndarray] = []
    null_predictions: list[np.ndarray] = []
    per_view_logits: list[np.ndarray] = []
    fit_rows: list[dict[str, object]] = []
    family_names = tuple(str(family) for family in families)
    for family in family_names:
        family_seed = _stable_family_seed(int(seed), family)
        model, normalizer, fit_metadata = train_per_view_candidate_probe(
            features=arrays["candidate_features"],
            view_valid=arrays["candidate_view_valid"],
            train_groups=train_groups,
            target_membership=all_target_membership,
            base_candidate_probabilities=(
                base["candidate_probabilities"][rows] if bool(prior_residual) else None
            ),
            base_null_probabilities=(
                base["null_probabilities"][rows] if bool(prior_residual) else None
            ),
            family=family,
            device=target_device,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            seed=family_seed,
            architecture=str(architecture),
            hidden_dim=int(hidden_dim),
            feature_names=tuple(str(value) for value in arrays["feature_names"].tolist()),
        )
        candidate_probability, null_probability, view_logits = predict_per_view_candidate_probe(
            model,
            features=arrays["candidate_features"],
            view_valid=arrays["candidate_view_valid"],
            normalizer=normalizer,
            device=target_device,
            batch_size=max(int(batch_size), 1),
            base_candidate_probabilities=(
                base["candidate_probabilities"][rows] if bool(prior_residual) else None
            ),
            base_null_probabilities=(
                base["null_probabilities"][rows] if bool(prior_residual) else None
            ),
        )
        if not np.allclose(
            candidate_probability.sum(axis=1) + null_probability, 1.0, atol=1e-5
        ):
            raise RuntimeError("per-view probe prediction does not conserve probability mass")
        predictions.append(candidate_probability)
        null_predictions.append(null_probability)
        per_view_logits.append(view_logits)
        model_metadata = {
            "features_sha256": file_sha256_short(features_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "training_supervision_split": "train",
            "validation_or_test_labels_used_by_fit": False,
            "target_audit": target_audit,
            "supervision_mode": mode,
            "training_objective": training_objective,
            "probability_semantics": probability_semantics,
            "candidate_probability_role": probability_role,
            "base_prior_residual": bool(prior_residual),
            "architecture": str(architecture),
            "hidden_dim": (None if str(architecture) == "linear" else int(hidden_dim)),
            "fit": fit_metadata,
        }
        _save_model(
            path=model_dir / f"{family}.pt",
            model=model,
            normalizer=normalizer,
            family=family,
            metadata=model_metadata,
        )
        fit_rows.append({"family": family, "supervision_mode": mode, **fit_metadata})
    candidate_probability_tensor = np.stack(predictions, axis=0).astype(np.float32)
    null_probability_tensor = np.stack(null_predictions, axis=0).astype(np.float32)
    view_logit_tensor = np.stack(per_view_logits, axis=0).astype(np.float32)
    prediction_metadata = {
        "format": PREDICTION_ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used_for_prediction": False,
        "training_supervision_split": "train",
        "supervision_mode": mode,
        "training_objective": training_objective,
        "validation_or_test_labels_used_by_fit": False,
        "test_used_for_model_selection": False,
        "features_sha256": file_sha256_short(features_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "base_prior_overlay_sha256": file_sha256_short(base_overlay_path),
        "families": list(family_names),
        "probability_semantics": probability_semantics,
        "candidate_probability_role": probability_role,
        "base_prior_residual": bool(prior_residual),
        "architecture": str(architecture),
        "hidden_dim": (None if str(architecture) == "linear" else int(hidden_dim)),
        "source_feature_protocol": {
            "image_retrieval_or_submap_used": feature_metadata.get(
                "image_retrieval_or_submap_used"
            ),
            "whole_image_summary_or_global_used": feature_metadata.get(
                "whole_image_summary_or_global_used"
            ),
            "soft_global_context_factor": feature_metadata.get("soft_global_context_factor"),
            "global_context_usage": feature_metadata.get("global_context_usage"),
            "global_context_hard_retrieval_or_candidate_reselection": feature_metadata.get(
                "global_context_hard_retrieval_or_candidate_reselection"
            ),
            "candidate_anchor_conditioned_spatial_grid_only": feature_metadata.get(
                "candidate_anchor_conditioned_spatial_grid_only"
            ),
            "cost_volume": feature_metadata.get("cost_volume"),
            "support_view_selection": feature_metadata.get("support_view_selection"),
        },
    }
    prediction_path = output_dir / "predictions_inference_only.npz"
    np.savez_compressed(
        prediction_path,
        source_row_indices=arrays["source_row_indices"],
        query_ids=arrays["query_ids"],
        split_names=arrays["split_names"],
        candidate_track_ids=arrays["candidate_track_ids"],
        candidate_view_valid=arrays["candidate_view_valid"],
        family_names=np.asarray(family_names, dtype=np.str_),
        candidate_probabilities=candidate_probability_tensor,
        null_probabilities=null_probability_tensor,
        per_view_logits=view_logit_tensor,
        metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
    )
    overlay_rows = []
    for family_index, family in enumerate(family_names):
        candidate, null, overlay_audit = _replace_overlay_rows(
            base=base,
            source_rows=arrays["source_row_indices"],
            source_tracks=arrays["candidate_track_ids"],
            probabilities=candidate_probability_tensor[family_index],
            null_probabilities=null_probability_tensor[family_index],
        )
        overlay_metadata = {
            "format": OVERLAY_ARTIFACT_FORMAT,
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "probability_semantics": probability_semantics,
            "candidate_probability_role": probability_role,
            "base_prior_residual": bool(prior_residual),
            "architecture": str(architecture),
            "hidden_dim": (None if str(architecture) == "linear" else int(hidden_dim)),
            "proposals_sha256": file_sha256_short(proposals_path),
            "features_sha256": file_sha256_short(features_path),
            "predictions_sha256": file_sha256_short(prediction_path),
            "base_prior_overlay_sha256": file_sha256_short(base_overlay_path),
            "family": family,
            "training_supervision_split": "train",
            "supervision_mode": mode,
            "training_objective": training_objective,
            "validation_or_test_labels_used_by_fit": False,
            "replaced_source_row_count": int(len(rows)),
            "replaced_source_rows_sha256": _array_sha256_short(rows),
            "audit": overlay_audit,
        }
        overlay_path = overlay_dir / f"{family}.npz"
        np.savez_compressed(
            overlay_path,
            candidate_track_ids=base["candidate_track_ids"],
            candidate_probabilities=candidate,
            null_probabilities=null,
            metadata_json=np.asarray(json.dumps(overlay_metadata, sort_keys=True)),
        )
        overlay_rows.append(
            {
                "family": family,
                "path": str(overlay_path),
                "sha256": file_sha256_short(overlay_path),
                "audit": overlay_audit,
            }
        )
    summary = {
        "stage": "train_only_frozen_multiscale_per_view_candidate_probe",
        "protocol": {
            "training_supervision_split": "train",
            "train_target": target_audit["supervision"],
            "supervision_mode": mode,
            "training_objective": training_objective,
            "probability_semantics": probability_semantics,
            "candidate_probability_role": probability_role,
            "base_prior_residual": bool(prior_residual),
            "architecture": str(architecture),
            "hidden_dim": (None if str(architecture) == "linear" else int(hidden_dim)),
            "validation_or_test_labels_used_by_fit": False,
            "test_used_for_model_selection": False,
            "image_retrieval": False,
            "render": False,
        },
        "target_audit": target_audit,
        "families": list(family_names),
        "fits": fit_rows,
        "inputs": {
            "features": str(features_path),
            "features_sha256": file_sha256_short(features_path),
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "base_prior_overlay": str(base_overlay_path),
            "base_prior_overlay_sha256": file_sha256_short(base_overlay_path),
            "base_prior_format": base_metadata.get("format"),
            "base_prior_probability_semantics": base_metadata.get("probability_semantics"),
            "colmap_model_dir": (
                None
                if mode == GEOMETRIC_SET_SUPERVISION_MODE
                else str(Path(colmap_model_dir))
            ),
        },
        "outputs": {
            "predictions": str(prediction_path),
            "predictions_sha256": file_sha256_short(prediction_path),
            "models": str(model_dir),
            "overlays": overlay_rows,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_multiscale_candidate_probe(
        features_path=Path(args.features),
        proposals_path=Path(args.proposals),
        base_overlay_path=Path(args.base_prior_overlay),
        output_dir=Path(args.output_dir),
        families=_resolve_families(args.families),
        geometric_positive_threshold_px=float(args.geometric_positive_threshold_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        device=str(args.device),
        allow_diagnostic_feature_artifact=bool(args.allow_diagnostic_feature_artifact),
        prior_residual=bool(args.prior_residual),
        architecture=str(args.architecture),
        hidden_dim=int(args.hidden_dim),
        supervision_mode=str(args.supervision_mode),
        colmap_model_dir=(
            None if not str(args.colmap_model_dir).strip() else Path(args.colmap_model_dir)
        ),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
