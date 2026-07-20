"""Strict contracts for mixed-point multi-scale candidate probes.

The mixed verification points are deliberately independent of the detector
rows used to fit pose hypotheses.  This module keeps the supervision boundary
equally explicit: all visual features and predictions are target-free; SfM
reprojection targets are materialized only for a requested train or validation
split after candidate predictions have been frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.current_v3_candidate_probe import (
    validate_candidate_probability_contract,
)
from feature_extract.vfm.localization.detector_landmark_proposals import (
    candidate_reprojection_residuals,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    MixedVerificationPoints,
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ABSOLUTE_GLOBAL_TRANSPORT_CANDIDATE_PROBE_FAMILIES,
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_candidate_identity_target_membership,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


MIXED_MULTISCALE_FROZEN_LAYOUT_FORMAT = (
    "mixed_multiscale_verification_frozen_candidate_layout_v1"
)
MIXED_MULTISCALE_MODEL_FORMAT = "mixed_multiscale_per_view_candidate_probe_v1"
MIXED_MULTISCALE_PREDICTION_FORMAT = "mixed_multiscale_candidate_probe_predictions_v1"
MIXED_GEOMETRIC_PROBABILITY_SEMANTICS = (
    "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one"
)
MIXED_EXACT_IDENTITY_PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
MIXED_GEOMETRIC_SET_SUPERVISION_MODE = "geometric_set"
MIXED_REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE = "registered_track_identity"
MIXED_SUPPORTED_SUPERVISION_MODES = frozenset(
    {MIXED_GEOMETRIC_SET_SUPERVISION_MODE, MIXED_REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE}
)
_ALLOWED_SPLITS = frozenset({"train", "validation"})

# A mixed-point probe can use only a family declared for the frozen feature
# schema that produced its per-view tensor.  This prevents a local-region
# family from accidentally being applied to a full-image transport tensor.
MIXED_MULTISCALE_LOCAL_REGION_FAMILIES = (
    "multisource_landmark_region_radio_intermediate_appearance_only",
    "multisource_landmark_region_alike_appearance_only",
    "multisource_landmark_region_candidate_specific_appearance_only",
    "multisource_landmark_region_with_anchor_appearance_only",
)
MIXED_ABSOLUTE_GLOBAL_TRANSPORT_FAMILIES = tuple(
    str(value) for value in ABSOLUTE_GLOBAL_TRANSPORT_CANDIDATE_PROBE_FAMILIES
)
MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT: Mapping[str, tuple[str, ...]] = {
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT: (
        MIXED_MULTISCALE_LOCAL_REGION_FAMILIES
    ),
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT: (
        MIXED_ABSOLUTE_GLOBAL_TRANSPORT_FAMILIES
    ),
}


@dataclass(frozen=True)
class MixedMultiscaleCandidateProbeFeatures:
    """Target-free point/candidate/support-view evidence with fixed priors."""

    path: Path
    source_point_ids: np.ndarray
    query_ids: np.ndarray
    split_names: np.ndarray
    point_sources: np.ndarray
    xy: np.ndarray
    candidate_tracks: np.ndarray
    candidate_bank_rows: np.ndarray
    candidate_features: np.ndarray
    candidate_view_valid: np.ndarray
    feature_names: tuple[str, ...]
    base_candidate_probabilities: np.ndarray
    base_null_probabilities: np.ndarray
    points: MixedVerificationPoints
    metadata: Mapping[str, Any]


def metadata_from_npz(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} lacks metadata_json")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata_json must contain an object")
    return metadata


def _load_layout_metadata(path: Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return metadata_from_npz(payload, context="mixed frozen layout")


def load_mixed_multiscale_candidate_probe_features(
    *, features_path: Path, verification_points_path: Path
) -> MixedMultiscaleCandidateProbeFeatures:
    """Load multi-scale evidence and prove it matches immutable point proposals."""

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
    with np.load(Path(features_path), allow_pickle=False) as payload:
        if "labels" in payload.files:
            raise ValueError("mixed multi-scale feature artifact unexpectedly contains labels")
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"mixed multi-scale features lack {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        metadata = metadata_from_npz(payload, context="mixed multi-scale features")
    feature_format = str(metadata.get("format", ""))
    if (
        feature_format not in MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("whole_image_summary_or_global_used", True))
        or bool(metadata.get("render", True))
        or metadata.get("is_complete_frozen_layout") is not True
    ):
        raise ValueError("mixed multi-scale feature artifact violates the frozen visual protocol")
    if feature_format == ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT and (
        metadata.get("feature_definition")
        != "masked_full_image_candidate_specific_absolute_transport_v1"
        or metadata.get("full_image_spatial_layout_used") is not True
    ):
        raise ValueError("mixed absolute transport artifact has an invalid visual contract")
    layout_value = str(metadata.get("source_frozen_layout", ""))
    if not layout_value:
        raise ValueError("mixed multi-scale features lack frozen-layout provenance")
    layout_path = Path(layout_value)
    if not layout_path.is_file():
        raise FileNotFoundError(f"mixed frozen layout is missing: {layout_path}")
    if str(metadata.get("source_frozen_layout_sha256", "")) != str(
        file_sha256_short(layout_path)
    ):
        raise ValueError("mixed multi-scale features reference a stale frozen layout")
    layout_metadata = _load_layout_metadata(layout_path)
    if layout_metadata.get("format") != MIXED_MULTISCALE_FROZEN_LAYOUT_FORMAT:
        raise ValueError("mixed multi-scale features were not built from a mixed frozen layout")

    points = load_mixed_verification_points(Path(verification_points_path))
    points_sha = file_sha256_short(Path(verification_points_path))
    if (
        str(points.metadata.get("format", "")) != MIXED_VERIFICATION_POINTS_FORMAT
        or str(layout_metadata.get("verification_points_sha256", "")) != points_sha
        or str(layout_metadata.get("projected_landmark_bank_sha256", ""))
        != str(points.metadata.get("projected_landmark_bank_sha256", ""))
    ):
        raise ValueError("mixed frozen layout and verification points differ")

    source_ids = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    bank_rows = np.asarray(arrays["candidate_canonical_rows"], dtype=np.int64)
    raw_features = np.asarray(arrays["candidate_features"])
    view_valid = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    feature_names = tuple(str(value) for value in np.asarray(arrays["feature_names"]).tolist())
    count = len(source_ids)
    if (
        count == 0
        or len(np.unique(source_ids)) != count
        or not (query_ids.shape == split_names.shape == (count,))
        or xy.shape != (count, 2)
        or tracks.ndim != 2
        or bank_rows.shape != tracks.shape
        or raw_features.ndim != 4
        or raw_features.shape[:3] != view_valid.shape
        or raw_features.shape[:2] != tracks.shape
        or raw_features.shape[0] != count
        or raw_features.shape[3] != len(feature_names)
        or not feature_names
        or set(split_names.tolist()) - _ALLOWED_SPLITS
    ):
        raise ValueError("mixed multi-scale feature arrays are not aligned")
    if not (
        np.array_equal(source_ids, points.source_point_ids)
        and np.array_equal(query_ids, points.query_ids)
        and np.array_equal(split_names, points.split_names)
        and np.allclose(xy, points.xy, rtol=0.0, atol=1e-4)
        and np.array_equal(tracks, points.candidate_track_ids)
        and np.array_equal(bank_rows, points.candidate_bank_rows)
    ):
        raise ValueError("mixed multi-scale features differ from frozen verification points")
    candidate_valid = tracks >= 0
    if (
        np.any(candidate_valid & (bank_rows < 0))
        or np.any(~candidate_valid & (bank_rows >= 0))
        or np.any(candidate_valid & ~np.any(view_valid, axis=2))
        or np.any(~candidate_valid & np.any(view_valid, axis=2))
        or np.any(np.isinf(raw_features[view_valid]))
    ):
        raise ValueError("mixed multi-scale candidate/support-view layout is invalid")
    base_candidate = np.asarray(points.candidate_prior_probabilities, dtype=np.float32)
    base_null = np.asarray(points.null_probabilities, dtype=np.float32)
    validate_candidate_probability_contract(base_candidate, base_null, candidate_valid)
    return MixedMultiscaleCandidateProbeFeatures(
        path=Path(features_path),
        source_point_ids=source_ids,
        query_ids=query_ids,
        split_names=split_names,
        point_sources=np.asarray(points.point_sources).astype(str),
        xy=xy,
        candidate_tracks=tracks,
        candidate_bank_rows=bank_rows,
        candidate_features=raw_features,
        candidate_view_valid=view_valid,
        feature_names=feature_names,
        base_candidate_probabilities=base_candidate,
        base_null_probabilities=base_null,
        points=points,
        metadata=metadata,
    )


def _pose_w2c(image: object) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def materialize_mixed_candidate_reprojection_residuals(
    *,
    features: MixedMultiscaleCandidateProbeFeatures,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    row_indices: np.ndarray,
    required_split: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Join only one requested split's geometric targets after feature freeze."""

    selected = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if (
        not len(selected)
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= len(features.source_point_ids))
        or str(required_split) not in _ALLOWED_SPLITS
        or np.any(features.split_names[selected] != str(required_split))
    ):
        raise ValueError("mixed reprojection target rows do not match the requested split")
    bank, bank_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
    if str(features.points.metadata.get("projected_landmark_bank_sha256", "")) != str(
        file_sha256_short(Path(projected_landmark_bank))
    ):
        raise ValueError("mixed points reference a different projected landmark bank")
    if str(features.points.metadata.get("descriptor_space_id", "")) != str(
        bank_metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("mixed points and projected landmark bank descriptor spaces differ")
    valid = features.candidate_tracks >= 0
    safe_rows = np.maximum(features.candidate_bank_rows, 0)
    if np.any(bank.track_ids[safe_rows[valid]] != features.candidate_tracks[valid]):
        raise ValueError("mixed candidate tracks differ from the projected landmark bank")
    model_dir = Path(colmap_model_dir)
    cameras_path = model_dir / "cameras.bin"
    images_path = model_dir / "images.bin"
    cameras = read_colmap_cameras_binary(cameras_path)
    images = read_colmap_images_binary(images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    output = np.full(
        (len(selected), features.candidate_tracks.shape[1]), np.inf, dtype=np.float32
    )
    local_row = {int(row): index for index, row in enumerate(selected.tolist())}
    for query_id in sorted(set(features.query_ids[selected].tolist())):
        image = images_by_name.get(str(query_id))
        if image is None:
            raise KeyError(f"COLMAP model is missing mixed query image {query_id}")
        global_rows = selected[features.query_ids[selected] == str(query_id)]
        local_rows = np.asarray([local_row[int(row)] for row in global_rows], dtype=np.int64)
        output[local_rows] = candidate_reprojection_residuals(
            features.xy[global_rows],
            features.candidate_bank_rows[global_rows],
            bank,
            _pose_w2c(image),
            cameras[int(image.camera_id)],
        )
    if np.any(np.isnan(output)) or np.any(output[valid[selected]] < 0.0):
        raise RuntimeError("mixed candidate reprojection target materialization is invalid")
    return output, {
        "target_source": "sfm_candidate_reprojection_residual_px",
        "target_split": str(required_split),
        "target_row_count": int(len(selected)),
        "projected_landmark_bank": str(Path(projected_landmark_bank)),
        "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
        "colmap_model_dir": str(model_dir),
        "colmap_cameras_sha256": file_sha256_short(cameras_path),
        "colmap_images_sha256": file_sha256_short(images_path),
    }


def geometric_membership_from_residuals(
    *, residuals: np.ndarray, candidate_valid: np.ndarray, threshold_px: float
) -> np.ndarray:
    """Return set-valued candidate positives plus an explicit null target."""

    values = np.asarray(residuals, dtype=np.float32)
    valid = np.asarray(candidate_valid, dtype=bool)
    if (
        values.ndim != 2
        or values.shape != valid.shape
        or not np.isfinite(float(threshold_px))
        or float(threshold_px) <= 0.0
        or np.any(np.isnan(values))
        or np.any(values[valid] < 0.0)
    ):
        raise ValueError("mixed geometric membership inputs are invalid")
    membership = np.zeros((len(values), values.shape[1] + 1), dtype=bool)
    membership[:, :-1] = valid & np.isfinite(values) & (values <= float(threshold_px))
    membership[~np.any(membership[:, :-1], axis=1), -1] = True
    if np.any(~np.any(membership, axis=1)):
        raise RuntimeError("mixed geometric membership has an empty group")
    return membership


def _registered_identity_images(
    colmap_model_dir: Path,
) -> tuple[Path, Mapping[str, object]]:
    images_path = Path(colmap_model_dir) / "images.bin"
    if not images_path.is_file():
        raise FileNotFoundError(f"registered identity supervision is missing {images_path}")
    images = read_colmap_images_binary(images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    if len(images_by_name) != len(images):
        raise ValueError("registered identity supervision has duplicate image names")
    return images_path, images_by_name


def materialize_mixed_registered_identity_membership(
    *,
    features: MixedMultiscaleCandidateProbeFeatures,
    colmap_model_dir: Path,
    split_name: str,
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Join one split's registered point2D-to-point3D identities after freeze.

    No camera pose is consumed here: this is a direct real-image observation
    identity target.  Anchors without a nearby registered observation remain
    unsupervised rather than being silently labelled as null.
    """

    if str(split_name) not in _ALLOWED_SPLITS or float(identity_radius_px) <= 0.0:
        raise ValueError("mixed registered identity split/radius is invalid")
    split_rows = np.flatnonzero(features.split_names == str(split_name))
    if not len(split_rows):
        raise ValueError(f"mixed features have no {split_name} rows")
    images_path, images_by_name = _registered_identity_images(Path(colmap_model_dir))
    targets = registered_query_observation_targets(
        query_ids=features.query_ids[split_rows],
        query_xy=features.xy[split_rows],
        images_by_name=images_by_name,
        max_distance_px=float(identity_radius_px),
    )
    labels = registered_candidate_identity_labels(
        features.candidate_tracks[split_rows], targets
    )
    membership = registered_candidate_identity_target_membership(
        features.candidate_tracks[split_rows], targets
    )
    supervised = np.asarray(targets.supervised, dtype=bool)
    supervised_rows = split_rows[supervised]
    supervised_membership = membership[supervised]
    supervised_labels = labels[supervised]
    if not len(supervised_rows) or np.any(np.sum(supervised_membership, axis=1) != 1):
        raise ValueError("mixed registered identity found no valid singleton-or-null targets")
    identity_summary = summarize_registered_candidate_identity(labels, targets)
    return supervised_rows, supervised_membership, supervised_labels, {
        "target_source": "registered_query_observation_point2d_to_point3d_identity",
        "target_split": str(split_name),
        "registered_identity_radius_px": float(identity_radius_px),
        "split_row_count": int(len(split_rows)),
        "registered_supervised_row_count": int(np.sum(supervised)),
        "registered_supervised_row_rate": float(np.mean(supervised)),
        "unsupervised_row_count": int(np.sum(~supervised)),
        "exact_track_retrieved_row_count": int(np.sum(np.any(labels, axis=1))),
        "explicit_null_registered_row_count": int(
            np.sum(supervised & ~np.any(labels, axis=1))
        ),
        "positive_candidate_count": int(np.sum(labels)),
        "registered_identity_target_coverage": identity_summary,
        "colmap_model_dir": str(Path(colmap_model_dir)),
        "colmap_images_sha256": file_sha256_short(images_path),
    }


def candidate_probe_gate(
    baseline: Mapping[str, Any],
    probe: Mapping[str, Any],
    paired: Mapping[str, Any],
    *,
    top1_metric_key: str = "top1_geometry_valid_rate",
    target_name: str = "geometry",
) -> dict[str, Any]:
    """Conservative candidate-level promotion gate before any pose experiment."""

    baseline_top1 = baseline.get(str(top1_metric_key))
    probe_top1 = probe.get(str(top1_metric_key))
    nll_improved = float(probe["group_target_nll"]) < float(baseline["group_target_nll"])
    top1_improved = (
        baseline_top1 is not None
        and probe_top1 is not None
        and float(probe_top1) > float(baseline_top1)
    )
    rank_improved = int(paired["rank_win_count"]) > int(paired["rank_loss_count"])
    rescue_improved = int(paired["top1_rescue_count"]) > int(paired["top1_harm_count"])
    top1_gate_field = (
        "top1_geometry_valid_rate_improved"
        if str(target_name) == "geometry"
        else f"top1_{str(target_name)}_rate_improved"
    )
    return {
        "nll_improved": bool(nll_improved),
        top1_gate_field: bool(top1_improved),
        "paired_rank_wins_exceed_losses": bool(rank_improved),
        "top1_rescues_exceed_harms": bool(rescue_improved),
        "passed": bool(nll_improved and top1_improved and rank_improved and rescue_improved),
        "policy": (
            "validation-only all-four gate; pass is necessary but not sufficient "
            "for frozen pose-rank evaluation"
        ),
    }


def exact_identity_metric_names(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Avoid labelling strict observation identity metrics as geometry."""

    result = dict(metrics)
    result["top1_exact_identity_rate"] = result.pop("top1_geometry_valid_rate")
    result["mean_exact_track_probability_mass_when_present"] = result.pop(
        "mean_correct_probability_mass_when_present"
    )
    result["median_exact_track_rank"] = result.pop("median_first_positive_rank")
    result["p90_exact_track_rank"] = result.pop("p90_first_positive_rank")
    result["exact_track_top_l_recall"] = result.pop("positive_top_l_recall")
    return result


def load_mixed_multiscale_candidate_predictions(
    *, path: Path, features: MixedMultiscaleCandidateProbeFeatures
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Load a frozen target-free prediction artifact and prove its lineage."""

    required = {
        "source_point_ids",
        "query_ids",
        "split_names",
        "point_sources",
        "candidate_track_ids",
        "candidate_bank_rows",
        "candidate_view_valid",
        "family_names",
        "candidate_probabilities",
        "null_probabilities",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"mixed multi-scale predictions lack {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        metadata = metadata_from_npz(payload, context="mixed multi-scale predictions")
    if (
        metadata.get("format") != MIXED_MULTISCALE_PREDICTION_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("supervision_mode", MIXED_GEOMETRIC_SET_SUPERVISION_MODE)
        not in MIXED_SUPPORTED_SUPERVISION_MODES
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("render", True))
        or str(metadata.get("features_sha256", "")) != str(file_sha256_short(features.path))
    ):
        raise ValueError("mixed multi-scale predictions violate the frozen validation protocol")
    supervision_mode = str(
        metadata.get("supervision_mode", MIXED_GEOMETRIC_SET_SUPERVISION_MODE)
    )
    expected_semantics = (
        MIXED_GEOMETRIC_PROBABILITY_SEMANTICS
        if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE
        else MIXED_EXACT_IDENTITY_PROBABILITY_SEMANTICS
    )
    if metadata.get("probability_semantics") != expected_semantics:
        raise ValueError("mixed multi-scale prediction probability semantics are invalid")
    expected_arrays = {
        "source_point_ids": features.source_point_ids,
        "query_ids": features.query_ids,
        "split_names": features.split_names,
        "point_sources": features.point_sources,
        "candidate_track_ids": features.candidate_tracks,
        "candidate_bank_rows": features.candidate_bank_rows,
        "candidate_view_valid": features.candidate_view_valid,
    }
    for key, expected in expected_arrays.items():
        if not np.array_equal(np.asarray(arrays[key]), expected):
            raise ValueError(f"mixed multi-scale prediction {key} differs from frozen features")
    families = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    if (
        not families
        or len(set(families)) != len(families)
        or candidate.shape != (len(families), *features.candidate_tracks.shape)
        or null.shape != (len(families), len(features.source_point_ids))
    ):
        raise ValueError("mixed multi-scale prediction probability arrays are invalid")
    valid = features.candidate_tracks >= 0
    for index in range(len(families)):
        validate_candidate_probability_contract(candidate[index], null[index], valid)
    return candidate, null, families, metadata
