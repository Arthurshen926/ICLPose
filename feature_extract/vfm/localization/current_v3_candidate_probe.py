"""Strict current-V3 contracts for frozen candidate-appearance probes.

The feature exporter is intentionally target-free.  This module keeps that
boundary explicit: it loads only inference-time candidate evidence until a
caller has frozen its prediction artifact, then exposes split-specific target
joins for either train fitting or validation audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapImageObservation,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_candidate_identity_target_membership,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


CURRENT_V3_EVIDENCE_FORMAT = "candidate_evidence_v3"
CURRENT_V3_PREDICTION_FORMAT = "current_v3_multisource_candidate_probe_predictions_v1"
CURRENT_V3_MODEL_FORMAT = "current_v3_multisource_per_view_candidate_probe_v1"
CURRENT_V3_FEATURE_FORMAT = MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT
CURRENT_V3_POSE_OVERLAY_FORMAT = "current_v3_multisource_candidate_probe_prior_overlay_v1"
GEOMETRIC_PROBABILITY_SEMANTICS = (
    "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one"
)
EXACT_IDENTITY_PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
GEOMETRIC_SET_SUPERVISION_MODE = "geometric_set"
REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE = "registered_track_identity"
SUPPORTED_SUPERVISION_MODES = frozenset(
    {GEOMETRIC_SET_SUPERVISION_MODE, REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE}
)
SET_MEMBERSHIP_OBJECTIVE = "set_log_mass_nll_over_target_membership_v1"
REGISTERED_TRACK_IDENTITY_OBJECTIVE = (
    "registered_query_observation_exact_track_or_explicit_null_nll_v1"
)
_ALLOWED_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class CurrentV3Features:
    path: Path
    source_rows: np.ndarray
    query_ids: np.ndarray
    split_names: np.ndarray
    xy: np.ndarray
    candidate_tracks: np.ndarray
    candidate_canonical_rows: np.ndarray
    candidate_features: np.ndarray
    candidate_view_valid: np.ndarray
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CurrentV3EvidenceInference:
    path: Path
    source_rows: np.ndarray
    query_ids: np.ndarray
    xy: np.ndarray
    split_names: np.ndarray
    candidate_valid: np.ndarray
    candidate_tracks: np.ndarray
    candidate_bank_rows: np.ndarray
    candidate_priors: np.ndarray
    unknown_probability: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CurrentV3AlignedInference:
    features: CurrentV3Features
    evidence: CurrentV3EvidenceInference
    evidence_rows: np.ndarray
    base_candidate_probabilities: np.ndarray
    base_null_probabilities: np.ndarray


def metadata_from_npz(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} lacks metadata_json")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata_json must contain an object")
    return metadata


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _load_arrays(path: Path, *, keys: set[str], context: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = keys - set(payload.files)
        if missing:
            raise ValueError(f"{context} lacks {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in keys}
        metadata = metadata_from_npz(payload, context=context)
    return arrays, metadata


def _feature_contract(metadata: Mapping[str, Any]) -> None:
    if metadata.get("format") != CURRENT_V3_FEATURE_FORMAT:
        raise ValueError("current-V3 probe needs a multi-source frozen-layout feature artifact")
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("whole_image_summary_or_global_used", True))
        or bool(metadata.get("render", True))
        or metadata.get("is_complete_frozen_layout") is not True
    ):
        raise ValueError("current-V3 feature artifact violates the frozen visual-evidence protocol")


def load_current_v3_features(path: Path) -> CurrentV3Features:
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
    arrays, metadata = _load_arrays(Path(path), keys=required, context="current-V3 features")
    _feature_contract(metadata)
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    canonical = np.asarray(arrays["candidate_canonical_rows"], dtype=np.int64)
    features = np.asarray(arrays["candidate_features"])
    view_valid = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    names = tuple(str(value) for value in np.asarray(arrays["feature_names"]).tolist())
    if (
        len(rows) == 0
        or np.unique(rows).size != len(rows)
        or not (query_ids.shape == splits.shape == (len(rows),))
        or xy.shape != (len(rows), 2)
        or tracks.ndim != 2
        or canonical.shape != tracks.shape
        or features.ndim != 4
        or features.shape[:3] != view_valid.shape
        or features.shape[:2] != tracks.shape
        or features.shape[0] != len(rows)
        or features.shape[3] != len(names)
        or not names
    ):
        raise ValueError("current-V3 frozen feature arrays are not aligned")
    if set(splits.tolist()) - _ALLOWED_SPLITS or not np.any(splits == "train") or not np.any(
        splits == "validation"
    ):
        raise ValueError("current-V3 frozen features must contain only train and validation rows")
    candidate_valid = tracks >= 0
    if np.any(candidate_valid & (canonical < 0)) or np.any(~candidate_valid & (canonical >= 0)):
        raise ValueError("current-V3 feature candidate canonical rows are invalid")
    if np.any(candidate_valid & ~np.any(view_valid, axis=2)) or np.any(
        ~candidate_valid & np.any(view_valid, axis=2)
    ):
        raise ValueError("current-V3 feature candidate support views are invalid")
    if np.any(np.isinf(features[view_valid])):
        raise ValueError("current-V3 feature artifact contains infinite visual evidence")
    return CurrentV3Features(
        path=Path(path),
        source_rows=rows,
        query_ids=query_ids,
        split_names=splits,
        xy=xy,
        candidate_tracks=tracks,
        candidate_canonical_rows=canonical,
        candidate_features=features,
        candidate_view_valid=view_valid,
        feature_names=names,
        metadata=metadata,
    )


def load_current_v3_evidence_inference(path: Path) -> CurrentV3EvidenceInference:
    """Load only target-free V3 fields; labels stay behind the audit boundary."""

    required = {
        "selected_rows",
        "query_ids",
        "query_xy",
        "split_names",
        "candidate_valid",
        "candidate_track_ids",
        "candidate_bank_rows",
        "candidate_prior_probabilities",
        "unknown_probability",
    }
    arrays, metadata = _load_arrays(Path(path), keys=required, context="current-V3 evidence")
    if metadata.get("format") != CURRENT_V3_EVIDENCE_FORMAT:
        raise ValueError("current-V3 evidence has an unsupported format")
    if (
        bool(metadata.get("pose_used_for_selection", True))
        or bool(metadata.get("image_retrieval", True))
        or bool(metadata.get("render", True))
        or metadata.get("candidate_probability_semantics")
        != "factorized_top_l_availability_times_conditional_identity"
    ):
        raise ValueError("current-V3 evidence violates the fixed global no-render protocol")
    rows = np.asarray(arrays["selected_rows"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["query_xy"], dtype=np.float32)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    bank_rows = np.asarray(arrays["candidate_bank_rows"], dtype=np.int64)
    priors = np.asarray(arrays["candidate_prior_probabilities"], dtype=np.float32)
    unknown = np.asarray(arrays["unknown_probability"], dtype=np.float32).reshape(-1)
    if (
        len(rows) == 0
        or np.unique(rows).size != len(rows)
        or not (query_ids.shape == splits.shape == unknown.shape == (len(rows),))
        or xy.shape != (len(rows), 2)
        or not (valid.shape == tracks.shape == bank_rows.shape == priors.shape)
        or valid.shape[0] != len(rows)
        or set(splits.tolist()) - {"train", "validation", "test"}
        or np.any(valid != (tracks >= 0))
        or np.any(valid & (bank_rows < 0))
        or np.any(~valid & (bank_rows >= 0))
        or np.any(~np.isfinite(priors))
        or np.any(priors < 0.0)
        or np.any(~np.isfinite(unknown))
        or np.any(unknown <= 0.0)
        or np.any(priors[~valid] != 0.0)
    ):
        raise ValueError("current-V3 target-free evidence arrays are invalid")
    mass = priors.sum(axis=1, dtype=np.float64) + unknown.astype(np.float64)
    if np.max(np.abs(mass - 1.0)) > 3e-5:
        raise ValueError("current-V3 candidate and unknown prior mass is not conserved")
    return CurrentV3EvidenceInference(
        path=Path(path),
        source_rows=rows,
        query_ids=query_ids,
        xy=xy,
        split_names=splits,
        candidate_valid=valid,
        candidate_tracks=tracks,
        candidate_bank_rows=bank_rows,
        candidate_priors=priors,
        unknown_probability=unknown,
        metadata=metadata,
    )


def align_current_v3_features_and_evidence(
    features: CurrentV3Features, evidence: CurrentV3EvidenceInference
) -> CurrentV3AlignedInference:
    """Prove that target-free visual features use the same frozen V3 candidates."""

    if str(features.metadata.get("maplet_support_index_sha256", "")) != str(
        evidence.metadata.get("maplet_support_index_sha256", "")
    ):
        raise ValueError("current-V3 feature and evidence maplet provenance differs")
    if str(features.metadata.get("proposals_sha256", "")) != str(
        evidence.metadata.get("proposals_sha256", "")
    ):
        raise ValueError("current-V3 feature and evidence proposal provenance differs")
    layout_path_value = str(features.metadata.get("source_frozen_layout", ""))
    if not layout_path_value:
        raise ValueError("current-V3 feature artifact lacks its frozen-layout source")
    layout_path = Path(layout_path_value)
    if not layout_path.is_file():
        raise FileNotFoundError(f"current-V3 frozen layout is missing: {layout_path}")
    if str(features.metadata.get("source_frozen_layout_sha256", "")) != str(
        file_sha256_short(layout_path)
    ):
        raise ValueError("current-V3 feature artifact references a stale frozen layout")
    with np.load(layout_path, allow_pickle=False) as payload:
        layout_metadata = metadata_from_npz(payload, context="current-V3 frozen layout")
    if str(layout_metadata.get("candidate_evidence_sha256", "")) != str(
        file_sha256_short(evidence.path)
    ):
        raise ValueError("current-V3 frozen layout and candidate evidence differ")
    lookup = {int(row): index for index, row in enumerate(evidence.source_rows.tolist())}
    if len(lookup) != len(evidence.source_rows):
        raise ValueError("current-V3 evidence source rows are duplicated")
    try:
        evidence_rows = np.asarray(
            [lookup[int(row)] for row in features.source_rows.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("current-V3 visual feature row is missing from evidence") from error
    if not (
        np.array_equal(features.query_ids, evidence.query_ids[evidence_rows])
        and np.array_equal(features.split_names, evidence.split_names[evidence_rows])
        and np.allclose(features.xy, evidence.xy[evidence_rows], rtol=0.0, atol=1e-4)
        and np.array_equal(features.candidate_tracks, evidence.candidate_tracks[evidence_rows])
        and np.array_equal(
            features.candidate_canonical_rows,
            np.where(
                evidence.candidate_valid[evidence_rows],
                evidence.candidate_bank_rows[evidence_rows],
                -1,
            ),
        )
        and np.array_equal(
            np.any(features.candidate_view_valid, axis=2), evidence.candidate_valid[evidence_rows]
        )
    ):
        raise ValueError("current-V3 visual features do not match frozen evidence identities")
    return CurrentV3AlignedInference(
        features=features,
        evidence=evidence,
        evidence_rows=evidence_rows,
        base_candidate_probabilities=evidence.candidate_priors[evidence_rows],
        base_null_probabilities=evidence.unknown_probability[evidence_rows],
    )


def _load_current_v3_predictions_for_pose_overlay(
    *,
    predictions_path: Path,
    features: CurrentV3Features,
    candidate_evidence_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Load frozen train-only probe predictions without joining any targets."""

    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "candidate_track_ids",
        "candidate_canonical_rows",
        "candidate_view_valid",
        "family_names",
        "candidate_probabilities",
        "null_probabilities",
    }
    arrays, metadata = _load_arrays(
        Path(predictions_path), keys=required, context="current-V3 pose overlay predictions"
    )
    supervision_mode = str(
        metadata.get("supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE)
    )
    if supervision_mode not in SUPPORTED_SUPERVISION_MODES:
        raise ValueError("current-V3 prediction has an unsupported supervision mode")
    expected_probability_semantics = (
        GEOMETRIC_PROBABILITY_SEMANTICS
        if supervision_mode == GEOMETRIC_SET_SUPERVISION_MODE
        else EXACT_IDENTITY_PROBABILITY_SEMANTICS
    )
    if (
        metadata.get("format") != CURRENT_V3_PREDICTION_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("probability_semantics") != expected_probability_semantics
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("render", True))
    ):
        raise ValueError("current-V3 predictions violate the frozen pose-overlay protocol")
    if str(metadata.get("features_sha256", "")) != str(file_sha256_short(features.path)):
        raise ValueError("current-V3 predictions reference different visual features")
    if str(metadata.get("candidate_evidence_sha256", "")) != str(
        file_sha256_short(Path(candidate_evidence_path))
    ):
        raise ValueError("current-V3 predictions reference different candidate evidence")
    for key, expected in (
        ("source_row_indices", features.source_rows),
        ("query_ids", features.query_ids),
        ("split_names", features.split_names),
        ("candidate_track_ids", features.candidate_tracks),
        ("candidate_canonical_rows", features.candidate_canonical_rows),
        ("candidate_view_valid", features.candidate_view_valid),
    ):
        if not np.array_equal(np.asarray(arrays[key]), expected):
            raise ValueError(f"current-V3 prediction {key} differs from frozen features")
    families = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    unknown = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    if (
        not families
        or len(set(families)) != len(families)
        or candidate.shape != (len(families), *features.candidate_tracks.shape)
        or unknown.shape != (len(families), len(features.source_rows))
    ):
        raise ValueError("current-V3 pose-overlay prediction probabilities are invalid")
    for family_index in range(len(families)):
        validate_candidate_probability_contract(
            candidate[family_index],
            unknown[family_index],
            features.candidate_tracks >= 0,
        )
    return candidate, unknown, families, metadata


def _load_current_v3_base_prior_overlay(
    path: Path,
    *,
    evidence: CurrentV3EvidenceInference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the proposal-row base posterior and prove V3 compact alignment."""

    required = {
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
    }
    arrays, metadata = _load_arrays(
        Path(path), keys=required, context="current-V3 base prior overlay"
    )
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or str(metadata.get("proposals_sha256", ""))
        != str(evidence.metadata.get("proposals_sha256", ""))
    ):
        raise ValueError("current-V3 base overlay violates proposal provenance")
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    unknown = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    if (
        tracks.ndim != 2
        or candidate.shape != tracks.shape
        or unknown.shape != (tracks.shape[0],)
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(unknown))
        or np.any(candidate < 0.0)
        or np.any(unknown < 0.0)
    ):
        raise ValueError("current-V3 base overlay arrays are invalid")
    valid = tracks >= 0
    if np.any(candidate[~valid] != 0.0):
        raise ValueError("current-V3 base overlay gives invalid candidates mass")
    mass = candidate.sum(axis=1, dtype=np.float64) + unknown.astype(np.float64)
    if np.max(np.abs(mass - 1.0)) > 1e-4:
        raise ValueError("current-V3 base overlay does not conserve mass")
    return tracks, candidate, unknown, metadata


def _current_v3_source_columns(
    path: Path,
    *,
    evidence: CurrentV3EvidenceInference,
) -> np.ndarray:
    """Read compact-to-proposal columns, which are not part of probe features."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if "candidate_source_columns" not in payload.files:
            raise ValueError("current-V3 evidence lacks candidate source columns")
        source_columns = np.asarray(payload["candidate_source_columns"], dtype=np.int64)
    if source_columns.shape != evidence.candidate_tracks.shape:
        raise ValueError("current-V3 candidate source columns are misaligned")
    if not np.array_equal(source_columns >= 0, evidence.candidate_valid):
        raise ValueError("current-V3 candidate source-column validity differs from evidence")
    return source_columns


def _validate_current_v3_base_mapping(
    *,
    evidence: CurrentV3EvidenceInference,
    source_columns: np.ndarray,
    base_tracks: np.ndarray,
    base_candidate: np.ndarray,
    base_unknown: np.ndarray,
) -> None:
    """Require compact V3 candidates to cover every proposal candidate exactly once."""

    rows = np.asarray(evidence.source_rows, dtype=np.int64)
    columns = np.asarray(source_columns, dtype=np.int64)
    valid = np.asarray(evidence.candidate_valid, dtype=bool)
    if (
        np.any(rows < 0)
        or np.any(rows >= len(base_tracks))
        or np.any(columns[valid] >= base_tracks.shape[1])
    ):
        raise ValueError("current-V3 evidence refers outside the base prior overlay")
    safe_columns = np.maximum(columns, 0)
    base_rows = np.broadcast_to(rows[:, None], columns.shape)
    mapped_tracks = base_tracks[base_rows, safe_columns].copy()
    mapped_candidate = base_candidate[base_rows, safe_columns].copy()
    mapped_tracks[~valid] = -1
    mapped_candidate[~valid] = 0.0
    if not np.array_equal(mapped_tracks, evidence.candidate_tracks):
        raise ValueError("current-V3 evidence tracks do not map to base overlay columns")
    if not np.allclose(
        mapped_candidate, evidence.candidate_priors, rtol=0.0, atol=2e-5
    ) or not np.allclose(
        base_unknown[rows], evidence.unknown_probability, rtol=0.0, atol=2e-5
    ):
        raise ValueError("current-V3 evidence priors differ from the base overlay")
    base_valid = base_tracks[rows] >= 0
    covered = np.zeros_like(base_valid, dtype=bool)
    compact_rows = np.broadcast_to(
        np.arange(len(rows), dtype=np.int64)[:, None], columns.shape
    )
    covered[compact_rows[valid], columns[valid]] = True
    if not np.array_equal(covered, base_valid):
        raise ValueError(
            "current-V3 evidence must cover every valid proposal candidate for a pose overlay"
        )


def build_current_v3_validation_pose_overlay(
    *,
    features_path: Path,
    candidate_evidence_path: Path,
    predictions_path: Path,
    base_prior_overlay_path: Path,
    family: str,
    output_path: Path,
) -> dict[str, Any]:
    """Overlay one frozen probe only on validation proposal rows.

    Train rows are deliberately left at the base posterior despite having model
    predictions, and test rows were never materialized by the visual probe.
    This makes the artifact safe for a validation-only pose-rank audit.
    """

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite current-V3 pose overlay: {output}")
    features = load_current_v3_features(Path(features_path))
    evidence = load_current_v3_evidence_inference(Path(candidate_evidence_path))
    aligned = align_current_v3_features_and_evidence(features, evidence)
    prediction, predicted_unknown, families, prediction_metadata = (
        _load_current_v3_predictions_for_pose_overlay(
            predictions_path=Path(predictions_path),
            features=features,
            candidate_evidence_path=Path(candidate_evidence_path),
        )
    )
    if str(family) not in families:
        raise ValueError(f"current-V3 prediction has no requested family: {family}")
    family_index = families.index(str(family))
    base_tracks, base_candidate, base_unknown, base_metadata = (
        _load_current_v3_base_prior_overlay(
            Path(base_prior_overlay_path), evidence=evidence
        )
    )
    source_columns = _current_v3_source_columns(
        Path(candidate_evidence_path), evidence=evidence
    )
    _validate_current_v3_base_mapping(
        evidence=evidence,
        source_columns=source_columns,
        base_tracks=base_tracks,
        base_candidate=base_candidate,
        base_unknown=base_unknown,
    )
    validation_mask = features.split_names == "validation"
    if not np.any(validation_mask):
        raise ValueError("current-V3 pose overlay has no validation prediction rows")
    if np.any(features.split_names == "test"):
        raise ValueError("current-V3 pose overlay refuses materialized test predictions")
    update_evidence_rows = aligned.evidence_rows[validation_mask]
    update_rows = features.source_rows[validation_mask]
    update_columns = source_columns[update_evidence_rows]
    update_valid = features.candidate_tracks[validation_mask] >= 0
    update_candidate = prediction[family_index, validation_mask]
    update_unknown = predicted_unknown[family_index, validation_mask]
    if not np.array_equal(update_valid, update_columns >= 0):
        raise RuntimeError("current-V3 validation update columns differ from prediction validity")
    output_candidate = np.array(base_candidate, copy=True)
    output_unknown = np.array(base_unknown, copy=True)
    proposal_rows = np.broadcast_to(update_rows[:, None], update_columns.shape)
    output_candidate[proposal_rows[update_valid], update_columns[update_valid]] = (
        update_candidate[update_valid]
    )
    output_unknown[update_rows] = update_unknown
    output_valid = base_tracks >= 0
    output_mass = output_candidate.sum(axis=1, dtype=np.float64) + output_unknown.astype(
        np.float64
    )
    maximum_probability_mass_error = float(np.max(np.abs(output_mass - 1.0)))
    if maximum_probability_mass_error > 1e-4:
        raise RuntimeError("current-V3 pose overlay no longer conserves probability mass")
    unchanged = np.ones((len(base_tracks),), dtype=bool)
    unchanged[update_rows] = False
    if not (
        np.array_equal(output_candidate[unchanged], base_candidate[unchanged])
        and np.array_equal(output_unknown[unchanged], base_unknown[unchanged])
        and np.all(output_candidate[~output_valid] == 0.0)
    ):
        raise RuntimeError("current-V3 pose overlay changed non-validation proposal rows")
    metadata: dict[str, Any] = {
        "format": CURRENT_V3_POSE_OVERLAY_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used_for_prediction": False,
        "probability_semantics": prediction_metadata["probability_semantics"],
        "proposals_sha256": evidence.metadata.get("proposals_sha256"),
        "candidate_evidence": str(Path(candidate_evidence_path)),
        "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence_path)),
        "features": str(Path(features_path)),
        "features_sha256": file_sha256_short(Path(features_path)),
        "predictions": str(Path(predictions_path)),
        "predictions_sha256": file_sha256_short(Path(predictions_path)),
        "base_prior_overlay": str(Path(base_prior_overlay_path)),
        "base_prior_overlay_sha256": file_sha256_short(Path(base_prior_overlay_path)),
        "base_prior_format": base_metadata.get("format"),
        "family": str(family),
        "family_index": int(family_index),
        "training_supervision_split": prediction_metadata.get("training_supervision_split"),
        "training_supervision_mode": prediction_metadata.get(
            "supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE
        ),
        "training_objective": prediction_metadata.get("training_objective"),
        "validation_or_test_labels_used_by_fit": False,
        "fixed_global_top_l": True,
        "candidate_retrieval_or_reselection": False,
        "candidate_compact_to_proposal_mapping": "candidate_source_columns_exact_cover_v1",
        "updated_split_names": ["validation"],
        "updated_source_row_count": int(len(update_rows)),
        "updated_source_rows_sha256": _array_sha256_short(update_rows),
        "train_prediction_rows_overwritten": False,
        "test_prediction_rows_materialized": False,
        "nonvalidation_base_rows_unchanged": True,
        "maximum_probability_mass_error": maximum_probability_mass_error,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_track_ids=base_tracks,
            candidate_probabilities=output_candidate.astype(np.float32),
            null_probabilities=output_unknown.astype(np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "build_current_v3_validation_pose_overlay",
        "protocol": {
            "fixed_global_top_l": True,
            "validation_rows_only": True,
            "train_rows_left_at_base_posterior": True,
            "test_rows_not_materialized": True,
            "candidate_retrieval_or_reselection": False,
            "pose_or_ground_truth_used": False,
            "render": False,
        },
        "inputs": metadata,
        "outputs": {
            "overlay": str(output),
            "overlay_sha256": file_sha256_short(output),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def train_geometric_target_membership(
    evidence_path: Path,
    aligned: CurrentV3AlignedInference,
    *,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Join V3 residual labels for train rows only after feature validation."""

    if float(threshold_px) <= 0.0:
        raise ValueError("current-V3 geometric positive threshold must be positive")
    train_mask = aligned.features.split_names == "train"
    train_rows = np.flatnonzero(train_mask)
    if len(train_rows) == 0:
        raise ValueError("current-V3 frozen features have no train rows")
    with np.load(Path(evidence_path), allow_pickle=False) as payload:
        metadata = metadata_from_npz(payload, context="current-V3 train evidence")
        if metadata.get("format") != CURRENT_V3_EVIDENCE_FORMAT:
            raise ValueError("current-V3 train labels have an unsupported evidence format")
        if "candidate_target_gt_residuals_px" not in payload.files:
            raise ValueError("current-V3 evidence lacks target residuals")
        # This is the sole training-label read.  Only rows selected by the
        # frozen train mask below are used to form the supervision tensor.
        residuals = np.asarray(payload["candidate_target_gt_residuals_px"], dtype=np.float32)
    selected = residuals[aligned.evidence_rows[train_rows]]
    tracks = aligned.features.candidate_tracks[train_rows]
    valid = aligned.evidence.candidate_valid[aligned.evidence_rows[train_rows]]
    if (
        selected.shape != tracks.shape
        or not np.array_equal(valid, tracks >= 0)
        or np.any(np.isnan(selected))
        or np.any(selected[valid] < 0.0)
    ):
        raise ValueError("current-V3 train target residuals are not aligned")
    positive = valid & np.isfinite(selected) & (selected <= float(threshold_px))
    membership = np.zeros((len(train_rows), tracks.shape[1] + 1), dtype=bool)
    membership[:, :-1] = positive
    no_positive = ~np.any(positive, axis=1)
    membership[no_positive, -1] = True
    if np.any(np.sum(membership, axis=1) == 0):
        raise RuntimeError("current-V3 train target membership is empty")
    return train_rows, membership, {
        "target_source": "candidate_target_gt_residuals_px",
        "target_split": "train",
        "positive_threshold_px": float(threshold_px),
        "training_row_count": int(len(train_rows)),
        "positive_training_row_count": int(np.sum(~no_positive)),
        "positive_training_row_rate": float(np.mean(~no_positive)),
        "multi_positive_training_row_count": int(np.sum(np.sum(positive, axis=1) > 1)),
        "positive_candidate_training_count": int(np.sum(positive)),
        "explicit_null_training_row_count": int(np.sum(no_positive)),
        "validation_or_test_target_used": False,
    }


def validation_geometric_labels(
    evidence_path: Path,
    aligned: CurrentV3AlignedInference,
    *,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Join validation residual labels only for an already frozen prediction."""

    if float(threshold_px) <= 0.0:
        raise ValueError("current-V3 validation threshold must be positive")
    validation_mask = aligned.features.split_names == "validation"
    if not np.any(validation_mask):
        raise ValueError("current-V3 frozen features have no validation rows")
    with np.load(Path(evidence_path), allow_pickle=False) as payload:
        metadata = metadata_from_npz(payload, context="current-V3 validation evidence")
        if metadata.get("format") != CURRENT_V3_EVIDENCE_FORMAT:
            raise ValueError("current-V3 validation labels have an unsupported evidence format")
        if "candidate_target_gt_residuals_px" not in payload.files:
            raise ValueError("current-V3 evidence lacks target residuals")
        residuals = np.asarray(payload["candidate_target_gt_residuals_px"], dtype=np.float32)
    selected = residuals[aligned.evidence_rows]
    valid = aligned.evidence.candidate_valid[aligned.evidence_rows]
    if (
        selected.shape != aligned.features.candidate_tracks.shape
        or np.any(np.isnan(selected))
        or np.any(selected[valid] < 0.0)
    ):
        raise ValueError("current-V3 validation target residuals are not aligned")
    labels = valid & np.isfinite(selected) & (selected <= float(threshold_px))
    return validation_mask, labels


def _registered_identity_images(
    colmap_model_dir: Path,
) -> tuple[Path, dict[str, ColmapImageObservation]]:
    """Load SfM observations for a supervised exact-track target join.

    The callers below only consume point2D-to-point3D identities and image
    coordinates.  They deliberately never consume the qvec/tvec fields from
    the parsed model, so query pose is not available to fitting or auditing.
    """

    images_path = Path(colmap_model_dir) / "images.bin"
    if not images_path.is_file():
        raise FileNotFoundError(f"registered-track supervision is missing {images_path}")
    images = read_colmap_images_binary(images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    if len(images_by_name) != len(images):
        raise ValueError("COLMAP registered-track images have duplicate names")
    return images_path, images_by_name


def _registered_identity_subset(
    features: CurrentV3Features,
    *,
    split_name: str,
    images_by_name: Mapping[str, ColmapImageObservation],
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return exact identity labels for exactly one frozen split.

    Anchors without a nearby registered SfM observation are intentionally
    unsupervised.  A registered target outside fixed top-L instead receives the
    explicit null target; it is not silently dropped.
    """

    if str(split_name) not in _ALLOWED_SPLITS:
        raise ValueError(f"current-V3 registered identity split is unsupported: {split_name}")
    if float(identity_radius_px) <= 0.0:
        raise ValueError("current-V3 registered identity radius must be positive")
    subset_rows = np.flatnonzero(features.split_names == str(split_name))
    if subset_rows.size == 0:
        raise ValueError(f"current-V3 frozen features have no {split_name} rows")
    targets = registered_query_observation_targets(
        query_ids=features.query_ids[subset_rows],
        query_xy=features.xy[subset_rows],
        images_by_name=images_by_name,
        max_distance_px=float(identity_radius_px),
    )
    labels = registered_candidate_identity_labels(
        features.candidate_tracks[subset_rows], targets
    )
    membership = registered_candidate_identity_target_membership(
        features.candidate_tracks[subset_rows], targets
    )
    audit = {
        "target_source": "registered_query_observation_point2d_to_point3d_identity",
        "target_split": str(split_name),
        "registered_identity_radius_px": float(identity_radius_px),
        "split_row_count": int(len(subset_rows)),
        "registered_supervised_row_count": int(np.sum(targets.supervised)),
        "registered_supervised_row_rate": float(np.mean(targets.supervised)),
        "unsupervised_row_count": int(np.sum(~targets.supervised)),
        "exact_track_retrieved_row_count": int(np.sum(np.any(labels, axis=1))),
        "explicit_null_registered_row_count": int(
            np.sum(targets.supervised & ~np.any(labels, axis=1))
        ),
        "positive_candidate_count": int(np.sum(labels)),
        "registered_identity_target_coverage": summarize_registered_candidate_identity(
            labels, targets
        ),
    }
    return subset_rows, membership, labels, audit


def train_registered_track_identity_membership_from_images(
    features: CurrentV3Features,
    *,
    images_by_name: Mapping[str, ColmapImageObservation],
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build train-only exact-track-or-null targets from SfM observations."""

    rows, membership, _labels, audit = _registered_identity_subset(
        features,
        split_name="train",
        images_by_name=images_by_name,
        identity_radius_px=float(identity_radius_px),
    )
    supervised = np.any(membership, axis=1)
    supervised_rows = rows[supervised]
    supervised_membership = membership[supervised]
    if supervised_rows.size == 0:
        raise ValueError("registered-track supervision found no train anchors")
    if np.any(np.sum(supervised_membership, axis=1) != 1):
        raise RuntimeError("registered-track train membership is not singleton-or-null")
    result = {
        "supervision": "registered_query_observation_exact_track_or_explicit_null_v1",
        "training_objective": REGISTERED_TRACK_IDENTITY_OBJECTIVE,
        "training_supervision_mode": REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
        "train_split_row_count": int(len(rows)),
        "registered_supervised_train_row_count": int(len(supervised_rows)),
        "registered_supervised_train_row_rate": float(np.mean(supervised)),
        "unsupervised_train_row_count": int(np.sum(~supervised)),
        "exact_track_retrieved_train_row_count": int(
            audit["exact_track_retrieved_row_count"]
        ),
        "explicit_null_registered_train_row_count": int(
            audit["explicit_null_registered_row_count"]
        ),
        "positive_candidate_train_count": int(audit["positive_candidate_count"]),
        "registered_identity_target_coverage": audit[
            "registered_identity_target_coverage"
        ],
        "registered_identity_radius_px": float(identity_radius_px),
        "validation_or_test_target_used": False,
    }
    return supervised_rows, supervised_membership, result


def train_registered_track_identity_membership(
    features: CurrentV3Features,
    *,
    colmap_model_dir: Path,
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load train observations and build strict current-V3 identity targets."""

    images_path, images_by_name = _registered_identity_images(Path(colmap_model_dir))
    rows, membership, audit = train_registered_track_identity_membership_from_images(
        features,
        images_by_name=images_by_name,
        identity_radius_px=float(identity_radius_px),
    )
    audit["colmap_images_sha256"] = file_sha256_short(images_path)
    audit["colmap_model_dir"] = str(Path(colmap_model_dir))
    return rows, membership, audit


def validation_registered_track_identity_labels_from_images(
    features: CurrentV3Features,
    *,
    images_by_name: Mapping[str, ColmapImageObservation],
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Join validation exact-track labels after target-free prediction freeze."""

    rows, membership, labels, audit = _registered_identity_subset(
        features,
        split_name="validation",
        images_by_name=images_by_name,
        identity_radius_px=float(identity_radius_px),
    )
    supervised = np.any(membership, axis=1)
    row_mask = np.zeros((len(features.source_rows),), dtype=bool)
    full_labels = np.zeros(features.candidate_tracks.shape, dtype=bool)
    row_mask[rows] = supervised
    full_labels[rows] = labels
    if not np.any(row_mask):
        raise ValueError("registered-track validation audit found no supervised anchors")
    audit.update(
        {
            "validation_supervised_row_count": int(np.sum(row_mask)),
            "validation_supervised_row_rate": float(np.mean(row_mask[rows])),
            "prediction_frozen_before_validation_target_join": True,
            "test_target_used": False,
        }
    )
    return row_mask, full_labels, audit


def validation_registered_track_identity_labels(
    features: CurrentV3Features,
    *,
    colmap_model_dir: Path,
    identity_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load validation observations only after the prediction artifact is frozen."""

    images_path, images_by_name = _registered_identity_images(Path(colmap_model_dir))
    mask, labels, audit = validation_registered_track_identity_labels_from_images(
        features,
        images_by_name=images_by_name,
        identity_radius_px=float(identity_radius_px),
    )
    audit["colmap_images_sha256"] = file_sha256_short(images_path)
    audit["colmap_model_dir"] = str(Path(colmap_model_dir))
    return mask, labels, audit


def validate_candidate_probability_contract(
    candidate: np.ndarray, null: np.ndarray, valid: np.ndarray
) -> None:
    values = np.asarray(candidate, dtype=np.float64)
    unknown = np.asarray(null, dtype=np.float64).reshape(-1)
    candidate_valid = np.asarray(valid, dtype=bool)
    if (
        values.shape != candidate_valid.shape
        or unknown.shape != (len(values),)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(unknown))
        or np.any(values < 0.0)
        or np.any(unknown <= 0.0)
        or np.any(values[~candidate_valid] != 0.0)
    ):
        raise ValueError("current-V3 candidate probability contract is invalid")
    mass = values.sum(axis=1) + unknown
    if np.max(np.abs(mass - 1.0)) > 3e-5:
        raise ValueError("current-V3 candidate probability mass is not conserved")


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positive.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("current-V3 average-precision inputs are invalid")
    count = int(np.sum(positive))
    if count == 0:
        return None
    ranked = positive[np.argsort(-values, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / count)


def candidate_top_and_first_positive_rank(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool).copy()
    candidate_valid = np.asarray(valid, dtype=bool)
    if values.shape != positive.shape or values.shape != candidate_valid.shape:
        raise ValueError("current-V3 candidate rank inputs are incompatible")
    if np.any(~np.isfinite(values[candidate_valid])):
        raise ValueError("current-V3 valid candidate score is non-finite")
    ordered = np.argsort(
        np.where(candidate_valid, -values, np.inf), axis=1, kind="stable"
    )
    top = ordered[:, 0]
    ranked_positive = np.take_along_axis(positive, ordered, axis=1)
    ranked_valid = np.take_along_axis(candidate_valid, ordered, axis=1)
    first = np.full((len(values),), -1, dtype=np.int64)
    for column in range(values.shape[1]):
        take = (first < 0) & ranked_valid[:, column] & ranked_positive[:, column]
        first[take] = column + 1
    return top, first


def set_valued_metrics(
    candidate: np.ndarray,
    null: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(candidate, dtype=np.float64)
    unknown = np.asarray(null, dtype=np.float64).reshape(-1)
    positive = np.asarray(labels, dtype=bool).copy()
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1)
    if not (
        values.shape == positive.shape == candidate_valid.shape
        and unknown.shape == selected.shape == values.shape[:1]
    ):
        raise ValueError("current-V3 set metrics inputs are incompatible")
    if not np.any(selected):
        raise ValueError("current-V3 set metrics has no selected rows")
    positive &= candidate_valid
    target_mass = np.sum(np.where(positive, values, 0.0), axis=1)
    no_positive = ~np.any(positive, axis=1)
    target_mass[no_positive] = unknown[no_positive]
    top, ranks = candidate_top_and_first_positive_rank(values, positive, candidate_valid)
    edge_mask = selected[:, None] & candidate_valid
    positive_rows = selected & ~no_positive
    return {
        "row_count": int(np.sum(selected)),
        "positive_row_count": int(np.sum(positive_rows)),
        "explicit_null_row_count": int(np.sum(selected & no_positive)),
        "group_target_nll": float(np.mean(-np.log(np.clip(target_mass[selected], 1e-12, 1.0)))),
        "candidate_pair_average_precision": average_precision(
            positive[edge_mask], values[edge_mask]
        ),
        "top1_geometry_valid_rate": (
            None
            if not np.any(positive_rows)
            else float(np.mean(positive[positive_rows, top[positive_rows]]))
        ),
        "mean_correct_probability_mass_when_present": (
            None
            if not np.any(positive_rows)
            else float(np.mean(target_mass[positive_rows]))
        ),
        "median_first_positive_rank": (
            None if not np.any(positive_rows) else float(np.median(ranks[positive_rows]))
        ),
        "p90_first_positive_rank": (
            None
            if not np.any(positive_rows)
            else float(np.quantile(ranks[positive_rows], 0.9))
        ),
        "positive_top_l_recall": float(np.mean(~no_positive[selected])),
    }


def paired_rank_audit(
    baseline: np.ndarray,
    probe: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    base = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(probe, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1)
    if not (
        base.shape == candidate.shape == positive.shape == candidate_valid.shape
        and selected.shape == base.shape[:1]
    ):
        raise ValueError("current-V3 paired rank inputs are incompatible")
    _, base_rank = candidate_top_and_first_positive_rank(base, positive, candidate_valid)
    _, probe_rank = candidate_top_and_first_positive_rank(candidate, positive, candidate_valid)
    eligible = selected & (base_rank > 0)
    if not np.any(eligible):
        return {
            "eligible_positive_row_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "median_rank_delta_baseline_minus_probe": None,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
        }
    return {
        "eligible_positive_row_count": int(np.sum(eligible)),
        "rank_win_count": int(np.sum(probe_rank[eligible] < base_rank[eligible])),
        "rank_loss_count": int(np.sum(probe_rank[eligible] > base_rank[eligible])),
        "rank_tie_count": int(np.sum(probe_rank[eligible] == base_rank[eligible])),
        "median_rank_delta_baseline_minus_probe": float(
            np.median(base_rank[eligible] - probe_rank[eligible])
        ),
        "top1_rescue_count": int(
            np.sum((base_rank[eligible] > 1) & (probe_rank[eligible] == 1))
        ),
        "top1_harm_count": int(
            np.sum((base_rank[eligible] == 1) & (probe_rank[eligible] > 1))
        ),
    }


def rank2_to_l_rescue_audit(
    baseline: np.ndarray,
    probe: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    base = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(probe, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1)
    _, base_rank = candidate_top_and_first_positive_rank(base, positive, candidate_valid)
    eligible = selected & (base_rank >= 2)
    if not np.any(eligible):
        return {
            "eligible_positive_row_count": 0,
            "baseline": None,
            "probe": None,
            "paired_rank": paired_rank_audit(
                base, candidate, labels=positive, valid=candidate_valid, row_mask=eligible
            ),
        }
    baseline_metrics = set_valued_metrics(
        base,
        np.full((len(base),), 1e-12, dtype=np.float64),
        labels=positive,
        valid=candidate_valid,
        row_mask=eligible,
    )
    probe_metrics = set_valued_metrics(
        candidate,
        np.full((len(candidate),), 1e-12, dtype=np.float64),
        labels=positive,
        valid=candidate_valid,
        row_mask=eligible,
    )
    return {
        "eligible_positive_row_count": int(np.sum(eligible)),
        "baseline": baseline_metrics,
        "probe": probe_metrics,
        "paired_rank": paired_rank_audit(
            base, candidate, labels=positive, valid=candidate_valid, row_mask=eligible
        ),
    }


def stable_family_seed(seed: int, family: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{str(family)}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)
