"""Calibrate whether a target-free RGB measurement should update a 2D point.

The calibrated probability is an action gate.  It is not a landmark identity
likelihood and must never be multiplied into the candidate identity posterior.
All inference features are computed from frozen RGB spatial predictions and map
support metadata.  Pose-derived targets are joined only by the fit/audit paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.candidate_rgb_spatial_inference import (
    SPATIAL_INFERENCE_FORMAT,
)


UTILITY_MODEL_FORMAT = "candidate_measurement_utility_gate_v2"
LEGACY_UTILITY_MODEL_FORMAT = "candidate_measurement_utility_gate_v1"
UTILITY_APPLY_STAGE = "candidate_measurement_utility_apply"
UTILITY_FEATURE_SET = "independent_rgb_geometry_v1"
UTILITY_FEATURE_NAMES = (
    "log_track_length",
    "available_support_view_mass",
    "accepted_rgb_view_mass",
    "weighted_normalized_local_entropy",
    "log_weighted_covariance_trace_px2",
    "weighted_local_peak_probability",
    "weighted_local_peak_margin",
    "mixture_peak_probability",
    "mixture_normalized_entropy",
    "proposed_offset_norm_px",
    "log_posterior_mean_disagreement_px2",
    "log_view_mode_disagreement_px2",
    "log_support_view_count",
    "support_view_prior_max",
    "support_view_prior_entropy",
    "weighted_support_reprojection_error",
    "weighted_log_absolute_frame_gap",
)

_SPATIAL_ARRAY_SCHEMA = {
    "source_row_indices",
    "supervision_source_row_indices",
    "query_ids",
    "source_query_rows",
    "candidate_identity_keys",
    "candidate_measurement_cache_keys",
    "candidate_measurement_ranks",
    "candidate_track_ids",
    "candidate_prototype_ids",
    "support_image_ids",
    "support_view_ranks",
    "support_view_probabilities",
    "candidate_prior_probabilities",
    "center_xy",
    "offsets_xy",
    "local_log_probabilities",
    "dustbin_probabilities",
    "likelihood_entropy",
    "likelihood_covariance_trace_px2",
    "measurement_geometry_probabilities",
}


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    text = str(row.get(key, "")).strip()
    value = float(default) if not text else float(text)
    if not math.isfinite(value):
        raise ValueError(f"candidate measurement row has non-finite {key}")
    return value


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class CandidateMeasurementUtilityGate:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    update_threshold: float
    minimum_baseline_residual_px: float
    minimum_improvement_px: float
    measurement_checkpoint_sha256: str
    candidate_evidence_sha256: str
    candidate_inference_evidence_sha256: str
    coordinate_space_id: str
    offsets_sha256: str
    coordinate_proposal_policy: str = "posterior_mean"

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        if self.feature_names != UTILITY_FEATURE_NAMES:
            raise ValueError("candidate measurement utility feature schema mismatch")
        if any(
            len(values) != count
            for values in (self.feature_mean, self.feature_scale, self.coefficients)
        ):
            raise ValueError("candidate measurement utility vector lengths differ")
        if any(float(value) <= 0.0 for value in self.feature_scale):
            raise ValueError("candidate measurement utility scales must be positive")
        if not 0.0 <= float(self.update_threshold) <= 1.0:
            raise ValueError("candidate measurement utility threshold is invalid")
        if str(self.coordinate_proposal_policy) not in {
            "posterior_mean",
            "mixture_map",
        }:
            raise ValueError("unsupported measurement coordinate proposal policy")
        if float(self.minimum_baseline_residual_px) < 0.0:
            raise ValueError("minimum baseline residual must be non-negative")
        if float(self.minimum_improvement_px) < 0.0:
            raise ValueError("minimum improvement must be non-negative")
        for name, value in (
            ("measurement checkpoint", self.measurement_checkpoint_sha256),
            ("candidate evidence", self.candidate_evidence_sha256),
            ("candidate inference evidence", self.candidate_inference_evidence_sha256),
            ("coordinate space", self.coordinate_space_id),
            ("offset grid", self.offsets_sha256),
        ):
            if len(str(value)) < 8:
                raise ValueError(f"candidate measurement utility lacks {name} identity")

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(
            -1, len(self.feature_names)
        )
        standardized = (
            values - np.asarray(self.feature_mean, dtype=np.float64)[None]
        ) / np.asarray(self.feature_scale, dtype=np.float64)[None]
        base_logit = (
            standardized @ np.asarray(self.coefficients, dtype=np.float64)
            + float(self.intercept)
        )
        calibrated_logit = (
            float(self.calibration_slope) * base_logit
            + float(self.calibration_intercept)
        )
        return _sigmoid(calibrated_logit)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": UTILITY_MODEL_FORMAT,
            "feature_set": UTILITY_FEATURE_SET,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "update_threshold": float(self.update_threshold),
            "minimum_baseline_residual_px": float(
                self.minimum_baseline_residual_px
            ),
            "minimum_improvement_px": float(self.minimum_improvement_px),
            "measurement_checkpoint_sha256": self.measurement_checkpoint_sha256,
            "candidate_evidence_sha256": self.candidate_evidence_sha256,
            "candidate_inference_evidence_sha256": (
                self.candidate_inference_evidence_sha256
            ),
            "coordinate_space_id": self.coordinate_space_id,
            "offsets_sha256": self.offsets_sha256,
            "coordinate_proposal_policy": self.coordinate_proposal_policy,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "CandidateMeasurementUtilityGate":
        if payload.get("format") not in {
            UTILITY_MODEL_FORMAT,
            LEGACY_UTILITY_MODEL_FORMAT,
        }:
            raise ValueError("unsupported candidate measurement utility model")
        if payload.get("feature_set") != UTILITY_FEATURE_SET:
            raise ValueError("unsupported candidate measurement utility feature set")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            calibration_slope=float(payload["calibration_slope"]),
            calibration_intercept=float(payload["calibration_intercept"]),
            update_threshold=float(payload["update_threshold"]),
            minimum_baseline_residual_px=float(
                payload["minimum_baseline_residual_px"]
            ),
            minimum_improvement_px=float(payload["minimum_improvement_px"]),
            measurement_checkpoint_sha256=str(
                payload["measurement_checkpoint_sha256"]
            ),
            candidate_evidence_sha256=str(payload["candidate_evidence_sha256"]),
            candidate_inference_evidence_sha256=str(
                payload["candidate_inference_evidence_sha256"]
            ),
            coordinate_space_id=str(payload["coordinate_space_id"]),
            offsets_sha256=str(payload["offsets_sha256"]),
            coordinate_proposal_policy=str(
                payload.get("coordinate_proposal_policy", "posterior_mean")
            ),
        )


@dataclass(frozen=True)
class _SpatialInput:
    arrays: Mapping[str, np.ndarray]
    offsets_xy: np.ndarray
    observable_rows: tuple[Mapping[str, str], ...]
    metadata: Mapping[str, object]
    split: str
    paths: tuple[Path, ...]
    hashes: tuple[str, ...]


def _validate_spatial_metadata(metadata: Mapping[str, object]) -> None:
    if metadata.get("format") != SPATIAL_INFERENCE_FORMAT:
        raise ValueError("measurement utility requires target-free v7 RGB predictions")
    if (
        bool(metadata.get("contains_ground_truth_arrays"))
        or bool(metadata.get("ground_truth_loaded_by_inference_process"))
        or bool(metadata.get("pose_or_ground_truth_used_for_inference"))
        or not bool(metadata.get("prediction_frozen_before_target_join"))
        or not bool(metadata.get("input_schema_allowlisted"))
    ):
        raise ValueError("RGB spatial artifact does not prove target-free inference")
    if bool(metadata.get("render")) or bool(metadata.get("image_retrieval")) or bool(
        metadata.get("submap")
    ):
        raise ValueError("measurement utility mainline requires real-image global mode")


def _load_target_free_spatial(paths: Sequence[Path]) -> _SpatialInput:
    spatial_paths = tuple(Path(path) for path in paths)
    if not spatial_paths:
        raise ValueError("at least one RGB spatial artifact is required")
    blocks: list[dict[str, np.ndarray]] = []
    hashes: list[str] = []
    offsets: np.ndarray | None = None
    common_metadata: dict[str, object] | None = None
    split: str | None = None
    for path in spatial_paths:
        summary_path = path.parent / "summary.json"
        if not path.exists() or not summary_path.exists():
            raise ValueError("RGB spatial artifact is incomplete")
        summary = json.loads(summary_path.read_text())
        if summary.get("stage") != "independent_rgb_candidate_spatial_target_free_inference":
            raise ValueError("unsupported RGB spatial inference stage")
        artifact_hash = file_sha256_short(path)
        if summary.get("outputs", {}).get("spatial_likelihood_sha256") != artifact_hash:
            raise ValueError("RGB spatial artifact is stale")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != _SPATIAL_ARRAY_SCHEMA | {"metadata_json"}:
                raise ValueError("RGB spatial artifact schema differs from v7 contract")
            metadata = json.loads(str(payload["metadata_json"].item()))
            arrays = {
                key: np.asarray(payload[key])
                for key in payload.files
                if key not in {"metadata_json", "offsets_xy"}
            }
            local_offsets = np.asarray(payload["offsets_xy"], dtype=np.float32)
        _validate_spatial_metadata(metadata)
        forbidden = [
            key
            for key in arrays
            if key.startswith("target_") or "ground_truth" in key.lower()
        ]
        if forbidden:
            raise ValueError(f"RGB spatial artifact exposes targets: {forbidden}")
        local_split = str(summary.get("split", ""))
        if not local_split:
            raise ValueError("RGB spatial summary lacks split identity")
        if split is None:
            split = local_split
        elif split != local_split:
            raise ValueError("RGB spatial shards mix dataset splits")
        contract_keys = (
            "measurement_checkpoint_sha256",
            "candidate_evidence_sha256",
            "candidate_inference_evidence_sha256",
            "coordinate_space_id",
            "rows_csv",
            "rows_csv_sha256",
            "support_view_probability_semantics",
        )
        current = {key: metadata.get(key) for key in contract_keys}
        if common_metadata is None:
            common_metadata = {**metadata, "_contract": current}
        elif common_metadata["_contract"] != current:
            raise ValueError("RGB spatial shards have different inference contracts")
        if offsets is None:
            offsets = local_offsets
        elif not np.array_equal(offsets, local_offsets):
            raise ValueError("RGB spatial shards use different offset grids")
        row_count = len(arrays["candidate_identity_keys"])
        if any(
            len(value) != row_count
            for key, value in arrays.items()
            if key != "local_log_probabilities"
        ):
            raise ValueError("RGB spatial arrays have different row counts")
        if arrays["local_log_probabilities"].shape != (row_count, len(local_offsets)):
            raise ValueError("RGB spatial local probability shape is invalid")
        blocks.append(arrays)
        hashes.append(artifact_hash)
    assert offsets is not None and common_metadata is not None and split is not None

    keys = blocks[0].keys()
    if any(block.keys() != keys for block in blocks[1:]):
        raise ValueError("RGB spatial shard array keys differ")
    arrays = {key: np.concatenate([block[key] for block in blocks]) for key in keys}
    source_indices = np.asarray(
        arrays["supervision_source_row_indices"], dtype=np.int64
    )
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("RGB spatial shards overlap supervision rows")
    order = np.argsort(source_indices, kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    source_indices = source_indices[order]

    rows_path = Path(str(common_metadata.get("rows_csv", "")))
    rows_summary_path = rows_path.with_suffix(".summary.json")
    if not rows_path.exists() or not rows_summary_path.exists():
        raise ValueError("target-free RGB inference rows are incomplete")
    if common_metadata.get("rows_csv_sha256") != file_sha256_short(rows_path):
        raise ValueError("target-free RGB inference rows are stale")
    rows_summary = json.loads(rows_summary_path.read_text())
    if (
        rows_summary.get("stage") != "candidate_specific_real_rgb_inference_rows"
        or bool(rows_summary.get("protocol", {}).get("contains_ground_truth"))
        or bool(rows_summary.get("protocol", {}).get("contains_pose_derived_features"))
        or rows_summary.get("split") != split
    ):
        raise ValueError("RGB inference rows do not satisfy the target-free contract")
    with rows_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            "target" in str(name).lower() or "ground_truth" in str(name).lower()
            for name in reader.fieldnames
        ):
            raise ValueError("RGB inference rows expose supervision fields")
        observable_rows = tuple(dict(row) for row in reader)
    if np.any((source_indices < 0) | (source_indices >= len(observable_rows))):
        raise ValueError("RGB spatial source row join indices are invalid")
    for index, source_index in enumerate(source_indices.tolist()):
        row = observable_rows[int(source_index)]
        if (
            int(row.get("supervision_source_row_index", -1)) != int(source_index)
            or str(row.get("candidate_identity_key", ""))
            != str(arrays["candidate_identity_keys"][index])
            or str(row.get("query_id", "")) != str(arrays["query_ids"][index])
            or int(row.get("source_query_row", -1))
            != int(arrays["source_query_rows"][index])
            or int(row.get("candidate_measurement_rank", -1))
            != int(arrays["candidate_measurement_ranks"][index])
            or int(row.get("track_id", -1))
            != int(arrays["candidate_track_ids"][index])
            or str(row.get("support_image_id", ""))
            != str(arrays["support_image_ids"][index])
        ):
            raise ValueError("RGB spatial prediction differs from sanitized input row")
    return _SpatialInput(
        arrays=arrays,
        offsets_xy=offsets.astype(np.float64),
        observable_rows=observable_rows,
        metadata=common_metadata,
        split=split,
        paths=spatial_paths,
        hashes=tuple(hashes),
    )


def aggregate_candidate_measurement_views(
    *,
    offsets_xy: np.ndarray,
    local_log_probabilities: np.ndarray,
    dustbin_probabilities: np.ndarray,
    support_view_probabilities: np.ndarray,
    likelihood_entropy: np.ndarray,
    likelihood_covariance_trace_px2: np.ndarray,
    observable_rows: Sequence[Mapping[str, object]],
    coordinate_proposal_policy: str = "posterior_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-free utility features and one multi-view coordinate offset."""

    offsets = np.asarray(offsets_xy, dtype=np.float64)
    local_log = np.asarray(local_log_probabilities, dtype=np.float64)
    dustbin = np.asarray(dustbin_probabilities, dtype=np.float64).reshape(-1)
    view_prior = np.asarray(
        support_view_probabilities, dtype=np.float64
    ).reshape(-1)
    entropy = np.asarray(likelihood_entropy, dtype=np.float64).reshape(-1)
    covariance = np.asarray(
        likelihood_covariance_trace_px2, dtype=np.float64
    ).reshape(-1)
    view_count = len(local_log)
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 2
        or local_log.shape != (view_count, len(offsets))
        or len(dustbin) != view_count
        or len(view_prior) != view_count
        or len(entropy) != view_count
        or len(covariance) != view_count
        or len(observable_rows) != view_count
        or view_count == 0
    ):
        raise ValueError("candidate measurement view arrays are not aligned")
    if (
        np.any(~np.isfinite(local_log))
        or np.any(~np.isfinite(dustbin))
        or np.any(~np.isfinite(view_prior))
        or np.any(~np.isfinite(entropy))
        or np.any(~np.isfinite(covariance))
        or np.any((dustbin < 0.0) | (dustbin > 1.0))
        or np.any(view_prior < 0.0)
    ):
        raise ValueError("candidate measurement view values are invalid")
    available_mass = float(np.sum(view_prior))
    if available_mass <= 0.0 or available_mass > 1.0 + 2e-5:
        raise ValueError("candidate support-view probability mass is invalid")
    normalized_prior = view_prior / available_mass
    local_log = local_log - np.max(local_log, axis=1, keepdims=True)
    local_probability = np.exp(local_log)
    local_probability /= np.sum(local_probability, axis=1, keepdims=True)
    accepted_weights = view_prior * (1.0 - dustbin)
    accepted_mass = float(np.sum(accepted_weights))
    if accepted_mass > 1e-12:
        mixture = np.sum(accepted_weights[:, None] * local_probability, axis=0)
    else:
        mixture = np.sum(view_prior[:, None] * local_probability, axis=0)
    mixture /= max(float(np.sum(mixture)), 1e-12)
    proposal_policy = str(coordinate_proposal_policy)
    if proposal_policy == "posterior_mean":
        proposed_offset = mixture @ offsets
    elif proposal_policy == "mixture_map":
        proposed_offset = offsets[int(np.argmax(mixture))]
    else:
        raise ValueError("unsupported measurement coordinate proposal policy")

    local_means = local_probability @ offsets
    local_modes = offsets[np.argmax(local_probability, axis=1)]
    agreement_weights = (
        accepted_weights / accepted_mass
        if accepted_mass > 1e-12
        else normalized_prior
    )
    mean_center = np.sum(agreement_weights[:, None] * local_means, axis=0)
    mode_center = np.sum(agreement_weights[:, None] * local_modes, axis=0)
    posterior_disagreement = float(
        np.sum(
            agreement_weights
            * np.sum(np.square(local_means - mean_center[None]), axis=1)
        )
    )
    mode_disagreement = float(
        np.sum(
            agreement_weights
            * np.sum(np.square(local_modes - mode_center[None]), axis=1)
        )
    )
    peak = np.max(local_probability, axis=1)
    if len(offsets) > 1:
        partitioned = np.partition(local_probability, -2, axis=1)
        margin = partitioned[:, -1] - partitioned[:, -2]
    else:
        margin = peak.copy()
    bin_count_log = max(math.log(max(len(offsets), 2)), 1e-12)

    def weighted_observable(key: str, *, transform=lambda value: value) -> float:
        values = np.asarray(
            [transform(_float(row, key)) for row in observable_rows],
            dtype=np.float64,
        )
        return float(np.sum(normalized_prior * values))

    track_lengths = [_float(row, "track_length") for row in observable_rows]
    if not np.allclose(track_lengths, track_lengths[:1], rtol=0.0, atol=1e-5):
        raise ValueError("candidate support views disagree on track length")
    features = np.asarray(
        [
            math.log1p(track_lengths[0]),
            available_mass,
            accepted_mass,
            float(np.sum(normalized_prior * entropy)) / bin_count_log,
            math.log1p(float(np.sum(normalized_prior * covariance))),
            float(np.sum(normalized_prior * peak)),
            float(np.sum(normalized_prior * margin)),
            float(np.max(mixture)),
            float(-np.sum(mixture * np.log(np.maximum(mixture, 1e-12))))
            / bin_count_log,
            float(np.linalg.norm(proposed_offset)),
            math.log1p(posterior_disagreement),
            math.log1p(mode_disagreement),
            math.log1p(view_count),
            float(np.max(normalized_prior)),
            float(
                -np.sum(
                    normalized_prior * np.log(np.maximum(normalized_prior, 1e-12))
                )
            ),
            weighted_observable("support_reprojection_error"),
            weighted_observable(
                "support_frame_gap", transform=lambda value: math.log1p(abs(value))
            ),
        ],
        dtype=np.float64,
    )
    if len(features) != len(UTILITY_FEATURE_NAMES) or np.any(~np.isfinite(features)):
        raise ValueError("candidate measurement utility features are invalid")
    return features, np.asarray(proposed_offset, dtype=np.float64)


def build_candidate_measurement_utility_examples(
    spatial_paths: Sequence[Path],
    *,
    coordinate_proposal_policy: str = "posterior_mean",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    spatial = _load_target_free_spatial(spatial_paths)
    arrays = spatial.arrays
    source_indices = np.asarray(
        arrays["supervision_source_row_indices"], dtype=np.int64
    )
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(arrays["candidate_identity_keys"].astype(str)):
        if not identity:
            raise ValueError("candidate measurement identity is empty")
        grouped.setdefault(identity, []).append(index)
    examples: list[dict[str, object]] = []
    for identity, positions in grouped.items():
        indices = np.asarray(positions, dtype=np.int64)
        first = int(indices[0])
        consistent = (
            "query_ids",
            "source_query_rows",
            "candidate_measurement_ranks",
            "candidate_track_ids",
            "candidate_prototype_ids",
        )
        for key in consistent:
            if np.any(arrays[key][indices] != arrays[key][first]):
                raise ValueError(f"candidate support views disagree on {key}")
        center = np.asarray(arrays["center_xy"][indices], dtype=np.float64)
        if not np.allclose(center, center[:1], rtol=0.0, atol=1e-4):
            raise ValueError("candidate support views disagree on center coordinate")
        view_ranks = np.asarray(arrays["support_view_ranks"][indices], dtype=np.int64)
        if len(np.unique(view_ranks)) != len(view_ranks):
            raise ValueError("candidate support-view ranks are not unique")
        order = np.argsort(view_ranks, kind="stable")
        indices = indices[order]
        observable = [
            spatial.observable_rows[int(source_indices[index])] for index in indices
        ]
        features, offset = aggregate_candidate_measurement_views(
            offsets_xy=spatial.offsets_xy,
            local_log_probabilities=arrays["local_log_probabilities"][indices],
            dustbin_probabilities=arrays["dustbin_probabilities"][indices],
            support_view_probabilities=arrays["support_view_probabilities"][indices],
            likelihood_entropy=arrays["likelihood_entropy"][indices],
            likelihood_covariance_trace_px2=arrays[
                "likelihood_covariance_trace_px2"
            ][indices],
            observable_rows=observable,
            coordinate_proposal_policy=str(coordinate_proposal_policy),
        )
        center_xy = center[0]
        examples.append(
            {
                "candidate_identity_key": identity,
                "query_id": str(arrays["query_ids"][first]),
                "source_query_row": int(arrays["source_query_rows"][first]),
                "candidate_measurement_rank": int(
                    arrays["candidate_measurement_ranks"][first]
                ),
                "track_id": int(arrays["candidate_track_ids"][first]),
                "prototype_id": int(arrays["candidate_prototype_ids"][first]),
                "center_xy": center_xy,
                "refined_xy": center_xy + offset,
                "features": features,
                "supervision_source_row_indices": tuple(
                    int(source_indices[index]) for index in indices
                ),
            }
        )
    examples.sort(
        key=lambda example: (
            str(example["query_id"]),
            int(example["source_query_row"]),
            int(example["candidate_measurement_rank"]),
        )
    )
    contract = {
        "split": spatial.split,
        "measurement_checkpoint_sha256": str(
            spatial.metadata["measurement_checkpoint_sha256"]
        ),
        "candidate_evidence_sha256": str(
            spatial.metadata["candidate_evidence_sha256"]
        ),
        "candidate_inference_evidence_sha256": str(
            spatial.metadata["candidate_inference_evidence_sha256"]
        ),
        "coordinate_space_id": str(spatial.metadata["coordinate_space_id"]),
        "offsets_sha256": _array_sha256_short(spatial.offsets_xy),
        "coordinate_proposal_policy": str(coordinate_proposal_policy),
        "offset_min_xy": np.min(spatial.offsets_xy, axis=0).tolist(),
        "offset_max_xy": np.max(spatial.offsets_xy, axis=0).tolist(),
        "spatial_paths": [str(path) for path in spatial.paths],
        "spatial_sha256": list(spatial.hashes),
        "sanitized_rows": str(spatial.metadata["rows_csv"]),
        "sanitized_rows_sha256": str(spatial.metadata["rows_csv_sha256"]),
    }
    return examples, contract


def _feature_matrix(examples: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.stack(
        [np.asarray(example["features"], dtype=np.float64) for example in examples]
    )


def _load_targets(
    *,
    examples: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    supervision_rows_csv: Path,
    minimum_baseline_residual_px: float,
    minimum_improvement_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sanitized_summary_path = Path(str(contract["sanitized_rows"])).with_suffix(
        ".summary.json"
    )
    sanitized_summary = json.loads(sanitized_summary_path.read_text())
    target_path = Path(supervision_rows_csv)
    inputs = sanitized_summary.get("inputs", {})
    if (
        inputs.get("supervision_rows_sha256") != file_sha256_short(target_path)
        or str(Path(str(inputs.get("supervision_rows", "")))) != str(target_path)
    ):
        raise ValueError("utility targets differ from sanitized-row source manifest")
    with target_path.open(newline="") as handle:
        target_rows = list(csv.DictReader(handle))
    offset_min = np.asarray(contract["offset_min_xy"], dtype=np.float64)
    offset_max = np.asarray(contract["offset_max_xy"], dtype=np.float64)
    labels = np.zeros((len(examples),), dtype=bool)
    evaluable = np.zeros((len(examples),), dtype=bool)
    baseline = np.full((len(examples),), np.nan, dtype=np.float64)
    refined = np.full((len(examples),), np.nan, dtype=np.float64)
    for index, example in enumerate(examples):
        source_indices = tuple(
            int(value) for value in example["supervision_source_row_indices"]
        )
        if not source_indices or any(
            value < 0 or value >= len(target_rows) for value in source_indices
        ):
            raise ValueError("candidate utility target join index is invalid")
        rows = [target_rows[value] for value in source_indices]
        first = rows[0]
        if any(
            str(row.get("candidate_identity_key", ""))
            != str(example["candidate_identity_key"])
            for row in rows
        ):
            raise ValueError("candidate utility target identity differs")
        target_values = np.asarray(
            [
                [_float(row, "target_gt_projected_x"), _float(row, "target_gt_projected_y")]
                for row in rows
            ],
            dtype=np.float64,
        )
        if not np.allclose(target_values, target_values[:1], rtol=0.0, atol=1e-5):
            raise ValueError("candidate support views disagree on pose target")
        target_xy = target_values[0]
        center_xy = np.asarray(example["center_xy"], dtype=np.float64)
        refined_xy = np.asarray(example["refined_xy"], dtype=np.float64)
        physical = all(
            _bool(row.get("target_gt_projection_in_front"))
            and _bool(row.get("target_gt_projection_in_image"))
            and _float(row, "geometry_supervision_weight") > 0.0
            for row in rows
        )
        evaluable[index] = bool(physical and np.all(np.isfinite(target_xy)))
        if not evaluable[index]:
            continue
        target_offset = target_xy - center_xy
        inside = bool(
            np.all(target_offset >= offset_min) and np.all(target_offset <= offset_max)
        )
        baseline[index] = float(np.linalg.norm(center_xy - target_xy))
        refined[index] = float(np.linalg.norm(refined_xy - target_xy))
        labels[index] = bool(
            inside
            and baseline[index] > float(minimum_baseline_residual_px)
            and refined[index] + float(minimum_improvement_px) < baseline[index]
        )
    if not np.any(evaluable):
        raise ValueError("candidate utility target join produced no evaluable examples")
    return labels, evaluable, baseline, refined


def _fit_base(
    features: np.ndarray, labels: np.ndarray, *, c_value: float
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(
        C=float(c_value), solver="lbfgs", max_iter=2000, random_state=0
    )
    model.fit((features - mean[None]) / scale[None], labels)
    return mean, scale, model


def _fit_calibration(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    model = LogisticRegression(
        C=1000.0, solver="lbfgs", max_iter=2000, random_state=0
    )
    model.fit(_logit(probabilities).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _threshold_at_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_precision: float,
    minimum_selected_fraction: float,
) -> tuple[float, dict[str, object]]:
    order = np.argsort(-probabilities, kind="stable")
    precision = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    minimum_count = max(1, int(math.ceil(len(labels) * minimum_selected_fraction)))
    valid = np.flatnonzero(
        (precision >= float(target_precision))
        & (np.arange(1, len(labels) + 1) >= minimum_count)
    )
    if not len(valid):
        return 1.0, {
            "selected_count": 0,
            "selected_fraction": 0.0,
            "precision": None,
            "target_precision": float(target_precision),
            "minimum_selected_fraction": float(minimum_selected_fraction),
        }
    end = int(valid[-1])
    return float(probabilities[order[end]]), {
        "selected_count": int(end + 1),
        "selected_fraction": float((end + 1) / len(labels)),
        "precision": float(precision[end]),
        "target_precision": float(target_precision),
        "minimum_selected_fraction": float(minimum_selected_fraction),
    }


def _evaluate(
    *,
    labels: np.ndarray,
    evaluable: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    minimum_improvement_px: float,
) -> dict[str, object]:
    valid = np.asarray(evaluable, dtype=bool)
    target = np.asarray(labels, dtype=bool)[valid]
    values = np.asarray(probabilities, dtype=np.float64)[valid]
    selected = values >= float(threshold)
    before = np.asarray(baseline, dtype=np.float64)[valid]
    after = np.asarray(refined, dtype=np.float64)[valid]
    report = confidence_metrics(target, values)
    return {
        **report,
        "evaluable_count": int(np.sum(valid)),
        "positive_prior": float(np.mean(target)),
        "selected_count": int(np.sum(selected)),
        "selected_fraction": float(np.mean(selected)),
        "selected_safe_update_precision": (
            None if not np.any(selected) else float(np.mean(target[selected]))
        ),
        "selected_actual_improve_fraction": (
            None
            if not np.any(selected)
            else float(
                np.mean(
                    after[selected] + float(minimum_improvement_px)
                    < before[selected]
                )
            )
        ),
        "selected_actual_worsen_fraction": (
            None
            if not np.any(selected)
            else float(
                np.mean(
                    after[selected]
                    > before[selected] + float(minimum_improvement_px)
                )
            )
        ),
        "selected_baseline_median_px": (
            None if not np.any(selected) else float(np.median(before[selected]))
        ),
        "selected_refined_median_px": (
            None if not np.any(selected) else float(np.median(after[selected]))
        ),
        "selected_mean_improvement_px": (
            None if not np.any(selected) else float(np.mean(before[selected] - after[selected]))
        ),
    }


def _write_predictions(
    path: Path,
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> None:
    fields = [
        "candidate_identity_key",
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "update_beneficial_probability",
        "update_approved",
        "center_x",
        "center_y",
        "refined_x",
        "refined_y",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example, probability in zip(examples, probabilities.tolist()):
            center = np.asarray(example["center_xy"], dtype=np.float64)
            refined = np.asarray(example["refined_xy"], dtype=np.float64)
            writer.writerow(
                {
                    **{name: example[name] for name in fields[:6]},
                    "update_beneficial_probability": float(probability),
                    "update_approved": bool(float(probability) >= float(threshold)),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "refined_x": float(refined[0]),
                    "refined_y": float(refined[1]),
                }
            )


def _validate_model_contract(
    model: CandidateMeasurementUtilityGate, contract: Mapping[str, object]
) -> None:
    expected = {
        "measurement_checkpoint_sha256": model.measurement_checkpoint_sha256,
        "candidate_evidence_sha256": model.candidate_evidence_sha256,
        "candidate_inference_evidence_sha256": (
            model.candidate_inference_evidence_sha256
        ),
        "coordinate_space_id": model.coordinate_space_id,
        "offsets_sha256": model.offsets_sha256,
        "coordinate_proposal_policy": model.coordinate_proposal_policy,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if str(contract.get(key, "")) != str(value)
    }
    if mismatches:
        raise ValueError(
            "candidate measurement utility contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def fit_candidate_measurement_utility(
    *,
    train_spatial_paths: Sequence[Path],
    train_supervision_rows_csv: Path,
    output_dir: Path,
    c_value: float = 0.1,
    fold_count: int = 5,
    target_precision: float = 0.8,
    minimum_selected_fraction: float = 0.01,
    minimum_baseline_residual_px: float = 1.0,
    minimum_improvement_px: float = 0.1,
    coordinate_proposal_policy: str = "posterior_mean",
) -> dict[str, object]:
    examples, contract = build_candidate_measurement_utility_examples(
        train_spatial_paths,
        coordinate_proposal_policy=str(coordinate_proposal_policy),
    )
    if contract["split"] != "train":
        raise ValueError("candidate measurement utility fit requires train artifacts")
    labels, evaluable, baseline, refined = _load_targets(
        examples=examples,
        contract=contract,
        supervision_rows_csv=Path(train_supervision_rows_csv),
        minimum_baseline_residual_px=float(minimum_baseline_residual_px),
        minimum_improvement_px=float(minimum_improvement_px),
    )
    examples = [example for example, keep in zip(examples, evaluable) if bool(keep)]
    labels = labels[evaluable].astype(np.int64)
    baseline = baseline[evaluable]
    refined = refined[evaluable]
    if len(np.unique(labels)) != 2:
        raise ValueError("candidate measurement utility fit requires both target classes")
    features = _feature_matrix(examples)
    groups = np.asarray([str(example["query_id"]) for example in examples], dtype=object)
    split_count = min(int(fold_count), len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("candidate measurement utility fit needs two query groups")
    oof = np.zeros((len(examples),), dtype=np.float64)
    splitter = GroupKFold(n_splits=split_count)
    for fit_indices, heldout_indices in splitter.split(features, labels, groups):
        mean, scale, model = _fit_base(
            features[fit_indices], labels[fit_indices], c_value=float(c_value)
        )
        oof[heldout_indices] = model.predict_proba(
            (features[heldout_indices] - mean[None]) / scale[None]
        )[:, 1]
    calibration_slope, calibration_intercept = _fit_calibration(oof, labels)
    calibrated_oof = _sigmoid(
        calibration_slope * _logit(oof) + calibration_intercept
    )
    threshold, threshold_audit = _threshold_at_precision(
        labels,
        calibrated_oof,
        target_precision=float(target_precision),
        minimum_selected_fraction=float(minimum_selected_fraction),
    )
    mean, scale, base_model = _fit_base(features, labels, c_value=float(c_value))
    gate = CandidateMeasurementUtilityGate(
        feature_names=UTILITY_FEATURE_NAMES,
        feature_mean=tuple(mean.tolist()),
        feature_scale=tuple(scale.tolist()),
        coefficients=tuple(base_model.coef_[0].tolist()),
        intercept=float(base_model.intercept_[0]),
        calibration_slope=float(calibration_slope),
        calibration_intercept=float(calibration_intercept),
        update_threshold=float(threshold),
        minimum_baseline_residual_px=float(minimum_baseline_residual_px),
        minimum_improvement_px=float(minimum_improvement_px),
        measurement_checkpoint_sha256=str(
            contract["measurement_checkpoint_sha256"]
        ),
        candidate_evidence_sha256=str(contract["candidate_evidence_sha256"]),
        candidate_inference_evidence_sha256=str(
            contract["candidate_inference_evidence_sha256"]
        ),
        coordinate_space_id=str(contract["coordinate_space_id"]),
        offsets_sha256=str(contract["offsets_sha256"]),
        coordinate_proposal_policy=str(coordinate_proposal_policy),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "candidate_measurement_utility_gate.json"
    model_path.write_text(json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n")
    predictions_path = output / "train_oof_utility_predictions.csv"
    _write_predictions(
        predictions_path, examples, calibrated_oof, threshold=float(threshold)
    )
    train_metrics = _evaluate(
        labels=labels,
        evaluable=np.ones((len(labels),), dtype=bool),
        baseline=baseline,
        refined=refined,
        probabilities=calibrated_oof,
        threshold=float(threshold),
        minimum_improvement_px=float(minimum_improvement_px),
    )
    summary = {
        "stage": "candidate_measurement_utility_fit",
        "protocol": {
            "probability_semantics": (
                f"P({coordinate_proposal_policy}_RGB_coordinate_improves_GT_pose_projection_"
                "within_local_support)"
            ),
            "identity_likelihood": False,
            "features_use_candidate_identity_prior": False,
            "features_use_pose_or_ground_truth": False,
            "target_joined_after_frozen_RGB_inference": True,
            "grouped_oof_by_query": True,
            "probability_calibration": "Platt_scaling_on_grouped_train_OOF",
            "threshold_selection": "train_OOF_precision_gate",
            "validation_used": False,
            "test_used": False,
        },
        "feature_set": UTILITY_FEATURE_SET,
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "config": {
            "c_value": float(c_value),
            "fold_count": int(split_count),
            "target_precision": float(target_precision),
            "minimum_selected_fraction": float(minimum_selected_fraction),
            "minimum_baseline_residual_px": float(minimum_baseline_residual_px),
            "minimum_improvement_px": float(minimum_improvement_px),
            "coordinate_proposal_policy": str(coordinate_proposal_policy),
        },
        "threshold_audit": threshold_audit,
        "train_oof_TARGET_ONLY": train_metrics,
        "inputs": {
            **contract,
            "supervision_rows": str(train_supervision_rows_csv),
            "supervision_rows_sha256": file_sha256_short(
                Path(train_supervision_rows_csv)
            ),
        },
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "train_oof_predictions": str(predictions_path),
            "train_oof_predictions_sha256": file_sha256_short(predictions_path),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def apply_candidate_measurement_utility(
    *,
    model_path: Path,
    spatial_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, object]:
    model_file = Path(model_path)
    gate = CandidateMeasurementUtilityGate.from_dict(
        json.loads(model_file.read_text())
    )
    examples, contract = build_candidate_measurement_utility_examples(
        spatial_paths,
        coordinate_proposal_policy=str(gate.coordinate_proposal_policy),
    )
    _validate_model_contract(gate, contract)
    probabilities = gate.predict(_feature_matrix(examples))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        output,
        examples,
        probabilities,
        threshold=float(gate.update_threshold),
    )
    query_update_counts: dict[str, int] = {}
    for example, probability in zip(examples, probabilities.tolist()):
        if float(probability) >= float(gate.update_threshold):
            query = str(example["query_id"])
            query_update_counts[query] = query_update_counts.get(query, 0) + 1
    all_queries = {str(example["query_id"]) for example in examples}
    summary = {
        "stage": UTILITY_APPLY_STAGE,
        "split": contract["split"],
        "protocol": {
            "ground_truth_loaded": False,
            "pose_loaded": False,
            "target_free_spatial_predictions": True,
            "target_free_observable_features": True,
            "candidate_identity_prior_used": False,
            "probability_is_action_gate_not_identity_likelihood": True,
            "coordinate_update_applied": False,
        },
        "update_threshold": float(gate.update_threshold),
        "counts": {
            "candidate_count": int(len(examples)),
            "approved_candidate_count": int(
                np.sum(probabilities >= float(gate.update_threshold))
            ),
            "approved_candidate_fraction": float(
                np.mean(probabilities >= float(gate.update_threshold))
            ),
            "query_count": int(len(all_queries)),
            "queries_with_at_least_four_approved_candidates": int(
                sum(query_update_counts.get(query, 0) >= 4 for query in all_queries)
            ),
        },
        "inputs": {
            **contract,
            "model": str(model_file),
            "model_sha256": file_sha256_short(model_file),
        },
        "outputs": {
            "predictions": str(output),
            "predictions_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def audit_candidate_measurement_utility(
    *,
    predictions_path: Path,
    spatial_paths: Sequence[Path],
    supervision_rows_csv: Path,
    output_path: Path,
) -> dict[str, object]:
    prediction_path = Path(predictions_path)
    summary_path = prediction_path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != UTILITY_APPLY_STAGE:
        raise ValueError("measurement utility audit requires a target-free apply artifact")
    if summary.get("outputs", {}).get("predictions_sha256") != file_sha256_short(
        prediction_path
    ):
        raise ValueError("measurement utility predictions are stale")
    model_path = Path(str(summary.get("inputs", {}).get("model", "")))
    if summary.get("inputs", {}).get("model_sha256") != file_sha256_short(model_path):
        raise ValueError("measurement utility model is stale")
    gate = CandidateMeasurementUtilityGate.from_dict(json.loads(model_path.read_text()))
    examples, contract = build_candidate_measurement_utility_examples(
        spatial_paths,
        coordinate_proposal_policy=str(gate.coordinate_proposal_policy),
    )
    _validate_model_contract(gate, contract)
    labels, evaluable, baseline, refined = _load_targets(
        examples=examples,
        contract=contract,
        supervision_rows_csv=Path(supervision_rows_csv),
        minimum_baseline_residual_px=float(gate.minimum_baseline_residual_px),
        minimum_improvement_px=float(gate.minimum_improvement_px),
    )
    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(examples):
        raise ValueError("measurement utility prediction/example counts differ")
    probabilities = np.zeros((len(rows),), dtype=np.float64)
    for index, (row, example) in enumerate(zip(rows, examples)):
        if any(
            str(row.get(name, "")) != str(example[name])
            for name in (
                "candidate_identity_key",
                "query_id",
                "source_query_row",
                "candidate_measurement_rank",
                "track_id",
                "prototype_id",
            )
        ):
            raise ValueError("measurement utility prediction identity differs")
        probabilities[index] = _float(row, "update_beneficial_probability")
        observed_refined = np.asarray(
            [_float(row, "refined_x"), _float(row, "refined_y")], dtype=np.float64
        )
        if not np.allclose(
            observed_refined,
            np.asarray(example["refined_xy"], dtype=np.float64),
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError("measurement utility refined coordinate differs")
    metrics = _evaluate(
        labels=labels,
        evaluable=evaluable,
        baseline=baseline,
        refined=refined,
        probabilities=probabilities,
        threshold=float(gate.update_threshold),
        minimum_improvement_px=float(gate.minimum_improvement_px),
    )
    result = {
        "stage": "candidate_measurement_utility_external_target_audit",
        "split": contract["split"],
        "protocol": {
            "predictions_frozen_before_target_join": True,
            "apply_process_loaded_ground_truth": False,
            "threshold_frozen_from_train_OOF": True,
            "probability_is_action_gate_not_identity_likelihood": True,
            "reused_test_is_diagnostic_not_untouched": True,
        },
        "metrics_TARGET_ONLY": metrics,
        "inputs": {
            "predictions": str(prediction_path),
            "predictions_sha256": file_sha256_short(prediction_path),
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "spatial_paths": contract["spatial_paths"],
            "spatial_sha256": contract["spatial_sha256"],
            "supervision_rows": str(supervision_rows_csv),
            "supervision_rows_sha256": file_sha256_short(
                Path(supervision_rows_csv)
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
