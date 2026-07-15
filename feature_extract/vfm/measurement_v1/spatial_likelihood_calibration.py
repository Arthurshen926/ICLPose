"""Post-hoc calibration for candidate-specific RGB spatial likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from feature_extract.vfm.artifacts import file_sha256_short


SPATIAL_LIKELIHOOD_CALIBRATION_FORMAT = (
    "candidate_spatial_likelihood_calibration_v1"
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values) - np.log1p(-values)


def binary_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float | int]:
    values = np.clip(
        np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-6, 1.0 - 1e-6
    )
    targets = np.asarray(labels, dtype=bool).reshape(-1)
    if values.shape != targets.shape or len(values) == 0:
        raise ValueError("binary calibration inputs must be non-empty and aligned")
    target_float = targets.astype(np.float64)
    nll = -np.mean(
        target_float * np.log(values) + (1.0 - target_float) * np.log1p(-values)
    )
    brier = np.mean(np.square(values - target_float))
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        selected = (values >= boundaries[index]) & (
            values < boundaries[index + 1]
            if index + 1 < int(bins)
            else values <= boundaries[index + 1]
        )
        if not np.any(selected):
            continue
        ece += float(np.mean(selected)) * abs(
            float(np.mean(values[selected])) - float(np.mean(target_float[selected]))
        )
    return {
        "sample_count": int(len(values)),
        "positive_count": int(np.count_nonzero(targets)),
        "positive_prior": float(np.mean(target_float)),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
    }


def fit_dustbin_platt_scaling(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=bool).reshape(-1)
    if values.shape != targets.shape or len(values) == 0:
        raise ValueError("dustbin calibration inputs must be non-empty and aligned")
    logits = _logit(values)
    target_float = targets.astype(np.float64)

    def objective(parameters: np.ndarray) -> float:
        scale = float(np.exp(parameters[0]))
        calibrated = np.clip(
            _sigmoid(scale * logits + float(parameters[1])), 1e-9, 1.0 - 1e-9
        )
        return float(
            -np.mean(
                target_float * np.log(calibrated)
                + (1.0 - target_float) * np.log1p(-calibrated)
            )
        )

    result = minimize(
        objective,
        np.asarray([0.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-4.0, 4.0), (-12.0, 12.0)),
    )
    if not bool(result.success) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"dustbin calibration failed: {result.message}")
    scale = float(np.exp(result.x[0]))
    bias = float(result.x[1])
    calibrated = _sigmoid(scale * logits + bias)
    return {
        "logit_scale": scale,
        "logit_bias": bias,
        "optimizer_success": True,
        "optimizer_iterations": int(result.nit),
        "raw_metrics": binary_calibration_metrics(values, targets),
        "calibrated_metrics": binary_calibration_metrics(calibrated, targets),
    }


def _bilinear_target_indices(
    offsets_xy: np.ndarray,
    target_offsets_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(offsets_xy, dtype=np.float64).reshape(-1, 2)
    targets = np.asarray(target_offsets_xy, dtype=np.float64).reshape(-1, 2)
    xs = np.unique(offsets[:, 0])
    ys = np.unique(offsets[:, 1])
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(offsets):
        raise ValueError("spatial offsets must form a complete 2D grid")
    if not np.allclose(np.diff(xs), np.diff(xs)[0], rtol=0.0, atol=1e-8) or not np.allclose(
        np.diff(ys), np.diff(ys)[0], rtol=0.0, atol=1e-8
    ):
        raise ValueError("spatial calibration requires a regular grid")
    order = np.lexsort((offsets[:, 0], offsets[:, 1]))
    expected = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if not np.allclose(offsets[order], expected, rtol=0.0, atol=1e-8):
        raise ValueError("spatial offset ordering is incomplete")
    inside = (
        np.all(np.isfinite(targets), axis=1)
        & (targets[:, 0] >= xs[0])
        & (targets[:, 0] <= xs[-1])
        & (targets[:, 1] >= ys[0])
        & (targets[:, 1] <= ys[-1])
    )
    if not np.all(inside):
        raise ValueError("non-dustbin calibration targets must lie inside the spatial grid")
    gx = np.clip((targets[:, 0] - xs[0]) / (xs[1] - xs[0]), 0.0, len(xs) - 1.0)
    gy = np.clip((targets[:, 1] - ys[0]) / (ys[1] - ys[0]), 0.0, len(ys) - 1.0)
    x0 = np.minimum(np.floor(gx).astype(np.int64), len(xs) - 2)
    y0 = np.minimum(np.floor(gy).astype(np.int64), len(ys) - 2)
    tx = gx - x0
    ty = gy - y0
    index_grid = order.reshape(len(ys), len(xs))
    indices = np.column_stack(
        [
            index_grid[y0, x0],
            index_grid[y0, x0 + 1],
            index_grid[y0 + 1, x0],
            index_grid[y0 + 1, x0 + 1],
        ]
    )
    weights = np.column_stack(
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ]
    )
    return indices.astype(np.int64), weights.astype(np.float64)


def spatial_target_nll(
    local_log_probabilities: np.ndarray,
    offsets_xy: np.ndarray,
    target_offsets_xy: np.ndarray,
    *,
    temperature: float,
    chunk_size: int = 2048,
) -> float:
    logits = np.asarray(local_log_probabilities, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("spatial calibration logits must have shape (N, K)")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("spatial calibration temperature must be positive")
    indices, weights = _bilinear_target_indices(offsets_xy, target_offsets_xy)
    if len(indices) != len(logits) or logits.shape[1] != len(offsets_xy):
        raise ValueError("spatial calibration targets and logits are misaligned")
    total = 0.0
    for start in range(0, len(logits), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(logits))
        scaled = logits[start:stop] / float(temperature)
        scaled -= np.max(scaled, axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= np.maximum(
            np.sum(probabilities, axis=1, keepdims=True), 1e-300
        )
        row_indices = np.arange(stop - start, dtype=np.int64)[:, None]
        target_probability = np.sum(
            probabilities[row_indices, indices[start:stop]] * weights[start:stop],
            axis=1,
        )
        total += float(np.sum(-np.log(np.maximum(target_probability, 1e-300))))
    return total / float(len(logits))


def fit_spatial_temperature(
    local_log_probabilities: np.ndarray,
    offsets_xy: np.ndarray,
    target_offsets_xy: np.ndarray,
) -> dict[str, object]:
    raw_nll = spatial_target_nll(
        local_log_probabilities,
        offsets_xy,
        target_offsets_xy,
        temperature=1.0,
    )
    result = minimize_scalar(
        lambda log_temperature: spatial_target_nll(
            local_log_probabilities,
            offsets_xy,
            target_offsets_xy,
            temperature=float(np.exp(log_temperature)),
        ),
        bounds=(float(np.log(0.05)), float(np.log(20.0))),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 80},
    )
    if not bool(result.success) or not np.isfinite(result.x):
        raise RuntimeError("spatial temperature calibration failed")
    temperature = float(np.exp(result.x))
    return {
        "temperature": temperature,
        "raw_nll": float(raw_nll),
        "calibrated_nll": float(result.fun),
        "sample_count": int(len(local_log_probabilities)),
        "optimizer_success": True,
        "optimizer_iterations": int(result.nfev),
    }


@dataclass(frozen=True)
class CandidateSpatialLikelihoodCalibration:
    spatial_temperature: float
    dustbin_logit_scale: float
    dustbin_logit_bias: float
    measurement_checkpoint_sha256: str
    search_radius_px: float
    step_px: float
    context_radius_px: float
    query_source: str
    support_patch_warp: str
    fit_query_manifest_sha256: str
    audit_query_manifest_sha256: str
    production_eligible: bool
    base_model_query_overlap_count: int
    dustbin_probability_semantics: str = "legacy_binary_visibility_reliability"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.spatial_temperature)) or float(
            self.spatial_temperature
        ) <= 0.0:
            raise ValueError("spatial temperature must be positive")
        if not np.isfinite(float(self.dustbin_logit_scale)) or float(
            self.dustbin_logit_scale
        ) <= 0.0:
            raise ValueError("dustbin logit scale must be positive")
        if not np.isfinite(float(self.dustbin_logit_bias)):
            raise ValueError("dustbin logit bias must be finite")
        if int(self.base_model_query_overlap_count) < 0:
            raise ValueError("base-model query overlap cannot be negative")
        if bool(self.production_eligible) and int(
            self.base_model_query_overlap_count
        ) != 0:
            raise ValueError("production calibration cannot overlap base-model training queries")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": SPATIAL_LIKELIHOOD_CALIBRATION_FORMAT,
            "spatial_temperature": float(self.spatial_temperature),
            "dustbin_logit_scale": float(self.dustbin_logit_scale),
            "dustbin_logit_bias": float(self.dustbin_logit_bias),
            "measurement_checkpoint_sha256": str(self.measurement_checkpoint_sha256),
            "search_radius_px": float(self.search_radius_px),
            "step_px": float(self.step_px),
            "context_radius_px": float(self.context_radius_px),
            "query_source": str(self.query_source),
            "support_patch_warp": str(self.support_patch_warp),
            "fit_query_manifest_sha256": str(self.fit_query_manifest_sha256),
            "audit_query_manifest_sha256": str(self.audit_query_manifest_sha256),
            "production_eligible": bool(self.production_eligible),
            "base_model_query_overlap_count": int(self.base_model_query_overlap_count),
            "dustbin_probability_semantics": str(
                self.dustbin_probability_semantics
            ),
            "probability_contract": {
                "identity_prior_modified": False,
                "null_identity_mass_modified": False,
                "spatial_probability": "conditional_on_non_dustbin",
                "dustbin_probability": str(self.dustbin_probability_semantics),
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "CandidateSpatialLikelihoodCalibration":
        if payload.get("format") != SPATIAL_LIKELIHOOD_CALIBRATION_FORMAT:
            raise ValueError("unsupported candidate spatial calibration format")
        return cls(
            spatial_temperature=float(payload["spatial_temperature"]),
            dustbin_logit_scale=float(payload["dustbin_logit_scale"]),
            dustbin_logit_bias=float(payload["dustbin_logit_bias"]),
            measurement_checkpoint_sha256=str(
                payload["measurement_checkpoint_sha256"]
            ),
            search_radius_px=float(payload["search_radius_px"]),
            step_px=float(payload["step_px"]),
            context_radius_px=float(payload["context_radius_px"]),
            query_source=str(payload["query_source"]),
            support_patch_warp=str(payload["support_patch_warp"]),
            fit_query_manifest_sha256=str(payload["fit_query_manifest_sha256"]),
            audit_query_manifest_sha256=str(payload["audit_query_manifest_sha256"]),
            production_eligible=bool(payload["production_eligible"]),
            base_model_query_overlap_count=int(
                payload["base_model_query_overlap_count"]
            ),
            dustbin_probability_semantics=str(
                payload.get(
                    "dustbin_probability_semantics",
                    "legacy_binary_visibility_reliability",
                )
            ),
        )

    def validate_spatial_metadata(self, metadata: Mapping[str, object]) -> None:
        expected = {
            "measurement_checkpoint_sha256": str(self.measurement_checkpoint_sha256),
            "query_source": str(self.query_source),
            "support_patch_warp": str(self.support_patch_warp),
        }
        if (
            "dustbin_probability_semantics" in metadata
            or self.dustbin_probability_semantics
            != "legacy_binary_visibility_reliability"
        ):
            expected["dustbin_probability_semantics"] = str(
                self.dustbin_probability_semantics
            )
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        for key, expected_value in (
            ("search_radius_px", self.search_radius_px),
            ("step_px", self.step_px),
            ("context_radius_px", self.context_radius_px),
        ):
            actual = metadata.get(key)
            if actual is None or not np.isclose(
                float(actual), float(expected_value), rtol=0.0, atol=1e-8
            ):
                mismatches[key] = {"expected": float(expected_value), "actual": actual}
        if mismatches:
            raise ValueError(
                "candidate spatial calibration is incompatible with likelihood artifact: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )

    def apply(
        self,
        local_log_probabilities: np.ndarray,
        dustbin_probabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local = np.asarray(local_log_probabilities, dtype=np.float32) / float(
            self.spatial_temperature
        )
        dustbin = _sigmoid(
            float(self.dustbin_logit_scale)
            * _logit(np.asarray(dustbin_probabilities, dtype=np.float64))
            + float(self.dustbin_logit_bias)
        )
        return local.astype(np.float32), dustbin.astype(np.float32)


def load_candidate_spatial_likelihood_calibration(
    path: Path,
    *,
    require_production: bool = True,
    verify_artifact_manifest: bool = True,
) -> CandidateSpatialLikelihoodCalibration:
    model_path = Path(path)
    if verify_artifact_manifest:
        summary_path = model_path.parent / "summary.json"
        if not summary_path.exists():
            raise ValueError("candidate spatial calibration summary is missing")
        summary = json.loads(summary_path.read_text())
        outputs = summary.get("outputs")
        if summary.get("stage") != "candidate_spatial_likelihood_calibration_fit" or not isinstance(
            outputs, dict
        ):
            raise ValueError("unsupported candidate spatial calibration artifact")
        if outputs.get("model_sha256") != file_sha256_short(model_path):
            raise ValueError("candidate spatial calibration artifact is stale")
    payload = json.loads(model_path.read_text())
    calibration = CandidateSpatialLikelihoodCalibration.from_dict(payload)
    if bool(require_production) and not bool(calibration.production_eligible):
        raise ValueError(
            "candidate spatial calibration is diagnostic because its calibration "
            "queries overlap base-model training queries"
        )
    return calibration
