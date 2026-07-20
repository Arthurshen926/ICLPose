"""Train-only residual probes for frozen absolute-appearance evidence.

The raw C-RADIO/ALIKE appearance probes are intentionally weaker than the
already-trained mapper posterior when used alone.  This module tests the
narrower and more useful question: after conditioning on that immutable
top-L-plus-null posterior, do the frozen per-view appearance values contain
additional identity information?

The probe has deliberately limited capacity.  It learns one linear residual
per support view, marginalizes with the *fixed* maplet view weights, and keeps
the null logit fixed.  Therefore an all-zero residual exactly reproduces the
input posterior; a reported gain cannot be caused by re-normalizing top-L,
averaging descriptors, or globally re-calibrating null mass.
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

from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT,
    LOFTR_ANCHOR_FEATURE_NAMES,
)


FROZEN_APPEARANCE_RESIDUAL_MODEL_FORMAT = (
    "frozen_multiscale_candidate_appearance_fixedprior_residual_model_v1"
)
FROZEN_APPEARANCE_RESIDUAL_PREDICTION_FORMAT = (
    "frozen_multiscale_candidate_appearance_fixedprior_residual_prediction_v1"
)
FROZEN_APPEARANCE_ARTIFACT_FORMAT = (
    "frozen_multiscale_candidate_absolute_appearance_v1"
)

# These families are declared before any residual fit.  They separate the
# fine ALIKE evidence from larger C-RADIO context, while retaining one complete
# multiscale control.  They are not a validation-selected profile sweep.
FROZEN_APPEARANCE_RESIDUAL_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "fixedprior_alike_local": (
        "alike_center",
        "alike_context3",
        "alike_context5",
        "alike_context9",
    ),
    "fixedprior_radio_context": (
        "radio_final_center",
        "radio_final_context3",
        "radio_final_context5",
        "radio_intermediate_center",
        "radio_intermediate_context5",
        "radio_intermediate_context9",
        "radio_intermediate_context13",
    ),
    "fixedprior_multiscale": (
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
    "fixedprior_loftr_anchor": LOFTR_ANCHOR_FEATURE_NAMES,
}


@dataclass(frozen=True)
class FrozenAppearanceProbeFeatures:
    """Aligned target-free rows from one or more frozen appearance shards."""

    paths: tuple[Path, ...]
    query_ids: np.ndarray
    split_names: np.ndarray
    source_row_indices: np.ndarray
    xy: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    candidate_view_weights: np.ndarray
    candidate_view_scores: np.ndarray
    candidate_view_usable: np.ndarray
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        paths = tuple(Path(path) for path in self.paths)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        split_names = np.asarray(self.split_names).astype(str).reshape(-1)
        rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        probabilities = np.asarray(self.candidate_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        weights = np.asarray(self.candidate_view_weights, dtype=np.float32)
        scores = np.asarray(self.candidate_view_scores, dtype=np.float32)
        usable = np.asarray(self.candidate_view_usable, dtype=bool)
        names = tuple(str(value) for value in self.feature_names)
        count = len(query_ids)
        if (
            not paths
            or len(set(paths)) != len(paths)
            or split_names.shape != (count,)
            or rows.shape != (count,)
            or xy.shape != (count, 2)
            or tracks.ndim != 2
            or tracks.shape[0] != count
            or probabilities.shape != tracks.shape
            or null.shape != (count,)
            or weights.ndim != 3
            or weights.shape[:2] != tracks.shape
            or scores.shape != (*weights.shape, len(names))
            or usable.shape != scores.shape
            or not names
            or len(set(names)) != len(names)
            or np.any(~np.isfinite(xy))
            or np.any(~np.isfinite(probabilities))
            or np.any(~np.isfinite(null))
            or np.any(~np.isfinite(weights))
            or np.any(probabilities < 0.0)
            or np.any(null <= 0.0)
            or np.any(weights < 0.0)
            or np.any(~np.isfinite(scores[usable]))
        ):
            raise ValueError("frozen appearance probe arrays are invalid")
        mass = probabilities.sum(axis=1, dtype=np.float64) + null.astype(np.float64)
        if np.max(np.abs(mass - 1.0)) > 2e-4:
            raise ValueError("frozen appearance posterior does not conserve mass")
        supported = np.sum(weights, axis=2)
        if np.any((probabilities > 0.0) & ~np.isclose(supported, 1.0, atol=2e-4)):
            raise ValueError("positive frozen candidate lacks normalized support-view mass")
        if np.any((probabilities <= 0.0) & (supported > 2e-4)):
            raise ValueError("zero-mass frozen candidate has support-view mass")
        keys = tuple((str(query_id), int(row)) for query_id, row in zip(query_ids, rows))
        if len(keys) != len(set(keys)):
            raise ValueError("frozen appearance rows repeat query/source identities")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "split_names", split_names)
        object.__setattr__(self, "source_row_indices", rows)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_probabilities", probabilities)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "candidate_view_weights", weights)
        object.__setattr__(self, "candidate_view_scores", scores)
        object.__setattr__(self, "candidate_view_usable", usable)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class AppearanceFeatureNormalizer:
    """Train-only standardization for finite per-view raw appearance values."""

    mean: np.ndarray
    scale: np.ndarray
    feature_indices: np.ndarray

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
        ):
            raise ValueError("appearance feature normalizer is invalid")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_indices", indices)


def _metadata_from_payload(
    payload: Mapping[str, np.ndarray], *, path: Path
) -> tuple[dict[str, Any], str, str]:
    """Validate a supported target-free per-view evidence schema."""

    if "metadata_json" not in payload:
        raise ValueError(f"{path}: frozen appearance artifact lacks metadata")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: frozen appearance metadata is not an object")
    common = (
        metadata.get("contains_target_fields") is False
        and metadata.get("pose_or_ground_truth_used") is False
        and metadata.get("supervision_arrays_loaded") is False
    )
    if metadata.get("format") == FROZEN_APPEARANCE_ARTIFACT_FORMAT:
        strict = metadata.get("strict_frozen_appearance_contract")
        required = {
            "heldout_s0_verification_rows": True,
            "fixed_global_topl": True,
            "candidate_identity_fixed": True,
            "candidate_3d_projection_or_pose_used": False,
            "candidate_reselection": False,
            "support_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        }
        if (
            not common
            or not isinstance(strict, Mapping)
            or any(strict.get(key) is not expected for key, expected in required.items())
        ):
            raise ValueError(f"{path}: invalid target-free frozen appearance contract")
        return metadata, "candidate_view_aligned_ncc", "family_names"
    if metadata.get("format") == FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT:
        strict = metadata.get("strict_frozen_loftr_anchor_contract")
        required = {
            "heldout_s0_verification_rows": True,
            "fixed_global_topl": True,
            "candidate_identity_fixed": True,
            "candidate_3d_projection_or_pose_used": False,
            "candidate_reselection": False,
            "support_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "support_view_descriptor_averaging": False,
            "all_mapping_images_pair_cached": True,
            "pair_cache_image_level_selection": False,
            "anchor_evidence_pose_free": True,
        }
        if (
            not common
            or int(metadata.get("fixed_candidate_top_k", -1)) != 20
            or metadata.get("feature_field") != "candidate_view_features"
            or not isinstance(strict, Mapping)
            or any(strict.get(key) is not expected for key, expected in required.items())
        ):
            raise ValueError(f"{path}: invalid target-free frozen LoFTR anchor contract")
        return metadata, "candidate_view_features", "feature_names"
    raise ValueError(f"{path}: unsupported frozen per-view evidence format")


def load_frozen_appearance_probe_features(
    paths: Sequence[Path],
) -> FrozenAppearanceProbeFeatures:
    """Merge complete direct or LoFTR per-view shards with strict lineage checks."""

    artifacts = tuple(Path(path) for path in paths)
    if not artifacts or len(set(artifacts)) != len(artifacts):
        raise ValueError("frozen appearance probe paths must be unique and non-empty")
    required_base = (
        "metadata_json",
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_view_weights",
        "candidate_view_usable",
    )
    merged: dict[str, list[np.ndarray]] = {
        name: [] for name in required_base if name != "metadata_json"
    }
    merged["candidate_view_scores"] = []
    reference_metadata: dict[str, Any] | None = None
    reference_names: np.ndarray | None = None
    compatibility: dict[str, Any] | None = None
    for path in artifacts:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(set(required_base).difference(payload.files))
            if missing:
                raise ValueError(f"{path}: frozen appearance artifact lacks {missing}")
            arrays = {name: np.asarray(payload[name]).copy() for name in required_base}
            metadata, score_field, name_field = _metadata_from_payload(arrays, path=path)
            dynamic_missing = sorted({score_field, name_field}.difference(payload.files))
            if dynamic_missing:
                raise ValueError(f"{path}: frozen appearance artifact lacks {dynamic_missing}")
            arrays["candidate_view_scores"] = np.asarray(payload[score_field]).copy()
            arrays["feature_names"] = np.asarray(payload[name_field]).copy()
            raw_usable = np.asarray(payload["candidate_view_usable"], dtype=bool).copy()
            if (
                metadata.get("format") == FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT
                and raw_usable.shape == arrays["candidate_view_scores"].shape[:3]
            ):
                raw_usable = np.broadcast_to(
                    raw_usable[..., None], arrays["candidate_view_scores"].shape
                ).copy()
            arrays["candidate_view_usable"] = raw_usable
        names = np.asarray(arrays["feature_names"]).astype(str).reshape(-1)
        row_count = int(metadata.get("row_count", -1))
        query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
        split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
        source_rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
        xy = np.asarray(arrays["verification_xy"], dtype=np.float32)
        tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
        probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
        null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
        weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float32)
        scores = np.asarray(arrays["candidate_view_scores"], dtype=np.float32)
        usable = np.asarray(arrays["candidate_view_usable"], dtype=bool)
        if (
            len(names) == 0
            or len(set(names.tolist())) != len(names)
            or row_count != 192
            or len(query_ids) != row_count
            or split_names.shape != (row_count,)
            or source_rows.shape != (row_count,)
            or xy.shape != (row_count, 2)
            or tracks.ndim != 2
            or tracks.shape[0] != row_count
            or probabilities.shape != tracks.shape
            or null.shape != (row_count,)
            or weights.ndim != 3
            or weights.shape[:2] != tracks.shape
            or scores.shape != (*weights.shape, len(names))
            or usable.shape != scores.shape
        ):
            raise ValueError(f"{path}: frozen appearance artifact tensor shapes differ from metadata")
        item_compatibility = {
            "format": metadata.get("format"),
            "version": metadata.get("version"),
            "strict_contract": metadata.get("strict_frozen_appearance_contract")
            if metadata.get("format") == FROZEN_APPEARANCE_ARTIFACT_FORMAT
            else metadata.get("strict_frozen_loftr_anchor_contract"),
            "score_field": score_field,
            "name_field": name_field,
            "profiles": metadata.get("profiles"),
            "appearance_config": metadata.get("appearance_config"),
            "implementation_hash": metadata.get("implementation_hash"),
            "implementation_source_sha256": (
                metadata.get("implementation", {}).get("source_sha256")
                if isinstance(metadata.get("implementation"), Mapping)
                else None
            ),
        }
        if reference_names is None:
            reference_names = names
            reference_metadata = metadata
            compatibility = item_compatibility
        elif not np.array_equal(names, reference_names) or item_compatibility != compatibility:
            raise ValueError(f"{path}: frozen appearance probe configuration differs")
        for name in merged:
            merged[name].append(arrays[name])
    if reference_metadata is None or reference_names is None:
        raise RuntimeError("frozen appearance probe has no loaded artifacts")
    output = {name: np.concatenate(parts, axis=0) for name, parts in merged.items()}
    return FrozenAppearanceProbeFeatures(
        paths=artifacts,
        query_ids=output["verification_query_ids"],
        split_names=output["split_names"],
        source_row_indices=output["verification_source_row_indices"],
        xy=output["verification_xy"],
        candidate_track_ids=output["candidate_track_ids"],
        candidate_probabilities=output["candidate_probabilities"],
        null_probabilities=output["null_probabilities"],
        candidate_view_weights=output["candidate_view_weights"],
        candidate_view_scores=output["candidate_view_scores"],
        candidate_view_usable=output["candidate_view_usable"],
        feature_names=tuple(reference_names.tolist()),
        metadata=reference_metadata,
    )


def appearance_feature_indices_for_family(
    family: str, *, feature_names: Sequence[str]
) -> np.ndarray:
    """Resolve a predeclared residual family against one frozen schema."""

    requested = FROZEN_APPEARANCE_RESIDUAL_FAMILIES.get(str(family))
    names = tuple(str(value) for value in feature_names)
    if requested is None:
        raise ValueError(f"unsupported frozen appearance residual family: {family!r}")
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"frozen appearance schema lacks family fields: {missing}")
    return np.asarray([names.index(name) for name in requested], dtype=np.int64)


def fit_appearance_feature_normalizer(
    features: FrozenAppearanceProbeFeatures,
    *,
    feature_indices: np.ndarray,
    train_rows: np.ndarray,
) -> AppearanceFeatureNormalizer:
    """Fit standardization only from real finite train support-view values."""

    indices = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    if (
        len(indices) == 0
        or np.any((indices < 0) | (indices >= len(features.feature_names)))
        or len(rows) == 0
        or np.any((rows < 0) | (rows >= len(features.query_ids)))
        or len(set(rows.tolist())) != len(rows)
    ):
        raise ValueError("appearance normalizer rows or feature indices are invalid")
    values = np.asarray(features.candidate_view_scores[rows][..., indices], dtype=np.float32)
    supported = features.candidate_view_weights[rows][..., None] > 0.0
    finite = supported & np.isfinite(values)
    counts = finite.sum(axis=(0, 1, 2), dtype=np.int64)
    if np.any(counts == 0):
        missing = [features.feature_names[index] for index, count in zip(indices, counts) if count == 0]
        raise ValueError(f"appearance normalizer has no finite train values for {missing}")
    masked = np.where(finite, values, 0.0).astype(np.float64, copy=False)
    mean64 = masked.sum(axis=(0, 1, 2), dtype=np.float64) / counts
    variance = np.maximum(
        np.square(masked).sum(axis=(0, 1, 2), dtype=np.float64) / counts - mean64**2,
        0.0,
    )
    return AppearanceFeatureNormalizer(
        mean=mean64.astype(np.float32),
        scale=np.maximum(np.sqrt(variance), 1e-3).astype(np.float32),
        feature_indices=indices,
    )


def normalized_appearance_model_input(
    features: FrozenAppearanceProbeFeatures,
    normalizer: AppearanceFeatureNormalizer,
    *,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Return standardized raw values plus explicit finite-evidence flags."""

    row_indices = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    if np.any((row_indices < 0) | (row_indices >= len(features.query_ids))):
        raise ValueError("appearance model rows are out of range")
    raw = np.asarray(
        features.candidate_view_scores[row_indices][..., normalizer.feature_indices],
        dtype=np.float32,
    )
    supported = features.candidate_view_weights[row_indices][..., None] > 0.0
    finite = supported & np.isfinite(raw)
    standardized = (
        np.where(finite, raw, normalizer.mean) - normalizer.mean
    ) / normalizer.scale
    return np.concatenate(
        [standardized.astype(np.float32), finite.astype(np.float32)], axis=-1
    )


def fixed_prior_log_probabilities(
    features: FrozenAppearanceProbeFeatures,
    *,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return immutable candidate/null logs and fixed support-view log weights."""

    row_indices = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    candidate = np.asarray(features.candidate_probabilities[row_indices], dtype=np.float32)
    null = np.asarray(features.null_probabilities[row_indices], dtype=np.float32)
    weights = np.asarray(features.candidate_view_weights[row_indices], dtype=np.float32)
    candidate_log = np.full(candidate.shape, -np.inf, dtype=np.float32)
    positive_candidate = candidate > 0.0
    candidate_log[positive_candidate] = np.log(candidate[positive_candidate])
    weight_log = np.full(weights.shape, -np.inf, dtype=np.float32)
    positive_weight = weights > 0.0
    weight_log[positive_weight] = np.log(weights[positive_weight])
    return candidate_log, np.log(null).astype(np.float32), weight_log


class FixedPriorPerViewLinearResidual(nn.Module):
    """A zero-preserving per-view residual over immutable candidate priors."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("appearance residual input dimension must be positive")
        self.linear = nn.Linear(int(input_dim), 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(
        self,
        model_input: torch.Tensor,
        candidate_log_prior: torch.Tensor,
        null_log_prior: torch.Tensor,
        view_log_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            model_input.ndim != 4
            or candidate_log_prior.ndim != 2
            or null_log_prior.shape != (model_input.shape[0],)
            or view_log_weights.shape != model_input.shape[:3]
            or candidate_log_prior.shape != model_input.shape[:2]
        ):
            raise ValueError("appearance residual tensors are incompatible")
        residual = self.linear(model_input).squeeze(-1)
        mixture = torch.logsumexp(view_log_weights + residual, dim=2)
        candidates = candidate_log_prior + mixture
        logits = torch.cat([candidates, null_log_prior[:, None]], dim=1)
        return logits, residual


def set_membership_nll(log_probabilities: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """Negative log mass of an exact track or explicit-null training target."""

    if log_probabilities.shape != membership.shape or log_probabilities.ndim != 2:
        raise ValueError("appearance residual target membership is incompatible")
    selected = torch.where(
        membership.to(dtype=torch.bool),
        log_probabilities,
        torch.full_like(log_probabilities, -torch.inf),
    )
    return -torch.logsumexp(selected, dim=1).mean()


def fit_fixedprior_linear_residual(
    *,
    features: FrozenAppearanceProbeFeatures,
    family: str,
    train_rows: np.ndarray,
    target_membership: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[FixedPriorPerViewLinearResidual, AppearanceFeatureNormalizer, dict[str, Any]]:
    """Fit one train-only residual model without touching validation targets."""

    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    targets = np.asarray(target_membership, dtype=bool)
    if (
        len(rows) == 0
        or targets.shape != (len(rows), features.candidate_track_ids.shape[1] + 1)
        or np.any(targets.sum(axis=1) != 1)
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
    ):
        raise ValueError("appearance residual fit arguments are invalid")
    indices = appearance_feature_indices_for_family(
        family, feature_names=features.feature_names
    )
    normalizer = fit_appearance_feature_normalizer(
        features, feature_indices=indices, train_rows=rows
    )
    inputs = normalized_appearance_model_input(features, normalizer, rows=rows)
    candidate_log, null_log, view_log = fixed_prior_log_probabilities(features, rows=rows)
    candidate_tensor = torch.from_numpy(candidate_log).to(device)
    null_tensor = torch.from_numpy(null_log).to(device)
    view_tensor = torch.from_numpy(view_log).to(device)
    input_tensor = torch.from_numpy(inputs).to(device)
    target_tensor = torch.from_numpy(targets).to(device=device, dtype=torch.bool)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = FixedPriorPerViewLinearResidual(input_tensor.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed))
    last_loss = float("nan")
    model.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(rows))
        for begin in range(0, len(order), int(batch_size)):
            batch = torch.as_tensor(order[begin : begin + int(batch_size)], device=device)
            logits, _residual = model(
                input_tensor[batch],
                candidate_tensor[batch],
                null_tensor[batch],
                view_tensor[batch],
            )
            loss = set_membership_nll(F.log_softmax(logits, dim=1), target_tensor[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    model.eval()
    return model, normalizer, {
        "family": str(family),
        "architecture": "fixed_prior_per_view_linear_residual_no_bias_v1",
        "feature_names": [features.feature_names[index] for index in indices],
        "train_row_count": int(len(rows)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "last_train_loss": last_loss,
        "null_logit": "immutable_base_null_prior",
        "view_marginalization": "fixed_maplet_view_weight_logsumexp_v1",
        "zero_residual_reproduces_fixed_posterior": True,
    }


@torch.inference_mode()
def predict_fixedprior_linear_residual(
    *,
    model: FixedPriorPerViewLinearResidual,
    features: FrozenAppearanceProbeFeatures,
    normalizer: AppearanceFeatureNormalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one frozen residual model to every train/validation row."""

    if int(batch_size) <= 0:
        raise ValueError("appearance residual prediction batch size must be positive")
    candidate_log, null_log, view_log = fixed_prior_log_probabilities(features)
    output_candidate: list[np.ndarray] = []
    output_null: list[np.ndarray] = []
    output_residual: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(features.query_ids), int(batch_size)):
        end = min(start + int(batch_size), len(features.query_ids))
        inputs = normalized_appearance_model_input(
            features, normalizer, rows=np.arange(start, end, dtype=np.int64)
        )
        logits, residual = model(
            torch.from_numpy(inputs).to(device),
            torch.from_numpy(candidate_log[start:end]).to(device),
            torch.from_numpy(null_log[start:end]).to(device),
            torch.from_numpy(view_log[start:end]).to(device),
        )
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        output_candidate.append(probabilities[:, :-1])
        output_null.append(probabilities[:, -1])
        output_residual.append(residual.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(output_candidate, axis=0),
        np.concatenate(output_null, axis=0),
        np.concatenate(output_residual, axis=0),
    )


def zero_residual_posterior(
    features: FrozenAppearanceProbeFeatures,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact invariant baseline for zero visual residual evidence."""

    return (
        np.asarray(features.candidate_probabilities, dtype=np.float32).copy(),
        np.asarray(features.null_probabilities, dtype=np.float32).copy(),
    )
