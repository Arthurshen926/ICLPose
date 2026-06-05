"""Confidence-supervised descriptor refinement for Stage C2.9."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.correspondence_confidence import (
    CalibratedLogisticConfidence,
    vectorize_match_rows,
)
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.tokens import TokenBankManifest


@dataclass(frozen=True)
class ConfidenceRefinementTrainingSet:
    query_features: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    teacher_scores: np.ndarray
    reprojection_strides: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        query = np.asarray(self.query_features, dtype=np.float32)
        candidates = np.asarray(self.candidate_features, dtype=np.float32)
        mask = np.asarray(self.candidate_mask, dtype=bool)
        labels = np.asarray(self.labels, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        teacher = np.asarray(self.teacher_scores, dtype=np.float32)
        reproj = np.asarray(self.reprojection_strides, dtype=np.float32)
        if query.ndim != 2:
            raise ValueError("query_features must have shape (N, C)")
        if candidates.ndim != 3:
            raise ValueError("candidate_features must have shape (N, K, C)")
        if candidates.shape[0] != query.shape[0] or candidates.shape[2] != query.shape[1]:
            raise ValueError("candidate_features must match query sample count and feature dim")
        if mask.shape != candidates.shape[:2]:
            raise ValueError("candidate_mask must have shape (N, K)")
        for name, value in (
            ("labels", labels),
            ("weights", weights),
            ("teacher_scores", teacher),
            ("reprojection_strides", reproj),
        ):
            if value.shape != candidates.shape[:2]:
                raise ValueError(f"{name} must have shape (N, K)")
        if query.shape[0] and not np.all(np.any(mask & (labels > 0.5), axis=1)):
            raise ValueError("each sample must contain at least one positive candidate")
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "candidate_features", candidates)
        object.__setattr__(self, "candidate_mask", mask)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "weights", np.clip(weights, 0.0, None))
        object.__setattr__(self, "teacher_scores", np.clip(teacher, 1e-4, 1.0 - 1e-4))
        object.__setattr__(self, "reprojection_strides", reproj)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def sample_count(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.query_features.shape[1]) if self.query_features.ndim == 2 else 0

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_features.shape[1]) if self.candidate_features.ndim == 3 else 0


@dataclass(frozen=True)
class ConfidenceDescriptorRefinementConfig:
    output_dim: int = 128
    hidden_dim: int = 256
    steps: int = 800
    batch_size: int = 512
    lr: float = 1e-3
    temperature: float = 0.07
    margin_loss_weight: float = 0.2
    margin: float = 0.2
    distill_loss_weight: float = 0.2
    anchor_loss_weight: float = 0.05
    score_anchor_loss_weight: float = 0.0
    eval_split_fraction: float = 0.1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if min(
            float(self.margin_loss_weight),
            float(self.distill_loss_weight),
            float(self.anchor_loss_weight),
            float(self.score_anchor_loss_weight),
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")


@dataclass(frozen=True)
class ConfidenceDescriptorRefinementSummary:
    initial_loss: float
    final_loss: float
    raw_train_top1_acc: float
    raw_eval_top1_acc: float
    train_top1_acc: float
    eval_top1_acc: float
    sample_count: int
    train_sample_count: int
    eval_sample_count: int
    input_dim: int
    output_dim: int
    candidate_count: int
    steps: int
    batch_size: int
    temperature: float
    margin_loss_weight: float
    distill_loss_weight: float
    anchor_loss_weight: float
    score_anchor_loss_weight: float


@dataclass(frozen=True)
class ConfidenceDescriptorRefinementRun:
    model: "ConfidenceDescriptorRefiner"
    summary: ConfidenceDescriptorRefinementSummary

    def encode_rows(self, rows: np.ndarray, device: str = "cpu", batch_size: int = 65536) -> np.ndarray:
        return encode_rows_with_confidence_refiner(self.model, rows, device=device, batch_size=batch_size)


class ConfidenceDescriptorRefiner(nn.Module):
    """Shared descriptor refiner initialized close to identity when possible."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.Identity()
        self.projection = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.residual = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        if self.input_dim == self.output_dim:
            nn.init.eye_(self.projection.weight)
        else:
            nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.input_norm(features)
        projected = self.projection(values)
        return F.normalize(projected + self.residual(projected), dim=-1, eps=1e-8)


@dataclass(frozen=True)
class DiagonalDescriptorSelectionConfig:
    steps: int = 600
    batch_size: int = 512
    lr: float = 5e-2
    temperature: float = 0.07
    margin_loss_weight: float = 0.2
    margin: float = 0.2
    distill_loss_weight: float = 0.1
    anchor_loss_weight: float = 0.05
    scale_regularization_weight: float = 0.01
    max_log_scale: float = 0.5
    eval_split_fraction: float = 0.1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if min(
            float(self.margin_loss_weight),
            float(self.distill_loss_weight),
            float(self.anchor_loss_weight),
            float(self.scale_regularization_weight),
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.max_log_scale <= 0.0:
            raise ValueError("max_log_scale must be positive")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")


@dataclass(frozen=True)
class DiagonalDescriptorSelectionSummary:
    initial_loss: float
    final_loss: float
    raw_train_top1_acc: float
    raw_eval_top1_acc: float
    train_top1_acc: float
    eval_top1_acc: float
    sample_count: int
    train_sample_count: int
    eval_sample_count: int
    input_dim: int
    output_dim: int
    candidate_count: int
    steps: int
    batch_size: int
    temperature: float
    margin_loss_weight: float
    distill_loss_weight: float
    anchor_loss_weight: float
    scale_regularization_weight: float
    max_log_scale: float
    channel_scale_min: float
    channel_scale_max: float
    channel_scale_mean: float
    channel_scale_std: float
    effective_channel_count: float


@dataclass(frozen=True)
class DiagonalDescriptorSelectionRun:
    model: "DiagonalDescriptorSelector"
    summary: DiagonalDescriptorSelectionSummary

    def encode_rows(self, rows: np.ndarray, device: str = "cpu", batch_size: int = 65536) -> np.ndarray:
        return encode_rows_with_diagonal_selector(self.model, rows, device=device, batch_size=batch_size)


class DiagonalDescriptorSelector(nn.Module):
    """Conservative channel reweighting selector that cannot rotate descriptor space."""

    def __init__(self, feature_dim: int, max_log_scale: float = 0.5) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if float(max_log_scale) <= 0.0:
            raise ValueError("max_log_scale must be positive")
        self.input_dim = int(feature_dim)
        self.output_dim = int(feature_dim)
        self.max_log_scale = float(max_log_scale)
        self.log_channel_scale = nn.Parameter(torch.zeros((self.input_dim,), dtype=torch.float32))

    def channel_scales(self) -> torch.Tensor:
        clipped = torch.tanh(self.log_channel_scale) * float(self.max_log_scale)
        return torch.exp(clipped)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features * self.channel_scales().to(features.device), dim=-1, eps=1e-8)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _query_feature_for_token(feature_map: np.ndarray, token_index: int) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = values.shape
    y_idx, x_idx = divmod(int(token_index), int(width))
    if y_idx < 0 or y_idx >= height:
        raise ValueError(f"token_index {token_index} is outside feature map")
    return values[:, y_idx, x_idx].reshape(channels).astype(np.float32, copy=True)


def _row_float(row: Mapping[str, Any], key: str, default: float) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _match_label_weight(row: Mapping[str, Any], weak_positive_weight: float) -> tuple[float, float, bool]:
    if bool(row.get("strong_positive_label", False)) or bool(row.get("patch_positive_label", False)):
        return 1.0, 1.0, False
    if bool(row.get("weak_positive_label", False)):
        return 1.0, float(weak_positive_weight), False
    if bool(row.get("ignore_label", False)):
        return 0.0, 0.0, True
    if bool(row.get("hard_negative_label", False)) or _row_float(row, "gt_reproj_error_stride", 1e9) > 2.0:
        return 0.0, 1.0, False
    return 0.0, 0.5, False


def _precision_weighted_positive_weight(
    label: float,
    weight: float,
    reprojection_stride: float,
    positive_reprojection_weight: float,
    positive_reprojection_scale_stride: float,
) -> float:
    if label <= 0.5 or positive_reprojection_weight <= 0.0:
        return float(weight)
    if not np.isfinite(reprojection_stride):
        return float(weight)
    scale = max(float(positive_reprojection_scale_stride), 1e-6)
    strength = min(max(float(positive_reprojection_weight), 0.0), 1.0)
    precision = float(np.exp(-max(float(reprojection_stride), 0.0) / scale))
    return float(weight) * ((1.0 - strength) + strength * precision)


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, float, float]:
    label, weight, ignore = _match_label_weight(row, weak_positive_weight=0.35)
    if ignore:
        bucket = 3
    elif label > 0.5:
        bucket = 0
    elif bool(row.get("hard_negative_label", False)):
        bucket = 1
    else:
        bucket = 2
    confidence = float(row.get("teacher_score", row.get("similarity", 0.0)) or 0.0)
    reproj = _row_float(row, "gt_reproj_error_stride", 1e9)
    return bucket, -confidence, reproj


def build_confidence_refinement_samples(
    match_jsonl: str | Path,
    query_manifest: str | Path,
    landmark_bank: str | Path,
    layer_name: str = "radio_final",
    max_candidates_per_token: int = 16,
    max_samples: int = 0,
    min_negative_candidates: int = 0,
    negative_sampling_mode: str = "default",
    weak_positive_weight: float = 0.35,
    positive_reprojection_weight: float = 0.0,
    positive_reprojection_scale_stride: float = 1.0,
    teacher_model: CalibratedLogisticConfidence | None = None,
    teacher_feature_set: str = "full",
    seed: int = 0,
) -> tuple[ConfidenceRefinementTrainingSet, dict[str, object]]:
    if int(max_candidates_per_token) <= 1:
        raise ValueError("max_candidates_per_token must be greater than 1")
    if int(min_negative_candidates) < 0:
        raise ValueError("min_negative_candidates must be non-negative")
    if str(negative_sampling_mode) not in {"default", "semi_hard"}:
        raise ValueError("negative_sampling_mode must be one of: default, semi_hard")
    if float(positive_reprojection_weight) < 0.0 or float(positive_reprojection_weight) > 1.0:
        raise ValueError("positive_reprojection_weight must be in [0, 1]")
    if float(positive_reprojection_scale_stride) <= 0.0:
        raise ValueError("positive_reprojection_scale_stride must be positive")
    rows = _load_jsonl(Path(match_jsonl))
    if teacher_model is not None and rows:
        matrix, _labels, _keep, _names = vectorize_match_rows(rows, feature_set=teacher_feature_set)
        teacher_scores = teacher_model.predict_proba(matrix)
        for row, score in zip(rows, teacher_scores):
            row["teacher_score"] = float(score)
    manifest = TokenBankManifest.from_json(Path(query_manifest))
    manifest.validate(verify_checksums=False)
    record_by_query = {record.image_id: record for record in manifest.records}
    bank = load_selected_track_bank_npz(Path(landmark_bank))
    track_ids = sorted(bank.tracks)
    track_row = {int(track_id): idx for idx, track_id in enumerate(track_ids)}
    features = np.stack([np.asarray(bank.tracks[int(track_id)].mean_feature, dtype=np.float32) for track_id in track_ids], axis=0)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if query_id not in record_by_query:
            continue
        track_id = int(row.get("track_id", -1))
        if track_id not in track_row:
            continue
        grouped[(query_id, int(row.get("token_index", 0)))].append(row)

    rng = np.random.default_rng(int(seed))
    keys = sorted(grouped)
    if max_samples > 0 and len(keys) > int(max_samples):
        keys = [keys[int(idx)] for idx in sorted(rng.choice(len(keys), size=int(max_samples), replace=False).tolist())]

    query_cache: dict[str, np.ndarray] = {}
    query_features = []
    candidate_features = []
    candidate_mask = []
    labels = []
    weights = []
    teacher = []
    reproj = []
    hard_negative_count = 0
    semi_hard_negative_count = 0
    weak_positive_count = 0
    positive_reprojection_weighted_count = 0
    skipped_without_positive = 0
    for query_id, token_index in keys:
        eligible_rows = []
        for row in sorted(grouped[(query_id, token_index)], key=_row_sort_key):
            label, weight, ignore = _match_label_weight(row, weak_positive_weight=weak_positive_weight)
            if ignore or weight <= 0.0:
                continue
            item = dict(row)
            item["_label"] = label
            reproj_stride = _row_float(item, "gt_reproj_error_stride", np.inf)
            item["_weight"] = _precision_weighted_positive_weight(
                label,
                weight,
                reproj_stride,
                positive_reprojection_weight=float(positive_reprojection_weight),
                positive_reprojection_scale_stride=float(positive_reprojection_scale_stride),
            )
            eligible_rows.append(item)
        if int(min_negative_candidates) > 0:
            positives = [row for row in eligible_rows if float(row["_label"]) > 0.5]
            negatives = [row for row in eligible_rows if float(row["_label"]) <= 0.5]
            negative_count = min(
                int(min_negative_candidates),
                max(int(max_candidates_per_token) - 1, 0),
                len(negatives),
            )
            positive_count = min(len(positives), int(max_candidates_per_token) - negative_count)
            if str(negative_sampling_mode) == "semi_hard":
                semi_hard_negatives = [
                    row for row in negatives if _row_float(row, "gt_reproj_error_stride", np.inf) > 2.0
                ]
                other_negatives = [row for row in negatives if id(row) not in {id(item) for item in semi_hard_negatives}]
                selected_negatives = semi_hard_negatives[:negative_count]
                if len(selected_negatives) < negative_count:
                    selected_negatives.extend(other_negatives[: negative_count - len(selected_negatives)])
            else:
                selected_negatives = negatives[:negative_count]
            selected_rows = positives[:positive_count] + selected_negatives
            selected_ids = {id(row) for row in selected_rows}
            for row in eligible_rows:
                if id(row) in selected_ids:
                    continue
                selected_rows.append(row)
                selected_ids.add(id(row))
                if len(selected_rows) >= int(max_candidates_per_token):
                    break
        else:
            selected_rows = []
            for item in eligible_rows:
                selected_rows.append(item)
                if len(selected_rows) >= int(max_candidates_per_token):
                    break
        if len(selected_rows) > int(max_candidates_per_token):
            selected_rows = selected_rows[: int(max_candidates_per_token)]
        if not any(float(row["_label"]) > 0.5 for row in selected_rows):
            skipped_without_positive += 1
            continue
        record = record_by_query[query_id]
        if query_id not in query_cache:
            with np.load(record.token_path) as data:
                query_cache[query_id] = np.asarray(data[layer_name], dtype=np.float32)
        q = _query_feature_for_token(query_cache[query_id], token_index)
        cand = np.zeros((int(max_candidates_per_token), q.shape[0]), dtype=np.float32)
        mask = np.zeros((int(max_candidates_per_token),), dtype=bool)
        lab = np.zeros((int(max_candidates_per_token),), dtype=np.float32)
        w = np.zeros((int(max_candidates_per_token),), dtype=np.float32)
        t = np.full((int(max_candidates_per_token),), 1e-4, dtype=np.float32)
        d = np.full((int(max_candidates_per_token),), np.inf, dtype=np.float32)
        for idx, row in enumerate(selected_rows):
            cand[idx] = features[track_row[int(row["track_id"])]]
            mask[idx] = True
            lab[idx] = float(row["_label"])
            w[idx] = float(row["_weight"])
            t[idx] = float(row.get("teacher_score", 0.9 if lab[idx] > 0.5 else 0.1))
            d[idx] = _row_float(row, "gt_reproj_error_stride", np.inf)
            if bool(row.get("hard_negative_label", False)):
                hard_negative_count += 1
            if lab[idx] <= 0.5 and _row_float(row, "gt_reproj_error_stride", np.inf) > 2.0:
                semi_hard_negative_count += 1
            if bool(row.get("weak_positive_label", False)):
                weak_positive_count += 1
            if (
                lab[idx] > 0.5
                and float(positive_reprojection_weight) > 0.0
                and np.isfinite(_row_float(row, "gt_reproj_error_stride", np.inf))
            ):
                positive_reprojection_weighted_count += 1
        query_features.append(q)
        candidate_features.append(cand)
        candidate_mask.append(mask)
        labels.append(lab)
        weights.append(w)
        teacher.append(t)
        reproj.append(d)
    samples = ConfidenceRefinementTrainingSet(
        query_features=np.stack(query_features, axis=0) if query_features else np.zeros((0, bank.feature_dim), dtype=np.float32),
        candidate_features=np.stack(candidate_features, axis=0) if candidate_features else np.zeros((0, max_candidates_per_token, bank.feature_dim), dtype=np.float32),
        candidate_mask=np.stack(candidate_mask, axis=0) if candidate_mask else np.zeros((0, max_candidates_per_token), dtype=bool),
        labels=np.stack(labels, axis=0) if labels else np.zeros((0, max_candidates_per_token), dtype=np.float32),
        weights=np.stack(weights, axis=0) if weights else np.zeros((0, max_candidates_per_token), dtype=np.float32),
        teacher_scores=np.stack(teacher, axis=0) if teacher else np.zeros((0, max_candidates_per_token), dtype=np.float32),
        reprojection_strides=np.stack(reproj, axis=0) if reproj else np.zeros((0, max_candidates_per_token), dtype=np.float32),
        metadata={
            "source_match_jsonl": str(match_jsonl),
            "max_candidates_per_token": int(max_candidates_per_token),
            "min_negative_candidates": int(min_negative_candidates),
            "negative_sampling_mode": str(negative_sampling_mode),
            "weak_positive_weight": float(weak_positive_weight),
            "positive_reprojection_weight": float(positive_reprojection_weight),
            "positive_reprojection_scale_stride": float(positive_reprojection_scale_stride),
        },
    )
    meta = {
        "sample_count": int(samples.sample_count),
        "candidate_count": int(samples.candidate_count),
        "hard_negative_count": int(hard_negative_count),
        "semi_hard_negative_count": int(semi_hard_negative_count),
        "weak_positive_count": int(weak_positive_count),
        "positive_reprojection_weighted_count": int(positive_reprojection_weighted_count),
        "skipped_without_positive": int(skipped_without_positive),
        "query_count": int(len({key[0] for key in keys})),
    }
    return samples, meta


def _split_indices(count: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    eval_count = int(round(count * float(eval_fraction)))
    if eval_fraction > 0.0 and count > 1:
        eval_count = max(1, eval_count)
    eval_count = min(eval_count, max(count - 1, 0))
    return indices[eval_count:], indices[:eval_count]


def _subset(samples: ConfidenceRefinementTrainingSet, indices: np.ndarray) -> ConfidenceRefinementTrainingSet:
    return ConfidenceRefinementTrainingSet(
        query_features=samples.query_features[indices],
        candidate_features=samples.candidate_features[indices],
        candidate_mask=samples.candidate_mask[indices],
        labels=samples.labels[indices],
        weights=samples.weights[indices],
        teacher_scores=samples.teacher_scores[indices],
        reprojection_strides=samples.reprojection_strides[indices],
        metadata=samples.metadata,
    )


def _descriptor_loss(
    model: ConfidenceDescriptorRefiner,
    query: torch.Tensor,
    candidates: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    teacher_scores: torch.Tensor,
    config: ConfidenceDescriptorRefinementConfig,
) -> torch.Tensor:
    query_z = model(query)
    cand_z = model(candidates.reshape(-1, candidates.shape[-1])).reshape(candidates.shape[0], candidates.shape[1], -1)
    logits = torch.einsum("bd,bkd->bk", query_z, cand_z) / float(config.temperature)
    valid_logits = logits.masked_fill(~mask, -1e9)
    pos = mask & (labels > 0.5)
    log_weights = torch.log(torch.clamp(weights, min=1e-6))
    weighted_logits = valid_logits + log_weights
    numerator = torch.logsumexp(weighted_logits.masked_fill(~pos, -1e9), dim=1)
    denominator = torch.logsumexp(weighted_logits, dim=1)
    loss = torch.mean(denominator - numerator)
    if float(config.margin_loss_weight) > 0.0:
        pos_scores = logits.masked_fill(~pos, -1e9)
        neg = mask & (labels <= 0.5)
        neg_scores = logits.masked_fill(~neg, -1e9)
        best_pos = torch.max(pos_scores, dim=1).values
        best_neg = torch.max(neg_scores, dim=1).values
        has_neg = torch.any(neg, dim=1)
        if bool(has_neg.any().item()):
            loss = loss + float(config.margin_loss_weight) * F.relu(best_neg[has_neg] - best_pos[has_neg] + float(config.margin)).mean()
    if float(config.distill_loss_weight) > 0.0:
        teacher_logits = torch.logit(torch.clamp(teacher_scores, 1e-4, 1.0 - 1e-4)).masked_fill(~mask, -1e9)
        teacher_prob = F.softmax(teacher_logits / max(float(config.temperature), 1e-6), dim=1)
        student_log = F.log_softmax(valid_logits, dim=1)
        loss = loss + float(config.distill_loss_weight) * F.kl_div(student_log, teacher_prob, reduction="batchmean")
    if float(config.anchor_loss_weight) > 0.0 and model.input_dim == model.output_dim:
        anchor_q = F.normalize(query, dim=-1, eps=1e-8)
        anchor_c = F.normalize(candidates, dim=-1, eps=1e-8)
        anchor_loss = 1.0 - torch.sum(query_z * anchor_q, dim=-1)
        cand_anchor_loss = 1.0 - torch.sum(cand_z * anchor_c, dim=-1)
        loss = loss + float(config.anchor_loss_weight) * (
            anchor_loss.mean() + cand_anchor_loss[mask].mean()
        )
    score_anchor_weight = float(getattr(config, "score_anchor_loss_weight", 0.0))
    if score_anchor_weight > 0.0 and model.input_dim == model.output_dim:
        raw_q = F.normalize(query, dim=-1, eps=1e-8)
        raw_c = F.normalize(candidates, dim=-1, eps=1e-8)
        raw_scores = torch.einsum("bd,bkd->bk", raw_q, raw_c)
        student_scores = torch.einsum("bd,bkd->bk", query_z, cand_z)
        score_anchor = F.mse_loss(student_scores[mask], raw_scores[mask])
        loss = loss + score_anchor_weight * score_anchor
    return loss


def _top1_acc(model: ConfidenceDescriptorRefiner | None, samples: ConfidenceRefinementTrainingSet, device: torch.device) -> float:
    if samples.sample_count == 0:
        return 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, samples.sample_count, 2048):
            end = min(start + 2048, samples.sample_count)
            q = torch.as_tensor(samples.query_features[start:end], dtype=torch.float32, device=device)
            c = torch.as_tensor(samples.candidate_features[start:end], dtype=torch.float32, device=device)
            mask = torch.as_tensor(samples.candidate_mask[start:end], dtype=torch.bool, device=device)
            labels = torch.as_tensor(samples.labels[start:end], dtype=torch.float32, device=device)
            if model is None:
                qz = F.normalize(q, dim=-1, eps=1e-8)
                cz = F.normalize(c, dim=-1, eps=1e-8)
            else:
                qz = model(q)
                cz = model(c.reshape(-1, c.shape[-1])).reshape(c.shape[0], c.shape[1], -1)
            scores = torch.einsum("bd,bkd->bk", qz, cz).masked_fill(~mask, -1e9)
            top = torch.argmax(scores, dim=1)
            correct += int((labels[torch.arange(labels.shape[0], device=device), top] > 0.5).sum().item())
    return float(correct / max(samples.sample_count, 1))


def train_confidence_descriptor_refiner(
    samples: ConfidenceRefinementTrainingSet,
    config: ConfidenceDescriptorRefinementConfig | None = None,
) -> ConfidenceDescriptorRefinementRun:
    config = config or ConfidenceDescriptorRefinementConfig(output_dim=samples.feature_dim)
    if samples.sample_count == 0:
        raise ValueError("at least one training sample is required")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, float(config.eval_split_fraction), int(config.seed))
    train_samples = _subset(samples, train_idx)
    eval_samples = _subset(samples, eval_idx) if eval_idx.size else _subset(samples, train_idx[:0])
    model = ConfidenceDescriptorRefiner(samples.feature_dim, int(config.output_dim), int(config.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))

    def loss_for(subset: ConfidenceRefinementTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, subset.sample_count, min(int(config.batch_size), 2048)):
                end = min(start + min(int(config.batch_size), 2048), subset.sample_count)
                loss = _descriptor_loss(
                    model,
                    torch.as_tensor(subset.query_features[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.candidate_features[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.candidate_mask[start:end], dtype=torch.bool, device=device),
                    torch.as_tensor(subset.labels[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.weights[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.teacher_scores[start:end], dtype=torch.float32, device=device),
                    config,
                )
                total += float(loss.detach().cpu()) * int(end - start)
                count += int(end - start)
        return float(total / max(count, 1))

    initial_loss = loss_for(train_samples)
    for _step in range(int(config.steps)):
        count = min(int(config.batch_size), train_samples.sample_count)
        batch = rng.choice(train_samples.sample_count, size=count, replace=False)
        loss = _descriptor_loss(
            model,
            torch.as_tensor(train_samples.query_features[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.candidate_features[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.candidate_mask[batch], dtype=torch.bool, device=device),
            torch.as_tensor(train_samples.labels[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.weights[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.teacher_scores[batch], dtype=torch.float32, device=device),
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = loss_for(train_samples)
    summary = ConfidenceDescriptorRefinementSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        raw_train_top1_acc=_top1_acc(None, train_samples, device),
        raw_eval_top1_acc=_top1_acc(None, eval_samples, device),
        train_top1_acc=_top1_acc(model, train_samples, device),
        eval_top1_acc=_top1_acc(model, eval_samples, device),
        sample_count=int(samples.sample_count),
        train_sample_count=int(train_samples.sample_count),
        eval_sample_count=int(eval_samples.sample_count),
        input_dim=int(samples.feature_dim),
        output_dim=int(config.output_dim),
        candidate_count=int(samples.candidate_count),
        steps=int(config.steps),
        batch_size=int(config.batch_size),
        temperature=float(config.temperature),
        margin_loss_weight=float(config.margin_loss_weight),
        distill_loss_weight=float(config.distill_loss_weight),
        anchor_loss_weight=float(config.anchor_loss_weight),
        score_anchor_loss_weight=float(config.score_anchor_loss_weight),
    )
    return ConfidenceDescriptorRefinementRun(model=model.cpu(), summary=summary)


def _diagonal_descriptor_loss(
    model: DiagonalDescriptorSelector,
    query: torch.Tensor,
    candidates: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    teacher_scores: torch.Tensor,
    config: DiagonalDescriptorSelectionConfig,
) -> torch.Tensor:
    loss = _descriptor_loss(
        model,  # type: ignore[arg-type]
        query,
        candidates,
        mask,
        labels,
        weights,
        teacher_scores,
        config,  # type: ignore[arg-type]
    )
    if float(config.scale_regularization_weight) > 0.0:
        loss = loss + float(config.scale_regularization_weight) * torch.mean(model.log_channel_scale**2)
    return loss


def _effective_channel_count(scales: np.ndarray) -> float:
    values = np.asarray(scales, dtype=np.float64)
    if values.size == 0:
        return 0.0
    energy = np.square(values)
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    probability = energy / total
    return float(1.0 / max(float(np.sum(np.square(probability))), 1e-12))


def train_diagonal_descriptor_selector(
    samples: ConfidenceRefinementTrainingSet,
    config: DiagonalDescriptorSelectionConfig | None = None,
) -> DiagonalDescriptorSelectionRun:
    config = config or DiagonalDescriptorSelectionConfig()
    if samples.sample_count == 0:
        raise ValueError("at least one training sample is required")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, float(config.eval_split_fraction), int(config.seed))
    train_samples = _subset(samples, train_idx)
    eval_samples = _subset(samples, eval_idx) if eval_idx.size else _subset(samples, train_idx[:0])
    model = DiagonalDescriptorSelector(samples.feature_dim, max_log_scale=float(config.max_log_scale)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=0.0)
    rng = np.random.default_rng(int(config.seed))

    def loss_for(subset: ConfidenceRefinementTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, subset.sample_count, min(int(config.batch_size), 2048)):
                end = min(start + min(int(config.batch_size), 2048), subset.sample_count)
                loss = _diagonal_descriptor_loss(
                    model,
                    torch.as_tensor(subset.query_features[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.candidate_features[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.candidate_mask[start:end], dtype=torch.bool, device=device),
                    torch.as_tensor(subset.labels[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.weights[start:end], dtype=torch.float32, device=device),
                    torch.as_tensor(subset.teacher_scores[start:end], dtype=torch.float32, device=device),
                    config,
                )
                total += float(loss.detach().cpu()) * int(end - start)
                count += int(end - start)
        return float(total / max(count, 1))

    initial_loss = loss_for(train_samples)
    for _step in range(int(config.steps)):
        count = min(int(config.batch_size), train_samples.sample_count)
        batch = rng.choice(train_samples.sample_count, size=count, replace=False)
        loss = _diagonal_descriptor_loss(
            model,
            torch.as_tensor(train_samples.query_features[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.candidate_features[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.candidate_mask[batch], dtype=torch.bool, device=device),
            torch.as_tensor(train_samples.labels[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.weights[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.teacher_scores[batch], dtype=torch.float32, device=device),
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = loss_for(train_samples)
    scales = model.channel_scales().detach().cpu().numpy().astype(np.float64)
    summary = DiagonalDescriptorSelectionSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        raw_train_top1_acc=_top1_acc(None, train_samples, device),
        raw_eval_top1_acc=_top1_acc(None, eval_samples, device),
        train_top1_acc=_top1_acc(model, train_samples, device),  # type: ignore[arg-type]
        eval_top1_acc=_top1_acc(model, eval_samples, device),  # type: ignore[arg-type]
        sample_count=int(samples.sample_count),
        train_sample_count=int(train_samples.sample_count),
        eval_sample_count=int(eval_samples.sample_count),
        input_dim=int(samples.feature_dim),
        output_dim=int(samples.feature_dim),
        candidate_count=int(samples.candidate_count),
        steps=int(config.steps),
        batch_size=int(config.batch_size),
        temperature=float(config.temperature),
        margin_loss_weight=float(config.margin_loss_weight),
        distill_loss_weight=float(config.distill_loss_weight),
        anchor_loss_weight=float(config.anchor_loss_weight),
        scale_regularization_weight=float(config.scale_regularization_weight),
        max_log_scale=float(config.max_log_scale),
        channel_scale_min=float(np.min(scales)),
        channel_scale_max=float(np.max(scales)),
        channel_scale_mean=float(np.mean(scales)),
        channel_scale_std=float(np.std(scales)),
        effective_channel_count=_effective_channel_count(scales),
    )
    return DiagonalDescriptorSelectionRun(model=model.cpu(), summary=summary)


def encode_rows_with_confidence_refiner(
    model: ConfidenceDescriptorRefiner,
    rows: np.ndarray,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    torch_device = torch.device(device)
    was_training = model.training
    model = model.to(torch_device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, values.shape[0], int(batch_size)):
            batch = torch.as_tensor(values[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            chunks.append(model(batch).detach().cpu().numpy().astype(np.float32))
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.output_dim), dtype=np.float32)


def encode_rows_with_diagonal_selector(
    model: DiagonalDescriptorSelector,
    rows: np.ndarray,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    torch_device = torch.device(device)
    was_training = model.training
    model = model.to(torch_device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, values.shape[0], int(batch_size)):
            batch = torch.as_tensor(values[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            chunks.append(model(batch).detach().cpu().numpy().astype(np.float32))
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.output_dim), dtype=np.float32)


def save_diagonal_descriptor_selector_checkpoint(run: DiagonalDescriptorSelectionRun, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "vfm_stage_c216_diagonal_descriptor_selector_v1",
            "model_config": {
                "feature_dim": int(run.model.input_dim),
                "max_log_scale": float(run.model.max_log_scale),
            },
            "state_dict": run.model.state_dict(),
            "summary": asdict(run.summary),
        },
        output,
    )


def load_diagonal_descriptor_selector_checkpoint(path: str | Path, device: str = "cpu") -> DiagonalDescriptorSelectionRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "vfm_stage_c216_diagonal_descriptor_selector_v1":
        raise ValueError("unsupported diagonal descriptor selector checkpoint format")
    cfg = payload["model_config"]
    model = DiagonalDescriptorSelector(int(cfg["feature_dim"]), max_log_scale=float(cfg["max_log_scale"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    summary = DiagonalDescriptorSelectionSummary(**payload["summary"])
    return DiagonalDescriptorSelectionRun(model=model, summary=summary)


def save_confidence_refiner_checkpoint(run: ConfidenceDescriptorRefinementRun, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "vfm_stage_c29_confidence_descriptor_refiner_v1",
            "model_config": {
                "input_dim": int(run.model.input_dim),
                "output_dim": int(run.model.output_dim),
                "hidden_dim": int(run.model.hidden_dim),
            },
            "state_dict": run.model.state_dict(),
            "summary": asdict(run.summary),
        },
        output,
    )


def load_confidence_refiner_checkpoint(path: str | Path, device: str = "cpu") -> ConfidenceDescriptorRefinementRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "vfm_stage_c29_confidence_descriptor_refiner_v1":
        raise ValueError("unsupported confidence descriptor refiner checkpoint format")
    cfg = payload["model_config"]
    model = ConfidenceDescriptorRefiner(int(cfg["input_dim"]), int(cfg["output_dim"]), int(cfg["hidden_dim"]))
    model.load_state_dict(payload["state_dict"], strict=False)
    summary = ConfidenceDescriptorRefinementSummary(**payload["summary"])
    return ConfidenceDescriptorRefinementRun(model=model, summary=summary)
