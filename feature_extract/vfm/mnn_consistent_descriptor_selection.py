"""MNN-consistent descriptor selection for Stage C2.10."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.confidence_descriptor_refinement import ConfidenceDescriptorRefiner
from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence, vectorize_match_rows
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.tokens import TokenBankManifest


@dataclass(frozen=True)
class MNNConsistentDescriptorTrainingSet:
    query_features: np.ndarray
    landmark_features: np.ndarray
    valid_mask: np.ndarray
    hard_labels: np.ndarray
    soft_labels: np.ndarray
    pair_weights: np.ndarray
    teacher_scores: np.ndarray
    wrong_pose_negative_weights: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        query = np.asarray(self.query_features, dtype=np.float32)
        landmarks = np.asarray(self.landmark_features, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        hard = np.asarray(self.hard_labels, dtype=np.float32)
        soft = np.asarray(self.soft_labels, dtype=np.float32)
        weights = np.asarray(self.pair_weights, dtype=np.float32)
        teacher = np.asarray(self.teacher_scores, dtype=np.float32)
        wrong = np.asarray(self.wrong_pose_negative_weights, dtype=np.float32)
        if query.ndim != 3:
            raise ValueError("query_features must have shape (G, T, C)")
        if landmarks.ndim != 3:
            raise ValueError("landmark_features must have shape (G, L, C)")
        if query.shape[0] != landmarks.shape[0] or query.shape[2] != landmarks.shape[2]:
            raise ValueError("query and landmark feature groups must share group count and feature dim")
        expected = (query.shape[0], query.shape[1], landmarks.shape[1])
        for name, value in (
            ("valid_mask", valid),
            ("hard_labels", hard),
            ("soft_labels", soft),
            ("pair_weights", weights),
            ("teacher_scores", teacher),
            ("wrong_pose_negative_weights", wrong),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        has_positive = np.any(valid & (hard > 0.5), axis=(1, 2))
        if query.shape[0] and not np.all(has_positive):
            raise ValueError("each group must contain at least one hard positive pair")
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "landmark_features", landmarks)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "hard_labels", np.clip(hard, 0.0, 1.0))
        object.__setattr__(self, "soft_labels", np.clip(soft, 0.0, 1.0))
        object.__setattr__(self, "pair_weights", np.clip(weights, 0.0, None))
        object.__setattr__(self, "teacher_scores", np.clip(teacher, 1e-4, 1.0 - 1e-4))
        object.__setattr__(self, "wrong_pose_negative_weights", np.clip(wrong, 0.0, None))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def group_count(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def max_tokens_per_group(self) -> int:
        return int(self.query_features.shape[1])

    @property
    def max_landmarks_per_group(self) -> int:
        return int(self.landmark_features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.query_features.shape[2]) if self.query_features.ndim == 3 else 0


@dataclass(frozen=True)
class MNNConsistentDescriptorSelectionConfig:
    output_dim: int = 128
    hidden_dim: int = 256
    steps: int = 800
    batch_size: int = 16
    lr: float = 1e-4
    temperature: float = 0.07
    dual_loss_weight: float = 1.0
    soft_reproj_loss_weight: float = 0.5
    wrong_pose_loss_weight: float = 0.1
    teacher_distill_loss_weight: float = 0.2
    anchor_loss_weight: float = 0.1
    margin: float = 0.2
    teacher_temperature: float = 0.2
    eval_split_fraction: float = 0.1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.output_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive")
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.lr <= 0.0 or self.temperature <= 0.0 or self.teacher_temperature <= 0.0:
            raise ValueError("lr and temperatures must be positive")
        for name, value in (
            ("dual_loss_weight", self.dual_loss_weight),
            ("soft_reproj_loss_weight", self.soft_reproj_loss_weight),
            ("wrong_pose_loss_weight", self.wrong_pose_loss_weight),
            ("teacher_distill_loss_weight", self.teacher_distill_loss_weight),
            ("anchor_loss_weight", self.anchor_loss_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")


@dataclass(frozen=True)
class MNNConsistentDescriptorSelectionSummary:
    initial_loss: float
    final_loss: float
    raw_train_dual_top1_acc: float
    raw_eval_dual_top1_acc: float
    train_dual_top1_acc: float
    eval_dual_top1_acc: float
    sample_group_count: int
    train_group_count: int
    eval_group_count: int
    input_dim: int
    output_dim: int
    max_tokens_per_group: int
    max_landmarks_per_group: int
    steps: int
    batch_size: int
    temperature: float
    dual_loss_weight: float
    soft_reproj_loss_weight: float
    wrong_pose_loss_weight: float
    teacher_distill_loss_weight: float
    anchor_loss_weight: float
    wrong_pose_negative_count: int


@dataclass(frozen=True)
class MNNConsistentDescriptorSelectionRun:
    model: ConfidenceDescriptorRefiner
    summary: MNNConsistentDescriptorSelectionSummary

    def encode_rows(self, rows: np.ndarray, device: str = "cpu", batch_size: int = 65536) -> np.ndarray:
        return encode_rows_with_mnn_consistent_selector(self.model, rows, device=device, batch_size=batch_size)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _query_feature_for_token(feature_map: np.ndarray, token_index: int) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = values.shape
    y_idx, x_idx = divmod(int(token_index), int(width))
    if y_idx < 0 or y_idx >= height:
        raise ValueError(f"token_index {token_index} is outside feature map")
    return values[:, y_idx, x_idx].reshape(channels).astype(np.float32, copy=True)


def _is_ignored(row: Mapping[str, Any]) -> bool:
    return bool(row.get("ignore_label", False))


def _gt_reproj_stride(row: Mapping[str, Any]) -> float:
    value = row.get("gt_reproj_error_stride", row.get("gt_error_stride", np.inf))
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = np.inf
    return result if np.isfinite(result) else np.inf


def _is_hard_positive(row: Mapping[str, Any], positive_stride: float) -> bool:
    return (
        bool(row.get("strong_positive_label", False))
        or bool(row.get("patch_positive_label", False))
        or _gt_reproj_stride(row) <= float(positive_stride)
    )


def _soft_label(row: Mapping[str, Any], sigma_stride: float) -> float:
    distance = _gt_reproj_stride(row)
    if not np.isfinite(distance):
        return 0.0
    sigma = max(float(sigma_stride), 1e-6)
    return float(np.exp(-0.5 * (distance / sigma) ** 2))


def _is_wrong_pose_negative(row: Mapping[str, Any], negative_stride: float) -> bool:
    inlier = (
        bool(row.get("baseline_ransac_inlier", False))
        or bool(row.get("baseline_inlier", False))
        or bool(row.get("ransac_inlier", False))
        or bool(row.get("pnp_inlier", False))
    )
    return bool(inlier and _gt_reproj_stride(row) > float(negative_stride))


def _row_priority(row: Mapping[str, Any], positive_stride: float, negative_stride: float) -> tuple[int, float, float]:
    if _is_ignored(row):
        return 5, 0.0, np.inf
    if _is_hard_positive(row, positive_stride):
        bucket = 0
    elif _is_wrong_pose_negative(row, negative_stride):
        bucket = 1
    elif bool(row.get("hard_negative_label", False)):
        bucket = 2
    else:
        bucket = 3
    score = float(row.get("teacher_score", row.get("confidence", row.get("similarity", 0.0))) or 0.0)
    return bucket, -score, _gt_reproj_stride(row)


def build_mnn_consistent_descriptor_samples(
    match_jsonl: str | Path,
    query_manifest: str | Path,
    landmark_bank: str | Path,
    layer_name: str = "radio_final",
    max_tokens_per_query: int = 64,
    max_landmarks_per_query: int = 256,
    max_candidates_per_token: int = 16,
    max_query_groups: int = 0,
    positive_stride: float = 1.0,
    negative_stride: float = 2.0,
    soft_label_sigma_stride: float = 1.0,
    teacher_model: CalibratedLogisticConfidence | None = None,
    teacher_feature_set: str = "full",
    seed: int = 0,
) -> tuple[MNNConsistentDescriptorTrainingSet, dict[str, object]]:
    if int(max_tokens_per_query) <= 0:
        raise ValueError("max_tokens_per_query must be positive")
    if int(max_landmarks_per_query) <= 1:
        raise ValueError("max_landmarks_per_query must be greater than 1")
    if int(max_candidates_per_token) <= 1:
        raise ValueError("max_candidates_per_token must be greater than 1")
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
    track_features = np.stack([np.asarray(bank.tracks[int(track_id)].mean_feature, dtype=np.float32) for track_id in track_ids], axis=0)

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if query_id not in record_by_query or _is_ignored(row):
            continue
        track_id = int(row.get("track_id", -1))
        if track_id not in track_row:
            continue
        grouped[query_id][int(row.get("token_index", 0))].append(row)

    rng = np.random.default_rng(int(seed))
    query_ids = sorted(grouped)
    if max_query_groups > 0 and len(query_ids) > int(max_query_groups):
        query_ids = [query_ids[int(idx)] for idx in sorted(rng.choice(len(query_ids), size=int(max_query_groups), replace=False).tolist())]

    query_cache: dict[str, np.ndarray] = {}
    group_queries: list[np.ndarray] = []
    group_landmarks: list[np.ndarray] = []
    group_valid: list[np.ndarray] = []
    group_hard: list[np.ndarray] = []
    group_soft: list[np.ndarray] = []
    group_weights: list[np.ndarray] = []
    group_teacher: list[np.ndarray] = []
    group_wrong: list[np.ndarray] = []
    wrong_pose_count = 0
    skipped_without_positive = 0
    selected_pair_count = 0
    for query_id in query_ids:
        token_rows: dict[int, list[dict[str, Any]]] = {}
        positive_tokens = []
        for token_index, token_items in grouped[query_id].items():
            ordered = [dict(row) for row in sorted(token_items, key=lambda item: _row_priority(item, positive_stride, negative_stride))]
            selected = ordered[: int(max_candidates_per_token)]
            if any(_is_hard_positive(row, positive_stride) for row in selected):
                positive_tokens.append(int(token_index))
                token_rows[int(token_index)] = selected
        if not positive_tokens:
            skipped_without_positive += 1
            continue
        positive_tokens = sorted(positive_tokens)
        if len(positive_tokens) > int(max_tokens_per_query):
            positive_tokens = [positive_tokens[int(idx)] for idx in sorted(rng.choice(len(positive_tokens), size=int(max_tokens_per_query), replace=False).tolist())]

        landmark_order: list[int] = []
        landmark_set: set[int] = set()
        # Keep positives first so every retained token can still contribute a target.
        for pass_positive in (True, False):
            for token_index in positive_tokens:
                for row in token_rows[token_index]:
                    if _is_hard_positive(row, positive_stride) != pass_positive:
                        continue
                    track_id = int(row["track_id"])
                    if track_id not in landmark_set:
                        landmark_set.add(track_id)
                        landmark_order.append(track_id)
                    if len(landmark_order) >= int(max_landmarks_per_query):
                        break
                if len(landmark_order) >= int(max_landmarks_per_query):
                    break
            if len(landmark_order) >= int(max_landmarks_per_query):
                break
        if len(landmark_order) <= 1:
            skipped_without_positive += 1
            continue
        landmark_col = {track_id: idx for idx, track_id in enumerate(landmark_order)}
        valid = np.zeros((len(positive_tokens), len(landmark_order)), dtype=bool)
        hard = np.zeros_like(valid, dtype=np.float32)
        soft = np.zeros_like(valid, dtype=np.float32)
        weights = np.zeros_like(valid, dtype=np.float32)
        teacher = np.full_like(valid, 1e-4, dtype=np.float32)
        wrong = np.zeros_like(valid, dtype=np.float32)
        for token_idx, token_index in enumerate(positive_tokens):
            for row in token_rows[token_index]:
                track_id = int(row["track_id"])
                if track_id not in landmark_col:
                    continue
                col = landmark_col[track_id]
                valid[token_idx, col] = True
                hard[token_idx, col] = 1.0 if _is_hard_positive(row, positive_stride) else 0.0
                soft_value = _soft_label(row, soft_label_sigma_stride)
                soft[token_idx, col] = 1.0 if hard[token_idx, col] > 0.5 and soft_value <= 0.0 else soft_value
                weights[token_idx, col] = 1.0
                teacher[token_idx, col] = float(row.get("teacher_score", 0.9 if hard[token_idx, col] > 0.5 else 0.1))
                if _is_wrong_pose_negative(row, negative_stride):
                    wrong[token_idx, col] = 1.0
                    wrong_pose_count += 1
                selected_pair_count += 1
        if not bool(np.any(valid & (hard > 0.5))):
            skipped_without_positive += 1
            continue
        if query_id not in query_cache:
            with np.load(record_by_query[query_id].token_path) as data:
                query_cache[query_id] = np.asarray(data[layer_name], dtype=np.float32)
        query_matrix = np.stack([_query_feature_for_token(query_cache[query_id], token_index) for token_index in positive_tokens], axis=0)
        landmark_matrix = np.stack([track_features[track_row[track_id]] for track_id in landmark_order], axis=0)
        group_queries.append(query_matrix)
        group_landmarks.append(landmark_matrix)
        group_valid.append(valid)
        group_hard.append(hard)
        group_soft.append(soft)
        group_weights.append(weights)
        group_teacher.append(teacher)
        group_wrong.append(wrong)

    max_tokens = max((item.shape[0] for item in group_queries), default=0)
    max_landmarks = max((item.shape[0] for item in group_landmarks), default=0)
    feature_dim = int(bank.feature_dim)

    def pad_3d(groups: list[np.ndarray], first: int, second: int, fill: float = 0.0) -> np.ndarray:
        out = np.full((len(groups), first, second), fill, dtype=np.float32)
        for idx, item in enumerate(groups):
            out[idx, : item.shape[0], : item.shape[1]] = item
        return out

    query_out = np.zeros((len(group_queries), max_tokens, feature_dim), dtype=np.float32)
    landmark_out = np.zeros((len(group_queries), max_landmarks, feature_dim), dtype=np.float32)
    for idx, item in enumerate(group_queries):
        query_out[idx, : item.shape[0], :] = item
    for idx, item in enumerate(group_landmarks):
        landmark_out[idx, : item.shape[0], :] = item
    valid_out = np.zeros((len(group_queries), max_tokens, max_landmarks), dtype=bool)
    for idx, item in enumerate(group_valid):
        valid_out[idx, : item.shape[0], : item.shape[1]] = item
    samples = MNNConsistentDescriptorTrainingSet(
        query_features=query_out,
        landmark_features=landmark_out,
        valid_mask=valid_out,
        hard_labels=pad_3d(group_hard, max_tokens, max_landmarks),
        soft_labels=pad_3d(group_soft, max_tokens, max_landmarks),
        pair_weights=pad_3d(group_weights, max_tokens, max_landmarks),
        teacher_scores=pad_3d(group_teacher, max_tokens, max_landmarks, fill=1e-4),
        wrong_pose_negative_weights=pad_3d(group_wrong, max_tokens, max_landmarks),
        metadata={
            "source_match_jsonl": str(match_jsonl),
            "max_tokens_per_query": int(max_tokens_per_query),
            "max_landmarks_per_query": int(max_landmarks_per_query),
            "max_candidates_per_token": int(max_candidates_per_token),
            "positive_stride": float(positive_stride),
            "negative_stride": float(negative_stride),
            "soft_label_sigma_stride": float(soft_label_sigma_stride),
        },
    )
    meta = {
        "group_count": int(samples.group_count),
        "max_tokens_per_group": int(samples.max_tokens_per_group),
        "max_landmarks_per_group": int(samples.max_landmarks_per_group),
        "selected_pair_count": int(selected_pair_count),
        "wrong_pose_negative_count": int(wrong_pose_count),
        "skipped_without_positive": int(skipped_without_positive),
        "mean_valid_pairs_per_group": float(np.mean(np.sum(samples.valid_mask, axis=(1, 2)))) if samples.group_count else 0.0,
        "mean_soft_label": float(np.mean(samples.soft_labels[samples.valid_mask])) if bool(np.any(samples.valid_mask)) else 0.0,
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


def _subset(samples: MNNConsistentDescriptorTrainingSet, indices: np.ndarray) -> MNNConsistentDescriptorTrainingSet:
    return MNNConsistentDescriptorTrainingSet(
        query_features=samples.query_features[indices],
        landmark_features=samples.landmark_features[indices],
        valid_mask=samples.valid_mask[indices],
        hard_labels=samples.hard_labels[indices],
        soft_labels=samples.soft_labels[indices],
        pair_weights=samples.pair_weights[indices],
        teacher_scores=samples.teacher_scores[indices],
        wrong_pose_negative_weights=samples.wrong_pose_negative_weights[indices],
        metadata=samples.metadata,
    )


def _encode_group(model: ConfidenceDescriptorRefiner, query: torch.Tensor, landmarks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query_z = model(query.reshape(-1, query.shape[-1])).reshape(query.shape[0], query.shape[1], -1)
    landmark_z = model(landmarks.reshape(-1, landmarks.shape[-1])).reshape(landmarks.shape[0], landmarks.shape[1], -1)
    return query_z, landmark_z


def _mnn_consistent_loss(
    model: ConfidenceDescriptorRefiner,
    query: torch.Tensor,
    landmarks: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_labels: torch.Tensor,
    soft_labels: torch.Tensor,
    pair_weights: torch.Tensor,
    teacher_scores: torch.Tensor,
    wrong_pose_negative_weights: torch.Tensor,
    config: MNNConsistentDescriptorSelectionConfig,
) -> torch.Tensor:
    query_z, landmark_z = _encode_group(model, query, landmarks)
    cosine = torch.einsum("gtd,gld->gtl", query_z, landmark_z)
    logits = cosine / float(config.temperature)
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    loss = torch.zeros((), dtype=torch.float32, device=query.device)

    target = torch.clamp(soft_labels * pair_weights, min=0.0).masked_fill(~valid_mask, 0.0)
    has_target = torch.sum(target, dim=(1, 2)) > 0.0
    if float(config.dual_loss_weight) > 0.0 and bool(has_target.any().item()):
        row_log = F.log_softmax(masked_logits, dim=2)
        col_log = F.log_softmax(masked_logits, dim=1)
        dual_log = row_log + col_log
        log_target = torch.log(torch.clamp(target, min=1e-8))
        numerator = torch.logsumexp((dual_log + log_target).masked_fill(target <= 0.0, -1e9), dim=(1, 2))
        normalizer = torch.logsumexp(log_target.masked_fill(target <= 0.0, -1e9), dim=(1, 2))
        dual_loss = -(numerator[has_target] - normalizer[has_target]).mean()
        loss = loss + float(config.dual_loss_weight) * dual_loss

    row_has_target = torch.sum(target, dim=2) > 0.0
    if float(config.soft_reproj_loss_weight) > 0.0 and bool(row_has_target.any().item()):
        row_log = F.log_softmax(masked_logits, dim=2)
        target_prob = target / torch.clamp(torch.sum(target, dim=2, keepdim=True), min=1e-8)
        soft_ce = -torch.sum(target_prob * row_log, dim=2)
        loss = loss + float(config.soft_reproj_loss_weight) * soft_ce[row_has_target].mean()

    wrong = (wrong_pose_negative_weights > 0.0) & valid_mask
    positives = (hard_labels > 0.5) & valid_mask
    if float(config.wrong_pose_loss_weight) > 0.0 and bool(wrong.any().item()):
        margins = []
        for group_idx in range(cosine.shape[0]):
            for token_idx in range(cosine.shape[1]):
                if not bool(positives[group_idx, token_idx].any().item()):
                    continue
                pos_score = cosine[group_idx, token_idx][positives[group_idx, token_idx]].max()
                neg_scores = cosine[group_idx, token_idx][wrong[group_idx, token_idx]]
                if neg_scores.numel():
                    weights = wrong_pose_negative_weights[group_idx, token_idx][wrong[group_idx, token_idx]]
                    margins.append(weights * F.relu(neg_scores - pos_score + float(config.margin)))
        if margins:
            loss = loss + float(config.wrong_pose_loss_weight) * torch.cat([item.reshape(-1) for item in margins]).mean()

    if float(config.teacher_distill_loss_weight) > 0.0:
        row_valid = torch.any(valid_mask, dim=2)
        if bool(row_valid.any().item()):
            teacher_logits = torch.logit(torch.clamp(teacher_scores, 1e-4, 1.0 - 1e-4))
            teacher_logits = teacher_logits.masked_fill(~valid_mask, -1e9)
            teacher_prob = F.softmax(teacher_logits / float(config.teacher_temperature), dim=2)
            student_log = F.log_softmax(masked_logits, dim=2)
            kl = torch.sum(teacher_prob * (torch.log(torch.clamp(teacher_prob, min=1e-8)) - student_log), dim=2)
            loss = loss + float(config.teacher_distill_loss_weight) * kl[row_valid].mean()

    if float(config.anchor_loss_weight) > 0.0 and model.input_dim == model.output_dim:
        anchor_q = F.normalize(query, dim=-1, eps=1e-8)
        anchor_l = F.normalize(landmarks, dim=-1, eps=1e-8)
        anchor_loss = 1.0 - torch.sum(query_z * anchor_q, dim=-1)
        landmark_anchor_loss = 1.0 - torch.sum(landmark_z * anchor_l, dim=-1)
        query_used = torch.any(valid_mask, dim=2)
        landmark_used = torch.any(valid_mask, dim=1)
        loss = loss + float(config.anchor_loss_weight) * (
            anchor_loss[query_used].mean() + landmark_anchor_loss[landmark_used].mean()
        )
    return loss


def _dual_top1_acc(model: ConfidenceDescriptorRefiner | None, samples: MNNConsistentDescriptorTrainingSet, device: torch.device) -> float:
    if samples.group_count == 0:
        return 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, samples.group_count, 256):
            end = min(start + 256, samples.group_count)
            query = torch.as_tensor(samples.query_features[start:end], dtype=torch.float32, device=device)
            landmarks = torch.as_tensor(samples.landmark_features[start:end], dtype=torch.float32, device=device)
            valid = torch.as_tensor(samples.valid_mask[start:end], dtype=torch.bool, device=device)
            labels = torch.as_tensor(samples.hard_labels[start:end], dtype=torch.float32, device=device)
            if model is None:
                qz = F.normalize(query, dim=-1, eps=1e-8)
                lz = F.normalize(landmarks, dim=-1, eps=1e-8)
            else:
                qz, lz = _encode_group(model, query, landmarks)
            scores = torch.einsum("gtd,gld->gtl", qz, lz).masked_fill(~valid, -1e9)
            row_prob = F.softmax(scores, dim=2)
            col_prob = F.softmax(scores, dim=1)
            dual = (row_prob * col_prob).masked_fill(~valid, -1.0)
            top = torch.argmax(dual, dim=2)
            rows = torch.any(valid & (labels > 0.5), dim=2)
            if bool(rows.any().item()):
                row_indices = torch.arange(labels.shape[1], device=device).view(1, -1).expand(labels.shape[0], -1)
                group_indices = torch.arange(labels.shape[0], device=device).view(-1, 1).expand(-1, labels.shape[1])
                chosen = labels[group_indices, row_indices, top] > 0.5
                correct += int(chosen[rows].sum().item())
                total += int(rows.sum().item())
    return float(correct / max(total, 1))


def _loss_for_subset(
    model: ConfidenceDescriptorRefiner,
    subset: MNNConsistentDescriptorTrainingSet,
    config: MNNConsistentDescriptorSelectionConfig,
    device: torch.device,
) -> float:
    if subset.group_count == 0:
        return 0.0
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, subset.group_count, min(int(config.batch_size), 64)):
            end = min(start + min(int(config.batch_size), 64), subset.group_count)
            loss = _mnn_consistent_loss(
                model,
                torch.as_tensor(subset.query_features[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(subset.landmark_features[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(subset.valid_mask[start:end], dtype=torch.bool, device=device),
                torch.as_tensor(subset.hard_labels[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(subset.soft_labels[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(subset.pair_weights[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(subset.teacher_scores[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(subset.wrong_pose_negative_weights[start:end], dtype=torch.float32, device=device),
                config,
            )
            total += float(loss.detach().cpu()) * int(end - start)
            count += int(end - start)
    return float(total / max(count, 1))


def train_mnn_consistent_descriptor_selector(
    samples: MNNConsistentDescriptorTrainingSet,
    config: MNNConsistentDescriptorSelectionConfig | None = None,
) -> MNNConsistentDescriptorSelectionRun:
    config = config or MNNConsistentDescriptorSelectionConfig(output_dim=samples.feature_dim)
    if samples.group_count == 0:
        raise ValueError("at least one training group is required")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.group_count, float(config.eval_split_fraction), int(config.seed))
    train_samples = _subset(samples, train_idx)
    eval_samples = _subset(samples, eval_idx) if eval_idx.size else _subset(samples, train_idx[:0])
    model = ConfidenceDescriptorRefiner(samples.feature_dim, int(config.output_dim), int(config.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))
    initial_loss = _loss_for_subset(model, train_samples, config, device)
    for _step in range(int(config.steps)):
        count = min(int(config.batch_size), train_samples.group_count)
        batch = rng.choice(train_samples.group_count, size=count, replace=False)
        loss = _mnn_consistent_loss(
            model,
            torch.as_tensor(train_samples.query_features[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.landmark_features[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.valid_mask[batch], dtype=torch.bool, device=device),
            torch.as_tensor(train_samples.hard_labels[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.soft_labels[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.pair_weights[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.teacher_scores[batch], dtype=torch.float32, device=device),
            torch.as_tensor(train_samples.wrong_pose_negative_weights[batch], dtype=torch.float32, device=device),
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = _loss_for_subset(model, train_samples, config, device)
    summary = MNNConsistentDescriptorSelectionSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        raw_train_dual_top1_acc=_dual_top1_acc(None, train_samples, device),
        raw_eval_dual_top1_acc=_dual_top1_acc(None, eval_samples, device),
        train_dual_top1_acc=_dual_top1_acc(model, train_samples, device),
        eval_dual_top1_acc=_dual_top1_acc(model, eval_samples, device),
        sample_group_count=int(samples.group_count),
        train_group_count=int(train_samples.group_count),
        eval_group_count=int(eval_samples.group_count),
        input_dim=int(samples.feature_dim),
        output_dim=int(config.output_dim),
        max_tokens_per_group=int(samples.max_tokens_per_group),
        max_landmarks_per_group=int(samples.max_landmarks_per_group),
        steps=int(config.steps),
        batch_size=int(config.batch_size),
        temperature=float(config.temperature),
        dual_loss_weight=float(config.dual_loss_weight),
        soft_reproj_loss_weight=float(config.soft_reproj_loss_weight),
        wrong_pose_loss_weight=float(config.wrong_pose_loss_weight),
        teacher_distill_loss_weight=float(config.teacher_distill_loss_weight),
        anchor_loss_weight=float(config.anchor_loss_weight),
        wrong_pose_negative_count=int(np.sum(samples.wrong_pose_negative_weights > 0.0)),
    )
    return MNNConsistentDescriptorSelectionRun(model=model.cpu(), summary=summary)


def encode_rows_with_mnn_consistent_selector(
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


def save_mnn_consistent_selector_checkpoint(run: MNNConsistentDescriptorSelectionRun, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "vfm_stage_c210_mnn_consistent_descriptor_selector_v1",
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


def load_mnn_consistent_selector_checkpoint(path: str | Path, device: str = "cpu") -> MNNConsistentDescriptorSelectionRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "vfm_stage_c210_mnn_consistent_descriptor_selector_v1":
        raise ValueError("unsupported C2.10 selector checkpoint format")
    cfg = payload["model_config"]
    model = ConfidenceDescriptorRefiner(int(cfg["input_dim"]), int(cfg["output_dim"]), int(cfg["hidden_dim"]))
    model.load_state_dict(payload["state_dict"], strict=False)
    summary = MNNConsistentDescriptorSelectionSummary(**payload["summary"])
    return MNNConsistentDescriptorSelectionRun(model=model, summary=summary)
