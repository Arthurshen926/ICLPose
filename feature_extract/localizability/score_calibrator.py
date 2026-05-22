"""Lightweight hypothesis-score calibrator utilities."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from feature_extract.localizability.bank_schema import validate_no_forbidden_training_inputs


@dataclass
class CandidateScoreTable:
    sample_names: list[str]
    feature_keys: tuple[str, ...]
    features: torch.Tensor
    pose_cost_m: torch.Tensor
    trans_err_m: torch.Tensor
    rot_err_deg: torch.Tensor
    valid_mask: torch.Tensor
    basin_label: torch.Tensor


class HypothesisScoreCalibrator(nn.Module):
    """Small candidate-wise scorer for cached POFD-FS hypothesis evidence."""

    def __init__(
        self,
        *,
        feature_dim: int,
        model_type: str = "linear",
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        score_residual_weight: float = 0.0,
        score_feature_index: int = 0,
    ) -> None:
        super().__init__()
        self.model_type = str(model_type)
        self.score_residual_weight = float(score_residual_weight)
        self.score_feature_index = int(score_feature_index)
        dim = int(feature_dim)
        if dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.model_type == "linear":
            self.net = nn.Linear(dim, 1)
        elif self.model_type == "mlp":
            depth = max(1, int(num_layers))
            hidden = max(1, int(hidden_dim))
            layers: list[nn.Module] = []
            in_dim = dim
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.GELU())
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported score calibrator model_type: {model_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (B,K,F)")
        logits = self.net(features.float()).squeeze(-1)
        if self.score_residual_weight != 0.0:
            idx = max(0, min(int(self.score_feature_index), features.shape[-1] - 1))
            logits = logits + float(self.score_residual_weight) * features[..., idx].float()
        return logits


def load_candidate_table_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def group_candidate_table_rows(
    rows: Iterable[dict],
    *,
    feature_keys: Sequence[str],
) -> CandidateScoreTable:
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        sample_name = str(row["sample_name"])
        grouped.setdefault(sample_name, []).append(row)
    if not grouped:
        raise ValueError("candidate table is empty")

    sample_names = list(grouped)
    num_candidates = max(int(row["candidate_idx"]) for items in grouped.values() for row in items) + 1
    keys = tuple(str(key) for key in feature_keys)
    if not keys:
        raise ValueError("feature_keys must not be empty")
    validate_no_forbidden_training_inputs(keys)

    bsz = len(sample_names)
    features = torch.zeros(bsz, num_candidates, len(keys), dtype=torch.float32)
    pose_cost = torch.full((bsz, num_candidates), float("inf"), dtype=torch.float32)
    trans_err = torch.zeros(bsz, num_candidates, dtype=torch.float32)
    rot_err = torch.zeros(bsz, num_candidates, dtype=torch.float32)
    valid = torch.zeros(bsz, num_candidates, dtype=torch.bool)
    basin = torch.zeros(bsz, num_candidates, dtype=torch.bool)

    for batch_idx, sample_name in enumerate(sample_names):
        sample_rows = sorted(grouped[sample_name], key=lambda item: int(item["candidate_idx"]))
        sample_scores = torch.zeros(num_candidates, dtype=torch.float32)
        sample_score_ranks = torch.full((num_candidates,), float(num_candidates - 1), dtype=torch.float32)
        sample_valid = torch.zeros(num_candidates, dtype=torch.bool)
        for row in sample_rows:
            candidate_idx = int(row["candidate_idx"])
            sample_scores[candidate_idx] = float(row.get("score", 0.0))
            sample_score_ranks[candidate_idx] = float(row.get("score_rank", candidate_idx))
            sample_valid[candidate_idx] = bool(row.get("valid", True))
        if sample_valid.any():
            valid_scores = sample_scores[sample_valid]
            top_score = valid_scores.max()
            score_mean = valid_scores.mean()
            score_std = valid_scores.std(unbiased=False).clamp_min(1.0e-6)
        else:
            top_score = sample_scores.max()
            score_mean = sample_scores.mean()
            score_std = sample_scores.std(unbiased=False).clamp_min(1.0e-6)

        for row in sample_rows:
            candidate_idx = int(row["candidate_idx"])
            for feature_idx, key in enumerate(keys):
                if key == "score_margin_to_top1":
                    value = float(sample_scores[candidate_idx] - top_score) if bool(sample_valid[candidate_idx]) else 0.0
                elif key == "score_rank_norm":
                    denom = max(float(num_candidates - 1), 1.0)
                    value = float(sample_score_ranks[candidate_idx] / denom)
                elif key == "score_zscore":
                    value = float((sample_scores[candidate_idx] - score_mean) / score_std) if bool(sample_valid[candidate_idx]) else 0.0
                else:
                    value = float(row.get(key, 0.0))
                features[batch_idx, candidate_idx, feature_idx] = value
            pose_cost[batch_idx, candidate_idx] = float(row["pose_cost_m"])
            trans_err[batch_idx, candidate_idx] = float(row.get("trans_err_m", row["pose_cost_m"]))
            rot_err[batch_idx, candidate_idx] = float(row.get("rot_err_deg", 0.0))
            valid[batch_idx, candidate_idx] = bool(row.get("valid", True))
            basin[batch_idx, candidate_idx] = bool(row.get("in_basin", False))

    return CandidateScoreTable(
        sample_names=sample_names,
        feature_keys=keys,
        features=features,
        pose_cost_m=pose_cost,
        trans_err_m=trans_err,
        rot_err_deg=rot_err,
        valid_mask=valid,
        basin_label=basin,
    )


def normalize_features(
    train: torch.Tensor,
    val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = train.reshape(-1, train.shape[-1]).float()
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1.0e-6)
    return (train.float() - mean) / std, (val.float() - mean) / std, mean, std
