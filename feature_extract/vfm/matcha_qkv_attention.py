"""Residual QKV cross-attention for rendered/query descriptor maps."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_FORMAT = "vfm_matcha_qkv_cross_attention_v1"


class ResidualQKVCrossAttention(nn.Module):
    """Small residual bidirectional QKV attention over query/render descriptors."""

    def __init__(
        self,
        input_dim: int,
        attention_dim: int | None = None,
        *,
        alpha: float = 0.25,
        logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.attention_dim = int(attention_dim or input_dim)
        self.alpha = float(alpha)
        self.logit_scale = float(logit_scale)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.logit_scale <= 0.0:
            raise ValueError("logit_scale must be positive")
        self.query_proj = nn.Linear(self.input_dim, self.attention_dim, bias=False)
        self.render_proj = nn.Linear(self.input_dim, self.attention_dim, bias=False)
        self.query_value = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.render_value = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.query_proj.weight.shape[0] == self.query_proj.weight.shape[1]:
            nn.init.eye_(self.query_proj.weight)
        else:
            nn.init.xavier_uniform_(self.query_proj.weight)
        if self.render_proj.weight.shape[0] == self.render_proj.weight.shape[1]:
            nn.init.eye_(self.render_proj.weight)
        else:
            nn.init.xavier_uniform_(self.render_proj.weight)
        nn.init.eye_(self.query_value.weight)
        nn.init.eye_(self.render_value.weight)

    def forward_rows(self, query_rows: torch.Tensor, render_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhance descriptor rows with bidirectional cross-attention."""

        if query_rows.ndim != 2 or render_rows.ndim != 2:
            raise ValueError("query_rows and render_rows must have shape (N, C)")
        if query_rows.shape[1] != self.input_dim or render_rows.shape[1] != self.input_dim:
            raise ValueError("row feature dimension does not match model input_dim")
        query = F.normalize(query_rows, dim=1)
        render = F.normalize(render_rows, dim=1)
        q = F.normalize(self.query_proj(query), dim=1)
        k = F.normalize(self.render_proj(render), dim=1)
        logits = (q @ k.T) * float(self.logit_scale)
        q_to_r = torch.softmax(logits, dim=1)
        r_to_q = torch.softmax(logits.T, dim=1)
        q_context = q_to_r @ self.render_value(render)
        r_context = r_to_q @ self.query_value(query)
        query_out = F.normalize((1.0 - self.alpha) * query + self.alpha * q_context, dim=1)
        render_out = F.normalize((1.0 - self.alpha) * render + self.alpha * r_context, dim=1)
        return query_out, render_out

    def forward(self, query_rows: torch.Tensor, render_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_rows(query_rows, render_rows)


def _flatten_feature_map(feature_map: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    return fmap.reshape(channels, height * width).T.astype(np.float32, copy=False), (channels, height, width)


def apply_qkv_attention_to_feature_maps(
    model: ResidualQKVCrossAttention,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply residual QKV cross-attention to two CHW descriptor maps."""

    query_rows, query_shape = _flatten_feature_map(query_feature_map)
    render_rows, render_shape = _flatten_feature_map(render_feature_map)
    if query_shape[0] != render_shape[0]:
        raise ValueError("query and render descriptor dimensions must match")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    with torch.no_grad():
        query_tensor = torch.as_tensor(query_rows, dtype=torch.float32, device=torch_device)
        render_tensor = torch.as_tensor(render_rows, dtype=torch.float32, device=torch_device)
        query_out, render_out = model(query_tensor, render_tensor)
    if was_training:
        model.train()
    q_channels, qh, qw = query_shape
    r_channels, rh, rw = render_shape
    return (
        query_out.detach().cpu().numpy().T.reshape(q_channels, qh, qw).astype(np.float32, copy=False),
        render_out.detach().cpu().numpy().T.reshape(r_channels, rh, rw).astype(np.float32, copy=False),
    )


def save_qkv_attention_checkpoint(model: ResidualQKVCrossAttention, path: Path, *, summary: dict[str, object] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.save(
        {
            "format": _FORMAT,
            "model_config": {
                "input_dim": int(model.input_dim),
                "attention_dim": int(model.attention_dim),
                "alpha": float(model.alpha),
                "logit_scale": float(model.logit_scale),
            },
            "state_dict": model.state_dict(),
            "summary": dict(summary or {}),
        },
        output,
    )


def load_qkv_attention_checkpoint(path: Path, *, device: str = "cpu") -> tuple[ResidualQKVCrossAttention, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != _FORMAT:
        raise ValueError(f"unsupported QKV attention checkpoint format in {path}")
    cfg = dict(payload["model_config"])
    model = ResidualQKVCrossAttention(
        input_dim=int(cfg["input_dim"]),
        attention_dim=int(cfg["attention_dim"]),
        alpha=float(cfg.get("alpha", 0.25)),
        logit_scale=float(cfg.get("logit_scale", 1.0)),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(torch.device(device)).eval(), dict(payload.get("summary", {}))


@dataclass(frozen=True)
class QKVAttentionTrainingConfig:
    attention_dim: int = 128
    alpha: float = 0.25
    logit_scale: float = 1.0
    steps: int = 300
    batch_size: int = 512
    lr: float = 1e-5
    temperature: float = 0.07
    eval_split_fraction: float = 0.1
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.attention_dim) <= 0:
            raise ValueError("attention_dim must be positive")
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if float(self.logit_scale) <= 0.0:
            raise ValueError("logit_scale must be positive")
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive")
        if int(self.batch_size) <= 1:
            raise ValueError("batch_size must be greater than one")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")


@dataclass
class QKVAttentionTrainingRun:
    model: ResidualQKVCrossAttention
    summary: dict[str, object]


def _validate_training_arrays(
    query_features: np.ndarray,
    render_features: np.ndarray,
    negative_render_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query_features, dtype=np.float32)
    render = np.asarray(render_features, dtype=np.float32)
    negatives = np.asarray(negative_render_features, dtype=np.float32)
    if query.ndim != 2 or render.ndim != 2:
        raise ValueError("query_features and render_features must have shape (N, C)")
    if query.shape != render.shape:
        raise ValueError("query_features and render_features must have the same shape")
    if negatives.ndim != 3 or negatives.shape[0] != query.shape[0] or negatives.shape[2] != query.shape[1]:
        raise ValueError("negative_render_features must have shape (N, K, C)")
    return query, render, negatives


def _split_indices(count: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(count), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    eval_count = int(round(float(eval_fraction) * int(count)))
    eval_count = min(max(eval_count, 0), max(int(count) - 1, 0))
    return indices[eval_count:], indices[:eval_count]


def _attention_loss(
    model: ResidualQKVCrossAttention,
    query: torch.Tensor,
    render: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    batch = int(query.shape[0])
    candidates = torch.cat([render, negatives.reshape(-1, negatives.shape[-1])], dim=0)
    query_z, candidate_z = model(query, candidates)
    logits = query_z @ candidate_z.T / float(temperature)
    labels = torch.arange(batch, device=query.device)
    loss = F.cross_entropy(logits, labels)
    render_z = candidate_z[:batch]
    reverse_logits = render_z @ query_z.T / float(temperature)
    loss = loss + F.cross_entropy(reverse_logits, labels)
    return 0.5 * loss


def _evaluate_attention(
    model: ResidualQKVCrossAttention,
    query_features: np.ndarray,
    render_features: np.ndarray,
    negative_render_features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> dict[str, float]:
    if query_features.shape[0] == 0:
        return {"loss": 0.0, "top1_acc": 0.0}
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, query_features.shape[0], int(batch_size)):
            end = min(start + int(batch_size), query_features.shape[0])
            query = torch.as_tensor(query_features[start:end], dtype=torch.float32, device=device)
            render = torch.as_tensor(render_features[start:end], dtype=torch.float32, device=device)
            negatives = torch.as_tensor(negative_render_features[start:end], dtype=torch.float32, device=device)
            loss = _attention_loss(model, query, render, negatives, temperature=float(temperature))
            losses.append(float(loss.detach().cpu()))
            candidates = torch.cat([render, negatives.reshape(-1, negatives.shape[-1])], dim=0)
            query_z, candidate_z = model(query, candidates)
            pred = torch.argmax(query_z @ candidate_z.T, dim=1)
            labels = torch.arange(end - start, device=device)
            correct += int(torch.sum(pred == labels).item())
            total += int(end - start)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1_acc": float(correct) / max(float(total), 1.0),
    }


def train_qkv_attention(
    query_features: np.ndarray,
    render_features: np.ndarray,
    negative_render_features: np.ndarray,
    config: QKVAttentionTrainingConfig | None = None,
) -> QKVAttentionTrainingRun:
    """Train residual QKV attention on descriptor-level positive/negative sets."""

    config = config or QKVAttentionTrainingConfig()
    query, render, negatives = _validate_training_arrays(query_features, render_features, negative_render_features)
    if query.shape[0] == 0:
        raise ValueError("at least one training sample is required")
    torch.manual_seed(int(config.seed))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device if torch.cuda.is_available() or not str(config.device).startswith("cuda") else "cpu")
    train_idx, eval_idx = _split_indices(query.shape[0], float(config.eval_split_fraction), int(config.seed))
    train_q, train_r, train_n = query[train_idx], render[train_idx], negatives[train_idx]
    eval_q, eval_r, eval_n = query[eval_idx], render[eval_idx], negatives[eval_idx]
    model = ResidualQKVCrossAttention(
        input_dim=int(query.shape[1]),
        attention_dim=int(config.attention_dim),
        alpha=float(config.alpha),
        logit_scale=float(config.logit_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))
    initial = _evaluate_attention(
        model,
        train_q,
        train_r,
        train_n,
        device=device,
        batch_size=min(int(config.batch_size), 1024),
        temperature=float(config.temperature),
    )
    model.train()
    for _ in range(int(config.steps)):
        batch_count = min(int(config.batch_size), train_q.shape[0])
        batch_idx = rng.choice(train_q.shape[0], size=batch_count, replace=False)
        query_batch = torch.as_tensor(train_q[batch_idx], dtype=torch.float32, device=device)
        render_batch = torch.as_tensor(train_r[batch_idx], dtype=torch.float32, device=device)
        negative_batch = torch.as_tensor(train_n[batch_idx], dtype=torch.float32, device=device)
        loss = _attention_loss(model, query_batch, render_batch, negative_batch, temperature=float(config.temperature))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = _evaluate_attention(
        model,
        train_q,
        train_r,
        train_n,
        device=device,
        batch_size=min(int(config.batch_size), 1024),
        temperature=float(config.temperature),
    )
    eval_summary = _evaluate_attention(
        model,
        eval_q,
        eval_r,
        eval_n,
        device=device,
        batch_size=min(int(config.batch_size), 1024),
        temperature=float(config.temperature),
    )
    summary = {
        "initial_loss": float(initial["loss"]),
        "final_loss": float(final["loss"]),
        "train_top1_acc": float(final["top1_acc"]),
        "eval_loss": float(eval_summary["loss"]),
        "eval_top1_acc": float(eval_summary["top1_acc"]),
        "sample_count": int(query.shape[0]),
        "train_sample_count": int(train_q.shape[0]),
        "eval_sample_count": int(eval_q.shape[0]),
        "input_dim": int(query.shape[1]),
        "attention_dim": int(config.attention_dim),
        "alpha": float(config.alpha),
        "steps": int(config.steps),
        "batch_size": int(config.batch_size),
    }
    return QKVAttentionTrainingRun(model=model.cpu().eval(), summary=summary)
