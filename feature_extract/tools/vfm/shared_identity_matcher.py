"""Identity-only residual matcher with shared physical anchor messages.

Anchor IDs are equality keys, never embedding indices. No absolute map position
or provisional query pose is consumed. Unknown labels remain unsupervised.
"""
import torch
from torch import nn


class SharedIdentityMatcher(nn.Module):
    def __init__(self, width=48, shared=True):
        super().__init__()
        self.shared = shared
        self.query = nn.Sequential(nn.Linear(136, width), nn.LayerNorm(width), nn.GELU())
        # Only intrinsic appearance: the last five map inputs depend on the query's candidate set.
        self.anchor = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width), nn.GELU())
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.head = nn.Sequential(nn.Linear(width * 3 + 2, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.feature_weight = nn.Parameter(torch.tensor([10., 0.]))

    def forward(self, query, map, edges, similarity, ids):
        valid = (query.abs().sum(-1) > 0)[..., None] & (map.abs().sum(-1) > 0) & (ids >= 0)
        q = self.query(query)
        m = self.anchor(map[..., :128])
        appearance = (similarity * self.feature_weight).sum(-1)
        weights = (10 * similarity[..., 0]).masked_fill(~valid, -1e4).softmax(-1) * valid
        pooled = []
        for b in range(len(query)):
            mask = valid[b].flatten()
            # Pool weighted query messages onto each unique physical anchor, then scatter back.
            msg = q[b, :, None].expand_as(m[b]).reshape(-1, q.shape[-1])
            if self.shared and mask.any():
                _, inverse = torch.unique(ids[b].flatten()[mask], sorted=True, return_inverse=True)
                w = weights[b].flatten()[mask]
                sums = torch.zeros((int(inverse.max()) + 1, q.shape[-1]), device=q.device, dtype=q.dtype)
                mass = torch.zeros((len(sums), 1), device=q.device, dtype=q.dtype)
                sums.index_add_(0, inverse, msg[mask] * w[:, None])
                mass.index_add_(0, inverse, w[:, None])
                messages = torch.zeros_like(msg)
                messages[mask] = (sums / mass.clamp_min(1e-8))[inverse]
            else:
                messages = msg * mask[:, None]
            pooled.append(messages.reshape_as(m[b]))
        m = m + self.update(torch.cat([m, torch.stack(pooled)], -1))
        # Query adjacency is pose-free and depth-gated; it does not use predicted overlap.
        qvalid = valid.any(-1)
        graph = edges * qvalid[:, :, None] * qvalid[:, None, :]
        context = torch.bmm(graph, q) / graph.sum(-1, keepdim=True).clamp_min(1)
        q = (q + context)[:, :, None].expand_as(m)
        residual = self.head(torch.cat([q, m, q * m, similarity], -1)).squeeze(-1)
        return (appearance + residual).masked_fill(~valid, -1e4)


def identity_objective(logits, similarity, positive, known, valid):
    """Set-valued known loss plus a weak full-candidate appearance trust penalty.

The KL term is a teacher prior, not extra positive/negative supervision.
    """
    available = positive.any(-1)
    if available.any():
        denom = logits.masked_fill(~known, -1e4).logsumexp(-1)
        numer = logits.masked_fill(~positive, -1e4).logsumexp(-1)
        supervised = (denom - numer)[available].mean()
    else:
        supervised = logits.sum() * 0
    teacher = (10 * similarity[..., 0]).masked_fill(~valid, -1e4).softmax(-1).detach()
    kl = (teacher * (teacher.clamp_min(1e-12).log() - logits.log_softmax(-1))).sum(-1)
    active = valid.any(-1)
    trust = kl[active].mean() if active.any() else logits.sum() * 0
    return supervised + .1 * trust, supervised, trust


def reciprocal_identity(logits, ids, coarse_similarity, threshold, normalize=True):
    """Sparse candidate-graph reciprocity, with the frozen appearance floor.

Row log-softmax fixes the additive score gauge left unconstrained by training.
This is not probability calibration. normalize=False is an explicit raw-score control.
Ties admit all equal maxima; final PnP still counts each query token once.
    """
    import numpy as np
    logits = np.asarray(logits)
    ids = np.asarray(ids)
    if normalize:
        shifted = logits - logits.max(-1, keepdims=True)
        logits = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    selected = logits.argmax(-1)
    row = np.arange(len(ids))
    _, inverse = np.unique(ids, return_inverse=True)
    best = np.full(int(inverse.max()) + 1, -np.inf)
    np.maximum.at(best, inverse, logits.ravel())
    winner = best[inverse.reshape(ids.shape)[row, selected]]
    keep = (logits[row, selected] >= winner) & (coarse_similarity[row, selected] >= threshold)
    return np.flatnonzero(keep), selected[keep]


class AppearanceIdentityMatcher(nn.Module):
    """Two global similarity weights; no anchor-specific trainable information."""
    def __init__(self):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.tensor([10., 0.]))

    def forward(self, query, map, edges, similarity, ids):
        valid = (query.abs().sum(-1) > 0)[..., None] & (map.abs().sum(-1) > 0) & (ids >= 0)
        return (similarity * self.feature_weight).sum(-1).masked_fill(~valid, -1e4)
