"""Mapability diagnostics for selected localization features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def track_feature_variance(
    features: torch.Tensor,
    track_ids: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure within-track feature variance for 3D aggregation diagnostics.

    If features are shaped (V,N,C) and track_ids are shaped (N,), the function
    first averages observations across V for each point before computing
    per-track variance across points. If features are shaped (N,C), it uses
    them directly.
    """
    if features.ndim == 3 and track_ids.numel() == features.shape[1]:
        point_features = features.float().mean(dim=0)
    elif features.ndim == 2 and track_ids.numel() == features.shape[0]:
        point_features = features.float()
    else:
        raise ValueError("features must be (N,C) or (V,N,C), with track_ids length N")
    track_ids = track_ids.to(device=point_features.device)
    valid = valid_mask.to(device=point_features.device).bool() if valid_mask is not None else torch.ones_like(track_ids, dtype=torch.bool)
    variances = []
    for track_id in torch.unique(track_ids[valid]):
        mask = valid & (track_ids == track_id)
        if int(mask.sum()) < 2:
            continue
        vals = point_features[mask]
        variances.append(vals.var(dim=0, unbiased=False).mean())
    if variances:
        variance = torch.stack(variances).mean()
        num_tracks = point_features.new_tensor(float(len(variances)))
    else:
        variance = point_features.sum() * 0.0
        num_tracks = point_features.new_tensor(0.0)
    return variance, {"num_tracks": num_tracks}


def observation_track_feature_variance(
    features: torch.Tensor,
    track_ids: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure feature consistency across observations of the same 3D track.

    Unlike :func:`track_feature_variance`, this function expects every row in
    ``features`` to be one observation of a 3D primitive/track.  It computes the
    within-track variance across observations and averages over tracks with at
    least two valid observations.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape (N,C)")
    if track_ids.numel() != features.shape[0]:
        raise ValueError("track_ids length must match number of observations")
    point_features = features.float()
    track_ids = track_ids.to(device=point_features.device)
    valid = (
        valid_mask.to(device=point_features.device).bool()
        if valid_mask is not None
        else torch.ones_like(track_ids, dtype=torch.bool)
    )
    variances = []
    num_observations = point_features.new_tensor(float(valid.sum().item()))
    for track_id in torch.unique(track_ids[valid]):
        mask = valid & (track_ids == track_id)
        if int(mask.sum()) < 2:
            continue
        vals = point_features[mask]
        variances.append(vals.var(dim=0, unbiased=False).mean())
    if variances:
        variance = torch.stack(variances).mean()
        num_tracks = point_features.new_tensor(float(len(variances)))
    else:
        variance = point_features.sum() * 0.0
        num_tracks = point_features.new_tensor(0.0)
    return variance, {"num_tracks": num_tracks, "num_observations": num_observations}


def observation_track_feature_separability(
    features: torch.Tensor,
    track_ids: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure between-track separation relative to within-track variance."""
    if features.ndim != 2:
        raise ValueError("features must have shape (N,C)")
    if track_ids.numel() != features.shape[0]:
        raise ValueError("track_ids length must match number of observations")
    point_features = features.float()
    track_ids = track_ids.to(device=point_features.device)
    valid = (
        valid_mask.to(device=point_features.device).bool()
        if valid_mask is not None
        else torch.ones_like(track_ids, dtype=torch.bool)
    )
    centroids = []
    variances = []
    for track_id in torch.unique(track_ids[valid]):
        mask = valid & (track_ids == track_id)
        vals = point_features[mask]
        if vals.numel() == 0:
            continue
        centroids.append(vals.mean(dim=0))
        if int(vals.shape[0]) >= 2:
            variances.append(vals.var(dim=0, unbiased=False).mean())
    if len(centroids) >= 2:
        centroid_tensor = torch.stack(centroids, dim=0)
        pairwise = torch.pdist(centroid_tensor, p=2).pow(2)
        between = pairwise.mean()
    else:
        between = point_features.sum() * 0.0
    if variances:
        within = torch.stack(variances).mean()
    else:
        within = point_features.sum() * 0.0
    ratio = between / within.clamp_min(1.0e-12)
    return ratio, {
        "num_tracks": point_features.new_tensor(float(len(centroids))),
        "num_observations": point_features.new_tensor(float(valid.sum().item())),
        "between_track_distance": between,
        "within_track_variance": within,
    }


def render_query_feature_consistency(
    query_features: torch.Tensor,
    render_features: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average dense query/render cosine consistency over valid candidates.

    ``query_features`` is shaped ``(B,C,H,W)``. ``render_features`` may be
    ``(B,C,H,W)`` for one render per query or ``(B,K,C,H,W)`` for candidate
    banks.  The optional mask follows the render spatial shape without the
    channel dimension: ``(B,H,W)`` or ``(B,K,H,W)``.
    """
    if query_features.ndim != 4:
        raise ValueError("query_features must have shape (B,C,H,W)")
    if render_features.ndim == 4:
        render = render_features[:, None]
        query = query_features[:, None]
    elif render_features.ndim == 5:
        render = render_features
        query = query_features[:, None].expand(-1, render.shape[1], -1, -1, -1)
    else:
        raise ValueError("render_features must have shape (B,C,H,W) or (B,K,C,H,W)")
    if query.shape != render.shape:
        raise ValueError("query and render feature shapes are incompatible")
    cosine = (F.normalize(query.float(), dim=2, eps=1.0e-6) * F.normalize(render.float(), dim=2, eps=1.0e-6)).sum(dim=2)
    if valid_mask is None:
        valid = torch.ones_like(cosine, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=cosine.device).bool()
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.shape != cosine.shape:
            raise ValueError("valid_mask must have shape (B,H,W) or (B,K,H,W)")
    if valid.any():
        consistency = cosine[valid].mean()
    else:
        consistency = cosine.sum() * 0.0
    return consistency, {"num_valid": cosine.new_tensor(float(valid.sum().item()))}
