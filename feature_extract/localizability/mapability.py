"""Mapability diagnostics for selected localization features."""

from __future__ import annotations

import torch


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
