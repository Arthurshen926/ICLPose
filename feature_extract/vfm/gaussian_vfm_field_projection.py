"""Projection helpers for Gaussian VFM feature fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.patch_selector_training import SafePatchSelectorTrainingRun


def project_gaussian_vfm_field_features(
    field: GaussianVFMField,
    selector_run: SafePatchSelectorTrainingRun,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
    selector_path: str | Path = "",
) -> GaussianVFMField:
    """Apply a safe selector to all feature-bearing Gaussians."""

    if int(field.feature_dim) != int(selector_run.summary.input_dim):
        raise ValueError("field feature_dim must match selector input_dim")
    encoded = selector_run.encode_rows(field.features, device=device, batch_size=int(batch_size))
    metadata = dict(field.metadata or {})
    metadata.update(
        {
            "stage": "gaussian_vfm_field_safe_selector_projection",
            "source_feature_dim": int(field.feature_dim),
            "selector_output_dim": int(selector_run.summary.output_dim),
            "selector_path": str(selector_path),
            "active_group_count": int(selector_run.summary.active_group_count),
            "group_count": int(selector_run.summary.group_count),
        }
    )
    return GaussianVFMField(
        xyz=field.xyz,
        features=np.asarray(encoded, dtype=np.float32),
        opacity=field.opacity,
        scale=field.scale,
        gaussian_indices=field.gaussian_indices,
        nearest_track_ids=field.nearest_track_ids,
        support_counts=field.support_counts,
        mean_distances=field.mean_distances,
        metadata=metadata,
    )
