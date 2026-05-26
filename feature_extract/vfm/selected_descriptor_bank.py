"""Build cached descriptors from dense tokens through a trained selector."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


def _load_layer(record: TokenBankRecord, layer_name: str) -> np.ndarray:
    with np.load(record.token_path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
        feature = np.asarray(data[layer_name], dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("dense token feature map must have shape (C, H, W)")
    return feature


def _selected_descriptors_for_batch(
    selector: LocalizableFeatureSelector,
    features: np.ndarray,
    device: torch.device,
    utility_weighted_pooling: bool = False,
    utility_spatial_mask: str = "none",
    utility_spatial_mask_fraction: float = 0.0,
) -> np.ndarray:
    if utility_spatial_mask not in {"none", "high", "low"}:
        raise ValueError("utility_spatial_mask must be one of: none, high, low")
    if utility_spatial_mask != "none" and not 0.0 < utility_spatial_mask_fraction <= 1.0:
        raise ValueError("utility_spatial_mask_fraction must be in (0, 1] when masking is enabled")
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = selector(tensor)
        selected = output.selected
        batch, _dim, height, width = selected.shape
        spatial_count = height * width
        if utility_weighted_pooling:
            weights = output.utility.clamp_min(1e-6)
        else:
            weights = torch.ones((batch, 1, height, width), dtype=selected.dtype, device=selected.device)
        if utility_spatial_mask != "none":
            remove_count = max(1, int(round(spatial_count * float(utility_spatial_mask_fraction))))
            utilities = output.utility.flatten(2)
            top_largest = utility_spatial_mask == "high"
            indices = torch.topk(utilities, k=remove_count, dim=2, largest=top_largest).indices
            keep = torch.ones((batch, 1, spatial_count), dtype=selected.dtype, device=selected.device)
            keep.scatter_(2, indices, 0.0)
            weights = weights.flatten(2) * keep
            descriptors = (selected.flatten(2) * weights).sum(dim=2)
            descriptors = descriptors / weights.sum(dim=2).clamp_min(1e-6)
        else:
            descriptors = (selected * weights).flatten(2).sum(dim=2)
            descriptors = descriptors / weights.flatten(2).sum(dim=2).clamp_min(1e-6)
        descriptors = F.normalize(descriptors, p=2, dim=1, eps=1e-6)
    return descriptors.detach().cpu().numpy().astype(np.float32, copy=False)


def build_selected_descriptor_bank(
    manifest: TokenBankManifest,
    selector: LocalizableFeatureSelector,
    layer_name: str,
    device: str = "cpu",
    batch_size: int = 8,
    utility_weighted_pooling: bool = False,
    utility_spatial_mask: str = "none",
    utility_spatial_mask_fraction: float = 0.0,
    metadata: Mapping[str, object] | None = None,
) -> TokenDescriptorBank:
    """Cache mean-pooled selected descriptors for every image in a token manifest."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if utility_spatial_mask not in {"none", "high", "low"}:
        raise ValueError("utility_spatial_mask must be one of: none, high, low")
    if utility_spatial_mask != "none" and not 0.0 < utility_spatial_mask_fraction <= 1.0:
        raise ValueError("utility_spatial_mask_fraction must be in (0, 1] when masking is enabled")
    manifest.validate(verify_checksums=False)
    torch_device = torch.device(device)
    selector = selector.to(torch_device)
    selector.eval()

    image_ids: list[str] = []
    descriptor_chunks: list[np.ndarray] = []
    records = list(manifest.records)
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        features = [_load_layer(record, layer_name) for record in chunk]
        shapes = {feature.shape for feature in features}
        if len(shapes) != 1:
            raise ValueError("all dense token maps in one batch must have the same shape")
        channels = int(features[0].shape[0])
        if channels != selector.input_dim:
            raise ValueError(f"selector expects {selector.input_dim} channels, got {channels}")
        image_ids.extend(record.image_id for record in chunk)
        descriptor_chunks.append(
            _selected_descriptors_for_batch(
                selector=selector,
                features=np.stack(features, axis=0),
                device=torch_device,
                utility_weighted_pooling=utility_weighted_pooling,
                utility_spatial_mask=utility_spatial_mask,
                utility_spatial_mask_fraction=utility_spatial_mask_fraction,
            )
        )

    descriptors = (
        np.concatenate(descriptor_chunks, axis=0)
        if descriptor_chunks
        else np.zeros((0, selector.output_dim), dtype=np.float32)
    )
    bank_metadata = {
        "batch_size": int(batch_size),
        "selector_input_dim": int(selector.input_dim),
        "selector_output_dim": int(selector.output_dim),
        "selector_group_size": int(selector.group_size),
        "utility_weighted_pooling": bool(utility_weighted_pooling),
        "utility_spatial_mask": utility_spatial_mask,
        "utility_spatial_mask_fraction": float(utility_spatial_mask_fraction),
    }
    bank_metadata.update(dict(metadata or {}))
    return TokenDescriptorBank(
        image_ids=tuple(image_ids),
        descriptors=descriptors,
        layer_name=layer_name,
        pooling=(
            f"selected_{utility_spatial_mask}_utility_masked"
            if utility_spatial_mask != "none"
            else "selected_utility_weighted_mean"
            if utility_weighted_pooling
            else "selected_mean"
        ),
        normalized=True,
        metadata=bank_metadata,
    )
