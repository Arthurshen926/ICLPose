"""Descriptor-bank controls for feature-selection causality experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np

from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


DescriptorControl = Literal["identity", "query_shuffle", "map_shuffle", "wrong_scene"]


def _shuffled_rows(descriptors: np.ndarray, seed: int) -> np.ndarray:
    array = np.asarray(descriptors, dtype=np.float32)
    if array.shape[0] < 2:
        raise ValueError("descriptor shuffle controls require at least two descriptors")
    rng = np.random.default_rng(seed)
    order = np.arange(array.shape[0])
    for _ in range(8):
        rng.shuffle(order)
        if np.any(order != np.arange(array.shape[0])):
            break
    return array[order].astype(np.float32, copy=True)


def _replacement_rows(target: TokenDescriptorBank, replacement: TokenDescriptorBank, seed: int) -> np.ndarray:
    if target.descriptors.shape[1] != replacement.descriptors.shape[1]:
        raise ValueError("replacement descriptor dimension must match target")
    if replacement.descriptors.shape[0] == 0:
        raise ValueError("replacement descriptor bank must be non-empty")
    rng = np.random.default_rng(seed)
    order = np.arange(replacement.descriptors.shape[0])
    rng.shuffle(order)
    tiled = np.resize(order, target.descriptors.shape[0])
    return replacement.descriptors[tiled].astype(np.float32, copy=True)


def _normalize_rows(descriptors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-6)


def apply_descriptor_control(
    bank: TokenDescriptorBank,
    control: DescriptorControl,
    seed: int,
    replacement_bank: TokenDescriptorBank | None = None,
) -> TokenDescriptorBank:
    """Return a descriptor bank with image ids preserved but evidence corrupted."""

    if control == "identity":
        descriptors = bank.descriptors.copy()
    elif control in {"query_shuffle", "map_shuffle"}:
        descriptors = _shuffled_rows(bank.descriptors, seed=seed)
    elif control == "wrong_scene":
        if replacement_bank is None:
            raise ValueError("wrong_scene control requires replacement_bank")
        descriptors = _replacement_rows(bank, replacement_bank, seed=seed)
    else:
        raise ValueError(f"unsupported descriptor control: {control}")
    metadata = dict(bank.metadata or {})
    metadata.update(
        {
            "control": control,
            "control_seed": int(seed),
            "replacement_bank": dict(replacement_bank.metadata or {}) if replacement_bank is not None else None,
        }
    )
    return TokenDescriptorBank(
        image_ids=bank.image_ids,
        descriptors=descriptors,
        layer_name=bank.layer_name,
        pooling=bank.pooling,
        normalized=bank.normalized,
        metadata=metadata,
    )


def mask_descriptor_channels_by_utility(
    bank: TokenDescriptorBank,
    utility: np.ndarray,
    fraction: float,
    remove: Literal["high", "low"],
    renormalize: bool = True,
) -> TokenDescriptorBank:
    """Zero selected descriptor columns according to a channel-utility vector."""

    descriptors = np.asarray(bank.descriptors, dtype=np.float32).copy()
    channel_utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    if descriptors.shape[1] != channel_utility.size:
        raise ValueError("descriptor dimension must match utility length")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(round(channel_utility.size * fraction)))
    order = np.argsort(channel_utility, kind="mergesort")
    if remove == "high":
        masked = order[-count:]
    elif remove == "low":
        masked = order[:count]
    else:
        raise ValueError("remove must be 'high' or 'low'")
    descriptors[:, masked] = 0.0
    if renormalize:
        descriptors = _normalize_rows(descriptors).astype(np.float32, copy=False)
    metadata = dict(bank.metadata or {})
    metadata.update(
        {
            "control": f"{remove}_utility_channels_removed",
            "mask_fraction": float(fraction),
            "masked_channel_count": int(count),
            "masked_channel_indices": [int(idx) for idx in np.sort(masked)],
            "renormalized": bool(renormalize),
        }
    )
    return TokenDescriptorBank(
        image_ids=bank.image_ids,
        descriptors=descriptors,
        layer_name=bank.layer_name,
        pooling=bank.pooling,
        normalized=bool(renormalize or bank.normalized),
        metadata=metadata,
    )
