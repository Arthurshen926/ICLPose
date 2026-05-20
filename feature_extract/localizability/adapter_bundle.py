"""Helpers for POFD-FS pose-adapter training boundaries."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def _set_module_trainable(module: nn.Module | None, enabled: bool) -> list[torch.nn.Parameter]:
    if module is None:
        return []
    params = list(module.parameters())
    for param in params:
        param.requires_grad_(bool(enabled))
    return params if bool(enabled) else []


def collect_pose_adapter_trainable_parameters(
    adapter: nn.Module,
    pair_matcher: nn.Module | None = None,
    *,
    train_query_adapter: bool = False,
    train_render_adapter: bool = False,
    train_rgb_context: bool = False,
    train_texture_branch: bool = False,
    train_uncertainty: bool = False,
    train_pair_matcher: bool = True,
) -> list[torch.nn.Parameter]:
    """Freeze everything, then expose only the requested localization modules."""
    for param in adapter.parameters():
        param.requires_grad_(False)
    if pair_matcher is not None:
        for param in pair_matcher.parameters():
            param.requires_grad_(False)

    trainable: list[torch.nn.Parameter] = []
    trainable.extend(_set_module_trainable(getattr(adapter, "query_adapter", None), train_query_adapter))
    trainable.extend(_set_module_trainable(getattr(adapter, "render_adapter", None), train_render_adapter))
    if train_rgb_context:
        trainable.extend(_set_module_trainable(getattr(adapter, "query_rgb_stem", None), True))
        trainable.extend(_set_module_trainable(getattr(adapter, "render_rgb_stem", None), True))
    if train_texture_branch:
        trainable.extend(_set_module_trainable(getattr(adapter, "query_texture_branch", None), True))
        trainable.extend(_set_module_trainable(getattr(adapter, "render_texture_branch", None), True))
    if train_uncertainty:
        trainable.extend(_set_module_trainable(getattr(adapter, "query_uncertainty", None), True))
        trainable.extend(_set_module_trainable(getattr(adapter, "render_uncertainty", None), True))
    if pair_matcher is not None:
        trainable.extend(_set_module_trainable(pair_matcher, train_pair_matcher))
    return _unique_params(trainable)


def _unique_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    unique: list[torch.nn.Parameter] = []
    for param in params:
        ident = id(param)
        if ident not in seen:
            seen.add(ident)
            unique.append(param)
    return unique
