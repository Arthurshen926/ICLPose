"""Helpers for loading RADIO via torch.hub with a local-or-remote fallback."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Tuple

import torch


OFFICIAL_RADIO_REPO = "NVlabs/RADIO"


def ensure_load_state_dict_from_url_weights_only_compatible() -> None:
    """Allow RADIO hubconf to run on torch versions without weights_only support."""
    load_fn = torch.hub.load_state_dict_from_url
    if getattr(load_fn, "_iclp_accepts_weights_only", False):
        return
    if "weights_only" in inspect.signature(load_fn).parameters:
        return

    def compatible_load_state_dict_from_url(*args, weights_only=None, **kwargs):
        return load_fn(*args, **kwargs)

    compatible_load_state_dict_from_url._iclp_accepts_weights_only = True  # type: ignore[attr-defined]
    torch.hub.load_state_dict_from_url = compatible_load_state_dict_from_url


def resolve_radio_repo(radio_repo: str | None) -> Tuple[str, str]:
    """Return (repo_or_dir, source) for torch.hub.load."""
    repo = str(radio_repo or "").strip()
    if not repo:
        return OFFICIAL_RADIO_REPO, "github"

    repo_path = Path(repo)
    if repo_path.exists():
        return str(repo_path), "local"

    if "/" in repo:
        return repo, "github"

    return OFFICIAL_RADIO_REPO, "github"


def load_radio_model(
    *,
    version: str = "c-radio_v4-h",
    radio_repo: str | None = None,
    printer: Callable[[str], None] | None = print,
):
    """Load RADIO from a local repo if present, else fall back to NVlabs/RADIO."""
    repo_or_dir, source = resolve_radio_repo(radio_repo)
    if printer is not None:
        printer(f"  RADIO hub source: {source} ({repo_or_dir})")

    ensure_load_state_dict_from_url_weights_only_compatible()
    try:
        load_kwargs = {
            "repo_or_dir": repo_or_dir,
            "model": "radio_model",
            "version": version,
            "source": source,
            "skip_validation": True,
        }
        if source == "github":
            load_kwargs["trust_repo"] = True
        return torch.hub.load(
            **load_kwargs,
        )
    except Exception:
        if source == "local":
            if printer is not None:
                printer(
                    f"  Local RADIO repo unavailable or broken, falling back to {OFFICIAL_RADIO_REPO}"
                )
            fallback_kwargs = {
                "repo_or_dir": OFFICIAL_RADIO_REPO,
                "model": "radio_model",
                "version": version,
                "source": "github",
                "skip_validation": True,
                "trust_repo": True,
            }
            return torch.hub.load(
                **fallback_kwargs,
            )
        raise
