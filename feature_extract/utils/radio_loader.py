"""Helpers for loading RADIO via torch.hub with a local-or-remote fallback."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Sequence, Tuple

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
    adaptor_names: str | Sequence[str] | None = None,
    visual_only_siglip2: bool = False,
    printer: Callable[[str], None] | None = print,
):
    """Load RADIO from a local repo if present, else fall back to NVlabs/RADIO."""
    repo_or_dir, source = resolve_radio_repo(radio_repo)
    if printer is not None:
        printer(f"  RADIO hub source: {source} ({repo_or_dir})")

    ensure_load_state_dict_from_url_weights_only_compatible()
    requested_adaptors = (
        [adaptor_names]
        if isinstance(adaptor_names, str)
        else list(adaptor_names or ())
    )
    use_visual_siglip = bool(
        visual_only_siglip2 and "siglip2-g" in requested_adaptors
    )
    hub_adaptors = (
        [name for name in requested_adaptors if name != "siglip2-g"]
        if use_visual_siglip
        else requested_adaptors
    )

    def finish_loaded(value):
        if not use_visual_siglip:
            return value
        model, checkpoint = value
        # The official SigLIP2 adaptor unconditionally downloads and retains
        # its text tower.  Localization only needs the frozen visual readout,
        # whose complete weights already live in the C-RADIO checkpoint.
        # Reconstruct that exact GenericAdaptor and omit the unused text model.
        from timm.models import clean_state_dict
        from radio.adaptor_generic import GenericAdaptor

        state_dict = checkpoint.get(
            "state_dict_ema", checkpoint.get("state_dict")
        )
        state_dict = clean_state_dict(state_dict)
        teacher_index = next(
            index
            for index, config in enumerate(checkpoint["args"].teachers)
            if config["name"] == "siglip2-g"
        )
        teacher_config = checkpoint["args"].teachers[teacher_index]
        adaptor_state = {}
        for key, tensor in state_dict.items():
            for prefix, output_prefix in (
                (f"_heads.{teacher_index}", "summary"),
                ("_heads.siglip2-g", "summary"),
                (f"_feature_projections.{teacher_index}", "feature"),
                ("_feature_projections.siglip2-g", "feature"),
            ):
                if key.startswith(prefix):
                    adaptor_state[output_prefix + key[len(prefix) :]] = tensor
        adaptor = GenericAdaptor(
            checkpoint["args"], teacher_config, adaptor_state
        )
        adaptor.head_idx = int(teacher_config.get("token_slot", teacher_index))
        model.adaptors["siglip2-g"] = adaptor
        return model

    try:
        load_kwargs = {
            "repo_or_dir": repo_or_dir,
            "model": "radio_model",
            "version": version,
            "source": source,
            "skip_validation": True,
            "adaptor_names": hub_adaptors,
            "return_checkpoint": use_visual_siglip,
        }
        if source == "github":
            load_kwargs["trust_repo"] = True
        return finish_loaded(torch.hub.load(**load_kwargs))
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
                "adaptor_names": hub_adaptors,
                "return_checkpoint": use_visual_siglip,
            }
            return finish_loaded(torch.hub.load(**fallback_kwargs))
        raise
