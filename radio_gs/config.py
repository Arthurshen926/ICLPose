"""
RADIO-GS configuration system.

YAML-based config with dataclass schema. No Hydra — matches existing ICLPose pattern.

Usage:
    config = load_config("radio_gs/configs/my_exp.yaml")
    config = override_from_args(config, args)
    save_config(config, "output/radio_gs/config_snapshot.yaml")
"""

import argparse
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

import yaml


@dataclass
class RadioGSConfig:
    """Configuration for RADIO-GS training and evaluation."""

    # Experiment
    exp_name: str = "radio_gs_default"
    output_dir: str = "output/radio_gs"
    seed: int = 42

    # Scene
    scene: str = "room_0"
    ply_path: str = ""

    # RADIO
    radio_version: str = "c-radio_v4-h"
    radio_repo: str = "/root/RADIO"
    radio_feature_dim: int = 1280

    # Architecture
    architecture: str = "explicit"  # "explicit" or "hybrid"
    latent_dim: int = 64  # for explicit
    hybrid_latent_dim: int = 16  # for hybrid
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_log2_size: int = 19
    hash_base_resolution: int = 16
    hash_max_resolution: int = 2048
    hash_output_dim: int = 48
    fine_dim: int = 64
    coarse_dim: int = 64
    hybrid_output_dim: int = 128

    # HCD Codec
    bottleneck_dim: int = 64
    dual_stream: bool = True
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 3

    # FeatSharp
    featsharp_mode: str = "analytical"  # "none", "analytical", "learned", "multiview"
    featsharp_strength: float = 0.5
    featsharp_num_source_views: int = 2

    # Screen-space refiner
    use_refiner: bool = False
    refiner_hidden_dim: int = 128
    refiner_num_blocks: int = 4
    refiner_dropout: float = 0.1
    refiner_rgb_guide: bool = False  # Use RGB as additional input to refiner
    refiner_depth_guide: bool = False  # Use rendered depth as guide (always available)
    refiner_depth_grad: bool = False   # Use depth gradients (3ch: depth+dx+dy) instead of 1ch
    refiner_norm_type: str = "gn"      # "gn" (GroupNorm, stable) or "bn" (BatchNorm, legacy)
    self_guided: bool = False  # Use rendered RGB (not GT) as refiner guide
    lr_refiner: float = 5e-4

    # Joint RGB training (V10)
    train_sh: bool = False  # Unfreeze SH coefficients for joint RGB training
    rgb_loss_weight: float = 0.0  # Weight for RGB reconstruction loss
    lr_sh: float = 5e-4  # Learning rate for SH coefficients

    # Rendering
    image_height: int = 480
    image_width: int = 640
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 319.5
    cy: float = 239.5
    feature_height: int = 30  # H/16 for RADIO patch_size=16
    feature_width: int = 40  # W/16
    max_channels_per_chunk: int = 32
    use_2dgs: bool = False

    # Training
    epochs: int = 100
    batch_size: int = 4
    lr_features: float = 1e-3
    lr_decoder: float = 1e-4
    lr_hash: float = 1e-3
    lr_heads: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 10.0
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    train_mode: str = "decoded"  # "decoded" (1280d) or "latent" (64d)

    # Loss weights
    l2_weight: float = 1.0
    cosine_weight: float = 0.5
    consistency_weight: float = 0.1
    adaptor_weight: float = 0.1
    tv_weight: float = 0.01
    feat_norm_weight: float = 0.0

    # Data
    feature_dir: str = ""  # pre-extracted RADIO features
    depth_dir: str = ""
    semantics_dir: str = ""
    train_split: str = "Sequence_1"
    val_split: str = "Sequence_2"
    mixed_split: bool = False
    mixed_train_ratio: float = 0.8
    mixed_seed: int = 42
    num_workers: int = 4

    # Downstream tasks
    depth_head_type: str = "mlp"  # "linear", "mlp", "dpt"
    depth_num_classes: int = 1
    seg_num_classes: int = 40
    seg_head_type: str = "mlp"
    grounding_use_adaptor: bool = True

    # Checkpointing
    save_every: int = 10
    eval_every: int = 5
    log_every: int = 100  # iterations
    resume_from: str = ""  # Resume training from checkpoint (model + optimizer)
    warmstart_from: str = ""  # Warmstart model weights only


def config_to_dict(config: RadioGSConfig) -> Dict[str, Any]:
    """Convert config dataclass to a plain dict."""
    return asdict(config)


def _coerce_value(field_type: type, value: Any) -> Any:
    """Coerce a parsed YAML value to the expected dataclass field type."""
    if field_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return field_type(value)


def load_config(yaml_path: str) -> RadioGSConfig:
    """Load a RadioGSConfig from a YAML file.

    Unknown keys in the YAML are silently ignored so that experiment
    configs can carry extra metadata without breaking the loader.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    valid_fields = {fld.name: fld for fld in fields(RadioGSConfig)}
    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in valid_fields:
            kwargs[key] = _coerce_value(valid_fields[key].type, value)

    return RadioGSConfig(**kwargs)


def save_config(config: RadioGSConfig, path: str) -> None:
    """Save a RadioGSConfig to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)


def override_from_args(
    config: RadioGSConfig, args: argparse.Namespace
) -> RadioGSConfig:
    """Override config fields with non-None values from argparse.

    Only fields that exist in both the config and the Namespace are
    considered, so extra CLI flags (e.g. ``--config``) are harmless.
    """
    valid_fields = {fld.name: fld for fld in fields(RadioGSConfig)}
    updates = config_to_dict(config)
    for key, value in vars(args).items():
        if value is not None and key in valid_fields:
            updates[key] = _coerce_value(valid_fields[key].type, value)
    return RadioGSConfig(**updates)


# ---------------------------------------------------------------------------
# Quick CLI: ``python -m radio_gs.config <yaml>`` prints the resolved config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    parser = argparse.ArgumentParser(description="Print resolved RadioGSConfig")
    parser.add_argument("yaml", nargs="?", help="Path to YAML config file")
    cli_args = parser.parse_args()

    if cli_args.yaml:
        cfg = load_config(cli_args.yaml)
    else:
        cfg = RadioGSConfig()

    print(json.dumps(config_to_dict(cfg), indent=2))
