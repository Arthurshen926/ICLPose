from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from feature_field.utils.project_paths import resolve_checkpoint_path, resolve_repo_path


MAINLINE_PATH_SPECS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dataset", "feature_dir"), "repo"),
    (("dataset", "teacher_feature_dir"), "repo"),
    (("dataset", "colmap_dir"), "path"),
    (("dataset", "source_dir"), "path"),
    (("dataset", "train_split"), "path"),
    (("dataset", "test_split"), "path"),
    (("dataset", "val_split"), "path"),
    (("dataset", "train_traj_path"), "path"),
    (("dataset", "val_traj_path"), "path"),
    (("dataset", "train_depth_dir"), "path"),
    (("dataset", "val_depth_dir"), "path"),
    (("dataset", "feature_base_dir"), "repo"),
    (("dcff", "checkpoint"), "checkpoint"),
    (("dcff", "joint_checkpoint"), "checkpoint"),
    (("dcff", "ply_path"), "repo"),
    (("map_supervision", "config_path"), "repo"),
    (("map_supervision", "colmap_dir"), "repo"),
    (("renderer", "ply_path"), "repo"),
    (("renderer", "scale_model_paths", "coarse"), "checkpoint"),
    (("renderer", "scale_model_paths", "mid"), "checkpoint"),
    (("renderer", "scale_model_paths", "fine_sd"), "checkpoint"),
    (("renderer", "scale_model_paths", "fine_dino"), "checkpoint"),
    (("retrieval", "feature_dir"), "repo"),
)


JOINT_RADIO_PATH_SPECS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dataset", "source_dir"), "path"),
    (("dataset", "feature_dir"), "repo"),
    (("dataset", "train_split"), "path"),
    (("dataset", "val_split"), "path"),
    (("retrieval", "feature_dir"), "repo"),
    (("map_supervision", "config_path"), "repo"),
    (("map_supervision", "colmap_dir"), "path"),
)


FEATURE_FIELD_PATH_SPECS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dataset", "source_dir"), "path"),
    (("dataset", "feature_dir"), "repo"),
    (("teacher", "radio_repo"), "repo"),
    (("teacher", "pca_init_dir"), "repo"),
    (("training", "init_ply"), "repo"),
)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_config(path: str) -> dict[str, Any]:
    resolved = resolve_repo_path(path, must_exist=True)
    assert resolved is not None
    with open(resolved, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _get_nested(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(mapping: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    current: dict[str, Any] = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def normalize_config_paths(
    config: dict[str, Any],
    path_specs: Iterable[tuple[Sequence[str], str]],
) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    for keys, kind in path_specs:
        value = _get_nested(normalized, keys)
        if value in (None, ""):
            continue
        if kind == "checkpoint":
            resolved = resolve_checkpoint_path(value)
        elif kind == "path":
            resolved = resolve_repo_path(value, enforce_local=False)
        else:
            resolved = resolve_repo_path(value, enforce_local=True)
        if resolved is not None:
            _set_nested(normalized, keys, str(resolved))
    return normalized


def load_mainline_config(path: str) -> dict[str, Any]:
    resolved = resolve_repo_path(path, must_exist=True)
    assert resolved is not None
    config = load_yaml_config(str(resolved))
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = Path(resolved).parent / base_path
        config = deep_merge(load_mainline_config(str(base_path)), config)
    return normalize_config_paths(config, MAINLINE_PATH_SPECS)


def load_joint_radio_config(
    path: str,
    *,
    default_config: dict[str, Any],
) -> dict[str, Any]:
    resolved = resolve_repo_path(path, must_exist=True)
    assert resolved is not None
    user_cfg = load_yaml_config(str(resolved))
    base_config = user_cfg.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = Path(resolved).parent / base_path
        base_cfg = load_joint_radio_config(str(base_path), default_config=default_config)
    else:
        base_cfg = default_config
    merged = deep_merge(base_cfg, user_cfg)
    return normalize_config_paths(merged, JOINT_RADIO_PATH_SPECS)


def load_scene_feature_field_config(path: str) -> dict[str, Any]:
    config = load_yaml_config(path)
    return normalize_config_paths(config, FEATURE_FIELD_PATH_SPECS)


def load_feature_field_config(path: str) -> dict[str, Any]:
    return load_scene_feature_field_config(path)
