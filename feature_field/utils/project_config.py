from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from feature_field.utils.project_paths import resolve_checkpoint_path, resolve_repo_path


MAINLINE_PATH_SPECS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dataset", "feature_dir"), "path"),
    (("dataset", "teacher_feature_dir"), "path"),
    (("dataset", "colmap_dir"), "path"),
    (("dataset", "source_dir"), "path"),
    (("dataset", "train_split"), "path"),
    (("dataset", "test_split"), "path"),
    (("dataset", "val_split"), "path"),
    (("dataset", "train_traj_path"), "path"),
    (("dataset", "val_traj_path"), "path"),
    (("dataset", "train_depth_dir"), "path"),
    (("dataset", "val_depth_dir"), "path"),
    (("dataset", "feature_base_dir"), "path"),
    (("dataset", "train_init_poses_path"), "path"),
    (("dataset", "val_init_poses_path"), "path"),
    (("dataset", "init_poses_path"), "path"),
    (("dataset", "teacher_correspondence_path"), "path"),
    (("dataset", "teacher_correspondence_train_path"), "path"),
    (("dataset", "teacher_correspondence_val_path"), "path"),
    (("dcff", "checkpoint"), "path"),
    (("dcff", "joint_checkpoint"), "path"),
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
    (("dataset", "teacher_correspondence_path"), "path"),
    (("dataset", "teacher_correspondence_train_path"), "path"),
    (("dataset", "teacher_correspondence_val_path"), "path"),
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


def sha256_file_or_none(path: str | Path | None) -> str | None:
    if not path:
        return None
    resolved = resolve_repo_path(str(path), enforce_local=False)
    if resolved is None or not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_localization_manifest(path: str) -> dict[str, Any]:
    resolved = resolve_repo_path(path, must_exist=True, enforce_local=False)
    assert resolved is not None
    with open(resolved, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Localization manifest must be a JSON object: {path}")
    manifest = copy.deepcopy(manifest)
    manifest.setdefault("schema_version", 1)
    manifest["path"] = str(resolved)
    return manifest


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


def _manifest_dataset_overrides(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset = copy.deepcopy(manifest.get("dataset") or {})
    init_caches = manifest.get("init_caches") or {}
    train_cache = init_caches.get("train") if isinstance(init_caches, dict) else None
    val_cache = init_caches.get("val") if isinstance(init_caches, dict) else None
    if isinstance(train_cache, dict) and train_cache.get("path"):
        dataset.setdefault("train_init_poses_path", train_cache["path"])
    if isinstance(val_cache, dict) and val_cache.get("path"):
        dataset.setdefault("val_init_poses_path", val_cache["path"])
    return dataset


def _normalize_manifest_hashes(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(manifest)
    init_caches = normalized.get("init_caches")
    if isinstance(init_caches, dict):
        for cache in init_caches.values():
            if isinstance(cache, dict) and cache.get("path"):
                digest = sha256_file_or_none(cache["path"])
                if digest is not None:
                    cache["sha256"] = digest
    dataset = normalized.get("dataset")
    if isinstance(dataset, dict):
        for key in ("train_init_poses_path", "val_init_poses_path", "init_poses_path"):
            path = dataset.get(key)
            if path:
                digest = sha256_file_or_none(path)
                if digest is not None:
                    normalized.setdefault("init_cache_sha256", {})[key] = digest
    return normalized


def apply_localization_manifest_overrides(
    config: dict[str, Any],
    manifest: dict[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if not manifest:
        return copy.deepcopy(config)
    loaded = load_localization_manifest(str(manifest)) if isinstance(manifest, (str, Path)) else copy.deepcopy(manifest)
    loaded = _normalize_manifest_hashes(loaded)
    overrides: dict[str, Any] = {}
    dataset = _manifest_dataset_overrides(loaded)
    if dataset:
        overrides["dataset"] = dataset
    for section in ("dcff", "model", "renderer", "export"):
        value = loaded.get(section)
        if isinstance(value, dict) and value:
            overrides[section] = copy.deepcopy(value)
    merged = deep_merge(config, overrides)
    merged["localization_manifest"] = loaded
    return normalize_config_paths(merged, MAINLINE_PATH_SPECS)


def load_mainline_config(path: str, localization_manifest: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = resolve_repo_path(path, must_exist=True)
    assert resolved is not None
    config = load_yaml_config(str(resolved))
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = Path(resolved).parent / base_path
        config = deep_merge(load_mainline_config(str(base_path)), config)
    config = normalize_config_paths(config, MAINLINE_PATH_SPECS)
    manifest_path = localization_manifest or _get_nested(config, ("localization_manifest", "path"))
    if manifest_path:
        config = apply_localization_manifest_overrides(config, manifest_path)
    return config


def should_restore_pose_checkpoint_map_state(config: dict[str, Any]) -> bool:
    """Return whether a pose-refine checkpoint is allowed to override the DCFF map.

    A configured/exported ``dcff.joint_checkpoint`` is the source of truth for
    localization map features. Restoring map tensors embedded in an older
    pose-refine checkpoint after loading that joint checkpoint silently replaces
    the evaluated map, which breaks fixed-manifest experiments.
    """
    dcff_cfg = config.get("dcff", {}) if isinstance(config, dict) else {}
    explicit = dcff_cfg.get("restore_pose_checkpoint_map_state")
    if explicit is not None:
        return bool(explicit)
    return not bool(dcff_cfg.get("joint_checkpoint"))


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
