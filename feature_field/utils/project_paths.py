from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LEGACY_PREFIX_REDIRECTS: dict[str, str] = {
    "scene_feature_field": "feature_field",
    "dcff": "feature_field/dcff",
    "utils": "feature_field/utils",
    "feature_3dgs": "feature_gaussian/legacy_3dgs",
}

LEGACY_CONFIG_REDIRECTS: tuple[tuple[str, str], ...] = (
    ("dcff_", "feature_field/configs"),
    ("joint_radio_", "feature_extract/configs"),
    ("concat_loc_", "pose_refine/configs"),
)

LEGACY_OUTPUT_REDIRECTS: tuple[tuple[str, str], ...] = (
    ("2dgs_models", "output/feature_gaussian"),
    ("dcff_", "output/feature_field"),
    ("reconstruction_visuals", "output/feature_field"),
    ("joint_radio", "output/feature_extract"),
    ("features_", "output/feature_extract"),
    ("scr_radio_", "output/feature_extract"),
    ("radio_loc_", "output/feature_retrieval"),
    ("experiment_reports", "output/feature_retrieval"),
    ("concat_loc_", "output/pose_refine"),
    ("pipeline_eval", "output/pose_refine"),
    ("pnp_refine_eval", "output/pose_refine"),
    ("render_compare_eval", "output/pose_refine"),
    ("eval_step_tmp", "output/pose_refine"),
    ("eval_step_tmp2", "output/pose_refine"),
    ("eval_sweep_tmp", "output/pose_refine"),
)

UNIFIED_OUTPUT_PREFIXES: dict[str, str] = {
    "dcff_": "output/feature_field",
    "joint_radio": "output/feature_extract",
    "features_": "output/feature_extract",
    "scr_radio_": "output/feature_extract",
    "concat_loc_": "output/pose_refine",
    "radio_loc_": "output/feature_retrieval",
    "pipeline_eval": "output/pose_refine",
    "pnp_refine_eval": "output/pose_refine",
    "render_compare_eval": "output/pose_refine",
    "experiment_reports": "output/feature_retrieval",
}

MODULE_OUTPUT_REDIRECTS: dict[str, str] = {
    "feature_field/output": "output/feature_field",
    "feature_extract/output": "output/feature_extract",
    "feature_gaussian/output": "output/feature_gaussian",
    "feature_retrieval/output": "output/feature_retrieval",
    "pose_refine/output": "output/pose_refine",
}

UNIFIED_OUTPUT_DIRS: set[str] = {
    "feature_field",
    "feature_extract",
    "feature_gaussian",
    "feature_retrieval",
    "pose_refine",
}


def repo_root() -> Path:
    return REPO_ROOT


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _redirect_legacy_relative_path(path: Path) -> Path | None:
    if path.is_absolute() or not path.parts:
        return None

    root = path.parts[0]
    rest = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()

    if root in LEGACY_PREFIX_REDIRECTS:
        return REPO_ROOT / LEGACY_PREFIX_REDIRECTS[root] / rest

    if root == "configs" and len(path.parts) > 1:
        name = path.parts[1]
        for prefix, target in LEGACY_CONFIG_REDIRECTS:
            if name.startswith(prefix):
                return REPO_ROOT / target / rest
        return REPO_ROOT / "legacy/configs" / rest

    if root == "output" and len(path.parts) > 1:
        name = path.parts[1]
        if name in UNIFIED_OUTPUT_DIRS:
            suffix = Path(*path.parts[2:]) if len(path.parts) > 2 else Path()
            return REPO_ROOT / "output" / name / suffix
        for prefix, target in LEGACY_OUTPUT_REDIRECTS:
            if name == prefix or name.startswith(prefix):
                return REPO_ROOT / target / rest
        for prefix, target in UNIFIED_OUTPUT_PREFIXES.items():
            if name.startswith(prefix):
                suffix = Path(*path.parts[2:]) if len(path.parts) > 2 else Path()
                return REPO_ROOT / target / name / suffix
        return REPO_ROOT / "legacy/output" / rest

    path_str = path.as_posix()
    for prefix, target in MODULE_OUTPUT_REDIRECTS.items():
        if path_str == prefix:
            return REPO_ROOT / target
        if path_str.startswith(prefix + "/"):
            suffix = path_str[len(prefix) + 1 :]
            return REPO_ROOT / target / suffix

    return None


def _relocate_repo_absolute_path(path: Path) -> Path | None:
    if not path.is_absolute():
        return None
    parts = path.parts
    if len(parts) < 4:
        return None
    if parts[1] != "root":
        return None
    repo_name = parts[2]
    if repo_name in {REPO_ROOT.name, "ICLPose", "ICLPose-Loc", "ICLPose-loc"}:
        candidate = REPO_ROOT / Path(*parts[3:])
        return candidate
    return None


def resolve_repo_path(
    path: str | Path | None,
    *,
    enforce_local: bool = False,
    must_exist: bool = False,
) -> Path | None:
    if path in (None, ""):
        return None

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        if _is_relative_to(candidate, REPO_ROOT):
            resolved = candidate
        elif (relocated := _relocate_repo_absolute_path(candidate)) is not None:
            resolved = relocated
        elif enforce_local:
            raise ValueError(
                f"Expected a repository-local path, but got external path: {candidate}"
            )
        else:
            resolved = candidate
    else:
        direct = REPO_ROOT / candidate
        redirected = _redirect_legacy_relative_path(candidate)
        if direct.exists() or redirected is None:
            resolved = direct
        else:
            resolved = redirected

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Resolved path does not exist: {resolved}")
    return resolved


def resolve_checkpoint_path(
    path: str | Path | None,
    *,
    must_exist: bool = False,
) -> Path | None:
    return resolve_repo_path(
        path,
        enforce_local=True,
        must_exist=must_exist,
    )


def stringify_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
