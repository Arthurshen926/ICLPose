"""Small, deterministic artifact-lineage helpers for Goal-Maplet."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


FORBIDDEN_DEPLOYMENT_FLAGS = (
    "stores_mapping_rgb",
    "stores_mapping_image_paths",
    "stores_mapping_image_ids",
    "uses_alike_descriptors",
    "uses_radio_intermediate",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_point_correspondences",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays independent of NPZ compression and dictionary order."""

    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value with stable key and separator semantics."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def capture_repository_state(repository_root: Path) -> dict[str, object]:
    """Snapshot the loaded run's source state before long computation starts."""

    root = Path(repository_root).resolve()

    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = git("status", "--short", "--untracked-files=no")
    tracked_diff = git("diff", "--binary", "HEAD", "--")
    untracked_listing = git("ls-files", "--others", "--exclude-standard")
    source_suffixes = {
        ".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".patch",
    }
    source_prefixes = (
        "configs/", "feature_extract/", "scripts/", "tests/", "docs/",
    )
    untracked_source_hashes = {}
    if untracked_listing != "unavailable":
        for relative in sorted(untracked_listing.splitlines()):
            path = root / relative
            if (
                path.is_file()
                and relative.startswith(source_prefixes)
                and path.suffix.lower() in source_suffixes
            ):
                untracked_source_hashes[relative] = file_sha256(path)
    return {
        "git_commit_sha": git("rev-parse", "HEAD"),
        "git_tracked_worktree_dirty": bool(status and status != "unavailable"),
        "git_tracked_status_sha256": canonical_json_sha256(status),
        "git_tracked_diff_sha256": hashlib.sha256(
            tracked_diff.encode("utf8")
        ).hexdigest(),
        "untracked_source_file_count": len(untracked_source_hashes),
        "untracked_source_files_sha256": canonical_json_sha256(
            untracked_source_hashes
        ),
        "repository_state_capture": "process_start_before_long_computation",
    }


def build_run_manifest(
    *,
    repository_root: Path,
    argv: Sequence[str],
    configuration: Mapping[str, object],
    input_artifacts: Mapping[str, str | None],
    query_ids: Sequence[str],
    device: str,
    numeric_contract: Mapping[str, object],
    candidate_counts: Mapping[str, object] | None = None,
    repository_state_at_start: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the immutable replay identity embedded in a result artifact.

    Artifact values are already-computed content or file hashes.  The helper
    deliberately does not hash large files itself, which keeps manifest
    creation cheap and makes the caller choose the authoritative lineage for
    each artifact type.
    """

    source_state = dict(
        repository_state_at_start
        if repository_state_at_start is not None
        else capture_repository_state(repository_root)
    )
    required_source_fields = {
        "git_commit_sha", "git_tracked_worktree_dirty",
        "git_tracked_status_sha256", "git_tracked_diff_sha256",
        "untracked_source_file_count", "untracked_source_files_sha256",
        "repository_state_capture",
    }
    if set(source_state) != required_source_fields:
        raise ValueError("repository source-state snapshot has an invalid schema")
    config = dict(configuration)
    queries = [str(value) for value in query_ids]
    artifacts = {str(key): value for key, value in sorted(input_artifacts.items())}
    return {
        "schema": "goal_maplet_run_manifest_v1",
        **source_state,
        "repository_state_capture": "process_start_before_long_computation",
        "argv": [str(value) for value in argv],
        "configuration_sha256": canonical_json_sha256(config),
        "configuration": config,
        "input_artifacts": artifacts,
        "query_list_sha256": canonical_json_sha256(queries),
        "query_ids": queries,
        "device": str(device),
        "numeric_contract": dict(numeric_contract),
        "candidate_counts": dict(candidate_counts or {}),
    }


def validate_deployment_metadata(metadata: Mapping[str, object]) -> None:
    for key in FORBIDDEN_DEPLOYMENT_FLAGS:
        if bool(metadata.get(key, False)):
            raise ValueError(f"Goal-Maplet deployment artifact violates contract: {key}")
    if metadata.get("vfm_layer", "radio_final") != "radio_final":
        raise ValueError("Goal-Maplet permits only RADIO-final canonical features")
