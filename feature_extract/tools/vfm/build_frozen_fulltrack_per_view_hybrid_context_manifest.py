"""Build a target-free manifest joining aligned local and absolute evidence.

This command does not materialize another multi-gigabyte cache.  It validates
that each translation-mode shard and absolute-phase shard have exactly the
same frozen full-track CSR, then writes a small immutable manifest containing
both source hashes.  The downstream loader repeats the checks before use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
    FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_hybrid_context import (
    FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_HYBRID_CONTEXT_MANIFEST_FORMAT,
    HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES,
    HYBRID_CONTEXT_PROFILE_NAMES,
    HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES,
    hybrid_component_query_id,
    validate_hybrid_component_pair,
)
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    _load_artifact,
)


MANIFEST_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-artifacts", required=True)
    parser.add_argument("--absolute-artifacts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("hybrid component artifact paths must be non-empty and unique")
    return paths


def _component_entries(paths: Mapping[str, Path]) -> list[dict[str, str]]:
    return [
        {
            "query_id": query_id,
            "path": str(path.resolve()),
            "sha256": file_sha256_short(path),
        }
        for query_id, path in sorted(paths.items())
    ]


def build_hybrid_context_manifest(
    *,
    translation_artifacts: Sequence[Path],
    absolute_artifacts: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite hybrid context manifest")
    translation_loaded = {
        hybrid_component_query_id(arrays, context=str(path)): (path, arrays, metadata)
        for path in tuple(Path(value) for value in translation_artifacts)
        for arrays, metadata in (_load_artifact(path),)
    }
    absolute_loaded = {
        hybrid_component_query_id(arrays, context=str(path)): (path, arrays, metadata)
        for path in tuple(Path(value) for value in absolute_artifacts)
        for arrays, metadata in (_load_artifact(path),)
    }
    if (
        not translation_loaded
        or len(translation_loaded) != len(tuple(translation_artifacts))
        or len(absolute_loaded) != len(tuple(absolute_artifacts))
        or set(translation_loaded) != set(absolute_loaded)
    ):
        raise ValueError("hybrid component query-shard sets differ or contain duplicates")
    for query_id in sorted(translation_loaded):
        translation_path, translation_arrays, translation_metadata = translation_loaded[query_id]
        absolute_path, absolute_arrays, absolute_metadata = absolute_loaded[query_id]
        validate_hybrid_component_pair(
            translation_arrays=translation_arrays,
            translation_metadata=translation_metadata,
            translation_context=str(translation_path),
            absolute_arrays=absolute_arrays,
            absolute_metadata=absolute_metadata,
            absolute_context=str(absolute_path),
        )
    result: dict[str, Any] = {
        "format": FULLTRACK_HYBRID_CONTEXT_MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "translation_artifacts": _component_entries(
            {key: value[0] for key, value in translation_loaded.items()}
        ),
        "absolute_artifacts": _component_entries(
            {key: value[0] for key, value in absolute_loaded.items()}
        ),
        "selected_profiles": {
            "translation": list(HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES),
            "absolute": list(HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES),
            "combined": list(HYBRID_CONTEXT_PROFILE_NAMES),
        },
        "component_contracts": {
            "translation_format": FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
            "translation_edge_feature_semantics": (
                FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
            ),
            "absolute_format": FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
            "absolute_edge_feature_semantics": FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
            "hybrid_edge_feature_semantics": FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS,
        },
        "protocol": {
            "feature_export_target_free": True,
            "identity_or_pose_targets_loaded": False,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "support_view_features_averaged_before_inference": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "virtual_manifest_no_feature_cache_materialized": True,
            "translation_and_absolute_csr_exactly_aligned": True,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_hybrid_context_manifest(
        translation_artifacts=_paths(args.translation_artifacts),
        absolute_artifacts=_paths(args.absolute_artifacts),
        output=Path(args.output),
    )
    print(
        json.dumps(
            {
                "format": result["format"],
                "translation_artifact_count": len(result["translation_artifacts"]),
                "absolute_artifact_count": len(result["absolute_artifacts"]),
                "profile_count": len(result["selected_profiles"]["combined"]),
                "protocol": result["protocol"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
