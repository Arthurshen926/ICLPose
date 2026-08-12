"""Render a train-OOF selected candidate configuration as strict-run env lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS,
)


ENV_ARGUMENTS = {
    "mapping_view_candidates": "G23_MAPPING_VIEW_CANDIDATES",
    "mapping_view_anchors": "G23_MAPPING_VIEW_ANCHORS",
    "mapping_view_support_pairs": "G23_MAPPING_VIEW_SUPPORT_PAIRS",
    "mapping_view_hypotheses": "G23_MAPPING_VIEW_HYPOTHESES",
    "view_geometry_prescore_per_anchor": "G23_PRESCORE_PER_ANCHOR",
    "view_geometry_exact_verify_count": "G23_EXACT_VERIFY_COUNT",
    "view_geometry_exact_keep_per_anchor": "G23_EXACT_KEEP_PER_ANCHOR",
    "view_geometry_exact_protected_anchors": "G23_EXACT_PROTECTED_ANCHORS",
    "sparse_vfm_primitives_per_child": "G23_SPARSE_PRIMITIVES_PER_CHILD",
    "sparse_vfm_maximum_splat_radius_tokens": "G23_SPARSE_SPLAT_RADIUS",
    "geometry_proposal_confidence": "G23_GEOMETRY_CONFIDENCE",
    "translation_nms_m": "G23_TRANSLATION_NMS_M",
    "rotation_nms_deg": "G23_ROTATION_NMS_DEG",
}


def selected_environment(payload: dict[str, object]) -> dict[str, str]:
    if payload.get("artifact_type") != (
        "goal_maplet_candidate_oof_development_selection_v1"
    ):
        raise ValueError("unsupported candidate-selection report")
    recommended = payload["recommended"]
    arguments = recommended.get("inference_arguments", {})
    missing = [key for key in ENV_ARGUMENTS if key not in arguments]
    if missing:
        raise ValueError(f"selected candidate lacks strict-run arguments: {missing}")
    result = {"G23_CANDIDATE_TAG": str(payload["recommended_tag"])}
    result.update({env: str(arguments[key]) for key, env in ENV_ARGUMENTS.items()})
    return result


def selected_arguments(payload: dict[str, object]) -> list[str]:
    if payload.get("artifact_type") != (
        "goal_maplet_candidate_oof_development_selection_v1"
    ):
        raise ValueError("unsupported candidate-selection report")
    values = payload["recommended"].get("inference_arguments", {})
    if set(values) != set(DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS):
        raise ValueError("selected candidate argument set differs from executable schema")
    tokens = []
    for key in DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS:
        value = values[key]
        if isinstance(value, bool):
            if value:
                tokens.append(f"--{key}")
        elif value is not None:
            tokens.extend((f"--{key}", str(value)))
    return tokens


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--format", choices=("env", "args"), default="env")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.selection).read_text())
    if str(args.format) == "args":
        print("\n".join(selected_arguments(payload)))
        return
    for name, value in selected_environment(payload).items():
        if "\n" in value or "=" in name:
            raise ValueError("candidate environment contains unsafe text")
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
