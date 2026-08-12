"""Render path-free refinement CLI arguments from a frozen G23 configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


DIRECT_KEYS = (
    "mode_name", "refine_topk", "refinement_candidate_policy",
    "refinement_score_margin", "minimum_refinement_candidates",
    "refinement_basin_translation_m", "refinement_basin_rotation_deg",
    "passthrough_without_additional_expert", "baseline_mode_count",
    "additional_expert_mode_count", "maximum_splat_radius_tokens",
    "validation_splat_radius_tokens", "require_cross_splat_winner_consistency",
)
OPTIMIZER_KEYS = (
    "translation_steps_m", "rotation_steps_deg", "iterations_per_scale",
    "minimum_score_improvement",
)


def frozen_refinement_arguments(payload: dict[str, object]) -> list[str]:
    if payload.get("artifact_type") != "goal_maplet_g23_frozen_configuration_v1":
        raise ValueError("unsupported frozen G23 configuration")
    configuration = payload.get("refinement_execution_configuration", {})
    if not isinstance(configuration, dict) or any(
        key not in configuration for key in DIRECT_KEYS
    ):
        raise ValueError("frozen refinement execution fields are incomplete")
    optimizer = configuration.get("refinement_optimizer", {})
    if not isinstance(optimizer, dict) or any(
        key not in optimizer for key in OPTIMIZER_KEYS
    ):
        raise ValueError("frozen refinement optimizer fields are incomplete")
    values = {
        **{key: configuration[key] for key in DIRECT_KEYS},
        **{key: optimizer[key] for key in OPTIMIZER_KEYS},
    }
    tokens = []
    for key in (*DIRECT_KEYS, *OPTIMIZER_KEYS):
        value = values[key]
        if isinstance(value, bool):
            if value:
                tokens.append(f"--{key}")
        elif isinstance(value, list):
            tokens.extend((f"--{key}", ",".join(str(item) for item in value)))
        elif value is not None:
            tokens.extend((f"--{key}", str(value)))
    return tokens


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.frozen).read_text())
    print("\n".join(frozen_refinement_arguments(payload)))


if __name__ == "__main__":
    main()
