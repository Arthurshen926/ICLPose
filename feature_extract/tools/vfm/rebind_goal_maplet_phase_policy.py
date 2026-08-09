"""Bind one frozen phase operator to a cross-fitted map without refitting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)


MAP_KEYS = (
    "physical_map_sha256",
    "canonical_field_sha256",
    "physical_instance_readout_sha256",
)


def _rebound_metadata(
    source: dict[str, object],
    *,
    source_sha256: str,
    physical_map_sha256: str,
    canonical_field_sha256: str,
    physical_instance_readout_sha256: str,
) -> dict[str, object]:
    artifact_type = str(source.get("artifact_type", ""))
    if artifact_type not in {
        "goal_maplet_phase_readout_policy_v1",
        "goal_maplet_phase_readout_policy_v2",
    }:
        raise ValueError("unsupported phase-policy artifact")
    if not source.get("component_names") or not source.get("coefficient"):
        raise ValueError("source phase policy lacks a frozen operator")
    old_lineage = {key: source.get(key) for key in MAP_KEYS}
    result = dict(source)
    result.update({
        "physical_map_sha256": str(physical_map_sha256),
        "canonical_field_sha256": str(canonical_field_sha256),
        "physical_instance_readout_sha256": str(
            physical_instance_readout_sha256
        ),
        "map_crossfit_lineage_rebind": True,
        "operator_parameters_refit": False,
        "source_policy_sha256": str(source_sha256),
        "source_map_lineage": old_lineage,
        "rebind_semantics": (
            "frozen phase operator; only map/readout lineage identifiers changed"
        ),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_policy", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite rebound phase policy")
    source_path = Path(args.source_policy)
    source = json.loads(source_path.read_text())
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    result = _rebound_metadata(
        source,
        source_sha256=file_sha256(source_path),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=field.content_sha256,
        physical_instance_readout_sha256=file_sha256(
            Path(args.physical_instance_readout)
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_type": result["artifact_type"],
        "component_names": result["component_names"],
        "operator_parameters_refit": result["operator_parameters_refit"],
        "output_json": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
