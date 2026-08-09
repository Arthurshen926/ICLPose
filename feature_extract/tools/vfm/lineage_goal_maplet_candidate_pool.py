"""Attach current exact-map lineage to a legacy frozen candidate-pose pool.

Only top-level provenance is added.  Candidate poses, scores, ordering and
evaluation labels are copied byte-for-byte through JSON values and are never
recomputed by this migration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite lineaged candidate pool")
    input_path = Path(args.input_pool)
    pool = json.loads(input_path.read_text())
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    if str(pool.get("physical_map_sha256", "")) != physical.content_sha256:
        raise ValueError("legacy candidate pool and physical map differ")
    if str(pool.get("canonical_field_sha256", "")) != field.content_sha256:
        raise ValueError("legacy candidate pool and canonical field differ")
    if contract.canonical_field_sha256 != field.content_sha256:
        raise ValueError("feature contract and canonical field differ")
    pool["field_feature_contract_sha256"] = contract.content_sha256
    pool["physical_instance_readout_sha256"] = file_sha256(Path(args.physical_instance_readout))
    pool["candidate_pool_lineage_migration"] = {
        "source_sha256": file_sha256(input_path),
        "candidate_pose_values_changed": False,
        "candidate_score_values_changed": False,
        "ground_truth_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "query_count": len(pool.get("rows", [])),
        "field_feature_contract_sha256": contract.content_sha256,
        "physical_instance_readout_sha256": pool["physical_instance_readout_sha256"],
        "candidate_pool_lineage_migration": pool["candidate_pool_lineage_migration"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
