"""Merge trajectory-cross-fitted Goal-Maplet configuration evidence shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged configuration evidence")
    shards = [json.loads(Path(value).read_text()) for value in args.inputs]
    for key in (
        "stage", "physical_map_sha256", "canonical_field_sha256",
        "validity_calibration_sha256", "proposal_method", "maximum_modes",
        "typed_graph_sha256", "field_feature_contract_sha256", "proposal_seed_policy",
    ):
        if len({json.dumps(item.get(key), sort_keys=True) for item in shards}) != 1:
            raise ValueError(f"configuration evidence shards differ: {key}")
    contracts = [dict(item.get("configuration_evidence_contract", {})) for item in shards]
    semantic_contract_keys = (
        "feature_names", "maximum_groups", "maximum_children", "child_local_mode_source",
        "evidence_version", "fixed_group_denominator", "one_mode_per_group",
        "typed_null_marginalization", "child_capacity", "primitive_capacity",
        "pose_likelihood_pairing_contract", "soft_assignment", "soft_capacity",
    )
    for key in semantic_contract_keys:
        if len({json.dumps(item.get(key), sort_keys=True) for item in contracts}) != 1:
            raise ValueError(f"configuration evidence contracts differ: {key}")
    rows = [row for shard in shards for row in shard.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("configuration evidence shards contain duplicate queries")
    rows.sort(key=lambda row: str(row["image_id"]))
    application = sorted({
        value for contract in contracts for value in contract.get("application_trajectories", ())
    })
    merged_contract = {
        **{key: contracts[0].get(key) for key in semantic_contract_keys},
        "application_trajectories": application,
        "child_local_factor_calibrator_sha256": sorted({
            str(contract.get("child_local_factor_calibrator_sha256")) for contract in contracts
        }),
        "pose_likelihood_ratio_sha256": sorted({
            str(contract.get("pose_likelihood_ratio_sha256")) for contract in contracts
            if contract.get("pose_likelihood_ratio_sha256") is not None
        }),
        "factor_training_pool_disjoint": bool(all(
            contract.get("factor_training_pool_disjoint", False) for contract in contracts
        )),
        "outer_cross_fit": bool(all(
            contract.get("outer_cross_fit", False) for contract in contracts
        )),
        "non_crossfit_diagnostic_only": bool(any(
            contract.get("non_crossfit_diagnostic_only", False) for contract in contracts
        )),
        "component_contracts": contracts,
    }
    result = {
        **{key: value for key, value in shards[0].items() if key not in (
            "rows", "query_count", "configuration_evidence_contract",
            "summary", "source_shards", "shard_count",
        )},
        "query_count": len(rows), "rows": rows,
        "configuration_evidence_contract": merged_contract,
        "source_shards": list(args.inputs), "shard_count": len(shards),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
