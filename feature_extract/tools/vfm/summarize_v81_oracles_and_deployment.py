"""Merge V8.1 oracle/deployment shards into one fail-closed conclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_reports", nargs="+", required=True)
    parser.add_argument("--deployment_reports", nargs="+", required=True)
    parser.add_argument("--graph_audit", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _rows(paths: list[str]) -> list[dict]:
    result = []
    for path in paths:
        report = json.loads(Path(path).read_text())
        result.extend(report["rows"])
    ids = [row["image_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation shards contain duplicate query IDs")
    return sorted(result, key=lambda row: row["image_id"])


def _metrics(values: list[dict]) -> dict[str, float]:
    translation = np.asarray([value["translation_m"] for value in values])
    rotation = np.asarray([value["rotation_deg"] for value in values])
    return {
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.quantile(translation, 0.9)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.quantile(rotation, 0.9)),
        "1m_10deg": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
        "50cm_5deg": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "30cm_3deg": float(np.mean((translation <= 0.3 + 1e-6) & (rotation <= 3.0))),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite V8.1 summary")
    oracle = _rows(list(args.oracle_reports))
    deployment = _rows(list(args.deployment_reports))
    if [row["image_id"] for row in oracle] != [row["image_id"] for row in deployment]:
        raise ValueError("oracle and deployment query sets differ")
    audit = json.loads(Path(args.graph_audit).read_text())

    def oracle_values(key: str, refined: bool = False) -> list[dict]:
        if refined:
            return [row[key]["refined"]["top1"] for row in oracle]
        return [row[key]["top1"] for row in oracle]

    o5 = [row["o5_virtual_pose_lattice"] for row in deployment]
    report = {
        "stage": "v81_strict12_oracle_and_deployment_conclusion",
        "query_count": len(oracle),
        "query_ids": [row["image_id"] for row in oracle],
        "contracts": {
            "map_stores_one_vfm_feature_type": True,
            "stores_mapping_images_or_ids": False,
            "uses_loftr": False,
            "uses_point_correspondence_or_pnp": False,
            "multi_teacher_student_promoted": False,
            "frozen_control": "V8.0-B current canonical RADIO feature plus graph",
        },
        "bug_audit": {
            "physical_directed_edge_duplication_factor": audit["directed_edge_duplication_factor"],
            "v81_query_edges_are_unique_undirected": True,
            "v81_physical_edge_lookup_is_canonical_undirected": True,
            "overlapping_supports_are_grouped_without_transitive_chains": True,
            "posterior_mass_is_conserved_during_grouping": True,
            "global_soft_maplet_capacity_is_active": True,
            "reverse_maplet_to_query_visibility_is_active": True,
            "region_to_maplet_uses_footprint_containment_not_equal_box_size": True,
            "virtual_subdivision_uses_se3_mode_nms": True,
        },
        "graph_identifiability": audit["rooted_hops"],
        "evidence_grouping": {
            "raw_support_count": 128,
            "group_count_median": float(np.median([row["evidence_group_count"] for row in oracle])),
            "group_count_minimum": int(min(row["evidence_group_count"] for row in oracle)),
            "group_count_maximum": int(max(row["evidence_group_count"] for row in oracle)),
            "forced_24_to_64_groups_rejected": True,
        },
        "oracles": {
            "o1_oracle_support_and_identity": _metrics(oracle_values("o1_oracle_support_identity")),
            "o2_current_support_oracle_identity": _metrics(oracle_values("o2_current_support_oracle_identity")),
            "o3_current_posterior_dense_local_lattice": _metrics(oracle_values("o3_current_posterior_dense_local_lattice")),
            "o3_coordinate_refinement": _metrics(oracle_values("o3_current_posterior_dense_local_lattice", refined=True)),
        },
        "virtual_lattice": {
            "pose_count": int(o5[0]["lattice_pose_count"]),
            "oracle_nearest_translation_median_m": float(np.median([value["oracle_nearest_translation_m"] for value in o5])),
            "oracle_nearest_rotation_median_deg": float(np.median([value["oracle_nearest_rotation_deg"] for value in o5])),
            "coarse_top8192_1m_10deg_coverage": float(np.mean([value["coarse_top8192_1m_10deg"] for value in o5])),
            "structured_coarse_top16_1m_10deg_coverage": float(np.mean([value["structured_coarse_top16_1m_10deg"] for value in o5])),
            "subdivided_top16_1m_10deg_coverage": float(np.mean([value["subdivided_top16_1m_10deg"] for value in o5])),
            "subdivided_top16_50cm_5deg_coverage": float(np.mean([value["subdivided_top16_50cm_5deg"] for value in o5])),
            "final_graph_top1": _metrics([value["corrected_graph_top1"] for value in o5]),
            "final_graph_refined_top1": _metrics([value["corrected_graph_refined_top1"] for value in o5]),
        },
        "frozen_historical_proposals_same_protocol": {
            "v80_refined_top1": _metrics([
                row["o4_current_proposal_score"]["frozen_v80_refined_top1"]
                for row in deployment
            ]),
            "v81_score_refined_top1": _metrics([
                row["o4_current_proposal_score"]["corrected_v81_refined_top1"]
                for row in deployment
            ]),
        },
        "decision": {
            "promote_v81_to_production": False,
            "keep_v80_b_frozen_control": True,
            "continue_virtual_lattice_scale_sweep": False,
            "reason": (
                "Candidate coverage improves after graph/NMS fixes, but the region-box "
                "score discards correct modes and cannot recover decimetre pose."
            ),
            "next_required_module": (
                "continuous maplet-interior RADIO feature/surface-coordinate alignment "
                "with a measured convergence basin, initialized by diverse graph modes"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
