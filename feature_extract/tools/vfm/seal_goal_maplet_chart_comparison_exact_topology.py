"""Seal explicit common vertex/face indices over a paired comparison domain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    seal_exact_comparison_domain,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_domain", type=Path, required=True)
    parser.add_argument("--expected_comparison_domain_content_sha256", required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--expected_plan_content_sha256", required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument("--expected_disjoint_authority_content_sha256", required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--expected_source_tree_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite exact comparison domain")
    arrays, metadata = seal_exact_comparison_domain(
        args.comparison_domain,
        expected_upstream_content_sha256=(
            args.expected_comparison_domain_content_sha256
        ),
        plan_path=args.frozen_submap_plan,
        expected_plan_content_sha256=args.expected_plan_content_sha256,
        authority_path=args.disjoint_upstream_authority,
        expected_authority_content_sha256=(
            args.expected_disjoint_authority_content_sha256
        ),
        source_root=args.source_root,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, args.output)
    report = {
        "artifact_type": "goal_maplet_chart_comparison_exact_topology_seal_audit_v1",
        "output": str(args.output.resolve()),
        "output_file_sha256": file_sha256(args.output),
        "output_content_sha256": metadata["content_sha256"],
        "chart_count": metadata["chart_count"],
        "selected_chart_names_in_order_sha256": metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "frozen_submap_plan_content_sha256": metadata[
            "frozen_submap_plan_content_sha256"
        ],
        "disjoint_upstream_authority_content_sha256": metadata[
            "disjoint_upstream_authority_content_sha256"
        ],
        "source_tree_sha256": metadata["source_tree_sha256"],
        "mapping_source_ordered_names_sha256": metadata[
            "mapping_source_ordered_names_sha256"
        ],
        "exact_topology_arrays_sha256": metadata[
            "exact_topology_arrays_sha256"
        ],
        "full_submap_gate_eligible": metadata["full_submap_gate_eligible"],
        "uses_query_or_ground_truth": metadata["uses_query_or_ground_truth"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
