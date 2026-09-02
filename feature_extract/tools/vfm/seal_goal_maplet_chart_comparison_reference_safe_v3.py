"""Seal source-only MASt3R edge safety into a paired exact chart topology."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_comparison_reference_safe_domain import (
    seal_reference_safe_comparison_domain,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream_exact_topology_v2", type=Path, required=True)
    parser.add_argument("--expected_upstream_v2_content_sha256", required=True)
    parser.add_argument("--optimizer_comparison_domain_v1", type=Path, required=True)
    parser.add_argument("--expected_optimizer_v1_content_sha256", required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument("--expected_disjoint_authority_content_sha256", required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--expected_source_tree_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite reference-safe comparison domain")
    arrays, metadata = seal_reference_safe_comparison_domain(
        args.upstream_exact_topology_v2,
        expected_upstream_v2_content_sha256=(
            args.expected_upstream_v2_content_sha256
        ),
        optimizer_v1_path=args.optimizer_comparison_domain_v1,
        expected_optimizer_v1_content_sha256=(
            args.expected_optimizer_v1_content_sha256
        ),
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
    print(
        json.dumps(
            {
                "artifact_type": metadata["artifact_type"],
                "output": str(args.output.resolve()),
                "output_file_sha256": file_sha256(args.output),
                "output_content_sha256": metadata["content_sha256"],
                "arrays_sha256": metadata["arrays_sha256"],
                "exact_topology_arrays_sha256": metadata[
                    "exact_topology_arrays_sha256"
                ],
                "source_reference_edge_safety_metrics": metadata[
                    "source_reference_edge_safety_metrics"
                ],
                "full_submap_gate_eligible": metadata[
                    "full_submap_gate_eligible"
                ],
                "uses_query_or_ground_truth": metadata[
                    "uses_query_or_ground_truth"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
