"""Freeze source-only seam correspondences on the exact physical topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.source_seam_correspondence import (
    build_source_seam_correspondence_authority,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_domain_v3", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite source seam authority")
    authority = build_source_seam_correspondence_authority(
        args.comparison_domain_v3,
        expected_comparison_domain_content_sha256=(
            args.expected_comparison_domain_content_sha256
        ),
        frozen_submap_plan_path=args.frozen_submap_plan,
        expected_plan_content_sha256=args.expected_plan_content_sha256,
        disjoint_upstream_authority_path=args.disjoint_upstream_authority,
        expected_disjoint_authority_content_sha256=(
            args.expected_disjoint_authority_content_sha256
        ),
        source_root=args.source_root,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
    )
    metadata = authority.save_npz(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_file_sha256": file_sha256(args.output),
                "content_sha256": metadata["content_sha256"],
                "chart_count": metadata["chart_count"],
                "frozen_edge_count": metadata["frozen_edge_count"],
                "frozen_correspondence_count": metadata[
                    "frozen_correspondence_count"
                ],
                "m0_reachable_edge_count": metadata["m0_reachable_edge_count"],
                "m0_unreachable_edge_count": metadata[
                    "m0_unreachable_edge_count"
                ],
                "formal_arm_gate_reachability_eligible": metadata[
                    "formal_arm_gate_reachability_eligible"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
