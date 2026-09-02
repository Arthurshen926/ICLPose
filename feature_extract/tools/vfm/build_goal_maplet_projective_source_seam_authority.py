"""Build the source-ray projected exact-face seam authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    build_projective_exact_face_seam_authority,
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
        raise FileExistsError("refusing to overwrite projective seam authority")
    authority = build_projective_exact_face_seam_authority(
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
                "authority_semantics_version": metadata[
                    "authority_semantics_version"
                ],
                "chart_count": metadata["chart_count"],
                "frozen_edge_count": metadata["frozen_edge_count"],
                "frozen_correspondence_count": metadata[
                    "frozen_correspondence_count"
                ],
                "edge_with_any_correspondence_count": metadata[
                    "edge_with_any_correspondence_count"
                ],
                "edge_with_both_directions_count": metadata[
                    "edge_with_both_directions_count"
                ],
                "edge_formal_valid_count": metadata[
                    "edge_formal_valid_count"
                ],
                "projective_correspondence_production_candidate": metadata[
                    "projective_correspondence_production_candidate"
                ],
                "legacy_plan_phase_diagnostic_only": metadata[
                    "legacy_plan_phase_diagnostic_only"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
