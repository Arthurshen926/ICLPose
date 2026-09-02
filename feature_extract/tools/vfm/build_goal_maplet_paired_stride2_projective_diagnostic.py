"""Build the strict paired stride-2 topology and projective diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.paired_stride2_topology_diagnostic import (
    PairedStride2TopologyDiagnostic,
    build_paired_stride2_topology_diagnostic,
    build_stride2_projective_authority,
    diagnostic_report,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_domain_v1", type=Path, required=True)
    parser.add_argument("--expected_v1_content_sha256", required=True)
    parser.add_argument("--comparison_domain_v3", type=Path, required=True)
    parser.add_argument("--expected_v3_content_sha256", required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--expected_plan_content_sha256", required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument(
        "--expected_disjoint_authority_content_sha256", required=True
    )
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--expected_source_tree_sha256", required=True)
    parser.add_argument("--dav2_initializer_root", type=Path, required=True)
    parser.add_argument("--moge3_initializer_root", type=Path, required=True)
    parser.add_argument("--topology_output", type=Path, required=True)
    parser.add_argument("--authority_output", type=Path, required=True)
    parser.add_argument("--report_output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.topology_output, args.authority_output, args.report_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")

    topology = build_paired_stride2_topology_diagnostic(
        args.comparison_domain_v1,
        expected_v1_content_sha256=args.expected_v1_content_sha256,
        comparison_domain_v3_path=args.comparison_domain_v3,
        expected_v3_content_sha256=args.expected_v3_content_sha256,
        frozen_submap_plan_path=args.frozen_submap_plan,
        expected_plan_content_sha256=args.expected_plan_content_sha256,
        disjoint_upstream_authority_path=args.disjoint_upstream_authority,
        expected_disjoint_authority_content_sha256=(
            args.expected_disjoint_authority_content_sha256
        ),
        source_root=args.source_root,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        dav2_initializer_root=args.dav2_initializer_root,
        moge3_initializer_root=args.moge3_initializer_root,
    )
    topology_metadata = topology.save_npz(args.topology_output)
    # Immediate self-contained load is mandatory before the projective build.
    topology = PairedStride2TopologyDiagnostic.load_npz(args.topology_output)

    authority = build_stride2_projective_authority(
        args.topology_output,
        expected_topology_content_sha256=topology_metadata["content_sha256"],
        frozen_submap_plan_path=args.frozen_submap_plan,
        expected_plan_content_sha256=args.expected_plan_content_sha256,
        source_root=args.source_root,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
    )
    authority_metadata = authority.save_npz(args.authority_output)
    # Immediate semantic replay catches any builder/loader drift.
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority_output)
    report = diagnostic_report(topology, authority)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "topology_output": str(args.topology_output),
                "topology_file_sha256": file_sha256(args.topology_output),
                "topology_content_sha256": topology_metadata["content_sha256"],
                "authority_output": str(args.authority_output),
                "authority_file_sha256": file_sha256(args.authority_output),
                "authority_content_sha256": authority_metadata["content_sha256"],
                "report_output": str(args.report_output),
                "report_file_sha256": file_sha256(args.report_output),
                "report_content_sha256": report["content_sha256"],
                "stride2_face_quad_count": report["stride2_face_quad_count"],
                "stride2_packed_vertex_count": report[
                    "stride2_packed_vertex_count"
                ],
                "source_eligible_count_stride2": report[
                    "source_eligible_count_stride2"
                ],
                "formal_component_largest_chart_count": report["graph_summary"][
                    "formal_valid_both_directions"
                ]["largest_component_chart_count"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
