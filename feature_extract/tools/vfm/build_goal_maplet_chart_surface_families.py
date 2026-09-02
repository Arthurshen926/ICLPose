"""Build canonical surface-family IDs over an aligned explicit chart atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_surface_families import (
    SurfaceFamilyConfig,
    build_canonical_surface_families,
)
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import (
    ExplicitChartAtlas,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage_output", type=Path, required=True)
    parser.add_argument("--maximum_intra_chart_face_angle_deg", type=float, default=35.0)
    parser.add_argument("--minimum_patch_faces", type=int, default=2)
    parser.add_argument("--minimum_patch_area_m2", type=float, default=0.02)
    parser.add_argument("--maximum_cross_chart_distance_m", type=float, default=0.75)
    parser.add_argument("--maximum_cross_chart_normal_angle_deg", type=float, default=50.0)
    parser.add_argument("--minimum_smaller_patch_overlap", type=float, default=0.20)
    parser.add_argument("--minimum_larger_patch_overlap", type=float, default=0.03)
    parser.add_argument("--maximum_overlap_point_to_plane_median_m", type=float, default=0.40)
    parser.add_argument("--minimum_cross_chart_matches", type=int, default=8)
    parser.add_argument("--maximum_overlap_samples_per_patch", type=int, default=512)
    parser.add_argument("--minimum_online_family_parameterizations", type=int, default=2)
    parser.add_argument("--minimum_online_supported_patch_area_fraction", type=float, default=0.50)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if any(path.exists() for path in (args.output, args.lineage_output, report_path)):
        raise FileExistsError("refusing to reuse surface-family output")
    atlas = ExplicitChartAtlas.load_npz(args.atlas)
    atlas_content_sha256 = str(atlas.metadata.get("content_sha256"))
    if not atlas_content_sha256 or atlas_content_sha256 == "None":
        raise ValueError("source atlas lacks content lineage")
    atlas_metadata = dict(atlas.metadata)
    atlas_metadata.pop("content_sha256", None)
    if canonical_json_sha256(atlas_metadata) != atlas_content_sha256:
        raise ValueError("source atlas metadata content hash does not replay")
    config = SurfaceFamilyConfig(
        maximum_intra_chart_face_angle_deg=args.maximum_intra_chart_face_angle_deg,
        minimum_patch_faces=args.minimum_patch_faces,
        minimum_patch_area_m2=args.minimum_patch_area_m2,
        maximum_cross_chart_distance_m=args.maximum_cross_chart_distance_m,
        maximum_cross_chart_normal_angle_deg=args.maximum_cross_chart_normal_angle_deg,
        minimum_smaller_patch_overlap=args.minimum_smaller_patch_overlap,
        minimum_larger_patch_overlap=args.minimum_larger_patch_overlap,
        maximum_overlap_point_to_plane_median_m=args.maximum_overlap_point_to_plane_median_m,
        minimum_cross_chart_matches=args.minimum_cross_chart_matches,
        maximum_overlap_samples_per_patch=args.maximum_overlap_samples_per_patch,
        minimum_online_family_parameterizations=args.minimum_online_family_parameterizations,
        minimum_online_supported_patch_area_fraction=args.minimum_online_supported_patch_area_fraction,
    ).validated()
    carrier, lineage = build_canonical_surface_families(
        atlas,
        config=config,
        source_atlas_content_sha256=atlas_content_sha256,
    )
    carrier_metadata = carrier.save_npz(args.output)
    lineage.update(
        {
            "carrier_path": str(args.output),
            "carrier_file_sha256": file_sha256(args.output),
            "carrier_content_sha256": carrier_metadata["content_sha256"],
            "source_atlas_path": str(args.atlas),
            "source_atlas_file_sha256": file_sha256(args.atlas),
        }
    )
    lineage.pop("content_sha256", None)
    lineage["content_sha256"] = canonical_json_sha256(lineage)
    args.lineage_output.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_output.write_text(json.dumps(lineage, indent=2, sort_keys=True))
    report = {
        "artifact_type": "goal_maplet_canonical_surface_family_build_audit_v1",
        "carrier_path": str(args.output),
        "carrier_file_sha256": file_sha256(args.output),
        "carrier_content_sha256": carrier_metadata["content_sha256"],
        "lineage_path": str(args.lineage_output),
        "lineage_file_sha256": file_sha256(args.lineage_output),
        "lineage_content_sha256": lineage["content_sha256"],
        "source_atlas_path": str(args.atlas),
        "source_atlas_file_sha256": file_sha256(args.atlas),
        "parameterization_count": carrier_metadata["parameterization_count"],
        "patch_count": carrier.patch_count,
        "family_count": carrier.family_count,
        "online_eligible_family_count": int(carrier.family_online_eligible.sum()),
        "multi_parameterization_patch_fraction": carrier_metadata[
            "multi_parameterization_patch_fraction"
        ],
        "online_supported_patch_area_fraction": carrier_metadata[
            "online_supported_patch_area_fraction"
        ],
        "overlap_edge_count": len(carrier.overlap_edges),
        "family_support_histogram": {
            str(count): int(np.sum(carrier.family_parameterization_count == count))
            for count in sorted(set(carrier.family_parameterization_count.tolist()))
        },
        "runtime_candidate_unit": "canonical_surface_family",
        "source_views_are_runtime_candidate_ids": False,
        "canonical_geometry_fused": False,
        "decision": (
            "GO_family_carrier_to_geometry_fusion_and_RADIO_UV_gate"
            if carrier_metadata["family_canonicalization_gate_pass"]
            else "KILL_current_atlas_family_canonicalization_insufficient"
        ),
        "limitation": (
            "This carrier canonicalizes identity and parameterization membership only; it does "
            "not yet fuse member patches into one shared runtime mesh."
        ),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
