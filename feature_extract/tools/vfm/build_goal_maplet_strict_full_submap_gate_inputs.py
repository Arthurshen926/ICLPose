"""Build strict held rays plus a fair source-only bounded surface control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.full_submap_gate_inputs import (
    StrictGateInputBuildConfig,
    bounded_submap_authority_from_artifacts,
    build_strict_full_submap_gate_inputs,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument("--comparison_domain_v3", type=Path, required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--dav2_initializers", type=Path, required=True)
    parser.add_argument("--moge3_initializers", type=Path, required=True)
    parser.add_argument("--topology_stride", type=int, choices=(4, 8), default=4)
    parser.add_argument("--bound_margin_m", type=float, default=1.0)
    parser.add_argument("--held_confidence_threshold", type=float, default=0.25)
    parser.add_argument("--temporal_block_size", type=int, default=3)
    parser.add_argument("--output_held_rays", type=Path, required=True)
    parser.add_argument("--output_source_surface_m0", type=Path, required=True)
    parser.add_argument("--output_bounded_submap", type=Path, required=True)
    parser.add_argument("--output_audit", type=Path, required=True)
    args = parser.parse_args()
    outputs = (
        args.output_held_rays,
        args.output_source_surface_m0,
        args.output_bounded_submap,
        args.output_audit,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite strict gate outputs: " + ", ".join(existing))
    rays, surface, audit = build_strict_full_submap_gate_inputs(
        authority_path=args.disjoint_upstream_authority,
        comparison_domain_path=args.comparison_domain_v3,
        frozen_submap_plan_path=args.frozen_submap_plan,
        dav2_initializers=args.dav2_initializers,
        moge3_initializers=args.moge3_initializers,
        config=StrictGateInputBuildConfig(
            topology_stride=args.topology_stride,
            bound_margin_m=args.bound_margin_m,
            held_confidence_threshold=args.held_confidence_threshold,
            temporal_block_size=args.temporal_block_size,
        ),
    )
    rays_metadata = rays.save_npz(args.output_held_rays)
    surface_metadata = surface.save_npz(args.output_source_surface_m0)
    bounded = bounded_submap_authority_from_artifacts(rays, surface)
    args.output_bounded_submap.parent.mkdir(parents=True, exist_ok=True)
    bounded_temporary = args.output_bounded_submap.with_name(
        args.output_bounded_submap.name + ".temporary"
    )
    bounded_temporary.write_text(json.dumps(bounded, indent=2, sort_keys=True))
    os.replace(bounded_temporary, args.output_bounded_submap)
    report = dict(audit)
    report.update(
        {
            "held_ray_inventory": str(args.output_held_rays.resolve()),
            "held_ray_inventory_file_sha256": file_sha256(args.output_held_rays),
            "held_ray_inventory_content_sha256": rays_metadata["content_sha256"],
            "source_surface_M0": str(args.output_source_surface_m0.resolve()),
            "source_surface_M0_file_sha256": file_sha256(
                args.output_source_surface_m0
            ),
            "source_surface_M0_content_sha256": surface_metadata["content_sha256"],
            "bounded_submap_authority": str(args.output_bounded_submap.resolve()),
            "bounded_submap_authority_file_sha256": file_sha256(
                args.output_bounded_submap
            ),
            "bounded_submap_authority_content_sha256": bounded["content_sha256"],
        }
    )
    report.pop("content_sha256", None)
    from feature_extract.vfm.localization_goal_maplet.lineage import (
        canonical_json_sha256,
    )

    report["content_sha256"] = canonical_json_sha256(report)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
