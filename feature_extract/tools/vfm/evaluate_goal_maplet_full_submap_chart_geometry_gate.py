"""Run the strict M0/M1/M2 full-submap chart geometry gate.

Invalid, absent, or mismatched lineage is written as ``KILL_INPUT_CONTRACT``
with ``NOT_EVALUATED`` performance semantics.  Such a report must never be
interpreted as evidence that either geometry representation is good or bad.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    FullSubmapGeometryGateConfig,
    blocked_input_report,
    evaluate_full_submap_geometry_gate_from_paths,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0_bounded_map", type=Path, required=True)
    parser.add_argument("--m1_initial_atlas", type=Path, required=True)
    parser.add_argument("--m1_atlas", type=Path, required=True)
    parser.add_argument("--m2_initial_atlas", type=Path, required=True)
    parser.add_argument("--m2_atlas", type=Path, required=True)
    parser.add_argument("--held_ray_inventory", type=Path, required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument("--comparison_domain", type=Path, required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=260830)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite full-submap geometry gate report")
    inputs = {
        "disjoint_upstream_authority": args.disjoint_upstream_authority,
        "comparison_domain": args.comparison_domain,
        "held_ray_inventory": args.held_ray_inventory,
        "M0_bounded_map": args.m0_bounded_map,
        "M1_initial_atlas": args.m1_initial_atlas,
        "M1_atlas": args.m1_atlas,
        "M2_initial_atlas": args.m2_initial_atlas,
        "M2_atlas": args.m2_atlas,
        "frozen_submap_plan": args.frozen_submap_plan,
    }
    status = 0
    try:
        report = evaluate_full_submap_geometry_gate_from_paths(
            authority_path=args.disjoint_upstream_authority,
            comparison_domain_path=args.comparison_domain,
            held_ray_inventory_path=args.held_ray_inventory,
            m0_bounded_map_path=args.m0_bounded_map,
            m1_initial_atlas_path=args.m1_initial_atlas,
            m1_atlas_path=args.m1_atlas,
            m2_initial_atlas_path=args.m2_initial_atlas,
            m2_atlas_path=args.m2_atlas,
            frozen_submap_plan_path=args.frozen_submap_plan,
            config=FullSubmapGeometryGateConfig(
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            ),
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        report = blocked_input_report(error, inputs)
        status = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "input_contract": report["input_contract"],
                "scientific_performance_conclusion": report["scientific_performance_conclusion"],
                "decisions": report["decisions"],
                "content_sha256": report["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
