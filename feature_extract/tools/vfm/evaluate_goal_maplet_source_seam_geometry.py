"""Evaluate aligned chart arms on frozen source-only seam correspondences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.source_seam_correspondence import (
    SourceSeamCorrespondenceAuthority,
    evaluate_source_seam_geometry,
)


def _replay_manifest(path: Path, expected_content_sha256: str) -> dict:
    manifest = json.loads(path.read_text())
    content = dict(manifest)
    claimed = content.pop("content_sha256", None)
    if claimed != canonical_json_sha256(content):
        raise ValueError("alignment manifest content hash differs")
    if claimed != expected_content_sha256:
        raise ValueError("alignment manifest differs from experiment pin")
    return manifest


def _load_arm(
    label: str,
    charts_path: Path,
    manifest_path: Path,
    expected_manifest_content_sha256: str,
    authority: SourceSeamCorrespondenceAuthority,
) -> tuple[np.ndarray, dict]:
    manifest = _replay_manifest(manifest_path, expected_manifest_content_sha256)
    if manifest.get("artifact_type") != "goal_maplet_masked_chart_alignment_gate_v1":
        raise ValueError(f"{label} is not a strict chart-alignment run")
    if manifest.get("uses_query_or_ground_truth") is not False:
        raise ValueError(f"{label} alignment is not source-only")
    if manifest.get("paired_common_pixel_domain") is not True:
        raise ValueError(f"{label} alignment lacks a paired common pixel domain")
    if manifest.get("charts_data_file_sha256") != file_sha256(charts_path):
        raise ValueError(f"{label} charts data differs from its manifest")
    metadata = authority.metadata
    required = {
        "disjoint_upstream_authority_content_sha256": metadata[
            "disjoint_upstream_authority_content_sha256"
        ],
        "frozen_submap_plan_content_sha256": metadata[
            "frozen_submap_plan_content_sha256"
        ],
        "selected_chart_names_in_order_sha256": metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "comparison_domain_content_sha256": metadata[
            "upstream_optimizer_comparison_domain_content_sha256"
        ],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{label} alignment {key} differs from seam authority")
    with np.load(charts_path, allow_pickle=False) as data:
        required_arrays = (
            "pts",
            "scale_factor",
            "chart_names",
            "valid",
            "comparison_face_valid_stride4",
        )
        if any(name not in data.files for name in required_arrays):
            raise ValueError(f"{label} charts data lacks strict paired arrays")
        points = np.asarray(data["pts"], np.float64)
        scale = float(data["scale_factor"])
        names = np.asarray(data["chart_names"]).astype(str)
        valid = np.asarray(data["valid"], bool)
    if names.tolist() != authority.chart_names.astype(str).tolist():
        raise ValueError(f"{label} chart order differs from seam authority")
    if (
        points.ndim != 4
        or points.shape[-1] != 3
        or valid.shape != points.shape[:3]
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(f"{label} chart grid contract differs")
    height, width = valid.shape[1:]
    packed = []
    for chart in range(len(names)):
        lo, hi = map(int, authority.chart_vertex_offsets[chart : chart + 2])
        pixels = authority.sampled_vertex_pixel_indices[lo:hi]
        if np.any((pixels < 0) | (pixels >= height * width)):
            raise ValueError(f"{label} seam pixel leaves chart grid")
        yy, xx = pixels // width, pixels % width
        if not valid[chart, yy, xx].all():
            raise ValueError(f"{label} seam topology leaves common valid pixels")
        packed.append(points[chart, yy, xx] / scale)
    packed_points = np.concatenate(packed)
    if not np.isfinite(packed_points).all():
        raise ValueError(f"{label} seam geometry is nonfinite")
    return packed_points, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_seam_authority", type=Path, required=True)
    parser.add_argument("--expected_source_seam_authority_content_sha256", required=True)
    parser.add_argument("--m1_charts", type=Path, required=True)
    parser.add_argument("--m1_manifest", type=Path, required=True)
    parser.add_argument("--expected_m1_manifest_content_sha256", required=True)
    parser.add_argument("--m2_charts", type=Path, required=True)
    parser.add_argument("--m2_manifest", type=Path, required=True)
    parser.add_argument("--expected_m2_manifest_content_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite source seam evaluation")
    authority = SourceSeamCorrespondenceAuthority.load_npz(
        args.source_seam_authority
    )
    if authority.metadata.get("content_sha256") != (
        args.expected_source_seam_authority_content_sha256
    ):
        raise ValueError("source seam authority differs from experiment pin")
    m1_points, m1_manifest = _load_arm(
        "M1",
        args.m1_charts,
        args.m1_manifest,
        args.expected_m1_manifest_content_sha256,
        authority,
    )
    m2_points, m2_manifest = _load_arm(
        "M2",
        args.m2_charts,
        args.m2_manifest,
        args.expected_m2_manifest_content_sha256,
        authority,
    )
    if m1_manifest.get("alignment_code_inventory_sha256") != m2_manifest.get(
        "alignment_code_inventory_sha256"
    ):
        raise ValueError("M1/M2 alignment code inventories differ")
    if m1_manifest.get("initializer") != "dav2" or m2_manifest.get("initializer") != "moge3":
        raise ValueError("source seam evaluator requires DAV2 M1 and MoGe3 M2")
    report = evaluate_source_seam_geometry(
        authority, {"M1_DAV2": m1_points, "M2_MoGe3": m2_points}
    )
    report.update(
        {
            "source_seam_authority_file_sha256": file_sha256(
                args.source_seam_authority
            ),
            "source_seam_authority_content_sha256": authority.metadata[
                "content_sha256"
            ],
            "m1_charts_file_sha256": file_sha256(args.m1_charts),
            "m1_manifest_file_sha256": file_sha256(args.m1_manifest),
            "m1_manifest_content_sha256": m1_manifest["content_sha256"],
            "m2_charts_file_sha256": file_sha256(args.m2_charts),
            "m2_manifest_file_sha256": file_sha256(args.m2_manifest),
            "m2_manifest_content_sha256": m2_manifest["content_sha256"],
            "alignment_code_inventory_sha256": m1_manifest[
                "alignment_code_inventory_sha256"
            ],
        }
    )
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_file_sha256": file_sha256(args.output),
                "content_sha256": report["content_sha256"],
                "m0_authority_reachable_and_geometrically_valid": report[
                    "m0_authority_reachable_and_geometrically_valid"
                ],
                "formal_arm_gate_eligible": report["formal_arm_gate_eligible"],
                "M1": report["arms"]["M1_DAV2"]["formal_decision"],
                "M2": report["arms"]["M2_MoGe3"]["formal_decision"],
                "M1_conditional_reachable_edges": report["arms"]["M1_DAV2"][
                    "conditional_reachable_edge_decision"
                ],
                "M2_conditional_reachable_edges": report["arms"]["M2_MoGe3"][
                    "conditional_reachable_edge_decision"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
