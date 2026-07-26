"""Apply conservative feature-only refinement to frozen 2DGS pose results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_feature_refinement import (
    SurfaceFeatureRefinementConfig,
    refine_surface_pose_featuremetric,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_image_root", required=True)
    parser.add_argument("--query_camera_manifest", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--descriptor_bank", required=True)
    parser.add_argument("--input_results_jsonl", required=True)
    parser.add_argument("--output_results_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _read_results(path: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = dict(json.loads(line))
        image_id = str(row["image_id"])
        if image_id in output:
            raise ValueError(f"duplicate input result: {image_id}")
        output[image_id] = row
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    image_ids = [record.image_id for record in manifest.records]
    input_rows = _read_results(Path(args.input_results_jsonl))
    missing = set(image_ids) - set(input_rows)
    if missing:
        raise ValueError(f"input results miss queries: {sorted(missing)[:3]}")
    cameras, intrinsic_audit = _load_query_camera_manifest(
        Path(args.query_camera_manifest)
    )
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    if "signed_toward" not in str(
        dict(anchors.metadata or {}).get("normal_orientation", "")
    ):
        raise ValueError("refinement requires oriented 2DGS anchor normals")
    descriptor_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.descriptor_bank)
    )
    config = SurfaceFeatureRefinementConfig()
    alike = AlikeDenseObservationExtractor(device=str(args.device))
    output_path = Path(args.output_results_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
    success_count = 0
    failure_counts: dict[str, int] = {}
    with output_path.open("w") as handle:
        for index, image_id in enumerate(image_ids):
            row = dict(input_rows[image_id])
            if bool(row.get("success")) and row.get("pose_w2c") is not None:
                result = refine_surface_pose_featuremetric(
                    initial_pose_w2c=np.asarray(
                        row["pose_w2c"], dtype=np.float64
                    ),
                    image_path=Path(args.query_image_root) / image_id,
                    image_id=image_id,
                    camera=cameras[image_id],
                    anchors=anchors,
                    descriptor_bank=descriptor_bank,
                    alike=alike,
                    config=config,
                )
                success_count += int(result.success)
                accepted_count += int(result.accepted)
                reason = result.failure_reason or "accepted"
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
                if result.accepted:
                    row["pose_w2c"] = result.pose_w2c.tolist()
                diagnostics = dict(row.get("diagnostics") or {})
                diagnostics["surface_feature_refinement"] = {
                    key: value
                    for key, value in asdict(result).items()
                    if key != "pose_w2c"
                }
                diagnostics["surface_feature_refinement"][
                    "uses_ground_truth"
                ] = False
                row["diagnostics"] = diagnostics
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "query": index + 1,
                        "query_count": len(image_ids),
                        "image_id": image_id,
                        "accepted_count": accepted_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "stage": "refine_surface_pose_results",
        "query_count": len(image_ids),
        "refinement_success_count": success_count,
        "accepted_count": accepted_count,
        "outcome_counts": failure_counts,
        "config": asdict(config),
        "intrinsic_audit": intrinsic_audit,
        "production_contract": {
            "query_rgb_only": True,
            "stores_mapping_rgb": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_mapping_pose_at_inference": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
        "output_results_jsonl": str(output_path),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
