"""Anchor-free RADIO/maplet localization by continuous 2DGS field alignment.

The coarse pose input must be produced by a RADIO-final maplet stage. This
program has no anchor-map, local-descriptor-bank, pairwise matcher, or PnP
input. ALIKE descriptor tensors produced by its third-party detector API are
discarded; only detection coordinates and scalar scores are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
    _load_raw_final,
)
from feature_extract.vfm.localization.continuous_surface_alignment import (
    ContinuousSurfaceAlignmentConfig,
    align_surface_feature_field,
    build_detector_heatmap,
)
from feature_extract.vfm.localization.alike_detector_only import AlikeDetectorOnly
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
    retrieve_surface_maplets,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_image_root", required=True)
    parser.add_argument("--query_camera_manifest", required=True)
    parser.add_argument("--coarse_pose_jsonl", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--metric_mapper_checkpoint", required=True)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--surface_feature_field", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum_maplets", type=int, default=12)
    parser.add_argument("--maximum_samples", type=int, default=2048)
    parser.add_argument("--detector_top_k", type=int, default=1024)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _coarse_poses(path: Path) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not bool(record.get("success", True)) or record.get("pose_w2c") is None:
            continue
        contract = dict(record.get("production_contract") or {})
        if contract.get("coarse_pose_source") not in {
            "radio_final_maplet_retrieval",
            "radio_final_maplet_pose_mixture",
        }:
            raise ValueError(
                "coarse pose must declare a RADIO-final maplet retrieval source"
            )
        for forbidden in (
            "uses_mapping_rgb_at_inference",
            "uses_pairwise_image_matching",
            "uses_radio_intermediate",
            "uses_sfm_points",
            "uses_sfm_tracks",
        ):
            if bool(contract.get(forbidden, False)):
                raise ValueError(f"coarse pose violates production contract: {forbidden}")
        output[str(record["image_id"])] = np.asarray(
            record["pose_w2c"], dtype=np.float64
        ).reshape(4, 4)
    return output


def _retrieved_maplets(
    raw: np.ndarray,
    mapped: np.ndarray,
    maplets: SurfaceRetrievalMapletBank,
    region_config: RadioFinalRegionConfig,
    maximum_maplets: int,
) -> tuple[np.ndarray, dict[str, object]]:
    _indices, xy = select_spatially_balanced_radio_final_regions(raw)
    descriptors = encode_radio_final_regions(mapped, xy, region_config)
    return retrieve_surface_maplets(
        descriptors,
        maplets,
        maximum_maplets=int(maximum_maplets),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite localization outputs")
    field = SurfaceFeatureField.load_npz(Path(args.surface_feature_field))
    maplets = SurfaceRetrievalMapletBank.load_npz(Path(args.maplets))
    coarse = _coarse_poses(Path(args.coarse_pose_jsonl))
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.query_camera_manifest)
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    metric = load_surface_metric_feature_mapper(
        Path(args.metric_mapper_checkpoint), device=str(args.device)
    )
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    records = [
        record
        for record in manifest.records
        if record.image_id in coarse and record.image_id in camera_by_image
    ]
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    detector = AlikeDetectorOnly(
        device=str(args.device), matcha_repo=Path("/root/matcha"), model_name="alike-t"
    )
    region_config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))),
    )
    alignment_config = ContinuousSurfaceAlignmentConfig(
        maximum_maplets=int(args.maximum_maplets),
        maximum_samples=int(args.maximum_samples),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    with output_path.open("w") as handle:
        for record in records:
            camera = camera_by_image[record.image_id]
            raw = _load_raw_final(Path(record.token_path), "radio_final")
            mapped_coarse = mapper.project(raw).measurement_context
            mapped_metric = metric.project_map(mapped_coarse)
            selected_ids, retrieval = _retrieved_maplets(
                raw,
                (
                    mapped_metric
                    if maplets.metadata.get("feature_space")
                    == "surface_metric_radio_final"
                    else mapped_coarse
                ),
                maplets,
                region_config,
                int(args.maximum_maplets),
            )
            detections = detector.detect(
                Path(args.query_image_root) / record.image_id,
                image_width=int(camera.width),
                image_height=int(camera.height),
                top_k=int(args.detector_top_k),
                candidate_top_k=max(4096, int(args.detector_top_k)),
                nms_radius_px=2.0,
                grid_rows=8,
                grid_cols=8,
            )
            detector_heatmap = build_detector_heatmap(
                detections.xy,
                detections.scores,
                image_width=int(camera.width),
                image_height=int(camera.height),
                output_width=256,
                output_height=144,
            )
            result = align_surface_feature_field(
                field,
                mapped_metric,
                coarse[record.image_id],
                camera,
                selected_ids,
                config=alignment_config,
                device=str(args.device),
                detector_heatmap=detector_heatmap,
            )
            row = {
                "image_id": record.image_id,
                "success": bool(result.converged),
                "pose_w2c": result.pose_w2c.reshape(-1).tolist(),
                "coarse_pose_w2c": coarse[record.image_id].reshape(-1).tolist(),
                "selected_maplet_ids": selected_ids.tolist(),
                "initial_score": result.initial_score,
                "final_score": result.final_score,
                "sample_count": result.sample_count,
                "retrieval": retrieval,
                "diagnostics": result.diagnostics,
                "production_contract": {
                    "coarse_representation": "radio_final_maplet_pose",
                    "coarse_pose_source": "radio_final_maplet_pose_mixture",
                    "fine_pose": "continuous_2dgs_surface_feature_alignment",
                    "computes_alike_descriptor_map": False,
                    "uses_alike_descriptors": False,
                    "uses_stable_anchor_identity": False,
                    "uses_point_correspondence_pnp": False,
                    "uses_mapping_rgb_at_inference": False,
                    "uses_pairwise_image_matching": False,
                    "uses_radio_intermediate": False,
                    "uses_sfm_points": False,
                    "uses_sfm_tracks": False,
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            success_count += int(result.converged)
    summary = {
        "stage": "localize_anchor_free_2dgs_surface_feature_field",
        "query_count": len(records),
        "success_count": success_count,
        "output_jsonl": str(output_path),
        "camera_audit": camera_audit,
        "production_contract": {
            "map": "compact_radio_final_maplet_mixtures_and_clean_2dgs_surface_feature_field",
            "alike_role": "detector_coordinates_and_weights_only",
            "computes_alike_descriptor_map": False,
            "uses_alike_descriptors": False,
            "uses_stable_anchor_identity": False,
            "uses_point_correspondence_pnp": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_pairwise_image_matching": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
