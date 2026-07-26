"""Evaluate the held-out convergence basin of feature-only 2DGS pose refinement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
)
from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
    pose_w2c_from_center_rotation,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_feature_refinement import (
    SurfaceFeatureRefinementConfig,
    _pose_distance,
    refine_surface_pose_featuremetric,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap


DEFAULT_LEVELS = (
    (0.00, 0.00),
    (0.05, 0.25),
    (0.10, 0.50),
    (0.20, 1.00),
    (0.30, 2.00),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--descriptor_bank", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_images", type=int, default=32)
    parser.add_argument("--image_stride", type=int, default=1)
    parser.add_argument("--maximum_anchors", type=int, default=192)
    parser.add_argument("--search_radius_px", type=float, default=24.0)
    parser.add_argument("--minimum_measurement_llr", type=float, default=0.0)
    parser.add_argument("--minimum_peak_margin", type=float, default=0.05)
    return parser.parse_args(argv)


def _deterministic_direction(key: str, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(int(dimension),))
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


def _perturb_pose(
    pose_w2c: np.ndarray,
    *,
    image_id: str,
    level_index: int,
    translation_m: float,
    rotation_deg: float,
) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    center = -rotation.T @ pose[:3, 3]
    translation_direction = _deterministic_direction(
        f"{image_id}:translation:{level_index}",
        3,
    )
    rotation_axis = _deterministic_direction(
        f"{image_id}:rotation:{level_index}",
        3,
    )
    delta_rotation, _jacobian = cv2.Rodrigues(
        rotation_axis.reshape(3, 1) * np.radians(float(rotation_deg))
    )
    return pose_w2c_from_center_rotation(
        center + float(translation_m) * translation_direction,
        delta_rotation @ rotation,
    )


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for level in sorted({int(row["level_index"]) for row in rows}):
        selected = [row for row in rows if int(row["level_index"]) == level]
        accepted = [row for row in selected if bool(row["accepted"])]
        output[str(level)] = {
            "translation_m": selected[0]["perturbation_translation_m"],
            "rotation_deg": selected[0]["perturbation_rotation_deg"],
            "count": len(selected),
            "success_count": int(sum(bool(row["success"]) for row in selected)),
            "accepted_count": len(accepted),
            "accepted_fraction": float(len(accepted) / max(len(selected), 1)),
            "improved_both_count": int(
                sum(bool(row["improved_both"]) for row in selected)
            ),
            "worsened_any_count": int(
                sum(bool(row["worsened_any"]) for row in selected)
            ),
            "converged_4cm_1deg_count": int(
                sum(bool(row["converged_4cm_1deg"]) for row in selected)
            ),
            "final_translation_median": float(
                np.median(
                    [float(row["final_translation_error_m"]) for row in selected]
                )
            ),
            "final_rotation_median_deg": float(
                np.median(
                    [float(row["final_rotation_error_deg"]) for row in selected]
                )
            ),
        }
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    orientation = str(
        dict(anchors.metadata or {}).get("normal_orientation", "")
    )
    if not orientation.startswith("signed_toward_"):
        raise ValueError("refinement requires offline-oriented anchor normals")
    descriptor_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.descriptor_bank)
    )
    cameras, camera_audit = _load_query_camera_manifest(
        Path(args.camera_manifest)
    )
    records = sorted(
        parse_cambridge_pose_file(Path(args.mapping_pose_file)),
        key=lambda record: record.image_id,
    )
    records = [
        record
        for record in records[:: max(int(args.image_stride), 1)]
        if record.image_id in cameras
        and (Path(args.image_root) / record.image_id).is_file()
    ][: int(args.max_images)]
    if not records:
        raise ValueError("no evaluable mapping hold-out images")
    config = SurfaceFeatureRefinementConfig(
        maximum_anchors=int(args.maximum_anchors),
        search_radius_px=float(args.search_radius_px),
        minimum_measurement_llr=float(args.minimum_measurement_llr),
        minimum_peak_margin=float(args.minimum_peak_margin),
    )
    alike = AlikeDenseObservationExtractor(device=str(args.device))
    rows: list[dict[str, object]] = []
    for record in records:
        image_path = Path(args.image_root) / record.image_id
        for level_index, (translation_m, rotation_deg) in enumerate(
            DEFAULT_LEVELS
        ):
            initial = _perturb_pose(
                record.pose_w2c,
                image_id=record.image_id,
                level_index=level_index,
                translation_m=translation_m,
                rotation_deg=rotation_deg,
            )
            result = refine_surface_pose_featuremetric(
                initial_pose_w2c=initial,
                image_path=image_path,
                image_id=record.image_id,
                camera=cameras[record.image_id],
                anchors=anchors,
                descriptor_bank=descriptor_bank,
                alike=alike,
                config=config,
            )
            initial_translation, initial_rotation = _pose_distance(
                initial,
                record.pose_w2c,
            )
            final_translation, final_rotation = _pose_distance(
                result.pose_w2c,
                record.pose_w2c,
            )
            rows.append(
                {
                    "image_id": record.image_id,
                    "level_index": level_index,
                    "perturbation_translation_m": translation_m,
                    "perturbation_rotation_deg": rotation_deg,
                    "initial_translation_error_m": initial_translation,
                    "initial_rotation_error_deg": initial_rotation,
                    "final_translation_error_m": final_translation,
                    "final_rotation_error_deg": final_rotation,
                    "improved_both": (
                        final_translation < initial_translation - 1e-9
                        and final_rotation < initial_rotation - 1e-9
                    ),
                    "worsened_any": (
                        final_translation > initial_translation + 1e-9
                        or final_rotation > initial_rotation + 1e-9
                    ),
                    "converged_4cm_1deg": (
                        final_translation <= 0.04
                        and final_rotation <= 1.0
                    ),
                    **{
                        key: value
                        for key, value in asdict(result).items()
                        if key != "pose_w2c"
                    },
                }
            )
            print(json.dumps(rows[-1], sort_keys=True))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "stage": "surface_feature_refinement_basin",
        "held_out_protocol": (
            "mapping image descriptor from the evaluated image is excluded "
            "from every anchor view mixture"
        ),
        "image_count": len(records),
        "trial_count": len(rows),
        "config": asdict(config),
        "levels": _summarize(rows),
        "camera_audit": camera_audit,
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
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
