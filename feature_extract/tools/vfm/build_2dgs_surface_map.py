"""Build the track-free VFM maplet + stable 2DGS anchor representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import _load_camera_by_image
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    SurfaceMapletBuildConfig,
    build_track_free_surface_map,
    load_2dgs_primitive_quality,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    Vfm2DgsAnchorMap,
    Vfm2DgsObservationBank,
)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("integer tuple cannot be empty")
    return values


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("float tuple cannot be empty")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SfM-track-free RADIO-final/2DGS surface localization map"
    )
    parser.add_argument("--surface_elements", required=True)
    parser.add_argument("--region_map", required=True)
    parser.add_argument("--observation_bank", required=True)
    parser.add_argument("--radio_final_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument(
        "--require_camera_for_every_view", action="store_true",
    )
    parser.add_argument("--gaussian_ply", default="")
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--matcha_joint_checkpoint", default="")
    parser.add_argument(
        "--surface_maplet_mapper_checkpoint",
        default="",
        help="Production mapper trained from 2DGS surface-maplet cross-view identity.",
    )
    parser.add_argument("--mapper_device", default="cuda")
    parser.add_argument("--require_full_map_mapper", action="store_true")
    parser.add_argument("--pool_sizes", default="1,3,5,9")
    parser.add_argument("--pool_weights", default="0.4,0.3,0.2,0.1")
    parser.add_argument("--global_context_weight", type=float, default=0.0)
    parser.add_argument("--assignment_neighbor_count", type=int, default=8)
    parser.add_argument("--assignment_max_center_distance", type=float, default=1.0)
    parser.add_argument("--assignment_min_surface_iou", type=float, default=0.05)
    parser.add_argument("--min_anchor_views", type=int, default=2)
    parser.add_argument("--min_anchors_per_maplet", type=int, default=4)
    parser.add_argument("--max_anchors_per_maplet", type=int, default=64)
    parser.add_argument("--min_anchor_opacity", type=float, default=0.05)
    parser.add_argument("--min_geometry_confidence", type=float, default=0.5)
    parser.add_argument("--allowed_primitive_classes", default="0")
    parser.add_argument("--allow_all_primitive_classes", action="store_true")
    parser.add_argument("--min_anchor_normal_cosine", type=float, default=0.5)
    parser.add_argument("--min_anchor_separation", type=float, default=0.02)
    parser.add_argument("--max_anchor_scale", type=float, default=0.0)
    parser.add_argument("--observation_min_element_weight", type=float, default=0.01)
    parser.add_argument("--normal_variance_scale", type=float, default=0.02)
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--output_anchors", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def _validate_final_only_manifest(manifest: TokenBankManifest, layer_name: str) -> None:
    if "intermediate" in str(layer_name).lower():
        raise ValueError("RADIO intermediate is forbidden by the 2DGS production protocol")
    found = False
    for record in manifest.records:
        for layer in record.layers:
            if layer.name != layer_name:
                continue
            found = True
            if str(layer.layer).lower() != "final":
                raise ValueError("the requested production VFM layer is not RADIO final")
    if not found:
        raise ValueError(f"RADIO-final layer {layer_name!r} is absent from the token manifest")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = TokenBankManifest.from_json(Path(args.radio_final_manifest))
    manifest.validate(verify_checksums=False)
    _validate_final_only_manifest(manifest, str(args.radio_final_layer))
    mapper_paths = [
        value
        for value in (str(args.matcha_joint_checkpoint), str(args.surface_maplet_mapper_checkpoint))
        if value
    ]
    if len(mapper_paths) > 1:
        raise ValueError("matcha and surface-maplet mapper checkpoints are mutually exclusive")
    if bool(args.require_full_map_mapper) and not mapper_paths:
        raise ValueError("--require_full_map_mapper requires a full-map mapper checkpoint")
    feature_mapper = None
    mapper_metadata: dict[str, object] = {}
    if str(args.matcha_joint_checkpoint):
        mapper_run = load_matcha_joint_model(
            Path(args.matcha_joint_checkpoint),
            device=str(args.mapper_device),
        )
        feature_mapper = JointFeatureMapper(mapper_run.model, device=str(args.mapper_device))
        mapper_metadata = {"mapper_type": "legacy_matcha_joint"}
    elif str(args.surface_maplet_mapper_checkpoint):
        feature_mapper, checkpoint_metadata = load_surface_maplet_mapper(
            Path(args.surface_maplet_mapper_checkpoint),
            device=str(args.mapper_device),
        )
        if bool(checkpoint_metadata.get("uses_radio_intermediate", False)):
            raise ValueError("surface-maplet mapper checkpoint illegally uses RADIO intermediate")
        if bool(checkpoint_metadata.get("uses_sfm_tracks", False)):
            raise ValueError("surface-maplet mapper checkpoint illegally uses SfM tracks")
        mapper_metadata = {
            "mapper_type": "surface_maplet",
            "surface_maplet_mapper_metadata": checkpoint_metadata,
        }
    surface_elements = SurfaceElementMap.load_npz(Path(args.surface_elements))
    region_map = Vfm2DgsAnchorMap.load_npz(Path(args.region_map))
    observation_bank = Vfm2DgsObservationBank.load_npz(Path(args.observation_bank))

    record_by_image = {record.image_id: record for record in manifest.records}
    required_images = sorted(set(observation_bank.image_ids))
    missing = [image_id for image_id in required_images if image_id not in record_by_image]
    if missing:
        raise ValueError(f"observation image is absent from RADIO-final manifest: {missing[0]}")
    feature_maps: dict[str, np.ndarray] = {}
    for image_id in required_images:
        record = record_by_image[image_id]
        with np.load(record.token_path) as data:
            if str(args.radio_final_layer) not in data:
                raise ValueError(
                    f"RADIO-final layer {args.radio_final_layer!r} is absent from {record.token_path}"
                )
            feature = np.asarray(data[str(args.radio_final_layer)], dtype=np.float32)
        if feature.ndim == 4 and int(feature.shape[0]) == 1:
            feature = feature[0]
        if feature_mapper is not None:
            feature = feature_mapper.project(feature).coarse_descriptors
        feature_maps[image_id] = feature

    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.reference_pose_file))
    }
    camera_by_image = _load_camera_by_image(str(args.camera_model_dir))
    missing_camera_ids = sorted(set(required_images) - set(camera_by_image))
    if bool(args.require_camera_for_every_view) and missing_camera_ids:
        raise ValueError(
            f"camera model is absent for surface observation: {missing_camera_ids[0]}"
        )
    primitive_quality = (
        load_2dgs_primitive_quality(Path(args.gaussian_ply))
        if str(args.gaussian_ply)
        else None
    )
    allowed_classes = (
        None
        if bool(args.allow_all_primitive_classes)
        else _parse_int_tuple(args.allowed_primitive_classes)
    )
    build_config = SurfaceMapletBuildConfig(
        assignment_neighbor_count=int(args.assignment_neighbor_count),
        assignment_max_center_distance=float(args.assignment_max_center_distance),
        assignment_min_surface_iou=float(args.assignment_min_surface_iou),
        min_anchor_views=int(args.min_anchor_views),
        min_anchors_per_maplet=int(args.min_anchors_per_maplet),
        max_anchors_per_maplet=int(args.max_anchors_per_maplet),
        min_anchor_opacity=float(args.min_anchor_opacity),
        min_geometry_confidence=float(args.min_geometry_confidence),
        allowed_primitive_classes=allowed_classes,
        min_anchor_normal_cosine=float(args.min_anchor_normal_cosine),
        min_anchor_separation=float(args.min_anchor_separation),
        max_anchor_scale=float(args.max_anchor_scale),
        observation_min_element_weight=float(args.observation_min_element_weight),
        normal_variance_scale=float(args.normal_variance_scale),
    )
    region_config = RadioFinalRegionConfig(
        pool_sizes=_parse_int_tuple(args.pool_sizes),
        pool_weights=_parse_float_tuple(args.pool_weights),
        global_context_weight=float(args.global_context_weight),
    )
    maplets, anchors, summary = build_track_free_surface_map(
        surface_elements=surface_elements,
        region_map=region_map,
        observation_bank=observation_bank,
        radio_final_feature_maps=feature_maps,
        pose_w2c_by_image=pose_by_image,
        camera_by_image=camera_by_image,
        primitive_quality=primitive_quality,
        build_config=build_config,
        region_config=region_config,
        descriptor_space_metadata={
            "full_map_mapper_applied": bool(feature_mapper is not None),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "surface_maplet_mapper_checkpoint": str(args.surface_maplet_mapper_checkpoint),
            "mapper_device": str(args.mapper_device),
            **mapper_metadata,
        },
    )
    maplets.save_npz(Path(args.output_maplets))
    anchors.save_npz(Path(args.output_anchors))
    summary.update(
        {
            "inputs": {
                "surface_elements": str(args.surface_elements),
                "region_map": str(args.region_map),
                "observation_bank": str(args.observation_bank),
                "radio_final_manifest": str(args.radio_final_manifest),
                "reference_pose_file": str(args.reference_pose_file),
                "camera_model_dir": str(args.camera_model_dir),
                "require_camera_for_every_view": bool(args.require_camera_for_every_view),
                "missing_camera_view_count": len(missing_camera_ids),
                "gaussian_ply": str(args.gaussian_ply),
                "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
                "surface_maplet_mapper_checkpoint": str(args.surface_maplet_mapper_checkpoint),
                "full_map_mapper_applied": bool(feature_mapper is not None),
                "mapper_device": str(args.mapper_device),
                **mapper_metadata,
            },
            "outputs": {
                "maplets": str(args.output_maplets),
                "anchors": str(args.output_anchors),
                "summary": str(args.summary_json),
            },
            "production_contract": {
                "map_representation": "2dgs_surface",
                "coarse_identity": "surface_maplet_id",
                "fine_identity": "stable_surface_anchor_id",
                "vfm_layer": "radio_final",
                "uses_radio_intermediate": False,
                "uses_sfm_points": False,
                "uses_sfm_tracks": False,
            },
        }
    )
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
