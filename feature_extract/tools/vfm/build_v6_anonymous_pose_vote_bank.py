"""Build image-free appearance-mode pose statistics for V6 coarse voting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
    _rotation_angle_deg,
    _weighted_rotation_mean,
    retrieval_descriptor_sha256,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction_maplets", required=True)
    parser.add_argument("--retrieval_maplets", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_pose_modes", type=int, default=2)
    parser.add_argument("--translation_scale_m", type=float, default=0.75)
    parser.add_argument("--rotation_scale_deg", type=float, default=12.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _pose_clusters(
    centers: np.ndarray,
    rotations: np.ndarray,
    weights: np.ndarray,
    maximum_modes: int,
    translation_scale_m: float,
    rotation_scale_deg: float,
) -> list[tuple[np.ndarray, np.ndarray, float, float, float]]:
    count = int(centers.shape[0])
    cluster_count = min(max(int(maximum_modes), 1), count)
    translation_distance = np.linalg.norm(
        centers[:, None] - centers[None], axis=2
    )
    rotation_distance = _rotation_angle_deg(
        rotations[:, None], rotations[None]
    )
    distance = (
        translation_distance / max(float(translation_scale_m), 1e-4)
    ) ** 2 + (
        rotation_distance / max(float(rotation_scale_deg), 1e-4)
    ) ** 2
    chosen = [int(np.argmax(weights))]
    while len(chosen) < cluster_count:
        nearest = np.min(distance[:, chosen], axis=1)
        nearest[chosen] = -1.0
        chosen.append(int(np.argmax(nearest * weights)))
    center_modes = centers[chosen].copy()
    rotation_modes = rotations[chosen].copy()
    labels = np.zeros((count,), dtype=np.int64)
    for _iteration in range(6):
        translation = np.linalg.norm(
            centers[:, None] - center_modes[None], axis=2
        ) / max(float(translation_scale_m), 1e-4)
        rotation = _rotation_angle_deg(
            rotations[:, None], rotation_modes[None]
        ) / max(float(rotation_scale_deg), 1e-4)
        labels = np.argmin(translation * translation + rotation * rotation, axis=1)
        for mode in range(cluster_count):
            local = labels == mode
            if not np.any(local):
                continue
            local_weights = weights[local]
            local_weights /= max(float(np.sum(local_weights)), 1e-8)
            center_modes[mode] = np.sum(
                centers[local] * local_weights[:, None], axis=0
            )
            rotation_modes[mode] = _weighted_rotation_mean(
                rotations[local], local_weights
            )
    output = []
    total = max(float(np.sum(weights)), 1e-8)
    for mode in range(cluster_count):
        local = labels == mode
        if not np.any(local):
            continue
        local_weights = weights[local]
        local_weights /= max(float(np.sum(local_weights)), 1e-8)
        translation_sigma = np.sqrt(
            np.sum(
                local_weights
                * np.sum(
                    (centers[local] - center_modes[mode]) ** 2, axis=1
                )
            )
        )
        rotation_sigma = np.sqrt(
            np.sum(
                local_weights
                * _rotation_angle_deg(
                    rotations[local], rotation_modes[mode][None]
                )
                ** 2
            )
        )
        output.append(
            (
                center_modes[mode],
                rotation_modes[mode],
                float(np.sum(weights[local]) / total),
                float(translation_sigma),
                float(rotation_sigma),
            )
        )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_bank)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite anonymous pose-vote bank")
    source = VfmSurfaceMapletBank.load_npz(Path(args.construction_maplets))
    retrieval = SurfaceRetrievalMapletBank.load_npz(
        Path(args.retrieval_maplets)
    )
    if (
        not np.array_equal(source.maplet_ids, retrieval.maplet_ids)
        or not np.allclose(source.centers, retrieval.centers, atol=1e-5)
    ):
        raise ValueError("construction/retrieval maplet geometry differs")
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    component_ids = np.repeat(
        retrieval.maplet_ids, np.diff(retrieval.descriptor_offsets)
    )
    offsets = [0]
    output_centers = []
    output_rotations = []
    output_weights = []
    output_translation_sigma = []
    output_rotation_sigma = []
    missing_observations = 0
    for maplet_row in range(len(source)):
        view_begin, view_end = (
            int(source.view_offsets[maplet_row]),
            int(source.view_offsets[maplet_row + 1]),
        )
        descriptor_begin, descriptor_end = (
            int(retrieval.descriptor_offsets[maplet_row]),
            int(retrieval.descriptor_offsets[maplet_row + 1]),
        )
        descriptors = source.view_descriptors[view_begin:view_end]
        quality = np.maximum(
            source.view_quality_scores[view_begin:view_end], 1e-4
        ).astype(np.float64)
        valid = np.asarray(
            [
                image_id in pose_by_image
                for image_id in source.view_image_ids[view_begin:view_end]
            ],
            dtype=bool,
        )
        descriptors = descriptors[valid]
        quality = quality[valid]
        image_ids = [
            image_id
            for image_id, keep in zip(
                source.view_image_ids[view_begin:view_end], valid.tolist()
            )
            if keep
        ]
        if not image_ids:
            raise ValueError("retrieval component has no pose-bearing support")
        component_descriptors = retrieval.descriptors[
            descriptor_begin:descriptor_end
        ]
        assignments = np.argmax(
            descriptors @ component_descriptors.T, axis=1
        )
        centers = np.asarray(
            [
                -pose_by_image[image_id][:3, :3].T
                @ pose_by_image[image_id][:3, 3]
                for image_id in image_ids
            ],
            dtype=np.float64,
        )
        rotations = np.asarray(
            [pose_by_image[image_id][:3, :3] for image_id in image_ids],
            dtype=np.float64,
        )
        for local_component in range(component_descriptors.shape[0]):
            members = assignments == local_component
            if not np.any(members):
                # This can occur only through a numerical tie after rebuilding
                # the stripped descriptor bank.  Use its nearest observation.
                nearest = int(
                    np.argmax(descriptors @ component_descriptors[local_component])
                )
                members[nearest] = True
                missing_observations += 1
            modes = _pose_clusters(
                centers[members],
                rotations[members],
                quality[members].copy(),
                int(args.maximum_pose_modes),
                float(args.translation_scale_m),
                float(args.rotation_scale_deg),
            )
            for center, rotation, weight, t_sigma, r_sigma in modes:
                output_centers.append(center)
                output_rotations.append(rotation)
                output_weights.append(weight)
                output_translation_sigma.append(t_sigma)
                output_rotation_sigma.append(r_sigma)
            offsets.append(len(output_centers))
    bank = AnonymousMapletPoseVoteBank(
        component_maplet_ids=component_ids,
        vote_offsets=np.asarray(offsets, dtype=np.int64),
        camera_centers=np.asarray(output_centers, dtype=np.float64),
        rotations_w2c=np.asarray(output_rotations, dtype=np.float64),
        vote_weights=np.asarray(output_weights, dtype=np.float32),
        translation_sigma_m=np.asarray(
            output_translation_sigma, dtype=np.float32
        ),
        rotation_sigma_deg=np.asarray(
            output_rotation_sigma, dtype=np.float32
        ),
        component_descriptor_sha256=retrieval_descriptor_sha256(retrieval),
        metadata={
            "artifact_type": "v6_anonymous_maplet_pose_vote_bank",
            "representation": "maplet_component_pose_sufficient_statistics",
            "maximum_pose_modes_per_component": int(args.maximum_pose_modes),
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "stores_observation_descriptors": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
        },
    )
    bank.save_npz(output_path)
    report = {
        "stage": "build_v6_anonymous_maplet_pose_vote_bank",
        "component_count": int(component_ids.size),
        "pose_vote_count": int(bank.camera_centers.shape[0]),
        "maximum_pose_modes_per_component": int(args.maximum_pose_modes),
        "component_without_direct_assignment_count": int(
            missing_observations
        ),
        "translation_sigma_m": {
            "median": float(np.median(bank.translation_sigma_m)),
            "p90": float(np.quantile(bank.translation_sigma_m, 0.9)),
        },
        "rotation_sigma_deg": {
            "median": float(np.median(bank.rotation_sigma_deg)),
            "p90": float(np.quantile(bank.rotation_sigma_deg, 0.9)),
        },
        "production_contract": dict(bank.metadata or {}),
        "output_bank": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
