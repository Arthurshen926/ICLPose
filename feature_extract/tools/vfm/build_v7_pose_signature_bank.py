"""Build or merge V7 pose-aware maplet sufficient-statistic prototypes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _retrieve,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)
from feature_extract.vfm.localization_v7.pose_signature import (
    PoseSignatureBank,
    file_sha256,
    pose_signature_from_retrieval,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping_manifest", default="")
    parser.add_argument("--mapping_pose_file", default="")
    parser.add_argument("--retrieval_regions", default="")
    parser.add_argument("--spatial_maplets", default="")
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument("--probability_calibration", default="")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=576)
    parser.add_argument("--aggregation", default="topq_nms")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--maximum_records", type=int, default=0)
    parser.add_argument("--merge_inputs", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _contract_metadata() -> dict[str, bool]:
    return {
        "stores_mapping_rgb": False,
        "stores_mapping_image_ids": False,
        "stores_mapping_image_paths": False,
        "stores_observation_descriptors": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_mapping_pose_statistics": True,
    }


def _merge(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("--merge_inputs requires at least one shard")
    banks = [PoseSignatureBank.load_npz(path) for path in paths]
    first = banks[0]
    for bank in banks[1:]:
        if not np.array_equal(bank.maplet_ids, first.maplet_ids):
            raise ValueError("V7 pose-signature shards use different maplet bases")
        for key in (
            "mapping_manifest_sha256",
            "mapping_pose_file_sha256",
            "retrieval_regions_sha256",
            "spatial_maplets_sha256",
            "surface_mapper_checkpoint_sha256",
            "probability_calibration_sha256",
        ):
            if bank.metadata.get(key) != first.metadata.get(key):
                raise ValueError(f"V7 pose-signature shard lineage differs at {key}")
    shard_indices = sorted(int(bank.metadata["shard_index"]) for bank in banks)
    shard_count = int(first.metadata["shard_count"])
    if shard_indices != list(range(shard_count)):
        raise ValueError(
            f"incomplete pose-signature shards: {shard_indices}/{shard_count}"
        )
    order = np.concatenate(
        [
            np.stack(
                [
                    np.full(bank.poses_w2c.shape[0], int(bank.metadata["shard_index"])),
                    np.arange(bank.poses_w2c.shape[0]),
                ],
                axis=1,
            )
            for bank in banks
        ]
    )
    # Restore the original round-robin manifest order without storing image IDs.
    global_index = order[:, 0] + shard_count * order[:, 1]
    permutation = np.argsort(global_index, kind="mergesort")
    metadata = dict(first.metadata)
    metadata.pop("shard_index", None)
    metadata.pop("shard_count", None)
    metadata["prototype_count"] = int(sum(bank.poses_w2c.shape[0] for bank in banks))
    metadata["merged_shard_count"] = shard_count
    metadata["prototype_order"] = "sorted_mapping_image_id"
    metadata["mapping_trajectory_ids"] = sorted(
        {
            str(value)
            for bank in banks
            for value in bank.metadata.get("mapping_trajectory_ids", [])
        }
    )
    def concatenate(field: str) -> np.ndarray:
        return np.concatenate([getattr(bank, field) for bank in banks], axis=0)[
            permutation
        ]

    PoseSignatureBank(
        maplet_ids=first.maplet_ids,
        poses_w2c=concatenate("poses_w2c"),
        identity=concatenate("identity"),
        layout_mean_xy=concatenate("layout_mean_xy"),
        layout_extent_xy=concatenate("layout_extent_xy"),
        layout_variance_xy=concatenate("layout_variance_xy"),
        layout_mass=concatenate("layout_mass"),
        metadata=metadata,
    ).save_npz(output)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_npz)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite V7 pose-signature bank")
    if args.merge_inputs:
        _merge([Path(value) for value in args.merge_inputs], output)
        return
    required = {
        "mapping_manifest": args.mapping_manifest,
        "mapping_pose_file": args.mapping_pose_file,
        "retrieval_regions": args.retrieval_regions,
        "spatial_maplets": args.spatial_maplets,
        "surface_mapper_checkpoint": args.surface_mapper_checkpoint,
        "probability_calibration": args.probability_calibration,
    }
    missing = sorted(key for key, value in required.items() if not str(value))
    if missing:
        raise ValueError(f"missing V7 build inputs: {missing}")
    shard_count = max(int(args.shard_count), 1)
    shard_index = int(args.shard_index)
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")

    manifest_path = Path(args.mapping_manifest)
    pose_path = Path(args.mapping_pose_file)
    identity_path = Path(args.retrieval_regions)
    spatial_path = Path(args.spatial_maplets)
    mapper_path = Path(args.surface_mapper_checkpoint)
    calibration_path = Path(args.probability_calibration)
    manifest = TokenBankManifest.from_json(manifest_path)
    manifest.validate(verify_checksums=False)
    pose_by_id = {
        str(record.image_id): np.asarray(record.pose_w2c, dtype=np.float64)
        for record in parse_cambridge_pose_file(pose_path)
    }
    records = sorted(
        [record for record in manifest.records if record.image_id in pose_by_id],
        key=lambda record: record.image_id,
    )
    records = records[shard_index::shard_count]
    if int(args.maximum_records) > 0:
        records = records[: int(args.maximum_records)]
    if not records:
        raise ValueError("V7 pose-signature shard contains no mapping records")

    identity_bank = SurfaceRetrievalMapletBank.load_npz(identity_path)
    spatial_bank = SurfaceRetrievalMapletBank.load_npz(spatial_path)
    mapper, mapper_metadata = load_surface_maplet_mapper(
        mapper_path, device=str(args.device)
    )
    calibration = V6ProbabilityCalibration.load_json(calibration_path)
    identity_values = []
    mean_values = []
    extent_values = []
    variance_values = []
    mass_values = []
    poses = []
    for index, record in enumerate(records):
        with np.load(record.token_path, allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        retrieval = _retrieve(
            raw,
            identity_bank,
            spatial_bank,
            mapper,
            mapper_metadata,
            calibration,
            (int(args.image_width), int(args.image_height)),
            str(args.aggregation),
        )
        signature = pose_signature_from_retrieval(
            retrieval,
            identity_bank.maplet_ids,
            (int(args.image_width), int(args.image_height)),
        )
        identity_values.append(signature.identity)
        mean_values.append(signature.layout_mean_xy)
        extent_values.append(signature.layout_extent_xy)
        variance_values.append(signature.layout_variance_xy)
        mass_values.append(signature.layout_mass)
        poses.append(pose_by_id[record.image_id])
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "shard": shard_index,
                        "processed": index + 1,
                        "total": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    metadata: dict[str, object] = {
        "artifact_type": "v7_pose_signature_bank",
        "representation": "pose_aware_maplet_sufficient_statistics",
        "prototype_count": len(records),
        "maplet_count": int(identity_bank.maplet_ids.size),
        "image_size_wh": [int(args.image_width), int(args.image_height)],
        "scene_evidence_aggregation": str(args.aggregation),
        "prototype_order": "round_robin_shard_of_sorted_mapping_image_id",
        "mapping_trajectory_ids": sorted(
            {record.image_id.split("/", 1)[0] for record in records}
        ),
        "mapping_manifest_sha256": file_sha256(manifest_path),
        "mapping_pose_file_sha256": file_sha256(pose_path),
        "retrieval_regions_sha256": file_sha256(identity_path),
        "spatial_maplets_sha256": file_sha256(spatial_path),
        "surface_mapper_checkpoint_sha256": file_sha256(mapper_path),
        "probability_calibration_sha256": file_sha256(calibration_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        **_contract_metadata(),
    }
    PoseSignatureBank(
        maplet_ids=identity_bank.maplet_ids,
        poses_w2c=np.stack(poses),
        identity=np.stack(identity_values),
        layout_mean_xy=np.stack(mean_values),
        layout_extent_xy=np.stack(extent_values),
        layout_variance_xy=np.stack(variance_values),
        layout_mass=np.stack(mass_values),
        metadata=metadata,
    ).save_npz(output)


if __name__ == "__main__":
    main()
