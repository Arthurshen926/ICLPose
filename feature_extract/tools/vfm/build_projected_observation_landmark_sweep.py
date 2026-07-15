"""Build several landmark representations from one full-map projection pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_projected_observation_landmark_bank import (
    resolve_descriptor_source_config,
    source_image_list_hash,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.landmark_feature_aggregation import (
    LandmarkAggregationConfig,
    LandmarkViewClusteringConfig,
    TrackPrototypeBuilder,
    build_multi_prototype_track_bank,
)
from feature_extract.vfm.localization.descriptor_space import (
    NO_VIEW_CLUSTERING_CONFIG,
    PROTOTYPE_BUILDER_VERSION,
    TRACK_IMAGE_OBSERVATION_SELECTION_V1,
    descriptor_space_manifest,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.landmark_hybrid import (
    _track_xyz_and_reprojection_stats,
    sample_projected_track_observations,
    save_landmark_index_npz,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


REPRESENTATIONS = (
    "mean",
    "normalized_mean",
    "medoid",
    "descriptor_kmeans_2",
)
PROTOTYPE_VIEW_GEOMETRY_FORMAT = "landmark_prototype_view_geometry_v1"


def parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value).split(",") if item.strip())
    unknown = sorted(set(values) - set(REPRESENTATIONS))
    if unknown:
        raise ValueError(f"unsupported landmark representations: {unknown!r}")
    if not values:
        raise ValueError("at least one landmark representation is required")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--utility_mode", default="inverse_reprojection")
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--projection_image_batch_size", type=int, default=8)
    parser.add_argument("--projection_load_workers", type=int, default=4)
    parser.add_argument("--weight_floor", type=float, default=1e-6)
    parser.add_argument("--cluster_iterations", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve_device(value: str) -> str:
    requested = torch.device(str(value))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def _representation_config(
    name: str,
    *,
    min_observations: int,
    weight_floor: float,
    cluster_iterations: int,
) -> tuple[LandmarkAggregationConfig, LandmarkViewClusteringConfig | None]:
    if str(name) == "mean":
        return (
            LandmarkAggregationConfig(
                method="mean",
                min_observations=int(min_observations),
                l2_normalize_observations=False,
                weight_floor=float(weight_floor),
            ),
            None,
        )
    if str(name) == "normalized_mean":
        return (
            LandmarkAggregationConfig(
                method="mean",
                min_observations=int(min_observations),
                l2_normalize_observations=True,
                weight_floor=float(weight_floor),
            ),
            None,
        )
    if str(name) == "medoid":
        return (
            LandmarkAggregationConfig(
                method="medoid",
                min_observations=int(min_observations),
                l2_normalize_observations=True,
                weight_floor=float(weight_floor),
            ),
            None,
        )
    if str(name) == "descriptor_kmeans_2":
        return (
            LandmarkAggregationConfig(
                method="mean",
                min_observations=int(min_observations),
                l2_normalize_observations=True,
                weight_floor=float(weight_floor),
            ),
            LandmarkViewClusteringConfig(
                method="descriptor_spherical_kmeans",
                max_prototypes_per_track=2,
                min_observations_per_prototype=2,
                iterations=int(cluster_iterations),
            ),
        )
    raise ValueError(f"unsupported representation: {name}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    representations = parse_csv(str(args.representations))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / f"projected_observations_{name}.npz" for name in representations}
    sidecar_paths = {
        name: output_dir / f"projected_observations_{name}.view_geometry.npz"
        for name in representations
        if name.startswith("descriptor_kmeans_")
    }
    existing = [
        str(path)
        for path in (*output_paths.values(), *sidecar_paths.values())
        if path.exists()
    ]
    if existing and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing landmark banks: {existing!r}")

    device = _resolve_device(str(args.device))
    start = time.time()
    observations = load_colmap_track_observations_jsonl(
        Path(args.track_observations),
        deduplicate_track_images=False,
    )
    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device)
    source_config, source_config_audit = resolve_descriptor_source_config(
        manifest=manifest,
        feature_key=str(args.feature_key),
        checkpoint_summary=dict(joint_run.summary),
        model_input_dim=int(joint_run.model.input_dim),
        override_json="",
        allow_override=False,
    )
    mapper = JointFeatureMapper(joint_run.model, device=device)
    sampled, sampling_metadata = sample_projected_track_observations(
        observations,
        manifest,
        mapper,
        feature_key=str(args.feature_key),
        missing="error",
        utility_mode=str(args.utility_mode),
        weight_floor=float(args.weight_floor),
        sample_mode=str(args.sample_mode),
        projection_image_batch_size=int(args.projection_image_batch_size),
        projection_load_workers=int(args.projection_load_workers),
    )
    projection_seconds = float(time.time() - start)
    xyz_by_track, reprojection_error_by_track = _track_xyz_and_reprojection_stats(observations)

    checkpoint_sha256 = file_sha256_short(Path(args.matcha_joint_checkpoint))
    track_sha256 = file_sha256_short(Path(args.track_observations))
    token_manifest_sha256 = file_sha256_short(Path(args.token_manifest))
    source_image_ids = {str(observation.image_id) for observation in observations}
    image_manifest_hash = source_image_list_hash(manifest, source_image_ids)
    checkpoint_builder_value = joint_run.summary.get("track_prototype_builder")
    checkpoint_builder = dict(checkpoint_builder_value) if isinstance(checkpoint_builder_value, dict) else {}

    summaries: dict[str, object] = {}
    for name in representations:
        aggregation, clustering = _representation_config(
            name,
            min_observations=int(args.min_observations),
            weight_floor=float(args.weight_floor),
            cluster_iterations=int(args.cluster_iterations),
        )
        builder = TrackPrototypeBuilder(aggregation=aggregation, normalize_final_prototypes=True)
        representation_start = time.time()
        if clustering is None:
            bank = builder.build_bank_torch(sampled, device=device)
            index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
            view_clustering = dict(NO_VIEW_CLUSTERING_CONFIG)
            unique_track_count = int(len(index))
        else:
            bank = build_multi_prototype_track_bank(
                sampled,
                aggregation=aggregation,
                clustering=clustering,
                normalize_final_prototypes=True,
            )
            index = LandmarkMapIndex.from_multi_prototype_bank(bank, xyz_by_track, reprojection_error_by_track)
            view_clustering = clustering.to_dict()
            unique_track_count = int(bank.track_count)
        build_seconds = float(time.time() - representation_start)
        prototype_builder = builder.to_dict()
        space_manifest = descriptor_space_manifest(
            checkpoint_sha256=checkpoint_sha256,
            mapper_mode="joint_full_map",
            feature_key=str(args.feature_key),
            projection_source="projected_observation_full_map",
            aggregation_method=str(aggregation.method),
            l2_normalize_observations=bool(aggregation.l2_normalize_observations),
            image_manifest_hash=image_manifest_hash,
            sfm_track_hash=track_sha256,
            descriptor_dimension=int(index.feature_dim),
            normalization_mode="row_l2_normalized_search",
            output_branch="coarse_descriptors",
            radio_model=str(source_config["radio_model"]),
            radio_model_version=str(source_config["radio_model_version"]),
            radio_intermediate_layer_index=source_config.get("radio_intermediate_layer_index"),
            preprocessing=dict(source_config["preprocessing"]),
            feature_grid_stride=int(source_config["feature_grid_stride"]),
            input_channels=int(source_config["input_channels"]),
            sampling_mode=str(args.sample_mode),
            sampling_convention="pixel_endpoint_to_token_endpoint_v1",
            sampling_align_corners=True,
            prototype_builder_version=PROTOTYPE_BUILDER_VERSION,
            prototype_builder_config=prototype_builder,
            view_clustering=view_clustering,
            maplet_bank_version="none",
            observation_selection=TRACK_IMAGE_OBSERVATION_SELECTION_V1,
        )
        view_geometry_path = sidecar_paths.get(name)
        view_geometry_sha256 = None
        view_geometry_valid_fraction = None
        if clustering is not None:
            if view_geometry_path is None:
                raise RuntimeError("multi-prototype representation has no view sidecar path")
            view_arrays = bank.view_geometry_arrays()
            if not np.array_equal(view_arrays["track_ids"], index.track_ids):
                raise RuntimeError("prototype view sidecar track rows differ from index")
            if not np.array_equal(
                view_arrays["prototype_ids"], index.prototype_ids
            ):
                raise RuntimeError("prototype view sidecar IDs differ from index")
            view_metadata = {
                "format": PROTOTYPE_VIEW_GEOMETRY_FORMAT,
                "row_count": int(len(index)),
                "descriptor_space_id": str(space_manifest["descriptor_space_id"]),
                "ray_convention": "camera_to_landmark_world",
                "cluster_assignment": str(clustering.method),
                "view_geometry_semantics": (
                    "mean observation ray and p90 angular radius inside each "
                    "descriptor prototype cluster"
                ),
                "projected_landmark_bank": str(output_paths[name]),
                "matcha_joint_checkpoint_sha256": checkpoint_sha256,
                "track_observations_sha256": track_sha256,
            }
            np.savez_compressed(
                view_geometry_path,
                **view_arrays,
                metadata_json=np.asarray(
                    json.dumps(view_metadata, sort_keys=True), dtype=np.str_
                ),
            )
            view_geometry_sha256 = file_sha256_short(view_geometry_path)
            view_geometry_valid_fraction = float(
                np.mean(view_arrays["view_geometry_valid"])
            ) if len(index) else 0.0
        metadata = {
            **sampling_metadata,
            "aggregation": aggregation.to_dict(),
            "prototype_builder": prototype_builder,
            "view_clustering": view_clustering,
            "representation_name": str(name),
            "aggregation_device": device,
            "landmark_count": int(len(index)),
            "unique_track_count": int(unique_track_count),
            "max_prototypes_per_track": int(max(1, max(view_clustering.get("max_prototypes_per_track", 1), 1))),
            "feature_dim": int(index.feature_dim),
            "projection_mode": "full_map_projected_observations",
            "feature_key": str(args.feature_key),
            "mapper_class": "JointFeatureMapper",
            "mapper_config_hash": checkpoint_sha256,
            "normalization_mode": "row_l2_normalized_search",
            "track_observations": str(args.track_observations),
            "track_observations_sha256": track_sha256,
            "token_manifest": str(args.token_manifest),
            "token_manifest_sha256": token_manifest_sha256,
            "source_image_list_hash": image_manifest_hash,
            "source_image_count": int(len(source_image_ids)),
            "descriptor_source_config": source_config,
            "descriptor_source_config_audit": source_config_audit,
            "checkpoint_track_prototype_builder": checkpoint_builder,
            "prototype_builder_override": bool(checkpoint_builder and checkpoint_builder != prototype_builder),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "matcha_joint_checkpoint_sha256": checkpoint_sha256,
            "descriptor_space_manifest": space_manifest,
            "descriptor_space_id": str(space_manifest["descriptor_space_id"]),
            "descriptor_dimension": int(index.feature_dim),
            "observation_selection": dict(TRACK_IMAGE_OBSERVATION_SELECTION_V1),
            "projection_seconds_shared": projection_seconds,
            "representation_build_seconds": build_seconds,
            "device": device,
            "prototype_view_geometry": (
                None if view_geometry_path is None else str(view_geometry_path)
            ),
            "prototype_view_geometry_sha256": view_geometry_sha256,
            "prototype_view_geometry_valid_fraction": view_geometry_valid_fraction,
        }
        output_path = output_paths[name]
        save_landmark_index_npz(index, output_path, metadata=metadata)
        representation_summary = {
            "stage": "projected_observation_landmark_representation_sweep",
            "representation": str(name),
            "output_index": str(output_path),
            "metadata": metadata,
            "prototype_view_geometry": (
                None if view_geometry_path is None else str(view_geometry_path)
            ),
        }
        summary_path = output_dir / f"projected_observations_{name}.summary.json"
        summary_path.write_text(json.dumps(representation_summary, indent=2, sort_keys=True) + "\n")
        summaries[name] = representation_summary
        print(
            f"[landmark-sweep] {name}: rows={len(index)} tracks={unique_track_count} "
            f"build={build_seconds:.1f}s output={output_path}",
            flush=True,
        )

    summary = {
        "stage": "projected_observation_landmark_representation_sweep",
        "representations": list(representations),
        "projection_seconds": projection_seconds,
        "total_seconds": float(time.time() - start),
        "sampled_observation_count": int(len(sampled)),
        "outputs": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
