"""Build projected-observation 3D landmark descriptors from a trained joint mapper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence
import warnings

import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.descriptor_space import (
    NO_VIEW_CLUSTERING_CONFIG,
    PROTOTYPE_BUILDER_VERSION,
    TRACK_IMAGE_OBSERVATION_SELECTION_V1,
    descriptor_space_manifest,
    token_feature_source_config,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.landmark_hybrid import (
    build_projected_observation_landmark_index,
    save_landmark_index_npz,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.landmark_feature_aggregation import (
    AGGREGATION_METHODS,
    LandmarkAggregationConfig,
    TrackPrototypeBuilder,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument(
        "--descriptor_source_override_json",
        default="",
        help="Optional JSON object/file overriding recorded feature-source fields; changes the descriptor-space id.",
    )
    parser.add_argument(
        "--allow_descriptor_source_override",
        action="store_true",
        help="Acknowledge that descriptor source overrides are diagnostic and must be recorded in the cache id.",
    )
    parser.add_argument(
        "--method",
        default="mean",
        choices=tuple(sorted(AGGREGATION_METHODS)),
    )
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--missing", default="skip", choices=("skip", "error"))
    parser.add_argument(
        "--utility_mode",
        default="inverse_reprojection",
        choices=(
            "inverse_reprojection",
            "center",
            "inverse_reprojection_center",
            "view_consistency",
            "inverse_reprojection_center_view",
        ),
    )
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--projection_image_batch_size", type=int, default=8)
    parser.add_argument("--projection_load_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim_fraction", type=float, default=0.2)
    parser.add_argument("--view_consistent_keep", type=int, default=4)
    parser.add_argument("--geometric_median_iterations", type=int, default=32)
    parser.add_argument("--l2_normalize_observations", action="store_true")
    parser.add_argument("--allow_prototype_builder_override", action="store_true")
    parser.add_argument("--weight_floor", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def source_image_list_hash(
    manifest: TokenBankManifest,
    image_ids: set[str] | None = None,
) -> str:
    payload = [
        {
            "image_id": str(record.image_id),
            "token_path": str(record.token_path),
            "layers": [str(layer.name) for layer in record.layers],
        }
        for record in manifest.records
        if image_ids is None or str(record.image_id) in image_ids
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_json_object(value: str) -> dict[str, object]:
    text = str(value).strip()
    if not text:
        return {}
    path = Path(text)
    payload = json.loads(path.read_text() if path.exists() else text)
    if not isinstance(payload, dict):
        raise ValueError("descriptor_source_override_json must contain a JSON object")
    return dict(payload)


def resolve_descriptor_source_config(
    *,
    manifest: TokenBankManifest,
    feature_key: str,
    checkpoint_summary: dict[str, object],
    model_input_dim: int,
    override_json: str = "",
    allow_override: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve source semantics without silently trusting CLI defaults."""

    inferred = token_feature_source_config(manifest, str(feature_key))
    checkpoint_value = checkpoint_summary.get("descriptor_source_config")
    checkpoint_config = dict(checkpoint_value) if isinstance(checkpoint_value, dict) else {}
    resolution = "checkpoint_and_token_manifest"
    if checkpoint_config:
        for key, expected in checkpoint_config.items():
            if key in inferred and inferred[key] != expected:
                raise ValueError(
                    f"checkpoint/token descriptor source mismatch for {key}: "
                    f"checkpoint={expected!r}, token_manifest={inferred[key]!r}"
                )
        resolved = {**inferred, **checkpoint_config}
    else:
        resolution = "token_manifest_legacy_checkpoint_fallback"
        resolved = dict(inferred)
        warnings.warn(
            "checkpoint has no descriptor_source_config; using the validated token manifest as a legacy fallback",
            stacklevel=2,
        )
    if int(resolved.get("input_channels", -1)) != int(model_input_dim):
        raise ValueError(
            "mapper input dimension does not match token feature source: "
            f"model={int(model_input_dim)}, token={resolved.get('input_channels')!r}"
        )
    override = _load_json_object(str(override_json))
    allowed_override_keys = set(inferred)
    unknown = sorted(set(override) - allowed_override_keys)
    if unknown:
        raise ValueError(f"unsupported descriptor source override fields: {unknown!r}")
    if override and not bool(allow_override):
        raise ValueError("descriptor source override requires --allow_descriptor_source_override")
    if override:
        warnings.warn(
            "descriptor source override is active; the override is embedded in a distinct descriptor-space id",
            stacklevel=2,
        )
        resolved.update(override)
        resolution += "+explicit_override"
    audit = {
        "resolution": resolution,
        "token_manifest_inferred": inferred,
        "checkpoint_recorded": checkpoint_config,
        "explicit_override": override,
    }
    return resolved, audit


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(str(args.device))
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
        override_json=str(args.descriptor_source_override_json),
        allow_override=bool(args.allow_descriptor_source_override),
    )
    planned_prototype_builder = TrackPrototypeBuilder(
        aggregation=LandmarkAggregationConfig(
            method=str(args.method),
            min_observations=int(args.min_observations),
            seed=int(args.seed),
            trim_fraction=float(args.trim_fraction),
            view_consistent_keep=int(args.view_consistent_keep),
            geometric_median_iterations=int(args.geometric_median_iterations),
            l2_normalize_observations=bool(args.l2_normalize_observations),
            weight_floor=float(args.weight_floor),
        ),
        normalize_final_prototypes=True,
    ).to_dict()
    checkpoint_builder_value = joint_run.summary.get("track_prototype_builder")
    checkpoint_builder = dict(checkpoint_builder_value) if isinstance(checkpoint_builder_value, dict) else {}
    if checkpoint_builder and checkpoint_builder != planned_prototype_builder:
        if not bool(args.allow_prototype_builder_override):
            raise ValueError(
                "checkpoint/bank TrackPrototypeBuilder mismatch; use identical settings or explicitly pass "
                "--allow_prototype_builder_override for an ablation"
            )
        warnings.warn(
            "TrackPrototypeBuilder override is active; the bank receives a distinct descriptor-space id",
            stacklevel=2,
        )
    mapper = JointFeatureMapper(joint_run.model, device=device)
    index, metadata = build_projected_observation_landmark_index(
        observations,
        manifest,
        mapper,
        feature_key=str(args.feature_key),
        aggregation_method=str(args.method),
        min_observations=int(args.min_observations),
        missing=str(args.missing),
        utility_mode=str(args.utility_mode),
        weight_floor=float(args.weight_floor),
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
        trim_fraction=float(args.trim_fraction),
        view_consistent_keep=int(args.view_consistent_keep),
        geometric_median_iterations=int(args.geometric_median_iterations),
        l2_normalize_observations=bool(args.l2_normalize_observations),
        projection_image_batch_size=int(args.projection_image_batch_size),
        projection_load_workers=int(args.projection_load_workers),
    )
    output_index = Path(args.output_index)
    checkpoint_sha256 = file_sha256_short(Path(args.matcha_joint_checkpoint))
    track_sha256 = file_sha256_short(Path(args.track_observations))
    token_manifest_sha256 = file_sha256_short(Path(args.token_manifest))
    effective_source_image_ids = {str(observation.image_id) for observation in observations}
    image_manifest_hash = source_image_list_hash(manifest, effective_source_image_ids)
    aggregation = dict(metadata.get("aggregation", {}))
    prototype_builder = dict(metadata.get("prototype_builder", {}))
    space_manifest = descriptor_space_manifest(
        checkpoint_sha256=checkpoint_sha256,
        mapper_mode="joint_full_map",
        feature_key=str(args.feature_key),
        projection_source="projected_observation_full_map",
        aggregation_method=str(aggregation.get("method", args.method)),
        l2_normalize_observations=bool(aggregation.get("l2_normalize_observations", bool(args.l2_normalize_observations))),
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
        view_clustering=NO_VIEW_CLUSTERING_CONFIG,
        maplet_bank_version="none",
        observation_selection=TRACK_IMAGE_OBSERVATION_SELECTION_V1,
    )
    output_metadata = {
        **metadata,
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
        "source_image_count": int(len(effective_source_image_ids)),
        "descriptor_source_config": source_config,
        "descriptor_source_config_audit": source_config_audit,
        "checkpoint_track_prototype_builder": checkpoint_builder,
        "prototype_builder_override": bool(checkpoint_builder and checkpoint_builder != planned_prototype_builder),
        "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
        "matcha_joint_checkpoint_sha256": checkpoint_sha256,
        "descriptor_space_manifest": space_manifest,
        "descriptor_space_id": str(space_manifest["descriptor_space_id"]),
        "descriptor_dimension": int(index.feature_dim),
        "observation_selection": dict(TRACK_IMAGE_OBSERVATION_SELECTION_V1),
        "device": device,
    }
    save_landmark_index_npz(index, output_index, metadata=output_metadata)
    summary = {
        "stage": "projected_observation_3d_landmark_feature_aggregation",
        "projection_mode": "full_map_projected_observations",
        "input_files": {
            "track_observations": {
                "path": str(args.track_observations),
                "sha256": track_sha256,
            },
            "token_manifest": {
                "path": str(args.token_manifest),
                "sha256": token_manifest_sha256,
            },
            "matcha_joint_checkpoint": {
                "path": str(args.matcha_joint_checkpoint),
                "sha256": checkpoint_sha256,
            },
        },
        "output_files": {
            "projected_landmark_index": str(output_index),
        },
        "metadata": output_metadata,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
