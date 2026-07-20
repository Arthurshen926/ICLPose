"""Build a raw RADIO-intermediate projected-observation landmark bank.

Unlike the legacy 1x1 post-aggregate projection, this command samples the
frozen full-image RADIO-intermediate PCA grid at every mapping SfM observation
and aggregates descriptors per physical track.  It never accesses query pose,
query labels, image retrieval, render output, or a candidate ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_observation_landmark_bank import (
    build_context_observation_landmark_index,
)
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
    save_landmark_index_npz,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.spatial_image_context import (
    load_spatial_image_context_cache,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    image_source_contract_signature,
)


STAGE = "radio_intermediate_pca_projected_observation_landmark_bank_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-cache", required=True)
    parser.add_argument("--mapping-support-manifest", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--source-landmark-bank", required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _short_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()[:16]


def _mapping_support_image_ids(path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "maplet_support_image_manifest_v1":
        raise ValueError("mapping support manifest has an unsupported format")
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise ValueError("mapping support manifest is malformed")
    required = {
        "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1",
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("mapping support manifest violates the no-retrieval protocol")
    image_ids = tuple(sorted({str(row.get("image_id", "")) for row in records}))
    if not image_ids or "" in image_ids or len(image_ids) != len(records):
        raise ValueError("mapping support image IDs are invalid")
    if int(metadata.get("support_image_count", -1)) != len(image_ids):
        raise ValueError("mapping support manifest image count is stale")
    return image_ids, metadata


def _descriptor_manifest(
    *,
    cache_metadata: Mapping[str, Any],
    context_cache: Path,
    mapping_support_manifest: Path,
    support_geometry_index: Path,
    source_landmark_bank: Path,
    grid_size: int,
    source_bank_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": 1,
        "mapper_mode": "raw_radio_intermediate_pca_full_image_grid",
        "projection_source": "raw_radio_intermediate_pca_projected_observation_full_map",
        "feature_key": "radio_intermediate_pca",
        "radio_checkpoint_sha256": cache_metadata.get("radio_checkpoint_sha256"),
        "intermediate_index": cache_metadata.get("intermediate_index"),
        "pca_source_context_sha256": cache_metadata.get("source_context_sha256"),
        "pca_training_manifest_sha256": cache_metadata.get("pca_training_manifest_sha256"),
        "pca_training_image_list_sha256": cache_metadata.get(
            "pca_training_image_list_sha256"
        ),
        "pca_fit_scope": cache_metadata.get("pca_fit_scope"),
        "context_cache_sha256": file_sha256_short(context_cache),
        "mapping_support_manifest_sha256": file_sha256_short(mapping_support_manifest),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
        "source_landmark_bank_sha256": file_sha256_short(source_landmark_bank),
        "source_image_manifest_sha256": cache_metadata.get("source_image_manifest_sha256"),
        "sfm_track_hash": source_bank_metadata.get("track_observations_sha256"),
        "descriptor_dimension": int(cache_metadata.get("projection_dim", 0)),
        "grid_size": int(grid_size),
        "sampling_convention": (
            "sfm_pixel_endpoint_to_grid_endpoint_bilinear_border_clamp_v1"
        ),
        "normalization_mode": "bilinear_sample_l2_then_track_normalized_mean_l2",
        "aggregation_method": "normalized_mean",
        "coordinate_source": "mapping_sfm_observation_xy",
    }
    if (
        not manifest["radio_checkpoint_sha256"]
        or not manifest["pca_training_manifest_sha256"]
        or not manifest["source_image_manifest_sha256"]
        or not manifest["sfm_track_hash"]
        or int(manifest["descriptor_dimension"]) <= 0
    ):
        raise ValueError("context cache or source bank lacks descriptor-space lineage")
    manifest["descriptor_space_id"] = _short_hash(manifest)
    return manifest


def build_radio_intermediate_observation_landmark_bank(
    *,
    context_cache: Path,
    mapping_support_manifest: Path,
    support_geometry_index: Path,
    source_landmark_bank: Path,
    grid_size: int,
    output_index: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """Materialize a descriptor-space-compatible raw intermediate landmark bank."""

    output = Path(output_index)
    summary = Path(summary_json)
    if output.exists() or summary.exists():
        raise FileExistsError("refusing to overwrite intermediate landmark-bank outputs")
    cache_path = Path(context_cache)
    cache = load_spatial_image_context_cache(cache_path, expected_format=PCA_FORMAT)
    metadata = dict(cache.metadata)
    if (
        bool(metadata.get("pose_or_ground_truth_used", True))
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("render", True))
        or int(grid_size) not in cache.grids
        or int(metadata.get("projection_dim", -1)) != cache.descriptor_dim
    ):
        raise ValueError("RADIO intermediate context cache violates the frozen protocol")
    image_contract = image_source_contract_signature(
        dict(metadata.get("image_source_contract", {}))
    )
    if str(metadata.get("source_image_manifest_sha256", "")) != str(
        image_contract["sampled_content_manifest_sha256"]
    ):
        raise ValueError("RADIO intermediate context cache has stale image provenance")
    support_ids, support_metadata = _mapping_support_image_ids(
        Path(mapping_support_manifest)
    )
    if not set(support_ids).issubset(set(cache.image_ids.tolist())):
        raise ValueError("RADIO intermediate context cache lacks mapping support images")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if tuple(geometry.image_ids) != support_ids:
        raise ValueError("support geometry images differ from the frozen mapping manifest")
    source_index, source_metadata = load_landmark_index_npz(Path(source_landmark_bank))
    if (
        str(source_metadata.get("track_observations_sha256", ""))
        != str(geometry_metadata.get("support_track_observations_sha256", ""))
        or int(source_metadata.get("sampled_observation_count", -1))
        != int(np_sum(geometry_metadata.get("observation_count")))
    ):
        raise ValueError("source landmark bank and support geometry have different tracks")
    started = time.monotonic()
    index, aggregation = build_context_observation_landmark_index(
        cache=cache,
        geometry=geometry,
        source_landmark_index=source_index,
        support_image_ids=support_ids,
        grid_size=int(grid_size),
    )
    descriptor_manifest = _descriptor_manifest(
        cache_metadata=metadata,
        context_cache=cache_path,
        mapping_support_manifest=Path(mapping_support_manifest),
        support_geometry_index=Path(support_geometry_index),
        source_landmark_bank=Path(source_landmark_bank),
        grid_size=int(grid_size),
        source_bank_metadata=source_metadata,
    )
    bank_metadata: dict[str, Any] = {
        "stage": STAGE,
        "descriptor_space_id": descriptor_manifest["descriptor_space_id"],
        "descriptor_space_manifest": descriptor_manifest,
        "context_cache": str(cache_path),
        "context_cache_sha256": file_sha256_short(cache_path),
        "mapping_support_manifest": str(Path(mapping_support_manifest)),
        "mapping_support_manifest_sha256": file_sha256_short(
            Path(mapping_support_manifest)
        ),
        "support_geometry_index": str(Path(support_geometry_index)),
        "support_geometry_index_sha256": file_sha256_short(
            Path(support_geometry_index)
        ),
        "source_landmark_bank": str(Path(source_landmark_bank)),
        "source_landmark_bank_sha256": file_sha256_short(Path(source_landmark_bank)),
        "source_descriptor_space_id": source_metadata.get("descriptor_space_id"),
        "feature_dim": int(index.feature_dim),
        "landmark_count": int(len(index)),
        "sampled_observation_count": aggregation["sampled_observation_count"],
        "excluded_noncanonical_observation_count": aggregation[
            "excluded_observation_count"
        ],
        "border_clamped_observation_count": aggregation[
            "border_clamped_observation_count"
        ],
        "maximum_boundary_excursion_px": aggregation[
            "maximum_boundary_excursion_px"
        ],
        "source_image_count": int(len(support_ids)),
        "source_image_manifest_sha256": image_contract[
            "sampled_content_manifest_sha256"
        ],
        "mapping_support_protocol": support_metadata,
        "query_pose_or_target_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    save_landmark_index_npz(index, output, metadata=bank_metadata)
    result: dict[str, Any] = {
        "stage": STAGE,
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "feature_dim": int(index.feature_dim),
        "landmark_count": int(len(index)),
        "aggregation": aggregation,
        "elapsed_seconds": float(time.monotonic() - started),
        "descriptor_space_manifest": descriptor_manifest,
        "protocol": {
            "projected_observation_full_map": True,
            "post_aggregate_1x1_projection": False,
            "query_pose_or_target_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def np_sum(value: object) -> int:
    """Parse one non-negative scalar from untrusted cache metadata."""

    try:
        output = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("support geometry observation count is invalid") from error
    if output <= 0:
        raise ValueError("support geometry observation count is invalid")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_radio_intermediate_observation_landmark_bank(
        context_cache=Path(args.context_cache),
        mapping_support_manifest=Path(args.mapping_support_manifest),
        support_geometry_index=Path(args.support_geometry_index),
        source_landmark_bank=Path(args.source_landmark_bank),
        grid_size=int(args.grid_size),
        output_index=Path(args.output_index),
        summary_json=Path(args.summary_json),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
