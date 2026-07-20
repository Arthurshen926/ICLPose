"""Fit explicitly scoped PCA for pose-free RADIO-intermediate image grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_radio_final_context_pca_cache import _array_hash, _project
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.radio_intermediate_context import fit_normalized_pca
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
    save_spatial_image_context_cache,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    image_source_contract_signature,
)


RAW_FORMAT = "radio_intermediate_image_spatial_context_v1"
PCA_FORMAT = "radio_intermediate_image_context_pca_v1"
_SAFE_PCA_FIT_SCOPES = frozenset(
    {
        "mapping_train_images_only",
        "mapping_support_images_excluding_all_query_splits_v1",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_context", required=True)
    parser.add_argument("--pca_training_manifest", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.projection_dim) <= 0:
        raise ValueError("projection_dim must be positive")
    source_path = Path(args.source_context)
    source = load_spatial_image_context_cache(source_path, expected_format=RAW_FORMAT)
    if bool(source.metadata.get("image_retrieval_or_submap_used", True)):
        raise ValueError("RADIO intermediate source cache violates no-retrieval protocol")
    source_image_contract = dict(source.metadata.get("image_source_contract", {}))
    source_image_signature = image_source_contract_signature(source_image_contract)
    if str(source.metadata.get("source_image_manifest_sha256", "")) != str(
        source_image_signature["sampled_content_manifest_sha256"]
    ):
        raise ValueError("RADIO intermediate source cache has a stale image source manifest")
    manifest_path = Path(args.pca_training_manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest_metadata = manifest.get("metadata", {})
    if not isinstance(manifest_metadata, dict):
        raise ValueError("PCA training manifest metadata must be an object")
    pca_fit_scope = str(
        manifest_metadata.get("pca_fit_scope", "mapping_train_images_only")
    )
    if pca_fit_scope not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError(f"unsupported PCA training fit scope: {pca_fit_scope}")
    training_ids = sorted({str(row["image_id"]) for row in manifest.get("records", [])})
    if not training_ids:
        raise ValueError("PCA training manifest contains no image records")
    positions = {value: row for row, value in enumerate(source.image_ids.tolist())}
    missing = [value for value in training_ids if value not in positions]
    if missing:
        raise ValueError(f"PCA training images are absent from source context: {missing[:10]}")
    training_rows = np.asarray([positions[value] for value in training_ids], dtype=np.int64)
    projections: dict[str, dict[str, object]] = {}
    projection_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    projected_grids: dict[int, np.ndarray] = {}
    for grid_size, values in sorted(source.grids.items()):
        training_values = values[training_rows].reshape(-1, values.shape[-1])
        mean, components, metrics = fit_normalized_pca(
            training_values,
            output_dim=int(args.projection_dim),
            seed=int(args.seed),
        )
        projection_arrays[int(grid_size)] = (mean, components)
        projected_grids[int(grid_size)] = _project(values, mean, components)
        projections[f"grid{int(grid_size)}"] = {
            "mean_sha256": _array_hash(mean),
            "components_sha256": _array_hash(components),
            **metrics,
        }
    metadata = {
        "format": PCA_FORMAT,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "source_context_sha256": file_sha256_short(source_path),
        "source_image_manifest_sha256": source_image_signature[
            "sampled_content_manifest_sha256"
        ],
        "image_source_contract": source_image_contract,
        "pca_training_manifest_sha256": file_sha256_short(manifest_path),
        "pca_training_image_list_sha256": hashlib.sha256(
            "\n".join(training_ids).encode()
        ).hexdigest()[:16],
        "pca_training_image_count": int(len(training_ids)),
        "pca_fit_scope": pca_fit_scope,
        "radio_version": source.metadata.get("radio_version"),
        "radio_checkpoint_sha256": source.metadata.get("radio_checkpoint_sha256"),
        "intermediate_index": source.metadata.get("intermediate_index"),
        "normalization": "input_row_l2_centered_pca_output_row_l2",
        "spatial_grid_sizes": sorted(int(size) for size in projected_grids),
        "projection_dim": int(args.projection_dim),
        "projections": projections,
    }
    output = Path(args.output_cache)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output}")
    cache = SpatialImageContextCache(
        image_ids=source.image_ids,
        image_sizes=source.image_sizes,
        grids=projected_grids,
        metadata=metadata,
    )
    extra_arrays: dict[str, np.ndarray] = {}
    for grid_size, (mean, components) in projection_arrays.items():
        extra_arrays[f"grid{int(grid_size)}_pca_mean"] = mean.astype(np.float32)
        extra_arrays[f"grid{int(grid_size)}_pca_components"] = components.astype(np.float32)
    save_spatial_image_context_cache(
        cache,
        output,
        cache_dtype=str(args.cache_dtype),
        extra_arrays=extra_arrays,
    )
    summary = {
        "stage": "radio_intermediate_image_context_scoped_mapping_pca",
        "output_cache": str(output),
        "output_cache_sha256": file_sha256_short(output),
        "metadata": metadata,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
