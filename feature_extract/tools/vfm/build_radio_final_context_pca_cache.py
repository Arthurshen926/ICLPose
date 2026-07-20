"""Project pose-free RADIO-final context into an explicitly scoped PCA sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
)
from feature_extract.vfm.localization.radio_intermediate_context import fit_normalized_pca
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    image_source_contract_signature,
    require_compatible_image_source_contracts,
)


_SAFE_PCA_FIT_SCOPES = frozenset(
    {
        "mapping_train_images_only",
        "mapping_support_images_excluding_all_query_splits_v1",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_context", required=True)
    pca = parser.add_mutually_exclusive_group(required=True)
    pca.add_argument("--pca_training_manifest")
    pca.add_argument("--projection_source_cache")
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args(argv)


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _project(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    shape = rows.shape[:-1]
    flat = rows.reshape(-1, rows.shape[-1])
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    projected = (flat - mean[None]) @ components.T
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    return projected.reshape(*shape, components.shape[0]).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.projection_dim) <= 0:
        raise ValueError("projection_dim must be positive")
    source_path = Path(args.source_context)
    with np.load(source_path, allow_pickle=False) as data:
        image_ids = np.asarray(data["image_ids"]).astype(str)
        context_values = {
            "summary": np.asarray(data["summary_descriptors"], dtype=np.float32),
            "global": np.asarray(data["global_local_descriptors"], dtype=np.float32),
        }
        for key in data.files:
            match = re.fullmatch(r"grid([1-9][0-9]*)_descriptors", str(key))
            if match is not None:
                context_values[f"grid{int(match.group(1))}"] = np.asarray(
                    data[key], dtype=np.float32
                )
        source_metadata = json.loads(str(data["metadata_json"].item()))
    if source_metadata.get("format") != "radio_image_multiscale_context_v1":
        raise ValueError("unsupported source RADIO image context cache")
    if bool(source_metadata.get("pose_or_ground_truth_used", True)):
        raise ValueError("source RADIO image context must be pose/GT free")
    source_image_contract = dict(source_metadata.get("image_source_contract", {}))
    source_image_signature = image_source_contract_signature(source_image_contract)
    if str(source_metadata.get("source_image_manifest_sha256", "")) != str(
        source_image_signature["sampled_content_manifest_sha256"]
    ):
        raise ValueError("source RADIO image context has a stale image source manifest")
    if "grid4" not in context_values:
        raise ValueError("source RADIO image context must retain grid4 descriptors")
    projections = {}
    outputs = {}
    projection_arrays = {}
    projection_source_path = (
        None if args.projection_source_cache is None else Path(args.projection_source_cache)
    )
    if projection_source_path is None:
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
        training_ids = sorted(
            {str(row["image_id"]) for row in manifest.get("records", [])}
        )
        if not training_ids:
            raise ValueError("PCA training manifest contains no image records")
        position = {value: row for row, value in enumerate(image_ids.tolist())}
        missing = [value for value in training_ids if value not in position]
        if missing:
            raise ValueError(
                f"PCA training images are absent from source context: {missing[:10]}"
            )
        training_rows = np.asarray(
            [position[value] for value in training_ids], dtype=np.int64
        )
        source_projection_metadata = None
    else:
        with np.load(projection_source_path, allow_pickle=False) as data:
            source_projection_metadata = json.loads(str(data["metadata_json"].item()))
            if source_projection_metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
                raise ValueError("unsupported RADIO final projection source cache")
            for name in context_values:
                projection_arrays[name] = (
                    np.asarray(data[f"{name}_pca_mean"], dtype=np.float32),
                    np.asarray(data[f"{name}_pca_components"], dtype=np.float32),
                )
        if str(source_projection_metadata.get("radio_checkpoint_sha256", "")) != str(
            source_metadata.get("radio_checkpoint_sha256", "")
        ):
            raise ValueError("source context and reused PCA use different RADIO checkpoints")
        if int(source_projection_metadata.get("projection_dim", -1)) != int(
            args.projection_dim
        ):
            raise ValueError("reused RADIO final PCA dimension differs")
        require_compatible_image_source_contracts(
            source_image_contract,
            dict(source_projection_metadata.get("image_source_contract", {})),
            context="source context and reused RADIO final PCA",
        )
        pca_fit_scope = str(source_projection_metadata.get("pca_fit_scope", ""))
        if pca_fit_scope not in _SAFE_PCA_FIT_SCOPES:
            raise ValueError(f"reused PCA has an unsupported fit scope: {pca_fit_scope}")
        training_ids = []
        manifest_path = None

    for name, values in context_values.items():
        if projection_source_path is None:
            training_values = values[training_rows].reshape(-1, values.shape[-1])
            mean, components, metrics = fit_normalized_pca(
                training_values,
                output_dim=int(args.projection_dim),
                seed=int(args.seed),
            )
            projection_arrays[name] = (mean, components)
        else:
            mean, components = projection_arrays[name]
            metrics = dict(source_projection_metadata["projections"][name])
        projections[name] = {
            "mean_sha256": _array_hash(mean),
            "components_sha256": _array_hash(components),
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"mean_sha256", "components_sha256"}
            },
        }
        outputs[name] = _project(values, mean, components)

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    image_by_name = {str(image.image_name): image for image in images.values()}
    sizes = np.asarray(
        [
            [cameras[int(image_by_name[value].camera_id)].width, cameras[int(image_by_name[value].camera_id)].height]
            for value in image_ids.tolist()
        ],
        dtype=np.int64,
    )
    training_list_hash = (
        hashlib.sha256("\n".join(training_ids).encode()).hexdigest()[:16]
        if projection_source_path is None
        else source_projection_metadata.get("pca_training_image_list_sha256")
    )
    metadata = {
        "format": RADIO_FINAL_CONTEXT_PCA_FORMAT,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "source_context_sha256": file_sha256_short(source_path),
        "source_image_manifest_sha256": source_image_signature[
            "sampled_content_manifest_sha256"
        ],
        "image_source_contract": source_image_contract,
        "pca_training_manifest_sha256": (
            file_sha256_short(manifest_path)
            if manifest_path is not None
            else source_projection_metadata.get("pca_training_manifest_sha256")
        ),
        "pca_training_image_list_sha256": training_list_hash,
        "pca_training_image_count": (
            int(len(training_ids))
            if projection_source_path is None
            else int(source_projection_metadata.get("pca_training_image_count", 0))
        ),
        "pca_fit_scope": pca_fit_scope,
        "radio_checkpoint_sha256": source_metadata.get("radio_checkpoint_sha256"),
        "radio_version": source_metadata.get("radio_version"),
        "projection_dim": int(args.projection_dim),
        "normalization": "input_row_l2_centered_pca_output_row_l2",
        "grid": "adaptive_pool_spatial_cell_lookup_v2",
        "spatial_grid_sizes": sorted(
            int(name[len("grid") :])
            for name in context_values
            if name.startswith("grid")
        ),
        "projections": projections,
        "projection_source_cache_sha256": (
            None
            if projection_source_path is None
            else file_sha256_short(projection_source_path)
        ),
    }
    dtype = np.float16 if str(args.cache_dtype) == "float16" else np.float32
    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=image_ids,
        image_sizes=sizes,
        summary_descriptors=outputs["summary"].astype(dtype),
        global_descriptors=outputs["global"].astype(dtype),
        **{
            f"{name}_descriptors": outputs[name].astype(dtype)
            for name in context_values
            if name.startswith("grid")
        },
        **{
            f"{name}_pca_mean": values[0].astype(np.float32)
            for name, values in projection_arrays.items()
        },
        **{
            f"{name}_pca_components": values[1].astype(np.float32)
            for name, values in projection_arrays.items()
        },
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary_payload = {
        "stage": "radio_final_context_scoped_mapping_pca",
        "output_cache": str(output_path),
        "output_cache_sha256": file_sha256_short(output_path),
        "metadata": metadata,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
