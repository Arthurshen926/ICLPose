"""Append pose-free query/rendered typed consistency to candidate features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS,
    QUERY_TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    TYPED_FULLTOKEN_POSE_RANKING_CHANNELS,
    TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.query_typed_geometry import (
    append_query_typed_geometry_consistency,
    load_query_typed_geometry_predictor,
)


SCHEMA = "goal_maplet_query_typed_consistency_pose_ranking_features_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed_features", required=True)
    parser.add_argument("--typed_manifest", required=True)
    parser.add_argument("--query_features", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_geometry_model", required=True)
    parser.add_argument("--output_features", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--query_batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    typed_path = Path(args.typed_features).resolve()
    query_path = Path(args.query_features).resolve()
    typed_manifest = json.loads(Path(args.typed_manifest).read_text())
    query_manifest = json.loads(Path(args.query_manifest).read_text())
    if (
        Path(str(typed_manifest.get("feature_file", ""))).resolve() != typed_path
        or typed_manifest.get("feature_file_sha256") != file_sha256(typed_path)
        or typed_manifest.get("feature_semantics")
        != TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS
        or typed_manifest.get("feature_channels")
        != list(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS)
        or Path(str(query_manifest.get("feature_file", ""))).resolve() != query_path
        or query_manifest.get("feature_file_sha256") != file_sha256(query_path)
        or typed_manifest.get("dataset_content_sha256")
        != query_manifest.get("dataset_content_sha256")
    ):
        raise ValueError("query typed consistency parent lineage differs")
    typed = np.load(typed_path, mmap_mode="r", allow_pickle=False)
    query = np.load(query_path, mmap_mode="r", allow_pickle=False)
    if (
        typed.dtype != np.float16 or typed.shape[2:] != (18, 36, 64)
        or query.dtype != np.float16 or query.shape != (typed.shape[0], 128, 36, 64)
    ):
        raise ValueError("query typed consistency parent arrays differ")
    device = torch.device(str(args.device))
    model_path = Path(args.query_geometry_model)
    model, model_metadata = load_query_typed_geometry_predictor(model_path, device=device)
    if (
        model_metadata.get("dataset_content_sha256")
        != typed_manifest.get("dataset_content_sha256")
        or model_metadata.get("query_feature_file_sha256")
        != query_manifest.get("feature_file_sha256")
        or model_metadata.get("typed_feature_file_sha256")
        != typed_manifest.get("feature_file_sha256")
    ):
        raise ValueError("query geometry model parent lineage differs")

    output_path = Path(args.output_features)
    manifest_path = Path(args.output_manifest)
    partial_path = output_path.with_suffix(".partial.npy")
    if output_path.suffix != ".npy" or any(
        value.exists() for value in (output_path, manifest_path, partial_path)
    ):
        raise FileExistsError("refusing to overwrite query typed consistency features")
    batch_size = int(args.query_batch_size)
    if batch_size <= 0:
        raise ValueError("query batch size must be positive")
    shape = (
        typed.shape[0], typed.shape[1],
        len(QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float16, shape=shape)
    with torch.no_grad():
        for begin in range(0, int(typed.shape[0]), batch_size):
            end = min(begin + batch_size, int(typed.shape[0]))
            query_tensor = torch.as_tensor(
                np.asarray(query[begin:end], dtype=np.float32), device=device,
            )
            prediction = model(query_tensor)
            candidate = torch.as_tensor(
                np.asarray(typed[begin:end], dtype=np.float32), device=device,
            )
            augmented = append_query_typed_geometry_consistency(candidate, prediction)
            output[begin:end] = augmented.cpu().numpy().astype(np.float16, copy=False)
            print(json.dumps({"completed": end, "total": int(typed.shape[0])}), flush=True)
    output.flush()
    del output
    os.replace(partial_path, output_path)
    manifest = {
        "artifact_type": SCHEMA,
        "feature_semantics": QUERY_TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
        "feature_channels": list(QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS),
        "feature_file": str(output_path.resolve()),
        "feature_file_sha256": file_sha256(output_path),
        "feature_shape": list(shape),
        "feature_dtype": "float16",
        "dataset_file_sha256": typed_manifest["dataset_file_sha256"],
        "dataset_content_sha256": typed_manifest["dataset_content_sha256"],
        "typed_feature_file_sha256": typed_manifest["feature_file_sha256"],
        "query_feature_file_sha256": query_manifest["feature_file_sha256"],
        "query_geometry_model_file_sha256": file_sha256(model_path),
        "query_geometry_model_content_sha256": model_metadata["model_content_sha256"],
        "query_pose_is_model_input": False,
        "candidate_pose_values_are_model_inputs": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "production_eligible": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_features": str(output_path),
        "output_manifest": str(manifest_path),
        "feature_file_sha256": manifest["feature_file_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
