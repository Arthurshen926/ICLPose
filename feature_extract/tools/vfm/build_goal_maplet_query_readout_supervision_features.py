"""Materialize pose-free 36x64 query RADIO readouts for typed supervision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_local_supervision_dataset,
)
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_pose_free_query_readout_supervision_features_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--supervision_labels", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--output_features", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source_path = Path(args.source_dataset)
    label_path = Path(args.supervision_labels)
    output_path = Path(args.output_features)
    manifest_path = Path(args.output_manifest)
    partial_path = output_path.with_suffix(".partial.npy")
    if output_path.suffix != ".npy" or any(
        value.exists() for value in (output_path, manifest_path, partial_path)
    ):
        raise FileExistsError("refusing to overwrite query readout features")
    source, source_metadata = load_pose_candidate_dataset(
        source_path, require_rendered_targets=False,
    )
    labels, label_metadata = _load_local_supervision_dataset(label_path)
    source_rows = np.asarray(labels["source_query_rows"], dtype=np.int64)
    if (
        np.any(source_rows < 0) or np.any(source_rows >= source["image_ids"].size)
        or not np.array_equal(source["image_ids"][source_rows], labels["image_ids"])
        or label_metadata.get("source_dataset_content_sha256")
        != source_metadata["content_sha256"]
    ):
        raise ValueError("query readout labels differ from the source inventory")

    mapper_path = Path(args.surface_mapper)
    mapper_sha = file_sha256(mapper_path)
    contract_path = Path(args.field_feature_contract)
    contract = json.loads(contract_path.read_text())
    if contract.get("query_readout_sha256") != mapper_sha:
        raise ValueError("query readout mapper lineage differs")
    device = torch.device(str(args.device))
    mapper, _ = load_surface_maplet_mapper(mapper_path, device=str(device))
    mapper.model.to(device).eval()
    shape = (int(source_rows.size), 128, 36, 64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float16, shape=shape)
    with torch.no_grad():
        for output_row, source_row in enumerate(source_rows.tolist()):
            token_path = Path(str(source["radio_token_paths"][source_row])).resolve()
            if file_sha256(token_path) != str(source["radio_file_sha256"][source_row]):
                raise ValueError("query RADIO token lineage differs")
            with np.load(token_path, allow_pickle=False) as data:
                if set(data.files) != {"radio_final"}:
                    raise ValueError("query RADIO token members differ")
                raw = np.asarray(data["radio_final"], dtype=np.float32)
            mapped = mapper.model(torch.as_tensor(raw, device=device)[None])[0]
            if mapped.shape != (128, 36, 64) or not torch.isfinite(mapped).all():
                raise ValueError("query mapped RADIO grid differs")
            output[output_row] = mapped.cpu().numpy().astype(np.float16, copy=False)
    output.flush()
    del output
    os.replace(partial_path, output_path)
    manifest = {
        "artifact_type": SCHEMA,
        "feature_semantics": "pose_free_surface_mapper_radio_readout_128x36x64_v1",
        "feature_file": str(output_path.resolve()),
        "feature_file_sha256": file_sha256(output_path),
        "feature_shape": list(shape),
        "feature_dtype": "float16",
        "image_ids": labels["image_ids"].tolist(),
        "source_dataset_file_sha256": file_sha256(source_path),
        "source_dataset_content_sha256": source_metadata["content_sha256"],
        "dataset_file_sha256": file_sha256(label_path),
        "dataset_content_sha256": label_metadata["content_sha256"],
        "surface_mapper_file_sha256": mapper_sha,
        "field_feature_contract_file_sha256": file_sha256(contract_path),
        "query_pose_is_input": False,
        "candidate_pose_is_input": False,
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
