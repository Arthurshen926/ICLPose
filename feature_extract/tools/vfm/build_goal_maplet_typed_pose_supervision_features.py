"""Render typed full-token features for an existing pose-supervision inventory.

This builder reuses frozen candidate poses and labels.  It adds only
candidate-rendered camera-frame unsigned normal axes, relative log depth, and
boundary evidence to the existing RADIO/canonical appearance grid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_local_supervision_dataset,
)
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    TYPED_FULLTOKEN_POSE_RANKING_CHANNELS,
    TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    compact_typed_fulltoken_pose_ranking_features,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)


FEATURE_SCHEMA = "goal_maplet_typed_fulltoken_pose_ranking_features_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--supervision_labels", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--output_features", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--render_batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source_path = Path(args.source_dataset)
    label_path = Path(args.supervision_labels)
    output_path = Path(args.output_features)
    manifest_path = Path(args.output_manifest)
    partial_path = output_path.with_suffix(".partial.npy")
    if output_path.suffix != ".npy" or any(
        path.exists() for path in (output_path, manifest_path, partial_path)
    ):
        raise FileExistsError("refusing to overwrite typed pose features")
    if int(args.render_batch_size) <= 0:
        raise ValueError("render batch size must be positive")
    source, source_metadata = load_pose_candidate_dataset(
        source_path, require_rendered_targets=False,
    )
    labels, label_metadata = _load_local_supervision_dataset(label_path)
    source_rows = np.asarray(labels["source_query_rows"], dtype=np.int64)
    if (
        np.any(source_rows < 0) or np.any(source_rows >= source["image_ids"].size)
        or not np.array_equal(source["image_ids"][source_rows], labels["image_ids"])
        or label_metadata.get("source_dataset_content_sha256") != source_metadata["content_sha256"]
    ):
        raise ValueError("typed feature labels differ from the source query inventory")

    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = json.loads(Path(args.field_feature_contract).read_text())
    mapper_sha = file_sha256(Path(args.surface_mapper))
    if (
        physical.content_sha256 != source_metadata.get("physical_map_sha256")
        or field.content_sha256 != source_metadata.get("canonical_field_sha256")
        or field.physical_map_sha256 != physical.content_sha256
        or contract.get("canonical_field_sha256") != field.content_sha256
        or contract.get("query_readout_sha256") != mapper_sha
    ):
        raise ValueError("typed feature map/field/mapper lineage differs")
    device = torch.device(str(args.device))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(device))
    mapper.model.to(device).eval()
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(device))
    contributors = _load_contributors(Path(args.contributors))
    candidate_count = int(labels["candidate_valid"].shape[1])
    shape = (
        int(source_rows.size), candidate_count,
        len(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float16, shape=shape,
    )
    timing_rows = []
    with torch.no_grad():
        for output_row, query_index in enumerate(source_rows.tolist()):
            image_id = str(source["image_ids"][query_index])
            contributor = contributors.get(image_id)
            if contributor is None or file_sha256(contributor) != str(
                source["contributor_file_sha256"][query_index]
            ):
                raise ValueError(f"typed feature contributor lineage differs for {image_id}")
            token_path = Path(str(source["radio_token_paths"][query_index])).resolve()
            if file_sha256(token_path) != str(source["radio_file_sha256"][query_index]):
                raise ValueError("typed feature RADIO lineage differs")
            with np.load(token_path, allow_pickle=False) as data:
                if set(data.files) != {"radio_final"}:
                    raise ValueError("typed feature RADIO members differ")
                raw = np.asarray(data["radio_final"], dtype=np.float32)
            query = mapper.model(torch.as_tensor(raw, device=device)[None])[0]
            query = query.permute(1, 2, 0).reshape(2304, 128)
            poses = np.asarray(labels["candidate_poses_w2c"][output_row], dtype=np.float64)
            render_seconds = 0.0
            for begin in range(0, candidate_count, int(args.render_batch_size)):
                rendered = scene.render_direct_typed_canonical_grid_batch(
                    poses[begin:begin + int(args.render_batch_size)], _camera(contributor),
                )
                render_seconds += float(rendered.total_seconds)
                for local in range(rendered.batch_size):
                    features[output_row, begin + local] = (
                        compact_typed_fulltoken_pose_ranking_features(
                            query,
                            torch.as_tensor(
                                rendered.feature[local].reshape(2304, 1, -1), device=device,
                            ),
                            torch.as_tensor(
                                rendered.mass[local].reshape(2304, 1), device=device,
                            ),
                            torch.as_tensor(
                                rendered.valid[local].reshape(2304, 1), device=device,
                            ),
                            torch.as_tensor(rendered.normal_axis_moment[local], device=device),
                            torch.as_tensor(rendered.relative_log_depth[local], device=device),
                            torch.as_tensor(rendered.log_depth_std[local], device=device),
                            torch.as_tensor(rendered.boundary[local], device=device),
                        ).cpu().numpy().astype(np.float16, copy=False)
                    )
            features.flush()
            timing_rows.append({"image_id": image_id, "render_seconds": render_seconds})
            print(json.dumps({
                "completed": output_row + 1, "total": int(source_rows.size),
                "image_id": image_id, "render_seconds": render_seconds,
            }), flush=True)
    del features
    os.replace(partial_path, output_path)
    manifest = {
        "artifact_type": FEATURE_SCHEMA,
        "feature_semantics": TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
        "feature_channels": list(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS),
        "feature_shape": list(shape),
        "feature_dtype": "float16",
        "feature_file": str(output_path.resolve()),
        "feature_file_sha256": file_sha256(output_path),
        "dataset_file_sha256": file_sha256(label_path),
        "dataset_content_sha256": label_metadata["content_sha256"],
        "source_dataset_file_sha256": file_sha256(source_path),
        "source_dataset_content_sha256": source_metadata["content_sha256"],
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_file_sha256": mapper_sha,
        "mean_render_seconds_per_query": float(np.mean([
            row["render_seconds"] for row in timing_rows
        ])),
        "normal_semantics": "camera_frame_unsigned_axis_second_moment_xx_yy_zz_xy_xz_yz_v1",
        "depth_semantics": "candidate_view_mass_centered_log_primitive_center_depth_v1",
        "uses_gt_for_training_candidate_generation": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_features": str(output_path),
        "output_manifest": str(manifest_path),
        "feature_file_sha256": manifest["feature_file_sha256"],
        "mean_render_seconds_per_query": manifest["mean_render_seconds_per_query"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
