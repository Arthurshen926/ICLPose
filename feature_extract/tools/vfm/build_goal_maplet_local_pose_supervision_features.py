"""Render multi-scale candidate-relative local pose supervision features."""

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
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FULLTOKEN_POSE_RANKING_CHANNELS,
    FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    compact_fulltoken_pose_ranking_features,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.local_pose_supervision import (
    LOCAL_POSE_SUPERVISION_SEMANTICS,
    build_local_pose_supervision_candidates,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)


LABEL_SCHEMA = "goal_maplet_local_pose_supervision_dataset_v1"
FEATURE_SCHEMA = "goal_maplet_compact_fulltoken_pose_ranking_features_v1"


def _parse_route_limits(values: list[str]) -> list[tuple[str, int]]:
    result = []
    for value in values:
        route, separator, count = str(value).partition(":")
        if not separator or not route or not count.isdigit() or int(count) <= 0:
            raise ValueError("route limits must use ROUTE:POSITIVE_COUNT")
        result.append((route, int(count)))
    if len({route for route, _ in result}) != len(result):
        raise ValueError("route limits must be unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--route_limits", nargs="+", default=["seq13:64", "seq3:32", "seq5:32"])
    parser.add_argument("--output_features", required=True)
    parser.add_argument("--output_labels", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--render_batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    feature_path = Path(args.output_features)
    label_path = Path(args.output_labels)
    manifest_path = Path(args.output_manifest)
    partial_path = feature_path.with_suffix(".partial.npy")
    if feature_path.suffix != ".npy" or label_path.suffix != ".npz":
        raise ValueError("feature/label outputs must use .npy/.npz")
    if any(path.exists() for path in (feature_path, label_path, manifest_path, partial_path)):
        raise FileExistsError("refusing to overwrite local pose supervision artifacts")
    arrays, source_metadata = load_pose_candidate_dataset(
        dataset_path, require_rendered_targets=False,
    )
    limits = _parse_route_limits(list(args.route_limits))
    selected_rows = []
    for route, count in limits:
        rows = [
            row for row, value in enumerate(arrays["image_ids"].tolist())
            if str(value).split("/", 1)[0] == route
        ]
        if len(rows) < count:
            raise ValueError(f"route {route} lacks {count} supervision queries")
        selected_rows.extend(rows[:count])
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
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
        raise ValueError("local supervision dataset/map/field/mapper lineage differs")
    device = torch.device(str(args.device))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(device))
    mapper.model.to(device).eval()
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(device))
    contributors = _load_contributors(Path(args.contributors))
    example = build_local_pose_supervision_candidates(
        arrays["candidate_poses_w2c"][selected_rows[0], 0]
    )
    candidate_count = int(example[0].shape[0])
    feature_shape = (
        int(selected_rows.size), candidate_count,
        len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64,
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float16, shape=feature_shape,
    )
    image_ids = np.asarray(arrays["image_ids"][selected_rows], dtype=str)
    candidate_poses = np.zeros((selected_rows.size, candidate_count, 4, 4), dtype=np.float64)
    translation = np.zeros((selected_rows.size, candidate_count), dtype=np.float32)
    rotation = np.zeros_like(translation)
    timing_rows = []
    with torch.no_grad():
        for output_row, query_index in enumerate(selected_rows.tolist()):
            image_id = str(arrays["image_ids"][query_index])
            contributor_path = contributors.get(image_id)
            if contributor_path is None:
                raise ValueError(f"missing contributor camera for {image_id}")
            declared = Path(str(arrays["contributor_paths"][query_index])).resolve()
            if declared != contributor_path or file_sha256(declared) != str(
                arrays["contributor_file_sha256"][query_index]
            ):
                raise ValueError("local supervision contributor lineage differs")
            token_path = Path(str(arrays["radio_token_paths"][query_index])).resolve()
            if file_sha256(token_path) != str(arrays["radio_file_sha256"][query_index]):
                raise ValueError("local supervision RADIO lineage differs")
            with np.load(token_path, allow_pickle=False) as data:
                if set(data.files) != {"radio_final"}:
                    raise ValueError("local supervision RADIO members differ")
                raw_np = np.asarray(data["radio_final"], dtype=np.float32)
            query = mapper.model(torch.as_tensor(raw_np, device=device)[None])[0]
            query = query.permute(1, 2, 0).reshape(2304, 128)
            poses, translation_row, rotation_row = build_local_pose_supervision_candidates(
                arrays["candidate_poses_w2c"][query_index, 0]
            )
            candidate_poses[output_row] = poses
            translation[output_row] = translation_row
            rotation[output_row] = rotation_row
            render_seconds = 0.0
            for begin in range(0, candidate_count, int(args.render_batch_size)):
                rendered = scene.render_direct_canonical_grid_batch(
                    poses[begin:begin + int(args.render_batch_size)],
                    _camera(contributor_path),
                )
                render_seconds += float(rendered.total_seconds)
                for local in range(int(rendered.batch_size)):
                    feature[output_row, begin + local] = (
                        compact_fulltoken_pose_ranking_features(
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
                        ).cpu().numpy().astype(np.float16, copy=False)
                    )
            timing_rows.append({"image_id": image_id, "render_seconds": render_seconds})
            print(json.dumps({
                "completed": output_row + 1, "total": int(selected_rows.size),
                "image_id": image_id, "render_seconds": render_seconds,
            }), flush=True)
    feature.flush()
    del feature
    os.replace(partial_path, feature_path)
    label_arrays = {
        "image_ids": image_ids,
        "source_query_rows": selected_rows,
        "candidate_poses_w2c": candidate_poses,
        "translation_m": translation,
        "rotation_deg": rotation,
        "candidate_valid": np.ones(translation.shape, dtype=bool),
    }
    label_content_sha256 = arrays_sha256(label_arrays)
    label_metadata = {
        "artifact_type": LABEL_SCHEMA,
        "content_sha256": label_content_sha256,
        "source_dataset_file_sha256": file_sha256(dataset_path),
        "source_dataset_content_sha256": source_metadata["content_sha256"],
        "supervision_semantics": LOCAL_POSE_SUPERVISION_SEMANTICS,
        "route_limits": [{"route": route, "count": count} for route, count in limits],
        "uses_gt_for_training_candidate_generation": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    temporary_label = label_path.with_name(label_path.name + ".tmp.npz")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        temporary_label,
        **label_arrays,
        metadata_json=np.asarray(json.dumps(label_metadata, sort_keys=True)),
    )
    os.replace(temporary_label, label_path)
    manifest = {
        "artifact_type": FEATURE_SCHEMA,
        "feature_semantics": FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
        "feature_channels": list(FULLTOKEN_POSE_RANKING_CHANNELS),
        "feature_shape": list(feature_shape),
        "feature_dtype": "float16",
        "feature_file": str(feature_path.resolve()),
        "feature_file_sha256": file_sha256(feature_path),
        "dataset_file_sha256": file_sha256(label_path),
        "dataset_content_sha256": label_content_sha256,
        "local_pose_supervision_semantics": LOCAL_POSE_SUPERVISION_SEMANTICS,
        "mean_render_seconds_per_query": float(np.mean([
            row["render_seconds"] for row in timing_rows
        ])),
        "uses_gt_for_training_candidate_generation": True,
        "production_eligible": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": int(selected_rows.size),
        "candidate_count": candidate_count,
        "feature_file_sha256": manifest["feature_file_sha256"],
        "label_content_sha256": label_content_sha256,
        "mean_render_seconds_per_query": manifest["mean_render_seconds_per_query"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
