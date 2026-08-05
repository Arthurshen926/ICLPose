"""Measure the convergence basin of continuous Goal-Maplet surface alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.canonical_codec import CanonicalRadioCodec
from feature_extract.vfm.localization_goal_maplet.local_head import load_child_local_head
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels, _primitive_to_child_links
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_refiner import refine_pose_with_canonical_surface
from feature_extract.vfm.localization_v6.se3_update import se3_exp
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            0,
            int(data["camera_model_id"]),
            int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _oracle_children(labels: ContributorLabels, physical: GoalMapletPhysicalMap) -> np.ndarray:
    row_by_id = {int(value): row for row, value in enumerate(physical.primitive_ids.tolist())}
    offsets, children, _ = _primitive_to_child_links(physical)
    selected: set[int] = set()
    for primitive_id in np.unique(labels.topk_primitive_ids).tolist():
        row = row_by_id.get(int(primitive_id))
        if row is not None:
            selected.update(children[int(offsets[row]) : int(offsets[row + 1])].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _parse_perturbations(value: str) -> list[tuple[float, float]]:
    result = []
    for item in str(value).split(","):
        translation, rotation = item.split(":", 1)
        result.append((float(translation), float(rotation)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--canonical_codec", default="")
    parser.add_argument("--child_local_head", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--candidate_surface_mode", choices=("oracle_children", "all_field"), default="oracle_children")
    parser.add_argument("--perturbations", default="0:0,0.1:0,0.2:0,0.3:0,0:1,0:3,0.1:1")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--render_supersample_factor", type=int, default=1)
    parser.add_argument("--acceptance_supersample_factor", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite surface-basin report")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    local_head = None
    if args.child_local_head:
        artifact = load_child_local_head(Path(args.child_local_head), device=str(args.device))
        if str(artifact.metadata.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("child-local head and physical map lineage differ")
        if str(artifact.metadata.get("canonical_field_sha256", "")) != field.content_sha256:
            raise ValueError("child-local head and canonical field lineage differ")
        local_head = artifact.model
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    codec = CanonicalRadioCodec.load_npz(Path(args.canonical_codec)) if args.canonical_codec else None
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        paths = paths[: int(args.maximum_queries)]
    translation_direction = np.asarray([1.0, 0.5, -0.25], dtype=np.float64)
    translation_direction /= np.linalg.norm(translation_direction)
    rotation_direction = np.asarray([0.2, 1.0, 0.1], dtype=np.float64)
    rotation_direction /= np.linalg.norm(rotation_direction)
    perturbations = _parse_perturbations(args.perturbations)
    rows = []
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        if codec is not None:
            if str(field.metadata.get("canonical_codec_sha256", "")) != codec.content_sha256:
                raise ValueError("canonical codec and field lineage differ")
            query = codec.transform_map(raw)
        elif int(field.feature_dim) == int(raw.shape[0]):
            query = raw / np.maximum(np.linalg.norm(raw, axis=0, keepdims=True), 1e-8)
        else:
            query = mapper.project(raw).measurement_context
        if int(query.shape[0]) != int(field.feature_dim):
            raise ValueError("query feature and canonical field dimensions differ")
        selected_children = (
            _oracle_children(labels, physical)
            if str(args.candidate_surface_mode) == "oracle_children"
            else None
        )
        for translation_m, rotation_deg in perturbations:
            delta = np.concatenate([
                rotation_direction * np.deg2rad(rotation_deg),
                translation_direction * translation_m,
            ])
            initial_pose = se3_exp(delta) @ labels.pose_w2c
            initial_error = pnp_pose_error(initial_pose, labels.pose_w2c)
            refined = refine_pose_with_canonical_surface(
                query,
                initial_pose,
                camera,
                physical,
                field,
                selected_child_rows=selected_children,
                local_head=local_head,
                rounds=int(args.rounds),
                render_supersample_factor=int(args.render_supersample_factor),
                acceptance_supersample_factor=(
                    int(args.acceptance_supersample_factor)
                    if int(args.acceptance_supersample_factor) > 0 else None
                ),
                device=str(args.device),
            )
            final_error = pnp_pose_error(refined.pose_w2c, labels.pose_w2c)
            row = {
                "image_id": str(metadata["image_id"]),
                "translation_perturbation_m": translation_m,
                "rotation_perturbation_deg": rotation_deg,
                "selected_child_count": int(selected_children.size) if selected_children is not None else None,
                "initial_translation_m": float(initial_error.translation_m),
                "initial_rotation_deg": float(initial_error.rotation_deg),
                "final_translation_m": float(final_error.translation_m),
                "final_rotation_deg": float(final_error.rotation_deg),
                "improved_translation": bool(final_error.translation_m < initial_error.translation_m),
                "improved_rotation": bool(final_error.rotation_deg < initial_error.rotation_deg),
                "success": bool(refined.success),
                "history": list(refined.history),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for translation_m, rotation_deg in perturbations:
        selected = [
            row for row in rows
            if row["translation_perturbation_m"] == translation_m
            and row["rotation_perturbation_deg"] == rotation_deg
        ]
        key = f"{translation_m:g}m_{rotation_deg:g}deg"
        summary[key] = {
            "count": len(selected),
            "final_translation_median_m": float(np.median([row["final_translation_m"] for row in selected])) if selected else None,
            "final_translation_p90_m": float(np.percentile([row["final_translation_m"] for row in selected], 90.0)) if selected else None,
            "final_rotation_median_deg": float(np.median([row["final_rotation_deg"] for row in selected])) if selected else None,
            "final_rotation_p90_deg": float(np.percentile([row["final_rotation_deg"] for row in selected], 90.0)) if selected else None,
            "translation_improvement_fraction": float(np.mean([row["improved_translation"] for row in selected])) if selected else 0.0,
            "rotation_improvement_fraction": float(np.mean([row["improved_rotation"] for row in selected])) if selected else 0.0,
        }
    result = {
        "stage": "goal_maplet_continuous_surface_refiner_basin",
        "query_count": len(paths),
        "candidate_surface_mode": str(args.candidate_surface_mode),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "child_local_head": str(args.child_local_head) if args.child_local_head else None,
        "render_supersample_factor": int(args.render_supersample_factor),
        "acceptance_supersample_factor": int(args.acceptance_supersample_factor),
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
