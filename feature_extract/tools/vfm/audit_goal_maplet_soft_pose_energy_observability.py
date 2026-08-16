"""Audit local SE(3) observability of correspondence-free RADIO/3DGS energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    rotate_camera_local,
    score_soft_surface_pose_energy,
    translate_camera_world,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    render_canonical_surface_field,
)


def _camera(contributor: Path) -> ColmapCamera:
    with np.load(contributor, allow_pickle=False) as data:
        return ColmapCamera(
            0,
            int(data["camera_model_id"]),
            int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--retrieval_run", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_ids", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--translation_step_m", type=float, default=0.5)
    parser.add_argument("--rotation_step_deg", type=float, default=5.0)
    parser.add_argument("--radio_weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite pose-energy observability audit")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    run = json.loads(Path(args.retrieval_run).read_text())
    artifact_by_id = {str(row["image_id"]): Path(row["artifact"]) for row in run["rows"]}
    gt = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    axis_names = ("tx", "ty", "tz", "rx", "ry", "rz")
    axes = np.eye(3, dtype=np.float64)
    rows_out = []
    for image_id in args.image_ids:
        if image_id not in artifact_by_id or image_id not in gt:
            raise KeyError(f"missing retrieval/GT for {image_id}")
        contributor = Path(args.contributors) / image_id.replace("/", "__")
        contributor = contributor.with_suffix(contributor.suffix + ".npz")
        if not contributor.exists():
            raise FileNotFoundError(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        query = np.asarray(
            mapper.project(_load_raw_final(Path(str(metadata["token_path"])), "radio_final")).measurement_context,
            dtype=np.float32,
        )
        retrieval = PureRadioPhysicalRetrieval.load_npz(artifact_by_id[image_id])
        poses = [("gt", gt[image_id])]
        for index, axis in enumerate(axes):
            poses.append((f"{axis_names[index]}+", translate_camera_world(gt[image_id], axis * float(args.translation_step_m))))
            poses.append((f"{axis_names[index]}-", translate_camera_world(gt[image_id], -axis * float(args.translation_step_m))))
        # Replace the last six translation labels by three local rotation pairs.
        poses = poses[:7]
        for index, axis in enumerate(axes):
            poses.append((f"r{'xyz'[index]}+", rotate_camera_local(gt[image_id], axis, float(args.rotation_step_deg))))
            poses.append((f"r{'xyz'[index]}-", rotate_camera_local(gt[image_id], axis, -float(args.rotation_step_deg))))
        scores = []
        camera = _camera(contributor)
        for name, pose in poses:
            rendered = render_canonical_surface_field(
                physical, field, pose, camera,
                width=query.shape[2], height=query.shape[1],
                selected_child_rows=retrieval.scene_child_rows,
                device=str(args.device),
            )
            energy = score_soft_surface_pose_energy(
                query, retrieval, rendered, radio_weight=float(args.radio_weight)
            )
            scores.append({"probe": name, **energy.__dict__})
        by_name = {row["probe"]: row for row in scores}
        curvature = {}
        for axis_name in ("tx", "ty", "tz"):
            curvature[axis_name] = float(
                2.0 * by_name["gt"]["combined_score"]
                - by_name[axis_name + "+"]["combined_score"]
                - by_name[axis_name + "-"]["combined_score"]
            ) / float(args.translation_step_m) ** 2
        for axis_name in ("rx", "ry", "rz"):
            key = "r" + axis_name[1]
            curvature[axis_name] = float(
                2.0 * by_name["gt"]["combined_score"]
                - by_name[key + "+"]["combined_score"]
                - by_name[key + "-"]["combined_score"]
            ) / float(args.rotation_step_deg) ** 2
        ranked = sorted(scores, key=lambda row: (-float(row["combined_score"]), str(row["probe"])))
        rows_out.append({
            "image_id": image_id,
            "gt_probe_rank": 1 + next(i for i, row in enumerate(ranked) if row["probe"] == "gt"),
            "positive_curvature_axis_count": int(sum(value > 0.0 for value in curvature.values())),
            "curvature": curvature,
            "scores": scores,
        })
    report = {
        "artifact_type": "goal_maplet_soft_pose_energy_observability_audit_v1",
        "query_count": len(rows_out),
        "translation_step_m": float(args.translation_step_m),
        "rotation_step_deg": float(args.rotation_step_deg),
        "radio_weight": float(args.radio_weight),
        "mean_positive_curvature_axis_count": float(np.mean([row["positive_curvature_axis_count"] for row in rows_out])),
        "gt_top1_fraction": float(np.mean([row["gt_probe_rank"] == 1 for row in rows_out])),
        "claims": {
            "uses_alike": False,
            "uses_pnp": False,
            "uses_hard_correspondences": False,
            "uses_gt_only_to_place_observability_probes": True,
            "is_global_localization_result": False,
        },
        "rows": rows_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
