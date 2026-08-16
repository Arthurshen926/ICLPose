"""Verify retrieved pose modes with correspondence-free RADIO/3DGS energy.

The query pose is opened only after both the child-atlas proposal scores and
the rendered soft-surface scores have been computed.  This is a bounded
diagnostic of the retrieval-to-pose handoff, not a localization benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_proposal import _rotation_distance_degrees
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    score_soft_surface_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    render_canonical_surface_field,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    diverse_pose_rows,
    score_visibility_pose_atlas,
)


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            0,
            int(data["camera_model_id"]),
            int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _pose_error(pose: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target, dtype=np.float64).reshape(4, 4)
    center = -pose[:3, :3].T @ pose[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    return (
        float(np.linalg.norm(center - target_center)),
        float(_rotation_distance_degrees(pose, target)),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--retrieval_run", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--radio_weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite soft pose verification")
    atlas = ChildVisibilityPoseAtlas.load_npz(Path(args.atlas))
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    mapper_path = Path(args.surface_mapper)
    mapper, mapper_metadata = load_surface_maplet_mapper(
        mapper_path, device=str(args.device)
    )
    run = json.loads(Path(args.retrieval_run).read_text())
    artifact = next(
        (Path(row["artifact"]) for row in run["rows"] if row["image_id"] == args.image_id),
        None,
    )
    if artifact is None:
        raise KeyError(f"retrieval run lacks {args.image_id}")
    retrieval = PureRadioPhysicalRetrieval.load_npz(artifact)
    contributor = Path(args.contributors) / args.image_id.replace("/", "__")
    contributor = contributor.with_suffix(contributor.suffix + ".npz")
    with np.load(contributor, allow_pickle=False) as data:
        contributor_metadata = json.loads(str(data["metadata_json"].item()))
    raw = _load_raw_final(Path(str(contributor_metadata["token_path"])), "radio_final")
    query = np.asarray(mapper.project(raw).measurement_context, dtype=np.float32)
    proposal, global_score, _ = score_visibility_pose_atlas(
        atlas, retrieval, layout_weight=0.0, selected_children_only=True
    )
    mode_rows = diverse_pose_rows(
        atlas.poses_w2c, proposal, maximum_modes=int(args.candidate_count)
    )
    camera = _camera(contributor)
    candidate_rows = []
    for proposal_rank, atlas_row in enumerate(mode_rows.tolist(), start=1):
        rendered = render_canonical_surface_field(
            physical,
            field,
            atlas.poses_w2c[atlas_row],
            camera,
            width=query.shape[2],
            height=query.shape[1],
            selected_child_rows=retrieval.scene_child_rows,
            device=str(args.device),
        )
        energy = score_soft_surface_pose_energy(
            query, retrieval, rendered, radio_weight=float(args.radio_weight)
        )
        candidate_rows.append({
            "atlas_row": int(atlas_row),
            "proposal_rank": int(proposal_rank),
            "proposal_score": float(proposal[atlas_row]),
            "global_child_score": float(global_score[atlas_row]),
            **energy.__dict__,
        })
    order = sorted(
        range(len(candidate_rows)),
        key=lambda row: (-candidate_rows[row]["combined_score"], candidate_rows[row]["atlas_row"]),
    )
    gt = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.query_pose_file))
    }
    if args.image_id not in gt:
        raise KeyError(f"query pose file lacks {args.image_id}")
    for rerank, row in enumerate(order, start=1):
        translation, rotation = _pose_error(
            atlas.poses_w2c[candidate_rows[row]["atlas_row"]], gt[args.image_id]
        )
        candidate_rows[row].update({
            "soft_energy_rank": int(rerank),
            "translation_m": translation,
            "rotation_deg": rotation,
        })
    proposal_top = candidate_rows[0]
    energy_top = candidate_rows[order[0]]
    report = {
        "artifact_type": "goal_maplet_soft_pose_verification_evaluation_v1",
        "image_id": args.image_id,
        "candidate_count": len(candidate_rows),
        "radio_weight": float(args.radio_weight),
        "proposal_top1_translation_m": proposal_top["translation_m"],
        "proposal_top1_rotation_deg": proposal_top["rotation_deg"],
        "soft_energy_top1_translation_m": energy_top["translation_m"],
        "soft_energy_top1_rotation_deg": energy_top["rotation_deg"],
        "soft_energy_improved_joint_scale": bool(
            max(energy_top["translation_m"] / 2.0, energy_top["rotation_deg"] / 45.0)
            < max(proposal_top["translation_m"] / 2.0, proposal_top["rotation_deg"] / 45.0)
        ),
        "mapper_identity": {
            "path": str(mapper_path.resolve()),
            "file_sha256": _file_sha256(mapper_path),
            "supervision": mapper_metadata.get("supervision"),
            "vfm_layer": mapper_metadata.get("vfm_layer"),
        },
        "claims": {
            "uses_alike": False,
            "uses_pnp": False,
            "uses_hard_correspondences": False,
            "uses_query_pose_for_scoring": False,
            "gt_opened_only_after_all_candidate_scores": True,
            "is_full_localization_benchmark": False,
        },
        "candidates": candidate_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
