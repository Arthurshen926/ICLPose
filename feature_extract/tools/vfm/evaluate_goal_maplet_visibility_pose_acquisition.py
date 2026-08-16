"""Evaluate pose-basin acquisition after pose-free child retrieval.

Ground-truth poses are opened only here.  The visibility atlas and retrieval
artifacts remain pose-free inputs, so this evaluator cannot improve their
ranking or fit a parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.pose_proposal import (
    _rotation_distance_degrees,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    diverse_pose_rows,
    score_visibility_pose_atlas,
)


THRESHOLDS = {
    "strict_0_5m_5deg": (0.5, 5.0),
    "loose_1m_10deg": (1.0, 10.0),
    "acquisition_2m_20deg": (2.0, 20.0),
    "wide_2m_45deg": (2.0, 45.0),
}
K_VALUES = (1, 5, 10, 32, 64)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_errors(poses_w2c: np.ndarray, target_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(poses_w2c, dtype=np.float64)
    target = np.asarray(target_w2c, dtype=np.float64).reshape(4, 4)
    centers = -np.swapaxes(pose[:, :3, :3], 1, 2) @ pose[:, :3, 3, None]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation = np.linalg.norm(centers[..., 0] - target_center[None], axis=1)
    rotation = np.asarray(
        [_rotation_distance_degrees(value, target) for value in pose], dtype=np.float64
    )
    return translation, rotation


def _hit(translation: np.ndarray, rotation: np.ndarray, threshold: tuple[float, float]) -> bool:
    return bool(np.any(
        (np.asarray(translation) <= float(threshold[0]))
        & (np.asarray(rotation) <= float(threshold[1]))
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--retrieval_run", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--layout_weight", type=float, default=0.5)
    parser.add_argument("--maximum_modes", type=int, default=32)
    parser.add_argument("--translation_nms_m", type=float, default=0.5)
    parser.add_argument("--rotation_nms_deg", type=float, default=5.0)
    parser.add_argument(
        "--all_token_candidates",
        action="store_true",
        help="diagnostic candidate-universe ceiling; ignore the selected child set",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite pose-acquisition evaluation")
    atlas_path = Path(args.atlas)
    retrieval_run_path = Path(args.retrieval_run)
    pose_path = Path(args.query_pose_file)
    atlas = ChildVisibilityPoseAtlas.load_npz(atlas_path)
    run = json.loads(retrieval_run_path.read_text())
    run_rows = run.get("rows")
    if run.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1" or not isinstance(run_rows, list):
        raise ValueError("not a pure RADIO retrieval run")
    gt_records = parse_cambridge_pose_file(pose_path)
    gt_by_id = {record.image_id: record.pose_w2c for record in gt_records}
    if len(gt_by_id) != len(gt_records):
        raise ValueError("query pose file contains duplicate image IDs")
    matrices = atlas.sparse_matrices()
    rows_out: list[dict[str, object]] = []
    recall_counts = {
        name: {str(k): 0 for k in K_VALUES} for name in THRESHOLDS
    }
    oracle_counts = {name: 0 for name in THRESHOLDS}
    top1_translation: list[float] = []
    top1_rotation: list[float] = []
    best_pool_translation: list[float] = []
    best_pool_rotation: list[float] = []
    for source in run_rows:
        image_id = str(source["image_id"])
        if image_id not in gt_by_id:
            raise KeyError(f"query pose file lacks {image_id}")
        artifact = Path(source["artifact"])
        retrieval = PureRadioPhysicalRetrieval.load_npz(artifact)
        if retrieval.image_id != image_id:
            raise ValueError("retrieval row and artifact image IDs differ")
        score, global_score, layout_score = score_visibility_pose_atlas(
            atlas,
            retrieval,
            layout_weight=float(args.layout_weight),
            selected_children_only=not bool(args.all_token_candidates),
            matrices=matrices,
        )
        selected = diverse_pose_rows(
            atlas.poses_w2c,
            score,
            maximum_modes=int(args.maximum_modes),
            translation_nms_m=float(args.translation_nms_m),
            rotation_nms_deg=float(args.rotation_nms_deg),
        )
        translation_all, rotation_all = _pose_errors(
            atlas.poses_w2c, gt_by_id[image_id]
        )
        translation = translation_all[selected]
        rotation = rotation_all[selected]
        for name, threshold in THRESHOLDS.items():
            oracle_counts[name] += int(_hit(translation_all, rotation_all, threshold))
            for k in K_VALUES:
                recall_counts[name][str(k)] += int(
                    _hit(translation[:k], rotation[:k], threshold)
                )
        top1_translation.append(float(translation[0]))
        top1_rotation.append(float(rotation[0]))
        scale = np.maximum(translation / 2.0, rotation / 45.0)
        best = int(np.argmin(scale))
        best_pool_translation.append(float(translation[best]))
        best_pool_rotation.append(float(rotation[best]))
        rows_out.append({
            "image_id": image_id,
            "retrieval_content_sha256": retrieval.content_sha256,
            "selected_child_count": int(retrieval.scene_child_rows.size),
            "mode_rows": selected.tolist(),
            "mode_scores": score[selected].astype(float).tolist(),
            "mode_global_scores": global_score[selected].astype(float).tolist(),
            "mode_layout_scores": layout_score[selected].astype(float).tolist(),
            "translation_m": translation.astype(float).tolist(),
            "rotation_deg": rotation.astype(float).tolist(),
            "oracle_best_translation_m": float(translation_all[np.argmin(np.maximum(translation_all / 2.0, rotation_all / 45.0))]),
            "oracle_best_rotation_deg": float(rotation_all[np.argmin(np.maximum(translation_all / 2.0, rotation_all / 45.0))]),
        })
    query_count = len(rows_out)
    if query_count != len(run_rows) or query_count == 0:
        raise ValueError("empty or incomplete pose-acquisition evaluation")
    metrics = {}
    for name in THRESHOLDS:
        values = {
            f"recall_at_{k}": float(recall_counts[name][str(k)] / query_count)
            for k in K_VALUES
        }
        values["atlas_oracle_recall"] = float(oracle_counts[name] / query_count)
        metrics[name] = values
    report = {
        "artifact_type": "goal_maplet_visibility_pose_acquisition_evaluation_v1",
        "method": "pure_radio_child_set_to_feature_free_visibility_chart_centres",
        "query_count": query_count,
        "atlas": str(atlas_path.resolve()),
        "atlas_file_sha256": _file_sha256(atlas_path),
        "atlas_content_sha256": atlas.content_sha256,
        "retrieval_run": str(retrieval_run_path.resolve()),
        "retrieval_run_file_sha256": _file_sha256(retrieval_run_path),
        "query_pose_file": str(pose_path.resolve()),
        "query_pose_file_sha256": _file_sha256(pose_path),
        "gt_access_scope": "evaluation_only_after_pose_free_scores_and_nms",
        "score_semantics": "uncalibrated_proposal_affinity_not_pose_posterior",
        "layout_weight": float(args.layout_weight),
        "selected_children_only": not bool(args.all_token_candidates),
        "maximum_modes": int(args.maximum_modes),
        "translation_nms_m": float(args.translation_nms_m),
        "rotation_nms_deg": float(args.rotation_nms_deg),
        "metrics": metrics,
        "error_summary": {
            "top1_median_translation_m": float(np.median(top1_translation)),
            "top1_median_rotation_deg": float(np.median(top1_rotation)),
            "best_pool_median_translation_m": float(np.median(best_pool_translation)),
            "best_pool_median_rotation_deg": float(np.median(best_pool_rotation)),
            "best_pool_size": int(args.maximum_modes),
            "top1_p90_translation_m": float(np.percentile(top1_translation, 90.0)),
            "top1_p90_rotation_deg": float(np.percentile(top1_rotation, 90.0)),
        },
        "claims": {
            "uses_alike": False,
            "uses_pnp": False,
            "uses_query_pose_for_ranking": False,
            "uses_mapping_rgb_or_image_retrieval": False,
            "chart_centres_are_final_pose": False,
            "measures_pose_basin_acquisition_not_localization_success": True,
        },
        "rows": rows_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
