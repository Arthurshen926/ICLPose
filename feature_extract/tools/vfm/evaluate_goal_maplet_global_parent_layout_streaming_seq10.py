"""Phase-2 seq10 basin retention for frozen global streaming layout scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    EXPECTED_SEQ10_QUERY_COUNT,
    RETURNED_TOPK,
    RUN_SCHEMA,
    load_json_no_duplicate_keys,
    load_streaming_score,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_global_parent_layout_streaming_seq10_basin_retention_v1"
EVALUATION_K = (64, 256, 1024, 4096)
THRESHOLDS = (
    ("region_2m_45deg", 2.0, 45.0),
    ("loose_1m_10deg", 1.0, 10.0),
    ("strict_0_5m_5deg", 0.5, 5.0),
)
DIRECT_MEMBERS = {
    "image_ids.npy", "radio_token_paths.npy", "radio_file_sha256.npy",
    "contributor_paths.npy", "contributor_file_sha256.npy",
    "candidate_poses_w2c.npy", "translation_m.npy", "rotation_deg.npy",
    "candidate_valid.npy", "metadata_json.npy",
}


def _rotation_error_deg(rotation: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
    relative = np.asarray(rotation, dtype=np.float64) @ target_rotation.T
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
    )
    return np.degrees(np.arccos(cosine))


def _load_frozen_phase1(path: Path) -> tuple[dict, list[tuple[dict, dict]]]:
    report = load_json_no_duplicate_keys(path)
    unhashed = dict(report)
    content = unhashed.pop("content_sha256", "")
    rows = report.get("rows")
    if (
        report.get("artifact_type") != RUN_SCHEMA
        or report.get("query_route") != "seq10"
        or int(report.get("query_count", -1)) != EXPECTED_SEQ10_QUERY_COUNT
        or not isinstance(rows, list) or len(rows) != EXPECTED_SEQ10_QUERY_COUNT
        or content != canonical_json_sha256(unhashed)
        or report.get("control_only") is not True
        or report.get("production_eligible") is not False
        or report.get("score_before_label_contract", {}).get(
            "phase2_labels_opened"
        ) is not False
    ):
        raise ValueError("seq10 Phase-1 run contract differs")
    frozen = []
    image_ids = []
    # All score bytes and their rank contracts are verified before this
    # function returns.  The caller must not open the label dataset earlier.
    for expected_index, row in enumerate(rows):
        artifact = Path(str(row.get("artifact", ""))).resolve()
        if (
            int(row.get("query_index", -1)) != expected_index
            or file_sha256(artifact) != row.get("artifact_file_sha256")
        ):
            raise ValueError("Phase-1 score inventory differs")
        arrays, metadata = load_streaming_score(artifact)
        image_id = str(np.asarray(arrays["image_id"]).item())
        if (
            image_id != row.get("image_id")
            or metadata.get("query_index") != expected_index
            or metadata.get("content_sha256") != row.get("content_sha256")
            or metadata.get("proposal_content_sha256")
            != report.get("proposal_content_sha256")
        ):
            raise ValueError("Phase-1 score row binding differs")
        image_ids.append(image_id)
        frozen.append((arrays, metadata))
    if image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Phase-1 image inventory differs")
    return report, frozen


def _load_direct_labels_after_freeze(
    path: Path, expected_image_ids: list[str], physical_file: str, physical_content: str,
) -> tuple[dict[str, np.ndarray], dict]:
    artifact = Path(path)
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("direct label dataset is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != DIRECT_MEMBERS:
        raise ValueError("direct label exact NPZ members differ")
    arrays, metadata = load_pose_candidate_dataset(
        artifact, require_rendered_targets=False,
    )
    poses = np.asarray(arrays["candidate_poses_w2c"], dtype=np.float64)
    target = poses[:, 0]
    rotations = target[:, :3, :3]
    if (
        metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA
        or metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True
        or metadata.get("pose_errors_computed_only_after_candidate_freeze") is not True
        or metadata.get("physical_map_file_sha256") != physical_file
        or metadata.get("physical_map_sha256") != physical_content
        or np.asarray(arrays["image_ids"]).astype(str).tolist() != expected_image_ids
        or target.shape != (EXPECTED_SEQ10_QUERY_COUNT, 4, 4)
        or np.any(~np.isfinite(target))
        or not np.allclose(target[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        or not np.allclose(
            np.einsum("bij,bjk->bik", rotations.transpose(0, 2, 1), rotations),
            np.eye(3), atol=1e-7,
        )
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-7)
    ):
        raise ValueError("direct diagnostic GT-anchor contract differs")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1_run", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite seq10 retention evaluation")

    phase1_path = Path(args.phase1_run).resolve()
    # P0 phase boundary: freeze and exact-validate every score before even
    # resolving/opening the label dataset.
    phase1, frozen = _load_frozen_phase1(phase1_path)
    proposal_path = Path(args.proposal).resolve()
    proposal, proposal_metadata = load_all_parent_union_support(proposal_path)
    if (
        phase1.get("proposal_file_sha256") != file_sha256(proposal_path)
        or phase1.get("proposal_content_sha256") != proposal_metadata["content_sha256"]
    ):
        raise ValueError("Phase-2 proposal differs from frozen score run")
    physical_file = str(phase1["physical_map_file_sha256"])
    physical_content = str(phase1["physical_map_content_sha256"])
    image_ids = [str(row[1]["image_id"]) for row in frozen]
    direct_path = Path(args.direct_dataset).resolve()
    direct, direct_metadata = _load_direct_labels_after_freeze(
        direct_path, image_ids, physical_file, physical_content,
    )

    target = np.asarray(direct["candidate_poses_w2c"], dtype=np.float64)[:, 0]
    target_centers = np.stack([camera_center_from_pose_w2c(value) for value in target])
    origin = np.asarray(proposal["lattice_origin_world"], dtype=np.float64)
    all_positions = origin + (
        np.asarray(proposal["cell_indices_world"], dtype=np.float64) + 0.5
    ) * float(proposal["lattice_spacing_m"])
    all_rotations = np.asarray(proposal["orientation_rotations_w2c"], dtype=np.float64)
    translation = np.empty((EXPECTED_SEQ10_QUERY_COUNT, RETURNED_TOPK), dtype=np.float64)
    rotation = np.empty_like(translation)
    raw_translation = np.empty((EXPECTED_SEQ10_QUERY_COUNT,), dtype=np.float64)
    raw_rotation = np.empty_like(raw_translation)
    for query_index, (arrays, _) in enumerate(frozen):
        position_rows = np.asarray(arrays["top_position_factor_indices"], dtype=np.int64)
        orientation_rows = np.asarray(
            arrays["top_orientation_factor_indices"], dtype=np.int64,
        )
        translation[query_index] = np.linalg.norm(
            all_positions[position_rows] - target_centers[query_index], axis=1,
        )
        rotation[query_index] = _rotation_error_deg(
            all_rotations[orientation_rows], target[query_index, :3, :3],
        )
        raw_translation[query_index] = float(np.min(np.linalg.norm(
            all_positions - target_centers[query_index], axis=1,
        )))
        raw_rotation[query_index] = float(np.min(_rotation_error_deg(
            all_rotations, target[query_index, :3, :3],
        )))

    metrics = {}
    main_first_hit = np.full((EXPECTED_SEQ10_QUERY_COUNT,), -1, dtype=np.int64)
    for name, translation_limit, rotation_limit in THRESHOLDS:
        joint = (translation <= translation_limit) & (rotation <= rotation_limit)
        raw_position_hit = raw_translation <= translation_limit
        raw_orientation_hit = raw_rotation <= rotation_limit
        raw_joint_hit = raw_position_hit & raw_orientation_hit
        raw_joint_count = int(np.sum(raw_joint_hit))
        first = np.where(np.any(joint, axis=1), np.argmax(joint, axis=1) + 1, -1)
        if name == "region_2m_45deg":
            main_first_hit = first
        by_k = {}
        for budget in EVALUATION_K:
            position_hit = np.any(translation[:, :budget] <= translation_limit, axis=1)
            orientation_hit = np.any(rotation[:, :budget] <= rotation_limit, axis=1)
            joint_hit = np.any(joint[:, :budget], axis=1)
            by_k[str(budget)] = {
                "position_hit_count": int(np.sum(position_hit)),
                "orientation_hit_count": int(np.sum(orientation_hit)),
                "joint_hit_count": int(np.sum(joint_hit)),
                "joint_hit_rate": float(np.mean(joint_hit)),
                "conditional_retention_of_raw_global_support": (
                    float(np.sum(joint_hit) / raw_joint_count)
                    if raw_joint_count > 0 else None
                ),
            }
        metrics[name] = {
            "translation_limit_m": translation_limit,
            "rotation_limit_deg": rotation_limit,
            "by_k": by_k,
            "raw_global_support": {
                "position_hit_count": int(np.sum(raw_position_hit)),
                "orientation_hit_count": int(np.sum(raw_orientation_hit)),
                "joint_hit_count": raw_joint_count,
                "joint_hit_rate": float(np.mean(raw_joint_hit)),
            },
            "first_hit_rank": first.tolist(),
        }
    required = int(math.ceil(0.95 * EXPECTED_SEQ10_QUERY_COUNT - 1e-12))
    selected = next(
        (budget for budget in EVALUATION_K
         if metrics["region_2m_45deg"]["by_k"][str(budget)]["joint_hit_count"]
         >= required),
        None,
    )
    report = {
        "artifact_type": SCHEMA,
        "query_route": "seq10",
        "query_count": EXPECTED_SEQ10_QUERY_COUNT,
        "phase1_run": str(phase1_path),
        "phase1_run_file_sha256": file_sha256(phase1_path),
        "phase1_run_content_sha256": phase1["content_sha256"],
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": proposal_metadata["content_sha256"],
        "direct_dataset": str(direct_path),
        "direct_dataset_file_sha256": file_sha256(direct_path),
        "direct_dataset_content_sha256": direct_metadata["content_sha256"],
        "phase_separation_audit": {
            "all_88_score_files_exact_loaded_and_hash_frozen_before_label_open": True,
            "phase1_api_has_no_label_or_pose_input": True,
            "phase1_camera_from_pose_free_manifest": True,
            "direct_dataset_preexisted_score_run_allowed": True,
            "only_diagnostic_candidate_zero_pose_consumed": True,
            "nonanchor_candidate_poses_and_stored_error_arrays_consumed": False,
        },
        "evaluation_k_preregistered": list(EVALUATION_K),
        "selection_metric": "region_2m_45deg_joint_hit_count",
        "selection_required_hits": required,
        "selection_threshold": 0.95,
        "selected_k": selected,
        "decision": "GO" if selected is not None else "KILL",
        "metrics": metrics,
        "main_first_hit_rank_summary": {
            "hit_count_at_4096": int(np.sum(main_first_hit > 0)),
            "median_among_hits": (
                float(np.median(main_first_hit[main_first_hit > 0]))
                if np.any(main_first_hit > 0) else None
            ),
            "maximum_among_hits": (
                int(np.max(main_first_hit)) if np.any(main_first_hit > 0) else None
            ),
        },
        "raw_global_region_2m45_support_hit_count": metrics[
            "region_2m_45deg"
        ]["raw_global_support"]["joint_hit_count"],
        "held_route_labels_opened": False,
        "held_evaluation_authorized": selected is not None,
        "control_only": True,
        "production_eligible": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_optimizer": False,
        "uses_renderer": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output), "decision": report["decision"],
        "selected_k": selected,
        "region_2m45_hits": {
            key: value["joint_hit_count"]
            for key, value in metrics["region_2m_45deg"]["by_k"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
