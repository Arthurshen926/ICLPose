"""Phase-2 seq10 retention for frozen parent/child/fusion global scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.tools.vfm.evaluate_goal_maplet_global_parent_layout_streaming_seq10 import (
    EVALUATION_K, THRESHOLDS, _load_direct_labels_after_freeze,
    _rotation_error_deg,
)
from feature_extract.vfm.localization_goal_maplet.global_child_layout_streaming_contract import (
    MODES, RUN_SCHEMA, TOPK, load_hierarchy_score,
)
from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    EXPECTED_SEQ10_QUERY_COUNT, load_json_no_duplicate_keys,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256, file_sha256,
)


SCHEMA = "goal_maplet_global_hierarchy_layout_streaming_seq10_basin_retention_v1"
SELECTION_MODE_PRIORITY = ("child", "geometric_mean")


def _freeze_phase1(path: Path) -> tuple[dict, list[tuple[dict, dict]]]:
    report = load_json_no_duplicate_keys(path)
    unhashed = dict(report)
    content = unhashed.pop("content_sha256", "")
    rows = report.get("rows")
    if (
        report.get("artifact_type") != RUN_SCHEMA
        or report.get("query_route") != "seq10"
        or report.get("query_count") != EXPECTED_SEQ10_QUERY_COUNT
        or not isinstance(rows, list) or len(rows) != EXPECTED_SEQ10_QUERY_COUNT
        or content != canonical_json_sha256(unhashed)
        or report.get("control_only") is not True
        or report.get("production_eligible") is not False
        or report.get("score_before_label_contract", {}).get("phase2_labels_opened") is not False
    ):
        raise ValueError("hierarchy seq10 Phase-1 run contract differs")
    frozen = []
    for expected, row in enumerate(rows):
        artifact = Path(str(row.get("artifact", ""))).resolve()
        if int(row.get("query_index", -1)) != expected or file_sha256(artifact) != row.get("artifact_file_sha256"):
            raise ValueError("hierarchy Phase-1 score inventory differs")
        arrays, metadata = load_hierarchy_score(artifact)
        if (
            str(np.asarray(arrays["image_id"]).item()) != row.get("image_id")
            or metadata.get("query_index") != expected
            or metadata.get("content_sha256") != row.get("content_sha256")
        ):
            raise ValueError("hierarchy Phase-1 score row differs")
        frozen.append((arrays, metadata))
    image_ids = [str(np.asarray(value[0]["image_id"]).item()) for value in frozen]
    if image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("hierarchy Phase-1 image inventory differs")
    return report, frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1_run", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite hierarchy retention report")
    phase1_path = Path(args.phase1_run).resolve()
    phase1, frozen = _freeze_phase1(phase1_path)
    proposal_path = Path(args.proposal).resolve()
    proposal, proposal_metadata = load_all_parent_union_support(proposal_path)
    if (
        phase1.get("proposal_file_sha256") != file_sha256(proposal_path)
        or phase1.get("proposal_content_sha256") != proposal_metadata["content_sha256"]
    ):
        raise ValueError("hierarchy Phase-2 proposal differs")
    # This is the first point at which the GT-anchor artifact is resolved/opened.
    image_ids = [str(np.asarray(value[0]["image_id"]).item()) for value in frozen]
    direct_path = Path(args.direct_dataset).resolve()
    direct, direct_metadata = _load_direct_labels_after_freeze(
        direct_path, image_ids, str(phase1["physical_map_file_sha256"]),
        str(phase1["physical_map_content_sha256"]),
    )
    target = np.asarray(direct["candidate_poses_w2c"], dtype=np.float64)[:, 0]
    target_center = np.stack([camera_center_from_pose_w2c(value) for value in target])
    positions = np.asarray(proposal["lattice_origin_world"], np.float64) + (
        np.asarray(proposal["cell_indices_world"], np.float64) + 0.5
    ) * float(proposal["lattice_spacing_m"])
    rotations = np.asarray(proposal["orientation_rotations_w2c"], np.float64)
    raw_translation = np.min(
        np.linalg.norm(positions[None] - target_center[:, None], axis=2), axis=1,
    )
    raw_rotation = np.asarray([
        np.min(_rotation_error_deg(rotations, pose[:3, :3])) for pose in target
    ])
    metrics: dict[str, dict] = {}
    first_hit: dict[str, list[int]] = {}
    for mode in MODES:
        translation = np.empty((EXPECTED_SEQ10_QUERY_COUNT, TOPK), np.float64)
        rotation = np.empty_like(translation)
        for query_index, (arrays, _) in enumerate(frozen):
            position_rows = np.asarray(arrays[f"{mode}_top_position_factor_indices"], np.int64)
            orientation_rows = np.asarray(arrays[f"{mode}_top_orientation_factor_indices"], np.int64)
            translation[query_index] = np.linalg.norm(
                positions[position_rows] - target_center[query_index], axis=1,
            )
            rotation[query_index] = _rotation_error_deg(
                rotations[orientation_rows], target[query_index, :3, :3],
            )
        mode_metrics = {}
        for name, translation_limit, rotation_limit in THRESHOLDS:
            joint = (translation <= translation_limit) & (rotation <= rotation_limit)
            raw_joint = (raw_translation <= translation_limit) & (raw_rotation <= rotation_limit)
            first = np.where(np.any(joint, axis=1), np.argmax(joint, axis=1) + 1, -1)
            if name == "region_2m_45deg":
                first_hit[mode] = first.tolist()
            mode_metrics[name] = {
                "translation_limit_m": translation_limit,
                "rotation_limit_deg": rotation_limit,
                "raw_global_joint_hit_count": int(np.sum(raw_joint)),
                "by_k": {str(k): {
                    "position_hit_count": int(np.sum(np.any(translation[:, :k] <= translation_limit, axis=1))),
                    "orientation_hit_count": int(np.sum(np.any(rotation[:, :k] <= rotation_limit, axis=1))),
                    "joint_hit_count": int(np.sum(np.any(joint[:, :k], axis=1))),
                    "joint_hit_rate": float(np.mean(np.any(joint[:, :k], axis=1))),
                } for k in EVALUATION_K},
            }
        metrics[mode] = mode_metrics
    required = int(math.ceil(0.95 * EXPECTED_SEQ10_QUERY_COUNT - 1e-12))
    selected = next(
        ({"mode": mode, "k": k} for k in EVALUATION_K for mode in SELECTION_MODE_PRIORITY
         if metrics[mode]["region_2m_45deg"]["by_k"][str(k)]["joint_hit_count"] >= required),
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
        "evaluation_k_preregistered": list(EVALUATION_K),
        "selection_mode_priority_preregistered": list(SELECTION_MODE_PRIORITY),
        "selection_required_hits": required,
        "selection_threshold": 0.95,
        "selected": selected,
        "decision": "GO" if selected is not None else "KILL",
        "metrics": metrics,
        "region_2m45_first_hit_rank": first_hit,
        "phase_separation_audit": {
            "all_88_score_files_exact_loaded_and_hash_frozen_before_label_open": True,
            "phase1_api_has_no_label_or_pose_input": True,
            "only_diagnostic_candidate_zero_pose_consumed": True,
        },
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
        "output": str(output), "decision": report["decision"], "selected": selected,
        "region_2m45_hits": {
            mode: {str(k): metrics[mode]["region_2m_45deg"]["by_k"][str(k)]["joint_hit_count"] for k in EVALUATION_K}
            for mode in MODES
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
