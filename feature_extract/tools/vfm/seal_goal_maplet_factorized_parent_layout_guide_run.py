"""Seal a complete score-before-label parent-layout guide run.

The command has deliberately no label, contributor, or target-pose argument.
It validates every per-query score artifact, its strict retrieval lineage, and
the frozen factor proposal, then publishes a content-hashed run inventory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_factorized_parent_layout_guide import (
    _load_score,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    camera_intrinsics_content_sha256,
    load_pose_free_candidate_pool,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)


SCHEMA = "goal_maplet_factorized_parent_support_layout_guide_phase1_run_v1"


def seal_parent_layout_score_run(
    *,
    scores_dir: Path,
    proposal_path: Path,
    require_strict_v4: bool,
    parallel_runner_wall_seconds: float | None = None,
) -> dict[str, object]:
    if parallel_runner_wall_seconds is not None and (
        not np.isfinite(float(parallel_runner_wall_seconds))
        or float(parallel_runner_wall_seconds) <= 0.0
    ):
        raise ValueError("parallel runner wall time must be finite and positive")
    proposal_path = Path(proposal_path).resolve()
    factors, proposal_metadata = load_factorized_pose_free_proposal(proposal_path)
    image_ids = [str(value) for value in np.asarray(factors["image_ids"]).tolist()]
    pool_path = Path(str(proposal_metadata["candidate_pool"])).resolve()
    pool = load_pose_free_candidate_pool(pool_path)
    pool_rows = {str(row["image_id"]): row for row in pool["rows"]}
    paths = sorted(Path(scores_dir).resolve().glob("*.npz"))
    if len(paths) != len(image_ids):
        raise ValueError("parent layout Phase-1 score count differs from proposal")

    rows_by_index: dict[int, dict[str, object]] = {}
    physical_lineage: tuple[str, str, str] | None = None
    total_pair_count: int | None = None
    returned_topk: int | None = None
    for path in paths:
        score, metadata = _load_score(path)
        query_index = int(metadata.get("query_index", -1))
        image_id = str(np.asarray(score["image_id"]).item())
        if (
            query_index < 0
            or query_index >= len(image_ids)
            or query_index in rows_by_index
            or image_ids[query_index] != image_id
            or metadata.get("image_id") != image_id
            or Path(str(metadata.get("proposal", ""))).resolve() != proposal_path
            or metadata.get("proposal_file_sha256") != file_sha256(proposal_path)
            or metadata.get("proposal_content_sha256")
            != proposal_metadata.get("content_sha256")
        ):
            raise ValueError("parent layout Phase-1 query/proposal binding differs")
        if require_strict_v4 and (
            metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed") is not True
            or metadata.get("protocol_status")
            != "strict_v4_seq10_calibrated_route_disjoint_retrieval"
        ):
            raise ValueError("parent layout Phase-1 score is not strict v4")
        expected_camera_hash = camera_intrinsics_content_sha256(
            image_id,
            int(metadata["camera_model_id"]),
            int(metadata["camera_width"]),
            int(metadata["camera_height"]),
            tuple(float(value) for value in metadata["camera_params"]),
        )
        if expected_camera_hash != metadata.get("camera_intrinsics_only_content_sha256"):
            raise ValueError("parent layout Phase-1 camera binding differs")
        pool_row = pool_rows.get(image_id)
        retrieval_path = Path(str(metadata.get("retrieval", ""))).resolve()
        if (
            pool_row is None
            or retrieval_path
            != Path(str(pool_row.get("retrieval_artifact", ""))).resolve()
            or metadata.get("retrieval_content_sha256")
            != pool_row.get("retrieval_content_sha256")
            or metadata.get("retrieval_file_sha256") != file_sha256(retrieval_path)
        ):
            raise ValueError("parent layout Phase-1 retrieval/pool binding differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if retrieval.content_sha256 != metadata.get("retrieval_content_sha256"):
            raise ValueError("parent layout Phase-1 retrieval content differs")

        lineage = (
            str(metadata.get("physical_map", "")),
            str(metadata.get("physical_map_file_sha256", "")),
            str(metadata.get("physical_map_content_sha256", "")),
        )
        if physical_lineage is None:
            physical_lineage = lineage
        elif lineage != physical_lineage:
            raise ValueError("parent layout Phase-1 physical-map lineage differs")
        pair_count = int(metadata.get("total_factor_pair_count", -1))
        topk = int(metadata.get("returned_topk", -1))
        if total_pair_count is None:
            total_pair_count = pair_count
            returned_topk = topk
        elif pair_count != total_pair_count or topk != returned_topk:
            raise ValueError("parent layout Phase-1 factor budget differs by query")
        if (
            pair_count <= 0
            or topk <= 0
            or int(np.asarray(score["top_scores"]).size) != topk
            or metadata.get("contributor_file_bytes_hashed_during_scoring", False)
            or metadata.get("contributor_pose_member_opened_during_scoring", False)
            or metadata.get("uses_query_pose") is not False
            or metadata.get("uses_query_ground_truth") is not False
        ):
            raise ValueError("parent layout Phase-1 score boundary differs")
        rows_by_index[query_index] = {
            "query_index": query_index,
            "image_id": image_id,
            "score": str(path.resolve()),
            "score_file_sha256": file_sha256(path),
            "score_content_sha256": str(metadata["content_sha256"]),
            "score_file_mtime_ns": int(path.stat().st_mtime_ns),
            "camera_intrinsics_only_content_sha256": expected_camera_hash,
            "retrieval": str(retrieval_path),
            "retrieval_file_sha256": str(metadata["retrieval_file_sha256"]),
            "retrieval_content_sha256": str(metadata["retrieval_content_sha256"]),
            "elapsed_seconds": float(metadata["elapsed_seconds"]),
        }
    if sorted(rows_by_index) != list(range(len(image_ids))):
        raise ValueError("parent layout Phase-1 query indices are not complete")
    rows = [rows_by_index[index] for index in range(len(image_ids))]
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_count": len(rows),
        "query_route": str(proposal_metadata.get("query_route", "")),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(proposal_metadata["content_sha256"]),
        "candidate_pool": str(pool_path),
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "candidate_pool_content_sha256": str(pool["content_sha256"]),
        "strict_v4_required": bool(require_strict_v4),
        "strict_v4_seq10_calibrated_retrieval_confirmed": bool(
            require_strict_v4
            and proposal_metadata.get(
                "strict_v4_seq10_calibrated_retrieval_confirmed"
            ) is True
        ),
        "physical_map": physical_lineage[0] if physical_lineage else None,
        "physical_map_file_sha256": physical_lineage[1] if physical_lineage else None,
        "physical_map_content_sha256": physical_lineage[2] if physical_lineage else None,
        "total_factor_pair_count_per_query": int(total_pair_count or 0),
        "returned_topk_per_query": int(returned_topk or 0),
        "total_elapsed_seconds": float(sum(float(row["elapsed_seconds"]) for row in rows)),
        "parallel_runner_wall_seconds": (
            None
            if parallel_runner_wall_seconds is None
            else float(parallel_runner_wall_seconds)
        ),
        "latest_score_file_mtime_ns": max(int(row["score_file_mtime_ns"]) for row in rows),
        "score_before_label_contract": {
            "label_or_direct_dataset_argument_accepted": False,
            "query_pose_or_gt_argument_accepted": False,
            "contributor_argument_accepted": False,
            "contributor_file_bytes_hashed": False,
            "contributor_pose_member_opened": False,
            "camera_binding_is_intrinsics_and_image_id_only": True,
            "ranked_scores_are_frozen_by_file_and_content_hash": True,
        },
        "rows": rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--require_strict_v4", action="store_true")
    parser.add_argument("--parallel_runner_wall_seconds", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent layout Phase-1 run")
    report = seal_parent_layout_score_run(
        scores_dir=Path(args.scores_dir),
        proposal_path=Path(args.proposal),
        require_strict_v4=bool(args.require_strict_v4),
        parallel_runner_wall_seconds=args.parallel_runner_wall_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "artifact_type", "query_count", "query_route", "content_sha256",
            "total_factor_pair_count_per_query", "returned_topk_per_query",
            "total_elapsed_seconds",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
