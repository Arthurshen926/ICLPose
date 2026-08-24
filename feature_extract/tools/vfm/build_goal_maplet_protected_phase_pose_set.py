"""Build a replayable protected, physically deduplicated phase pose set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import _pose_errors
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


SCHEMA = "goal_maplet_protected_phase_pose_set_v1"
SUPPORTED_PHASE_SCORE_SEMANTICS = (
    "conditional_phase_control",
    "conditional_phase_shift_control",
    "conservative_phase",
    "conservative_phase_shift",
    "conservative_phase_maxmin",
    "conservative_phase_shift_maxmin",
)


def _same_basin(a: np.ndarray, b: np.ndarray) -> bool:
    translation, rotation = _pose_errors(np.asarray(a)[None], np.asarray(b))
    return bool(translation[0] <= 0.5 + 1.0e-6 and rotation[0] <= 5.0 + 1.0e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase_report", required=True)
    parser.add_argument("--refinement_report", action="append", default=[])
    parser.add_argument(
        "--expected_phase_score_semantics",
        choices=SUPPORTED_PHASE_SCORE_SEMANTICS,
        required=True,
    )
    parser.add_argument(
        "--expected_refinement_score_semantics",
        choices=("conservative_phase", "conditional_phase"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int)
    parser.add_argument("--maximum_seed_basins", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite protected phase pose set")
    dataset_path = Path(args.dataset)
    arrays, _ = load_pose_candidate_dataset(dataset_path)
    dataset_sha = file_sha256(dataset_path)
    phase_path = Path(args.phase_report)
    phase = json.loads(phase_path.read_text())
    if (
        phase.get("dataset_file_sha256") != dataset_sha
        or phase.get("score_semantics") != str(args.expected_phase_score_semantics)
        or phase.get("all_candidate_scores_built_before_pose_error_metrics") is not True
    ):
        raise ValueError("phase report differs from the dataset/scorer")
    phase_score = np.asarray(phase["candidate_score"], dtype=np.float64)
    if phase_score.shape != arrays["candidate_valid"].shape:
        raise ValueError("phase score matrix differs from the candidate inventory")
    expected_shift_radius = (
        int(phase.get("local_radius_tokens", 0))
        if "phase_shift" in str(args.expected_phase_score_semantics)
        else 0
    )
    refinements: dict[str, list[dict[str, object]]] = {}
    refinement_bindings = []
    if args.refinement_report and args.expected_refinement_score_semantics is None:
        raise ValueError("refinement reports require an explicit expected scorer")
    for value in args.refinement_report:
        path = Path(value)
        report = json.loads(path.read_text())
        if (
            report.get("dataset_file_sha256") != dataset_sha
            or report.get("score_semantics") != str(args.expected_refinement_score_semantics)
            or report.get("uses_alike") is not False
            or report.get("uses_pnp") is not False
            or int(report.get("phase_shift_radius_tokens", 0)) != expected_shift_radius
        ):
            raise ValueError("refinement report differs from the protected phase contract")
        refinement_bindings.append({"path": str(path.resolve()), "file_sha256": file_sha256(path)})
        for row in report["rows"]:
            if "initial_pose_w2c" not in row or "final_pose_w2c" not in row:
                raise ValueError("refinement report lacks replayable poses")
            refinements.setdefault(str(row["image_id"]), []).append(row)
    begin = int(args.query_start)
    end = int(arrays["image_ids"].size)
    if args.maximum_queries is not None:
        end = min(end, begin + int(args.maximum_queries))
    rows = []
    for query in range(begin, end):
        image_id = str(arrays["image_ids"][query])
        valid = np.flatnonzero(arrays["candidate_valid"][query])
        order = valid[valid != 0]
        order = order[np.argsort(-phase_score[query, order], kind="stable")]
        # The retrieval first proposal is protected. Fill the remaining seed
        # budget with the best phase-ranked physically independent basins.
        seed_indices = [1]
        for candidate in order.tolist():
            if candidate == 1 or any(_same_basin(
                arrays["candidate_poses_w2c"][query, candidate],
                arrays["candidate_poses_w2c"][query, previous],
            ) for previous in seed_indices):
                continue
            seed_indices.append(int(candidate))
            if len(seed_indices) >= int(args.maximum_seed_basins):
                break
        hypotheses = [
            {
                "pose_w2c": np.asarray(arrays["candidate_poses_w2c"][query, candidate]),
                "score": float(phase_score[query, candidate]),
                "role": "protected_seed",
                "candidate_index": int(candidate),
                "source": "dataset",
            }
            for candidate in seed_indices
        ]
        for refinement in refinements.get(image_id, []):
            candidate = refinement.get("initial_candidate_index")
            if candidate is None or int(candidate) not in seed_indices:
                # Only refinement of a protected seed belongs to this set.
                continue
            initial = np.asarray(refinement["initial_pose_w2c"], dtype=np.float64)
            if not np.allclose(
                initial, arrays["candidate_poses_w2c"][query, int(candidate)], atol=1.0e-12, rtol=0.0
            ):
                raise ValueError("refinement initial pose differs from its protected seed")
            hypotheses.append({
                "pose_w2c": np.asarray(refinement["final_pose_w2c"], dtype=np.float64),
                "score": float(refinement["final_score"]),
                "role": "refined_hypothesis",
                "candidate_index": int(candidate),
                "source": "refinement_report",
            })
        ranked = sorted(
            enumerate(hypotheses), key=lambda item: (-float(item[1]["score"]), item[0])
        )
        retained: list[dict[str, object]] = []
        for _, hypothesis in ranked:
            if any(_same_basin(hypothesis["pose_w2c"], old["pose_w2c"]) for old in retained):
                continue
            retained.append(hypothesis)
        # GT is opened only after the pose-free set and physical NMS are frozen.
        poses = np.stack([np.asarray(item["pose_w2c"]) for item in retained])
        translation, rotation = _pose_errors(poses, arrays["candidate_poses_w2c"][query, 0])
        serial = []
        for item, t, r in zip(retained, translation.tolist(), rotation.tolist()):
            serial.append({
                **{key: value for key, value in item.items() if key != "pose_w2c"},
                "pose_w2c": np.asarray(item["pose_w2c"]).tolist(),
                "translation_m": float(t), "rotation_deg": float(r),
            })
        rows.append({
            "image_id": image_id,
            "protected_seed_candidate_indices": seed_indices,
            "pre_nms_hypothesis_count": len(hypotheses),
            "retained_physical_basin_count": len(retained),
            "strict_acquired": bool(np.any((translation <= 0.5 + 1.0e-6) & (rotation <= 5.0 + 1.0e-5))),
            "loose_acquired": bool(np.any((translation <= 1.0 + 1.0e-6) & (rotation <= 10.0 + 1.0e-5))),
            "retained": serial,
        })
    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": dataset_sha,
        "phase_report": {"path": str(phase_path.resolve()), "file_sha256": file_sha256(phase_path)},
        "phase_score_semantics": str(args.expected_phase_score_semantics),
        "phase_energy_semantics": str(phase.get("energy_semantics")),
        "phase_shift_radius_tokens": expected_shift_radius,
        "refinement_score_semantics": args.expected_refinement_score_semantics,
        "refinement_reports": refinement_bindings,
        "query_range": [begin, end],
        "maximum_seed_basins": int(args.maximum_seed_basins),
        "seed_selection": "retrieval_first_plus_best_phase_ranked_independent_basins_v1",
        "physical_nms": {"translation_m": 0.5, "rotation_deg": 5.0, "score_ordered": True},
        "query_count": len(rows),
        "strict_acquisition_rate": float(np.mean([row["strict_acquired"] for row in rows])),
        "loose_acquisition_rate": float(np.mean([row["loose_acquired"] for row in rows])),
        "mean_retained_physical_basin_count": float(np.mean([
            row["retained_physical_basin_count"] for row in rows
        ])),
        "rows": rows,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "selection_or_top1_claim": False,
        "production_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "query_count", "strict_acquisition_rate", "loose_acquisition_rate",
        "mean_retained_physical_basin_count",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
