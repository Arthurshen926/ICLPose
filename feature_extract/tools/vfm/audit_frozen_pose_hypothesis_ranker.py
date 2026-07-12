"""Audit frozen multi-hypothesis rank regret against the l97 pose backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.fit_eval_pose_backend_fallback_gate import (
    _examples_for_run,
    _load_ranker_and_config,
)
from feature_extract.tools.vfm.fit_eval_pose_hypothesis_ranker import (
    _gt_pose,
    _load_frozen_inputs,
    _mainline_pose,
    _run_split,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_hypothesis_ranker", required=True)
    parser.add_argument("--selected_policy_artifact", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--proposal_score_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_key", default=None)
    parser.add_argument("--split_name", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--baseline_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--baseline_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _metrics(rows: Sequence[dict[str, object]], prefix: str) -> dict[str, object]:
    translation = np.asarray(
        [float(row[f"{prefix}_translation_m"]) for row in rows], dtype=np.float64
    )
    rotation = np.asarray(
        [float(row[f"{prefix}_rotation_deg"]) for row in rows], dtype=np.float64
    )
    return {
        "query_count": int(len(rows)),
        "median_translation_m": float(np.median(translation)),
        "p90_translation_m": float(np.quantile(translation, 0.9)),
        "median_rotation_deg": float(np.median(rotation)),
        "recall_25cm_2deg": float(
            np.mean((translation <= 0.25) & (rotation <= 2.0))
        ),
        "recall_10cm_5deg": float(
            np.mean((translation <= 0.10) & (rotation <= 5.0))
        ),
        "recall_5cm_5deg": float(
            np.mean((translation <= 0.05) & (rotation <= 5.0))
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_inputs(SimpleNamespace(**vars(args)))
    ranker_path = Path(args.pose_hypothesis_ranker)
    ranker, config, _payload = _load_ranker_and_config(ranker_path, args)
    split = json.loads(Path(args.split_json).read_text())
    query_ids = [str(value) for value in split[str(args.split_name)]]
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    groups, _learned_rows, learned_results = _run_split(
        split_name=f"frozen_ranker_regret_{args.split_name}",
        query_ids=query_ids,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=ranker,
    )
    examples, _baseline_results = _examples_for_run(
        query_ids=query_ids,
        groups=groups,
        learned_results=learned_results,
        ranker=ranker,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        baseline_reprojection_error_px=float(args.baseline_reprojection_error_px),
        baseline_iterations=int(args.baseline_iterations),
    )
    rows: list[dict[str, object]] = []
    for query_id, group, result, example in zip(
        query_ids, groups, learned_results, examples
    ):
        image = images_by_name[str(query_id)]
        gt_pose = _gt_pose(image)
        pre_error = pnp_pose_error(result.pre_refine_pose_w2c, gt_pose)
        finite = [
            index
            for index, value in enumerate(group.translation_errors_m)
            if np.isfinite(value) and np.isfinite(group.rotation_errors_deg[index])
        ]
        oracle_index = min(
            finite,
            key=lambda index: (
                float(group.translation_errors_m[index]),
                float(group.rotation_errors_deg[index]),
            ),
        )
        chosen_index = int(result.chosen_hypothesis_index)
        chosen_rank = 1 + int(
            np.sum(
                np.asarray(group.translation_errors_m, dtype=np.float64)[finite]
                < float(group.translation_errors_m[chosen_index])
            )
        )
        learned_quality = (
            float(example.learned_translation_m)
            + 0.02 * float(example.learned_rotation_deg)
        )
        baseline_quality = (
            float(example.baseline_translation_m)
            + 0.02 * float(example.baseline_rotation_deg)
        )
        hybrid_uses_learned = learned_quality < baseline_quality
        rows.append(
            {
                "query_id": str(query_id),
                "chosen_hypothesis_index": chosen_index,
                "chosen_hypothesis_translation_rank": int(chosen_rank),
                "hypothesis_count": int(len(group.records)),
                "oracle_hypothesis_index_TARGET_ONLY": int(oracle_index),
                "oracle_translation_m": float(
                    group.translation_errors_m[oracle_index]
                ),
                "oracle_rotation_deg": float(group.rotation_errors_deg[oracle_index]),
                "pre_refine_translation_m": float(pre_error.translation_m),
                "pre_refine_rotation_deg": float(pre_error.rotation_deg),
                "learned_translation_m": float(example.learned_translation_m),
                "learned_rotation_deg": float(example.learned_rotation_deg),
                "baseline_translation_m": float(example.baseline_translation_m),
                "baseline_rotation_deg": float(example.baseline_rotation_deg),
                "hybrid_oracle_translation_m": float(
                    example.learned_translation_m
                    if hybrid_uses_learned
                    else example.baseline_translation_m
                ),
                "hybrid_oracle_rotation_deg": float(
                    example.learned_rotation_deg
                    if hybrid_uses_learned
                    else example.baseline_rotation_deg
                ),
                "refit_translation_delta_m": float(
                    example.learned_translation_m - pre_error.translation_m
                ),
            }
        )
    summary = {
        "stage": "frozen_pose_hypothesis_rank_regret_audit",
        "protocol": {
            "split": str(args.split_name),
            "ground_truth_used_for_inference": False,
            "oracle_fields_are_target_only": True,
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "production_promoted": False,
        },
        "inputs": {
            "pose_hypothesis_ranker_sha256": file_sha256_short(ranker_path),
            "selected_policy_artifact_sha256": file_sha256_short(
                Path(args.selected_policy_artifact)
            ),
        },
        "metrics": {
            "hypothesis_oracle_TARGET_ONLY": _metrics(rows, "oracle"),
            "selected_pre_refine": _metrics(rows, "pre_refine"),
            "learned_final": _metrics(rows, "learned"),
            "frozen_l97_baseline": _metrics(rows, "baseline"),
            "two_backend_oracle_TARGET_ONLY": _metrics(rows, "hybrid_oracle"),
            "median_chosen_hypothesis_translation_rank": float(
                np.median(
                    [int(row["chosen_hypothesis_translation_rank"]) for row in rows]
                )
            ),
            "refit_worsened_query_count_TARGET_ONLY": int(
                np.sum([float(row["refit_translation_delta_m"]) > 1e-9 for row in rows])
            ),
            "refit_improved_query_count_TARGET_ONLY": int(
                np.sum([float(row["refit_translation_delta_m"]) < -1e-9 for row in rows])
            ),
        },
        "frozen_mainline_summary": (
            None
            if str(args.split_name) == "train"
            else _mainline_pose(
                frozen.selected_policy_summary,
                "validation" if str(args.split_name) == "validation" else "test",
            )
        ),
        "outputs": {
            "rows": str(output_dir / "regret_rows.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    rows_path = output_dir / "regret_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary["outputs"]["rows_sha256"] = file_sha256_short(rows_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
