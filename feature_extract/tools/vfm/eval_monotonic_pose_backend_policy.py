"""Select a monotonic pose fallback policy, then replay it once on late dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.fit_eval_pose_backend_fallback_gate import (
    _examples_for_run,
    _load_ranker_and_config,
)
from feature_extract.tools.vfm.fit_eval_pose_hypothesis_ranker import (
    _load_frozen_inputs,
    _mainline_pose,
    _run_split,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.pose_backend_selection import (
    POSE_BACKEND_GATE_FEATURE_NAMES,
    PoseBackendSelectionExample,
    pose_backend_selection_metrics,
    select_monotonic_pose_backend_policy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development_gate_summary", required=True)
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
    parser.add_argument("--baseline_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--baseline_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _examples(rows: Sequence[Mapping[str, object]]) -> list[PoseBackendSelectionExample]:
    output = []
    for row in rows:
        features = dict(row["features"])
        if tuple(features) != POSE_BACKEND_GATE_FEATURE_NAMES:
            if set(features) != set(POSE_BACKEND_GATE_FEATURE_NAMES):
                raise ValueError("development decision rows have an incompatible schema")
        output.append(
            PoseBackendSelectionExample(
                query_id=str(row["query_id"]),
                features=tuple(
                    float(features[name]) for name in POSE_BACKEND_GATE_FEATURE_NAMES
                ),
                learned_translation_m=float(
                    row["learned_translation_m_TARGET_ONLY"]
                ),
                learned_rotation_deg=float(row["learned_rotation_deg_TARGET_ONLY"]),
                baseline_translation_m=float(
                    row["baseline_translation_m_TARGET_ONLY"]
                ),
                baseline_rotation_deg=float(
                    row["baseline_rotation_deg_TARGET_ONLY"]
                ),
            )
        )
    return output


def _verify_late_baseline_replay(
    examples: Sequence[PoseBackendSelectionExample], policy_path: Path
) -> dict[str, object]:
    rows_path = policy_path.parent / "chosen_pose_rows.json"
    expected_rows = json.loads(rows_path.read_text())["late_development"]
    expected = {str(row["query_id"]): row for row in expected_rows}
    translation_delta = []
    rotation_delta = []
    for example in examples:
        row = expected[str(example.query_id)]
        translation_delta.append(
            abs(float(example.baseline_translation_m) - float(row["translation_m"]))
        )
        rotation_delta.append(
            abs(float(example.baseline_rotation_deg) - float(row["rotation_deg"]))
        )
    maximum_translation_delta = max(translation_delta, default=0.0)
    maximum_rotation_delta = max(rotation_delta, default=0.0)
    if maximum_translation_delta > 1e-9 or maximum_rotation_delta > 1e-9:
        raise ValueError("generated late fallback does not replay frozen l97 exactly")
    return {
        "maximum_translation_delta_m": float(maximum_translation_delta),
        "maximum_rotation_delta_deg": float(maximum_rotation_delta),
        "source_rows_sha256": file_sha256_short(rows_path),
    }


def _policy_rows(
    examples: Sequence[PoseBackendSelectionExample], decisions: np.ndarray
) -> list[dict[str, object]]:
    return [
        {
            "query_id": str(example.query_id),
            "features": {
                name: float(value)
                for name, value in zip(POSE_BACKEND_GATE_FEATURE_NAMES, example.features)
            },
            "selected_backend": (
                "learned_multi_hypothesis" if bool(decision) else "l97_all_match"
            ),
            "learned_translation_m_TARGET_ONLY": float(
                example.learned_translation_m
            ),
            "learned_rotation_deg_TARGET_ONLY": float(example.learned_rotation_deg),
            "baseline_translation_m_TARGET_ONLY": float(
                example.baseline_translation_m
            ),
            "baseline_rotation_deg_TARGET_ONLY": float(
                example.baseline_rotation_deg
            ),
        }
        for example, decision in zip(examples, decisions.tolist())
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    development_summary_path = Path(args.development_gate_summary)
    development_summary = json.loads(development_summary_path.read_text())
    rows_path = Path(development_summary["outputs"]["decision_rows"])
    if file_sha256_short(rows_path) != str(
        development_summary["outputs"]["decision_rows_sha256"]
    ):
        raise ValueError("development decision rows hash differs from its summary")
    ranker_path = Path(args.pose_hypothesis_ranker)
    if file_sha256_short(ranker_path) != str(
        development_summary["inputs"]["pose_hypothesis_ranker_sha256"]
    ):
        raise ValueError("development rows and requested ranker differ")
    row_payload = json.loads(rows_path.read_text())
    train_examples = _examples(row_payload["train_oof"])
    validation_examples = _examples(row_payload["validation"])
    policy, policy_selection = select_monotonic_pose_backend_policy(
        train_examples, validation_examples
    )
    train_features = np.asarray(
        [example.features for example in train_examples], dtype=np.float64
    )
    validation_features = np.asarray(
        [example.features for example in validation_examples], dtype=np.float64
    )
    train_decisions = policy.choose_learned(train_features)
    validation_decisions = policy.choose_learned(validation_features)
    train_metrics = pose_backend_selection_metrics(train_examples, train_decisions)
    validation_metrics = pose_backend_selection_metrics(
        validation_examples, validation_decisions
    )

    frozen = _load_frozen_inputs(SimpleNamespace(**vars(args)))
    ranker, config, _ranker_payload = _load_ranker_and_config(
        ranker_path, args
    )
    split = json.loads(Path(args.split_json).read_text())
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    late_ids = [str(value) for value in split["test"]]
    late_groups, _late_rows, late_results = _run_split(
        split_name="monotonic_backend_policy_late_development",
        query_ids=late_ids,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=ranker,
    )
    late_examples, _baseline_results = _examples_for_run(
        query_ids=late_ids,
        groups=late_groups,
        learned_results=late_results,
        ranker=ranker,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        baseline_reprojection_error_px=float(args.baseline_reprojection_error_px),
        baseline_iterations=int(args.baseline_iterations),
    )
    late_replay = _verify_late_baseline_replay(
        late_examples, Path(args.selected_policy_artifact)
    )
    late_features = np.asarray(
        [example.features for example in late_examples], dtype=np.float64
    )
    late_decisions = policy.choose_learned(late_features)
    late_metrics = pose_backend_selection_metrics(late_examples, late_decisions)

    policy_path = output_dir / "monotonic_pose_backend_policy.json"
    policy_payload = {
        "format": "monotonic_pose_backend_policy_v1",
        "policy": policy.to_dict(),
        "development_gate_summary_sha256": file_sha256_short(
            development_summary_path
        ),
        "development_decision_rows_sha256": file_sha256_short(rows_path),
        "pose_hypothesis_ranker_sha256": file_sha256_short(ranker_path),
        "selected_policy_artifact_sha256": file_sha256_short(
            Path(args.selected_policy_artifact)
        ),
        "selection": policy_selection,
    }
    policy_path.write_text(json.dumps(policy_payload, indent=2, sort_keys=True) + "\n")
    output_rows_path = output_dir / "backend_decision_rows.json"
    output_rows = {
        "train_oof": _policy_rows(train_examples, train_decisions),
        "validation": _policy_rows(validation_examples, validation_decisions),
        "late_development": _policy_rows(late_examples, late_decisions),
    }
    output_rows_path.write_text(
        json.dumps(output_rows, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "stage": "validation_selected_monotonic_pose_backend_policy",
        "protocol": {
            "policy_form": "monotonic_primary_plus_conservative_rescue",
            "candidate_policy_count": int(policy_selection["candidate_count"]),
            "required_selection_gates": ["train_oof", "validation"],
            "late_used_for_selection": False,
            "late_block_is_untouched_test": False,
            "ground_truth_available_to_inference_policy": False,
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "production_promoted": False,
        },
        "inputs": {
            "development_gate_summary_sha256": file_sha256_short(
                development_summary_path
            ),
            "development_decision_rows_sha256": file_sha256_short(rows_path),
            "pose_hypothesis_ranker_sha256": file_sha256_short(ranker_path),
            "selected_policy_artifact_sha256": file_sha256_short(
                Path(args.selected_policy_artifact)
            ),
        },
        "policy_selection": policy_selection,
        "train_oof": {key: value for key, value in train_metrics.items() if key != "rows"},
        "validation": {
            "metrics": {
                key: value for key, value in validation_metrics.items() if key != "rows"
            },
            "frozen_l97_mainline_pose": _mainline_pose(
                frozen.selected_policy_summary, "validation"
            ),
        },
        "late_development_replay": {
            "metrics": {key: value for key, value in late_metrics.items() if key != "rows"},
            "frozen_l97_mainline_pose": _mainline_pose(
                frozen.selected_policy_summary, "test"
            ),
            "exact_fallback_replay": late_replay,
            "used_for_selection": False,
        },
        "outputs": {
            "policy": str(policy_path),
            "policy_sha256": file_sha256_short(policy_path),
            "decision_rows": str(output_rows_path),
            "decision_rows_sha256": file_sha256_short(output_rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
