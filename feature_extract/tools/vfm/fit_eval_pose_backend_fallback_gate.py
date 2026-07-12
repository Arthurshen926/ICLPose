"""Fit an OOF train-only gate between multi-hypothesis and l97 pose backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.fit_eval_pose_hypothesis_ranker import (
    FrozenPoseInputs,
    _gt_pose,
    _load_frozen_inputs,
    _mainline_pose,
    _query_matches_and_pool,
    _run_split,
    _strict_pose_gate,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.pose_backend_selection import (
    POSE_BACKEND_GATE_FEATURE_NAMES,
    PoseBackendSelectionExample,
    fit_pose_backend_gate,
    pose_backend_selection_metrics,
)
from feature_extract.vfm.localization.pose_hypothesis_ranking import (
    POSE_HYPOTHESIS_FEATURE_NAMES,
    PoseHypothesisRanker,
    fit_pose_hypothesis_ranker_fixed,
    pose_hypothesis_features,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    VerifiedPnPConfig,
    VerifiedPnPResult,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import (
    PnPResult,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)


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
    parser.add_argument("--baseline_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--baseline_iterations", type=int, default=5000)
    parser.add_argument("--gate_rotation_equivalent_m_per_deg", type=float, default=0.02)
    parser.add_argument("--skip_late_on_validation_pass", action="store_true")
    return parser.parse_args(argv)


def _load_ranker_and_config(
    path: Path, args: argparse.Namespace
) -> tuple[PoseHypothesisRanker, VerifiedPnPConfig, dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    if str(payload.get("format")) != "pose_hypothesis_ranker_v1":
        raise ValueError("unsupported pose hypothesis ranker artifact")
    expected = {
        "selected_policy_artifact_sha256": file_sha256_short(
            Path(args.selected_policy_artifact)
        ),
        "proposals_sha256": file_sha256_short(Path(args.proposals)),
        "candidate_artifact_sha256": file_sha256_short(Path(args.candidate_artifact)),
        "proposal_score_artifact_sha256": file_sha256_short(
            Path(args.proposal_score_artifact)
        ),
        "projected_landmark_bank_sha256": file_sha256_short(
            Path(args.projected_landmark_bank)
        ),
        "split_json_sha256": file_sha256_short(Path(args.split_json)),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get("inputs", {}).get(key)}
        for key, value in expected.items()
        if payload.get("inputs", {}).get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"pose hypothesis ranker inputs are stale: {json.dumps(mismatches, sort_keys=True)}"
        )
    config_payload = dict(payload["verified_pnp_config"])
    for key in (
        "fit_match_counts",
        "selection_modes",
        "ransac_thresholds_px",
        "rng_seed_offsets",
    ):
        config_payload[key] = tuple(config_payload[key])
    config = VerifiedPnPConfig(**config_payload)
    if config.final_audit_fold is None:
        raise ValueError("fallback gate requires a strict final-audit ranker")
    return PoseHypothesisRanker.from_dict(payload["model"]), config, payload


def _set_cv2_seed(query_id: str) -> None:
    try:
        import cv2

        seed = int.from_bytes(
            hashlib.sha256(str(query_id).encode("utf8")).digest()[:4], "little"
        )
        cv2.setRNGSeed(int(seed % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def _baseline_result(
    frozen: FrozenPoseInputs,
    query_id: str,
    camera,
    *,
    reprojection_error_px: float,
    iterations: int,
) -> PnPResult:
    matches, _pool = _query_matches_and_pool(frozen, str(query_id))
    selected = select_pose_safe_matches(
        matches,
        max_matches=int(frozen.selected_policy_metadata["max_matches"]),
        image_width=int(camera.width),
        image_height=int(camera.height),
        mode=str(frozen.selected_policy_metadata["selection_mode"]),
    )
    selected = stable_uniform_ransac_order(selected)
    _set_cv2_seed(str(query_id))
    return estimate_pose_pnp_ransac(
        selected,
        camera,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
        refine_method="LM",
    )


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3] @ np.asarray(second)[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _ratio(value: float, count: int) -> float:
    return float(value) / max(float(count), 1.0)


def _gate_features(
    ranker: PoseHypothesisRanker,
    group,
    learned: VerifiedPnPResult,
    baseline: PnPResult,
) -> tuple[float, ...]:
    if (
        learned.chosen_hypothesis_index is None
        or learned.pose_w2c is None
        or baseline.pose_w2c is None
        or learned.final_verification is None
    ):
        raise ValueError("pose backend gate requires successful learned and baseline poses")
    indices, scores = ranker.scores(group.records, group.poses_w2c)
    order = np.argsort(-scores, kind="mergesort")
    chosen_index = int(indices[int(order[0])])
    if chosen_index != int(learned.chosen_hypothesis_index):
        raise RuntimeError("learned pose result does not match the frozen ranker")
    top_score = float(scores[int(order[0])])
    margin = (
        0.0
        if len(order) < 2
        else top_score - float(scores[int(order[1])])
    )
    probabilities = np.exp(np.clip(scores - np.max(scores), -80.0, 0.0))
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    entropy = float(
        -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        / max(math.log(max(len(probabilities), 2)), 1e-12)
    )
    feature_indices, feature_rows = pose_hypothesis_features(
        group.records, group.poses_w2c, eligible_indices=(chosen_index,)
    )
    if feature_indices.tolist() != [chosen_index]:
        raise RuntimeError("chosen hypothesis feature identity changed")
    selected_features = {
        name: float(value)
        for name, value in zip(POSE_HYPOTHESIS_FEATURE_NAMES, feature_rows[0])
    }
    chosen_verification = group.records[chosen_index].verification
    if chosen_verification is None:
        raise RuntimeError("chosen hypothesis is missing rank verification")
    audit = learned.final_verification
    disagreement_translation = float(
        np.linalg.norm(
            _camera_center(learned.pose_w2c) - _camera_center(baseline.pose_w2c)
        )
    )
    disagreement_rotation = _rotation_distance_deg(
        learned.pose_w2c, baseline.pose_w2c
    )
    return (
        top_score,
        margin,
        entropy,
        selected_features["pose_cluster_50cm_5deg_fraction"],
        _ratio(
            chosen_verification.strict_inlier_count,
            chosen_verification.verification_count,
        ),
        _ratio(
            chosen_verification.loose_inlier_count,
            chosen_verification.verification_count,
        ),
        float(chosen_verification.selected_descriptor_score_mean or 0.0),
        float(chosen_verification.selected_descriptor_rank_score_mean or 0.0),
        _ratio(audit.strict_inlier_count, audit.verification_count),
        _ratio(audit.loose_inlier_count, audit.verification_count),
        _ratio(audit.soft_consensus, audit.verification_count),
        math.log1p(max(float(audit.clipped_median_residual_px), 0.0)),
        float(learned.inlier_count) / max(float(learned.match_count), 1.0),
        float(baseline.inlier_count) / max(float(baseline.match_count), 1.0),
        math.log1p(disagreement_translation),
        math.log1p(disagreement_rotation),
    )


def _examples_for_run(
    *,
    query_ids: Sequence[str],
    groups,
    learned_results: Sequence[VerifiedPnPResult],
    ranker: PoseHypothesisRanker,
    frozen: FrozenPoseInputs,
    cameras,
    images_by_name,
    baseline_reprojection_error_px: float,
    baseline_iterations: int,
) -> tuple[list[PoseBackendSelectionExample], list[PnPResult]]:
    examples: list[PoseBackendSelectionExample] = []
    baseline_results: list[PnPResult] = []
    by_group = {str(group.query_id): group for group in groups}
    by_learned = {
        str(query_id): result
        for query_id, result in zip(query_ids, learned_results)
    }
    for query_id in query_ids:
        image = images_by_name[str(query_id)]
        camera = cameras[int(image.camera_id)]
        baseline = _baseline_result(
            frozen,
            str(query_id),
            camera,
            reprojection_error_px=baseline_reprojection_error_px,
            iterations=baseline_iterations,
        )
        learned = by_learned[str(query_id)]
        if not baseline.success or baseline.pose_w2c is None or not learned.success:
            raise RuntimeError(f"pose backend failed on {query_id}")
        gt_pose = _gt_pose(image)
        learned_error = pnp_pose_error(learned.pose_w2c, gt_pose)
        baseline_error = pnp_pose_error(baseline.pose_w2c, gt_pose)
        examples.append(
            PoseBackendSelectionExample(
                query_id=str(query_id),
                features=_gate_features(
                    ranker, by_group[str(query_id)], learned, baseline
                ),
                learned_translation_m=float(learned_error.translation_m),
                learned_rotation_deg=float(learned_error.rotation_deg),
                baseline_translation_m=float(baseline_error.translation_m),
                baseline_rotation_deg=float(baseline_error.rotation_deg),
            )
        )
        baseline_results.append(baseline)
    return examples, baseline_results


def _verify_validation_baseline_replay(
    examples: Sequence[PoseBackendSelectionExample], policy_path: Path
) -> dict[str, float]:
    rows_path = policy_path.parent / "chosen_pose_rows.json"
    rows = json.loads(rows_path.read_text())["validation"]
    expected = {str(row["query_id"]): row for row in rows}
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
        raise ValueError("generated fallback does not exactly replay the frozen l97 pose")
    return {
        "maximum_translation_delta_m": float(maximum_translation_delta),
        "maximum_rotation_delta_deg": float(maximum_rotation_delta),
        "source_rows_sha256": file_sha256_short(rows_path),
    }


def _decision_rows(
    examples: Sequence[PoseBackendSelectionExample], probabilities: np.ndarray, decisions: np.ndarray
) -> list[dict[str, object]]:
    return [
        {
            "query_id": str(example.query_id),
            "features": {
                name: float(value)
                for name, value in zip(POSE_BACKEND_GATE_FEATURE_NAMES, example.features)
            },
            "learned_probability": float(probability),
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
        for example, probability, decision in zip(
            examples, probabilities.tolist(), decisions.tolist()
        )
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_inputs(SimpleNamespace(**vars(args)))
    ranker_path = Path(args.pose_hypothesis_ranker)
    ranker, config, ranker_payload = _load_ranker_and_config(ranker_path, args)
    split = json.loads(Path(args.split_json).read_text())
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    fold_assignments = {
        str(key): int(value)
        for key, value in ranker_payload["fit_summary"]["fold_assignments"].items()
    }

    train_ids = [str(value) for value in split["train"]]
    train_groups, _train_rows, _train_results = _run_split(
        split_name="fallback_gate_train_hypothesis_generation",
        query_ids=train_ids,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=None,
    )
    group_by_query = {str(group.query_id): group for group in train_groups}
    oof_groups = []
    oof_results = []
    oof_rankers: dict[str, PoseHypothesisRanker] = {}
    for fold in sorted(set(fold_assignments.values())):
        fold_train = [
            group
            for group in train_groups
            if int(fold_assignments[str(group.query_id)]) != int(fold)
        ]
        fold_ranker = fit_pose_hypothesis_ranker_fixed(
            fold_train,
            c_value=float(ranker.c_value),
            rotation_equivalent_m_per_deg=float(
                ranker.rotation_equivalent_m_per_deg
            ),
            minimum_pair_gap_m=float(ranker.minimum_pair_gap_m),
        )
        fold_ids = [
            query_id
            for query_id in train_ids
            if int(fold_assignments[query_id]) == int(fold)
        ]
        groups, _rows, results = _run_split(
            split_name=f"fallback_gate_train_oof_fold{fold}",
            query_ids=fold_ids,
            frozen=frozen,
            cameras=cameras,
            images_by_name=images_by_name,
            config=config,
            ranker=fold_ranker,
        )
        for group in groups:
            oof_rankers[str(group.query_id)] = fold_ranker
        oof_groups.extend(groups)
        oof_results.extend(results)
    result_by_query = {
        str(group.query_id): result for group, result in zip(oof_groups, oof_results)
    }
    train_examples: list[PoseBackendSelectionExample] = []
    for query_id in train_ids:
        examples, _baseline = _examples_for_run(
            query_ids=(query_id,),
            groups=(group_by_query[query_id],),
            learned_results=(result_by_query[query_id],),
            ranker=oof_rankers[query_id],
            frozen=frozen,
            cameras=cameras,
            images_by_name=images_by_name,
            baseline_reprojection_error_px=float(args.baseline_reprojection_error_px),
            baseline_iterations=int(args.baseline_iterations),
        )
        train_examples.extend(examples)
    gate, gate_fit = fit_pose_backend_gate(
        train_examples,
        fold_assignments=fold_assignments,
        rotation_equivalent_m_per_deg=float(
            args.gate_rotation_equivalent_m_per_deg
        ),
    )

    validation_ids = [str(value) for value in split["validation"]]
    validation_groups, _validation_rows, validation_results = _run_split(
        split_name="fallback_gate_validation_frozen_ranker",
        query_ids=validation_ids,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=ranker,
    )
    validation_examples, _validation_baseline = _examples_for_run(
        query_ids=validation_ids,
        groups=validation_groups,
        learned_results=validation_results,
        ranker=ranker,
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        baseline_reprojection_error_px=float(args.baseline_reprojection_error_px),
        baseline_iterations=int(args.baseline_iterations),
    )
    replay = _verify_validation_baseline_replay(
        validation_examples, Path(args.selected_policy_artifact)
    )
    validation_features = np.asarray(
        [example.features for example in validation_examples], dtype=np.float64
    )
    validation_probabilities = gate.probabilities(validation_features)
    validation_decisions = gate.choose_learned(validation_features)
    validation_metrics = pose_backend_selection_metrics(
        validation_examples, validation_decisions
    )
    validation_mainline = _mainline_pose(frozen.selected_policy_summary, "validation")
    validation_passes = _strict_pose_gate(validation_metrics, validation_mainline)

    late_examples: list[PoseBackendSelectionExample] = []
    late_probabilities = np.empty((0,), dtype=np.float64)
    late_decisions = np.empty((0,), dtype=bool)
    late_metrics = None
    if validation_passes and not bool(args.skip_late_on_validation_pass):
        late_ids = [str(value) for value in split["test"]]
        late_groups, _late_rows, late_results = _run_split(
            split_name="fallback_gate_late_frozen_ranker",
            query_ids=late_ids,
            frozen=frozen,
            cameras=cameras,
            images_by_name=images_by_name,
            config=config,
            ranker=ranker,
        )
        late_examples, _late_baseline = _examples_for_run(
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
        late_features = np.asarray(
            [example.features for example in late_examples], dtype=np.float64
        )
        late_probabilities = gate.probabilities(late_features)
        late_decisions = gate.choose_learned(late_features)
        late_metrics = pose_backend_selection_metrics(late_examples, late_decisions)

    model_path = output_dir / "pose_backend_fallback_gate.json"
    model_payload = {
        "format": "pose_backend_fallback_gate_v1",
        "model": gate.to_dict(),
        "pose_hypothesis_ranker_sha256": file_sha256_short(ranker_path),
        "selected_policy_artifact_sha256": file_sha256_short(
            Path(args.selected_policy_artifact)
        ),
        "split_json_sha256": file_sha256_short(Path(args.split_json)),
        "fit": gate_fit,
    }
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    rows_path = output_dir / "backend_decision_rows.json"
    rows_payload = {
        "train_oof": _decision_rows(
            train_examples,
            gate.probabilities(
                np.asarray([example.features for example in train_examples])
            ),
            gate.choose_learned(
                np.asarray([example.features for example in train_examples])
            ),
        ),
        "validation": _decision_rows(
            validation_examples, validation_probabilities, validation_decisions
        ),
        "late_development": _decision_rows(
            late_examples, late_probabilities, late_decisions
        ),
    }
    rows_path.write_text(json.dumps(rows_payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "train_oof_pose_backend_fallback_gate",
        "protocol": {
            "ranker_features_for_gate_train": "out_of_fold",
            "gate_fit_split": "train_queries_only",
            "gate_hyperparameter_selection": "train_query_grouped_cross_validation_only",
            "validation_used_for_fit": False,
            "late_used_for_selection": False,
            "ground_truth_available_to_inference_gate": False,
            "fallback_backend": "exact_frozen_l97_all_match_uniform_ransac",
            "learned_backend": "strict_final_audit_multi_hypothesis_pnp",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "production_promoted": False,
        },
        "inputs": {
            "pose_hypothesis_ranker": str(ranker_path),
            "pose_hypothesis_ranker_sha256": file_sha256_short(ranker_path),
            "selected_policy_artifact_sha256": file_sha256_short(
                Path(args.selected_policy_artifact)
            ),
        },
        "gate_fit": gate_fit,
        "validation": {
            "metrics": validation_metrics,
            "frozen_l97_mainline_pose": validation_mainline,
            "exact_fallback_replay": replay,
            "passes_strict_l97_pose_gate": bool(validation_passes),
        },
        "late_development_replay": {
            "executed": bool(late_examples),
            "metrics": late_metrics,
            "frozen_l97_mainline_pose": (
                None
                if not late_examples
                else _mainline_pose(frozen.selected_policy_summary, "test")
            ),
            "used_for_selection": False,
        },
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "decision_rows": str(rows_path),
            "decision_rows_sha256": file_sha256_short(rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
