from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    ARTIFACT_FORMAT,
    main,
    validate_inference_artifact,
)
from feature_extract.vfm.colmap_tracks import (
    ColmapImageObservation,
    write_colmap_images_binary,
)


def _artifact_arrays() -> dict[str, np.ndarray]:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[0, 0, 3] = 0.50
    poses[1, 0, 3] = 0.02
    poses[2, 0, 3] = 0.04
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "row_count": 3,
        "inputs": {"proposal_sha256": "frozen"},
        "grouped_config": {"candidate_limits": [20]},
    }
    return {
        "query_ids": np.asarray(["q0.jpg", "q0.jpg", "q1.jpg"]),
        "split_names": np.asarray(["validation", "validation", "validation"]),
        "evaluation_labels": np.asarray(["optional", "optional", "optional"]),
        "hypothesis_indices": np.asarray([0, 1, 0], dtype=np.int64),
        "generation_profiles": np.asarray(["p", "p", "p"]),
        "selection_modes": np.asarray(["raw", "raw", "raw"]),
        "shortlisted_for_verification": np.asarray([True, False, True]),
        "chosen_for_optional_pose": np.asarray([True, False, True]),
        "poses_w2c": poses,
        "preliminary_log_likelihood_means": np.asarray([2.0, 1.0, 1.0]),
        "shortlist_log_likelihood_means": np.asarray([2.0, np.nan, 1.0]),
        "verification_log_likelihood_means": np.asarray([2.0, np.nan, 1.0]),
        "verification_relation_log_likelihood_ratio_means": np.asarray(
            [-0.2, np.nan, -0.1]
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def _write_artifact(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _write_gt(model_dir: Path) -> None:
    images = {}
    for image_id, name in enumerate(("q0.jpg", "q1.jpg"), start=1):
        images[image_id] = ColmapImageObservation(
            image_id=image_id,
            image_name=name,
            camera_id=1,
            qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros((3,), dtype=np.float64),
            xys=np.zeros((0, 2), dtype=np.float64),
            point3d_ids=np.zeros((0,), dtype=np.int64),
        )
    write_colmap_images_binary(images, model_dir / "images.bin")


def test_inference_artifact_rejects_target_fields() -> None:
    arrays = _artifact_arrays()
    arrays["translation_errors_m"] = np.zeros((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="target fields"):
        validate_inference_artifact(arrays)


def test_gt_join_keeps_targets_separate_and_reports_stage_gaps(tmp_path: Path) -> None:
    source = tmp_path / "hypotheses.npz"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "evaluation"
    _write_artifact(source, _artifact_arrays())
    _write_gt(model_dir)

    assert (
        main(
            [
                "--hypothesis_artifacts",
                str(source),
                "--colmap_model_dir",
                str(model_dir),
                "--output_dir",
                str(output_dir),
            ]
        )
        == 0
    )

    with np.load(source, allow_pickle=False) as inference:
        assert "translation_errors_m" not in inference.files
        assert "correct_10cm_5deg" not in inference.files
    with np.load(
        output_dir / "grouped_hypothesis_targets_v1.npz", allow_pickle=False
    ) as targets:
        assert targets["translation_errors_m"].tolist() == pytest.approx(
            [0.50, 0.02, 0.04]
        )
        assert targets["correct_10cm_5deg"].tolist() == [False, True, True]
        metadata = json.loads(str(targets["metadata_json"].item()))
        assert metadata["targets_joined_after_inference"] is True

    summary = json.loads(
        (output_dir / "grouped_hypothesis_evaluation_summary.json").read_text()
    )
    metrics = summary["metrics"]["validation::optional"]
    assert metrics["full_oracle"]["median_translation_error_cm"] == pytest.approx(3.0)
    assert metrics["shortlist_oracle"]["median_translation_error_cm"] == pytest.approx(27.0)
    assert metrics["chosen"]["median_translation_error_cm"] == pytest.approx(27.0)
    assert metrics["full_oracle"]["recall_10cm_5deg"] == pytest.approx(1.0)
    assert metrics["chosen"]["recall_10cm_5deg"] == pytest.approx(0.5)
    assert metrics["relation_DIAGNOSTIC_ONLY_top1"][
        "median_translation_error_cm"
    ] == pytest.approx(27.0)
    assert summary["protocol"][
        "pairwise_relation_score_is_diagnostic_only"
    ] is True
