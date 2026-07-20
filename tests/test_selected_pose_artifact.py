from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.run_independent_crossfit_pose_alignment import (
    _load_immutable_source_pose_artifact,
)
from feature_extract.vfm.localization.selected_pose_artifact import (
    selected_pose_rows_from_grouped_hypotheses,
    write_selected_pose_artifact,
)


def _write_grouped_hypotheses(
    path: Path,
    *,
    query_id: str,
    split_name: str,
    chosen_count: int = 1,
) -> None:
    count = 2
    chosen = np.zeros((count,), dtype=bool)
    chosen[:chosen_count] = True
    poses = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    metadata = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "inputs": {
            "colmap_cameras_bin_sha256": "camera-hash",
            "colmap_images_bin_sha256": "image-hash",
        },
        "grouped_config": {"generation_mode": "grouped_prosac"},
    }
    np.savez_compressed(
        path,
        query_ids=np.asarray([query_id, query_id]),
        split_names=np.asarray([split_name, split_name]),
        evaluation_labels=np.asarray(["optional", "optional"]),
        chosen_for_optional_pose=chosen,
        poses_w2c=poses,
        verification_effective_group_counts=np.asarray([12, 9], dtype=np.int64),
        verification_strict_inlier_counts=np.asarray([7, 4], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_grouped_hypotheses_build_a_loader_compatible_source(tmp_path: Path) -> None:
    source_a = tmp_path / "a.npz"
    source_b = tmp_path / "b.npz"
    _write_grouped_hypotheses(
        source_a, query_id="train.png", split_name="train"
    )
    _write_grouped_hypotheses(
        source_b, query_id="validation.png", split_name="validation"
    )
    rows, manifest = selected_pose_rows_from_grouped_hypotheses(
        (source_a, source_b),
        expected_query_ids={
            "train": ["train.png"],
            "validation": ["validation.png"],
            "test": [],
        },
    )
    assert len(rows) == 2
    assert rows[0]["match_count"] == 12
    assert rows[0]["inlier_count"] == 7
    artifact = tmp_path / "selected.npz"
    write_selected_pose_artifact(artifact, rows, source_manifest=manifest)
    loaded = _load_immutable_source_pose_artifact(
        artifact,
        expected_colmap_cameras_sha256="camera-hash",
        expected_colmap_images_sha256="image-hash",
        evaluation_label="optional",
    )
    assert set(loaded["records"]) == {
        ("train", "train.png"),
        ("validation", "validation.png"),
    }


def test_grouped_hypotheses_reject_non_unique_chosen_pose(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.npz"
    _write_grouped_hypotheses(
        source, query_id="validation.png", split_name="validation", chosen_count=2
    )
    with pytest.raises(ValueError, match="has 2 chosen poses"):
        selected_pose_rows_from_grouped_hypotheses((source,))
