from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_coordinate_pose_hypothesis_set import (
    _load_hypotheses,
    _seal_hypotheses,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256


def _pose_file(path: Path, names: list[str], *, postlabel: bool = False) -> None:
    arrays = {
        "names": np.asarray(names),
        "pose_w2c": np.repeat(np.eye(4)[None], len(names), axis=0),
        "usable": np.ones(len(names), bool),
    }
    metadata = {
        "artifact_type": "goal_maplet_uncertainty_weighted_plane_pose_refinement_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_pose_or_ground_truth_read": postlabel,
        "source_rgb_stored_or_consumed_at_runtime": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(path, **arrays, metadata_json=np.asarray(__import__("json").dumps(metadata)))


def test_hypothesis_set_preserves_primary_first_and_two_candidates(tmp_path: Path) -> None:
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _pose_file(first, ["a.npz", "b.npz"])
    _pose_file(second, ["a.npz", "b.npz"])
    arrays, metadata = _seal_hypotheses(first, second)
    assert arrays["pose_w2c"].shape == (2, 2, 4, 4)
    assert arrays["branch_names"].tolist() == [
        "point_coordinate_V5", "continuous_chart_coordinate_V11",
    ]
    assert metadata["primary_branch_index"] == 0
    assert metadata["query_pose_or_ground_truth_read"] is False


def test_hypothesis_set_rejects_order_drift_and_postlabel_input(tmp_path: Path) -> None:
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _pose_file(first, ["a.npz", "b.npz"])
    _pose_file(second, ["b.npz", "a.npz"])
    with pytest.raises(ValueError, match="order"):
        _seal_hypotheses(first, second)
    _pose_file(second, ["a.npz", "b.npz"], postlabel=True)
    with pytest.raises(ValueError, match="contract"):
        _seal_hypotheses(first, second)


def test_hypothesis_set_loader_rejects_pose_tamper(tmp_path: Path) -> None:
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _pose_file(first, ["a.npz"])
    _pose_file(second, ["a.npz"])
    arrays, metadata = _seal_hypotheses(first, second)
    output = tmp_path / "set.npz"
    np.savez_compressed(
        output, **arrays,
        metadata_json=np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
    )
    loaded, _ = _load_hypotheses(output)
    np.testing.assert_array_equal(loaded["pose_w2c"], arrays["pose_w2c"])
    arrays["pose_w2c"][0, 1, 0, 3] = 1.0
    np.savez_compressed(
        output, **arrays,
        metadata_json=np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="contract"):
        _load_hypotheses(output)
