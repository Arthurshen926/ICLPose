from pathlib import Path
import json

import pytest

from feature_extract.tools.vfm.run_goal_maplet_strict_matcha_fold import (
    _commands,
    _fold_contract,
    _validate_tracked_patch,
)


def _dataset(tmp_path: Path, *, points_bytes: bytes = b"\x00" * 8) -> Path:
    root = tmp_path / "posed"
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    for index in range(3):
        (root / "images" / f"image{index}.png").write_bytes(b"image")
    for name in ("cameras.bin", "images.bin"):
        (root / "sparse" / "0" / name).write_bytes(b"model")
    (root / "sparse" / "0" / "points3D.bin").write_bytes(points_bytes)
    (root / "fold_colmap_dataset.json").write_text(json.dumps({
        "artifact_type": "goal_maplet_fold_clean_posed_colmap_dataset_v2",
        "fold_id": "fold0",
        "mapping_image_count": 3,
        "dense_supervision_image_count": 3,
        "dense_supervision_uses_all_mapping_images": True,
        "selected_contains_held_query": False,
        "selected_chart_image_indices_zero_based": [0, 2],
        "held_query_trajectories": ["seq2"],
    }))
    return root


def test_fold_contract_requires_all_mapping_images_and_point_free_model(tmp_path):
    contract = _fold_contract(_dataset(tmp_path))
    assert contract["mapping_image_count"] == 3
    assert contract["chart_indices"] == [0, 2]


def test_fold_contract_rejects_sfm_points(tmp_path):
    with pytest.raises(ValueError, match="SfM points"):
        _fold_contract(_dataset(tmp_path, points_bytes=b"not-empty-model"))


def test_commands_use_direct_checked_stages_and_full_dense_dataset(tmp_path):
    dataset = tmp_path / "posed"
    output = tmp_path / "out"
    commands = _commands(
        conda_env="matcha", dataset=dataset, output=output,
        chart_indices=[0, 7], gaussian_iterations=30000,
    )
    assert commands["sfm"][6] == "mast3r/run_mast3r.py"
    assert "scripts/run_sfm.py" not in commands["sfm"]
    assert "--image_idx" in commands["sfm"]
    assert commands["alignment"][6] == "scripts/align_charts.py"
    assert commands["gaussians"][6] == "2d-gaussian-splatting/train_with_charts.py"
    dense_at = commands["gaussians"].index("--dense_data_path")
    assert commands["gaussians"][dense_at + 1] == str(dataset)
    iteration_at = commands["gaussians"].index("--iterations")
    assert commands["gaussians"][iteration_at + 1] == "30000"


def test_matcha_runner_rejects_unversioned_tracked_source_drift(tmp_path):
    contract = tmp_path / "declared.patch"
    contract.write_bytes(b"declared\n")
    result = _validate_tracked_patch(b"declared\n", contract)
    assert result["sha256"] == result["applied_diff_sha256"]
    with pytest.raises(ValueError, match="versioned"):
        _validate_tracked_patch(b"different\n", contract)
