from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries, save_retrieval_init_entries
from feature_retrieval.tools.export_selected_init_cache import export_selected_init_cache


def _pose_at(value: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(value)
    return pose


def _entry(query_idx: int) -> dict:
    poses = np.stack(
        [
            _pose_at(query_idx * 10.0 + 1.0),
            _pose_at(query_idx * 10.0 + 2.0),
            _pose_at(query_idx * 10.0 + 3.0),
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "query_img_id": query_idx,
        "query_image_name": f"query/{query_idx:06d}.png",
        "query_image_stem": f"query_{query_idx:06d}",
        "pose_init": poses[0].copy(),
        "init_source": "original",
        "retrieval_frame_id": 100 + query_idx,
        "retrieval_image_name": f"ref/original_{query_idx:06d}.png",
        "retrieval_score": 0.1 + query_idx,
        "pose_init_candidates": poses,
        "candidate_valid_mask": np.array([True, True, False], dtype=bool),
        "retrieval_frame_ids_candidates": np.array(
            [1000 + query_idx, 2000 + query_idx, 3000 + query_idx],
            dtype=np.int64,
        ),
        "retrieval_image_names_candidates": np.array(
            [
                f"ref/a_{query_idx:06d}.png",
                f"ref/b_{query_idx:06d}.png",
                f"ref/c_{query_idx:06d}.png",
            ]
        ),
        "retrieval_scores_candidates": np.array([0.9, 0.8, 0.7], dtype=np.float32),
        "retrieval_pnp_num_inliers_candidates": np.array([11.0, 22.0, 33.0], dtype=np.float32),
    }


def _write_cache(tmp_path: Path) -> Path:
    init_cache_path = tmp_path / "init_cache.npz"
    save_retrieval_init_entries(
        [_entry(1), _entry(2)],
        {"method_used": "source_cache", "existing": 123},
        str(init_cache_path),
    )
    return init_cache_path


def test_export_selected_init_cache_selects_candidate_and_preserves_candidate_arrays(tmp_path):
    init_cache_path = _write_cache(tmp_path)
    save_path = tmp_path / "selected_cache.npz"

    exported_entries, exported_stats = export_selected_init_cache(
        str(init_cache_path),
        [1, 0],
        str(save_path),
        source_name="reranker_choice",
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert len(exported_entries) == 2
    assert exported_stats["selection_source"] == "selected_candidate"
    assert exported_stats["init_cache"] == str(init_cache_path)
    assert exported_stats["num_selected"] == 2
    assert exported_stats["source_name"] == "reranker_choice"
    assert exported_stats["existing"] == 123
    assert loaded_stats == exported_stats

    assert loaded_entries[0]["init_source"] == "reranker_choice"
    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(12.0))
    assert loaded_entries[0]["retrieval_frame_id"] == 2001
    assert loaded_entries[0]["retrieval_image_name"] == "ref/b_000001.png"
    assert loaded_entries[0]["retrieval_score"] == pytest.approx(0.8)
    assert np.allclose(loaded_entries[0]["pose_init_candidates"], _entry(1)["pose_init_candidates"])
    assert loaded_entries[0]["candidate_valid_mask"].tolist() == [True, True, False]
    assert np.allclose(
        loaded_entries[0]["retrieval_pnp_num_inliers_candidates"],
        np.array([11.0, 22.0, 33.0], dtype=np.float32),
    )

    assert loaded_entries[1]["retrieval_frame_id"] == 1002
    assert loaded_entries[1]["retrieval_image_name"] == "ref/a_000002.png"
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(21.0))


def test_export_selected_init_cache_rejects_bad_selection_lengths_and_indices(tmp_path):
    init_cache_path = _write_cache(tmp_path)

    with pytest.raises(ValueError, match="selected_indices length"):
        export_selected_init_cache(str(init_cache_path), [0], str(tmp_path / "short.npz"))

    with pytest.raises(ValueError, match="out of range"):
        export_selected_init_cache(str(init_cache_path), [3, 0], str(tmp_path / "oob.npz"))

    with pytest.raises(ValueError, match="not valid"):
        export_selected_init_cache(str(init_cache_path), [2, 0], str(tmp_path / "invalid.npz"))


def test_export_selected_init_cache_cli_reads_json_indices(tmp_path):
    init_cache_path = _write_cache(tmp_path)
    save_path = tmp_path / "cli_selected_cache.npz"
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "cv2.py").write_text("", encoding="utf-8")
    (stub_dir / "faiss.py").write_text("", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{stub_dir}{os.pathsep}{Path(__file__).resolve().parents[1]}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_retrieval.tools.export_selected_init_cache",
            "--init_cache",
            str(init_cache_path),
            "--indices_json",
            json.dumps([0, 1]),
            "--save_path",
            str(save_path),
            "--source_name",
            "cli_choice",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert "Exported 2 selected init entries" in result.stdout
    assert loaded_stats["source_name"] == "cli_choice"
    assert loaded_entries[0]["init_source"] == "cli_choice"
    assert loaded_entries[1]["retrieval_frame_id"] == 2002
