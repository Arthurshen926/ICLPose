from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.pose_cache_report import (  # noqa: E402
    build_pose_cache_comparison,
    pose_cache_comparison_to_markdown,
    query_names_from_cache,
    summarize_pose_cache_for_queries,
)


def _pose_at(center_x_m: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = -float(center_x_m)
    return pose


def _write_cache(path: Path, names: list[str], centers: list[float], *, refine_success=None) -> None:
    payload = {
        "query_image_names": np.asarray(names),
        "query_image_stems": np.asarray([Path(name).with_suffix("").as_posix().replace("/", "_") for name in names]),
        "pose_inits": np.stack([_pose_at(center) for center in centers]).astype(np.float32),
        "init_sources": np.asarray(["test_solver"] * len(names)),
    }
    if refine_success is not None:
        payload["refine_success"] = np.asarray(refine_success, dtype=bool)
    np.savez_compressed(path, **payload)


def test_pose_cache_report_filters_to_reference_query_order_and_solver_success(tmp_path):
    ref_path = tmp_path / "reference.npz"
    method_path = tmp_path / "method.npz"
    _write_cache(ref_path, ["seq/q2.png", "seq/q1.png"], [0.0, 0.0])
    _write_cache(
        method_path,
        ["seq/q1.png", "seq/q2.png", "seq/q3.png"],
        [0.04, 0.20, 8.0],
        refine_success=[True, False, True],
    )
    gt = {
        "seq/q1.png": _pose_at(0.0),
        "seq/q2.png": _pose_at(0.0),
        "seq/q3.png": _pose_at(0.0),
    }

    summary = summarize_pose_cache_for_queries(
        cache_path=method_path,
        label="method",
        gt_poses_by_name=gt,
        query_names=query_names_from_cache(ref_path),
    )

    assert summary["num_samples"] == 2
    assert summary["solver_success_frac"] == pytest.approx(0.5)
    assert summary["metrics"]["trans_median"] == pytest.approx(120.0)
    assert summary["metrics"]["joint_1deg_50mm"] == pytest.approx(50.0)


def test_pose_cache_comparison_markdown_contains_real_pose_metrics(tmp_path):
    cache_path = tmp_path / "method.npz"
    _write_cache(cache_path, ["seq/q1.png"], [0.04], refine_success=[True])
    gt = {"seq/q1.png": _pose_at(0.0)}

    report = build_pose_cache_comparison(
        caches=[("method", cache_path)],
        gt_poses_by_name=gt,
        query_names=["seq/q1.png"],
    )
    markdown = pose_cache_comparison_to_markdown(report)

    assert report["rows"][0]["metrics"]["trans_median"] == pytest.approx(40.0)
    assert "method" in markdown
    assert "40.0" in markdown
