from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.report_hard_case_final_pose_table import (  # noqa: E402
    build_hard_case_final_pose_report,
    format_hard_case_final_pose_markdown,
    hard_case_query_names,
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
        "init_sources": np.asarray(["test"] * len(names)),
    }
    if refine_success is not None:
        payload["refine_success"] = np.asarray(refine_success, dtype=bool)
    np.savez_compressed(path, **payload)


def test_hard_case_final_pose_report_filters_unique_case_queries(tmp_path):
    hard_case = tmp_path / "case.jsonl"
    hard_case.write_text(
        "\n".join(
            [
                json.dumps({"sample_name": "seq/q2.png", "candidate_idx": 0}),
                json.dumps({"sample_name": "seq/q2.png", "candidate_idx": 1}),
                json.dumps({"sample_name": "seq/q1.png", "candidate_idx": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    identity = tmp_path / "identity.npz"
    refined = tmp_path / "refined.npz"
    _write_cache(identity, ["seq/q1.png", "seq/q2.png"], [0.20, 0.20])
    _write_cache(refined, ["seq/q1.png", "seq/q2.png"], [0.02, 0.04], refine_success=[True, False])
    gt = {"seq/q1.png": _pose_at(0.0), "seq/q2.png": _pose_at(0.0)}

    report = build_hard_case_final_pose_report(
        hard_cases=[("false_accept", hard_case)],
        caches=[("identity", identity), ("refined", refined)],
        gt_poses_by_name=gt,
    )
    markdown = format_hard_case_final_pose_markdown(report)

    assert hard_case_query_names(hard_case) == ["seq/q2.png", "seq/q1.png"]
    assert report["cases"][0]["case"] == "false_accept"
    assert report["cases"][0]["rows"][0]["label"] == "identity"
    assert report["cases"][0]["rows"][0]["num_samples"] == 2
    assert report["cases"][0]["rows"][1]["solver_success_frac"] == pytest.approx(0.5)
    assert report["cases"][0]["rows"][1]["metrics"]["trans_median"] == pytest.approx(30.0)
    assert "false_accept" in markdown
    assert "refined" in markdown
