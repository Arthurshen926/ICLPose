"""Prove sparse-first dense rendering is exactly decision-equivalent.

Only rows for which the frozen sparse test can still permit V11 are rendered.
This audit replays the external plan, checks every retained render diagnostic
against a complete-render authority, verifies omitted rows carry no score, and
then proves that both branch decisions and selected poses are unchanged.  No
query pose or ground truth is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_resident_render_equivalence import (
    _IMMUTABLE_RENDER_FIELDS,
)
from feature_extract.tools.vfm.build_goal_maplet_sparse_first_render_plan import (
    _load_sparse_first_plan,
)
from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import (
    _load_plane_geometry,
    _load_render,
    _load_selected,
    _normal_good_ray_score,
    _select_geometry_consensus,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


OUTPUT_ARTIFACT = "goal_maplet_sparse_first_dense_render_equivalence_v1"


def _validate_sparse_rows(
    full: dict[str, object], sparse: dict[str, object], required: np.ndarray,
) -> None:
    mask = np.asarray(required, bool)
    if len(full["rows"]) != len(mask) or len(sparse["rows"]) != len(mask):
        raise ValueError("sparse-first render row count differs")
    for key in _IMMUTABLE_RENDER_FIELDS:
        if full.get(key) != sparse.get(key):
            raise ValueError(f"sparse-first render lineage differs at {key}")
    for index, (full_row, sparse_row) in enumerate(zip(full["rows"], sparse["rows"])):
        evaluated = sparse_row.get("dense_score_evaluated")
        expected = bool(mask[index])
        if evaluated is not expected:
            raise ValueError("sparse-first evaluated mask differs")
        comparison = dict(sparse_row)
        comparison.pop("dense_score_evaluated")
        full_comparison = {
            key: value for key, value in full_row.items() if not key.startswith("planar_")
        }
        if expected:
            if comparison != full_comparison:
                raise ValueError("sparse-first retained render diagnostics differ")
        elif set(comparison) != {"name", "usable"}:
            raise ValueError("sparse-first omitted row contains a dense score")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose", type=Path, required=True)
    parser.add_argument("--alternate_pose", type=Path, required=True)
    parser.add_argument("--plane_geometry", type=Path, required=True)
    parser.add_argument("--render_plan", type=Path, required=True)
    parser.add_argument("--full_primary_render", type=Path, required=True)
    parser.add_argument("--full_alternate_render", type=Path, required=True)
    parser.add_argument("--sparse_primary_render", type=Path, required=True)
    parser.add_argument("--sparse_alternate_render", type=Path, required=True)
    parser.add_argument("--full_consensus", type=Path, required=True)
    parser.add_argument("--sparse_consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sparse-first render audit")

    primary, primary_meta = _load_pose_candidate(args.primary_pose)
    alternate, alternate_meta = _load_pose_candidate(args.alternate_pose)
    plane, plane_meta = _load_plane_geometry(args.plane_geometry)
    plan, plan_meta = _load_sparse_first_plan(args.render_plan)
    full_primary = _load_render(args.full_primary_render)
    full_alternate = _load_render(args.full_alternate_render)
    sparse_primary = _load_render(args.sparse_primary_render)
    sparse_alternate = _load_render(args.sparse_alternate_render)
    full_selected, full_selected_meta = _load_selected(args.full_consensus)
    sparse_selected, sparse_selected_meta = _load_selected(args.sparse_consensus)
    names = primary["names"].astype(str)
    required = np.asarray(plan["dense_render_required"], bool)
    if not (
        np.array_equal(names, alternate["names"].astype(str))
        and np.array_equal(names, plane["names"].astype(str))
        and np.array_equal(names, plan["names"].astype(str))
        and np.array_equal(names, full_selected["names"].astype(str))
        and np.array_equal(names, sparse_selected["names"].astype(str))
        and plan_meta.get("primary_pose_file_sha256") == file_sha256(args.primary_pose)
        and plan_meta.get("primary_pose_content_sha256") == primary_meta.get("content_sha256")
        and plan_meta.get("alternate_pose_file_sha256") == file_sha256(args.alternate_pose)
        and plan_meta.get("alternate_pose_content_sha256") == alternate_meta.get("content_sha256")
        and plan_meta.get("plane_geometry_file_sha256") == file_sha256(args.plane_geometry)
        and plan_meta.get("plane_geometry_content_sha256") == plane_meta.get("content_sha256")
    ):
        raise ValueError("sparse-first audit inventory differs")
    for report in (sparse_primary, sparse_alternate):
        if (
            report.get("sparse_first_render_plan_file_sha256")
            != file_sha256(args.render_plan)
            or report.get("sparse_first_render_plan_content_sha256")
            != plan_meta.get("content_sha256")
            or int(report.get("dense_render_query_count", -1)) != int(np.sum(required))
        ):
            raise ValueError("sparse render does not bind the external plan")
    _validate_sparse_rows(full_primary, sparse_primary, required)
    _validate_sparse_rows(full_alternate, sparse_alternate, required)

    usable = np.stack([primary["usable"], alternate["usable"]], axis=1).astype(bool)
    objective = np.asarray(plane["moge3_plane_geometry_objective"], np.float64)
    expected_required = usable[:, 0] & usable[:, 1] & (objective[:, 1] < objective[:, 0])
    if not np.array_equal(required, expected_required):
        raise ValueError("render plan does not replay the sparse decision gate")

    def replay(first: dict[str, object], second: dict[str, object]) -> np.ndarray:
        score = np.stack([
            np.asarray([_normal_good_ray_score(row) for row in first["rows"]]),
            np.asarray([_normal_good_ray_score(row) for row in second["rows"]]),
        ], axis=1)
        return _select_geometry_consensus(objective, score, usable)

    full_branch = replay(full_primary, full_alternate)
    sparse_branch = replay(sparse_primary, sparse_alternate)
    if not (
        np.array_equal(full_branch, sparse_branch)
        and np.array_equal(full_branch, full_selected["selected_branch"])
        and np.array_equal(sparse_branch, sparse_selected["selected_branch"])
    ):
        raise ValueError("sparse-first rendering changes the selected branch")
    for key in (
        "names", "pose_w2c", "usable", "selected_branch",
        "candidate_plane_geometry_objective", "candidate_moge3_depth_scale",
    ):
        first = np.asarray(full_selected[key])
        second = np.asarray(sparse_selected[key])
        equal = (
            np.array_equal(first, second, equal_nan=True)
            if np.issubdtype(first.dtype, np.inexact)
            else np.array_equal(first, second)
        )
        if not equal:
            raise ValueError(f"sparse-first selected artifact differs at {key}")
    if not (
        full_selected_meta.get("primary_pose_file_sha256") == file_sha256(args.primary_pose)
        and sparse_selected_meta.get("primary_pose_file_sha256") == file_sha256(args.primary_pose)
    ):
        raise ValueError("selected artifacts do not bind the primary pose")

    full_seconds = np.asarray([
        full_primary["elapsed_seconds"], full_alternate["elapsed_seconds"],
    ], np.float64)
    sparse_seconds = np.asarray([
        sparse_primary["elapsed_seconds"], sparse_alternate["elapsed_seconds"],
    ], np.float64)
    if not (np.isfinite(full_seconds).all() and np.isfinite(sparse_seconds).all()
            and np.all(full_seconds > 0.0) and np.all(sparse_seconds > 0.0)):
        raise ValueError("render runtime differs")
    report: dict[str, object] = {
        "artifact_type": OUTPUT_ARTIFACT,
        "query_count": int(len(names)),
        "dense_render_query_count": int(np.sum(required)),
        "dense_render_fraction": float(np.mean(required)),
        "query_pose_or_ground_truth_read": False,
        "retained_render_rows_exact": True,
        "omitted_rows_have_no_dense_score": True,
        "consensus_branches_exact": True,
        "selected_pose_arrays_exact": True,
        "selected_alternate_count": int(np.sum(sparse_branch == 1)),
        "full_elapsed_seconds_primary_alternate": full_seconds.tolist(),
        "sparse_elapsed_seconds_primary_alternate": sparse_seconds.tolist(),
        "speedup_primary_alternate": (full_seconds / sparse_seconds).tolist(),
        "render_plan_file_sha256": file_sha256(args.render_plan),
        "render_plan_content_sha256": plan_meta.get("content_sha256"),
        "full_consensus_file_sha256": file_sha256(args.full_consensus),
        "sparse_consensus_file_sha256": file_sha256(args.sparse_consensus),
        "full_render_file_sha256_primary_alternate": [
            file_sha256(args.full_primary_render), file_sha256(args.full_alternate_render),
        ],
        "sparse_render_file_sha256_primary_alternate": [
            file_sha256(args.sparse_primary_render), file_sha256(args.sparse_alternate_render),
        ],
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
