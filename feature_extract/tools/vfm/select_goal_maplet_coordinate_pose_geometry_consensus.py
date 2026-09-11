"""Collapse V5/V11 using sparse and dense MoGe/map geometry evidence.

The sparse plane test uses frozen query-region/map-plane associations and one
fitted MoGe3 scale.  The dense test renders the complete 2DGS surface and
measures query-valid pixels whose rendered/MoGe3 normals agree within the
already established 20 degree convention.  V11 replaces V5 only when it is
strictly better on both tests; there is no learned or continuous fusion weight.

An opt-in policy uses dense evidence alone only when both sparse objectives
are positive infinity; default selection remains unchanged. These two geometry
tests share MoGe input and are not statistically independent.

The selected NPZ is written and strongly reloaded before label-bearing endpoint
evaluation reports are opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import THRESHOLDS
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


PLANE_ARTIFACT = "goal_maplet_cross_coordinate_moge3_geometry_pose_selection_v1"
RENDER_ARTIFACT = "goal_maplet_frozen_pnp_moge3_2dgs_render_consistency_v1"
OUTPUT_ARTIFACT = "goal_maplet_coordinate_pose_geometry_consensus_v1"


def _canonical_report(path: Path, artifact_type: str) -> dict[str, object]:
    report = json.loads(path.read_text())
    expected = canonical_json_sha256({
        key: value for key, value in report.items() if key != "content_sha256"
    })
    if (
        report.get("artifact_type") != artifact_type
        or report.get("content_sha256") != expected
    ):
        raise ValueError(f"non-canonical {artifact_type} report")
    return report


def _load_plane_geometry(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    expected = canonical_json_sha256({
        key: value for key, value in metadata.items() if key != "content_sha256"
    })
    count = len(arrays.get("names", []))
    if (
        metadata.get("artifact_type") != PLANE_ARTIFACT
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("candidate_specific_association_selection") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or metadata.get("content_sha256") != expected
        or arrays.get("moge3_plane_geometry_objective", np.empty(0)).shape != (count, 2)
        or arrays.get("moge3_fitted_depth_scale", np.empty(0)).shape != (count, 2)
    ):
        raise ValueError("sparse plane-geometry authority differs")
    return arrays, metadata


def _load_render(path: Path) -> dict[str, object]:
    report = _canonical_report(path, RENDER_ARTIFACT)
    rows = report.get("rows")
    if (
        report.get("query_pose_or_ground_truth_read") is not False
        or report.get("query_depth_changes_frozen_pose") is not False
        or report.get("query_depth_or_scale_used_by_pose_solver") is not False
        or not isinstance(rows, list)
        or len(rows) != int(report.get("query_count", -1))
    ):
        raise ValueError("dense rendered-geometry authority differs")
    return report


def _normal_good_ray_score(row: dict[str, object], prefix: str = "") -> float:
    if prefix not in {"", "planar_"}:
        raise ValueError("normal-score domain differs")
    query = int(row.get(f"{prefix}query_valid_pixel_count", 0))
    common = int(row.get(f"{prefix}common_valid_pixel_count", 0))
    conditional = float(row.get(f"{prefix}normal_within_20deg", 0.0))
    if query <= 0 or common < 0 or common > query or not np.isfinite(conditional):
        return -np.inf
    return float(conditional * common / query)


def _sparse_allows_alternate(objective: np.ndarray, policy: str = "retain_primary") -> np.ndarray:
    value = np.asarray(objective, np.float64)
    if value.ndim != 2 or value.shape[1:] != (2,):
        raise ValueError("sparse objective shape differs")
    if policy not in {"retain_primary", "dense_when_both_missing"}:
        raise ValueError("unknown missing sparse policy")
    allowed = value[:, 1] < value[:, 0]
    if policy == "dense_when_both_missing":
        # Missing evidence is represented by positive infinity, not NaN or -inf.
        allowed |= np.isposinf(value).all(axis=1)
    return allowed


def _select_geometry_consensus(
    objectives: np.ndarray,
    normal_good_ray: np.ndarray,
    usable: np.ndarray,
    missing_sparse_policy: str = "retain_primary",
) -> np.ndarray:
    objective = np.asarray(objectives, np.float64)
    normal = np.asarray(normal_good_ray, np.float64)
    valid = np.asarray(usable, bool)
    if objective.shape != normal.shape or objective.shape != valid.shape or objective.shape[1:] != (2,):
        raise ValueError("geometry-consensus candidate arrays differ")
    prefer_alternate = _sparse_allows_alternate(objective, missing_sparse_policy) & (normal[:, 1] > normal[:, 0])
    return ((~valid[:, 0] & valid[:, 1]) | (valid[:, 1] & prefer_alternate)).astype(np.int8)


def _load_selected(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    expected = canonical_json_sha256({
        key: value for key, value in metadata.items() if key != "content_sha256"
    })
    count = len(arrays.get("names", []))
    if (
        metadata.get("artifact_type") != OUTPUT_ARTIFACT
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("selection_has_continuous_fusion_weight") is not False
        or metadata.get("missing_sparse_policy", "retain_primary") not in {"retain_primary", "dense_when_both_missing"}
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or metadata.get("content_sha256") != expected
        or arrays.get("pose_w2c", np.empty(0)).shape != (count, 4, 4)
        or arrays.get("usable", np.empty(0)).shape != (count,)
        or arrays.get("selected_branch", np.empty(0)).shape != (count,)
        or arrays.get("candidate_plane_geometry_objective", np.empty(0)).shape != (count, 2)
        or arrays.get("candidate_dense_normal_good_ray_recall", np.empty(0)).shape != (count, 2)
        or np.any(~np.isin(arrays.get("selected_branch", np.empty(0)), [0, 1]))
    ):
        raise ValueError("geometry-consensus selected-pose contract differs")
    return arrays, metadata


def _load_endpoint_evaluation(
    path: Path, pose_path: Path, pose_content: str, names: np.ndarray,
) -> list[dict[str, object]]:
    report = json.loads(path.read_text())
    rows = report.get("rows")
    if (
        report.get("pose_frozen_before_query_pose_or_ground_truth_open") is not True
        or report.get("frozen_pose_inventory_file_sha256") != file_sha256(pose_path)
        or report.get("frozen_pose_inventory_content_sha256") != pose_content
        or not isinstance(rows, list)
        or [str(row.get("name")) for row in rows] != names.astype(str).tolist()
    ):
        raise ValueError("endpoint label evaluation does not bind the candidate pose")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose", type=Path, required=True)
    parser.add_argument("--alternate_pose", type=Path, required=True)
    parser.add_argument("--plane_geometry", type=Path, required=True)
    parser.add_argument("--primary_render", type=Path, required=True)
    parser.add_argument("--alternate_render", type=Path, required=True)
    parser.add_argument(
        "--sparse_first_render_plan", type=Path,
        help="Required whenever the render reports use sparse-first evaluation.",
    )
    parser.add_argument("--primary_evaluation", type=Path, required=True)
    parser.add_argument("--alternate_evaluation", type=Path, required=True)
    parser.add_argument(
        "--dense_normal_domain",
        choices=("all_valid_query", "observed_query_planes"),
        default="all_valid_query",
    )
    parser.add_argument("--missing_sparse_policy", choices=("retain_primary", "dense_when_both_missing"), default="retain_primary")
    parser.add_argument("--output_frozen_pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose.exists():
        raise FileExistsError("refusing to overwrite geometry-consensus output")

    primary, primary_meta = _load_pose_candidate(args.primary_pose)
    alternate, alternate_meta = _load_pose_candidate(args.alternate_pose)
    plane, plane_meta = _load_plane_geometry(args.plane_geometry)
    primary_render = _load_render(args.primary_render)
    alternate_render = _load_render(args.alternate_render)
    names = primary["names"].astype(str)
    if not (
        np.array_equal(names, alternate["names"].astype(str))
        and np.array_equal(names, plane["names"].astype(str))
        and [str(row.get("name")) for row in primary_render["rows"]] == names.tolist()
        and [str(row.get("name")) for row in alternate_render["rows"]] == names.tolist()
        and plane_meta.get("point_pose_file_sha256") == file_sha256(args.primary_pose)
        and plane_meta.get("surface_pose_file_sha256") == file_sha256(args.alternate_pose)
        and primary_render.get("frozen_pose_inventory_file_sha256") == file_sha256(args.primary_pose)
        and alternate_render.get("frozen_pose_inventory_file_sha256") == file_sha256(args.alternate_pose)
    ):
        raise ValueError("geometry-consensus query or candidate lineage differs")
    candidate_pose = np.stack([primary["pose_w2c"], alternate["pose_w2c"]], axis=1)
    candidate_usable = np.stack([primary["usable"], alternate["usable"]], axis=1).astype(bool)
    normal_prefix = "" if args.dense_normal_domain == "all_valid_query" else "planar_"
    normal_good_ray = np.stack([
        np.asarray([_normal_good_ray_score(row, normal_prefix) for row in primary_render["rows"]]),
        np.asarray([_normal_good_ray_score(row, normal_prefix) for row in alternate_render["rows"]]),
    ], axis=1)
    objectives = np.asarray(plane["moge3_plane_geometry_objective"], np.float64)
    sparse_plan_sha = primary_render.get("sparse_first_render_plan_file_sha256")
    alternate_sparse_plan_sha = alternate_render.get("sparse_first_render_plan_file_sha256")
    sparse_plan_content = primary_render.get("sparse_first_render_plan_content_sha256")
    alternate_sparse_plan_content = alternate_render.get(
        "sparse_first_render_plan_content_sha256"
    )
    sparse_first_enabled = sparse_plan_sha is not None or alternate_sparse_plan_sha is not None
    if sparse_first_enabled:
        if args.sparse_first_render_plan is None:
            raise ValueError("sparse-first render reports require the external render plan")
        from feature_extract.tools.vfm.build_goal_maplet_sparse_first_render_plan import (
            _load_sparse_first_plan,
        )
        sparse_plan, sparse_plan_meta = _load_sparse_first_plan(
            args.sparse_first_render_plan,
        )
        required = (
            candidate_usable[:, 0]
            & candidate_usable[:, 1]
            & _sparse_allows_alternate(objectives, args.missing_sparse_policy)
        )
        primary_evaluated = np.asarray([
            row.get("dense_score_evaluated") is True for row in primary_render["rows"]
        ], bool)
        alternate_evaluated = np.asarray([
            row.get("dense_score_evaluated") is True for row in alternate_render["rows"]
        ], bool)
        if (
            sparse_plan_meta.get("missing_sparse_policy", "retain_primary") != args.missing_sparse_policy
            or sparse_plan_sha != alternate_sparse_plan_sha
            or sparse_plan_sha != file_sha256(args.sparse_first_render_plan)
            or sparse_plan_content is None
            or sparse_plan_content != alternate_sparse_plan_content
            or sparse_plan_content != sparse_plan_meta.get("content_sha256")
            or not np.array_equal(names, sparse_plan["names"].astype(str))
            or not np.array_equal(required, sparse_plan["dense_render_required"])
            or sparse_plan_meta.get("primary_pose_file_sha256") != file_sha256(args.primary_pose)
            or sparse_plan_meta.get("primary_pose_content_sha256") != primary_meta.get("content_sha256")
            or sparse_plan_meta.get("alternate_pose_file_sha256") != file_sha256(args.alternate_pose)
            or sparse_plan_meta.get("alternate_pose_content_sha256") != alternate_meta.get("content_sha256")
            or sparse_plan_meta.get("plane_geometry_file_sha256") != file_sha256(args.plane_geometry)
            or sparse_plan_meta.get("plane_geometry_content_sha256") != plane_meta.get("content_sha256")
            or not np.array_equal(primary_evaluated, required)
            or not np.array_equal(alternate_evaluated, required)
            or int(primary_render.get("dense_render_query_count", -1)) != int(np.sum(required))
            or int(alternate_render.get("dense_render_query_count", -1)) != int(np.sum(required))
        ):
            raise ValueError("sparse-first dense-render short circuit differs from selection rule")
    else:
        if args.sparse_first_render_plan is not None:
            raise ValueError("external sparse-first plan supplied for complete render reports")
        if any(
            "dense_score_evaluated" in row
            for report in (primary_render, alternate_render)
            for row in report["rows"]
        ):
            raise ValueError("dense-render evaluation mask lacks a sparse-first authority")
    selected = _select_geometry_consensus(objectives, normal_good_ray, candidate_usable, args.missing_sparse_policy)
    row = np.arange(len(names))
    arrays = {
        "names": primary["names"],
        "pose_w2c": candidate_pose[row, selected],
        "usable": candidate_usable[row, selected],
        "selected_branch": selected,
        "candidate_plane_geometry_objective": objectives,
        "candidate_dense_normal_good_ray_recall": normal_good_ray,
        "candidate_moge3_depth_scale": np.asarray(plane["moge3_fitted_depth_scale"], np.float64),
    }
    metadata: dict[str, object] = {
        "artifact_type": OUTPUT_ARTIFACT,
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "candidate_order": "retained_point_coordinate_V5_then_continuous_chart_coordinate_V11",
        "selection_rule": (
            "V11_only_if_strictly_better_on_frozen_sparse_plane_scale_geometry_AND_"
            f"dense_2DGS_MoGe3_normal20_{args.dense_normal_domain}_good_ray_recall_else_V5;"
            f"missing_sparse_policy={args.missing_sparse_policy}"
        ),
        "missing_sparse_policy": args.missing_sparse_policy,
        "missing_sparse_override": "both_positive_infinity_only_then_strict_dense_improvement" if args.missing_sparse_policy == "dense_when_both_missing" else "none",
        "dense_normal_domain": args.dense_normal_domain,
        "dense_normal_score_denominator": (
            "all_valid_query_MoGe3_pixels_missing_render_is_failure"
            if args.dense_normal_domain == "all_valid_query"
            else "observed_finite_query_plane_pixels_missing_render_is_failure"
        ),
        "normal_threshold_degrees": 20.0,
        "dense_render_evaluation_policy": (
            "sparse_first_exact_logical_short_circuit"
            if sparse_first_enabled else "all_usable_candidates"
        ),
        "sparse_first_render_plan_file_sha256": sparse_plan_sha,
        "sparse_first_render_plan_content_sha256": sparse_plan_content,
        "dense_render_query_count": (
            int(np.sum(required)) if sparse_first_enabled else int(len(names))
        ),
        "selection_has_continuous_fusion_weight": False,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_for_selection": True,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "primary_pose_file_sha256": file_sha256(args.primary_pose),
        "primary_pose_content_sha256": primary_meta.get("content_sha256"),
        "alternate_pose_file_sha256": file_sha256(args.alternate_pose),
        "alternate_pose_content_sha256": alternate_meta.get("content_sha256"),
        "plane_geometry_file_sha256": file_sha256(args.plane_geometry),
        "plane_geometry_content_sha256": plane_meta.get("content_sha256"),
        "primary_render_file_sha256": file_sha256(args.primary_render),
        "primary_render_content_sha256": primary_render.get("content_sha256"),
        "alternate_render_file_sha256": file_sha256(args.alternate_render),
        "alternate_render_content_sha256": alternate_render.get("content_sha256"),
        "selected_alternate_count": int(np.sum(selected == 1)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_frozen_pose.with_name(args.output_frozen_pose.name + ".temporary.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(args.output_frozen_pose)
    arrays, metadata = _load_selected(args.output_frozen_pose)

    # Phase 2 starts only after the immutable selected-pose artifact reloads.
    primary_rows = _load_endpoint_evaluation(
        args.primary_evaluation, args.primary_pose,
        str(primary_meta.get("content_sha256")), arrays["names"],
    )
    alternate_rows = _load_endpoint_evaluation(
        args.alternate_evaluation, args.alternate_pose,
        str(alternate_meta.get("content_sha256")), arrays["names"],
    )
    primary_error = np.asarray([
        [float(item["translation_error_m"]), float(item["rotation_error_deg"])]
        for item in primary_rows
    ])
    alternate_error = np.asarray([
        [float(item["translation_error_m"]), float(item["rotation_error_deg"])]
        for item in alternate_rows
    ])
    selected_error = np.where(selected[:, None] == 1, alternate_error, primary_error)
    primary_hits: dict[str, int] = {}
    selected_hits: dict[str, int] = {}
    for translation_limit, rotation_limit in THRESHOLDS:
        key = f"{translation_limit:g}m_{rotation_limit:g}deg"
        primary_hits[key] = int(np.sum(
            (primary_error[:, 0] <= translation_limit)
            & (primary_error[:, 1] <= rotation_limit)
        ))
        selected_hits[key] = int(np.sum(
            (selected_error[:, 0] <= translation_limit)
            & (selected_error[:, 1] <= rotation_limit)
        ))
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_coordinate_pose_geometry_consensus_evaluation_v1",
        "pose_frozen_before_label_evaluations_open": True,
        "frozen_pose_file_sha256": file_sha256(args.output_frozen_pose),
        "frozen_pose_content_sha256": metadata["content_sha256"],
        "query_count": int(len(names)),
        "selected_alternate_count": int(np.sum(selected == 1)),
        "primary_threshold_hit_counts": primary_hits,
        "threshold_hit_counts": selected_hits,
        "threshold_hit_count_delta": {
            key: int(selected_hits[key] - primary_hits[key]) for key in selected_hits
        },
        "primary_median_translation_m": float(np.median(primary_error[:, 0])),
        "primary_median_rotation_deg": float(np.median(primary_error[:, 1])),
        "median_translation_m": float(np.median(selected_error[:, 0])),
        "median_rotation_deg": float(np.median(selected_error[:, 1])),
        "p90_translation_m": float(np.quantile(selected_error[:, 0], 0.9)),
        "p90_rotation_deg": float(np.quantile(selected_error[:, 1], 0.9)),
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
