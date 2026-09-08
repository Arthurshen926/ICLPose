"""Post-label stage attribution for frozen chart-surface localization outputs.

This is a diagnostic, never a selector.  It separates final 2m/45deg failures
into missing retrieved coordinate support, degenerate candidate geometry,
initialization failure, downstream refinement regression, and final selector
failure.  All pose/correspondence inputs are frozen and hash-bound before a
query contributor (the only label-bearing input) is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_continuous_coordinate_pose_oracle import (
    _oracle_candidate_rows,
    _pose_error,
    _solve_rows,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_correspondences,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _token_pixels
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


COARSE_TRANSLATION_M = 2.0
COARSE_ROTATION_DEG = 45.0
ORACLE_REPROJECTION_PX = 4.0
MINIMUM_ORACLE_ROWS = 6
MINIMUM_ORACLE_PLANES = 2


def _hit(error: tuple[float, float]) -> bool:
    return bool(error[0] <= COARSE_TRANSLATION_M and error[1] <= COARSE_ROTATION_DEG)


def _failure_category(
    *,
    selected_hit: bool,
    oracle_row_count: int,
    oracle_plane_count: int,
    oracle_exact_hit: bool,
    primary_pnp_hit: bool,
    alternate_pnp_hit: bool,
    primary_final_hit: bool,
    alternate_final_hit: bool,
) -> str:
    """Return the first causal stage supported by the frozen evidence."""
    if selected_hit:
        return "selected_pose_coarse_success"
    if oracle_row_count < MINIMUM_ORACLE_ROWS:
        return "retrieved_chart_or_uv_support_insufficient"
    if oracle_plane_count < MINIMUM_ORACLE_PLANES or not oracle_exact_hit:
        return "retrieved_candidate_geometry_degenerate"
    if primary_final_hit or alternate_final_hit:
        return "final_v5_v11_selector_failure"
    if not (primary_pnp_hit or alternate_pnp_hit):
        return "hard_coordinate_or_pnp_initialization_failure"
    return "pnp_candidate_collapse_or_post_pnp_refinement_regression"


def _pose_stage(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    try:
        arrays, metadata = _load_pose_candidate(path)
    except ValueError:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        if (
            metadata.get("artifact_type") != "goal_maplet_direct_plane_pnp_grouped_multihypothesis_v1"
            or not {
                "names", "candidate_offsets", "candidate_pose_w2c",
                "raw_inliers_pose_w2c", "balanced_support_pose_w2c",
                "supported_entities_pose_w2c",
            }.issubset(arrays)
            or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        ):
            raise ValueError("stage pose inventory contract differs")
    if metadata.get("query_pose_or_ground_truth_read") not in (False, None):
        raise ValueError("stage pose inventory is not label-free")
    return arrays, metadata


def _best_stage_error(
    arrays: dict[str, np.ndarray], query: int, target: np.ndarray,
) -> tuple[float, float]:
    """Return post-label best error within a frozen stage candidate inventory."""
    if "pose_w2c" in arrays:
        if not bool(arrays["usable"][query]):
            return float("inf"), float("inf")
        return _pose_error(arrays["pose_w2c"][query], target)
    lo, hi = map(int, arrays["candidate_offsets"][query:query + 2])
    candidates = [np.asarray(value, np.float64) for value in arrays["candidate_pose_w2c"][lo:hi]]
    candidates.extend([
        np.asarray(arrays["raw_inliers_pose_w2c"][query], np.float64),
        np.asarray(arrays["balanced_support_pose_w2c"][query], np.float64),
        np.asarray(arrays["supported_entities_pose_w2c"][query], np.float64),
    ])
    errors = [_pose_error(candidate, target) for candidate in candidates]
    if not errors:
        return float("inf"), float("inf")
    score = np.asarray([
        np.hypot(value[0] / COARSE_TRANSLATION_M, value[1] / COARSE_ROTATION_DEG)
        for value in errors
    ])
    return errors[int(np.argmin(score))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--selected_pose", type=Path, required=True)
    parser.add_argument("--primary_pnp", type=Path, required=True)
    parser.add_argument("--alternate_pnp", type=Path, required=True)
    parser.add_argument("--primary_final", type=Path, required=True)
    parser.add_argument("--alternate_final", type=Path, required=True)
    parser.add_argument("--point_correspondences", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite pose failure attribution")

    paths = {
        "selected": args.selected_pose,
        "primary_pnp": args.primary_pnp,
        "alternate_pnp": args.alternate_pnp,
        "primary_final": args.primary_final,
        "alternate_final": args.alternate_final,
    }
    stages = {key: _pose_stage(path) for key, path in paths.items()}
    corr, corr_meta = _load_correspondences(args.point_correspondences)
    names = np.asarray(corr["names"]).astype(str)
    for key, (arrays, _) in stages.items():
        if not np.array_equal(names, np.asarray(arrays["names"]).astype(str)):
            raise ValueError(f"{key} query order differs")

    rows: list[dict[str, object]] = []
    for query, name in enumerate(names.tolist()):
        contributor_path = args.query_contributors / name
        with np.load(contributor_path, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        errors: dict[str, tuple[float, float]] = {}
        for key, (arrays, _) in stages.items():
            errors[key] = _best_stage_error(arrays, query, target)

        lo, hi = map(int, corr["correspondence_offsets"][query:query + 2])
        world = np.asarray(corr["world_points"][lo:hi], np.float64)
        token = np.asarray(corr["query_tokens"][lo:hi], np.int64)
        provenance = np.asarray(
            corr["provenance_region_plane_atlas_row"][lo:hi], np.int64,
        )
        grid = tuple(corr_meta.get("token_grid", (36, 64)))
        pixel = _token_pixels(token, grid)
        if len(world):
            oracle_rows, projected, reprojection = _oracle_candidate_rows(
                target, world, token, pixel,
                np.asarray(corr["camera_matrices"][query], np.float64),
                float(corr["radial_k1"][query]),
                maximum_error_px=ORACLE_REPROJECTION_PX,
            )
        else:
            oracle_rows = np.zeros(0, np.int64)
            projected = np.zeros((0, 2), np.float64)
            reprojection = np.zeros(0, np.float64)
        oracle_planes = np.unique(provenance[oracle_rows, 1]) if len(oracle_rows) else np.zeros(0)
        oracle_pose = (
            _solve_rows(
                world[oracle_rows], token[oracle_rows], projected[oracle_rows],
                np.asarray(corr["camera_matrices"][query], np.float64),
                float(corr["radial_k1"][query]),
            )
            if len(oracle_rows) >= MINIMUM_ORACLE_ROWS and len(oracle_planes) >= MINIMUM_ORACLE_PLANES
            else None
        )
        oracle_error = _pose_error(oracle_pose, target)
        hits = {key: _hit(value) for key, value in errors.items()}
        category = _failure_category(
            selected_hit=hits["selected"],
            oracle_row_count=int(len(oracle_rows)),
            oracle_plane_count=int(len(oracle_planes)),
            oracle_exact_hit=_hit(oracle_error),
            primary_pnp_hit=hits["primary_pnp"],
            alternate_pnp_hit=hits["alternate_pnp"],
            primary_final_hit=hits["primary_final"],
            alternate_final_hit=hits["alternate_final"],
        )
        finite_reprojection = reprojection[np.isfinite(reprojection)]
        rows.append({
            "name": name,
            "final_failure_category": category,
            "selected_is_2m_45deg_hit": hits["selected"],
            "oracle_candidate_row_count": int(len(oracle_rows)),
            "oracle_physical_plane_count": int(len(oracle_planes)),
            "oracle_exact_projection_is_2m_45deg_hit": _hit(oracle_error),
            "minimum_candidate_gt_reprojection_px": (
                None if not len(finite_reprojection) else float(np.min(finite_reprojection))
            ),
            "oracle_exact_projection_translation_error_m": float(oracle_error[0]),
            "oracle_exact_projection_rotation_error_deg": float(oracle_error[1]),
            "stage_errors": {
                key: {"translation_m": float(value[0]), "rotation_deg": float(value[1]),
                      "is_2m_45deg_hit": hits[key]}
                for key, value in errors.items()
            },
        })

    categories = sorted({row["final_failure_category"] for row in rows})
    category_counts = {
        category: int(sum(row["final_failure_category"] == category for row in rows))
        for category in categories
    }
    failure_rows = [row for row in rows if not row["selected_is_2m_45deg_hit"]]
    input_lineage = {
        key: {
            "file_sha256": file_sha256(path),
            "content_sha256": stages[key][1].get("content_sha256"),
        }
        for key, path in paths.items()
    }
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_pose_failure_stage_postlabel_audit_v1",
        "evaluation_role": "POSTLABEL_DIAGNOSTIC_ONLY_NOT_SELECTION_OR_TRAINING",
        "split_name": args.split_name,
        "query_count": int(len(names)),
        "coarse_threshold": {"translation_m": COARSE_TRANSLATION_M,
                             "rotation_deg": COARSE_ROTATION_DEG},
        "oracle_definition": "one_existing_retrieved_hypothesis_per_token_with_GT_reprojection_at_most_4px_then_exact_GT_projected_pixel_PnP",
        "correct_chart_vs_correct_uv_separable_without_depth_gt": False,
        "candidate_absence_category_combines_chart_retrieval_and_within_chart_uv_support": True,
        "selected_coarse_success_count": int(sum(row["selected_is_2m_45deg_hit"] for row in rows)),
        "selected_coarse_failure_count": int(len(failure_rows)),
        "failure_category_counts": {
            category: int(sum(row["final_failure_category"] == category for row in failure_rows))
            for category in categories if category != "selected_pose_coarse_success"
        },
        "all_category_counts": category_counts,
        "frozen_input_lineage": input_lineage,
        "point_correspondence_file_sha256": file_sha256(args.point_correspondences),
        "point_correspondence_content_sha256": corr_meta.get("content_sha256"),
        "query_pose_or_ground_truth_read": True,
        "selection_or_training_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
