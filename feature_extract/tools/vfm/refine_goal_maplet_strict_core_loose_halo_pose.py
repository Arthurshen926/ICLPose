"""Refine one pose with a strict correspondence core and low-weight loose halo.

The 0.25 m and 1.0 m homography inventories use the same anonymous atlas and
mapping-only subtoken head.  The strict inventory remains the pose authority.
For a query token, a loose-only hypothesis is admitted only when no strict
hypothesis is geometrically valid at the frozen input pose, and its standard
deviation is inflated by the fixed homography-threshold ratio.  Acceptance is
scored only on the unchanged strict inventory, preventing added halo support
from grading its own update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_correspondences,
    _score,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import (
    MAXIMUM_REPROJECTION_ERROR_PX,
    MINIMUM_GLOBAL_SUPPORT_FRACTION,
    _evaluate,
    _fixed_reprojection_sigma_px,
    _project,
    _refine_pose,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


STRICT_THRESHOLD_M = 0.25
LOOSE_THRESHOLD_M = 1.0
LOOSE_SIGMA_MULTIPLIER = LOOSE_THRESHOLD_M / STRICT_THRESHOLD_M


def _paired_contract(
    strict: dict[str, np.ndarray], strict_meta: dict[str, object],
    loose: dict[str, np.ndarray], loose_meta: dict[str, object],
) -> None:
    schema = "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5"
    if strict_meta.get("artifact_type") != schema or loose_meta.get("artifact_type") != schema:
        raise ValueError("core/halo refinement requires two V5 inventories")
    if not (
        float(strict_meta.get("homography_threshold_m", -1.0)) == STRICT_THRESHOLD_M
        and float(loose_meta.get("homography_threshold_m", -1.0)) == LOOSE_THRESHOLD_M
    ):
        raise ValueError("core/halo homography thresholds differ")
    for key in ("names", "camera_matrices", "radial_k1"):
        if not np.array_equal(strict[key], loose[key]):
            raise ValueError(f"core/halo query contract differs: {key}")
    for key in (
        "plane_uv_atlas_content_sha256", "mapping_subtoken_head_content_sha256",
        "plane_ranking_file_sha256", "query_camera_only_inventory_content_sha256",
        "chart_local_radio_projection_content_sha256", "hypotheses_per_query_token",
        "topk_planes", "query_support_policy",
    ):
        if strict_meta.get(key) != loose_meta.get(key):
            raise ValueError(f"core/halo lineage differs: {key}")


def _keys(corr: dict[str, np.ndarray], lo: int, hi: int) -> np.ndarray:
    return np.c_[
        np.asarray(corr["query_tokens"][lo:hi], np.int64),
        np.asarray(corr["provenance_region_plane_atlas_row"][lo:hi], np.int64),
        np.asarray(corr["prototype_atlas_row"][lo:hi], np.int64),
    ]


def _exclusive_loose_rows(
    strict: dict[str, np.ndarray], strict_lo: int, strict_hi: int,
    loose: dict[str, np.ndarray], loose_lo: int, loose_hi: int,
) -> np.ndarray:
    strict_key = {tuple(row.tolist()) for row in _keys(strict, strict_lo, strict_hi)}
    loose_key = _keys(loose, loose_lo, loose_hi)
    selected = np.asarray([
        loose_lo + row for row, key in enumerate(loose_key.tolist())
        if tuple(key) not in strict_key
    ], np.int64)
    return selected


def _core_first_hypotheses(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    query_pixels: np.ndarray,
    is_loose_halo: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    *,
    maximum_error_px: float = MAXIMUM_REPROJECTION_ERROR_PX,
) -> tuple[np.ndarray, np.ndarray]:
    token = np.asarray(query_tokens, np.int64).reshape(-1)
    pixel = np.asarray(query_pixels, np.float64).reshape(-1, 2)
    halo = np.asarray(is_loose_halo, bool).reshape(-1)
    if not (len(token) == len(pixel) == len(halo)):
        raise ValueError("core/halo selection arrays differ")
    projected, camera = _project(pose_w2c, world_points, camera_matrix, radial_k1)
    error = np.linalg.norm(projected - pixel, axis=1)
    valid = (camera[:, 2] > 0.0) & np.isfinite(error) & (error <= float(maximum_error_px))
    chosen: list[int] = []
    for token_id in np.unique(token):
        core = np.flatnonzero((token == token_id) & valid & ~halo)
        rows = core if len(core) else np.flatnonzero((token == token_id) & valid & halo)
        if len(rows):
            chosen.append(int(rows[np.argmin(error[rows])]))
    return np.asarray(chosen, np.int64), error


def _strict_acceptance(
    initial_score: dict[str, object], final_score: dict[str, object],
) -> bool:
    initial_median = initial_score["reprojection_median_px"]
    final_median = final_score["reprojection_median_px"]
    return bool(
        int(final_score["inlier_count"]) >= max(
            6,
            int(np.floor(
                MINIMUM_GLOBAL_SUPPORT_FRACTION * int(initial_score["inlier_count"])
            )),
        )
        and initial_median is not None and final_median is not None
        and float(final_median) <= float(initial_median) + 0.25
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--strict_correspondences", type=Path, required=True)
    parser.add_argument("--loose_correspondences", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite core/halo refinement")

    poses, pose_meta = _load_frozen_poses(args.frozen_pose_inventory)
    strict, strict_meta = _load_correspondences(args.strict_correspondences)
    loose, loose_meta = _load_correspondences(args.loose_correspondences)
    _paired_contract(strict, strict_meta, loose, loose_meta)
    names = strict["names"].astype(str)
    if not np.array_equal(names, poses["names"].astype(str)):
        raise ValueError("core/halo pose query order differs")

    output_pose = np.asarray(poses["pose_w2c"], np.float64).copy()
    usable = np.asarray(poses["usable"], bool).copy()
    output_inliers = np.asarray(poses["pnp_inlier_count"], np.int64).copy()
    accepted = np.zeros(len(names), bool)
    selected_core = np.zeros(len(names), np.int64)
    selected_halo = np.zeros(len(names), np.int64)
    union_count = np.zeros(len(names), np.int64)
    diagnostics: list[dict[str, object]] = []
    for query, name in enumerate(names.tolist()):
        row: dict[str, object] = {"name": name, "usable": bool(usable[query])}
        if not usable[query]:
            diagnostics.append(row)
            continue
        slo, shi = map(int, strict["correspondence_offsets"][query:query + 2])
        llo, lhi = map(int, loose["correspondence_offsets"][query:query + 2])
        halo_rows = _exclusive_loose_rows(strict, slo, shi, loose, llo, lhi)

        def concatenate(key: str) -> np.ndarray:
            return np.concatenate((strict[key][slo:shi], loose[key][halo_rows]), axis=0)

        world = concatenate("world_points").astype(np.float64)
        token = concatenate("query_tokens").astype(np.int64)
        pixel = concatenate("query_measurements_xy").astype(np.float64)
        provenance = concatenate("provenance_region_plane_atlas_row").astype(np.int64)
        purity = concatenate("prototype_plane_pixel_purity").astype(np.float64)
        footprint = concatenate("prototype_world_covariance_m2").astype(np.float64)
        dispersion = concatenate("prototype_plane_depth_dispersion_m").astype(np.float64)
        query_variance = concatenate("query_measurement_variance_px2").astype(np.float64)
        is_halo = np.r_[np.zeros(shi - slo, bool), np.ones(len(halo_rows), bool)]
        K = np.asarray(strict["camera_matrices"][query], np.float64)
        k1 = float(strict["radial_k1"][query])
        union_count[query] = len(world)
        sigma = _fixed_reprojection_sigma_px(
            output_pose[query], world, footprint, purity, dispersion, K,
            query_measurement_variance_px2=query_variance,
            include_map_footprint_scatter=False,
            radial_k1=k1,
        )
        sigma[is_halo] *= LOOSE_SIGMA_MULTIPLIER
        selected, _ = _core_first_hypotheses(
            output_pose[query], world, token, pixel, is_halo, K, k1,
        )
        selected_core[query] = int(np.sum(~is_halo[selected]))
        selected_halo[query] = int(np.sum(is_halo[selected]))
        initial_score = _score(
            output_pose[query], strict["world_points"][slo:shi],
            strict["query_tokens"][slo:shi],
            strict["provenance_region_plane_atlas_row"][slo:shi], K, k1,
            strict["query_measurements_xy"][slo:shi],
        )
        refined, use, detail = _refine_pose(
            output_pose[query], world[selected], pixel[selected],
            provenance[selected, 1], sigma[selected], K, k1,
        )
        final_score = _score(
            refined, strict["world_points"][slo:shi],
            strict["query_tokens"][slo:shi],
            strict["provenance_region_plane_atlas_row"][slo:shi], K, k1,
            strict["query_measurements_xy"][slo:shi],
        )
        strict_pass = _strict_acceptance(initial_score, final_score)
        use = bool(use and strict_pass)
        if use:
            output_pose[query] = refined
            output_inliers[query] = int(final_score["inlier_count"])
            accepted[query] = True
        row.update({
            **detail,
            "refinement_accepted": bool(use),
            "strict_acceptance_pass": bool(strict_pass),
            "strict_correspondence_count": int(shi - slo),
            "loose_exclusive_correspondence_count": int(len(halo_rows)),
            "selected_strict_core_count": int(selected_core[query]),
            "selected_loose_halo_count": int(selected_halo[query]),
            "initial_strict_inlier_count": int(initial_score["inlier_count"]),
            "final_strict_inlier_count": int(final_score["inlier_count"]),
            "initial_strict_reprojection_median_px": initial_score["reprojection_median_px"],
            "final_strict_reprojection_median_px": final_score["reprojection_median_px"],
        })
        diagnostics.append(row)

    arrays = {
        "names": strict["names"],
        "pose_w2c": output_pose,
        "usable": usable,
        "candidate_correspondence_count": union_count,
        "pnp_inlier_count": output_inliers,
        "core_halo_refinement_accepted": accepted,
        "selected_strict_core_count": selected_core,
        "selected_loose_halo_count": selected_halo,
    }
    metadata = {
        "artifact_type": "goal_maplet_strict_core_loose_halo_surface_pose_refinement_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "strict_core_threshold_m": STRICT_THRESHOLD_M,
        "loose_halo_threshold_m": LOOSE_THRESHOLD_M,
        "loose_halo_sigma_multiplier": LOOSE_SIGMA_MULTIPLIER,
        "fixed_hypothesis_rule": (
            "one_nearest_strict_valid_hypothesis_per_token_else_one_nearest_loose_exclusive_valid_hypothesis"
        ),
        "acceptance_evidence": "strict_core_only_raw_support_and_reprojection",
        "frozen_pose_inventory_file_sha256": file_sha256(args.frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": pose_meta.get("content_sha256"),
        "strict_correspondence_file_sha256": file_sha256(args.strict_correspondences),
        "strict_correspondence_content_sha256": strict_meta.get("content_sha256"),
        "loose_correspondence_file_sha256": file_sha256(args.loose_correspondences),
        "loose_correspondence_content_sha256": loose_meta.get("content_sha256"),
        "accepted_count": int(np.sum(accepted)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    rows, translation, rotation, finite = _evaluate(
        arrays["names"], arrays["pose_w2c"], arrays["usable"], diagnostics,
        args.query_contributors,
    )
    report = {
        "artifact_type": "goal_maplet_strict_core_loose_halo_surface_pose_evaluation_v1",
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(len(rows)),
        "usable_count": int(np.sum(finite)),
        "accepted_count": int(np.sum(accepted)),
        "selected_strict_core_total": int(np.sum(selected_core)),
        "selected_loose_halo_total": int(np.sum(selected_halo)),
        "median_translation_m": float(np.median(translation[finite])),
        "median_rotation_deg": float(np.median(rotation[finite])),
        "p90_translation_m": float(np.quantile(translation[finite], 0.9)),
        "p90_rotation_deg": float(np.quantile(rotation[finite], 0.9)),
        "recall_2m45": float(np.mean(finite & (translation <= 2.0) & (rotation <= 45.0))),
        "recall_1m10": float(np.mean(finite & (translation <= 1.0) & (rotation <= 10.0))),
        "recall_0p5m5": float(np.mean(finite & (translation <= 0.5) & (rotation <= 5.0))),
        "recall_0p25m2": float(np.mean(finite & (translation <= 0.25) & (rotation <= 2.0))),
        "recall_0p1m1": float(np.mean(finite & (translation <= 0.1) & (rotation <= 1.0))),
        "rows": rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
