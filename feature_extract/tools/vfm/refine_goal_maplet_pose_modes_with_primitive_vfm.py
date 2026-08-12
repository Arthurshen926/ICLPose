"""Refine selected Goal-Maplet poses with the sole canonical primitive VFM field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import (
    build_run_manifest,
    canonical_json_sha256,
    capture_repository_state,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.primitive_pose_refiner import (
    refine_pose_with_primitive_vfm_score,
)
from feature_extract.vfm.localization_goal_maplet.pose_proposal import (
    _rotation_distance_degrees,
)
from feature_extract.vfm.localization_goal_maplet.sparse_vfm_pose_likelihood import (
    PRIMITIVE_Z_TIE_BREAK,
    score_pose_conditioned_sparse_primitives,
)
from feature_extract.vfm.postselection_success_calibration import (
    candidate_source_postselection_evidence,
    typed_null_diagnostics,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _parse_positive_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in str(value).split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise ValueError("refinement trust-region steps must be positive")
    return result


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=np.asarray(data["camera_params"], dtype=np.float64),
        )


def _summary(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int]:
    translation = np.asarray([row[f"{prefix}_translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row[f"{prefix}_rotation_deg"] for row in rows], dtype=np.float64)
    return {
        "query_count": int(len(rows)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "strict_count": int(np.sum((translation <= 0.5) & (rotation <= 5.0))),
        "within_1m_10deg_count": int(np.sum((translation <= 1.0) & (rotation <= 10.0))),
        "catastrophic_count": int(np.sum((translation > 2.0) | (rotation > 20.0))),
    }


def _load_deployed_baseline_selection(
    payload: dict[str, object], baseline_report_sha256: str,
) -> dict[str, int]:
    """Read only the frozen policy decision, never its evaluation errors."""
    selected: dict[str, int] = {}
    matching_transfer_count = 0
    for transfer in payload.get("transfer_evaluation", []):
        if str(transfer.get("report_sha256")) != str(baseline_report_sha256):
            continue
        matching_transfer_count += 1
        for prediction in transfer.get("predictions", []):
            image_id = str(prediction["image_id"])
            if image_id in selected:
                raise ValueError(f"duplicate frozen baseline selection: {image_id}")
            selected[image_id] = int(prediction["selected_index"])
    if matching_transfer_count != 1:
        raise ValueError(
            "baseline selection report must contain exactly one transfer block "
            "for the first candidate report"
        )
    return selected


def _deployed_baseline_first(
    source: dict[str, object], mode_name: str, selected_source_index: int,
) -> list[dict[str, object]]:
    """Move the deployed pre-order candidate to rank zero without dropping states."""
    details = list(source.get("mode_details", {}).get(str(mode_name), []))
    original_order = [
        int(value) for value in source.get("ranking_diagnostics", {})
        .get(str(mode_name), {})
        .get("surface_alignment_original_indices", [])
    ]
    try:
        selected_ranked_index = original_order.index(int(selected_source_index))
    except ValueError as error:
        raise ValueError("frozen selected source index is absent from ranked modes") from error
    if selected_ranked_index >= len(details):
        raise ValueError("frozen selected source index resolves outside ranked modes")
    return [details[selected_ranked_index]] + [
        detail for index, detail in enumerate(details)
        if index != selected_ranked_index
    ]


def _select_refinement_candidate_indices(
    *,
    common_order: np.ndarray,
    common_initial_scores: np.ndarray,
    candidate_poses: np.ndarray,
    candidate_details: list[tuple[int, int, dict[str, object]]],
    maximum_count: int,
    policy: str,
    score_margin: float,
    minimum_count: int,
    translation_radius_m: float,
    rotation_radius_deg: float,
) -> tuple[list[int], dict[str, object]]:
    """Allocate refinement to score-competitive, structurally distinct basins."""

    order = np.asarray(common_order, dtype=np.int64).reshape(-1)
    scores = np.asarray(common_initial_scores, dtype=np.float64).reshape(-1)
    poses = np.asarray(candidate_poses, dtype=np.float64).reshape(-1, 4, 4)
    if scores.size != poses.shape[0] or order.size != scores.size:
        raise ValueError("refinement candidate arrays differ")
    if sorted(order.tolist()) != list(range(scores.size)):
        raise ValueError("common_order must be a permutation")
    budget = min(max(int(maximum_count), 1), int(order.size))
    minimum = min(max(int(minimum_count), 1), budget)
    margin = float(score_margin)
    eligible = [
        int(row)
        for rank, row in enumerate(order.tolist())
        if rank < minimum
        or margin < 0.0
        or float(scores[int(row)]) >= float(scores[int(order[0])]) - margin
    ]
    if policy == "score_topk":
        selected = eligible[:budget]
        return selected, {
            "policy": policy,
            "maximum_count": budget,
            "selected_count_before_baseline_guard": len(selected),
            "score_margin": margin,
            "minimum_count": minimum,
            "represented_anchor_count": 0,
            "selected_anchor_count": 0,
            "selected_pose_basin_count": len(selected),
        }
    if policy != "anchor_basin_cover":
        raise ValueError(f"unknown refinement candidate policy: {policy}")

    centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    anchors = [
        (int(report_index), int(detail.get("mapping_view_anchor_label", -1)))
        for report_index, _, detail in candidate_details
    ]
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_anchors: set[tuple[int, int]] = set()

    def is_new_basin(row: int) -> bool:
        return not any(
            np.linalg.norm(centers[row] - centers[other])
            < float(translation_radius_m)
            and _rotation_distance_degrees(poses[row], poses[other])
            < float(rotation_radius_deg)
            for other in selected
        )

    # First reserve one high-score basin per structural anchor.  If the budget
    # is smaller than the anchor count, stable score order chooses the anchors.
    for row in order.tolist():
        anchor = anchors[int(row)]
        if anchor in selected_anchors or not is_new_basin(int(row)):
            continue
        selected.append(int(row))
        selected_set.add(int(row))
        selected_anchors.add(anchor)
        if len(selected) >= budget:
            break
    anchor_reserved_count = len(selected)
    # Then add score-competitive distinct basins.  A finite margin makes the
    # actual workload query-adaptive while the anchor reservation prevents a
    # repeated facade from monopolizing it.
    for row in eligible:
        if len(selected) >= budget:
            break
        if row in selected_set or not is_new_basin(row):
            continue
        selected.append(row)
        selected_set.add(row)
    basin_count = len(selected)
    # A requested minimum is a compute contract even for degenerate duplicate
    # inputs, so fill it in global score order without claiming extra basins.
    for row in order.tolist():
        if len(selected) >= minimum:
            break
        if int(row) not in selected_set:
            selected.append(int(row))
            selected_set.add(int(row))
    return selected, {
        "policy": policy,
        "maximum_count": budget,
        "selected_count_before_baseline_guard": len(selected),
        "score_margin": margin,
        "minimum_count": minimum,
        "translation_radius_m": float(translation_radius_m),
        "rotation_radius_deg": float(rotation_radius_deg),
        "represented_anchor_count": len(set(anchors)),
        "selected_anchor_count": len(selected_anchors),
        "anchor_reserved_count": anchor_reserved_count,
        "selected_pose_basin_count": basin_count,
        "duplicate_minimum_fill_count": len(selected) - basin_count,
    }


def _baseline_refinement_index(
    refinements: list[dict[str, object]],
) -> int:
    """Resolve the deployed union candidate, not any mode from report zero."""

    matches = [
        index
        for index, value in enumerate(refinements)
        if int(value.get("union_candidate_index", -1)) == 0
    ]
    if len(matches) != 1:
        raise ValueError("refinements must contain deployed union candidate zero once")
    return int(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument(
        "--candidate_report",
        required=True,
        nargs="+",
        help="One or more proposal experts; the first is the frozen baseline.",
    )
    parser.add_argument(
        "--baseline_selection_report",
        default="",
        help=(
            "Optional frozen query-local likelihood report. Its deployed selected_index "
            "is resolved through the baseline report's original-index permutation, so "
            "passthrough and expert comparison start from the actual deployed baseline."
        ),
    )
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument(
        "--refine_topk",
        type=int,
        default=1,
        help="Refine this many Stage-B modes, then select by the common final surface score.",
    )
    parser.add_argument(
        "--refinement_candidate_policy",
        choices=("score_topk", "anchor_basin_cover"),
        default="score_topk",
    )
    parser.add_argument(
        "--refinement_score_margin",
        type=float,
        default=-1.0,
        help=(
            "With a non-negative value, refine score-competitive modes within "
            "this margin; anchor_basin_cover still reserves one basin per anchor."
        ),
    )
    parser.add_argument("--minimum_refinement_candidates", type=int, default=1)
    parser.add_argument("--refinement_basin_translation_m", type=float, default=0.5)
    parser.add_argument("--refinement_basin_rotation_deg", type=float, default=5.0)
    parser.add_argument("--translation_steps_m", default="0.60,0.40,0.25,0.12")
    parser.add_argument("--rotation_steps_deg", default="5,3,2,1")
    parser.add_argument("--iterations_per_scale", type=int, default=2)
    parser.add_argument("--minimum_score_improvement", type=float, default=1.0e-6)
    parser.add_argument(
        "--passthrough_without_additional_expert",
        action="store_true",
        help="Keep the frozen baseline unchanged when the deployment gate did not invoke another expert.",
    )
    parser.add_argument(
        "--baseline_mode_count",
        type=int,
        default=0,
        help="Retain this many frozen-baseline modes in a multi-expert union; zero keeps all.",
    )
    parser.add_argument(
        "--additional_expert_mode_count",
        type=int,
        default=0,
        help="Retain this many modes from each additional expert; zero keeps all.",
    )
    parser.add_argument("--maximum_splat_radius_tokens", type=int, default=0)
    parser.add_argument(
        "--validation_splat_radius_tokens",
        type=int,
        default=-1,
        help="Second renderer discretization used for robust state selection; negative disables it.",
    )
    parser.add_argument(
        "--baseline_null_calibration_report",
        default="",
        help="Held-out calibration refinements used to derive a lower control limit.",
    )
    parser.add_argument("--baseline_null_sigma", type=float, default=3.0)
    parser.add_argument(
        "--require_cross_splat_winner_consistency",
        action="store_true",
        help="Allow expert override only when both renderer discretizations select one state.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite primitive VFM refinement")
    translation_steps = _parse_positive_floats(args.translation_steps_m)
    rotation_steps = _parse_positive_floats(args.rotation_steps_deg)
    if len(translation_steps) != len(rotation_steps):
        raise ValueError("translation and rotation trust-region schedules differ")
    if int(args.iterations_per_scale) <= 0:
        raise ValueError("iterations per trust-region scale must be positive")
    if float(args.minimum_score_improvement) < 0.0:
        raise ValueError("minimum score improvement must be non-negative")
    repository_root = Path(__file__).resolve().parents[3]
    repository_state_at_start = capture_repository_state(repository_root)

    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    report_paths = [Path(value) for value in args.candidate_report]
    reports = [json.loads(path.read_text()) for path in report_paths]
    for report in reports:
        if report.get("physical_map_sha256") != physical.content_sha256:
            raise ValueError("candidate report and physical map differ")
        if report.get("canonical_field_sha256") != field.content_sha256:
            raise ValueError("candidate report and canonical field differ")
    baseline_selected_source_index: dict[str, int] = {}
    baseline_selection_path = None
    baseline_selection_sha256 = None
    if args.baseline_selection_report:
        baseline_selection_path = Path(args.baseline_selection_report)
        baseline_selection = json.loads(baseline_selection_path.read_text())
        baseline_report_sha256 = file_sha256(report_paths[0])
        baseline_selected_source_index = _load_deployed_baseline_selection(
            baseline_selection, baseline_report_sha256,
        )
        baseline_selection_sha256 = file_sha256(baseline_selection_path)
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    baseline_null_threshold = None
    baseline_null_calibration = None
    if args.baseline_null_calibration_report:
        calibration_path = Path(args.baseline_null_calibration_report)
        calibration = json.loads(calibration_path.read_text())
        calibration_scores = np.asarray([
            float(row["final_score"]) for row in calibration.get("rows", [])
        ], dtype=np.float64)
        if calibration_scores.size < 3 or np.any(~np.isfinite(calibration_scores)):
            raise ValueError("baseline-null calibration requires at least three finite scores")
        score_mean = float(np.mean(calibration_scores))
        score_std = float(np.std(calibration_scores, ddof=1))
        baseline_null_threshold = float(
            score_mean - float(args.baseline_null_sigma) * score_std
        )
        baseline_null_calibration = {
            "artifact": str(calibration_path),
            "sha256": file_sha256(calibration_path),
            "count": int(calibration_scores.size),
            "score_mean": score_mean,
            "score_std": score_std,
            "sigma": float(args.baseline_null_sigma),
            "lower_control_limit": baseline_null_threshold,
        }
    contributor_by_image: dict[str, Path] = {}
    for path in Path(args.contributors).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        contributor_by_image[str(metadata["image_id"])] = path

    sources_by_image: dict[str, list[tuple[int, dict[str, object]]]] = {}
    image_order: list[str] = []
    for report_index, report in enumerate(reports):
        for source in report.get("rows", []):
            image_id = str(source["image_id"])
            if image_id not in sources_by_image:
                sources_by_image[image_id] = []
                image_order.append(image_id)
            sources_by_image[image_id].append((int(report_index), source))

    rows: list[dict[str, object]] = []
    contributor_hashes: dict[str, str] = {}
    token_hashes: dict[str, str] = {}
    for image_id in image_order:
        source_reports = sources_by_image[image_id]
        source_evidence = candidate_source_postselection_evidence(
            source_reports[0][1], mode_name=str(args.mode_name)
        )
        contributor = contributor_by_image.get(image_id)
        if contributor is None:
            raise ValueError(f"missing contributor for candidate query: {image_id}")
        labels = ContributorLabels.load_npz(contributor)
        camera = _camera(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        token_path = Path(str(metadata["token_path"]))
        contributor_hashes[image_id] = file_sha256(contributor)
        token_hashes[image_id] = file_sha256(token_path)
        with np.load(token_path, allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        query = mapper.project(raw).measurement_context
        candidate_details: list[tuple[int, int, dict[str, object]]] = []
        for report_index, source in source_reports:
            details = source.get("mode_details", {}).get(str(args.mode_name), [])
            if int(report_index) == 0 and baseline_selected_source_index:
                if image_id not in baseline_selected_source_index:
                    raise ValueError(f"missing frozen baseline selection: {image_id}")
                details = _deployed_baseline_first(
                    source, str(args.mode_name),
                    baseline_selected_source_index[image_id],
                )
            maximum = (
                int(args.baseline_mode_count)
                if int(report_index) == 0 else int(args.additional_expert_mode_count)
            )
            if maximum > 0:
                details = details[:maximum]
            candidate_details.extend(
                (int(report_index), int(mode_index + 1), detail)
                for mode_index, detail in enumerate(details)
            )
        if not candidate_details:
            continue
        baseline_details = source_reports[0][1].get("mode_details", {}).get(
            str(args.mode_name), []
        )
        if baseline_selected_source_index:
            source = source_reports[0][1]
            baseline_details = _deployed_baseline_first(
                source, str(args.mode_name), baseline_selected_source_index[image_id],
            )
        if not baseline_details:
            raise ValueError(f"first candidate report lacks baseline mode: {image_id}")
        source_top1_pose = np.asarray(baseline_details[0]["pose_w2c"], dtype=np.float64)
        if bool(args.passthrough_without_additional_expert) and len(source_reports) == 1:
            source_top1_error = pnp_pose_error(source_top1_pose, labels.pose_w2c)
            baseline_score = float(baseline_details[0].get("score", 0.0))
            row = {
                "image_id": image_id,
                "initial_translation_m": float(source_top1_error.translation_m),
                "initial_rotation_deg": float(source_top1_error.rotation_deg),
                "final_translation_m": float(source_top1_error.translation_m),
                "final_rotation_deg": float(source_top1_error.rotation_deg),
                "initial_score": baseline_score,
                "final_score": baseline_score,
                "accepted_steps": 0,
                "selected_source_mode_rank": 1,
                "selected_source_report_index": 0,
                "refined_mode_count": 0,
                "union_candidate_count": int(len(baseline_details)),
                "seconds": 0.0,
                "pose_w2c": source_top1_pose.tolist(),
                "history": [],
                "candidate_refinements": [],
                "gate_passthrough": True,
                "postselection_source_evidence": source_evidence,
            }
            row["typed_null_diagnostics"] = typed_null_diagnostics(row)
            rows.append(row)
            print(json.dumps({key: row[key] for key in row if key not in {
                "pose_w2c", "history", "candidate_refinements",
            }}))
            continue
        token_height, token_width = int(query.shape[1]), int(query.shape[2])
        query_flat = query.transpose(1, 2, 0).reshape(-1, query.shape[0])

        def evidence_at_radius(poses: np.ndarray, radius: int):
            return score_pose_conditioned_sparse_primitives(
                poses,
                query_flat,
                field.primitive_rows,
                field.codes,
                field.confidence,
                physical,
                camera,
                token_height=token_height,
                token_width=token_width,
                primitives_per_child=0,
                batch_size=16,
                maximum_splat_radius_tokens=int(radius),
                score_semantics="fixed_grid",
                device=str(args.device),
            )

        def score_at_radius(poses: np.ndarray, radius: int) -> np.ndarray:
            return evidence_at_radius(poses, radius).scores

        def score(poses: np.ndarray) -> np.ndarray:
            return score_at_radius(poses, int(args.maximum_splat_radius_tokens))

        started = time.perf_counter()
        candidate_poses = np.asarray(
            [value[2]["pose_w2c"] for value in candidate_details], dtype=np.float64,
        )
        common_initial_scores = score(candidate_poses)
        common_order = np.argsort(-common_initial_scores, kind="stable")
        refinements = []
        selected_candidate_indices, selection_diagnostic = (
            _select_refinement_candidate_indices(
                common_order=common_order,
                common_initial_scores=common_initial_scores,
                candidate_poses=candidate_poses,
                candidate_details=candidate_details,
                maximum_count=max(int(args.refine_topk), 1),
                policy=str(args.refinement_candidate_policy),
                score_margin=float(args.refinement_score_margin),
                minimum_count=int(args.minimum_refinement_candidates),
                translation_radius_m=float(args.refinement_basin_translation_m),
                rotation_radius_deg=float(args.refinement_basin_rotation_deg),
            )
        )
        # An optional expert proposes alternatives to the deployed state; it
        # must never silently delete that state before typed-null gating.
        baseline_guard_added = 0 not in selected_candidate_indices
        if baseline_guard_added:
            selected_candidate_indices.append(0)
        rank_by_candidate = {
            int(candidate_index): int(rank + 1)
            for rank, candidate_index in enumerate(common_order.tolist())
        }
        selection_diagnostic["baseline_guard_added"] = baseline_guard_added
        selection_diagnostic["selected_count_after_baseline_guard"] = len(
            selected_candidate_indices
        )
        selection_diagnostic["selected_common_initial_ranks"] = [
            rank_by_candidate[int(value)] for value in selected_candidate_indices
        ]
        for candidate_index in selected_candidate_indices:
            report_index, source_mode_rank, detail = candidate_details[int(candidate_index)]
            initial_pose = np.asarray(detail["pose_w2c"], dtype=np.float64)
            refined = refine_pose_with_primitive_vfm_score(
                initial_pose, score,
                translation_steps_m=translation_steps,
                rotation_steps_deg=rotation_steps,
                iterations_per_scale=int(args.iterations_per_scale),
                minimum_score_improvement=float(args.minimum_score_improvement),
            )
            initial_error = pnp_pose_error(initial_pose, labels.pose_w2c)
            final_error = pnp_pose_error(refined.pose_w2c, labels.pose_w2c)
            refinements.append({
                "union_candidate_index": int(candidate_index),
                "common_initial_rank": rank_by_candidate[int(candidate_index)],
                "source_report_index": int(report_index),
                "source_mode_rank": int(source_mode_rank),
                "mapping_view_anchor_label": int(
                    detail.get("mapping_view_anchor_label", -1)
                ),
                "initial_pose_w2c": initial_pose.tolist(),
                "initial_translation_m": float(initial_error.translation_m),
                "initial_rotation_deg": float(initial_error.rotation_deg),
                "final_translation_m": float(final_error.translation_m),
                "final_rotation_deg": float(final_error.rotation_deg),
                "initial_score": float(refined.initial_score),
                "final_score": float(refined.final_score),
                "accepted_steps": int(refined.accepted_steps),
                "pose_w2c": refined.pose_w2c.tolist(),
                "history": list(refined.history),
            })
        refined_evidence = evidence_at_radius(
            np.asarray([value["pose_w2c"] for value in refinements], dtype=np.float64),
            int(args.maximum_splat_radius_tokens),
        )
        for index, value in enumerate(refinements):
            value["rendered_coverage"] = float(
                refined_evidence.rendered_coverage[index]
            )
            value["feature_coverage"] = float(
                refined_evidence.feature_coverage[index]
            )
        # Every mode is evaluated by the same fixed-grid primitive likelihood;
        # selection therefore happens only after continuous optimization and
        # does not compare proposal-expert scores from different classes.
        selected = max(
            refinements,
            key=lambda value: (
                float(value["final_score"]), -int(value["common_initial_rank"]),
            ),
        )
        selection_decision = "maximum_primary_discretization_score"
        baseline_index = _baseline_refinement_index(refinements)
        validation_radius = int(args.validation_splat_radius_tokens)
        if validation_radius >= 0:
            validation_scores = score_at_radius(
                np.asarray([value["pose_w2c"] for value in refinements], dtype=np.float64),
                validation_radius,
            )
            for index, value in enumerate(refinements):
                value["validation_score"] = float(validation_scores[index])
            primary_winner = int(np.argmax([
                float(value["final_score"]) for value in refinements
            ]))
            validation_winner = int(np.argmax(validation_scores))
            baseline_is_null = (
                baseline_null_threshold is not None
                and float(refinements[baseline_index]["final_score"])
                < float(baseline_null_threshold)
            )
            if baseline_null_threshold is not None and not baseline_is_null:
                selected = refinements[baseline_index]
                selection_decision = "baseline_explained_query_control_limit"
            elif (
                bool(args.require_cross_splat_winner_consistency)
                and primary_winner != validation_winner
            ):
                selected = refinements[baseline_index]
                selection_decision = "cross_discretization_disagreement_baseline_fallback"
            else:
                selected = refinements[primary_winner]
                selection_decision = (
                    "cross_discretization_consistent_expert_override"
                    if primary_winner == validation_winner
                    and primary_winner != baseline_index
                    else "cross_discretization_consistent_baseline"
                )
        source_top1_error = pnp_pose_error(source_top1_pose, labels.pose_w2c)
        source_evidence = dict(source_evidence)
        source_evidence.update({
            "selected_exact_rendered_coverage": float(
                selected["rendered_coverage"]
            ),
            "selected_exact_feature_coverage": float(
                selected["feature_coverage"]
            ),
            "coverage_semantics": "final_selected_pose_under_primary_exact_renderer",
        })
        row = {
            "image_id": image_id,
            "initial_translation_m": float(source_top1_error.translation_m),
            "initial_rotation_deg": float(source_top1_error.rotation_deg),
            "final_translation_m": float(selected["final_translation_m"]),
            "final_rotation_deg": float(selected["final_rotation_deg"]),
            "initial_score": float(refinements[baseline_index]["initial_score"]),
            "final_score": float(selected["final_score"]),
            "accepted_steps": int(selected["accepted_steps"]),
            "selected_source_mode_rank": int(selected["source_mode_rank"]),
            "selected_source_report_index": int(selected["source_report_index"]),
            "selected_union_candidate_index": int(
                selected["union_candidate_index"]
            ),
            "refined_mode_count": int(len(refinements)),
            "union_candidate_count": int(len(candidate_details)),
            "selection_decision": selection_decision,
            "baseline_null_threshold": baseline_null_threshold,
            "seconds": float(time.perf_counter() - started),
            "pose_w2c": selected["pose_w2c"],
            "history": selected["history"],
            "candidate_refinements": refinements,
            "refinement_candidate_selection": selection_diagnostic,
            "postselection_source_evidence": source_evidence,
        }
        row["typed_null_diagnostics"] = typed_null_diagnostics(row)
        rows.append(row)
        print(json.dumps({key: row[key] for key in row if key not in {"pose_w2c", "history"}}))
    payload = {
        "stage": "goal_maplet_sparse_primitive_vfm_pose_refinement",
        "candidate_reports": [str(path) for path in report_paths],
        "candidate_report_sha256": [file_sha256(path) for path in report_paths],
        "baseline_selection_report": (
            str(baseline_selection_path) if baseline_selection_path is not None else None
        ),
        "baseline_selection_report_sha256": baseline_selection_sha256,
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "map_feature_contract": (
            "single_canonical_primitive_vfm_field_no_rgb_no_hard_correspondence"
        ),
        "postselection_evidence_contract": (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ),
        "typed_null_contract": (
            "diagnostic_intervention_vector_not_independent_likelihoods_v1"
        ),
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "alignment_semantics": (
            "hard_correspondence_free_pnp_free_direct_fixed_grid_surface_"
            "score_coordinate_search_with_monotonic_acceptance"
        ),
        "exact_score_acceptance_is_monotonic_by_construction": True,
        "selection_uses_query_ground_truth": False,
        "ground_truth_pose_used_for_offline_error_only": True,
        "refine_topk": int(max(int(args.refine_topk), 1)),
        "mode_name": str(args.mode_name),
        "refinement_candidate_policy": str(args.refinement_candidate_policy),
        "refinement_score_margin": float(args.refinement_score_margin),
        "minimum_refinement_candidates": int(args.minimum_refinement_candidates),
        "refinement_basin_translation_m": float(
            args.refinement_basin_translation_m
        ),
        "refinement_basin_rotation_deg": float(args.refinement_basin_rotation_deg),
        "refinement_optimizer": {
            "translation_steps_m": list(translation_steps),
            "rotation_steps_deg": list(rotation_steps),
            "iterations_per_scale": int(args.iterations_per_scale),
            "minimum_score_improvement": float(args.minimum_score_improvement),
        },
        "selection_semantics": (
            "calibrated_baseline_typed_null_then_cross_discretization_consistent_"
            "common_surface_likelihood"
            if baseline_null_threshold is not None
            else "maximum_common_fixed_grid_primitive_vfm_after_continuous_refinement"
        ),
        "passthrough_without_additional_expert": bool(
            args.passthrough_without_additional_expert
        ),
        "baseline_mode_count": int(args.baseline_mode_count),
        "additional_expert_mode_count": int(args.additional_expert_mode_count),
        "maximum_splat_radius_tokens": int(args.maximum_splat_radius_tokens),
        "validation_splat_radius_tokens": int(args.validation_splat_radius_tokens),
        "require_cross_splat_winner_consistency": bool(
            args.require_cross_splat_winner_consistency
        ),
        "baseline_null_calibration": baseline_null_calibration,
        "initial_summary": _summary(rows, "initial"),
        "final_summary": _summary(rows, "final"),
        "rows": rows,
    }
    manifest_configuration = {
        key: value for key, value in vars(args).items()
        if key not in {"output_json", "force"}
    }
    payload["run_manifest"] = build_run_manifest(
        repository_root=repository_root,
        argv=[sys.executable, *sys.argv],
        configuration=manifest_configuration,
        input_artifacts={
            "physical_map_content_sha256": physical.content_sha256,
            "canonical_field_content_sha256": field.content_sha256,
            "surface_mapper_file_sha256": file_sha256(Path(args.surface_mapper)),
            "candidate_reports_sha256": canonical_json_sha256(
                [file_sha256(path) for path in report_paths]
            ),
            "query_contributors_sha256": canonical_json_sha256(contributor_hashes),
            "query_radio_tokens_sha256": canonical_json_sha256(token_hashes),
        },
        query_ids=image_order,
        device=str(args.device),
        numeric_contract={
            "pose_dtype": "float64",
            "feature_and_render_dtype": "float32",
            "surface_visibility": "all_clean_primitives_depth_prepass",
            "primitive_tie_break": PRIMITIVE_Z_TIE_BREAK,
            "refinement_update": "left_multiplicative_camera_frame_se3",
            "acceptance": "strict_monotonic_common_exact_surface_score",
        },
        candidate_counts={
            str(row["image_id"]): {
                "union_candidate_count": int(row.get("union_candidate_count", 0)),
                "refined_mode_count": int(row.get("refined_mode_count", 0)),
            }
            for row in rows
        },
        repository_state_at_start=repository_state_at_start,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
