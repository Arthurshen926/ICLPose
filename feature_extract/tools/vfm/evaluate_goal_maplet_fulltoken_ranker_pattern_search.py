"""Refine learned full-token basins by bounded derivative-free SE(3) search.

The scorer renders the frozen 3DGS/canonical RADIO field at each proposal and
applies the trained full-layout ranker.  It uses no keypoints,
correspondences, PnP, or absolute pose regression.  Target poses are consumed
only after each query search has completed to compute diagnostic errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_feature_artifact,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    compact_fulltoken_pose_ranking_features,
    compact_typed_fulltoken_pose_ranking_features,
    greedy_distinct_pose_basin_order,
    load_factorized_fulltoken_candidate_pose_ranker,
    load_fulltoken_candidate_pose_ranker,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_initialization import (
    factorized_hierarchical_pose_initialization,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_surface_pose_energy import (
    conditional_fulltoken_phase_pose_energy_control,
    conservative_fulltoken_phase_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.local_pose_supervision import (
    LOCAL_POSE_SUPERVISION_SEMANTICS,
    deterministic_global_joint_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.multibasin_pattern_search import (
    batched_multibasin_beam_pattern_search,
    batched_multibasin_pattern_search,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.query_typed_geometry import (
    conjunct_phase_and_typed_geometry_score,
    fixed_denominator_query_typed_geometry_score,
    load_query_typed_geometry_predictor,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    left_retract_pose_w2c,
)
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)
from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    ViewConditionedPrimitiveField,
)
from feature_extract.vfm.localization_goal_maplet.relative_pose_correction import (
    apply_left_pose_correction_coordinate,
    load_fulltoken_relative_pose_correction,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp


SCHEMA = "goal_maplet_fulltoken_ranker_bounded_multibasin_pattern_search_v1"


def _factorized_trace_diagnostics(
    trace: list[tuple[str, np.ndarray, np.ndarray]],
    target_pose_w2c: np.ndarray,
) -> list[dict[str, object]]:
    """Explain geometric coverage and score pruning at every hierarchy call.

    The target is supplied only after the pose-free search has completed.  The
    resulting rows are therefore diagnostics, never inputs to proposal or
    scoring.  Stable score ranks make it possible to distinguish a missing
    probe from an evaluated near pose that the live objective ranked poorly.
    """

    target = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    rows: list[dict[str, object]] = []
    for stage_index, (head, raw_pose, raw_score) in enumerate(trace):
        poses = np.asarray(raw_pose, dtype=np.float64)
        scores = np.asarray(raw_score, dtype=np.float64).reshape(-1)
        if (
            poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or scores.shape != (poses.shape[0],)
            or poses.shape[0] == 0
            or np.any(~np.isfinite(poses)) or np.any(~np.isfinite(scores))
        ):
            raise ValueError("factorized trace pose/score rows differ")
        translation, rotation = _pose_errors(poses, target)
        joint = np.maximum(translation / 1.0, rotation / 10.0)
        oracle = int(np.argmin(joint))
        score_order = np.lexsort((np.arange(scores.size), -scores))
        oracle_rank = int(np.flatnonzero(score_order == oracle)[0]) + 1
        top = int(score_order[0])
        loose = (translation <= 1.0 + 1.0e-6) & (rotation <= 10.0 + 1.0e-5)
        strict = (translation <= 0.5 + 1.0e-6) & (rotation <= 5.0 + 1.0e-5)
        rows.append({
            "stage_index": int(stage_index),
            "head": str(head),
            "pose_count": int(poses.shape[0]),
            "oracle_translation_m": float(translation[oracle]),
            "oracle_rotation_deg": float(rotation[oracle]),
            "oracle_score": float(scores[oracle]),
            "oracle_score_rank": oracle_rank,
            "topscore_translation_m": float(translation[top]),
            "topscore_rotation_deg": float(rotation[top]),
            "topscore": float(scores[top]),
            "loose_pose_count": int(np.sum(loose)),
            "strict_pose_count": int(np.sum(strict)),
            "best_loose_score": None if not np.any(loose) else float(np.max(scores[loose])),
            "best_strict_score": None if not np.any(strict) else float(np.max(scores[strict])),
        })
    return rows


def _evaluated_pose_trace_diagnostics(
    poses_w2c: np.ndarray,
    scores: np.ndarray,
    target_pose_w2c: np.ndarray,
) -> dict[str, object]:
    """Summarize all actually rendered poses, consuming target only afterward."""

    poses = np.asarray(poses_w2c, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if (
        poses.ndim != 3 or poses.shape[1:] != (4, 4)
        or values.shape != (poses.shape[0],) or poses.shape[0] == 0
        or np.any(~np.isfinite(poses)) or np.any(~np.isfinite(values))
    ):
        raise ValueError("evaluated pose trace arrays differ")
    unique_rows: list[int] = []
    unique_scores_list: list[float] = []
    seen: dict[bytes, int] = {}
    duplicate_score_max_absolute_drift = 0.0
    duplicate_score_drift_count = 0
    for row in range(poses.shape[0]):
        key = np.ascontiguousarray(poses[row]).tobytes()
        previous = seen.get(key)
        if previous is None:
            seen[key] = len(unique_rows)
            unique_rows.append(row)
            unique_scores_list.append(float(values[row]))
        else:
            drift = abs(float(values[row]) - unique_scores_list[previous])
            duplicate_score_max_absolute_drift = max(
                duplicate_score_max_absolute_drift, drift,
            )
            if drift > 2.0e-6 + 2.0e-6 * abs(unique_scores_list[previous]):
                duplicate_score_drift_count += 1
            # The trace records what the running search actually observed.  A
            # duplicate pose therefore retains its maximum observed score for
            # the conservative post-search rank diagnostic while separately
            # exposing any replay drift.
            unique_scores_list[previous] = max(
                unique_scores_list[previous], float(values[row]),
            )
    keep = np.asarray(unique_rows, dtype=np.int64)
    unique_poses = poses[keep]
    unique_scores = np.asarray(unique_scores_list, dtype=np.float64)
    translation, rotation = _pose_errors(unique_poses, target_pose_w2c)
    joint = np.maximum(translation / 1.0, rotation / 10.0)
    oracle = int(np.argmin(joint))
    return {
        "unique_rendered_pose_count": int(keep.size),
        "duplicate_rendered_pose_count": int(poses.shape[0] - keep.size),
        "duplicate_score_drift_count": int(duplicate_score_drift_count),
        "duplicate_score_max_absolute_drift": float(
            duplicate_score_max_absolute_drift
        ),
        "best_translation_m": float(translation[oracle]),
        "best_rotation_deg": float(rotation[oracle]),
        "any_strict_0_5m_5deg": bool(np.any(
            (translation <= 0.5 + 1.0e-6) & (rotation <= 5.0 + 1.0e-5)
        )),
        "any_loose_1m_10deg": bool(np.any(
            (translation <= 1.0 + 1.0e-6) & (rotation <= 10.0 + 1.0e-5)
        )),
        "oracle_pose_score_rank": int(
            1 + np.sum(unique_scores > unique_scores[oracle] + 1.0e-8)
        ),
    }


def _global_joint_seed_poses(
    domain_seed_poses: np.ndarray,
    *,
    translation_step_m: float = 8.0,
    rotation_step_degrees: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the frozen joint design at an explicit search-domain scale.

    The default is the exact 8 m/45 degree supervision distribution. Medium
    refinement may explicitly request 2 m/20 degrees; normalized coordinates
    and deterministic source ordering remain unchanged. No hidden
    ``1/sqrt(3)`` shrink is applied to either domain.
    """

    seeds = np.asarray(domain_seed_poses, dtype=np.float64)
    if seeds.ndim != 3 or seeds.shape[1:] != (4, 4) or np.any(~np.isfinite(seeds)):
        raise ValueError("global-joint seed poses must have shape [B,4,4]")
    if (
        not 0.0 < float(translation_step_m) < float("inf")
        or not 0.0 < float(rotation_step_degrees) < float("inf")
    ):
        raise ValueError("global-joint search-domain scales must be positive")
    coordinates = deterministic_global_joint_coordinates()
    rows: list[np.ndarray] = []
    sources: list[int] = []
    for source, seed_pose in enumerate(seeds):
        rows.append(seed_pose.copy())
        sources.append(source)
        for coordinate in coordinates:
            rows.append(left_retract_pose_w2c(
                seed_pose, coordinate,
                translation_step_m=float(translation_step_m),
                rotation_step_degrees=float(rotation_step_degrees),
            ))
            sources.append(source)
    return np.asarray(rows, dtype=np.float64), np.asarray(sources, dtype=np.int64)


def _protected_hypothesis_union(
    domain_seed_poses: np.ndarray,
    domain_seed_scores: np.ndarray,
    search_initial_poses: np.ndarray,
    search_initial_scores: np.ndarray,
    refined_states,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep original seeds and initialized hypotheses alongside refinements.

    Refinement is allowed to add a hypothesis but never to erase the physical
    basin that justified evaluating it.  Duplicate removal and ranking happen
    only after this protected union has been formed.
    """

    seed_pose = np.asarray(domain_seed_poses, dtype=np.float64)
    seed_score = np.asarray(domain_seed_scores, dtype=np.float64).reshape(-1)
    initial_pose = np.asarray(search_initial_poses, dtype=np.float64)
    initial_score = np.asarray(search_initial_scores, dtype=np.float64).reshape(-1)
    refined_pose = np.asarray([state.pose_w2c for state in refined_states], dtype=np.float64)
    refined_score = np.asarray([state.score for state in refined_states], dtype=np.float64)
    if (
        seed_pose.shape != (seed_score.size, 4, 4)
        or initial_pose.shape != (initial_score.size, 4, 4)
        or refined_pose.shape != (refined_score.size, 4, 4)
        or seed_score.size == 0 or initial_score.size == 0 or refined_score.size == 0
        or any(np.any(~np.isfinite(value)) for value in (
            seed_pose, seed_score, initial_pose, initial_score,
            refined_pose, refined_score,
        ))
    ):
        raise ValueError("protected search hypothesis arrays differ")
    pose = np.concatenate((seed_pose, initial_pose, refined_pose), axis=0)
    score = np.concatenate((seed_score, initial_score, refined_score), axis=0)
    accepted = np.concatenate((
        np.zeros(seed_score.size + initial_score.size, dtype=np.int64),
        np.asarray([state.accepted_updates for state in refined_states], dtype=np.int64),
    ))
    kind = np.asarray(
        ["retrieval_seed"] * seed_score.size
        + ["search_initializer"] * initial_score.size
        + ["refined_hypothesis"] * refined_score.size,
    )
    return pose, score, accepted, kind


def _inside_seed_domain_union(
    poses_w2c: np.ndarray,
    seed_poses_w2c: np.ndarray,
    *,
    translation_half_extent_m: float,
    rotation_radius_deg: float,
) -> np.ndarray:
    poses = np.asarray(poses_w2c, dtype=np.float64)
    seeds = np.asarray(seed_poses_w2c, dtype=np.float64)
    rotation = poses[:, :3, :3]
    center = (-np.swapaxes(rotation, 1, 2) @ poses[:, :3, 3, None])[:, :, 0]
    seed_rotation = seeds[:, :3, :3]
    seed_center = (-np.swapaxes(seed_rotation, 1, 2) @ seeds[:, :3, 3, None])[:, :, 0]
    delta = center[:, None, :] - seed_center[None, :, :]
    translation_inside = np.all(
        np.abs(delta) <= float(translation_half_extent_m) + 1.0e-9, axis=2,
    )
    relative = rotation[:, None] @ np.swapaxes(seed_rotation[None], 2, 3)
    cosine = np.clip(
        (np.trace(relative, axis1=2, axis2=3) - 1.0) / 2.0, -1.0, 1.0,
    )
    rotation_inside = np.degrees(np.arccos(cosine)) <= float(rotation_radius_deg) + 1.0e-8
    return np.any(translation_inside & rotation_inside, axis=1)


def _validate_natural_transfer_model_pair(
    coarse_metadata: dict[str, object],
    local_metadata: dict[str, object] | None,
    *,
    evaluation_dataset_content_sha256: str,
) -> None:
    """Validate the diagnostic coarse-basin/local-energy role separation.

    A locally supervised model may be applied to a distinct, pose-free natural
    candidate pool only as a diagnostic transfer control.  When a second model
    is supplied, it must be a direct refinement of the frozen coarse model and
    its supervision must describe either the original local SE(3) probes or a
    frozen-natural-seed-to-GT trajectory.  This keeps retrieval ordering and
    local energy optimization separate without allowing an unrelated scorer to
    be substituted after looking at the evaluation candidates.
    """

    if (
        not bool(coarse_metadata.get("local_supervision", False))
        or not bool(coarse_metadata.get("multiscale_supervision", False))
        or coarse_metadata.get("dataset_content_sha256")
        == str(evaluation_dataset_content_sha256)
    ):
        raise ValueError(
            "natural-transfer control requires a distinct multiscale local model"
        )
    if local_metadata is None:
        return
    semantics = str(local_metadata.get("local_pose_supervision_semantics", ""))
    supported_semantics = (
        semantics == LOCAL_POSE_SUPERVISION_SEMANTICS
        or semantics.startswith("frozen_natural_seed_to_gt_")
    )
    if (
        not bool(local_metadata.get("local_supervision", False))
        or not bool(local_metadata.get("multiscale_supervision", False))
        or local_metadata.get("initial_model_content_sha256")
        != coarse_metadata.get("model_content_sha256")
        or not supported_semantics
    ):
        raise ValueError(
            "local scorer is not a direct trajectory/local refinement of the "
            "selected coarse basin model"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--local_model", default="",
        help=(
            "Optional locally supervised scorer.  --model remains the coarse "
            "basin selector; this model is used only inside selected domains."
        ),
    )
    parser.add_argument(
        "--allow_local_model_as_natural_transfer_control", action="store_true",
        help=(
            "diagnostic-only: apply a multiscale locally supervised ranker "
            "to a distinct pose-free natural pool; an optional --local_model "
            "may refine energy inside basins while --model keeps their ordering"
        ),
    )
    parser.add_argument("--relative_correction_model", default="")
    parser.add_argument("--relative_correction_modes_per_basin", type=int, default=4)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--view_conditioned_field", default="")
    parser.add_argument("--query_typed_geometry_predictor", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include_route", required=True)
    parser.add_argument("--maximum_queries", type=int, default=20)
    parser.add_argument("--initial_basins", type=int, default=4)
    parser.add_argument(
        "--initialization_semantics",
        choices=("learned_topk", "gt_offset_1m10_diagnostic"),
        default="learned_topk",
    )
    parser.add_argument("--maximum_sweeps", type=int, default=6)
    parser.add_argument(
        "--search_semantics", choices=("greedy_pattern_v1", "beam_pattern_v2"),
        default="greedy_pattern_v1",
    )
    parser.add_argument("--beam_width_per_initial_basin", type=int, default=4)
    parser.add_argument(
        "--global_joint_initialization", action="store_true",
        help="Poll the frozen symmetric full-domain design before local search.",
    )
    parser.add_argument("--global_joint_survivors_per_basin", type=int, default=2)
    parser.add_argument(
        "--factorized_model", default="",
        help=(
            "Optional three-head full-token ranker for position-beam then "
            "orientation-codebook initialization."
        ),
    )
    parser.add_argument("--factorized_position_beam_per_basin", type=int, default=2)
    parser.add_argument("--factorized_survivors_per_basin", type=int, default=2)
    parser.add_argument(
        "--phase_factorized_initialization", action="store_true",
        help="Use the live full-token phase objective for both hierarchy stages.",
    )
    parser.add_argument(
        "--live_score_semantics",
        choices=(
            "learned_ranker", "conditional_phase_shift_control",
            "conditional_phase_query_typed_conjunction",
            "conservative_phase_shift",
        ),
        default="learned_ranker",
    )
    parser.add_argument("--translation_radius_m", type=float, default=4.0)
    parser.add_argument("--rotation_radius_deg", type=float, default=22.5)
    parser.add_argument("--minimum_translation_radius_m", type=float, default=0.25)
    parser.add_argument("--minimum_rotation_radius_deg", type=float, default=2.5)
    parser.add_argument(
        "--search_domain_translation_half_extent_m", type=float, default=8.0,
    )
    parser.add_argument(
        "--search_domain_rotation_radius_deg", type=float, default=45.0,
    )
    parser.add_argument("--render_batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite learned pattern-search report")
    dataset_path = Path(args.dataset)
    arrays, metadata = load_pose_candidate_dataset(
        dataset_path, require_rendered_targets=False,
    )
    features, feature_manifest = _load_feature_artifact(
        Path(args.features), Path(args.feature_manifest), dataset_path=dataset_path,
        dataset_content_sha256=str(metadata["content_sha256"]),
    )
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = json.loads(Path(args.field_feature_contract).read_text())
    mapper_sha = file_sha256(Path(args.surface_mapper))
    if (
        physical.content_sha256 != metadata.get("physical_map_sha256")
        or field.content_sha256 != metadata.get("canonical_field_sha256")
        or field.physical_map_sha256 != physical.content_sha256
        or contract.get("canonical_field_sha256") != field.content_sha256
        or contract.get("query_readout_sha256") != mapper_sha
    ):
        raise ValueError("pattern-search dataset/map/field/mapper lineage differs")
    view_conditioned_field = None
    view_conditioned_path = None
    if str(args.view_conditioned_field):
        view_conditioned_path = Path(args.view_conditioned_field)
        view_conditioned_field = ViewConditionedPrimitiveField.load_npz(
            view_conditioned_path
        )
        view_conditioned_field.validate_alignment(
            physical_map_sha256=physical.content_sha256,
            canonical_field_sha256=field.content_sha256,
            canonical_primitive_rows=field.primitive_rows,
            canonical_feature_dim=field.feature_dim,
        )
        if (
            view_conditioned_field.metadata.get("coordinate_correct") is not True
            or str(args.include_route) in set(
                view_conditioned_field.metadata.get("mapping_trajectory_ids", [])
            )
        ):
            raise ValueError("view-conditioned field is not query-route disjoint/coordinate-correct")
    device = torch.device(str(args.device))
    query_typed_model = None
    query_typed_metadata = None
    query_typed_path = None
    if str(args.query_typed_geometry_predictor):
        query_typed_path = Path(args.query_typed_geometry_predictor)
        query_typed_model, query_typed_metadata = load_query_typed_geometry_predictor(
            query_typed_path, device=device,
        )
    typed_phase = (
        str(args.live_score_semantics)
        == "conditional_phase_query_typed_conjunction"
    )
    if typed_phase != (query_typed_model is not None):
        raise ValueError("query typed predictor is required only by typed phase conjunction")
    if typed_phase and view_conditioned_field is not None:
        raise ValueError("typed phase and view-conditioned field are separate diagnostic arms")
    model, model_metadata = load_fulltoken_candidate_pose_ranker(Path(args.model), device=device)
    natural_transfer_control = bool(args.allow_local_model_as_natural_transfer_control)
    if natural_transfer_control:
        _validate_natural_transfer_model_pair(
            model_metadata, None,
            evaluation_dataset_content_sha256=str(metadata["content_sha256"]),
        )
    elif (
        model_metadata.get("feature_file_sha256") != feature_manifest["feature_file_sha256"]
        or model_metadata.get("dataset_content_sha256") != metadata["content_sha256"]
        or bool(model_metadata.get("local_supervision", False))
    ):
        raise ValueError("pattern-search model/feature lineage differs")
    score_model = model
    score_model_metadata = model_metadata
    local_model_path = None
    if str(args.local_model):
        local_model_path = Path(args.local_model)
        score_model, score_model_metadata = load_fulltoken_candidate_pose_ranker(
            local_model_path, device=device,
        )
        if natural_transfer_control:
            _validate_natural_transfer_model_pair(
                model_metadata, score_model_metadata,
                evaluation_dataset_content_sha256=str(metadata["content_sha256"]),
            )
        elif (
            not bool(score_model_metadata.get("local_supervision", False))
            or score_model_metadata.get("initial_model_content_sha256")
            != model_metadata.get("model_content_sha256")
            or score_model_metadata.get("local_pose_supervision_semantics")
            != LOCAL_POSE_SUPERVISION_SEMANTICS
        ):
            raise ValueError("local scorer is not bound to the selected coarse basin model")
    correction_model = None
    correction_model_metadata = None
    correction_model_path = None
    if str(args.relative_correction_model):
        correction_model_path = Path(args.relative_correction_model)
        correction_model, correction_model_metadata = load_fulltoken_relative_pose_correction(
            correction_model_path, device=device,
        )
        if (
            correction_model_metadata.get("local_pose_supervision_semantics")
            != LOCAL_POSE_SUPERVISION_SEMANTICS
            or correction_model_metadata.get("translation_scale_m") != 8.0
            or correction_model_metadata.get("rotation_scale_deg") != 45.0
        ):
            raise ValueError("relative correction model domain lineage differs")
    if correction_model is not None and bool(args.global_joint_initialization):
        raise ValueError("relative correction and global-joint initialization are separate arms")
    factorized_model = None
    factorized_model_metadata = None
    factorized_model_path = None
    if str(args.factorized_model):
        factorized_model_path = Path(args.factorized_model)
        factorized_model, factorized_model_metadata = (
            load_factorized_fulltoken_candidate_pose_ranker(
                factorized_model_path, device=device,
            )
        )
        local_score_ancestor = factorized_model_metadata.get(
            "local_score_model_content_sha256",
            factorized_model_metadata.get("initial_model_content_sha256"),
        )
        if (
            local_score_ancestor
            != score_model_metadata.get("model_content_sha256")
            or factorized_model_metadata.get("translation_scale_m")
            != float(args.search_domain_translation_half_extent_m)
            or factorized_model_metadata.get("rotation_scale_deg")
            != float(args.search_domain_rotation_radius_deg)
        ):
            raise ValueError("factorized initializer is not bound to the local scorer/domain")
    if sum((
        correction_model is not None,
        bool(args.global_joint_initialization),
        factorized_model is not None,
        bool(args.phase_factorized_initialization),
    )) > 1:
        raise ValueError(
            "relative, global-joint, learned-factorized, and phase-factorized "
            "initialization are separate arms"
        )
    if factorized_model is not None and factorized_model.input_channels not in (9, 18):
        raise ValueError("live query-typed factorized inference requires its query head")
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(device))
    mapper.model.to(device).eval()
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(device))
    contributors = _load_contributors(Path(args.contributors))
    query_rows = np.asarray([
        row for row, value in enumerate(arrays["image_ids"].tolist())
        if str(value).split("/", 1)[0] == str(args.include_route)
    ], dtype=np.int64)
    if int(args.maximum_queries) > 0:
        query_rows = query_rows[:int(args.maximum_queries)]
    if query_rows.size == 0 or int(args.initial_basins) <= 0:
        raise ValueError("pattern-search query/basin inventory is empty")

    result_rows = []
    with torch.no_grad():
        for query_index in query_rows.tolist():
            image_id = str(arrays["image_ids"][query_index])
            contributor_path = contributors.get(image_id)
            if contributor_path is None:
                raise ValueError(f"missing contributor camera for {image_id}")
            camera = _camera(contributor_path)
            token_path = Path(str(arrays["radio_token_paths"][query_index])).resolve()
            if file_sha256(token_path) != str(arrays["radio_file_sha256"][query_index]):
                raise ValueError("pattern-search RADIO token lineage differs")
            with np.load(token_path, allow_pickle=False) as data:
                raw_np = np.asarray(data["radio_final"], dtype=np.float32)
            query_descriptor = mapper.model(
                torch.as_tensor(raw_np, device=device)[None]
            )[0].permute(1, 2, 0).reshape(2304, 128)
            query_typed_prediction = None
            if query_typed_model is not None:
                query_typed_prediction = query_typed_model(
                    query_descriptor.reshape(36, 64, 128).permute(2, 0, 1)[None]
                )

            target_pose = None
            if str(args.initialization_semantics) == "learned_topk":
                valid_rows = np.flatnonzero(arrays["candidate_valid"][query_index])
                valid_rows = valid_rows[valid_rows != 0]
                frozen = torch.as_tensor(
                    np.asarray(features[query_index, valid_rows], dtype=np.float32),
                    device=device,
                )
                frozen_score = model(frozen).cpu().numpy()
                order = valid_rows[np.lexsort((valid_rows, -frozen_score))]
                selected = order[:int(args.initial_basins)]
                if selected.size != int(args.initial_basins):
                    raise ValueError("insufficient learned initial basins")
                initial_poses = np.asarray(
                    arrays["candidate_poses_w2c"][query_index, selected], dtype=np.float64,
                )
            else:
                if int(args.initial_basins) != 1:
                    raise ValueError("GT-offset diagnostic requires exactly one initial basin")
                target_pose = np.asarray(
                    arrays["candidate_poses_w2c"][query_index, 0], dtype=np.float64,
                )
                initial_poses = np.asarray([
                    se3_exp(np.asarray([
                        0.0, np.radians(10.0), 0.0, 1.0, 0.0, 0.0,
                    ])) @ target_pose
                ])
                selected = np.asarray([-1], dtype=np.int64)
            domain_seed_poses = initial_poses.copy()
            search_initial_poses = initial_poses
            relative_correction_proposal_count = 0
            if correction_model is not None:
                if str(args.initialization_semantics) != "learned_topk":
                    raise ValueError("relative correction requires learned basin initialization")
                mode_budget = int(args.relative_correction_modes_per_basin)
                if mode_budget <= 0 or mode_budget > int(correction_model.mode_count):
                    raise ValueError("relative correction mode budget differs from the model")
                correction_coordinate, correction_logit = correction_model(
                    torch.as_tensor(
                        np.asarray(features[query_index, selected], dtype=np.float32),
                        device=device,
                    )
                )
                correction_coordinate = correction_coordinate.cpu().numpy()
                correction_logit = correction_logit.cpu().numpy()
                corrected = []
                for basin in range(initial_poses.shape[0]):
                    mode_order = np.lexsort((
                        np.arange(correction_logit.shape[1]), -correction_logit[basin],
                    ))[:mode_budget]
                    corrected.extend(
                        apply_left_pose_correction_coordinate(
                            initial_poses[basin], correction_coordinate[basin, mode]
                        )
                        for mode in mode_order.tolist()
                    )
                search_initial_poses = np.asarray(corrected, dtype=np.float64)
                relative_correction_proposal_count = int(search_initial_poses.shape[0])
            rendered_pose_count = 0
            rejected_outside_domain = 0
            render_seconds = 0.0
            initial_live_scores = None
            evaluated_pose_trace: list[np.ndarray] = []
            evaluated_score_trace: list[np.ndarray] = []
            live_score_cache: dict[bytes, float] = {}

            def render_score_batch(
                poses: np.ndarray, scoring_function, *, typed_geometry: bool = False,
            ) -> np.ndarray:
                nonlocal rendered_pose_count, rejected_outside_domain, render_seconds
                pose = np.asarray(poses, dtype=np.float64)
                inside = _inside_seed_domain_union(
                    pose, domain_seed_poses,
                    translation_half_extent_m=float(
                        args.search_domain_translation_half_extent_m
                    ),
                    rotation_radius_deg=float(args.search_domain_rotation_radius_deg),
                )
                score = np.full((pose.shape[0],), -1.0e6, dtype=np.float64)
                rejected_outside_domain += int(np.sum(~inside))
                rows = np.flatnonzero(inside)
                for begin in range(0, int(rows.size), int(args.render_batch_size)):
                    chosen = rows[begin:begin + int(args.render_batch_size)]
                    if typed_geometry:
                        rendered = scene.render_direct_typed_canonical_grid_batch(
                            pose[chosen], camera,
                        )
                    else:
                        rendered = scene.render_direct_canonical_grid_batch(
                            pose[chosen], camera,
                        )
                    render_seconds += float(rendered.total_seconds)
                    rendered_pose_count += int(chosen.size)
                    compact = []
                    for local in range(int(chosen.size)):
                        common = (
                            query_descriptor,
                            torch.as_tensor(
                                rendered.feature[local].reshape(2304, 1, -1),
                                device=device,
                            ),
                            torch.as_tensor(
                                rendered.mass[local].reshape(2304, 1), device=device,
                            ),
                            torch.as_tensor(
                                rendered.valid[local].reshape(2304, 1), device=device,
                            ),
                        )
                        if typed_geometry:
                            compact.append(compact_typed_fulltoken_pose_ranking_features(
                                *common,
                                torch.as_tensor(
                                    rendered.normal_axis_moment[local], device=device,
                                ),
                                torch.as_tensor(
                                    rendered.relative_log_depth[local], device=device,
                                ),
                                torch.as_tensor(
                                    rendered.log_depth_std[local], device=device,
                                ),
                                torch.as_tensor(rendered.boundary[local], device=device),
                                height=36, width=64,
                            ))
                        else:
                            compact.append(compact_fulltoken_pose_ranking_features(
                                *common, height=36, width=64,
                            ))
                    score[chosen] = scoring_function(torch.stack(compact)).cpu().numpy()
                return score

            def render_phase_score_batch(poses: np.ndarray) -> np.ndarray:
                nonlocal rendered_pose_count, rejected_outside_domain, render_seconds
                pose = np.asarray(poses, dtype=np.float64)
                inside = _inside_seed_domain_union(
                    pose, domain_seed_poses,
                    translation_half_extent_m=float(
                        args.search_domain_translation_half_extent_m
                    ),
                    rotation_radius_deg=float(args.search_domain_rotation_radius_deg),
                )
                score = np.full((pose.shape[0],), -1.0e6, dtype=np.float64)
                rejected_outside_domain += int(np.sum(~inside))
                rows = np.flatnonzero(inside)
                for begin in range(0, int(rows.size), int(args.render_batch_size)):
                    chosen = rows[begin:begin + int(args.render_batch_size)]
                    if typed_phase:
                        rendered = scene.render_direct_typed_canonical_grid_batch(
                            pose[chosen], camera,
                        )
                    else:
                        rendered = scene.render_direct_canonical_grid_batch(
                            pose[chosen], camera,
                            view_conditioned_field=view_conditioned_field,
                        )
                    render_seconds += float(rendered.total_seconds)
                    rendered_rows = tuple(
                        (
                            rendered.feature[local].reshape(2304, 1, -1),
                            rendered.mass[local].reshape(2304, 1),
                            rendered.valid[local].reshape(2304, 1),
                        )
                        for local in range(int(chosen.size))
                    )
                    rendered_pose_count += int(chosen.size)
                    values = []
                    for local in range(int(chosen.size)):
                        target_feature, target_mass, target_valid = rendered_rows[local]
                        common = (
                            query_descriptor,
                            torch.as_tensor(
                                target_feature, device=device,
                            ),
                            torch.as_tensor(
                                target_mass, device=device,
                            ),
                            torch.as_tensor(
                                target_valid, device=device,
                            ),
                        )
                        if str(args.live_score_semantics) in (
                            "conditional_phase_shift_control",
                            "conditional_phase_query_typed_conjunction",
                        ):
                            value = conditional_fulltoken_phase_pose_energy_control(
                                *common, height=36, width=64, maximum_shift_tokens=1,
                            )
                        elif str(args.live_score_semantics) == "conservative_phase_shift":
                            value = conservative_fulltoken_phase_pose_energy(
                                *common, height=36, width=64, maximum_shift_tokens=1,
                            )
                        else:
                            raise AssertionError("phase scorer called for learned semantics")
                        values.append(value.score)
                    phase_values = torch.stack(values)
                    if typed_phase:
                        prediction = {
                            key: value.expand(int(chosen.size), *value.shape[1:])
                            for key, value in query_typed_prediction.items()
                        }
                        geometry_values = fixed_denominator_query_typed_geometry_score(
                            prediction,
                            torch.as_tensor(rendered.normal_axis_moment, device=device),
                            torch.as_tensor(rendered.relative_log_depth, device=device),
                            torch.as_tensor(rendered.log_depth_std, device=device),
                            torch.as_tensor(rendered.boundary, device=device),
                            torch.as_tensor(rendered.mass, device=device),
                        )
                        phase_values = conjunct_phase_and_typed_geometry_score(
                            phase_values, geometry_values,
                        )
                    score[chosen] = phase_values.cpu().numpy()
                return score

            def score_batch(poses: np.ndarray) -> np.ndarray:
                nonlocal initial_live_scores
                pose = np.asarray(poses, dtype=np.float64)
                if pose.ndim != 3 or pose.shape[1:] != (4, 4):
                    raise ValueError("live score poses must have shape [B,4,4]")
                keys = [np.ascontiguousarray(value).tobytes() for value in pose]
                missing_keys: list[bytes] = []
                missing_poses: list[np.ndarray] = []
                pending: set[bytes] = set()
                for key, value in zip(keys, pose):
                    if key not in live_score_cache and key not in pending:
                        pending.add(key)
                        missing_keys.append(key)
                        missing_poses.append(value.copy())
                if missing_poses:
                    new_pose = np.asarray(missing_poses, dtype=np.float64)
                    new_score = (
                        render_score_batch(new_pose, score_model)
                        if str(args.live_score_semantics) == "learned_ranker"
                        else render_phase_score_batch(new_pose)
                    )
                    for key, value in zip(missing_keys, new_score.tolist()):
                        live_score_cache[key] = float(value)
                    inside = _inside_seed_domain_union(
                        new_pose, domain_seed_poses,
                        translation_half_extent_m=float(
                            args.search_domain_translation_half_extent_m
                        ),
                        rotation_radius_deg=float(args.search_domain_rotation_radius_deg),
                    )
                    if np.any(inside):
                        evaluated_pose_trace.append(new_pose[inside].copy())
                        evaluated_score_trace.append(new_score[inside].copy())
                score = np.asarray([live_score_cache[key] for key in keys], dtype=np.float64)
                if (
                    initial_live_scores is None
                    and pose.shape == search_initial_poses.shape
                    and np.array_equal(pose, search_initial_poses)
                ):
                    initial_live_scores = score.copy()
                return score

            global_joint_rendered_pose_count = 0
            global_pose = None
            factorized_result = None
            factorized_evaluated_poses: list[np.ndarray] = []
            factorized_trace: list[tuple[str, np.ndarray, np.ndarray]] = []
            if bool(args.global_joint_initialization):
                survivor_count = int(args.global_joint_survivors_per_basin)
                if survivor_count <= 0:
                    raise ValueError("global joint survivor count must be positive")
                global_pose, source_rows = _global_joint_seed_poses(
                    domain_seed_poses,
                    translation_step_m=float(args.search_domain_translation_half_extent_m),
                    rotation_step_degrees=float(args.search_domain_rotation_radius_deg),
                )
                global_score = score_batch(global_pose)
                global_joint_rendered_pose_count = int(global_pose.shape[0])
                retained = []
                for source in range(domain_seed_poses.shape[0]):
                    rows = np.flatnonzero(source_rows == source)
                    order = rows[np.lexsort((rows, -global_score[rows]))]
                    retained.extend(order[:survivor_count].tolist())
                search_initial_poses = global_pose[np.asarray(retained, dtype=np.int64)]
                initial_live_scores = None
            elif factorized_model is not None or bool(args.phase_factorized_initialization):
                def factorized_score(poses: np.ndarray, head: str) -> np.ndarray:
                    pose = np.asarray(poses, dtype=np.float64)
                    factorized_evaluated_poses.append(pose.copy())
                    if bool(args.phase_factorized_initialization):
                        score = render_phase_score_batch(pose)
                    else:
                        score = render_score_batch(
                            pose, lambda feature: factorized_model(feature)[head],
                            typed_geometry=(factorized_model.input_channels == 18),
                        )
                    factorized_trace.append((head, pose.copy(), score.copy()))
                    return score

                factorized_result = factorized_hierarchical_pose_initialization(
                    domain_seed_poses,
                    lambda poses: factorized_score(poses, "location"),
                    lambda poses: factorized_score(poses, "joint"),
                    translation_half_extent_m=float(
                        args.search_domain_translation_half_extent_m
                    ),
                    coarse_translation_step_m=float(
                        args.search_domain_translation_half_extent_m
                    ) / 2.0,
                    refinement_translation_steps_m=tuple(
                        float(args.search_domain_translation_half_extent_m) / divisor
                        for divisor in (4.0, 8.0, 16.0)
                    ),
                    rotation_radius_deg=float(
                        args.search_domain_rotation_radius_deg
                    ),
                    position_beam_width_per_seed=int(
                        args.factorized_position_beam_per_basin
                    ),
                    survivors_per_seed=int(args.factorized_survivors_per_basin),
                )
                search_initial_poses = factorized_result.poses_w2c
                initial_live_scores = None

            if str(args.search_semantics) == "greedy_pattern_v1":
                search = batched_multibasin_pattern_search(
                    search_initial_poses,
                    score_batch,
                    translation_radius_m=float(args.translation_radius_m),
                    rotation_radius_deg=float(args.rotation_radius_deg),
                    shrink_factor=0.5,
                    minimum_translation_radius_m=float(args.minimum_translation_radius_m),
                    minimum_rotation_radius_deg=float(args.minimum_rotation_radius_deg),
                    maximum_sweeps=int(args.maximum_sweeps),
                )
                completed_search_levels = int(search.completed_sweeps)
            else:
                translation_schedule = []
                rotation_schedule = []
                translation_radius = float(args.translation_radius_m)
                rotation_radius = float(args.rotation_radius_deg)
                while (
                    len(translation_schedule) < int(args.maximum_sweeps)
                    and translation_radius >= float(args.minimum_translation_radius_m)
                    and rotation_radius >= float(args.minimum_rotation_radius_deg)
                ):
                    translation_schedule.append(translation_radius)
                    rotation_schedule.append(rotation_radius)
                    translation_radius *= 0.5
                    rotation_radius *= 0.5
                search = batched_multibasin_beam_pattern_search(
                    search_initial_poses, score_batch,
                    translation_radii_m=translation_schedule,
                    rotation_radii_deg=rotation_schedule,
                    beam_width_per_source=int(args.beam_width_per_initial_basin),
                )
                completed_search_levels = int(search.completed_levels)
            # The learned initialization consumes the target only now.  The
            # explicit GT-offset control consumed it solely to construct its
            # declared capture-basin initializer; the scorer never receives it.
            if target_pose is None:
                target_pose = np.asarray(
                    arrays["candidate_poses_w2c"][query_index, 0], dtype=np.float64,
                )
            evaluated_diagnostics = _evaluated_pose_trace_diagnostics(
                np.concatenate(evaluated_pose_trace, axis=0),
                np.concatenate(evaluated_score_trace, axis=0),
                target_pose,
            )
            if global_pose is not None:
                global_t, global_r = _pose_errors(global_pose, target_pose)
                global_joint = np.maximum(global_t / 1.0, global_r / 10.0)
                global_oracle = int(np.argmin(global_joint))
            else:
                global_t = global_r = None
                global_oracle = None
            if factorized_evaluated_poses:
                factorized_pose = np.concatenate(factorized_evaluated_poses, axis=0)
                factorized_t, factorized_r = _pose_errors(factorized_pose, target_pose)
                factorized_joint = np.maximum(factorized_t / 1.0, factorized_r / 10.0)
                factorized_oracle = int(np.argmin(factorized_joint))
            else:
                factorized_t = factorized_r = None
                factorized_oracle = None
            factorized_stage_diagnostics = (
                None if not factorized_trace
                else _factorized_trace_diagnostics(factorized_trace, target_pose)
            )
            initial_t, initial_r = _pose_errors(search_initial_poses, target_pose)
            if initial_live_scores is None:
                raise RuntimeError("pattern search did not evaluate its initial basins")
            initial_top = int(np.argmax(initial_live_scores))
            # A refined state is an additional hypothesis, never a replacement
            # for the retrieval basin or the initialized state.  Score the
            # original domains with the same local scorer and build the
            # protected union before physical NMS.
            domain_seed_scores = score_batch(domain_seed_poses)
            searched_poses, searched_scores, searched_updates, searched_kind = (
                _protected_hypothesis_union(
                    domain_seed_poses, domain_seed_scores,
                    search_initial_poses, initial_live_scores, search.basins,
                )
            )
            distinct_order = greedy_distinct_pose_basin_order(
                searched_scores, searched_poses,
                np.ones((searched_scores.size,), dtype=bool),
                excluded_candidate_index=None,
            )
            if not distinct_order.size:
                raise RuntimeError("pattern search produced no distinct basin")
            distinct_poses = searched_poses[distinct_order]
            distinct_scores = searched_scores[distinct_order]
            distinct_updates = searched_updates[distinct_order]
            distinct_kind = searched_kind[distinct_order]
            final_t, final_r = _pose_errors(distinct_poses[:1], target_pose)
            distinct_t, distinct_r = _pose_errors(
                distinct_poses, target_pose,
            )
            distinct_joint = np.maximum(distinct_t / 1.0, distinct_r / 10.0)
            final_oracle = int(np.argmin(distinct_joint))
            best_initial = int(np.argmin(np.maximum(initial_t / 1.0, initial_r / 10.0)))
            result_rows.append({
                "image_id": image_id,
                "initial_candidate_indices": selected.tolist(),
                "search_initial_pose_count": int(search_initial_poses.shape[0]),
                "global_joint_rendered_pose_count": global_joint_rendered_pose_count,
                "factorized_initialization_evaluated_pose_count": (
                    0 if factorized_result is None else factorized_result.evaluated_pose_count
                ),
                "factorized_position_stage_pose_counts": (
                    None if factorized_result is None
                    else list(factorized_result.position_stage_pose_counts)
                ),
                "factorized_orientation_stage_pose_count": (
                    None if factorized_result is None
                    else factorized_result.orientation_stage_pose_count
                ),
                "factorized_stage_diagnostics": factorized_stage_diagnostics,
                "evaluated_pose_trace_diagnostics": evaluated_diagnostics,
                "relative_correction_proposal_count": relative_correction_proposal_count,
                "global_joint_oracle_best_translation_m": (
                    None if global_oracle is None else float(global_t[global_oracle])
                ),
                "global_joint_oracle_best_rotation_deg": (
                    None if global_oracle is None else float(global_r[global_oracle])
                ),
                "global_joint_any_strict_0_5m_5deg": (
                    None if global_oracle is None else bool(np.any(
                        (global_t <= 0.5 + 1e-6) & (global_r <= 5.0 + 1e-5)
                    ))
                ),
                "global_joint_any_loose_1m_10deg": (
                    None if global_oracle is None else bool(np.any(
                        (global_t <= 1.0 + 1e-6) & (global_r <= 10.0 + 1e-5)
                    ))
                ),
                "factorized_oracle_best_translation_m": (
                    None if factorized_oracle is None
                    else float(factorized_t[factorized_oracle])
                ),
                "factorized_oracle_best_rotation_deg": (
                    None if factorized_oracle is None
                    else float(factorized_r[factorized_oracle])
                ),
                "factorized_any_strict_0_5m_5deg": (
                    None if factorized_oracle is None else bool(np.any(
                        (factorized_t <= 0.5 + 1e-6) & (factorized_r <= 5.0 + 1e-5)
                    ))
                ),
                "factorized_any_loose_1m_10deg": (
                    None if factorized_oracle is None else bool(np.any(
                        (factorized_t <= 1.0 + 1e-6) & (factorized_r <= 10.0 + 1e-5)
                    ))
                ),
                "initial_topscore_translation_m": float(initial_t[initial_top]),
                "initial_topscore_rotation_deg": float(initial_r[initial_top]),
                "initial_oracle_best_translation_m": float(initial_t[best_initial]),
                "initial_oracle_best_rotation_deg": float(initial_r[best_initial]),
                "final_translation_m": float(final_t[0]),
                "final_rotation_deg": float(final_r[0]),
                "final_strict_0_5m_5deg": bool(final_t[0] <= 0.5 + 1e-6 and final_r[0] <= 5.0 + 1e-5),
                "final_loose_1m_10deg": bool(final_t[0] <= 1.0 + 1e-6 and final_r[0] <= 10.0 + 1e-5),
                "final_any_strict_0_5m_5deg": bool(np.any(
                    (distinct_t <= 0.5 + 1e-6) & (distinct_r <= 5.0 + 1e-5)
                )),
                "final_any_loose_1m_10deg": bool(np.any(
                    (distinct_t <= 1.0 + 1e-6) & (distinct_r <= 10.0 + 1e-5)
                )),
                "final_oracle_best_translation_m": float(distinct_t[final_oracle]),
                "final_oracle_best_rotation_deg": float(distinct_r[final_oracle]),
                "final_oracle_best_score_rank": final_oracle + 1,
                "objective_drift_from_topscore": bool(
                    max(final_t[0] / 1.0, final_r[0] / 10.0)
                    > max(initial_t[initial_top] / 1.0, initial_r[initial_top] / 10.0) + 1e-8
                    and float(distinct_scores[0])
                    > float(initial_live_scores[initial_top]) + 1e-8
                ),
                "final_score": float(distinct_scores[0]),
                "final_source_semantics": str(distinct_kind[0]),
                "protected_hypothesis_count_before_nms": int(searched_scores.size),
                "final_distinct_basin_count": int(distinct_order.size),
                "final_distinct_score_margin": (
                    None if distinct_order.size < 2
                    else float(distinct_scores[0] - distinct_scores[1])
                ),
                "final_duplicate_basin_count_removed": int(
                    searched_scores.size - distinct_order.size
                ),
                "accepted_updates": int(distinct_updates[0]),
                "evaluated_pose_count": int(
                    search.evaluated_pose_count + global_joint_rendered_pose_count
                    + (0 if factorized_result is None else factorized_result.evaluated_pose_count)
                ),
                "local_search_evaluated_pose_count": int(search.evaluated_pose_count),
                "rendered_pose_count": rendered_pose_count,
                "rejected_outside_domain_count": rejected_outside_domain,
                "completed_sweeps": completed_search_levels,
                "render_seconds": render_seconds,
            })
            print(json.dumps(result_rows[-1]), flush=True)
    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "coarse_basin_model_file_sha256": file_sha256(Path(args.model)),
        "coarse_basin_model_content_sha256": model_metadata["model_content_sha256"],
        "local_scorer_model_file_sha256": (
            None if local_model_path is None else file_sha256(local_model_path)
        ),
        "local_scorer_model_content_sha256": score_model_metadata["model_content_sha256"],
        "local_scorer_uses_explicit_axis_supervision": bool(
            score_model_metadata.get("local_supervision", False)
        ),
        "relative_correction_model_file_sha256": (
            None if correction_model_path is None else file_sha256(correction_model_path)
        ),
        "relative_correction_model_content_sha256": (
            None if correction_model_metadata is None
            else correction_model_metadata["model_content_sha256"]
        ),
        "relative_correction_modes_per_basin": (
            None if correction_model is None
            else int(args.relative_correction_modes_per_basin)
        ),
        "factorized_model_file_sha256": (
            None if factorized_model_path is None else file_sha256(factorized_model_path)
        ),
        "factorized_model_content_sha256": (
            None if factorized_model_metadata is None
            else factorized_model_metadata["model_content_sha256"]
        ),
        "factorized_feature_semantics": (
            None if factorized_model_metadata is None
            else factorized_model_metadata.get("feature_semantics")
        ),
        "factorized_input_channels": (
            None if factorized_model is None else int(factorized_model.input_channels)
        ),
        "local_refinement_feature_semantics": (
            score_model_metadata.get("feature_semantics")
            if str(args.live_score_semantics) == "learned_ranker"
            else str(args.live_score_semantics)
        ),
        "factorized_position_beam_per_basin": (
            None if factorized_model is None and not bool(args.phase_factorized_initialization)
            else int(args.factorized_position_beam_per_basin)
        ),
        "factorized_survivors_per_basin": (
            None if factorized_model is None and not bool(args.phase_factorized_initialization)
            else int(args.factorized_survivors_per_basin)
        ),
        "phase_factorized_initialization": bool(args.phase_factorized_initialization),
        "live_score_semantics": str(args.live_score_semantics),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_file_sha256": mapper_sha,
        "view_conditioned_field_file_sha256": (
            None if view_conditioned_path is None else file_sha256(view_conditioned_path)
        ),
        "view_conditioned_field_content_sha256": (
            None if view_conditioned_field is None else view_conditioned_field.content_sha256
        ),
        "query_typed_geometry_predictor_file_sha256": (
            None if query_typed_path is None else file_sha256(query_typed_path)
        ),
        "query_typed_geometry_predictor_content_sha256": (
            None if query_typed_metadata is None
            else query_typed_metadata["model_content_sha256"]
        ),
        "include_route": str(args.include_route),
        "query_count": len(result_rows),
        "initial_basins": int(args.initial_basins),
        "initialization_semantics": str(args.initialization_semantics),
        "translation_radius_m": float(args.translation_radius_m),
        "rotation_radius_deg": float(args.rotation_radius_deg),
        "minimum_translation_radius_m": float(args.minimum_translation_radius_m),
        "minimum_rotation_radius_deg": float(args.minimum_rotation_radius_deg),
        "continuous_domain_translation_half_extent_m": float(
            args.search_domain_translation_half_extent_m
        ),
        "continuous_domain_rotation_radius_deg": float(
            args.search_domain_rotation_radius_deg
        ),
        "maximum_sweeps": int(args.maximum_sweeps),
        "search_semantics": str(args.search_semantics),
        "beam_width_per_initial_basin": (
            None if str(args.search_semantics) == "greedy_pattern_v1"
            else int(args.beam_width_per_initial_basin)
        ),
        "global_joint_initialization": bool(args.global_joint_initialization),
        "local_model_as_natural_transfer_control": natural_transfer_control,
        "global_joint_survivors_per_basin": (
            int(args.global_joint_survivors_per_basin)
            if bool(args.global_joint_initialization) else None
        ),
        "strict_capture_rate": float(np.mean([row["final_strict_0_5m_5deg"] for row in result_rows])),
        "loose_capture_rate": float(np.mean([row["final_loose_1m_10deg"] for row in result_rows])),
        "any_basin_strict_capture_rate": float(np.mean([
            row["final_any_strict_0_5m_5deg"] for row in result_rows
        ])),
        "any_basin_loose_capture_rate": float(np.mean([
            row["final_any_loose_1m_10deg"] for row in result_rows
        ])),
        "global_joint_probe_strict_oracle_rate": (
            None if not bool(args.global_joint_initialization) else float(np.mean([
                row["global_joint_any_strict_0_5m_5deg"] for row in result_rows
            ]))
        ),
        "global_joint_probe_loose_oracle_rate": (
            None if not bool(args.global_joint_initialization) else float(np.mean([
                row["global_joint_any_loose_1m_10deg"] for row in result_rows
            ]))
        ),
        "factorized_probe_strict_oracle_rate": (
            None if factorized_model is None and not bool(args.phase_factorized_initialization)
            else float(np.mean([
                row["factorized_any_strict_0_5m_5deg"] for row in result_rows
            ]))
        ),
        "factorized_probe_loose_oracle_rate": (
            None if factorized_model is None and not bool(args.phase_factorized_initialization)
            else float(np.mean([
                row["factorized_any_loose_1m_10deg"] for row in result_rows
            ]))
        ),
        "median_final_translation_m": float(np.median([row["final_translation_m"] for row in result_rows])),
        "median_final_rotation_deg": float(np.median([row["final_rotation_deg"] for row in result_rows])),
        "mean_render_seconds_per_query": float(np.mean([row["render_seconds"] for row in result_rows])),
        "rows": result_rows,
        "target_pose_consumed_only_after_each_search": bool(
            str(args.initialization_semantics) == "learned_topk"
        ),
        "target_pose_used_by_scorer": False,
        "search_is_bounded_to_union_of_selected_continuous_domains": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "bounded_multibasin_search_diagnostic_not_calibrated_localization",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "query_count", "strict_capture_rate", "loose_capture_rate",
        "any_basin_strict_capture_rate", "any_basin_loose_capture_rate",
        "median_final_translation_m", "median_final_rotation_deg",
        "mean_render_seconds_per_query",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
