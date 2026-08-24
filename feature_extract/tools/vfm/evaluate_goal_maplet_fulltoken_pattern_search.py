"""Oracle-capture pattern search using full-token surface RADIO energy.

The initial pose is a fixed 1m/10deg perturbation of GT solely to measure the
local capture basin. GT is never passed to the scorer. The frozen 3DGS map,
surface field, query RADIO mapper, and derivative-free SE(3) search are the
only scoring inputs; there are no keypoints, correspondences, or PnP.
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
from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import _pose_errors
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.fulltoken_surface_pose_energy import (
    CONDITIONAL_PHASE_CONTROL_SEMANTICS,
    CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS,
    CONTRASTIVE_SEMANTICS,
    PHASE_SEMANTICS,
    SHIFT_TOLERANT_PHASE_SEMANTICS,
    SEMANTICS,
    PARENT_GATED_SEMANTICS,
    PARENT_GATED_CONTRASTIVE_SEMANTICS,
    fulltoken_surface_pose_energy,
    conservative_fulltoken_phase_pose_energy,
    conditional_fulltoken_phase_pose_energy_control,
    parent_gated_fulltoken_surface_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.multibasin_pattern_search import (
    batched_multibasin_pattern_search,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import FrozenSoftSurfaceSceneGPU
from feature_extract.vfm.localization_v6.se3_update import se3_exp


SCHEMA = "goal_maplet_fulltoken_surface_pattern_search_oracle_capture_v1"


def _wide_oracle_candidate_index(
    candidate_valid: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
) -> int:
    """Diagnostic-only nearest candidate under the declared 2m/45deg scale."""

    valid_mask = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    translation = np.asarray(translation_m, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation_deg, dtype=np.float64).reshape(-1)
    if valid_mask.shape != translation.shape or valid_mask.shape != rotation.shape:
        raise ValueError("oracle candidate arrays differ")
    valid = np.flatnonzero(valid_mask)
    valid = valid[valid != 0]
    if valid.size == 0 or not np.isfinite(translation[valid]).all() or not np.isfinite(rotation[valid]).all():
        raise ValueError("oracle diagnostic lacks finite deployable candidates")
    return int(valid[np.argmin(np.maximum(translation[valid] / 2.0, rotation[valid] / 45.0))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_start", type=int, default=6)
    parser.add_argument("--maximum_queries", type=int, default=5)
    parser.add_argument("--maximum_sweeps", type=int, default=3)
    parser.add_argument("--render_batch_size", type=int, default=8)
    parser.add_argument("--minimum_cosine_evidence", type=float, default=-1.0)
    parser.add_argument("--phase_shift_radius_tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--score_semantics",
        choices=(
            "appearance_only", "appearance_parent_product",
            "conservative_phase", "conditional_phase",
        ),
        default="appearance_parent_product",
    )
    parser.add_argument(
        "--initialization_semantics",
        choices=(
            "gt_offset_1m10_oracle", "dataset_first_proposal", "dataset_best_phase",
            "dataset_best_phase_independent_of_first", "dataset_oracle_best_wide_candidate",
        ),
        default="gt_offset_1m10_oracle",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= int(args.phase_shift_radius_tokens) <= 4:
        raise ValueError("phase_shift_radius_tokens must lie in [0,4]")
    if (
        int(args.phase_shift_radius_tokens) != 0
        and str(args.score_semantics) not in {"conservative_phase", "conditional_phase"}
    ):
        raise ValueError("phase shift tolerance is only defined for phase scorers")
    output = Path(args.output)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite full-token search report")
    arrays, dataset_metadata = load_pose_candidate_dataset(Path(args.dataset))
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    if (
        physical.content_sha256 != dataset_metadata.get("physical_map_sha256")
        or field.content_sha256 != dataset_metadata.get("canonical_field_sha256")
        or field.physical_map_sha256 != physical.content_sha256
    ):
        raise ValueError("dataset/map/field lineage differs")
    contract = json.loads(Path(args.field_feature_contract).read_text())
    mapper_sha = file_sha256(Path(args.surface_mapper))
    if (
        contract.get("canonical_field_sha256") != field.content_sha256
        or contract.get("query_readout_sha256") != mapper_sha
    ):
        raise ValueError("surface mapper/field contract differs")
    device = torch.device(str(args.device))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(device))
    mapper.model.to(device).eval()
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(device))
    contributors = _load_contributors(Path(args.contributors))
    begin = int(args.query_start)
    end = min(begin + int(args.maximum_queries), int(arrays["image_ids"].size))
    if not 0 <= begin < end:
        raise ValueError("query range is empty")
    rows = []
    total_render_seconds = 0.0
    for query_index in range(begin, end):
        image_id = str(arrays["image_ids"][query_index])
        contributor_path = contributors.get(image_id)
        if contributor_path is None:
            raise ValueError(f"missing contributor for {image_id}")
        camera = _camera(contributor_path)
        target_pose = np.asarray(arrays["candidate_poses_w2c"][query_index, 0], dtype=np.float64)
        with torch.no_grad():
            raw = torch.as_tensor(
                arrays["radio_final"][query_index], device=device, dtype=torch.float32
            )[None]
            query_descriptor = mapper.model(raw)[0].permute(1, 2, 0).reshape(2304, 128)
        initial_candidate_index = None
        if str(args.initialization_semantics) in {
            "dataset_best_phase", "dataset_best_phase_independent_of_first",
        }:
            frozen_score = np.full(arrays["candidate_valid"].shape[1], -np.inf)
            for candidate in range(1, int(frozen_score.size)):
                if not bool(arrays["candidate_valid"][query_index, candidate]):
                    continue
                phase_function = (
                    conditional_fulltoken_phase_pose_energy_control
                    if str(args.score_semantics) == "conditional_phase"
                    else conservative_fulltoken_phase_pose_energy
                )
                phase = phase_function(
                    query_descriptor,
                    torch.as_tensor(
                        arrays["target_canonical_features"][query_index, candidate],
                        device=device, dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        arrays["target_child_weights"][query_index, candidate],
                        device=device, dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        arrays["target_modality_valid"][query_index, candidate, ..., 0],
                        device=device,
                    ),
                    height=36, width=64,
                    maximum_shift_tokens=int(args.phase_shift_radius_tokens),
                )
                frozen_score[candidate] = float(phase.score.detach().cpu())
            order = np.argsort(-frozen_score, kind="stable")
            if str(args.initialization_semantics) == "dataset_best_phase_independent_of_first":
                initial_candidate_index = None
                for candidate in order.tolist():
                    if not np.isfinite(frozen_score[candidate]) or candidate == 1:
                        continue
                    translation, rotation = _pose_errors(
                        arrays["candidate_poses_w2c"][query_index, candidate][None],
                        arrays["candidate_poses_w2c"][query_index, 1],
                    )
                    if float(translation[0]) > 0.5 + 1.0e-6 or float(rotation[0]) > 5.0 + 1.0e-5:
                        initial_candidate_index = int(candidate)
                        break
                if initial_candidate_index is None:
                    raise ValueError(f"no phase basin independent of the first proposal for {image_id}")
            else:
                initial_candidate_index = int(order[0])
            if not np.isfinite(frozen_score[initial_candidate_index]):
                raise ValueError(f"no valid phase-ranked proposal for {image_id}")
            initial_pose = np.asarray(
                arrays["candidate_poses_w2c"][query_index, initial_candidate_index],
                dtype=np.float64,
            )
        elif str(args.initialization_semantics) == "dataset_first_proposal":
            initial_candidate_index = 1
            if not bool(arrays["candidate_valid"][query_index, initial_candidate_index]):
                raise ValueError(f"first proposal is invalid for {image_id}")
            initial_pose = np.asarray(
                arrays["candidate_poses_w2c"][query_index, initial_candidate_index],
                dtype=np.float64,
            )
        elif str(args.initialization_semantics) == "dataset_oracle_best_wide_candidate":
            initial_candidate_index = _wide_oracle_candidate_index(
                arrays["candidate_valid"][query_index],
                arrays["translation_m"][query_index],
                arrays["rotation_deg"][query_index],
            )
            initial_pose = np.asarray(
                arrays["candidate_poses_w2c"][query_index, initial_candidate_index],
                dtype=np.float64,
            )
        else:
            initial_delta = np.asarray([0.0, np.radians(10.0), 0.0, 1.0, 0.0, 0.0])
            initial_pose = se3_exp(initial_delta) @ target_pose
        reliability = torch.as_tensor(
            arrays["query_reliability"][query_index], device=device, dtype=torch.float32
        )
        retrieval_path = Path(dataset_metadata["retrieval_directory"]) / (
            image_id.replace("/", "__") + ".npz"
        )
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        query_parent_ids = torch.as_tensor(retrieval.token_parent_ids, device=device)
        query_parent_probability = torch.as_tensor(
            retrieval.token_parent_probabilities, device=device, dtype=torch.float32
        )
        token_xy = arrays["token_xy"][query_index]
        render_seconds = 0.0
        initial_score_cache: list[float] = []

        def score_batch(poses: np.ndarray) -> np.ndarray:
            nonlocal render_seconds
            score = []
            for start in range(0, int(poses.shape[0]), int(args.render_batch_size)):
                batch = scene.render_exact_batch(
                    poses[start : start + int(args.render_batch_size)],
                    camera, width=64, height=36, top_l=4,
                )
                render_seconds += float(batch.audit.total_seconds)
                for rendered in batch.rendered:
                    target_descriptor = torch.as_tensor(
                        np.asarray(rendered.child_features).reshape(2304, 4, 128),
                        device=device, dtype=torch.float32,
                    )
                    target_mass = torch.as_tensor(
                        np.asarray(rendered.child_weights).reshape(2304, 4),
                        device=device, dtype=torch.float32,
                    )
                    target_valid = torch.as_tensor(
                        np.asarray(rendered.child_feature_valid).reshape(2304, 4),
                        device=device,
                    )
                    if str(args.score_semantics) == "appearance_parent_product":
                        child_rows = np.asarray(rendered.child_rows).reshape(2304, 4)
                        safe_child = np.maximum(child_rows, 0)
                        target_parent_ids = physical.maplet_ids[
                            physical.child_parent_rows[safe_child]
                        ]
                        value = parent_gated_fulltoken_surface_pose_energy(
                            query_descriptor, reliability, token_xy,
                            query_parent_ids, query_parent_probability,
                            target_descriptor, target_mass, target_valid,
                            torch.as_tensor(target_parent_ids, device=device),
                            minimum_cosine_evidence=float(args.minimum_cosine_evidence),
                        )
                    elif str(args.score_semantics) == "appearance_only":
                        value = fulltoken_surface_pose_energy(
                            query_descriptor, reliability, token_xy,
                            target_descriptor, target_mass, target_valid,
                            local_radius_tokens=0,
                            minimum_cosine_evidence=float(args.minimum_cosine_evidence),
                        )
                    elif str(args.score_semantics) == "conservative_phase":
                        value = conservative_fulltoken_phase_pose_energy(
                            query_descriptor, target_descriptor, target_mass, target_valid,
                            height=36, width=64,
                            maximum_shift_tokens=int(args.phase_shift_radius_tokens),
                        )
                    else:
                        value = conditional_fulltoken_phase_pose_energy_control(
                            query_descriptor, target_descriptor, target_mass, target_valid,
                            height=36, width=64,
                            maximum_shift_tokens=int(args.phase_shift_radius_tokens),
                        )
                    score.append(float(value.score.detach().cpu()))
            result = np.asarray(score, dtype=np.float64)
            if not initial_score_cache and result.shape == (1,):
                initial_score_cache.append(float(result[0]))
            return result

        search = batched_multibasin_pattern_search(
            initial_pose[None], score_batch,
            translation_radius_m=0.5,
            rotation_radius_deg=5.0,
            shrink_factor=0.5,
            minimum_translation_radius_m=0.125,
            minimum_rotation_radius_deg=0.625,
            maximum_sweeps=int(args.maximum_sweeps),
        )
        final = search.basins[0]
        initial_t, initial_r = _pose_errors(initial_pose[None], target_pose)
        final_t, final_r = _pose_errors(final.pose_w2c[None], target_pose)
        row = {
            "image_id": image_id,
            "initial_translation_m": float(initial_t[0]),
            "initial_rotation_deg": float(initial_r[0]),
            "initial_candidate_index": initial_candidate_index,
            "initial_pose_w2c": np.asarray(initial_pose, dtype=np.float64).tolist(),
            "final_translation_m": float(final_t[0]),
            "final_rotation_deg": float(final_r[0]),
            "final_pose_w2c": np.asarray(final.pose_w2c, dtype=np.float64).tolist(),
            # Pose errors are reconstructed through floating-point SE(3)
            # composition.  Treat the published closed thresholds as closed
            # despite a few arithmetic ulps (for example 5.000000000000283).
            "final_strict_0_5m_5deg": bool(
                final_t[0] <= 0.5 + 1.0e-6 and final_r[0] <= 5.0 + 1.0e-5
            ),
            "final_loose_1m_10deg": bool(
                final_t[0] <= 1.0 + 1.0e-6 and final_r[0] <= 10.0 + 1.0e-5
            ),
            "accepted_updates": int(final.accepted_updates),
            "evaluated_pose_count": int(search.evaluated_pose_count),
            "completed_sweeps": int(search.completed_sweeps),
            "initial_score": float(initial_score_cache[0]),
            "final_score": float(final.score),
            "render_seconds": float(render_seconds),
        }
        total_render_seconds += render_seconds
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "artifact_type": SCHEMA,
        "energy_semantics": (
            (
                PARENT_GATED_CONTRASTIVE_SEMANTICS
                if float(args.minimum_cosine_evidence) > -1.0
                else PARENT_GATED_SEMANTICS
            )
            if str(args.score_semantics) == "appearance_parent_product"
            else (
                (
                    (
                        PHASE_SEMANTICS
                        if int(args.phase_shift_radius_tokens) == 0
                        else SHIFT_TOLERANT_PHASE_SEMANTICS
                    )
                    if str(args.score_semantics) == "conservative_phase"
                    else (
                        CONDITIONAL_PHASE_CONTROL_SEMANTICS
                        if int(args.phase_shift_radius_tokens) == 0
                        else CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS
                    )
                )
                if str(args.score_semantics) in (
                    "conservative_phase", "conditional_phase"
                )
                else (
                    CONTRASTIVE_SEMANTICS
                    if float(args.minimum_cosine_evidence) > -1.0
                    else SEMANTICS
                )
            )
        ),
        "score_semantics": str(args.score_semantics),
        "phase_shift_radius_tokens": int(args.phase_shift_radius_tokens),
        "minimum_cosine_evidence": float(args.minimum_cosine_evidence),
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_file_sha256": mapper_sha,
        "initialization_semantics": (
            f"dataset_{args.score_semantics}_shift{int(args.phase_shift_radius_tokens)}_best_basin_independent_of_first_no_gt_v2"
            if str(args.initialization_semantics) == "dataset_best_phase_independent_of_first"
            else (
                f"dataset_{args.score_semantics}_shift{int(args.phase_shift_radius_tokens)}_top1_no_gt_initialization_v2"
                if str(args.initialization_semantics) == "dataset_best_phase"
                else (
                    "dataset_first_retrieval_proposal_no_gt_initialization_v1"
                    if str(args.initialization_semantics) == "dataset_first_proposal"
                    else (
                        "dataset_gt_oracle_best_candidate_by_2m45_joint_scale_v1"
                        if str(args.initialization_semantics) == "dataset_oracle_best_wide_candidate"
                        else "gt_left_se3_plus_1m_camera_x_plus_10deg_camera_y_oracle_v1"
                    )
                )
            )
        ),
        "search_semantics": "six_axis_derivative_free_pattern_search_no_gt_in_scorer_v1",
        "query_count": len(rows),
        "strict_capture_rate": float(np.mean([row["final_strict_0_5m_5deg"] for row in rows])),
        "loose_capture_rate": float(np.mean([row["final_loose_1m_10deg"] for row in rows])),
        "median_final_translation_m": float(np.median([row["final_translation_m"] for row in rows])),
        "median_final_rotation_deg": float(np.median([row["final_rotation_deg"] for row in rows])),
        "objective_drift_rate": float(np.mean([
            row["final_score"] > row["initial_score"] and
            max(row["final_translation_m"] / 1.0, row["final_rotation_deg"] / 10.0)
            > max(row["initial_translation_m"] / 1.0, row["initial_rotation_deg"] / 10.0) + 1.0e-8
            for row in rows
        ])),
        "total_render_seconds": float(total_render_seconds),
        "rows": rows,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": (
            "map_disjoint_natural_proposal_local_refinement_diagnostic"
            if str(args.initialization_semantics) not in (
                "gt_offset_1m10_oracle", "dataset_oracle_best_wide_candidate"
            )
            else "map_disjoint_gt_relative_oracle_capture_not_end_to_end_localization"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
