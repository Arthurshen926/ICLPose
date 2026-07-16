"""Mine train-only coherent wrong modes from frozen global landmark proposals.

Hypotheses are generated and saved before ground-truth poses are joined.  The
generation stage uses one immutable global top-L denominator per query point;
pose-dependent candidate reselection is deliberately unsupported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.mine_pose_conditioned_system_hard_negatives import (
    PoseConditionedHardNegativeConfig,
    _gt_pose_w2c,
    _project_candidate_residuals,
    mine_query_system_error_modes,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    GroupedCandidatePnPConfig,
    GroupedProsacProfile,
    PoseVerificationCandidatePool,
    generate_grouped_prosac_hypotheses,
    verify_pose_candidate_pool,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


HYPOTHESIS_FORMAT = "coherent_global_proposal_hypotheses_inference_only_v1"
HARD_MODE_FORMAT = "pose_conditioned_system_hard_modes_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--candidate_temperature", type=float, default=0.05)
    parser.add_argument("--null_probability", type=float, default=0.05)
    parser.add_argument("--candidate_limits", default="1,3,5,10,20")
    parser.add_argument("--sampling_temperatures", default="0.5,1.0")
    parser.add_argument("--hypotheses_per_limit", type=int, default=16)
    parser.add_argument("--minimal_set_sizes", default="4,5,6")
    parser.add_argument("--min_bad_translation_m", type=float, default=0.25)
    parser.add_argument("--max_bad_translation_m", type=float, default=3.0)
    parser.add_argument("--min_bad_rotation_deg", type=float, default=1.0)
    parser.add_argument("--bad_pose_consistency_px", type=float, default=3.0)
    parser.add_argument("--hard_score_log_margin", type=float, default=0.7)
    parser.add_argument("--min_consistent_groups", type=int, default=6)
    parser.add_argument("--min_consistent_grid_cells", type=int, default=3)
    parser.add_argument("--max_bad_modes_per_query", type=int, default=8)
    return parser.parse_args(argv)


def _csv_ints(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise ValueError("integer list must not be empty")
    return output


def _csv_floats(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise ValueError("float list must not be empty")
    return output


def _candidate_probabilities(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    *,
    temperature: float,
    null_probability: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("candidate scores and valid mask must share shape [N,L]")
    if float(temperature) <= 0.0:
        raise ValueError("candidate temperature must be positive")
    if not 0.0 < float(null_probability) < 1.0:
        raise ValueError("null probability must be in (0,1)")
    logits = np.where(valid, values / float(temperature), -np.inf)
    maxima = np.max(logits, axis=1, keepdims=True)
    if np.any(~np.isfinite(maxima)):
        raise ValueError("every query group must contain a valid candidate")
    weights = np.where(valid, np.exp(logits - maxima), 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights * (1.0 - float(null_probability))


def _pool_for_rows(
    rows: np.ndarray,
    *,
    xy: np.ndarray,
    track_ids: np.ndarray,
    prototype_ids: np.ndarray,
    bank_rows: np.ndarray,
    probabilities: np.ndarray,
    bank_xyz: np.ndarray,
    null_probability: float,
) -> PoseVerificationCandidatePool:
    selected_bank_rows = np.asarray(bank_rows, dtype=np.int64)[rows]
    valid = selected_bank_rows >= 0
    safe_rows = np.maximum(selected_bank_rows, 0)
    xyz = np.asarray(bank_xyz, dtype=np.float64)[safe_rows]
    xyz[~valid] = np.nan
    return PoseVerificationCandidatePool(
        token_indices=np.asarray(rows, dtype=np.int64),
        xy=np.asarray(xy, dtype=np.float64)[rows],
        track_ids=np.asarray(track_ids, dtype=np.int64)[rows],
        prototype_ids=np.asarray(prototype_ids, dtype=np.int64)[rows],
        xyz=xyz,
        descriptor_scores=np.asarray(probabilities, dtype=np.float64)[rows],
        valid_mask=valid,
        null_scores=np.full((len(rows),), float(null_probability), dtype=np.float64),
    )


def _query_seed(query_id: str) -> int:
    return int.from_bytes(hashlib.sha256(str(query_id).encode("utf-8")).digest()[:8], "little")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    proposals_path = Path(args.proposals)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    model_dir = Path(args.colmap_model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(proposals_path, allow_pickle=False) as data:
        proposal_metadata = json.loads(str(data["metadata_json"].item()))
        query_ids = np.asarray(data["query_ids"]).astype(str)
        xy = np.asarray(data["xy"], dtype=np.float64)
        bank_rows = np.asarray(data["bank_row_indices"], dtype=np.int64)
        track_ids = np.asarray(data["candidate_track_ids"], dtype=np.int64)
        prototype_ids = np.asarray(data["candidate_prototype_ids"], dtype=np.int64)
        coarse_scores = np.asarray(data["coarse_scores"], dtype=np.float64)
        pose_keep = np.asarray(data["pose_keep_mask"], dtype=bool)
    if bool(proposal_metadata.get("ground_truth_geometry_present", True)):
        raise ValueError("proposal artifact must be inference-only and target-free")
    if not (
        bank_rows.shape == track_ids.shape == prototype_ids.shape == coarse_scores.shape
        and len(query_ids) == len(xy) == len(pose_keep) == len(bank_rows)
    ):
        raise ValueError("proposal arrays are misaligned")

    landmark_index, bank_metadata = load_landmark_index_npz(bank_path)
    if str(proposal_metadata.get("descriptor_space_id", "")) != str(
        bank_metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("proposal and landmark bank descriptor spaces differ")
    valid = bank_rows >= 0
    safe_rows = np.maximum(bank_rows, 0)
    if np.any(track_ids[valid] != landmark_index.track_ids[safe_rows[valid]]):
        raise ValueError("proposal bank rows do not resolve to their recorded track ids")
    probabilities = _candidate_probabilities(
        coarse_scores,
        valid,
        temperature=float(args.candidate_temperature),
        null_probability=float(args.null_probability),
    )

    split = json.loads(split_path.read_text())
    split_name = str(args.split_name)
    if split_name != "train" or split_name not in split:
        raise ValueError("coherent hard-mode mining is restricted to the train split")
    allowed_queries = {str(value) for value in split[split_name]}
    available_queries = set(query_ids.tolist())
    missing_queries = sorted(allowed_queries.difference(available_queries))
    if missing_queries:
        raise ValueError(f"proposals miss train queries: {missing_queries[:5]!r}")

    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    pnp_config = GroupedCandidatePnPConfig(
        candidate_limits=_csv_ints(args.candidate_limits),
        sampling_temperatures=_csv_floats(args.sampling_temperatures),
        generation_mode="grouped_prosac",
        prosac_profiles=(
            GroupedProsacProfile(
                name="coherent_mining_raw",
                minimal_set_sizes=_csv_ints(args.minimal_set_sizes),
                hypotheses_per_limit=int(args.hypotheses_per_limit),
                candidate_probability_power=1.0,
                candidate_uniform_mix=0.0,
                local_optimization=False,
                use_spatial_modes=False,
            ),
        ),
        prosac_local_optimization=False,
    )

    hypothesis_query_ids: list[str] = []
    hypothesis_poses: list[np.ndarray] = []
    hypothesis_scores: list[float] = []
    hypothesis_sample_tracks: list[np.ndarray] = []
    query_rows_by_id: dict[str, np.ndarray] = {}
    per_query_generation: list[dict[str, object]] = []
    for query_id in sorted(allowed_queries):
        image = images_by_name.get(query_id)
        if image is None or int(image.camera_id) not in cameras:
            raise ValueError(f"query {query_id!r} is absent from the COLMAP model")
        rows = np.flatnonzero((query_ids == query_id) & pose_keep)
        if len(rows) < int(pnp_config.min_fit_matches):
            raise ValueError(f"query {query_id!r} has too few pose-kept groups")
        query_rows_by_id[query_id] = rows
        pool = _pool_for_rows(
            rows,
            xy=xy,
            track_ids=track_ids,
            prototype_ids=prototype_ids,
            bank_rows=bank_rows,
            probabilities=probabilities,
            bank_xyz=landmark_index.xyz,
            null_probability=float(args.null_probability),
        )
        generated = generate_grouped_prosac_hypotheses(
            pool,
            cameras[int(image.camera_id)],
            config=pnp_config,
            query_seed=_query_seed(query_id),
        )
        success_count = 0
        for item in generated:
            if not item.solver_success or item.pose_w2c is None:
                continue
            verification = verify_pose_candidate_pool(
                item.pose_w2c,
                pool,
                cameras[int(image.camera_id)],
                residual_sigma_px=2.0,
                candidate_outlier_likelihood=1e-3,
                null_likelihood=1e-3,
            )
            if verification is None or verification.fixed_posterior_log_likelihood_mean is None:
                continue
            score = float(verification.fixed_posterior_log_likelihood_mean)
            if not np.isfinite(score):
                continue
            sample_tracks = np.full((8,), -1, dtype=np.int64)
            values = np.asarray(item.sample.candidate_columns, dtype=np.int64)
            sample_rows = np.asarray(item.sample.row_indices, dtype=np.int64)
            take = min(len(values), len(sample_tracks))
            sample_tracks[:take] = pool.track_ids[sample_rows[:take], values[:take]]
            hypothesis_query_ids.append(query_id)
            hypothesis_poses.append(np.asarray(item.pose_w2c, dtype=np.float64))
            hypothesis_scores.append(score)
            hypothesis_sample_tracks.append(sample_tracks)
            success_count += 1
        per_query_generation.append(
            {
                "query_id": query_id,
                "generated_count": int(len(generated)),
                "successful_scored_count": int(success_count),
            }
        )
    if not hypothesis_poses:
        raise RuntimeError("grouped PROSAC produced no valid hypotheses")

    hypothesis_path = output_dir / "coherent_hypotheses_inference_only_v1.npz"
    hypothesis_metadata = {
        "format": HYPOTHESIS_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "candidate_denominator": "fixed_global_top_l_with_explicit_null",
        "inputs": {
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json": str(split_path),
            "split_json_sha256": file_sha256_short(split_path),
            "colmap_cameras_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_camera_assignment_sha256": file_sha256_short(model_dir / "images.bin"),
        },
        "config": {
            "candidate_temperature": float(args.candidate_temperature),
            "null_probability": float(args.null_probability),
            "grouped_pnp": asdict(pnp_config),
        },
    }
    np.savez_compressed(
        hypothesis_path,
        query_ids=np.asarray(hypothesis_query_ids),
        poses_w2c=np.stack(hypothesis_poses, axis=0),
        fixed_posterior_log_likelihood_means=np.asarray(hypothesis_scores, dtype=np.float64),
        sample_track_ids=np.stack(hypothesis_sample_tracks, axis=0),
        metadata_json=np.asarray(json.dumps(hypothesis_metadata, sort_keys=True)),
    )

    # Target join starts only after the inference-only artifact is immutable on disk.
    mining_config = PoseConditionedHardNegativeConfig(
        min_bad_translation_m=float(args.min_bad_translation_m),
        max_bad_translation_m=float(args.max_bad_translation_m),
        min_bad_rotation_deg=float(args.min_bad_rotation_deg),
        bad_pose_consistency_px=float(args.bad_pose_consistency_px),
        hard_score_log_margin=float(args.hard_score_log_margin),
        min_consistent_groups=int(args.min_consistent_groups),
        min_consistent_grid_cells=int(args.min_consistent_grid_cells),
        max_bad_modes_per_query=int(args.max_bad_modes_per_query),
    )
    mining_config.validate()
    selected_rows = np.concatenate(
        [query_rows_by_id[query_id] for query_id in sorted(allowed_queries)], axis=0
    )
    selected_columns = np.broadcast_to(
        np.arange(track_ids.shape[1], dtype=np.int64),
        (len(selected_rows), track_ids.shape[1]),
    ).copy()
    selected_valid = valid[selected_rows]
    selected_xyz = landmark_index.xyz[np.maximum(bank_rows[selected_rows], 0)]
    selected_xyz[~selected_valid] = np.nan
    positive = np.zeros_like(selected_valid)
    hard_mask = np.zeros_like(selected_valid)
    mode_counts = np.zeros(selected_valid.shape, dtype=np.uint16)
    group_hard = np.zeros((len(selected_rows),), dtype=bool)
    group_mode_counts = np.zeros((len(selected_rows),), dtype=np.uint16)
    mode_ids = np.full(
        (len(selected_rows), int(mining_config.max_bad_modes_per_query)),
        -1,
        dtype=np.int32,
    )
    mode_candidate_masks = np.zeros(
        (
            len(selected_rows),
            int(mining_config.max_bad_modes_per_query),
            selected_valid.shape[1],
        ),
        dtype=bool,
    )
    mode_query_ids: list[str] = []
    mode_support_group_counts: list[int] = []
    per_query_targets: list[dict[str, object]] = []
    hypothesis_query_array = np.asarray(hypothesis_query_ids)
    hypothesis_pose_array = np.stack(hypothesis_poses, axis=0)
    hypothesis_score_array = np.asarray(hypothesis_scores, dtype=np.float64)
    offset = 0
    for query_id in sorted(allowed_queries):
        image = images_by_name[query_id]
        rows = query_rows_by_id[query_id]
        count = len(rows)
        local = slice(offset, offset + count)
        gt_pose = _gt_pose_w2c(image)
        gt_residuals, gt_depth = _project_candidate_residuals(
            selected_xyz[local],
            xy[rows],
            selected_valid[local],
            gt_pose,
            cameras[int(image.camera_id)],
        )
        positive[local] = selected_valid[local] & gt_depth & (gt_residuals <= 2.0)
        hypothesis_rows = np.flatnonzero(hypothesis_query_array == query_id)
        poses = hypothesis_pose_array[hypothesis_rows]
        errors = [pnp_pose_error(pose, gt_pose) for pose in poses]
        result = mine_query_system_error_modes(
            query_xy=xy[rows],
            candidate_xyz=selected_xyz[local],
            candidate_scores=probabilities[rows],
            valid_mask=selected_valid[local],
            positive_mask=positive[local],
            hypothesis_poses_w2c=poses,
            hypothesis_scores=hypothesis_score_array[hypothesis_rows],
            hypothesis_translation_errors_m=np.asarray(
                [value.translation_m for value in errors], dtype=np.float64
            ),
            hypothesis_rotation_errors_deg=np.asarray(
                [value.rotation_deg for value in errors], dtype=np.float64
            ),
            camera=cameras[int(image.camera_id)],
            config=mining_config,
        )
        hard_mask[local] = result["hard_negative_mask"]
        mode_counts[local] = result["candidate_mode_counts"]
        group_hard[local] = result["group_hard_mask"]
        group_mode_counts[local] = result["group_mode_counts"]
        selected_mode_masks = np.asarray(
            result["selected_mode_hard_masks"], dtype=bool
        )
        for local_mode_index, local_mask in enumerate(selected_mode_masks):
            participating = np.any(local_mask, axis=1)
            if not np.any(participating):
                raise RuntimeError("selected coherent hard mode has no groups")
            global_mode_id = len(mode_query_ids)
            local_rows = np.flatnonzero(participating) + int(offset)
            mode_ids[local_rows, local_mode_index] = global_mode_id
            mode_candidate_masks[local, local_mode_index] = local_mask
            mode_query_ids.append(query_id)
            mode_support_group_counts.append(int(np.sum(participating)))
        per_query_targets.append(
            {
                "query_id": query_id,
                "hypothesis_count": int(len(hypothesis_rows)),
                "eligible_bad_hypothesis_count": int(result["eligible_bad_hypothesis_count"]),
                "selected_bad_mode_count": int(len(result["selected_modes"])),
                "hard_group_count": int(np.sum(result["group_hard_mask"])),
                "hard_candidate_count": int(np.sum(result["hard_negative_mask"])),
            }
        )
        offset += count

    hard_mode_path = output_dir / "pose_conditioned_system_hard_modes_v2.npz"
    hard_metadata = {
        "format": HARD_MODE_FORMAT,
        "training_only_target_artifact": True,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "ground_truth_joined_after_generation": True,
        "split_names": ["train"],
        "config": asdict(mining_config),
        "inputs": {
            "hypothesis_artifact": str(hypothesis_path),
            "hypothesis_artifact_sha256": file_sha256_short(hypothesis_path),
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json": str(split_path),
            "split_json_sha256": file_sha256_short(split_path),
        },
    }
    np.savez_compressed(
        hard_mode_path,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        query_ids=query_ids[selected_rows],
        valid_edges=selected_valid,
        positive_mask_TARGET_ONLY=positive,
        hard_negative_mask_TARGET_ONLY=hard_mask,
        candidate_bad_mode_counts_TARGET_ONLY=mode_counts,
        group_hard_mask_TARGET_ONLY=group_hard,
        group_bad_mode_counts_TARGET_ONLY=group_mode_counts,
        hard_mode_ids_TARGET_ONLY=mode_ids,
        hard_mode_candidate_mask_TARGET_ONLY=mode_candidate_masks,
        hard_mode_query_ids_TARGET_ONLY=np.asarray(mode_query_ids),
        hard_mode_support_group_counts_TARGET_ONLY=np.asarray(
            mode_support_group_counts, dtype=np.uint16
        ),
        metadata_json=np.asarray(json.dumps(hard_metadata, sort_keys=True)),
    )
    summary = {
        "stage": "coherent_global_proposal_hard_mode_mining",
        "protocol": {
            "hypotheses_saved_before_target_join": True,
            "hypothesis_generation_is_target_free": True,
            "candidate_denominator": "fixed_global_top_l_with_explicit_null",
            "train_only_target_join": True,
        },
        "metrics_TARGET_ONLY": {
            "query_count": int(len(allowed_queries)),
            "hypothesis_count": int(len(hypothesis_poses)),
            "coherent_mode_count": int(len(mode_query_ids)),
            "coherent_mode_support_group_count": int(
                np.sum(mode_support_group_counts, dtype=np.int64)
            ),
            "coherent_mode_support_group_min": int(
                min(mode_support_group_counts, default=0)
            ),
            "coherent_mode_support_group_median": float(
                np.median(mode_support_group_counts)
                if mode_support_group_counts
                else 0.0
            ),
            "query_with_bad_mode_count": int(
                sum(row["selected_bad_mode_count"] > 0 for row in per_query_targets)
            ),
            "hard_group_count": int(np.sum(group_hard)),
            "hard_candidate_count": int(np.sum(hard_mask)),
        },
        "generation": per_query_generation,
        "target_join": per_query_targets,
        "outputs": {
            "hypotheses": str(hypothesis_path),
            "hypotheses_sha256": file_sha256_short(hypothesis_path),
            "hard_modes": str(hard_mode_path),
            "hard_modes_sha256": file_sha256_short(hard_mode_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
