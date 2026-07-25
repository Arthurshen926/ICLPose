"""Build current-P1 coherent wrong-mode supervision without target leakage.

The frozen P1 layout contains only query locations, a fixed global top-L
landmark denominator, and coarse posterior mass.  This tool first writes an
inference-only proposal contract and target-free cross-fit PnP hypotheses.
Only after those files are durable does it join train-image GT poses to mine
coherent wrong landmark configurations for mapper training.

The output hard-mode artifact is intentionally train-only.  It is compatible
with ``--landmark_coherent_hard_negative_artifact`` in
``train_real_radio_joint_localization.py``; the paired proposal artifact must
be passed through ``--landmark_coherent_hard_negative_proposals``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.mine_pose_conditioned_system_hard_negatives import (
    PoseConditionedHardNegativeConfig,
    _gt_pose_w2c,
    _project_candidate_residuals,
    mine_query_system_error_modes,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapCamera,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    GroupedCandidateCrossfitPools,
    GroupedCandidatePnPConfig,
    GroupedProsacProfile,
    PoseVerificationCandidatePool,
    generate_grouped_prosac_hypotheses,
    partition_grouped_candidate_pool,
    verify_pose_candidate_pool,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


P1_PROPOSAL_FORMAT = "current_p1_coherent_mapper_proposals_inference_only_v1"
P1_HYPOTHESIS_FORMAT = "current_p1_coherent_mapper_hypotheses_inference_only_v1"
P1_HYPOTHESIS_EVIDENCE_VERSION = "fixed_p1_global_top20_coarse_prior_v1"
HARD_MODE_FORMAT = "pose_conditioned_system_hard_modes_v2"

_TARGET_ARRAY_TOKENS = ("target", "residual", "label", "ground_truth")

# The P1 bank is sizeable but immutable.  Linux ``fork`` lets worker processes
# share it copy-on-write, avoiding a descriptor-bank pickle per query.  This is
# deliberately scoped to target-free PnP generation; GT mining stays serial
# and visibly after the inference-only files are saved.
_P1_GENERATION_WORKER_STATE: dict[str, object] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-layout", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--candidate-limits", default="1,3,5,10,20")
    parser.add_argument("--sampling-temperatures", default="0.5,1.0")
    parser.add_argument("--hypotheses-per-limit", type=int, default=48)
    parser.add_argument("--minimal-set-sizes", default="4,5,6")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Query-level CPU workers for target-free grouped PnP generation.",
    )
    parser.add_argument("--holdout-folds", type=int, default=4)
    parser.add_argument("--verification-fold", type=int, default=0)
    parser.add_argument("--final-audit-fold", type=int, default=1)
    parser.add_argument(
        "--crossfit-spatial-fold-policy", default="cell_rotated_balanced"
    )
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--min-fit-matches", type=int, default=8)
    parser.add_argument("--min-fit-grid-cells", type=int, default=4)
    parser.add_argument("--min-bearing-span-deg", type=float, default=3.0)
    parser.add_argument("--residual-sigma-px", type=float, default=2.0)
    parser.add_argument("--candidate-outlier-likelihood", type=float, default=1e-3)
    parser.add_argument("--null-likelihood", type=float, default=1e-3)
    parser.add_argument("--min-bad-translation-m", type=float, default=0.25)
    parser.add_argument("--max-bad-translation-m", type=float, default=3.0)
    parser.add_argument("--min-bad-rotation-deg", type=float, default=1.0)
    parser.add_argument("--bad-pose-consistency-px", type=float, default=3.0)
    parser.add_argument("--hard-score-log-margin", type=float, default=0.7)
    parser.add_argument("--min-consistent-groups", type=int, default=6)
    parser.add_argument("--min-consistent-grid-cells", type=int, default=3)
    parser.add_argument("--max-bad-modes-per-query", type=int, default=8)
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


def _canonical_query_id(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _json_sha256_short(value: Mapping[str, object]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)


def _load_train_query_ids(path: Path, split_name: str) -> tuple[str, ...]:
    if str(split_name) != "train":
        raise ValueError("current P1 coherent hard-mode mining is train-only")
    payload = json.loads(Path(path).read_text())
    values = payload.get("train") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("split JSON has no non-empty train query list")
    query_ids = tuple(_canonical_query_id(value) for value in values)
    if any(not value for value in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("train query identifiers are invalid")
    return query_ids


def _validate_layout_train_contract(
    layout: CandidatePoseRGBSpatialLayout,
    train_query_ids: Sequence[str],
) -> np.ndarray:
    """Return train-layout rows after rejecting split or posterior drift."""

    expected = {_canonical_query_id(value) for value in train_query_ids}
    query_ids = np.asarray([_canonical_query_id(value) for value in layout.query_ids])
    split_names = np.asarray(layout.split_names).astype(str)
    rows = np.flatnonzero(split_names == "train").astype(np.int64)
    if rows.size == 0:
        raise ValueError("P1 layout has no train rows")
    actual = set(query_ids[rows].tolist())
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            "P1 layout/train split mismatch: "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    if np.any(np.isin(query_ids, np.asarray(sorted(expected))) & (split_names != "train")):
        raise ValueError("P1 train query appears outside its declared train split")
    if np.any(np.abs(
        layout.candidate_prior_probabilities[rows].sum(axis=1)
        + layout.null_probabilities[rows]
        - 1.0
    ) > 2e-4):
        raise ValueError("P1 candidate/null posterior is not normalized")
    return rows


def _proposal_arrays_from_layout(
    layout: CandidatePoseRGBSpatialLayout,
    train_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialize only train P1 rows in the trainer-compatible proposal schema."""

    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= layout.row_count):
        raise ValueError("P1 proposal rows are invalid")
    query_ids = np.asarray([_canonical_query_id(value) for value in layout.query_ids[rows]])
    arrays = {
        "layout_row_indices": rows,
        "source_point_ids": np.asarray(layout.source_point_ids[rows], dtype=np.int64),
        "query_ids": query_ids,
        "xy": np.asarray(layout.xy[rows], dtype=np.float32),
        "candidate_track_ids": np.asarray(
            layout.candidate_track_ids[rows], dtype=np.int64
        ),
        "candidate_bank_rows": np.asarray(
            layout.candidate_bank_rows[rows], dtype=np.int64
        ),
        "candidate_prior_probabilities": np.asarray(
            layout.candidate_prior_probabilities[rows], dtype=np.float32
        ),
        "null_probabilities": np.asarray(
            layout.null_probabilities[rows], dtype=np.float32
        ),
    }
    _validate_target_free_proposal_arrays(arrays)
    return arrays


def _validate_target_free_proposal_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Reject targets and verify a fixed explicit-null candidate denominator."""

    names = {str(name).lower() for name in arrays}
    forbidden = sorted(
        name for name in names if any(token in name for token in _TARGET_ARRAY_TOKENS)
    )
    if forbidden:
        raise ValueError(f"proposal contract exposes target-side fields: {forbidden}")
    required = {
        "layout_row_indices",
        "source_point_ids",
        "query_ids",
        "xy",
        "candidate_track_ids",
        "candidate_bank_rows",
        "candidate_prior_probabilities",
        "null_probabilities",
    }
    if set(arrays) != required:
        raise ValueError("proposal contract has an unexpected schema")
    rows = np.asarray(arrays["layout_row_indices"], dtype=np.int64).reshape(-1)
    source_ids = np.asarray(arrays["source_point_ids"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float64)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    bank_rows = np.asarray(arrays["candidate_bank_rows"], dtype=np.int64)
    priors = np.asarray(arrays["candidate_prior_probabilities"], dtype=np.float64)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float64).reshape(-1)
    count = len(rows)
    if (
        count == 0
        or len(np.unique(rows)) != count
        or len(np.unique(source_ids)) != count
        or len(query_ids) != count
        or np.any(query_ids == "")
        or xy.shape != (count, 2)
        or tracks.ndim != 2
        or tracks.shape[0] != count
        or bank_rows.shape != tracks.shape
        or priors.shape != tracks.shape
        or null.shape != (count,)
        or not np.isfinite(xy).all()
        or not np.isfinite(priors).all()
        or not np.isfinite(null).all()
        or np.any(priors < 0.0)
        or np.any(null < 0.0)
    ):
        raise ValueError("proposal contract arrays are invalid")
    valid = tracks >= 0
    if (
        np.any(valid & (bank_rows < 0))
        or np.any(~valid & (bank_rows != -1))
        or np.any(~valid & (priors > 1e-7))
        or np.any(np.abs(priors.sum(axis=1) + null - 1.0) > 2e-4)
    ):
        raise ValueError("proposal contract candidate posterior is invalid")


def _validate_bank_alignment(
    proposal_arrays: Mapping[str, np.ndarray],
    *,
    bank_track_ids: np.ndarray,
) -> None:
    bank_rows = np.asarray(proposal_arrays["candidate_bank_rows"], dtype=np.int64)
    tracks = np.asarray(proposal_arrays["candidate_track_ids"], dtype=np.int64)
    valid = bank_rows >= 0
    if np.any(bank_rows[valid] >= len(bank_track_ids)):
        raise ValueError("P1 proposal references a missing projected-bank row")
    if np.any(np.asarray(bank_track_ids, dtype=np.int64)[bank_rows[valid]] != tracks[valid]):
        raise ValueError("P1 proposal bank rows resolve to different physical tracks")


def _candidate_pool_from_proposal_rows(
    proposal_arrays: Mapping[str, np.ndarray],
    proposal_rows: np.ndarray,
    *,
    bank_xyz: np.ndarray,
) -> PoseVerificationCandidatePool:
    rows = np.asarray(proposal_rows, dtype=np.int64).reshape(-1)
    candidate_rows = np.asarray(proposal_arrays["candidate_bank_rows"], dtype=np.int64)[
        rows
    ]
    valid = candidate_rows >= 0
    safe_rows = np.maximum(candidate_rows, 0)
    xyz = np.asarray(bank_xyz, dtype=np.float64)[safe_rows].copy()
    xyz[~valid] = np.nan
    return PoseVerificationCandidatePool(
        token_indices=np.asarray(proposal_arrays["layout_row_indices"], dtype=np.int64)[
            rows
        ],
        xy=np.asarray(proposal_arrays["xy"], dtype=np.float64)[rows],
        track_ids=np.asarray(proposal_arrays["candidate_track_ids"], dtype=np.int64)[
            rows
        ],
        prototype_ids=candidate_rows,
        xyz=xyz,
        descriptor_scores=np.asarray(
            proposal_arrays["candidate_prior_probabilities"], dtype=np.float64
        )[rows],
        valid_mask=valid,
        null_scores=np.asarray(proposal_arrays["null_probabilities"], dtype=np.float64)[
            rows
        ],
    )


def _validate_crossfit_partition(
    partition: GroupedCandidateCrossfitPools,
    *,
    require_track_purge: bool,
) -> None:
    role_sets = {
        "fit": set(map(int, partition.fit_tokens)),
        "verification": set(map(int, partition.verification_tokens)),
        "audit": set(map(int, partition.audit_tokens)),
    }
    role_names = tuple(role_sets)
    for first_index, first_name in enumerate(role_names):
        for second_name in role_names[first_index + 1 :]:
            if role_sets[first_name].intersection(role_sets[second_name]):
                raise RuntimeError("cross-fit P1 token roles overlap")
    if min(len(values) for values in role_sets.values()) <= 0:
        raise RuntimeError("cross-fit P1 partition has an empty role")
    if require_track_purge and not bool(
        partition.partition_audit.get("strict_track_disjoint", False)
    ):
        raise RuntimeError("cross-fit P1 partition did not enforce track disjointness")


def _camera_assignments_only(
    images_path: Path,
) -> dict[str, int]:
    """Read only query image -> camera ID before target-free generation.

    ``images.bin`` physically stores registered poses, so this deliberately
    returns and retains no pose field.  The later GT join reads the file again
    in a visibly separate target-only block.
    """

    images = read_colmap_images_binary(Path(images_path))
    assignments = {
        _canonical_query_id(image.image_name): int(image.camera_id)
        for image in images.values()
    }
    if not assignments:
        raise ValueError("COLMAP image camera assignment is empty")
    return assignments


def _pnp_config_from_args(args: argparse.Namespace) -> GroupedCandidatePnPConfig:
    minimal_set_sizes = _csv_ints(args.minimal_set_sizes)
    pnp_config = GroupedCandidatePnPConfig(
        candidate_limits=_csv_ints(args.candidate_limits),
        sampling_temperatures=_csv_floats(args.sampling_temperatures),
        holdout_folds=int(args.holdout_folds),
        verification_fold=int(args.verification_fold),
        verification_fold_count=1,
        final_audit_fold=int(args.final_audit_fold),
        crossfit_mode="token_spatial_track_purged",
        crossfit_role_assignment="fixed",
        crossfit_spatial_fold_policy=str(args.crossfit_spatial_fold_policy),
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
        min_fit_matches=int(args.min_fit_matches),
        min_fit_grid_cells=int(args.min_fit_grid_cells),
        min_xyz_second_singular_ratio=1e-3,
        candidate_pool_residual_sigma_px=float(args.residual_sigma_px),
        candidate_pose_outlier_likelihood=float(args.candidate_outlier_likelihood),
        candidate_pose_null_likelihood=float(args.null_likelihood),
        generation_mode="grouped_prosac",
        prosac_min_bearing_span_deg=float(args.min_bearing_span_deg),
        prosac_local_optimization=False,
        prosac_use_spatial_modes=False,
        prosac_profiles=(
            GroupedProsacProfile(
                name="p1_coarse_prior",
                hypotheses_per_limit=int(args.hypotheses_per_limit),
                minimal_set_sizes=minimal_set_sizes,
                candidate_probability_power=1.0,
                candidate_uniform_mix=0.0,
                local_optimization=False,
                use_spatial_modes=False,
            ),
            GroupedProsacProfile(
                name="p1_tempered_exploration",
                hypotheses_per_limit=int(args.hypotheses_per_limit),
                minimal_set_sizes=minimal_set_sizes,
                candidate_probability_power=0.5,
                candidate_uniform_mix=0.05,
                local_optimization=False,
                use_spatial_modes=False,
            ),
        ),
    )
    # Dataclass validation is invoked by construction.  Keeping the explicit
    # local binding makes the target-free config hash below unambiguous.
    return pnp_config


def _query_seed(query_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(str(query_id).encode("utf-8")).digest()[:8], "little"
    )


def _finite_verification_score(
    pose_w2c: np.ndarray,
    pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig,
) -> float | None:
    verification = verify_pose_candidate_pool(
        pose_w2c,
        pool,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(config.candidate_pose_outlier_likelihood),
        null_likelihood=float(config.candidate_pose_null_likelihood),
        strict_threshold_px=float(config.verification_strict_px),
        loose_threshold_px=float(config.verification_loose_px),
        grid_rows=int(config.grid_rows),
        grid_cols=int(config.grid_cols),
    )
    if verification is None:
        return None
    value = verification.fixed_posterior_log_likelihood_mean
    return None if value is None or not np.isfinite(value) else float(value)


def _validate_target_free_hypothesis_payload(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> None:
    """Prevent accidental target serialization before the explicit GT join."""

    forbidden = sorted(
        str(name)
        for name in arrays
        if str(name) != "evaluation_labels"
        and any(token in str(name).lower() for token in _TARGET_ARRAY_TOKENS)
    )
    if forbidden:
        raise ValueError(f"target-free hypothesis artifact exposes {forbidden}")
    if (
        metadata.get("format") != P1_HYPOTHESIS_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_generation") is not False
        or metadata.get("candidate_denominator")
        != "fixed_global_top20_coarse_prior_with_explicit_null"
        or metadata.get("ranking_pool") != "crossfit_verification"
        or metadata.get("audit_pool") != "crossfit_audit_diagnostic_only"
    ):
        raise ValueError("target-free hypothesis metadata is invalid")
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "poses_w2c",
        "verification_log_likelihood_means",
        "audit_log_likelihood_means",
        "sample_track_ids",
        "sample_token_indices",
        "generation_profiles",
        "metadata_json",
    }
    if set(arrays) != required:
        raise ValueError("target-free hypothesis artifact schema is invalid")
    count = len(np.asarray(arrays["query_ids"]).reshape(-1))
    if (
        count == 0
        or np.asarray(arrays["poses_w2c"]).shape != (count, 4, 4)
        or np.asarray(arrays["verification_log_likelihood_means"]).shape != (count,)
        or np.asarray(arrays["audit_log_likelihood_means"]).shape != (count,)
        or np.asarray(arrays["hypothesis_indices"]).shape != (count,)
        or np.asarray(arrays["sample_track_ids"]).shape[0] != count
        or np.asarray(arrays["sample_token_indices"]).shape
        != np.asarray(arrays["sample_track_ids"]).shape
        or np.asarray(arrays["generation_profiles"]).shape != (count,)
        or not np.isfinite(np.asarray(arrays["poses_w2c"], dtype=np.float64)).all()
        or not np.isfinite(
            np.asarray(arrays["verification_log_likelihood_means"], dtype=np.float64)
        ).all()
    ):
        raise ValueError("target-free hypothesis arrays are invalid")


def _generation_metadata(
    *,
    layout_path: Path,
    proposal_path: Path,
    bank_path: Path,
    split_path: Path,
    model_dir: Path,
    layout_metadata: Mapping[str, object],
    pnp_config: GroupedCandidatePnPConfig,
) -> dict[str, object]:
    grouped_config = asdict(pnp_config)
    return {
        "format": P1_HYPOTHESIS_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_pose_evidence_version": P1_HYPOTHESIS_EVIDENCE_VERSION,
        "candidate_denominator": "fixed_global_top20_coarse_prior_with_explicit_null",
        "ranking_pool": "crossfit_verification",
        "audit_pool": "crossfit_audit_diagnostic_only",
        "generation_pool": "crossfit_fit",
        "crossfit_roles": "token_spatial_track_purged",
        "grouped_config": grouped_config,
        "grouped_config_sha256": _json_sha256_short(grouped_config),
        "inputs": {
            "p1_layout": str(layout_path.resolve()),
            "p1_layout_sha256": file_sha256_short(layout_path),
            "proposal_contract": str(proposal_path.resolve()),
            "proposal_contract_sha256": file_sha256_short(proposal_path),
            "projected_landmark_bank": str(bank_path.resolve()),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json": str(split_path.resolve()),
            "split_json_sha256": file_sha256_short(split_path),
            "colmap_cameras_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_camera_assignment_sha256": file_sha256_short(
                model_dir / "images.bin"
            ),
            "matcha_joint_checkpoint_sha256": layout_metadata.get(
                "matcha_joint_checkpoint_sha256"
            ),
            "descriptor_space_id": layout_metadata.get("descriptor_space_id"),
            "projection_space_id": layout_metadata.get("projection_space_id"),
            "projection_method": "projected_observation_full_map",
            "mapper_mode": "joint_full_map",
        },
    }


def _build_target_free_hypotheses(
    *,
    proposal_arrays: Mapping[str, np.ndarray],
    train_query_ids: Sequence[str],
    bank_xyz: np.ndarray,
    cameras: Mapping[int, ColmapCamera],
    camera_assignments: Mapping[str, int],
    pnp_config: GroupedCandidatePnPConfig,
    workers: int = 1,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Generate on fit groups, score only on held-out verification groups."""

    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("P1 target-free generation worker count must be positive")
    normalized_queries = tuple(_canonical_query_id(value) for value in train_query_ids)
    if len(set(normalized_queries)) != len(normalized_queries):
        raise ValueError("P1 target-free generation query ids are not unique")
    state: dict[str, object] = {
        "proposal_arrays": proposal_arrays,
        "bank_xyz": np.asarray(bank_xyz, dtype=np.float64),
        "cameras": cameras,
        "camera_assignments": camera_assignments,
        "pnp_config": pnp_config,
    }
    if worker_count == 1:
        records = [
            _build_target_free_hypotheses_for_query(query_id, **state)
            for query_id in normalized_queries
        ]
    else:
        # ``fork`` is intentional here: it preserves the immutable in-memory
        # bank without serializing it to every worker.  The repository targets
        # Linux training hosts; fail loudly on a platform without this contract
        # rather than silently switching to a memory-amplifying spawn path.
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel P1 generation requires multiprocessing fork")
        global _P1_GENERATION_WORKER_STATE
        if _P1_GENERATION_WORKER_STATE is not None:
            raise RuntimeError("P1 generation worker state is already active")
        _P1_GENERATION_WORKER_STATE = state
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                # executor.map preserves query order, which in turn makes the
                # artifact byte-stable for identical frozen inputs/configs.
                records = list(executor.map(_build_target_free_hypotheses_worker, normalized_queries))
        finally:
            _P1_GENERATION_WORKER_STATE = None

    all_hypothesis_query_ids: list[str] = []
    poses: list[np.ndarray] = []
    verification_scores: list[float] = []
    audit_scores: list[float] = []
    sample_tracks: list[np.ndarray] = []
    sample_tokens: list[np.ndarray] = []
    profiles: list[str] = []
    per_query: list[dict[str, object]] = []
    for record in records:
        all_hypothesis_query_ids.extend(record["hypothesis_query_ids"])
        poses.extend(record["poses"])
        verification_scores.extend(record["verification_scores"])
        audit_scores.extend(record["audit_scores"])
        sample_tracks.extend(record["sample_tracks"])
        sample_tokens.extend(record["sample_tokens"])
        profiles.extend(record["profiles"])
        per_query.append(record["summary"])
    if not poses:
        raise RuntimeError("current P1 grouped PROSAC produced no scored hypotheses")
    count = len(poses)
    arrays = {
        "query_ids": np.asarray(all_hypothesis_query_ids),
        "split_names": np.full((count,), "train"),
        "evaluation_labels": np.full((count,), "p1_train_crossfit"),
        "hypothesis_indices": np.arange(count, dtype=np.int64),
        "poses_w2c": np.stack(poses, axis=0),
        "verification_log_likelihood_means": np.asarray(
            verification_scores, dtype=np.float64
        ),
        "audit_log_likelihood_means": np.asarray(audit_scores, dtype=np.float64),
        "sample_track_ids": np.stack(sample_tracks, axis=0),
        "sample_token_indices": np.stack(sample_tokens, axis=0),
        "generation_profiles": np.asarray(profiles),
    }
    return arrays, per_query


def _build_target_free_hypotheses_worker(query_id: str) -> dict[str, object]:
    """Fork-worker entry point with only immutable target-free state."""

    state = _P1_GENERATION_WORKER_STATE
    if state is None:
        raise RuntimeError("P1 generation worker lacks immutable parent state")
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        # OpenCV is already required by the PnP path.  Do not make its optional
        # thread-setting API a separate runtime dependency.
        pass
    return _build_target_free_hypotheses_for_query(query_id, **state)


def _build_target_free_hypotheses_for_query(
    query_id: str,
    *,
    proposal_arrays: Mapping[str, np.ndarray],
    bank_xyz: np.ndarray,
    cameras: Mapping[int, ColmapCamera],
    camera_assignments: Mapping[str, int],
    pnp_config: GroupedCandidatePnPConfig,
) -> dict[str, object]:
    """One deterministic query-level target-free PnP generation job."""

    normalized = _canonical_query_id(query_id)
    query_ids = np.asarray(proposal_arrays["query_ids"]).astype(str)
    rows = np.flatnonzero(query_ids == normalized).astype(np.int64)
    if rows.size == 0:
        raise ValueError(f"proposal contract misses train query {query_id!r}")
    camera_id = camera_assignments.get(normalized)
    if camera_id is None or int(camera_id) not in cameras:
        raise ValueError(f"train query {query_id!r} has no valid camera assignment")
    camera = cameras[int(camera_id)]
    pool = _candidate_pool_from_proposal_rows(proposal_arrays, rows, bank_xyz=bank_xyz)
    crossfit = partition_grouped_candidate_pool(
        pool,
        camera,
        config=pnp_config,
        query_seed=_query_seed(normalized),
    )
    _validate_crossfit_partition(crossfit, require_track_purge=True)
    if crossfit.fit.query_count < int(pnp_config.min_fit_matches):
        raise RuntimeError(f"train query {query_id!r} has too few fit groups")
    generated = generate_grouped_prosac_hypotheses(
        crossfit.fit,
        camera,
        config=pnp_config,
        query_seed=_query_seed(normalized),
    )
    sample_width = max(
        8,
        max(
            size
            for profile in pnp_config.prosac_profiles
            for size in profile.minimal_set_sizes
        ),
    )
    successful = 0
    scored = 0
    poses: list[np.ndarray] = []
    verification_scores: list[float] = []
    audit_scores: list[float] = []
    sample_tracks: list[np.ndarray] = []
    sample_tokens: list[np.ndarray] = []
    profiles: list[str] = []
    for item in generated:
        if not item.solver_success or item.pose_w2c is None:
            continue
        successful += 1
        verification_score = _finite_verification_score(
            item.pose_w2c, crossfit.verification, camera, config=pnp_config
        )
        if verification_score is None:
            continue
        audit_score = _finite_verification_score(
            item.pose_w2c, crossfit.audit, camera, config=pnp_config
        )
        columns = np.asarray(item.sample.candidate_columns, dtype=np.int64)
        sample_rows = np.asarray(item.sample.row_indices, dtype=np.int64)
        if columns.shape != sample_rows.shape or len(columns) > sample_width:
            raise RuntimeError("grouped PROSAC sample contract is invalid")
        padded_tracks = np.full((sample_width,), -1, dtype=np.int64)
        padded_tokens = np.full((sample_width,), -1, dtype=np.int64)
        padded_tracks[: len(columns)] = crossfit.fit.track_ids[sample_rows, columns]
        padded_tokens[: len(columns)] = crossfit.fit.token_indices[sample_rows]
        poses.append(np.asarray(item.pose_w2c, dtype=np.float64).reshape(4, 4))
        verification_scores.append(float(verification_score))
        audit_scores.append(np.nan if audit_score is None else float(audit_score))
        sample_tracks.append(padded_tracks)
        sample_tokens.append(padded_tokens)
        profiles.append(str(item.sample.generation_profile))
        scored += 1
    return {
        "hypothesis_query_ids": [normalized] * len(poses),
        "poses": poses,
        "verification_scores": verification_scores,
        "audit_scores": audit_scores,
        "sample_tracks": sample_tracks,
        "sample_tokens": sample_tokens,
        "profiles": profiles,
        "summary": {
            "query_id": normalized,
            "pool_group_count": int(pool.query_count),
            "generated_count": int(len(generated)),
            "solver_success_count": int(successful),
            "verification_scored_count": int(scored),
            "fit_token_indices": [int(value) for value in crossfit.fit_tokens],
            "verification_token_indices": [
                int(value) for value in crossfit.verification_tokens
            ],
            "audit_token_indices": [int(value) for value in crossfit.audit_tokens],
            "partition_audit": dict(crossfit.partition_audit),
        },
    }


def _train_gt_images(images_path: Path, train_query_ids: Sequence[str]) -> dict[str, Any]:
    """Load GT poses only after the inference-only hypothesis is already saved."""

    allowed = {_canonical_query_id(value) for value in train_query_ids}
    images = read_colmap_images_binary(Path(images_path))
    output = {
        _canonical_query_id(image.image_name): image
        for image in images.values()
        if _canonical_query_id(image.image_name) in allowed
    }
    missing = sorted(allowed.difference(output))
    if missing:
        raise ValueError(f"COLMAP model misses train GT images: {missing[:3]!r}")
    return output


def _mine_train_only_hard_modes(
    *,
    proposal_arrays: Mapping[str, np.ndarray],
    train_query_ids: Sequence[str],
    bank_xyz: np.ndarray,
    cameras: Mapping[int, ColmapCamera],
    gt_images: Mapping[str, Any],
    hypothesis_arrays: Mapping[str, np.ndarray],
    mining_config: PoseConditionedHardNegativeConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Join GT after generation and create the structured mapper loss targets."""

    mining_config.validate()
    query_ids = np.asarray(proposal_arrays["query_ids"]).astype(str)
    tracks = np.asarray(proposal_arrays["candidate_track_ids"], dtype=np.int64)
    bank_rows = np.asarray(proposal_arrays["candidate_bank_rows"], dtype=np.int64)
    valid = bank_rows >= 0
    xyz = np.asarray(bank_xyz, dtype=np.float64)[np.maximum(bank_rows, 0)].copy()
    xyz[~valid] = np.nan
    count, candidate_count = valid.shape
    positive = np.zeros((count, candidate_count), dtype=bool)
    hard_mask = np.zeros((count, candidate_count), dtype=bool)
    mode_counts = np.zeros((count, candidate_count), dtype=np.uint16)
    group_hard = np.zeros((count,), dtype=bool)
    group_mode_counts = np.zeros((count,), dtype=np.uint16)
    mode_ids = np.full(
        (count, int(mining_config.max_bad_modes_per_query)), -1, dtype=np.int32
    )
    mode_candidate_masks = np.zeros(
        (count, int(mining_config.max_bad_modes_per_query), candidate_count), dtype=bool
    )
    mode_query_ids: list[str] = []
    mode_support_group_counts: list[int] = []
    per_query: list[dict[str, object]] = []
    hypothesis_query_ids = np.asarray(hypothesis_arrays["query_ids"]).astype(str)
    hypothesis_poses = np.asarray(hypothesis_arrays["poses_w2c"], dtype=np.float64)
    hypothesis_scores = np.asarray(
        hypothesis_arrays["verification_log_likelihood_means"], dtype=np.float64
    )

    for query_id in train_query_ids:
        normalized = _canonical_query_id(query_id)
        rows = np.flatnonzero(query_ids == normalized).astype(np.int64)
        image = gt_images.get(normalized)
        if image is None:
            raise RuntimeError(f"train GT image vanished for {normalized!r}")
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            raise ValueError(f"train GT image {normalized!r} has no camera")
        gt_pose = _gt_pose_w2c(image)
        residuals, positive_depth = _project_candidate_residuals(
            xyz[rows],
            np.asarray(proposal_arrays["xy"], dtype=np.float64)[rows],
            valid[rows],
            gt_pose,
            camera,
        )
        positive[rows] = valid[rows] & positive_depth & (residuals <= 2.0)
        hypothesis_rows = np.flatnonzero(hypothesis_query_ids == normalized)
        if hypothesis_rows.size == 0:
            per_query.append(
                {
                    "query_id": normalized,
                    "hypothesis_count": 0,
                    "eligible_bad_hypothesis_count": 0,
                    "selected_bad_mode_count": 0,
                    "hard_group_count": 0,
                    "hard_candidate_count": 0,
                }
            )
            continue
        poses = hypothesis_poses[hypothesis_rows]
        pose_errors = [pnp_pose_error(pose, gt_pose) for pose in poses]
        result = mine_query_system_error_modes(
            query_xy=np.asarray(proposal_arrays["xy"], dtype=np.float64)[rows],
            candidate_xyz=xyz[rows],
            candidate_scores=np.asarray(
                proposal_arrays["candidate_prior_probabilities"], dtype=np.float64
            )[rows],
            valid_mask=valid[rows],
            positive_mask=positive[rows],
            hypothesis_poses_w2c=poses,
            hypothesis_scores=hypothesis_scores[hypothesis_rows],
            hypothesis_translation_errors_m=np.asarray(
                [item.translation_m for item in pose_errors], dtype=np.float64
            ),
            hypothesis_rotation_errors_deg=np.asarray(
                [item.rotation_deg for item in pose_errors], dtype=np.float64
            ),
            camera=camera,
            config=mining_config,
        )
        hard_mask[rows] = np.asarray(result["hard_negative_mask"], dtype=bool)
        mode_counts[rows] = np.asarray(result["candidate_mode_counts"], dtype=np.uint16)
        group_hard[rows] = np.asarray(result["group_hard_mask"], dtype=bool)
        group_mode_counts[rows] = np.asarray(
            result["group_mode_counts"], dtype=np.uint16
        )
        selected_mode_masks = np.asarray(result["selected_mode_hard_masks"], dtype=bool)
        for local_mode_index, local_mask in enumerate(selected_mode_masks):
            if local_mode_index >= mode_ids.shape[1]:
                raise RuntimeError("hard-mode miner exceeded declared mode capacity")
            participating = np.any(local_mask, axis=1)
            if not np.any(participating):
                raise RuntimeError("selected coherent hard mode has no groups")
            global_mode_id = len(mode_query_ids)
            mode_ids[rows[participating], local_mode_index] = global_mode_id
            mode_candidate_masks[rows, local_mode_index, :] = local_mask
            mode_query_ids.append(normalized)
            mode_support_group_counts.append(int(np.sum(participating)))
        per_query.append(
            {
                "query_id": normalized,
                "hypothesis_count": int(hypothesis_rows.size),
                "eligible_bad_hypothesis_count": int(
                    result["eligible_bad_hypothesis_count"]
                ),
                "selected_bad_mode_count": int(len(result["selected_modes"])),
                "hard_group_count": int(np.sum(result["group_hard_mask"])),
                "hard_candidate_count": int(np.sum(result["hard_negative_mask"])),
            }
        )

    selected_rows = np.arange(count, dtype=np.int64)
    selected_columns = np.broadcast_to(
        np.arange(candidate_count, dtype=np.int64), (count, candidate_count)
    ).copy()
    hard_arrays = {
        "selected_rows": selected_rows,
        "selected_columns": selected_columns,
        "query_ids": query_ids,
        "valid_edges": valid,
        "positive_mask_TARGET_ONLY": positive,
        "hard_negative_mask_TARGET_ONLY": hard_mask,
        "candidate_bad_mode_counts_TARGET_ONLY": mode_counts,
        "group_hard_mask_TARGET_ONLY": group_hard,
        "group_bad_mode_counts_TARGET_ONLY": group_mode_counts,
        "hard_mode_ids_TARGET_ONLY": mode_ids,
        "hard_mode_candidate_mask_TARGET_ONLY": mode_candidate_masks,
        "hard_mode_query_ids_TARGET_ONLY": np.asarray(mode_query_ids),
        "hard_mode_support_group_counts_TARGET_ONLY": np.asarray(
            mode_support_group_counts, dtype=np.uint16
        ),
    }
    _validate_train_only_hard_mode_arrays(hard_arrays, train_query_ids)
    return hard_arrays, per_query


def _validate_train_only_hard_mode_arrays(
    arrays: Mapping[str, np.ndarray], train_query_ids: Sequence[str]
) -> None:
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(arrays["selected_columns"], dtype=np.int64)
    valid = np.asarray(arrays["valid_edges"], dtype=bool)
    mode_ids = np.asarray(arrays["hard_mode_ids_TARGET_ONLY"], dtype=np.int64)
    mode_masks = np.asarray(
        arrays["hard_mode_candidate_mask_TARGET_ONLY"], dtype=bool
    )
    allowed = {_canonical_query_id(value) for value in train_query_ids}
    if (
        selected_rows.shape != (len(query_ids),)
        or not np.array_equal(selected_rows, np.arange(len(query_ids), dtype=np.int64))
        or selected_columns.shape != valid.shape
        or mode_ids.shape[0] != len(query_ids)
        or mode_masks.shape[:2] != mode_ids.shape
        or mode_masks.shape[2] != valid.shape[1]
        or set(query_ids.tolist()).difference(allowed)
        or np.any(mode_masks & ~valid[:, None, :])
    ):
        raise ValueError("train-only hard-mode arrays are invalid")


def build_current_p1_coherent_mapper_hard_modes(
    *,
    p1_layout_path: Path,
    projected_landmark_bank: Path,
    split_json: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Execute the target-free generation then the explicit train-only GT join."""

    layout_path = Path(p1_layout_path)
    bank_path = Path(projected_landmark_bank)
    split_path = Path(split_json)
    model_dir = Path(colmap_model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if str(args.split_name) != "train":
        raise ValueError("current P1 coherent hard-mode builder accepts only train")
    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    train_query_ids = _load_train_query_ids(split_path, args.split_name)
    train_rows = _validate_layout_train_contract(layout, train_query_ids)
    proposal_arrays = _proposal_arrays_from_layout(layout, train_rows)
    landmark_index, bank_metadata = load_landmark_index_npz(bank_path)
    if str(bank_metadata.get("descriptor_space_id", "")) != str(
        layout.metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("P1 layout and projected landmark bank descriptor spaces differ")
    _validate_bank_alignment(
        proposal_arrays, bank_track_ids=np.asarray(landmark_index.track_ids, dtype=np.int64)
    )
    proposal_path = output / "current_p1_train_proposals_inference_only_v1.npz"
    proposal_metadata = {
        "format": P1_PROPOSAL_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_denominator": "fixed_global_top20_coarse_prior_with_explicit_null",
        "split_names": ["train"],
        "inputs": {
            "p1_layout": str(layout_path.resolve()),
            "p1_layout_sha256": file_sha256_short(layout_path),
            "projected_landmark_bank": str(bank_path.resolve()),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json": str(split_path.resolve()),
            "split_json_sha256": file_sha256_short(split_path),
            "matcha_joint_checkpoint_sha256": layout.metadata.get(
                "matcha_joint_checkpoint_sha256"
            ),
            "descriptor_space_id": layout.metadata.get("descriptor_space_id"),
            "projection_space_id": layout.metadata.get("projection_space_id"),
            "projection_method": "projected_observation_full_map",
            "mapper_mode": "joint_full_map",
        },
    }
    _atomic_savez(
        proposal_path,
        **proposal_arrays,
        metadata_json=np.asarray(json.dumps(proposal_metadata, sort_keys=True)),
    )

    # This block receives camera intrinsics and image->camera assignment only;
    # no registered pose is retained until the target-only phase below.
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    camera_assignments = _camera_assignments_only(model_dir / "images.bin")
    pnp_config = _pnp_config_from_args(args)
    hypothesis_arrays, generation_per_query = _build_target_free_hypotheses(
        proposal_arrays=proposal_arrays,
        train_query_ids=train_query_ids,
        bank_xyz=np.asarray(landmark_index.xyz, dtype=np.float64),
        cameras=cameras,
        camera_assignments=camera_assignments,
        pnp_config=pnp_config,
        workers=int(args.workers),
    )
    hypothesis_path = output / "current_p1_train_crossfit_hypotheses_inference_only_v1.npz"
    hypothesis_metadata = _generation_metadata(
        layout_path=layout_path,
        proposal_path=proposal_path,
        bank_path=bank_path,
        split_path=split_path,
        model_dir=model_dir,
        layout_metadata=layout.metadata,
        pnp_config=pnp_config,
    )
    hypothesis_payload = {
        **hypothesis_arrays,
        "metadata_json": np.asarray(json.dumps(hypothesis_metadata, sort_keys=True)),
    }
    _validate_target_free_hypothesis_payload(hypothesis_payload, hypothesis_metadata)
    _atomic_savez(hypothesis_path, **hypothesis_payload)
    generation_manifest_path = (
        output / "current_p1_train_crossfit_manifest_inference_only_v1.json"
    )
    generation_manifest_path.write_text(
        json.dumps(
            {
                "format": "current_p1_crossfit_manifest_inference_only_v1",
                "contains_target_fields": False,
                "pose_or_ground_truth_used_for_generation": False,
                "hypothesis_artifact": str(hypothesis_path),
                "hypothesis_artifact_sha256": file_sha256_short(hypothesis_path),
                "per_query": generation_per_query,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # Explicit train-only boundary: GT poses are read only after the target-free
    # proposal/hypothesis files above are immutable and hash-addressed.
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
    gt_images = _train_gt_images(model_dir / "images.bin", train_query_ids)
    hard_arrays, target_per_query = _mine_train_only_hard_modes(
        proposal_arrays=proposal_arrays,
        train_query_ids=train_query_ids,
        bank_xyz=np.asarray(landmark_index.xyz, dtype=np.float64),
        cameras=cameras,
        gt_images=gt_images,
        hypothesis_arrays=hypothesis_arrays,
        mining_config=mining_config,
    )
    hard_mode_path = output / "pose_conditioned_system_hard_modes_v2.npz"
    hard_metadata = {
        "format": HARD_MODE_FORMAT,
        "training_only_target_artifact": True,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "ground_truth_joined_after_generation": True,
        "split_names": ["train"],
        "candidate_pose_evidence_version": P1_HYPOTHESIS_EVIDENCE_VERSION,
        "config": asdict(mining_config),
        "pnp_generation_config_sha256": _json_sha256_short(asdict(pnp_config)),
        "inputs": {
            "hypothesis_artifact": str(hypothesis_path.resolve()),
            "hypothesis_artifact_sha256": file_sha256_short(hypothesis_path),
            "crossfit_manifest": str(generation_manifest_path.resolve()),
            "crossfit_manifest_sha256": file_sha256_short(generation_manifest_path),
            "proposals": str(proposal_path.resolve()),
            "proposals_sha256": file_sha256_short(proposal_path),
            "p1_layout": str(layout_path.resolve()),
            "p1_layout_sha256": file_sha256_short(layout_path),
            "projected_landmark_bank": str(bank_path.resolve()),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json": str(split_path.resolve()),
            "split_json_sha256": file_sha256_short(split_path),
            "matcha_joint_checkpoint_sha256": layout.metadata.get(
                "matcha_joint_checkpoint_sha256"
            ),
            "descriptor_space_id": layout.metadata.get("descriptor_space_id"),
            "projection_space_id": layout.metadata.get("projection_space_id"),
            "projection_method": "projected_observation_full_map",
            "mapper_mode": "joint_full_map",
        },
    }
    _atomic_savez(
        hard_mode_path,
        **hard_arrays,
        metadata_json=np.asarray(json.dumps(hard_metadata, sort_keys=True)),
    )
    mode_support = np.asarray(
        hard_arrays["hard_mode_support_group_counts_TARGET_ONLY"], dtype=np.int64
    )
    summary = {
        "stage": "build_current_p1_coherent_mapper_hard_modes",
        "protocol": {
            "fixed_current_p1_global_top_l": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "rgb_or_measurement_used_for_generation": False,
            "generation_is_target_free": True,
            "generation_fit_verify_audit_crossfit": True,
            "track_purged_crossfit": True,
            "ground_truth_joined_after_generation": True,
            "hard_mode_artifact_train_only": True,
        },
        "generation": {
            "query_count": len(train_query_ids),
            "hypothesis_count": int(len(hypothesis_arrays["query_ids"])),
            "query_with_scored_hypothesis_count": int(
                sum(row["verification_scored_count"] > 0 for row in generation_per_query)
            ),
            "generated_count": int(sum(row["generated_count"] for row in generation_per_query)),
            "solver_success_count": int(
                sum(row["solver_success_count"] for row in generation_per_query)
            ),
        },
        "metrics_TARGET_ONLY": {
            "coherent_mode_count": int(mode_support.size),
            "coherent_mode_support_group_count": int(mode_support.sum()),
            "coherent_mode_support_group_min": int(mode_support.min())
            if mode_support.size
            else 0,
            "coherent_mode_support_group_median": float(np.median(mode_support))
            if mode_support.size
            else 0.0,
            "query_with_bad_mode_count": int(
                sum(row["selected_bad_mode_count"] > 0 for row in target_per_query)
            ),
            "hard_group_count": int(
                np.sum(hard_arrays["group_hard_mask_TARGET_ONLY"])
            ),
            "hard_candidate_count": int(
                np.sum(hard_arrays["hard_negative_mask_TARGET_ONLY"])
            ),
        },
        "outputs": {
            "proposals": str(proposal_path),
            "proposals_sha256": file_sha256_short(proposal_path),
            "hypotheses": str(hypothesis_path),
            "hypotheses_sha256": file_sha256_short(hypothesis_path),
            "crossfit_manifest": str(generation_manifest_path),
            "crossfit_manifest_sha256": file_sha256_short(generation_manifest_path),
            "hard_modes": str(hard_mode_path),
            "hard_modes_sha256": file_sha256_short(hard_mode_path),
        },
    }
    (output / "per_query_TARGET_ONLY.json").write_text(
        json.dumps(target_per_query, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_current_p1_coherent_mapper_hard_modes(
        p1_layout_path=Path(args.p1_layout),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        split_json=Path(args.split_json),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        args=args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
