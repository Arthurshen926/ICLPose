"""Mine coherent matcher hard negatives from target-free pose hypotheses.

Pose generation remains inference-only. This tool joins generated hypotheses
with GT afterwards, identifies high-scoring wrong pose modes, and marks the
wrong landmark candidates that jointly support those modes. The output is a
training-only target artifact and must never be consumed by pose inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    concatenate_hypothesis_shard_field,
    load_inference_artifact_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapCamera,
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
    pnp_pose_error,
    rotation_angle_deg,
)


ARTIFACT_FORMAT = "pose_conditioned_system_hard_negatives_v1"
STRUCTURED_ARTIFACT_FORMAT = "pose_conditioned_system_hard_modes_v2"
_HYPOTHESIS_FIELDS_REQUIRED_BY_MINING = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "poses_w2c",
    "verification_log_likelihood_means",
)


@dataclass(frozen=True)
class PoseConditionedHardNegativeConfig:
    min_bad_translation_m: float = 0.25
    max_bad_translation_m: float = 3.0
    min_bad_rotation_deg: float = 1.0
    bad_pose_consistency_px: float = 3.0
    hard_score_log_margin: float = 0.7
    max_hard_candidates_per_group_per_mode: int = 2
    max_hard_candidates_per_group: int = 2
    min_consistent_groups: int = 6
    min_consistent_grid_cells: int = 3
    max_bad_modes_per_query: int = 8
    mode_translation_diversity_m: float = 0.10
    mode_rotation_diversity_deg: float = 1.0
    grid_rows: int = 4
    grid_cols: int = 4

    def validate(self) -> None:
        finite_nonnegative = (
            self.min_bad_translation_m,
            self.max_bad_translation_m,
            self.min_bad_rotation_deg,
            self.bad_pose_consistency_px,
            self.hard_score_log_margin,
            self.mode_translation_diversity_m,
            self.mode_rotation_diversity_deg,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in finite_nonnegative):
            raise ValueError("hard-negative thresholds must be finite and non-negative")
        if (
            self.max_bad_translation_m > 0.0
            and self.max_bad_translation_m < self.min_bad_translation_m
        ):
            raise ValueError("max bad translation must not be below the minimum")
        if self.bad_pose_consistency_px <= 0.0:
            raise ValueError("bad-pose consistency threshold must be positive")
        if min(
            self.min_consistent_groups,
            self.min_consistent_grid_cells,
            self.max_bad_modes_per_query,
            self.max_hard_candidates_per_group_per_mode,
            self.max_hard_candidates_per_group,
            self.grid_rows,
            self.grid_cols,
        ) <= 0:
            raise ValueError("hard-negative count and grid settings must be positive")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis_artifacts",
        required=True,
        help="comma-separated inference-only grouped hypothesis NPZ files",
    )
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument(
        "--score_key", default="ensemble__factorized_set_candidate_probability"
    )
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_names", default="train")
    parser.add_argument("--evaluation_label", default="")
    parser.add_argument("--allow_partial_query_coverage", action="store_true")
    parser.add_argument("--min_bad_translation_m", type=float, default=0.25)
    parser.add_argument("--max_bad_translation_m", type=float, default=3.0)
    parser.add_argument("--min_bad_rotation_deg", type=float, default=1.0)
    parser.add_argument("--bad_pose_consistency_px", type=float, default=3.0)
    parser.add_argument("--hard_score_log_margin", type=float, default=0.7)
    parser.add_argument(
        "--max_hard_candidates_per_group_per_mode", type=int, default=2
    )
    parser.add_argument("--max_hard_candidates_per_group", type=int, default=2)
    parser.add_argument("--min_consistent_groups", type=int, default=6)
    parser.add_argument("--min_consistent_grid_cells", type=int, default=3)
    parser.add_argument("--max_bad_modes_per_query", type=int, default=8)
    parser.add_argument("--mode_translation_diversity_m", type=float, default=0.10)
    parser.add_argument("--mode_rotation_diversity_deg", type=float, default=1.0)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument(
        "--preserve_mode_membership",
        action="store_true",
        help=(
            "export the query-local bad-pose mode membership required by the "
            "structured mode loss"
        ),
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def _compact(
    values: np.ndarray, selected_rows: np.ndarray, selected_columns: np.ndarray
) -> np.ndarray:
    columns = np.asarray(selected_columns, dtype=np.int64)
    safe = np.maximum(columns, 0)
    output = np.take_along_axis(np.asarray(values)[selected_rows], safe, axis=1).copy()
    if np.issubdtype(output.dtype, np.floating):
        output[columns < 0] = np.nan
    else:
        output[columns < 0] = -1
    return output


def positive_mask_from_posthoc_gt_residuals(
    *,
    proposal_residuals: np.ndarray,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
    valid_edges: np.ndarray,
    positive_threshold_px: float,
) -> np.ndarray:
    """Join GT residual targets only after target-free candidates are frozen."""

    threshold = float(positive_threshold_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("positive threshold must be finite and positive")
    valid = np.asarray(valid_edges, dtype=bool)
    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    columns = np.asarray(selected_columns, dtype=np.int64)
    if columns.shape != valid.shape or rows.shape != (len(valid),):
        raise ValueError("selected candidate layout is not aligned with valid edges")
    residuals = _compact(
        np.asarray(proposal_residuals, dtype=np.float64), rows, columns
    )
    if residuals.shape != valid.shape:
        raise ValueError("compacted proposal residuals are not candidate aligned")
    return valid & np.isfinite(residuals) & (residuals <= threshold)


def select_candidate_rows_for_allowed_queries(
    *,
    query_ids: np.ndarray,
    allowed_query_ids: set[str],
    fields: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep all candidate-target rows inside the declared training split."""

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    allowed = {str(query_id) for query_id in allowed_query_ids}
    if len(ids) == 0 or not allowed:
        raise ValueError("candidate row selection needs non-empty query sets")
    selected = np.flatnonzero(np.isin(ids, np.asarray(sorted(allowed))))
    if len(selected) == 0:
        raise ValueError("candidate rows do not cover the requested query set")
    filtered: dict[str, np.ndarray] = {}
    for name, value in fields.items():
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != len(ids):
            raise ValueError(f"candidate field {name!r} is not query-row aligned")
        filtered[str(name)] = array[selected]
    return ids[selected], filtered


def _gt_pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(np.asarray(image.qvec, dtype=np.float64))
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    return pose


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]


def _project_candidate_residuals(
    xyz: np.ndarray,
    xy: np.ndarray,
    valid_mask: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64)
    pixels = np.asarray(xy, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("candidate xyz must have shape [G,L,3]")
    if pixels.shape != (points.shape[0], 2) or valid.shape != points.shape[:2]:
        raise ValueError("candidate projection arrays are not aligned")
    residuals = np.full(valid.shape, np.inf, dtype=np.float64)
    positive_depth = np.zeros(valid.shape, dtype=bool)
    flat_valid = np.flatnonzero(valid.reshape(-1))
    if len(flat_valid) == 0:
        return residuals, positive_depth
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for pose-conditioned mining") from exc
    flat_points = points.reshape(-1, 3)[flat_valid]
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    depths = (flat_points @ pose[:3, :3].T + pose[:3, 3])[:, 2]
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    projected, _ = cv2.projectPoints(
        flat_points, rvec, pose[:3, 3], camera_matrix, distortion
    )
    repeated_xy = np.repeat(pixels, points.shape[1], axis=0)[flat_valid]
    flat_residuals = np.linalg.norm(projected.reshape(-1, 2) - repeated_xy, axis=1)
    residuals.reshape(-1)[flat_valid] = flat_residuals
    positive_depth.reshape(-1)[flat_valid] = depths > 1e-6
    return residuals, positive_depth


def _grid_cell_count(
    xy: np.ndarray,
    selected_groups: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_cols: int,
) -> int:
    groups = np.asarray(selected_groups, dtype=bool).reshape(-1)
    if not np.any(groups):
        return 0
    points = np.asarray(xy, dtype=np.float64)[groups]
    cols = np.clip(
        np.floor(points[:, 0] / max(float(image_width), 1.0) * int(grid_cols)),
        0,
        int(grid_cols) - 1,
    ).astype(np.int64)
    rows = np.clip(
        np.floor(points[:, 1] / max(float(image_height), 1.0) * int(grid_rows)),
        0,
        int(grid_rows) - 1,
    ).astype(np.int64)
    return int(len(np.unique(rows * int(grid_cols) + cols)))


def mine_query_system_error_modes(
    *,
    query_xy: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_scores: np.ndarray,
    valid_mask: np.ndarray,
    positive_mask: np.ndarray,
    hypothesis_poses_w2c: np.ndarray,
    hypothesis_scores: np.ndarray,
    hypothesis_translation_errors_m: np.ndarray,
    hypothesis_rotation_errors_deg: np.ndarray,
    camera: ColmapCamera,
    config: PoseConditionedHardNegativeConfig,
) -> dict[str, object]:
    """Return candidate masks induced by diverse high-scoring wrong pose modes."""

    config.validate()
    xy = np.asarray(query_xy, dtype=np.float64)
    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    positive = np.asarray(positive_mask, dtype=bool)
    poses = np.asarray(hypothesis_poses_w2c, dtype=np.float64)
    pose_scores = np.asarray(hypothesis_scores, dtype=np.float64).reshape(-1)
    translation = np.asarray(
        hypothesis_translation_errors_m, dtype=np.float64
    ).reshape(-1)
    rotation = np.asarray(
        hypothesis_rotation_errors_deg, dtype=np.float64
    ).reshape(-1)
    if xyz.shape[:2] != valid.shape or scores.shape != valid.shape:
        raise ValueError("candidate arrays must share [G,L]")
    if positive.shape != valid.shape or xy.shape != (len(valid), 2):
        raise ValueError("candidate targets and query xy are not aligned")
    if poses.shape != (len(pose_scores), 4, 4):
        raise ValueError("hypothesis poses and scores are not aligned")
    if len(translation) != len(poses) or len(rotation) != len(poses):
        raise ValueError("hypothesis target errors are not aligned")
    if np.any(positive & ~valid):
        raise ValueError("positive candidates must be valid")

    bad = translation >= float(config.min_bad_translation_m)
    if float(config.min_bad_rotation_deg) > 0.0:
        bad |= rotation >= float(config.min_bad_rotation_deg)
    if float(config.max_bad_translation_m) > 0.0:
        bad &= translation <= float(config.max_bad_translation_m)
    eligible = np.flatnonzero(
        bad
        & np.isfinite(pose_scores)
        & np.isfinite(translation)
        & np.isfinite(rotation)
    )
    order = eligible[
        np.lexsort((eligible, translation[eligible], -pose_scores[eligible]))
    ]

    log_scores = np.log(np.clip(scores, 1e-12, None))
    best_positive_log_score = np.max(
        np.where(positive & valid, log_scores, -np.inf), axis=1
    )
    has_positive = np.any(positive & valid, axis=1)
    candidate_mode_counts = np.zeros(valid.shape, dtype=np.uint16)
    group_mode_counts = np.zeros((len(valid),), dtype=np.uint16)
    selected_modes: list[dict[str, object]] = []
    selected_mode_hard_masks: list[np.ndarray] = []
    selected_poses: list[np.ndarray] = []
    for hypothesis_index in order.tolist():
        pose = poses[hypothesis_index]
        if any(
            np.linalg.norm(_camera_center(pose) - _camera_center(previous))
            < float(config.mode_translation_diversity_m)
            and rotation_angle_deg(pose[:3, :3], previous[:3, :3])
            < float(config.mode_rotation_diversity_deg)
            for previous in selected_poses
        ):
            continue
        residuals, positive_depth = _project_candidate_residuals(
            xyz, xy, valid, pose, camera
        )
        score_plausible = log_scores >= (
            best_positive_log_score[:, None]
            - float(config.hard_score_log_margin)
        )
        hard_candidates = (
            valid
            & ~positive
            & positive_depth
            & (residuals <= float(config.bad_pose_consistency_px))
            & score_plausible
        )
        candidate_limit = int(config.max_hard_candidates_per_group_per_mode)
        if candidate_limit < hard_candidates.shape[1]:
            for group_index in np.flatnonzero(
                np.sum(hard_candidates, axis=1) > candidate_limit
            ).tolist():
                columns = np.flatnonzero(hard_candidates[group_index])
                keep = columns[
                    np.lexsort((columns, -scores[group_index, columns]))[
                        :candidate_limit
                    ]
                ]
                hard_candidates[group_index] = False
                hard_candidates[group_index, keep] = True
        hard_groups = has_positive & np.any(hard_candidates, axis=1)
        hard_candidates &= hard_groups[:, None]
        support_count = int(np.sum(hard_groups))
        grid_count = _grid_cell_count(
            xy,
            hard_groups,
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
        )
        if (
            support_count < int(config.min_consistent_groups)
            or grid_count < int(config.min_consistent_grid_cells)
        ):
            continue
        candidate_mode_counts += hard_candidates.astype(np.uint16)
        group_mode_counts += hard_groups.astype(np.uint16)
        selected_mode_hard_masks.append(hard_candidates.copy())
        selected_poses.append(pose.copy())
        selected_modes.append(
            {
                "hypothesis_index": int(hypothesis_index),
                "score": float(pose_scores[hypothesis_index]),
                "translation_error_m_TARGET_ONLY": float(
                    translation[hypothesis_index]
                ),
                "rotation_error_deg_TARGET_ONLY": float(rotation[hypothesis_index]),
                "consistent_group_count": support_count,
                "consistent_grid_cell_count": grid_count,
                "hard_candidate_count": int(np.sum(hard_candidates)),
            }
        )
        if len(selected_modes) >= int(config.max_bad_modes_per_query):
            break

    hard_negative_mask = candidate_mode_counts > 0
    final_candidate_limit = int(config.max_hard_candidates_per_group)
    for group_index in np.flatnonzero(
        np.sum(hard_negative_mask, axis=1) > final_candidate_limit
    ).tolist():
        columns = np.flatnonzero(hard_negative_mask[group_index])
        keep = columns[
            np.lexsort(
                (
                    columns,
                    -scores[group_index, columns],
                    -candidate_mode_counts[group_index, columns].astype(np.int64),
                )
            )[:final_candidate_limit]
        ]
        hard_negative_mask[group_index] = False
        hard_negative_mask[group_index, keep] = True
    group_hard_mask = group_mode_counts > 0
    return {
        "hard_negative_mask": hard_negative_mask,
        "candidate_mode_counts": candidate_mode_counts,
        "group_hard_mask": group_hard_mask,
        "group_mode_counts": group_mode_counts,
        "selected_modes": selected_modes,
        "selected_mode_hard_masks": (
            np.stack(selected_mode_hard_masks, axis=0)
            if selected_mode_hard_masks
            else np.zeros((0, *valid.shape), dtype=bool)
        ),
        "eligible_bad_hypothesis_count": int(len(eligible)),
    }


def _load_hypotheses(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    if not paths:
        raise ValueError("at least one hypothesis artifact is required")
    payloads = []
    manifests = []
    compatibility = None
    for path in paths:
        arrays, metadata = load_inference_artifact_fields(
            path, _HYPOTHESIS_FIELDS_REQUIRED_BY_MINING
        )
        evidence_version = metadata.get("candidate_pose_evidence_version")
        if not isinstance(evidence_version, str) or not evidence_version:
            raise ValueError("hypothesis artifact lacks a candidate pose evidence version")
        current = {
            "candidate_pose_evidence_version": evidence_version,
            "inputs": metadata.get("inputs"),
            "grouped_config": metadata.get("grouped_config"),
        }
        if compatibility is None:
            compatibility = current
        elif current != compatibility:
            raise ValueError("hypothesis artifacts use incompatible inputs/configs")
        payloads.append(arrays)
        manifests.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
                "row_count": int(len(arrays["query_ids"])),
            }
        )
    keys = set(payloads[0]).difference({"metadata_json"})
    if any(set(payload).difference({"metadata_json"}) != keys for payload in payloads):
        raise ValueError("hypothesis artifact schemas differ")
    merged = {
        key: concatenate_hypothesis_shard_field(
            key, [np.asarray(payload[key]) for payload in payloads]
        )
        for key in sorted(keys)
    }
    unique_keys = list(
        zip(
            merged["query_ids"].astype(str).tolist(),
            merged["evaluation_labels"].astype(str).tolist(),
            merged["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(unique_keys) != len(set(unique_keys)):
        raise ValueError("merged hypothesis artifacts contain duplicate rows")
    return merged, dict(compatibility or {}), manifests


def _validate_source_hashes(
    compatibility: Mapping[str, object],
    *,
    candidate_path: Path,
    proposals_path: Path,
    bank_path: Path,
    score_path: Path,
    split_path: Path,
) -> None:
    inputs = compatibility.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("hypothesis artifact has no input manifest")
    expected = {
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "score_artifact_sha256": file_sha256_short(score_path),
        "split_json_sha256": file_sha256_short(split_path),
    }
    mismatches = {
        key: {"expected": value, "actual": inputs.get(key)}
        for key, value in expected.items()
        if inputs.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "hypothesis artifact is stale or misaligned: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PoseConditionedHardNegativeConfig(
        min_bad_translation_m=float(args.min_bad_translation_m),
        max_bad_translation_m=float(args.max_bad_translation_m),
        min_bad_rotation_deg=float(args.min_bad_rotation_deg),
        bad_pose_consistency_px=float(args.bad_pose_consistency_px),
        hard_score_log_margin=float(args.hard_score_log_margin),
        max_hard_candidates_per_group_per_mode=int(
            args.max_hard_candidates_per_group_per_mode
        ),
        max_hard_candidates_per_group=int(args.max_hard_candidates_per_group),
        min_consistent_groups=int(args.min_consistent_groups),
        min_consistent_grid_cells=int(args.min_consistent_grid_cells),
        max_bad_modes_per_query=int(args.max_bad_modes_per_query),
        mode_translation_diversity_m=float(args.mode_translation_diversity_m),
        mode_rotation_diversity_deg=float(args.mode_rotation_diversity_deg),
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
    )
    config.validate()
    hypothesis_paths = tuple(
        Path(value.strip())
        for value in str(args.hypothesis_artifacts).split(",")
        if value.strip()
    )
    hypotheses, compatibility, hypothesis_manifests = _load_hypotheses(
        hypothesis_paths
    )
    candidate_path = Path(args.candidate_artifact)
    proposals_path = Path(args.proposals)
    bank_path = Path(args.projected_landmark_bank)
    score_path = Path(args.score_artifact)
    split_path = Path(args.split_json)
    _validate_source_hashes(
        compatibility,
        candidate_path=candidate_path,
        proposals_path=proposals_path,
        bank_path=bank_path,
        score_path=score_path,
        split_path=split_path,
    )

    candidate = _load_npz(candidate_path)
    proposals = _load_npz(proposals_path)
    scores_payload = _load_npz(score_path)
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate["selected_columns"], dtype=np.int64)
    valid = np.asarray(candidate["valid_edges"], dtype=bool)
    if selected_columns.shape != valid.shape or selected_rows.shape != (len(valid),):
        raise ValueError("candidate artifact arrays are not aligned")
    score_key = str(args.score_key)
    if score_key not in scores_payload:
        raise ValueError(f"score artifact has no key {score_key!r}")
    candidate_scores = np.asarray(scores_payload[score_key], dtype=np.float64)
    if candidate_scores.shape == np.asarray(proposals["candidate_track_ids"]).shape:
        candidate_scores = _compact(
            candidate_scores, selected_rows, selected_columns
        )
    if candidate_scores.shape != valid.shape:
        raise ValueError("candidate score matrix is not aligned with the candidate store")

    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)

    split = json.loads(split_path.read_text())
    split_names = tuple(
        value.strip() for value in str(args.split_names).split(",") if value.strip()
    )
    if not split_names or any(name not in split for name in split_names):
        raise ValueError("requested split names are absent from the split manifest")
    allowed_query_ids = {
        str(query_id) for name in split_names for query_id in split[name]
    }
    artifact_split_names = np.asarray(hypotheses["split_names"]).astype(str)
    artifact_query_ids = np.asarray(hypotheses["query_ids"]).astype(str)
    artifact_labels = np.asarray(hypotheses["evaluation_labels"]).astype(str)
    labels = sorted(
        set(
            artifact_labels[
                np.isin(artifact_split_names, np.asarray(split_names))
            ].tolist()
        )
    )
    evaluation_label = str(args.evaluation_label)
    if evaluation_label:
        if evaluation_label not in labels:
            raise ValueError("requested evaluation label is absent from hypotheses")
    elif len(labels) == 1:
        evaluation_label = labels[0]
    else:
        raise ValueError("multiple evaluation labels require --evaluation_label")
    artifact_query_set = set(
        artifact_query_ids[
            np.isin(artifact_split_names, np.asarray(split_names))
            & (artifact_labels == evaluation_label)
        ].tolist()
    )
    if not artifact_query_set.issubset(allowed_query_ids):
        raise ValueError("hypothesis artifact contains queries outside requested splits")
    if not bool(args.allow_partial_query_coverage) and artifact_query_set != allowed_query_ids:
        missing = sorted(allowed_query_ids.difference(artifact_query_set))
        raise ValueError(f"hypothesis artifacts do not cover requested split: {missing[:5]}")

    query_ids, selected_candidate_fields = select_candidate_rows_for_allowed_queries(
        query_ids=query_ids,
        allowed_query_ids=allowed_query_ids,
        fields={
            "selected_rows": selected_rows,
            "selected_columns": selected_columns,
            "valid": valid,
            "candidate_scores": candidate_scores,
        },
    )
    selected_rows = np.asarray(selected_candidate_fields["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(
        selected_candidate_fields["selected_columns"], dtype=np.int64
    )
    valid = np.asarray(selected_candidate_fields["valid"], dtype=bool)
    candidate_scores = np.asarray(
        selected_candidate_fields["candidate_scores"], dtype=np.float64
    )
    if np.any(~np.isfinite(candidate_scores[valid])):
        raise ValueError("valid candidate scores must be finite")
    query_xy = np.asarray(proposals["xy"], dtype=np.float64)[selected_rows]
    compact_tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    candidate_metadata = json.loads(str(candidate["metadata_json"].item()))
    positive_threshold = float(candidate_metadata.get("positive_threshold_px", 2.0))
    positive = positive_mask_from_posthoc_gt_residuals(
        proposal_residuals=proposals["candidate_gt_residuals_px"],
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        valid_edges=valid,
        positive_threshold_px=positive_threshold,
    )

    landmark_index, _ = load_landmark_index_npz(bank_path)
    canonical_rows = canonical_rows_for_track_candidates(
        compact_tracks, landmark_index.track_ids
    )
    valid &= canonical_rows >= 0
    candidate_xyz = np.zeros((*valid.shape, 3), dtype=np.float64)
    candidate_xyz[valid] = landmark_index.xyz[canonical_rows[valid]]

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    hard_negative_mask = np.zeros(valid.shape, dtype=bool)
    candidate_mode_counts = np.zeros(valid.shape, dtype=np.uint16)
    group_hard_mask = np.zeros((len(valid),), dtype=bool)
    group_mode_counts = np.zeros((len(valid),), dtype=np.uint16)
    mode_ids = np.full(
        (len(valid), int(config.max_bad_modes_per_query)), -1, dtype=np.int32
    )
    mode_candidate_masks = np.zeros(
        (
            len(valid),
            int(config.max_bad_modes_per_query),
            valid.shape[1],
        ),
        dtype=bool,
    )
    mode_query_ids: list[str] = []
    mode_support_group_counts: list[int] = []
    per_query_rows = []
    for query_id in sorted(artifact_query_set):
        image = images_by_name.get(query_id)
        if image is None or int(image.camera_id) not in cameras:
            raise ValueError(f"query {query_id!r} is absent from the COLMAP model")
        camera = cameras[int(image.camera_id)]
        group_rows = np.flatnonzero(query_ids == query_id)
        if len(group_rows) == 0:
            raise ValueError(f"query {query_id!r} is absent from candidate rows")
        hypothesis_rows = np.flatnonzero(
            (artifact_query_ids == query_id)
            & (artifact_labels == evaluation_label)
            & np.isin(artifact_split_names, np.asarray(split_names))
        )
        poses = np.asarray(hypotheses["poses_w2c"], dtype=np.float64)[hypothesis_rows]
        pose_scores = np.asarray(
            hypotheses["verification_log_likelihood_means"], dtype=np.float64
        )[hypothesis_rows]
        gt_pose = _gt_pose_w2c(image)
        errors = [pnp_pose_error(pose, gt_pose) for pose in poses]
        result = mine_query_system_error_modes(
            query_xy=query_xy[group_rows],
            candidate_xyz=candidate_xyz[group_rows],
            candidate_scores=candidate_scores[group_rows],
            valid_mask=valid[group_rows],
            positive_mask=positive[group_rows],
            hypothesis_poses_w2c=poses,
            hypothesis_scores=pose_scores,
            hypothesis_translation_errors_m=np.asarray(
                [error.translation_m for error in errors], dtype=np.float64
            ),
            hypothesis_rotation_errors_deg=np.asarray(
                [error.rotation_deg for error in errors], dtype=np.float64
            ),
            camera=camera,
            config=config,
        )
        hard_negative_mask[group_rows] = result["hard_negative_mask"]
        candidate_mode_counts[group_rows] = result["candidate_mode_counts"]
        group_hard_mask[group_rows] = result["group_hard_mask"]
        group_mode_counts[group_rows] = result["group_mode_counts"]
        selected_mode_masks = np.asarray(
            result["selected_mode_hard_masks"], dtype=bool
        )
        if selected_mode_masks.shape[0] > int(config.max_bad_modes_per_query):
            raise RuntimeError("selected hard modes exceed the configured maximum")
        for local_mode_index, local_mask in enumerate(selected_mode_masks):
            if local_mask.shape != valid[group_rows].shape:
                raise RuntimeError("selected hard-mode mask is not query aligned")
            participating = np.any(local_mask, axis=1)
            if not np.any(participating):
                raise RuntimeError("selected hard mode has no participating group")
            global_mode_id = len(mode_query_ids)
            mode_ids[group_rows[participating], local_mode_index] = global_mode_id
            mode_candidate_masks[
                group_rows, local_mode_index
            ] = local_mask
            mode_query_ids.append(query_id)
            mode_support_group_counts.append(int(np.sum(participating)))
        per_query_rows.append(
            {
                "query_id": query_id,
                "split_name": next(
                    name for name in split_names if query_id in set(split[name])
                ),
                "hypothesis_count": int(len(hypothesis_rows)),
                "eligible_bad_hypothesis_count": int(
                    result["eligible_bad_hypothesis_count"]
                ),
                "selected_bad_mode_count": int(len(result["selected_modes"])),
                "hard_group_count": int(np.sum(result["group_hard_mask"])),
                "hard_candidate_count": int(np.sum(result["hard_negative_mask"])),
                "selected_modes_TARGET_ONLY": result["selected_modes"],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structured = bool(args.preserve_mode_membership)
    output_path = output_dir / (
        "pose_conditioned_system_hard_modes_v2.npz"
        if structured
        else "pose_conditioned_system_hard_negatives_v1.npz"
    )
    score_order = np.argsort(
        -np.where(valid, candidate_scores, -np.inf), axis=1, kind="stable"
    )
    score_ranks = np.empty_like(score_order)
    score_ranks[
        np.arange(len(score_order))[:, None], score_order
    ] = np.arange(score_order.shape[1], dtype=np.int64)[None, :] + 1
    hard_ranks = score_ranks[hard_negative_mask]
    metadata = {
        "format": STRUCTURED_ARTIFACT_FORMAT if structured else ARTIFACT_FORMAT,
        "training_only_target_artifact": True,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "ground_truth_joined_after_generation": True,
        "candidate_pose_evidence_version": compatibility.get(
            "candidate_pose_evidence_version"
        ),
        "evaluation_label": evaluation_label,
        "split_names": list(split_names),
        "config": asdict(config),
        "inputs": {
            "hypothesis_artifacts": hypothesis_manifests,
            "candidate_artifact": str(candidate_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "score_artifact": str(score_path),
            "score_artifact_sha256": file_sha256_short(score_path),
            "score_key": score_key,
            "split_json": str(split_path),
            "split_json_sha256": file_sha256_short(split_path),
            "colmap_cameras_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_sha256": file_sha256_short(model_dir / "images.bin"),
        },
    }
    output_arrays = {
        "selected_rows": selected_rows,
        "selected_columns": selected_columns,
        "query_ids": query_ids,
        "valid_edges": valid,
        "positive_mask_TARGET_ONLY": positive,
        "hard_negative_mask_TARGET_ONLY": hard_negative_mask,
        "candidate_bad_mode_counts_TARGET_ONLY": candidate_mode_counts,
        "group_hard_mask_TARGET_ONLY": group_hard_mask,
        "group_bad_mode_counts_TARGET_ONLY": group_mode_counts,
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True), dtype=np.str_
        ),
    }
    if structured:
        output_arrays.update(
            {
                "hard_mode_ids_TARGET_ONLY": mode_ids,
                "hard_mode_candidate_mask_TARGET_ONLY": mode_candidate_masks,
                "hard_mode_query_ids_TARGET_ONLY": np.asarray(mode_query_ids),
                "hard_mode_support_group_counts_TARGET_ONLY": np.asarray(
                    mode_support_group_counts, dtype=np.uint16
                ),
            }
        )
    np.savez_compressed(output_path, **output_arrays)
    summary = {
        "stage": "pose_conditioned_system_hard_negative_mining",
        "protocol": {
            "hypothesis_generation_is_target_free": True,
            "target_join_is_training_only": True,
            "validation_or_test_training_leakage": bool(
                any(name != "train" for name in split_names)
            ),
        },
        "config": asdict(config),
        "metrics_TARGET_ONLY": {
            "query_count": int(len(per_query_rows)),
            "query_with_bad_mode_count": int(
                sum(row["selected_bad_mode_count"] > 0 for row in per_query_rows)
            ),
            "selected_bad_mode_count": int(
                sum(row["selected_bad_mode_count"] for row in per_query_rows)
            ),
            "structured_mode_membership_exported": structured,
            "hard_group_count": int(np.sum(group_hard_mask)),
            "hard_candidate_count": int(np.sum(hard_negative_mask)),
            "multi_mode_hard_group_count": int(np.sum(group_mode_counts >= 2)),
            "mean_hard_candidates_per_hard_group": float(
                np.sum(hard_negative_mask) / max(np.sum(group_hard_mask), 1)
            ),
            "hard_candidate_rank1_count": int(np.sum(hard_ranks == 1)),
            "hard_candidate_rank2_to_5_count": int(
                np.sum((hard_ranks >= 2) & (hard_ranks <= 5))
            ),
            "hard_candidate_rank6_to_20_count": int(
                np.sum((hard_ranks >= 6) & (hard_ranks <= 20))
            ),
        },
        "outputs": {
            "artifact": str(output_path),
            "artifact_sha256": file_sha256_short(output_path),
            "per_query": str(output_dir / "per_query_TARGET_ONLY.json"),
        },
        "inputs": metadata["inputs"],
    }
    (output_dir / "per_query_TARGET_ONLY.json").write_text(
        json.dumps(per_query_rows, indent=2, sort_keys=True)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
