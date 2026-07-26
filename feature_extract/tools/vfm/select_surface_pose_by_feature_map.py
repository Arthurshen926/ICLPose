"""Select map-only pose candidates with independent 2DGS feature evidence.

The selector opens only the query RGB.  Candidate generators may use different
feature-bearing 2DGS anchor indexes, but the selector evaluates every pose
against one fixed feature-aligned 2DGS map.  It never loads a mapping image,
SfM point/track, RADIO intermediate feature, or pairwise image matcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.special import logsumexp

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", default="")
    parser.add_argument("--query_image_root", default="")
    parser.add_argument("--query_camera_manifest", default="")
    parser.add_argument("--anchors", default="")
    parser.add_argument("--local_descriptor_bank", default="")
    parser.add_argument(
        "--candidate_results",
        nargs="*",
        default=(),
        help="Named result inputs in NAME=PATH form.",
    )
    parser.add_argument("--maximum_anchors", type=int, default=384)
    parser.add_argument("--maximum_anchors_per_cell", type=int, default=8)
    parser.add_argument("--grid_rows", type=int, default=8)
    parser.add_argument("--grid_cols", type=int, default=8)
    parser.add_argument("--minimum_normal_cosine", type=float, default=0.15)
    parser.add_argument("--minimum_similarity", type=float, default=0.65)
    parser.add_argument(
        "--fixed_evidence_union",
        action="store_true",
        help=(
            "Freeze the union of candidate-visible anchor IDs and score every "
            "pose with that identical denominator and explicit null mass."
        ),
    )
    parser.add_argument(
        "--signed_normal_visibility",
        action="store_true",
        help="Require normals oriented toward mapping observation cameras.",
    )
    parser.add_argument(
        "--view_conditioning_candidate",
        default="",
        help=(
            "Candidate whose query-level RADIO feature-mode scores define the "
            "fixed support-view mixture."
        ),
    )
    parser.add_argument("--similarity_scale", type=float, default=0.08)
    parser.add_argument("--anchor_null_probability", type=float, default=0.5)
    parser.add_argument("--view_prior_temperature", type=float, default=0.08)
    parser.add_argument(
        "--geometry_view_temperature", type=float, default=0.10
    )
    parser.add_argument(
        "--zbuffer_occlusion",
        action="store_true",
        help="Reject anchors hidden by the oriented 2DGS anchor depth field.",
    )
    parser.add_argument("--occlusion_cell_px", type=int, default=4)
    parser.add_argument(
        "--occlusion_depth_tolerance_m", type=float, default=0.15
    )
    parser.add_argument(
        "--safety_candidate",
        default="",
        help=(
            "Optional candidate retained unless another pose exceeds its "
            "fixed-map log likelihood by minimum_score_margin."
        ),
    )
    parser.add_argument(
        "--minimum_score_margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum_safety_disagreement_translation_m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum_safety_disagreement_rotation_deg",
        type=float,
        default=0.0,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Optional positive prefix length for bounded validation runs.",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--merge_results_jsonl",
        nargs="*",
        default=(),
        help="Merge already selected disjoint result shards without inference.",
    )
    return parser.parse_args(argv)


def _read_results(path: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = dict(json.loads(line))
        image_id = str(row["image_id"])
        if image_id in output:
            raise ValueError(f"duplicate result image ID: {image_id}")
        output[image_id] = row
    return output


def _named_result_paths(
    values: Sequence[str],
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = str(value).partition("=")
        if not separator or not name or not path:
            raise ValueError(
                "candidate_results entries must use NAME=PATH"
            )
        if name in output:
            raise ValueError(f"duplicate candidate name: {name}")
        output[name] = Path(path)
    if len(output) < 2:
        raise ValueError("at least two candidate result sets are required")
    return output


def _descriptor_rows_by_anchor(
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(
            descriptor_bank.anchor_ids.tolist()
        )
    }
    offsets = np.full((len(anchors), 2), -1, dtype=np.int64)
    descriptor_counts = np.zeros((len(anchors),), dtype=np.int64)
    for anchor_row, anchor_id in enumerate(anchors.anchor_ids.tolist()):
        bank_row = bank_row_by_id.get(int(anchor_id))
        if bank_row is None:
            continue
        start = int(descriptor_bank.descriptor_offsets[bank_row])
        end = int(descriptor_bank.descriptor_offsets[bank_row + 1])
        if end <= start:
            continue
        offsets[anchor_row] = (start, end)
        descriptor_counts[anchor_row] = end - start
    valid = descriptor_counts > 0
    quality = (
        np.maximum(anchors.quality_scores, 1e-8)
        * np.log1p(descriptor_counts.astype(np.float32))
    )
    # The feature-aligned anchor layer is detector-repeatable by construction.
    quality *= np.where(
        anchors.anchor_ids >= 1_000_000_000,
        2.0,
        1.0,
    )
    return offsets, descriptor_counts, quality.astype(np.float32) * valid


def _visible_anchor_rows(
    *,
    pose_w2c: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_quality: np.ndarray,
    camera,
    maximum_anchors: int,
    maximum_anchors_per_cell: int,
    grid_rows: int,
    grid_cols: int,
    minimum_normal_cosine: float,
    signed_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz = anchors.xyz
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        xyz,
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    view_direction = camera_center[None, :] - xyz
    view_direction /= np.maximum(
        np.linalg.norm(view_direction, axis=1, keepdims=True),
        1e-12,
    )
    signed_normal_cosine = np.sum(
        anchors.normals * view_direction,
        axis=1,
    )
    normal_cosine = (
        signed_normal_cosine
        if bool(signed_normals)
        else np.abs(signed_normal_cosine)
    )
    valid = (
        (descriptor_quality > 0.0)
        & (camera_xyz[:, 2] > 1e-6)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(camera.width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(camera.height - 1))
        & (normal_cosine >= float(minimum_normal_cosine))
    )
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    score = descriptor_quality[candidates] * (
        0.5 + 0.5 * normal_cosine[candidates]
    )
    order = candidates[
        np.lexsort((candidates, -score))
    ]
    cell_counts = np.zeros(
        (int(grid_rows), int(grid_cols)),
        dtype=np.int64,
    )
    selected: list[int] = []
    for anchor_row in order.tolist():
        col = int(
            np.clip(
                np.floor(
                    projected[anchor_row, 0]
                    / max(float(camera.width), 1.0)
                    * int(grid_cols)
                ),
                0,
                int(grid_cols) - 1,
            )
        )
        row = int(
            np.clip(
                np.floor(
                    projected[anchor_row, 1]
                    / max(float(camera.height), 1.0)
                    * int(grid_rows)
                ),
                0,
                int(grid_rows) - 1,
            )
        )
        if cell_counts[row, col] >= int(maximum_anchors_per_cell):
            continue
        selected.append(anchor_row)
        cell_counts[row, col] += 1
        if len(selected) >= int(maximum_anchors):
            break
    rows = np.asarray(selected, dtype=np.int64)
    return (
        rows,
        projected[rows].astype(np.float32),
        normal_cosine[rows].astype(np.float32),
    )


def _pose_feature_evidence(
    *,
    pose_w2c: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    descriptor_offsets: np.ndarray,
    descriptor_quality: np.ndarray,
    alike: AlikeDenseObservationExtractor,
    image_path: Path,
    camera,
    args: argparse.Namespace,
) -> dict[str, object]:
    rows, xy, normal_cosine = _visible_anchor_rows(
        pose_w2c=pose_w2c,
        anchors=anchors,
        descriptor_quality=descriptor_quality,
        camera=camera,
        maximum_anchors=int(args.maximum_anchors),
        maximum_anchors_per_cell=int(
            args.maximum_anchors_per_cell
        ),
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
        minimum_normal_cosine=float(args.minimum_normal_cosine),
        signed_normals=False,
    )
    if len(rows) < 8:
        return {
            "log_likelihood": None,
            "supported_anchor_count": 0,
            "evaluated_anchor_count": int(len(rows)),
            "supported_cell_count": 0,
            "median_similarity": None,
        }
    query_descriptors, detector_scores, _image_hash = alike.sample_points(
        image_path,
        xy,
        image_width=int(camera.width),
        image_height=int(camera.height),
    )
    similarities = np.zeros((len(rows),), dtype=np.float32)
    for query_row, anchor_row in enumerate(rows.tolist()):
        start, end = descriptor_offsets[int(anchor_row)]
        similarities[query_row] = float(
            np.max(
                descriptor_bank.descriptors[int(start) : int(end)]
                @ query_descriptors[query_row]
            )
        )
    logits = (
        similarities.astype(np.float64)
        - float(args.minimum_similarity)
    ) / 0.08
    log_match = -np.logaddexp(0.0, -logits)
    repeatability = np.tanh(
        np.log1p(
            np.maximum(
                np.asarray(detector_scores, dtype=np.float64),
                0.0,
            )
            * 1000.0
        )
        / 3.0
    )
    per_anchor = (
        log_match
        + 0.15 * repeatability
        + 0.05 * normal_cosine.astype(np.float64)
    )
    supported = (
        (similarities >= float(args.minimum_similarity))
        & (repeatability >= 0.05)
    )
    supported_xy = xy[supported]
    if len(supported_xy):
        supported_cells = {
            (
                int(
                    np.clip(
                        point[1]
                        / max(float(camera.height), 1.0)
                        * int(args.grid_rows),
                        0,
                        int(args.grid_rows) - 1,
                    )
                ),
                int(
                    np.clip(
                        point[0]
                        / max(float(camera.width), 1.0)
                        * int(args.grid_cols),
                        0,
                        int(args.grid_cols) - 1,
                    )
                ),
            )
            for point in supported_xy
        }
    else:
        supported_cells = set()
    # Coverage is independent of candidate generation and prevents a compact
    # repeated facade patch from defeating a spatially supported pose.
    coverage_bonus = 0.02 * len(supported_cells)
    return {
        "log_likelihood": float(np.mean(per_anchor) + coverage_bonus),
        "supported_anchor_count": int(np.sum(supported)),
        "evaluated_anchor_count": int(len(rows)),
        "supported_cell_count": int(len(supported_cells)),
        "median_similarity": float(np.median(similarities)),
    }


def _query_feature_mode_scores(
    candidate: dict[str, object],
) -> dict[str, float]:
    diagnostics = dict(candidate.get("diagnostics") or {})
    output: dict[str, float] = {}
    for value in diagnostics.get("feature_mode_diagnostics", []):
        row = dict(value)
        mode_id = str(row.get("mode_id") or "")
        score = row.get("vfm_score")
        if not mode_id or score is None or not np.isfinite(float(score)):
            continue
        output[mode_id] = max(
            output.get(mode_id, -float("inf")),
            float(score),
        )
    return output


def _fixed_anchor_union(
    *,
    candidate_rows: dict[str, dict[str, object]],
    anchors: StableSurfaceAnchorMap,
    descriptor_quality: np.ndarray,
    camera,
    args: argparse.Namespace,
) -> np.ndarray:
    rows: set[int] = set()
    for candidate in candidate_rows.values():
        if (
            not bool(candidate.get("success", False))
            or candidate.get("pose_w2c") is None
        ):
            continue
        selected, _xy, _normal = _visible_anchor_rows(
            pose_w2c=np.asarray(candidate["pose_w2c"], dtype=np.float64),
            anchors=anchors,
            descriptor_quality=descriptor_quality,
            camera=camera,
            maximum_anchors=int(args.maximum_anchors),
            maximum_anchors_per_cell=int(
                args.maximum_anchors_per_cell
            ),
            grid_rows=int(args.grid_rows),
            grid_cols=int(args.grid_cols),
            minimum_normal_cosine=float(args.minimum_normal_cosine),
            signed_normals=bool(args.signed_normal_visibility),
        )
        rows.update(int(value) for value in selected.tolist())
    return np.asarray(sorted(rows), dtype=np.int64)


def _fixed_pose_feature_evidence(
    *,
    pose_w2c: np.ndarray,
    fixed_anchor_rows: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    descriptor_offsets: np.ndarray,
    alike: AlikeDenseObservationExtractor,
    image_path: Path,
    camera,
    view_mode_scores: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, object]:
    fixed_rows = np.asarray(fixed_anchor_rows, dtype=np.int64).reshape(-1)
    if len(fixed_rows) == 0:
        return {
            "log_likelihood": None,
            "supported_anchor_count": 0,
            "evaluated_anchor_count": 0,
            "visible_anchor_count": 0,
            "supported_cell_count": 0,
            "median_similarity": None,
            "fixed_denominator": True,
        }
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz = anchors.xyz[fixed_rows]
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        xyz,
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    view_direction = camera_center[None, :] - xyz
    view_direction /= np.maximum(
        np.linalg.norm(view_direction, axis=1, keepdims=True),
        1e-12,
    )
    signed_normal = np.sum(
        anchors.normals[fixed_rows] * view_direction,
        axis=1,
    )
    normal_visibility = (
        signed_normal
        if bool(args.signed_normal_visibility)
        else np.abs(signed_normal)
    )
    visible = (
        (camera_xyz[:, 2] > 1e-6)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(camera.width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(camera.height - 1))
        & (
            normal_visibility
            >= float(args.minimum_normal_cosine)
        )
    )
    occlusion_visible = np.ones((len(fixed_rows),), dtype=bool)
    if bool(args.zbuffer_occlusion):
        all_camera_xyz = (
            anchors.xyz @ pose[:3, :3].T + pose[:3, 3]
        )
        all_projected, _jacobian = cv2.projectPoints(
            anchors.xyz,
            rvec,
            pose[:3, 3],
            matrix,
            distortion,
        )
        all_projected = all_projected.reshape(-1, 2)
        cell_size = max(int(args.occlusion_cell_px), 1)
        cell_cols = int(np.ceil(float(camera.width) / cell_size))
        cell_rows = int(np.ceil(float(camera.height) / cell_size))
        all_in_frame = (
            (all_camera_xyz[:, 2] > 1e-6)
            & (all_projected[:, 0] >= 0.0)
            & (all_projected[:, 0] <= float(camera.width - 1))
            & (all_projected[:, 1] >= 0.0)
            & (all_projected[:, 1] <= float(camera.height - 1))
        )
        zbuffer = np.full(
            (cell_rows * cell_cols,),
            float("inf"),
            dtype=np.float64,
        )
        all_cells = (
            np.floor(all_projected[all_in_frame, 1] / cell_size)
            .astype(np.int64)
            * cell_cols
            + np.floor(all_projected[all_in_frame, 0] / cell_size)
            .astype(np.int64)
        )
        np.minimum.at(
            zbuffer,
            all_cells,
            all_camera_xyz[all_in_frame, 2],
        )
        fixed_cells = (
            np.clip(
                np.floor(projected[:, 1] / cell_size),
                0,
                cell_rows - 1,
            ).astype(np.int64)
            * cell_cols
            + np.clip(
                np.floor(projected[:, 0] / cell_size),
                0,
                cell_cols - 1,
            ).astype(np.int64)
        )
        occlusion_visible = (
            camera_xyz[:, 2]
            <= zbuffer[fixed_cells]
            + float(args.occlusion_depth_tolerance_m)
        )
        visible &= occlusion_visible
    visible_local_rows = np.flatnonzero(visible)
    anchor_llr = np.zeros((len(fixed_rows),), dtype=np.float64)
    similarities = np.full(
        (len(fixed_rows),),
        np.nan,
        dtype=np.float64,
    )
    supported = np.zeros((len(fixed_rows),), dtype=bool)
    if len(visible_local_rows):
        query_descriptors, _detector_scores, _image_hash = (
            alike.sample_points(
                image_path,
                projected[visible_local_rows],
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
        )
        for query_row, local_row in enumerate(
            visible_local_rows.tolist()
        ):
            anchor_row = int(fixed_rows[local_row])
            start, end = descriptor_offsets[anchor_row]
            descriptors = descriptor_bank.descriptors[
                int(start) : int(end)
            ]
            edge_similarity = (
                descriptors @ query_descriptors[query_row]
            ).astype(np.float64)
            similarities[local_row] = float(
                np.max(edge_similarity)
            )
            support_ids = descriptor_bank.support_image_ids[
                int(start) : int(end)
            ]
            support_quality = np.maximum(
                descriptor_bank.descriptor_quality[
                    int(start) : int(end)
                ].astype(np.float64),
                1e-8,
            )
            prior_logits = np.log(support_quality)
            support_directions = (
                descriptor_bank.support_view_directions[
                    int(start) : int(end)
                ].astype(np.float64)
            )
            valid_direction = (
                np.linalg.norm(support_directions, axis=1) > 0.5
            )
            if np.any(valid_direction):
                geometry_cosine = (
                    support_directions @ view_direction[local_row]
                )
                prior_logits += np.where(
                    valid_direction,
                    geometry_cosine
                    / float(args.geometry_view_temperature),
                    np.min(geometry_cosine[valid_direction])
                    / float(args.geometry_view_temperature),
                )
            if view_mode_scores and any(
                value in view_mode_scores for value in support_ids
            ):
                prior_logits += np.asarray(
                    [
                        (
                            view_mode_scores[value]
                            if value in view_mode_scores
                            else min(view_mode_scores.values())
                            - float(args.view_prior_temperature)
                        )
                        / float(args.view_prior_temperature)
                        for value in support_ids
                    ],
                    dtype=np.float64,
                )
            log_weights = prior_logits - logsumexp(prior_logits)
            edge_llr = np.clip(
                (
                    edge_similarity
                    - float(args.minimum_similarity)
                )
                / float(args.similarity_scale),
                -12.0,
                12.0,
            )
            view_mixture_llr = float(
                logsumexp(log_weights + edge_llr)
            )
            null_probability = float(args.anchor_null_probability)
            anchor_llr[local_row] = float(
                np.logaddexp(
                    np.log(null_probability),
                    np.log1p(-null_probability)
                    + view_mixture_llr,
                )
            )
            supported[local_row] = bool(
                view_mixture_llr > 0.0
            )
    score = float(
        np.sum(np.clip(anchor_llr, -2.0, 4.0))
        / max(len(fixed_rows), 1)
    )
    supported_xy = projected[supported]
    supported_cells = {
        (
            int(
                np.clip(
                    point[1]
                    / max(float(camera.height), 1.0)
                    * int(args.grid_rows),
                    0,
                    int(args.grid_rows) - 1,
                )
            ),
            int(
                np.clip(
                    point[0]
                    / max(float(camera.width), 1.0)
                    * int(args.grid_cols),
                    0,
                    int(args.grid_cols) - 1,
                )
            ),
        )
        for point in supported_xy
    }
    finite_similarity = similarities[np.isfinite(similarities)]
    return {
        "log_likelihood": score,
        "supported_anchor_count": int(np.sum(supported)),
        "evaluated_anchor_count": int(len(fixed_rows)),
        "visible_anchor_count": int(np.sum(visible)),
        "occlusion_rejected_anchor_count": int(
            np.sum(~occlusion_visible)
            if bool(args.zbuffer_occlusion)
            else 0
        ),
        "supported_cell_count": int(len(supported_cells)),
        "median_similarity": (
            float(np.median(finite_similarity))
            if len(finite_similarity)
            else None
        ),
        "fixed_denominator": True,
        "explicit_null_probability": float(
            args.anchor_null_probability
        ),
        "view_conditioned_mixture": bool(
            view_mode_scores
            or np.any(
                np.linalg.norm(
                    descriptor_bank.support_view_directions,
                    axis=1,
                )
                > 0.5
            )
        ),
        "zbuffer_occlusion": bool(args.zbuffer_occlusion),
        "uses_max_over_views": False,
    }


def _pose_disagreement(
    first_w2c: np.ndarray,
    second_w2c: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first_w2c, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second_w2c, dtype=np.float64).reshape(4, 4)
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    translation = float(np.linalg.norm(first_center - second_center))
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float(
        np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    )
    return translation, float(np.degrees(np.arccos(cosine)))


def _select_candidate(
    *,
    candidate_rows: dict[str, dict[str, object]],
    evidence_by_name: dict[str, dict[str, object]],
    safety_candidate: str,
    minimum_score_margin: float,
    maximum_disagreement_translation_m: float,
    maximum_disagreement_rotation_deg: float,
) -> tuple[str, str | None, tuple[float, float] | None]:
    selected_name = max(
        candidate_rows,
        key=lambda name: (
            float(
                evidence_by_name[name]["log_likelihood"]
                if evidence_by_name[name]["log_likelihood"] is not None
                else -float("inf")
            ),
            int(evidence_by_name[name]["supported_cell_count"]),
            int(evidence_by_name[name]["supported_anchor_count"]),
            name,
        ),
    )
    safety_reason = None
    if safety_candidate and selected_name != safety_candidate:
        selected_score = evidence_by_name[selected_name][
            "log_likelihood"
        ]
        safety_score = evidence_by_name[safety_candidate][
            "log_likelihood"
        ]
        if (
            selected_score is None
            or safety_score is None
            or float(selected_score)
            < float(safety_score) + float(minimum_score_margin)
        ):
            return (
                safety_candidate,
                "insufficient_feature_score_margin",
                None,
            )
    candidate_disagreement = None
    if safety_candidate and selected_name != safety_candidate:
        selected_pose = candidate_rows[selected_name].get("pose_w2c")
        safety_pose = candidate_rows[safety_candidate].get("pose_w2c")
        if selected_pose is not None and safety_pose is not None:
            candidate_disagreement = _pose_disagreement(
                np.asarray(selected_pose, dtype=np.float64),
                np.asarray(safety_pose, dtype=np.float64),
            )
            if (
                (
                    float(maximum_disagreement_translation_m) > 0.0
                    and candidate_disagreement[0]
                    > float(maximum_disagreement_translation_m)
                )
                or (
                    float(maximum_disagreement_rotation_deg) > 0.0
                    and candidate_disagreement[1]
                    > float(maximum_disagreement_rotation_deg)
                )
            ):
                return (
                    safety_candidate,
                    "candidate_pose_disagreement_exceeds_limit",
                    candidate_disagreement,
                )
    return selected_name, safety_reason, candidate_disagreement


def _merge_result_shards(
    paths: Sequence[str],
    output_path: Path,
) -> dict[str, object]:
    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for value in paths:
        rows = _read_results(Path(value))
        overlap = set(merged) & set(rows)
        if overlap:
            raise ValueError(
                f"selected result shards overlap: {sorted(overlap)[:3]}"
            )
        order.extend(rows)
        merged.update(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for image_id in order:
            handle.write(
                json.dumps(merged[image_id], sort_keys=True) + "\n"
            )
    return {
        "stage": "select_surface_pose_by_feature_map_merge",
        "query_count": len(order),
        "input_shards": [str(value) for value in paths],
        "output_jsonl": str(output_path),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    if args.merge_results_jsonl:
        summary = _merge_result_shards(
            args.merge_results_jsonl,
            output_path,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    required = {
        "query_manifest": args.query_manifest,
        "query_image_root": args.query_image_root,
        "query_camera_manifest": args.query_camera_manifest,
        "anchors": args.anchors,
        "local_descriptor_bank": args.local_descriptor_bank,
    }
    missing = [name for name, value in required.items() if not str(value)]
    if missing:
        raise ValueError(f"missing selector inputs: {missing}")
    if (
        int(args.maximum_anchors) <= 0
        or int(args.maximum_anchors_per_cell) <= 0
        or int(args.grid_rows) <= 0
        or int(args.grid_cols) <= 0
    ):
        raise ValueError("selector anchor/grid limits must be positive")
    candidate_paths = _named_result_paths(args.candidate_results)
    if (
        str(args.safety_candidate)
        and str(args.safety_candidate) not in candidate_paths
    ):
        raise ValueError("safety_candidate is absent from candidate results")
    if (
        str(args.view_conditioning_candidate)
        and str(args.view_conditioning_candidate) not in candidate_paths
    ):
        raise ValueError(
            "view_conditioning_candidate is absent from candidate results"
        )
    if float(args.minimum_score_margin) < 0.0:
        raise ValueError("minimum_score_margin must be non-negative")
    if (
        float(args.maximum_safety_disagreement_translation_m) < 0.0
        or float(args.maximum_safety_disagreement_rotation_deg) < 0.0
    ):
        raise ValueError("safety disagreement limits must be non-negative")
    if (
        (
            float(args.maximum_safety_disagreement_translation_m) > 0.0
            or float(args.maximum_safety_disagreement_rotation_deg) > 0.0
        )
        and not str(args.safety_candidate)
    ):
        raise ValueError(
            "safety disagreement limits require safety_candidate"
        )
    if float(args.similarity_scale) <= 0.0:
        raise ValueError("similarity_scale must be positive")
    if not 0.0 < float(args.anchor_null_probability) < 1.0:
        raise ValueError("anchor_null_probability must be in (0, 1)")
    if float(args.view_prior_temperature) <= 0.0:
        raise ValueError("view_prior_temperature must be positive")
    if float(args.geometry_view_temperature) <= 0.0:
        raise ValueError("geometry_view_temperature must be positive")
    if (
        int(args.occlusion_cell_px) <= 0
        or float(args.occlusion_depth_tolerance_m) < 0.0
    ):
        raise ValueError("occlusion cell/tolerance is invalid")
    candidates = {
        name: _read_results(path)
        for name, path in candidate_paths.items()
    }
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    image_ids = [record.image_id for record in manifest.records]
    if int(args.max_queries) < 0:
        raise ValueError("max_queries must be non-negative")
    if int(args.max_queries) > 0:
        image_ids = image_ids[: int(args.max_queries)]
    for name, rows in candidates.items():
        missing_ids = set(image_ids) - set(rows)
        if missing_ids:
            raise ValueError(
                f"candidate {name} misses query IDs: "
                f"{sorted(missing_ids)[:3]}"
            )
    cameras, intrinsic_audit = _load_query_camera_manifest(
        Path(args.query_camera_manifest)
    )
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    if (
        bool(args.signed_normal_visibility)
        and "signed_toward" not in str(
            dict(anchors.metadata or {}).get(
                "normal_orientation", ""
            )
        )
    ):
        raise ValueError(
            "signed normal visibility requires offline-oriented anchor normals"
        )
    descriptor_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.local_descriptor_bank)
    )
    (
        descriptor_offsets,
        _descriptor_counts,
        descriptor_quality,
    ) = _descriptor_rows_by_anchor(anchors, descriptor_bank)
    alike = AlikeDenseObservationExtractor(
        device=str(args.device),
        matcha_repo=Path(args.matcha_repo),
        model_name=str(args.alike_model_name),
    )
    query_root = Path(args.query_image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_counts = {name: 0 for name in candidates}
    with output_path.open("w") as handle:
        for query_index, image_id in enumerate(image_ids):
            if image_id not in cameras:
                raise ValueError(f"missing query calibration: {image_id}")
            candidate_rows = {
                name: rows[image_id]
                for name, rows in candidates.items()
            }
            fixed_anchor_rows = (
                _fixed_anchor_union(
                    candidate_rows=candidate_rows,
                    anchors=anchors,
                    descriptor_quality=descriptor_quality,
                    camera=cameras[image_id],
                    args=args,
                )
                if bool(args.fixed_evidence_union)
                else np.zeros((0,), dtype=np.int64)
            )
            view_mode_scores = (
                _query_feature_mode_scores(
                    candidate_rows[
                        str(args.view_conditioning_candidate)
                    ]
                )
                if str(args.view_conditioning_candidate)
                else {}
            )
            evidence_by_name: dict[str, dict[str, object]] = {}
            for name, rows in candidates.items():
                candidate = rows[image_id]
                if (
                    not bool(candidate.get("success", False))
                    or candidate.get("pose_w2c") is None
                ):
                    evidence_by_name[name] = {
                        "log_likelihood": None,
                        "supported_anchor_count": 0,
                        "evaluated_anchor_count": 0,
                        "supported_cell_count": 0,
                        "median_similarity": None,
                    }
                    continue
                if bool(args.fixed_evidence_union):
                    evidence_by_name[name] = (
                        _fixed_pose_feature_evidence(
                            pose_w2c=np.asarray(
                                candidate["pose_w2c"],
                                dtype=np.float64,
                            ),
                            fixed_anchor_rows=fixed_anchor_rows,
                            anchors=anchors,
                            descriptor_bank=descriptor_bank,
                            descriptor_offsets=descriptor_offsets,
                            alike=alike,
                            image_path=query_root / image_id,
                            camera=cameras[image_id],
                            view_mode_scores=view_mode_scores,
                            args=args,
                        )
                    )
                else:
                    evidence_by_name[name] = _pose_feature_evidence(
                        pose_w2c=np.asarray(
                            candidate["pose_w2c"],
                            dtype=np.float64,
                        ),
                        anchors=anchors,
                        descriptor_bank=descriptor_bank,
                        descriptor_offsets=descriptor_offsets,
                        descriptor_quality=descriptor_quality,
                        alike=alike,
                        image_path=query_root / image_id,
                        camera=cameras[image_id],
                        args=args,
                    )
            (
                selected_name,
                safety_reason,
                candidate_disagreement,
            ) = _select_candidate(
                candidate_rows={
                    name: rows[image_id]
                    for name, rows in candidates.items()
                },
                evidence_by_name=evidence_by_name,
                safety_candidate=str(args.safety_candidate),
                minimum_score_margin=float(args.minimum_score_margin),
                maximum_disagreement_translation_m=float(
                    args.maximum_safety_disagreement_translation_m
                ),
                maximum_disagreement_rotation_deg=float(
                    args.maximum_safety_disagreement_rotation_deg
                ),
            )
            row = dict(candidates[selected_name][image_id])
            diagnostics = dict(row.get("diagnostics") or {})
            diagnostics["feature_map_pose_selector"] = {
                "selected_candidate": selected_name,
                "candidate_evidence": evidence_by_name,
                "fixed_evidence_anchor_count": int(
                    len(fixed_anchor_rows)
                ),
                "view_mode_prior_count": int(len(view_mode_scores)),
                "safety_candidate": (
                    str(args.safety_candidate)
                    if str(args.safety_candidate)
                    else None
                ),
                "minimum_score_margin": float(
                    args.minimum_score_margin
                ),
                "candidate_disagreement_translation_m": (
                    float(candidate_disagreement[0])
                    if candidate_disagreement is not None
                    else None
                ),
                "candidate_disagreement_rotation_deg": (
                    float(candidate_disagreement[1])
                    if candidate_disagreement is not None
                    else None
                ),
                "safety_reason": safety_reason,
                "uses_ground_truth": False,
                "uses_mapping_rgb": False,
                "uses_pairwise_image_matching": False,
            }
            row["diagnostics"] = diagnostics
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            selected_counts[selected_name] += 1
            print(
                json.dumps(
                    {
                        "query": query_index + 1,
                        "query_count": len(image_ids),
                        "image_id": image_id,
                        "selected_candidate": selected_name,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "stage": "select_surface_pose_by_feature_map",
        "query_count": len(image_ids),
        "selected_candidate_counts": selected_counts,
        "artifacts": {
            "anchors": {
                "path": str(args.anchors),
                "sha256": file_sha256_short(Path(args.anchors)),
            },
            "local_descriptor_bank": {
                "path": str(args.local_descriptor_bank),
                "sha256": file_sha256_short(
                    Path(args.local_descriptor_bank)
                ),
            },
            "candidate_results": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256_short(path),
                }
                for name, path in candidate_paths.items()
            },
        },
        "config": {
            "maximum_anchors": int(args.maximum_anchors),
            "maximum_anchors_per_cell": int(
                args.maximum_anchors_per_cell
            ),
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "minimum_normal_cosine": float(
                args.minimum_normal_cosine
            ),
            "minimum_similarity": float(args.minimum_similarity),
            "fixed_evidence_union": bool(args.fixed_evidence_union),
            "signed_normal_visibility": bool(
                args.signed_normal_visibility
            ),
            "view_conditioning_candidate": (
                str(args.view_conditioning_candidate)
                if str(args.view_conditioning_candidate)
                else None
            ),
            "similarity_scale": float(args.similarity_scale),
            "anchor_null_probability": float(
                args.anchor_null_probability
            ),
            "view_prior_temperature": float(
                args.view_prior_temperature
            ),
            "geometry_view_temperature": float(
                args.geometry_view_temperature
            ),
            "zbuffer_occlusion": bool(args.zbuffer_occlusion),
            "occlusion_cell_px": int(args.occlusion_cell_px),
            "occlusion_depth_tolerance_m": float(
                args.occlusion_depth_tolerance_m
            ),
            "safety_candidate": (
                str(args.safety_candidate)
                if str(args.safety_candidate)
                else None
            ),
            "minimum_score_margin": float(
                args.minimum_score_margin
            ),
            "maximum_safety_disagreement_translation_m": float(
                args.maximum_safety_disagreement_translation_m
            ),
            "maximum_safety_disagreement_rotation_deg": float(
                args.maximum_safety_disagreement_rotation_deg
            ),
            "intrinsic_audit": intrinsic_audit,
        },
        "production_contract": {
            "gate_uses_ground_truth": False,
            "map_representation": (
                "fixed_feature_aligned_2dgs_anchor_map"
            ),
            "stores_mapping_rgb": False,
            "uses_loftr": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_pairwise_image_matching": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
        "output_jsonl": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
