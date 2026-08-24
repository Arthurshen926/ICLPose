"""Train a small full-layout candidate ranker and evaluate fixed query routes.

The ranker sees only candidate-conditioned RADIO/canonical evidence grids.  It
does not receive pose matrices, pose errors, image coordinates, keypoints,
correspondences, or PnP outputs.  Candidate zero is a diagnostic rendered GT
anchor used for supervision; all reported selection metrics exclude it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _metrics,
    _spearman,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FULLTOKEN_POSE_RANKER_SEMANTICS,
    FULLTOKEN_POSE_RANKING_CHANNELS,
    FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    FullTokenCandidatePoseRanker,
    continuous_seed_domain_recall_metrics,
    distinct_pose_basin_recall_metrics,
    listwise_pose_ranking_loss,
    load_fulltoken_candidate_pose_ranker,
    multiscale_pose_ranking_loss,
    natural_hard_negative_pose_ranking_loss,
    trajectory_monotonic_pose_ranking_loss,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


REPORT_SCHEMA = "goal_maplet_fulltoken_candidate_pose_ranker_train_route_holdout_v1"
LOCAL_LABEL_SCHEMA = "goal_maplet_local_pose_supervision_dataset_v1"


def _route(image_id: object) -> str:
    return str(image_id).split("/", 1)[0]


def _mask_for_routes(image_ids: np.ndarray, routes: tuple[str, ...]) -> np.ndarray:
    allowed = frozenset(str(value) for value in routes)
    return np.asarray([_route(value) in allowed for value in image_ids.tolist()], dtype=bool)


def _load_feature_artifact(
    path: Path,
    manifest_path: Path,
    *,
    dataset_path: Path,
    dataset_content_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("artifact_type") != "goal_maplet_compact_fulltoken_pose_ranking_features_v1":
        raise ValueError("not a compact full-token ranking feature artifact")
    if manifest.get("feature_semantics") != FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS:
        raise ValueError("compact full-token feature semantics differ")
    if manifest.get("feature_channels") != list(FULLTOKEN_POSE_RANKING_CHANNELS):
        raise ValueError("compact full-token feature channels differ")
    resolved = Path(path).resolve()
    if Path(str(manifest.get("feature_file", ""))).resolve() != resolved:
        raise ValueError("compact full-token feature path differs")
    if manifest.get("feature_file_sha256") != file_sha256(resolved):
        raise ValueError("compact full-token feature bytes differ")
    if (
        manifest.get("dataset_file_sha256") != file_sha256(dataset_path)
        or manifest.get("dataset_content_sha256") != dataset_content_sha256
    ):
        raise ValueError("compact full-token feature dataset lineage differs")
    features = np.load(resolved, mmap_mode="r", allow_pickle=False)
    if (
        features.dtype != np.float16
        or list(features.shape) != manifest.get("feature_shape")
        or features.shape[2:] != (len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64)
    ):
        raise ValueError("compact full-token feature array differs")
    return features, manifest


def _load_local_supervision_dataset(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("local pose supervision dataset lacks metadata")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {
            name: np.asarray(data[name]) for name in data.files if name != "metadata_json"
        }
    artifact_type = metadata.get("artifact_type")
    controlled_trajectory = (
        artifact_type == "goal_maplet_controlled_pose_training_inventory_v1"
        and str(metadata.get("supervision_semantics", "")).startswith(
            "frozen_natural_seed_to_gt_"
        )
        and metadata.get("uses_gt_for_training_candidate_generation") is True
        and metadata.get("natural_candidate_pools_frozen_before_target_pose_opened") is True
        and metadata.get("deployment_candidate_pool") is False
        and metadata.get("production_eligible") is False
        and all(metadata.get(key) is False for key in (
            "uses_alike", "uses_point_correspondences", "uses_pnp",
            "uses_absolute_pose_regression",
        ))
    )
    if artifact_type != LOCAL_LABEL_SCHEMA and not controlled_trajectory:
        raise ValueError("not a local or controlled trajectory supervision dataset")
    if metadata.get("content_sha256") != arrays_sha256(arrays):
        raise ValueError("local pose supervision content differs")
    required = (
        "image_ids", "candidate_poses_w2c",
        "translation_m", "rotation_deg", "candidate_valid",
    )
    if artifact_type == LOCAL_LABEL_SCHEMA:
        required = required[:1] + ("source_query_rows",) + required[1:]
    if any(name not in arrays for name in required):
        raise ValueError("local pose supervision arrays are incomplete")
    shape = np.asarray(arrays["candidate_valid"]).shape
    if (
        len(shape) != 2
        or arrays["candidate_poses_w2c"].shape != shape + (4, 4)
        or arrays["translation_m"].shape != shape
        or arrays["rotation_deg"].shape != shape
        or not np.all(arrays["candidate_valid"])
        or not np.all(np.abs(arrays["translation_m"][:, 0]) <= 1.0e-7)
        or not np.all(np.abs(arrays["rotation_deg"][:, 0]) <= 2.0e-6)
    ):
        raise ValueError("local pose supervision geometry differs")
    return arrays, metadata


def _score_queries(
    model: FullTokenCandidatePoseRanker,
    features: np.ndarray,
    query_rows: np.ndarray,
    *,
    batch_queries: int,
    device: torch.device,
) -> np.ndarray:
    result = np.full((int(query_rows.size), int(features.shape[1])), -1.0e9, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, int(query_rows.size), int(batch_queries)):
            rows = query_rows[begin:begin + int(batch_queries)]
            value = torch.as_tensor(
                np.asarray(features[rows], dtype=np.float32), device=device,
            )
            batch, candidates = value.shape[:2]
            result[begin:begin + batch] = model(value.flatten(0, 1)).reshape(
                batch, candidates,
            ).cpu().numpy()
    return result


def _basin_recall_metrics(
    score: np.ndarray,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, object]:
    return distinct_pose_basin_recall_metrics(
        score,
        arrays["candidate_poses_w2c"][rows],
        arrays["translation_m"][rows],
        arrays["rotation_deg"][rows],
        arrays["candidate_valid"][rows],
    )


def _fuse_with_retrieval_order(
    appearance_score: np.ndarray,
    valid: np.ndarray,
    weight: float,
) -> np.ndarray:
    score = np.asarray(appearance_score, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    result = score.copy()
    ordinal = -np.arange(score.shape[1], dtype=np.float32)
    for query in range(score.shape[0]):
        rows = np.flatnonzero(mask[query])
        nonanchor = rows[rows != 0]
        content = score[query, nonanchor]
        content = (content - np.mean(content)) / max(float(np.std(content)), 1.0e-6)
        prior = ordinal[nonanchor]
        prior = (prior - np.mean(prior)) / max(float(np.std(prior)), 1.0e-6)
        result[query, nonanchor] = content + float(weight) * prior
    return result


def _tiered_energy_landscape_metrics(
    score_rows: np.ndarray,
    translation_rows: np.ndarray,
    rotation_rows: np.ndarray,
    valid_rows: np.ndarray,
) -> dict[str, object]:
    """Measure ranking quality at fixed, interpretable SE(3) scales.

    The aggregate metric mixes sub-metre curvature with 8 m / 45 degree
    discrimination.  That can hide the exact failure mode needed to decide
    whether a ranker is usable for local refinement, medium-range basin
    selection, or neither.  Every tier keeps the rendered GT anchor and only
    compares it with valid perturbations inside the stated closed domain.
    """

    tiers = (
        ("strict_0_5m_5deg", 0.5, 5.0),
        ("local_1m_10deg", 1.0, 10.0),
        ("medium_2m_20deg", 2.0, 20.0),
        ("wide_8m_45deg", 8.0, 45.0),
    )
    result: dict[str, object] = {}
    for name, maximum_translation_m, maximum_rotation_deg in tiers:
        rows = []
        for score, translation, rotation, valid in zip(
            score_rows, translation_rows, rotation_rows, valid_rows,
        ):
            eligible = np.flatnonzero(
                np.asarray(valid, dtype=bool)
                & (np.asarray(translation) <= maximum_translation_m + 1.0e-8)
                & (np.asarray(rotation) <= maximum_rotation_deg + 1.0e-8)
            )
            if 0 not in eligible:
                raise ValueError("tiered landscape metrics require a valid GT anchor")
            nonanchor = eligible[eligible != 0]
            if nonanchor.size == 0:
                rows.append({
                    "nonanchor_count": 0,
                    "gt_anchor_rank": 1,
                    "gt_anchor_margin": None,
                    "score_error_spearman": None,
                    "pairwise_correct": 0,
                    "pairwise_total": 0,
                })
                continue
            joint = np.maximum(
                np.asarray(translation, dtype=np.float64) / max(maximum_translation_m, 1.0e-8),
                np.asarray(rotation, dtype=np.float64) / max(maximum_rotation_deg, 1.0e-8),
            )
            ordered_pairs = 0
            correct_pairs = 0
            for left in eligible.tolist():
                for right in eligible.tolist():
                    if joint[left] + 1.0e-8 < joint[right]:
                        ordered_pairs += 1
                        correct_pairs += int(score[left] > score[right])
            rows.append({
                "nonanchor_count": int(nonanchor.size),
                "gt_anchor_rank": int(1 + np.sum(score[eligible] > score[0])),
                "gt_anchor_margin": float(score[0] - np.max(score[nonanchor])),
                "score_error_spearman": _spearman(score[eligible], -joint[eligible]),
                "pairwise_correct": correct_pairs,
                "pairwise_total": ordered_pairs,
            })
        informative = [row for row in rows if row["nonanchor_count"] > 0]
        result[name] = {
            "maximum_translation_m": maximum_translation_m,
            "maximum_rotation_deg": maximum_rotation_deg,
            "query_count": len(rows),
            "informative_query_count": len(informative),
            "mean_nonanchor_count": float(np.mean([row["nonanchor_count"] for row in rows])),
            "gt_anchor_top1_rate": (
                None if not informative else
                float(np.mean([row["gt_anchor_rank"] == 1 for row in informative]))
            ),
            "mean_gt_anchor_margin": (
                None if not informative else
                float(np.mean([row["gt_anchor_margin"] for row in informative]))
            ),
            "mean_score_error_spearman": (
                None if not informative else
                float(np.mean([row["score_error_spearman"] for row in informative]))
            ),
            "pairwise_order_accuracy": (
                None if not informative else float(
                    sum(row["pairwise_correct"] for row in informative)
                    / max(sum(row["pairwise_total"] for row in informative), 1)
                )
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--train_routes", nargs="+", default=["seq13"])
    parser.add_argument("--validation_routes", nargs="+", default=["seq3"])
    parser.add_argument(
        "--held_routes", nargs="*", default=["seq5"],
        help="Optional untouched route set; pass the flag without values for a train/validation-only diagnostic.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_queries", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--translation_supervision_scale_m", type=float, default=8.0)
    parser.add_argument("--rotation_supervision_scale_deg", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local_supervision", action="store_true")
    parser.add_argument(
        "--multiscale_supervision", action="store_true",
        help=(
            "equalize strict/local/medium/wide energy supervision and add a "
            "hardest-perturbation GT-anchor margin"
        ),
    )
    parser.add_argument(
        "--natural_hard_negative_weight", type=float, default=0.0,
        help=(
            "extra loss weight for the GT anchor versus the frozen natural "
            "retrieval rows recorded by a merged supervision artifact"
        ),
    )
    parser.add_argument(
        "--trajectory_monotonic_weight", type=float, default=0.0,
        help="extra monotonic score loss for frozen natural seed-to-GT training paths",
    )
    parser.add_argument("--initial_model", default="")
    args = parser.parse_args()
    model_path, report_path = Path(args.output_model), Path(args.output_report)
    if model_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite full-token pose ranker experiment")
    if int(args.epochs) <= 0 or int(args.batch_queries) <= 0:
        raise ValueError("training schedule must be positive")
    if float(args.learning_rate) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("optimizer constants are invalid")
    if not 0.0 <= float(args.natural_hard_negative_weight) < float("inf"):
        raise ValueError("natural hard-negative weight is invalid")
    if not 0.0 <= float(args.trajectory_monotonic_weight) < float("inf"):
        raise ValueError("trajectory monotonic weight is invalid")

    dataset_path = Path(args.dataset)
    if bool(args.local_supervision):
        arrays, metadata = _load_local_supervision_dataset(dataset_path)
    else:
        arrays, metadata = load_pose_candidate_dataset(
            dataset_path, require_rendered_targets=False,
        )
    features, feature_manifest = _load_feature_artifact(
        Path(args.features), Path(args.feature_manifest),
        dataset_path=dataset_path,
        dataset_content_sha256=str(metadata["content_sha256"]),
    )
    if features.shape[:2] != arrays["candidate_valid"].shape:
        raise ValueError("full-token features differ from candidate inventory")
    natural_start = metadata.get("natural_candidate_start_row")
    if float(args.natural_hard_negative_weight) > 0.0:
        if (
            not bool(args.local_supervision) or not bool(args.multiscale_supervision)
            or not isinstance(natural_start, int)
            or not 1 < natural_start < features.shape[1]
            or metadata.get("natural_candidate_pools_frozen_before_target_pose_opened") is not True
        ):
            raise ValueError("natural hard-negative loss requires a frozen merged inventory")
    if float(args.trajectory_monotonic_weight) > 0.0:
        if (
            not bool(args.local_supervision) or not bool(args.multiscale_supervision)
            or "trajectory_seed_candidate_index" not in arrays
            or "trajectory_alpha" not in arrays
            or np.asarray(arrays["trajectory_seed_candidate_index"]).shape
            != arrays["candidate_valid"].shape
            or np.asarray(arrays["trajectory_alpha"]).shape
            != arrays["candidate_valid"].shape
            or metadata.get("natural_candidate_pools_frozen_before_target_pose_opened") is not True
        ):
            raise ValueError("trajectory loss requires frozen controlled path supervision")
    route_sets = tuple(
        tuple(str(value) for value in routes)
        for routes in (args.train_routes, args.validation_routes, args.held_routes)
    )
    if len(set(route_sets[0]) | set(route_sets[1]) | set(route_sets[2])) != sum(map(len, route_sets)):
        raise ValueError("train, validation, and held routes must be disjoint")
    masks = tuple(_mask_for_routes(arrays["image_ids"], routes) for routes in route_sets)
    if (
        not np.any(masks[0]) or not np.any(masks[1])
        or (route_sets[2] and not np.any(masks[2]))
        or not np.all(masks[0] | masks[1] | masks[2])
    ):
        raise ValueError("fixed route split must cover the complete feature inventory")

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(args.device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FullTokenCandidatePoseRanker().to(device)
    initial_model_metadata = None
    initial_model_file_sha256 = None
    if str(args.initial_model):
        initial_model, initial_model_metadata = load_fulltoken_candidate_pose_ranker(
            Path(args.initial_model), device=device,
        )
        model.load_state_dict(initial_model.state_dict(), strict=True)
        initial_model_file_sha256 = file_sha256(Path(args.initial_model))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay),
    )
    train_rows = np.flatnonzero(masks[0])
    generator = np.random.default_rng(seed)
    epoch_rows = []
    for epoch in range(int(args.epochs)):
        order = train_rows.copy()
        generator.shuffle(order)
        model.train()
        losses = []
        for begin in range(0, int(order.size), int(args.batch_queries)):
            rows = order[begin:begin + int(args.batch_queries)]
            value = torch.as_tensor(np.asarray(features[rows], dtype=np.float32), device=device)
            batch, candidates = value.shape[:2]
            score = model(value.flatten(0, 1)).reshape(batch, candidates)
            translation_labels = torch.as_tensor(arrays["translation_m"][rows], device=device)
            rotation_labels = torch.as_tensor(arrays["rotation_deg"][rows], device=device)
            valid_labels = torch.as_tensor(arrays["candidate_valid"][rows], device=device)
            loss = (
                multiscale_pose_ranking_loss(
                    score, translation_labels, rotation_labels, valid_labels,
                )
                if bool(args.multiscale_supervision)
                else listwise_pose_ranking_loss(
                    score, translation_labels, rotation_labels, valid_labels,
                    translation_scale_m=float(args.translation_supervision_scale_m),
                    rotation_scale_deg=float(args.rotation_supervision_scale_deg),
                )
            )
            if float(args.natural_hard_negative_weight) > 0.0:
                loss = loss + float(args.natural_hard_negative_weight) * (
                    natural_hard_negative_pose_ranking_loss(
                        score, translation_labels, rotation_labels, valid_labels,
                        natural_candidate_start_row=int(natural_start),
                    )
                )
            if float(args.trajectory_monotonic_weight) > 0.0:
                loss = loss + float(args.trajectory_monotonic_weight) * (
                    trajectory_monotonic_pose_ranking_loss(
                        score,
                        torch.as_tensor(
                            arrays["trajectory_seed_candidate_index"][rows], device=device,
                        ),
                        torch.as_tensor(arrays["trajectory_alpha"][rows], device=device),
                        valid_labels,
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_rows.append({"epoch": epoch + 1, "mean_training_loss": float(np.mean(losses))})

    state_arrays = {
        name: value.detach().cpu().numpy() for name, value in model.state_dict().items()
    }
    model_content_sha256 = arrays_sha256(state_arrays)
    model_payload = {
        "artifact_type": "goal_maplet_fulltoken_candidate_pose_ranker_v1",
        "model_semantics": FULLTOKEN_POSE_RANKER_SEMANTICS,
        "model_content_sha256": model_content_sha256,
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "dataset_content_sha256": metadata["content_sha256"],
        "initial_model_file_sha256": initial_model_file_sha256,
        "initial_model_content_sha256": (
            None if initial_model_metadata is None
            else initial_model_metadata["model_content_sha256"]
        ),
        "local_supervision": bool(args.local_supervision),
        "multiscale_supervision": bool(args.multiscale_supervision),
        "natural_hard_negative_weight": float(args.natural_hard_negative_weight),
        "trajectory_monotonic_weight": float(args.trajectory_monotonic_weight),
        "natural_candidate_start_row": natural_start,
        "local_pose_supervision_semantics": metadata.get("supervision_semantics"),
        "state_dict": model.state_dict(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)

    all_scores = np.full(arrays["candidate_valid"].shape, -1.0e9, dtype=np.float32)
    for mask in masks:
        rows = np.flatnonzero(mask)
        score = _score_queries(
            model, features, rows,
            batch_queries=int(args.batch_queries), device=device,
        )
        all_scores[rows] = score

    fusion_candidates = (
        (0.0,) if bool(args.local_supervision)
        else (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    )
    fusion_rows = []
    for weight in fusion_candidates:
        fused = _fuse_with_retrieval_order(
            all_scores[masks[0]], arrays["candidate_valid"][masks[0]], weight,
        )
        acquisition = (
            {"strict_domain_acquisition_at_1": 0.0,
             "strict_domain_acquisition_at_4": 0.0,
             "loose_domain_acquisition_at_1": 0.0}
            if bool(args.local_supervision)
            else continuous_seed_domain_recall_metrics(
                fused,
                arrays["candidate_poses_w2c"][masks[0]],
                arrays["candidate_valid"][masks[0]],
                translation_half_extent_m=float(args.translation_supervision_scale_m),
                rotation_radius_deg=float(args.rotation_supervision_scale_deg),
            )
        )
        fusion_rows.append({"weight": weight, "training_domain_acquisition": acquisition})
    selected_fusion = max(
        fusion_rows,
        key=lambda row: (
            row["training_domain_acquisition"]["strict_domain_acquisition_at_1"],
            row["training_domain_acquisition"]["strict_domain_acquisition_at_4"],
            row["training_domain_acquisition"]["loose_domain_acquisition_at_1"],
            -row["weight"],
        ),
    )
    fusion_weight = float(selected_fusion["weight"])
    model_payload["retrieval_prior_fusion_weight"] = fusion_weight
    model_payload["retrieval_prior_fusion_selected_on_training_routes_only"] = True
    torch.save(model_payload, model_path)
    all_fused_scores = _fuse_with_retrieval_order(
        all_scores, arrays["candidate_valid"], fusion_weight,
    )

    split_reports = {}
    for name, mask, routes in zip(("train", "validation", "held"), masks, route_sets):
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            split_reports[name] = {
                "routes": list(routes),
                "query_count": 0,
                "metrics": None,
                "distinct_basin_recall": None,
                "continuous_domain_acquisition": None,
                "retrieval_prior_fused_metrics": None,
                "retrieval_prior_fused_continuous_domain_acquisition": None,
                "retrieval_order_distinct_basin_recall": None,
                "retrieval_order_continuous_domain_acquisition": None,
                "tiered_energy_landscape": None,
            }
            continue
        score = all_scores[rows]
        fused_score = all_fused_scores[rows]
        split_reports[name] = {
            "routes": list(routes),
            "metrics": _metrics(
                score,
                arrays["translation_m"][rows],
                arrays["rotation_deg"][rows],
                arrays["candidate_valid"][rows],
                arrays["image_ids"][rows],
            ),
            "distinct_basin_recall": _basin_recall_metrics(score, arrays, rows),
            "continuous_domain_acquisition": None if bool(args.local_supervision) else continuous_seed_domain_recall_metrics(
                score,
                arrays["candidate_poses_w2c"][rows],
                arrays["candidate_valid"][rows],
                translation_half_extent_m=float(args.translation_supervision_scale_m),
                rotation_radius_deg=float(args.rotation_supervision_scale_deg),
            ),
            "retrieval_prior_fused_metrics": _metrics(
                fused_score,
                arrays["translation_m"][rows],
                arrays["rotation_deg"][rows],
                arrays["candidate_valid"][rows],
                arrays["image_ids"][rows],
            ),
            "retrieval_prior_fused_continuous_domain_acquisition": None if bool(args.local_supervision) else (
                continuous_seed_domain_recall_metrics(
                    fused_score,
                    arrays["candidate_poses_w2c"][rows],
                    arrays["candidate_valid"][rows],
                    translation_half_extent_m=float(args.translation_supervision_scale_m),
                    rotation_radius_deg=float(args.rotation_supervision_scale_deg),
                )
            ),
            "retrieval_order_distinct_basin_recall": _basin_recall_metrics(
                np.broadcast_to(
                    -np.arange(features.shape[1], dtype=np.float32)[None],
                    score.shape,
                ),
                arrays,
                rows,
            ),
            "retrieval_order_continuous_domain_acquisition": None if bool(args.local_supervision) else (
                continuous_seed_domain_recall_metrics(
                    np.broadcast_to(
                        -np.arange(features.shape[1], dtype=np.float32)[None],
                        score.shape,
                    ),
                    arrays["candidate_poses_w2c"][rows],
                    arrays["candidate_valid"][rows],
                    translation_half_extent_m=float(args.translation_supervision_scale_m),
                    rotation_radius_deg=float(args.rotation_supervision_scale_deg),
                )
            ),
            "tiered_energy_landscape": _tiered_energy_landscape_metrics(
                score,
                arrays["translation_m"][rows],
                arrays["rotation_deg"][rows],
                arrays["candidate_valid"][rows],
            ),
        }
    report = {
        "artifact_type": REPORT_SCHEMA,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_manifest_file_sha256": file_sha256(Path(args.feature_manifest)),
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "feature_semantics": FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
        "model_semantics": FULLTOKEN_POSE_RANKER_SEMANTICS,
        "model_file_sha256": file_sha256(model_path),
        "model_content_sha256": model_content_sha256,
        "initial_model_file_sha256": initial_model_file_sha256,
        "initial_model_content_sha256": (
            None if initial_model_metadata is None
            else initial_model_metadata["model_content_sha256"]
        ),
        "seed": seed,
        "epochs": int(args.epochs),
        "batch_queries": int(args.batch_queries),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "translation_supervision_scale_m": float(args.translation_supervision_scale_m),
        "rotation_supervision_scale_deg": float(args.rotation_supervision_scale_deg),
        "local_supervision": bool(args.local_supervision),
        "multiscale_supervision": bool(args.multiscale_supervision),
        "natural_hard_negative_weight": float(args.natural_hard_negative_weight),
        "trajectory_monotonic_weight": float(args.trajectory_monotonic_weight),
        "natural_candidate_start_row": natural_start,
        "local_pose_supervision_semantics": metadata.get("supervision_semantics"),
        "epoch_rows": epoch_rows,
        "retrieval_prior_fusion_candidates": list(fusion_candidates),
        "retrieval_prior_fusion_training_rows": fusion_rows,
        "retrieval_prior_fusion_selected_weight": fusion_weight,
        "retrieval_prior_fusion_selected_on_training_routes_only": True,
        "split_reports": split_reports,
        "all_metrics": _metrics(
            all_scores,
            arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"], arrays["image_ids"],
        ),
        "all_distinct_basin_recall": _basin_recall_metrics(
            all_scores, arrays, np.arange(arrays["image_ids"].size, dtype=np.int64),
        ),
        "all_tiered_energy_landscape": _tiered_energy_landscape_metrics(
            all_scores,
            arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"],
        ),
        "all_continuous_domain_acquisition": None if bool(args.local_supervision) else continuous_seed_domain_recall_metrics(
            all_scores,
            arrays["candidate_poses_w2c"], arrays["candidate_valid"],
            translation_half_extent_m=float(args.translation_supervision_scale_m),
            rotation_radius_deg=float(args.rotation_supervision_scale_deg),
        ),
        "all_retrieval_prior_fused_metrics": _metrics(
            all_fused_scores,
            arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"], arrays["image_ids"],
        ),
        "all_retrieval_prior_fused_continuous_domain_acquisition": None if bool(args.local_supervision) else (
            continuous_seed_domain_recall_metrics(
                all_fused_scores,
                arrays["candidate_poses_w2c"], arrays["candidate_valid"],
                translation_half_extent_m=float(args.translation_supervision_scale_m),
                rotation_radius_deg=float(args.rotation_supervision_scale_deg),
            )
        ),
        "candidate_pose_values_are_model_inputs": False,
        "candidate_pose_errors_are_model_inputs": False,
        "held_labels_used_by_optimizer": False,
        "validation_labels_used_for_epoch_or_hyperparameter_selection": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "standard_test_was_historically_exposed": bool(
            metadata.get("contains_standard_test_queries", True)
        ),
        "claim": "fixed_route_diagnostic_candidate_ranking_not_final_localization",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_model": str(model_path),
        "output_report": str(report_path),
        "split_metrics": {
            name: (
                None if payload["metrics"] is None else
                {key: value for key, value in payload["metrics"].items() if key != "rows"}
            )
            for name, payload in split_reports.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
