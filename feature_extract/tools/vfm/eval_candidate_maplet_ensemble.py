"""Evaluate a validation-selected ensemble of candidate-maplet checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.probe_detector_maplet_geometry import _identity_metrics, _pose_gate
from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _assignment_identity_gate,
    _candidate_data_manifest,
    _compact_values,
    _load_frozen_global_baseline_policy,
    _predict_edges,
    _rescue_policy_report,
    _validate_frozen_baseline_pose,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.candidate_maplet_data import CandidateMapletEpisodeStore
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
)
from feature_extract.vfm.localization.local_assignment_linear import (
    resolve_rescue_policy_scores,
    selective_switch_scores,
)
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
)


def _float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return parsed


def _global_assignment_scores(
    *,
    candidates: UniqueTrackCandidateSet,
    scores: np.ndarray,
    query_ids: np.ndarray,
    valid_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one candidate per token and one token per physical track."""

    return global_assignment_score_matrix(
        candidates.track_ids,
        scores,
        query_ids,
        valid_mask=valid_edges & np.isfinite(scores),
        dustbin_score=None,
    )


def _baseline_assignment_scores(
    *,
    candidates: UniqueTrackCandidateSet,
    scores: np.ndarray,
    query_ids: np.ndarray,
    valid_edges: np.ndarray,
    formal_global_assignment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if formal_global_assignment:
        return _global_assignment_scores(
            candidates=candidates,
            scores=scores,
            query_ids=query_ids,
            valid_edges=valid_edges,
        )
    selected = np.argmax(np.where(valid_edges, scores, -np.inf), axis=1).astype(
        np.int64
    )
    return scores, selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--query_context_detector_cache", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--radio_intermediate_cache", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument(
        "--query_split_manifest",
        default="",
        help="original leakage-audited query split manifest used by the checkpoints",
    )
    parser.add_argument(
        "--global_assignment_baseline_summary",
        default="",
        help="frozen validation baseline summary bound into the checkpoint manifest",
    )
    parser.add_argument(
        "--prediction_splits",
        default="validation,test",
        help="comma-separated split names whose per-edge predictions are exported",
    )
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument(
        "--development_cross_block_audit",
        action="store_true",
        help="replay every validation trial on the reused development test block",
    )
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--support_view_count", type=int, default=2)
    parser.add_argument("--static_feature_count", type=int, default=17)
    parser.add_argument("--query_radius_px", type=float, default=96.0)
    parser.add_argument("--max_query_nodes", type=int, default=48)
    parser.add_argument("--max_support_tracks", type=int, default=33)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--assignment_threshold_px", type=float, default=5.0)
    parser.add_argument(
        "--geometry_validity_thresholds_px",
        type=_float_list,
        default=(1.0, 2.0, 5.0),
    )
    parser.add_argument(
        "--switch_margin_thresholds",
        type=_float_list,
        default=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7),
    )
    parser.add_argument(
        "--validation_strategies",
        default=(
            "set_candidate_probability,"
            "set_assignment_geomean,"
            "set_candidate_probability_dustbin_p50,"
            "set_candidate_probability_dustbin_p70,"
            "set_candidate_probability_dustbin_p90,"
            "set_assignment_geomean_dustbin_p50,"
            "set_assignment_geomean_dustbin_p70,"
            "set_assignment_geomean_dustbin_p90"
        ),
    )
    parser.add_argument(
        "--rescue_action_margin_thresholds",
        type=_float_list,
        default=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5),
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--export_checkpoint_latents",
        action="store_true",
        help="export per-support-view candidate embeddings without averaging checkpoint spaces",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.development_cross_block_audit) and args.evaluation_role != "development":
        raise ValueError("cross-block audit is only valid for development evaluation")
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = tuple(
        Path(value.strip()) for value in str(args.checkpoints).split(",") if value.strip()
    )
    devices = tuple(value.strip() for value in str(args.devices).split(",") if value.strip())
    if not checkpoint_paths or not devices:
        raise ValueError("evaluation requires at least one checkpoint and one device")
    split = json.loads(Path(args.split_json).read_text())
    for name in ("train", "validation", "test"):
        if name not in split or not isinstance(split[name], list):
            raise ValueError("split JSON is missing a query block")
    prediction_split_names = tuple(
        value.strip()
        for value in str(args.prediction_splits).split(",")
        if value.strip()
    )
    if not prediction_split_names or set(prediction_split_names) - {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("prediction_splits must use train, validation, and/or test")
    if not {"validation", "test"}.issubset(prediction_split_names):
        raise ValueError("evaluation export requires validation and test predictions")
    store = CandidateMapletEpisodeStore(
        proposals=Path(args.proposals),
        detector_query_cache=Path(args.detector_query_cache),
        query_context_detector_cache=Path(args.query_context_detector_cache),
        support_feature_cache=Path(args.support_feature_cache),
        support_geometry_index=Path(args.support_geometry_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        maplet_support_index=Path(args.maplet_support_index),
        feature_artifact=Path(args.feature_artifact),
        colmap_model_dir=Path(args.colmap_model_dir),
        radio_intermediate_cache=(
            None
            if args.radio_intermediate_cache is None
            else Path(args.radio_intermediate_cache)
        ),
        query_radius_px=float(args.query_radius_px),
        max_query_nodes=int(args.max_query_nodes),
        max_support_tracks=int(args.max_support_tracks),
        positive_threshold_px=float(args.positive_threshold_px),
        assignment_threshold_px=float(args.assignment_threshold_px),
        support_view_count=int(args.support_view_count),
        static_feature_count=int(args.static_feature_count),
    )
    data_manifest = _candidate_data_manifest(args, store)
    frozen_global_baseline_source = (
        None
        if not str(args.global_assignment_baseline_summary)
        else _load_frozen_global_baseline_policy(
            Path(args.global_assignment_baseline_summary),
            args=args,
            data_manifest=data_manifest,
        )
    )
    selected_query_ids = store.query_ids[store.selected_rows]
    split_row_masks = {
        name: np.isin(selected_query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("train", "validation", "test")
    }
    split_edges = {
        name: store.split_edge_indices(split[name])
        for name in ("train", "validation", "test")
    }
    inference_edges = np.concatenate(
        [split_edges[name] for name in prediction_split_names]
    )
    prototype_ids = _compact_values(store.proposals["candidate_prototype_ids"], store).astype(np.int64)
    candidates = UniqueTrackCandidateSet(
        store.compact_canonical_rows,
        store.compact_candidate_tracks,
        prototype_ids,
        _compact_values(store.proposals["coarse_scores"], store),
    )
    baseline_scores = _compact_values(
        store.proposals[f"strategy__{args.baseline_strategy}"], store
    )
    candidate_residuals = _compact_values(store.proposals["candidate_gt_residuals_px"], store)
    nearest_residuals = np.asarray(store.proposals["nearest_visible_residuals_px"], dtype=np.float32)[
        store.selected_rows
    ]
    query_observations = [
        ColmapTrackObservation(
            track_id=int(store.proposals["nearest_visible_track_ids"][global_row]),
            image_id=str(store.query_ids[global_row]),
            point2d_idx=int(global_row),
            xy=(float(store.query_xy[global_row, 0]), float(store.query_xy[global_row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for global_row in store.selected_rows.tolist()
    ]
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    def identity(scores: np.ndarray, split_name: str) -> dict[str, object]:
        mask = split_row_masks[split_name]
        return _identity_metrics(
            nearest_residuals=nearest_residuals[mask],
            candidate_residuals=candidate_residuals[mask],
            scores=scores[mask],
            query_ids=selected_query_ids[mask],
            labels=store.labels[mask],
            valid_edges=store.valid_edges[mask],
        )

    def pose(
        scores: np.ndarray,
        split_name: str,
        strategy: str,
        *,
        max_matches: int | None = None,
        selection_mode: str = "score_topk",
    ):
        rows = np.flatnonzero(split_row_masks[split_name])
        subset = UniqueTrackCandidateSet(
            candidates.bank_row_indices[rows],
            candidates.track_ids[rows],
            candidates.prototype_ids[rows],
            candidates.coarse_scores[rows],
        )
        return _evaluate_pose_strategy(
            strategy=str(strategy),
            scores=scores[rows],
            candidates=subset,
            query_observations=[query_observations[int(row)] for row in rows.tolist()],
            query_ids=selected_query_ids[rows].tolist(),
            landmark_index=store.landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
            max_matches=max_matches,
            pose_selection_mode=str(selection_mode),
        )

    checkpoint_predictions = []
    checkpoint_metadata = []
    checkpoint_latent_artifacts = []
    checkpoint_formats = set()
    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        device = torch.device(devices[checkpoint_index % len(devices)])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_format = str(checkpoint.get("format", ""))
        if checkpoint_format not in {
            "candidate_maplet_matcher_checkpoint_v5",
            "candidate_maplet_matcher_checkpoint_v6",
            "candidate_maplet_matcher_checkpoint_v7",
        }:
            raise ValueError(f"unsupported checkpoint: {checkpoint_path}")
        if checkpoint.get("data_manifest") != data_manifest:
            checkpoint_manifest = dict(checkpoint.get("data_manifest") or {})
            manifest_mismatches = {
                key: {
                    "checkpoint": checkpoint_manifest.get(key),
                    "evaluation": data_manifest.get(key),
                }
                for key in sorted(set(checkpoint_manifest) | set(data_manifest))
                if checkpoint_manifest.get(key) != data_manifest.get(key)
            }
            raise ValueError(
                "checkpoint data manifest does not match evaluation inputs: "
                f"{checkpoint_path}; mismatches={json.dumps(manifest_mismatches, sort_keys=True)}"
            )
        if checkpoint.get("split") != split:
            raise ValueError(
                f"checkpoint split does not match evaluation split JSON: {checkpoint_path}"
            )
        checkpoint_formats.add(checkpoint_format)
        config = CandidateMapletMatcherConfig(**dict(checkpoint["model_config"]))
        if (
            int(config.query_input_dim) != store.query_input_dim
            or int(config.support_input_dim) != store.support_input_dim
            or int(config.static_input_dim) != store.static_input_dim
        ):
            raise ValueError("checkpoint and episode-store feature dimensions differ")
        model = CandidateMapletMatcher(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions = _predict_edges(
            model,
            store,
            inference_edges,
            device=device,
            batch_size=int(args.eval_batch_size),
            use_amp=bool(device.type == "cuda" and not args.no_amp),
            export_candidate_view_embeddings=bool(args.export_checkpoint_latents),
        )
        if bool(args.export_checkpoint_latents):
            latent_keys = tuple(
                f"candidate_view_embedding_{view_rank}"
                for view_rank in range(store.support_view_count)
            )
            if not all(key in predictions for key in latent_keys):
                raise RuntimeError("candidate latent export is incomplete")
            latent_path = output_dir / f"checkpoint_{checkpoint_index}_candidate_latents.npz"
            latent_metadata = {
                "format": "candidate_maplet_view_latents_v1",
                "data_manifest": data_manifest,
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "checkpoint_format": checkpoint_format,
                "prediction_splits": list(prediction_split_names),
                "model_dim": int(config.model_dim),
                "support_view_count": int(store.support_view_count),
            }
            np.savez(
                latent_path,
                **{key: predictions.pop(key) for key in latent_keys},
                metadata_json=np.asarray(
                    json.dumps(latent_metadata, sort_keys=True), dtype=np.str_
                ),
            )
            checkpoint_latent_artifacts.append(
                {
                    "path": str(latent_path),
                    "sha256": file_sha256_short(latent_path),
                    "checkpoint_sha256": latent_metadata["checkpoint_sha256"],
                }
            )
        checkpoint_predictions.append(predictions)
        checkpoint_metadata.append(
            {
                "path": str(checkpoint_path),
                "format": checkpoint_format,
                "sha256": file_sha256_short(checkpoint_path),
                "seed": int(checkpoint.get("seed", -1)),
                "epoch": int(checkpoint.get("epoch", -1)),
                "training_selection": checkpoint.get("selection"),
                "checkpoint_role": checkpoint.get("checkpoint_role"),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(checkpoint_formats) != 1:
        raise ValueError("ensemble checkpoints must use the same model format")

    ensemble_predictions = {}
    for strategy_name in checkpoint_predictions[0]:
        stacked = np.stack(
            [prediction[strategy_name] for prediction in checkpoint_predictions], axis=0
        )
        finite = np.isfinite(stacked)
        counts = np.sum(finite, axis=0)
        sums = np.sum(np.where(finite, stacked, 0.0), axis=0)
        values = np.full(stacked.shape[1:], -np.inf, dtype=np.float32)
        valid = counts > 0
        values[valid] = (sums[valid] / counts[valid]).astype(np.float32)
        ensemble_predictions[strategy_name] = values.reshape(
            len(store.selected_rows), store.candidate_top_k
        )
    if {
        "rescue_candidate_probability",
        "rescue_keep_probability_DIAGNOSTIC_ONLY",
    }.issubset(ensemble_predictions):
        rescue_candidate_probabilities = ensemble_predictions[
            "rescue_candidate_probability"
        ]
        rescue_keep_probabilities = ensemble_predictions[
            "rescue_keep_probability_DIAGNOSTIC_ONLY"
        ][:, 0]
        rescue_active_rows = np.zeros((len(selected_query_ids),), dtype=bool)
        for split_name in prediction_split_names:
            rescue_active_rows |= split_row_masks[split_name]

        def resolve_ensemble_rescue(
            action_margin_threshold: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            selected = np.argmax(
                np.where(store.valid_edges, baseline_scores, -np.inf), axis=1
            ).astype(np.int64)
            resolved = np.full_like(baseline_scores, -np.inf, dtype=np.float32)
            switched = np.zeros((len(baseline_scores),), dtype=bool)
            action_margin = np.full((len(baseline_scores),), -np.inf, dtype=np.float32)
            action_scores = np.full_like(baseline_scores, -np.inf, dtype=np.float32)
            (
                active_selected,
                active_resolved,
                active_switched,
                active_action_margin,
                active_action_scores,
            ) = resolve_rescue_policy_scores(
                rescue_candidate_probabilities[rescue_active_rows],
                rescue_keep_probabilities[rescue_active_rows],
                baseline_scores[rescue_active_rows],
                action_margin_threshold=float(action_margin_threshold),
                valid_mask=store.valid_edges[rescue_active_rows],
            )
            selected[rescue_active_rows] = active_selected
            resolved[rescue_active_rows] = active_resolved
            switched[rescue_active_rows] = active_switched
            action_margin[rescue_active_rows] = active_action_margin
            action_scores[rescue_active_rows] = active_action_scores
            return selected, resolved, switched, action_margin, action_scores

        (
            _selected,
            rescue_resolved,
            _switched,
            _action_margin,
            rescue_action_scores,
        ) = resolve_ensemble_rescue(0.0)
        ensemble_predictions["rescue_policy_resolved"] = rescue_resolved
        ensemble_predictions["rescue_action_probability"] = rescue_action_scores

    baseline_validation_identity = identity(baseline_scores, "validation")
    row_baseline_validation_pose, _row_baseline_validation_rows = pose(
        baseline_scores, "validation", "baseline_validation"
    )
    formal_global_assignment = frozen_global_baseline_source is not None
    baseline_pose_scores, baseline_selected_columns = _baseline_assignment_scores(
        candidates=candidates,
        scores=baseline_scores,
        query_ids=selected_query_ids,
        valid_edges=store.valid_edges,
        formal_global_assignment=formal_global_assignment,
    )
    baseline_validation_pose = row_baseline_validation_pose
    baseline_validation_rows = _row_baseline_validation_rows
    if formal_global_assignment:
        baseline_validation_pose, baseline_validation_rows = pose(
            baseline_pose_scores,
            "validation",
            "frozen_global_baseline_validation",
            max_matches=int(args.global_assignment_match_count),
            selection_mode=str(args.global_assignment_selection_mode),
        )
        _validate_frozen_baseline_pose(
            baseline_validation_pose,
            frozen_global_baseline_source,
        )
    validation_strategy_names = tuple(
        value.strip()
        for value in str(args.validation_strategies).split(",")
        if value.strip()
    )
    missing_strategies = set(validation_strategy_names) - set(ensemble_predictions)
    if not validation_strategy_names or missing_strategies:
        raise ValueError(
            f"invalid validation strategies; missing={sorted(missing_strategies)}"
        )
    validation_trials = []
    validation_trial_pose_rows: list[dict[str, object]] = []
    formal_resolved_scores: dict[str, np.ndarray] = {}
    formal_selected_columns: dict[str, np.ndarray] = {}
    formal_pose_rows: dict[str, dict[str, list[dict[str, object]]]] = {}
    if formal_global_assignment:
        formal_pose_rows["validation"] = {
            "frozen_baseline": baseline_validation_rows,
        }
    for strategy_name in validation_strategy_names:
        scores = ensemble_predictions[strategy_name]
        if formal_global_assignment:
            resolved, selected_columns = _global_assignment_scores(
                candidates=candidates,
                scores=scores,
                query_ids=selected_query_ids,
                valid_edges=store.valid_edges,
            )
            selected_identity = identity(scores, "validation")
            formal_resolved_scores[str(strategy_name)] = resolved
            formal_selected_columns[str(strategy_name)] = selected_columns
            selected_pose, selected_rows = pose(
                resolved,
                "validation",
                f"ensemble_{strategy_name}_global_partial_assignment_validation",
                max_matches=int(args.global_assignment_match_count),
                selection_mode=str(args.global_assignment_selection_mode),
            )
            validation_mask = split_row_masks["validation"]
            trial = {
                "strategy": strategy_name,
                "mode": "global_partial_assignment",
                "margin_threshold": None,
                "action_margin_threshold": None,
                "switch_count": int(
                    np.sum(
                        selected_columns[validation_mask]
                        != baseline_selected_columns[validation_mask]
                    )
                ),
                "accepted_query_count": int(
                    np.sum(selected_columns[validation_mask] >= 0)
                ),
                "identity": selected_identity,
                "pose": selected_pose,
                "passes_pose_gate": _pose_gate(
                    selected_pose, baseline_validation_pose
                ),
                "passes_identity_gate": _assignment_identity_gate(
                    selected_identity, baseline_validation_identity
                ),
            }
            validation_trials.append(trial)
            validation_trial_pose_rows.append(
                {"trial": trial, "pose_rows": selected_rows}
            )
            formal_pose_rows["validation"][str(strategy_name)] = selected_rows
            continue
        if str(strategy_name) == "rescue_policy_resolved":
            for action_threshold in tuple(args.rescue_action_margin_thresholds):
                (
                    _selected,
                    resolved,
                    switched,
                    _action_margins,
                    _action_scores,
                ) = resolve_ensemble_rescue(float(action_threshold))
                selected_identity = identity(resolved, "validation")
                selected_pose, _selected_rows = pose(
                    resolved,
                    "validation",
                    (
                        "ensemble_rescue_policy_resolved_"
                        f"action_margin{float(action_threshold):g}_validation"
                    ),
                )
                trial = {
                    "strategy": strategy_name,
                    "mode": "rescue_action_margin",
                    "margin_threshold": None,
                    "action_margin_threshold": float(action_threshold),
                    "switch_count": int(
                        np.sum(switched[split_row_masks["validation"]])
                    ),
                    "identity": selected_identity,
                    "pose": selected_pose,
                    "rescue_policy": _rescue_policy_report(
                        ensemble_predictions,
                        store,
                        baseline_scores,
                        row_mask=split_row_masks["validation"],
                        candidate_threshold_px=float(
                            config.rescue_candidate_threshold_px
                        ),
                        baseline_invalid_threshold_px=float(
                            config.rescue_baseline_invalid_threshold_px
                        ),
                        action_margin_threshold=float(action_threshold),
                    ),
                    "passes_pose_gate": _pose_gate(
                        selected_pose, baseline_validation_pose
                    ),
                    "passes_identity_gate": _assignment_identity_gate(
                        selected_identity, baseline_validation_identity
                    ),
                }
                validation_trials.append(trial)
                validation_trial_pose_rows.append(
                    {"trial": trial, "pose_rows": _selected_rows}
                )
            continue
        unconditional_identity = identity(scores, "validation")
        unconditional_pose, _rows = pose(
            scores, "validation", f"ensemble_{strategy_name}_unconditional_validation"
        )
        unconditional_choices = np.argmax(scores, axis=1)
        baseline_choices = np.argmax(baseline_scores, axis=1)
        trial = {
            "strategy": strategy_name,
            "mode": "unconditional",
            "margin_threshold": None,
            "action_margin_threshold": None,
            "switch_count": int(
                np.sum(
                    (unconditional_choices != baseline_choices)
                    & split_row_masks["validation"]
                )
            ),
            "identity": unconditional_identity,
            "pose": unconditional_pose,
            "passes_pose_gate": _pose_gate(
                unconditional_pose, baseline_validation_pose
            ),
            "passes_identity_gate": _assignment_identity_gate(
                unconditional_identity, baseline_validation_identity
            ),
        }
        validation_trials.append(trial)
        validation_trial_pose_rows.append({"trial": trial, "pose_rows": _rows})
        strategy_switch_thresholds = tuple(args.switch_margin_thresholds)
        for threshold in strategy_switch_thresholds:
            _selected, resolved, switched, _margins = selective_switch_scores(
                scores,
                baseline_scores,
                margin_threshold=float(threshold),
                valid_mask=store.valid_edges,
                preserve_baseline_row_confidence=str(strategy_name).endswith(
                    "_prior_row_confidence"
                ),
            )
            selected_identity = identity(resolved, "validation")
            selected_pose, _selected_rows = pose(
                resolved,
                "validation",
                f"ensemble_{strategy_name}_margin{float(threshold):g}_validation",
            )
            trial = {
                "strategy": strategy_name,
                "mode": "selective",
                "margin_threshold": float(threshold),
                "action_margin_threshold": None,
                "switch_count": int(
                    np.sum(switched[split_row_masks["validation"]])
                ),
                "identity": selected_identity,
                "pose": selected_pose,
                "passes_pose_gate": _pose_gate(
                    selected_pose, baseline_validation_pose
                ),
                "passes_identity_gate": _assignment_identity_gate(
                    selected_identity, baseline_validation_identity
                ),
            }
            validation_trials.append(trial)
            validation_trial_pose_rows.append(
                {"trial": trial, "pose_rows": _selected_rows}
            )
    for trial in validation_trials:
        trial["passes_stage_gate"] = bool(
            trial["passes_pose_gate"] and trial["passes_identity_gate"]
        )
    eligible = [trial for trial in validation_trials if bool(trial["passes_stage_gate"])]
    if eligible:
        chosen = max(
            eligible,
            key=lambda trial: (
                float(trial["pose"]["recall_10cm_5deg"]),
                float(trial["pose"]["recall_5cm_5deg"]),
                -float(trial["pose"]["median_translation_m_success"]),
                -float(trial["pose"]["p90_translation_m_success"]),
                -float(trial["pose"]["median_rotation_deg_success"]),
                float(
                    trial["identity"]["geometry"]["thresholds_px"]["2"][
                        "recall_at_1_given_mappable"
                    ]
                ),
            ),
        )
        validation_gate_passed = True
    else:
        chosen = {
            "strategy": "baseline",
            "mode": (
                "global_baseline_fallback"
                if formal_global_assignment
                else "fallback"
            ),
            "margin_threshold": None,
            "action_margin_threshold": None,
            "switch_count": 0,
            "identity": baseline_validation_identity,
            "pose": baseline_validation_pose,
            "passes_pose_gate": False,
            "passes_identity_gate": False,
            "passes_stage_gate": False,
        }
        validation_gate_passed = False

    def resolve_trial_scores(
        trial: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pose scores, pre-conflict identity scores, and changed rows."""

        mode = str(trial["mode"])
        if mode == "global_baseline_fallback":
            return (
                baseline_pose_scores,
                baseline_scores,
                np.zeros((len(baseline_scores),), dtype=bool),
            )
        if mode == "fallback":
            return (
                baseline_scores,
                baseline_scores,
                np.zeros((len(baseline_scores),), dtype=bool),
            )
        strategy = str(trial["strategy"])
        model_scores = ensemble_predictions[strategy]
        if mode == "global_partial_assignment":
            resolved = formal_resolved_scores[strategy]
            selected_columns = formal_selected_columns[strategy]
            switched = selected_columns != baseline_selected_columns
            return resolved, model_scores, switched
        if mode == "rescue_action_margin":
            _selected, resolved, switched, _margins, _action_scores = (
                resolve_ensemble_rescue(float(trial["action_margin_threshold"]))
            )
            return resolved, resolved, switched
        if mode == "selective":
            _selected, resolved, switched, _margins = selective_switch_scores(
                model_scores,
                baseline_scores,
                margin_threshold=float(trial["margin_threshold"]),
                valid_mask=store.valid_edges,
                preserve_baseline_row_confidence=strategy.endswith(
                    "_prior_row_confidence"
                ),
            )
            return resolved, resolved, switched
        if mode == "unconditional":
            switched = np.argmax(model_scores, axis=1) != np.argmax(
                baseline_scores, axis=1
            )
            return model_scores, model_scores, switched
        raise ValueError(f"unsupported validation trial mode: {mode}")

    baseline_test_identity = identity(baseline_scores, "test")
    baseline_test_pose, baseline_test_rows = pose(
        baseline_pose_scores,
        "test",
        "baseline_test",
        max_matches=(
            int(args.global_assignment_match_count)
            if formal_global_assignment
            else None
        ),
        selection_mode=(
            str(args.global_assignment_selection_mode)
            if formal_global_assignment
            else "score_topk"
        ),
    )
    if formal_global_assignment:
        formal_pose_rows["test"] = {"frozen_baseline": baseline_test_rows}
    cross_block_trials = []
    cross_block_trial_pose_rows: list[dict[str, object]] = []
    if bool(args.development_cross_block_audit):
        for trial_index, trial in enumerate(validation_trials):
            trial_scores, trial_identity_scores, trial_switched = resolve_trial_scores(
                trial
            )
            trial_test_identity = identity(trial_identity_scores, "test")
            trial_test_pose, _trial_test_rows = pose(
                trial_scores,
                "test",
                f"development_cross_block_trial_{trial_index}",
                max_matches=(
                    int(args.global_assignment_match_count)
                    if formal_global_assignment
                    else None
                ),
                selection_mode=(
                    str(args.global_assignment_selection_mode)
                    if formal_global_assignment
                    else "score_topk"
                ),
            )
            trial_test_pose_gate = _pose_gate(trial_test_pose, baseline_test_pose)
            trial_test_identity_gate = _assignment_identity_gate(
                trial_test_identity, baseline_test_identity
            )
            trial_rescue_policy = None
            if str(trial["mode"]) == "rescue_action_margin":
                trial_rescue_policy = _rescue_policy_report(
                    ensemble_predictions,
                    store,
                    baseline_scores,
                    row_mask=split_row_masks["test"],
                    candidate_threshold_px=float(config.rescue_candidate_threshold_px),
                    baseline_invalid_threshold_px=float(
                        config.rescue_baseline_invalid_threshold_px
                    ),
                    action_margin_threshold=float(trial["action_margin_threshold"]),
                )
            cross_block_trials.append(
                {
                    "strategy": trial["strategy"],
                    "mode": trial["mode"],
                    "margin_threshold": trial["margin_threshold"],
                    "action_margin_threshold": trial["action_margin_threshold"],
                    "validation_passes_stage_gate": bool(trial["passes_stage_gate"]),
                    "test_switch_count": int(
                        np.sum(trial_switched[split_row_masks["test"]])
                    ),
                    "test_identity": trial_test_identity,
                    "test_pose": trial_test_pose,
                    "test_rescue_policy": trial_rescue_policy,
                    "test_passes_pose_gate": bool(trial_test_pose_gate),
                    "test_passes_identity_gate": bool(trial_test_identity_gate),
                    "test_passes_stage_gate": bool(
                        trial_test_pose_gate and trial_test_identity_gate
                    ),
                    "passes_both_development_blocks": bool(
                        trial["passes_stage_gate"]
                        and trial_test_pose_gate
                        and trial_test_identity_gate
                    ),
                }
            )
            cross_block_trial_pose_rows.append(
                {"trial": dict(cross_block_trials[-1]), "pose_rows": _trial_test_rows}
            )
            if formal_global_assignment:
                formal_pose_rows["test"][str(trial["strategy"])] = _trial_test_rows

    if formal_global_assignment and "train" in prediction_split_names:
        _baseline_train_pose, baseline_train_rows = pose(
            baseline_pose_scores,
            "train",
            "frozen_global_baseline_train_diagnostic",
            max_matches=int(args.global_assignment_match_count),
            selection_mode=str(args.global_assignment_selection_mode),
        )
        formal_pose_rows["train"] = {"frozen_baseline": baseline_train_rows}
        for strategy_name in validation_strategy_names:
            _strategy_train_pose, strategy_train_rows = pose(
                formal_resolved_scores[str(strategy_name)],
                "train",
                f"ensemble_{strategy_name}_global_partial_assignment_train_diagnostic",
                max_matches=int(args.global_assignment_match_count),
                selection_mode=str(args.global_assignment_selection_mode),
            )
            formal_pose_rows["train"][str(strategy_name)] = strategy_train_rows

    selected_test_rescue_policy = None
    if validation_gate_passed:
        selected_scores, selected_identity_scores, switched = resolve_trial_scores(
            chosen
        )
        test_switch_count = int(np.sum(switched[split_row_masks["test"]]))
        if str(chosen["mode"]) == "rescue_action_margin":
            selected_test_rescue_policy = _rescue_policy_report(
                ensemble_predictions,
                store,
                baseline_scores,
                row_mask=split_row_masks["test"],
                candidate_threshold_px=float(config.rescue_candidate_threshold_px),
                baseline_invalid_threshold_px=float(
                    config.rescue_baseline_invalid_threshold_px
                ),
                action_margin_threshold=float(chosen["action_margin_threshold"]),
            )
        selected_test_identity = identity(selected_identity_scores, "test")
        selected_test_pose, selected_test_rows = pose(
            selected_scores,
            "test",
            "ensemble_selected_test",
            max_matches=(
                int(args.global_assignment_match_count)
                if formal_global_assignment
                else None
            ),
            selection_mode=(
                str(args.global_assignment_selection_mode)
                if formal_global_assignment
                else "score_topk"
            ),
        )
    else:
        selected_scores = baseline_pose_scores
        selected_identity_scores = baseline_scores
        selected_test_identity = baseline_test_identity
        selected_test_pose = baseline_test_pose
        selected_test_rows = baseline_test_rows
        test_switch_count = 0
    test_pose_gate_passed = _pose_gate(selected_test_pose, baseline_test_pose)
    test_identity_gate_passed = _assignment_identity_gate(
        selected_test_identity, baseline_test_identity
    )
    test_gate_passed = bool(test_pose_gate_passed and test_identity_gate_passed)
    promoted = bool(
        validation_gate_passed
        and test_gate_passed
        and args.evaluation_role == "untouched_test"
    )
    score_metadata = {
        "format": "candidate_maplet_ensemble_scores_v2",
        "data_manifest": data_manifest,
        "checkpoint_sha256": [item["sha256"] for item in checkpoint_metadata],
        "prediction_splits": list(prediction_split_names),
    }
    np.savez(
        output_dir / "ensemble_scores.npz",
        **{
            f"ensemble__{name}": values
            for name, values in ensemble_predictions.items()
        },
        baseline_scores=baseline_scores,
        selected_scores=selected_scores,
        selected_identity_scores=selected_identity_scores,
        metadata_json=np.asarray(json.dumps(score_metadata, sort_keys=True), dtype=np.str_),
    )
    (output_dir / "pose_rows_test.json").write_text(
        json.dumps(
            {"baseline": baseline_test_rows, "selected": selected_test_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "validation_trial_pose_rows.json").write_text(
        json.dumps(validation_trial_pose_rows, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "cross_block_trial_pose_rows.json").write_text(
        json.dumps(cross_block_trial_pose_rows, indent=2, sort_keys=True) + "\n"
    )
    if formal_global_assignment:
        (output_dir / "formal_global_pose_rows.json").write_text(
            json.dumps(formal_pose_rows, indent=2, sort_keys=True) + "\n"
        )
    summary = {
        "stage": "candidate_maplet_checkpoint_ensemble",
        "protocol": {
            "evaluation_role": str(args.evaluation_role),
            "baseline_strategy": str(args.baseline_strategy),
            "development_data_reused": bool(args.evaluation_role == "development"),
            "ensemble": (
                "single_checkpoint"
                if len(checkpoint_paths) == 1
                else "arithmetic_mean_probability_across_validation_selected_seeds"
            ),
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "test_used_for_selection": False,
            "prediction_splits": list(prediction_split_names),
            "development_cross_block_audit": bool(
                args.development_cross_block_audit
            ),
            "cross_block_audit_used_for_selection": False,
            "formal_global_assignment": (
                None
                if not formal_global_assignment
                else {
                    "assignment": "whole_image_sparse_bipartite_with_per_query_dustbin",
                    "dustbin_score": None,
                    "max_matches": int(args.global_assignment_match_count),
                    "selection_mode": str(args.global_assignment_selection_mode),
                    "frozen_baseline_source": frozen_global_baseline_source,
                    "identity_evaluated_before_conflict_resolution": True,
                }
            ),
            "rescue_probability_fusion": (
                "average_candidate_and_keep_probabilities_then_resolve_once"
            ),
        },
        "split": split,
        "checkpoints": checkpoint_metadata,
        "data_manifest": data_manifest,
        "validation": {
            "baseline": {
                "identity": baseline_validation_identity,
                "pose": baseline_validation_pose,
                "row_argmax_pose_diagnostic": row_baseline_validation_pose,
            },
            "trials": validation_trials,
            "chosen": chosen,
            "passes_stage_gate": bool(validation_gate_passed),
        },
        "development_cross_block_audit": {
            "enabled": bool(args.development_cross_block_audit),
            "diagnostic_only": True,
            "trials": cross_block_trials,
            "passing_both_block_count": int(
                sum(
                    bool(trial["passes_both_development_blocks"])
                    for trial in cross_block_trials
                )
            ),
        },
        "test": {
            "baseline": {"identity": baseline_test_identity, "pose": baseline_test_pose},
            "selected": {
                "strategy": chosen["strategy"],
                "mode": chosen["mode"],
                "margin_threshold": chosen["margin_threshold"],
                "action_margin_threshold": chosen["action_margin_threshold"],
                "switch_count": test_switch_count,
                "identity": selected_test_identity,
                "pose": selected_test_pose,
                "rescue_policy": selected_test_rescue_policy,
                "passes_pose_gate": bool(test_pose_gate_passed),
                "passes_identity_gate": bool(test_identity_gate_passed),
                "passes_stage_gate": bool(test_gate_passed),
            },
        },
        "gate": {
            "validation_passed": bool(validation_gate_passed),
            "test_passed": bool(test_gate_passed),
            "test_pose_passed": bool(test_pose_gate_passed),
            "test_identity_passed": bool(test_identity_gate_passed),
            "evaluation_role_allows_production_promotion": bool(
                args.evaluation_role == "untouched_test"
            ),
            "production_promoted": bool(promoted),
        },
        "runtime_seconds": float(time.time() - start_time),
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "scores": str(output_dir / "ensemble_scores.npz"),
            "scores_sha256": file_sha256_short(output_dir / "ensemble_scores.npz"),
            "checkpoint_latents": checkpoint_latent_artifacts,
            "pose_rows_test": str(output_dir / "pose_rows_test.json"),
            "validation_trial_pose_rows": str(
                output_dir / "validation_trial_pose_rows.json"
            ),
            "validation_trial_pose_rows_sha256": file_sha256_short(
                output_dir / "validation_trial_pose_rows.json"
            ),
            "cross_block_trial_pose_rows": str(
                output_dir / "cross_block_trial_pose_rows.json"
            ),
            "cross_block_trial_pose_rows_sha256": file_sha256_short(
                output_dir / "cross_block_trial_pose_rows.json"
            ),
            "formal_global_pose_rows": (
                None
                if not formal_global_assignment
                else str(output_dir / "formal_global_pose_rows.json")
            ),
            "formal_global_pose_rows_sha256": (
                None
                if not formal_global_assignment
                else file_sha256_short(output_dir / "formal_global_pose_rows.json")
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": summary["gate"], "validation_chosen": chosen, "test": summary["test"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
