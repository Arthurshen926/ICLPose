"""Train a grid-local partial assignment matcher on detector hard proposals."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_grid_assignment_data import DetectorGridAssignmentStore
from feature_extract.vfm.localization.detector_landmark_proposals import (
    summarize_ranked_detector_proposal_geometry,
)
from feature_extract.vfm.localization.local_assignment_linear import selective_switch_scores
from feature_extract.vfm.localization.local_assignment_matcher import (
    LocalAssignmentMatcher,
    LocalAssignmentMatcherConfig,
    local_assignment_loss,
)
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
    binary_average_precision,
)


def _float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected at least one float")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dustbin_only_epochs", type=int, default=2)
    parser.add_argument("--dustbin_learning_rate", type=float, default=5e-3)
    parser.add_argument("--residual_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--pair_loss_weight", type=float, default=0.2)
    parser.add_argument("--no_match_loss_weight", type=float, default=1.0)
    parser.add_argument("--candidate_top_k", type=int, default=10)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--train_query_count", type=int, default=60)
    parser.add_argument("--validation_query_count", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument(
        "--switch_margin_thresholds",
        type=_float_list,
        default=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _query_rows(store: DetectorGridAssignmentStore, query_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([store.rows_by_query[str(query_id)] for query_id in query_ids], axis=0)


def _episode_keys(store: DetectorGridAssignmentStore, query_ids: Sequence[str]):
    return [key for query_id in query_ids for key in store.episode_keys_by_query[str(query_id)]]


def _geometry_metrics(
    store: DetectorGridAssignmentStore,
    rows: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    summary = summarize_ranked_detector_proposal_geometry(
        nearest_landmark_residuals=store.nearest_residuals[rows],
        candidate_residuals=store.candidate_residuals[rows],
        candidate_scores=np.asarray(scores, dtype=np.float32)[rows],
        query_ids=store.query_ids[rows],
        thresholds_px=(1.0, 2.0, 5.0, 8.0),
        top_ls=(1, 5, 10, store.candidates.top_l),
    )
    return summary


@torch.no_grad()
def evaluate_model(
    model: LocalAssignmentMatcher,
    store: DetectorGridAssignmentStore,
    query_ids: Sequence[str],
    *,
    device: str,
    pair_loss_weight: float,
    no_match_loss_weight: float,
) -> dict[str, object]:
    model.eval()
    rows = _query_rows(store, query_ids)
    row_allowed = np.zeros((store.candidates.query_count,), dtype=bool)
    row_allowed[rows] = True
    scores = np.full(store.candidates.coarse_scores.shape, -np.inf, dtype=np.float32)
    accepted_scores = np.full_like(scores, -np.inf)
    no_match_probabilities = np.full((store.candidates.query_count,), np.nan, dtype=np.float32)
    losses: list[float] = []
    target_correct = 0
    target_count = 0
    for key in _episode_keys(store, query_ids):
        episode_cpu = store.episode(key)
        episode = episode_cpu.to(device)
        output = model(episode)
        loss, _metrics = local_assignment_loss(
            output,
            episode,
            pair_loss_weight=float(pair_loss_weight),
            no_match_loss_weight=float(no_match_loss_weight),
        )
        losses.append(float(loss.cpu().item()))
        log_probabilities = output["query_log_probabilities"].cpu().numpy().astype(np.float32)
        predictions = np.argmax(log_probabilities, axis=1)
        targets = episode_cpu.target_track_indices.numpy()
        target_correct += int(np.sum(predictions == targets))
        target_count += int(len(targets))
        dustbin_prob = torch.sigmoid(output["no_match_logits"]).cpu().numpy().astype(np.float32)
        edge_queries = episode_cpu.edge_query_indices.numpy()
        edge_tracks = episode_cpu.edge_track_indices.numpy()
        columns = episode_cpu.candidate_columns.numpy()
        query_rows = episode_cpu.query_rows.numpy()
        for local_query, global_row in enumerate(query_rows.tolist()):
            no_match_probabilities[global_row] = float(dustbin_prob[local_query])
            edge_mask = edge_queries == int(local_query)
            for track_index, column in zip(edge_tracks[edge_mask], columns[edge_mask]):
                scores[global_row, int(column)] = float(log_probabilities[local_query, int(track_index)])
            if int(predictions[local_query]) < int(episode_cpu.track_features.shape[0]):
                accepted_scores[global_row] = scores[global_row]
    if not np.all(np.isfinite(no_match_probabilities[rows])):
        raise RuntimeError("evaluation did not cover every requested detector row")
    finite_edges = np.isfinite(scores[rows])
    edge_labels = store.candidate_residuals[rows] <= float(store.positive_threshold_px)
    row_has_positive = np.any(edge_labels & finite_edges, axis=1)
    metrics = {
        "row_count": int(len(rows)),
        "episode_count": int(len(_episode_keys(store, query_ids))),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "target_assignment_accuracy": float(target_correct / max(target_count, 1)),
        "selected_pool_positive_row_rate": float(np.mean(row_has_positive)),
        "pair_validity_average_precision": binary_average_precision(
            edge_labels[finite_edges],
            scores[rows][finite_edges],
        ),
        "wrong_pool_rejection_average_precision": binary_average_precision(
            ~row_has_positive,
            no_match_probabilities[rows],
        ),
        "accepted_rate_at_0p5": float(np.mean(no_match_probabilities[rows] < 0.5)),
        "geometry": _geometry_metrics(store, rows, scores),
        "accepted_geometry": _geometry_metrics(store, rows, accepted_scores),
    }
    return {
        "rows": rows,
        "scores": scores,
        "accepted_scores": accepted_scores,
        "no_match_probabilities": no_match_probabilities,
        "metrics": metrics,
    }


def _pose_observations(store: DetectorGridAssignmentStore, rows: np.ndarray):
    nearest_tracks = np.asarray(store.probe["nearest_visible_track_ids"], dtype=np.int64)
    observations = [
        ColmapTrackObservation(
            track_id=int(nearest_tracks[row]),
            image_id=str(store.query_ids[row]),
            point2d_idx=int(row),
            xy=(float(store.query_xy[row, 0]), float(store.query_xy[row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for row in rows.tolist()
    ]
    return observations, store.query_ids[rows].tolist()


def _evaluate_pose(
    *,
    name: str,
    store: DetectorGridAssignmentStore,
    rows: np.ndarray,
    scores: np.ndarray,
    cameras,
    images_by_name,
    reprojection_error_px: float,
    iterations: int,
):
    values = np.asarray(scores, dtype=np.float32).copy()
    values[~store.pose_keep_mask] = -np.inf
    observations, query_ids = _pose_observations(store, rows)
    subset = UniqueTrackCandidateSet(
        store.candidates.bank_row_indices[rows],
        store.candidates.track_ids[rows],
        store.candidates.prototype_ids[rows],
        store.candidates.coarse_scores[rows],
    )
    return _evaluate_pose_strategy(
        strategy=str(name),
        scores=values[rows],
        candidates=subset,
        query_observations=observations,
        query_ids=query_ids,
        landmark_index=store.landmark_index,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
    )


def _pose_gate(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
    lower_is_better = (
        "median_translation_m_success",
        "p90_translation_m_success",
        "median_rotation_deg_success",
    )
    higher_is_better = (
        "success_rate",
        "recall_25cm_2deg",
        "recall_10cm_5deg",
        "recall_5cm_5deg",
    )
    tolerance = 1e-12
    no_regression = all(
        float(candidate[name]) <= float(baseline[name]) + tolerance for name in lower_is_better
    ) and all(
        float(candidate[name]) + tolerance >= float(baseline[name]) for name in higher_is_better
    )
    strict_improvement = any(
        float(candidate[name]) < float(baseline[name]) - tolerance for name in lower_is_better
    ) or any(
        float(candidate[name]) > float(baseline[name]) + tolerance for name in higher_is_better
    )
    return bool(no_regression and strict_improvement)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.gradient_accumulation) <= 0:
        raise ValueError("gradient accumulation must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = DetectorGridAssignmentStore(
        proposals=Path(args.proposals),
        detector_query_cache=Path(args.detector_query_cache),
        support_feature_cache=Path(args.support_feature_cache),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        maplet_support_index=Path(args.maplet_support_index),
        candidate_top_k=int(args.candidate_top_k),
        positive_threshold_px=float(args.positive_threshold_px),
        baseline_strategy=str(args.baseline_strategy),
    )
    train_end = int(args.train_query_count)
    validation_end = train_end + int(args.validation_query_count)
    if train_end <= 0 or validation_end >= len(store.unique_query_ids):
        raise ValueError("split counts must leave a non-empty test block")
    split = {
        "strategy": "contiguous_temporal_blocks_v1",
        "train": list(store.unique_query_ids[:train_end]),
        "validation": list(store.unique_query_ids[train_end:validation_end]),
        "test": list(store.unique_query_ids[validation_end:]),
    }
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    config = LocalAssignmentMatcherConfig(
        query_input_dim=store.query_input_dim,
        track_input_dim=store.track_input_dim,
        support_input_dim=store.support_input_dim,
        edge_input_dim=store.edge_input_dim,
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        query_layers=1,
        track_layers=1,
        dropout=float(args.dropout),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
        edge_prior_feature_index=3,
        edge_prior_scale=10.0,
        edge_prior_center=0.8,
        query_dustbin_initial_bias=0.5,
    )
    device = torch.device(str(args.device))
    model = LocalAssignmentMatcher(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.residual_learning_rate),
        weight_decay=float(args.weight_decay),
    )
    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "training_log.jsonl"
    initial_validation = evaluate_model(
        model,
        store,
        split["validation"],
        device=str(device),
        pair_loss_weight=float(args.pair_loss_weight),
        no_match_loss_weight=float(args.no_match_loss_weight),
    )
    initial_two_px = initial_validation["metrics"]["geometry"]["thresholds_px"]["2"]
    best_key = (
        float(initial_two_px["recall_at_1_given_mappable"]),
        float(initial_validation["metrics"]["pair_validity_average_precision"]),
        float(initial_validation["metrics"]["wrong_pool_rejection_average_precision"]),
    )
    best_epoch = -1
    artifact_hashes = {
        name: file_sha256_short(Path(getattr(args, name)))
        for name in (
            "proposals",
            "detector_query_cache",
            "support_feature_cache",
            "projected_landmark_bank",
            "maplet_support_index",
        )
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": -1,
            "validation_metrics": initial_validation["metrics"],
            "promotion_key": list(best_key),
            "split": split,
            "seed": int(args.seed),
            "artifact_hashes": artifact_hashes,
        },
        checkpoint_path,
    )
    generator = np.random.default_rng(int(args.seed))
    stale_epochs = 0
    with log_path.open("w") as log_handle:
        log_handle.write(json.dumps({"epoch": -1, "validation": initial_validation["metrics"], "promotion_key": list(best_key)}, sort_keys=True) + "\n")
        for epoch in range(int(args.epochs)):
            dustbin_only = epoch < int(args.dustbin_only_epochs)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(not dustbin_only or name.startswith("no_match_head"))
            learning_rate = float(args.dustbin_learning_rate) if dustbin_only else float(args.residual_learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            train_keys = _episode_keys(store, split["train"])
            order = generator.permutation(len(train_keys))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for step, position in enumerate(order.tolist()):
                episode = store.episode(train_keys[int(position)]).to(device)
                output = model(episode)
                if dustbin_only:
                    targets = (
                        episode.target_track_indices == int(episode.track_features.shape[0])
                    ).to(dtype=output["no_match_logits"].dtype)
                    loss = F.binary_cross_entropy_with_logits(output["no_match_logits"], targets)
                else:
                    loss, _metrics = local_assignment_loss(
                        output,
                        episode,
                        pair_loss_weight=float(args.pair_loss_weight),
                        no_match_loss_weight=float(args.no_match_loss_weight),
                    )
                (loss / int(args.gradient_accumulation)).backward()
                losses.append(float(loss.detach().cpu().item()))
                should_step = (step + 1) % int(args.gradient_accumulation) == 0 or step + 1 == len(order)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            validation = evaluate_model(
                model,
                store,
                split["validation"],
                device=str(device),
                pair_loss_weight=float(args.pair_loss_weight),
                no_match_loss_weight=float(args.no_match_loss_weight),
            )
            two_px = validation["metrics"]["geometry"]["thresholds_px"]["2"]
            key = (
                float(two_px["recall_at_1_given_mappable"]),
                float(validation["metrics"]["pair_validity_average_precision"]),
                float(validation["metrics"]["wrong_pool_rejection_average_precision"]),
            )
            record = {
                "epoch": int(epoch),
                "phase": "dustbin_only" if dustbin_only else "assignment_residual",
                "learning_rate": learning_rate,
                "train_mean_loss": float(np.mean(losses)),
                "validation": validation["metrics"],
                "promotion_key": list(key),
            }
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()
            print(json.dumps({"epoch": epoch, "train_loss": record["train_mean_loss"], "key": key}), flush=True)
            if key > best_key:
                best_key = key
                best_epoch = int(epoch)
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "epoch": int(epoch),
                        "validation_metrics": validation["metrics"],
                        "promotion_key": list(key),
                        "split": split,
                        "seed": int(args.seed),
                        "artifact_hashes": artifact_hashes,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            if int(args.patience) > 0 and stale_epochs >= int(args.patience):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_model(
        model,
        store,
        split["validation"],
        device=str(device),
        pair_loss_weight=float(args.pair_loss_weight),
        no_match_loss_weight=float(args.no_match_loss_weight),
    )
    test = evaluate_model(
        model,
        store,
        split["test"],
        device=str(device),
        pair_loss_weight=float(args.pair_loss_weight),
        no_match_loss_weight=float(args.no_match_loss_weight),
    )
    baseline_scores = store.baseline_scores
    baseline_geometry = {
        name: _geometry_metrics(store, _query_rows(store, split[name]), baseline_scores)
        for name in ("validation", "test")
    }

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    validation_rows = np.asarray(validation["rows"], dtype=np.int64)
    baseline_validation_pose, baseline_validation_pose_rows = _evaluate_pose(
        name=str(args.baseline_strategy), store=store, rows=validation_rows, scores=baseline_scores,
        cameras=cameras, images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
    )
    threshold_trials = []
    resolved_trials = []
    for threshold in args.switch_margin_thresholds:
        _selected, resolved, switched, _margin = selective_switch_scores(
            np.asarray(validation["scores"], dtype=np.float32),
            baseline_scores,
            margin_threshold=float(threshold),
            valid_mask=store.candidates.valid_mask,
        )
        pose, _rows = _evaluate_pose(
            name=f"grid_assignment_switch_{float(threshold):g}", store=store,
            rows=validation_rows, scores=resolved, cameras=cameras, images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        )
        threshold_trials.append({
            "margin_threshold": float(threshold), "pose": pose,
            "switch_count": int(np.sum(switched[validation_rows])),
            "passes_pose_gate": _pose_gate(pose, baseline_validation_pose),
        })
        resolved_trials.append(resolved)
    eligible = [index for index, trial in enumerate(threshold_trials) if bool(trial["passes_pose_gate"])]
    if eligible:
        chosen_index = max(
            eligible,
            key=lambda index: (
                float(threshold_trials[index]["pose"]["recall_10cm_5deg"]),
                -float(threshold_trials[index]["pose"]["median_translation_m_success"]),
                -float(threshold_trials[index]["margin_threshold"]),
            ),
        )
        chosen_threshold = float(threshold_trials[chosen_index]["margin_threshold"])
        full_model_scores = np.full_like(baseline_scores, -np.inf)
        full_model_scores[validation_rows] = np.asarray(validation["scores"])[validation_rows]
        test_rows = np.asarray(test["rows"], dtype=np.int64)
        full_model_scores[test_rows] = np.asarray(test["scores"])[test_rows]
        _selected, resolved_scores, switched, _margins = selective_switch_scores(
            full_model_scores, baseline_scores, margin_threshold=chosen_threshold,
            valid_mask=store.candidates.valid_mask,
        )
        pose_gate_passed = True
    else:
        chosen_threshold = None
        resolved_scores = baseline_scores.copy()
        switched = np.zeros((store.candidates.query_count,), dtype=bool)
        pose_gate_passed = False
    test_rows = np.asarray(test["rows"], dtype=np.int64)
    test_baseline_pose, test_baseline_rows = _evaluate_pose(
        name=str(args.baseline_strategy), store=store, rows=test_rows, scores=baseline_scores,
        cameras=cameras, images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
    )
    test_selected_pose, test_selected_rows = _evaluate_pose(
        name="grid_assignment_selective_switch", store=store, rows=test_rows, scores=resolved_scores,
        cameras=cameras, images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps({
        "validation_baseline": baseline_validation_pose_rows,
        "test_baseline": test_baseline_rows,
        "test_selected": test_selected_rows,
    }, indent=2, sort_keys=True) + "\n")
    predictions_path = output_dir / "predictions.npz"
    np.savez(
        predictions_path,
        validation_rows=validation_rows,
        validation_scores=np.asarray(validation["scores"], dtype=np.float32)[validation_rows],
        validation_no_match_probabilities=np.asarray(validation["no_match_probabilities"], dtype=np.float32)[validation_rows],
        test_rows=test_rows,
        test_scores=np.asarray(test["scores"], dtype=np.float32)[test_rows],
        test_no_match_probabilities=np.asarray(test["no_match_probabilities"], dtype=np.float32)[test_rows],
    )
    summary = {
        "stage": "s4_l2_detector_grid_partial_assignment",
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "promotion_key": list(best_key),
        "config": config.to_dict(),
        "data_config": {
            "candidate_top_k": int(args.candidate_top_k),
            "positive_threshold_px": float(args.positive_threshold_px),
            "grid_rows": store.grid_rows,
            "grid_cols": store.grid_cols,
            "episode_count": len(store.episode_keys),
        },
        "split": split,
        "artifact_hashes": artifact_hashes,
        "initial_validation": initial_validation["metrics"],
        "validation": validation["metrics"],
        "test": test["metrics"],
        "baseline_geometry": baseline_geometry,
        "selective_switch": {
            "trials": threshold_trials,
            "pose_gate_passed": bool(pose_gate_passed),
            "chosen_margin_threshold": chosen_threshold,
            "validation_switch_count": int(np.sum(switched[validation_rows])),
            "test_switch_count": int(np.sum(switched[test_rows])),
        },
        "pose": {
            "validation_baseline": baseline_validation_pose,
            "validation_selected": baseline_validation_pose if not pose_gate_passed else threshold_trials[chosen_index]["pose"],
            "test_baseline": test_baseline_pose,
            "test_selected": test_selected_pose,
        },
        "runtime_seconds": float(time.time() - start_time),
        "limitations": [
            "the matcher resolves only the ALIKE-reranked top10 candidates per detector node",
            "grid episodes use maplet context features but do not yet add maplet-neighbor tracks outside the proposal pool",
            "no-match probabilities are diagnostic and are not used as PnP covariance",
            "pixel measurement is not active",
        ],
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "training_log": str(log_path),
            "predictions": str(predictions_path),
            "pose_rows": str(pose_rows_path),
            "split": str(split_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
