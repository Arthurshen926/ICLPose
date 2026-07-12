"""Train and evaluate an assignment-only local landmark matcher on real images."""

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
from feature_extract.vfm.localization.local_assignment_data import LocalAssignmentFeatureStore
from feature_extract.vfm.localization.local_assignment_matcher import (
    LocalAssignmentMatcher,
    LocalAssignmentMatcherConfig,
    local_assignment_loss,
)
from feature_extract.vfm.localization.local_assignment_probe import (
    binary_average_precision,
    summarize_assignment_strategy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_arrays", required=True)
    parser.add_argument("--real_feature_cache", required=True)
    parser.add_argument("--query_global_cache", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--pair_loss_weight", type=float, default=0.2)
    parser.add_argument("--no_match_loss_weight", type=float, default=1.0)
    parser.add_argument("--no_match_threshold", type=float, default=0.5)
    parser.add_argument("--dustbin_only_epochs", type=int, default=10)
    parser.add_argument("--residual_learning_rate", type=float, default=1e-4)
    parser.add_argument("--dustbin_learning_rate", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_support_views", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--query_layers", type=int, default=1)
    parser.add_argument("--track_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--edge_prior_feature_index", type=int, default=4)
    parser.add_argument("--edge_prior_scale", type=float, default=10.0)
    parser.add_argument("--edge_prior_center", type=float, default=0.8)
    parser.add_argument("--query_dustbin_initial_bias", type=float, default=-0.3)
    parser.add_argument("--train_query_count", type=int, default=60)
    parser.add_argument("--validation_query_count", type=int, default=15)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _query_rows(store: LocalAssignmentFeatureStore, query_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([store.rows_by_query[str(query_id)] for query_id in query_ids], axis=0)


@torch.no_grad()
def evaluate_model(
    model: LocalAssignmentMatcher,
    store: LocalAssignmentFeatureStore,
    query_ids: Sequence[str],
    *,
    device: str,
    pair_loss_weight: float,
    no_match_loss_weight: float,
    no_match_threshold: float,
) -> dict[str, object]:
    model.eval()
    rows = _query_rows(store, query_ids)
    row_position = {int(row): int(index) for index, row in enumerate(rows.tolist())}
    subset_candidates = store.subset_candidates(rows)
    raw_scores = np.full(subset_candidates.coarse_scores.shape, -np.inf, dtype=np.float32)
    accepted_scores = np.full_like(raw_scores, -np.inf)
    dustbin_probabilities = np.zeros((len(rows),), dtype=np.float32)
    predicted_track_ids = np.full((len(rows),), -1, dtype=np.int64)
    losses: list[float] = []
    for query_id in query_ids:
        episode_cpu = store.episode(str(query_id))
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
        episode_track_count = int(episode_cpu.track_features.shape[0])
        episode_predictions = np.argmax(log_probabilities[:, :episode_track_count], axis=1)
        no_match_probabilities = torch.sigmoid(output["no_match_logits"]).cpu().numpy().astype(np.float32)
        for local_query, global_row in enumerate(episode_cpu.query_rows.tolist()):
            output_row = row_position[int(global_row)]
            dustbin_probabilities[output_row] = float(no_match_probabilities[local_query])
            selected_track = int(episode_predictions[local_query])
            predicted_no_match = bool(no_match_probabilities[local_query] >= float(no_match_threshold))
            if not predicted_no_match:
                edge_mask = episode_cpu.edge_query_indices.numpy() == int(local_query)
                local_edge_tracks = episode_cpu.edge_track_indices.numpy()[edge_mask]
                local_columns = episode_cpu.candidate_columns.numpy()[edge_mask]
                match = np.flatnonzero(local_edge_tracks == selected_track)
                if match.size:
                    column = int(local_columns[int(match[0])])
                    predicted_track_ids[output_row] = int(subset_candidates.track_ids[output_row, column])
            edge_mask = episode_cpu.edge_query_indices.numpy() == int(local_query)
            for track_index, column in zip(
                episode_cpu.edge_track_indices.numpy()[edge_mask],
                episode_cpu.candidate_columns.numpy()[edge_mask],
            ):
                raw_scores[output_row, int(column)] = float(log_probabilities[local_query, int(track_index)])
            if not predicted_no_match:
                accepted_scores[output_row] = raw_scores[output_row]

    correct = store.correct_track_ids[rows]
    present = np.any(subset_candidates.track_ids == correct[:, None], axis=1)
    target_dustbin = ~present
    predicted_dustbin = predicted_track_ids < 0
    resolved_correct = predicted_track_ids == correct
    accepted = ~predicted_dustbin
    dustbin_true_positive = int(np.sum(predicted_dustbin & target_dustbin))
    dustbin_false_positive = int(np.sum(predicted_dustbin & ~target_dustbin))
    dustbin_false_negative = int(np.sum(~predicted_dustbin & target_dustbin))
    dustbin_true_negative = int(np.sum(~predicted_dustbin & ~target_dustbin))
    identity = summarize_assignment_strategy(
        candidates=subset_candidates,
        correct_track_ids=correct,
        query_ids=[store.query_ids[int(row)] for row in rows],
        scores=raw_scores,
    )
    metrics = {
        "query_count": int(len(rows)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "proposal_present_rate": float(np.mean(present)),
        "resolved_correct_rate": float(np.mean(resolved_correct)),
        "conditional_resolved_correct_rate": _safe_div(np.sum(resolved_correct & present), np.sum(present)),
        "accepted_rate": float(np.mean(accepted)),
        "accepted_identity_precision": _safe_div(np.sum(resolved_correct & accepted), np.sum(accepted)),
        "dustbin_accuracy": float(np.mean(predicted_dustbin == target_dustbin)),
        "dustbin_precision": _safe_div(dustbin_true_positive, dustbin_true_positive + dustbin_false_positive),
        "dustbin_recall": _safe_div(dustbin_true_positive, dustbin_true_positive + dustbin_false_negative),
        "dustbin_specificity": _safe_div(dustbin_true_negative, dustbin_true_negative + dustbin_false_positive),
        "wrong_maplet_rejection_auprc": binary_average_precision(target_dustbin, dustbin_probabilities),
        "identity_ranking": identity,
    }
    return {
        "rows": rows,
        "raw_scores": raw_scores,
        "accepted_scores": accepted_scores,
        "dustbin_probabilities": dustbin_probabilities,
        "predicted_track_ids": predicted_track_ids,
        "metrics": metrics,
    }


def _baseline_summary(
    store: LocalAssignmentFeatureStore,
    query_ids: Sequence[str],
    strategy: str,
) -> dict[str, object]:
    rows = _query_rows(store, query_ids)
    return summarize_assignment_strategy(
        candidates=store.subset_candidates(rows),
        correct_track_ids=store.correct_track_ids[rows],
        query_ids=[store.query_ids[int(row)] for row in rows],
        scores=np.asarray(store.probe[f"strategy__{strategy}"], dtype=np.float32)[rows],
    )


def _pose_inputs(store: LocalAssignmentFeatureStore, rows: np.ndarray):
    observations = [
        ColmapTrackObservation(
            track_id=int(store.correct_track_ids[row]),
            image_id=str(store.query_ids[row]),
            point2d_idx=int(store.query_point2d_indices[row]),
            xy=(float(store.query_xy[row, 0]), float(store.query_xy[row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for row in rows.tolist()
    ]
    return observations, [store.query_ids[int(row)] for row in rows]


def _evaluate_pose_set(
    *,
    store: LocalAssignmentFeatureStore,
    prediction: dict[str, object],
    baseline_strategy: str,
    cameras,
    images_by_name,
    reprojection_error_px: float,
    iterations: int,
) -> tuple[dict[str, object], dict[str, object]]:
    rows = np.asarray(prediction["rows"], dtype=np.int64)
    candidates = store.subset_candidates(rows)
    observations, query_ids = _pose_inputs(store, rows)
    model_summary, model_rows = _evaluate_pose_strategy(
        strategy="local_assignment_matcher_accepted",
        scores=np.asarray(prediction["accepted_scores"], dtype=np.float32),
        candidates=candidates,
        query_observations=observations,
        query_ids=query_ids,
        landmark_index=store.landmark_index,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
    )
    baseline_scores = np.asarray(store.probe[f"strategy__{baseline_strategy}"], dtype=np.float32)[rows]
    baseline_summary, baseline_rows = _evaluate_pose_strategy(
        strategy=baseline_strategy,
        scores=baseline_scores,
        candidates=candidates,
        query_observations=observations,
        query_ids=query_ids,
        landmark_index=store.landmark_index,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
    )
    return (
        {"local_assignment_matcher_accepted": model_summary, baseline_strategy: baseline_summary},
        {"local_assignment_matcher_accepted": model_rows, baseline_strategy: baseline_rows},
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.gradient_accumulation) <= 0:
        raise ValueError("--gradient_accumulation must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    store = LocalAssignmentFeatureStore(
        probe_arrays=Path(args.probe_arrays),
        real_feature_cache=Path(args.real_feature_cache),
        query_global_cache=Path(args.query_global_cache),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        maplet_support_index=Path(args.maplet_support_index),
        max_support_views=int(args.max_support_views),
    )
    query_ids = store.unique_query_ids
    train_end = int(args.train_query_count)
    validation_end = train_end + int(args.validation_query_count)
    if train_end <= 0 or validation_end >= len(query_ids):
        raise ValueError("train/validation counts must leave a non-empty contiguous test block")
    split = {
        "strategy": "contiguous_temporal_blocks_v1",
        "train": list(query_ids[:train_end]),
        "validation": list(query_ids[train_end:validation_end]),
        "test": list(query_ids[validation_end:]),
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
        query_layers=int(args.query_layers),
        track_layers=int(args.track_layers),
        dropout=float(args.dropout),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
        edge_prior_feature_index=int(args.edge_prior_feature_index),
        edge_prior_scale=float(args.edge_prior_scale),
        edge_prior_center=float(args.edge_prior_center),
        query_dustbin_initial_bias=float(args.query_dustbin_initial_bias),
    )
    device = torch.device(str(args.device) if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model = LocalAssignmentMatcher(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    artifact_hashes = {
        "probe_arrays": file_sha256_short(Path(args.probe_arrays)),
        "real_feature_cache": file_sha256_short(Path(args.real_feature_cache)),
        "query_global_cache": file_sha256_short(Path(args.query_global_cache)),
        "projected_landmark_bank": file_sha256_short(Path(args.projected_landmark_bank)),
        "maplet_support_index": file_sha256_short(Path(args.maplet_support_index)),
    }
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "training_log.jsonl"
    generator = np.random.default_rng(int(args.seed))
    with log_path.open("w") as log_handle:
        initial_validation = evaluate_model(
            model,
            store,
            split["validation"],
            device=str(device),
            pair_loss_weight=float(args.pair_loss_weight),
            no_match_loss_weight=float(args.no_match_loss_weight),
            no_match_threshold=float(args.no_match_threshold),
        )
        initial_metrics = dict(initial_validation["metrics"])
        initial_identity = dict(initial_metrics["identity_ranking"])
        best_key = (
            float(initial_identity["recall_at_1"]),
            float(initial_metrics["accepted_identity_precision"]),
            float(initial_metrics["wrong_maplet_rejection_auprc"]),
        )
        best_epoch = -1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "epoch": -1,
                "validation_metrics": initial_metrics,
                "promotion_key": list(best_key),
                "seed": int(args.seed),
                "split": split,
                "artifact_hashes": artifact_hashes,
            },
            checkpoint_path,
        )
        log_handle.write(
            json.dumps(
                {
                    "epoch": -1,
                    "phase": "frozen_prior_baseline",
                    "learning_rate": 0.0,
                    "train_mean_loss": None,
                    "validation": initial_metrics,
                    "promotion_key": list(best_key),
                },
                sort_keys=True,
            )
            + "\n"
        )
        log_handle.flush()
        for epoch in range(int(args.epochs)):
            dustbin_only = int(epoch) < int(args.dustbin_only_epochs)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(not dustbin_only or name.startswith("no_match_head"))
            phase_learning_rate = (
                float(args.dustbin_learning_rate) if dustbin_only else float(args.residual_learning_rate)
            )
            for group in optimizer.param_groups:
                group["lr"] = phase_learning_rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_losses: list[float] = []
            order = generator.permutation(len(split["train"]))
            for step, position in enumerate(order.tolist()):
                episode = store.episode(split["train"][int(position)]).to(device)
                output = model(episode)
                if dustbin_only:
                    no_match_targets = (
                        episode.target_track_indices == int(episode.track_features.shape[0])
                    ).to(dtype=output["no_match_logits"].dtype)
                    loss = F.binary_cross_entropy_with_logits(output["no_match_logits"], no_match_targets)
                else:
                    loss, _metrics = local_assignment_loss(
                        output,
                        episode,
                        pair_loss_weight=float(args.pair_loss_weight),
                        no_match_loss_weight=float(args.no_match_loss_weight),
                    )
                (loss / int(args.gradient_accumulation)).backward()
                train_losses.append(float(loss.detach().cpu().item()))
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
                no_match_threshold=float(args.no_match_threshold),
            )
            val_metrics = dict(validation["metrics"])
            identity = dict(val_metrics["identity_ranking"])
            key = (
                float(identity["recall_at_1"]),
                float(val_metrics["accepted_identity_precision"]),
                float(val_metrics["wrong_maplet_rejection_auprc"]),
            )
            record = {
                "epoch": int(epoch),
                "phase": "dustbin_only" if dustbin_only else "assignment_residual",
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_mean_loss": float(np.mean(train_losses)),
                "validation": val_metrics,
                "promotion_key": list(key),
            }
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()
            print(json.dumps({"epoch": epoch, "train_loss": record["train_mean_loss"], "key": key}, sort_keys=True))
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = int(epoch)
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "epoch": int(epoch),
                        "validation_metrics": val_metrics,
                        "promotion_key": list(key),
                        "seed": int(args.seed),
                        "split": split,
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
        no_match_threshold=float(args.no_match_threshold),
    )
    test = evaluate_model(
        model,
        store,
        split["test"],
        device=str(device),
        pair_loss_weight=float(args.pair_loss_weight),
        no_match_loss_weight=float(args.no_match_loss_weight),
        no_match_threshold=float(args.no_match_threshold),
    )
    baselines = {
        split_name: {
            strategy: _baseline_summary(store, split[split_name], strategy)
            for strategy in ("coarse_prototype", "all_support_top2_mean", "alike_support_top4_mean")
        }
        for split_name in ("validation", "test")
    }

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    pose: dict[str, object] = {}
    pose_rows: dict[str, object] = {}
    for split_name, prediction in (("validation", validation), ("test", test)):
        split_pose, split_rows = _evaluate_pose_set(
            store=store,
            prediction=prediction,
            baseline_strategy="alike_support_top4_mean",
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        pose[split_name] = split_pose
        pose_rows[split_name] = split_rows

    prediction_path = output_dir / "predictions.npz"
    np.savez(
        prediction_path,
        validation_rows=np.asarray(validation["rows"], dtype=np.int64),
        validation_raw_scores=np.asarray(validation["raw_scores"], dtype=np.float32),
        validation_accepted_scores=np.asarray(validation["accepted_scores"], dtype=np.float32),
        validation_dustbin_probabilities=np.asarray(validation["dustbin_probabilities"], dtype=np.float32),
        test_rows=np.asarray(test["rows"], dtype=np.int64),
        test_raw_scores=np.asarray(test["raw_scores"], dtype=np.float32),
        test_accepted_scores=np.asarray(test["accepted_scores"], dtype=np.float32),
        test_dustbin_probabilities=np.asarray(test["dustbin_probabilities"], dtype=np.float32),
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "s4_l1_assignment_only",
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "promotion_key": None if best_key is None else list(best_key),
        "config": config.to_dict(),
        "split": split,
        "validation": validation["metrics"],
        "test": test["metrics"],
        "baselines": baselines,
        "pose": pose,
        "runtime_seconds": float(time.time() - start),
        "limitations": [
            "L1 query nodes are held-out GT SfM observations; detector-point inference is not evaluated yet",
            "global mapper and ALIKE are frozen; only support selection/assignment/dustbin are trained",
            "full182 and val90 have prior development exposure and are not untouched test sets",
            "pose uses token/observation centers only; no local pixel update is active",
        ],
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "split": str(split_path),
            "training_log": str(log_path),
            "predictions": str(prediction_path),
            "pose_rows": str(pose_rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
