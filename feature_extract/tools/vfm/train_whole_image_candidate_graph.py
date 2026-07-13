"""Train a whole-image candidate/null graph on frozen P23 evidence."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.whole_image_candidate_graph import (
    WholeImageCandidateGraph,
    WholeImageLatentCandidateGraph,
    set_identity_nll,
)


def _load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files if k != "metadata_json"}
        metadata = {} if "metadata_json" not in z.files else json.loads(str(z["metadata_json"].item()))
    return arrays, metadata


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(np.count_nonzero(labels))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positive_count)


def _metrics(candidate_logits, null_logits, labels):
    with torch.no_grad():
        probs = torch.softmax(torch.cat([candidate_logits, null_logits.unsqueeze(2)], 2), 2)
        candidate = probs[:, :, :-1]
        null = probs[:, :, -1]
        any_positive = torch.any(labels, dim=2)
        correct_mass = torch.sum(candidate * labels.float(), dim=2)
        target_mass = torch.where(any_positive, correct_mass, null)
        order = torch.argsort(candidate, dim=2, descending=True)
        ranked = torch.gather(labels, 2, order)
        top1 = ranked[:, :, 0]
        state_unknown = null >= torch.max(candidate, dim=2).values
        state_correct = torch.where(any_positive, top1 & ~state_unknown, state_unknown)
        result = {
            "identity_nll": float(torch.mean(-torch.log(target_mass.clamp_min(1e-12))).cpu()),
            "candidate_top1_correct_rate": float(torch.mean(top1.float()).cpu()),
            "state_top1_correct_rate": float(torch.mean(state_correct.float()).cpu()),
            "unknown_top1_rate": float(torch.mean(state_unknown.float()).cpu()),
            "correct_available_rate": float(torch.mean(any_positive.float()).cpu()),
        }
        available_count = int(torch.count_nonzero(any_positive).cpu())
        for rank in (1, 5, 10, 20):
            capped = min(rank, candidate.shape[2])
            recalled = torch.any(ranked[:, :, :capped], dim=2)
            result[f"candidate_recall_at_{rank}_given_available"] = (
                float(torch.mean(recalled[any_positive].float()).cpu())
                if available_count
                else float("nan")
            )
        result["null_average_precision"] = _average_precision(
            null.cpu().numpy(), (~any_positive).cpu().numpy()
        )
        return result


def _require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _validate_manifests(
    args,
    candidate_meta: dict,
    score_meta: dict,
    landmark_meta: dict,
) -> dict:
    hashes = {
        "candidate_artifact_sha256": file_sha256_short(Path(args.candidate_artifact)),
        "score_artifact_sha256": file_sha256_short(Path(args.score_artifact)),
        "proposals_sha256": file_sha256_short(Path(args.proposals)),
        "landmark_bank_sha256": file_sha256_short(Path(args.landmark_bank)),
        "split_json_sha256": file_sha256_short(Path(args.split_json)),
    }
    _require_equal(
        "candidate artifact format",
        candidate_meta.get("format"),
        "detector_maplet_geometry_features_v1",
    )
    _require_equal(
        "score artifact format",
        score_meta.get("format"),
        "candidate_maplet_ensemble_scores_v2",
    )
    score_manifest = dict(score_meta.get("data_manifest") or {})
    required_splits = {"train", "validation", "test"}
    exported_splits = set(score_meta.get("prediction_splits") or ())
    if not required_splits.issubset(exported_splits):
        raise ValueError(
            "score artifact must contain frozen train/validation/test predictions; "
            f"found {sorted(exported_splits)}"
        )
    for metadata, label in (
        (candidate_meta, "candidate artifact"),
        (score_manifest, "score artifact"),
    ):
        _require_equal(
            f"{label} proposals hash",
            metadata.get("proposals_sha256"),
            hashes["proposals_sha256"],
        )
        _require_equal(
            f"{label} landmark bank hash",
            metadata.get("projected_landmark_bank_sha256"),
            hashes["landmark_bank_sha256"],
        )
    _require_equal(
        "score feature artifact hash",
        score_manifest.get("feature_artifact_sha256"),
        hashes["candidate_artifact_sha256"],
    )
    _require_equal(
        "candidate/score top-k",
        candidate_meta.get("candidate_top_k"),
        score_manifest.get("candidate_top_k"),
    )
    if not landmark_meta.get("descriptor_space_id"):
        raise ValueError("landmark bank is missing descriptor_space_id")
    return hashes


def _validate_latent_manifest(
    path: Path,
    latent_meta: dict,
    score_meta: dict,
) -> str:
    _require_equal(
        "candidate latent format",
        latent_meta.get("format"),
        "candidate_maplet_view_latents_v1",
    )
    _require_equal(
        "candidate latent data manifest",
        latent_meta.get("data_manifest"),
        score_meta.get("data_manifest"),
    )
    _require_equal(
        "candidate latent prediction splits",
        set(latent_meta.get("prediction_splits") or ()),
        {"train", "validation", "test"},
    )
    checkpoint_sha256 = latent_meta.get("checkpoint_sha256")
    if checkpoint_sha256 not in set(score_meta.get("checkpoint_sha256") or ()):
        raise ValueError("candidate latent checkpoint is not part of the score ensemble")
    return file_sha256_short(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument(
        "--candidate_latents",
        default=None,
        help="optional single-checkpoint per-support-view candidate latent artifact",
    )
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--model_dim", type=int, default=96)
    parser.add_argument("--neighbor_k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_width", type=float, default=1024.0)
    parser.add_argument("--image_height", type=float, default=576.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if int(args.batch_size) <= 0 or int(args.neighbor_k) <= 0:
        raise ValueError("batch_size and neighbor_k must be positive")
    if float(args.image_width) <= 1.0 or float(args.image_height) <= 1.0:
        raise ValueError("image dimensions must be greater than one pixel")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    candidates, candidate_meta = _load(Path(args.candidate_artifact))
    scores, score_meta = _load(Path(args.score_artifact))
    latent_arrays = None
    latent_meta = None
    if args.candidate_latents is not None:
        latent_arrays, latent_meta = _load(Path(args.candidate_latents))
    proposals, _ = _load(Path(args.proposals))
    landmark, landmark_meta = load_landmark_index_npz(Path(args.landmark_bank))
    input_hashes = _validate_manifests(
        args, candidate_meta, score_meta, landmark_meta
    )
    if latent_arrays is not None and latent_meta is not None:
        input_hashes["candidate_latents_sha256"] = _validate_latent_manifest(
            Path(args.candidate_latents), latent_meta, score_meta
        )
    selected_rows = candidates["selected_rows"].astype(np.int64)
    selected_columns = candidates["selected_columns"].astype(np.int64)
    if selected_columns.ndim != 2:
        raise ValueError("selected_columns must have shape (rows, candidates)")
    row_count, candidate_count = selected_columns.shape
    query_count = int(candidate_meta.get("query_points_per_image", 0))
    if query_count <= 0 or row_count % query_count:
        raise ValueError("candidate rows do not form fixed whole-image blocks")
    if int(candidate_meta.get("candidate_top_k", 0)) != candidate_count:
        raise ValueError("candidate count disagrees with candidate artifact manifest")
    for key in ("features", "labels", "valid_edges"):
        if candidates[key].shape[:2] != selected_columns.shape:
            raise ValueError(f"candidate array {key!r} is not aligned")
    if not np.all(candidates["valid_edges"]):
        raise ValueError("whole-image graph v1 requires a dense valid candidate pool")
    query_ids = proposals["query_ids"][selected_rows].astype(str)
    query_xy = proposals["xy"][selected_rows].astype(np.float32)
    image_count = len(query_ids) // query_count
    query_id_blocks = query_ids.reshape(image_count, query_count)
    if not np.all(query_id_blocks == query_id_blocks[:, :1]):
        raise ValueError("candidate rows are not contiguous whole images")
    static = candidates["features"].astype(np.float32)
    if static.ndim != 3 or static.shape[:2] != (row_count, candidate_count):
        raise ValueError("candidate feature tensor is not aligned")
    if not np.all(np.isfinite(static)):
        raise ValueError("candidate features contain non-finite values")
    prefix = "ensemble"
    typed = np.stack(
        [
            scores[f"{prefix}__set_candidate_probability"],
            scores[f"{prefix}__anchor_assignment_probability"],
            scores[f"{prefix}__geometry_p01px"],
            scores[f"{prefix}__geometry_p02px"],
            scores[f"{prefix}__geometry_p05px"],
            scores[f"{prefix}__candidate_visibility_probability"],
            scores[f"{prefix}__support_view_probability_0"],
            scores[f"{prefix}__support_view_probability_1"],
        ],
        axis=2,
    ).astype(np.float32)
    if typed.shape[:2] != (row_count, candidate_count):
        raise ValueError("typed score evidence is not aligned with candidates")
    latent_mode = latent_arrays is not None
    scalar_typed_count = 6 if latent_mode else typed.shape[2]
    features = np.concatenate([static, typed[:, :, :scalar_typed_count]], axis=2).reshape(
        image_count, query_count, candidate_count, -1
    )
    candidate_view_latents = None
    support_view_probability = None
    support_view_mask = None
    latent_mean = None
    latent_std = None
    if latent_mode:
        assert latent_arrays is not None and latent_meta is not None
        view_count = int(latent_meta.get("support_view_count", 0))
        expected_view_count = typed.shape[2] - scalar_typed_count
        if view_count != expected_view_count or view_count <= 0:
            raise ValueError("latent support-view count disagrees with score evidence")
        latent_keys = tuple(
            f"candidate_view_embedding_{view_rank}" for view_rank in range(view_count)
        )
        if not all(key in latent_arrays for key in latent_keys):
            raise ValueError("candidate latent artifact is missing support views")
        latent_values = np.stack(
            [latent_arrays[key] for key in latent_keys], axis=1
        ).astype(np.float32)
        expected_edge_count = row_count * candidate_count
        if latent_values.shape[:2] != (expected_edge_count, view_count):
            raise ValueError("candidate latent arrays are not aligned with candidate edges")
        if int(latent_meta.get("model_dim", 0)) != latent_values.shape[2]:
            raise ValueError("candidate latent dimension disagrees with its manifest")
        if not np.all(np.isfinite(latent_values)):
            raise ValueError("candidate latent artifact contains missing/non-finite values")
        candidate_view_latents = latent_values.reshape(
            image_count,
            query_count,
            candidate_count,
            view_count,
            latent_values.shape[2],
        )
        support_view_probability = typed[:, :, scalar_typed_count:].reshape(
            image_count, query_count, candidate_count, view_count
        )
        support_view_mask = np.isfinite(support_view_probability)
        if not np.all(support_view_mask):
            raise ValueError("whole-image latent graph v1 requires every support view")
        if np.any(support_view_probability < 0.0):
            raise ValueError("support-view probabilities must be non-negative")
        support_mass = np.sum(support_view_probability, axis=3)
        if not np.allclose(support_mass, 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError("support-view probabilities are not normalized")
    labels = candidates["labels"].astype(bool).reshape(
        image_count, query_count, candidate_count
    )
    prior = scores[f"{prefix}__set_candidate_probability"].astype(np.float32).reshape(
        image_count, query_count, candidate_count
    )
    dustbin_full = scores[
        f"{prefix}__set_dustbin_probability_DIAGNOSTIC_ONLY"
    ].astype(np.float32)
    if dustbin_full.shape != (row_count, candidate_count):
        raise ValueError("dustbin score tensor is not aligned with candidates")
    if not np.allclose(dustbin_full, dustbin_full[:, :1], atol=1e-7, rtol=1e-6):
        raise ValueError("per-query dustbin probability is not constant over candidates")
    null = dustbin_full[:, 0].reshape(image_count, query_count)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(null)):
        raise ValueError("full-split graph inputs contain non-finite values")
    if np.any(prior < 0.0) or np.any(null < 0.0):
        raise ValueError("candidate/null prior probabilities must be non-negative")
    prior_mass = np.sum(prior, axis=2) + null
    if not np.allclose(prior_mass, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError(
            "candidate/null prior probability mass is not normalized: "
            f"range=({prior_mass.min():.8f}, {prior_mass.max():.8f})"
        )
    bank_rows = np.take_along_axis(
        proposals["bank_row_indices"][selected_rows], selected_columns, axis=1
    ).reshape(image_count, query_count * candidate_count)
    xyz = landmark.xyz[bank_rows]
    total_candidates = query_count * candidate_count
    if int(args.neighbor_k) > total_candidates:
        raise ValueError("neighbor_k exceeds candidates in a whole image")
    neighbor_indices = np.empty(
        (image_count, total_candidates, int(args.neighbor_k)), dtype=np.int64
    )
    for image_index in range(image_count):
        _distance, indices = cKDTree(xyz[image_index]).query(xyz[image_index], k=int(args.neighbor_k))
        neighbor_indices[image_index] = np.asarray(indices).reshape(
            total_candidates, int(args.neighbor_k)
        )
    neighbor_indices = neighbor_indices.reshape(
        image_count, query_count, candidate_count, int(args.neighbor_k)
    )
    xy_normalized = query_xy.reshape(image_count, query_count, 2) / np.asarray(
        [float(args.image_width) - 1.0, float(args.image_height) - 1.0],
        dtype=np.float32,
    )
    if np.any(xy_normalized < -1e-4) or np.any(xy_normalized > 1.0001):
        raise ValueError("query coordinates lie outside the declared image dimensions")
    split = json.loads(Path(args.split_json).read_text())
    image_ids = query_id_blocks[:, 0]
    indices_by_split = {
        name: np.flatnonzero(np.isin(image_ids, split[name]))
        for name in ("train", "validation", "test")
    }
    assigned = np.zeros((image_count,), dtype=np.int64)
    for name, indices in indices_by_split.items():
        if len(indices) == 0:
            raise ValueError(f"split {name!r} has no whole images")
        assigned[indices] += 1
    if not np.all(assigned == 1):
        raise ValueError("split JSON must partition every candidate image exactly once")
    train_indices = indices_by_split["train"]
    mean = features[train_indices].mean(axis=(0, 1, 2), keepdims=True)
    std = features[train_indices].std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-4)
    features = (features - mean) / std
    if candidate_view_latents is not None:
        latent_mean = candidate_view_latents[train_indices].mean(
            axis=(0, 1, 2, 3), keepdims=True
        )
        latent_std = candidate_view_latents[train_indices].std(
            axis=(0, 1, 2, 3), keepdims=True
        )
        latent_std = np.maximum(latent_std, 1e-4)
        candidate_view_latents = (
            candidate_view_latents - latent_mean
        ) / latent_std
    device = torch.device(args.device)
    if candidate_view_latents is None:
        model = WholeImageCandidateGraph(
            features.shape[-1], model_dim=int(args.model_dim)
        ).to(device)
    else:
        model = WholeImageLatentCandidateGraph(
            candidate_view_latents.shape[-1],
            features.shape[-1],
            model_dim=int(args.model_dim),
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    rng = random.Random(int(args.seed))

    def forward(index):
        ids = np.asarray(index, dtype=np.int64)
        common = (
            torch.from_numpy(xy_normalized[ids]).to(device),
            torch.from_numpy(neighbor_indices[ids]).to(device),
            torch.from_numpy(prior[ids]).to(device),
            torch.from_numpy(null[ids]).to(device),
        )
        if candidate_view_latents is None:
            return model(torch.from_numpy(features[ids]).to(device), *common)
        assert support_view_probability is not None and support_view_mask is not None
        return model(
            torch.from_numpy(candidate_view_latents[ids]).to(device),
            torch.from_numpy(features[ids]).to(device),
            torch.from_numpy(support_view_probability[ids]).to(device),
            torch.from_numpy(support_view_mask[ids]).to(device),
            *common,
        )

    best_state = None
    best_epoch = -1
    best_nll = float("inf")
    history = []
    baseline_metrics = {
        name: _metrics(
            torch.from_numpy(np.log(np.maximum(prior[indices], 1e-12))).to(device),
            torch.from_numpy(np.log(np.maximum(null[indices], 1e-12))).to(device),
            torch.from_numpy(labels[indices]).to(device),
        )
        for name, indices in indices_by_split.items()
    }
    for epoch in range(int(args.epochs)):
        model.train()
        order = train_indices.tolist()
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            batch_indices = order[start : start + int(args.batch_size)]
            candidate_logits, null_logits = forward(batch_indices)
            target = torch.from_numpy(labels[batch_indices]).to(device)
            loss = set_identity_nll(candidate_logits, null_logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits, validation_null = forward(indices_by_split["validation"])
            validation_metrics = _metrics(
                validation_logits,
                validation_null,
                torch.from_numpy(labels[indices_by_split["validation"]]).to(device),
            )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation_metrics})
        if validation_metrics["identity_nll"] < best_nll:
            best_nll = validation_metrics["identity_nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    metrics = {}
    posterior = {}
    with torch.no_grad():
        for name in ("train", "validation", "test"):
            logits, null_logits = forward(indices_by_split[name])
            metrics[name] = _metrics(logits, null_logits, torch.from_numpy(labels[indices_by_split[name]]).to(device))
            posterior[name] = torch.softmax(torch.cat([logits, null_logits.unsqueeze(2)], 2), 2).cpu().numpy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    torch.save(
        {
            "format": (
                "whole_image_latent_candidate_graph_v1"
                if latent_mode
                else "whole_image_candidate_graph_v1"
            ),
            "model_state_dict": best_state,
            "input_dim": features.shape[-1],
            "latent_dim": (
                None
                if candidate_view_latents is None
                else int(candidate_view_latents.shape[-1])
            ),
            "model_dim": int(args.model_dim),
            "feature_mean": mean,
            "feature_std": std,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "best_epoch": best_epoch,
            "input_hashes": input_hashes,
            "descriptor_space_id": landmark_meta.get("descriptor_space_id"),
        },
        checkpoint,
    )
    posterior_path = output / "posterior.npz"
    posterior_metadata = {
        "format": (
            "whole_image_latent_candidate_graph_posterior_v1"
            if latent_mode
            else "whole_image_candidate_graph_posterior_v1"
        ),
        "checkpoint_sha256": file_sha256_short(checkpoint),
        "input_hashes": input_hashes,
        "descriptor_space_id": landmark_meta.get("descriptor_space_id"),
        "query_count": query_count,
        "candidate_count": candidate_count,
    }
    np.savez_compressed(
        posterior_path,
        image_ids=image_ids,
        metadata_json=np.asarray(json.dumps(posterior_metadata, sort_keys=True), dtype=np.str_),
        **{f"{name}_indices": indices_by_split[name] for name in indices_by_split},
        **{f"{name}_posterior": posterior[name] for name in posterior},
    )
    metric_delta = {
        name: {
            key: metrics[name][key] - baseline_metrics[name][key]
            for key in metrics[name]
            if key in baseline_metrics[name]
            and np.isfinite(metrics[name][key])
            and np.isfinite(baseline_metrics[name][key])
        }
        for name in metrics
    }
    summary = {
        "stage": (
            "whole_image_latent_candidate_graph_v1"
            if latent_mode
            else "whole_image_candidate_graph_v1"
        ),
        "best_epoch": best_epoch,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "metric_delta_graph_minus_prior": metric_delta,
        "history": history,
        "protocol": {
            "target": "2px_set_valued_identity",
            "pose_input": False,
            "test_used_for_selection": False,
            "explicit_null": True,
            "query_graph": True,
            "candidate_latent_input": bool(latent_mode),
            "support_view_mixture": bool(latent_mode),
            "support_view_marginalization": (
                "logsumexp_at_candidate_logit" if latent_mode else None
            ),
            "landmark_xyz_knn": int(args.neighbor_k),
            "query_count": query_count,
            "candidate_count": candidate_count,
            "image_dimensions": [float(args.image_width), float(args.image_height)],
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        "inputs": {
            **input_hashes,
            "descriptor_space_id": landmark_meta.get("descriptor_space_id"),
        },
        "outputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256_short(checkpoint),
            "posterior": str(posterior_path),
            "posterior_sha256": file_sha256_short(posterior_path),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
