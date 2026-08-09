"""Train G18 typed surface likelihood on same-query frozen candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    FEATURE_NAMES,
    SurfacePoseLikelihoodConfig,
    ViewGeometryConditionedSurfaceLikelihood,
    listwise_surface_pose_loss,
    save_surface_pose_likelihood,
)


def _load(paths: list[str]) -> dict[str, np.ndarray]:
    collections = {}
    metadata = []
    for name in paths:
        path = Path(name)
        with np.load(path, allow_pickle=False) as data:
            item = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            meta = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if meta.get("artifact_type") != "goal_maplet_surface_likelihood_samples_v1":
            raise ValueError("not a G18 surface-likelihood sample artifact")
        metadata.append((path, meta))
        for key, value in item.items():
            collections.setdefault(key, []).append(value)
    result = {key: np.concatenate(value, axis=0) for key, value in collections.items()}
    result["_metadata"] = metadata
    return result


def _lineage(data: dict[str, np.ndarray]) -> dict[str, object]:
    metadata = [value for _path, value in data["_metadata"]]
    keys = (
        "physical_map_sha256", "canonical_field_sha256", "surface_mapper_sha256",
        "physical_instance_readout_sha256", "candidate_count",
    )
    result = {}
    for key in keys:
        values = {json.dumps(item.get(key), sort_keys=True) for item in metadata}
        if len(values) != 1:
            raise ValueError(f"G18 sample lineage differs: {key}")
        result[key] = metadata[0].get(key)
    return result


def _metric_summary(
    model: ViewGeometryConditionedSurfaceLikelihood,
    data: dict[str, np.ndarray],
    *,
    device: str,
) -> dict[str, object]:
    translation = np.asarray(data["translation_m"], dtype=np.float64)
    rotation = np.asarray(data["rotation_deg"], dtype=np.float64)
    selected, confidences, entropies, null_probabilities, nll = [], [], [], [], []
    top3_success = []
    model.eval()
    with torch.inference_mode():
        for index in range(translation.shape[0]):
            feature = torch.from_numpy(
                np.asarray(data["token_feature"][index:index + 1], dtype=np.float32)
            ).to(device)
            summary = torch.from_numpy(
                np.asarray(data["query_summary"][index:index + 1], dtype=np.float32)
            ).to(device)
            validity = torch.from_numpy(
                np.asarray(data["candidate_valid"][index:index + 1], dtype=bool)
            ).to(device)
            candidate, null, _event = model.posterior(feature, summary, validity)
            candidate_np = candidate[0].cpu().numpy()
            null_value = float(null[0].cpu())
            logits_probability = np.r_[candidate_np, null_value]
            choice = int(np.argmax(logits_probability))
            selected.append(choice)
            candidate_confidence = float(np.max(candidate_np))
            entropy = float(-np.sum(
                logits_probability * np.log(np.maximum(logits_probability, 1.0e-12))
            ))
            normalized_entropy = entropy / max(float(np.log(logits_probability.size)), 1.0e-12)
            # Risk confidence always refers to accepting a pose.  A confident
            # null prediction must not become a confidently accepted pose.
            confidences.append(candidate_confidence * (1.0 - normalized_entropy))
            entropies.append(entropy)
            null_probabilities.append(null_value)
            target = int(data["target_index"][index])
            nll.append(float(-np.log(max(logits_probability[target], 1.0e-8))))
            top3 = np.argsort(-candidate_np)[:3]
            top3_success.append(bool(np.any(
                (translation[index, top3] <= 0.5) & (rotation[index, top3] <= 5.0)
            )))
    selected = np.asarray(selected, dtype=np.int64)
    nonnull = selected < translation.shape[1]
    safe = np.minimum(selected, translation.shape[1] - 1)
    selected_translation = translation[np.arange(translation.shape[0]), safe]
    selected_rotation = rotation[np.arange(rotation.shape[0]), safe]
    selected_translation[~nonnull] = np.inf
    selected_rotation[~nonnull] = np.inf
    finite_translation = selected_translation[np.isfinite(selected_translation)]
    catastrophic_pose = nonnull & (
        (selected_translation > 5.0) | (selected_rotation > 30.0)
    )
    risk_coverage = {}
    confidence = np.asarray(confidences)
    eligible = np.flatnonzero(nonnull)
    eligible = eligible[np.argsort(-confidence[eligible])]
    for coverage in (1.0, 0.9, 0.8, 0.5):
        requested = max(1, int(np.ceil(float(coverage) * confidence.size)))
        accepted = eligible[:requested]
        if accepted.size == 0:
            risk_coverage[f"coverage_{coverage:.1f}"] = {
                "requested_count": int(requested),
                "accepted_count": 0,
                "realized_coverage": 0.0,
                "strict_success": None,
                "strict_success_yield": 0.0,
                "catastrophic_pose_rate": None,
            }
            continue
        success = (
            (selected_translation[accepted] <= 0.5)
            & (selected_rotation[accepted] <= 5.0)
        )
        risk_coverage[f"coverage_{coverage:.1f}"] = {
            "requested_count": int(requested),
            "accepted_count": int(accepted.size),
            "realized_coverage": float(accepted.size / confidence.size),
            "strict_success": float(np.mean(success)),
            "strict_success_yield": float(np.sum(success) / confidence.size),
            "catastrophic_pose_rate": float(np.mean(catastrophic_pose[accepted])),
        }
    return {
        "query_count": int(translation.shape[0]),
        "translation_m": {
            "median": float(np.median(finite_translation)) if finite_translation.size else None,
            "p90": float(np.percentile(finite_translation, 90.0)) if finite_translation.size else None,
        },
        "rotation_deg": {
            "median": float(np.median(selected_rotation[np.isfinite(selected_rotation)])) if np.any(np.isfinite(selected_rotation)) else None,
            "p90": float(np.percentile(selected_rotation[np.isfinite(selected_rotation)], 90.0)) if np.any(np.isfinite(selected_rotation)) else None,
        },
        "strict_0.5m_5deg": float(np.mean((selected_translation <= 0.5) & (selected_rotation <= 5.0))),
        "success_1m_10deg": float(np.mean((selected_translation <= 1.0) & (selected_rotation <= 10.0))),
        "catastrophic_rate": float(np.mean(catastrophic_pose)),
        "catastrophic_or_abstain_rate": float(np.mean(catastrophic_pose | ~nonnull)),
        "top3_strict_success": float(np.mean(top3_success)),
        "null_selection_rate": float(np.mean(~nonnull)),
        "mean_null_probability": float(np.mean(null_probabilities)),
        "posterior_entropy_median": float(np.median(entropies)),
        "listwise_nll": float(np.mean(nll)),
        "risk_coverage": risk_coverage,
    }


def _selection_key(metrics: dict[str, object]) -> tuple[float, ...]:
    translation = metrics["translation_m"]
    return (
        float(metrics["catastrophic_rate"]),
        float(metrics["catastrophic_or_abstain_rate"]),
        float(translation["p90"] if translation["p90"] is not None else 1.0e6),
        -float(metrics["top3_strict_success"]),
        -float(metrics["strict_0.5m_5deg"]),
        float(metrics["listwise_nll"]),
    )


def _trajectory_balanced_order(
    trajectory_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an epoch with equal trajectory mass and near-original length."""

    trajectories = np.asarray(trajectory_ids).astype(str)
    names = sorted(set(trajectories.tolist()))
    per_trajectory = int(np.ceil(trajectories.size / max(len(names), 1)))
    selected = []
    for name in names:
        group = np.flatnonzero(trajectories == name)
        if group.size == 0:
            continue
        repeats = int(np.ceil(per_trajectory / group.size))
        values = np.concatenate([rng.permutation(group) for _ in range(repeats)])
        selected.extend(values[:per_trajectory].tolist())
    return rng.permutation(np.asarray(selected, dtype=np.int64))


def _baseline_metrics(data: dict[str, np.ndarray], policy: str) -> dict[str, object]:
    translation = np.asarray(data["translation_m"], dtype=np.float64)
    rotation = np.asarray(data["rotation_deg"], dtype=np.float64)
    if policy == "frozen_candidate_top1":
        selected = np.zeros((translation.shape[0],), dtype=np.int64)
    elif policy == "fixed_grid_cosine":
        feature = np.asarray(data["token_feature"], dtype=np.float32)
        cosine = feature[..., FEATURE_NAMES.index("cosine")]
        valid = feature[..., FEATURE_NAMES.index("render_valid")]
        score = np.mean(cosine * valid, axis=2)
        score[~np.asarray(data["candidate_valid"], dtype=bool)] = -np.inf
        selected = np.argmax(score, axis=1)
    else:
        raise ValueError("unknown G18 baseline policy")
    chosen_translation = translation[np.arange(translation.shape[0]), selected]
    chosen_rotation = rotation[np.arange(rotation.shape[0]), selected]
    return {
        "policy": policy,
        "translation_m": {
            "median": float(np.median(chosen_translation)),
            "p90": float(np.percentile(chosen_translation, 90.0)),
        },
        "rotation_deg": {
            "median": float(np.median(chosen_rotation)),
            "p90": float(np.percentile(chosen_rotation, 90.0)),
        },
        "strict_0.5m_5deg": float(np.mean(
            (chosen_translation <= 0.5) & (chosen_rotation <= 5.0)
        )),
        "success_1m_10deg": float(np.mean(
            (chosen_translation <= 1.0) & (chosen_rotation <= 10.0)
        )),
        "catastrophic_rate": float(np.mean(
            (chosen_translation > 5.0) | (chosen_rotation > 30.0)
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_samples", nargs="+", required=True)
    parser.add_argument("--selection_samples", nargs="+", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--typed_weight", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--trajectory_balanced", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary_path = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary_path.exists()):
        raise FileExistsError("refusing to overwrite G18 likelihood")
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    train = _load(list(args.train_samples))
    selection = _load(list(args.selection_samples))
    train_lineage, selection_lineage = _lineage(train), _lineage(selection)
    for key in train_lineage:
        if train_lineage[key] != selection_lineage[key]:
            raise ValueError(f"G18 train/selection lineage differs: {key}")
    train_trajectories = set(np.asarray(train["trajectory_ids"]).astype(str).tolist())
    selection_trajectories = set(np.asarray(selection["trajectory_ids"]).astype(str).tolist())
    if train_trajectories & selection_trajectories:
        raise ValueError("G18 train and selection trajectories overlap")
    model = ViewGeometryConditionedSurfaceLikelihood(
        SurfacePoseLikelihoodConfig(
            hidden_dim=int(args.hidden_dim),
            candidate_conditioned_null=True,
            candidate_contrast_scale=3.0,
            candidate_disagreement_weight=True,
        ),
    ).to(str(args.device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay),
    )
    best_state, best_key, history = None, None, []
    for epoch in range(max(int(args.epochs), 1)):
        model.train()
        order = (
            _trajectory_balanced_order(train["trajectory_ids"], rng)
            if bool(args.trajectory_balanced)
            else rng.permutation(train["target_index"].shape[0])
        )
        losses, listwise, typed = [], [], []
        for index in order.tolist():
            feature = torch.from_numpy(
                np.asarray(train["token_feature"][index:index + 1], dtype=np.float32)
            ).to(str(args.device))
            query_summary = torch.from_numpy(
                np.asarray(train["query_summary"][index:index + 1], dtype=np.float32)
            ).to(str(args.device))
            target = torch.from_numpy(
                np.asarray(train["target_index"][index:index + 1], dtype=np.int64)
            ).to(str(args.device))
            typed_target = torch.from_numpy(
                np.asarray(train["typed_target"][index:index + 1], dtype=np.int64)
            ).to(str(args.device))
            teacher_array = np.asarray(
                train["teacher_weight"][index:index + 1], dtype=np.float32,
            )
            if teacher_array.ndim == 2:
                teacher_array = np.repeat(
                    teacher_array[:, None, :], typed_target.shape[1], axis=1,
                )
            if teacher_array.shape != tuple(typed_target.shape):
                raise ValueError("typed teacher weight shape differs")
            teacher_weight = torch.from_numpy(teacher_array).to(str(args.device))
            candidate_valid = torch.from_numpy(
                np.asarray(train["candidate_valid"][index:index + 1], dtype=bool)
            ).to(str(args.device))
            teacher_weight = teacher_weight * candidate_valid[..., None].to(teacher_weight.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss, report = listwise_surface_pose_loss(
                model,
                feature,
                query_summary,
                target,
                typed_target,
                typed_sample_weight=teacher_weight,
                candidate_valid=candidate_valid,
                typed_weight=float(args.typed_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            listwise.append(report["listwise_nll"])
            typed.append(report["typed_nll"])
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(args.epochs):
            metrics = _metric_summary(model, selection, device=str(args.device))
            key = _selection_key(metrics)
            history.append({
                "epoch": int(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "train_listwise_nll": float(np.mean(listwise)),
                "train_typed_nll": float(np.mean(typed)),
                "selection": metrics,
            })
            print(json.dumps(history[-1]), flush=True)
            if best_key is None or key < best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("G18 training produced no selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    train_metrics = _metric_summary(model, train, device=str(args.device))
    selection_metrics = _metric_summary(model, selection, device=str(args.device))
    metadata = {
        **train_lineage,
        "candidate_set_frozen_before_scoring": True,
        "fixed_full_query_denominator": True,
        "same_query_listwise_normalization": True,
        "typed_null_hypothesis": True,
        "candidate_conditioned_typed_null": True,
        "same_token_candidate_contrast": True,
        "candidate_independent_denominator_shared_disagreement_weight": True,
        "training_trajectory_ids": sorted(train_trajectories),
        "selection_trajectory_ids": sorted(selection_trajectories),
        "training_sample_sha256": [file_sha256(Path(value)) for value in args.train_samples],
        "selection_sample_sha256": [file_sha256(Path(value)) for value in args.selection_samples],
        "teacher_use": "offline_typed_supervision_weights_only",
        "teacher_embeddings_stored": False,
        "trajectory_balanced_training": bool(args.trajectory_balanced),
        "selection_priority": (
            "catastrophic_pose,catastrophic_or_abstain,p90,"
            "top3_strict,top1_strict,nll"
        ),
    }
    save_surface_pose_likelihood(model, output, metadata=metadata)
    report = {
        "stage": "g18_view_geometry_conditioned_surface_pose_likelihood",
        "output_model": str(output),
        "model_sha256": file_sha256(output),
        "history": history,
        "train": train_metrics,
        "selection": selection_metrics,
        "selection_baselines": {
            policy: _baseline_metrics(selection, policy)
            for policy in ("frozen_candidate_top1", "fixed_grid_cosine")
        },
        "metadata": metadata,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
