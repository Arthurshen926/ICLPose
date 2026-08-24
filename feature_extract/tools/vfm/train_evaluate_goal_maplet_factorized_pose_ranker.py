"""Train separate location, orientation, and joint full-token pose rankers.

The three heads share one spatial evidence encoder but receive different
listwise objectives.  This is a candidate-conditioned ranking diagnostic: the
network never receives pose values and never regresses an absolute pose.  GT
pose errors are used only to form the explicitly non-production train/eval
objectives and fixed route metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_local_supervision_dataset,
    _mask_for_routes,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _spearman,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS,
    FULLTOKEN_POSE_RANKING_CHANNELS,
    FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    TYPED_FULLTOKEN_POSE_RANKING_CHANNELS,
    TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS,
    QUERY_TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    FactorizedFullTokenCandidatePoseRanker,
    listwise_pose_ranking_loss,
    load_factorized_fulltoken_candidate_pose_ranker,
    load_fulltoken_candidate_pose_ranker,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256


REPORT_SCHEMA = "goal_maplet_factorized_fulltoken_pose_ranker_train_route_holdout_v1"


def _load_feature_artifact(
    path: Path,
    manifest_path: Path,
    *,
    dataset_path: Path,
    dataset_content_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text())
    contracts = {
        FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS: list(FULLTOKEN_POSE_RANKING_CHANNELS),
        TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS: list(
            TYPED_FULLTOKEN_POSE_RANKING_CHANNELS
        ),
        QUERY_TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS: list(
            QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS
        ),
    }
    semantics = str(manifest.get("feature_semantics", ""))
    if semantics not in contracts or manifest.get("feature_channels") != contracts[semantics]:
        raise ValueError("factorized full-token feature contract differs")
    resolved = Path(path).resolve()
    if (
        Path(str(manifest.get("feature_file", ""))).resolve() != resolved
        or manifest.get("feature_file_sha256") != file_sha256(resolved)
        or manifest.get("dataset_file_sha256") != file_sha256(dataset_path)
        or manifest.get("dataset_content_sha256") != dataset_content_sha256
    ):
        raise ValueError("factorized full-token feature lineage differs")
    features = np.load(resolved, mmap_mode="r", allow_pickle=False)
    if (
        features.dtype != np.float16
        or list(features.shape) != manifest.get("feature_shape")
        or features.shape[2:] != (len(contracts[semantics]), 36, 64)
    ):
        raise ValueError("factorized full-token feature array differs")
    return features, manifest


def _score_queries(
    model: FactorizedFullTokenCandidatePoseRanker,
    features: np.ndarray,
    query_rows: np.ndarray,
    *,
    batch_queries: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    result = {
        name: np.full((int(query_rows.size), int(features.shape[1])), -1.0e9, dtype=np.float32)
        for name in model.HEAD_NAMES
    }
    model.eval()
    with torch.no_grad():
        for begin in range(0, int(query_rows.size), int(batch_queries)):
            rows = query_rows[begin:begin + int(batch_queries)]
            value = torch.as_tensor(
                np.asarray(features[rows], dtype=np.float32), device=device,
            )
            batch, candidates = value.shape[:2]
            score = model(value.flatten(0, 1))
            for name in model.HEAD_NAMES:
                result[name][begin:begin + batch] = score[name].reshape(
                    batch, candidates,
                ).cpu().numpy()
    return result


def _head_metrics(
    score_rows: np.ndarray,
    target_error_rows: np.ndarray,
    translation_rows: np.ndarray,
    rotation_rows: np.ndarray,
    valid_rows: np.ndarray,
    image_ids: np.ndarray,
) -> dict[str, object]:
    score = np.asarray(score_rows, dtype=np.float64)
    target = np.asarray(target_error_rows, dtype=np.float64)
    translation = np.asarray(translation_rows, dtype=np.float64)
    rotation = np.asarray(rotation_rows, dtype=np.float64)
    valid = np.asarray(valid_rows, dtype=bool)
    if not (
        score.shape == target.shape == translation.shape == rotation.shape == valid.shape
        and score.ndim == 2 and score.shape[0] == len(image_ids)
    ):
        raise ValueError("factorized pose-ranker metric arrays differ")
    rows = []
    pair_correct = pair_total = 0
    for image_id, values, error, t_m, r_deg, mask in zip(
        image_ids.tolist(), score, target, translation, rotation, valid,
    ):
        indices = np.flatnonzero(mask)
        nonanchor = indices[indices != 0]
        if nonanchor.size == 0:
            raise ValueError("factorized metrics require a non-anchor candidate")
        selected = int(nonanchor[np.argmax(values[nonanchor])])
        better = error[indices, None] + 1.0e-8 < error[None, indices]
        difference = values[indices, None] > values[None, indices]
        pair_correct += int(np.sum(better & difference))
        pair_total += int(np.sum(better))
        rows.append({
            "image_id": str(image_id),
            "target_error_spearman": _spearman(values[indices], -error[indices]),
            "anchor_is_highest": bool(values[0] >= np.max(values[indices]) - 1.0e-8),
            "selected_index_excluding_anchor": selected,
            "selected_translation_m": float(t_m[selected]),
            "selected_rotation_deg": float(r_deg[selected]),
            "selected_strict_0_5m_5deg": bool(t_m[selected] <= 0.5 and r_deg[selected] <= 5.0),
            "selected_loose_1m_10deg": bool(t_m[selected] <= 1.0 and r_deg[selected] <= 10.0),
        })
    return {
        "query_count": len(rows),
        "mean_target_error_spearman": float(np.mean([
            row["target_error_spearman"] for row in rows
        ])),
        "pairwise_order_accuracy": float(pair_correct / max(pair_total, 1)),
        "anchor_highest_rate": float(np.mean([row["anchor_is_highest"] for row in rows])),
        "selected_strict_0_5m_5deg": float(np.mean([
            row["selected_strict_0_5m_5deg"] for row in rows
        ])),
        "selected_loose_1m_10deg": float(np.mean([
            row["selected_loose_1m_10deg"] for row in rows
        ])),
        "median_selected_translation_m": float(np.median([
            row["selected_translation_m"] for row in rows
        ])),
        "median_selected_rotation_deg": float(np.median([
            row["selected_rotation_deg"] for row in rows
        ])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--initial_model", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--train_routes", nargs="+", default=["seq13"])
    parser.add_argument("--validation_routes", nargs="+", default=["seq3"])
    parser.add_argument(
        "--held_routes", nargs="*", default=["seq5"],
        help="Optional untouched routes; pass the flag without values for a two-way diagnostic.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_queries", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--translation_scale_m", type=float, default=8.0)
    parser.add_argument("--rotation_scale_deg", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    feature_path = Path(args.features)
    feature_manifest_path = Path(args.feature_manifest)
    initial_model_path = Path(args.initial_model)
    output_model = Path(args.output_model)
    output_report = Path(args.output_report)
    if output_model.exists() or output_report.exists():
        raise FileExistsError("refusing to overwrite factorized pose ranker experiment")
    if int(args.epochs) <= 0 or int(args.batch_queries) <= 0:
        raise ValueError("factorized ranker schedule must be positive")
    if float(args.translation_scale_m) <= 0.0 or float(args.rotation_scale_deg) <= 0.0:
        raise ValueError("factorized ranker supervision scales must be positive")

    arrays, metadata = _load_local_supervision_dataset(dataset_path)
    features, feature_manifest = _load_feature_artifact(
        feature_path, feature_manifest_path,
        dataset_path=dataset_path,
        dataset_content_sha256=str(metadata["content_sha256"]),
    )
    if features.shape[:2] != arrays["candidate_valid"].shape:
        raise ValueError("factorized ranker features differ from supervision inventory")
    route_sets = tuple(tuple(str(value) for value in routes) for routes in (
        args.train_routes, args.validation_routes, args.held_routes,
    ))
    if len(set().union(*map(set, route_sets))) != sum(map(len, route_sets)):
        raise ValueError("factorized ranker route splits must be disjoint")
    masks = tuple(_mask_for_routes(arrays["image_ids"], routes) for routes in route_sets)
    if (
        not np.any(masks[0]) or not np.any(masks[1])
        or (route_sets[2] and not np.any(masks[2]))
        or not np.all(masks[0] | masks[1] | masks[2])
    ):
        raise ValueError("factorized ranker route splits must cover the inventory")

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(args.device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    input_channels = int(features.shape[2])
    model = FactorizedFullTokenCandidatePoseRanker(input_channels=input_channels)
    try:
        initial_factorized, initial_metadata = load_factorized_fulltoken_candidate_pose_ranker(
            initial_model_path, device="cpu",
        )
    except ValueError:
        initial_single, initial_metadata = load_fulltoken_candidate_pose_ranker(
            initial_model_path, device="cpu",
        )
        if input_channels != len(FULLTOKEN_POSE_RANKING_CHANNELS):
            raise ValueError("typed factorized ranker requires a factorized warm start")
        model.initialize_from_single_ranker(initial_single)
        initial_artifact_type = "single_head"
    else:
        model.initialize_from_factorized_ranker(initial_factorized)
        initial_artifact_type = "factorized"
    if initial_artifact_type == "single_head":
        local_score_model_content_sha256 = initial_metadata["model_content_sha256"]
    else:
        local_score_model_content_sha256 = initial_metadata.get(
            "local_score_model_content_sha256",
            initial_metadata.get("initial_model_content_sha256"),
        )
        if not isinstance(local_score_model_content_sha256, str):
            raise ValueError("factorized warm start lacks local scorer ancestry")
    model.to(device)
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
        loss_rows = []
        for begin in range(0, int(order.size), int(args.batch_queries)):
            rows = order[begin:begin + int(args.batch_queries)]
            value = torch.as_tensor(np.asarray(features[rows], dtype=np.float32), device=device)
            batch, candidates = value.shape[:2]
            score = {
                name: tensor.reshape(batch, candidates)
                for name, tensor in model(value.flatten(0, 1)).items()
            }
            translation = torch.as_tensor(
                arrays["translation_m"][rows], device=device, dtype=torch.float32,
            )
            rotation = torch.as_tensor(
                arrays["rotation_deg"][rows], device=device, dtype=torch.float32,
            )
            valid = torch.as_tensor(arrays["candidate_valid"][rows], device=device)
            zeros = torch.zeros_like(translation)
            losses = {
                "location": listwise_pose_ranking_loss(
                    score["location"], translation, zeros, valid,
                    translation_scale_m=float(args.translation_scale_m),
                    rotation_scale_deg=1.0,
                ),
                "orientation": listwise_pose_ranking_loss(
                    score["orientation"], zeros, rotation, valid,
                    translation_scale_m=1.0,
                    rotation_scale_deg=float(args.rotation_scale_deg),
                ),
                "joint": listwise_pose_ranking_loss(
                    score["joint"], translation, rotation, valid,
                    translation_scale_m=float(args.translation_scale_m),
                    rotation_scale_deg=float(args.rotation_scale_deg),
                ),
            }
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_rows.append({name: float(value.detach().cpu()) for name, value in losses.items()})
        epoch_rows.append({
            "epoch": epoch + 1,
            **{f"mean_{name}_loss": float(np.mean([row[name] for row in loss_rows]))
               for name in model.HEAD_NAMES},
        })

    all_score = {
        name: np.full(arrays["candidate_valid"].shape, -1.0e9, dtype=np.float32)
        for name in model.HEAD_NAMES
    }
    for mask in masks:
        rows = np.flatnonzero(mask)
        split_score = _score_queries(
            model, features, rows, batch_queries=int(args.batch_queries), device=device,
        )
        for name in model.HEAD_NAMES:
            all_score[name][rows] = split_score[name]

    translation_target = arrays["translation_m"] / float(args.translation_scale_m)
    orientation_target = arrays["rotation_deg"] / float(args.rotation_scale_deg)
    targets = {
        "location": translation_target,
        "orientation": orientation_target,
        "joint": np.maximum(translation_target, orientation_target),
    }
    split_reports = {}
    for split, mask, routes in zip(("train", "validation", "held"), masks, route_sets):
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            split_reports[split] = {
                "routes": list(routes),
                "query_count": 0,
                "heads": None,
            }
            continue
        split_reports[split] = {
            "routes": list(routes),
            "query_count": int(rows.size),
            "heads": {
                name: _head_metrics(
                    all_score[name][rows], targets[name][rows],
                    arrays["translation_m"][rows], arrays["rotation_deg"][rows],
                    arrays["candidate_valid"][rows], arrays["image_ids"][rows],
                )
                for name in model.HEAD_NAMES
            },
        }

    state = model.state_dict()
    state_arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    model_content_sha256 = arrays_sha256(state_arrays)
    payload = {
        "artifact_type": "goal_maplet_factorized_fulltoken_candidate_pose_ranker_v1",
        "model_semantics": FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS,
        "model_content_sha256": model_content_sha256,
        "input_channels": input_channels,
        "feature_semantics": feature_manifest["feature_semantics"],
        "feature_channels": feature_manifest["feature_channels"],
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "initial_model_file_sha256": file_sha256(initial_model_path),
        "initial_model_content_sha256": initial_metadata["model_content_sha256"],
        "initial_model_artifact_type": initial_artifact_type,
        "local_score_model_content_sha256": local_score_model_content_sha256,
        "translation_scale_m": float(args.translation_scale_m),
        "rotation_scale_deg": float(args.rotation_scale_deg),
        "candidate_pose_values_are_model_inputs": False,
        "uses_absolute_pose_regression": False,
        "state_dict": state,
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_model)
    report = {
        "artifact_type": REPORT_SCHEMA,
        "model_semantics": FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS,
        "model_file_sha256": file_sha256(output_model),
        "model_content_sha256": model_content_sha256,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_manifest_file_sha256": file_sha256(feature_manifest_path),
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "feature_semantics": feature_manifest["feature_semantics"],
        "feature_channels": feature_manifest["feature_channels"],
        "input_channels": input_channels,
        "initial_model_file_sha256": file_sha256(initial_model_path),
        "initial_model_content_sha256": initial_metadata["model_content_sha256"],
        "initial_model_artifact_type": initial_artifact_type,
        "local_score_model_content_sha256": local_score_model_content_sha256,
        "seed": seed,
        "epochs": int(args.epochs),
        "batch_queries": int(args.batch_queries),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "translation_scale_m": float(args.translation_scale_m),
        "rotation_scale_deg": float(args.rotation_scale_deg),
        "epoch_rows": epoch_rows,
        "split_reports": split_reports,
        "candidate_pose_values_are_model_inputs": False,
        "candidate_pose_errors_are_model_inputs": False,
        "held_labels_used_by_optimizer": False,
        "validation_labels_used_for_epoch_or_hyperparameter_selection": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "factorized_candidate_ranking_diagnostic_not_final_localization",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_model": str(output_model),
        "output_report": str(output_report),
        "split_summary": {
            split: None if row["heads"] is None else {
                name: {
                    key: value for key, value in metrics.items() if key != "rows"
                }
                for name, metrics in row["heads"].items()
            }
            for split, row in split_reports.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
