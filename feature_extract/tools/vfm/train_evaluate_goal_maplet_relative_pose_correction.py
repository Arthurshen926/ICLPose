"""Train and evaluate a candidate-relative multimodal SE(3) correction head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_feature_artifact,
    _load_local_supervision_dataset,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.local_pose_supervision import (
    LOCAL_POSE_SUPERVISION_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.relative_pose_correction import (
    RELATIVE_POSE_CORRECTION_SEMANTICS,
    FullTokenRelativePoseCorrectionNet,
    apply_left_pose_correction_coordinate,
    left_pose_correction_coordinate,
    relative_pose_mixture_loss,
)


REPORT_SCHEMA = "goal_maplet_relative_pose_correction_route_holdout_report_v1"


def _route(value: object) -> str:
    return str(value).split("/", 1)[0]


def _predict(
    model: FullTokenRelativePoseCorrectionNet,
    features: np.ndarray,
    pairs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates, logits = [], []
    model.eval()
    with torch.no_grad():
        for begin in range(0, pairs.shape[0], int(batch_size)):
            rows = pairs[begin:begin + int(batch_size)]
            value = torch.as_tensor(
                np.asarray(features[rows[:, 0], rows[:, 1]], dtype=np.float32),
                device=device,
            )
            coordinate, logit = model(value)
            coordinates.append(coordinate.cpu().numpy())
            logits.append(logit.cpu().numpy())
    return np.concatenate(coordinates), np.concatenate(logits)


def _correction_metrics(
    arrays: dict[str, np.ndarray],
    pairs: np.ndarray,
    coordinate: np.ndarray,
    logit: np.ndarray,
) -> dict[str, object]:
    corrected, targets = [], []
    for pair, modes in zip(pairs.tolist(), coordinate):
        query, candidate = pair
        pose = arrays["candidate_poses_w2c"][query, candidate]
        corrected.append(np.asarray([
            apply_left_pose_correction_coordinate(pose, mode) for mode in modes
        ]))
        targets.append(arrays["candidate_poses_w2c"][query, 0])
    corrected = np.asarray(corrected)
    targets = np.asarray(targets)
    translation = np.zeros(corrected.shape[:2], dtype=np.float64)
    rotation = np.zeros_like(translation)
    for row in range(corrected.shape[0]):
        translation[row], rotation[row] = _pose_errors(corrected[row], targets[row])
    top = np.argmax(logit, axis=1)
    rows = np.arange(pairs.shape[0])
    input_translation = arrays["translation_m"][pairs[:, 0], pairs[:, 1]]
    input_rotation = arrays["rotation_deg"][pairs[:, 0], pairs[:, 1]]
    input_joint = np.maximum(input_translation / 1.0, input_rotation / 10.0)

    def summarize(mask: np.ndarray) -> dict[str, object]:
        chosen = np.flatnonzero(mask)
        if not chosen.size:
            return {"sample_count": 0}
        any_strict = np.any(
            (translation[chosen] <= 0.5 + 1e-6)
            & (rotation[chosen] <= 5.0 + 1e-5), axis=1,
        )
        any_loose = np.any(
            (translation[chosen] <= 1.0 + 1e-6)
            & (rotation[chosen] <= 10.0 + 1e-5), axis=1,
        )
        top_t = translation[chosen, top[chosen]]
        top_r = rotation[chosen, top[chosen]]
        oracle_joint = np.minimum.reduce(
            np.maximum(translation[chosen] / 1.0, rotation[chosen] / 10.0), axis=1,
        )
        return {
            "sample_count": int(chosen.size),
            "any_mode_strict_0_5m_5deg": float(np.mean(any_strict)),
            "any_mode_loose_1m_10deg": float(np.mean(any_loose)),
            "top_probability_mode_strict_0_5m_5deg": float(np.mean(
                (top_t <= 0.5 + 1e-6) & (top_r <= 5.0 + 1e-5)
            )),
            "top_probability_mode_loose_1m_10deg": float(np.mean(
                (top_t <= 1.0 + 1e-6) & (top_r <= 10.0 + 1e-5)
            )),
            "median_top_probability_translation_m": float(np.median(top_t)),
            "median_top_probability_rotation_deg": float(np.median(top_r)),
            "median_oracle_joint_error": float(np.median(oracle_joint)),
        }

    nonanchor = pairs[:, 1] != 0
    return {
        "all_nonanchor": summarize(nonanchor),
        "outside_1m_10deg_input": summarize(nonanchor & (input_joint > 1.0 + 1e-8)),
        "inside_8m_45deg_normalized_input": summarize(
            nonanchor & (input_translation <= np.sqrt(3.0) * 8.0 + 1e-5)
            & (input_rotation <= 45.0 + 1e-5)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--samples_per_query_per_epoch", type=int, default=96)
    parser.add_argument("--mode_count", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    model_path, report_path = Path(args.output_model), Path(args.output_report)
    if model_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite relative pose correction experiment")
    dataset_path = Path(args.dataset)
    arrays, metadata = _load_local_supervision_dataset(dataset_path)
    if metadata.get("supervision_semantics") != LOCAL_POSE_SUPERVISION_SEMANTICS:
        raise ValueError("relative correction requires current global-joint supervision")
    features, feature_manifest = _load_feature_artifact(
        Path(args.features), Path(args.feature_manifest), dataset_path=dataset_path,
        dataset_content_sha256=str(metadata["content_sha256"]),
    )
    query_count, candidate_count = arrays["candidate_valid"].shape
    target_coordinate = np.zeros((query_count, candidate_count, 6), dtype=np.float32)
    for query in range(query_count):
        target = arrays["candidate_poses_w2c"][query, 0]
        for candidate in range(candidate_count):
            target_coordinate[query, candidate] = left_pose_correction_coordinate(
                arrays["candidate_poses_w2c"][query, candidate], target,
            )
    if np.any(np.abs(target_coordinate[:, :, :3]) > 1.001) or np.any(
        np.linalg.norm(target_coordinate[:, :, 3:], axis=2) > 1.001
    ):
        raise ValueError("relative correction supervision exceeds its bounded domain")
    routes = np.asarray([_route(value) for value in arrays["image_ids"]])
    split_routes = {"train": "seq13", "validation": "seq3", "held": "seq5"}
    split_queries = {
        name: np.flatnonzero(routes == route) for name, route in split_routes.items()
    }
    if any(not value.size for value in split_queries.values()):
        raise ValueError("relative correction route split is incomplete")
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(args.device))
    model = FullTokenRelativePoseCorrectionNet(mode_count=int(args.mode_count)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    epoch_rows = []
    sample_count = min(int(args.samples_per_query_per_epoch), candidate_count)
    for epoch in range(int(args.epochs)):
        pairs = []
        for query in split_queries["train"].tolist():
            chosen = generator.choice(candidate_count, size=sample_count, replace=False)
            if 0 not in chosen:
                chosen[0] = 0
            pairs.extend((query, int(candidate)) for candidate in chosen.tolist())
        pairs = np.asarray(pairs, dtype=np.int64)
        generator.shuffle(pairs)
        losses = []
        model.train()
        for begin in range(0, pairs.shape[0], int(args.batch_size)):
            rows = pairs[begin:begin + int(args.batch_size)]
            value = torch.as_tensor(
                np.asarray(features[rows[:, 0], rows[:, 1]], dtype=np.float32),
                device=device,
            )
            target = torch.as_tensor(
                target_coordinate[rows[:, 0], rows[:, 1]], device=device,
            )
            coordinate, logit = model(value)
            loss, _ = relative_pose_mixture_loss(coordinate, logit, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_rows.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
    state_arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    model_content_sha256 = arrays_sha256(state_arrays)
    payload = {
        "artifact_type": "goal_maplet_fulltoken_relative_pose_correction_v1",
        "model_semantics": RELATIVE_POSE_CORRECTION_SEMANTICS,
        "model_content_sha256": model_content_sha256,
        "mode_count": int(args.mode_count),
        "translation_scale_m": 8.0,
        "rotation_scale_deg": 45.0,
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "dataset_content_sha256": metadata["content_sha256"],
        "local_pose_supervision_semantics": metadata["supervision_semantics"],
        "state_dict": model.state_dict(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    split_metrics = {}
    for name, query_rows in split_queries.items():
        pairs = np.asarray([
            (query, candidate) for query in query_rows.tolist()
            for candidate in range(candidate_count)
        ], dtype=np.int64)
        coordinate, logit = _predict(
            model, features, pairs, batch_size=int(args.batch_size), device=device,
        )
        split_metrics[name] = _correction_metrics(arrays, pairs, coordinate, logit)
    report = {
        "artifact_type": REPORT_SCHEMA,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "model_file_sha256": file_sha256(model_path),
        "model_content_sha256": model_content_sha256,
        "model_semantics": RELATIVE_POSE_CORRECTION_SEMANTICS,
        "supervision_semantics": metadata["supervision_semantics"],
        "mode_count": int(args.mode_count),
        "epochs": int(args.epochs),
        "samples_per_query_per_epoch": sample_count,
        "epoch_rows": epoch_rows,
        "split_routes": split_routes,
        "split_metrics": split_metrics,
        "candidate_pose_matrix_is_model_input": False,
        "absolute_pose_is_model_output": False,
        "output_is_candidate_relative_left_se3_mixture": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "production_eligible": False,
        "claim": "candidate_relative_correction_diagnostic_not_final_localization",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split_metrics": split_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
