"""Train a pose-free RADIO head to predict target-view token geometry."""

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
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.query_typed_geometry import (
    QUERY_TYPED_GEOMETRY_PREDICTOR_SEMANTICS,
    QueryTypedGeometryPredictor,
    query_typed_geometry_supervision_loss,
)


def _load_manifest_array(path: Path, manifest_path: Path, *, channels: int) -> tuple[np.ndarray, dict]:
    manifest = json.loads(manifest_path.read_text())
    resolved = path.resolve()
    if (
        Path(str(manifest.get("feature_file", ""))).resolve() != resolved
        or manifest.get("feature_file_sha256") != file_sha256(resolved)
    ):
        raise ValueError("query geometry feature lineage differs")
    value = np.load(resolved, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.float16 or value.shape[0] != 128:
        raise ValueError("query geometry feature inventory differs")
    if channels == 128 and value.shape[1:] != (128, 36, 64):
        raise ValueError("query readout feature shape differs")
    if channels == 18 and value.shape[2:] != (18, 36, 64):
        raise ValueError("typed candidate feature shape differs")
    return value, manifest


@torch.no_grad()
def _metrics(
    model: QueryTypedGeometryPredictor,
    query: np.ndarray,
    target: np.ndarray,
    mass: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    prediction = model(torch.as_tensor(np.asarray(query[rows], dtype=np.float32), device=device))
    truth = torch.as_tensor(target[rows], device=device, dtype=torch.float32)
    weight = torch.as_tensor(mass[rows], device=device, dtype=torch.float32)
    denominator = weight.sum().clamp_min(1.0)
    moment = prediction["normal_axis_moment"]
    normal_similarity = (
        torch.sum(moment[:, :3] * truth[:, :3], dim=1)
        + 2.0 * torch.sum(moment[:, 3:] * truth[:, 3:6], dim=1)
    ).clamp(0.0, 1.0)
    def weighted_mae(estimate, expected):
        return float((weight * torch.abs(estimate - expected)).sum().cpu() / denominator.cpu())
    return {
        "query_count": int(rows.size),
        "mass_weighted_normal_axis_similarity": float(
            (weight * normal_similarity).sum().cpu() / denominator.cpu()
        ),
        "mass_weighted_relative_log_depth_mae": weighted_mae(
            prediction["relative_log_depth"], truth[:, 6],
        ),
        "mass_weighted_log_depth_std_mae": weighted_mae(
            prediction["log_depth_std"], truth[:, 7],
        ),
        "mass_weighted_boundary_mae": weighted_mae(
            prediction["boundary"], truth[:, 8],
        ),
        "mass_confidence_mae": float(
            torch.mean(torch.abs(prediction["confidence"] - weight)).cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query_features", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--typed_features", required=True)
    parser.add_argument("--typed_manifest", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_queries", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    arrays, metadata = _load_local_supervision_dataset(dataset_path)
    query, query_manifest = _load_manifest_array(
        Path(args.query_features), Path(args.query_manifest), channels=128,
    )
    typed, typed_manifest = _load_manifest_array(
        Path(args.typed_features), Path(args.typed_manifest), channels=18,
    )
    if (
        not np.array_equal(arrays["image_ids"], query_manifest.get("image_ids"))
        or query_manifest.get("dataset_content_sha256") != metadata["content_sha256"]
        or typed_manifest.get("dataset_content_sha256") != metadata["content_sha256"]
        or typed.shape[:2] != arrays["candidate_valid"].shape
        or not np.all(arrays["candidate_valid"][:, 0])
        or np.any(arrays["translation_m"][:, 0] != 0.0)
        or np.any(arrays["rotation_deg"][:, 0] != 0.0)
    ):
        raise ValueError("query geometry target anchor or lineage differs")
    target = np.asarray(typed[:, 0, 9:18], dtype=np.float32)
    mass = np.asarray(typed[:, 0, 1], dtype=np.float32)
    route_sets = (("seq13",), ("seq3",), ("seq5",))
    masks = tuple(_mask_for_routes(arrays["image_ids"], routes) for routes in route_sets)
    if any(int(mask.sum()) != expected for mask, expected in zip(masks, (64, 32, 32))):
        raise ValueError("query geometry route split differs")

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(args.device))
    model = QueryTypedGeometryPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
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
            prediction = model(torch.as_tensor(
                np.asarray(query[rows], dtype=np.float32), device=device,
            ))
            loss = query_typed_geometry_supervision_loss(
                prediction,
                torch.as_tensor(target[rows], device=device),
                torch.as_tensor(mass[rows], device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_rows.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})

    split_metrics = {
        name: _metrics(model, query, target, mass, np.flatnonzero(mask), device)
        for name, mask in zip(("train", "validation", "held"), masks)
    }
    state = model.state_dict()
    model_content = arrays_sha256({
        key: value.detach().cpu().numpy() for key, value in state.items()
    })
    payload = {
        "artifact_type": "goal_maplet_query_typed_geometry_predictor_v1",
        "model_semantics": QUERY_TYPED_GEOMETRY_PREDICTOR_SEMANTICS,
        "model_content_sha256": model_content,
        "dataset_content_sha256": metadata["content_sha256"],
        "query_feature_file_sha256": query_manifest["feature_file_sha256"],
        "typed_feature_file_sha256": typed_manifest["feature_file_sha256"],
        "target_candidate_index": 0,
        "query_pose_is_model_input": False,
        "candidate_pose_is_model_input": False,
        "state_dict": state,
    }
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_model)
    report = {
        "artifact_type": "goal_maplet_query_typed_geometry_route_holdout_report_v1",
        "model_file_sha256": file_sha256(output_model),
        "model_content_sha256": model_content,
        "dataset_content_sha256": metadata["content_sha256"],
        "query_feature_file_sha256": query_manifest["feature_file_sha256"],
        "typed_feature_file_sha256": typed_manifest["feature_file_sha256"],
        "epochs": int(args.epochs),
        "seed": seed,
        "split_metrics": split_metrics,
        "epoch_rows": epoch_rows,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    Path(args.output_report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(split_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
