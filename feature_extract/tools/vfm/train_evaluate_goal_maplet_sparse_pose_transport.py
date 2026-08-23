"""Train and one-shot evaluate the real map-disjoint sparse transport backend.

The split is a single deterministic mapping-route train/dev split, not a
five-fold experiment.  Candidate 0 is a diagnostic GT anchor used only to
shape the local energy; all end-to-end-like reranking metrics exclude it.
The dev labels are consumed only after the fixed final epoch is serialized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    FrozenSparseTransportEdges,
    build_frozen_sparse_transport_edges,
    differentiable_sparse_pose_transport,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    HIERARCHY_SEMANTICS,
    build_pose_transport_hierarchy,
)
from feature_extract.vfm.localization_goal_maplet.pose_transport_training import (
    PoseTransportTrainingConfig,
    normalized_joint_pose_error,
    pose_transport_energy_landscape_loss,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    MinimalPoseTransportConfig,
    MinimalPoseTransportReadout,
    pose_transport_model_content_sha256,
    save_minimal_pose_transport_readout,
)


REPORT_SCHEMA = "goal_maplet_real_sparse_pose_transport_train_dev_report_v1"


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(value), kind="stable")
    rank = np.empty(order.size, dtype=np.float64)
    rank[order] = np.arange(order.size, dtype=np.float64)
    return rank


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _rankdata(left), _rankdata(right)
    if a.size < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("transport dataset lacks metadata")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
    if metadata.get("artifact_type") != "goal_maplet_real_sparse_pose_transport_dataset_v1":
        raise ValueError("not a real sparse pose transport dataset")
    if metadata.get("content_sha256") != arrays_sha256(arrays):
        raise ValueError("transport dataset content hash differs")
    required_false = ("uses_alike", "uses_point_correspondences", "uses_pnp", "uses_absolute_pose_regression")
    if any(metadata.get(key) is not False for key in required_false):
        raise ValueError("transport dataset violates method boundary")
    if metadata.get("canonical_map_excludes_query_route") is not True:
        raise ValueError("transport dataset is not map-disjoint")
    return arrays, metadata


def _ray_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height) * 2.0 - 1.0
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width) * 2.0 - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0)[None]


def _edges_for_query(
    arrays: dict[str, np.ndarray], query: int, hierarchy, *, stage: str,
) -> list[FrozenSparseTransportEdges]:
    result = []
    for candidate in range(arrays["candidate_valid"].shape[1]):
        if not bool(arrays["candidate_valid"][query, candidate]):
            result.append(FrozenSparseTransportEdges(
                source_index=np.zeros(0, dtype=np.int64), target_index=np.zeros(0, dtype=np.int64),
                hierarchy_score=np.zeros(0, dtype=np.float32), layout_score=np.zeros(0, dtype=np.float32),
                source_count=int(arrays["source_child_rows"][query].size),
                target_count=int(arrays["target_child_rows"][query, candidate].size), stage=str(stage),
            ))
            continue
        result.append(build_frozen_sparse_transport_edges(
            arrays["source_child_rows"][query],
            arrays["source_child_probabilities"][query],
            arrays["token_xy"][query],
            arrays["target_child_rows"][query, candidate],
            arrays["target_child_weights"][query, candidate],
            hierarchy, stage=str(stage),
            minimum_source_probability=1.0e-6,
            minimum_target_weight=1.0e-6,
        ))
    return result


def _query_scores(
    model: MinimalPoseTransportReadout,
    arrays: dict[str, np.ndarray],
    query_index: int,
    edge_rows: list[FrozenSparseTransportEdges],
    *,
    device: torch.device,
    gradient: bool,
) -> torch.Tensor:
    context = torch.enable_grad() if gradient else torch.no_grad()
    with context:
        query_feature_key = (
            "pose_query_features" if "pose_query_features" in arrays else "radio_final"
        )
        radio = torch.as_tensor(
            arrays[query_feature_key][query_index], device=device, dtype=torch.float32
        )[None]
        query = model(radio, _ray_grid(36, 64, device=device))
        source = torch.as_tensor(
            arrays["source_child_probabilities"][query_index], device=device, dtype=torch.float32
        )
        reliability = torch.as_tensor(
            arrays["query_reliability"][query_index], device=device, dtype=torch.float32
        )
        scores = []
        for candidate, edges in enumerate(edge_rows):
            if not bool(arrays["candidate_valid"][query_index, candidate]):
                scores.append(torch.full((), -1.0, device=device))
                continue
            scores.append(differentiable_sparse_pose_transport(
                model, query, source, reliability,
                torch.as_tensor(arrays["target_child_weights"][query_index, candidate], device=device),
                torch.as_tensor(
                    arrays["target_canonical_features"][query_index, candidate],
                    device=device, dtype=torch.float32,
                ),
                torch.as_tensor(arrays["target_normals_camera"][query_index, candidate], device=device),
                torch.as_tensor(arrays["target_double_sided"][query_index, candidate], device=device),
                torch.as_tensor(arrays["target_relative_depth"][query_index, candidate], device=device),
                torch.as_tensor(arrays["target_boundary"][query_index, candidate], device=device),
                torch.as_tensor(arrays["target_modality_valid"][query_index, candidate], device=device),
                torch.as_tensor(arrays["target_modality_confidence"][query_index, candidate], device=device),
                edges,
            ).combined_score)
        return torch.stack(scores)


def _metrics(
    score_rows: np.ndarray,
    translation_rows: np.ndarray,
    rotation_rows: np.ndarray,
    valid_rows: np.ndarray,
    image_ids: np.ndarray,
) -> dict[str, object]:
    rows = []
    for image_id, score, translation, rotation, valid in zip(
        image_ids.tolist(), score_rows, translation_rows, rotation_rows, valid_rows
    ):
        indices = np.flatnonzero(valid)
        nonanchor = indices[indices != 0]
        joint = np.maximum(translation / 1.0, rotation / 15.0)
        selected = int(nonanchor[np.argmax(score[nonanchor])])
        oracle = int(nonanchor[np.argmin(joint[nonanchor])])
        proposal = int(nonanchor[0])
        ordered_pairs = 0
        correct_pairs = 0
        for a in indices.tolist():
            for b in indices.tolist():
                if joint[a] + 1.0e-8 < joint[b]:
                    ordered_pairs += 1
                    correct_pairs += int(score[a] > score[b])
        rows.append({
            "image_id": str(image_id),
            "gt_anchor_rank": int(1 + np.sum(score[indices] > score[0])),
            "score_error_spearman": _spearman(score[indices], -joint[indices]),
            "pairwise_correct": correct_pairs,
            "pairwise_total": ordered_pairs,
            "selected_index_excluding_gt_anchor": selected,
            "proposal_index": proposal,
            "oracle_index_excluding_gt_anchor": oracle,
            "selected_translation_m": float(translation[selected]),
            "selected_rotation_deg": float(rotation[selected]),
            "proposal_translation_m": float(translation[proposal]),
            "proposal_rotation_deg": float(rotation[proposal]),
            "oracle_translation_m": float(translation[oracle]),
            "oracle_rotation_deg": float(rotation[oracle]),
            "selected_strict_0_5m_5deg": bool(translation[selected] <= 0.5 and rotation[selected] <= 5.0),
            "proposal_strict_0_5m_5deg": bool(translation[proposal] <= 0.5 and rotation[proposal] <= 5.0),
            "selected_loose_1m_10deg": bool(translation[selected] <= 1.0 and rotation[selected] <= 10.0),
            "proposal_loose_1m_10deg": bool(translation[proposal] <= 1.0 and rotation[proposal] <= 10.0),
        })
    def mean(key):
        return float(np.mean([float(row[key]) for row in rows]))
    return {
        "query_count": len(rows),
        "gt_anchor_top1_rate": float(np.mean([row["gt_anchor_rank"] == 1 for row in rows])),
        "mean_score_error_spearman": mean("score_error_spearman"),
        "pairwise_order_accuracy": float(sum(row["pairwise_correct"] for row in rows) / max(sum(row["pairwise_total"] for row in rows), 1)),
        "selected_strict_0_5m_5deg": mean("selected_strict_0_5m_5deg"),
        "proposal_strict_0_5m_5deg": mean("proposal_strict_0_5m_5deg"),
        "selected_loose_1m_10deg": mean("selected_loose_1m_10deg"),
        "proposal_loose_1m_10deg": mean("proposal_loose_1m_10deg"),
        "selected_median_translation_m": float(np.median([row["selected_translation_m"] for row in rows])),
        "selected_median_rotation_deg": float(np.median([row["selected_rotation_deg"] for row in rows])),
        "proposal_median_translation_m": float(np.median([row["proposal_translation_m"] for row in rows])),
        "proposal_median_rotation_deg": float(np.median([row["proposal_rotation_deg"] for row in rows])),
        "oracle_median_translation_m": float(np.median([row["oracle_translation_m"] for row in rows])),
        "oracle_median_rotation_deg": float(np.median([row["oracle_rotation_deg"] for row in rows])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--train_queries", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--surface_mapper", default="")
    parser.add_argument("--field_feature_contract", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_path, report_path = Path(args.output_model), Path(args.output_report)
    if (model_path.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite pose transport experiment")
    if int(args.epochs) <= 0 or float(args.learning_rate) <= 0.0:
        raise ValueError("training schedule must be positive")
    arrays, dataset_metadata = _load_dataset(Path(args.dataset))
    query_count = int(arrays["image_ids"].size)
    train_count = int(args.train_queries)
    if not 1 <= train_count < query_count:
        raise ValueError("train_queries must leave at least one dev query")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    if physical.content_sha256 != dataset_metadata.get("physical_map_sha256"):
        raise ValueError("dataset and physical map differ")
    hierarchy = build_pose_transport_hierarchy(physical)
    shared_surface_space = bool(str(args.surface_mapper))
    mapper_sha = None
    if shared_surface_space:
        if not str(args.field_feature_contract):
            raise ValueError("surface_mapper requires field_feature_contract")
        contract_path = Path(args.field_feature_contract)
        contract = json.loads(contract_path.read_text())
        mapper_path = Path(args.surface_mapper)
        mapper_sha = file_sha256(mapper_path)
        if (
            contract.get("artifact_type") != "goal_maplet_field_feature_contract_v1"
            or contract.get("canonical_field_sha256") != dataset_metadata.get("canonical_field_sha256")
            or contract.get("query_readout_type") != "surface_maplet_mapper"
            or contract.get("query_readout_sha256") != mapper_sha
        ):
            raise ValueError("surface mapper/canonical field contract differs from dataset")
        mapper, _mapper_metadata = load_surface_maplet_mapper(mapper_path, device=str(device))
        mapped_rows = []
        mapper.model.to(device).eval()
        with torch.no_grad():
            for query_index in range(query_count):
                raw = torch.as_tensor(
                    arrays["radio_final"][query_index], device=device, dtype=torch.float32
                )[None]
                mapped_rows.append(
                    mapper.model(raw)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                )
        arrays["pose_query_features"] = np.stack(mapped_rows, axis=0)
    print(json.dumps({"hierarchy_sha256": hierarchy.content_sha256, "building_edges": True}), flush=True)
    medium_edges = []
    for query in range(query_count):
        medium_edges.append(_edges_for_query(arrays, query, hierarchy, stage="medium"))
        print(json.dumps({
            "query": str(arrays["image_ids"][query]),
            "medium_edge_count": int(sum(value.source_index.size for value in medium_edges[-1])),
        }), flush=True)

    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=128 if shared_surface_space else 1280,
        shared_query_map_projection=shared_surface_space,
    )).to(device)
    if shared_surface_space:
        # P1 deliberately trains only the already aligned feature space plus
        # fixed hierarchy/layout evidence. Random, unsupervised query normal,
        # depth and boundary heads must not inject noise into this first gate.
        with torch.no_grad():
            model.edge_weight_unconstrained[1:4].fill_(-8.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-5)
    loss_config = PoseTransportTrainingConfig(stage="medium", attribution_weight=0.0)
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        epoch_loss = []
        order = np.roll(np.arange(train_count), epoch % train_count)
        for query_index in order.tolist():
            optimizer.zero_grad(set_to_none=True)
            scores = _query_scores(
                model, arrays, query_index, medium_edges[query_index], device=device, gradient=True
            )[None]
            valid = torch.as_tensor(arrays["candidate_valid"][query_index], device=device)[None]
            translation = torch.as_tensor(arrays["translation_m"][query_index], device=device)[None]
            rotation = torch.as_tensor(arrays["rotation_deg"][query_index], device=device)[None]
            error = normalized_joint_pose_error(translation, rotation, stage="medium")[0].detach().cpu().numpy()
            valid_indices = np.flatnonzero(arrays["candidate_valid"][query_index])
            sorted_indices = valid_indices[np.argsort(error[valid_indices], kind="stable")]
            monotonic = torch.as_tensor(
                [[0, int(a), int(b)] for a, b in zip(sorted_indices[:-1], sorted_indices[1:])],
                device=device, dtype=torch.long,
            )
            loss, stats = pose_transport_energy_landscape_loss(
                scores, translation, rotation, valid,
                config=loss_config, monotonic_pairs=monotonic,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite pose-transport loss at epoch={epoch + 1}, query={query_index}"
                )
            loss.backward()
            nonfinite_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite_gradients:
                raise FloatingPointError(
                    "nonfinite pose-transport gradients: " + ",".join(nonfinite_gradients)
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("nonfinite pose-transport gradient norm")
            optimizer.step()
            nonfinite_parameters = [
                name for name, parameter in model.named_parameters()
                if not torch.isfinite(parameter).all()
            ]
            if nonfinite_parameters:
                raise FloatingPointError(
                    "nonfinite pose-transport parameters: " + ",".join(nonfinite_parameters)
                )
            epoch_loss.append(float(loss.detach().cpu()))
        if epoch in {0, 1, 4, 9, 19, 39, int(args.epochs) - 1}:
            row = {"epoch": epoch + 1, "mean_train_loss": float(np.mean(epoch_loss))}
            history.append(row); print(json.dumps(row), flush=True)

    # Freeze the final fixed epoch before opening any dev pose labels.
    model.eval()
    model_sha = pose_transport_model_content_sha256(model)
    save_minimal_pose_transport_readout(model, model_path, metadata={
        "dataset_content_sha256": dataset_metadata["content_sha256"],
        "physical_map_sha256": physical.content_sha256,
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "surface_mapper_file_sha256": mapper_sha,
        "shared_query_map_projection": shared_surface_space,
        "p1_random_query_geometry_modalities_disabled_at_initialization": shared_surface_space,
        "train_image_ids": arrays["image_ids"][:train_count].tolist(),
        "dev_image_ids": arrays["image_ids"][train_count:].tolist(),
        "fixed_final_epoch": int(args.epochs),
        "dev_labels_opened_before_model_freeze": False,
    })

    def evaluate(indices: range, edges_by_query) -> np.ndarray:
        return np.stack([
            _query_scores(model, arrays, q, edges_by_query[q], device=device, gradient=False).cpu().numpy()
            for q in indices
        ])

    train_indices = range(0, train_count)
    dev_indices = range(train_count, query_count)
    train_score = evaluate(train_indices, medium_edges)
    dev_medium_score = evaluate(dev_indices, medium_edges)
    # Exact-child control is constructed only after the trained model is
    # frozen; it cannot change weights or hyperparameters.
    fine_edges = [None] * query_count
    for q in dev_indices:
        fine_edges[q] = _edges_for_query(arrays, q, hierarchy, stage="fine")
    dev_fine_score = evaluate(dev_indices, fine_edges)
    train_metrics = _metrics(
        train_score, arrays["translation_m"][:train_count], arrays["rotation_deg"][:train_count],
        arrays["candidate_valid"][:train_count], arrays["image_ids"][:train_count],
    )
    dev_medium_metrics = _metrics(
        dev_medium_score, arrays["translation_m"][train_count:], arrays["rotation_deg"][train_count:],
        arrays["candidate_valid"][train_count:], arrays["image_ids"][train_count:],
    )
    dev_fine_metrics = _metrics(
        dev_fine_score, arrays["translation_m"][train_count:], arrays["rotation_deg"][train_count:],
        arrays["candidate_valid"][train_count:], arrays["image_ids"][train_count:],
    )
    report = {
        "artifact_type": REPORT_SCHEMA,
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "dataset_content_sha256": dataset_metadata["content_sha256"],
        "physical_map_sha256": physical.content_sha256,
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "surface_mapper_file_sha256": mapper_sha,
        "shared_query_map_projection": shared_surface_space,
        "p1_random_query_geometry_modalities_disabled_at_initialization": shared_surface_space,
        "model_file_sha256": file_sha256(model_path),
        "model_content_sha256": model_sha,
        "seed": int(args.seed), "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "split_semantics": "single_seq11_temporal_mapping_train_dev_no_fivefold_v1",
        "train_image_ids": arrays["image_ids"][:train_count].tolist(),
        "dev_image_ids": arrays["image_ids"][train_count:].tolist(),
        "dev_labels_opened_before_model_freeze": False,
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "end_to_end_metrics_exclude_gt_anchor": True,
        "train_medium_transport": train_metrics,
        "dev_medium_sparse_transport": dev_medium_metrics,
        "dev_exact_child_edge_control": dev_fine_metrics,
        "history": history,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "map_disjoint_local_backend_diagnostic_not_end_to_end_localization",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
