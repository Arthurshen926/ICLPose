"""Train a fine RADIO adapter from repeated detector surface observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    SurfaceMetricFeatureMapper,
    SurfaceMetricFeatureMapperConfig,
    save_surface_metric_feature_mapper,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--identities_per_batch", type=int, default=192)
    parser.add_argument("--samples_per_identity", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _supervised_contrastive(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    views: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    similarity = descriptors @ descriptors.T / float(temperature)
    identity = torch.eye(
        descriptors.shape[0], dtype=torch.bool, device=descriptors.device
    )
    positive = (
        (labels[:, None] == labels[None, :])
        & (views[:, None] != views[None, :])
        & ~identity
    )
    denominator = ~identity
    valid = torch.any(positive, dim=1)
    negative_infinity = torch.finfo(similarity.dtype).min
    numerator = torch.logsumexp(
        torch.where(positive, similarity, negative_infinity), dim=1
    )
    normalizer = torch.logsumexp(
        torch.where(denominator, similarity, negative_infinity), dim=1
    )
    return torch.mean((normalizer - numerator)[valid])


@torch.no_grad()
def _heldout_retrieval(
    model: SurfaceMetricFeatureMapper | None,
    features: np.ndarray,
    views: np.ndarray,
    rows_by_label: list[np.ndarray],
    device: torch.device,
) -> dict[str, float | int]:
    prototypes: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    query_labels: list[int] = []
    for label, rows in enumerate(rows_by_label):
        heldout = (views[rows] % 5) == 0
        if not np.any(heldout) or not np.any(~heldout):
            continue
        prototypes.append(np.mean(features[rows[~heldout]], axis=0))
        chosen = rows[heldout][:4]
        queries.extend(features[chosen])
        query_labels.extend([label] * len(chosen))
    if not prototypes or not queries:
        return {"prototype_count": 0, "query_count": 0}
    prototype = torch.as_tensor(np.stack(prototypes), device=device)
    query = torch.as_tensor(np.stack(queries), device=device)
    if model is not None:
        model.eval()
        prototype = model(prototype)
        query = model(query)
    else:
        prototype = F.normalize(prototype, p=2, dim=1)
        query = F.normalize(query, p=2, dim=1)
    # Only labels with a held-out query were appended, in original order.
    retained_labels = [
        label
        for label, rows in enumerate(rows_by_label)
        if np.any((views[rows] % 5) == 0) and np.any((views[rows] % 5) != 0)
    ]
    target_column = {
        int(label): column for column, label in enumerate(retained_labels)
    }
    ranks: list[int] = []
    for start in range(0, query.shape[0], 512):
        score = query[start : start + 512] @ prototype.T
        order = torch.argsort(score, dim=1, descending=True).cpu().numpy()
        for local_row, columns in enumerate(order):
            label = int(query_labels[start + local_row])
            ranks.append(
                int(np.flatnonzero(columns == target_column[label])[0]) + 1
            )
    rank = np.asarray(ranks, dtype=np.int64)
    return {
        "prototype_count": int(prototype.shape[0]),
        "query_count": int(rank.size),
        "recall_at_1": float(np.mean(rank <= 1)),
        "recall_at_5": float(np.mean(rank <= 5)),
        "recall_at_10": float(np.mean(rank <= 10)),
        "median_rank": float(np.median(rank)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite metric mapper outputs")
    with np.load(Path(args.samples), allow_pickle=False) as data:
        features = np.asarray(data["radio_features"], dtype=np.float32)
        source_ids = np.asarray(data["source_indices"], dtype=np.int64)
        views = np.asarray(data["view_indices"], dtype=np.int64)
    unique, inverse, counts = np.unique(
        source_ids, return_inverse=True, return_counts=True
    )
    eligible = np.flatnonzero(counts >= int(args.samples_per_identity))
    rows_by_label = [
        np.flatnonzero(inverse == int(label)).astype(np.int64) for label in eligible
    ]
    rows_by_label = [
        rows
        for rows in rows_by_label
        if np.unique(views[rows]).size >= int(args.samples_per_identity)
    ]
    if len(rows_by_label) < int(args.identities_per_batch):
        raise ValueError("insufficient repeated surface identities for training")
    device = torch.device(str(args.device))
    model = SurfaceMetricFeatureMapper(
        SurfaceMetricFeatureMapperConfig(input_dim=int(features.shape[1]))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4
    )
    rng = np.random.default_rng(int(args.seed))
    baseline_validation = _heldout_retrieval(
        None, features, views, rows_by_label, device
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(int(args.epochs)):
        losses: list[float] = []
        model.train()
        for _step in range(int(args.steps_per_epoch)):
            labels = rng.choice(
                len(rows_by_label),
                size=int(args.identities_per_batch),
                replace=False,
            )
            batch_rows: list[int] = []
            batch_labels: list[int] = []
            for local_label, label in enumerate(labels.tolist()):
                candidates = rows_by_label[int(label)]
                candidate_views = views[candidates]
                chosen_views = rng.choice(
                    np.unique(candidate_views),
                    size=int(args.samples_per_identity),
                    replace=False,
                )
                for view in chosen_views.tolist():
                    rows = candidates[candidate_views == int(view)]
                    batch_rows.append(int(rng.choice(rows)))
                    batch_labels.append(int(local_label))
            row_array = np.asarray(batch_rows, dtype=np.int64)
            descriptors = model(
                torch.as_tensor(features[row_array], device=device)
            )
            loss = _supervised_contrastive(
                descriptors,
                torch.as_tensor(batch_labels, device=device),
                torch.as_tensor(views[row_array], device=device),
                float(args.temperature),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        record = {
            "epoch": int(epoch + 1),
            "mean_loss": float(np.mean(losses)),
        }
        history.append(record)
        print(json.dumps(record))
    metadata = {
        "supervision": "same_clean_2dgs_surfel_different_mapping_views",
        "alike_role": "detection_coordinates_only",
        "vfm_layer": "radio_final",
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "identity_count": int(len(rows_by_label)),
        "sample_count": int(features.shape[0]),
    }
    save_surface_metric_feature_mapper(output, model.eval(), metadata)
    learned_validation = _heldout_retrieval(
        model.eval(), features, views, rows_by_label, device
    )
    summary = {
        "stage": "train_detector_surface_radio_metric_mapper",
        "metadata": metadata,
        "config": vars(args),
        "history": history,
        "baseline_heldout_retrieval": baseline_validation,
        "learned_heldout_retrieval": learned_validation,
        "output_checkpoint": str(output),
    }
    summary["config"].pop("force", None)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
