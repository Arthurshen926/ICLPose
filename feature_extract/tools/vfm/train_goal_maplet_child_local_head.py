"""Train a regenerable child-local metric head over one canonical VFM field."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.local_head import ChildLocalReadoutHead, save_child_local_head
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels, contributor_multiscale_child_distribution
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


def _extract(
    directory: Path,
    mapper,
    physical: GoalMapletPhysicalMap,
    *,
    excluded_trajectories: set[str],
    maximum_images: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    query, parent_rows, target, weight = [], [], [], []
    image_ids, trajectories = [], []
    maximum_children = int(np.max(np.diff(physical.maplet_child_offsets)))
    config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    paths = sorted(Path(directory).glob("*.npz"))
    used_images = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        trajectory = str(metadata["trajectory_id"])
        if trajectory in excluded_trajectories:
            continue
        if maximum_images > 0 and used_images >= maximum_images:
            break
        labels = ContributorLabels.load_npz(path)
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = select_spatially_balanced_radio_final_regions(raw)
        descriptor = encode_radio_final_regions(mapped, token_xy, config)
        truth, _ = contributor_multiscale_child_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
        )
        parent_mass = np.zeros((truth.shape[0], physical.maplet_ids.size), dtype=np.float32)
        for parent in range(physical.maplet_ids.size):
            start, end = int(physical.maplet_child_offsets[parent]), int(physical.maplet_child_offsets[parent + 1])
            parent_mass[:, parent] = np.sum(truth[:, start:end], axis=1)
        selected_parent = np.argmax(parent_mass, axis=1)
        selected_mass = parent_mass[np.arange(parent_mass.shape[0]), selected_parent]
        total_mass = np.sum(truth, axis=1)
        for row in range(truth.shape[0]):
            parent = int(selected_parent[row])
            start, end = int(physical.maplet_child_offsets[parent]), int(physical.maplet_child_offsets[parent + 1])
            distribution = truth[row, start:end].astype(np.float32)
            if (
                float(selected_mass[row]) < 0.20
                or float(selected_mass[row] / max(total_mass[row], 1e-8)) < 0.55
                or float(np.max(distribution, initial=0.0) / max(selected_mass[row], 1e-8)) < 0.20
            ):
                continue
            padded = np.zeros((maximum_children,), dtype=np.float32)
            padded[: end - start] = distribution / max(float(np.sum(distribution)), 1e-8)
            query.append(descriptor[row])
            parent_rows.append(parent)
            target.append(padded)
            weight.append(float(selected_mass[row]))
        image_ids.append(str(metadata["image_id"]))
        trajectories.append(trajectory)
        used_images += 1
        print(json.dumps({"image_id": image_ids[-1], "retained_samples": len(query)}), flush=True)
    if not query:
        raise ValueError("no child-local training samples")
    return (
        np.asarray(query, dtype=np.float32),
        np.asarray(parent_rows, dtype=np.int64),
        np.asarray(target, dtype=np.float32),
        np.asarray(weight, dtype=np.float32),
        image_ids,
        sorted(set(trajectories)),
    )


def _candidate_table(physical: GoalMapletPhysicalMap) -> tuple[np.ndarray, np.ndarray]:
    maximum = int(np.max(np.diff(physical.maplet_child_offsets)))
    rows = np.full((physical.maplet_ids.size, maximum), -1, dtype=np.int64)
    mask = np.zeros_like(rows, dtype=bool)
    for parent in range(physical.maplet_ids.size):
        start, end = int(physical.maplet_child_offsets[parent]), int(physical.maplet_child_offsets[parent + 1])
        rows[parent, : end - start] = np.arange(start, end)
        mask[parent, : end - start] = True
    return rows, mask


@torch.no_grad()
def _evaluate(
    model: ChildLocalReadoutHead,
    query: np.ndarray,
    parent: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    child_descriptor: torch.Tensor,
    candidate_rows: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    map_code = model.encode_map(child_descriptor)
    losses, recalls1, recalls5, reciprocal, weights = [], [], [], [], []
    for start in range(0, query.shape[0], batch_size):
        end = min(query.shape[0], start + batch_size)
        q = model.encode_query(torch.as_tensor(query[start:end], device=device))
        p = torch.as_tensor(parent[start:end], device=device)
        rows = candidate_rows[p]
        mask = candidate_mask[p]
        code = map_code[torch.clamp(rows, min=0)]
        logits = torch.einsum("bd,bkd->bk", q, code) / float(temperature)
        logits = logits.masked_fill(~mask, -1e4)
        target_batch = torch.as_tensor(target[start:end], device=device)
        log_probability = F.log_softmax(logits, dim=1)
        loss = -torch.sum(target_batch * log_probability, dim=1)
        order = torch.argsort(logits, dim=1, descending=True)
        ranked_target = torch.gather(target_batch, 1, order)
        positive = ranked_target > 1e-8
        first = torch.argmax(positive.to(torch.int64), dim=1) + 1
        has_positive = torch.any(positive, dim=1)
        losses.extend(loss.cpu().tolist())
        recalls1.extend(torch.sum(ranked_target[:, :1], dim=1).cpu().tolist())
        recalls5.extend(torch.sum(ranked_target[:, :5], dim=1).cpu().tolist())
        reciprocal.extend(torch.where(has_positive, 1.0 / first.float(), torch.zeros_like(first, dtype=torch.float32)).cpu().tolist())
        weights.extend(weight[start:end].tolist())
    w = np.asarray(weights, dtype=np.float64)
    return {
        "loss": float(np.average(losses, weights=w)),
        "recall_at_1": float(np.average(recalls1, weights=w)),
        "recall_at_5": float(np.average(recalls5, weights=w)),
        "mrr": float(np.average(reciprocal, weights=w)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_contributors", required=True)
    parser.add_argument("--validation_contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_head", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--exclude_train_trajectories", nargs="*", default=["seq11", "seq3", "seq5", "seq13"])
    parser.add_argument("--maximum_train_images", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=194917)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_head), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite child-local head")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout = readout_canonical_field(field, physical)
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    train = _extract(
        Path(args.train_contributors), mapper, physical,
        excluded_trajectories=set(args.exclude_train_trajectories),
        maximum_images=int(args.maximum_train_images),
    )
    validation = _extract(
        Path(args.validation_contributors), mapper, physical,
        excluded_trajectories=set(), maximum_images=0,
    )
    if set(train[5]) & set(validation[5]):
        raise ValueError("child-local train and validation trajectories overlap")
    device = torch.device(str(args.device))
    model = ChildLocalReadoutHead(field.feature_dim).to(device)
    child_descriptor = torch.as_tensor(readout.child_descriptors, dtype=torch.float32, device=device)
    candidate_rows_np, candidate_mask_np = _candidate_table(physical)
    candidate_rows = torch.as_tensor(candidate_rows_np, device=device)
    candidate_mask = torch.as_tensor(candidate_mask_np, device=device)
    baseline = _evaluate(
        model, *validation[:4], child_descriptor, candidate_rows, candidate_mask,
        batch_size=int(args.batch_size), temperature=float(args.temperature), device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_metrics, best_epoch, stale = baseline, 0, 0
    order_generator = np.random.default_rng(int(args.seed))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = order_generator.permutation(train[0].shape[0])
        losses = []
        for offset in range(0, order.size, int(args.batch_size)):
            indices = order[offset : offset + int(args.batch_size)]
            q = model.encode_query(torch.as_tensor(train[0][indices], device=device))
            parent = torch.as_tensor(train[1][indices], device=device)
            rows = candidate_rows[parent]
            mask = candidate_mask[parent]
            map_code = model.encode_map(child_descriptor)
            candidate_code = map_code[torch.clamp(rows, min=0)]
            logits = torch.einsum("bd,bkd->bk", q, candidate_code) / float(args.temperature)
            logits = logits.masked_fill(~mask, -1e4)
            target = torch.as_tensor(train[2][indices], device=device)
            sample_weight = torch.as_tensor(train[3][indices], device=device)
            per_sample = -torch.sum(target * F.log_softmax(logits, dim=1), dim=1)
            loss = torch.sum(per_sample * sample_weight) / torch.clamp(torch.sum(sample_weight), min=1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _evaluate(
            model, *validation[:4], child_descriptor, candidate_rows, candidate_mask,
            batch_size=int(args.batch_size), temperature=float(args.temperature), device=device,
        )
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["recall_at_5"] > best_metrics["recall_at_5"] + 1e-5:
            best_metrics, best_epoch = metrics, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    model.load_state_dict(best_state)
    metadata = {
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_file_sha256": file_sha256(Path(args.surface_mapper)),
        "supervision": "exact_clean_2dgs_child_distribution_parent_conditioned",
        "train_trajectory_ids": train[5],
        "validation_trajectory_ids": validation[5],
        "strict_holdout_trajectory_ids": ["seq3", "seq5", "seq13"],
        "train_image_count": len(train[4]),
        "validation_image_count": len(validation[4]),
        "train_sample_count": int(train[0].shape[0]),
        "validation_sample_count": int(validation[0].shape[0]),
        "best_epoch": int(best_epoch),
        "best_validation": best_metrics,
        "baseline_validation": baseline,
    }
    save_child_local_head(output, model, metadata)
    report = {
        "stage": "train_goal_maplet_child_local_head",
        "output_head": str(output),
        "output_head_file_sha256": file_sha256(output),
        **metadata,
        "history": history,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
