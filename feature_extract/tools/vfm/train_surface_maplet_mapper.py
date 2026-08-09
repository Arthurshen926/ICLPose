"""Train a RADIO-final mapper from 2DGS surface-maplet cross-view identity.

The supervision source is deliberately narrow: a positive pair means that two
RADIO-final regions from different real mapping images were assigned to the
same persistent 2DGS surface maplet. No SfM point, feature track, query pose, or
RADIO intermediate tensor is accepted by this trainer.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.localization.surface_maplet_mapper import (
    SurfaceMapletMapper,
    SurfaceMapletMapperConfig,
    load_surface_maplet_mapper,
    maplet_prototype_retrieval_metrics,
    pool_radio_final_context_torch,
    save_surface_maplet_mapper,
    surface_maplet_contrastive_loss,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    VfmSurfaceMapletBank,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("integer tuple cannot be empty")
    return values


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("float tuple cannot be empty")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a track-free RADIO-final mapper from 2DGS surface maplets"
    )
    parser.add_argument("--surface_maplets", required=True)
    parser.add_argument("--radio_final_manifest", required=True)
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument(
        "--initial_checkpoint",
        default="",
        help="Optional surface-maplet mapper to fine-tune; epoch 0 remains a hard validation baseline.",
    )
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard_negative_radius", type=float, default=0.75)
    parser.add_argument("--hard_negative_margin", type=float, default=0.25)
    parser.add_argument("--hard_negative_weight", type=float, default=0.20)
    parser.add_argument("--pool_sizes", default="1,3,5,9")
    parser.add_argument("--pool_weights", default="0.4,0.3,0.2,0.1")
    parser.add_argument(
        "--validation_stride",
        type=int,
        default=4,
        help="Every Nth sorted mapping image is held out from mapper fitting.",
    )
    parser.add_argument("--validation_offset", type=int, default=0)
    parser.add_argument(
        "--training_trajectory_ids",
        nargs="*",
        default=(),
        help=(
            "Optional trajectory-level fit split. Must be paired with "
            "--validation_trajectory_ids; unlisted trajectories are ignored."
        ),
    )
    parser.add_argument(
        "--validation_trajectory_ids", nargs="*", default=()
    )
    parser.add_argument(
        "--prototype_trajectory_ids",
        nargs="*",
        default=(),
        help=(
            "Optional subset of training trajectories used as retrieval prototypes "
            "during validation. This must match the trajectories baked into the "
            "deployed identity map; mapper fitting still uses every training trajectory."
        ),
    )
    parser.add_argument(
        "--strict_holdout_trajectory_ids",
        nargs="*",
        default=("seq3", "seq5", "seq13"),
    )
    parser.add_argument(
        "--batch_maplets",
        type=int,
        default=64,
        help="For large maps, sample this many maplet identities per optimization step.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=8,
        help="Large-map optimization steps per epoch; small maps still use one full step.",
    )
    parser.add_argument(
        "--full_batch_max_observations",
        type=int,
        default=5000,
    )
    return parser.parse_args(argv)


def _validate_final_only_manifest(manifest: TokenBankManifest, layer_name: str) -> None:
    if "intermediate" in str(layer_name).lower():
        raise ValueError("RADIO intermediate is forbidden by the 2DGS production protocol")
    found = False
    for record in manifest.records:
        for layer in record.layers:
            if layer.name != layer_name:
                continue
            found = True
            if str(layer.layer).lower() != "final":
                raise ValueError("the requested production VFM layer is not RADIO final")
    if not found:
        raise ValueError(f"RADIO-final layer {layer_name!r} is absent from the token manifest")


def _observation_labels(
    bank: VfmSurfaceMapletBank,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels: list[int] = []
    centers: list[np.ndarray] = []
    for row, maplet_id in enumerate(bank.maplet_ids.tolist()):
        start = int(bank.view_offsets[row])
        end = int(bank.view_offsets[row + 1])
        labels.extend([int(maplet_id)] * (end - start))
        centers.extend([bank.centers[row]] * (end - start))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(bank.view_image_ids, dtype=object),
        np.asarray(bank.view_quality_scores, dtype=np.float32),
        np.asarray(centers, dtype=np.float32).reshape(-1, 3),
    )


def _load_raw_feature_maps(
    manifest: TokenBankManifest,
    required_images: set[str],
    layer_name: str,
) -> dict[str, np.ndarray]:
    record_by_image = {record.image_id: record for record in manifest.records}
    missing = sorted(required_images - set(record_by_image))
    if missing:
        raise ValueError(f"maplet view is absent from RADIO-final manifest: {missing[0]}")
    feature_maps: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, int, int] | None = None
    for image_id in sorted(required_images):
        record = record_by_image[image_id]
        with np.load(record.token_path) as data:
            if layer_name not in data:
                raise ValueError(f"RADIO-final layer {layer_name!r} is absent from {record.token_path}")
            feature = np.asarray(data[layer_name])
        if feature.ndim == 4 and int(feature.shape[0]) == 1:
            feature = feature[0]
        if feature.ndim != 3:
            raise ValueError(f"RADIO-final feature for {image_id} must have shape (C,H,W)")
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in feature.shape)
        elif tuple(feature.shape) != expected_shape:
            raise ValueError("all mapper-training RADIO-final maps must have the same shape")
        feature_maps[image_id] = feature
    return feature_maps


def _split_images(
    image_ids: np.ndarray,
    stride: int,
    offset: int,
) -> tuple[set[str], set[str]]:
    images = sorted(set(str(value) for value in image_ids.tolist()))
    if int(stride) < 2:
        raise ValueError("validation_stride must be at least 2")
    resolved_offset = int(offset) % int(stride)
    validation = {
        image_id for index, image_id in enumerate(images) if index % int(stride) == resolved_offset
    }
    training = set(images) - validation
    if not training or not validation:
        raise ValueError("image-disjoint split must contain both training and validation images")
    return training, validation


def _split_images_by_trajectory(
    image_ids: np.ndarray,
    training_trajectory_ids: Sequence[str],
    validation_trajectory_ids: Sequence[str],
    strict_holdout_trajectory_ids: Sequence[str],
) -> tuple[set[str], set[str]]:
    training_ids = {str(value) for value in training_trajectory_ids}
    validation_ids = {str(value) for value in validation_trajectory_ids}
    strict_ids = {str(value) for value in strict_holdout_trajectory_ids}
    if (
        not training_ids
        or not validation_ids
        or training_ids & validation_ids
        or training_ids & strict_ids
        or validation_ids & strict_ids
    ):
        raise ValueError(
            "mapper fit/validation/strict-holdout trajectory roles must be "
            "non-empty and mutually disjoint"
        )
    images = {str(value) for value in image_ids.tolist()}
    training = {
        image for image in images if image.split("/", 1)[0] in training_ids
    }
    validation = {
        image
        for image in images
        if image.split("/", 1)[0] in validation_ids
    }
    if not training or not validation:
        raise ValueError("trajectory-level mapper split has an empty role")
    return training, validation


def _resolve_prototype_images(
    training_images: set[str],
    validation_images: set[str],
    prototype_trajectory_ids: Sequence[str],
    strict_holdout_trajectory_ids: Sequence[str],
    *,
    trajectory_split: bool,
) -> set[str]:
    """Resolve the deployment-faithful prototype side of validation.

    A mapper may be fitted from more trajectories than are baked into the
    production map. Model selection must nevertheless use only the production
    map trajectories as prototypes, otherwise validation measures a richer map
    than the one available at localization time.
    """

    prototype_ids = {str(value) for value in prototype_trajectory_ids}
    if not prototype_ids:
        return set(training_images)
    if not trajectory_split:
        raise ValueError(
            "explicit prototype trajectory IDs require a trajectory-level mapper split"
        )
    training_ids = {
        str(image_id).split("/", 1)[0] for image_id in training_images
    }
    validation_ids = {
        str(image_id).split("/", 1)[0] for image_id in validation_images
    }
    strict_ids = {str(value) for value in strict_holdout_trajectory_ids}
    if (
        not prototype_ids.issubset(training_ids)
        or prototype_ids & validation_ids
        or prototype_ids & strict_ids
    ):
        raise ValueError(
            "prototype trajectories must be a non-empty subset of mapper-fit "
            "trajectories and disjoint from validation/strict holdout"
        )
    prototype_images = {
        image_id
        for image_id in training_images
        if str(image_id).split("/", 1)[0] in prototype_ids
    }
    if not prototype_images:
        raise ValueError("prototype trajectory selection contains no mapper observations")
    return prototype_images


def _mapped_observation_descriptors(
    model: SurfaceMapletMapper,
    feature_maps: dict[str, np.ndarray],
    observation_image_ids: np.ndarray,
    token_xy: np.ndarray,
    selected_mask: np.ndarray,
    device: torch.device,
    pool_sizes: tuple[int, ...],
    pool_weights: tuple[float, ...],
    image_batch_size: int = 16,
) -> tuple[torch.Tensor, np.ndarray]:
    descriptor_parts: list[torch.Tensor] = []
    row_parts: list[np.ndarray] = []
    selected_by_image = [
        (
            image_id,
            np.flatnonzero(selected_mask & (observation_image_ids == image_id)),
        )
        for image_id in sorted(feature_maps)
    ]
    selected_by_image = [
        (image_id, rows) for image_id, rows in selected_by_image if rows.size > 0
    ]
    if int(image_batch_size) <= 0:
        raise ValueError("image_batch_size must be positive")
    for batch_start in range(0, len(selected_by_image), int(image_batch_size)):
        batch = selected_by_image[
            batch_start : batch_start + int(image_batch_size)
        ]
        raw = torch.as_tensor(
            np.stack([feature_maps[image_id] for image_id, _rows in batch], axis=0),
            dtype=torch.float32,
            device=device,
        )
        mapped_batch = model(raw)
        for batch_row, (_image_id, rows) in enumerate(batch):
            xy = torch.as_tensor(
                token_xy[rows],
                dtype=torch.float32,
                device=device,
            )
            descriptor_parts.append(
                pool_radio_final_context_torch(
                    mapped_batch[batch_row],
                    xy,
                    pool_sizes=pool_sizes,
                    pool_weights=pool_weights,
                )
            )
            row_parts.append(rows)
    if not descriptor_parts:
        raise ValueError("selected observation split contains no descriptors")
    descriptors = torch.cat(descriptor_parts, dim=0)
    rows = np.concatenate(row_parts)
    order = np.argsort(rows, kind="mergesort")
    return descriptors[torch.as_tensor(order, dtype=torch.long, device=device)], rows[order]


def _evaluate_model(
    model: SurfaceMapletMapper,
    feature_maps: dict[str, np.ndarray],
    image_ids: np.ndarray,
    token_xy: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    prototype_mask: np.ndarray,
    validation_mask: np.ndarray,
    device: torch.device,
    pool_sizes: tuple[int, ...],
    pool_weights: tuple[float, ...],
) -> dict[str, float | int]:
    model.eval()
    evaluation_mask = np.asarray(prototype_mask, dtype=bool) | np.asarray(
        validation_mask, dtype=bool
    )
    expected_rows = np.flatnonzero(evaluation_mask)
    if not len(expected_rows):
        raise ValueError("mapper evaluation split contains no observations")
    with torch.no_grad():
        descriptors, rows = _mapped_observation_descriptors(
            model,
            feature_maps,
            image_ids,
            token_xy,
            evaluation_mask,
            device,
            pool_sizes,
            pool_weights,
        )
    descriptor_array = descriptors.detach().cpu().numpy()
    if not np.array_equal(rows, expected_rows):
        raise RuntimeError("mapper evaluation did not preserve observation order")
    return maplet_prototype_retrieval_metrics(
        descriptor_array,
        labels[rows],
        np.asarray(prototype_mask, dtype=bool)[rows],
        np.asarray(validation_mask, dtype=bool)[rows],
        quality[rows],
    )


def _evaluate_raw_radio(
    feature_maps: dict[str, np.ndarray],
    image_ids: np.ndarray,
    token_xy: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    prototype_mask: np.ndarray,
    validation_mask: np.ndarray,
    region_config: RadioFinalRegionConfig,
) -> dict[str, float | int]:
    descriptors: np.ndarray | None = None
    for image_id in sorted(feature_maps):
        rows = np.flatnonzero(image_ids == image_id)
        values = encode_radio_final_regions(feature_maps[image_id], token_xy[rows], region_config)
        if descriptors is None:
            descriptors = np.zeros((len(image_ids), values.shape[1]), dtype=np.float32)
        descriptors[rows] = values
    if descriptors is None:
        raise ValueError("no raw RADIO-final descriptors to evaluate")
    return maplet_prototype_retrieval_metrics(
        descriptors,
        labels,
        prototype_mask,
        validation_mask,
        quality,
    )


def _selection_key(metrics: dict[str, float | int]) -> tuple[float, float, float]:
    return (
        float(metrics["recall_at_5"]),
        float(metrics["recall_at_1"]),
        float(metrics["mean_reciprocal_rank"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(
        str(args.device)
        if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )

    bank = VfmSurfaceMapletBank.load_npz(Path(args.surface_maplets))
    manifest = TokenBankManifest.from_json(Path(args.radio_final_manifest))
    manifest.validate(verify_checksums=False)
    _validate_final_only_manifest(manifest, str(args.radio_final_layer))
    labels, image_ids, quality, centers = _observation_labels(bank)
    if len(labels) != len(bank.view_image_ids):
        raise RuntimeError("surface-maplet view offsets are inconsistent")
    use_trajectory_split = bool(args.training_trajectory_ids) or bool(
        args.validation_trajectory_ids
    )
    if use_trajectory_split:
        if not (
            bool(args.training_trajectory_ids)
            and bool(args.validation_trajectory_ids)
        ):
            raise ValueError(
                "training and validation trajectory IDs must be paired"
            )
        training_images, validation_images = _split_images_by_trajectory(
            image_ids,
            args.training_trajectory_ids,
            args.validation_trajectory_ids,
            args.strict_holdout_trajectory_ids,
        )
    else:
        training_images, validation_images = _split_images(
            image_ids,
            int(args.validation_stride),
            int(args.validation_offset),
        )
    prototype_images = _resolve_prototype_images(
        training_images,
        validation_images,
        args.prototype_trajectory_ids,
        args.strict_holdout_trajectory_ids,
        trajectory_split=bool(use_trajectory_split),
    )
    feature_maps = _load_raw_feature_maps(
        manifest,
        training_images | validation_images,
        str(args.radio_final_layer),
    )
    train_mask = np.asarray([str(value) in training_images for value in image_ids], dtype=bool)
    validation_mask = np.asarray(
        [str(value) in validation_images for value in image_ids],
        dtype=bool,
    )
    prototype_mask = np.asarray(
        [str(value) in prototype_images for value in image_ids],
        dtype=bool,
    )
    pool_sizes = _parse_int_tuple(args.pool_sizes)
    pool_weights = _parse_float_tuple(args.pool_weights)
    region_config = RadioFinalRegionConfig(pool_sizes=pool_sizes, pool_weights=pool_weights)
    raw_metrics = _evaluate_raw_radio(
        feature_maps,
        image_ids,
        bank.view_token_xy,
        labels,
        quality,
        prototype_mask,
        validation_mask,
        region_config,
    )
    input_descriptor_metrics = maplet_prototype_retrieval_metrics(
        bank.view_descriptors,
        labels,
        prototype_mask,
        validation_mask,
        quality,
    )

    input_dim = int(next(iter(feature_maps.values())).shape[0])
    initial_checkpoint_metadata: dict[str, object] = {}
    if str(args.initial_checkpoint):
        loaded_mapper, initial_checkpoint_metadata = load_surface_maplet_mapper(
            Path(args.initial_checkpoint),
            device=str(device),
        )
        model = loaded_mapper.model
        config = model.config
        if int(config.input_dim) != input_dim:
            raise ValueError("initial surface mapper input dimension differs from RADIO final")
    else:
        config = SurfaceMapletMapperConfig(
            input_dim=input_dim,
            hidden_dim=int(args.hidden_dim),
            output_dim=int(args.output_dim),
            dropout=float(args.dropout),
        )
        model = SurfaceMapletMapper(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_rows = np.flatnonzero(train_mask)
    image_index = {
        image_id: index for index, image_id in enumerate(sorted(set(str(value) for value in image_ids)))
    }
    image_label_array = np.asarray(
        [image_index[str(value)] for value in image_ids],
        dtype=np.int64,
    )
    use_full_batch = len(train_rows) <= int(args.full_batch_max_observations)
    eligible_rows_by_label: dict[int, np.ndarray] = {}
    if not use_full_batch:
        for label in np.unique(labels[train_rows]).tolist():
            rows = train_rows[labels[train_rows] == int(label)]
            if len(set(str(image_ids[row]) for row in rows.tolist())) >= 2:
                eligible_rows_by_label[int(label)] = rows
        if len(eligible_rows_by_label) < 2:
            raise ValueError("large-map training has too few cross-view maplet identities")
        if int(args.batch_maplets) < 2 or int(args.steps_per_epoch) <= 0:
            raise ValueError("large-map batch_maplets/steps_per_epoch are invalid")
    batch_rng = np.random.default_rng(int(args.seed) + 101)

    initial_metrics = _evaluate_model(
        model,
        feature_maps,
        image_ids,
        bank.view_token_xy,
        labels,
        quality,
        prototype_mask,
        validation_mask,
        device,
        pool_sizes,
        pool_weights,
    )
    best_metrics: dict[str, float | int] | None = dict(initial_metrics)
    best_epoch = 0
    best_loss = float("inf")
    history: list[dict[str, object]] = [
        {"epoch": 0, "loss": None, "loss_stats": None, "validation": initial_metrics}
    ]
    stale_epochs = 0
    checkpoint_metadata_base = {
        "supervision": "2dgs_surface_maplet_cross_view_identity",
        "vfm_layer": "radio_final",
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "training_images": sorted(training_images),
        "validation_images": sorted(validation_images),
        "prototype_images": sorted(prototype_images),
        "training_trajectory_ids": sorted(
            {value.split("/", 1)[0] for value in training_images}
        ),
        "validation_trajectory_ids": sorted(
            {value.split("/", 1)[0] for value in validation_images}
        ),
        "prototype_trajectory_ids": sorted(
            {value.split("/", 1)[0] for value in prototype_images}
        ),
        "strict_holdout_trajectory_ids": sorted(
            str(value) for value in args.strict_holdout_trajectory_ids
        ),
        "trajectory_split": bool(use_trajectory_split),
        "pool_sizes": list(pool_sizes),
        "pool_weights": list(pool_weights),
        "initial_checkpoint": str(args.initial_checkpoint),
        "initial_checkpoint_metadata": initial_checkpoint_metadata,
    }
    save_surface_maplet_mapper(
        Path(args.output_checkpoint),
        model,
        metadata={
            **checkpoint_metadata_base,
            "best_epoch": 0,
            "best_validation": best_metrics,
        },
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        step_losses: list[float] = []
        step_stats: list[dict[str, float | int]] = []
        optimization_steps = 1 if use_full_batch else int(args.steps_per_epoch)
        for _step in range(optimization_steps):
            if use_full_batch:
                selected_rows = train_rows
            else:
                eligible_labels = np.asarray(sorted(eligible_rows_by_label), dtype=np.int64)
                selected_labels = batch_rng.choice(
                    eligible_labels,
                    size=min(int(args.batch_maplets), len(eligible_labels)),
                    replace=False,
                )
                selected: list[int] = []
                for label in selected_labels.tolist():
                    rows_for_label = eligible_rows_by_label[int(label)]
                    first = int(batch_rng.choice(rows_for_label))
                    other = rows_for_label[
                        image_ids[rows_for_label] != image_ids[first]
                    ]
                    second = int(batch_rng.choice(other))
                    selected.extend((first, second))
                selected_rows = np.asarray(sorted(selected), dtype=np.int64)
            selected_mask = np.zeros((len(labels),), dtype=bool)
            selected_mask[selected_rows] = True
            optimizer.zero_grad(set_to_none=True)
            descriptors, rows = _mapped_observation_descriptors(
                model,
                feature_maps,
                image_ids,
                bank.view_token_xy,
                selected_mask,
                device,
                pool_sizes,
                pool_weights,
            )
            if not np.array_equal(rows, selected_rows):
                raise RuntimeError("mapper training did not preserve selected observation order")
            loss, loss_stats = surface_maplet_contrastive_loss(
                descriptors,
                torch.as_tensor(labels[rows], dtype=torch.long, device=device),
                torch.as_tensor(image_label_array[rows], dtype=torch.long, device=device),
                quality_weights=torch.as_tensor(quality[rows], dtype=torch.float32, device=device),
                maplet_centers=torch.as_tensor(centers[rows], dtype=torch.float32, device=device),
                temperature=float(args.temperature),
                hard_negative_radius=float(args.hard_negative_radius),
                hard_negative_margin=float(args.hard_negative_margin),
                hard_negative_weight=float(args.hard_negative_weight),
            )
            if int(loss_stats["valid_anchor_count"]) == 0:
                raise ValueError("training batch has no cross-view positive surface-maplet pairs")
            loss.backward()
            if float(args.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
            optimizer.step()
            step_losses.append(float(loss.detach().cpu().item()))
            step_stats.append(loss_stats)
        loss_value = float(np.mean(step_losses))
        loss_stats = {
            key: (
                float(np.mean([float(stats[key]) for stats in step_stats]))
                if "loss" in key
                else int(np.sum([int(stats[key]) for stats in step_stats]))
            )
            for key in step_stats[0]
        }

        should_evaluate = (
            epoch == 1
            or epoch == int(args.epochs)
            or epoch % max(int(args.eval_every), 1) == 0
        )
        if not should_evaluate:
            continue
        metrics = _evaluate_model(
            model,
            feature_maps,
            image_ids,
            bank.view_token_xy,
            labels,
            quality,
            prototype_mask,
            validation_mask,
            device,
            pool_sizes,
            pool_weights,
        )
        history.append(
            {
                "epoch": int(epoch),
                "loss": loss_value,
                "loss_stats": loss_stats,
                "validation": metrics,
            }
        )
        improved = best_metrics is None or _selection_key(metrics) > _selection_key(best_metrics)
        if improved:
            best_metrics = dict(metrics)
            best_epoch = int(epoch)
            best_loss = loss_value
            stale_epochs = 0
            save_surface_maplet_mapper(
                Path(args.output_checkpoint),
                model,
                metadata={
                    **checkpoint_metadata_base,
                    "best_epoch": int(best_epoch),
                    "best_validation": best_metrics,
                },
            )
        else:
            stale_epochs += max(int(args.eval_every), 1)
        if int(args.patience) > 0 and stale_epochs >= int(args.patience):
            break

    if best_metrics is None:
        raise RuntimeError("surface-maplet mapper training produced no validation measurement")
    summary = {
        "stage": "train_surface_maplet_mapper",
        "supervision": "2dgs_surface_maplet_cross_view_identity",
        "production_contract": {
            "vfm_layer": "radio_final",
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "query_or_test_pose_used": False,
            "validation_views_disjoint_from_fit": True,
            "trajectory_level_split": bool(use_trajectory_split),
            "validation_prototypes_match_deployed_map": True,
        },
        "split": {
            "training_images": sorted(training_images),
            "validation_images": sorted(validation_images),
            "prototype_images": sorted(prototype_images),
            "training_trajectory_ids": sorted(
                {value.split("/", 1)[0] for value in training_images}
            ),
            "validation_trajectory_ids": sorted(
                {value.split("/", 1)[0] for value in validation_images}
            ),
            "prototype_trajectory_ids": sorted(
                {value.split("/", 1)[0] for value in prototype_images}
            ),
            "strict_holdout_trajectory_ids": sorted(
                str(value) for value in args.strict_holdout_trajectory_ids
            ),
            "training_observations": int(np.sum(train_mask)),
            "validation_observations": int(np.sum(validation_mask)),
            "prototype_observations": int(np.sum(prototype_mask)),
        },
        "raw_radio_final_validation": raw_metrics,
        "input_maplet_descriptor_validation": input_descriptor_metrics,
        "initial_model_validation": initial_metrics,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss) if np.isfinite(best_loss) else None,
        "best_validation": best_metrics,
        "history": history,
        "inputs": {
            "surface_maplets": str(args.surface_maplets),
            "radio_final_manifest": str(args.radio_final_manifest),
            "radio_final_layer": str(args.radio_final_layer),
            "initial_checkpoint": str(args.initial_checkpoint),
        },
        "output_checkpoint": str(args.output_checkpoint),
        "config": {
            "model": config.__dict__,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "hard_negative_radius": float(args.hard_negative_radius),
            "hard_negative_margin": float(args.hard_negative_margin),
            "hard_negative_weight": float(args.hard_negative_weight),
            "pool_sizes": list(pool_sizes),
            "pool_weights": list(pool_weights),
            "optimization_mode": "full_batch" if use_full_batch else "sampled_maplet_batches",
            "batch_maplets": int(args.batch_maplets),
            "steps_per_epoch": int(args.steps_per_epoch),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
