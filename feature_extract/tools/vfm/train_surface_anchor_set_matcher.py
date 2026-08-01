"""Train the map-only ALIKE-set to stable-2DGS-anchor matcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.local_assignment_matcher import (
    local_assignment_loss,
)
from feature_extract.vfm.localization.anchor_feature_contract import (
    anchor_feature_kind,
    compose_anchor_query_descriptors,
)
from feature_extract.vfm.localization.surface_anchor_set_matcher import (
    SurfaceAnchorSetMatcher,
    SurfaceAnchorSetMatcherConfig,
    build_surface_anchor_episode,
    load_surface_anchor_set_matcher,
    save_surface_anchor_set_matcher,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
    LocalFeatureFrame,
)
from feature_extract.vfm.surface_maplet_bank import (
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--local_descriptor_bank", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument(
        "--deployment_replay_dir",
        default="",
        help=(
            "Optional feature-only replay cache produced by "
            "build_surface_anchor_deployment_replay.py. When set, real ALIKE "
            "detections, null points, and RADIO-final maplet priors replace "
            "projection-centered synthetic query nodes."
        ),
    )
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=384)
    parser.add_argument("--validation_episodes", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--pair_loss_weight", type=float, default=0.2)
    parser.add_argument("--no_match_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--positive_assignment_weight",
        type=float,
        default=0.0,
        help=(
            "Positive-node weight in assignment NLL; <=0 uses the per-episode "
            "null/positive ratio capped at 50."
        ),
    )
    parser.add_argument(
        "--disable_balanced_no_match_classes",
        action="store_true",
        help="Disable per-episode balancing of match versus null BCE.",
    )
    parser.add_argument("--negative_episode_fraction", type=float, default=0.35)
    parser.add_argument("--maximum_positive_nodes", type=int, default=64)
    parser.add_argument("--maximum_null_nodes", type=int, default=16)
    parser.add_argument("--minimum_positive_nodes", type=int, default=6)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args(argv)


def _split_value(image_id: str, seed: int) -> int:
    payload = f"{int(seed)}:{image_id}".encode("utf8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 5


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(target, dtype=bool).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels))
    if positives == 0:
        return 0.0
    order = np.argsort(-values, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positives)


class SurfaceAnchorTrainingCorpus:
    deployment_replay = False
    def __init__(
        self,
        *,
        maplets: VfmSurfaceMapletBank,
        anchors: StableSurfaceAnchorMap,
        bank: AnchorLocalDescriptorBank,
        camera_by_image,
        config: SurfaceAnchorSetMatcherConfig,
        minimum_positive_nodes: int,
        maximum_positive_nodes: int,
        maximum_null_nodes: int,
        validation_images: set[str],
    ) -> None:
        self.maplets = maplets
        self.anchors = anchors
        self.bank = bank
        self.feature_kind = anchor_feature_kind(bank.metadata)
        self.camera_by_image = camera_by_image
        self.config = config
        self.minimum_positive_nodes = int(minimum_positive_nodes)
        self.maximum_positive_nodes = int(maximum_positive_nodes)
        self.maximum_null_nodes = int(maximum_null_nodes)
        self.validation_images = set(validation_images)
        self.anchor_row_by_id = anchors.row_by_id()
        self.descriptor_bank_row_by_id = {
            int(anchor_id): int(row)
            for row, anchor_id in enumerate(bank.anchor_ids.tolist())
        }
        self.best_descriptor_by_anchor_image: dict[tuple[int, str], int] = {}
        self.visible_anchors_by_maplet_image: dict[
            tuple[int, str], list[int]
        ] = {}
        for bank_row, anchor_id_value in enumerate(bank.anchor_ids.tolist()):
            anchor_id = int(anchor_id_value)
            anchor_row = self.anchor_row_by_id.get(anchor_id)
            if anchor_row is None:
                continue
            maplet_id = int(anchors.owner_maplet_ids[anchor_row])
            start = int(bank.descriptor_offsets[bank_row])
            end = int(bank.descriptor_offsets[bank_row + 1])
            for descriptor_row in range(start, end):
                image_id = str(bank.support_image_ids[descriptor_row])
                key = (anchor_id, image_id)
                previous = self.best_descriptor_by_anchor_image.get(key)
                if previous is None or float(
                    bank.descriptor_quality[descriptor_row]
                ) > float(bank.descriptor_quality[previous]):
                    self.best_descriptor_by_anchor_image[key] = descriptor_row
            for image_id in {
                str(bank.support_image_ids[row]) for row in range(start, end)
            }:
                self.visible_anchors_by_maplet_image.setdefault(
                    (maplet_id, image_id), []
                ).append(anchor_id)
        self.observation_xy: dict[tuple[int, str], np.ndarray] = {}
        for anchor_row, anchor_id_value in enumerate(anchors.anchor_ids.tolist()):
            anchor_id = int(anchor_id_value)
            start = int(anchors.observation_offsets[anchor_row])
            end = int(anchors.observation_offsets[anchor_row + 1])
            best: dict[str, int] = {}
            for row in range(start, end):
                image_id = str(anchors.observation_image_ids[row])
                previous = best.get(image_id)
                if previous is None or float(
                    anchors.observation_weights[row]
                ) > float(anchors.observation_weights[previous]):
                    best[image_id] = row
            for image_id, row in best.items():
                self.observation_xy[(anchor_id, image_id)] = (
                    anchors.observation_xy[row].astype(np.float32)
                )
        self.keys = tuple(
            sorted(
                key
                for key, anchor_ids in self.visible_anchors_by_maplet_image.items()
                if len(set(anchor_ids)) >= self.minimum_positive_nodes
                and key[1] in camera_by_image
            )
        )
        if not self.keys:
            raise ValueError("no surface-anchor training episodes were found")
        deployable_maplet_ids = sorted({key[0] for key in self.keys})
        maplet_row = {
            int(maplet_id): int(row)
            for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
        }
        rows = np.asarray(
            [maplet_row[maplet_id] for maplet_id in deployable_maplet_ids],
            dtype=np.int64,
        )
        similarity = maplets.descriptors[rows] @ maplets.descriptors[rows].T
        np.fill_diagonal(similarity, -np.inf)
        self.hard_negative_maplet: dict[int, int] = {}
        for output_row, maplet_id in enumerate(deployable_maplet_ids):
            order = np.argsort(-similarity[output_row], kind="mergesort")
            for candidate_row in order.tolist():
                candidate_id = int(deployable_maplet_ids[candidate_row])
                if any(key[0] == candidate_id for key in self.keys):
                    self.hard_negative_maplet[int(maplet_id)] = candidate_id
                    break

    def _query_frame(
        self,
        *,
        source_maplet_id: int,
        image_id: str,
        add_null_from_maplet_id: int | None,
    ) -> tuple[LocalFeatureFrame, np.ndarray]:
        positive_ids = sorted(
            {
                int(anchor_id)
                for anchor_id in self.visible_anchors_by_maplet_image.get(
                    (int(source_maplet_id), str(image_id)), ()
                )
                if (int(anchor_id), str(image_id))
                in self.best_descriptor_by_anchor_image
                and (int(anchor_id), str(image_id)) in self.observation_xy
            },
            key=lambda anchor_id: (
                -float(
                    self.bank.descriptor_quality[
                        self.best_descriptor_by_anchor_image[
                            (int(anchor_id), str(image_id))
                        ]
                    ]
                ),
                int(anchor_id),
            ),
        )[: self.maximum_positive_nodes]
        descriptor_rows = [
            self.best_descriptor_by_anchor_image[(int(anchor_id), str(image_id))]
            for anchor_id in positive_ids
        ]
        xy = [self.observation_xy[(int(anchor_id), str(image_id))] for anchor_id in positive_ids]
        target_ids = list(positive_ids)
        if add_null_from_maplet_id is not None and self.maximum_null_nodes > 0:
            null_ids = [
                int(anchor_id)
                for anchor_id in sorted(
                    set(
                        self.visible_anchors_by_maplet_image.get(
                            (int(add_null_from_maplet_id), str(image_id)), ()
                        )
                    )
                )
                if (int(anchor_id), str(image_id))
                in self.best_descriptor_by_anchor_image
                and (int(anchor_id), str(image_id)) in self.observation_xy
            ][: self.maximum_null_nodes]
            descriptor_rows.extend(
                self.best_descriptor_by_anchor_image[
                    (int(anchor_id), str(image_id))
                ]
                for anchor_id in null_ids
            )
            xy.extend(
                self.observation_xy[(int(anchor_id), str(image_id))]
                for anchor_id in null_ids
            )
            target_ids.extend([-1] * len(null_ids))
        rows = np.asarray(descriptor_rows, dtype=np.int64)
        return (
            LocalFeatureFrame(
                image_id=str(image_id),
                keypoints_xy=np.asarray(xy, dtype=np.float32),
                descriptors=self.bank.descriptors[rows],
                scores=np.maximum(self.bank.descriptor_quality[rows], 0.0),
            ),
            np.asarray(target_ids, dtype=np.int64),
        )

    def episode(
        self,
        key: tuple[int, str],
        *,
        negative: bool,
    ):
        source_maplet_id, image_id = int(key[0]), str(key[1])
        negative_maplet_id = self.hard_negative_maplet.get(source_maplet_id)
        candidate_maplet_id = (
            int(negative_maplet_id)
            if negative and negative_maplet_id is not None
            else source_maplet_id
        )
        query, targets = self._query_frame(
            source_maplet_id=source_maplet_id,
            image_id=image_id,
            add_null_from_maplet_id=(
                negative_maplet_id
                if not negative and negative_maplet_id is not None
                else None
            ),
        )
        if negative:
            targets[:] = -1
        camera = self.camera_by_image[image_id]
        excluded = set(self.validation_images)
        excluded.add(image_id)
        return build_surface_anchor_episode(
            query=query,
            query_rows=np.arange(len(query.keypoints_xy), dtype=np.int64),
            query_image_size=(int(camera.width), int(camera.height)),
            maplet_id=int(candidate_maplet_id),
            maplet_probabilities=np.full(
                (len(query.keypoints_xy),), 0.85, dtype=np.float32
            ),
            region_distances=np.zeros(
                (len(query.keypoints_xy),), dtype=np.float32
            ),
            maplets=self.maplets,
            anchors=self.anchors,
            descriptor_bank=self.bank,
            config=self.config,
            target_anchor_ids=targets,
            excluded_support_image_ids=tuple(sorted(excluded)),
        )[0]


class DeploymentReplaySurfaceAnchorTrainingCorpus:
    """Image-disjoint matcher episodes from the exact online input distribution."""

    deployment_replay = True

    def __init__(
        self,
        *,
        replay_dir: Path,
        maplets: VfmSurfaceMapletBank,
        anchors: StableSurfaceAnchorMap,
        bank: AnchorLocalDescriptorBank,
        camera_by_image,
        config: SurfaceAnchorSetMatcherConfig,
        validation_images: set[str],
    ) -> None:
        self.maplets = maplets
        self.anchors = anchors
        self.bank = bank
        self.feature_kind = anchor_feature_kind(bank.metadata)
        self.camera_by_image = camera_by_image
        self.config = config
        self.validation_images = set(validation_images)
        self.path_by_image: dict[str, Path] = {}
        self.keys: list[tuple[int, str]] = []
        deployable_maplets = set(int(value) for value in maplets.maplet_ids.tolist())
        for path in sorted(Path(replay_dir).glob("*.npz")):
            with np.load(path) as data:
                image_id = str(np.asarray(data["image_id"]).item())
                candidate_ids = np.asarray(
                    data["candidate_maplet_ids"], dtype=np.int64
                )
                candidate_probabilities = np.asarray(
                    data["candidate_probabilities"], dtype=np.float32
                )
                target_owner = np.asarray(
                    data["target_owner_maplet_ids"], dtype=np.int64
                )
            if image_id not in camera_by_image or candidate_ids.ndim != 2:
                continue
            self.path_by_image[image_id] = path
            mass_by_maplet: dict[int, float] = {}
            for maplet_id in np.unique(candidate_ids).tolist():
                maplet_id = int(maplet_id)
                if maplet_id < 0 or maplet_id not in deployable_maplets:
                    continue
                mass_by_maplet[maplet_id] = float(
                    np.sum(
                        np.where(
                            candidate_ids == maplet_id,
                            candidate_probabilities,
                            0.0,
                        )
                    )
                )
            scene_maplets = [
                item[0]
                for item in sorted(
                    mass_by_maplet.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: int(config.maximum_scene_maplets)]
            ]
            positive_maplets = [
                maplet_id
                for maplet_id in scene_maplets
                if bool(np.any(target_owner == int(maplet_id)))
            ]
            negative_maplets = [
                maplet_id
                for maplet_id in scene_maplets
                if maplet_id not in set(positive_maplets)
            ][: max(2, len(positive_maplets))]
            self.keys.extend(
                (int(maplet_id), image_id)
                for maplet_id in positive_maplets + negative_maplets
            )
        self.keys = sorted(set(self.keys))
        if not self.keys:
            raise ValueError("deployment replay contains no matcher episodes")

    def episode(self, key: tuple[int, str], *, negative: bool):
        del negative
        maplet_id, image_id = int(key[0]), str(key[1])
        path = self.path_by_image[image_id]
        with np.load(path) as data:
            xy = np.asarray(data["xy"], dtype=np.float32)
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            vfm_descriptors = (
                np.asarray(data["vfm_descriptors"], dtype=np.float32)
                if "vfm_descriptors" in data
                else None
            )
            scores = np.asarray(data["scores"], dtype=np.float32)
            target_anchor_ids = np.asarray(
                data["target_anchor_ids"], dtype=np.int64
            )
            target_owner = np.asarray(
                data["target_owner_maplet_ids"], dtype=np.int64
            )
            candidate_ids = np.asarray(
                data["candidate_maplet_ids"], dtype=np.int64
            )
            candidate_probabilities = np.asarray(
                data["candidate_probabilities"], dtype=np.float32
            )
        descriptors = compose_anchor_query_descriptors(
            alike_descriptors=descriptors,
            radio_final_descriptors=vfm_descriptors,
            feature_kind=self.feature_kind,
            expected_dim=int(self.bank.feature_dim),
        )
        matches = candidate_ids == int(maplet_id)
        rows, columns = np.nonzero(matches)
        if len(rows) < 4:
            raise ValueError("deployment replay maplet has too few query nodes")
        unique_rows, first = np.unique(rows, return_index=True)
        columns = columns[first]
        probabilities = candidate_probabilities[unique_rows, columns]
        targets = np.where(
            target_owner[unique_rows] == int(maplet_id),
            target_anchor_ids[unique_rows],
            -1,
        ).astype(np.int64)
        query = LocalFeatureFrame(
            image_id=image_id,
            keypoints_xy=xy,
            descriptors=descriptors,
            scores=np.maximum(scores, 0.0),
        )
        camera = self.camera_by_image[image_id]
        excluded = set(self.validation_images)
        excluded.add(image_id)
        return build_surface_anchor_episode(
            query=query,
            query_rows=unique_rows.astype(np.int64),
            query_image_size=(int(camera.width), int(camera.height)),
            maplet_id=int(maplet_id),
            maplet_probabilities=probabilities.astype(np.float32),
            region_distances=(1.0 - probabilities).astype(np.float32),
            maplets=self.maplets,
            anchors=self.anchors,
            descriptor_bank=self.bank,
            config=self.config,
            target_anchor_ids=targets,
            excluded_support_image_ids=tuple(sorted(excluded)),
        )[0]


@torch.no_grad()
def _evaluate(
    model: SurfaceAnchorSetMatcher,
    corpus: SurfaceAnchorTrainingCorpus,
    keys: Sequence[tuple[int, str]],
    *,
    device: torch.device,
    maximum_episodes: int,
    pair_loss_weight: float,
    no_match_loss_weight: float,
    positive_assignment_weight: float,
    balance_no_match_classes: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    positive_ranks: list[int] = []
    cosine_positive_ranks: list[int] = []
    dustbin_targets: list[bool] = []
    dustbin_scores: list[float] = []
    evaluated = 0
    for key in list(keys)[: int(maximum_episodes)]:
        negative_modes = (False,) if corpus.deployment_replay else (False, True)
        for negative in negative_modes:
            try:
                episode = corpus.episode(key, negative=negative).to(device)
            except ValueError:
                continue
            output = model(episode)
            loss, _metrics = local_assignment_loss(
                output,
                episode,
                pair_loss_weight=float(pair_loss_weight),
                no_match_loss_weight=float(no_match_loss_weight),
                positive_assignment_weight=float(
                    positive_assignment_weight
                ),
                balance_no_match_classes=bool(
                    balance_no_match_classes
                ),
            )
            losses.append(float(loss.cpu().item()))
            probabilities = torch.exp(output["query_log_probabilities"]).cpu().numpy()
            targets = episode.target_track_indices.cpu().numpy()
            anchor_count = int(episode.track_features.shape[0])
            cosine_scores = (
                episode.edge_features[:, 0]
                .reshape(len(targets), anchor_count)
                .cpu()
                .numpy()
            )
            for row, target in enumerate(targets.tolist()):
                is_dustbin = int(target) == anchor_count
                dustbin_targets.append(is_dustbin)
                dustbin_scores.append(float(probabilities[row, anchor_count]))
                if not is_dustbin:
                    order = np.argsort(-probabilities[row, :anchor_count], kind="mergesort")
                    positive_ranks.append(
                        int(np.flatnonzero(order == int(target))[0]) + 1
                    )
                    cosine_order = np.argsort(
                        -cosine_scores[row], kind="mergesort"
                    )
                    cosine_positive_ranks.append(
                        int(
                            np.flatnonzero(
                                cosine_order == int(target)
                            )[0]
                        )
                        + 1
                    )
            evaluated += 1
    ranks = np.asarray(positive_ranks, dtype=np.int64)
    cosine_ranks = np.asarray(cosine_positive_ranks, dtype=np.int64)
    dustbin_target = np.asarray(dustbin_targets, dtype=bool)
    dustbin_score = np.asarray(dustbin_scores, dtype=np.float64)
    return {
        "episode_count": float(evaluated),
        "positive_query_count": float(len(ranks)),
        "mean_loss": float(np.mean(losses)) if losses else float("inf"),
        "positive_recall_at_1": float(np.mean(ranks <= 1)) if len(ranks) else 0.0,
        "positive_recall_at_3": float(np.mean(ranks <= 3)) if len(ranks) else 0.0,
        "positive_recall_at_5": float(np.mean(ranks <= 5)) if len(ranks) else 0.0,
        "cosine_positive_recall_at_1": (
            float(np.mean(cosine_ranks <= 1)) if len(cosine_ranks) else 0.0
        ),
        "cosine_positive_recall_at_3": (
            float(np.mean(cosine_ranks <= 3)) if len(cosine_ranks) else 0.0
        ),
        "cosine_positive_recall_at_5": (
            float(np.mean(cosine_ranks <= 5)) if len(cosine_ranks) else 0.0
        ),
        "dustbin_auprc": _average_precision(dustbin_target, dustbin_score),
        "dustbin_recall_at_0p5": (
            float(np.mean(dustbin_score[dustbin_target] >= 0.5))
            if np.any(dustbin_target)
            else 0.0
        ),
        "non_dustbin_specificity_at_0p5": (
            float(np.mean(dustbin_score[~dustbin_target] < 0.5))
            if np.any(~dustbin_target)
            else 0.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if (
        int(args.epochs) <= 0
        or int(args.steps_per_epoch) <= 0
        or int(args.validation_episodes) <= 0
        or int(args.gradient_accumulation) <= 0
    ):
        raise ValueError("training loop counts must be positive")
    if not 0.0 <= float(args.negative_episode_fraction) <= 1.0:
        raise ValueError("negative episode fraction must be in [0, 1]")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    bank = AnchorLocalDescriptorBank.load_npz(Path(args.local_descriptor_bank))
    camera_by_image = _load_camera_by_image(str(args.camera_model_dir))
    image_ids = sorted(set(bank.support_image_ids))
    validation_images = {
        image_id
        for image_id in image_ids
        if _split_value(image_id, int(args.seed)) == 0
    }
    training_images = set(image_ids) - validation_images
    if not training_images or not validation_images:
        raise ValueError("image-disjoint split is empty")
    config = SurfaceAnchorSetMatcherConfig(
        descriptor_dim=int(bank.feature_dim),
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        query_layers=int(args.layers),
        anchor_layers=int(args.layers),
        dropout=float(args.dropout),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
        maximum_query_nodes=int(args.maximum_positive_nodes)
        + int(args.maximum_null_nodes),
    )
    if str(args.deployment_replay_dir):
        corpus = DeploymentReplaySurfaceAnchorTrainingCorpus(
            replay_dir=Path(args.deployment_replay_dir),
            maplets=maplets,
            anchors=anchors,
            bank=bank,
            camera_by_image=camera_by_image,
            config=config,
            validation_images=validation_images,
        )
    else:
        corpus = SurfaceAnchorTrainingCorpus(
            maplets=maplets,
            anchors=anchors,
            bank=bank,
            camera_by_image=camera_by_image,
            config=config,
            minimum_positive_nodes=int(args.minimum_positive_nodes),
            maximum_positive_nodes=int(args.maximum_positive_nodes),
            maximum_null_nodes=int(args.maximum_null_nodes),
            validation_images=validation_images,
        )
    train_keys = [key for key in corpus.keys if key[1] in training_images]
    validation_keys = [key for key in corpus.keys if key[1] in validation_images]
    if not train_keys or not validation_keys:
        raise ValueError("no train/validation surface-anchor episodes remain")
    device = torch.device(
        str(args.device)
        if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )
    model = SurfaceAnchorSetMatcher(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    rng = np.random.default_rng(int(args.seed))
    output_checkpoint = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_key = None
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        completed = 0
        attempts = 0
        while completed < int(args.steps_per_epoch):
            attempts += 1
            if attempts > int(args.steps_per_epoch) * 10:
                raise RuntimeError("too many invalid sampled training episodes")
            key = train_keys[int(rng.integers(0, len(train_keys)))]
            negative = bool(
                not corpus.deployment_replay
                and rng.random() < float(args.negative_episode_fraction)
            )
            try:
                episode = corpus.episode(key, negative=negative).to(device)
            except ValueError:
                continue
            output = model(episode)
            loss, _metrics = local_assignment_loss(
                output,
                episode,
                pair_loss_weight=float(args.pair_loss_weight),
                no_match_loss_weight=float(args.no_match_loss_weight),
                positive_assignment_weight=float(
                    args.positive_assignment_weight
                ),
                balance_no_match_classes=not bool(
                    args.disable_balanced_no_match_classes
                ),
            )
            (loss / int(args.gradient_accumulation)).backward()
            losses.append(float(loss.detach().cpu().item()))
            completed += 1
            if (
                completed % int(args.gradient_accumulation) == 0
                or completed == int(args.steps_per_epoch)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.gradient_clip_norm)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = _evaluate(
            model,
            corpus,
            validation_keys,
            device=device,
            maximum_episodes=int(args.validation_episodes),
            pair_loss_weight=float(args.pair_loss_weight),
            no_match_loss_weight=float(args.no_match_loss_weight),
            positive_assignment_weight=float(
                args.positive_assignment_weight
            ),
            balance_no_match_classes=not bool(
                args.disable_balanced_no_match_classes
            ),
        )
        key = (
            float(validation["positive_recall_at_1"]),
            float(validation["dustbin_auprc"]),
            -float(validation["mean_loss"]),
        )
        record = {
            "epoch": int(epoch + 1),
            "train_mean_loss": float(np.mean(losses)),
            "validation": validation,
            "selection_key": list(key),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = int(epoch + 1)
            stale_epochs = 0
            save_surface_anchor_set_matcher(
                output_checkpoint,
                model,
                metadata={
                    "best_epoch": best_epoch,
                    "best_validation": validation,
                    "selection_key": list(key),
                    "split": {
                        "strategy": (
                            "sha256_image_disjoint_mod5_deployment_replay_v2"
                            if corpus.deployment_replay
                            else "sha256_image_disjoint_mod5_v1"
                        ),
                        "seed": int(args.seed),
                        "training_image_count": len(training_images),
                        "validation_image_count": len(validation_images),
                        "training_episode_count": len(train_keys),
                        "validation_episode_count": len(validation_keys),
                        "deployment_replay": bool(corpus.deployment_replay),
                    },
                    "artifacts": {
                        "maplets": {
                            "path": str(args.maplets),
                            "sha256": file_sha256_short(Path(args.maplets)),
                        },
                        "anchors": {
                            "path": str(args.anchors),
                            "sha256": file_sha256_short(Path(args.anchors)),
                        },
                        "local_descriptor_bank": {
                            "path": str(args.local_descriptor_bank),
                            "sha256": file_sha256_short(
                                Path(args.local_descriptor_bank)
                            ),
                        },
                    },
                },
            )
        else:
            stale_epochs += 1
        if int(args.patience) > 0 and stale_epochs >= int(args.patience):
            break
    _best_model, payload = load_surface_anchor_set_matcher(
        output_checkpoint, device=device
    )
    summary = {
        "stage": "train_surface_anchor_set_matcher",
        "format": payload["format"],
        "best_epoch": int(best_epoch),
        "best_selection_key": list(best_key) if best_key is not None else None,
        "best_validation": payload["metadata"]["best_validation"],
        "config": config.to_dict(),
        "split": payload["metadata"]["split"],
        "history": history,
        "production_contract": {
            "map_representation": "2dgs_surface_maplets_and_stable_anchors",
            "identity": "stable_surface_anchor_id",
            "anchor_identity_feature": anchor_feature_kind(bank.metadata),
            "alike_descriptor_used_for_identity": (
                anchor_feature_kind(bank.metadata)
                != "radio_final_at_alike_detection"
            ),
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
        "output_checkpoint": str(output_checkpoint),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
