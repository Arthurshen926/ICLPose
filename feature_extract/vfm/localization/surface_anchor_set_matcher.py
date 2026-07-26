"""Feature-only set matching from query ALIKE points to stable 2DGS anchors.

The deployment contract is deliberately strict: mapping RGB is used only while
building the descriptor bank.  This module consumes query features plus the
frozen maplet/anchor/descriptor artifacts and never accepts a mapping image
path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from feature_extract.vfm.localization.local_assignment_matcher import (
    LocalAssignmentEpisode,
    LocalAssignmentMatcher,
    LocalAssignmentMatcherConfig,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
    LocalFeatureFrame,
    SurfaceAnchorCandidatePool,
    SurfaceMapletMatchResult,
)
from feature_extract.vfm.surface_maplet_bank import (
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
)


SURFACE_ANCHOR_SET_MATCHER_FORMAT = "vfm_2dgs_surface_anchor_set_matcher_v1"


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("descriptor values must have shape (N, C)")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), eps)


@dataclass(frozen=True)
class SurfaceAnchorSetMatcherConfig:
    descriptor_dim: int = 64
    model_dim: int = 128
    num_heads: int = 4
    query_layers: int = 2
    anchor_layers: int = 2
    dropout: float = 0.1
    sinkhorn_iterations: int = 20
    edge_prior_scale: float = 12.0
    edge_prior_center: float = 0.65
    query_dustbin_initial_bias: float = -0.25
    maximum_support_descriptors: int = 8
    maximum_query_nodes: int = 96
    maximum_anchors: int = 64
    maximum_maplets_per_region: int = 3
    maximum_region_neighbors: int = 4
    maximum_maplets_per_query: int = 8
    maximum_scene_maplets: int = 12

    def __post_init__(self) -> None:
        for name in (
            "descriptor_dim",
            "model_dim",
            "num_heads",
            "query_layers",
            "anchor_layers",
            "sinkhorn_iterations",
            "maximum_support_descriptors",
            "maximum_query_nodes",
            "maximum_anchors",
            "maximum_maplets_per_region",
            "maximum_region_neighbors",
            "maximum_maplets_per_query",
            "maximum_scene_maplets",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.model_dim) % int(self.num_heads) != 0:
            raise ValueError("num_heads must divide model_dim")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def query_input_dim(self) -> int:
        return int(self.descriptor_dim) + 3

    @property
    def anchor_input_dim(self) -> int:
        return int(self.descriptor_dim) + 10

    @property
    def support_input_dim(self) -> int:
        return int(self.descriptor_dim) + 1

    @property
    def edge_input_dim(self) -> int:
        return 9

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceAnchorEpisodeIndex:
    maplet_id: int
    query_rows: np.ndarray
    anchor_ids: np.ndarray
    maplet_probabilities: np.ndarray

    def __post_init__(self) -> None:
        query_rows = np.asarray(self.query_rows, dtype=np.int64).reshape(-1)
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        probabilities = np.asarray(
            self.maplet_probabilities, dtype=np.float32
        ).reshape(-1)
        if probabilities.shape != query_rows.shape:
            raise ValueError("one maplet probability is required per query row")
        object.__setattr__(self, "query_rows", query_rows)
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "maplet_probabilities", probabilities)


class SurfaceAnchorSetMatcher(nn.Module):
    """Anchor-semantic wrapper around the shared partial-assignment backend."""

    def __init__(self, config: SurfaceAnchorSetMatcherConfig) -> None:
        super().__init__()
        self.config = config
        self.assignment = LocalAssignmentMatcher(
            LocalAssignmentMatcherConfig(
                query_input_dim=config.query_input_dim,
                track_input_dim=config.anchor_input_dim,
                support_input_dim=config.support_input_dim,
                edge_input_dim=config.edge_input_dim,
                model_dim=int(config.model_dim),
                num_heads=int(config.num_heads),
                query_layers=int(config.query_layers),
                track_layers=int(config.anchor_layers),
                dropout=float(config.dropout),
                sinkhorn_iterations=int(config.sinkhorn_iterations),
                edge_prior_feature_index=0,
                edge_prior_scale=float(config.edge_prior_scale),
                edge_prior_center=float(config.edge_prior_center),
                query_dustbin_initial_bias=float(
                    config.query_dustbin_initial_bias
                ),
            )
        )

    def forward(
        self, episode: LocalAssignmentEpisode
    ) -> dict[str, torch.Tensor]:
        return self.assignment(episode)


def _maplet_row(
    maplets: VfmSurfaceMapletBank, maplet_id: int
) -> int:
    rows = np.flatnonzero(maplets.maplet_ids == int(maplet_id))
    if len(rows) != 1:
        raise KeyError(f"surface maplet ID is absent or duplicated: {maplet_id}")
    return int(rows[0])


def _anchor_descriptor_rows(
    descriptor_bank: AnchorLocalDescriptorBank,
) -> dict[int, np.ndarray]:
    return {
        int(anchor_id): np.arange(
            int(descriptor_bank.descriptor_offsets[row]),
            int(descriptor_bank.descriptor_offsets[row + 1]),
            dtype=np.int64,
        )
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }


def _episode_anchor_ids(
    maplets: VfmSurfaceMapletBank,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    maplet_id: int,
    maximum_anchors: int,
    feature_preferred_base_fill_target: int | None = None,
) -> np.ndarray:
    maplet_row = _maplet_row(maplets, int(maplet_id))
    start = int(maplets.anchor_offsets[maplet_row])
    end = int(maplets.anchor_offsets[maplet_row + 1])
    bank_ids = set(int(value) for value in descriptor_bank.anchor_ids.tolist())
    anchor_rows = anchors.row_by_id()
    candidates: list[tuple[float, int]] = []
    for anchor_id_value in maplets.anchor_ids[start:end].tolist():
        anchor_id = int(anchor_id_value)
        anchor_row = anchor_rows.get(anchor_id)
        if (
            anchor_row is None
            or anchor_id not in bank_ids
            or int(anchors.owner_maplet_ids[anchor_row]) != int(maplet_id)
        ):
            continue
        candidates.append((float(anchors.quality_scores[anchor_row]), anchor_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if feature_preferred_base_fill_target is not None:
        feature = [
            item for item in candidates if int(item[1]) >= 1_000_000_000
        ][: int(maximum_anchors)]
        geometry_first = [
            item for item in candidates if int(item[1]) < 1_000_000_000
        ]
        fill_target = min(
            int(maximum_anchors),
            max(
                len(feature),
                int(feature_preferred_base_fill_target),
            ),
        )
        candidates = [
            *feature,
            *geometry_first[: max(fill_target - len(feature), 0)],
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
    return np.asarray(
        [item[1] for item in candidates[: int(maximum_anchors)]],
        dtype=np.int64,
    )


def build_surface_anchor_episode(
    *,
    query: LocalFeatureFrame,
    query_rows: np.ndarray,
    query_image_size: tuple[int, int],
    maplet_id: int,
    maplet_probabilities: np.ndarray,
    region_distances: np.ndarray,
    maplets: VfmSurfaceMapletBank,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    config: SurfaceAnchorSetMatcherConfig,
    target_anchor_ids: np.ndarray | None = None,
    excluded_support_image_ids: Sequence[str] = (),
    feature_preferred_base_fill_target: int | None = None,
) -> tuple[LocalAssignmentEpisode, SurfaceAnchorEpisodeIndex]:
    """Build one maplet-conditioned set episode without reading mapping RGB."""

    rows = np.asarray(query_rows, dtype=np.int64).reshape(-1)
    if len(rows) == 0:
        raise ValueError("surface-anchor episode requires query nodes")
    if np.any(rows < 0) or np.any(rows >= len(query.keypoints_xy)):
        raise ValueError("query row is outside the LocalFeatureFrame")
    probabilities = np.asarray(maplet_probabilities, dtype=np.float32).reshape(-1)
    distances = np.asarray(region_distances, dtype=np.float32).reshape(-1)
    if probabilities.shape != rows.shape or distances.shape != rows.shape:
        raise ValueError("maplet probability/distance must align with query rows")
    if len(rows) > int(config.maximum_query_nodes):
        ranking = np.lexsort(
            (
                rows,
                -query.scores[rows],
                -probabilities,
            )
        )[: int(config.maximum_query_nodes)]
        rows = rows[ranking]
        probabilities = probabilities[ranking]
        distances = distances[ranking]
        if target_anchor_ids is not None:
            target_anchor_ids = np.asarray(
                target_anchor_ids, dtype=np.int64
            ).reshape(-1)[ranking]

    anchor_ids = _episode_anchor_ids(
        maplets,
        anchors,
        descriptor_bank,
        int(maplet_id),
        int(config.maximum_anchors),
        feature_preferred_base_fill_target,
    )
    if len(anchor_ids) < 4:
        raise ValueError("maplet has fewer than four deployable stable anchors")
    width, height = int(query_image_size[0]), int(query_image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("query image dimensions must be positive")
    maplet_row = _maplet_row(maplets, int(maplet_id))
    anchor_row_by_id = anchors.row_by_id()
    descriptor_rows_by_id = _anchor_descriptor_rows(descriptor_bank)
    excluded = set(str(value) for value in excluded_support_image_ids)
    if excluded:
        anchor_ids = np.asarray(
            [
                int(anchor_id)
                for anchor_id in anchor_ids.tolist()
                if any(
                    descriptor_bank.support_image_ids[int(row)] not in excluded
                    for row in descriptor_rows_by_id[int(anchor_id)].tolist()
                )
            ],
            dtype=np.int64,
        )
        if len(anchor_ids) < 4:
            raise ValueError(
                "maplet has fewer than four anchors outside excluded support views"
            )

    query_descriptors = _normalize_rows(query.descriptors[rows])
    query_xy = query.keypoints_xy[rows].astype(np.float32)
    normalized_xy = query_xy / np.asarray(
        [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
    )
    query_scores = np.log1p(np.maximum(query.scores[rows], 0.0)).reshape(-1, 1)
    query_features = np.concatenate(
        [query_descriptors, query_scores, 2.0 * normalized_xy - 1.0],
        axis=1,
    ).astype(np.float32)

    anchor_features: list[np.ndarray] = []
    support_rows_per_anchor: list[np.ndarray] = []
    extent = np.maximum(
        np.asarray(maplets.extents[maplet_row], dtype=np.float32), 1e-3
    )
    tangent = np.asarray(maplets.tangent_frames[maplet_row], dtype=np.float32)
    center = np.asarray(maplets.centers[maplet_row], dtype=np.float32)
    for anchor_id in anchor_ids.tolist():
        anchor_row = anchor_row_by_id[int(anchor_id)]
        descriptor_rows = descriptor_rows_by_id[int(anchor_id)]
        if excluded:
            descriptor_rows = descriptor_rows[
                np.asarray(
                    [
                        descriptor_bank.support_image_ids[int(row)] not in excluded
                        for row in descriptor_rows.tolist()
                    ],
                    dtype=bool,
                )
            ]
        order = np.argsort(
            -descriptor_bank.descriptor_quality[descriptor_rows],
            kind="mergesort",
        )[: int(config.maximum_support_descriptors)]
        descriptor_rows = descriptor_rows[order]
        support_rows_per_anchor.append(descriptor_rows)
        weights = np.maximum(
            descriptor_bank.descriptor_quality[descriptor_rows], 1e-6
        )
        prototype = np.sum(
            descriptor_bank.descriptors[descriptor_rows] * weights[:, None],
            axis=0,
        ) / float(np.sum(weights))
        prototype = _normalize_rows(prototype[None])[0]
        local_xyz = (
            (np.asarray(anchors.xyz[anchor_row], dtype=np.float32) - center)
            @ tangent.T
        ) / extent
        local_normal = (
            np.asarray(anchors.normals[anchor_row], dtype=np.float32) @ tangent.T
        )
        geometry = np.asarray(
            [
                *local_xyz.tolist(),
                *local_normal.tolist(),
                np.log1p(max(float(anchors.quality_scores[anchor_row]), 0.0)),
                np.log1p(max(float(anchors.support_radii[anchor_row]), 0.0)),
                float(anchors.opacity[anchor_row]),
                float(anchors.geometry_confidence[anchor_row]),
            ],
            dtype=np.float32,
        )
        anchor_features.append(np.concatenate([prototype, geometry]))
    anchor_feature_array = np.stack(anchor_features).astype(np.float32)

    edge_query_indices = np.repeat(
        np.arange(len(rows), dtype=np.int64), len(anchor_ids)
    )
    edge_anchor_indices = np.tile(
        np.arange(len(anchor_ids), dtype=np.int64), len(rows)
    )
    edge_count = len(edge_query_indices)
    support_features = np.zeros(
        (
            edge_count,
            int(config.maximum_support_descriptors),
            config.support_input_dim,
        ),
        dtype=np.float32,
    )
    support_mask = np.zeros(
        (edge_count, int(config.maximum_support_descriptors)), dtype=bool
    )
    edge_features = np.zeros((edge_count, config.edge_input_dim), dtype=np.float32)
    for edge_row, (query_row, anchor_column) in enumerate(
        zip(edge_query_indices.tolist(), edge_anchor_indices.tolist())
    ):
        descriptor_rows = support_rows_per_anchor[int(anchor_column)]
        descriptor_values = descriptor_bank.descriptors[descriptor_rows]
        descriptor_quality = descriptor_bank.descriptor_quality[descriptor_rows]
        count = len(descriptor_rows)
        support_features[edge_row, :count, : config.descriptor_dim] = (
            descriptor_values
        )
        support_features[edge_row, :count, config.descriptor_dim] = (
            descriptor_quality
        )
        support_mask[edge_row, :count] = True
        similarities = descriptor_values @ query_descriptors[int(query_row)]
        ordered = np.sort(similarities)[::-1]
        maximum = float(ordered[0])
        second = float(ordered[1]) if len(ordered) > 1 else maximum
        temperature = 0.08
        shifted = similarities / temperature
        log_mean_exp = float(
            temperature
            * (
                np.max(shifted)
                + np.log(np.mean(np.exp(shifted - np.max(shifted))))
            )
        )
        anchor_row = anchor_row_by_id[int(anchor_ids[int(anchor_column)])]
        edge_features[edge_row] = np.asarray(
            [
                maximum,
                log_mean_exp,
                float(np.mean(similarities)),
                float(np.std(similarities)),
                maximum - second,
                float(probabilities[int(query_row)]),
                float(distances[int(query_row)]),
                float(anchors.quality_scores[anchor_row]),
                float(count / max(int(config.maximum_support_descriptors), 1)),
            ],
            dtype=np.float32,
        )

    if target_anchor_ids is None:
        targets = np.full((len(rows),), len(anchor_ids), dtype=np.int64)
    else:
        requested_targets = np.asarray(target_anchor_ids, dtype=np.int64).reshape(-1)
        if requested_targets.shape != rows.shape:
            raise ValueError("target anchor IDs must align with retained query rows")
        column_by_id = {
            int(anchor_id): column
            for column, anchor_id in enumerate(anchor_ids.tolist())
        }
        targets = np.asarray(
            [
                column_by_id.get(int(anchor_id), len(anchor_ids))
                if int(anchor_id) >= 0
                else len(anchor_ids)
                for anchor_id in requested_targets.tolist()
            ],
            dtype=np.int64,
        )
    episode = LocalAssignmentEpisode(
        query_features=torch.from_numpy(query_features),
        track_features=torch.from_numpy(anchor_feature_array),
        edge_query_indices=torch.from_numpy(edge_query_indices),
        edge_track_indices=torch.from_numpy(edge_anchor_indices),
        edge_features=torch.from_numpy(edge_features),
        support_features=torch.from_numpy(support_features),
        support_mask=torch.from_numpy(support_mask),
        target_track_indices=torch.from_numpy(targets),
        query_rows=torch.from_numpy(rows),
        candidate_columns=torch.from_numpy(edge_anchor_indices),
    )
    episode.validate()
    return (
        episode,
        SurfaceAnchorEpisodeIndex(
            maplet_id=int(maplet_id),
            query_rows=rows,
            anchor_ids=anchor_ids,
            maplet_probabilities=probabilities,
        ),
    )


@torch.no_grad()
def match_query_to_surface_anchors(
    *,
    model: SurfaceAnchorSetMatcher,
    query: LocalFeatureFrame,
    query_image_size: tuple[int, int],
    query_region_xy: np.ndarray,
    query_region_grid_size: tuple[int, int],
    maplet_match: SurfaceMapletMatchResult,
    maplets: VfmSurfaceMapletBank,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    device: str | torch.device,
    top_l: int = 5,
    query_aligned_maplet_match: SurfaceMapletMatchResult | None = None,
    feature_preferred_base_fill_target: int | None = None,
) -> tuple[SurfaceAnchorCandidatePool, dict[str, object]]:
    """Run maplet-first feature-only partial assignment over query ALIKE sets."""

    if int(top_l) <= 0:
        raise ValueError("top_l must be positive")
    if len(query.keypoints_xy) == 0:
        return (
            SurfaceAnchorCandidatePool(
                query_xy=query.keypoints_xy,
                anchor_ids=np.zeros((0, int(top_l)), dtype=np.int64),
                xyz=np.zeros((0, int(top_l), 3), dtype=np.float64),
                descriptor_scores=np.zeros((0, int(top_l)), dtype=np.float32),
                candidate_probabilities=np.zeros((0, int(top_l)), dtype=np.float32),
                null_probabilities=np.zeros((0,), dtype=np.float32),
                valid_mask=np.zeros((0, int(top_l)), dtype=bool),
            ),
            {"episode_count": 0, "matched_query_count": 0},
        )
    config = model.config
    region_xy = np.asarray(query_region_xy, dtype=np.float32).reshape(-1, 2)
    grid_width, grid_height = (
        int(query_region_grid_size[0]),
        int(query_region_grid_size[1]),
    )
    width, height = int(query_image_size[0]), int(query_image_size[1])
    query_grid_xy = query.keypoints_xy * np.asarray(
        [grid_width / max(width, 1), grid_height / max(height, 1)],
        dtype=np.float32,
    )
    episode_inputs: dict[int, dict[str, list[float] | list[int]]] = {}
    if query_aligned_maplet_match is not None:
        aligned = query_aligned_maplet_match
        if aligned.candidate_maplet_ids.shape[0] != len(query.keypoints_xy):
            raise ValueError(
                "query_aligned_maplet_match must have one row per query point"
            )
        for query_row in range(len(query.keypoints_xy)):
            keep = min(
                int(config.maximum_maplets_per_query),
                aligned.candidate_maplet_ids.shape[1],
            )
            for column in range(keep):
                maplet_id = int(
                    aligned.candidate_maplet_ids[query_row, column]
                )
                probability = float(
                    aligned.candidate_probabilities[query_row, column]
                )
                if maplet_id < 0 or probability <= 0.0:
                    continue
                entry = episode_inputs.setdefault(
                    maplet_id,
                    {"rows": [], "probabilities": [], "distances": []},
                )
                entry["rows"].append(int(query_row))
                entry["probabilities"].append(probability)
                # Query-aligned retrieval has no spatial proxy distance.  Its
                # calibrated uncertainty is the appropriate deployment-time
                # analogue and is also replayed during matcher training.
                entry["distances"].append(float(1.0 - probability))
    else:
        if len(region_xy) == 0:
            raise ValueError("query_region_xy cannot be empty")
        region_neighbor_count = min(
            int(config.maximum_region_neighbors), len(region_xy)
        )
        neighbor_distance, neighbor_region = cKDTree(region_xy).query(
            query_grid_xy, k=region_neighbor_count
        )
        neighbor_distance = np.asarray(
            neighbor_distance, dtype=np.float64
        ).reshape(len(query.keypoints_xy), region_neighbor_count)
        neighbor_region = np.asarray(
            neighbor_region, dtype=np.int64
        ).reshape(len(query.keypoints_xy), region_neighbor_count)
        region_scale = np.asarray(
            [max(grid_width - 1, 1), max(grid_height - 1, 1)],
            dtype=np.float32,
        )
        for query_row in range(len(query.keypoints_xy)):
            best_by_maplet: dict[int, tuple[float, float]] = {}
            neighbor_null: list[float] = []
            for neighbor_column in range(region_neighbor_count):
                region_row = int(
                    neighbor_region[query_row, neighbor_column]
                )
                distance = float(
                    np.linalg.norm(
                        (query_grid_xy[query_row] - region_xy[region_row])
                        / region_scale
                    )
                )
                spatial_weight = float(
                    np.exp(-0.5 * (distance / 0.10) ** 2)
                )
                neighbor_null.append(
                    spatial_weight
                    * float(maplet_match.null_probabilities[region_row])
                )
                for column in range(
                    min(
                        int(config.maximum_maplets_per_region),
                        maplet_match.candidate_maplet_ids.shape[1],
                    )
                ):
                    maplet_id = int(
                        maplet_match.candidate_maplet_ids[region_row, column]
                    )
                    score = (
                        spatial_weight
                        * float(
                            maplet_match.candidate_probabilities[
                                region_row, column
                            ]
                        )
                    )
                    if maplet_id < 0 or score <= 0.0:
                        continue
                    previous = best_by_maplet.get(maplet_id)
                    if previous is None or score > previous[0]:
                        best_by_maplet[maplet_id] = (score, distance)
            ranked = sorted(
                best_by_maplet.items(),
                key=lambda item: (-item[1][0], item[0]),
            )[: int(config.maximum_maplets_per_query)]
            raw_mass = sum(item[1][0] for item in ranked)
            null_mass = max(neighbor_null, default=0.0)
            normalizer = max(raw_mass + null_mass, 1e-12)
            for maplet_id, (score, distance) in ranked:
                probability = float(score / normalizer)
                entry = episode_inputs.setdefault(
                    maplet_id,
                    {"rows": [], "probabilities": [], "distances": []},
                )
                entry["rows"].append(int(query_row))
                entry["probabilities"].append(float(probability))
                entry["distances"].append(distance)

    raw_episode_count = len(episode_inputs)
    if len(episode_inputs) > int(config.maximum_scene_maplets):
        ranked_maplets = sorted(
            episode_inputs,
            key=lambda maplet_id: (
                -sum(
                    float(value)
                    for value in episode_inputs[maplet_id]["probabilities"]
                ),
                int(maplet_id),
            ),
        )[: int(config.maximum_scene_maplets)]
        episode_inputs = {
            int(maplet_id): episode_inputs[int(maplet_id)]
            for maplet_id in ranked_maplets
        }

    candidate_records: list[list[tuple[float, int, float]]] = [
        [] for _ in range(len(query.keypoints_xy))
    ]
    used_maplet_mass = np.zeros((len(query.keypoints_xy),), dtype=np.float64)
    conditional_null_mass = np.zeros_like(used_maplet_mass)
    episode_count = 0
    model.eval()
    for maplet_id in sorted(episode_inputs):
        values = episode_inputs[maplet_id]
        try:
            episode, index = build_surface_anchor_episode(
                query=query,
                query_rows=np.asarray(values["rows"], dtype=np.int64),
                query_image_size=query_image_size,
                maplet_id=int(maplet_id),
                maplet_probabilities=np.asarray(
                    values["probabilities"], dtype=np.float32
                ),
                region_distances=np.asarray(values["distances"], dtype=np.float32),
                maplets=maplets,
                anchors=anchors,
                descriptor_bank=descriptor_bank,
                config=config,
                feature_preferred_base_fill_target=(
                    feature_preferred_base_fill_target
                ),
            )
        except ValueError:
            continue
        output = model(episode.to(device))
        probabilities = torch.exp(output["query_log_probabilities"]).cpu().numpy()
        anchor_count = len(index.anchor_ids)
        for local_row, global_row in enumerate(index.query_rows.tolist()):
            maplet_probability = float(index.maplet_probabilities[local_row])
            used_maplet_mass[global_row] += maplet_probability
            conditional_null_mass[global_row] += (
                maplet_probability * float(probabilities[local_row, anchor_count])
            )
            for anchor_column, anchor_id in enumerate(index.anchor_ids.tolist()):
                joint = maplet_probability * float(
                    probabilities[local_row, anchor_column]
                )
                if joint <= 0.0:
                    continue
                candidate_records[global_row].append(
                    (
                        joint,
                        int(anchor_id),
                        float(
                            episode.edge_features[
                                local_row * anchor_count + anchor_column, 0
                            ].item()
                        ),
                    )
                )
        episode_count += 1

    anchor_row_by_id = anchors.row_by_id()
    output_ids = np.full(
        (len(query.keypoints_xy), int(top_l)), -1, dtype=np.int64
    )
    output_xyz = np.zeros(
        (len(query.keypoints_xy), int(top_l), 3), dtype=np.float64
    )
    output_scores = np.full(
        (len(query.keypoints_xy), int(top_l)), -np.inf, dtype=np.float32
    )
    output_probabilities = np.zeros(
        (len(query.keypoints_xy), int(top_l)), dtype=np.float32
    )
    valid = np.zeros_like(output_probabilities, dtype=bool)
    null = np.ones((len(query.keypoints_xy),), dtype=np.float32)
    conditional_null = np.ones(
        (len(query.keypoints_xy),), dtype=np.float32
    )
    conditional_retained_anchor = np.zeros(
        (len(query.keypoints_xy),), dtype=np.float32
    )
    for query_row, records in enumerate(candidate_records):
        by_anchor: dict[int, tuple[float, float]] = {}
        for probability, anchor_id, descriptor_score in records:
            previous = by_anchor.get(anchor_id)
            if previous is None:
                by_anchor[anchor_id] = (probability, descriptor_score)
            else:
                by_anchor[anchor_id] = (
                    previous[0] + probability,
                    max(previous[1], descriptor_score),
                )
        all_ranked = sorted(
            by_anchor.items(), key=lambda item: (-item[1][0], item[0])
        )
        ranked = all_ranked[: int(top_l)]
        null_mass = max(0.0, 1.0 - float(used_maplet_mass[query_row]))
        null_mass += float(conditional_null_mass[query_row])
        total_anchor_mass = sum(item[1][0] for item in all_ranked)
        candidate_mass = sum(item[1][0] for item in ranked)
        omitted_anchor_mass = max(0.0, total_anchor_mass - candidate_mass)
        null_mass += omitted_anchor_mass
        # Keep a second, conditional confidence for deciding whether enough
        # query groups can seed PnP.  The public posterior below remains the
        # calibrated scene posterior and therefore includes probability mass
        # from maplets outside the bounded scene budget.  Using its mean (or
        # comparing each retained anchor directly with that absolute null) as
        # a pose gate is invalid for a sparse anchor map: most real detector
        # nodes should be null, and the unprocessed maplet mass dominates even
        # when a small set of nodes has an unambiguous identity inside the
        # retrieved maplets.
        query_conditional_null_mass = (
            float(conditional_null_mass[query_row])
            + float(omitted_anchor_mass)
        )
        conditional_total_mass = (
            float(candidate_mass) + query_conditional_null_mass
        )
        conditional_normalizer = max(conditional_total_mass, 1e-12)
        if conditional_total_mass > 0.0:
            conditional_null[query_row] = float(
                query_conditional_null_mass / conditional_normalizer
            )
            conditional_retained_anchor[query_row] = float(
                candidate_mass / conditional_normalizer
            )
        normalizer = max(candidate_mass + null_mass, 1e-12)
        null[query_row] = float(null_mass / normalizer)
        for column, (anchor_id, (probability, descriptor_score)) in enumerate(
            ranked
        ):
            anchor_row = anchor_row_by_id.get(int(anchor_id))
            if anchor_row is None:
                continue
            output_ids[query_row, column] = int(anchor_id)
            output_xyz[query_row, column] = anchors.xyz[anchor_row]
            output_scores[query_row, column] = float(descriptor_score)
            output_probabilities[query_row, column] = float(
                probability / normalizer
            )
            valid[query_row, column] = True
        probability_sum = float(
            output_probabilities[query_row].sum(dtype=np.float64)
            + null[query_row]
        )
        if not np.isclose(probability_sum, 1.0, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "surface-anchor top-L posterior does not conserve probability"
            )
    pool = SurfaceAnchorCandidatePool(
        query_xy=query.keypoints_xy,
        anchor_ids=output_ids,
        xyz=output_xyz,
        descriptor_scores=output_scores,
        candidate_probabilities=output_probabilities,
        null_probabilities=null,
        valid_mask=valid,
    )
    return (
        pool,
        {
            "episode_count": int(episode_count),
            "raw_episode_count_before_scene_budget": int(raw_episode_count),
            "maximum_scene_maplets": int(config.maximum_scene_maplets),
            "matched_query_count": int(np.sum(np.any(valid, axis=1))),
            "mean_null_probability": float(np.mean(null)) if len(null) else 1.0,
            "mean_conditional_null_probability": (
                float(np.mean(conditional_null)) if len(conditional_null) else 1.0
            ),
            "conditionally_confident_group_count": int(
                np.sum(conditional_retained_anchor > conditional_null)
            ),
            "conditionally_matchable_group_count": int(
                np.sum(conditional_retained_anchor > 0.0)
            ),
            "conditional_pose_gate_ignores_unprocessed_maplet_mass": True,
            "omitted_top_l_anchor_mass_transferred_to_null": True,
            "posterior_probability_conserved": True,
            "uses_query_aligned_radio_final_maplets": bool(
                query_aligned_maplet_match is not None
            ),
            "feature_preferred_base_fill_target": (
                int(feature_preferred_base_fill_target)
                if feature_preferred_base_fill_target is not None
                else None
            ),
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )


def save_surface_anchor_set_matcher(
    path: Path,
    model: SurfaceAnchorSetMatcher,
    *,
    metadata: Mapping[str, object],
) -> None:
    payload = {
        "format": SURFACE_ANCHOR_SET_MATCHER_FORMAT,
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "metadata": {
            **dict(metadata),
            "uses_mapping_rgb_at_inference": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
            "identity": "stable_surface_anchor_id",
            "assignment": "maplet_conditioned_partial_optimal_transport",
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_surface_anchor_set_matcher(
    path: Path,
    *,
    device: str | torch.device,
) -> tuple[SurfaceAnchorSetMatcher, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device)
    if str(payload.get("format")) != SURFACE_ANCHOR_SET_MATCHER_FORMAT:
        raise ValueError("surface-anchor matcher checkpoint format is invalid")
    metadata = dict(payload.get("metadata") or {})
    forbidden_true = (
        "uses_mapping_rgb_at_inference",
        "uses_loftr",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_radio_intermediate",
    )
    if any(bool(metadata.get(name, False)) for name in forbidden_true):
        raise ValueError("surface-anchor matcher violates the map-only contract")
    config = SurfaceAnchorSetMatcherConfig(**dict(payload["config"]))
    model = SurfaceAnchorSetMatcher(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
