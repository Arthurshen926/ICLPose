"""Pose-free RADIO retrieval over the physical Goal-Maplet hierarchy.

This module is deliberately narrower than localization: it accepts a complete
RADIO token grid and returns physical parent/child rankings.  It has no pose,
keypoint, PnP, SfM, mapping-image, or renderer interface.  Keeping this seam
small makes retrieval quality measurable before any pose estimator can hide a
failure (or receive ground-truth information by accident).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .lineage import arrays_sha256
from .physical_map import GoalMapletPhysicalMap


SCHEMA = "goal_maplet_pure_radio_physical_retrieval_v1"
SCENE_AGGREGATION = "fixed_4x4_blocks_top4_max_sum_v1"
PARENT_SCENE_RANK_RAW = "raw_parent_evidence_v1"
PARENT_SCENE_RANK_SURFACE_DENSITY = "parent_evidence_per_surface_area_v1"

_REQUIRED_FALSE_CLAIMS = (
    "uses_query_pose",
    "uses_query_ground_truth",
    "uses_alike",
    "uses_pnp",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_mapping_rgb",
    "uses_image_retrieval",
)


def all_radio_token_coordinates(height: int, width: int) -> np.ndarray:
    """Return every token exactly once in row-major ``(x,y)`` order."""

    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("RADIO token dimensions must be positive")
    yy, xx = np.meshgrid(
        np.arange(int(height), dtype=np.int16),
        np.arange(int(width), dtype=np.int16),
        indexing="ij",
    )
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def aggregate_sparse_token_evidence(
    token_xy: np.ndarray,
    candidate_rows: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    entity_count: int,
    token_height: int,
    token_width: int,
    block_rows: int = 4,
    block_cols: int = 4,
    top_blocks: int = 4,
) -> np.ndarray:
    """Aggregate dense-token evidence without rewarding adjacent duplicates.

    For each physical entity we retain the maximum probability in each fixed
    image block, then sum its strongest blocks.  The fixed partition is
    query-independent, bounded, and deterministic.  It is a baseline scene
    aggregator, not a learned pose prior.
    """

    xy = np.asarray(token_xy, dtype=np.int64)
    rows = np.asarray(candidate_rows, dtype=np.int64)
    probability = np.asarray(candidate_probabilities, dtype=np.float64)
    expected_xy = all_radio_token_coordinates(token_height, token_width).astype(
        np.int64
    )
    if (
        xy.shape != expected_xy.shape
        or not np.array_equal(xy, expected_xy)
        or rows.ndim != 2
        or rows.shape[0] != xy.shape[0]
        or probability.shape != rows.shape
        or int(entity_count) <= 0
        or int(block_rows) <= 0
        or int(block_cols) <= 0
        or int(top_blocks) <= 0
        or np.any(~np.isfinite(probability))
        or np.any(probability < 0.0)
    ):
        raise ValueError("invalid full-token sparse evidence")
    valid = (rows >= 0) & (rows < int(entity_count)) & (probability > 0.0)
    y_block = np.minimum(
        xy[:, 1] * int(block_rows) // int(token_height), int(block_rows) - 1
    )
    x_block = np.minimum(
        xy[:, 0] * int(block_cols) // int(token_width), int(block_cols) - 1
    )
    block = y_block * int(block_cols) + x_block
    block_count = int(block_rows) * int(block_cols)
    block_max = np.zeros((int(entity_count), block_count), dtype=np.float64)
    if np.any(valid):
        entity = rows[valid]
        repeated_block = np.broadcast_to(block[:, None], rows.shape)[valid]
        np.maximum.at(block_max, (entity, repeated_block), probability[valid])
    keep = min(int(top_blocks), block_count)
    strongest = np.partition(block_max, block_count - keep, axis=1)[:, -keep:]
    return np.sum(strongest, axis=1)


def _sorted_members(physical: GoalMapletPhysicalMap, child_row: int) -> np.ndarray:
    start = int(physical.child_member_offsets[int(child_row)])
    end = int(physical.child_member_offsets[int(child_row) + 1])
    return np.unique(
        np.asarray(
            physical.child_member_primitive_rows[start:end], dtype=np.int64
        )
    )


def _sorted_intersection_size(left: np.ndarray, right: np.ndarray) -> int:
    i = j = count = 0
    while i < left.size and j < right.size:
        a, b = int(left[i]), int(right[j])
        if a == b:
            count += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return count


def rank_children_with_physical_iou_nms(
    scores: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_children: int = 64,
    maximum_primitive_iou: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rank child surfaces and suppress duplicate physical primitive support."""

    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    if (
        value.shape != physical.child_parent_rows.shape
        or np.any(~np.isfinite(value))
        or int(maximum_children) <= 0
        or not 0.0 <= float(maximum_primitive_iou) <= 1.0
    ):
        raise ValueError("invalid child scene scores")
    order = np.lexsort((np.arange(value.size, dtype=np.int64), -value))
    selected: list[int] = []
    selected_members: list[np.ndarray] = []
    suppressed = 0
    for row in order.tolist():
        if float(value[row]) <= 0.0:
            break
        members = _sorted_members(physical, int(row))
        duplicate = False
        for prior in selected_members:
            intersection = _sorted_intersection_size(members, prior)
            union = int(members.size + prior.size - intersection)
            iou = float(intersection / max(union, 1))
            if iou >= float(maximum_primitive_iou):
                duplicate = True
                break
        if duplicate:
            suppressed += 1
            continue
        selected.append(int(row))
        selected_members.append(members)
        if len(selected) >= int(maximum_children):
            break
    rows = np.asarray(selected, dtype=np.int64)
    return rows, value[rows].astype(np.float32), int(suppressed)


def rank_parent_regions(
    evidence_scores: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_parents: int = 64,
    semantics: str = PARENT_SCENE_RANK_RAW,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank physical regions by evidence or evidence per map-surface cost.

    A set-valued retriever is useful only relative to the physical region it
    returns.  The density control therefore divides a parent's scene evidence
    by the unique 2DGS ellipse area represented by that parent.  This is the
    fractional-knapsack priority for expected relevance under a surface-area
    budget; it uses map geometry only and never query pose or ground truth.
    """

    evidence = np.asarray(evidence_scores, dtype=np.float64).reshape(-1)
    if (
        evidence.shape != physical.maplet_ids.shape
        or np.any(~np.isfinite(evidence))
        or np.any(evidence < 0.0)
        or int(maximum_parents) <= 0
        or semantics not in (PARENT_SCENE_RANK_RAW, PARENT_SCENE_RANK_SURFACE_DENSITY)
    ):
        raise ValueError("invalid parent scene ranking")
    score = evidence.copy()
    if semantics == PARENT_SCENE_RANK_SURFACE_DENSITY:
        primitive_area = (
            np.pi
            * np.asarray(physical.primitive_scale1, dtype=np.float64)
            * np.asarray(physical.primitive_scale2, dtype=np.float64)
        )
        area = np.zeros((physical.maplet_ids.size,), dtype=np.float64)
        for parent in range(physical.maplet_ids.size):
            start = int(physical.membership_offsets[parent])
            end = int(physical.membership_offsets[parent + 1])
            members = np.unique(
                np.asarray(
                    physical.membership_primitive_rows[start:end], dtype=np.int64
                )
            )
            area[parent] = float(np.sum(primitive_area[members]))
        score = np.divide(
            evidence,
            area,
            out=np.zeros_like(evidence),
            where=area > 1e-12,
        )
    order = np.lexsort((physical.maplet_ids, -score))
    order = order[evidence[order] > 0.0][: int(maximum_parents)]
    return order.astype(np.int64), score[order].astype(np.float32)


@dataclass(frozen=True)
class PureRadioPhysicalRetrieval:
    image_id: str
    token_xy: np.ndarray
    token_parent_ids: np.ndarray
    token_parent_probabilities: np.ndarray
    token_out_of_map_probabilities: np.ndarray
    token_in_map_tail_probabilities: np.ndarray
    token_child_rows: np.ndarray
    token_child_probabilities: np.ndarray
    scene_parent_ids: np.ndarray
    scene_parent_scores: np.ndarray
    scene_child_rows: np.ndarray
    scene_child_scores: np.ndarray
    physical_map_sha256: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        xy = np.asarray(self.token_xy, dtype=np.int16)
        parent_ids = np.asarray(self.token_parent_ids, dtype=np.int64)
        parent_probability = np.asarray(
            self.token_parent_probabilities, dtype=np.float32
        )
        out_of_map = np.asarray(
            self.token_out_of_map_probabilities, dtype=np.float32
        ).reshape(-1)
        in_map_tail = np.asarray(
            self.token_in_map_tail_probabilities, dtype=np.float32
        ).reshape(-1)
        child_rows = np.asarray(self.token_child_rows, dtype=np.int64)
        child_probability = np.asarray(
            self.token_child_probabilities, dtype=np.float32
        )
        scene_parent_ids = np.asarray(self.scene_parent_ids, dtype=np.int64).reshape(
            -1
        )
        scene_parent_scores = np.asarray(
            self.scene_parent_scores, dtype=np.float32
        ).reshape(-1)
        scene_child_rows = np.asarray(self.scene_child_rows, dtype=np.int64).reshape(
            -1
        )
        scene_child_scores = np.asarray(
            self.scene_child_scores, dtype=np.float32
        ).reshape(-1)
        metadata = dict(self.metadata)
        height = int(metadata.get("token_height", 0))
        width = int(metadata.get("token_width", 0))
        expected_xy = all_radio_token_coordinates(height, width)
        if (
            xy.shape != expected_xy.shape
            or not np.array_equal(xy, expected_xy)
            or parent_ids.ndim != 2
            or parent_ids.shape[0] != xy.shape[0]
            or parent_probability.shape != parent_ids.shape
            or out_of_map.shape != (xy.shape[0],)
            or in_map_tail.shape != (xy.shape[0],)
            or child_rows.ndim != 2
            or child_rows.shape[0] != xy.shape[0]
            or child_probability.shape != child_rows.shape
            or scene_parent_ids.shape != scene_parent_scores.shape
            or scene_child_rows.shape != scene_child_scores.shape
        ):
            raise ValueError("pure retrieval arrays differ")
        probabilities = (
            parent_probability,
            out_of_map,
            in_map_tail,
            child_probability,
        )
        if any(
            np.any(~np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0))
            for value in probabilities
        ):
            raise ValueError("pure retrieval probabilities must lie in [0,1]")
        parent_mass = (
            np.sum(parent_probability, axis=1) + out_of_map + in_map_tail
        )
        if np.any(parent_mass > 1.0 + 2e-5):
            raise ValueError("parent probability mass is not conserved")
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a pure RADIO physical retrieval artifact")
        for key in _REQUIRED_FALSE_CLAIMS:
            if metadata.get(key) is not False:
                raise ValueError(f"pure retrieval contract requires {key}=false")
        if metadata.get("scene_aggregation") != SCENE_AGGREGATION:
            raise ValueError("unknown pure retrieval scene aggregation")
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "token_xy", xy)
        object.__setattr__(self, "token_parent_ids", parent_ids)
        object.__setattr__(self, "token_parent_probabilities", parent_probability)
        object.__setattr__(self, "token_out_of_map_probabilities", out_of_map)
        object.__setattr__(self, "token_in_map_tail_probabilities", in_map_tail)
        object.__setattr__(self, "token_child_rows", child_rows)
        object.__setattr__(self, "token_child_probabilities", child_probability)
        object.__setattr__(self, "scene_parent_ids", scene_parent_ids)
        object.__setattr__(self, "scene_parent_scores", scene_parent_scores)
        object.__setattr__(self, "scene_child_rows", scene_child_rows)
        object.__setattr__(self, "scene_child_scores", scene_child_scores)
        object.__setattr__(self, "metadata", metadata)
        declared = str(metadata.get("content_sha256", ""))
        if declared and declared != self.content_sha256:
            raise ValueError("pure retrieval content hash mismatch")

    @property
    def content_sha256(self) -> str:
        return arrays_sha256(
            {
                "token_xy": self.token_xy,
                "token_parent_ids": self.token_parent_ids,
                "token_parent_probabilities": self.token_parent_probabilities,
                "token_out_of_map_probabilities": self.token_out_of_map_probabilities,
                "token_in_map_tail_probabilities": self.token_in_map_tail_probabilities,
                "token_child_rows": self.token_child_rows,
                "token_child_probabilities": self.token_child_probabilities,
                "scene_parent_ids": self.scene_parent_ids,
                "scene_parent_scores": self.scene_parent_scores,
                "scene_child_rows": self.scene_child_rows,
                "scene_child_scores": self.scene_child_scores,
            }
        )

    def save_npz(self, path: Path) -> None:
        metadata = {
            **dict(self.metadata),
            "artifact_type": SCHEMA,
            "content_sha256": self.content_sha256,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            image_id=np.asarray(self.image_id),
            token_xy=self.token_xy,
            token_parent_ids=self.token_parent_ids,
            token_parent_probabilities=self.token_parent_probabilities,
            token_out_of_map_probabilities=self.token_out_of_map_probabilities,
            token_in_map_tail_probabilities=self.token_in_map_tail_probabilities,
            token_child_rows=self.token_child_rows,
            token_child_probabilities=self.token_child_probabilities,
            scene_parent_ids=self.scene_parent_ids,
            scene_parent_scores=self.scene_parent_scores,
            scene_child_rows=self.scene_child_rows,
            scene_child_scores=self.scene_child_scores,
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "PureRadioPhysicalRetrieval":
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {
                "image_id",
                "token_xy",
                "token_parent_ids",
                "token_parent_probabilities",
                "token_out_of_map_probabilities",
                "token_in_map_tail_probabilities",
                "token_child_rows",
                "token_child_probabilities",
                "scene_parent_ids",
                "scene_parent_scores",
                "scene_child_rows",
                "scene_child_scores",
                "physical_map_sha256",
                "metadata_json",
            }
            if set(data.files) != expected:
                raise ValueError("pure retrieval NPZ members differ")
            return cls(
                image_id=str(np.asarray(data["image_id"]).item()),
                token_xy=np.asarray(data["token_xy"]),
                token_parent_ids=np.asarray(data["token_parent_ids"]),
                token_parent_probabilities=np.asarray(
                    data["token_parent_probabilities"]
                ),
                token_out_of_map_probabilities=np.asarray(
                    data["token_out_of_map_probabilities"]
                ),
                token_in_map_tail_probabilities=np.asarray(
                    data["token_in_map_tail_probabilities"]
                ),
                token_child_rows=np.asarray(data["token_child_rows"]),
                token_child_probabilities=np.asarray(
                    data["token_child_probabilities"]
                ),
                scene_parent_ids=np.asarray(data["scene_parent_ids"]),
                scene_parent_scores=np.asarray(data["scene_parent_scores"]),
                scene_child_rows=np.asarray(data["scene_child_rows"]),
                scene_child_scores=np.asarray(data["scene_child_scores"]),
                physical_map_sha256=str(
                    np.asarray(data["physical_map_sha256"]).item()
                ),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )
