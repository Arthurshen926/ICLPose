"""Anonymous maplet appearance-mode pose voting for V6 coarse localization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_pose_proposal import (
    MapletPoseHypothesis,
)


def _rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ij,...kj->...ik", left, right)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _weighted_rotation_mean(
    rotations: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    matrix = np.einsum("n,nij->ij", weights, rotations)
    left, _singular, right = np.linalg.svd(matrix)
    result = left @ right
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


@dataclass(frozen=True)
class AnonymousMapletPoseVoteBank:
    """Pose sufficient statistics keyed by anonymous descriptor components.

    The bank stores neither mapping image identity nor an observation
    descriptor list.  Each retrieval component owns a small mixture of camera
    pose statistics learned offline from its mapping support.
    """

    component_maplet_ids: np.ndarray
    vote_offsets: np.ndarray
    camera_centers: np.ndarray
    rotations_w2c: np.ndarray
    vote_weights: np.ndarray
    translation_sigma_m: np.ndarray
    rotation_sigma_deg: np.ndarray
    component_descriptor_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        component_ids = np.asarray(
            self.component_maplet_ids, dtype=np.int64
        ).reshape(-1)
        offsets = np.asarray(self.vote_offsets, dtype=np.int64).reshape(-1)
        if (
            offsets.shape != (component_ids.size + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("vote_offsets must provide votes for every component")
        vote_count = int(offsets[-1])
        centers = np.asarray(self.camera_centers, dtype=np.float64)
        rotations = np.asarray(self.rotations_w2c, dtype=np.float64)
        weights = np.asarray(self.vote_weights, dtype=np.float32).reshape(-1)
        translation_sigma = np.asarray(
            self.translation_sigma_m, dtype=np.float32
        ).reshape(-1)
        rotation_sigma = np.asarray(
            self.rotation_sigma_deg, dtype=np.float32
        ).reshape(-1)
        if centers.shape != (vote_count, 3):
            raise ValueError("camera_centers must have shape (V,3)")
        if rotations.shape != (vote_count, 3, 3):
            raise ValueError("rotations_w2c must have shape (V,3,3)")
        if any(
            value.shape != (vote_count,)
            for value in (weights, translation_sigma, rotation_sigma)
        ):
            raise ValueError("vote statistics must have shape (V,)")
        if (
            np.any(weights < 0.0)
            or np.any(translation_sigma < 0.0)
            or np.any(rotation_sigma < 0.0)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(rotations))
        ):
            raise ValueError("invalid anonymous pose-vote statistics")
        normalized = weights.copy()
        for row in range(component_ids.size):
            begin, end = int(offsets[row]), int(offsets[row + 1])
            normalized[begin:end] /= max(
                float(np.sum(normalized[begin:end])), 1e-8
            )
        metadata = dict(self.metadata or {})
        for forbidden in (
            "stores_mapping_rgb",
            "stores_mapping_image_ids",
            "stores_mapping_image_paths",
            "stores_observation_descriptors",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_alike_descriptors",
            "uses_radio_intermediate",
        ):
            if bool(metadata.get(forbidden, False)):
                raise ValueError(f"pose-vote bank violates contract: {forbidden}")
        object.__setattr__(self, "component_maplet_ids", component_ids)
        object.__setattr__(self, "vote_offsets", offsets)
        object.__setattr__(self, "camera_centers", centers)
        object.__setattr__(self, "rotations_w2c", rotations)
        object.__setattr__(self, "vote_weights", normalized)
        object.__setattr__(self, "translation_sigma_m", translation_sigma)
        object.__setattr__(self, "rotation_sigma_deg", rotation_sigma)
        object.__setattr__(self, "metadata", metadata)

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            component_maplet_ids=self.component_maplet_ids,
            vote_offsets=self.vote_offsets,
            camera_centers=self.camera_centers.astype(np.float32),
            rotations_w2c=self.rotations_w2c.astype(np.float32),
            vote_weights=self.vote_weights,
            translation_sigma_m=self.translation_sigma_m,
            rotation_sigma_deg=self.rotation_sigma_deg,
            component_descriptor_sha256=np.asarray(
                self.component_descriptor_sha256
            ),
            metadata_json=np.asarray(
                json.dumps(dict(self.metadata or {}), sort_keys=True)
            ),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "AnonymousMapletPoseVoteBank":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                component_maplet_ids=data["component_maplet_ids"],
                vote_offsets=data["vote_offsets"],
                camera_centers=data["camera_centers"],
                rotations_w2c=data["rotations_w2c"],
                vote_weights=data["vote_weights"],
                translation_sigma_m=data["translation_sigma_m"],
                rotation_sigma_deg=data["rotation_sigma_deg"],
                component_descriptor_sha256=str(
                    data["component_descriptor_sha256"].item()
                ),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def retrieval_descriptor_sha256(bank: SurfaceRetrievalMapletBank) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(bank.maplet_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(bank.descriptor_offsets, dtype=np.int64).tobytes())
    digest.update(np.asarray(bank.descriptors, dtype=np.float32).tobytes())
    return digest.hexdigest()


def vote_maplet_poses(
    query_descriptors: np.ndarray,
    retrieval_bank: SurfaceRetrievalMapletBank,
    vote_bank: AnonymousMapletPoseVoteBank,
    *,
    descriptor_temperature: float = 0.08,
    candidates_per_region: int = 16,
    maximum_modes: int = 16,
    translation_kernel_m: float = 0.75,
    rotation_kernel_deg: float = 12.0,
) -> tuple[MapletPoseHypothesis, ...]:
    """Cluster anonymous appearance-conditioned votes directly in SE(3)."""

    query = np.asarray(query_descriptors, dtype=np.float32)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    if (
        vote_bank.component_descriptor_sha256
        != retrieval_descriptor_sha256(retrieval_bank)
    ):
        raise ValueError("pose-vote/retrieval descriptor lineage mismatch")
    component_maplets = np.repeat(
        retrieval_bank.maplet_ids,
        np.diff(retrieval_bank.descriptor_offsets),
    )
    if not np.array_equal(component_maplets, vote_bank.component_maplet_ids):
        raise ValueError("pose-vote components differ from retrieval bank")
    logits = query @ retrieval_bank.descriptors.T
    logits += np.log(
        np.maximum(
            np.repeat(
                retrieval_bank.descriptor_weights,
                1,
            )[None],
            1e-8,
        )
    ) * float(descriptor_temperature)
    scaled = logits / max(float(descriptor_temperature), 1e-4)
    posterior = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
    keep = min(max(int(candidates_per_region), 1), scaled.shape[1])
    candidate_rows = np.argpartition(
        -scaled, kth=keep - 1, axis=1
    )[:, :keep]
    component_evidence = np.zeros(
        (retrieval_bank.descriptors.shape[0],), dtype=np.float64
    )
    for query_row in range(query.shape[0]):
        rows = candidate_rows[query_row]
        # Max aggregation prevents a repetitive facade occupying many query
        # regions from multiplying the same pose vote without bound.
        component_evidence[rows] = np.maximum(
            component_evidence[rows], posterior[query_row, rows]
        )
    active_components = np.flatnonzero(component_evidence > 0.0)
    vote_rows = []
    vote_evidence = []
    vote_components = []
    for component in active_components.tolist():
        begin, end = (
            int(vote_bank.vote_offsets[component]),
            int(vote_bank.vote_offsets[component + 1]),
        )
        rows = np.arange(begin, end, dtype=np.int64)
        vote_rows.extend(rows.tolist())
        vote_evidence.extend(
            (
                component_evidence[component]
                * vote_bank.vote_weights[rows]
            ).tolist()
        )
        vote_components.extend([component] * rows.size)
    if not vote_rows:
        return tuple()
    vote_rows_array = np.asarray(vote_rows, dtype=np.int64)
    evidence = np.asarray(vote_evidence, dtype=np.float64)
    components = np.asarray(vote_components, dtype=np.int64)
    centers = vote_bank.camera_centers[vote_rows_array]
    rotations = vote_bank.rotations_w2c[vote_rows_array]
    translation_scale = np.maximum(
        float(translation_kernel_m)
        + vote_bank.translation_sigma_m[vote_rows_array],
        1e-3,
    )
    rotation_scale = np.maximum(
        float(rotation_kernel_deg)
        + vote_bank.rotation_sigma_deg[vote_rows_array],
        1e-3,
    )
    translation_distance = np.linalg.norm(
        centers[:, None] - centers[None], axis=2
    )
    rotation_distance = _rotation_angle_deg(
        rotations[:, None], rotations[None]
    )
    kernel = np.exp(
        -0.5
        * (
            (translation_distance / translation_scale[None]) ** 2
            + (rotation_distance / rotation_scale[None]) ** 2
        )
    )
    density = kernel @ evidence
    order = np.argsort(-density, kind="stable")
    results: list[MapletPoseHypothesis] = []
    for seed in order.tolist():
        local = (
            (translation_distance[seed] <= 2.0 * translation_scale[seed])
            & (rotation_distance[seed] <= 2.0 * rotation_scale[seed])
        )
        weights = evidence[local] * kernel[seed, local]
        if float(np.sum(weights)) <= 0.0:
            continue
        weights /= np.sum(weights)
        center = np.sum(centers[local] * weights[:, None], axis=0)
        rotation = _weighted_rotation_mean(rotations[local], weights)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = -rotation @ center
        duplicate = False
        for kept in results:
            kept_center = -kept.pose_w2c[:3, :3].T @ kept.pose_w2c[:3, 3]
            if (
                np.linalg.norm(center - kept_center)
                < 0.5 * float(translation_kernel_m)
                and float(
                    _rotation_angle_deg(
                        rotation[None], kept.pose_w2c[None, :3, :3]
                    )[0]
                )
                < 0.5 * float(rotation_kernel_deg)
            ):
                duplicate = True
                break
        if duplicate:
            continue
        local_components = np.unique(components[local])
        results.append(
            MapletPoseHypothesis(
                pose_w2c=pose,
                score=float(density[seed]),
                supporting_group_count=int(local_components.size),
                sampled_maplet_ids=np.unique(
                    component_maplets[local_components]
                ).astype(np.int64),
            )
        )
        if len(results) >= int(maximum_modes):
            break
    return tuple(results)
