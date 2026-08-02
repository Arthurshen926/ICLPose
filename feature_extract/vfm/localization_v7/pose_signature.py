"""Pose-aware whole-set signatures from one VFM maplet retrieval pass.

The representation deliberately stores no mapping image, image identifier, or
per-observation feature descriptor.  It retains only maplet posterior moments
and the associated mapping camera pose.  Query-to-map feature interaction ends
at Stage A; pose retrieval below compares fixed sufficient statistics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.cambridge_pose_lattice import (
    camera_center_from_pose_w2c,
    pose_w2c_from_center_rotation,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
)


def rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ij,...kj->...ik", left, right)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def weighted_rotation_mean(
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
class PoseSignature:
    """Sparse semantic-layout sufficient statistics in a dense maplet basis."""

    identity: np.ndarray
    layout_mean_xy: np.ndarray
    layout_extent_xy: np.ndarray
    layout_variance_xy: np.ndarray
    layout_mass: np.ndarray

    def __post_init__(self) -> None:
        identity = np.asarray(self.identity, dtype=np.float32).reshape(-1)
        count = identity.size
        mean = np.asarray(self.layout_mean_xy, dtype=np.float32)
        extent = np.asarray(self.layout_extent_xy, dtype=np.float32)
        variance = np.asarray(self.layout_variance_xy, dtype=np.float32)
        mass = np.asarray(self.layout_mass, dtype=np.float32).reshape(-1)
        if any(value.shape != (count, 2) for value in (mean, extent, variance)):
            raise ValueError("pose-signature layout fields must have shape (M,2)")
        if mass.shape != (count,):
            raise ValueError("pose-signature layout mass must have shape (M,)")
        if (
            np.any(identity < 0.0)
            or np.any(mass < 0.0)
            or np.any(variance < 0.0)
            or not all(
                np.all(np.isfinite(value))
                for value in (identity, mean, extent, variance, mass)
            )
        ):
            raise ValueError("pose signature contains invalid statistics")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "layout_mean_xy", mean)
        object.__setattr__(self, "layout_extent_xy", extent)
        object.__setattr__(self, "layout_variance_xy", variance)
        object.__setattr__(self, "layout_mass", mass)


def pose_signature_from_retrieval(
    retrieval: MapletRetrievalResult,
    maplet_ids: np.ndarray,
    image_size_wh: tuple[int, int],
) -> PoseSignature:
    """Compress Top-K identity posteriors and their query-region geometry."""

    basis = np.asarray(maplet_ids, dtype=np.int64).reshape(-1)
    if np.unique(basis).size != basis.size:
        raise ValueError("pose-signature maplet basis must be unique")
    row_by_id = {int(value): row for row, value in enumerate(basis.tolist())}
    count = basis.size
    identity = np.zeros(count, dtype=np.float64)
    for maplet_id, evidence in zip(
        retrieval.ranked_maplet_ids.tolist(), retrieval.evidence.tolist()
    ):
        row = row_by_id.get(int(maplet_id))
        if row is not None:
            identity[row] = max(float(evidence), 0.0)
    # Hellinger embedding is robust to a few dominant maplets and makes the
    # identity dot product a Bhattacharyya affinity.
    identity_sum = float(np.sum(identity))
    if identity_sum > 0.0:
        identity = np.sqrt(identity / identity_sum)
        identity /= max(float(np.linalg.norm(identity)), 1e-12)

    mass = np.zeros(count, dtype=np.float64)
    first = np.zeros((count, 2), dtype=np.float64)
    second = np.zeros((count, 2), dtype=np.float64)
    extent_sum = np.zeros((count, 2), dtype=np.float64)
    image_scale = np.maximum(
        np.asarray(image_size_wh, dtype=np.float64).reshape(2), 1.0
    )
    for group in retrieval.groups:
        xy = np.asarray(group.query_region_xy, dtype=np.float64) / image_scale
        extent = (
            np.asarray(group.query_region_extent, dtype=np.float64)
            / image_scale
        )
        for maplet_id, probability in zip(
            group.maplet_ids.tolist(), group.probabilities.tolist()
        ):
            row = row_by_id.get(int(maplet_id))
            weight = max(float(probability), 0.0)
            if row is None or weight <= 0.0:
                continue
            mass[row] += weight
            first[row] += weight * xy
            second[row] += weight * xy * xy
            extent_sum[row] += weight * extent
    available = mass > 1e-12
    mean = np.zeros((count, 2), dtype=np.float64)
    extent = np.zeros((count, 2), dtype=np.float64)
    variance = np.zeros((count, 2), dtype=np.float64)
    mean[available] = first[available] / mass[available, None]
    extent[available] = extent_sum[available] / mass[available, None]
    variance[available] = np.maximum(
        second[available] / mass[available, None]
        - mean[available] * mean[available],
        0.0,
    )
    # Only relative layout support is meaningful across images.
    mass_sum = float(np.sum(mass))
    if mass_sum > 0.0:
        mass /= mass_sum
    return PoseSignature(
        identity=identity.astype(np.float32),
        layout_mean_xy=mean.astype(np.float32),
        layout_extent_xy=extent.astype(np.float32),
        layout_variance_xy=variance.astype(np.float32),
        layout_mass=mass.astype(np.float32),
    )


@dataclass(frozen=True)
class PoseSignatureBank:
    """Mapping-pose prototypes expressed only by maplet sufficient statistics."""

    maplet_ids: np.ndarray
    poses_w2c: np.ndarray
    identity: np.ndarray
    layout_mean_xy: np.ndarray
    layout_extent_xy: np.ndarray
    layout_variance_xy: np.ndarray
    layout_mass: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        maplet_ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        poses = np.asarray(self.poses_w2c, dtype=np.float64)
        identity = np.asarray(self.identity, dtype=np.float32)
        prototype_count = poses.shape[0]
        maplet_count = maplet_ids.size
        if poses.shape != (prototype_count, 4, 4) or prototype_count == 0:
            raise ValueError("pose-signature bank requires poses (N,4,4)")
        if identity.shape != (prototype_count, maplet_count):
            raise ValueError("pose-signature identity must have shape (N,M)")
        expected_layout = (prototype_count, maplet_count, 2)
        layouts = (
            np.asarray(self.layout_mean_xy, dtype=np.float32),
            np.asarray(self.layout_extent_xy, dtype=np.float32),
            np.asarray(self.layout_variance_xy, dtype=np.float32),
        )
        if any(value.shape != expected_layout for value in layouts):
            raise ValueError("pose-signature layout fields must have shape (N,M,2)")
        mass = np.asarray(self.layout_mass, dtype=np.float32)
        if mass.shape != (prototype_count, maplet_count):
            raise ValueError("pose-signature layout mass must have shape (N,M)")
        metadata = dict(self.metadata)
        required_false = (
            "stores_mapping_rgb",
            "stores_mapping_image_ids",
            "stores_mapping_image_paths",
            "stores_observation_descriptors",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_alike_descriptors",
            "uses_radio_intermediate",
        )
        if metadata.get("representation") != "pose_aware_maplet_sufficient_statistics":
            raise ValueError("invalid V7 pose-signature representation")
        for key in required_false:
            if bool(metadata.get(key, False)):
                raise ValueError(f"pose-signature bank violates contract: {key}")
        if not bool(metadata.get("uses_mapping_pose_statistics", False)):
            raise ValueError("pose-signature bank must declare pose statistics")
        object.__setattr__(self, "maplet_ids", maplet_ids)
        object.__setattr__(self, "poses_w2c", poses)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "layout_mean_xy", layouts[0])
        object.__setattr__(self, "layout_extent_xy", layouts[1])
        object.__setattr__(self, "layout_variance_xy", layouts[2])
        object.__setattr__(self, "layout_mass", mass)
        object.__setattr__(self, "metadata", metadata)

    def save_npz(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            maplet_ids=self.maplet_ids,
            poses_w2c=self.poses_w2c,
            identity=self.identity.astype(np.float16),
            layout_mean_xy=self.layout_mean_xy.astype(np.float16),
            layout_extent_xy=self.layout_extent_xy.astype(np.float16),
            layout_variance_xy=self.layout_variance_xy.astype(np.float16),
            layout_mass=self.layout_mass.astype(np.float16),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "PoseSignatureBank":
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {
                "maplet_ids",
                "poses_w2c",
                "identity",
                "layout_mean_xy",
                "layout_extent_xy",
                "layout_variance_xy",
                "layout_mass",
                "metadata_json",
            }
            if set(data.files) != expected:
                raise ValueError("non-canonical V7 pose-signature artifact")
            return cls(
                maplet_ids=data["maplet_ids"],
                poses_w2c=data["poses_w2c"],
                identity=data["identity"],
                layout_mean_xy=data["layout_mean_xy"],
                layout_extent_xy=data["layout_extent_xy"],
                layout_variance_xy=data["layout_variance_xy"],
                layout_mass=data["layout_mass"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_pose_signatures(
    query: PoseSignature,
    bank: PoseSignatureBank,
    *,
    identity_weight: float = 1.0,
    layout_weight: float = 0.35,
    extent_weight: float = 0.10,
    layout_sigma: float = 0.18,
    extent_sigma: float = 0.12,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score a query without any query-to-map descriptor interaction."""

    if query.identity.shape != (bank.maplet_ids.size,):
        raise ValueError("query signature and bank maplet bases differ")
    identity = np.asarray(bank.identity, dtype=np.float64) @ np.asarray(
        query.identity, dtype=np.float64
    )
    query_mass = np.asarray(query.layout_mass, dtype=np.float64)
    prototype_mass = np.asarray(bank.layout_mass, dtype=np.float64)
    overlap = np.sqrt(np.maximum(prototype_mass * query_mass[None], 0.0))
    overlap_sum = np.sum(overlap, axis=1)
    xy_delta2 = np.sum(
        (
            np.asarray(bank.layout_mean_xy, dtype=np.float64)
            - np.asarray(query.layout_mean_xy, dtype=np.float64)[None]
        )
        ** 2,
        axis=2,
    )
    extent_delta2 = np.sum(
        (
            np.asarray(bank.layout_extent_xy, dtype=np.float64)
            - np.asarray(query.layout_extent_xy, dtype=np.float64)[None]
        )
        ** 2,
        axis=2,
    )
    layout = np.sum(
        overlap * np.exp(-0.5 * xy_delta2 / max(layout_sigma**2, 1e-12)),
        axis=1,
    ) / np.maximum(overlap_sum, 1e-12)
    extent = np.sum(
        overlap
        * np.exp(-0.5 * extent_delta2 / max(extent_sigma**2, 1e-12)),
        axis=1,
    ) / np.maximum(overlap_sum, 1e-12)
    layout[overlap_sum <= 1e-12] = 0.0
    extent[overlap_sum <= 1e-12] = 0.0
    total = (
        float(identity_weight) * identity
        + float(layout_weight) * layout
        + float(extent_weight) * extent
    )
    return total, {
        "identity": identity,
        "layout": layout,
        "extent": extent,
        "overlap": overlap_sum,
    }


@dataclass(frozen=True)
class PoseMode:
    pose_w2c: np.ndarray
    score: float
    prototype_rows: np.ndarray
    source: str


def pose_signature_vector(signature: PoseSignature) -> np.ndarray:
    """Vector used only for local pose regression on fixed statistics."""

    root_mass = np.sqrt(
        np.maximum(np.asarray(signature.layout_mass, dtype=np.float64), 0.0)
    )[:, None]
    layout = root_mass * (
        np.asarray(signature.layout_mean_xy, dtype=np.float64) - 0.5
    )
    extent = root_mass * np.asarray(
        signature.layout_extent_xy, dtype=np.float64
    )
    return np.concatenate(
        [
            np.asarray(signature.identity, dtype=np.float64),
            layout.reshape(-1),
            extent.reshape(-1),
        ]
    )


def _bank_signature_vectors(bank: PoseSignatureBank) -> np.ndarray:
    root_mass = np.sqrt(
        np.maximum(np.asarray(bank.layout_mass, dtype=np.float64), 0.0)
    )[:, :, None]
    layout = root_mass * (
        np.asarray(bank.layout_mean_xy, dtype=np.float64) - 0.5
    )
    extent = root_mass * np.asarray(
        bank.layout_extent_xy, dtype=np.float64
    )
    return np.concatenate(
        [
            np.asarray(bank.identity, dtype=np.float64),
            layout.reshape(layout.shape[0], -1),
            extent.reshape(extent.shape[0], -1),
        ],
        axis=1,
    )


def _local_linear_pose(
    bank: PoseSignatureBank,
    bank_vectors: np.ndarray,
    query_vector: np.ndarray,
    rows: np.ndarray,
    scores: np.ndarray,
    *,
    score_temperature: float,
    ridge_fraction: float,
    maximum_translation_extrapolation_m: float,
    maximum_rotation_extrapolation_deg: float,
) -> np.ndarray:
    local_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    local_scores = np.asarray(scores, dtype=np.float64)[local_rows]
    weights = np.exp(
        np.clip(
            (local_scores - float(np.max(local_scores)))
            / max(float(score_temperature), 1e-6),
            -60.0,
            0.0,
        )
    )
    weights /= max(float(np.sum(weights)), 1e-12)
    rotations = np.asarray(bank.poses_w2c[local_rows, :3, :3], dtype=np.float64)
    centers = np.stack(
        [camera_center_from_pose_w2c(bank.poses_w2c[row]) for row in local_rows]
    )
    mean_center = np.sum(weights[:, None] * centers, axis=0)
    mean_rotation = weighted_rotation_mean(rotations, weights)
    if local_rows.size < 4:
        return pose_w2c_from_center_rotation(mean_center, mean_rotation)
    x = np.asarray(bank_vectors[local_rows], dtype=np.float64)
    x_mean = np.sum(weights[:, None] * x, axis=0)
    root_weight = np.sqrt(weights)[:, None]
    x_weighted = root_weight * (x - x_mean[None])
    query_delta = np.asarray(query_vector, dtype=np.float64) - x_mean
    gram = x_weighted @ x_weighted.T
    ridge = max(
        float(ridge_fraction)
        * float(np.trace(gram))
        / max(local_rows.size, 1),
        1e-8,
    )
    projection = (
        query_delta
        @ x_weighted.T
        @ np.linalg.solve(
            gram + ridge * np.eye(local_rows.size, dtype=np.float64),
            np.eye(local_rows.size, dtype=np.float64),
        )
    )
    center_target = root_weight * (centers - mean_center[None])
    center_delta = projection @ center_target
    center_norm = float(np.linalg.norm(center_delta))
    if center_norm > float(maximum_translation_extrapolation_m):
        center_delta *= float(maximum_translation_extrapolation_m) / center_norm

    relative_rotation = Rotation.from_matrix(
        rotations @ mean_rotation.T
    ).as_rotvec()
    rotation_target = root_weight * relative_rotation
    rotation_delta = projection @ rotation_target
    maximum_rotation = np.deg2rad(
        max(float(maximum_rotation_extrapolation_deg), 0.0)
    )
    rotation_norm = float(np.linalg.norm(rotation_delta))
    if rotation_norm > maximum_rotation:
        rotation_delta *= maximum_rotation / rotation_norm
    predicted_rotation = (
        Rotation.from_rotvec(rotation_delta).as_matrix() @ mean_rotation
    )
    return pose_w2c_from_center_rotation(
        mean_center + center_delta, predicted_rotation
    )


def local_linear_pose_modes(
    query: PoseSignature,
    bank: PoseSignatureBank,
    scores: np.ndarray,
    *,
    maximum_prototypes: int = 64,
    maximum_modes: int = 16,
    translation_radius_m: float = 1.5,
    rotation_radius_deg: float = 18.0,
    score_temperature: float = 0.04,
    ridge_fraction: float = 0.05,
    maximum_translation_extrapolation_m: float = 0.75,
    maximum_rotation_extrapolation_deg: float = 8.0,
) -> tuple[PoseMode, ...]:
    """Locally refit continuous pose from mapping-sequence pose statistics."""

    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    if value.shape != (bank.poses_w2c.shape[0],):
        raise ValueError("signature score count differs from pose prototypes")
    selected = np.argsort(-value, kind="mergesort")[: max(int(maximum_prototypes), 1)]
    centers = np.stack(
        [camera_center_from_pose_w2c(bank.poses_w2c[row]) for row in selected]
    )
    rotations = np.asarray(bank.poses_w2c[selected, :3, :3], dtype=np.float64)
    bank_vectors = _bank_signature_vectors(bank)
    query_vector = pose_signature_vector(query)
    unassigned = np.ones(selected.size, dtype=bool)
    modes = []
    for seed in range(selected.size):
        if not bool(unassigned[seed]):
            continue
        translation = np.linalg.norm(centers - centers[seed], axis=1)
        rotation = rotation_angle_deg(rotations, rotations[seed])
        members = np.flatnonzero(
            unassigned
            & (translation <= max(float(translation_radius_m), 0.0))
            & (rotation <= max(float(rotation_radius_deg), 0.0))
        )
        if members.size == 0:
            members = np.asarray([seed], dtype=np.int64)
        unassigned[members] = False
        rows = selected[members]
        pose = _local_linear_pose(
            bank,
            bank_vectors,
            query_vector,
            rows,
            value,
            score_temperature=float(score_temperature),
            ridge_fraction=float(ridge_fraction),
            maximum_translation_extrapolation_m=float(
                maximum_translation_extrapolation_m
            ),
            maximum_rotation_extrapolation_deg=float(
                maximum_rotation_extrapolation_deg
            ),
        )
        modes.append(
            PoseMode(
                pose_w2c=pose,
                score=float(
                    np.max(value[rows]) + 0.01 * np.log(max(rows.size, 1))
                ),
                prototype_rows=rows,
                source="pose_signature_local_linear",
            )
        )
    modes.sort(key=lambda item: -item.score)
    return tuple(modes[: max(int(maximum_modes), 1)])


def geometry_only_pose_modes(
    bank: PoseSignatureBank,
    scores: np.ndarray,
    *,
    maximum_prototypes: int = 64,
    maximum_modes: int = 16,
) -> tuple[PoseMode, ...]:
    """Find pose-density modes after retrieval, with no feature interaction."""

    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    selected = np.argsort(-value, kind="mergesort")[: max(int(maximum_prototypes), 1)]
    centers = np.stack(
        [camera_center_from_pose_w2c(bank.poses_w2c[row]) for row in selected]
    )
    rotations = np.asarray(bank.poses_w2c[selected, :3, :3], dtype=np.float64)
    rank_weight = np.exp(-np.arange(selected.size, dtype=np.float64) / 16.0)
    candidates: list[PoseMode] = []
    for translation_sigma, rotation_sigma in (
        (0.75, 8.0),
        (1.5, 15.0),
        (3.0, 30.0),
    ):
        translation2 = np.sum(
            (centers[:, None] - centers[None]) ** 2, axis=2
        )
        rotation = np.stack(
            [rotation_angle_deg(rotations, seed) for seed in rotations], axis=1
        )
        kernel = np.exp(
            -0.5 * translation2 / translation_sigma**2
            -0.5 * rotation**2 / rotation_sigma**2
        )
        density = kernel @ rank_weight
        retained: list[int] = []
        for seed in np.argsort(-density, kind="mergesort").tolist():
            if any(
                np.linalg.norm(centers[seed] - centers[other])
                <= 0.5 * translation_sigma
                and float(rotation_angle_deg(rotations[seed], rotations[other]))
                <= 0.5 * rotation_sigma
                for other in retained
            ):
                continue
            retained.append(seed)
            weight = kernel[:, seed] * rank_weight
            weight /= max(float(np.sum(weight)), 1e-12)
            center = np.sum(weight[:, None] * centers, axis=0)
            rotation_mean = weighted_rotation_mean(rotations, weight)
            candidates.append(
                PoseMode(
                    pose_w2c=pose_w2c_from_center_rotation(
                        center, rotation_mean
                    ),
                    score=float(
                        np.log(
                            max(
                                float(density[seed])
                                / max(float(np.max(density)), 1e-12),
                                1e-12,
                            )
                        )
                        - 0.02 * len(retained)
                    ),
                    prototype_rows=selected[kernel[:, seed] >= np.exp(-2.0)],
                    source=(
                        "pose_signature_geometry_kde_"
                        f"t{translation_sigma:g}_r{rotation_sigma:g}"
                    ),
                )
            )
            if len(retained) >= 6:
                break
    candidates.sort(key=lambda item: -item.score)
    output: list[PoseMode] = []
    for candidate in candidates:
        center = camera_center_from_pose_w2c(candidate.pose_w2c)
        rotation = candidate.pose_w2c[:3, :3]
        duplicate = any(
            np.linalg.norm(center - camera_center_from_pose_w2c(other.pose_w2c))
            < 0.25
            and float(rotation_angle_deg(rotation, other.pose_w2c[:3, :3])) < 3.0
            for other in output
        )
        if not duplicate:
            output.append(candidate)
        if len(output) >= max(int(maximum_modes), 1):
            break
    return tuple(output)


def pose_modes_from_signature_scores(
    bank: PoseSignatureBank,
    scores: np.ndarray,
    *,
    maximum_prototypes: int = 64,
    maximum_modes: int = 16,
    translation_radius_m: float = 1.5,
    rotation_radius_deg: float = 18.0,
    score_temperature: float = 0.04,
) -> tuple[PoseMode, ...]:
    """Cluster retrieved prototypes and interpolate continuous SE(3) modes."""

    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    if value.shape != (bank.poses_w2c.shape[0],):
        raise ValueError("signature score count differs from pose prototypes")
    selected = np.argsort(-value, kind="mergesort")[: max(int(maximum_prototypes), 1)]
    centers = np.stack(
        [camera_center_from_pose_w2c(bank.poses_w2c[row]) for row in selected]
    )
    rotations = np.asarray(bank.poses_w2c[selected, :3, :3], dtype=np.float64)
    unassigned = np.ones(selected.size, dtype=bool)
    modes: list[PoseMode] = []
    for local_seed in range(selected.size):
        if not bool(unassigned[local_seed]):
            continue
        translation = np.linalg.norm(centers - centers[local_seed], axis=1)
        rotation = rotation_angle_deg(rotations, rotations[local_seed])
        members = np.flatnonzero(
            unassigned
            & (translation <= max(float(translation_radius_m), 0.0))
            & (rotation <= max(float(rotation_radius_deg), 0.0))
        )
        if members.size == 0:
            members = np.asarray([local_seed], dtype=np.int64)
        unassigned[members] = False
        member_rows = selected[members]
        logits = value[member_rows]
        weights = np.exp(
            np.clip(
                (logits - float(np.max(logits)))
                / max(float(score_temperature), 1e-6),
                -60.0,
                0.0,
            )
        )
        weights /= max(float(np.sum(weights)), 1e-12)
        center = np.sum(weights[:, None] * centers[members], axis=0)
        rotation_mean = weighted_rotation_mean(rotations[members], weights)
        modes.append(
            PoseMode(
                pose_w2c=pose_w2c_from_center_rotation(center, rotation_mean),
                score=float(np.max(logits) + np.log(max(members.size, 1)) * 0.01),
                prototype_rows=member_rows,
                source="pose_signature_cluster_mean",
            )
        )
    modes.sort(key=lambda item: -item.score)
    # Keep prototype modes as an oracle/coverage diagnostic.  Cluster means
    # are the actual continuous Top-1 estimates and appear first on score ties.
    prototypes = [
        PoseMode(
            pose_w2c=np.asarray(bank.poses_w2c[row], dtype=np.float64),
            score=float(value[row] - 1e-6),
            prototype_rows=np.asarray([row], dtype=np.int64),
            source="pose_signature_prototype",
        )
        for row in selected[: max(int(maximum_modes), 1)]
    ]
    combined = modes[: max(int(maximum_modes), 1)] + prototypes
    combined.sort(key=lambda item: -item.score)
    return tuple(combined[: max(int(maximum_modes), 1)])
