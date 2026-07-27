"""Anonymous maplet appearance-mode pose voting for V6 coarse localization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_pose_proposal import (
    MapletPoseHypothesis,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    QueryMapletGroup,
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


def _bearing_from_normalized_image_xy(
    normalized_xy: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    xy = np.asarray(normalized_xy, dtype=np.float64).reshape(-1, 2)
    pixel = xy * np.asarray(
        [camera.width, camera.height], dtype=np.float64
    )[None] - 0.5
    fx = float(camera.params[0])
    fy = fx
    cx = float(camera.params[1])
    cy = float(camera.params[2])
    bearing = np.stack(
        [
            (pixel[:, 0] - cx) / max(fx, 1e-8),
            (pixel[:, 1] - cy) / max(fy, 1e-8),
            np.ones(pixel.shape[0], dtype=np.float64),
        ],
        axis=1,
    )
    return bearing / np.maximum(
        np.linalg.norm(bearing, axis=1, keepdims=True), 1e-8
    )


def _minimal_bearing_rotation(
    source_bearing: np.ndarray,
    target_bearing: np.ndarray,
) -> np.ndarray:
    """Camera-frame rotations taking each source ray to its query ray."""

    source = np.asarray(source_bearing, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_bearing, dtype=np.float64).reshape(-1, 3)
    if target.shape[0] == 1 and source.shape[0] != 1:
        target = np.broadcast_to(target, source.shape)
    if target.shape != source.shape:
        raise ValueError("source and target bearing counts differ")
    cross = np.cross(source, target)
    cosine = np.clip(np.sum(source * target, axis=1), -1.0, 1.0)
    sine2 = np.sum(cross * cross, axis=1)
    output = np.tile(np.eye(3, dtype=np.float64)[None], (source.shape[0], 1, 1))
    regular = sine2 > 1e-12
    if np.any(regular):
        x, y, z = cross[regular].T
        skew = np.zeros((int(np.sum(regular)), 3, 3), dtype=np.float64)
        skew[:, 0, 1] = -z
        skew[:, 0, 2] = y
        skew[:, 1, 0] = z
        skew[:, 1, 2] = -x
        skew[:, 2, 0] = -y
        skew[:, 2, 1] = x
        output[regular] += skew + np.einsum(
            "nij,njk->nik", skew, skew
        ) * (
            (1.0 - cosine[regular])
            / sine2[regular]
        )[:, None, None]
    opposite = (~regular) & (cosine < 0.0)
    if np.any(opposite):
        # This branch is unreachable for an ordinary camera FOV, but retain a
        # deterministic 180-degree construction for completeness.
        values = source[opposite]
        reference = np.zeros_like(values)
        reference[:, 0] = 1.0
        use_y = np.abs(values[:, 0]) > 0.8
        reference[use_y] = np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(values, reference)
        axis /= np.maximum(
            np.linalg.norm(axis, axis=1, keepdims=True), 1e-8
        )
        output[opposite] = (
            2.0 * axis[:, :, None] * axis[:, None, :]
            - np.eye(3, dtype=np.float64)[None]
        )
    return output


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
    region_xy_mean: np.ndarray
    region_xy_covariance: np.ndarray
    region_support_count: np.ndarray
    component_descriptor_sha256: str
    retrieval_geometry_sha256: str
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
        region_mean = np.asarray(
            self.region_xy_mean, dtype=np.float32
        )
        region_covariance = np.asarray(
            self.region_xy_covariance, dtype=np.float32
        )
        region_support = np.asarray(
            self.region_support_count, dtype=np.int32
        ).reshape(-1)
        if centers.shape != (vote_count, 3):
            raise ValueError("camera_centers must have shape (V,3)")
        if rotations.shape != (vote_count, 3, 3):
            raise ValueError("rotations_w2c must have shape (V,3,3)")
        if any(
            value.shape != (vote_count,)
            for value in (
                weights,
                translation_sigma,
                rotation_sigma,
                region_support,
            )
        ):
            raise ValueError("vote statistics must have shape (V,)")
        if region_mean.shape != (vote_count, 2):
            raise ValueError("region_xy_mean must have shape (V,2)")
        if region_covariance.shape != (vote_count, 2, 2):
            raise ValueError(
                "region_xy_covariance must have shape (V,2,2)"
            )
        if (
            np.any(weights < 0.0)
            or np.any(translation_sigma < 0.0)
            or np.any(rotation_sigma < 0.0)
            or np.any(region_support <= 0)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(rotations))
            or not np.all(np.isfinite(region_mean))
            or not np.all(np.isfinite(region_covariance))
        ):
            raise ValueError("invalid anonymous pose-vote statistics")
        if len(str(self.retrieval_geometry_sha256)) != 64:
            raise ValueError("pose-vote bank requires retrieval geometry lineage")
        normalized = weights.copy()
        for row in range(component_ids.size):
            begin, end = int(offsets[row]), int(offsets[row + 1])
            normalized[begin:end] /= max(
                float(np.sum(normalized[begin:end])), 1e-8
            )
        metadata = dict(self.metadata or {})
        if metadata.get("representation") != (
            "maplet_component_pose_and_region_sufficient_statistics"
        ):
            raise ValueError(
                "anonymous pose votes require region-conditioned "
                "sufficient statistics"
            )
        if not bool(metadata.get("uses_query_region_geometry", False)):
            raise ValueError(
                "anonymous pose votes must condition on query region geometry"
            )
        mapping_trajectories = tuple(
            str(value)
            for value in metadata.get("mapping_trajectory_ids", [])
        )
        if not mapping_trajectories or len(set(mapping_trajectories)) != len(
            mapping_trajectories
        ):
            raise ValueError(
                "pose-vote bank must record unique mapping trajectories"
            )
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
        object.__setattr__(self, "region_xy_mean", region_mean)
        object.__setattr__(
            self, "region_xy_covariance", region_covariance
        )
        object.__setattr__(self, "region_support_count", region_support)
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
            region_xy_mean=self.region_xy_mean,
            region_xy_covariance=self.region_xy_covariance,
            region_support_count=self.region_support_count,
            component_descriptor_sha256=np.asarray(
                self.component_descriptor_sha256
            ),
            retrieval_geometry_sha256=np.asarray(
                self.retrieval_geometry_sha256
            ),
            metadata_json=np.asarray(
                json.dumps(dict(self.metadata or {}), sort_keys=True)
            ),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "AnonymousMapletPoseVoteBank":
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {
                "component_maplet_ids",
                "vote_offsets",
                "camera_centers",
                "rotations_w2c",
                "vote_weights",
                "translation_sigma_m",
                "rotation_sigma_deg",
                "region_xy_mean",
                "region_xy_covariance",
                "region_support_count",
                "component_descriptor_sha256",
                "retrieval_geometry_sha256",
                "metadata_json",
            }
            missing = expected - set(data.files)
            extra = set(data.files) - expected
            if missing or extra:
                raise ValueError(
                    "non-canonical anonymous pose-vote artifact; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            return cls(
                component_maplet_ids=data["component_maplet_ids"],
                vote_offsets=data["vote_offsets"],
                camera_centers=data["camera_centers"],
                rotations_w2c=data["rotations_w2c"],
                vote_weights=data["vote_weights"],
                translation_sigma_m=data["translation_sigma_m"],
                rotation_sigma_deg=data["rotation_sigma_deg"],
                region_xy_mean=data["region_xy_mean"],
                region_xy_covariance=data["region_xy_covariance"],
                region_support_count=data["region_support_count"],
                component_descriptor_sha256=str(
                    data["component_descriptor_sha256"].item()
                ),
                retrieval_geometry_sha256=str(
                    data["retrieval_geometry_sha256"].item()
                ),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def retrieval_descriptor_sha256(bank: SurfaceRetrievalMapletBank) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(bank.maplet_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(bank.descriptor_offsets, dtype=np.int64).tobytes())
    digest.update(np.asarray(bank.descriptors, dtype=np.float32).tobytes())
    return digest.hexdigest()


def retrieval_geometry_sha256(bank: SurfaceRetrievalMapletBank) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(bank.maplet_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(bank.centers, dtype=np.float32).tobytes())
    digest.update(np.asarray(bank.normals, dtype=np.float32).tobytes())
    digest.update(
        np.asarray(bank.tangent_frames, dtype=np.float32).tobytes()
    )
    digest.update(np.asarray(bank.extents, dtype=np.float32).tobytes())
    return digest.hexdigest()


def vote_maplet_poses(
    query_descriptors: np.ndarray,
    retrieval_bank: SurfaceRetrievalMapletBank,
    vote_bank: AnonymousMapletPoseVoteBank,
    *,
    candidate_groups: tuple[QueryMapletGroup, ...] | None = None,
    image_size_wh: tuple[int, int] | None = None,
    camera: ColmapCamera | None = None,
    region_match_probability: np.ndarray | None = None,
    descriptor_temperature: float = 0.08,
    candidates_per_region: int = 16,
    maximum_modes: int = 16,
    translation_kernel_m: float = 0.75,
    rotation_kernel_deg: float = 12.0,
    minimum_region_sigma: float = 0.02,
    region_prior_strength: float = 1.0,
) -> tuple[MapletPoseHypothesis, ...]:
    """Cluster anonymous appearance/region-conditioned votes in SE(3)."""

    query = np.asarray(query_descriptors, dtype=np.float32)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    if region_match_probability is None:
        match_probability = np.ones(query.shape[0], dtype=np.float64)
    else:
        match_probability = np.asarray(
            region_match_probability, dtype=np.float64
        ).reshape(-1)
        if match_probability.shape != (query.shape[0],):
            raise ValueError("region match probability count differs")
        match_probability = np.clip(match_probability, 0.0, 1.0)
    if (
        vote_bank.component_descriptor_sha256
        != retrieval_descriptor_sha256(retrieval_bank)
    ):
        raise ValueError("pose-vote/retrieval descriptor lineage mismatch")
    if (
        vote_bank.retrieval_geometry_sha256
        != retrieval_geometry_sha256(retrieval_bank)
    ):
        raise ValueError("pose-vote/retrieval geometry lineage mismatch")
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
    component_evidence = np.zeros(
        (retrieval_bank.descriptors.shape[0],), dtype=np.float64
    )
    direct_vote_evidence: np.ndarray | None = None
    keep = min(max(int(candidates_per_region), 1), scaled.shape[1])
    if candidate_groups is not None:
        if len(candidate_groups) != query.shape[0]:
            raise ValueError("candidate group/query region count differs")
        if image_size_wh is None:
            raise ValueError(
                "region-conditioned pose voting requires image_size_wh"
            )
        image_size = np.asarray(image_size_wh, dtype=np.float64).reshape(2)
        if np.any(image_size <= 0.0):
            raise ValueError("image_size_wh must be positive")
        direct_vote_evidence = np.zeros(
            vote_bank.camera_centers.shape[0], dtype=np.float64
        )
        direct_vote_rotations = np.asarray(
            vote_bank.rotations_w2c, dtype=np.float64
        ).copy()
        maplet_row_by_id = {
            int(value): int(row)
            for row, value in enumerate(retrieval_bank.maplet_ids.tolist())
        }
        for query_row, group in enumerate(candidate_groups):
            local_rows = []
            local_probability = []
            for maplet_id, maplet_probability in zip(
                group.maplet_ids.tolist(), group.probabilities.tolist()
            ):
                maplet_row = maplet_row_by_id.get(int(maplet_id))
                if maplet_row is None:
                    continue
                begin, end = (
                    int(retrieval_bank.descriptor_offsets[maplet_row]),
                    int(retrieval_bank.descriptor_offsets[maplet_row + 1]),
                )
                conditional = np.exp(
                    scaled[query_row, begin:end]
                    - logsumexp(scaled[query_row, begin:end])
                )
                local_rows.extend(range(begin, end))
                local_probability.extend(
                    (float(maplet_probability) * conditional).tolist()
                )
            if not local_rows:
                continue
            rows = np.asarray(local_rows, dtype=np.int64)
            probability = np.asarray(local_probability, dtype=np.float64)
            retained = min(keep, rows.size)
            selected = np.argpartition(
                -probability, retained - 1
            )[:retained]
            rows = rows[selected]
            probability = probability[selected]
            query_region = (
                np.asarray(group.query_region_xy, dtype=np.float64) + 0.5
            ) / image_size
            query_extent = (
                np.asarray(group.query_region_extent, dtype=np.float64)
                / image_size
            )
            query_variance = (
                query_extent * query_extent / 3.0
                + float(minimum_region_sigma) ** 2
            )
            for component, component_probability in zip(
                rows.tolist(), probability.tolist()
            ):
                vote_begin, vote_end = (
                    int(vote_bank.vote_offsets[component]),
                    int(vote_bank.vote_offsets[component + 1]),
                )
                vote_rows = np.arange(
                    vote_begin, vote_end, dtype=np.int64
                )
                covariance = (
                    np.asarray(
                        vote_bank.region_xy_covariance[vote_rows],
                        dtype=np.float64,
                    )
                    + np.eye(2, dtype=np.float64)[None]
                    * query_variance[None, :]
                )
                residual = (
                    query_region[None]
                    - vote_bank.region_xy_mean[vote_rows]
                )
                try:
                    inverse = np.linalg.inv(covariance)
                except np.linalg.LinAlgError:
                    continue
                mahalanobis2 = np.einsum(
                    "ni,nij,nj->n", residual, inverse, residual
                )
                conditional_compatibility = np.exp(
                    -0.5 * np.clip(mahalanobis2, 0.0, 160.0)
                )
                # Many anonymous pose modes are supported by only one mapping
                # observation.  Treat the learned 2D location as a mixture of
                # an informative Gaussian and an uninformative branch instead
                # of letting a single sample veto an otherwise valid pose.
                support = np.asarray(
                    vote_bank.region_support_count[vote_rows],
                    dtype=np.float64,
                )
                reliability = np.clip(
                    float(region_prior_strength)
                    * support
                    / (support + 4.0),
                    0.0,
                    0.8,
                )
                compatibility = (
                    1.0
                    - reliability
                    + reliability * conditional_compatibility
                )
                local_evidence = (
                    float(component_probability)
                    * vote_bank.vote_weights[vote_rows]
                    * compatibility
                )
                # Overlapping RADIO regions are correlated.  Repeated hits of
                # one component must not multiply the same pose vote without
                # bound.
                improved = (
                    local_evidence
                    > direct_vote_evidence[vote_rows]
                )
                if np.any(improved):
                    selected_votes = vote_rows[improved]
                    if camera is not None:
                        source_bearing = (
                            _bearing_from_normalized_image_xy(
                                vote_bank.region_xy_mean[selected_votes],
                                camera,
                            )
                        )
                        query_bearing = (
                            _bearing_from_normalized_image_xy(
                                query_region[None], camera
                            )
                        )
                        correction = _minimal_bearing_rotation(
                            source_bearing, query_bearing
                        )
                        direct_vote_rotations[selected_votes] = (
                            correction
                            @ vote_bank.rotations_w2c[selected_votes]
                        )
                    direct_vote_evidence[selected_votes] = (
                        local_evidence[improved]
                    )
    else:
        posterior = np.exp(
            scaled - logsumexp(scaled, axis=1, keepdims=True)
        )
        candidate_rows = np.argpartition(
            -scaled, kth=keep - 1, axis=1
        )[:, :keep]
        for query_row in range(query.shape[0]):
            rows = candidate_rows[query_row]
            # Max aggregation prevents a repetitive facade occupying many
            # query regions from multiplying the same vote without bound.
            component_evidence[rows] = np.maximum(
                component_evidence[rows],
                match_probability[query_row] * posterior[query_row, rows],
            )
    vote_component_ids = np.repeat(
        np.arange(retrieval_bank.descriptors.shape[0], dtype=np.int64),
        np.diff(vote_bank.vote_offsets),
    )
    if direct_vote_evidence is not None:
        vote_rows_array = np.flatnonzero(direct_vote_evidence > 0.0)
        evidence = direct_vote_evidence[vote_rows_array]
        components = vote_component_ids[vote_rows_array]
        rotations = direct_vote_rotations[vote_rows_array]
    else:
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
        vote_rows_array = np.asarray(vote_rows, dtype=np.int64)
        evidence = np.asarray(vote_evidence, dtype=np.float64)
        components = np.asarray(vote_components, dtype=np.int64)
        rotations = vote_bank.rotations_w2c[vote_rows_array]
    if vote_rows_array.size == 0:
        return tuple()
    centers = vote_bank.camera_centers[vote_rows_array]
    translation_sigma = np.asarray(
        vote_bank.translation_sigma_m[vote_rows_array],
        dtype=np.float64,
    )
    rotation_sigma = np.asarray(
        vote_bank.rotation_sigma_deg[vote_rows_array],
        dtype=np.float64,
    )
    # Convolving two vote uncertainties with the clustering kernel gives a
    # symmetric pairwise bandwidth.  The previous column-wise division was
    # asymmetric and systematically favoured high-variance vote modes.
    translation_scale = np.sqrt(
        float(translation_kernel_m) ** 2
        + translation_sigma[:, None] ** 2
        + translation_sigma[None, :] ** 2
    )
    rotation_scale = np.sqrt(
        float(rotation_kernel_deg) ** 2
        + rotation_sigma[:, None] ** 2
        + rotation_sigma[None, :] ** 2
    )
    translation_scale = np.maximum(translation_scale, 1e-3)
    rotation_scale = np.maximum(rotation_scale, 1e-3)
    translation_distance = np.linalg.norm(
        centers[:, None] - centers[None], axis=2
    )
    rotation_distance = _rotation_angle_deg(
        rotations[:, None], rotations[None]
    )
    kernel = np.exp(
        -0.5
        * (
            (translation_distance / translation_scale) ** 2
            + (rotation_distance / rotation_scale) ** 2
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
                source="anonymous_view_mode_vote",
            )
        )
        if len(results) >= int(maximum_modes):
            break
    return tuple(results)
