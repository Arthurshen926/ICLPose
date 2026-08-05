"""Query-independent virtual camera lattice and maplet/cell inverted evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.spatial.transform import Rotation

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
)
from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    PhysicalMapletGraph,
)
from feature_extract.vfm.localization_v81.region_surface_graph import (
    RegionEvidenceGraph,
)


LATTICE_ARTIFACT = "v81_query_independent_virtual_pose_lattice"


@dataclass(frozen=True)
class VirtualPoseLattice:
    """Cartesian camera-position/orientation lattice with sparse maplet cells."""

    positions: np.ndarray
    rotations_w2c: np.ndarray
    key_offsets: np.ndarray
    visibility_keys: np.ndarray
    maplet_ids: np.ndarray
    cell_grid_size: int
    physical_graph_sha256: str
    pose_vote_bank_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float32)
        rotations = np.asarray(self.rotations_w2c, dtype=np.float32)
        offsets = np.asarray(self.key_offsets, dtype=np.int64).reshape(-1)
        keys = np.asarray(self.visibility_keys, dtype=np.uint16).reshape(-1)
        ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        pose_count = int(positions.shape[0] * rotations.shape[0])
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("lattice positions must have shape (P,3)")
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("lattice rotations must have shape (R,3,3)")
        if offsets.shape != (pose_count + 1,) or offsets[0] != 0:
            raise ValueError("lattice visibility offsets differ")
        if offsets[-1] != keys.size or np.any(np.diff(offsets) < 0):
            raise ValueError("lattice visibility CSR is invalid")
        maximum_key = int(ids.size * int(self.cell_grid_size) ** 2)
        if keys.size and int(np.max(keys)) >= maximum_key:
            raise ValueError("lattice visibility key is out of range")
        if len(str(self.physical_graph_sha256)) != 64 or len(str(self.pose_vote_bank_sha256)) != 64:
            raise ValueError("lattice requires fail-closed geometry/proposal lineage")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", LATTICE_ARTIFACT) != LATTICE_ARTIFACT:
            raise ValueError("not a V8.1 virtual-pose lattice")
        for key in (
            "stores_mapping_rgb",
            "stores_mapping_image_ids",
            "stores_mapping_image_paths",
            "stores_downstream_embeddings",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_point_correspondence_pnp",
        ):
            if bool(metadata.get(key, False)):
                raise ValueError(f"virtual lattice violates runtime contract: {key}")
        metadata.update(
            artifact_type=LATTICE_ARTIFACT,
            stores_mapping_rgb=False,
            stores_mapping_image_ids=False,
            stores_mapping_image_paths=False,
            stores_downstream_embeddings=False,
            uses_sfm_points=False,
            uses_sfm_tracks=False,
            uses_point_correspondence_pnp=False,
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rotations_w2c", rotations)
        object.__setattr__(self, "key_offsets", offsets)
        object.__setattr__(self, "visibility_keys", keys)
        object.__setattr__(self, "maplet_ids", ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def pose_count(self) -> int:
        return int(self.positions.shape[0] * self.rotations_w2c.shape[0])

    def poses_w2c(self, indices: np.ndarray) -> np.ndarray:
        index = np.asarray(indices, dtype=np.int64).reshape(-1)
        rotations = int(self.rotations_w2c.shape[0])
        position_rows = index // rotations
        rotation_rows = index % rotations
        pose = np.tile(np.eye(4, dtype=np.float64)[None], (index.size, 1, 1))
        pose[:, :3, :3] = self.rotations_w2c[rotation_rows]
        pose[:, :3, 3] = -np.einsum(
            "nij,nj->ni", pose[:, :3, :3], self.positions[position_rows]
        )
        return pose

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            positions=self.positions,
            rotations_w2c=self.rotations_w2c,
            key_offsets=self.key_offsets,
            visibility_keys=self.visibility_keys,
            maplet_ids=self.maplet_ids,
            cell_grid_size=np.asarray(self.cell_grid_size, dtype=np.int32),
            physical_graph_sha256=np.asarray(self.physical_graph_sha256),
            pose_vote_bank_sha256=np.asarray(self.pose_vote_bank_sha256),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "VirtualPoseLattice":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                positions=data["positions"],
                rotations_w2c=data["rotations_w2c"],
                key_offsets=data["key_offsets"],
                visibility_keys=data["visibility_keys"],
                maplet_ids=data["maplet_ids"],
                cell_grid_size=int(data["cell_grid_size"].item()),
                physical_graph_sha256=str(data["physical_graph_sha256"].item()),
                pose_vote_bank_sha256=str(data["pose_vote_bank_sha256"].item()),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_virtual_pose_lattice(
    votes: AnonymousMapletPoseVoteBank,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
    *,
    physical_graph_path: Path,
    pose_vote_bank_path: Path,
    position_spacing_m: float = 1.0,
    expansion_radius_m: float = 3.0,
    rotation_prototypes: int = 128,
    cell_grid_size: int = 4,
    batch_size: int = 4096,
    device: str = "cuda",
) -> VirtualPoseLattice:
    """Precompute a coarse pose lattice from anonymous mapping statistics."""

    spacing = float(position_spacing_m)
    radius = float(expansion_radius_m)
    if spacing <= 0.0 or radius < 0.0:
        raise ValueError("invalid virtual-position spacing/radius")
    base = np.unique(
        np.rint(np.asarray(votes.camera_centers) / spacing).astype(np.int32), axis=0
    ).astype(np.float64) * spacing
    offset_values = np.arange(-radius, radius + 0.5 * spacing, spacing)
    offsets = np.asarray(
        [(x, 0.0, z) for x in offset_values for z in offset_values],
        dtype=np.float64,
    )
    positions = np.unique(
        np.rint((base[:, None] + offsets[None]).reshape(-1, 3) / spacing).astype(np.int32),
        axis=0,
    ).astype(np.float64) * spacing
    quaternion = Rotation.from_matrix(votes.rotations_w2c).as_quat()
    quaternion = np.where(quaternion[:, 3:4] < 0.0, -quaternion, quaternion)
    centroid, _label = kmeans2(
        quaternion,
        min(int(rotation_prototypes), quaternion.shape[0]),
        iter=50,
        minit="++",
        seed=77,
    )
    centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-8)
    rotations = Rotation.from_quat(centroid).as_matrix()
    rotation_count = int(rotations.shape[0])
    pose_count = int(positions.shape[0] * rotation_count)
    world = torch.as_tensor(physical.centers, dtype=torch.float32, device=device)
    normal = torch.as_tensor(physical.normals, dtype=torch.float32, device=device)
    position_tensor = torch.as_tensor(positions, dtype=torch.float32, device=device)
    rotation_tensor = torch.as_tensor(rotations, dtype=torch.float32, device=device)
    f, cx, cy, radial_k = [float(value) for value in camera.params[:4]]
    grid = int(cell_grid_size)
    offsets_csr = np.zeros((pose_count + 1,), dtype=np.int64)
    key_chunks: list[np.ndarray] = []
    running = 0
    for start in range(0, pose_count, max(int(batch_size), 1)):
        stop = min(start + max(int(batch_size), 1), pose_count)
        index = torch.arange(start, stop, dtype=torch.long, device=device)
        center = position_tensor[index // rotation_count]
        rotation = rotation_tensor[index % rotation_count]
        delta = world[None] - center[:, None]
        camera_xyz = torch.einsum("bij,bnj->bni", rotation, delta)
        depth = camera_xyz[:, :, 2]
        normalized = camera_xyz[:, :, :2] / torch.where(
            torch.abs(depth[:, :, None]) >= 1e-6,
            depth[:, :, None],
            torch.full_like(depth[:, :, None], 1e-6),
        )
        radial = 1.0 + radial_k * torch.sum(normalized ** 2, dim=2)
        pixel_x = f * normalized[:, :, 0] * radial + cx
        pixel_y = f * normalized[:, :, 1] * radial + cy
        camera_normal = torch.einsum("bij,nj->bni", rotation, normal)
        view = -camera_xyz / torch.clamp(torch.linalg.norm(camera_xyz, dim=2, keepdim=True), min=1e-8)
        incidence = torch.abs(torch.sum(camera_normal * view, dim=2))
        visible = (
            (depth > 0.1)
            & (pixel_x >= 0.0) & (pixel_x < camera.width)
            & (pixel_y >= 0.0) & (pixel_y < camera.height)
            & (incidence >= 0.10)
        )
        cell_x = torch.clamp((pixel_x * grid / camera.width).long(), 0, grid - 1)
        cell_y = torch.clamp((pixel_y * grid / camera.height).long(), 0, grid - 1)
        maplet_row = torch.arange(world.shape[0], device=device)[None]
        keys = maplet_row * (grid * grid) + cell_y * grid + cell_x
        counts = visible.sum(dim=1).cpu().numpy().astype(np.int64)
        selected = keys[visible].cpu().numpy().astype(np.uint16, copy=False)
        key_chunks.append(selected)
        offsets_csr[start + 1 : stop + 1] = running + np.cumsum(counts)
        running += int(selected.size)
    visibility_keys = (
        np.concatenate(key_chunks) if key_chunks else np.zeros((0,), dtype=np.uint16)
    )
    return VirtualPoseLattice(
        positions=positions,
        rotations_w2c=rotations,
        key_offsets=offsets_csr,
        visibility_keys=visibility_keys,
        maplet_ids=physical.maplet_ids,
        cell_grid_size=grid,
        physical_graph_sha256=_sha256(physical_graph_path),
        pose_vote_bank_sha256=_sha256(pose_vote_bank_path),
        metadata={
            "position_spacing_m": spacing,
            "expansion_radius_m": radius,
            "rotation_prototype_count": rotation_count,
            "position_count": int(positions.shape[0]),
            "pose_count": pose_count,
            "visibility_key_count": int(visibility_keys.size),
            "mapping_trajectory_ids": list((votes.metadata or {}).get("mapping_trajectory_ids", [])),
            "camera_model_id": int(camera.model_id),
            "camera_width": int(camera.width),
            "camera_height": int(camera.height),
            "camera_params": [float(value) for value in camera.params],
        },
    )


def rank_virtual_pose_lattice(
    lattice: VirtualPoseLattice,
    query: RegionEvidenceGraph,
    *,
    maximum_poses: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank lattice cells using posterior maplet identity and coarse image cell."""

    grid = int(lattice.cell_grid_size)
    key_weight = np.zeros((lattice.maplet_ids.size * grid * grid,), dtype=np.float32)
    for group in range(query.group_count):
        centre = np.clip(np.floor(query.xy[group] * grid).astype(int), 0, grid - 1)
        for local, row in enumerate(query.candidate_rows[group].tolist()):
            if row < 0:
                continue
            probability = float(query.candidate_probabilities[group, local])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = int(centre[0] + dx), int(centre[1] + dy)
                    if not (0 <= x < grid and 0 <= y < grid):
                        continue
                    spatial_weight = 1.0 if dx == 0 and dy == 0 else (0.55 if dx == 0 or dy == 0 else 0.30)
                    key = int(row) * grid * grid + y * grid + x
                    # Multiple overlapping supports are correlated evidence,
                    # not independent votes.  Retain the strongest observation
                    # for one physical maplet/cell key.
                    key_weight[key] = max(
                        float(key_weight[key]), probability * spatial_weight
                    )
    # Repeated facade keys appear in a large fraction of the virtual lattice.
    # Their inverse document frequency is the geometric analogue of burstiness
    # suppression in image retrieval and is computed entirely from the map.
    document_frequency = np.bincount(
        lattice.visibility_keys.astype(np.int64),
        minlength=key_weight.size,
    ).astype(np.float64)
    inverse_frequency = np.log1p(
        lattice.pose_count / np.maximum(document_frequency, 1.0)
    )
    value = key_weight[lattice.visibility_keys] * inverse_frequency[
        lattice.visibility_keys
    ]
    cumulative = np.r_[0.0, np.cumsum(value, dtype=np.float64)]
    score = cumulative[lattice.key_offsets[1:]] - cumulative[lattice.key_offsets[:-1]]
    take = min(max(int(maximum_poses), 1), score.size)
    selected = np.argpartition(score, -take)[-take:]
    selected = selected[np.argsort(-score[selected], kind="stable")]
    return selected.astype(np.int64), score[selected].astype(np.float32)


def expand_virtual_pose_seeds(
    seed_poses_w2c: np.ndarray,
    *,
    translation_step_m: float = 0.5,
    rotation_step_deg: float = 3.0,
) -> np.ndarray:
    """One query-independent coarse-to-fine lattice subdivision."""

    seed = np.asarray(seed_poses_w2c, dtype=np.float64).reshape(-1, 4, 4)
    translations = np.asarray(
        [(x, 0.0, z) for x in (-translation_step_m, 0.0, translation_step_m) for z in (-translation_step_m, 0.0, translation_step_m)]
    )
    angle = np.deg2rad(float(rotation_step_deg))
    rotation_delta = [np.eye(3)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            vector = np.zeros((3,))
            vector[axis] = sign * angle
            rotation_delta.append(Rotation.from_rotvec(vector).as_matrix())
    result = []
    for pose in seed:
        center = -pose[:3, :3].T @ pose[:3, 3]
        for translation in translations:
            candidate_center = center + translation
            for delta in rotation_delta:
                rotation = delta @ pose[:3, :3]
                candidate = np.eye(4)
                candidate[:3, :3] = rotation
                candidate[:3, 3] = -rotation @ candidate_center
                result.append(candidate)
    array = np.asarray(result)
    flattened = np.round(array[:, :3].reshape(array.shape[0], -1), decimals=5)
    _unique, indices = np.unique(flattened, axis=0, return_index=True)
    return array[np.sort(indices)]


def rank_pose_candidates_by_region_centres(
    poses_w2c: np.ndarray,
    query: RegionEvidenceGraph,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
    *,
    maximum_poses: int = 64,
    batch_size: int = 1024,
    device: str = "cuda",
    diversity_translation_m: float = 0.75,
    diversity_rotation_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-rank a subdivided lattice by marginalized regional centre density."""

    pose = np.asarray(poses_w2c, dtype=np.float32)
    used_rows = np.unique(query.candidate_rows[query.candidate_rows >= 0])
    local_by_row = {int(row): index for index, row in enumerate(used_rows.tolist())}
    local_candidate = np.vectorize(lambda row: local_by_row.get(int(row), 0))(
        np.maximum(query.candidate_rows, 0)
    )
    valid = query.candidate_rows >= 0
    world = torch.as_tensor(physical.centers[used_rows], dtype=torch.float32, device=device)
    group_xy = torch.as_tensor(query.xy, dtype=torch.float32, device=device)
    candidate = torch.as_tensor(local_candidate, dtype=torch.long, device=device)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
    probability = torch.as_tensor(query.candidate_probabilities, dtype=torch.float32, device=device)
    null = torch.as_tensor(query.null_probability, dtype=torch.float32, device=device)
    structured_edges = np.flatnonzero(query.edge_features[:, 5] >= 1.0)
    if structured_edges.size > 256:
        positions = np.linspace(0, structured_edges.size - 1, 256, dtype=np.int64)
        structured_edges = structured_edges[positions]
    edge_source = torch.as_tensor(
        query.edge_source[structured_edges], dtype=torch.long, device=device
    )
    edge_target = torch.as_tensor(
        query.edge_target[structured_edges], dtype=torch.long, device=device
    )
    edge_delta = torch.as_tensor(
        query.edge_features[structured_edges, :2], dtype=torch.float32, device=device
    )
    relation_candidates = min(4, int(query.candidate_rows.shape[1]))
    relation_probability = probability[:, :relation_candidates]
    relation_rows = torch.as_tensor(
        query.candidate_rows[:, :relation_candidates], dtype=torch.long, device=device
    )
    scores = []
    f, cx, cy, radial_k = [float(value) for value in camera.params[:4]]
    for start in range(0, pose.shape[0], max(int(batch_size), 1)):
        value = torch.as_tensor(pose[start : start + batch_size], device=device)
        camera_xyz = torch.einsum("bij,nj->bni", value[:, :3, :3], world) + value[:, None, :3, 3]
        depth = camera_xyz[:, :, 2]
        normalised = camera_xyz[:, :, :2] / torch.clamp(depth[:, :, None], min=1e-6)
        radial = 1.0 + radial_k * torch.sum(normalised ** 2, dim=2)
        projected = torch.stack(
            ((f * normalised[:, :, 0] * radial + cx) / camera.width,
             (f * normalised[:, :, 1] * radial + cy) / camera.height),
            dim=2,
        )
        selected = projected[:, candidate]
        residual = torch.sum(((selected - group_xy[None, :, None]) / 0.10) ** 2, dim=3)
        inside = (
            (depth[:, candidate] > 0.1)
            & (selected[:, :, :, 0] >= 0.0) & (selected[:, :, :, 0] <= 1.0)
            & (selected[:, :, :, 1] >= 0.0) & (selected[:, :, :, 1] <= 1.0)
            & valid_tensor[None]
        )
        ratio = torch.where(inside, 16.0 * torch.exp(torch.clamp(-0.5 * residual, min=-40.0)), 0.0)
        evidence = null[None] + torch.sum(probability[None] * ratio, dim=2)
        unary_score = torch.mean(
            torch.log(torch.clamp(evidence, min=1e-8)), dim=1
        )
        if edge_source.numel():
            source_xy = selected[:, edge_source, :relation_candidates]
            target_xy = selected[:, edge_target, :relation_candidates]
            predicted_delta = target_xy[:, :, None] - source_xy[:, :, :, None]
            relation_residual = torch.sum(
                ((predicted_delta - edge_delta[None, :, None, None]) / 0.14) ** 2,
                dim=4,
            )
            ratio = 16.0 * torch.exp(torch.clamp(-0.5 * relation_residual, min=-40.0))
            valid_relation = (
                relation_rows[edge_source, :, None] >= 0
            ) & (
                relation_rows[edge_target, None, :] >= 0
            )
            distinct = (
                relation_rows[edge_source, :, None]
                != relation_rows[edge_target, None, :]
            )
            ratio = torch.where(
                valid_relation[None],
                torch.where(distinct[None], ratio, torch.ones_like(ratio)),
                torch.zeros_like(ratio),
            )
            pair_probability = (
                relation_probability[edge_source, :, None]
                * relation_probability[edge_target, None, :]
            )
            unmatched = torch.clamp(
                1.0 - torch.sum(pair_probability, dim=(1, 2)), min=0.0
            )
            relation_evidence = unmatched[None] + torch.sum(
                pair_probability[None] * ratio, dim=(2, 3)
            )
            graph_score = torch.mean(
                torch.log(torch.clamp(relation_evidence, min=1e-8)), dim=1
            )
            total_score = unary_score + 0.5 * graph_score
        else:
            total_score = unary_score
        scores.append(total_score.cpu().numpy())
    score = np.concatenate(scores)
    take = min(max(int(maximum_poses), 1), score.size)
    pool_size = min(max(take * 32, take), score.size)
    pool = np.argpartition(score, -pool_size)[-pool_size:]
    pool = pool[np.argsort(-score[pool], kind="stable")]
    candidate_pose = pose[pool]
    candidate_center = -np.einsum(
        "nji,nj->ni", candidate_pose[:, :3, :3], candidate_pose[:, :3, 3]
    )
    retained: list[int] = []
    for local in range(pool.size):
        if retained:
            old = np.asarray(retained, dtype=np.int64)
            translation = np.linalg.norm(
                candidate_center[old] - candidate_center[local], axis=1
            )
            relative = np.einsum(
                "nij,kj->nki",
                candidate_pose[old, :3, :3],
                candidate_pose[local, :3, :3],
            )
            cosine = np.clip(
                (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
            rotation = np.degrees(np.arccos(cosine))
            if np.any(
                (translation < float(diversity_translation_m))
                & (rotation < float(diversity_rotation_deg))
            ):
                continue
        retained.append(local)
        if len(retained) >= take:
            break
    if len(retained) < take:
        used = set(retained)
        retained.extend(
            index for index in range(pool.size)
            if index not in used
        )
    selected = pool[np.asarray(retained[:take], dtype=np.int64)]
    return selected.astype(np.int64), score[selected].astype(np.float32)
