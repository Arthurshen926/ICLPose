"""Fit a compact cross-fitted view-conditioned residual over a canonical field."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.sparse_vfm_pose_likelihood import (
    _camera_parameters,
)
from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    PREDICTOR_COUNT,
    SCHEMA,
    ViewConditionedPrimitiveField,
)
from feature_extract.tools.vfm.build_goal_maplet_canonical_field_from_contributors import (
    _pixel_layout,
)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class ViewObservation:
    field_indices: np.ndarray
    descriptors: np.ndarray
    weights: np.ndarray
    local_directions: np.ndarray
    log_projected_scales: np.ndarray


def _eligible_contributors(
    root: Path,
    excluded: set[str],
) -> list[tuple[Path, dict[str, object]]]:
    result: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(Path(root).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if str(metadata["trajectory_id"]) in excluded:
            continue
        if not bool(metadata.get("uses_declared_clean_2dgs_for_occlusion", False)):
            raise ValueError(f"contributor cache is not declared-clean 2DGS: {path}")
        result.append((path, metadata))
    return result


def _observation(
    path: Path,
    metadata: dict[str, object],
    physical: GoalMapletPhysicalMap,
    dense_physical_lookup: np.ndarray,
    dense_field_lookup: np.ndarray,
    mapper,
) -> ViewObservation:
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["topk_ids"], dtype=np.int64)
        weights = np.asarray(data["topk_weights"], dtype=np.float32)
        pose_w2c = np.asarray(data["pose_w2c"], dtype=np.float64)
        camera = ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )
    with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
        raw = np.asarray(data["radio_final"], dtype=np.float32)
    mapped = mapper.project(raw).measurement_context
    rows, mass, contributing_pixel, starts, token_x, token_y = _pixel_layout(
        ids, weights, (int(raw.shape[1]), int(raw.shape[2])), dense_physical_lookup,
    )
    unique_rows = rows[starts]
    field_indices = dense_field_lookup[unique_rows]
    keep = field_indices >= 0
    if not np.any(keep):
        raise ValueError(f"contributor has no primitive aligned to canonical field: {path}")
    pixel_descriptor = mapped[:, token_y, token_x].T.astype(np.float32, copy=False)
    descriptor = pixel_descriptor[contributing_pixel]
    view_mass = np.add.reduceat(mass, starts)
    view_sum = np.add.reduceat(mass[:, None] * descriptor, starts, axis=0)
    view_descriptor = _normalize_rows(
        view_sum / np.maximum(view_mass[:, None], 1e-8),
    )
    unique_rows = unique_rows[keep]
    field_indices = field_indices[keep]
    view_descriptor = view_descriptor[keep]
    view_mass = view_mass[keep]

    rotation, translation = pose_w2c[:3, :3], pose_w2c[:3, 3]
    camera_center = -rotation.T @ translation
    view_world = camera_center[None] - physical.primitive_centers[unique_rows]
    view_world = _normalize_rows(view_world)
    tangent1 = _normalize_rows(physical.primitive_tangent1[unique_rows])
    tangent2 = _normalize_rows(physical.primitive_tangent2[unique_rows])
    normal = _normalize_rows(physical.primitive_normals[unique_rows])
    local_direction = np.stack((
        np.sum(view_world * tangent1, axis=1),
        np.sum(view_world * tangent2, axis=1),
        np.sum(view_world * normal, axis=1),
    ), axis=1)
    local_direction = _normalize_rows(local_direction)
    camera_xyz = physical.primitive_centers[unique_rows] @ rotation.T + translation[None]
    depth = camera_xyz[:, 2]
    fx, fy, _cx, _cy, _radial = _camera_parameters(camera)
    focal = float(np.sqrt(max(float(fx) * float(fy), 1e-8)))
    radius = np.sqrt(np.maximum(
        physical.primitive_scale1[unique_rows] * physical.primitive_scale2[unique_rows],
        1e-12,
    ))
    projected_scale = focal * radius / np.maximum(depth, 1e-4)
    log_projected_scale = np.log(np.maximum(projected_scale, 1e-6)).astype(np.float32)
    observation_weight = np.clip(np.sqrt(view_mass), 0.25, 4.0).astype(np.float32)
    return ViewObservation(
        field_indices=field_indices.astype(np.int64, copy=False),
        descriptors=view_descriptor.astype(np.float32, copy=False),
        weights=observation_weight,
        local_directions=local_direction.astype(np.float32, copy=False),
        log_projected_scales=log_projected_scale,
    )


def _fit_basis(
    residual_samples: np.ndarray,
    sample_weights: np.ndarray,
    *,
    rank: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    feature_dim = int(residual_samples.shape[1])
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    covariance = torch.zeros(
        (feature_dim, feature_dim), dtype=torch.float32, device=torch_device,
    )
    with torch.no_grad():
        for start in range(0, residual_samples.shape[0], 32_768):
            stop = min(start + 32_768, residual_samples.shape[0])
            value = torch.as_tensor(
                residual_samples[start:stop], dtype=torch.float32, device=torch_device,
            )
            weight = torch.as_tensor(
                sample_weights[start:stop], dtype=torch.float32, device=torch_device,
            )
            value = value * torch.sqrt(weight[:, None] / torch.clamp(
                torch.mean(weight), min=1e-8,
            ))
            covariance.addmm_(value.T, value)
        eigenvalue, eigenvector = torch.linalg.eigh(covariance)
    eigenvalue_np = eigenvalue.cpu().numpy().astype(np.float64)
    basis = eigenvector[:, -int(rank) :].T.cpu().numpy().astype(np.float32)
    # Eigenvector signs are arbitrary.  Canonicalizing them makes independent
    # fold builds byte-stable when the numeric backend returns the same axes.
    for row in range(basis.shape[0]):
        pivot = int(np.argmax(np.abs(basis[row])))
        if basis[row, pivot] < 0.0:
            basis[row] *= -1.0
    selected = eigenvalue_np[-int(rank) :][::-1]
    explained = float(np.sum(selected) / max(float(np.sum(eigenvalue_np)), 1e-12))
    return basis, selected, explained


def _batched_ridge_solve(
    normal_matrix: np.ndarray,
    normal_rhs: np.ndarray,
    weight_sum: np.ndarray,
    active: np.ndarray,
    *,
    intercept_ridge: float,
    slope_ridge: float,
    device: str,
) -> np.ndarray:
    coefficient = np.zeros(normal_rhs.shape, dtype=np.float32)
    active_rows = np.flatnonzero(active)
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    ridge = np.diag(np.asarray(
        [float(intercept_ridge)] + [float(slope_ridge)] * (PREDICTOR_COUNT - 1),
        dtype=np.float32,
    ))
    with torch.no_grad():
        for start in range(0, active_rows.size, 65_536):
            rows = active_rows[start : start + 65_536]
            denominator = np.maximum(weight_sum[rows], 1e-8)
            lhs = normal_matrix[rows] / denominator[:, None, None] + ridge[None]
            rhs = normal_rhs[rows] / denominator[:, None, None]
            solution = torch.linalg.solve(
                torch.as_tensor(lhs, dtype=torch.float32, device=torch_device),
                torch.as_tensor(rhs, dtype=torch.float32, device=torch_device),
            )
            coefficient[rows] = solution.cpu().numpy()
    return coefficient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--exclude_trajectories", nargs="*", default=[])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--minimum_views", type=int, default=3)
    parser.add_argument("--maximum_basis_samples", type=int, default=262_144)
    parser.add_argument("--intercept_ridge", type=float, default=0.02)
    parser.add_argument("--slope_ridge", type=float, default=0.25)
    parser.add_argument("--direction_cosine_margin", type=float, default=0.05)
    parser.add_argument("--log_scale_margin", type=float, default=0.25)
    parser.add_argument("--random_seed", type=int, default=194917)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_field), Path(args.summary_json)
    if not bool(args.force) and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite view-conditioned field")
    if int(args.rank) <= 0 or int(args.minimum_views) <= 0:
        raise ValueError("rank and minimum_views must be positive")
    if int(args.maximum_basis_samples) <= 0:
        raise ValueError("maximum_basis_samples must be positive")

    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    canonical = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    if canonical.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical and physical map lineage differ")
    if int(args.rank) > canonical.feature_dim:
        raise ValueError("view-conditioned rank exceeds canonical feature dimension")
    mapper_hash = file_sha256(Path(args.surface_mapper))
    declared_mapper_hash = str(canonical.metadata.get("surface_mapper_file_sha256", ""))
    if declared_mapper_hash and mapper_hash != declared_mapper_hash:
        raise ValueError("surface mapper differs from canonical field builder")
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    eligible = _eligible_contributors(
        Path(args.contributors), set(str(value) for value in args.exclude_trajectories),
    )
    if not eligible:
        raise ValueError("no eligible contributor views")
    trajectories = sorted({str(metadata["trajectory_id"]) for _, metadata in eligible})
    canonical_trajectories = sorted(str(value) for value in canonical.metadata.get(
        "mapping_trajectory_ids", [],
    ))
    if canonical_trajectories and trajectories != canonical_trajectories:
        raise ValueError(
            "eligible contributor trajectories differ from canonical cross-fit fold: "
            f"eligible={trajectories} canonical={canonical_trajectories}"
        )
    declared_images = int(canonical.metadata.get("mapping_image_count", len(eligible)))
    if declared_images != len(eligible):
        raise ValueError(
            "eligible contributor image count differs from canonical cross-fit fold: "
            f"eligible={len(eligible)} canonical={declared_images}"
        )

    dense_physical_lookup = np.full(
        (int(np.max(physical.primitive_ids)) + 1,), -1, dtype=np.int64,
    )
    dense_physical_lookup[physical.primitive_ids] = np.arange(
        physical.primitive_ids.size, dtype=np.int64,
    )
    dense_field_lookup = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    dense_field_lookup[canonical.primitive_rows] = np.arange(
        canonical.primitive_rows.size, dtype=np.int64,
    )
    primitive_count = canonical.primitive_rows.size
    observation_count = np.zeros((primitive_count,), dtype=np.int32)
    direction_sum = np.zeros((primitive_count, 3), dtype=np.float32)
    log_scale_sum = np.zeros((primitive_count,), dtype=np.float32)
    minimum_log_scale = np.full((primitive_count,), np.inf, dtype=np.float32)
    maximum_log_scale = np.full((primitive_count,), -np.inf, dtype=np.float32)

    rng = np.random.default_rng(int(args.random_seed))
    per_view_sample = max(
        int(np.ceil(int(args.maximum_basis_samples) / len(eligible))), 1,
    )
    sampled_residuals: list[np.ndarray] = []
    sampled_weights: list[np.ndarray] = []
    sampled_indices: list[np.ndarray] = []
    sampled_cosines: list[np.ndarray] = []
    sampled_directions: list[np.ndarray] = []
    sampled_log_scales: list[np.ndarray] = []
    total_observation_count = 0
    for view_index, (path, metadata) in enumerate(eligible, start=1):
        value = _observation(
            path, metadata, physical, dense_physical_lookup, dense_field_lookup, mapper,
        )
        rows = value.field_indices
        observation_count[rows] += 1
        direction_sum[rows] += value.local_directions
        log_scale_sum[rows] += value.log_projected_scales
        minimum_log_scale[rows] = np.minimum(
            minimum_log_scale[rows], value.log_projected_scales,
        )
        maximum_log_scale[rows] = np.maximum(
            maximum_log_scale[rows], value.log_projected_scales,
        )
        canonical_code = canonical.codes[rows]
        cosine = np.sum(value.descriptors * canonical_code, axis=1)
        tangent_residual = value.descriptors - cosine[:, None] * canonical_code
        if rows.size > per_view_sample:
            selected = np.sort(rng.choice(rows.size, size=per_view_sample, replace=False))
        else:
            selected = np.arange(rows.size, dtype=np.int64)
        sampled_residuals.append(tangent_residual[selected].astype(np.float32))
        sampled_weights.append(value.weights[selected].astype(np.float32))
        sampled_indices.append(rows[selected].astype(np.int64))
        sampled_cosines.append(cosine[selected].astype(np.float32))
        sampled_directions.append(value.local_directions[selected].astype(np.float32))
        sampled_log_scales.append(value.log_projected_scales[selected].astype(np.float32))
        total_observation_count += int(rows.size)
        print(json.dumps({
            "pass": "basis_and_chart",
            "view": int(view_index),
            "view_count": int(len(eligible)),
            "image_id": str(metadata["image_id"]),
            "observation_count": int(rows.size),
        }), flush=True)
    observed = observation_count > 0
    direction_center = np.zeros_like(direction_sum)
    direction_center[observed] = (
        direction_sum[observed] / observation_count[observed, None]
    )
    direction_concentration = np.linalg.norm(direction_center, axis=1).astype(np.float32)
    mean_direction = np.zeros_like(direction_center)
    mean_direction[:, 2] = 1.0
    stable_direction = observed & (direction_concentration > 1e-6)
    mean_direction[stable_direction] = (
        direction_center[stable_direction]
        / direction_concentration[stable_direction, None]
    )
    mean_log_scale = np.zeros((primitive_count,), dtype=np.float32)
    mean_log_scale[observed] = (
        log_scale_sum[observed] / observation_count[observed]
    )
    minimum_log_scale[~observed] = 0.0
    maximum_log_scale[~observed] = 0.0

    sample_residual = np.concatenate(sampled_residuals, axis=0)
    sample_weight = np.concatenate(sampled_weights, axis=0)
    sample_index = np.concatenate(sampled_indices, axis=0)
    sample_cosine = np.concatenate(sampled_cosines, axis=0)
    sample_direction = np.concatenate(sampled_directions, axis=0)
    sample_log_scale = np.concatenate(sampled_log_scales, axis=0)
    if sample_residual.shape[0] > int(args.maximum_basis_samples):
        selected = np.sort(rng.choice(
            sample_residual.shape[0], size=int(args.maximum_basis_samples), replace=False,
        ))
        sample_residual = sample_residual[selected]
        sample_weight = sample_weight[selected]
        sample_index = sample_index[selected]
        sample_cosine = sample_cosine[selected]
        sample_direction = sample_direction[selected]
        sample_log_scale = sample_log_scale[selected]
    basis, basis_eigenvalue, explained_residual_energy = _fit_basis(
        sample_residual, sample_weight, rank=int(args.rank), device=str(args.device),
    )

    normal_matrix = np.zeros(
        (primitive_count, PREDICTOR_COUNT, PREDICTOR_COUNT), dtype=np.float32,
    )
    normal_rhs = np.zeros(
        (primitive_count, PREDICTOR_COUNT, int(args.rank)), dtype=np.float32,
    )
    fit_weight_sum = np.zeros((primitive_count,), dtype=np.float32)
    minimum_direction_cosine = np.ones((primitive_count,), dtype=np.float32)
    for view_index, (path, metadata) in enumerate(eligible, start=1):
        value = _observation(
            path, metadata, physical, dense_physical_lookup, dense_field_lookup, mapper,
        )
        rows = value.field_indices
        canonical_code = canonical.codes[rows]
        cosine = np.sum(value.descriptors * canonical_code, axis=1)
        tangent_residual = value.descriptors - cosine[:, None] * canonical_code
        target = tangent_residual @ basis.T
        predictor = np.concatenate((
            np.ones((rows.size, 1), dtype=np.float32),
            value.local_directions - direction_center[rows],
            (value.log_projected_scales - mean_log_scale[rows])[:, None],
        ), axis=1)
        weighted_outer = (
            value.weights[:, None, None]
            * predictor[:, :, None] * predictor[:, None, :]
        )
        weighted_rhs = (
            value.weights[:, None, None] * predictor[:, :, None] * target[:, None, :]
        )
        # A physical primitive occurs at most once after per-view aggregation,
        # so direct indexed addition has no repeated-index write hazard.
        normal_matrix[rows] += weighted_outer
        normal_rhs[rows] += weighted_rhs
        fit_weight_sum[rows] += value.weights
        direction_cosine = np.sum(
            value.local_directions * mean_direction[rows], axis=1,
        )
        minimum_direction_cosine[rows] = np.minimum(
            minimum_direction_cosine[rows], direction_cosine,
        )
        print(json.dumps({
            "pass": "primitive_ridge",
            "view": int(view_index),
            "view_count": int(len(eligible)),
            "image_id": str(metadata["image_id"]),
            "observation_count": int(rows.size),
        }), flush=True)
    minimum_direction_cosine[~observed] = 1.0
    active = observation_count >= int(args.minimum_views)
    coefficient = _batched_ridge_solve(
        normal_matrix, normal_rhs, fit_weight_sum, active,
        intercept_ridge=float(args.intercept_ridge),
        slope_ridge=float(args.slope_ridge), device=str(args.device),
    )
    del normal_matrix, normal_rhs

    contributor_hash = hashlib.sha256()
    for path, _metadata in eligible:
        contributor_hash.update(path.name.encode("utf8"))
        contributor_hash.update(bytes.fromhex(file_sha256(path)))
    artifact = ViewConditionedPrimitiveField(
        primitive_rows=canonical.primitive_rows,
        residual_basis=basis,
        coefficients=coefficient.astype(np.float16),
        observation_count=observation_count,
        mean_local_direction=mean_direction,
        direction_concentration=direction_concentration,
        minimum_direction_cosine=minimum_direction_cosine,
        mean_log_projected_scale=mean_log_scale,
        minimum_log_projected_scale=minimum_log_scale,
        maximum_log_projected_scale=maximum_log_scale,
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=canonical.content_sha256,
        metadata={
            "artifact_type": SCHEMA,
            "representation": (
                "single_canonical_code_plus_shared_low_rank_tangent_residual_basis_"
                "and_per_primitive_view_scale_coefficients"
            ),
            "vfm_layer": "radio_final",
            "canonical_feature_space": canonical.metadata.get("canonical_feature_space"),
            "rank": int(args.rank),
            "minimum_views": int(args.minimum_views),
            "predictors": [
                "intercept", "centered_local_view_tangent1",
                "centered_local_view_tangent2", "centered_local_view_normal",
                "centered_log_projected_scale",
            ],
            "residual_geometry": "canonical_code_tangent_space",
            "validity": "observed_direction_cone_and_projected_scale_interval_else_canonical_fallback",
            "direction_cosine_margin": float(args.direction_cosine_margin),
            "log_scale_margin": float(args.log_scale_margin),
            "intercept_ridge": float(args.intercept_ridge),
            "slope_ridge": float(args.slope_ridge),
            "mapping_image_count": int(len(eligible)),
            "mapping_trajectory_ids": trajectories,
            "contributor_set_sha256": contributor_hash.hexdigest(),
            "surface_mapper_file_sha256": mapper_hash,
            "stores_per_view_descriptors": False,
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_point_correspondences": False,
        },
    )
    artifact.save_npz(output)

    sample_canonical = canonical.codes[sample_index]
    sample_descriptor = _normalize_rows(
        sample_cosine[:, None] * sample_canonical + sample_residual,
    )
    conditioned, sample_active = artifact.condition_codes_numpy(
        canonical.codes, sample_index, sample_direction, sample_log_scale,
    )
    baseline_similarity = np.sum(sample_descriptor * sample_canonical, axis=1)
    conditioned_similarity = np.sum(sample_descriptor * conditioned, axis=1)
    report = {
        "stage": "build_goal_maplet_low_rank_view_conditioned_field",
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": canonical.content_sha256,
        "view_conditioned_field_sha256": artifact.content_sha256,
        "mapping_image_count": int(len(eligible)),
        "mapping_trajectory_ids": trajectories,
        "total_view_primitive_observation_count": int(total_observation_count),
        "basis_sample_count": int(sample_residual.shape[0]),
        "rank": int(artifact.rank),
        "basis_eigenvalues": basis_eigenvalue.tolist(),
        "explained_tangent_residual_energy_fraction": explained_residual_energy,
        "conditioned_primitive_count": int(np.sum(active)),
        "conditioned_canonical_coverage_fraction": float(np.mean(active)),
        "observation_count": {
            "median": float(np.median(observation_count[observed])),
            "p90": float(np.percentile(observation_count[observed], 90.0)),
            "maximum": int(np.max(observation_count)),
        },
        "sample_reconstruction": {
            "active_fraction": float(np.mean(sample_active)),
            "canonical_cosine_mean": float(np.mean(baseline_similarity[sample_active])),
            "conditioned_cosine_mean": float(np.mean(conditioned_similarity[sample_active])),
            "mean_cosine_gain": float(np.mean(
                conditioned_similarity[sample_active] - baseline_similarity[sample_active]
            )),
        },
        "storage_bytes_uncompressed": int(
            sum(np.asarray(value).nbytes for value in (
                artifact.residual_basis, artifact.coefficients,
                artifact.observation_count, artifact.mean_local_direction,
                artifact.direction_concentration, artifact.minimum_direction_cosine,
                artifact.mean_log_projected_scale,
                artifact.minimum_log_projected_scale,
                artifact.maximum_log_projected_scale,
            ))
        ),
        "storage_contract": dict(artifact.metadata),
        "output_field": str(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

