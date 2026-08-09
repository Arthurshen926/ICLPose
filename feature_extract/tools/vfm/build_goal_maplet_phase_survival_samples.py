"""Build G19-A GT/physical-phase pairs at every Goal-Maplet feature stage."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_codec import CanonicalRadioCodec
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    FEATURE_NAMES,
    extract_surface_likelihood_features,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    render_canonical_surface_field,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
    pnp_pose_error,
)


MAP_LEVELS = (
    "radio_final_canonical",
    "radio_pca256_canonical",
    "mapper_canonical",
    "local_flat_transform",
    "local_spatial_readout",
    "context_spatial_readout",
)
G18_LEVELS = (
    "g18_summary_no_absolute_xy",
    "g18_summary_grid_xy",
    "g18_summary_camera_ray",
)
LEVELS = MAP_LEVELS + G18_LEVELS


def _phase_negative_index(details: list[dict[str, object]]) -> int:
    """Select a pose-defined nearby wrong phase without looking at appearance."""

    if len(details) < 2:
        raise ValueError("phase audit requires at least two frozen candidates")
    translation = np.asarray([float(item["translation_m"]) for item in details])
    rotation = np.asarray([float(item["rotation_deg"]) for item in details])
    eligible = np.flatnonzero(
        (translation >= 0.50) & (translation <= 2.50) & (rotation <= 10.0)
    )
    if eligible.size == 0:
        eligible = np.flatnonzero(translation >= 0.50)
    if eligible.size == 0:
        # A frozen pool is not required to place its best candidate at index
        # zero.  If every pose is already inside 0.5 m, retain the most
        # displaced pose instead of silently excluding an arbitrary row.
        eligible = np.asarray([int(np.argmax(translation))], dtype=np.int64)
    # A facade period in this sequence is usually sub-metre to roughly one
    # metre.  Rotation is a soft tie breaker so the pair remains phase-like.
    objective = np.abs(translation[eligible] - 0.75) + 0.03 * rotation[eligible]
    return int(eligible[int(np.argmin(objective))])


def _normalize_map(value: np.ndarray) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32)
    return feature / np.maximum(np.linalg.norm(feature, axis=0, keepdims=True), 1.0e-8)


def _pool_grid(value: np.ndarray, rows: int = 6, cols: int = 8) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    height, width = source.shape[-2:]
    if height % int(rows) or width % int(cols):
        raise ValueError("G19-A probe grid must divide the token map")
    output = source.reshape(
        *source.shape[:-2], rows, height // rows, cols, width // cols,
    ).mean(axis=(-3, -1))
    return output.reshape(-1)


def _map_probe_descriptor(
    query: np.ndarray,
    rendered: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    query_unit = _normalize_map(query)
    render_unit = _normalize_map(rendered)
    mask = np.asarray(valid, dtype=np.float32)
    product = query_unit * render_unit * mask[None]
    absolute = np.abs(query_unit - render_unit) * mask[None]
    cosine = np.sum(query_unit * render_unit, axis=0) * mask
    difference = np.mean(np.abs(query_unit - render_unit), axis=0) * mask
    descriptor = np.concatenate([
        np.mean(product, axis=(1, 2)),
        np.mean(absolute, axis=(1, 2)),
        _pool_grid(cosine),
        _pool_grid(difference),
        _pool_grid(mask),
    ]).astype(np.float32)
    return descriptor, float(np.mean(cosine)), cosine.astype(np.float32)


def _camera_ray_grid(camera, height: int, width: int) -> np.ndarray:
    grid_y, grid_x = np.mgrid[:height, :width]
    pixels = np.stack([
        (grid_x.reshape(-1) + 0.5) * float(camera.width) / float(width) - 0.5,
        (grid_y.reshape(-1) + 0.5) * float(camera.height) / float(height) - 0.5,
    ], axis=1).astype(np.float64)
    matrix, distortion = camera_matrix_and_distortion(camera)
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), matrix, distortion,
    ).reshape(-1, 2)
    rays = np.concatenate([
        normalized, np.ones((normalized.shape[0], 1), dtype=np.float64),
    ], axis=1)
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1.0e-12)
    return rays.astype(np.float32)


def _g18_probe_descriptors(
    token_feature: np.ndarray,
    camera_rays: np.ndarray,
) -> dict[str, np.ndarray]:
    value = np.asarray(token_feature, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise ValueError("G18 token feature shape differs")
    keep = [
        index for index, name in enumerate(FEATURE_NAMES)
        if name not in {"grid_x", "grid_y"}
    ]
    base = value[:, keep]
    moments = np.concatenate([
        np.mean(base, axis=0),
        np.std(base, axis=0),
        np.percentile(base, 90.0, axis=0),
    ])
    grid_x = value[:, FEATURE_NAMES.index("grid_x")]
    grid_y = value[:, FEATURE_NAMES.index("grid_y")]
    grid_interaction = np.concatenate([
        np.mean(base * grid_x[:, None], axis=0),
        np.mean(base * grid_y[:, None], axis=0),
    ])
    ray_interaction = np.concatenate([
        np.mean(base * camera_rays[:, axis:axis + 1], axis=0)
        for axis in range(3)
    ])
    return {
        "g18_summary_no_absolute_xy": moments.astype(np.float32),
        "g18_summary_grid_xy": np.concatenate([moments, grid_interaction]).astype(np.float32),
        "g18_summary_camera_ray": np.concatenate([moments, ray_interaction]).astype(np.float32),
    }


def _replace_feature(rendered, feature: np.ndarray):
    return replace(rendered, feature=np.asarray(feature, dtype=np.float32))


def _joint_diagnostic_field(
    raw_field: CanonicalSurfaceField,
    pca_field: CanonicalSurfaceField,
    mapper_field: CanonicalSurfaceField,
) -> tuple[CanonicalSurfaceField, tuple[slice, slice, slice]]:
    """Concatenate aligned spaces so one pose needs one geometry render."""

    if not (
        np.array_equal(raw_field.primitive_rows, pca_field.primitive_rows)
        and np.array_equal(raw_field.primitive_rows, mapper_field.primitive_rows)
    ):
        raise ValueError("G19-A diagnostic fields do not share primitive support")
    dimensions = (raw_field.feature_dim, pca_field.feature_dim, mapper_field.feature_dim)
    boundaries = np.cumsum((0,) + dimensions)
    joint = CanonicalSurfaceField(
        primitive_rows=raw_field.primitive_rows,
        codes=np.concatenate([raw_field.codes, pca_field.codes, mapper_field.codes], axis=1),
        confidence=mapper_field.confidence,
        uncertainty=mapper_field.uncertainty,
        physical_map_sha256=mapper_field.physical_map_sha256,
        metadata={
            "representation": "ephemeral_g19_a_joint_diagnostic_render_only",
            "stored_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "uses_point_correspondences": False,
        },
    )
    return joint, tuple(
        slice(int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(3)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--mapper_field", required=True)
    parser.add_argument("--radio_raw_field", required=True)
    parser.add_argument("--radio_pca_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--canonical_codec", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--maximum_modes", type=int, default=16)
    parser.add_argument("--views_per_trajectory", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite G19-A samples")

    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    mapper_field = CanonicalSurfaceField.load_npz(Path(args.mapper_field))
    raw_field = CanonicalSurfaceField.load_npz(Path(args.radio_raw_field))
    pca_field = CanonicalSurfaceField.load_npz(Path(args.radio_pca_field))
    if mapper_field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("mapper canonical field and physical map differ")
    if raw_field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("raw RADIO canonical field and physical map differ")
    if pca_field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("RADIO PCA field and physical map differ")
    if str(raw_field.metadata.get("canonical_feature_space", "")) != "raw_radio_final":
        raise ValueError("G19-A raw RADIO field is not in RADIO-final space")
    mapper, _mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device),
    )
    codec = CanonicalRadioCodec.load_npz(Path(args.canonical_codec))
    readout, readout_metadata = load_physical_instance_readout(
        Path(args.physical_instance_readout), device=str(args.device),
    )
    if readout_metadata.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("physical-instance readout and map differ")
    if readout_metadata.get("canonical_field_sha256") != mapper_field.content_sha256:
        raise ValueError("physical-instance readout and mapper field differ")
    if str(pca_field.metadata.get("canonical_codec_sha256", "")) != codec.content_sha256:
        raise ValueError("RADIO PCA field and codec differ")
    joint_field, (raw_slice, pca_slice, mapper_slice) = _joint_diagnostic_field(
        raw_field, pca_field, mapper_field,
    )

    pool_path = Path(args.candidate_pool)
    pool = json.loads(pool_path.read_text())
    if pool.get("physical_instance_readout_sha256") != file_sha256(Path(args.physical_instance_readout)):
        raise ValueError("candidate pool and physical-instance readout differ")
    contributor_by_image = {}
    for path in Path(args.contributors).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        contributor_by_image[str(metadata["image_id"])] = path
    rows = sorted(pool.get("rows", []), key=lambda item: str(item["image_id"]))
    if int(args.views_per_trajectory) > 0:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(str(row["image_id"]).split("/", 1)[0], []).append(row)
        selected_rows = []
        for trajectory in sorted(grouped):
            values = grouped[trajectory]
            count = min(int(args.views_per_trajectory), len(values))
            indices = np.linspace(0, len(values) - 1, num=count).round().astype(np.int64)
            selected_rows.extend(values[int(index)] for index in np.unique(indices))
        rows = sorted(selected_rows, key=lambda item: str(item["image_id"]))
    rows = rows[int(args.shard_index) :: int(args.shard_count)]

    descriptors = {level: [] for level in LEVELS}
    scores = {level: [] for level in LEVELS}
    discriminative_fraction = {level: [] for level in MAP_LEVELS}
    positive_mass = {level: [] for level in MAP_LEVELS}
    image_ids, trajectories = [], []
    phase_translation, phase_rotation, phase_rank = [], [], []
    skipped_single_candidate = []
    for row in rows:
        image_id = str(row["image_id"])
        contributor = contributor_by_image.get(image_id)
        if contributor is None:
            raise ValueError(f"missing G19-A contributor: {image_id}")
        labels = ContributorLabels.load_npz(contributor)
        camera = _camera(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        query_pca = codec.transform_map(raw)
        query_mapper = mapper.project(raw).measurement_context
        height, width = query_mapper.shape[1:]
        grid_y, grid_x = np.mgrid[:height, :width]
        token_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        query_flat = readout.project_numpy(
            query_mapper.transpose(1, 2, 0).reshape(-1, query_mapper.shape[0]),
            role="local", device=str(args.device),
        ).reshape(height, width, -1).transpose(2, 0, 1)
        query_local = encode_physical_instance_regions(
            readout, query_mapper, token_xy, role="local", device=str(args.device),
        ).reshape(height, width, -1).transpose(2, 0, 1)
        query_context = encode_physical_instance_regions(
            readout, query_mapper, token_xy, role="context", device=str(args.device),
        ).reshape(height, width, -1).transpose(2, 0, 1)
        rays = _camera_ray_grid(camera, height, width)

        details = list(row.get("mode_details", {}).get(str(args.mode_name), []))[
            : int(args.maximum_modes)
        ]
        if len(details) < 2:
            skipped_single_candidate.append(image_id)
            print(json.dumps({
                "image_id": image_id,
                "skipped": "fewer_than_two_frozen_candidates",
            }), flush=True)
            continue
        negative_index = _phase_negative_index(details)
        negative_pose = np.asarray(details[negative_index]["pose_w2c"], dtype=np.float64)
        named_pose = (("gt", labels.pose_w2c), ("phase", negative_pose))
        level_pair_descriptors = {level: [] for level in LEVELS}
        level_pair_scores = {level: [] for level in LEVELS}
        level_cosine = {level: [] for level in MAP_LEVELS}
        for _name, pose in named_pose:
            rendered_joint = render_canonical_surface_field(
                physical, joint_field, pose, camera,
                width=width, height=height, device=str(args.device),
            )
            rendered_raw = _replace_feature(
                rendered_joint, _normalize_map(rendered_joint.feature[raw_slice]),
            )
            rendered_pca = _replace_feature(
                rendered_joint, _normalize_map(rendered_joint.feature[pca_slice]),
            )
            rendered_mapper = _replace_feature(
                rendered_joint, _normalize_map(rendered_joint.feature[mapper_slice]),
            )
            raw_render = np.asarray(rendered_mapper.feature, dtype=np.float32)
            render_flat = readout.project_numpy(
                raw_render.transpose(1, 2, 0).reshape(-1, raw_render.shape[0]),
                role="local", device=str(args.device),
            ).reshape(height, width, -1).transpose(2, 0, 1)
            render_local = encode_physical_instance_regions(
                readout, raw_render, token_xy, role="local", device=str(args.device),
                spatial_valid_mask=np.asarray(rendered_mapper.mask, dtype=bool),
            ).reshape(height, width, -1).transpose(2, 0, 1)
            render_context = encode_physical_instance_regions(
                readout, raw_render, token_xy, role="context", device=str(args.device),
                spatial_valid_mask=np.asarray(rendered_mapper.mask, dtype=bool),
            ).reshape(height, width, -1).transpose(2, 0, 1)
            map_pairs = {
                "radio_final_canonical": (
                    raw, np.asarray(rendered_raw.feature), rendered_raw.mask,
                ),
                "radio_pca256_canonical": (
                    query_pca, np.asarray(rendered_pca.feature), rendered_pca.mask,
                ),
                "mapper_canonical": (query_mapper, raw_render, rendered_mapper.mask),
                "local_flat_transform": (query_flat, render_flat, rendered_mapper.mask),
                "local_spatial_readout": (query_local, render_local, rendered_mapper.mask),
                "context_spatial_readout": (query_context, render_context, rendered_mapper.mask),
            }
            for level, (query_feature, render_feature, mask) in map_pairs.items():
                descriptor, score, cosine = _map_probe_descriptor(
                    query_feature, render_feature, mask,
                )
                level_pair_descriptors[level].append(descriptor)
                level_pair_scores[level].append(score)
                level_cosine[level].append(cosine)
            rendered_context = _replace_feature(rendered_mapper, render_context)
            token_feature, _typed_target, _summary = extract_surface_likelihood_features(
                query_context, rendered_context, pose,
            )
            for level, descriptor in _g18_probe_descriptors(token_feature, rays).items():
                level_pair_descriptors[level].append(descriptor)
                level_pair_scores[level].append(level_pair_scores["context_spatial_readout"][-1])

        for level in LEVELS:
            descriptors[level].append(np.stack(level_pair_descriptors[level]).astype(np.float16))
            scores[level].append(np.asarray(level_pair_scores[level], dtype=np.float32))
        for level in MAP_LEVELS:
            delta = level_cosine[level][0] - level_cosine[level][1]
            discriminative_fraction[level].append(float(np.mean(np.abs(delta) >= 0.05)))
            denominator = float(np.sum(np.abs(delta)))
            positive_mass[level].append(
                float(np.sum(np.maximum(delta, 0.0)) / max(denominator, 1.0e-8))
            )
        phase_error = pnp_pose_error(negative_pose, labels.pose_w2c)
        image_ids.append(image_id)
        trajectories.append(image_id.split("/", 1)[0])
        phase_translation.append(float(phase_error.translation_m))
        phase_rotation.append(float(phase_error.rotation_deg))
        phase_rank.append(int(negative_index + 1))
        print(json.dumps({
            "image_id": image_id,
            "phase_rank": int(negative_index + 1),
            "phase_translation_m": float(phase_error.translation_m),
            "phase_rotation_deg": float(phase_error.rotation_deg),
            "context_margin": float(
                level_pair_scores["context_spatial_readout"][0]
                - level_pair_scores["context_spatial_readout"][1]
            ),
        }), flush=True)

    metadata = {
        "artifact_type": "goal_maplet_phase_survival_samples_v1",
        "stage": "g19_a_phase_information_survival",
        "levels": LEVELS,
        "pair_definition": "gt_vs_pose_defined_nearest_frozen_phase_negative",
        "phase_negative_uses_appearance": False,
        "skipped_single_candidate_image_ids": skipped_single_candidate,
        "candidate_pool_sha256": file_sha256(pool_path),
        "physical_map_sha256": physical.content_sha256,
        "mapper_field_sha256": mapper_field.content_sha256,
        "radio_raw_field_sha256": raw_field.content_sha256,
        "radio_pca_field_sha256": pca_field.content_sha256,
        "canonical_codec_sha256": codec.content_sha256,
        "surface_mapper_sha256": file_sha256(Path(args.surface_mapper)),
        "physical_instance_readout_sha256": file_sha256(Path(args.physical_instance_readout)),
        "radio_level_limitation": (
            "The 1280D RADIO-final level is measured after view-balanced canonical "
            "fusion. It isolates mapper/PCA/readout losses but cannot by itself "
            "separate native RADIO invariance from multi-view fusion loss."
        ),
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "stores_mapping_rgb": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "trajectory_ids": sorted(set(trajectories)),
        "views_per_trajectory": int(args.views_per_trajectory),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "image_ids": np.asarray(image_ids),
        "trajectory_ids": np.asarray(trajectories),
        "phase_translation_m": np.asarray(phase_translation, dtype=np.float32),
        "phase_rotation_deg": np.asarray(phase_rotation, dtype=np.float32),
        "phase_rank": np.asarray(phase_rank, dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for level in LEVELS:
        arrays[f"descriptor__{level}"] = np.stack(descriptors[level])
        arrays[f"score__{level}"] = np.stack(scores[level])
    for level in MAP_LEVELS:
        arrays[f"discriminative_fraction__{level}"] = np.asarray(
            discriminative_fraction[level], dtype=np.float32,
        )
        arrays[f"positive_mass__{level}"] = np.asarray(
            positive_mass[level], dtype=np.float32,
        )
    np.savez_compressed(output, **arrays)
    print(json.dumps({**metadata, "query_count": len(image_ids)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
