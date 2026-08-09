"""Render frozen candidates into G18 listwise surface-likelihood samples."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    EVENT_NAMES,
    FEATURE_NAMES,
    assign_pose_defined_typed_targets,
    extract_surface_likelihood_features,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import render_canonical_surface_field


def _compress_teacher(value: np.ndarray, dimensions: int = 64) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] % int(dimensions):
        raise ValueError("teacher width is not divisible by compact width")
    compact = array.reshape(*array.shape[:-1], int(dimensions), -1).mean(axis=-1)
    return compact / np.maximum(np.linalg.norm(compact, axis=-1, keepdims=True), 1.0e-8)


def _teacher_contrast(
    value: np.ndarray,
    token_xy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Interpolate sparse teacher boundary/affinity cues to the runtime grid."""

    feature = np.asarray(value, dtype=np.float32)
    xy = np.asarray(token_xy, dtype=np.float32).reshape(-1, 2)
    if feature.shape[0] != xy.shape[0] or feature.shape[0] < 2:
        raise ValueError("invalid sparse teacher support")
    distance2 = np.sum(np.square(xy[:, None, :] - xy[None, :, :]), axis=2)
    np.fill_diagonal(distance2, np.inf)
    neighbour_count = min(4, feature.shape[0] - 1)
    neighbours = np.argpartition(distance2, kth=neighbour_count - 1, axis=1)[:, :neighbour_count]
    sparse_contrast = np.mean(
        1.0 - np.sum(feature[:, None, :] * feature[neighbours], axis=2), axis=1,
    )
    grid_y, grid_x = np.mgrid[:height, :width]
    grid = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
    grid_distance2 = np.sum(np.square(grid[:, None, :] - xy[None, :, :]), axis=2)
    interpolation_count = min(4, xy.shape[0])
    nearest = np.argpartition(
        grid_distance2, kth=interpolation_count - 1, axis=1,
    )[:, :interpolation_count]
    weight = 1.0 / np.maximum(np.take_along_axis(grid_distance2, nearest, axis=1), 0.25)
    weight /= np.maximum(np.sum(weight, axis=1, keepdims=True), 1.0e-8)
    return np.sum(weight * sparse_contrast[nearest], axis=1).astype(np.float32)


def _normalize_cue(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    low, high = np.percentile(source, [10.0, 90.0])
    return np.clip((source - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)


def _teacher_cues(
    path: Path,
    height: int,
    width: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(path, allow_pickle=False) as data:
        dino = _compress_teacher(np.asarray(data["dino_v3_7b"], dtype=np.float32))
        sam = _compress_teacher(np.asarray(data["sam3"], dtype=np.float32))
        siglip = _compress_teacher(np.asarray(data["siglip2-g"], dtype=np.float32))
        token_xy = np.asarray(data["token_xy"], dtype=np.float32)
    cue = {
        "dino_local_affinity": _normalize_cue(_teacher_contrast(dino, token_xy, height, width)),
        "sam_boundary": _normalize_cue(_teacher_contrast(sam, token_xy, height, width)),
        "siglip_context": _normalize_cue(_teacher_contrast(siglip, token_xy, height, width)),
    }
    return cue, {
        key: float(np.mean(value)) for key, value in cue.items()
    }


def _teacher_weight(path: Path, height: int, width: int) -> tuple[np.ndarray, dict[str, float]]:
    """Backward-compatible aggregate used only for audits/tests."""

    cue, audit = _teacher_cues(path, height, width)
    weight = (
        1.0
        + 0.35 * cue["dino_local_affinity"]
        + 0.40 * cue["sam_boundary"]
        + 0.25 * cue["siglip_context"]
    )
    return weight.astype(np.float32), audit


def _typed_teacher_weight(
    typed_target: np.ndarray,
    cue: dict[str, np.ndarray],
) -> np.ndarray:
    """Route each offline teacher only to its declared event role.

    This is supervision metadata, not a runtime feature.  It avoids the
    rejected static-teacher-reliability average: DINO/SigLIP emphasize phase
    competition, while SAM emphasizes boundary/visibility null events.
    """

    target = np.asarray(typed_target, dtype=np.uint8)
    weight = np.ones(target.shape, dtype=np.float32)
    match = target == EVENT_NAMES.index("surface_match")
    wrong_phase = target == EVENT_NAMES.index("wrong_phase")
    support_event = np.isin(target, [
        EVENT_NAMES.index("occluded"),
        EVENT_NAMES.index("field_missing"),
        EVENT_NAMES.index("query_unmapped_dynamic"),
    ])
    weight += 0.35 * cue["dino_local_affinity"][None] * (match | wrong_phase)
    weight += 0.25 * cue["siglip_context"][None] * match
    weight += 0.50 * cue["siglip_context"][None] * wrong_phase
    weight += 0.50 * cue["sam_boundary"][None] * support_event
    return weight.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--teacher_cache", default="")
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--maximum_modes", type=int, default=16)
    parser.add_argument("--views_per_trajectory", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite G18 samples")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout, metadata = load_physical_instance_readout(
        Path(args.physical_instance_readout), device=str(args.device),
    )
    if metadata.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("readout and physical map differ")
    if metadata.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("readout and canonical field differ")
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    pool_path = Path(args.candidate_pool)
    pool = json.loads(pool_path.read_text())
    if pool.get("physical_instance_readout_sha256") != file_sha256(Path(args.physical_instance_readout)):
        raise ValueError("candidate pool and readout differ")
    contributors = {}
    for path in Path(args.contributors).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            item = json.loads(str(np.asarray(data["metadata_json"]).item()))
        contributors[str(item["image_id"])] = path
    rows = sorted(pool.get("rows", []), key=lambda item: str(item["image_id"]))
    if int(args.views_per_trajectory) > 0:
        grouped = {}
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
    maximum_modes = int(args.maximum_modes)
    feature_rows, type_rows, query_rows = [], [], []
    translation_rows, rotation_rows, valid_rows, target_rows = [], [], [], []
    teacher_rows, image_ids, trajectories, teacher_audit = [], [], [], []
    for row in rows:
        image_id = str(row["image_id"])
        contributor = contributors.get(image_id)
        if contributor is None:
            raise ValueError(f"missing G18 contributor: {image_id}")
        camera = _camera(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            item = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(item["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        height, width = mapped.shape[1:]
        grid_y, grid_x = np.mgrid[:height, :width]
        token_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        query_flat = encode_physical_instance_regions(
            readout, mapped, token_xy, role="context", device=str(args.device),
        )
        query = query_flat.reshape(height, width, mapped.shape[0]).transpose(2, 0, 1)
        details = list(row.get("mode_details", {}).get(str(args.mode_name), []))[:maximum_modes]
        if not details:
            raise ValueError(f"G18 requires at least one frozen candidate: {image_id}")
        candidate_feature, candidate_type = [], []
        translation, rotation = [], []
        query_summary = None
        for detail in details:
            pose = np.asarray(detail["pose_w2c"], dtype=np.float64)
            rendered = render_canonical_surface_field(
                physical, field, pose, camera,
                width=width, height=height, device=str(args.device),
            )
            render_flat = encode_physical_instance_regions(
                readout,
                np.asarray(rendered.feature, dtype=np.float32),
                token_xy,
                role="context",
                device=str(args.device),
                spatial_valid_mask=np.asarray(rendered.mask, dtype=bool),
            )
            render_feature = render_flat.reshape(height, width, mapped.shape[0]).transpose(2, 0, 1)
            rendered = replace(rendered, feature=render_feature)
            token_feature, typed_target, summary = extract_surface_likelihood_features(query, rendered, pose)
            candidate_feature.append(token_feature.astype(np.float16))
            candidate_type.append(typed_target.astype(np.uint8))
            translation.append(float(detail["translation_m"]))
            rotation.append(float(detail["rotation_deg"]))
            query_summary = summary
        candidate_count = len(details)
        token_count = int(height * width)
        while len(candidate_feature) < maximum_modes:
            candidate_feature.append(np.zeros((token_count, len(FEATURE_NAMES)), dtype=np.float16))
            candidate_type.append(np.full(
                (token_count,), EVENT_NAMES.index("unresolved"), dtype=np.uint8,
            ))
            translation.append(1.0e6)
            rotation.append(1.0e6)
        quality = np.asarray(translation[:candidate_count]) / 0.5 + np.asarray(rotation[:candidate_count]) / 5.0
        has_usable = np.any((np.asarray(translation) <= 1.0) & (np.asarray(rotation) <= 10.0))
        target = int(np.argmin(quality)) if has_usable else maximum_modes
        for candidate_index in range(candidate_count):
            candidate_type[candidate_index] = assign_pose_defined_typed_targets(
                candidate_type[candidate_index],
                candidate_feature[candidate_index],
                translation_m=float(translation[candidate_index]),
                rotation_deg=float(rotation[candidate_index]),
                is_listwise_target=bool(candidate_index == target),
            )
        teacher_weight = np.ones((maximum_modes, height * width), dtype=np.float32)
        teacher_report = {"teacher_available": False}
        if args.teacher_cache:
            teacher_path = Path(args.teacher_cache) / (image_id.replace("/", "__") + ".npz")
            if teacher_path.exists():
                cue, cue_audit = _teacher_cues(teacher_path, height, width)
                teacher_weight = _typed_teacher_weight(np.stack(candidate_type), cue)
                teacher_report = {"teacher_available": True, **cue_audit}
        feature_rows.append(np.stack(candidate_feature))
        type_rows.append(np.stack(candidate_type))
        query_rows.append(query_summary)
        translation_rows.append(translation)
        rotation_rows.append(rotation)
        valid_rows.append(np.arange(maximum_modes) < candidate_count)
        target_rows.append(target)
        teacher_rows.append(teacher_weight.astype(np.float16))
        teacher_audit.append(teacher_report)
        image_ids.append(image_id)
        trajectories.append(image_id.split("/", 1)[0])
        print(json.dumps({
            "image_id": image_id,
            "target_index": target,
            "candidate_count": candidate_count,
            "oracle_translation_m": float(np.asarray(translation)[np.argmin(quality)]),
            "teacher_available": bool(teacher_report["teacher_available"]),
        }), flush=True)
    if not feature_rows:
        raise ValueError("no G18 surface samples")
    metadata = {
        "artifact_type": "goal_maplet_surface_likelihood_samples_v1",
        "candidate_pool_sha256": file_sha256(pool_path),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_sha256": file_sha256(Path(args.surface_mapper)),
        "physical_instance_readout_sha256": file_sha256(Path(args.physical_instance_readout)),
        "feature_names": FEATURE_NAMES,
        "event_names": EVENT_NAMES,
        "candidate_count": maximum_modes,
        "candidate_set_frozen_before_scoring": True,
        "fixed_full_query_denominator": True,
        "typed_phase_target_semantics": (
            "pose_defined_frozen_candidate_error_not_feature_cosine"
        ),
        "typed_phase_target_uses_appearance": False,
        "training_teacher_roles": {
            "dino_v3_7b": "local_surface_affinity_hard_negative_weight",
            "sam3": "support_boundary_weight",
            "siglip2-g": "context_phase_weight",
        },
        "teacher_embeddings_stored": False,
        "teacher_supervision": "event_conditioned_spatial_weights_not_runtime_features",
        "teacher_available_fraction": float(np.mean([
            value["teacher_available"] for value in teacher_audit
        ])),
        "trajectory_ids": sorted(set(trajectories)),
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "stores_mapping_rgb": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        token_feature=np.stack(feature_rows),
        typed_target=np.stack(type_rows),
        query_summary=np.stack(query_rows),
        translation_m=np.asarray(translation_rows, dtype=np.float32),
        rotation_deg=np.asarray(rotation_rows, dtype=np.float32),
        candidate_valid=np.stack(valid_rows),
        target_index=np.asarray(target_rows, dtype=np.int64),
        teacher_weight=np.stack(teacher_rows),
        image_ids=np.asarray(image_ids),
        trajectory_ids=np.asarray(trajectories),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "query_count": len(image_ids)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
