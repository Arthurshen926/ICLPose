"""Audit the six-DoF local basin of the selected Goal-Maplet phase score.

This is a no-training correctness gate.  It evaluates the serialized runtime
readout around ground truth in a local surface frame, rather than judging a
refiner from whichever coarse candidates retrieval happened to provide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import (
    DualBandPhaseEvidence,
    dual_band_phase_evidence,
    load_phase_readout_policy,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_renderer import render_canonical_surface_field


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            0,
            int(data["camera_model_id"]),
            int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(np.asarray(data["metadata_json"]).item()))


def _query_mapper(path: Path, mapper) -> np.ndarray:
    metadata = _metadata(path)
    with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
        raw = np.asarray(data["radio_final"], dtype=np.float32)
    return np.asarray(mapper.project(raw).measurement_context, dtype=np.float32)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64)
    value /= np.maximum(np.linalg.norm(value), 1.0e-12)
    x, y, z = value
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _translate_camera(pose_w2c: np.ndarray, world_delta: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).copy()
    rotation = pose[:3, :3]
    center = -(rotation.T @ pose[:3, 3]) + np.asarray(world_delta, dtype=np.float64)
    pose[:3, 3] = -(rotation @ center)
    return pose


def _rotate_camera(pose_w2c: np.ndarray, camera_axis: np.ndarray, angle: float) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).copy()
    center = -(pose[:3, :3].T @ pose[:3, 3])
    pose[:3, :3] = _axis_angle(camera_axis, angle) @ pose[:3, :3]
    pose[:3, 3] = -(pose[:3, :3] @ center)
    return pose


def _score(
    query: np.ndarray,
    pose: np.ndarray,
    camera: ColmapCamera,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    policy,
    device: str,
) -> tuple[float, DualBandPhaseEvidence, object]:
    rendered = render_canonical_surface_field(
        physical,
        field,
        pose,
        camera,
        width=int(query.shape[2]),
        height=int(query.shape[1]),
        device=device,
    )
    evidence = dual_band_phase_evidence(
        query,
        np.asarray(rendered.feature, dtype=np.float32),
        query,
        np.asarray(rendered.feature, dtype=np.float32),
        np.asarray(rendered.mask, dtype=bool),
    )
    return float(policy.score(evidence)), evidence, rendered


def _sequence_metrics(
    rows: list[dict[str, object]],
    translation_magnitudes: list[float],
    rotation_magnitudes: list[float],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for axis in ("tangent1", "tangent2", "normal", "roll", "pitch", "yaw"):
        axis_rows = [row for row in rows if row["axis"] == axis]
        query_ids = sorted({str(row["image_id"]) for row in axis_rows})
        magnitudes = (
            translation_magnitudes
            if axis in ("tangent1", "tangent2", "normal")
            else rotation_magnitudes
        )
        smallest = min(magnitudes)
        local_max = []
        monotonic = []
        inward = []
        margins = []
        for image_id in query_ids:
            selected = [row for row in axis_rows if row["image_id"] == image_id]
            by_key = {(int(row["sign"]), float(row["magnitude"])): row for row in selected}
            gt_score = float(selected[0]["gt_score"])
            if all((sign, smallest) in by_key for sign in (-1, 1)):
                local_max.append(all(gt_score > float(by_key[(sign, smallest)]["score"]) for sign in (-1, 1)))
                margins.extend(gt_score - float(by_key[(sign, smallest)]["score"]) for sign in (-1, 1))
            for sign in (-1, 1):
                values = [gt_score] + [
                    float(by_key[(sign, magnitude)]["score"])
                    for magnitude in magnitudes if (sign, magnitude) in by_key
                ]
                if len(values) == len(magnitudes) + 1:
                    monotonic.append(all(values[index] > values[index + 1] for index in range(len(values) - 1)))
                    inward.extend(values[index] > values[index + 1] for index in range(len(values) - 1))
        result[axis] = {
            "query_count": len(query_ids),
            "gt_strict_local_max_fraction": float(np.mean(local_max)) if local_max else None,
            "fully_monotonic_ray_fraction": float(np.mean(monotonic)) if monotonic else None,
            "inward_step_correct_fraction": float(np.mean(inward)) if inward else None,
            "smallest_offset_margin_median": float(np.median(margins)) if margins else None,
            "smallest_offset_margin_p10": float(np.percentile(margins, 10.0)) if margins else None,
        }
    translation_axes = ("tangent1", "tangent2", "normal")
    rotation_axes = ("roll", "pitch", "yaw")
    result["gate"] = {
        "translation_0.25m_local_max_fraction": float(np.mean([
            row["gt_score"] > row["score"]
            for row in rows
            if row["axis"] in translation_axes and row["magnitude"] == 0.25
        ])),
        "translation_0.5m_local_max_fraction": float(np.mean([
            row["gt_score"] > row["score"]
            for row in rows
            if row["axis"] in translation_axes and row["magnitude"] == 0.5
        ])),
        "rotation_3deg_local_max_fraction": float(np.mean([
            row["gt_score"] > row["score"]
            for row in rows
            if row["axis"] in rotation_axes and row["magnitude"] == 3.0
        ])),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--phase_readout_model", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--translation_magnitudes", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--rotation_magnitudes_deg", default="1,3,5")
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase-basin report")
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid basin shard")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    policy = load_phase_readout_policy(Path(args.phase_readout_model))
    for key, expected in (("physical_map_sha256", physical.content_sha256), ("canonical_field_sha256", field.content_sha256)):
        if str(policy.metadata.get(key, "")) != expected:
            raise ValueError(f"phase policy lineage differs: {key}")
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index)::int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        paths = paths[:int(args.maximum_queries)]
    translation_magnitudes = [float(value) for value in str(args.translation_magnitudes).split(",")]
    rotation_magnitudes = [float(value) for value in str(args.rotation_magnitudes_deg).split(",")]
    rows: list[dict[str, object]] = []
    gt_rows: list[dict[str, object]] = []
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        metadata = _metadata(path)
        image_id = str(metadata["image_id"])
        camera = _camera(path)
        query = _query_mapper(path, mapper)
        gt_score, gt_evidence, gt_render = _score(
            query, labels.pose_w2c, camera, physical, field, policy, str(args.device),
        )
        primitive = np.asarray(gt_render.surface_id, dtype=np.int64)
        valid_rows = primitive[np.asarray(gt_render.mask, dtype=bool) & (primitive >= 0)]
        if not valid_rows.size:
            raise ValueError(f"ground-truth render has no physical support: {image_id}")
        dominant_row = int(np.bincount(valid_rows).argmax())
        surface_axes = {
            "tangent1": physical.primitive_tangent1[dominant_row],
            "tangent2": physical.primitive_tangent2[dominant_row],
            "normal": physical.primitive_normals[dominant_row],
        }
        gt_rows.append({
            "image_id": image_id,
            "score": gt_score,
            "dominant_primitive_row": dominant_row,
            "phase": gt_evidence.as_dict(),
        })
        for axis_name, axis in surface_axes.items():
            for magnitude in translation_magnitudes:
                for sign in (-1, 1):
                    pose = _translate_camera(labels.pose_w2c, float(sign) * magnitude * axis)
                    score, evidence, _ = _score(query, pose, camera, physical, field, policy, str(args.device))
                    rows.append({"image_id": image_id, "axis": axis_name, "unit": "m", "sign": sign, "magnitude": magnitude, "gt_score": gt_score, "score": score, "margin_to_gt": gt_score - score, "phase": evidence.as_dict()})
        camera_axes = {"roll": np.asarray([0.0, 0.0, 1.0]), "pitch": np.asarray([1.0, 0.0, 0.0]), "yaw": np.asarray([0.0, 1.0, 0.0])}
        for axis_name, axis in camera_axes.items():
            for magnitude in rotation_magnitudes:
                for sign in (-1, 1):
                    pose = _rotate_camera(labels.pose_w2c, axis, np.deg2rad(float(sign) * magnitude))
                    score, evidence, _ = _score(query, pose, camera, physical, field, policy, str(args.device))
                    rows.append({"image_id": image_id, "axis": axis_name, "unit": "deg", "sign": sign, "magnitude": magnitude, "gt_score": gt_score, "score": score, "margin_to_gt": gt_score - score, "phase": evidence.as_dict()})
        print(json.dumps({"image_id": image_id, "gt_score": gt_score, "row_count": len(rows)}), flush=True)
    result = {
        "stage": "goal_maplet_phase_basin_g19_c",
        "definition": "serialized_phase_score_around_ground_truth_in_surface_and_camera_frames",
        "query_count": len(paths),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "translation_magnitudes": translation_magnitudes,
        "rotation_magnitudes_deg": rotation_magnitudes,
        "summary": _sequence_metrics(rows, translation_magnitudes, rotation_magnitudes),
        "ground_truth": gt_rows,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "ground_truth"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
