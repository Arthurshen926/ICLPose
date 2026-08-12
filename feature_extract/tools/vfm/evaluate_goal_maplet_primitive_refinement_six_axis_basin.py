"""Measure the production primitive-VFM refiner's six-axis convergence basin.

The diagnostic starts from declared perturbations around ground truth, but the
refiner and its exact-score acceptance rule never receive ground truth.  It is
therefore an oracle attribution experiment, not a deployable localization path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.primitive_pose_refiner import (
    refine_pose_with_primitive_vfm_score,
)
from feature_extract.vfm.localization_goal_maplet.sparse_vfm_pose_likelihood import (
    score_pose_conditioned_sparse_primitives,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


AXES = ("rx", "ry", "rz", "tx", "ty", "tz")


def _parse_positive_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in str(value).split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise ValueError("perturbation levels must be positive")
    return result


def _trial_definitions(
    translation_levels_m: Sequence[float],
    rotation_levels_deg: Sequence[float],
    *,
    include_zero: bool,
    signs: Sequence[int] = (-1, 1),
) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    if include_zero:
        trials.append({
            "axis": "zero", "kind": "zero", "sign": 0, "magnitude": 0.0,
            "delta": np.zeros((6,), dtype=np.float64),
        })
    for axis_index, axis in enumerate(AXES):
        levels = rotation_levels_deg if axis_index < 3 else translation_levels_m
        for magnitude in levels:
            for sign in signs:
                delta = np.zeros((6,), dtype=np.float64)
                delta[axis_index] = (
                    np.deg2rad(float(magnitude)) if axis_index < 3
                    else float(magnitude)
                ) * int(sign)
                trials.append({
                    "axis": axis,
                    "kind": "rotation" if axis_index < 3 else "translation",
                    "sign": int(sign),
                    "magnitude": float(magnitude),
                    "delta": delta,
                })
    return trials


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=np.asarray(data["camera_params"], dtype=np.float64),
        )


def _metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(np.asarray(data["metadata_json"]).item()))


def _stable_order(image_id: str) -> str:
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def _select_queries(
    contributors: Path,
    *,
    trajectories: set[str] | None,
    maximum_per_trajectory: int,
    maximum_queries: int,
    shard_index: int,
    shard_count: int,
) -> list[tuple[str, Path, dict[str, object]]]:
    records = []
    for path in contributors.glob("*.npz"):
        metadata = _metadata(path)
        image_id = str(metadata["image_id"])
        trajectory = image_id.split("/", 1)[0]
        if trajectories is None or trajectory in trajectories:
            records.append((image_id, path, metadata))
    by_trajectory: dict[str, list[tuple[str, Path, dict[str, object]]]] = {}
    for record in records:
        by_trajectory.setdefault(record[0].split("/", 1)[0], []).append(record)
    selected = []
    for trajectory in sorted(by_trajectory):
        values = sorted(by_trajectory[trajectory], key=lambda value: _stable_order(value[0]))
        if int(maximum_per_trajectory) > 0:
            values = values[: int(maximum_per_trajectory)]
        selected.extend(values)
    selected.sort(key=lambda value: value[0])
    if int(maximum_queries) > 0:
        selected = selected[: int(maximum_queries)]
    if int(shard_count) <= 0 or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("invalid query shard")
    return selected[int(shard_index) :: int(shard_count)]


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        return [float("nan"), float("nan")]
    probability = float(successes) / float(count)
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    radius = z * sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count**2)
    ) / denominator
    return [float(center - radius), float(center + radius)]


def _group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    strict = sum(
        float(row["final_translation_m"]) <= 0.5
        and float(row["final_rotation_deg"]) <= 5.0
        for row in rows
    )
    loose = sum(
        float(row["final_translation_m"]) <= 1.0
        and float(row["final_rotation_deg"]) <= 10.0
        for row in rows
    )
    improved = sum(bool(row["improved_normalized_error"]) for row in rows)
    monotonic = sum(bool(row["exact_score_monotonic"]) for row in rows)
    return {
        "trial_count": count,
        "strict_success_count": int(strict),
        "strict_success_rate": float(strict / count) if count else 0.0,
        "strict_success_wilson95": _wilson(strict, count),
        "loose_success_count": int(loose),
        "loose_success_rate": float(loose / count) if count else 0.0,
        "loose_success_wilson95": _wilson(loose, count),
        "improved_normalized_error_count": int(improved),
        "exact_score_monotonic_count": int(monotonic),
        "accepted_step_mean": (
            float(np.mean([row["accepted_steps"] for row in rows])) if rows else 0.0
        ),
        "final_translation_median_m": (
            float(np.median([row["final_translation_m"] for row in rows]))
            if rows else None
        ),
        "final_rotation_median_deg": (
            float(np.median([row["final_rotation_deg"] for row in rows]))
            if rows else None
        ),
    }


def _score_history_is_monotonic(
    initial_score: float, history: Sequence[dict[str, object]], final_score: float,
    tolerance: float = 1.0e-9,
) -> bool:
    previous = float(initial_score)
    for value in history:
        current = float(value["score"])
        if current < previous - float(tolerance):
            return False
        previous = current
    return float(final_score) >= previous - float(tolerance)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--include_trajectories", nargs="*")
    parser.add_argument("--translation_levels_m", default="0.1,0.25,0.5,1,2")
    parser.add_argument("--rotation_levels_deg", default="2,5,10,20")
    parser.add_argument("--exclude_zero", action="store_true")
    parser.add_argument("--positive_only", action="store_true")
    parser.add_argument("--translation_steps_m", default="0.60,0.40,0.25,0.12")
    parser.add_argument("--rotation_steps_deg", default="5,3,2,1")
    parser.add_argument("--iterations_per_scale", type=int, default=2)
    parser.add_argument("--minimum_score_improvement", type=float, default=1.0e-6)
    parser.add_argument("--maximum_splat_radius_tokens", type=int, default=0)
    parser.add_argument("--maximum_queries_per_trajectory", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite six-axis basin report")
    physical_path = Path(args.physical_map)
    field_path = Path(args.canonical_field)
    mapper_path = Path(args.surface_mapper)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    mapper, _ = load_surface_maplet_mapper(mapper_path, device=str(args.device))
    translation_levels = _parse_positive_floats(args.translation_levels_m)
    rotation_levels = _parse_positive_floats(args.rotation_levels_deg)
    translation_steps = _parse_positive_floats(args.translation_steps_m)
    rotation_steps = _parse_positive_floats(args.rotation_steps_deg)
    if len(translation_steps) != len(rotation_steps):
        raise ValueError("refinement trust-region schedules differ")
    trials = _trial_definitions(
        translation_levels,
        rotation_levels,
        include_zero=not bool(args.exclude_zero),
        signs=(1,) if bool(args.positive_only) else (-1, 1),
    )
    trajectories = (
        set(str(value) for value in args.include_trajectories)
        if args.include_trajectories else None
    )
    queries = _select_queries(
        Path(args.contributors),
        trajectories=trajectories,
        maximum_per_trajectory=int(args.maximum_queries_per_trajectory),
        maximum_queries=int(args.maximum_queries),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    if not queries:
        raise ValueError("six-axis basin selection contains no queries")

    rows: list[dict[str, object]] = []
    for image_id, contributor_path, metadata in queries:
        labels = ContributorLabels.load_npz(contributor_path)
        camera = _camera(contributor_path)
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        query = mapper.project(raw).measurement_context
        query_flat = query.transpose(1, 2, 0).reshape(-1, query.shape[0])

        def score(poses: np.ndarray) -> np.ndarray:
            return score_pose_conditioned_sparse_primitives(
                poses,
                query_flat,
                field.primitive_rows,
                field.codes,
                field.confidence,
                physical,
                camera,
                token_height=int(query.shape[1]),
                token_width=int(query.shape[2]),
                primitives_per_child=0,
                batch_size=16,
                maximum_splat_radius_tokens=int(args.maximum_splat_radius_tokens),
                score_semantics="fixed_grid",
                device=str(args.device),
            ).scores

        for trial in trials:
            initial_pose = se3_exp(np.asarray(trial["delta"])) @ labels.pose_w2c
            initial_error = pnp_pose_error(initial_pose, labels.pose_w2c)
            refined = refine_pose_with_primitive_vfm_score(
                initial_pose,
                score,
                translation_steps_m=translation_steps,
                rotation_steps_deg=rotation_steps,
                iterations_per_scale=int(args.iterations_per_scale),
                minimum_score_improvement=float(args.minimum_score_improvement),
            )
            final_error = pnp_pose_error(refined.pose_w2c, labels.pose_w2c)
            initial_normalized = (
                float(initial_error.translation_m) / 0.5
                + float(initial_error.rotation_deg) / 5.0
            )
            final_normalized = (
                float(final_error.translation_m) / 0.5
                + float(final_error.rotation_deg) / 5.0
            )
            row = {
                "image_id": image_id,
                "trajectory_id": image_id.split("/", 1)[0],
                "perturbation_axis": str(trial["axis"]),
                "perturbation_kind": str(trial["kind"]),
                "perturbation_sign": int(trial["sign"]),
                "perturbation_magnitude": float(trial["magnitude"]),
                "left_camera_frame_twist": np.asarray(trial["delta"]).tolist(),
                "initial_translation_m": float(initial_error.translation_m),
                "initial_rotation_deg": float(initial_error.rotation_deg),
                "final_translation_m": float(final_error.translation_m),
                "final_rotation_deg": float(final_error.rotation_deg),
                "improved_normalized_error": bool(final_normalized < initial_normalized),
                "initial_score": float(refined.initial_score),
                "final_score": float(refined.final_score),
                "score_gain": float(refined.final_score - refined.initial_score),
                "accepted_steps": int(refined.accepted_steps),
                "exact_score_monotonic": _score_history_is_monotonic(
                    refined.initial_score, refined.history, refined.final_score
                ),
                "history": list(refined.history),
            }
            if not row["exact_score_monotonic"]:
                raise AssertionError("production exact-score acceptance regressed")
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "history"}))

    axis_levels = sorted({
        (str(row["perturbation_axis"]), float(row["perturbation_magnitude"]))
        for row in rows
    })
    directions = sorted({
        (
            str(row["perturbation_axis"]), float(row["perturbation_magnitude"]),
            int(row["perturbation_sign"]),
        )
        for row in rows
    })
    payload = {
        "artifact_type": "goal_maplet_primitive_vfm_six_axis_basin_v1",
        "oracle_only_initialization": True,
        "refinement_uses_ground_truth": False,
        "perturbation_coordinates": "left_multiplicative_camera_frame_se3_twist",
        "surface_score_semantics": "all_geometry_occludes_fixed_query_grid_primitive_vfm",
        "exact_score_acceptance_is_monotonic": all(
            bool(row["exact_score_monotonic"]) for row in rows
        ),
        "physical_map": str(physical_path),
        "physical_map_sha256": file_sha256(physical_path),
        "canonical_field": str(field_path),
        "canonical_field_sha256": file_sha256(field_path),
        "surface_mapper": str(mapper_path),
        "surface_mapper_sha256": file_sha256(mapper_path),
        "query_count": len(queries),
        "query_image_ids_sha256": ordered_id_sha256([value[0] for value in queries]),
        "trial_count": len(rows),
        "configuration": {
            "translation_levels_m": list(translation_levels),
            "rotation_levels_deg": list(rotation_levels),
            "signs": [1] if bool(args.positive_only) else [-1, 1],
            "include_zero": not bool(args.exclude_zero),
            "translation_steps_m": list(translation_steps),
            "rotation_steps_deg": list(rotation_steps),
            "iterations_per_scale": int(args.iterations_per_scale),
            "minimum_score_improvement": float(args.minimum_score_improvement),
            "maximum_splat_radius_tokens": int(args.maximum_splat_radius_tokens),
        },
        "summary": _group_summary(rows),
        "by_axis_level": {
            f"{axis}_{magnitude:g}": _group_summary([
                row for row in rows
                if row["perturbation_axis"] == axis
                and float(row["perturbation_magnitude"]) == magnitude
            ])
            for axis, magnitude in axis_levels
        },
        "by_signed_direction": {
            f"{axis}_{magnitude:g}_{sign:+d}": _group_summary([
                row for row in rows
                if row["perturbation_axis"] == axis
                and float(row["perturbation_magnitude"]) == magnitude
                and int(row["perturbation_sign"]) == sign
            ])
            for axis, magnitude, sign in directions
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
