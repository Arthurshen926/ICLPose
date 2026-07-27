"""Select scene-covering, view-diverse mapping images for V6 atlas baking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--contribution_dirs", nargs="+", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--output_image_ids", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_views", type=int, default=96)
    parser.add_argument("--target_views_per_maplet", type=int, default=3)
    parser.add_argument("--exclude_trajectories", nargs="*", default=[])
    parser.add_argument("--view_diversity_weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _camera_centres(pose_file: Path) -> dict[str, np.ndarray]:
    result = {}
    for record in parse_cambridge_pose_file(pose_file):
        pose = np.asarray(record.pose_w2c, dtype=np.float64)
        result[record.image_id] = -pose[:3, :3].T @ pose[:3, 3]
    return result


def _maplet_by_element(maplets: VfmSurfaceMapletBank) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for row in range(len(maplets)):
        begin = int(maplets.support_offsets[row])
        end = int(maplets.support_offsets[row + 1])
        for element_id in np.unique(maplets.support_element_ids[begin:end]).tolist():
            result.setdefault(int(element_id), []).append(row)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_ids = Path(args.output_image_ids)
    output_summary = Path(args.summary_json)
    if (
        (output_ids.exists() or output_summary.exists())
        and not bool(args.force)
    ):
        raise FileExistsError("refusing to overwrite V6 view selection")
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    centres = _camera_centres(Path(args.mapping_pose_file))
    inverse = _maplet_by_element(maplets)
    excluded = {str(value) for value in args.exclude_trajectories}
    candidates = []
    seen_images = set()
    for directory in args.contribution_dirs:
        for path in sorted(Path(directory).glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                if "element_ids" not in data.files:
                    continue
                image_id = (
                    str(data["image_id"].item())
                    if "image_id" in data.files
                    else path.stem.replace("__", "/", 1)
                )
                if image_id in seen_images or image_id not in centres:
                    continue
                trajectory = image_id.split("/", 1)[0]
                if trajectory in excluded:
                    continue
                rows = sorted(
                    {
                        maplet_row
                        for element_id in np.unique(data["element_ids"]).tolist()
                        for maplet_row in inverse.get(int(element_id), ())
                    }
                )
            if not rows:
                continue
            seen_images.add(image_id)
            rows_array = np.asarray(rows, dtype=np.int64)
            directions = (
                centres[image_id][None]
                - np.asarray(maplets.centers[rows_array], dtype=np.float64)
            )
            directions /= np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), 1e-8
            )
            candidates.append(
                {
                    "image_id": image_id,
                    "trajectory": trajectory,
                    "maplet_rows": rows_array,
                    "directions": directions.astype(np.float32),
                }
            )
    if not candidates:
        raise ValueError("no usable contribution records")
    target = max(int(args.target_views_per_maplet), 1)
    support_count = np.zeros((len(maplets),), dtype=np.int32)
    selected_directions: list[list[np.ndarray]] = [
        [] for _ in range(len(maplets))
    ]
    selected = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < int(args.maximum_views):
        best_index = -1
        best_score = 0.0
        for index in remaining:
            candidate = candidates[index]
            rows = candidate["maplet_rows"]
            deficit = np.maximum(target - support_count[rows], 0)
            coverage_score = float(
                np.sum((deficit > 0).astype(np.float64))
                + 2.0 * np.sum(support_count[rows] == 0)
            )
            diversity_score = 0.0
            for local_index, maplet_row in enumerate(rows.tolist()):
                previous = selected_directions[maplet_row]
                if not previous:
                    continue
                cosine = np.clip(
                    np.asarray(previous)
                    @ candidate["directions"][local_index],
                    -1.0,
                    1.0,
                )
                diversity_score += float(np.min(np.arccos(cosine)) / np.pi)
            score = coverage_score + float(args.view_diversity_weight) * diversity_score
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            break
        remaining.remove(best_index)
        candidate = candidates[best_index]
        selected.append(candidate)
        for local_index, maplet_row in enumerate(
            candidate["maplet_rows"].tolist()
        ):
            support_count[maplet_row] += 1
            selected_directions[maplet_row].append(
                candidate["directions"][local_index]
            )
    selected_ids = [str(value["image_id"]) for value in selected]
    covered = support_count > 0
    reached_target = support_count >= target
    angular_spans = []
    for directions in selected_directions:
        if len(directions) < 2:
            continue
        values = np.asarray(directions)
        angular_spans.append(
            float(
                np.max(
                    np.arccos(
                        np.clip(values @ values.T, -1.0, 1.0)
                    )
                )
            )
        )
    report = {
        "stage": "v6_atlas_baking_view_set_cover",
        "candidate_view_count": len(candidates),
        "selected_view_count": len(selected),
        "selected_trajectory_ids": sorted(
            {str(value["trajectory"]) for value in selected}
        ),
        "excluded_trajectory_ids": sorted(excluded),
        "maplet_count": len(maplets),
        "covered_maplet_count": int(np.sum(covered)),
        "covered_maplet_fraction": float(np.mean(covered)),
        "target_views_per_maplet": target,
        "target_reached_maplet_count": int(np.sum(reached_target)),
        "target_reached_maplet_fraction": float(np.mean(reached_target)),
        "support_quantiles": {
            "median": float(np.median(support_count[covered]))
            if np.any(covered)
            else 0.0,
            "p10": float(np.quantile(support_count[covered], 0.1))
            if np.any(covered)
            else 0.0,
            "p90": float(np.quantile(support_count[covered], 0.9))
            if np.any(covered)
            else 0.0,
        },
        "maximum_pairwise_view_angle_deg": {
            "median": float(np.degrees(np.median(angular_spans)))
            if angular_spans
            else 0.0,
            "p90": float(np.degrees(np.quantile(angular_spans, 0.9)))
            if angular_spans
            else 0.0,
        },
        "selected_image_ids": selected_ids,
    }
    output_ids.parent.mkdir(parents=True, exist_ok=True)
    output_ids.write_text("\n".join(selected_ids) + "\n")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
