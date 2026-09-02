"""Diversify raw finite-plane rankings using mapping-only merged families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _raw_to_family(raw: GeometryNativePlanarMap, merged: GeometryNativePlanarMap) -> np.ndarray:
    primitive_count = int(max(
        np.max(raw.member_primitive_rows, initial=-1),
        np.max(merged.member_primitive_rows, initial=-1),
    ) + 1)
    owner = np.full(primitive_count, -1, np.int32)
    for family in range(len(merged.plane_ids)):
        lo, hi = map(int, merged.member_offsets[family:family + 2])
        members = merged.member_primitive_rows[lo:hi]
        if np.any(owner[members] >= 0):
            raise ValueError("merged plane families overlap")
        owner[members] = family
    output = np.full(len(raw.plane_ids), -1, np.int32)
    for plane in range(len(raw.plane_ids)):
        lo, hi = map(int, raw.member_offsets[plane:plane + 2])
        families = np.unique(owner[raw.member_primitive_rows[lo:hi]])
        if len(families) != 1 or int(families[0]) < 0:
            raise ValueError("raw plane does not map to exactly one merged family")
        output[plane] = int(families[0])
    return output


def _family_first(
    ranking: list[int], family: np.ndarray, *, maximum_per_family: int = 1
) -> list[int]:
    if maximum_per_family < 1:
        raise ValueError("maximum_per_family must be positive")
    seen: dict[int, int] = {}
    preferred: list[int] = []
    deferred: list[int] = []
    for plane in ranking:
        group = int(family[int(plane)])
        count = seen.get(group, 0)
        seen[group] = count + 1
        if count >= maximum_per_family:
            deferred.append(int(plane))
        else:
            preferred.append(int(plane))
    return preferred + deferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--raw_planar_map", type=Path, required=True)
    parser.add_argument("--merged_planar_map", type=Path, required=True)
    parser.add_argument("--maximum_per_family", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite diversified plane ranking")
    raw = GeometryNativePlanarMap.load_npz(args.raw_planar_map)
    merged = GeometryNativePlanarMap.load_npz(args.merged_planar_map)
    family = _raw_to_family(raw, merged)
    report = json.loads(args.ranking.read_text())
    if (
        report.get("uses_pose_or_ground_truth") is not False
        or report.get("contains_postlabel_fields") is not False
        or int(report.get("plane_count", -1)) != len(family)
    ):
        raise ValueError("input plane ranking contract differs")
    duplicate_top5_before = duplicate_top5_after = 0
    for query in report["rows"]:
        for region in query["regions"]:
            planes = list(map(int, region["top10"]))
            scores = list(map(float, region["top10_scores"]))
            if len(planes) != len(scores) or len(set(planes)) != len(planes):
                raise ValueError("plane ranking rows differ")
            score_by_plane = dict(zip(planes, scores))
            duplicate_top5_before += len(planes[:5]) - len(set(family[planes[:5]].tolist()))
            reordered = _family_first(
                planes, family, maximum_per_family=int(args.maximum_per_family)
            )
            duplicate_top5_after += len(reordered[:5]) - len(set(family[reordered[:5]].tolist()))
            region["top10"] = reordered
            region["top10_scores"] = [score_by_plane[plane] for plane in reordered]
    report.update({
        "plane_family_diversity": "mapping_only_covisible_coplanar_family_capped_then_excess",
        "maximum_per_family": int(args.maximum_per_family),
        "raw_planar_map_file_sha256": file_sha256(args.raw_planar_map),
        "merged_planar_map_file_sha256": file_sha256(args.merged_planar_map),
        "merged_family_count": int(len(merged.plane_ids)),
        "duplicate_family_top5_slots_before": int(duplicate_top5_before),
        "duplicate_family_top5_slots_after": int(duplicate_top5_after),
        "uses_pose_or_ground_truth": False,
        "contains_postlabel_fields": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": len(report["rows"]),
        "merged_family_count": int(len(merged.plane_ids)),
        "duplicate_family_top5_slots_before": int(duplicate_top5_before),
        "duplicate_family_top5_slots_after": int(duplicate_top5_after),
        "output_file_sha256": file_sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
