"""Build pose/label-free query-plane -> finite-map-plane RADIO rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def _validate_plane_inventory(
    planar_map: GeometryNativePlanarMap,
    incidence_metadata: dict[str, object],
    incidence_plane_count: int,
) -> None:
    map_count = len(planar_map.plane_ids)
    if (
        map_count != int(incidence_plane_count)
        or int(incidence_metadata.get("plane_count", -1)) != int(incidence_plane_count)
    ):
        raise ValueError(
            f"incidence/planar-map row mismatch: incidence={incidence_plane_count}, "
            f"map={map_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidence", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--retrieval_dir", type=Path, required=True)
    parser.add_argument("--expected_retrieval_physical_map_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite RADIO plane ranking")
    with np.load(args.incidence, allow_pickle=False) as data:
        offsets = np.asarray(data["plane_cell_offsets"], np.int64)
        child_rows = np.asarray(data["child_rows"], np.int64)
        mass = np.asarray(data["surface_mass_m2"], np.float64)
        incidence_meta = json.loads(str(data["metadata_json"].item()))
    plane_count = offsets.size - 1
    planar_map = GeometryNativePlanarMap.load_npz(args.planar_map)
    _validate_plane_inventory(planar_map, incidence_meta, plane_count)
    child_to_planes: dict[int, list[tuple[int, float]]] = {}
    for plane in range(plane_count):
        lo, hi = map(int, offsets[plane:plane + 2])
        total = max(float(mass[lo:hi].sum()), 1e-15)
        for child, value in zip(child_rows[lo:hi].tolist(), mass[lo:hi].tolist()):
            child_to_planes.setdefault(int(child), []).append((plane, float(value / total)))
    rows = []
    retrieval_hashes = []
    for path in sorted(args.query_plane_dir.glob("*.npz")):
        name = path.name
        retrieval = args.retrieval_dir / name
        if not retrieval.exists():
            continue
        planes, _ = QueryPlaneRegions.load_npz(path)
        with np.load(retrieval, allow_pickle=False) as data:
            token_xy = np.asarray(data["token_xy"], np.int64)
            token_child = np.asarray(data["token_child_rows"], np.int64)
            token_probability = np.asarray(data["token_child_probabilities"], np.float64)
            retrieval_physical_map = str(data["physical_map_sha256"].item())
        if retrieval_physical_map != args.expected_retrieval_physical_map_sha256:
            raise ValueError("retrieval physical-map identity differs")
        retrieval_hashes.append(file_sha256(retrieval))
        records = []
        for region in range(len(planes.normals_camera)):
            mask = planes.labels == region
            block = mask.reshape(36, 4, 64, 4).mean((1, 3))
            token_weight = block[token_xy[:, 1], token_xy[:, 0]]
            child_mass: dict[int, float] = {}
            repeated_weight = np.repeat(token_weight, token_child.shape[1])
            for child, probability, weight in zip(
                token_child.reshape(-1).tolist(),
                token_probability.reshape(-1).tolist(),
                repeated_weight.tolist(),
            ):
                if child >= 0 and weight > 0:
                    child_mass[int(child)] = child_mass.get(int(child), 0.0) + probability * weight
            total = sum(child_mass.values())
            score = np.zeros(plane_count, np.float64)
            if total > 0:
                for child, value in child_mass.items():
                    for plane, fraction in child_to_planes.get(child, ()):
                        score[plane] += np.sqrt(value / total * fraction)
            ranking = np.lexsort((np.arange(plane_count), -score))[: int(args.topk)]
            records.append({
                "region": region,
                "pixels": int(mask.sum()),
                "top10": ranking.tolist(),
                "top10_scores": score[ranking].tolist(),
            })
        rows.append({"image": name, "query_plane_count": len(records), "regions": records})
    report = {
        "artifact_type": "goal_maplet_pose_free_radio_to_physical_plane_ranking_v2",
        "query_count": len(rows),
        "plane_count": int(plane_count),
        "uses_pose_or_ground_truth": False,
        "contains_postlabel_fields": False,
        "incidence_file_sha256": file_sha256(args.incidence),
        "incidence_content_sha256": incidence_meta.get("content_sha256"),
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "planar_map_content_sha256": planar_map.metadata.get("content_sha256"),
        "retrieval_physical_map_sha256": args.expected_retrieval_physical_map_sha256,
        "retrieval_file_sha256_in_order": retrieval_hashes,
        "topk": int(args.topk),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "retrieval_file_sha256_in_order")}, indent=2))


if __name__ == "__main__":
    main()
