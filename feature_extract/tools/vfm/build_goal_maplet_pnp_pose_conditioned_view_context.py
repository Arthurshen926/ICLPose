"""Select paired PnP candidates by nearby mapping-view RADIO context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _radio,
    _records,
    _region_tokens,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import (
    PlaneVisibilityAtlas,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def _pose_center_forward(pose_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, np.float64).reshape(4, 4)
    center = -pose[:3, :3].T @ pose[:3, 3]
    forward = pose[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
    return center, forward


def _candidate_context_score(
    pose_w2c: np.ndarray,
    query_descriptor: np.ndarray,
    view_centers: np.ndarray,
    view_forwards: np.ndarray,
    view_descriptors: np.ndarray,
    *,
    maximum_distance_m: float = 10.0,
    maximum_direction_degrees: float = 45.0,
    top_views: int = 4,
) -> tuple[float, int]:
    center, forward = _pose_center_forward(pose_w2c)
    distance = np.linalg.norm(np.asarray(view_centers) - center, axis=1)
    cosine_direction = np.clip(np.asarray(view_forwards) @ forward, -1.0, 1.0)
    eligible = (
        (distance <= float(maximum_distance_m))
        & (cosine_direction >= np.cos(np.deg2rad(float(maximum_direction_degrees))))
    )
    rows = np.flatnonzero(eligible)
    if len(rows) == 0:
        return -1.0, 0
    score = np.asarray(view_descriptors)[rows] @ np.asarray(query_descriptor)
    selected = np.sort(score)[-min(int(top_views), len(score)):]
    return float(np.mean(selected)), int(len(rows))


def _query_context(name: str, plane_dir: Path, records: dict[str, dict[str, object]]) -> np.ndarray:
    planes, _ = QueryPlaneRegions.load_npz(plane_dir / name)
    radio = _radio(name, records)
    total = np.zeros(1280, np.float64)
    weight = 0
    for region in range(len(planes.normals_camera)):
        tokens = _region_tokens(planes.labels, region)
        if len(tokens) == 0:
            continue
        descriptor = np.mean(radio[tokens], axis=0)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
        total += descriptor * len(tokens)
        weight += len(tokens)
    total /= max(weight, 1)
    total /= max(float(np.linalg.norm(total)), 1e-8)
    return total.astype(np.float32)


def _query_global_context(name: str, records: dict[str, dict[str, object]]) -> np.ndarray:
    radio = _radio(name, records)
    descriptor = np.mean(radio, axis=0)
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
    return descriptor.astype(np.float32)


def _load_global_view_field(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            key: np.asarray(data[key])
            for key in (
                "names",
                "global_radio_descriptors",
                "camera_centers_world",
                "camera_forwards_world",
                "contributor_file_sha256",
            )
        }
    if (
        metadata.get("artifact_type") != "goal_maplet_global_radio_mapping_view_field_v1"
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or len(arrays["names"]) != int(metadata.get("view_count", -1))
        or len(np.unique(arrays["names"].astype(str))) != len(arrays["names"])
        or arrays["global_radio_descriptors"].shape != (len(arrays["names"]), 1280)
        or arrays["camera_centers_world"].shape != (len(arrays["names"]), 3)
        or arrays["camera_forwards_world"].shape != (len(arrays["names"]), 3)
    ):
        raise ValueError("global RADIO view field differs")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path)
    parser.add_argument("--context_field", type=Path)
    parser.add_argument("--mapping_contributors", type=Path)
    parser.add_argument("--query_plane_dir", type=Path)
    parser.add_argument("--global_view_field", type=Path)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum_distance_m", type=float, default=10.0)
    parser.add_argument("--maximum_direction_degrees", type=float, default=45.0)
    parser.add_argument("--top_views", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite pose-conditioned context scores")
    top5, meta5 = _load_frozen_poses(args.top5_pose_inventory)
    top10, meta10 = _load_frozen_poses(args.top10_pose_inventory)
    names = top5["names"].astype(str)
    if not np.array_equal(names, top10["names"].astype(str)):
        raise ValueError("paired candidate names differ")
    using_global = args.global_view_field is not None
    if using_global:
        if any(value is not None for value in (
            args.visibility_atlas, args.context_field, args.mapping_contributors, args.query_plane_dir,
        )):
            raise ValueError("global view field mode cannot be mixed with planar context inputs")
        field, field_meta = _load_global_view_field(args.global_view_field)
        view_names = field["names"].astype(str).tolist()
        descriptors = np.asarray(field["global_radio_descriptors"], np.float32)
        centers = np.asarray(field["camera_centers_world"], np.float64)
        forwards = np.asarray(field["camera_forwards_world"], np.float64)
        atlas_meta: dict[str, object] = {}
        context_meta: dict[str, object] = {}
    else:
        if any(value is None for value in (
            args.visibility_atlas, args.context_field, args.mapping_contributors, args.query_plane_dir,
        )):
            raise ValueError("planar context mode requires atlas, field, contributors, and query planes")
        atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
        with np.load(args.context_field, allow_pickle=False) as data:
            context_meta = json.loads(str(data["metadata_json"].item()))
            context_arrays = {name: np.asarray(data[name]) for name in (
                "plane_offsets", "observation_context_descriptors",
            )}
        if (
            context_meta.get("artifact_type") != "goal_maplet_plane_observation_context_field_v1"
            or context_meta.get("visibility_atlas_content_sha256") != atlas_meta.get("content_sha256")
            or arrays_sha256(context_arrays) != context_meta.get("arrays_sha256")
            or not np.array_equal(context_arrays["plane_offsets"], atlas.plane_offsets)
        ):
            raise ValueError("view context field differs")
        observation_context = np.asarray(context_arrays["observation_context_descriptors"], np.float32)
        view_names = []
        view_descriptors = []
        for name in sorted(set(atlas.view_names.astype(str).tolist())):
            rows = np.flatnonzero(atlas.view_names.astype(str) == name)
            value = observation_context[rows[0]]
            if not np.allclose(observation_context[rows], value, atol=2e-7, rtol=0.0):
                raise ValueError("source-view context is inconsistent across plane observations")
            view_names.append(name)
            view_descriptors.append(value)
        centers = []
        forwards = []
        for name in view_names:
            with np.load(args.mapping_contributors / name, allow_pickle=False) as data:
                pose = np.asarray(data["pose_w2c"], np.float64)
            center, forward = _pose_center_forward(pose)
            centers.append(center); forwards.append(forward)
        centers = np.asarray(centers, np.float64)
        forwards = np.asarray(forwards, np.float64)
        descriptors = np.asarray(view_descriptors, np.float32)
    records = _records(args.radio_manifest)
    rows = []
    for index, name in enumerate(names.tolist()):
        query = (
            _query_global_context(name, records)
            if using_global
            else _query_context(name, args.query_plane_dir, records)
        )
        payload: dict[str, object] = {"name": name}
        scores = {}
        for branch, candidate in ((5, top5), (10, top10)):
            if bool(candidate["usable"][index]):
                score, count = _candidate_context_score(
                    candidate["pose_w2c"][index], query, centers, forwards, descriptors,
                    maximum_distance_m=float(args.maximum_distance_m),
                    maximum_direction_degrees=float(args.maximum_direction_degrees),
                    top_views=int(args.top_views),
                )
            else:
                score, count = -1.0, 0
            scores[branch] = score
            payload[f"top{branch}_context_score"] = score
            payload[f"top{branch}_eligible_mapping_view_count"] = count
        payload["selected_branch"] = 10 if scores[10] > scores[5] else 5
        rows.append(payload)
    report = {
        "artifact_type": (
            "goal_maplet_pnp_pose_conditioned_global_radio_selection_v1"
            if using_global
            else "goal_maplet_pnp_pose_conditioned_view_context_selection_v1"
        ),
        "query_count": int(len(rows)),
        "query_pose_or_ground_truth_read": False,
        "selection_rule": (
            "maximum_mean_top4_nearby_mapping_view_global_radio_cosine_tie_top5"
            if using_global
            else "maximum_mean_top4_nearby_mapping_view_planar_context_cosine_tie_top5"
        ),
        "maximum_distance_m": float(args.maximum_distance_m),
        "maximum_direction_degrees": float(args.maximum_direction_degrees),
        "top_views": int(args.top_views),
        "top5_pose_inventory_file_sha256": file_sha256(args.top5_pose_inventory),
        "top5_pose_inventory_content_sha256": meta5.get("content_sha256"),
        "top10_pose_inventory_file_sha256": file_sha256(args.top10_pose_inventory),
        "top10_pose_inventory_content_sha256": meta10.get("content_sha256"),
        "visibility_atlas_file_sha256": None if using_global else file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "context_field_file_sha256": None if using_global else file_sha256(args.context_field),
        "context_field_content_sha256": context_meta.get("content_sha256"),
        "global_view_field_file_sha256": (
            file_sha256(args.global_view_field) if using_global else None
        ),
        "global_view_field_content_sha256": (
            field_meta.get("content_sha256") if using_global else None
        ),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "mapping_view_count": int(len(view_names)),
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
