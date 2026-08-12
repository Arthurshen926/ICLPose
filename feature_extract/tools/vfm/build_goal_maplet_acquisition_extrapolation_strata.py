"""Build fixed OOF strata measuring departure from mapping acquisition poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def _bin(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    if len(labels) != len(edges) + 1:
        raise ValueError("stratum labels and edges differ")
    return str(labels[int(np.searchsorted(np.asarray(edges), value, side="right"))])


def _forward_world(rotation_w2c: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation_w2c, dtype=np.float64).reshape(3, 3).T[:, 2]
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _fold_rows(records, fold: dict[str, object]) -> list[dict[str, object]]:
    held = set(str(value) for value in fold["held_query_trajectories"])
    mapping = [
        value for value in records if value.image_id.split("/", 1)[0] not in held
    ]
    queries = [
        value for value in records if value.image_id.split("/", 1)[0] in held
    ]
    if len(mapping) != int(fold["mapping_count"]) or len(queries) != int(
        fold["query_count"]
    ):
        raise ValueError("acquisition stratum fold differs from protocol")
    mapping_center = np.stack([value.camera_center for value in mapping])
    mapping_forward = np.stack([_forward_world(value.rotation_w2c) for value in mapping])
    rows = []
    for query in queries:
        difference = mapping_center - query.camera_center[None]
        distance = np.linalg.norm(difference, axis=1)
        nearest = int(np.argmin(distance))
        query_forward = _forward_world(query.rotation_w2c)
        nearest_cosine = float(np.clip(
            np.dot(query_forward, mapping_forward[nearest]), -1.0, 1.0
        ))
        nearest_angle = float(np.degrees(np.arccos(nearest_cosine)))
        all_cosine = np.clip(mapping_forward @ query_forward, -1.0, 1.0)
        minimum_angle = float(np.degrees(np.arccos(float(np.max(all_cosine)))))
        height_offset = abs(float(
            query.camera_center[1] - mapping_center[nearest, 1]
        ))
        center_distance = float(distance[nearest])
        outside_proxy = bool(
            center_distance > 2.0 or nearest_angle > 60.0 or height_offset > 0.75
        )
        rows.append({
            "image_id": query.image_id,
            "trajectory_id": query.image_id.split("/", 1)[0],
            "fold_id": str(fold["fold_id"]),
            "nearest_mapping_image_id": mapping[nearest].image_id,
            "nearest_mapping_center_distance_m": center_distance,
            "nearest_mapping_forward_angle_deg": nearest_angle,
            "minimum_mapping_forward_angle_deg": minimum_angle,
            "nearest_mapping_height_offset_m": height_offset,
            "center_distance_stratum": _bin(
                center_distance, (0.5, 2.0, 5.0),
                ("overlap_le_0.5m", "near_0.5_2m", "far_2_5m", "extrapolated_gt_5m"),
            ),
            "view_direction_stratum": _bin(
                nearest_angle, (30.0, 60.0, 90.0),
                ("same_le_30deg", "oblique_30_60deg", "large_60_90deg", "opposite_ge_90deg"),
            ),
            "height_offset_stratum": _bin(
                height_offset, (0.25, 0.75),
                ("same_le_0.25m", "shifted_0.25_0.75m", "large_gt_0.75m"),
            ),
            "outside_mapping_view_manifold_proxy": outside_proxy,
        })
    return rows


def build_strata(protocol_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("artifact_type") != "goal_maplet_official_train_oof_protocol_v1":
        raise ValueError("unsupported OOF protocol")
    train = protocol["official_train"]
    pose_path = Path(train["pose_file"])
    if file_sha256(pose_path) != str(train["pose_file_sha256"]):
        raise ValueError("official train poses changed after protocol creation")
    records = parse_cambridge_pose_file(pose_path)
    rows = [
        row for fold in protocol["development"]["folds"]
        for row in _fold_rows(records, fold)
    ]
    image_ids = [str(row["image_id"]) for row in rows]
    if (
        len(image_ids) != len(set(image_ids))
        or len(image_ids) != int(train["count"])
        or ordered_id_sha256(image_ids) != str(train["image_ids_sha256"])
    ):
        raise ValueError("acquisition strata do not cover official train once")
    fields = (
        "center_distance_stratum", "view_direction_stratum",
        "height_offset_stratum", "outside_mapping_view_manifold_proxy",
    )
    return {
        "artifact_type": "goal_maplet_oof_acquisition_extrapolation_strata_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "oracle_pose_used_for_evaluation_stratification_only": True,
        "mapping_view_prior_claim": "map_conditioned_acquisition_prior_not_image_retrieval",
        "vertical_axis": "Cambridge_world_y",
        "outside_mapping_view_manifold_proxy_definition": (
            "nearest_center_distance>2m or nearest_forward_angle>60deg or "
            "nearest_height_offset>0.75m"
        ),
        "query_count": len(rows),
        "image_ids_sha256": ordered_id_sha256(image_ids),
        "counts": {
            field: {
                str(value): sum(row[field] == value for row in rows)
                for value in sorted({row[field] for row in rows}, key=str)
            }
            for field in fields
        },
        "rows": sorted(rows, key=lambda value: str(value["image_id"])),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite acquisition strata")
    payload = build_strata(Path(args.protocol))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
