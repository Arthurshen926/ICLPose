"""Evaluate anchor-free RADIO-final maplet retrieval against GT visibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_feature_field import (
    _retrieved_maplets,
)
from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
    _load_raw_final,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.continuous_surface_alignment import (
    select_visible_maplets,
)
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--query_camera_manifest", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--metric_mapper_checkpoint", default="")
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--surface_feature_field", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum_maplets", type=int, default=12)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite retrieval report")
    bank = SurfaceRetrievalMapletBank.load_npz(Path(args.maplets))
    field = SurfaceFeatureField.load_npz(Path(args.surface_feature_field))
    mapper, metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    metric = (
        load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device=str(args.device)
        )
        if str(args.metric_mapper_checkpoint)
        else None
    )
    if (
        bank.metadata.get("feature_space") == "surface_metric_radio_final"
        and metric is None
    ):
        raise ValueError("metric-space maplets require --metric_mapper_checkpoint")
    region_config = RadioFinalRegionConfig(
        pool_sizes=tuple(metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))),
    )
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.query_pose_file))
    }
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.query_camera_manifest)
    )
    records = [
        record
        for record in manifest.records
        if record.image_id in pose_by_image and record.image_id in camera_by_image
    ]
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    rows: list[dict[str, object]] = []
    for record in records:
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        mapped = mapper.project(raw).measurement_context
        if metric is not None:
            mapped = metric.project_map(mapped)
        retrieved, diagnostics = _retrieved_maplets(
            raw,
            mapped,
            bank,
            region_config,
            int(args.maximum_maplets),
        )
        visible = select_visible_maplets(
            field,
            pose_by_image[record.image_id],
            camera_by_image[record.image_id],
            maximum_maplets=int(args.maximum_maplets),
        )
        intersection = np.intersect1d(retrieved, visible)
        rows.append(
            {
                "image_id": record.image_id,
                "retrieved_maplet_ids": retrieved.tolist(),
                "visible_maplet_ids": visible.tolist(),
                "intersection_count": int(intersection.size),
                "any_visible_hit": bool(intersection.size > 0),
                "visible_recall": float(intersection.size / max(visible.size, 1)),
                "retrieved_precision": float(intersection.size / max(retrieved.size, 1)),
                "retrieval": diagnostics,
            }
        )
    any_hit = np.asarray([row["any_visible_hit"] for row in rows], dtype=np.float64)
    visible_recall = np.asarray([row["visible_recall"] for row in rows], dtype=np.float64)
    precision = np.asarray(
        [row["retrieved_precision"] for row in rows], dtype=np.float64
    )
    report = {
        "stage": "anchor_free_radio_final_maplet_retrieval_visibility",
        "query_count": len(rows),
        "maximum_maplets": int(args.maximum_maplets),
        "any_visible_hit": float(np.mean(any_hit)) if len(rows) else 0.0,
        "visible_recall_mean": float(np.mean(visible_recall)) if len(rows) else 0.0,
        "visible_recall_median": (
            float(np.median(visible_recall)) if len(rows) else 0.0
        ),
        "retrieved_precision_mean": float(np.mean(precision)) if len(rows) else 0.0,
        "camera_audit": camera_audit,
        "production_contract": {
            "representation": "compact_radio_final_mixture_per_maplet",
            "uses_ground_truth_for_visibility_evaluation_only": True,
            "uses_stable_anchor_identity": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
