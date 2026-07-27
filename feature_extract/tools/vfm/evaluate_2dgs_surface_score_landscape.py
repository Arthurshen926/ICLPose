"""Audit the local pose score landscape before running an optimizer.

Ground truth and oracle-visible maplets are used only for this diagnostic. A
valid fine representation must place the GT pose near a local maximum on all
three translation axes and give the correct direction at small perturbations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
    _load_raw_final,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.continuous_surface_alignment import (
    ContinuousSurfaceAlignmentConfig,
    _left_pose_step,
    score_surface_alignment,
    select_render_samples,
    select_visible_maplets,
)
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.highres_surface_metric_decoder import (
    decode_highres_surface_metric,
    load_highres_surface_metric_decoder,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--query_camera_manifest", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--metric_mapper_checkpoint", default="")
    parser.add_argument("--highres_metric_decoder_checkpoint", default="")
    parser.add_argument("--query_image_root", default="")
    parser.add_argument("--surface_feature_field", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--translation_offsets_m", default="-0.30,-0.20,-0.10,-0.05,0,0.05,0.10,0.20,0.30")
    parser.add_argument("--maximum_maplets", type=int, default=12)
    parser.add_argument("--maximum_samples", type=int, default=2048)
    parser.add_argument("--max_queries", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite score-landscape report")
    offsets = np.asarray(
        [float(value) for value in str(args.translation_offsets_m).split(",")],
        dtype=np.float64,
    )
    zero_rows = np.flatnonzero(np.isclose(offsets, 0.0))
    if zero_rows.size != 1 or not np.any(offsets < 0.0) or not np.any(offsets > 0.0):
        raise ValueError("translation offsets must contain one zero and both signs")
    zero_row = int(zero_rows[0])
    field = SurfaceFeatureField.load_npz(Path(args.surface_feature_field))
    mapper, _metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    metric = (
        load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device=str(args.device)
        )
        if str(args.metric_mapper_checkpoint)
        else None
    )
    highres_decoder = (
        load_highres_surface_metric_decoder(
            Path(args.highres_metric_decoder_checkpoint), device=str(args.device)
        )[0]
        if str(args.highres_metric_decoder_checkpoint)
        else None
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
    ][: int(args.max_queries)]
    config = ContinuousSurfaceAlignmentConfig(
        maximum_maplets=int(args.maximum_maplets),
        maximum_samples=int(args.maximum_samples),
    )
    rows: list[dict[str, object]] = []
    for record in records:
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        branch = field.metadata.get("feature_branch")
        if branch == "raw_radio_final":
            feature = raw
        elif branch == "highres_metric_decoded":
            if highres_decoder is None or not str(args.query_image_root):
                raise ValueError(
                    "highres field requires decoder checkpoint and query image root"
                )
            bgr = cv2.imread(
                str(Path(args.query_image_root) / record.image_id),
                cv2.IMREAD_COLOR,
            )
            if bgr is None:
                raise ValueError(f"failed to decode query RGB for {record.image_id}")
            feature = decode_highres_surface_metric(
                highres_decoder,
                raw,
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                device=str(args.device),
            )["fine"]
        else:
            feature = mapper.project(raw).measurement_context
            if (
                branch == "metric_mapped"
                or (
                    "feature_branch" not in field.metadata
                    and metric is not None
                )
            ):
                if metric is None:
                    raise ValueError("metric field requires --metric_mapper_checkpoint")
                feature = metric.project_map(feature)
        target = pose_by_image[record.image_id]
        camera = camera_by_image[record.image_id]
        maplets = select_visible_maplets(
            field, target, camera, maximum_maplets=int(args.maximum_maplets)
        )
        samples = select_render_samples(
            field,
            target,
            camera,
            maplets,
            feature_height=int(feature.shape[1]),
            feature_width=int(feature.shape[2]),
            config=config,
        )
        axis_scores: list[list[float]] = []
        axis_gt_max: list[bool] = []
        axis_small_direction: list[bool] = []
        for axis in (3, 4, 5):
            scores = [
                score_surface_alignment(
                    field,
                    feature,
                    _left_pose_step(target, axis, float(offset)),
                    camera,
                    samples,
                    device=str(args.device),
                )
                for offset in offsets.tolist()
            ]
            values = np.asarray(scores, dtype=np.float64)
            nonzero = np.arange(offsets.size) != zero_row
            axis_scores.append(scores)
            axis_gt_max.append(bool(values[zero_row] >= np.max(values[nonzero])))
            negative_near = int(np.argmax(offsets[offsets < 0.0]))
            negative_row = int(np.flatnonzero(offsets < 0.0)[negative_near])
            positive_candidates = np.flatnonzero(offsets > 0.0)
            positive_row = int(positive_candidates[np.argmin(offsets[positive_candidates])])
            axis_small_direction.append(
                bool(
                    values[zero_row] > values[negative_row]
                    and values[zero_row] > values[positive_row]
                )
            )
        rows.append(
            {
                "image_id": record.image_id,
                "sample_count": int(samples.size),
                "visible_maplet_count": int(maplets.size),
                "axis_scores": axis_scores,
                "axis_gt_is_maximum": axis_gt_max,
                "axis_small_direction_correct": axis_small_direction,
            }
        )
    gt_flags = np.asarray(
        [flag for row in rows for flag in row["axis_gt_is_maximum"]], dtype=np.float64
    )
    direction_flags = np.asarray(
        [flag for row in rows for flag in row["axis_small_direction_correct"]],
        dtype=np.float64,
    )
    report = {
        "stage": "oracle_2dgs_surface_translation_score_landscape",
        "query_count": len(rows),
        "axis_count": int(gt_flags.size),
        "translation_offsets_m": offsets.tolist(),
        "gt_axis_maximum_fraction": (
            float(np.mean(gt_flags)) if gt_flags.size else 0.0
        ),
        "small_direction_correct_fraction": (
            float(np.mean(direction_flags)) if direction_flags.size else 0.0
        ),
        "camera_audit": camera_audit,
        "production_contract": {
            "uses_ground_truth_for_diagnostic_only": True,
            "uses_oracle_visible_maplets": True,
            "fixed_surface_sample_denominator": True,
            "map_metric_protocol": field.metadata.get("metric_protocol", "legacy"),
            "uses_stable_anchor_identity": False,
            "uses_point_correspondence_pnp": False,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
