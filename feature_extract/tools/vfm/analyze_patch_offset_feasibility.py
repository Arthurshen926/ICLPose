"""Strict feasibility diagnostics for patch-to-pixel offset refinement."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_offset_feasibility import (
    calibration_summary,
    offset_error_bucket_summary,
    pose_metric_summary,
)
from feature_extract.vfm.patch_offset_refiner import (
    apply_predicted_patch_offsets,
    load_patch_offset_refiner_checkpoint,
    predict_patch_offsets_for_matches,
    refine_matches_with_oracle_offsets,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatch,
    estimate_pose_pnp_fixed,
    pnp_pose_error,
)
from feature_extract.vfm.rendered_map_verifier import project_xyz_to_image
from feature_extract.vfm.tokens import TokenBankManifest


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _group_rows(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_query: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    return dict(by_query)


def _match_from_row(row: dict[str, object]) -> QueryTo3DMatch:
    return QueryTo3DMatch(
        token_index=int(row.get("token_index", 0)),
        xy=np.asarray(row.get("xy", [0.0, 0.0]), dtype=np.float64).reshape(2),
        track_id=int(row.get("track_id", -1)),
        xyz=np.asarray(row.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3),
        similarity=float(row.get("similarity", 0.0) or 0.0),
        ratio=0.0,
        landmark_variance=float(row.get("landmark_variance", 0.0) or 0.0),
        source=str(row.get("source", "sparse_landmark")),
        observation_count=None if row.get("observation_count") is None else int(row.get("observation_count", 0)),
        visibility_count=None if row.get("visibility_count") is None else int(row.get("visibility_count", 0)),
        landmark_reprojection_error=None
        if row.get("landmark_reprojection_error") is None
        else float(row.get("landmark_reprojection_error", 0.0)),
        landmark_quality=None if row.get("landmark_quality") is None else float(row.get("landmark_quality", 0.0)),
        landmark_ambiguity=None if row.get("landmark_ambiguity") is None else float(row.get("landmark_ambiguity", 0.0)),
        similarity_margin=None if row.get("similarity_margin") is None else float(row.get("similarity_margin", 0.0)),
        distance_to_boundary_px=None
        if row.get("distance_to_boundary_px") is None
        else float(row.get("distance_to_boundary_px", 0.0)),
    )


def _stride_from_rows(rows: Sequence[dict[str, object]], fallback: float) -> float:
    values = []
    for row in rows:
        px = row.get("gt_reproj_error_px")
        stride = row.get("gt_reproj_error_stride")
        if px is None or stride is None:
            continue
        px_value = float(px)
        stride_value = float(stride)
        if np.isfinite(px_value) and np.isfinite(stride_value) and abs(stride_value) > 1e-6:
            values.append(px_value / stride_value)
    return float(np.median(values)) if values else float(fallback)


def _patch_positive_by_token(rows: Sequence[dict[str, object]]) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if bool(row.get("patch_positive_label", row.get("patch_correct", False))):
            positives[int(row.get("token_index", 0))].add(int(row.get("track_id", -1)))
    return dict(positives)


def _pose_row(
    variant: str,
    query_id: str,
    pnp,
    gt_pose_w2c: np.ndarray,
    selected_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    error = pnp_pose_error(pnp.pose_w2c, gt_pose_w2c)
    patch_values = [bool(row.get("patch_positive_label", row.get("patch_correct", False))) for row in selected_rows]
    gt_stride_values = [
        bool(row.get("gt_reproj_error_stride") is not None and float(row.get("gt_reproj_error_stride", 9999.0)) <= 1.0)
        for row in selected_rows
    ]
    return {
        "variant": variant,
        "query_id": query_id,
        "success": bool(pnp.success),
        "translation_error_m": None if not np.isfinite(error.translation_m) else float(error.translation_m),
        "rotation_error_deg": None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg),
        "inlier_count": int(pnp.inlier_count),
        "same_inlier_patch_at1": None if not patch_values else float(np.mean(patch_values)),
        "same_inlier_gt_at_stride": None if not gt_stride_values else float(np.mean(gt_stride_values)),
    }


def _fixed_pose_for_mask(
    variant: str,
    query_id: str,
    matches: Sequence[QueryTo3DMatch],
    rows: Sequence[dict[str, object]],
    mask: np.ndarray,
    camera,
    gt_pose_w2c: np.ndarray,
    min_inliers: int,
    pnp_method: str,
    refine_method: str,
) -> dict[str, object]:
    selected_matches = [match for match, keep in zip(matches, mask) if bool(keep)]
    selected_rows = [row for row, keep in zip(rows, mask) if bool(keep)]
    pnp = estimate_pose_pnp_fixed(
        selected_matches,
        camera,
        min_inliers=int(min_inliers),
        pnp_method=pnp_method,
        refine_method=refine_method,
    )
    return _pose_row(variant, query_id, pnp, gt_pose_w2c, selected_rows)


def _landmark_index_from_rows_and_bank(rows: Sequence[dict[str, object]], landmark_bank: Path) -> LandmarkMapIndex:
    bank = load_selected_track_bank_npz(landmark_bank)
    xyz_by_track = {}
    for row in rows:
        track_id = int(row.get("track_id", -1))
        if track_id not in xyz_by_track and track_id in bank.tracks:
            xyz_by_track[track_id] = np.asarray(row.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    track_ids = []
    xyz = []
    features = []
    variances = []
    counts = []
    image_ids = []
    for track_id in sorted(xyz_by_track):
        track = bank.tracks[int(track_id)]
        track_ids.append(int(track_id))
        xyz.append(xyz_by_track[int(track_id)])
        features.append(np.asarray(track.mean_feature, dtype=np.float32))
        variances.append(float(np.mean(track.variance)))
        counts.append(int(track.observation_count))
        image_ids.append(tuple(str(item) for item in track.observation_image_ids))
    if not track_ids:
        return LandmarkMapIndex(
            track_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, bank.feature_dim), dtype=np.float32),
            mean_variances=np.zeros((0,), dtype=np.float32),
            observation_counts=np.zeros((0,), dtype=np.int64),
            observation_image_ids=(),
        )
    return LandmarkMapIndex(
        track_ids=np.asarray(track_ids, dtype=np.int64),
        xyz=np.stack(xyz, axis=0),
        features=np.stack(features, axis=0),
        mean_variances=np.asarray(variances, dtype=np.float32),
        observation_counts=np.asarray(counts, dtype=np.int64),
        observation_image_ids=tuple(image_ids),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze strict patch offset feasibility from canonical match rows")
    parser.add_argument("--matches_jsonl", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--bounded_stride_values", default="0.5,1.0")
    parser.add_argument("--noise_sigmas_px", default="0,1,2,4,8,12")
    parser.add_argument("--noise_base_stride", type=float, default=0.5)
    parser.add_argument("--fallback_stride_px", type=float, default=16.0)
    parser.add_argument("--min_inliers", type=int, default=4)
    parser.add_argument("--pnp_method", default="EPNP")
    parser.add_argument("--refine_method", default="LM")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--learned_checkpoint", default="")
    parser.add_argument("--query_manifest", default="")
    parser.add_argument("--landmark_bank", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--learned_device", default="cpu")
    parser.add_argument("--learned_batch_size", type=int, default=4096)
    parser.add_argument("--learned_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--learned_max_stride", type=float, default=0.5)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_jsonl", default="")
    args = parser.parse_args(argv)

    all_rows = _load_jsonl(Path(args.matches_jsonl))
    by_query = _group_rows(all_rows)
    query_ids = sorted(by_query)
    if args.max_queries > 0:
        query_ids = query_ids[: int(args.max_queries)]
    pose_by_query = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    bounded_values = _parse_float_list(args.bounded_stride_values)
    noise_sigmas = _parse_float_list(args.noise_sigmas_px)

    learned_run = None
    manifest_by_query = {}
    learned_index = None
    if args.learned_checkpoint:
        if not args.query_manifest or not args.landmark_bank:
            raise ValueError("query_manifest and landmark_bank are required for learned offset diagnostics")
        learned_run = load_patch_offset_refiner_checkpoint(Path(args.learned_checkpoint), device=args.learned_device)
        manifest = TokenBankManifest.from_json(Path(args.query_manifest))
        manifest.validate(verify_checksums=False)
        manifest_by_query = {record.image_id: record for record in manifest.records}
        learned_index = _landmark_index_from_rows_and_bank(all_rows, Path(args.landmark_bank))

    variant_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    output_rows = []
    learned_error_values = []
    learned_confidence_values = []
    learned_bucket_rows = []

    for query_ord, query_id in enumerate(query_ids):
        rows = by_query[query_id]
        gt_pose = pose_by_query.get(query_id)
        if gt_pose is None:
            continue
        matches = [_match_from_row(row) for row in rows]
        inlier_mask = np.asarray([bool(row.get("pnp_inlier", False)) for row in rows], dtype=bool)
        stride = _stride_from_rows(rows, fallback=float(args.fallback_stride_px))
        patch_positive = _patch_positive_by_token(rows)

        query_variant_rows = []
        center = _fixed_pose_for_mask(
            "same_inlier_patch_center",
            query_id,
            matches,
            rows,
            inlier_mask,
            camera,
            gt_pose,
            int(args.min_inliers),
            args.pnp_method,
            args.refine_method,
        )
        query_variant_rows.append(center)
        variant_rows[center["variant"]].append(center)

        free_matches, free_summary = refine_matches_with_oracle_offsets(
            matches,
            gt_pose,
            camera,
            stride_px=stride,
            inlier_mask=inlier_mask,
            max_offset_stride=None,
            rng_seed=int(args.seed) + query_ord,
        )
        free_row = _fixed_pose_for_mask(
            "same_inlier_oracle_free",
            query_id,
            free_matches,
            rows,
            inlier_mask,
            camera,
            gt_pose,
            int(args.min_inliers),
            args.pnp_method,
            args.refine_method,
        )
        free_row["offset_summary"] = free_summary
        query_variant_rows.append(free_row)
        variant_rows[free_row["variant"]].append(free_row)

        for value in bounded_values:
            bounded_matches, bounded_summary = refine_matches_with_oracle_offsets(
                matches,
                gt_pose,
                camera,
                stride_px=stride,
                inlier_mask=inlier_mask,
                max_offset_stride=float(value),
                bound_metric="linf",
                rng_seed=int(args.seed) + query_ord,
            )
            name = f"same_inlier_oracle_bounded_linf_{value:g}stride"
            row = _fixed_pose_for_mask(
                name,
                query_id,
                bounded_matches,
                rows,
                inlier_mask,
                camera,
                gt_pose,
                int(args.min_inliers),
                args.pnp_method,
                args.refine_method,
            )
            row["offset_summary"] = bounded_summary
            query_variant_rows.append(row)
            variant_rows[name].append(row)

            positive_matches, positive_summary = refine_matches_with_oracle_offsets(
                matches,
                gt_pose,
                camera,
                stride_px=stride,
                inlier_mask=inlier_mask,
                max_offset_stride=float(value),
                bound_metric="linf",
                patch_positive_by_token=patch_positive,
                require_patch_positive=True,
                rng_seed=int(args.seed) + query_ord,
            )
            positive_name = f"same_inlier_patch_positive_oracle_bounded_linf_{value:g}stride"
            positive_row = _fixed_pose_for_mask(
                positive_name,
                query_id,
                positive_matches,
                rows,
                inlier_mask,
                camera,
                gt_pose,
                int(args.min_inliers),
                args.pnp_method,
                args.refine_method,
            )
            positive_row["offset_summary"] = positive_summary
            query_variant_rows.append(positive_row)
            variant_rows[positive_name].append(positive_row)

        for sigma in noise_sigmas:
            noisy_matches, noisy_summary = refine_matches_with_oracle_offsets(
                matches,
                gt_pose,
                camera,
                stride_px=stride,
                inlier_mask=inlier_mask,
                max_offset_stride=float(args.noise_base_stride),
                bound_metric="linf",
                noise_sigma_px=float(sigma),
                rng_seed=int(args.seed) + query_ord * 1000 + int(round(float(sigma) * 100.0)),
            )
            name = f"same_inlier_oracle_bounded_linf_{args.noise_base_stride:g}stride_noise_{sigma:g}px"
            noisy_row = _fixed_pose_for_mask(
                name,
                query_id,
                noisy_matches,
                rows,
                inlier_mask,
                camera,
                gt_pose,
                int(args.min_inliers),
                args.pnp_method,
                args.refine_method,
            )
            noisy_row["offset_summary"] = noisy_summary
            query_variant_rows.append(noisy_row)
            variant_rows[name].append(noisy_row)

        if learned_run is not None and learned_index is not None and query_id in manifest_by_query:
            query_feature = _load_query_feature(manifest_by_query[query_id].token_path, args.layer_name)
            offsets, confidences, sigmas = predict_patch_offsets_for_matches(
                query_feature,
                learned_index,
                matches,
                learned_run.model,
                device=args.learned_device,
                batch_size=int(args.learned_batch_size),
            )
            learned_matches, learned_summary = apply_predicted_patch_offsets(
                matches,
                offsets,
                confidences,
                stride_px=stride,
                confidence_threshold=float(args.learned_confidence_threshold),
                inlier_mask=inlier_mask,
                max_offset_stride=float(args.learned_max_stride),
            )
            learned_row = _fixed_pose_for_mask(
                "same_inlier_learned_offset",
                query_id,
                learned_matches,
                rows,
                inlier_mask,
                camera,
                gt_pose,
                int(args.min_inliers),
                args.pnp_method,
                args.refine_method,
            )
            learned_row["offset_summary"] = learned_summary
            query_variant_rows.append(learned_row)
            variant_rows[learned_row["variant"]].append(learned_row)
            for row_idx, (row, match) in enumerate(zip(rows, matches)):
                projected = project_xyz_to_image(np.asarray(match.xyz, dtype=np.float64), gt_pose, camera)
                if projected is None:
                    continue
                target_delta = np.asarray(projected, dtype=np.float64).reshape(2) - np.asarray(match.xy, dtype=np.float64).reshape(2)
                pred_delta = np.asarray(offsets[row_idx], dtype=np.float64).reshape(2) * float(stride)
                learned_error_values.append(float(np.linalg.norm(pred_delta - target_delta)))
                learned_confidence_values.append(float(confidences[row_idx]))
                learned_bucket_rows.append(row)

        output_rows.append(
            {
                "query_id": query_id,
                "first_pass_inlier_count": int(np.sum(inlier_mask)),
                "stride_px": float(stride),
                "variants": query_variant_rows,
            }
        )

    summary = {
        "stage": "patch_offset_strict_feasibility",
        "inputs": {
            "matches_jsonl": str(args.matches_jsonl),
            "query_pose_file": str(args.query_pose_file),
            "camera_source": str(camera_source),
            "learned_checkpoint": str(args.learned_checkpoint),
        },
        "query_count": int(len(output_rows)),
        "bounded_stride_values": bounded_values,
        "noise_sigmas_px": noise_sigmas,
        "variant_metrics": {name: pose_metric_summary(rows) for name, rows in sorted(variant_rows.items())},
        "learned_offset_error_buckets": None
        if not learned_error_values
        else offset_error_bucket_summary(learned_bucket_rows, learned_error_values, learned_confidence_values),
        "learned_confidence_calibration": None
        if not learned_error_values
        else calibration_summary(
            learned_error_values,
            learned_confidence_values,
            good_threshold_px=4.0,
            bin_count=10,
        ),
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_jsonl:
        rows_path = Path(args.output_jsonl)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in output_rows) + "\n")


if __name__ == "__main__":
    main()
