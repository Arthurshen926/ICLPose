"""Fit Goal-Maplet score-to-validity calibration on an independent trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    contributor_multiscale_maplet_distribution,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.query_support import all_token_coordinates
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    fit_validity_calibration,
    retrieve_maplet_posterior,
)
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


POOLING = {
    "mapped_1x1": ((1,), (1.0,)),
    "mapped_1x1_3x3": ((1, 3), (0.65, 0.35)),
    "mapped_1x1_3x3_5x5": ((1, 3, 5), (0.50, 0.30, 0.20)),
    "current_1x1_3x3_5x5_9x9": ((1, 3, 5, 9), (0.40, 0.30, 0.20, 0.10)),
}


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probability >= low) & (probability < high if index + 1 < bins else probability <= high)
        if np.any(selected):
            value += float(np.mean(selected)) * abs(float(np.mean(probability[selected]) - np.mean(target[selected])))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--physical_instance_readout", default="")
    parser.add_argument("--pooling", choices=tuple(POOLING), required=True)
    parser.add_argument("--output_calibration", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include_trajectories", nargs="+", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_calibration), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite Goal-Maplet calibration")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout = readout_canonical_field(field, physical)
    instance_readout = None
    if args.physical_instance_readout:
        instance_readout, instance_metadata = load_physical_instance_readout(
            Path(args.physical_instance_readout), device=str(args.device),
        )
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
        ):
            if instance_metadata.get(key) != expected:
                raise ValueError(f"physical-instance readout lineage differs: {key}")
        readout = type(readout)(
            instance_readout.project_numpy(readout.parent_descriptors, role="context", device=str(args.device)),
            readout.parent_coverage,
            instance_readout.project_numpy(readout.child_descriptors, role="local", device=str(args.device)),
            readout.child_coverage,
        )
    valid_maplets = readout.parent_coverage > 0.0
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    pool_sizes, pool_weights = POOLING[str(args.pooling)]
    config = RadioFinalRegionConfig(pool_sizes=pool_sizes, pool_weights=pool_weights)
    all_score, all_target = [], []
    image_ids = []
    included = set(str(value) for value in args.include_trajectories)
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if included and str(metadata["trajectory_id"]) not in included:
            continue
        labels = ContributorLabels.load_npz(path)
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        descriptor = (
            encode_radio_final_regions(mapped, token_xy, config)
            if instance_readout is None else encode_physical_instance_regions(
                instance_readout, mapped, token_xy, role="context", device=str(args.device),
            )
        )
        _, _, _, best = retrieve_maplet_posterior(
            descriptor,
            readout.parent_descriptors,
            physical.maplet_ids,
            valid_maplets,
            maximum_candidates=64,
            temperature=0.07,
            null_similarity_center=0.35,
            null_similarity_scale=0.08,
        )
        _, truth_null = contributor_multiscale_maplet_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            pool_sizes=pool_sizes,
            pool_weights=pool_weights,
        )
        all_score.append(best.astype(np.float64))
        all_target.append(1.0 - truth_null.astype(np.float64))
        image_ids.append(str(metadata["image_id"]))
        print(json.dumps({"image_id": image_ids[-1], "support_count": int(best.size)}), flush=True)
    score = np.concatenate(all_score)
    target = np.concatenate(all_target)
    calibration = fit_validity_calibration(
        score,
        target,
        metadata={
            "physical_map_sha256": physical.content_sha256,
            "canonical_field_sha256": field.content_sha256,
            "surface_mapper_file_sha256": file_sha256(Path(args.surface_mapper)),
            "physical_instance_readout_sha256": (
                file_sha256(Path(args.physical_instance_readout))
                if args.physical_instance_readout else None
            ),
            "pooling": str(args.pooling),
            "fit_support_mode": "all_tokens_exact_multiscale_masks",
            "fit_image_ids": image_ids,
            "fit_trajectory_ids": sorted({value.split("/", 1)[0] for value in image_ids}),
            "stores_scores_or_query_features": False,
        },
    )
    calibrated = calibration.predict_valid(score)
    uncalibrated = 1.0 / (1.0 + np.exp(-(score - 0.35) / 0.08))
    report = {
        "stage": "calibrate_goal_maplet_validity",
        "support_count": int(score.size),
        "image_count": len(image_ids),
        "center": float(calibration.center),
        "scale": float(calibration.scale),
        "target_valid_mean": float(np.mean(target)),
        "uncalibrated": {
            "predicted_valid_mean": float(np.mean(uncalibrated)),
            "ece": _ece(uncalibrated, target),
            "brier": float(np.mean(np.square(uncalibrated - target))),
        },
        "calibrated": {
            "predicted_valid_mean": float(np.mean(calibrated)),
            "ece": _ece(calibrated, target),
            "brier": float(np.mean(np.square(calibrated - target))),
        },
        "calibration_sha256": calibration.content_sha256,
        "calibration_metadata": dict(calibration.metadata),
    }
    calibration.save_json(output)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
