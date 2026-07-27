"""Fit trajectory-disjoint identity and conditional-spatial null calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from feature_extract.tools.vfm.evaluate_v6_retrieval_pose_basin import (
    _region_geometry,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    DistributionNullCalibration,
    V6ProbabilityCalibration,
    null_calibration_features,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--contributor_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--identity_maplets", required=True)
    parser.add_argument("--spatial_maplets", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--output_calibration", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--calibration_trajectory_ids", nargs="+", default=["seq11"]
    )
    parser.add_argument(
        "--strict_holdout_trajectory_ids",
        nargs="+",
        default=["seq3", "seq5", "seq13"],
    )
    parser.add_argument("--identity_candidates", type=int, default=16)
    parser.add_argument("--surface_resolution_m", type=float, default=0.30)
    parser.add_argument("--descriptor_temperature", type=float, default=0.08)
    parser.add_argument("--mixture_temperature", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / np.maximum(
        np.linalg.norm(array, axis=1, keepdims=True), 1e-8
    )


def _identity_logits(
    query: np.ndarray,
    bank: SurfaceRetrievalMapletBank,
    *,
    descriptor_temperature: float,
    mixture_temperature: float,
) -> np.ndarray:
    query = _normalize(query)
    score = query @ np.asarray(bank.descriptors, dtype=np.float64).T
    tau = max(float(mixture_temperature), 1e-4)
    logits = np.empty((query.shape[0], len(bank)), dtype=np.float64)
    for row in range(len(bank)):
        begin, end = (
            int(bank.descriptor_offsets[row]),
            int(bank.descriptor_offsets[row + 1]),
        )
        component = (
            score[:, begin:end] / tau
            + np.log(
                np.clip(
                    bank.descriptor_weights[begin:end], 1e-12, 1.0
                )
            )[None]
        )
        logits[:, row] = tau * logsumexp(component, axis=1)
    logits += 0.15 * np.log(
        np.clip(bank.quality_scores[None], 1e-4, 1.0)
    )
    logits -= 0.10 * np.clip(
        bank.descriptor_uncertainties[None], 0.0, 10.0
    )
    return logits / max(float(descriptor_temperature), 1e-4)


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.shape != (y.size, 5) or np.unique(y).size != 2:
        raise ValueError("calibration needs both null and resolved examples")
    mean = np.mean(x[:, 1:], axis=0)
    scale = np.maximum(np.std(x[:, 1:], axis=0), 1e-6)
    standardized = np.c_[
        np.ones(x.shape[0]), (x[:, 1:] - mean[None]) / scale[None]
    ]
    def objective(weight):
        logit = standardized @ weight
        loss = np.logaddexp(0.0, logit) - y * logit
        regularizer = 0.5 * float(l2) * np.sum(weight[1:] ** 2)
        # This artifact is consumed as a probability, not merely as a
        # balanced classifier score.  Class-balanced cross entropy changes
        # the implied class prior (the old conditional-spatial calibrator
        # predicted ~0.49 null for a 0.12 null prevalence), so use the proper
        # unweighted Bernoulli likelihood.
        gradient = standardized.T @ (expit(logit) - y) / y.size
        gradient[1:] += float(l2) * weight[1:]
        return float(np.mean(loss) + regularizer), gradient

    result = minimize(
        lambda value: objective(value),
        np.zeros(5, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success:
        raise RuntimeError(f"null calibration failed: {result.message}")
    standardized_weight = np.asarray(result.x, dtype=np.float64)
    weight = np.empty_like(standardized_weight)
    weight[1:] = standardized_weight[1:] / scale
    weight[0] = standardized_weight[0] - np.sum(
        standardized_weight[1:] * mean / scale
    )
    probability = expit(x @ weight)
    prediction = probability >= 0.5
    order = np.argsort(-probability, kind="stable")
    ranked = y[order] > 0.5
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    auprc = float(np.sum(precision[ranked]) / max(np.sum(ranked), 1))
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probability >= lower) & (
            (probability <= upper) if upper >= 1.0 else (probability < upper)
        )
        if np.any(selected):
            calibration_error += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected]))
                - float(np.mean(y[selected]))
            )
    return weight, {
        "example_count": int(y.size),
        "null_fraction": float(np.mean(y)),
        "balanced_accuracy": float(
            0.5
            * (
                np.mean(prediction[y > 0.5])
                + np.mean(~prediction[y <= 0.5])
            )
        ),
        "brier": float(np.mean((probability - y) ** 2)),
        "auprc": auprc,
        "ece_10bin": float(calibration_error),
        "mean_predicted_null": float(np.mean(probability)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_calibration)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite probability calibration")
    calibration_ids = {str(value) for value in args.calibration_trajectory_ids}
    strict_ids = {str(value) for value in args.strict_holdout_trajectory_ids}
    if calibration_ids & strict_ids:
        raise ValueError("strict holdout overlaps probability calibration")
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    identity_bank = SurfaceRetrievalMapletBank.load_npz(
        Path(args.identity_maplets)
    )
    spatial_bank = SurfaceRetrievalMapletBank.load_npz(
        Path(args.spatial_maplets)
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    identity_config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(
            mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))
        ),
        global_context_weight=float(
            mapper_metadata.get("global_context_weight", 0.0)
        ),
    )
    views = _load_views(
        Path(args.contributor_dir),
        atlas,
        Path(args.image_root),
        trajectory_ids=sorted(calibration_ids),
    )
    if not views:
        raise ValueError("probability calibration has no views")
    flat_cells = atlas.height * atlas.width
    spatial_row_by_id = {
        int(value): int(row)
        for row, value in enumerate(spatial_bank.maplet_ids.tolist())
    }
    identity_features = []
    identity_labels = []
    spatial_features = []
    spatial_labels = []
    for view in views:
        raw = np.asarray(view.radio, dtype=np.float32)
        identity_map = mapper.project(raw).measurement_context
        spatial_map = spatial_bank.project_query_feature_map(raw)
        _indices, token_xy = select_spatially_balanced_radio_final_regions(raw)
        identity_descriptor = encode_radio_final_regions(
            identity_map, token_xy, identity_config
        )
        spatial_descriptor = _normalize(
            encode_radio_final_regions(
                spatial_map,
                token_xy,
                RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,)),
            )
        )
        region_xy, region_extent = _region_geometry(
            token_xy,
            token_width=int(raw.shape[2]),
            token_height=int(raw.shape[1]),
            image_width=int(view.camera.width),
            image_height=int(view.camera.height),
            config=identity_config,
        )
        logits = _identity_logits(
            identity_descriptor,
            identity_bank,
            descriptor_temperature=float(args.descriptor_temperature),
            mixture_temperature=float(args.mixture_temperature),
        )
        identity_features.append(null_calibration_features(logits))
        surface_maplet_ids = atlas.maplet_ids[
            np.asarray(view.visible_rows, dtype=np.int64) // flat_cells
        ]
        surface_xy = np.asarray(view.image_xy, dtype=np.float64)
        surface_xyz = atlas.xyz.reshape(-1, 3)[view.visible_rows]
        for region in range(token_xy.shape[0]):
            normalized = np.abs(
                surface_xy - region_xy[region][None]
            ) / np.maximum(region_extent[region][None], 1.0)
            inside = np.max(normalized, axis=1) <= 1.0
            true_ids = np.unique(surface_maplet_ids[inside])
            identity_labels.append(float(true_ids.size == 0))
            candidate_count = min(
                max(int(args.identity_candidates), 1), len(identity_bank)
            )
            candidates = np.argpartition(
                -logits[region], candidate_count - 1
            )[:candidate_count]
            candidates = candidates[
                np.argsort(-logits[region, candidates], kind="stable")
            ]
            for identity_row in candidates.tolist():
                maplet_id = int(identity_bank.maplet_ids[identity_row])
                # Spatial null is a conditional event:
                #   p(spatial unresolved | this maplet identity is correct).
                # Wrong identity candidates belong to the identity posterior
                # and must not be relabelled as spatial negatives.  Mixing
                # them here made the old calibration overwhelmingly null and
                # produced a misleading near-perfect all-candidate AUPRC.
                if maplet_id not in true_ids:
                    continue
                spatial_row = spatial_row_by_id.get(maplet_id)
                if spatial_row is None:
                    # Atlas-unavailable is represented explicitly at runtime
                    # with spatial_available=False and spatial_null=1.  There
                    # is no score distribution to calibrate in this branch.
                    continue
                begin, end = (
                    int(spatial_bank.descriptor_offsets[spatial_row]),
                    int(spatial_bank.descriptor_offsets[spatial_row + 1]),
                )
                local_logit = (
                    spatial_descriptor[region]
                    @ spatial_bank.descriptors[begin:end].T
                ) / max(float(args.mixture_temperature), 1e-4)
                local_logit += np.log(
                    np.clip(
                        spatial_bank.descriptor_weights[begin:end],
                        1e-12,
                        1.0,
                    )
                )
                spatial_features.append(
                    null_calibration_features(local_logit)[0]
                )
                target = inside & (surface_maplet_ids == maplet_id)
                if not np.any(target):
                    raise AssertionError(
                        "identity-conditioned spatial target disappeared"
                    )
                component = int(np.argmax(local_logit))
                error = np.min(
                    np.linalg.norm(
                        surface_xyz[target]
                        - spatial_bank.descriptor_centers[begin + component],
                        axis=1,
                    )
                )
                spatial_labels.append(
                    float(error > float(args.surface_resolution_m))
                )
    identity_feature_array = np.concatenate(identity_features, axis=0)
    identity_label_array = np.asarray(identity_labels, dtype=np.float64)
    spatial_feature_array = np.asarray(spatial_features, dtype=np.float64)
    spatial_label_array = np.asarray(spatial_labels, dtype=np.float64)
    identity_weight, identity_metrics = _fit_logistic(
        identity_feature_array, identity_label_array, l2=float(args.l2)
    )
    spatial_weight, spatial_metrics = _fit_logistic(
        spatial_feature_array, spatial_label_array, l2=float(args.l2)
    )
    metadata = {
        "artifact_type": "v6_probability_calibration",
        "calibration_trajectory_ids": sorted(calibration_ids),
        "strict_holdout_trajectory_ids": sorted(strict_ids),
        "identity_bank_sha256": _sha256(Path(args.identity_maplets)),
        "spatial_bank_sha256": _sha256(Path(args.spatial_maplets)),
        "atlas_geometry_sha256": _sha256(Path(args.atlas_geometry)),
        "surface_mapper_sha256": _sha256(
            Path(args.surface_mapper_checkpoint)
        ),
        "surface_resolution_m": float(args.surface_resolution_m),
        "identity_candidates": int(args.identity_candidates),
        "identity_metrics": identity_metrics,
        "spatial_metrics": spatial_metrics,
        "spatial_calibration_condition": "identity_is_true_and_atlas_available",
        "calibration_objective": "unweighted_bernoulli_nll",
        "stores_mapping_rgb": False,
        "stores_mapping_image_ids": False,
        "uses_strict_holdout": False,
    }
    calibration = V6ProbabilityCalibration(
        identity=DistributionNullCalibration(identity_weight),
        spatial=DistributionNullCalibration(spatial_weight),
        metadata=metadata,
    )
    calibration.save_json(output)
    report = {
        "stage": "v6_trajectory_disjoint_probability_calibration",
        "output_calibration": str(output),
        "calibration_view_count": len(views),
        "identity": identity_metrics,
        "spatial": spatial_metrics,
        "metadata": metadata,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
