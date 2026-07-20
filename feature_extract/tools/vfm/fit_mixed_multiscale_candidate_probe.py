"""Fit target-free mixed-point appearance probes using train geometry only.

The input point/candidate/support-view evidence has a fixed global top-L
denominator.  Only train rows are joined with SfM reprojection residuals.  The
saved prediction artifact is then generated for train and validation rows
without exposing validation targets to optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.current_v3_candidate_probe import stable_family_seed
from feature_extract.vfm.localization.mixed_multiscale_candidate_probe import (
    MIXED_EXACT_IDENTITY_PROBABILITY_SEMANTICS,
    MIXED_GEOMETRIC_PROBABILITY_SEMANTICS,
    MIXED_GEOMETRIC_SET_SUPERVISION_MODE,
    MIXED_MULTISCALE_MODEL_FORMAT,
    MIXED_MULTISCALE_PREDICTION_FORMAT,
    MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT,
    MIXED_REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
    MIXED_SUPPORTED_SUPERVISION_MODES,
    geometric_membership_from_residuals,
    load_mixed_multiscale_candidate_probe_features,
    materialize_mixed_candidate_reprojection_residuals,
    materialize_mixed_registered_identity_membership,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    PerViewFeatureNormalizer,
    predict_per_view_candidate_probe,
    train_per_view_candidate_probe,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--verification_points", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--families", default="all")
    parser.add_argument("--geometric_positive_threshold_px", type=float, default=5.0)
    parser.add_argument(
        "--supervision_mode",
        choices=tuple(sorted(MIXED_SUPPORTED_SUPERVISION_MODES)),
        default=MIXED_GEOMETRIC_SET_SUPERVISION_MODE,
    )
    parser.add_argument("--registered_identity_radius_px", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--architecture", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--prior_mode",
        choices=("appearance_only", "fixed_coarse_residual"),
        default="appearance_only",
        help=(
            "appearance_only tests new visual evidence alone; fixed_coarse_residual "
            "adds residuals over the immutable diagnostic top-L posterior"
        ),
    )
    parser.add_argument(
        "--no_precompute_normalized_input",
        action="store_true",
        help=(
            "avoid the compact-probe fast path that standardizes each feature row once "
            "before epochs; useful only when host memory is constrained"
        ),
    )
    parser.add_argument(
        "--cache_train_input_on_device",
        action="store_true",
        help=(
            "cache only the precomputed train rows on the selected GPU; useful for "
            "wide frozen feature schemas when device memory permits"
        ),
    )
    return parser.parse_args(argv)


def _resolve_families(value: str, *, feature_format: str) -> tuple[str, ...]:
    supported = MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT.get(str(feature_format))
    if supported is None:
        raise ValueError(f"unsupported mixed probe feature artifact format: {feature_format!r}")
    families = (
        tuple(supported)
        if str(value).strip() == "all"
        else tuple(item.strip() for item in str(value).split(",") if item.strip())
    )
    if not families or len(set(families)) != len(families):
        raise ValueError("mixed multi-scale families must be a non-empty unique list")
    unsupported = set(families) - set(supported)
    if unsupported:
        raise ValueError(
            "mixed probe family is not declared for this feature artifact: "
            f"{sorted(unsupported)}"
        )
    if set(families) - set(ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES):
        raise RuntimeError("a mixed multi-scale family is not registered")
    return families


def _save_model(
    *,
    path: Path,
    model: torch.nn.Module,
    normalizer: PerViewFeatureNormalizer,
    family: str,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "format": MIXED_MULTISCALE_MODEL_FORMAT,
            "family": str(family),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "normalizer": {
                "mean": normalizer.mean,
                "scale": normalizer.scale,
                "feature_indices": normalizer.feature_indices,
            },
            "metadata": metadata,
        },
        path,
    )


def fit_mixed_multiscale_candidate_probe(
    *,
    features_path: Path,
    verification_points_path: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    families: Sequence[str],
    geometric_positive_threshold_px: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    architecture: str,
    hidden_dim: int,
    prior_mode: str,
    precompute_normalized_input: bool,
    cache_train_input_on_device: bool,
    supervision_mode: str = MIXED_GEOMETRIC_SET_SUPERVISION_MODE,
    registered_identity_radius_px: float = 3.0,
) -> dict[str, Any]:
    if (
        float(geometric_positive_threshold_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or int(hidden_dim) <= 0
        or str(prior_mode) not in {"appearance_only", "fixed_coarse_residual"}
        or str(supervision_mode) not in MIXED_SUPPORTED_SUPERVISION_MODES
        or float(registered_identity_radius_px) <= 0.0
        or (bool(cache_train_input_on_device) and not bool(precompute_normalized_input))
    ):
        raise ValueError("mixed multi-scale probe optimization arguments are invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    features = load_mixed_multiscale_candidate_probe_features(
        features_path=Path(features_path), verification_points_path=Path(verification_points_path)
    )
    resolved_families = _resolve_families(
        ",".join(str(value) for value in families),
        feature_format=str(features.metadata.get("format", "")),
    )
    all_train_rows = np.flatnonzero(features.split_names == "train")
    if not len(all_train_rows):
        raise ValueError("mixed multi-scale features contain no train rows")
    mode = str(supervision_mode)
    if mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE:
        train_rows = all_train_rows
        train_residuals, train_target_audit = materialize_mixed_candidate_reprojection_residuals(
            features=features,
            projected_landmark_bank=Path(projected_landmark_bank),
            colmap_model_dir=Path(colmap_model_dir),
            row_indices=train_rows,
            required_split="train",
        )
        train_membership = geometric_membership_from_residuals(
            residuals=train_residuals,
            candidate_valid=(features.candidate_tracks[train_rows] >= 0),
            threshold_px=float(geometric_positive_threshold_px),
        )
        probability_semantics = MIXED_GEOMETRIC_PROBABILITY_SEMANTICS
        training_target = "sfm_candidate_reprojection_residual_set_membership"
        train_target_audit = {
            **train_target_audit,
            "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
        }
    else:
        train_rows, train_membership, _train_labels, train_target_audit = (
            materialize_mixed_registered_identity_membership(
                features=features,
                colmap_model_dir=Path(colmap_model_dir),
                split_name="train",
                identity_radius_px=float(registered_identity_radius_px),
            )
        )
        probability_semantics = MIXED_EXACT_IDENTITY_PROBABILITY_SEMANTICS
        training_target = "registered_query_observation_exact_track_or_explicit_null"
    projected_bank_sha = str(
        train_target_audit.get(
            "projected_landmark_bank_sha256",
            features.points.metadata.get("projected_landmark_bank_sha256", ""),
        )
    )
    if not projected_bank_sha:
        raise ValueError("mixed probe lacks projected landmark-bank provenance")
    target_membership = np.zeros(
        (len(features.source_point_ids), features.candidate_tracks.shape[1] + 1), dtype=bool
    )
    target_membership[:, -1] = True
    target_membership[train_rows] = train_membership
    train_groups = np.zeros((len(features.source_point_ids),), dtype=bool)
    train_groups[train_rows] = True
    use_base_prior = str(prior_mode) == "fixed_coarse_residual"
    device_value = torch.device(str(device))
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("mixed multi-scale probe requested CUDA but CUDA is unavailable")
    if device_value.type == "cuda":
        torch.cuda.set_device(device_value)

    output.mkdir(parents=True, exist_ok=False)
    family_candidates: list[np.ndarray] = []
    family_nulls: list[np.ndarray] = []
    family_view_logits: list[np.ndarray] = []
    family_train_metadata: dict[str, Any] = {}
    family_models: dict[str, str] = {}
    for family in resolved_families:
        model, normalizer, fit_metadata = train_per_view_candidate_probe(
            features=features.candidate_features,
            view_valid=features.candidate_view_valid,
            train_groups=train_groups,
            target_membership=target_membership,
            base_candidate_probabilities=(
                features.base_candidate_probabilities if use_base_prior else None
            ),
            base_null_probabilities=(features.base_null_probabilities if use_base_prior else None),
            family=family,
            device=device_value,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            seed=stable_family_seed(int(seed), family),
            architecture=str(architecture),
            hidden_dim=int(hidden_dim),
            feature_names=features.feature_names,
            precompute_normalized_model_input=bool(precompute_normalized_input),
            cache_precomputed_train_input_on_device=bool(cache_train_input_on_device),
        )
        candidate, null, view_logits = predict_per_view_candidate_probe(
            model,
            features=features.candidate_features,
            view_valid=features.candidate_view_valid,
            normalizer=normalizer,
            device=device_value,
            batch_size=int(batch_size),
            base_candidate_probabilities=(
                features.base_candidate_probabilities if use_base_prior else None
            ),
            base_null_probabilities=(features.base_null_probabilities if use_base_prior else None),
        )
        model_path = output / f"{family}.pt"
        model_metadata = {
            **fit_metadata,
            "features_sha256": file_sha256_short(features.path),
            "verification_points_sha256": file_sha256_short(Path(verification_points_path)),
            "projected_landmark_bank_sha256": projected_bank_sha,
            "training_supervision_split": "train",
            "supervision_mode": mode,
            "registered_identity_radius_px": (
                None
                if mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE
                else float(registered_identity_radius_px)
            ),
            "validation_or_test_labels_used_by_fit": False,
            "prior_mode": str(prior_mode),
            "probability_semantics": probability_semantics,
        }
        _save_model(
            path=model_path,
            model=model,
            normalizer=normalizer,
            family=family,
            metadata=model_metadata,
        )
        family_models[family] = str(model_path)
        family_train_metadata[family] = model_metadata
        family_candidates.append(candidate)
        family_nulls.append(null)
        family_view_logits.append(view_logits)
        del model
        if device_value.type == "cuda":
            torch.cuda.empty_cache()

    candidates = np.stack(family_candidates, axis=0).astype(np.float32, copy=False)
    nulls = np.stack(family_nulls, axis=0).astype(np.float32, copy=False)
    view_logits = np.stack(family_view_logits, axis=0).astype(np.float32, copy=False)
    metadata: dict[str, Any] = {
        "format": MIXED_MULTISCALE_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "prediction_frozen_before_validation_target_join": True,
        "training_supervision_split": "train",
        "training_target": training_target,
        "training_objective": "set_log_mass_nll_over_target_membership_v1",
        "validation_or_test_labels_used_by_fit": False,
        "supervision_mode": mode,
        "registered_identity_radius_px": (
            None
            if mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE
            else float(registered_identity_radius_px)
        ),
        "probability_semantics": probability_semantics,
        "prior_mode": str(prior_mode),
        "precompute_normalized_input": bool(precompute_normalized_input),
        "cache_train_input_on_device": bool(cache_train_input_on_device),
        "base_coarse_posterior_role": (
            "not_used" if not use_base_prior else "fixed_diagnostic_residual_baseline_only"
        ),
        "features": str(features.path),
        "features_sha256": file_sha256_short(features.path),
        "verification_points": str(Path(verification_points_path)),
        "verification_points_sha256": file_sha256_short(Path(verification_points_path)),
        "projected_landmark_bank": str(Path(projected_landmark_bank)),
        "projected_landmark_bank_sha256": projected_bank_sha,
        "candidate_top_k": int(features.candidate_tracks.shape[1]),
        "fixed_global_top_l": True,
        "candidate_reselection": False,
        "support_view_marginalization": "per_view_logits_preserved_v1",
        "image_retrieval_or_submap_used": False,
        "render": False,
        "families": list(resolved_families),
        "models": family_models,
    }
    prediction_path = output / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_point_ids=features.source_point_ids,
            query_ids=features.query_ids,
            split_names=features.split_names,
            point_sources=features.point_sources,
            candidate_track_ids=features.candidate_tracks,
            candidate_bank_rows=features.candidate_bank_rows,
            candidate_view_valid=features.candidate_view_valid,
            family_names=np.asarray(resolved_families, dtype=np.str_),
            candidate_probabilities=candidates,
            null_probabilities=nulls,
            per_view_logits=view_logits,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    positive_rows = np.any(train_membership[:, :-1], axis=1)
    summary = {
        "stage": "fit_mixed_multiscale_candidate_probe_train_only",
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256_short(prediction_path),
        "models": family_models,
        "families": family_train_metadata,
        "train_target_audit": {
            **train_target_audit,
            "train_row_count": int(len(train_rows)),
            "positive_train_row_count": int(np.sum(positive_rows)),
            "explicit_null_train_row_count": int(np.sum(~positive_rows)),
            "positive_candidate_train_count": int(np.sum(train_membership[:, :-1])),
            "by_point_source": {
                source: {
                    "row_count": int(np.sum(features.point_sources[train_rows] == source)),
                    "positive_row_count": int(
                        np.sum(positive_rows[features.point_sources[train_rows] == source])
                    ),
                }
                for source in sorted(set(features.point_sources[train_rows].tolist()))
            },
        },
        "protocol": {
            "feature_export_target_free": True,
            "fit_uses_train_targets_only": True,
            "validation_or_test_targets_used": False,
            "prediction_frozen_before_validation_target_join": True,
            "fixed_global_top_l": True,
            "candidate_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            fit_mixed_multiscale_candidate_probe(
                features_path=Path(args.features),
                verification_points_path=Path(args.verification_points),
                projected_landmark_bank=Path(args.projected_landmark_bank),
                colmap_model_dir=Path(args.colmap_model_dir),
                output_dir=Path(args.output_dir),
                families=(args.families,),
                geometric_positive_threshold_px=float(args.geometric_positive_threshold_px),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                seed=int(args.seed),
                device=str(args.device),
                architecture=str(args.architecture),
                hidden_dim=int(args.hidden_dim),
                prior_mode=str(args.prior_mode),
                precompute_normalized_input=not bool(args.no_precompute_normalized_input),
                cache_train_input_on_device=bool(args.cache_train_input_on_device),
                supervision_mode=str(args.supervision_mode),
                registered_identity_radius_px=float(args.registered_identity_radius_px),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
