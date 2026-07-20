"""Fit train-only aggregate residual probes on frozen full-track summaries.

This is a diagnostic identity experiment, not a pose optimizer.  It reads
registered SfM identities only for train rows, preserves each input row's
global top-20 candidates and null probability exactly, and freezes predictions
for train and validation before any validation identity is loaded.  The
companion audit is the sole validation-label boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_MODEL_FORMAT,
    FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
    FROZEN_FULLTRACK_RESIDUAL_FAMILIES,
    FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES,
    FulltrackAppearanceFeatureNormalizer,
    FulltrackSummaryTop4RelativeNormalizer,
    SUMMARY_TOP4_FEATURE_GRANULARITY,
    fit_fixedprior_fulltrack_linear_residual,
    fit_fixedprior_fulltrack_summary_top4_probe,
    load_frozen_fulltrack_appearance_features,
    predict_fixedprior_fulltrack_linear_residual,
    predict_fixedprior_fulltrack_summary_top4_probe,
    zero_residual_fulltrack_posterior,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
)


EXPECTED_QUERY_COUNTS = {"train": 63, "validation": 21}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--appearance-artifacts",
        required=True,
        help="comma-separated complete train and validation full-track artifacts",
    )
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument(
        "--expected-identity-colmap-images-sha256",
        required=True,
        help=(
            "immutable images.bin hash declared by the frozen hypothesis/source "
            "lineage used for registered identity supervision"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", default="all")
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("full-track appearance artifacts must be non-empty and unique")
    return paths


def _resolve_families(value: str) -> tuple[str, ...]:
    # Keep the historical default bound to local-context summaries.  A global
    # context factor is an explicit post-local-failure diagnostic and must not
    # silently enter an existing ``--families all`` run.
    default_families = tuple(
        name
        for name, spec in FROZEN_FULLTRACK_RESIDUAL_FAMILIES.items()
        if spec.summary_kind == "local_context_summary"
    )
    families = (
        default_families
        if str(value).strip() == "all"
        else tuple(item.strip() for item in str(value).split(",") if item.strip())
    )
    if not families or len(set(families)) != len(families):
        raise ValueError("full-track residual families must be non-empty and unique")
    supported = set(FROZEN_FULLTRACK_RESIDUAL_FAMILIES).union(
        FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES
    )
    unsupported = set(families).difference(supported)
    if unsupported:
        raise ValueError(f"unsupported full-track residual families: {sorted(unsupported)}")
    return families


def _family_seed(seed: int, family: str) -> int:
    digest = hashlib.sha256(str(family).encode()).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def _artifact_entries(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path), "sha256": file_sha256_short(path)} for path in paths
    ]


def _expected_query_counts(features: Any) -> dict[str, int]:
    return {
        split: int(len(set(features.query_ids[features.split_names == split].tolist())))
        for split in EXPECTED_QUERY_COUNTS
    }


def _save_model(
    *,
    path: Path,
    model: torch.nn.Module,
    normalizer: Any,
    metadata: Mapping[str, Any],
) -> None:
    if isinstance(normalizer, FulltrackAppearanceFeatureNormalizer):
        normalizer_state = {
            "kind": "summary_feature_standardization_v1",
            "mean": np.asarray(normalizer.mean, dtype=np.float32),
            "scale": np.asarray(normalizer.scale, dtype=np.float32),
            "feature_indices": np.asarray(
                normalizer.feature_indices, dtype=np.int64
            ),
            "input_mode": str(normalizer.input_mode),
        }
    elif isinstance(normalizer, FulltrackSummaryTop4RelativeNormalizer):
        normalizer_state = {
            "kind": "summary_top4_relative_scale_v1",
            "scale": np.asarray(normalizer.scale, dtype=np.float32),
            "feature_indices": np.asarray(
                normalizer.feature_indices, dtype=np.int64
            ),
        }
    else:
        raise TypeError(f"unsupported full-track summary normalizer: {type(normalizer)!r}")
    torch.save(
        {
            "format": FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_MODEL_FORMAT,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "normalizer": normalizer_state,
            "metadata": dict(metadata),
        },
        path,
    )


def fit_frozen_fulltrack_candidate_appearance_residual(
    *,
    appearance_artifacts: Sequence[Path],
    colmap_model_dir: Path,
    expected_identity_colmap_images_sha256: str,
    output_dir: Path,
    families: Sequence[str],
    registered_identity_radius_px: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Fit predeclared aggregate residuals without materializing validation labels."""

    if (
        float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
        or not str(expected_identity_colmap_images_sha256).strip()
    ):
        raise ValueError("full-track residual fit arguments are invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_appearance_features(paths)
    unexpected_splits = set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS)
    if unexpected_splits:
        raise ValueError(f"full-track residual fit has unsupported splits: {unexpected_splits}")
    actual_query_counts = _expected_query_counts(features)
    if actual_query_counts != EXPECTED_QUERY_COUNTS:
        raise ValueError(
            "full-track residual fit needs the complete frozen train/validation query set"
        )
    train_rows = np.flatnonzero(features.split_names == "train")
    validation_rows = np.flatnonzero(features.split_names == "validation")
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("full-track residual fit needs both train and validation artifacts")
    identity_images_path = Path(colmap_model_dir) / "images.bin"
    identity_images_sha256 = file_sha256_short(identity_images_path)
    if identity_images_sha256 != str(expected_identity_colmap_images_sha256).strip():
        raise ValueError(
            "full-track residual identity COLMAP model differs from the frozen "
            "hypothesis/source lineage"
        )
    images = read_colmap_images_binary(identity_images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}

    # This is the only target join in the fit command.  The normalizer below is
    # fit from all frozen train values, not from labels or hard-pair membership.
    train_targets = registered_query_observation_targets(
        query_ids=features.query_ids[train_rows],
        query_xy=features.xy[train_rows],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    train_labels_all = registered_candidate_identity_labels(
        features.candidate_track_ids[train_rows], train_targets
    )
    retrieved_identity = train_targets.supervised & np.any(train_labels_all, axis=1)
    target_train_rows = train_rows[retrieved_identity]
    target_membership = train_labels_all[retrieved_identity]
    if len(target_train_rows) == 0:
        raise ValueError("no retrieved registered train identities are available")
    if np.any(target_membership.sum(axis=1) != 1):
        raise RuntimeError("retrieved train identity does not map to one frozen candidate")

    device_value = torch.device(str(device))
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("full-track residual fit requested unavailable CUDA device")
    if device_value.type == "cuda":
        torch.cuda.set_device(device_value)
    output.mkdir(parents=True, exist_ok=False)
    resolved_families = tuple(str(family) for family in families)
    summary_top4_family_set = set(FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES)
    if summary_top4_family_set.intersection(resolved_families) and not set(
        resolved_families
    ).issubset(summary_top4_family_set):
        raise ValueError(
            "summary top-four and historical linear residual families must not mix"
        )
    family_candidates: list[np.ndarray] = []
    family_nulls: list[np.ndarray] = []
    family_residuals: list[np.ndarray] = []
    model_paths: dict[str, str] = {}
    fit_details: dict[str, Any] = {}
    artifact_entries = _artifact_entries(paths)
    for family in resolved_families:
        if family in FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES:
            family_spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES[family]
            family_seed_key = str(family_spec.training_seed_key)
            model, normalizer, detail = fit_fixedprior_fulltrack_summary_top4_probe(
                features=features,
                family=family,
                train_normalizer_rows=train_rows,
                target_train_rows=target_train_rows,
                target_candidate_membership=target_membership,
                device=device_value,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                seed=_family_seed(int(seed), family_seed_key),
            )
            candidate, null, residual = predict_fixedprior_fulltrack_summary_top4_probe(
                model=model,
                features=features,
                normalizer=normalizer,
                device=device_value,
                batch_size=int(batch_size),
            )
        else:
            family_seed_key = family
            model, normalizer, detail = fit_fixedprior_fulltrack_linear_residual(
                features=features,
                family=family,
                train_normalizer_rows=train_rows,
                target_train_rows=target_train_rows,
                target_candidate_membership=target_membership,
                device=device_value,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                seed=_family_seed(int(seed), family_seed_key),
            )
            candidate, null, residual = predict_fixedprior_fulltrack_linear_residual(
                model=model,
                features=features,
                normalizer=normalizer,
                device=device_value,
                batch_size=int(batch_size),
            )
        if (
            np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
            or np.max(np.abs(null - features.null_probabilities)) > 0.0
        ):
            raise RuntimeError("full-track residual prediction violated fixed-mass contract")
        model_path = output / f"{family}.pt"
        model_metadata = {
            **detail,
            "training_seed_key": family_seed_key,
            "appearance_artifacts": artifact_entries,
            "fulltrack_compatibility": dict(features.compatibility),
            "training_supervision_split": "train",
            "validation_or_test_labels_used_by_fit": False,
            "registered_identity_radius_px": float(registered_identity_radius_px),
            "identity_supervision_colmap_images_bin": str(identity_images_path),
            "identity_supervision_colmap_images_sha256": identity_images_sha256,
            "diagnostic_only": True,
            "pose_scoring": False,
        }
        _save_model(
            path=model_path,
            model=model,
            normalizer=normalizer,
            metadata=model_metadata,
        )
        model_paths[family] = str(model_path)
        fit_details[family] = model_metadata
        family_candidates.append(candidate)
        family_nulls.append(null)
        family_residuals.append(residual)
        del model
        if device_value.type == "cuda":
            torch.cuda.empty_cache()

    baseline_candidate, baseline_null = zero_residual_fulltrack_posterior(features)
    metadata: dict[str, Any] = {
        "format": FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "prediction_frozen_before_validation_target_join": True,
        "training_supervision_split": "train",
        "validation_or_test_labels_used_by_fit": False,
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "identity_supervision_colmap_images_bin": str(identity_images_path),
        "identity_supervision_colmap_images_sha256": identity_images_sha256,
        "appearance_artifacts": artifact_entries,
        "fulltrack_compatibility": dict(features.compatibility),
        "families": list(resolved_families),
        "family_architectures": {
            family: str(fit_details[family]["architecture"])
            for family in resolved_families
        },
        "family_evidence_contracts": {
            family: {
                key: fit_details[family][key]
                for key in (
                    "architecture",
                    "candidate_evidence_transform",
                    "summary_statistic",
                    "profile_names",
                    "profile_feature_names",
                    "training_objective",
                    "rank2_hard_pair_weight",
                    "coarse_top1_stability_weight",
                    "residual_scale",
                    "residual_cap",
                    "missing_evidence_semantics",
                    "null_handling",
                    "candidate_mass_handling",
                    "per_view_model",
                )
                if key in fit_details[family]
            }
            for family in resolved_families
        },
        "models": model_paths,
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "support_reselection": False,
        "support_view_selection": False,
        "all_observation_aggregation_preserved": True,
        "feature_granularity": (
            SUMMARY_TOP4_FEATURE_GRANULARITY
            if set(resolved_families).issubset(summary_top4_family_set)
            else "candidate_summary_aggregate_not_per_view_v1"
        ),
        "per_view_model": False,
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "zero_residual_reproduces_input_posterior": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "pose_scoring": False,
    }
    prediction_path = output / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_ids=features.query_ids,
            split_names=features.split_names,
            source_row_indices=features.source_row_indices,
            candidate_track_ids=features.candidate_track_ids,
            family_names=np.asarray(resolved_families, dtype=np.str_),
            candidate_probabilities=np.stack(family_candidates, axis=0).astype(np.float32),
            null_probabilities=np.stack(family_nulls, axis=0).astype(np.float32),
            candidate_residuals=np.stack(family_residuals, axis=0).astype(np.float32),
            baseline_candidate_probabilities=baseline_candidate,
            baseline_null_probabilities=baseline_null,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "fit_frozen_fulltrack_candidate_appearance_conditional_residual",
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256_short(prediction_path),
        "families": fit_details,
        "train_target_audit": {
            "train_row_count": int(len(train_rows)),
            "supervised_train_row_count": int(np.sum(train_targets.supervised)),
            "retrieved_identity_train_row_count": int(np.sum(retrieved_identity)),
            "explicit_null_or_topl_miss_train_row_count": int(
                np.sum(train_targets.supervised & ~np.any(train_labels_all, axis=1))
            ),
            "unsupervised_train_row_count": int(np.sum(~train_targets.supervised)),
            "normalizer_uses_all_frozen_train_rows": True,
            "validation_or_test_target_used": False,
        },
        "identity_supervision": {
            "colmap_images_bin": str(identity_images_path),
            "colmap_images_sha256": identity_images_sha256,
        },
        "baseline_equivalence": {
            "candidate_max_abs_error": float(
                np.max(np.abs(baseline_candidate - features.candidate_probabilities))
            ),
            "null_max_abs_error": float(
                np.max(np.abs(baseline_null - features.null_probabilities))
            ),
        },
        "protocol": {
            "feature_export_target_free": True,
            "fit_uses_train_targets_only": True,
            "validation_or_test_targets_used": False,
            "prediction_frozen_before_validation_target_join": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "all_observation_aggregation_preserved": True,
            "per_view_s2_claimed": False,
            "input_null_probability_exactly_preserved": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_scoring": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = fit_frozen_fulltrack_candidate_appearance_residual(
        appearance_artifacts=_paths(args.appearance_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        expected_identity_colmap_images_sha256=str(
            args.expected_identity_colmap_images_sha256
        ),
        output_dir=Path(args.output_dir),
        families=_resolve_families(args.families),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
