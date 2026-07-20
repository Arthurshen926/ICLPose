"""Fit current-V3 multi-source appearance probes on train rows only.

The input feature artifact contains fixed global top-L candidates and real
query/support image evidence only.  V3 reprojection residuals are read solely
for its train rows; validation labels are deliberately unavailable to fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.current_v3_candidate_probe import (
    CURRENT_V3_MODEL_FORMAT,
    CURRENT_V3_PREDICTION_FORMAT,
    EXACT_IDENTITY_PROBABILITY_SEMANTICS,
    GEOMETRIC_SET_SUPERVISION_MODE,
    GEOMETRIC_PROBABILITY_SEMANTICS,
    REGISTERED_TRACK_IDENTITY_OBJECTIVE,
    SET_MEMBERSHIP_OBJECTIVE,
    SUPPORTED_SUPERVISION_MODES,
    align_current_v3_features_and_evidence,
    load_current_v3_evidence_inference,
    load_current_v3_features,
    stable_family_seed,
    train_geometric_target_membership,
    train_registered_track_identity_membership,
    validate_candidate_probability_contract,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    PerViewFeatureNormalizer,
    predict_per_view_candidate_probe,
    train_per_view_candidate_probe,
)


DEFAULT_FAMILIES = (
    "multisource_landmark_region_radio_intermediate_appearance_only",
    "multisource_landmark_region_alike_appearance_only",
    "multisource_landmark_region_candidate_specific_appearance_only",
    "multisource_landmark_region_with_anchor_appearance_only",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--families",
        default="all",
        help="all or a comma-separated subset of the fixed appearance-only families",
    )
    parser.add_argument("--geometric_positive_threshold_px", type=float, default=5.0)
    parser.add_argument(
        "--supervision_mode",
        choices=tuple(sorted(SUPPORTED_SUPERVISION_MODES)),
        default=GEOMETRIC_SET_SUPERVISION_MODE,
        help=(
            "geometric_set uses 2D residual membership; "
            "registered_track_identity uses train SfM point2D-to-point3D identities"
        ),
    )
    parser.add_argument(
        "--colmap_model_dir",
        default="",
        help="required only for --supervision_mode=registered_track_identity",
    )
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--architecture", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--no_prior_residual",
        action="store_true",
        help="fit an appearance-only posterior instead of residuals over frozen V3 priors",
    )
    return parser.parse_args(argv)


def _resolve_families(value: str) -> tuple[str, ...]:
    names = (
        DEFAULT_FAMILIES
        if str(value).strip() == "all"
        else tuple(part.strip() for part in str(value).split(",") if part.strip())
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("current-V3 probe families must be a non-empty unique list")
    unsupported = set(names) - set(DEFAULT_FAMILIES)
    if unsupported:
        raise ValueError(
            "current-V3 probe permits only predeclared appearance-only families: "
            f"{sorted(unsupported)}"
        )
    for name in names:
        if name not in ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES:
            raise RuntimeError(f"current-V3 probe family is not registered: {name}")
    return names


def _save_model(
    *,
    path: Path,
    model: torch.nn.Module,
    normalizer: PerViewFeatureNormalizer,
    family: str,
    metadata: dict[str, Any],
) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "format": CURRENT_V3_MODEL_FORMAT,
            "family": str(family),
            "state_dict": state,
            "normalizer": {
                "mean": normalizer.mean,
                "scale": normalizer.scale,
                "feature_indices": normalizer.feature_indices,
            },
            "metadata": metadata,
        },
        path,
    )


def fit_current_v3_multisource_candidate_probe(
    *,
    features_path: Path,
    candidate_evidence_path: Path,
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
    prior_residual: bool,
    supervision_mode: str = GEOMETRIC_SET_SUPERVISION_MODE,
    colmap_model_dir: Path | None = None,
    registered_identity_radius_px: float = 2.0,
) -> dict[str, Any]:
    mode = str(supervision_mode)
    if (
        float(geometric_positive_threshold_px) <= 0.0
        or float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or int(hidden_dim) <= 0
        or mode not in SUPPORTED_SUPERVISION_MODES
    ):
        raise ValueError("current-V3 probe optimization arguments are invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    resolved_families = _resolve_families(",".join(str(value) for value in families))
    features = load_current_v3_features(Path(features_path))
    evidence = load_current_v3_evidence_inference(Path(candidate_evidence_path))
    aligned = align_current_v3_features_and_evidence(features, evidence)
    if mode == GEOMETRIC_SET_SUPERVISION_MODE:
        train_rows, train_membership, target_audit = train_geometric_target_membership(
            Path(candidate_evidence_path),
            aligned,
            threshold_px=float(geometric_positive_threshold_px),
        )
        if not np.array_equal(train_rows, np.flatnonzero(features.split_names == "train")):
            raise RuntimeError("current-V3 train label rows differ from frozen train split")
        probability_semantics = GEOMETRIC_PROBABILITY_SEMANTICS
        training_objective = SET_MEMBERSHIP_OBJECTIVE
    else:
        if colmap_model_dir is None or not str(colmap_model_dir).strip():
            raise ValueError("registered-track supervision requires --colmap_model_dir")
        train_rows, train_membership, target_audit = train_registered_track_identity_membership(
            features,
            colmap_model_dir=Path(colmap_model_dir),
            identity_radius_px=float(registered_identity_radius_px),
        )
        if np.any(features.split_names[train_rows] != "train"):
            raise RuntimeError("registered-track targets include a held-out feature row")
        probability_semantics = EXACT_IDENTITY_PROBABILITY_SEMANTICS
        training_objective = REGISTERED_TRACK_IDENTITY_OBJECTIVE
    train_groups = np.zeros((len(features.source_rows),), dtype=bool)
    train_groups[train_rows] = True
    membership = np.zeros(
        (len(features.source_rows), features.candidate_tracks.shape[1] + 1), dtype=bool
    )
    membership[:, -1] = True
    membership[train_rows] = train_membership
    selected_device = torch.device(str(device))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {selected_device}")
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    candidate_predictions: list[np.ndarray] = []
    null_predictions: list[np.ndarray] = []
    view_logits: list[np.ndarray] = []
    fit_rows: list[dict[str, Any]] = []
    for family in resolved_families:
        family_seed = stable_family_seed(int(seed), family)
        model, normalizer, fit_metadata = train_per_view_candidate_probe(
            features=features.candidate_features,
            view_valid=features.candidate_view_valid,
            train_groups=train_groups,
            target_membership=membership,
            base_candidate_probabilities=(
                aligned.base_candidate_probabilities if bool(prior_residual) else None
            ),
            base_null_probabilities=(
                aligned.base_null_probabilities if bool(prior_residual) else None
            ),
            family=family,
            device=selected_device,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            seed=family_seed,
            architecture=str(architecture),
            hidden_dim=int(hidden_dim),
            feature_names=features.feature_names,
        )
        candidate, null, per_view = predict_per_view_candidate_probe(
            model,
            features=features.candidate_features,
            view_valid=features.candidate_view_valid,
            normalizer=normalizer,
            device=selected_device,
            batch_size=int(batch_size),
            base_candidate_probabilities=(
                aligned.base_candidate_probabilities if bool(prior_residual) else None
            ),
            base_null_probabilities=(
                aligned.base_null_probabilities if bool(prior_residual) else None
            ),
        )
        validate_candidate_probability_contract(
            candidate, null, features.candidate_tracks >= 0
        )
        model_metadata = {
            "features_sha256": file_sha256_short(Path(features_path)),
            "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence_path)),
            "family": family,
            "fit": fit_metadata,
            "training_supervision_split": "train",
            "validation_or_test_labels_used_by_fit": False,
            "prior_residual": bool(prior_residual),
            "supervision_mode": mode,
            "training_objective": training_objective,
            "probability_semantics": probability_semantics,
        }
        _save_model(
            path=model_dir / f"{family}.pt",
            model=model,
            normalizer=normalizer,
            family=family,
            metadata=model_metadata,
        )
        candidate_predictions.append(candidate.astype(np.float32, copy=False))
        null_predictions.append(null.astype(np.float32, copy=False))
        view_logits.append(per_view.astype(np.float16, copy=False))
        fit_rows.append({"family": family, **fit_metadata})
    prediction_metadata = {
        "format": CURRENT_V3_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "prediction_frozen_before_validation_target_join": True,
        "features": str(Path(features_path)),
        "features_sha256": file_sha256_short(Path(features_path)),
        "candidate_evidence": str(Path(candidate_evidence_path)),
        "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence_path)),
        "candidate_top_k": int(features.candidate_tracks.shape[1]),
        "support_view_count": int(features.candidate_view_valid.shape[2]),
        "family_names": list(resolved_families),
        "feature_names": list(features.feature_names),
        "training_supervision": target_audit,
        "training_supervision_split": "train",
        "supervision_mode": mode,
        "training_objective": training_objective,
        "validation_or_test_labels_used_by_fit": False,
        "source_test_rows_materialized": False,
        "prior_residual": bool(prior_residual),
        "probability_semantics": probability_semantics,
        "candidate_set": "fixed_current_v3_global_top_l",
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    predictions_path = output / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        source_row_indices=features.source_rows,
        query_ids=features.query_ids,
        split_names=features.split_names,
        candidate_track_ids=features.candidate_tracks,
        candidate_canonical_rows=features.candidate_canonical_rows,
        candidate_view_valid=features.candidate_view_valid,
        family_names=np.asarray(resolved_families),
        candidate_probabilities=np.stack(candidate_predictions, axis=0),
        null_probabilities=np.stack(null_predictions, axis=0),
        per_view_logits=np.stack(view_logits, axis=0),
        metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
    )
    summary = {
        "stage": "fit_current_v3_multisource_candidate_probe",
        "predictions": str(predictions_path),
        "predictions_sha256": file_sha256_short(predictions_path),
        "families": fit_rows,
        "protocol": {
            "fixed_global_top_l": True,
            "candidate_retrieval_or_reselection": False,
            "feature_export_target_free": True,
            "training_target_split": "train",
            "supervision_mode": mode,
            "training_objective": training_objective,
            "validation_or_test_labels_used_by_fit": False,
            "prior_residual": bool(prior_residual),
            "render": False,
            "colmap_model_dir": (
                None
                if mode == GEOMETRIC_SET_SUPERVISION_MODE
                else str(Path(colmap_model_dir))
            ),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_current_v3_multisource_candidate_probe(
        features_path=Path(args.features),
        candidate_evidence_path=Path(args.candidate_evidence),
        output_dir=Path(args.output_dir),
        families=_resolve_families(str(args.families)),
        geometric_positive_threshold_px=float(args.geometric_positive_threshold_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        device=str(args.device),
        architecture=str(args.architecture),
        hidden_dim=int(args.hidden_dim),
        prior_residual=not bool(args.no_prior_residual),
        supervision_mode=str(args.supervision_mode),
        colmap_model_dir=(
            None if not str(args.colmap_model_dir).strip() else Path(args.colmap_model_dir)
        ),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
