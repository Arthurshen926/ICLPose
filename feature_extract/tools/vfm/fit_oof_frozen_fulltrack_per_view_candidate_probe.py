"""Build sequence-grouped OOF train predictions for one frozen per-view family.

This is a calibration/selection diagnostic, not a validation evaluation.  It
fits one model per held-out *training sequence*, uses labels only from the
other sequences, and writes OOF probabilities into the original full-row
layout.  Validation rows retain the immutable baseline and never receive a
target join.  The resulting artifact can therefore be audited with
``--audit-split train`` to select a predeclared residual capacity or bound
without consuming validation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
    fit_fixedprior_fulltrack_per_view_probe,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
    predict_fixedprior_fulltrack_per_view_probe,
    zero_residual_fulltrack_per_view_posterior,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
)


EXPECTED_QUERY_COUNTS = {"train": 63, "validation": 21}
EXPECTED_ROWS_PER_QUERY = 192


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--expected-identity-colmap-images-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument(
        "--per-view-residual-architecture",
        choices=("mlp", "linear"),
        required=True,
    )
    parser.add_argument("--per-view-residual-cap", type=float, required=True)
    parser.add_argument("--per-view-residual-cap-provenance", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("OOF per-view artifacts must be non-empty and unique")
    return paths


def _artifact_entries(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": file_sha256_short(path)} for path in paths]


def _validate_complete_rows(features: Any) -> None:
    for split, expected_query_count in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique) != expected_query_count:
            raise ValueError("OOF per-view input has an incomplete query split")
        for query_id in unique:
            query_rows = rows[query_ids == query_id]
            if (
                len(query_rows) != EXPECTED_ROWS_PER_QUERY
                or len(np.unique(features.source_row_indices[query_rows]))
                != len(query_rows)
            ):
                raise ValueError("OOF per-view input has an incomplete query shard")


def _sequence_group(query_id: str) -> str:
    """Return the sequence prefix without using a target or pose."""

    normalized = str(query_id).replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("OOF query ID is empty")
    first = normalized.split("/", 1)[0]
    if not first.startswith("seq"):
        raise ValueError("OOF sequence grouping requires seq*/image query IDs")
    return first


def _stable_sequence_folds(
    query_ids: Sequence[str], *, folds: int, seed: int
) -> dict[str, int]:
    """Assign every training sequence to a deterministic, balanced fold."""

    if int(folds) < 2:
        raise ValueError("OOF sequence folding needs at least two folds")
    sequences = sorted({_sequence_group(str(query_id)) for query_id in query_ids})
    if len(sequences) < int(folds):
        raise ValueError("OOF sequence folding has fewer groups than folds")
    ordered = sorted(
        sequences,
        key=lambda sequence: hashlib.sha256(
            f"{int(seed)}:{sequence}".encode("utf-8")
        ).digest(),
    )
    return {sequence: index % int(folds) for index, sequence in enumerate(ordered)}


def _family_seed(seed: int, family: str, fold: int) -> int:
    digest = hashlib.sha256(f"{family}:{fold}".encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def fit_oof_frozen_fulltrack_per_view_candidate_probe(
    *,
    appearance_artifacts: Sequence[Path],
    colmap_model_dir: Path,
    expected_identity_colmap_images_sha256: str,
    output_dir: Path,
    family: str,
    folds: int,
    registered_identity_radius_px: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    residual_architecture: str,
    residual_cap: float,
    residual_cap_provenance: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if (
        str(family) not in FULLTRACK_PER_VIEW_FAMILIES
        or FULLTRACK_PER_VIEW_FAMILIES[str(family)].architecture
        != "sparse_per_view_mixture"
        or int(folds) < 2
        or float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
        or int(hidden_dim) <= 0
        or str(residual_architecture) not in {"mlp", "linear"}
        or not np.isfinite(float(residual_cap))
        or float(residual_cap) <= 0.0
        or not str(residual_cap_provenance).strip()
    ):
        raise ValueError("OOF per-view configuration is invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_per_view_appearance_features(paths)
    _validate_complete_rows(features)
    spec = FULLTRACK_PER_VIEW_FAMILIES[str(family)]
    if str(spec.edge_feature_semantics) != str(
        features.compatibility.get("per_view_edge_feature_semantics", "")
    ):
        raise ValueError("OOF family and appearance artifact semantics differ")
    identity_images_path = Path(colmap_model_dir) / "images.bin"
    identity_images_sha256 = file_sha256_short(identity_images_path)
    if identity_images_sha256 != str(expected_identity_colmap_images_sha256).strip():
        raise ValueError("OOF identity supervision images.bin differs from source lineage")
    torch_device = torch.device(str(device))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("OOF probe requested an unavailable CUDA device")
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    images = read_colmap_images_binary(identity_images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    train_rows = np.flatnonzero(features.split_names == "train")
    validation_rows = np.flatnonzero(features.split_names == "validation")
    train_query_ids = features.query_ids[train_rows].astype(str)
    sequence_folds = _stable_sequence_folds(train_query_ids, folds=int(folds), seed=int(seed))
    row_folds = np.asarray(
        [sequence_folds[_sequence_group(query_id)] for query_id in train_query_ids],
        dtype=np.int64,
    )
    baseline_candidate, baseline_null = zero_residual_fulltrack_per_view_posterior(features)
    oof_candidate = baseline_candidate.copy()
    oof_residual = np.zeros_like(oof_candidate, dtype=np.float32)
    fold_summaries: list[dict[str, Any]] = []
    representative_detail: dict[str, Any] | None = None
    for fold in range(int(folds)):
        fit_rows = train_rows[row_folds != fold]
        heldout_rows = train_rows[row_folds == fold]
        if not len(fit_rows) or not len(heldout_rows):
            raise RuntimeError("OOF sequence fold is empty")
        # Crucially, targets are joined only for the fit sequences.  Held-out
        # sequence labels are not materialized by this loop at all.
        fit_targets = registered_query_observation_targets(
            query_ids=features.query_ids[fit_rows],
            query_xy=features.xy[fit_rows],
            images_by_name=images_by_name,
            max_distance_px=float(registered_identity_radius_px),
        )
        fit_labels = registered_candidate_identity_labels(
            features.candidate_track_ids[fit_rows], fit_targets
        )
        retrieved = fit_targets.supervised & np.any(fit_labels, axis=1)
        target_rows = fit_rows[retrieved]
        membership = fit_labels[retrieved]
        if not len(target_rows) or np.any(membership.sum(axis=1) != 1):
            raise RuntimeError("OOF fit fold has no singleton retrieved identity targets")
        model, normalizer, detail = fit_fixedprior_fulltrack_per_view_probe(
            features=features,
            family=str(family),
            train_normalizer_rows=fit_rows,
            target_train_rows=target_rows,
            target_candidate_membership=membership,
            device=torch_device,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            hidden_dim=int(hidden_dim),
            seed=_family_seed(int(seed), str(family), fold),
            # ``None`` retains the predeclared family setting.  The absolute
            # phase family is identity-NLL-only (weight zero); passing an
            # explicit zero is correctly rejected by the core API because
            # explicit overrides are required to be positive.
            rank2_hard_pair_weight=None,
            residual_architecture=str(residual_architecture),
            residual_cap=float(residual_cap),
        )
        candidate, null, residual = predict_fixedprior_fulltrack_per_view_probe(
            model=model,
            features=features,
            normalizer=normalizer,
            device=torch_device,
            batch_size=int(batch_size),
            rows=heldout_rows,
        )
        if (
            np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
            or np.max(np.abs(null - features.null_probabilities[heldout_rows])) > 0.0
        ):
            raise RuntimeError("OOF held-out prediction changed fixed posterior mass")
        oof_candidate[heldout_rows] = candidate
        oof_residual[heldout_rows] = residual
        representative_detail = detail
        fold_summaries.append(
            {
                "fold": int(fold),
                "heldout_sequences": sorted(
                    sequence
                    for sequence, assigned_fold in sequence_folds.items()
                    if int(assigned_fold) == int(fold)
                ),
                "fit_row_count": int(len(fit_rows)),
                "heldout_row_count": int(len(heldout_rows)),
                "fit_retrieved_identity_row_count": int(len(target_rows)),
                "detail": detail,
            }
        )
        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    if representative_detail is None:
        raise RuntimeError("OOF probe did not fit a fold")
    if (
        np.max(np.abs(oof_candidate.sum(axis=1) + baseline_null - 1.0)) > 2e-5
        or np.max(np.abs(baseline_null - features.null_probabilities)) > 0.0
        or np.any(oof_candidate[features.candidate_probabilities <= 0.0] > 2e-6)
    ):
        raise RuntimeError("OOF prediction violated fixed candidate/null mass")
    output.mkdir(parents=True, exist_ok=False)
    contract = {
        "architecture": str(representative_detail["architecture"]),
        "support_view_marginalization": str(
            representative_detail["support_view_marginalization"]
        ),
        "missing_evidence_semantics": str(
            representative_detail["missing_evidence_semantics"]
        ),
        "zero_residual_reproduces_fixed_posterior": bool(
            representative_detail["zero_residual_reproduces_fixed_posterior"]
        ),
        "residual_architecture": str(representative_detail["residual_architecture"]),
        "residual_cap": float(residual_cap),
        "residual_cap_semantics": str(
            representative_detail["residual_cap_semantics"]
        ),
        "residual_cap_provenance": str(residual_cap_provenance).strip(),
    }
    metadata: dict[str, Any] = {
        "format": FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "prediction_frozen_before_validation_target_join": True,
        "training_supervision_split": "train",
        "validation_or_test_labels_used_by_fit": False,
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "appearance_artifacts": _artifact_entries(paths),
        "fulltrack_compatibility": dict(features.compatibility),
        "identity_supervision_colmap_images_bin": str(identity_images_path),
        "identity_supervision_colmap_images_sha256": identity_images_sha256,
        "families": [str(family)],
        "family_architectures": {str(family): str(spec.architecture)},
        "family_edge_feature_semantics": {
            str(family): str(spec.edge_feature_semantics)
        },
        "family_evidence_contracts": {str(family): contract},
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "support_reselection": False,
        "support_view_selection": False,
        "all_real_sfm_support_observations_retained": True,
        "feature_granularity": fulltrack_per_view_feature_granularity(features),
        "per_view_model": True,
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "missing_evidence_semantics": "joint_profile_missing_edge_omitted_neutral_v1",
        "stored_baseline_reproduces_input_posterior": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "pose_scoring": False,
        "train_prediction_mode": "sequence_grouped_oof_v1",
        "train_oof": {
            "fold_count": int(folds),
            "grouping": "query_sequence_prefix_v1",
            "sequence_folds": sequence_folds,
            "all_train_rows_predicted_out_of_fold": True,
            "validation_rows_are_immutable_baseline": True,
            "heldout_fold_labels_used_by_fit": False,
            "residual_cap_provenance": str(residual_cap_provenance).strip(),
        },
    }
    prediction_path = output / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_ids=features.query_ids,
            split_names=features.split_names,
            source_row_indices=features.source_row_indices,
            candidate_track_ids=features.candidate_track_ids,
            family_names=np.asarray([str(family)], dtype=np.str_),
            candidate_probabilities=oof_candidate[None].astype(np.float32),
            null_probabilities=baseline_null[None].astype(np.float32),
            candidate_residuals=oof_residual[None].astype(np.float32),
            baseline_candidate_probabilities=baseline_candidate,
            baseline_null_probabilities=baseline_null,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "fit_oof_frozen_fulltrack_per_view_candidate_probe",
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256_short(prediction_path),
        "family": str(family),
        "residual_architecture": str(residual_architecture),
        "residual_cap": float(residual_cap),
        "residual_cap_provenance": str(residual_cap_provenance).strip(),
        "folds": fold_summaries,
        "protocol": {
            "feature_export_target_free": True,
            "fit_uses_fit_fold_train_targets_only": True,
            "heldout_train_fold_targets_used": False,
            "validation_or_test_targets_used": False,
            "all_train_rows_predicted_out_of_fold": True,
            "validation_rows_immutable_baseline": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "all_real_sfm_support_observations_retained": True,
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
    summary = fit_oof_frozen_fulltrack_per_view_candidate_probe(
        appearance_artifacts=_paths(args.appearance_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        expected_identity_colmap_images_sha256=str(
            args.expected_identity_colmap_images_sha256
        ),
        output_dir=Path(args.output_dir),
        family=str(args.family),
        folds=int(args.folds),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        residual_architecture=str(args.per_view_residual_architecture),
        residual_cap=float(args.per_view_residual_cap),
        residual_cap_provenance=str(args.per_view_residual_cap_provenance),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
