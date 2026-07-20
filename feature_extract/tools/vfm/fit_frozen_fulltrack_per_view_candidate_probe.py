"""Fit train-only sparse per-view full-track appearance probes.

The command only learns a diagnostic residual inside a frozen top-20 candidate
mass.  It cannot change candidate availability, null probability, support
observations, or pose hypotheses.  Validation identities remain unavailable
until the separate audit process reads the frozen prediction artifact.
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
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_MODEL_FORMAT,
    FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
    FulltrackPerViewNormalizer,
    FulltrackRawTop4RelativeNormalizer,
    RAW_TOP4_ARCHITECTURES,
    fit_fixedprior_fulltrack_per_view_probe,
    fit_fixedprior_fulltrack_rawtop4_probe,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
    predict_fixedprior_fulltrack_per_view_probe,
    predict_fixedprior_fulltrack_rawtop4_probe,
    zero_residual_fulltrack_per_view_posterior,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
)


EXPECTED_QUERY_COUNTS = {"train": 63, "validation": 21}
EXPECTED_ROWS_PER_QUERY = 192
_COMPATIBLE_FAMILIES_SENTINEL = "__compatible_families__"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--per-view-residual-architecture",
        choices=("mlp", "linear"),
        default="mlp",
        help="train-only residual capacity; candidate/null mass remains frozen",
    )
    parser.add_argument(
        "--per-view-residual-cap",
        type=float,
        default=None,
        help="optional symmetric log-likelihood-ratio bound applied before softmax",
    )
    parser.add_argument(
        "--per-view-residual-cap-provenance",
        default="",
        help="required train-only/OFF provenance whenever a residual cap is used",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--rank2-hard-pair-weight",
        type=float,
        default=None,
        help="train-only loss ablation; default keeps the family-fixed weight",
    )
    parser.add_argument("--raw-top4-postfit-residual-scale", type=float, default=1.0)
    parser.add_argument("--raw-top4-postfit-scale-provenance", default="")
    parser.add_argument("--raw-top4-postfit-residual-cap", type=float, default=None)
    parser.add_argument("--raw-top4-postfit-cap-provenance", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("per-view appearance artifacts must be non-empty and unique")
    return paths


def _families(value: str) -> tuple[str, ...]:
    result = (
        (_COMPATIBLE_FAMILIES_SENTINEL,)
        if str(value).strip() == "all"
        else tuple(item.strip() for item in str(value).split(",") if item.strip())
    )
    if not result or len(set(result)) != len(result):
        raise ValueError("per-view probe families must be non-empty and unique")
    unsupported = set(result).difference(
        {*FULLTRACK_PER_VIEW_FAMILIES, _COMPATIBLE_FAMILIES_SENTINEL}
    )
    if unsupported:
        raise ValueError(f"unsupported per-view probe families: {sorted(unsupported)}")
    return result


def _artifact_entries(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": file_sha256_short(path)} for path in paths]


def _family_seed(seed: int, family: str) -> int:
    digest = hashlib.sha256(str(family).encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def _query_counts(features: Any) -> dict[str, int]:
    return {
        split: int(len(set(features.query_ids[features.split_names == split].tolist())))
        for split in EXPECTED_QUERY_COUNTS
    }


def _validate_complete_query_rows(features: Any) -> None:
    """Reject partial shards before train targets or validation predictions exist."""

    for split, expected_query_count in EXPECTED_QUERY_COUNTS.items():
        rows = np.flatnonzero(features.split_names == split)
        query_ids = features.query_ids[rows]
        unique_query_ids = tuple(dict.fromkeys(query_ids.tolist()))
        if len(unique_query_ids) != expected_query_count:
            raise ValueError("per-view probe query count is incomplete")
        for query_id in unique_query_ids:
            query_rows = rows[query_ids == query_id]
            if len(query_rows) != EXPECTED_ROWS_PER_QUERY:
                raise ValueError("per-view probe query shard has an incomplete row count")
            if len(np.unique(features.source_row_indices[query_rows])) != len(query_rows):
                raise ValueError("per-view probe query shard repeats source rows")


def _save_model(
    *, path: Path, model: torch.nn.Module, normalizer: Any, metadata: Mapping[str, Any]
) -> None:
    if isinstance(normalizer, FulltrackPerViewNormalizer):
        normalizer_state = {
            "kind": "edge_standardization_v1",
            "mean": np.asarray(normalizer.mean, dtype=np.float32),
            "scale": np.asarray(normalizer.scale, dtype=np.float32),
            "profile_indices": np.asarray(
                normalizer.profile_indices, dtype=np.int64
            ),
        }
    elif isinstance(normalizer, FulltrackRawTop4RelativeNormalizer):
        normalizer_state = {
            "kind": "top1_relative_raw_top4_scale_v1",
            "scale": np.asarray(normalizer.scale, dtype=np.float32),
            "profile_indices": np.asarray(
                normalizer.profile_indices, dtype=np.int64
            ),
            "top_k": int(normalizer.top_k),
            "aggregation": str(normalizer.aggregation),
        }
    else:
        raise TypeError(f"unsupported per-view normalizer: {type(normalizer)!r}")
    torch.save(
        {
            "format": FULLTRACK_PER_VIEW_MODEL_FORMAT,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "normalizer": normalizer_state,
            "metadata": dict(metadata),
        },
        path,
    )


def fit_frozen_fulltrack_per_view_candidate_probe(
    *,
    appearance_artifacts: Sequence[Path],
    colmap_model_dir: Path,
    output_dir: Path,
    families: Sequence[str],
    registered_identity_radius_px: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    seed: int,
    device: str,
    expected_identity_colmap_images_sha256: str,
    rank2_hard_pair_weight: float | None = None,
    per_view_residual_architecture: str = "mlp",
    per_view_residual_cap: float | None = None,
    per_view_residual_cap_provenance: str = "",
    raw_top4_postfit_residual_scale: float = 1.0,
    raw_top4_postfit_scale_provenance: str = "",
    raw_top4_postfit_residual_cap: float | None = None,
    raw_top4_postfit_cap_provenance: str = "",
) -> dict[str, Any]:
    if (
        float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
        or int(hidden_dim) <= 0
        or str(per_view_residual_architecture) not in {"mlp", "linear"}
        or (
            per_view_residual_cap is not None
            and (
                not np.isfinite(float(per_view_residual_cap))
                or float(per_view_residual_cap) <= 0.0
            )
        )
        or not np.isfinite(float(raw_top4_postfit_residual_scale))
        or float(raw_top4_postfit_residual_scale) <= 0.0
        or (
            raw_top4_postfit_residual_cap is not None
            and (
                not np.isfinite(float(raw_top4_postfit_residual_cap))
                or float(raw_top4_postfit_residual_cap) <= 0.0
            )
        )
        or not str(expected_identity_colmap_images_sha256).strip()
        or (
            rank2_hard_pair_weight is not None
            and float(rank2_hard_pair_weight) <= 0.0
        )
    ):
        raise ValueError("full-track per-view fit arguments are invalid")
    if per_view_residual_cap is not None and not str(
        per_view_residual_cap_provenance
    ).strip():
        raise ValueError("bounded per-view residual needs train-only provenance")
    uses_postfit_scale = float(raw_top4_postfit_residual_scale) != 1.0
    uses_postfit_cap = raw_top4_postfit_residual_cap is not None
    if uses_postfit_scale and not str(raw_top4_postfit_scale_provenance).strip():
        raise ValueError("non-unit raw top-4 scale needs train-only provenance")
    if uses_postfit_cap and not str(raw_top4_postfit_cap_provenance).strip():
        raise ValueError("raw top-4 residual cap needs train-only provenance")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_per_view_appearance_features(paths)
    artifact_semantics = str(
        features.compatibility.get("per_view_edge_feature_semantics", "")
    )
    requested_families = tuple(str(item) for item in families)
    if requested_families == (_COMPATIBLE_FAMILIES_SENTINEL,):
        family_names = tuple(
            name
            for name, spec in FULLTRACK_PER_VIEW_FAMILIES.items()
            if str(spec.edge_feature_semantics) == artifact_semantics
        )
    else:
        family_names = requested_families
    if (
        not family_names
        or len(set(family_names)) != len(family_names)
        or set(family_names).difference(FULLTRACK_PER_VIEW_FAMILIES)
        or any(
            str(FULLTRACK_PER_VIEW_FAMILIES[family].edge_feature_semantics)
            != artifact_semantics
            for family in family_names
        )
    ):
        raise ValueError("per-view probe families are incompatible with artifact semantics")
    if set(features.split_names.tolist()).difference(EXPECTED_QUERY_COUNTS):
        raise ValueError("per-view probe has unsupported split names")
    if _query_counts(features) != EXPECTED_QUERY_COUNTS:
        raise ValueError("per-view probe needs the complete frozen train/validation set")
    _validate_complete_query_rows(features)
    train_rows = np.flatnonzero(features.split_names == "train")
    validation_rows = np.flatnonzero(features.split_names == "validation")
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("per-view probe needs both train and validation artifacts")
    identity_images_path = Path(colmap_model_dir) / "images.bin"
    identity_images_sha256 = file_sha256_short(identity_images_path)
    if identity_images_sha256 != str(expected_identity_colmap_images_sha256).strip():
        raise ValueError(
            "identity supervision COLMAP images.bin differs from the declared "
            "frozen source lineage"
        )
    images = read_colmap_images_binary(identity_images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}

    # The only target join occurs here and only for train rows.  The feature
    # normalizer uses every frozen train edge, regardless of identity label.
    train_targets = registered_query_observation_targets(
        query_ids=features.query_ids[train_rows],
        query_xy=features.xy[train_rows],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels_all = registered_candidate_identity_labels(
        features.candidate_track_ids[train_rows], train_targets
    )
    retrieved = train_targets.supervised & np.any(labels_all, axis=1)
    target_rows = train_rows[retrieved]
    membership = labels_all[retrieved]
    if len(target_rows) == 0 or np.any(membership.sum(axis=1) != 1):
        raise ValueError("per-view probe has no singleton retrieved train identities")

    torch_device = torch.device(str(device))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("per-view probe requested an unavailable CUDA device")
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    output.mkdir(parents=True, exist_ok=False)
    artifact_entries = _artifact_entries(paths)
    candidate_blocks: list[np.ndarray] = []
    null_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    models: dict[str, str] = {}
    details: dict[str, Any] = {}
    for family in family_names:
        family_spec = FULLTRACK_PER_VIEW_FAMILIES[family]
        family_seed_key = family_spec.training_seed_key or family
        if family_spec.architecture == "sparse_per_view_mixture":
            if uses_postfit_scale or uses_postfit_cap:
                raise ValueError("raw top-4 post-fit calibration is incompatible with an MLP family")
            model, normalizer, detail = fit_fixedprior_fulltrack_per_view_probe(
                features=features,
                family=family,
                train_normalizer_rows=train_rows,
                target_train_rows=target_rows,
                target_candidate_membership=membership,
                device=torch_device,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                hidden_dim=int(hidden_dim),
                seed=_family_seed(int(seed), family_seed_key),
                rank2_hard_pair_weight=rank2_hard_pair_weight,
                residual_architecture=str(per_view_residual_architecture),
                residual_cap=per_view_residual_cap,
            )
            candidate, null, residual = predict_fixedprior_fulltrack_per_view_probe(
                model=model,
                features=features,
                normalizer=normalizer,
                device=torch_device,
                batch_size=int(batch_size),
            )
        elif family_spec.architecture in RAW_TOP4_ARCHITECTURES:
            model, normalizer, detail = fit_fixedprior_fulltrack_rawtop4_probe(
                features=features,
                family=family,
                train_normalizer_rows=train_rows,
                target_train_rows=target_rows,
                target_candidate_membership=membership,
                device=torch_device,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                seed=_family_seed(int(seed), family_seed_key),
                rank2_hard_pair_weight=rank2_hard_pair_weight,
                postfit_residual_scale=float(raw_top4_postfit_residual_scale),
                postfit_residual_cap=raw_top4_postfit_residual_cap,
            )
            candidate, null, residual = predict_fixedprior_fulltrack_rawtop4_probe(
                model=model,
                features=features,
                normalizer=normalizer,
                device=torch_device,
                batch_size=int(batch_size),
            )
        else:
            raise RuntimeError(f"unsupported per-view architecture: {family_spec.architecture}")
        if (
            np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
            or np.max(np.abs(null - features.null_probabilities)) > 0.0
        ):
            raise RuntimeError("per-view probe violated fixed posterior mass")
        model_path = output / f"{family}.pt"
        model_metadata = {
            **detail,
            "training_seed_key": str(family_seed_key),
            "appearance_artifacts": artifact_entries,
            "fulltrack_compatibility": dict(features.compatibility),
            "identity_supervision_colmap_images_bin": str(identity_images_path),
            "identity_supervision_colmap_images_sha256": identity_images_sha256,
            "training_supervision_split": "train",
            "validation_or_test_labels_used_by_fit": False,
            "registered_identity_radius_px": float(registered_identity_radius_px),
            "diagnostic_only": True,
            "pose_scoring": False,
            "per_view_residual_cap_provenance": (
                str(per_view_residual_cap_provenance).strip()
                if per_view_residual_cap is not None
                else "unbounded_legacy_log_likelihood_ratio_v1"
            ),
        }
        if family_spec.architecture in RAW_TOP4_ARCHITECTURES:
            model_metadata["postfit_scale_provenance"] = (
                str(raw_top4_postfit_scale_provenance).strip()
                if uses_postfit_scale
                else "unit_scale_no_postfit_calibration_v1"
            )
            model_metadata["postfit_cap_provenance"] = (
                str(raw_top4_postfit_cap_provenance).strip()
                if uses_postfit_cap
                else "no_postfit_residual_cap_v1"
            )
        _save_model(
            path=model_path, model=model, normalizer=normalizer, metadata=model_metadata
        )
        models[family] = str(model_path)
        details[family] = model_metadata
        candidate_blocks.append(candidate)
        null_blocks.append(null)
        residual_blocks.append(residual)
        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_candidate, baseline_null = zero_residual_fulltrack_per_view_posterior(features)
    family_evidence_contracts = {
        family: {
            "architecture": str(details[family]["architecture"]),
            "support_view_marginalization": str(
                details[family]["support_view_marginalization"]
            ),
            "missing_evidence_semantics": str(
                details[family]["missing_evidence_semantics"]
            ),
            "zero_residual_reproduces_fixed_posterior": bool(
                details[family]["zero_residual_reproduces_fixed_posterior"]
            ),
            **(
                {
                    "residual_architecture": str(
                        details[family]["residual_architecture"]
                    ),
                    "residual_cap": details[family]["residual_cap"],
                    "residual_cap_semantics": str(
                        details[family]["residual_cap_semantics"]
                    ),
                    "residual_cap_provenance": str(
                        details[family]["per_view_residual_cap_provenance"]
                    ),
                }
                if "residual_architecture" in details[family]
                else {}
            ),
            **(
                {
                    "training_objective": str(
                        details[family]["training_objective"]
                    ),
                    "coarse_top1_stability_weight": float(
                        details[family]["coarse_top1_stability_weight"]
                    ),
                }
                if (
                    FULLTRACK_PER_VIEW_FAMILIES[family].architecture
                    == "sparse_per_view_mixture"
                    and float(details[family].get("coarse_top1_stability_weight", 0.0))
                    > 0.0
                )
                else {}
            ),
            **(
                {
                    "postfit_residual_scale": float(
                        details[family]["postfit_residual_scale"]
                    ),
                    "postfit_scale_provenance": str(
                        details[family]["postfit_scale_provenance"]
                    ),
                    "raw_topk_aggregation": str(
                        details[family]["raw_topk_aggregation"]
                    ),
                    "postfit_residual_cap": details[family][
                        "postfit_residual_cap"
                    ],
                    "postfit_cap_provenance": str(
                        details[family].get(
                            "postfit_cap_provenance",
                            "no_postfit_residual_cap_v1",
                        )
                    ),
                    "candidate_evidence_transform": str(
                        details[family]["candidate_evidence_transform"]
                    ),
                    "postfit_cap_applied_after_train": bool(
                        details[family]["postfit_cap_applied_after_train"]
                    ),
                    "residual_calibration_applied_during_train": bool(
                        details[family][
                            "residual_calibration_applied_during_train"
                        ]
                    ),
                    "training_objective": str(
                        details[family]["training_objective"]
                    ),
                    "coarse_top1_stability_weight": float(
                        details[family]["coarse_top1_stability_weight"]
                    ),
                }
                if "postfit_residual_scale" in details[family]
                else {}
            ),
        }
        for family in family_names
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
        "appearance_artifacts": artifact_entries,
        "fulltrack_compatibility": dict(features.compatibility),
        "identity_supervision_colmap_images_bin": str(identity_images_path),
        "identity_supervision_colmap_images_sha256": identity_images_sha256,
        "families": list(family_names),
        "family_architectures": {
            str(family): FULLTRACK_PER_VIEW_FAMILIES[str(family)].architecture
            for family in family_names
        },
        "family_edge_feature_semantics": {
            str(family): str(
                FULLTRACK_PER_VIEW_FAMILIES[str(family)].edge_feature_semantics
            )
            for family in family_names
        },
        "models": models,
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "support_reselection": False,
        "support_view_selection": False,
        "all_real_sfm_support_observations_retained": True,
        "feature_granularity": fulltrack_per_view_feature_granularity(features),
        "per_view_model": True,
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "missing_evidence_semantics": "family_specific_no_availability_cue_v1",
        "family_evidence_contracts": family_evidence_contracts,
        "stored_baseline_reproduces_input_posterior": True,
        "raw_top4_postfit_residual_scale": float(raw_top4_postfit_residual_scale),
        "per_view_residual_architecture": str(per_view_residual_architecture),
        "per_view_residual_cap": per_view_residual_cap,
        "per_view_residual_cap_provenance": (
            str(per_view_residual_cap_provenance).strip()
            if per_view_residual_cap is not None
            else "unbounded_legacy_log_likelihood_ratio_v1"
        ),
        "raw_top4_postfit_scale_provenance": (
            str(raw_top4_postfit_scale_provenance).strip()
            if uses_postfit_scale
            else "unit_scale_no_postfit_calibration_v1"
        ),
        "raw_top4_postfit_residual_cap": raw_top4_postfit_residual_cap,
        "raw_top4_postfit_cap_provenance": (
            str(raw_top4_postfit_cap_provenance).strip()
            if uses_postfit_cap
            else "no_postfit_residual_cap_v1"
        ),
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
            family_names=np.asarray(family_names, dtype=np.str_),
            candidate_probabilities=np.stack(candidate_blocks, axis=0).astype(np.float32),
            null_probabilities=np.stack(null_blocks, axis=0).astype(np.float32),
            candidate_residuals=np.stack(residual_blocks, axis=0).astype(np.float32),
            baseline_candidate_probabilities=baseline_candidate,
            baseline_null_probabilities=baseline_null,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "fit_frozen_fulltrack_per_view_candidate_probe",
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256_short(prediction_path),
        "families": details,
        "train_target_audit": {
            "train_row_count": int(len(train_rows)),
            "supervised_train_row_count": int(np.sum(train_targets.supervised)),
            "retrieved_identity_train_row_count": int(np.sum(retrieved)),
            "explicit_null_or_topl_miss_train_row_count": int(
                np.sum(train_targets.supervised & ~np.any(labels_all, axis=1))
            ),
            "unsupervised_train_row_count": int(np.sum(~train_targets.supervised)),
            "normalizer_uses_all_frozen_train_rows": True,
            "validation_or_test_target_used": False,
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
            "all_real_sfm_support_observations_retained": True,
            "per_view_s2_claimed": False,
            "input_null_probability_exactly_preserved": True,
            "per_view_residual_cap_is_train_only_predeclared": (
                per_view_residual_cap is None
                or bool(str(per_view_residual_cap_provenance).strip())
            ),
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_scoring": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = fit_frozen_fulltrack_per_view_candidate_probe(
        appearance_artifacts=_paths(args.appearance_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        families=_families(args.families),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        device=str(args.device),
        expected_identity_colmap_images_sha256=str(
            args.expected_identity_colmap_images_sha256
        ),
        rank2_hard_pair_weight=args.rank2_hard_pair_weight,
        per_view_residual_architecture=str(args.per_view_residual_architecture),
        per_view_residual_cap=args.per_view_residual_cap,
        per_view_residual_cap_provenance=str(args.per_view_residual_cap_provenance),
        raw_top4_postfit_residual_scale=float(args.raw_top4_postfit_residual_scale),
        raw_top4_postfit_scale_provenance=str(args.raw_top4_postfit_scale_provenance),
        raw_top4_postfit_residual_cap=args.raw_top4_postfit_residual_cap,
        raw_top4_postfit_cap_provenance=str(args.raw_top4_postfit_cap_provenance),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
