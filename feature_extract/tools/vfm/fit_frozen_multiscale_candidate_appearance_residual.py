"""Fit fixed-prior appearance residual probes using train identities only.

This command is intentionally a small conditional-information experiment, not
a pose optimizer.  It receives target-free frozen per-view C-RADIO/ALIKE
appearance artifacts for train and validation queries, joins registered SfM
track identities for *train rows only*, and writes frozen predictions for both
splits.  The companion audit is the first code allowed to read validation
identities.
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
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    FROZEN_APPEARANCE_RESIDUAL_FAMILIES,
    FROZEN_APPEARANCE_RESIDUAL_MODEL_FORMAT,
    FROZEN_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
    fit_fixedprior_linear_residual,
    load_frozen_appearance_probe_features,
    predict_fixedprior_linear_residual,
    zero_residual_posterior,
)
from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_target_membership,
    registered_query_observation_targets,
)


FROZEN_LOFTR_FULL_QUERY_COUNTS = {"train": 63, "validation": 21}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--appearance_artifacts",
        required=True,
        help="comma-separated complete train and validation frozen appearance shards",
    )
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--families", default="all")
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frozen-loftr-anchor-manifest-audit")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("appearance artifacts must be non-empty and unique")
    return paths


def _resolve_families(value: str) -> tuple[str, ...]:
    families = (
        (
            "fixedprior_alike_local",
            "fixedprior_radio_context",
            "fixedprior_multiscale",
        )
        if str(value).strip() == "all"
        else tuple(item.strip() for item in str(value).split(",") if item.strip())
    )
    if not families or len(set(families)) != len(families):
        raise ValueError("residual probe families must be non-empty and unique")
    unsupported = set(families).difference(FROZEN_APPEARANCE_RESIDUAL_FAMILIES)
    if unsupported:
        raise ValueError(f"unsupported residual probe families: {sorted(unsupported)}")
    return families


def _family_seed(seed: int, family: str) -> int:
    digest = hashlib.sha256(str(family).encode()).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def _validate_frozen_loftr_anchor_manifest_audit(
    *, features: Any, path: Path
) -> dict[str, Any]:
    """Require a complete fail-closed manifest before fitting LoFTR evidence."""

    audit_path = Path(path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"LoFTR anchor manifest audit is absent: {audit_path}")
    payload = json.loads(audit_path.read_text())
    protocol = payload.get("protocol") if isinstance(payload, Mapping) else None
    expected_protocol = {
        "target_free": True,
        "full_mapping_pair_cache": True,
        "fixed_global_top_l": 20,
        "image_level_selection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_identity_fixed": True,
        "source_image_manifest_revalidated": True,
    }
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "frozen_loftr_anchor_manifest_audit_v1"
        or not isinstance(protocol, Mapping)
        or any(protocol.get(key) != value for key, value in expected_protocol.items())
    ):
        raise ValueError("LoFTR anchor manifest audit does not satisfy the frozen protocol")
    expected_paths = {str(Path(item).resolve()): file_sha256_short(Path(item)) for item in features.paths}
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("LoFTR anchor manifest audit records are absent")
    audited_paths: dict[str, str] = {}
    audited_queries: set[tuple[str, str]] = set()
    for record in records:
        anchor = record.get("anchor") if isinstance(record, Mapping) else None
        if not isinstance(anchor, Mapping):
            raise ValueError("LoFTR anchor manifest audit record lacks anchor provenance")
        artifact_path = Path(str(anchor.get("path", ""))).resolve(strict=True)
        artifact_sha = str(anchor.get("sha256", ""))
        key = str(artifact_path)
        if not artifact_sha or key in audited_paths:
            raise ValueError("LoFTR anchor manifest audit repeats an anchor artifact")
        audited_paths[key] = artifact_sha
        audited_queries.add((str(record.get("query_id", "")), str(record.get("split_name", ""))))
    expected_queries = set(zip(features.query_ids.tolist(), features.split_names.tolist()))
    expected_counts = {
        "train": int(len(set(features.query_ids[features.split_names == "train"].tolist()))),
        "validation": int(
            len(set(features.query_ids[features.split_names == "validation"].tolist()))
        ),
    }
    if (
        audited_paths != expected_paths
        or audited_queries != expected_queries
        or payload.get("actual_query_counts") != expected_counts
        or payload.get("expected_query_counts") != FROZEN_LOFTR_FULL_QUERY_COUNTS
        or expected_counts != FROZEN_LOFTR_FULL_QUERY_COUNTS
    ):
        raise ValueError("LoFTR anchor manifest audit does not match the requested fit artifacts")
    return {
        "path": str(audit_path),
        "sha256": file_sha256_short(audit_path),
        "actual_query_counts": dict(expected_counts),
        "mapping_support_image_count": int(payload.get("mapping_support_image_count", -1)),
        "implementation": dict(payload.get("implementation", {})),
    }


def _save_model(
    *,
    path: Path,
    model: torch.nn.Module,
    normalizer: Any,
    metadata: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "format": FROZEN_APPEARANCE_RESIDUAL_MODEL_FORMAT,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "normalizer": {
                "mean": np.asarray(normalizer.mean, dtype=np.float32),
                "scale": np.asarray(normalizer.scale, dtype=np.float32),
                "feature_indices": np.asarray(normalizer.feature_indices, dtype=np.int64),
            },
            "metadata": dict(metadata),
        },
        path,
    )


def fit_frozen_multiscale_candidate_appearance_residual(
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
    seed: int,
    device: str,
    frozen_loftr_anchor_manifest_audit: Path | None = None,
) -> dict[str, Any]:
    """Fit predeclared linear residuals without loading validation targets."""

    if (
        float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
    ):
        raise ValueError("appearance residual fit arguments are invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    feature_paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_appearance_probe_features(feature_paths)
    is_loftr = (
        features.metadata.get("format") == FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT
    )
    if is_loftr and frozen_loftr_anchor_manifest_audit is None:
        raise ValueError("LoFTR residual fit requires a complete frozen anchor manifest audit")
    if not is_loftr and frozen_loftr_anchor_manifest_audit is not None:
        raise ValueError("direct C-RADIO/ALIKE residual fit must not consume a LoFTR manifest audit")
    loftr_manifest = (
        _validate_frozen_loftr_anchor_manifest_audit(
            features=features, path=Path(frozen_loftr_anchor_manifest_audit)
        )
        if is_loftr
        else None
    )
    resolved_families = tuple(str(family) for family in families)
    if set(features.split_names.tolist()) - {"train", "validation"}:
        raise ValueError("appearance residual fit accepts train/validation artifacts only")
    train_mask = features.split_names == "train"
    validation_mask = features.split_names == "validation"
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("appearance residual fit needs both train and validation rows")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    # This is the only target join in the fit command.  Validation/query labels
    # are neither materialized nor used to select an architecture or epoch.
    train_source_rows = np.flatnonzero(train_mask)
    train_targets = registered_query_observation_targets(
        query_ids=features.query_ids[train_source_rows],
        query_xy=features.xy[train_source_rows],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    train_membership_all = registered_candidate_identity_target_membership(
        features.candidate_track_ids[train_source_rows], train_targets
    )
    target_train_rows = train_source_rows[train_targets.supervised]
    target_membership = train_membership_all[train_targets.supervised]
    if len(target_train_rows) == 0:
        raise ValueError("no supervised train appearance anchors are available")

    device_value = torch.device(str(device))
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("appearance residual fit requested unavailable CUDA device")
    if device_value.type == "cuda":
        torch.cuda.set_device(device_value)
    output.mkdir(parents=True, exist_ok=False)
    family_candidates: list[np.ndarray] = []
    family_nulls: list[np.ndarray] = []
    family_residuals: list[np.ndarray] = []
    model_paths: dict[str, str] = {}
    fit_details: dict[str, Any] = {}
    for family in resolved_families:
        model, normalizer, detail = fit_fixedprior_linear_residual(
            features=features,
            family=family,
            train_rows=target_train_rows,
            target_membership=target_membership,
            device=device_value,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=_family_seed(int(seed), family),
        )
        candidate, null, residual = predict_fixedprior_linear_residual(
            model=model,
            features=features,
            normalizer=normalizer,
            device=device_value,
            batch_size=int(batch_size),
        )
        if np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5:
            raise RuntimeError("appearance residual prediction does not conserve posterior mass")
        model_path = output / f"{family}.pt"
        model_metadata = {
            **detail,
            "appearance_artifact_sha256": {
                str(path): file_sha256_short(path) for path in feature_paths
            },
            "training_supervision_split": "train",
            "validation_or_test_labels_used_by_fit": False,
            "registered_identity_radius_px": float(registered_identity_radius_px),
            "frozen_loftr_anchor_manifest_audit": loftr_manifest,
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

    base_candidate, base_null = zero_residual_posterior(features)
    metadata: dict[str, Any] = {
        "format": FROZEN_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "prediction_frozen_before_validation_target_join": True,
        "training_supervision_split": "train",
        "validation_or_test_labels_used_by_fit": False,
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "appearance_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in feature_paths
        ],
        "families": list(resolved_families),
        "models": model_paths,
        "fixed_global_top_l": True,
        "candidate_reselection": False,
        "support_reselection": False,
        "support_view_marginalization": "fixed_maplet_view_weight_logsumexp_v1",
        "null_handling": "immutable_input_null_log_prior_v1",
        "zero_residual_reproduces_input_posterior": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "frozen_loftr_anchor_manifest_audit": loftr_manifest,
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
            per_view_residuals=np.stack(family_residuals, axis=0).astype(np.float32),
            baseline_candidate_probabilities=base_candidate,
            baseline_null_probabilities=base_null,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "fit_frozen_multiscale_candidate_appearance_fixedprior_residual",
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256_short(prediction_path),
        "families": fit_details,
        "train_target_audit": {
            "train_row_count": int(np.sum(train_mask)),
            "supervised_train_row_count": int(np.sum(train_targets.supervised)),
            "unsupervised_train_row_count": int(np.sum(~train_targets.supervised)),
            "candidate_identity_train_row_count": int(
                np.sum(np.any(train_membership_all[:, :-1], axis=1))
            ),
            "explicit_null_train_row_count": int(
                np.sum(train_membership_all[:, -1])
            ),
            "validation_or_test_target_used": False,
        },
        "baseline_equivalence": {
            "candidate_max_abs_error": float(
                np.max(np.abs(base_candidate - features.candidate_probabilities))
            ),
            "null_max_abs_error": float(
                np.max(np.abs(base_null - features.null_probabilities))
            ),
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
            "full_loftr_manifest_audit_required": bool(is_loftr),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = fit_frozen_multiscale_candidate_appearance_residual(
        appearance_artifacts=_paths(args.appearance_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        families=_resolve_families(args.families),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        frozen_loftr_anchor_manifest_audit=(
            None
            if args.frozen_loftr_anchor_manifest_audit is None
            else Path(args.frozen_loftr_anchor_manifest_audit)
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
