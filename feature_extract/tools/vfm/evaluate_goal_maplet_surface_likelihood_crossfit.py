"""Evaluate typed pose likelihood under strict canonical-feature cross-fit.

This audit does not claim that the fixed physical 2DGS geometry was rebuilt in
each outer fold. It therefore closes feature/calibrator leakage only, not the
stronger end-to-end map/query separation required for a paper test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_goal_maplet_surface_pose_likelihood import (
    _lineage,
    _load,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    FEATURE_NAMES,
    load_surface_pose_likelihood,
)


def _threshold_oracle_choice(
    translation: np.ndarray,
    rotation: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Choose a pool oracle consistent with the reported success thresholds."""

    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if translation.shape != rotation.shape or translation.shape != valid.shape:
        raise ValueError("oracle pose arrays differ")
    if not np.any(valid):
        raise ValueError("oracle requires at least one valid candidate")
    quality = translation / 0.5 + rotation / 5.0
    quality[~valid] = np.inf
    strict = valid & (translation <= 0.5) & (rotation <= 5.0)
    loose = valid & (translation <= 1.0) & (rotation <= 10.0)
    subset = strict if np.any(strict) else loose if np.any(loose) else valid
    return int(np.argmin(np.where(subset, quality, np.inf)))


def _pose_metrics(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    accepted = np.asarray([bool(row[key]["accepted"]) for row in rows], dtype=bool)
    translation = np.asarray([float(row[key]["translation_m"]) for row in rows])
    rotation = np.asarray([float(row[key]["rotation_deg"]) for row in rows])
    strict = accepted & (translation <= 0.5) & (rotation <= 5.0)
    loose = accepted & (translation <= 1.0) & (rotation <= 10.0)
    catastrophic = accepted & ((translation > 5.0) | (rotation > 30.0))
    finite_t = translation[accepted]
    finite_r = rotation[accepted]
    return {
        "query_count": len(rows),
        "accepted_count": int(np.sum(accepted)),
        "acceptance_rate": float(np.mean(accepted)),
        "strict_0.5m_5deg": float(np.mean(strict)),
        "success_1m_10deg": float(np.mean(loose)),
        "strict_precision_given_accept": (
            float(np.mean(strict[accepted])) if np.any(accepted) else None
        ),
        "success_precision_given_accept": (
            float(np.mean(loose[accepted])) if np.any(accepted) else None
        ),
        "catastrophic_rate": float(np.mean(catastrophic)),
        "catastrophic_or_abstain_rate": float(np.mean(catastrophic | ~accepted)),
        "translation_m": {
            "median_accepted": float(np.median(finite_t)) if finite_t.size else None,
            "p90_accepted": float(np.percentile(finite_t, 90.0)) if finite_t.size else None,
        },
        "rotation_deg": {
            "median_accepted": float(np.median(finite_r)) if finite_r.size else None,
            "p90_accepted": float(np.percentile(finite_r, 90.0)) if finite_r.size else None,
        },
    }


def _risk_coverage(rows: list[dict[str, object]]) -> dict[str, object]:
    order = sorted(
        range(len(rows)),
        key=lambda index: -float(rows[index]["typed_likelihood"]["confidence"]),
    )
    accepted_order = [
        index for index in order if bool(rows[index]["typed_likelihood"]["accepted"])
    ]
    result = {}
    for coverage in (1.0, 0.9, 0.8, 0.5):
        requested = max(1, int(np.ceil(coverage * len(rows))))
        selected = accepted_order[:requested]
        strict = [
            float(rows[index]["typed_likelihood"]["translation_m"]) <= 0.5
            and float(rows[index]["typed_likelihood"]["rotation_deg"]) <= 5.0
            for index in selected
        ]
        catastrophic = [
            float(rows[index]["typed_likelihood"]["translation_m"]) > 5.0
            or float(rows[index]["typed_likelihood"]["rotation_deg"]) > 30.0
            for index in selected
        ]
        result[f"coverage_{coverage:.1f}"] = {
            "requested_count": requested,
            "accepted_count": len(selected),
            "realized_coverage": float(len(selected) / max(len(rows), 1)),
            "strict_precision": float(np.mean(strict)) if strict else None,
            "strict_success_yield": float(np.sum(strict) / max(len(rows), 1)),
            "catastrophic_pose_rate": (
                float(np.mean(catastrophic)) if catastrophic else None
            ),
        }
    return result


def _fold_predictions(
    model_path: Path,
    sample_path: Path,
    *,
    target: str,
    canonical_feature_trajectories: set[str],
    device: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model, metadata = load_surface_pose_likelihood(model_path, device=device)
    data = _load([str(sample_path)])
    lineage = _lineage(data)
    sample_metadata = data["_metadata"][0][1]
    for key, expected in lineage.items():
        if metadata.get(key) != expected:
            raise ValueError(f"model/test sample lineage differs: {key}")
    if metadata.get("checkpoint_protocol") != "fixed_epoch_no_selection":
        raise ValueError("outer-fold likelihood must use a fixed-epoch checkpoint")
    if metadata.get("checkpoint_selection_uses_pose_labels") is not False:
        raise ValueError("outer-fold checkpoint selection consulted pose labels")
    if metadata.get("selection_trajectory_ids"):
        raise ValueError("fixed-epoch outer-fold model unexpectedly has a selection set")
    training_trajectories = set(str(value) for value in metadata.get(
        "training_trajectory_ids", (),
    ))
    target_trajectories = set(
        np.asarray(data["trajectory_ids"]).astype(str).tolist()
    )
    if target_trajectories != {str(target)}:
        raise ValueError("fold target and sample trajectories differ")
    if training_trajectories & target_trajectories:
        raise ValueError("outer-fold model was trained on its target trajectory")
    if (training_trajectories | target_trajectories) & canonical_feature_trajectories:
        raise ValueError("likelihood train/test query appears in the canonical map")
    if sample_metadata.get("candidate_set_frozen_before_scoring") is not True:
        raise ValueError("surface-likelihood candidates were not frozen before scoring")
    if sample_metadata.get("fixed_full_query_denominator") is not True:
        raise ValueError("surface-likelihood sample changed its token denominator")

    translation = np.asarray(data["translation_m"], dtype=np.float64)
    rotation = np.asarray(data["rotation_deg"], dtype=np.float64)
    valid = np.asarray(data["candidate_valid"], dtype=bool)
    token = np.asarray(data["token_feature"], dtype=np.float32)
    summary = np.asarray(data["query_summary"], dtype=np.float32)
    image_ids = np.asarray(data["image_ids"]).astype(str)
    cosine = token[..., FEATURE_NAMES.index("cosine")]
    render_valid = token[..., FEATURE_NAMES.index("render_valid")]
    cosine_score = np.mean(cosine * render_valid, axis=2)
    cosine_score[~valid] = -np.inf
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for index in range(translation.shape[0]):
            probability, null, _event = model.posterior(
                torch.from_numpy(token[index:index + 1]).to(device),
                torch.from_numpy(summary[index:index + 1]).to(device),
                torch.from_numpy(valid[index:index + 1]).to(device),
            )
            candidate = probability[0].cpu().numpy().astype(np.float64)
            null_probability = float(null[0].cpu())
            distribution = np.r_[candidate, null_probability]
            choice = int(np.argmax(distribution))
            accepted = choice < candidate.size
            safe_choice = min(choice, candidate.size - 1)
            entropy = float(-np.sum(
                distribution * np.log(np.maximum(distribution, 1.0e-12))
            ))
            normalized_entropy = entropy / max(float(np.log(distribution.size)), 1.0e-12)
            confidence = float(np.max(candidate) * (1.0 - normalized_entropy))
            cosine_choice = int(np.argmax(cosine_score[index]))
            # Threshold recall asks whether the frozen pool contains a
            # successful basin. A minimum weighted pose-error candidate can
            # lie just outside 0.5 m even when a strict-success candidate is
            # present, so it is not a valid oracle for success-rate metrics.
            oracle_choice = _threshold_oracle_choice(
                translation[index], rotation[index], valid[index],
            )

            def pose(candidate_index: int, is_accepted: bool = True) -> dict[str, object]:
                return {
                    "candidate_index": candidate_index,
                    "accepted": bool(is_accepted),
                    # Keep the best candidate error for diagnostics even when
                    # the typed null wins; ``accepted`` alone defines yield.
                    "translation_m": float(translation[index, candidate_index]),
                    "rotation_deg": float(rotation[index, candidate_index]),
                }

            rows.append({
                "image_id": str(image_ids[index]),
                "target_trajectory": str(target),
                "typed_likelihood": {
                    **pose(safe_choice, accepted),
                    "null_probability": null_probability,
                    "candidate_probability": float(candidate[safe_choice]),
                    "posterior_entropy": entropy,
                    "confidence": confidence,
                },
                "frozen_exact_top1": pose(0),
                "fixed_grid_cosine": pose(cosine_choice),
                "oracle_exact128": pose(oracle_choice),
            })
    contract = {
        "target_trajectory": str(target),
        "training_trajectory_ids": sorted(training_trajectories),
        "canonical_feature_trajectory_ids": sorted(canonical_feature_trajectories),
        "target_is_canonical_feature_disjoint": True,
        "training_queries_are_canonical_feature_disjoint": True,
        "physical_geometry_outer_crossfit": False,
        "target_is_model_disjoint": True,
        "fixed_epoch_no_selection": True,
        "model": str(model_path.resolve()),
        "model_sha256": file_sha256(model_path),
        "samples": str(sample_path.resolve()),
        "samples_sha256": file_sha256(sample_path),
        "checkpoint_epoch": int(metadata["checkpoint_epoch"]),
    }
    return rows, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical_field_summary", required=True)
    parser.add_argument(
        "--fold",
        nargs=3,
        action="append",
        metavar=("TARGET", "MODEL", "SAMPLES"),
        required=True,
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite cross-fit likelihood report")
    field_summary_path = Path(args.canonical_field_summary)
    field_summary = json.loads(field_summary_path.read_text())
    canonical_feature_trajectories = set(str(value) for value in field_summary.get(
        "mapping_trajectory_ids", (),
    ))
    if not canonical_feature_trajectories:
        raise ValueError("canonical field summary has no mapping trajectories")
    rows: list[dict[str, object]] = []
    contracts = []
    seen_targets: set[str] = set()
    for target, model, samples in args.fold:
        if str(target) in seen_targets:
            raise ValueError("duplicate outer-fold target")
        seen_targets.add(str(target))
        fold_rows, contract = _fold_predictions(
            Path(model), Path(samples), target=str(target),
            canonical_feature_trajectories=canonical_feature_trajectories,
            device=str(args.device),
        )
        rows.extend(fold_rows)
        contracts.append(contract)
    metrics = {
        key: _pose_metrics(rows, key)
        for key in (
            "typed_likelihood", "frozen_exact_top1", "fixed_grid_cosine",
            "oracle_exact128",
        )
    }
    metrics["typed_likelihood"]["risk_coverage"] = _risk_coverage(rows)
    typed = metrics["typed_likelihood"]
    baseline = metrics["frozen_exact_top1"]
    report = {
        "stage": (
            "goal_maplet_typed_surface_likelihood_outer_feature_crossfit_v1"
        ),
        "query_count": len(rows),
        "fold_count": len(contracts),
        "metrics": metrics,
        "paired_delta_vs_frozen_exact_top1": {
            "strict_0.5m_5deg": float(
                typed["strict_0.5m_5deg"] - baseline["strict_0.5m_5deg"]
            ),
            "success_1m_10deg": float(
                typed["success_1m_10deg"] - baseline["success_1m_10deg"]
            ),
            "catastrophic_or_abstain_rate": float(
                typed["catastrophic_or_abstain_rate"]
                - baseline["catastrophic_or_abstain_rate"]
            ),
        },
        "crossfit_contract": {
            "canonical_field_summary": str(field_summary_path.resolve()),
            "canonical_field_summary_sha256": file_sha256(field_summary_path),
            "canonical_feature_trajectory_ids": sorted(
                canonical_feature_trajectories
            ),
            "candidate_sets_frozen_before_scoring": True,
            "target_pose_labels_never_select_checkpoint": True,
            "physical_geometry_outer_crossfit": False,
            "post_selection_candidate_reranker": True,
            "calibrated_success_probability": False,
            "confidence_semantics": (
                "candidate_posterior_concentration_proxy_not_P(success_tau)"
            ),
            "paper_or_deployment_promotion_allowed": False,
            "folds": contracts,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
