"""Probe soft candidate support-view context without retrieval or submaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz


ARTIFACT_FORMAT = "candidate_image_context_prior_overlay_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--base_prior_overlay", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(target))
    if positives == 0:
        return 0.0
    order = np.argsort(-values, kind="mergesort")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _identity_metrics(
    probabilities: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    *,
    positive_threshold_px: float,
) -> dict[str, float]:
    scores = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(residuals, dtype=np.float64) <= float(
        positive_threshold_px
    )
    mask = np.asarray(valid, dtype=bool)
    ranked_scores = np.where(mask, scores, -np.inf)
    top = np.argmax(ranked_scores, axis=1)
    top_correct = target[np.arange(len(target)), top]
    mappable = np.any(target & mask, axis=1)
    ranks = []
    for row in np.flatnonzero(mappable):
        order = np.argsort(-ranked_scores[row], kind="mergesort")
        ranks.append(int(np.flatnonzero(target[row, order])[0] + 1))
    return {
        "pair_average_precision": _average_precision(target[mask], scores[mask]),
        "top1_positive_rate_all": float(np.mean(top_correct)),
        "top1_positive_rate_mappable": (
            float(np.mean(top_correct[mappable])) if np.any(mappable) else 0.0
        ),
        "median_first_positive_rank_mappable": (
            float(np.median(ranks)) if ranks else float("nan")
        ),
        "mappable_rate": float(np.mean(mappable)),
    }


def _spatial_chamfer_grid2(
    query: np.ndarray, reference: np.ndarray, device: str
) -> np.ndarray:
    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    r = torch.as_tensor(reference, dtype=torch.float32, device=device)
    output = np.empty((len(q), len(r)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(r), 128):
            block = r[start : start + 128]
            similarity = torch.einsum("qad,rbd->qrab", q, block)
            score = 0.5 * (
                similarity.amax(dim=3).mean(dim=2)
                + similarity.amax(dim=2).mean(dim=2)
            )
            output[:, start : start + len(block)] = score.cpu().numpy()
    return output


def _posterior(
    base: np.ndarray,
    null: np.ndarray,
    context: np.ndarray,
    valid: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_logits = np.log(np.maximum(base, 1e-12)) + float(alpha) * context
    candidate_logits[~valid] = -np.inf
    null_logits = np.log(np.maximum(null, 1e-12))
    maximum = np.maximum(np.max(candidate_logits, axis=1), null_logits)
    candidate_mass = np.exp(candidate_logits - maximum[:, None])
    candidate_mass[~valid] = 0.0
    null_mass = np.exp(null_logits - maximum)
    denominator = np.sum(candidate_mass, axis=1) + null_mass
    return (
        (candidate_mass / denominator[:, None]).astype(np.float32),
        (null_mass / denominator).astype(np.float32),
    )


def _support_view_score(values: np.ndarray, aggregation: str) -> float:
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return -1.0
    if aggregation == "top1":
        return float(np.max(scores))
    if aggregation == "top2_mean":
        count = min(2, len(scores))
        return float(np.mean(np.partition(scores, len(scores) - count)[-count:]))
    raise ValueError(f"unsupported support-view aggregation: {aggregation}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "candidate_image_context_prior_overlay_v1.npz"
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    proposal_path = Path(args.proposals)
    with np.load(proposal_path, allow_pickle=False) as data:
        proposals = {
            key: np.asarray(data[key]).copy()
            for key in (
                "query_ids",
                "candidate_track_ids",
                "candidate_gt_residuals_px",
            )
        }
    base, base_metadata = _load_candidate_prior_overlay(
        Path(args.base_prior_overlay),
        proposals_path=proposal_path,
        proposals=proposals,
    )
    with np.load(Path(args.candidate_artifact), allow_pickle=False) as data:
        selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
    with np.load(Path(args.context_cache), allow_pickle=False) as data:
        image_ids = data["image_ids"].astype(str)
        summary = np.asarray(data["summary_descriptors"], dtype=np.float32)
        global_local = np.asarray(
            data["global_local_descriptors"], dtype=np.float32
        )
        grid2 = np.asarray(data["grid2_descriptors"], dtype=np.float32)
        context_metadata = json.loads(str(data["metadata_json"].item()))
    if context_metadata.get("pose_or_ground_truth_used") is not False:
        raise ValueError("image context cache is not pose-free")
    query_image_ids = np.unique(proposals["query_ids"].astype(str))
    image_position = {str(value): index for index, value in enumerate(image_ids)}
    query_positions = np.asarray(
        [image_position[str(value)] for value in query_image_ids], dtype=np.int64
    )
    pair_scores = {
        "summary": summary[query_positions] @ summary.T,
        "global_local": global_local[query_positions] @ global_local.T,
        "grid2_chamfer": _spatial_chamfer_grid2(
            grid2[query_positions], grid2, str(args.device)
        ),
    }
    pair_scores["summary_global_mean"] = 0.5 * (
        pair_scores["summary"] + pair_scores["global_local"]
    )
    query_lookup = {value: index for index, value in enumerate(query_image_ids)}
    bank, bank_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    # Candidate scoring needs original image names while pair-score rows use the
    # compact query ordering. Build it directly to keep that distinction explicit.
    context_scores = {}
    support_position = {str(value): index for index, value in enumerate(image_ids)}
    track_position = {int(value): index for index, value in enumerate(bank.track_ids)}
    for name, values in pair_scores.items():
        for aggregation in ("top1", "top2_mean"):
            scores = np.full(
                proposals["candidate_track_ids"].shape, -1.0, dtype=np.float32
            )
            cache: dict[tuple[int, int], float] = {}
            for row, query_id in enumerate(proposals["query_ids"].astype(str)):
                query_row = query_lookup[str(query_id)]
                for column, track_id in enumerate(
                    proposals["candidate_track_ids"][row]
                ):
                    if int(track_id) < 0:
                        continue
                    key = (query_row, int(track_id))
                    score = cache.get(key)
                    if score is None:
                        bank_row = track_position[int(track_id)]
                        support_rows = [
                            support_position[str(image_id)]
                            for image_id in bank.observation_image_ids[bank_row]
                            if str(image_id) != str(query_id)
                            and str(image_id) in support_position
                        ]
                        score = _support_view_score(
                            values[query_row, support_rows], aggregation
                        )
                        cache[key] = score
                    scores[row, column] = score
            context_scores[f"{name}_{aggregation}"] = scores

    split = json.loads(Path(args.split_json).read_text())
    selected_query_ids = proposals["query_ids"][selected_rows].astype(str)
    residuals = proposals["candidate_gt_residuals_px"][selected_rows]
    tracks = proposals["candidate_track_ids"]
    valid = tracks >= 0
    selected_valid = valid[selected_rows]
    alphas = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    sweeps = []
    for family, scores in context_scores.items():
        for alpha in alphas:
            probabilities, null = _posterior(
                base["candidate_probabilities"],
                base["null_probabilities"],
                scores,
                valid,
                alpha,
            )
            metrics = {
                role: _identity_metrics(
                    probabilities[selected_rows][
                        np.isin(selected_query_ids, split[role])
                    ],
                    residuals[np.isin(selected_query_ids, split[role])],
                    selected_valid[np.isin(selected_query_ids, split[role])],
                    positive_threshold_px=float(args.positive_threshold_px),
                )
                for role in ("train", "validation", "test")
            }
            sweeps.append(
                {"family": family, "alpha": alpha, "metrics": metrics}
            )
    selected = max(
        sweeps,
        key=lambda row: (
            row["metrics"]["train"]["pair_average_precision"],
            row["metrics"]["train"]["top1_positive_rate_mappable"],
            -float(row["alpha"]),
        ),
    )
    final_probabilities, final_null = _posterior(
        base["candidate_probabilities"],
        base["null_probabilities"],
        context_scores[str(selected["family"])],
        valid,
        float(selected["alpha"]),
    )
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "calibration_ground_truth_used": True,
        "calibration_split": "train",
        "validation_or_test_used_for_selection": False,
        "positive_threshold_px": float(args.positive_threshold_px),
        "probability_semantics": (
            "candidate_identity_probability_plus_explicit_null_equals_one"
        ),
        "image_retrieval": False,
        "submap": False,
        "candidate_denominator_changed": False,
        "selection_split": "train",
        "split_manifest_sha256": file_sha256_short(Path(args.split_json)),
        "selected_family": selected["family"],
        "selected_alpha": selected["alpha"],
        "support_view_posterior_pose_independent": True,
        "proposals_sha256": file_sha256_short(proposal_path),
        "base_prior_overlay_sha256": file_sha256_short(
            Path(args.base_prior_overlay)
        ),
        "context_cache_sha256": file_sha256_short(Path(args.context_cache)),
        "projected_landmark_bank_sha256": file_sha256_short(
            Path(args.projected_landmark_bank)
        ),
        "descriptor_space_id": bank_metadata.get("descriptor_space_id"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidate_track_ids=tracks,
        candidate_probabilities=final_probabilities,
        null_probabilities=final_null,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary_payload = {
        "stage": "radio_image_context_candidate_prior_probe",
        "selected": selected,
        "sweeps": sweeps,
        "metadata": metadata,
        "base_prior_metadata": base_metadata,
        "outputs": {"overlay": str(output_path)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary_payload["selected"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
