from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.local_likelihood import LocalLikelihoodResult


def decodability_metrics(result: LocalLikelihoodResult, *, gt_xy_px: np.ndarray) -> dict[str, float | int]:
    """Summarize whether a local likelihood decodes the GT query pixel."""

    gt = np.asarray(gt_xy_px, dtype=np.float64).reshape(2)
    xy = np.asarray(result.xy_px, dtype=np.float64).reshape(-1, 2)
    log_probs = np.asarray(result.local_log_probs, dtype=np.float64).reshape(-1)
    if xy.shape[0] != log_probs.shape[0]:
        raise ValueError("xy_px and local_log_probs must contain the same number of samples")
    distances = np.linalg.norm(xy - gt.reshape(1, 2), axis=1)
    gt_idx = int(np.argmin(distances))
    order = np.argsort(-log_probs)
    rank = int(np.where(order == gt_idx)[0][0]) + 1
    probabilities = np.exp(log_probs)
    probabilities = probabilities / max(float(np.sum(probabilities)), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))))
    epe = float(np.linalg.norm(np.asarray(result.mean_xy_px, dtype=np.float64).reshape(2) - gt))
    mode_error = float(np.linalg.norm(np.asarray(result.mode_xy_px, dtype=np.float64).reshape(2) - gt))
    top_count = min(5, order.shape[0])
    top5_distances = distances[order[:top_count]] if top_count else np.asarray([], dtype=np.float64)
    sorted_probs = np.sort(probabilities)[::-1]
    peak_second_margin = float(sorted_probs[0] - sorted_probs[1]) if sorted_probs.shape[0] >= 2 else float(sorted_probs[0])
    return {
        "gt_rank": rank,
        "epe_px": epe,
        "nll": float(-log_probs[gt_idx]),
        "entropy": entropy,
        "recall_0p5px": float(epe <= 0.5),
        "recall_1px": float(epe <= 1.0),
        "recall_2px": float(epe <= 2.0),
        "recall_5px": float(epe <= 5.0),
        "mode_error_px": mode_error,
        "mode_recall_0p5px": float(mode_error <= 0.5),
        "mode_recall_1px": float(mode_error <= 1.0),
        "mode_recall_2px": float(mode_error <= 2.0),
        "mode_recall_5px": float(mode_error <= 5.0),
        "top5_mode_recall_0p5px": float(bool(top5_distances.size) and np.min(top5_distances) <= 0.5),
        "top5_mode_recall_1px": float(bool(top5_distances.size) and np.min(top5_distances) <= 1.0),
        "top5_mode_recall_2px": float(bool(top5_distances.size) and np.min(top5_distances) <= 2.0),
        "top5_mode_recall_5px": float(bool(top5_distances.size) and np.min(top5_distances) <= 5.0),
        "peak_second_margin": peak_second_margin,
    }
