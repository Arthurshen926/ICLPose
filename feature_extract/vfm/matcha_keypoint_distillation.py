"""MATCHA-style 65-bin keypoint distillation targets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from feature_extract.vfm.matcha_coarse_supervision import cell_offset_labels


NON_KEYPOINT_LABEL = 64


def build_keypoint_label_map(
    keypoints_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    offset_bins: int = 8,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a MATCHA keypoint label map.

    Labels 0..63 encode the keypoint sub-cell offset. Label 64 means
    non-keypoint, matching MATCHA's ALIKE distillation convention.
    """

    grid_w, grid_h = int(grid_width), int(grid_height)
    labels = np.full((grid_h, grid_w), NON_KEYPOINT_LABEL, dtype=np.int64)
    keypoints = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    if keypoints.shape[0] == 0:
        return labels, {
            "keypoint_count": 0,
            "positive_count": 0,
            "valid_keypoint_count": 0,
            "collision_count": 0,
        }
    score_values = (
        np.ones((keypoints.shape[0],), dtype=np.float64)
        if scores is None
        else np.asarray(scores, dtype=np.float64).reshape(-1)
    )
    if score_values.shape[0] != keypoints.shape[0]:
        raise ValueError("scores must have one value per keypoint")
    finite_in_frame = (
        np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 0] >= 0.0)
        & (keypoints[:, 0] < float(image_width))
        & (keypoints[:, 1] >= 0.0)
        & (keypoints[:, 1] < float(image_height))
    )
    compact_keypoints = keypoints[finite_in_frame]
    compact_scores = score_values[finite_in_frame]
    if compact_keypoints.shape[0] == 0:
        return labels, {
            "keypoint_count": int(keypoints.shape[0]),
            "positive_count": 0,
            "valid_keypoint_count": 0,
            "collision_count": 0,
        }
    offset_labels, valid = cell_offset_labels(
        compact_keypoints,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_width=grid_w,
        grid_height=grid_h,
        offset_bins=int(offset_bins),
    )
    cell_w = float(image_width) / float(grid_w)
    cell_h = float(image_height) / float(grid_h)
    cols = np.floor(compact_keypoints[:, 0] / cell_w).astype(np.int64)
    rows = np.floor(compact_keypoints[:, 1] / cell_h).astype(np.int64)
    best_score = np.full((grid_h, grid_w), -np.inf, dtype=np.float64)
    collisions = 0
    positives = 0
    for idx in np.flatnonzero(valid).tolist():
        row, col = int(rows[idx]), int(cols[idx])
        if labels[row, col] != NON_KEYPOINT_LABEL:
            collisions += 1
        score = float(compact_scores[idx])
        if not np.isfinite(score):
            score = -np.inf
        if score >= float(best_score[row, col]):
            labels[row, col] = int(offset_labels[idx])
            best_score[row, col] = score
        positives += 1
    return labels, {
        "keypoint_count": int(keypoints.shape[0]),
        "positive_count": int(np.count_nonzero(labels != NON_KEYPOINT_LABEL)),
        "valid_keypoint_count": int(positives),
        "collision_count": int(collisions),
    }


def extract_alike_keypoints(
    rgb: np.ndarray,
    *,
    matcha_repo: str = "/root/matcha",
    model_name: str = "alike-t",
    top_k: int = 4096,
    scores_th: float = 0.1,
    n_limit: int = 8000,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the local MATCHA/ALIKE implementation and return keypoints/scores."""

    repo = Path(matcha_repo)
    if not repo.exists():
        raise FileNotFoundError(f"MATCHA repo not found: {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from third_party.alike.alike import ALike, configs  # type: ignore

    if str(model_name) not in configs:
        raise ValueError(f"unknown ALIKE model '{model_name}'")
    cfg: dict[str, Any] = dict(configs[str(model_name)])
    cfg.update(device=str(device), top_k=int(top_k), scores_th=float(scores_th), n_limit=int(n_limit))
    model = ALike(**cfg)
    image = np.asarray(rgb, dtype=np.uint8)
    pred = model(image, sub_pixel=True, sort=True)
    keypoints = np.asarray(pred["keypoints"], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(pred.get("scores", np.ones((keypoints.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
    return keypoints, scores


class AlikeKeypointExtractor:
    """Reusable local ALIKE keypoint extractor."""

    def __init__(
        self,
        *,
        matcha_repo: str = "/root/matcha",
        model_name: str = "alike-t",
        top_k: int = 4096,
        scores_th: float = 0.1,
        n_limit: int = 8000,
        device: str = "cuda",
    ) -> None:
        repo = Path(matcha_repo)
        if not repo.exists():
            raise FileNotFoundError(f"MATCHA repo not found: {repo}")
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from third_party.alike.alike import ALike, configs  # type: ignore

        if str(model_name) not in configs:
            raise ValueError(f"unknown ALIKE model '{model_name}'")
        cfg: dict[str, Any] = dict(configs[str(model_name)])
        cfg.update(device=str(device), top_k=int(top_k), scores_th=float(scores_th), n_limit=int(n_limit))
        self.model = ALike(**cfg)

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(rgb, dtype=np.uint8)
        pred = self.model(image, sub_pixel=True, sort=True)
        keypoints = np.asarray(pred["keypoints"], dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(pred.get("scores", np.ones((keypoints.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        return keypoints, scores
