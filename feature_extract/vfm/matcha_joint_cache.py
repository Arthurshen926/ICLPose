"""Full-map MATCHA joint-cache construction utilities."""

from __future__ import annotations

import numpy as np

from feature_extract.vfm.matcha_coarse_fine_adapter import _flatten_feature_map
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    MatchaCoarseFineTrainingSet,
    build_matcha_coarse_fine_training_set,
)
from feature_extract.vfm.matcha_coarse_supervision import MatchaCoarseSupervision
from feature_extract.vfm.matcha_joint_training import IndexOnlyCoarseFineRows, MatchaJointTrainingSet
from feature_extract.vfm.query_to_3d_matching import normalize_rows


def heatmap_targets_from_coarse_supervision(
    supervision: MatchaCoarseSupervision,
    *,
    query_grid_hw: tuple[int, int],
    render_grid_hw: tuple[int, int],
    roundtrip_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build MATCHA-style dense reliability targets from coarse correspondences."""

    qh, qw = int(query_grid_hw[0]), int(query_grid_hw[1])
    rh, rw = int(render_grid_hw[0]), int(render_grid_hw[1])
    query = np.zeros((qh, qw), dtype=np.float32)
    render = np.zeros((rh, rw), dtype=np.float32)
    if supervision.count == 0:
        return query, render
    threshold = max(float(roundtrip_threshold_px), 1e-6)
    weights = np.clip(1.0 - np.asarray(supervision.roundtrip_errors_px, dtype=np.float32) / threshold, 0.0, 1.0)
    for qidx, ridx, weight in zip(supervision.query_indices, supervision.render_indices, weights):
        qidx = int(qidx)
        ridx = int(ridx)
        if 0 <= qidx < qh * qw:
            qrow, qcol = divmod(qidx, qw)
            query[qrow, qcol] = max(float(query[qrow, qcol]), float(weight))
        if 0 <= ridx < rh * rw:
            rrow, rcol = divmod(ridx, rw)
            render[rrow, rcol] = max(float(render[rrow, rcol]), float(weight))
    return query, render


def _resize_rgb_to_grid(rgb: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(rgb)
    target_h = int(grid_hw[0]) * 8
    target_w = int(grid_hw[1]) * 8
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to resize RGB images for MATCHA joint cache") from exc
    interpolation = cv2.INTER_AREA if image.shape[0] > target_h or image.shape[1] > target_w else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def _rgb_to_bchw_float(rgb: np.ndarray, *, grid_hw: tuple[int, int]) -> np.ndarray:
    rgb = _resize_rgb_to_grid(rgb, grid_hw)
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if image.dtype == np.uint8:
        value = image.astype(np.float32) / 255.0
    else:
        value = image.astype(np.float32)
        if np.max(value, initial=0.0) > 2.0:
            value = value / 255.0
    height, width = int(value.shape[0]), int(value.shape[1])
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("rgb height and width must be divisible by 8")
    return value.transpose(2, 0, 1)[None].astype(np.float32, copy=False)


def _label_bhw(label_map: np.ndarray | None) -> np.ndarray | None:
    if label_map is None:
        return None
    labels = np.asarray(label_map, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("keypoint label map must have shape (H, W)")
    if labels.size and (np.any(labels < 0) or np.any(labels > 64)):
        raise ValueError("keypoint labels must be in [0, 64]")
    return labels[None].astype(np.int64, copy=False)


def _rgb_to_bchw_float_if_labeled(rgb: np.ndarray | None, label_map: np.ndarray | None, *, grid_hw: tuple[int, int]) -> np.ndarray | None:
    if rgb is None or label_map is None:
        return None
    return _rgb_to_bchw_float(rgb, grid_hw=grid_hw)


def _mine_negative_render_indices(
    query_features: np.ndarray,
    render_features: np.ndarray,
    positive_render_indices: np.ndarray,
    *,
    count: int,
    excluded_render_indices_by_query: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    qnorm, qvalid = normalize_rows(query_features)
    rnorm, rvalid = normalize_rows(render_features)
    positives = np.asarray(positive_render_indices, dtype=np.int64).reshape(-1)
    if positives.shape[0] != query_features.shape[0]:
        raise ValueError("positive_render_indices must contain one index per query feature")
    if excluded_render_indices_by_query is not None and len(excluded_render_indices_by_query) != query_features.shape[0]:
        raise ValueError("excluded_render_indices_by_query must contain one exclusion set per query feature")
    scores = qnorm @ rnorm.T
    scores[~qvalid, :] = -np.inf
    scores[:, ~rvalid] = -np.inf
    output = np.zeros((query_features.shape[0], int(count)), dtype=np.int64)
    for row in range(query_features.shape[0]):
        row_scores = scores[row].copy()
        row_scores[int(positives[row])] = -np.inf
        if excluded_render_indices_by_query is not None:
            excluded = np.asarray(excluded_render_indices_by_query[row], dtype=np.int64).reshape(-1)
            excluded = excluded[(excluded >= 0) & (excluded < row_scores.shape[0])]
            row_scores[excluded] = -np.inf
        order = np.argsort(-row_scores)
        order = order[np.isfinite(row_scores[order])]
        if order.size == 0:
            order = np.asarray([int(positives[row])], dtype=np.int64)
        if order.size < int(count):
            order = np.resize(order, int(count))
        output[row] = order[: int(count)]
    return output


def _same_query_render_exclusion_sets(query_indices: np.ndarray, render_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return render cells that should not be mined as negatives for each query cell."""

    qidx = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    ridx = np.asarray(render_indices, dtype=np.int64).reshape(-1)
    if qidx.shape[0] != ridx.shape[0]:
        raise ValueError("query_indices and render_indices must have the same length")
    by_query: dict[int, list[int]] = {}
    for query_id, render_id in zip(qidx.tolist(), ridx.tolist()):
        if int(query_id) < 0 or int(render_id) < 0:
            continue
        by_query.setdefault(int(query_id), []).append(int(render_id))
    return tuple(np.unique(np.asarray(by_query.get(int(query_id), [int(render_id)]), dtype=np.int64)) for query_id, render_id in zip(qidx.tolist(), ridx.tolist()))


def _dustbin_soft_labels(count: int) -> np.ndarray:
    output = np.zeros((int(count), 65), dtype=np.float32)
    if int(count) > 0:
        output[:, 64] = 1.0
    return output


def _confidence_ignore_mask_from_supervision(
    supervision: MatchaCoarseSupervision,
    *,
    mined_no_match_count: int,
) -> np.ndarray:
    """Build one joint-sample confidence ignore mask.

    `MatchaCoarseFineTrainingSet` intentionally stores only targets. The
    ignore mask lives on `MatchaJointTrainingSet`, so cache builders must keep
    it aligned with the positive, explicit no-match, and mined no-match rows.
    """

    return np.concatenate(
        [
            np.asarray(supervision.confidence_ignore_mask, dtype=bool),
            np.asarray(supervision.no_match_confidence_ignore_mask, dtype=bool),
            np.zeros((int(mined_no_match_count),), dtype=bool),
        ],
        axis=0,
    )


def build_matcha_joint_training_set_from_maps(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    supervision: MatchaCoarseSupervision,
    *,
    query_rgb: np.ndarray | None = None,
    render_rgb: np.ndarray | None = None,
    query_keypoint_label_map: np.ndarray | None = None,
    render_keypoint_label_map: np.ndarray | None = None,
    hard_negatives_per_match: int = 16,
    roundtrip_heatmap_threshold_px: float = 2.0,
    seed: int = 0,
) -> MatchaJointTrainingSet:
    """Build one full-map joint training sample from query/render maps."""

    query = np.asarray(query_feature_map, dtype=np.float32)
    render = np.asarray(render_feature_map, dtype=np.float32)
    if query.ndim != 3 or render.ndim != 3:
        raise ValueError("feature maps must have shape (C, H, W)")
    if int(query.shape[0]) != int(render.shape[0]):
        raise ValueError("query and render feature-map channels must match")
    samples = build_matcha_coarse_fine_training_set(
        query,
        render,
        supervision,
        hard_negatives_per_match=int(hard_negatives_per_match),
        seed=int(seed),
    )
    positive_count = int(samples.sample_count)
    supervision_no_match_count = int(getattr(supervision, "no_match_count", 0))
    no_match_count = 0
    mined_no_match_count = 0
    if supervision_no_match_count > 0:
        no_query = np.asarray(supervision.no_match_query_indices, dtype=np.int64)
        no_render = np.asarray(supervision.no_match_render_indices, dtype=np.int64)
        query_rows = _flatten_feature_map(query)
        render_rows = _flatten_feature_map(render)
        no_query_features = query_rows[no_query]
        no_render_features = render_rows[no_render]
        samples = MatchaCoarseFineTrainingSet(
            query_features=np.concatenate([samples.query_features, no_query_features], axis=0),
            render_features=np.concatenate([samples.render_features, no_render_features], axis=0),
            query_offset_labels=np.concatenate(
                [samples.query_offset_labels, np.full((supervision_no_match_count,), 64, dtype=np.int64)],
                axis=0,
            ),
            render_offset_labels=np.concatenate(
                [samples.render_offset_labels, np.full((supervision_no_match_count,), 64, dtype=np.int64)],
                axis=0,
            ),
            negative_render_features=np.concatenate(
                [
                    samples.negative_render_features,
                    np.resize(
                        samples.negative_render_features,
                        (
                            supervision_no_match_count,
                            int(samples.negative_render_features.shape[1]),
                            int(samples.negative_render_features.shape[2]),
                        ),
                    ),
                ],
                axis=0,
            ),
            roundtrip_errors_px=np.concatenate(
                [samples.roundtrip_errors_px, np.asarray(supervision.no_match_roundtrip_errors_px, dtype=np.float32)],
                axis=0,
            ),
            query_offset_soft_labels=np.concatenate(
                [
                    np.asarray(supervision.query_offset_soft_labels, dtype=np.float32),
                    _dustbin_soft_labels(supervision_no_match_count),
                ],
                axis=0,
            ),
            render_offset_soft_labels=np.concatenate(
                [
                    np.asarray(supervision.render_offset_soft_labels, dtype=np.float32),
                    _dustbin_soft_labels(supervision_no_match_count),
                ],
                axis=0,
            ),
            sample_confidence_targets=np.concatenate(
                [
                    np.asarray(supervision.confidence_targets, dtype=np.float32),
                    np.asarray(supervision.no_match_confidence_targets, dtype=np.float32),
                ],
                axis=0,
            ),
            sample_uncertainty_px=np.concatenate(
                [
                    np.asarray(supervision.uncertainty_px, dtype=np.float32),
                    np.asarray(supervision.no_match_roundtrip_errors_px, dtype=np.float32),
                ],
                axis=0,
            ),
            metadata={
                **dict(samples.metadata or {}),
                "positive_match_count": positive_count,
                "supervision_no_match_count": supervision_no_match_count,
            },
        )
        no_match_count += supervision_no_match_count
    if positive_count > 0 and int(samples.negative_render_features.shape[1]) > 0:
        # Promote the strongest mined negative into an explicit dustbin row.
        # Descriptor/full-map losses ignore these rows; the pair-confidence
        # head learns that this high-similarity query/render pair is not a
        # geometrically valid correspondence.
        no_match_render = np.asarray(samples.negative_render_features[:, 0, :], dtype=np.float32)
        samples = MatchaCoarseFineTrainingSet(
            query_features=np.concatenate([samples.query_features, samples.query_features], axis=0),
            render_features=np.concatenate([samples.render_features, no_match_render], axis=0),
            query_offset_labels=np.concatenate(
                [samples.query_offset_labels, np.full((positive_count,), 64, dtype=np.int64)],
                axis=0,
            ),
            render_offset_labels=np.concatenate(
                [samples.render_offset_labels, np.full((positive_count,), 64, dtype=np.int64)],
                axis=0,
            ),
            negative_render_features=np.concatenate(
                [samples.negative_render_features, samples.negative_render_features],
                axis=0,
            ),
            roundtrip_errors_px=np.concatenate(
                [samples.roundtrip_errors_px, np.full((positive_count,), np.inf, dtype=np.float32)],
                axis=0,
            ),
            query_offset_soft_labels=np.concatenate(
                [
                    np.asarray(samples.query_offset_soft_labels, dtype=np.float32)
                    if samples.query_offset_soft_labels is not None
                    else np.eye(65, dtype=np.float32)[np.asarray(samples.query_offset_labels, dtype=np.int64)],
                    _dustbin_soft_labels(positive_count),
                ],
                axis=0,
            ),
            render_offset_soft_labels=np.concatenate(
                [
                    np.asarray(samples.render_offset_soft_labels, dtype=np.float32)
                    if samples.render_offset_soft_labels is not None
                    else np.eye(65, dtype=np.float32)[np.asarray(samples.render_offset_labels, dtype=np.int64)],
                    _dustbin_soft_labels(positive_count),
                ],
                axis=0,
            ),
            sample_confidence_targets=np.concatenate(
                [
                    np.asarray(samples.sample_confidence_targets, dtype=np.float32)
                    if samples.sample_confidence_targets is not None
                    else np.ones((samples.sample_count,), dtype=np.float32),
                    np.zeros((positive_count,), dtype=np.float32),
                ],
                axis=0,
            ),
            sample_uncertainty_px=np.concatenate(
                [
                    np.asarray(samples.sample_uncertainty_px, dtype=np.float32)
                    if samples.sample_uncertainty_px is not None
                    else np.asarray(samples.roundtrip_errors_px, dtype=np.float32),
                    np.full((positive_count,), np.inf, dtype=np.float32),
                ],
                axis=0,
            ),
            query_keypoint_features=samples.query_keypoint_features,
            query_keypoint_labels=samples.query_keypoint_labels,
            render_keypoint_features=samples.render_keypoint_features,
            render_keypoint_labels=samples.render_keypoint_labels,
            metadata={
                **dict(samples.metadata or {}),
                "positive_match_count": positive_count,
                "supervision_no_match_count": supervision_no_match_count,
                "mined_no_match_count": positive_count,
                "no_match_count": int(no_match_count + positive_count),
            },
        )
        no_match_count += positive_count
        mined_no_match_count = positive_count
    qheat, rheat = heatmap_targets_from_coarse_supervision(
        supervision,
        query_grid_hw=(int(query.shape[1]), int(query.shape[2])),
        render_grid_hw=(int(render.shape[1]), int(render.shape[2])),
        roundtrip_threshold_px=float(roundtrip_heatmap_threshold_px),
    )
    return MatchaJointTrainingSet(
        coarse_fine_samples=samples,
        query_feature_maps=query[None],
        render_feature_maps=render[None],
        query_heatmap_targets=qheat[None],
        render_heatmap_targets=rheat[None],
        query_cell_indices=np.concatenate(
            [
                np.asarray(supervision.query_indices, dtype=np.int64),
                np.asarray(supervision.no_match_query_indices, dtype=np.int64),
                np.asarray(supervision.query_indices[:positive_count], dtype=np.int64),
            ],
            axis=0,
        ),
        render_cell_indices=np.concatenate(
            [
                np.asarray(supervision.render_indices, dtype=np.int64),
                np.asarray(supervision.no_match_render_indices, dtype=np.int64),
                np.asarray(supervision.render_indices[:positive_count], dtype=np.int64),
            ],
            axis=0,
        ),
        query_rgb_images=_rgb_to_bchw_float_if_labeled(
            query_rgb,
            query_keypoint_label_map,
            grid_hw=(int(query.shape[1]), int(query.shape[2])),
        ),
        render_rgb_images=_rgb_to_bchw_float_if_labeled(
            render_rgb,
            render_keypoint_label_map,
            grid_hw=(int(render.shape[1]), int(render.shape[2])),
        ),
        query_rgb_keypoint_labels=_label_bhw(query_keypoint_label_map),
        render_rgb_keypoint_labels=_label_bhw(render_keypoint_label_map),
        sample_no_match_labels=np.concatenate(
            [
                np.zeros((positive_count,), dtype=np.int64),
                np.ones((no_match_count,), dtype=np.int64),
            ],
            axis=0,
        ),
        sample_ignore_mask=np.zeros((int(samples.sample_count),), dtype=bool),
        sample_confidence_ignore_mask=_confidence_ignore_mask_from_supervision(
            supervision,
            mined_no_match_count=int(mined_no_match_count),
        ),
        query_repeatability_targets=qheat[None],
        render_repeatability_targets=rheat[None],
    )


def build_matcha_joint_index_training_set_from_maps(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    supervision: MatchaCoarseSupervision,
    *,
    query_rgb: np.ndarray | None = None,
    render_rgb: np.ndarray | None = None,
    query_keypoint_label_map: np.ndarray | None = None,
    render_keypoint_label_map: np.ndarray | None = None,
    hard_negatives_per_match: int = 16,
    roundtrip_heatmap_threshold_px: float = 2.0,
) -> MatchaJointTrainingSet:
    """Build one index-only full-map joint training sample.

    The returned set stores full feature maps plus cell indices. Row descriptors
    and hard-negative descriptors are gathered dynamically during training.
    """

    query = np.asarray(query_feature_map, dtype=np.float32)
    render = np.asarray(render_feature_map, dtype=np.float32)
    if query.ndim != 3 or render.ndim != 3:
        raise ValueError("feature maps must have shape (C, H, W)")
    if int(query.shape[0]) != int(render.shape[0]):
        raise ValueError("query and render feature-map channels must match")
    if supervision.count == 0:
        raise ValueError("index-only joint training requires at least one supervised match")
    query_rows = _flatten_feature_map(query)
    render_rows = _flatten_feature_map(render)
    if np.max(supervision.query_indices, initial=-1) >= query_rows.shape[0]:
        raise ValueError("supervision query index exceeds query feature map size")
    if np.max(supervision.render_indices, initial=-1) >= render_rows.shape[0]:
        raise ValueError("supervision render index exceeds render feature map size")
    positive_count = int(supervision.count)
    supervision_no_match_count = int(getattr(supervision, "no_match_count", 0))
    positive_query_indices = np.asarray(supervision.query_indices, dtype=np.int64)
    positive_render_indices = np.asarray(supervision.render_indices, dtype=np.int64)
    supervision_no_query_indices = np.asarray(supervision.no_match_query_indices, dtype=np.int64)
    supervision_no_render_indices = np.asarray(supervision.no_match_render_indices, dtype=np.int64)
    base_query_indices = np.concatenate([positive_query_indices, supervision_no_query_indices], axis=0)
    base_render_indices = np.concatenate([positive_render_indices, supervision_no_render_indices], axis=0)
    negative_indices = _mine_negative_render_indices(
        query_rows[base_query_indices],
        render_rows,
        base_render_indices,
        count=int(hard_negatives_per_match),
        excluded_render_indices_by_query=_same_query_render_exclusion_sets(base_query_indices, base_render_indices),
    )
    no_match_count = positive_count if negative_indices.shape[1] > 0 else 0
    query_cell_indices = np.concatenate([base_query_indices, positive_query_indices[:no_match_count]], axis=0)
    render_cell_indices = np.concatenate(
        [base_render_indices, negative_indices[:no_match_count, 0]],
        axis=0,
    )
    negative_render_indices = np.concatenate([negative_indices, negative_indices[:no_match_count]], axis=0)
    query_offset_labels = np.concatenate(
        [
            np.asarray(supervision.query_offset_labels, dtype=np.int64),
            np.full((supervision_no_match_count,), 64, dtype=np.int64),
            np.full((no_match_count,), 64, dtype=np.int64),
        ],
        axis=0,
    )
    render_offset_labels = np.concatenate(
        [
            np.asarray(supervision.render_offset_labels, dtype=np.int64),
            np.full((supervision_no_match_count,), 64, dtype=np.int64),
            np.full((no_match_count,), 64, dtype=np.int64),
        ],
        axis=0,
    )
    roundtrip_errors = np.concatenate(
        [
            np.asarray(supervision.roundtrip_errors_px, dtype=np.float32),
            np.asarray(supervision.no_match_roundtrip_errors_px, dtype=np.float32),
            np.full((no_match_count,), np.inf, dtype=np.float32),
        ],
        axis=0,
    )
    query_soft = np.concatenate(
        [
            np.asarray(supervision.query_offset_soft_labels, dtype=np.float32),
            _dustbin_soft_labels(supervision_no_match_count),
            _dustbin_soft_labels(no_match_count),
        ],
        axis=0,
    )
    render_soft = np.concatenate(
        [
            np.asarray(supervision.render_offset_soft_labels, dtype=np.float32),
            _dustbin_soft_labels(supervision_no_match_count),
            _dustbin_soft_labels(no_match_count),
        ],
        axis=0,
    )
    confidence_targets = np.concatenate(
        [
            np.asarray(supervision.confidence_targets, dtype=np.float32),
            np.asarray(supervision.no_match_confidence_targets, dtype=np.float32),
            np.zeros((no_match_count,), dtype=np.float32),
        ],
        axis=0,
    )
    uncertainty_px = np.concatenate(
        [
            np.asarray(supervision.uncertainty_px, dtype=np.float32),
            np.asarray(supervision.no_match_roundtrip_errors_px, dtype=np.float32),
            np.full((no_match_count,), np.inf, dtype=np.float32),
        ],
        axis=0,
    )
    sample_pair_indices = np.zeros((positive_count + supervision_no_match_count + no_match_count,), dtype=np.int64)
    qheat, rheat = heatmap_targets_from_coarse_supervision(
        supervision,
        query_grid_hw=(int(query.shape[1]), int(query.shape[2])),
        render_grid_hw=(int(render.shape[1]), int(render.shape[2])),
        roundtrip_threshold_px=float(roundtrip_heatmap_threshold_px),
    )
    base = IndexOnlyCoarseFineRows(
        query_feature_maps=query[None],
        render_feature_maps=render[None],
        query_cell_indices=query_cell_indices,
        render_cell_indices=render_cell_indices,
        negative_render_indices=negative_render_indices,
        query_offset_labels=query_offset_labels,
        render_offset_labels=render_offset_labels,
        roundtrip_errors_px=roundtrip_errors,
        query_offset_soft_labels=query_soft,
        render_offset_soft_labels=render_soft,
        sample_confidence_targets=confidence_targets,
        sample_uncertainty_px=uncertainty_px,
        sample_pair_indices=sample_pair_indices,
        metadata={
            "source": "matcha_coarse_supervision_index_only",
            "sample_count": int(positive_count + supervision_no_match_count + no_match_count),
            "positive_match_count": int(positive_count),
            "supervision_no_match_count": int(supervision_no_match_count),
            "mined_no_match_count": int(no_match_count),
            "no_match_count": int(supervision_no_match_count + no_match_count),
            "hard_negatives_per_match": int(hard_negatives_per_match),
        },
    )
    return MatchaJointTrainingSet(
        coarse_fine_samples=base,
        query_feature_maps=query[None],
        render_feature_maps=render[None],
        query_heatmap_targets=qheat[None],
        render_heatmap_targets=rheat[None],
        sample_pair_indices=sample_pair_indices,
        query_cell_indices=query_cell_indices,
        render_cell_indices=render_cell_indices,
        query_rgb_images=_rgb_to_bchw_float_if_labeled(
            query_rgb,
            query_keypoint_label_map,
            grid_hw=(int(query.shape[1]), int(query.shape[2])),
        ),
        render_rgb_images=_rgb_to_bchw_float_if_labeled(
            render_rgb,
            render_keypoint_label_map,
            grid_hw=(int(render.shape[1]), int(render.shape[2])),
        ),
        query_rgb_keypoint_labels=_label_bhw(query_keypoint_label_map),
        render_rgb_keypoint_labels=_label_bhw(render_keypoint_label_map),
        sample_no_match_labels=np.concatenate(
            [
                np.zeros((positive_count,), dtype=np.int64),
                np.ones((supervision_no_match_count,), dtype=np.int64),
                np.ones((no_match_count,), dtype=np.int64),
            ],
            axis=0,
        ),
        sample_ignore_mask=np.zeros((positive_count + supervision_no_match_count + no_match_count,), dtype=bool),
        sample_confidence_ignore_mask=_confidence_ignore_mask_from_supervision(
            supervision,
            mined_no_match_count=int(no_match_count),
        ),
        query_repeatability_targets=qheat[None],
        render_repeatability_targets=rheat[None],
    )
