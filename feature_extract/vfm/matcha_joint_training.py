"""MATCHA-first joint training loop for RADIO render/query matching.

This module intentionally keeps the old coarse-fine adapter usable, but trains
the MATCHA-inspired heads in one optimizer step:

* descriptor selector with dual-softmax correspondence loss,
* 8x8 offset and pair-fine heads,
* pair confidence head,
* dense heatmap/reliability head on selected feature maps,
* RGB-local 65-bin keypoint detector trained by ALIKE-style labels.
"""

from __future__ import annotations

import json
import random
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.landmark_retrieval_training import (
    LandmarkPrototypeMemoryBank,
    LandmarkRetrievalLossConfig,
    landmark_retrieval_loss,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    MatchaCoarseFineAdapter,
    MatchaCoarseFineTrainingRun,
    MatchaCoarseFineTrainingSet,
    _OriginalMatchaFineMatcher,
    _dual_softmax_descriptor_loss_and_confidence,
    _fine_coordinate_loss_and_metrics,
)
from feature_extract.vfm.matcha_patch_fine import PatchCorrelationFineHead
from feature_extract.vfm.matcha_rgb_keypoint_detector import (
    BasicConvLayer,
    MatchaRgbKeypointDetector,
    matcha_alike_distillation_loss,
    matcha_keypoint_position_loss,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementBranch,
    continuous_offset_nll_with_dustbin,
    crop_rgb_windows_by_owner,
    residual_delta_gaussian_nll,
)


_JOINT_FORMAT = "vfm_matcha_joint_training_set_v1"
_JOINT_INDEX_FORMAT = "vfm_matcha_joint_index_training_set_v2"
_JOINT_MODEL_FORMAT = "vfm_matcha_joint_model_v1"
_JOINT_MANIFEST_FORMAT = "vfm_matcha_joint_training_manifest_v1"


class _IndexedFeatureRows:
    def __init__(self, feature_maps: np.ndarray, cell_indices: np.ndarray, pair_indices: np.ndarray | None = None) -> None:
        maps = np.asarray(feature_maps, dtype=np.float32)
        if maps.ndim != 4:
            raise ValueError("feature_maps must have shape (B, C, H, W)")
        indices = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
        pairs = np.zeros((indices.shape[0],), dtype=np.int64) if pair_indices is None else np.asarray(pair_indices, dtype=np.int64).reshape(-1)
        if pairs.shape[0] != indices.shape[0]:
            raise ValueError("pair_indices must contain one value per cell index")
        if np.any(pairs < 0) or np.any(pairs >= maps.shape[0]):
            raise ValueError("pair_indices contains an out-of-range pair")
        if np.any(indices < 0) or np.any(indices >= int(maps.shape[2] * maps.shape[3])):
            raise ValueError("cell_indices contains an out-of-range cell")
        self.feature_maps = maps
        self.cell_indices = indices
        self.pair_indices = pairs
        self.shape = (int(indices.shape[0]), int(maps.shape[1]))

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, item) -> np.ndarray:
        rows = np.asarray(np.arange(len(self))[item], dtype=np.int64).reshape(-1)
        maps = self.feature_maps
        height, width = int(maps.shape[2]), int(maps.shape[3])
        pairs = self.pair_indices[rows]
        cells = self.cell_indices[rows]
        y = cells // width
        x = cells % width
        return maps[pairs, :, y, x].astype(np.float32, copy=False)


class _IndexedNegativeFeatureRows:
    def __init__(self, render_feature_maps: np.ndarray, negative_render_indices: np.ndarray, pair_indices: np.ndarray | None = None) -> None:
        maps = np.asarray(render_feature_maps, dtype=np.float32)
        if maps.ndim != 4:
            raise ValueError("render_feature_maps must have shape (B, C, H, W)")
        negative = np.asarray(negative_render_indices, dtype=np.int64)
        if negative.ndim != 2:
            raise ValueError("negative_render_indices must have shape (N, K)")
        pairs = np.zeros((negative.shape[0],), dtype=np.int64) if pair_indices is None else np.asarray(pair_indices, dtype=np.int64).reshape(-1)
        if pairs.shape[0] != negative.shape[0]:
            raise ValueError("pair_indices must contain one value per negative row")
        if np.any(pairs < 0) or np.any(pairs >= maps.shape[0]):
            raise ValueError("pair_indices contains an out-of-range pair")
        if np.any(negative < 0) or np.any(negative >= int(maps.shape[2] * maps.shape[3])):
            raise ValueError("negative_render_indices contains an out-of-range cell")
        self.render_feature_maps = maps
        self.negative_render_indices = negative
        self.pair_indices = pairs
        self.shape = (int(negative.shape[0]), int(negative.shape[1]), int(maps.shape[1]))

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, item) -> np.ndarray:
        rows = np.asarray(np.arange(len(self))[item], dtype=np.int64).reshape(-1)
        maps = self.render_feature_maps
        _batch, _channels, _height, width = maps.shape
        pairs = self.pair_indices[rows]
        cells = self.negative_render_indices[rows]
        y = cells // int(width)
        x = cells % int(width)
        gathered = maps[pairs[:, None], :, y, x]
        return gathered.astype(np.float32, copy=False)


class IndexOnlyCoarseFineRows:
    """Coarse/fine rows backed by full feature maps and integer cell indices."""

    def __init__(
        self,
        *,
        query_feature_maps: np.ndarray,
        render_feature_maps: np.ndarray,
        query_cell_indices: np.ndarray,
        render_cell_indices: np.ndarray,
        negative_render_indices: np.ndarray,
        query_offset_labels: np.ndarray,
        render_offset_labels: np.ndarray,
        roundtrip_errors_px: np.ndarray,
        query_offset_soft_labels: np.ndarray | None = None,
        render_offset_soft_labels: np.ndarray | None = None,
        sample_confidence_targets: np.ndarray | None = None,
        sample_uncertainty_px: np.ndarray | None = None,
        sample_pair_indices: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.query_features = _IndexedFeatureRows(query_feature_maps, query_cell_indices, sample_pair_indices)
        self.render_features = _IndexedFeatureRows(render_feature_maps, render_cell_indices, sample_pair_indices)
        self.negative_render_features = _IndexedNegativeFeatureRows(render_feature_maps, negative_render_indices, sample_pair_indices)
        self.query_offset_labels = np.asarray(query_offset_labels, dtype=np.int64).reshape(-1)
        self.render_offset_labels = np.asarray(render_offset_labels, dtype=np.int64).reshape(-1)
        self.roundtrip_errors_px = np.asarray(roundtrip_errors_px, dtype=np.float32).reshape(-1)
        self.query_offset_soft_labels = None if query_offset_soft_labels is None else np.asarray(query_offset_soft_labels, dtype=np.float32)
        self.render_offset_soft_labels = None if render_offset_soft_labels is None else np.asarray(render_offset_soft_labels, dtype=np.float32)
        self.sample_confidence_targets = None if sample_confidence_targets is None else np.asarray(sample_confidence_targets, dtype=np.float32).reshape(-1)
        self.sample_uncertainty_px = None if sample_uncertainty_px is None else np.asarray(sample_uncertainty_px, dtype=np.float32).reshape(-1)
        self.negative_render_indices = np.asarray(negative_render_indices, dtype=np.int64)
        self.metadata = dict(metadata or {})
        sample_count = int(self.query_features.shape[0])
        if self.render_features.shape[0] != sample_count or self.negative_render_features.shape[0] != sample_count:
            raise ValueError("index-only row views must have the same sample count")
        for name, value in (
            ("query_offset_labels", self.query_offset_labels),
            ("render_offset_labels", self.render_offset_labels),
            ("roundtrip_errors_px", self.roundtrip_errors_px),
        ):
            if value.shape[0] != sample_count:
                raise ValueError(f"{name} must contain one value per sample")
        for name, value in (("query_offset_soft_labels", self.query_offset_soft_labels), ("render_offset_soft_labels", self.render_offset_soft_labels)):
            if value is not None and value.shape != (sample_count, 65):
                raise ValueError(f"{name} must have shape (N, 65)")
        for name, value in (("sample_confidence_targets", self.sample_confidence_targets), ("sample_uncertainty_px", self.sample_uncertainty_px)):
            if value is not None and value.shape[0] != sample_count:
                raise ValueError(f"{name} must contain one value per sample")

    @property
    def sample_count(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.query_features.shape[1])


@dataclass(frozen=True)
class MatchaJointTrainingSet:
    coarse_fine_samples: MatchaCoarseFineTrainingSet
    query_feature_maps: np.ndarray | None = None
    render_feature_maps: np.ndarray | None = None
    query_heatmap_targets: np.ndarray | None = None
    render_heatmap_targets: np.ndarray | None = None
    sample_pair_indices: np.ndarray | None = None
    query_cell_indices: np.ndarray | None = None
    render_cell_indices: np.ndarray | None = None
    fine_sample_pair_indices: np.ndarray | None = None
    fine_query_cell_indices: np.ndarray | None = None
    fine_render_cell_indices: np.ndarray | None = None
    fine_query_offset_labels: np.ndarray | None = None
    fine_render_offset_labels: np.ndarray | None = None
    fine_query_offset_soft_labels: np.ndarray | None = None
    fine_render_offset_soft_labels: np.ndarray | None = None
    fine_query_xy: np.ndarray | None = None
    fine_render_xy: np.ndarray | None = None
    fine_render_depth: np.ndarray | None = None
    fine_support_view_count: np.ndarray | None = None
    fine_validity_weight: np.ndarray | None = None
    query_rgb_images: np.ndarray | None = None
    render_rgb_images: np.ndarray | None = None
    query_rgb_keypoint_labels: np.ndarray | None = None
    render_rgb_keypoint_labels: np.ndarray | None = None
    pair_type_ids: np.ndarray | None = None
    pair_type_names: np.ndarray | None = None
    pair_query_ids: np.ndarray | None = None
    pair_split_names: np.ndarray | None = None
    pair_candidate_ids: np.ndarray | None = None
    pair_translation_errors_m: np.ndarray | None = None
    pair_rotation_errors_deg: np.ndarray | None = None
    pair_query_image_sizes: np.ndarray | None = None
    pair_reference_image_sizes: np.ndarray | None = None
    sample_no_match_labels: np.ndarray | None = None
    sample_ignore_mask: np.ndarray | None = None
    sample_confidence_ignore_mask: np.ndarray | None = None
    sample_track_ids: np.ndarray | None = None
    sample_track_xyz: np.ndarray | None = None
    landmark_sample_pair_indices: np.ndarray | None = None
    landmark_query_xy: np.ndarray | None = None
    landmark_reference_xy: np.ndarray | None = None
    landmark_track_ids: np.ndarray | None = None
    landmark_track_xyz: np.ndarray | None = None
    landmark_support_view_counts: np.ndarray | None = None
    # Coarse-cell tracks are ambiguity exclusions: they must not become
    # negatives, but are too far apart to be interchangeable pose positives.
    landmark_known_positive_offsets: np.ndarray | None = None
    landmark_known_positive_track_ids: np.ndarray | None = None
    # Strict spatial positives use the cache's pixel reprojection threshold
    # and are valid set-valued retrieval targets.
    landmark_strict_positive_offsets: np.ndarray | None = None
    landmark_strict_positive_track_ids: np.ndarray | None = None
    # Train-only wrong track identities jointly supporting coherent bad poses.
    landmark_coherent_hard_negative_offsets: np.ndarray | None = None
    landmark_coherent_hard_negative_track_ids: np.ndarray | None = None
    landmark_coherent_hard_negative_mode_ids: np.ndarray | None = None
    query_repeatability_targets: np.ndarray | None = None
    render_repeatability_targets: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.coarse_fine_samples.sample_count <= 0:
            raise ValueError("coarse_fine_samples must contain at least one sample")

        def check_maps(name: str, maps: np.ndarray | None, targets: np.ndarray | None) -> None:
            if maps is None and targets is None:
                return
            if maps is None or targets is None:
                raise ValueError(f"{name} feature maps and heatmap targets must be provided together")
            arr = np.asarray(maps, dtype=np.float32)
            tgt = np.asarray(targets, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"{name} feature maps must have shape (B, C, H, W)")
            if tgt.shape != (arr.shape[0], arr.shape[2], arr.shape[3]):
                raise ValueError(f"{name} heatmap targets must have shape (B, H, W)")
            if int(arr.shape[1]) != int(self.coarse_fine_samples.input_dim):
                raise ValueError(f"{name} feature-map channels must match sample input_dim")

        def check_rgb(name: str, images: np.ndarray | None, labels: np.ndarray | None) -> None:
            if images is None and labels is None:
                return
            if images is None:
                raise ValueError(f"{name} keypoint labels require RGB images")
            img = np.asarray(images, dtype=np.float32)
            if img.ndim != 4 or int(img.shape[1]) != 3:
                raise ValueError(f"{name} RGB images must have shape (B, 3, H, W)")
            if int(img.shape[2]) % 8 != 0 or int(img.shape[3]) % 8 != 0:
                raise ValueError(f"{name} RGB image height and width must be divisible by 8")
            object.__setattr__(self, f"{name}_rgb_images", img)
            if labels is None:
                return
            lab = np.asarray(labels, dtype=np.int64)
            if lab.shape != (img.shape[0], img.shape[2] // 8, img.shape[3] // 8):
                raise ValueError(f"{name} labels must have shape (B, H/8, W/8)")
            if lab.size and (np.any(lab < 0) or np.any(lab > 64)):
                raise ValueError(f"{name} labels must be in [0, 64]")
            object.__setattr__(self, f"{name}_rgb_keypoint_labels", lab)

        check_maps("query", self.query_feature_maps, self.query_heatmap_targets)
        check_maps("render", self.render_feature_maps, self.render_heatmap_targets)
        check_rgb("query", self.query_rgb_images, self.query_rgb_keypoint_labels)
        check_rgb("render", self.render_rgb_images, self.render_rgb_keypoint_labels)
        for name, targets, maps in (
            ("query_repeatability_targets", self.query_repeatability_targets, self.query_feature_maps),
            ("render_repeatability_targets", self.render_repeatability_targets, self.render_feature_maps),
        ):
            if targets is None:
                continue
            if maps is None:
                raise ValueError(f"{name} requires corresponding feature maps")
            tgt = np.asarray(targets, dtype=np.float32)
            arr = np.asarray(maps, dtype=np.float32)
            if tgt.shape != (arr.shape[0], arr.shape[2], arr.shape[3]):
                raise ValueError(f"{name} must have shape (B, H, W)")
            object.__setattr__(self, name, tgt)
        sample_count = int(self.coarse_fine_samples.sample_count)
        for name, value, dtype in (
            ("sample_no_match_labels", self.sample_no_match_labels, np.int64),
            ("sample_ignore_mask", self.sample_ignore_mask, bool),
            ("sample_confidence_ignore_mask", self.sample_confidence_ignore_mask, bool),
            ("sample_track_ids", self.sample_track_ids, np.int64),
        ):
            if value is None:
                continue
            arr = np.asarray(value, dtype=dtype).reshape(-1)
            if arr.shape[0] != sample_count:
                raise ValueError(f"{name} must contain one value per coarse-fine sample")
            object.__setattr__(self, name, arr)
        if self.sample_track_xyz is not None:
            track_xyz = np.asarray(self.sample_track_xyz, dtype=np.float64).reshape(-1, 3)
            if track_xyz.shape[0] != sample_count:
                raise ValueError("sample_track_xyz must contain one 3D point per coarse-fine sample")
            object.__setattr__(self, "sample_track_xyz", track_xyz)
        if (self.query_cell_indices is None) != (self.render_cell_indices is None):
            raise ValueError("query_cell_indices and render_cell_indices must be provided together")
        if self.query_cell_indices is not None and self.render_cell_indices is not None:
            qidx = np.asarray(self.query_cell_indices, dtype=np.int64).reshape(-1)
            ridx = np.asarray(self.render_cell_indices, dtype=np.int64).reshape(-1)
            if qidx.shape[0] != self.coarse_fine_samples.sample_count or ridx.shape[0] != self.coarse_fine_samples.sample_count:
                raise ValueError("cell indices must contain one value per coarse-fine sample")
            object.__setattr__(self, "query_cell_indices", qidx)
            object.__setattr__(self, "render_cell_indices", ridx)
            if self.sample_pair_indices is None:
                pair_indices = np.zeros((self.coarse_fine_samples.sample_count,), dtype=np.int64)
            else:
                pair_indices = np.asarray(self.sample_pair_indices, dtype=np.int64).reshape(-1)
                if pair_indices.shape[0] != self.coarse_fine_samples.sample_count:
                    raise ValueError("sample_pair_indices must contain one value per coarse-fine sample")
            if np.any(pair_indices < 0):
                raise ValueError("sample_pair_indices must be non-negative")
            object.__setattr__(self, "sample_pair_indices", pair_indices)
        pair_count = int(self.query_feature_maps.shape[0]) if self.query_feature_maps is not None else 0
        for name in ("pair_query_image_sizes", "pair_reference_image_sizes"):
            value = getattr(self, name)
            if value is None:
                continue
            sizes = np.asarray(value, dtype=np.int64).reshape(-1, 2)
            if pair_count and sizes.shape[0] != pair_count:
                raise ValueError(f"{name} must contain one (width, height) pair per image pair")
            if sizes.size and np.any(sizes <= 0):
                raise ValueError(f"{name} must contain positive image dimensions")
            object.__setattr__(self, name, sizes)
        landmark_required = (
            "landmark_sample_pair_indices",
            "landmark_query_xy",
            "landmark_reference_xy",
            "landmark_track_ids",
        )
        landmark_present = [getattr(self, name) is not None for name in landmark_required]
        if any(landmark_present):
            if not all(landmark_present):
                missing = ", ".join(name for name in landmark_required if getattr(self, name) is None)
                raise ValueError(f"landmark retrieval supervision requires: {missing}")
            if self.query_feature_maps is None or self.render_feature_maps is None:
                raise ValueError("landmark retrieval supervision requires query/reference feature maps")
            landmark_pairs = np.asarray(self.landmark_sample_pair_indices, dtype=np.int64).reshape(-1)
            landmark_query_xy = np.asarray(self.landmark_query_xy, dtype=np.float64).reshape(-1, 2)
            landmark_reference_xy = np.asarray(self.landmark_reference_xy, dtype=np.float64).reshape(-1, 2)
            landmark_track_ids = np.asarray(self.landmark_track_ids, dtype=np.int64).reshape(-1)
            landmark_count = int(landmark_pairs.shape[0])
            if landmark_query_xy.shape[0] != landmark_count or landmark_reference_xy.shape[0] != landmark_count:
                raise ValueError("landmark query/reference xy must contain one coordinate per retrieval sample")
            if landmark_track_ids.shape[0] != landmark_count:
                raise ValueError("landmark_track_ids must contain one value per retrieval sample")
            if landmark_count and (np.any(landmark_pairs < 0) or np.any(landmark_pairs >= pair_count)):
                raise ValueError("landmark_sample_pair_indices contains an out-of-range pair")
            object.__setattr__(self, "landmark_sample_pair_indices", landmark_pairs)
            object.__setattr__(self, "landmark_query_xy", landmark_query_xy)
            object.__setattr__(self, "landmark_reference_xy", landmark_reference_xy)
            object.__setattr__(self, "landmark_track_ids", landmark_track_ids)
            if self.landmark_track_xyz is not None:
                landmark_xyz = np.asarray(self.landmark_track_xyz, dtype=np.float64).reshape(-1, 3)
                if landmark_xyz.shape[0] != landmark_count:
                    raise ValueError("landmark_track_xyz must contain one 3D point per retrieval sample")
                object.__setattr__(self, "landmark_track_xyz", landmark_xyz)
            if self.landmark_support_view_counts is not None:
                view_counts = np.asarray(self.landmark_support_view_counts, dtype=np.int64).reshape(-1)
                if view_counts.shape[0] != landmark_count:
                    raise ValueError("landmark_support_view_counts must contain one value per retrieval sample")
                object.__setattr__(self, "landmark_support_view_counts", np.maximum(view_counts, 0))
            def validate_track_csr(
                *,
                label: str,
                offsets_value: np.ndarray | None,
                track_ids_value: np.ndarray | None,
            ) -> tuple[np.ndarray | None, np.ndarray | None]:
                present = (offsets_value is not None, track_ids_value is not None)
                if not any(present):
                    return None, None
                if not all(present):
                    raise ValueError(f"{label} requires both offsets and track ids")
                offsets = np.asarray(offsets_value, dtype=np.int64).reshape(-1)
                track_ids = np.asarray(track_ids_value, dtype=np.int64).reshape(-1)
                if offsets.shape[0] != landmark_count + 1:
                    raise ValueError(
                        f"{label} offsets must contain one offset per row plus a sentinel"
                    )
                if (
                    int(offsets[0]) != 0
                    or np.any(np.diff(offsets) < 0)
                    or int(offsets[-1]) != int(track_ids.size)
                ):
                    raise ValueError(f"{label} CSR offsets are invalid")
                if track_ids.size and np.any(track_ids < 0):
                    raise ValueError(f"{label} track ids must be non-negative")
                for row, target_track_id in enumerate(landmark_track_ids.tolist()):
                    values = track_ids[int(offsets[row]) : int(offsets[row + 1])]
                    if values.size and int(target_track_id) not in set(values.tolist()):
                        raise ValueError(
                            f"each non-empty {label} row must include its supervised track"
                        )
                return offsets, track_ids

            known_offsets, known_track_ids = validate_track_csr(
                label="landmark coarse-cell ambiguity",
                offsets_value=self.landmark_known_positive_offsets,
                track_ids_value=self.landmark_known_positive_track_ids,
            )
            strict_offsets, strict_track_ids = validate_track_csr(
                label="landmark strict positive",
                offsets_value=self.landmark_strict_positive_offsets,
                track_ids_value=self.landmark_strict_positive_track_ids,
            )
            object.__setattr__(self, "landmark_known_positive_offsets", known_offsets)
            object.__setattr__(self, "landmark_known_positive_track_ids", known_track_ids)
            object.__setattr__(self, "landmark_strict_positive_offsets", strict_offsets)
            object.__setattr__(self, "landmark_strict_positive_track_ids", strict_track_ids)
            coherent_present = (
                self.landmark_coherent_hard_negative_offsets is not None,
                self.landmark_coherent_hard_negative_track_ids is not None,
            )
            if any(coherent_present):
                if not all(coherent_present):
                    raise ValueError(
                        "landmark coherent hard negatives require offsets and track ids"
                    )
                coherent_offsets = np.asarray(
                    self.landmark_coherent_hard_negative_offsets, dtype=np.int64
                ).reshape(-1)
                coherent_tracks = np.asarray(
                    self.landmark_coherent_hard_negative_track_ids, dtype=np.int64
                ).reshape(-1)
                coherent_modes = (
                    None
                    if self.landmark_coherent_hard_negative_mode_ids is None
                    else np.asarray(
                        self.landmark_coherent_hard_negative_mode_ids,
                        dtype=np.int64,
                    ).reshape(-1)
                )
                if coherent_offsets.shape[0] != landmark_count + 1:
                    raise ValueError(
                        "landmark coherent hard-negative offsets must contain one "
                        "offset per row plus a sentinel"
                    )
                if (
                    int(coherent_offsets[0]) != 0
                    or np.any(np.diff(coherent_offsets) < 0)
                    or int(coherent_offsets[-1]) != int(coherent_tracks.size)
                ):
                    raise ValueError("landmark coherent hard-negative CSR is invalid")
                if coherent_tracks.size and np.any(coherent_tracks < 0):
                    raise ValueError(
                        "landmark coherent hard-negative track ids must be non-negative"
                    )
                if coherent_modes is not None:
                    if coherent_modes.shape != coherent_tracks.shape:
                        raise ValueError(
                            "landmark coherent hard-negative mode ids must align "
                            "with track ids"
                        )
                    if coherent_modes.size and np.any(coherent_modes < 0):
                        raise ValueError(
                            "landmark coherent hard-negative mode ids must be non-negative"
                        )
                for row, target_track_id in enumerate(landmark_track_ids.tolist()):
                    values = coherent_tracks[
                        int(coherent_offsets[row]) : int(coherent_offsets[row + 1])
                    ]
                    if int(target_track_id) in set(values.tolist()):
                        raise ValueError(
                            "a supervised landmark track cannot be a coherent hard negative"
                        )
                object.__setattr__(
                    self,
                    "landmark_coherent_hard_negative_offsets",
                    coherent_offsets,
                )
                object.__setattr__(
                    self,
                    "landmark_coherent_hard_negative_track_ids",
                    coherent_tracks,
                )
                object.__setattr__(
                    self,
                    "landmark_coherent_hard_negative_mode_ids",
                    coherent_modes,
                )
            elif self.landmark_coherent_hard_negative_mode_ids is not None:
                raise ValueError(
                    "landmark coherent hard-negative mode ids require the track CSR"
                )
        fine_required = (
            "fine_sample_pair_indices",
            "fine_query_cell_indices",
            "fine_render_cell_indices",
            "fine_query_offset_labels",
            "fine_render_offset_labels",
        )
        fine_present = [getattr(self, name) is not None for name in fine_required]
        if any(fine_present):
            if not all(fine_present):
                missing = ", ".join(name for name in fine_required if getattr(self, name) is None)
                raise ValueError(f"dense fine supervision requires: {missing}")
            if self.query_feature_maps is None or self.render_feature_maps is None:
                raise ValueError("dense fine supervision requires query/render feature maps")
            fine_pairs = np.asarray(self.fine_sample_pair_indices, dtype=np.int64).reshape(-1)
            fine_qidx = np.asarray(self.fine_query_cell_indices, dtype=np.int64).reshape(-1)
            fine_ridx = np.asarray(self.fine_render_cell_indices, dtype=np.int64).reshape(-1)
            fine_qlabels = np.asarray(self.fine_query_offset_labels, dtype=np.int64).reshape(-1)
            fine_rlabels = np.asarray(self.fine_render_offset_labels, dtype=np.int64).reshape(-1)
            fine_count = int(fine_pairs.shape[0])
            for name, value in (
                ("fine_query_cell_indices", fine_qidx),
                ("fine_render_cell_indices", fine_ridx),
                ("fine_query_offset_labels", fine_qlabels),
                ("fine_render_offset_labels", fine_rlabels),
            ):
                if value.shape[0] != fine_count:
                    raise ValueError(f"{name} must contain one value per dense fine sample")
            if fine_count and (np.any(fine_pairs < 0) or np.any(fine_pairs >= pair_count)):
                raise ValueError("fine_sample_pair_indices contains an out-of-range pair")
            q_cell_count = int(self.query_feature_maps.shape[2] * self.query_feature_maps.shape[3])
            r_cell_count = int(self.render_feature_maps.shape[2] * self.render_feature_maps.shape[3])
            if fine_count and (np.any(fine_qidx < 0) or np.any(fine_qidx >= q_cell_count)):
                raise ValueError("fine_query_cell_indices contains an out-of-range cell")
            if fine_count and (np.any(fine_ridx < 0) or np.any(fine_ridx >= r_cell_count)):
                raise ValueError("fine_render_cell_indices contains an out-of-range cell")
            if fine_count and (np.any(fine_qlabels < 0) or np.any(fine_qlabels > 64)):
                raise ValueError("fine_query_offset_labels must be in [0, 64]")
            if fine_count and (np.any(fine_rlabels < 0) or np.any(fine_rlabels > 64)):
                raise ValueError("fine_render_offset_labels must be in [0, 64]")
            object.__setattr__(self, "fine_sample_pair_indices", fine_pairs)
            object.__setattr__(self, "fine_query_cell_indices", fine_qidx)
            object.__setattr__(self, "fine_render_cell_indices", fine_ridx)
            object.__setattr__(self, "fine_query_offset_labels", fine_qlabels)
            object.__setattr__(self, "fine_render_offset_labels", fine_rlabels)
            for name in ("fine_query_offset_soft_labels", "fine_render_offset_soft_labels"):
                value = getattr(self, name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32).reshape(-1, 65)
                if arr.shape[0] != fine_count:
                    raise ValueError(f"{name} must contain one distribution per dense fine sample")
                object.__setattr__(self, name, arr)
            for name in ("fine_query_xy", "fine_render_xy"):
                value = getattr(self, name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float64).reshape(-1, 2)
                if arr.shape[0] != fine_count:
                    raise ValueError(f"{name} must contain one xy coordinate per dense fine sample")
                object.__setattr__(self, name, arr)
            for name, dtype in (
                ("fine_render_depth", np.float32),
                ("fine_validity_weight", np.float32),
                ("fine_support_view_count", np.int64),
            ):
                value = getattr(self, name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=dtype).reshape(-1)
                if arr.shape[0] != fine_count:
                    raise ValueError(f"{name} must contain one value per dense fine sample")
                object.__setattr__(self, name, arr)
        for name, value, dtype in (
            ("pair_type_ids", self.pair_type_ids, np.int64),
            ("pair_translation_errors_m", self.pair_translation_errors_m, np.float32),
            ("pair_rotation_errors_deg", self.pair_rotation_errors_deg, np.float32),
        ):
            if value is None:
                continue
            arr = np.asarray(value, dtype=dtype).reshape(-1)
            if pair_count and arr.shape[0] != pair_count:
                raise ValueError(f"{name} must contain one value per query/render pair")
            object.__setattr__(self, name, arr)
        if self.pair_type_names is not None:
            names = np.asarray(self.pair_type_names, dtype=object).reshape(-1)
            if pair_count and names.shape[0] != pair_count:
                raise ValueError("pair_type_names must contain one value per query/render pair")
            object.__setattr__(self, "pair_type_names", names)
        for name in ("pair_query_ids", "pair_split_names", "pair_candidate_ids"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=object).reshape(-1)
            if pair_count and arr.shape[0] != pair_count:
                raise ValueError(f"{name} must contain one value per query/render pair")
            object.__setattr__(self, name, arr)


def subset_landmark_known_positive_csr(
    offsets: np.ndarray,
    track_ids: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select ragged known-positive rows while preserving CSR invariants."""

    source_offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    source_track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    selection = np.asarray(keep)
    if selection.dtype == bool:
        if selection.shape != (source_offsets.size - 1,):
            raise ValueError("known-positive CSR boolean selector has the wrong length")
        row_indices = np.flatnonzero(selection)
    else:
        row_indices = selection.astype(np.int64, copy=False).reshape(-1)
    if row_indices.size and (
        np.any(row_indices < 0) or np.any(row_indices >= source_offsets.size - 1)
    ):
        raise ValueError("known-positive CSR row selector is out of range")
    pieces = [
        source_track_ids[int(source_offsets[row]) : int(source_offsets[row + 1])]
        for row in row_indices.tolist()
    ]
    counts = np.asarray([piece.size for piece in pieces], dtype=np.int64)
    output_offsets = np.zeros((row_indices.size + 1,), dtype=np.int64)
    if counts.size:
        output_offsets[1:] = np.cumsum(counts)
    output_track_ids = (
        np.zeros((0,), dtype=np.int64)
        if not pieces or int(output_offsets[-1]) == 0
        else np.concatenate(pieces, axis=0).astype(np.int64, copy=False)
    )
    return output_offsets, output_track_ids


def _landmark_known_positive_csr_to_padded(
    offsets: np.ndarray,
    track_ids: np.ndarray,
) -> np.ndarray:
    source_offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    source_track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    counts = np.diff(source_offsets)
    width = int(counts.max()) if counts.size else 0
    output = np.full((counts.size, width), -1, dtype=np.int64)
    for row, count in enumerate(counts.tolist()):
        if int(count) > 0:
            output[row, : int(count)] = source_track_ids[
                int(source_offsets[row]) : int(source_offsets[row + 1])
            ]
    return output


@dataclass(frozen=True)
class MatchaJointTrainingConfig:
    model_type: str = "residual_adapter"
    output_dim: int = 128
    residual_hidden_dim: int = 256
    fine_input_dim: int = 0
    coarse_input_dim: int = 0
    attention_hidden_dim: int = 256
    attention_depth: int = 2
    attention_heads: int = 4
    attention_patch_size: int = 2
    attention_upsample_mode: str = "bilinear"
    attention_fusion_mode: str = "matcha_original"
    context_hidden_dim: int = 256
    context_broad_kernel_size: int = 7
    context_freeze_base_steps: int = 0
    steps: int = 300
    batch_size: int = 512
    lr: float = 5e-5
    temperature: float = 0.07
    dual_softmax_weight: float = 1.0
    offset_loss_weight: float = 0.25
    pair_fine_loss_weight: float = 0.25
    query_pair_fine_loss_weight: float = 0.0
    fine_continuous_loss_weight: float = 0.25
    fine_loss_mode: str = "ce_plus_continuous"
    fine_uncertainty_loss_weight: float = 0.05
    pair_confidence_loss_weight: float = 0.1
    dense_heatmap_loss_weight: float = 0.25
    rgb_keypoint_loss_weight: float = 0.25
    rgb_keypoint_position_loss_weight: float = 0.0
    repeatability_loss_weight: float = 0.0
    local_fine_transformer_loss_weight: float = 0.0
    local_window_fine_loss_weight: float = 0.0
    local_window_fine_mode: str = "mlp"
    patch_corr_fine_loss_weight: float = 0.0
    patch_corr_fine_epe_weight: float = 0.1
    patch_corr_fine_batch_size: int = 256
    patch_corr_fine_max_samples_per_pair: int = 512
    patch_corr_fine_detach_context: bool = True
    measurement_patch_loss_weight: float = 0.0
    measurement_patch_direct_loss_weight: float = 0.25
    measurement_patch_epe_weight: float = 0.05
    measurement_patch_dustbin_bce_weight: float = 0.25
    measurement_patch_batch_size: int = 128
    measurement_patch_max_samples_per_pair: int = 512
    measurement_patch_search_radius_px: float = 8.0
    measurement_patch_context_radius_px: float = 8.0
    measurement_patch_step_px: float = 1.0
    measurement_patch_coarse_search_radius_px: float = 0.0
    measurement_patch_coarse_step_px: float = 0.0
    measurement_patch_feature_dim: int = 32
    measurement_patch_hidden_dim: int = 64
    measurement_patch_target_heatmap_sigma_px: float = 0.5
    measurement_patch_dustbin_positive_weight: float = 1.0
    measurement_patch_encoder_arch: str = "simple"
    measurement_patch_input_mode: str = "rgb"
    patch_correlation_loss_weight: float = 0.0
    patch_correlation_window_size: int = 3
    hard_negative_weight: float = 0.2
    hard_negative_margin: float = 0.2
    coarse_candidate_rank_loss_weight: float = 0.0
    coarse_candidate_rank_margin: float = 0.2
    hard_false_match_weight: float = 0.0
    hard_false_match_margin: float = 0.2
    landmark_retrieval_loss_weight: float = 0.0
    landmark_retrieval_temperature: float = 0.07
    landmark_prototype_history_mix: float = 0.5
    landmark_prototype_aggregation_method: str = "mean"
    landmark_l2_normalize_observations: bool = False
    landmark_normalize_final_prototypes: bool = True
    landmark_min_support_observations: int = 1
    landmark_positive_prototype_source: str = "episode_support_observations"
    landmark_set_valued_cell_positives: bool = True
    landmark_exclude_known_cell_positives_from_memory: bool = True
    landmark_memory_capacity: int = 65536
    landmark_frozen_negative_bank: str = ""
    # Mutable full-map initialization for retrieval training. Unlike the
    # read-only negative bank, this preserves the deployment track universe
    # while allowing EMA descriptor refreshes during training.
    landmark_memory_warm_start_bank: str = ""
    landmark_memory_sync_ddp: bool = False
    landmark_memory_momentum: float = 0.9
    landmark_memory_candidate_pool_size: int = 0
    landmark_semantic_hard_negatives_per_query: int = 16
    landmark_geometry_hard_negatives_per_track: int = 8
    landmark_random_negatives: int = 128
    landmark_max_memory_negatives: int = 2048
    landmark_memory_negative_merge_policy: str = "source_balanced_round_robin"
    landmark_system_hard_negative_margin: float = 0.05
    landmark_system_hard_negative_margin_weight: float = 0.0
    landmark_coherent_hard_negative_margin: float = 0.05
    landmark_coherent_hard_negative_margin_weight: float = 0.0
    landmark_coherent_hard_negative_min_mode_rows: int = 4
    landmark_dustbin_logit: float = 0.0
    landmark_dustbin_samples_per_image: int = 0
    landmark_dustbin_exclusion_radius_cells: int = 1
    landmark_dustbin_max_heatmap_target: float = 0.01
    landmark_dustbin_loss_weight: float = 0.25
    landmark_dustbin_detach_descriptors: bool = True
    group_size: int = 64
    input_norm_mode: str = "identity"
    gate_mode: str = "residual"
    residual_gate_scale: float = 0.1
    rgb_non_keypoint_divisor: int = 32
    map_pair_batch_size: int = 8
    validation_selection_metric: str = "total_loss"
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if str(self.model_type) not in {
            "residual_adapter",
            "radio_spatial_context",
            "radio_dual_attention",
        }:
            raise ValueError(
                "model_type must be 'residual_adapter', "
                "'radio_spatial_context', or 'radio_dual_attention'"
            )
        if int(self.output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        if int(self.residual_hidden_dim) <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if int(self.context_hidden_dim) <= 0:
            raise ValueError("context_hidden_dim must be positive")
        if (
            int(self.context_broad_kernel_size) <= 0
            or int(self.context_broad_kernel_size) % 2 == 0
        ):
            raise ValueError("context_broad_kernel_size must be a positive odd integer")
        if int(self.context_freeze_base_steps) < 0:
            raise ValueError("context_freeze_base_steps must be non-negative")
        if int(self.context_freeze_base_steps) > 0 and str(self.model_type) != "radio_spatial_context":
            raise ValueError(
                "context_freeze_base_steps is only valid for radio_spatial_context"
            )
        if int(self.context_freeze_base_steps) > int(self.steps):
            raise ValueError("context_freeze_base_steps cannot exceed training steps")
        if str(self.model_type) == "radio_dual_attention":
            if int(self.fine_input_dim) <= 0 or int(self.coarse_input_dim) <= 0:
                raise ValueError("fine_input_dim and coarse_input_dim must be positive for radio_dual_attention")
            if int(self.attention_hidden_dim) <= 0:
                raise ValueError("attention_hidden_dim must be positive")
            if int(self.attention_depth) <= 0:
                raise ValueError("attention_depth must be positive")
            if int(self.attention_heads) <= 0:
                raise ValueError("attention_heads must be positive")
            if int(self.attention_patch_size) <= 0:
                raise ValueError("attention_patch_size must be positive")
            if str(self.attention_upsample_mode) not in {"bilinear", "pixel_shuffle"}:
                raise ValueError("attention_upsample_mode must be 'bilinear' or 'pixel_shuffle'")
            if str(self.attention_fusion_mode) not in {"legacy", "matcha_original"}:
                raise ValueError("attention_fusion_mode must be 'legacy' or 'matcha_original'")
            if str(self.attention_upsample_mode) == "pixel_shuffle" and int(self.attention_hidden_dim) % (int(self.attention_patch_size) ** 2) != 0:
                raise ValueError("attention_hidden_dim must be divisible by attention_patch_size^2 for pixel_shuffle")
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive")
        if int(self.batch_size) <= 1:
            raise ValueError("batch_size must be greater than one")
        if int(self.map_pair_batch_size) <= 0:
            raise ValueError("map_pair_batch_size must be positive")
        if str(self.validation_selection_metric) not in {
            "total_loss",
            "landmark_retrieval_loss",
        }:
            raise ValueError(
                "validation_selection_metric must be 'total_loss' or "
                "'landmark_retrieval_loss'"
            )
        if int(self.patch_corr_fine_batch_size) <= 0:
            raise ValueError("patch_corr_fine_batch_size must be positive")
        if int(self.patch_corr_fine_max_samples_per_pair) < 0:
            raise ValueError("patch_corr_fine_max_samples_per_pair must be non-negative")
        if int(self.measurement_patch_batch_size) <= 0:
            raise ValueError("measurement_patch_batch_size must be positive")
        if int(self.measurement_patch_max_samples_per_pair) < 0:
            raise ValueError("measurement_patch_max_samples_per_pair must be non-negative")
        if float(self.measurement_patch_search_radius_px) <= 0.0:
            raise ValueError("measurement_patch_search_radius_px must be positive")
        if float(self.measurement_patch_context_radius_px) < 0.0:
            raise ValueError("measurement_patch_context_radius_px must be non-negative")
        if float(self.measurement_patch_step_px) <= 0.0:
            raise ValueError("measurement_patch_step_px must be positive")
        if float(self.measurement_patch_coarse_search_radius_px) < 0.0:
            raise ValueError("measurement_patch_coarse_search_radius_px must be non-negative")
        if float(self.measurement_patch_coarse_step_px) < 0.0:
            raise ValueError("measurement_patch_coarse_step_px must be non-negative")
        if (float(self.measurement_patch_coarse_search_radius_px) > 0.0) != (float(self.measurement_patch_coarse_step_px) > 0.0):
            raise ValueError("measurement coarse search radius and step must be enabled together")
        if int(self.measurement_patch_feature_dim) <= 0:
            raise ValueError("measurement_patch_feature_dim must be positive")
        if int(self.measurement_patch_hidden_dim) <= 0:
            raise ValueError("measurement_patch_hidden_dim must be positive")
        if float(self.measurement_patch_target_heatmap_sigma_px) < 0.0:
            raise ValueError("measurement_patch_target_heatmap_sigma_px must be non-negative")
        if float(self.measurement_patch_dustbin_positive_weight) <= 0.0:
            raise ValueError("measurement_patch_dustbin_positive_weight must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if float(self.landmark_retrieval_temperature) <= 0.0:
            raise ValueError("landmark_retrieval_temperature must be positive")
        if not 0.0 <= float(self.landmark_prototype_history_mix) <= 1.0:
            raise ValueError("landmark_prototype_history_mix must be in [0, 1]")
        if str(self.landmark_prototype_aggregation_method) not in {
            "mean",
            "cosine_weighted_mean",
            "geometry_weighted",
        }:
            raise ValueError("unsupported landmark_prototype_aggregation_method")
        if int(self.landmark_min_support_observations) <= 0:
            raise ValueError("landmark_min_support_observations must be positive")
        if str(self.landmark_positive_prototype_source) not in {
            "episode_support_observations",
            "query_disjoint_frozen_bank",
        }:
            raise ValueError("unsupported landmark_positive_prototype_source")
        if not 0.0 <= float(self.landmark_memory_momentum) < 1.0:
            raise ValueError("landmark_memory_momentum must be in [0, 1)")
        if int(self.landmark_memory_capacity) <= 0:
            raise ValueError("landmark_memory_capacity must be positive")
        if str(self.landmark_frozen_negative_bank) and str(self.landmark_memory_warm_start_bank):
            raise ValueError(
                "landmark_frozen_negative_bank and landmark_memory_warm_start_bank are mutually exclusive"
            )
        if bool(self.landmark_memory_sync_ddp) and not str(self.landmark_memory_warm_start_bank):
            raise ValueError(
                "landmark_memory_sync_ddp requires landmark_memory_warm_start_bank "
                "so every rank starts from the same full track universe"
            )
        if (
            str(self.landmark_memory_warm_start_bank)
            and str(self.landmark_positive_prototype_source) == "query_disjoint_frozen_bank"
        ):
            raise ValueError(
                "query_disjoint_frozen_bank positives require landmark_frozen_negative_bank, "
                "not a mutable warm-start bank"
            )
        for name in (
            "landmark_memory_candidate_pool_size",
            "landmark_semantic_hard_negatives_per_query",
            "landmark_geometry_hard_negatives_per_track",
            "landmark_random_negatives",
            "landmark_max_memory_negatives",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if str(self.landmark_memory_negative_merge_policy) not in {
            "legacy_source_concat",
            "source_balanced_round_robin",
        }:
            raise ValueError("unsupported landmark memory negative merge policy")
        if float(self.landmark_system_hard_negative_margin) < 0.0:
            raise ValueError("landmark_system_hard_negative_margin must be non-negative")
        if float(self.landmark_system_hard_negative_margin_weight) < 0.0:
            raise ValueError("landmark_system_hard_negative_margin_weight must be non-negative")
        if float(self.landmark_coherent_hard_negative_margin) < 0.0:
            raise ValueError("landmark_coherent_hard_negative_margin must be non-negative")
        if float(self.landmark_coherent_hard_negative_margin_weight) < 0.0:
            raise ValueError(
                "landmark_coherent_hard_negative_margin_weight must be non-negative"
            )
        if int(self.landmark_coherent_hard_negative_min_mode_rows) < 2:
            raise ValueError(
                "landmark_coherent_hard_negative_min_mode_rows must be at least 2"
            )
        if np.isnan(float(self.landmark_dustbin_logit)):
            raise ValueError("landmark_dustbin_logit must not be NaN")
        if int(self.landmark_dustbin_samples_per_image) < 0:
            raise ValueError("landmark_dustbin_samples_per_image must be non-negative")
        if int(self.landmark_dustbin_exclusion_radius_cells) < 0:
            raise ValueError("landmark_dustbin_exclusion_radius_cells must be non-negative")
        if not 0.0 <= float(self.landmark_dustbin_max_heatmap_target) <= 1.0:
            raise ValueError("landmark_dustbin_max_heatmap_target must be in [0, 1]")
        if float(self.landmark_dustbin_loss_weight) < 0.0:
            raise ValueError("landmark_dustbin_loss_weight must be non-negative")
        for name in (
            "dual_softmax_weight",
            "offset_loss_weight",
            "pair_fine_loss_weight",
            "query_pair_fine_loss_weight",
            "fine_continuous_loss_weight",
            "fine_uncertainty_loss_weight",
            "pair_confidence_loss_weight",
            "dense_heatmap_loss_weight",
            "rgb_keypoint_loss_weight",
            "rgb_keypoint_position_loss_weight",
            "repeatability_loss_weight",
            "local_fine_transformer_loss_weight",
            "local_window_fine_loss_weight",
            "patch_corr_fine_loss_weight",
            "patch_corr_fine_epe_weight",
            "measurement_patch_loss_weight",
            "measurement_patch_direct_loss_weight",
            "measurement_patch_epe_weight",
            "measurement_patch_dustbin_bce_weight",
            "patch_correlation_loss_weight",
            "hard_negative_weight",
            "coarse_candidate_rank_loss_weight",
            "hard_false_match_weight",
            "landmark_retrieval_loss_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.patch_correlation_window_size) <= 0 or int(self.patch_correlation_window_size) % 2 == 0:
            raise ValueError("patch_correlation_window_size must be a positive odd integer")
        if str(self.local_window_fine_mode) not in {"mlp", "correlation"}:
            raise ValueError("local_window_fine_mode must be 'mlp' or 'correlation'")
        if str(self.fine_loss_mode) not in {"ce", "ce_plus_continuous", "continuous"}:
            raise ValueError("fine_loss_mode must be 'ce', 'ce_plus_continuous', or 'continuous'")
        if int(self.rgb_non_keypoint_divisor) <= 0:
            raise ValueError("rgb_non_keypoint_divisor must be positive")


@dataclass
class MatchaJointTrainingRun:
    model: "MatchaStyleJointModel"
    summary: dict[str, object]


class MatchaStyleJointModel(nn.Module):
    """MATCHA-first wrapper around the existing RADIO selector adapter."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        residual_hidden_dim: int = 256,
        group_size: int = 64,
        input_norm_mode: str = "identity",
        gate_mode: str = "residual",
        residual_gate_scale: float = 0.1,
        local_window_fine_mode: str = "mlp",
        measurement_patch_search_radius_px: float = 8.0,
        measurement_patch_context_radius_px: float = 8.0,
        measurement_patch_step_px: float = 1.0,
        measurement_patch_coarse_search_radius_px: float = 0.0,
        measurement_patch_coarse_step_px: float = 0.0,
        measurement_patch_feature_dim: int = 32,
        measurement_patch_hidden_dim: int = 64,
        measurement_patch_encoder_arch: str = "simple",
        measurement_patch_input_mode: str = "rgb",
    ) -> None:
        super().__init__()
        self.local_window_fine_mode = str(local_window_fine_mode)
        self.measurement_patch_config = {
            "search_radius_px": float(measurement_patch_search_radius_px),
            "context_radius_px": float(measurement_patch_context_radius_px),
            "step_px": float(measurement_patch_step_px),
            "coarse_search_radius_px": float(measurement_patch_coarse_search_radius_px),
            "coarse_step_px": float(measurement_patch_coarse_step_px),
            "feature_dim": int(measurement_patch_feature_dim),
            "hidden_dim": int(measurement_patch_hidden_dim),
            "encoder_arch": str(measurement_patch_encoder_arch),
            "input_mode": str(measurement_patch_input_mode),
        }
        self.adapter = MatchaCoarseFineAdapter(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            residual_hidden_dim=int(residual_hidden_dim),
            group_size=int(group_size),
            input_norm_mode=str(input_norm_mode),
            gate_mode=str(gate_mode),
            residual_gate_scale=float(residual_gate_scale),
        )
        self.rgb_keypoint_detector = MatchaRgbKeypointDetector()
        self.feature_fusion = nn.Sequential(
            BasicConvLayer(int(input_dim), int(input_dim), 1, padding=0),
            nn.Conv2d(int(input_dim), int(input_dim), 1),
        )
        self.heatmap_head = nn.Sequential(
            BasicConvLayer(int(output_dim), int(residual_hidden_dim), 3, padding=1),
            BasicConvLayer(int(residual_hidden_dim), int(residual_hidden_dim), 1, padding=0),
            nn.Conv2d(int(residual_hidden_dim), 1, 1),
        )
        self.landmark_dustbin_head = nn.Sequential(
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), 1),
        )
        nn.init.zeros_(self.landmark_dustbin_head[-1].weight)
        nn.init.constant_(self.landmark_dustbin_head[-1].bias, 0.7)
        heads = 4
        while int(output_dim) % heads != 0 and heads > 1:
            heads -= 1
        self.local_fine_attention = nn.MultiheadAttention(int(output_dim), num_heads=heads, batch_first=True)
        self.local_fine_norm = nn.LayerNorm(int(output_dim))
        self.local_fine_head = nn.Sequential(
            nn.Linear(int(output_dim), int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 64),
        )
        self.query_pair_fine_head = _OriginalMatchaFineMatcher(
            descriptor_dim=int(output_dim),
            hidden_dim=int(residual_hidden_dim),
            output_bins=64,
        )
        self.local_window_query_proj = nn.Linear(int(output_dim), int(residual_hidden_dim))
        self.local_window_render_proj = nn.Linear(int(output_dim), int(residual_hidden_dim))
        self.local_window_score_head = nn.Sequential(
            nn.LayerNorm(int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.local_window_uncertainty_head = nn.Sequential(
            nn.LayerNorm(int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.local_window_logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.patch_corr_fine_head = PatchCorrelationFineHead(
            context_dim=int(output_dim),
            hidden_dim=int(residual_hidden_dim),
            patch_size=32,
            offset_bins=8,
        )
        coarse_radius = float(measurement_patch_coarse_search_radius_px)
        coarse_step = float(measurement_patch_coarse_step_px)
        self.measurement_patch_branch = RGBPatchMeasurementBranch(
            search_radius_px=float(measurement_patch_search_radius_px),
            context_radius_px=float(measurement_patch_context_radius_px),
            step_px=float(measurement_patch_step_px),
            coarse_search_radius_px=None if coarse_radius <= 0.0 else coarse_radius,
            coarse_step_px=None if coarse_step <= 0.0 else coarse_step,
            feature_dim=int(measurement_patch_feature_dim),
            hidden_dim=int(measurement_patch_hidden_dim),
            encoder_arch=str(measurement_patch_encoder_arch),
            input_mode=str(measurement_patch_input_mode),
        )

    @property
    def input_dim(self) -> int:
        return int(self.adapter.input_dim)

    @property
    def output_dim(self) -> int:
        return int(self.adapter.output_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter.encode(features)

    def landmark_dustbin_logits(self, query_descriptors: torch.Tensor) -> torch.Tensor:
        return self.landmark_dustbin_head(query_descriptors).reshape(-1)

    def forward_rows(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(features)

    def pair_confidence_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_confidence_logits(query_descriptors, render_descriptors)

    def pair_fine_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_fine_logits(query_descriptors, render_descriptors)

    def query_pair_fine_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.query_pair_fine_head(query_descriptors, render_descriptors)

    def pair_fine_uncertainty_log_sigma(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_fine_uncertainty_log_sigma(query_descriptors, render_descriptors)

    def query_pair_fine_uncertainty_log_sigma(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_fine_uncertainty_log_sigma(render_descriptors, query_descriptors)

    def fuse_feature_map(self, feature_maps: torch.Tensor) -> torch.Tensor:
        fused = self.feature_fusion(feature_maps)
        return feature_maps + 0.1 * fused

    def forward_feature_map(self, feature_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature_maps.ndim != 4 or int(feature_maps.shape[1]) != self.input_dim:
            raise ValueError("feature_maps must have shape (B, input_dim, H, W)")
        feature_maps = self.fuse_feature_map(feature_maps)
        batch, _channels, height, width = feature_maps.shape
        rows = feature_maps.permute(0, 2, 3, 1).reshape(-1, self.input_dim)
        descriptors, offset_logits = self.adapter(rows)
        descriptor_map = descriptors.reshape(batch, height, width, self.output_dim).permute(0, 3, 1, 2).contiguous()
        offset_map = offset_logits.reshape(batch, height, width, 65).permute(0, 3, 1, 2).contiguous()
        heatmap_logits = self.heatmap_head(descriptor_map)
        return descriptor_map, heatmap_logits, offset_map


    def forward_rgb_keypoints(self, images: torch.Tensor) -> torch.Tensor:
        return self.rgb_keypoint_detector(images)

    def local_fine_logits_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        batch, channels, qh, qw = query_desc.shape
        if int(render_desc.shape[0]) != int(batch):
            raise ValueError("query/render feature maps must have the same batch size")
        query_rows = query_desc.permute(0, 2, 3, 1).reshape(batch, qh * qw, channels)
        _rb, _rc, rh, rw = render_desc.shape
        render_rows = render_desc.permute(0, 2, 3, 1).reshape(batch, rh * rw, channels)
        pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
        qidx = query_indices.long().reshape(-1).clamp(0, qh * qw - 1)
        ridx = render_indices.long().reshape(-1).clamp(0, rh * rw - 1)
        qtoken = query_rows[pairs, qidx]
        rtoken = render_rows[pairs, ridx]
        sequence = torch.stack([qtoken, rtoken], dim=1)
        attended, _weights = self.local_fine_attention(sequence, sequence, sequence, need_weights=False)
        attended = self.local_fine_norm(attended[:, 0] + qtoken)
        return self.local_fine_head(attended)

    def local_window_fine_logits_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        query_tokens = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_candidates = _sample_local_window_descriptors(render_desc, pair_indices, render_indices)
        if str(self.local_window_fine_mode) == "correlation":
            logits, _sigma = _score_local_window_correlation_candidates_with_uncertainty(
                query_tokens,
                render_candidates,
                self.local_window_query_proj,
                self.local_window_render_proj,
                self.local_window_uncertainty_head,
                logit_scale=torch.exp(torch.clamp(self.local_window_logit_scale, min=-4.0, max=4.0)),
            )
            return logits
        return _score_local_window_candidates(
            query_tokens,
            render_candidates,
            self.local_window_query_proj,
            self.local_window_render_proj,
            self.local_window_score_head,
        )

    def local_window_fine_logits_uncertainty_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        query_tokens = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_candidates = _sample_local_window_descriptors(render_desc, pair_indices, render_indices)
        if str(self.local_window_fine_mode) == "correlation":
            return _score_local_window_correlation_candidates_with_uncertainty(
                query_tokens,
                render_candidates,
                self.local_window_query_proj,
                self.local_window_render_proj,
                self.local_window_uncertainty_head,
                logit_scale=torch.exp(torch.clamp(self.local_window_logit_scale, min=-4.0, max=4.0)),
            )
        return _score_local_window_candidates_with_uncertainty(
            query_tokens,
            render_candidates,
            self.local_window_query_proj,
            self.local_window_render_proj,
            self.local_window_score_head,
            self.local_window_uncertainty_head,
        )

    def local_window_correlation_logits_uncertainty_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        query_tokens = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_candidates = _sample_local_window_descriptors(render_desc, pair_indices, render_indices)
        return _score_local_window_correlation_candidates_with_uncertainty(
            query_tokens,
            render_candidates,
            self.local_window_query_proj,
            self.local_window_render_proj,
            self.local_window_uncertainty_head,
            logit_scale=torch.exp(torch.clamp(self.local_window_logit_scale, min=-4.0, max=4.0)),
        )

    def patch_corr_fine_logits_from_maps_and_rgb(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        query_rgb_images: torch.Tensor,
        render_rgb_images: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
        *,
        query_xy: torch.Tensor,
    ) -> torch.Tensor:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        _batch, _channels, rh, rw = render_desc.shape
        query_context = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_context = _select_descriptor_rows_from_map(render_desc, pair_indices, render_indices)
        return self.patch_corr_fine_head(
            query_rgb_images,
            render_rgb_images,
            pair_indices,
            query_xy,
            render_indices,
            render_grid_hw=(int(rh), int(rw)),
            query_context=query_context,
            render_context=render_context,
        )

class RadioSpatialContextJointModel(MatchaStyleJointModel):
    """Single-RADIO mapper with explicit local-to-broad spatial context.

    This descriptor space is defined only for full feature maps. In
    particular, a landmark vector must never be reshaped to 1x1 and projected
    through this model; projected-observation banks must sample descriptors
    after this complete map forward.
    """

    def __init__(
        self,
        *args,
        context_hidden_dim: int = 256,
        context_broad_kernel_size: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        channels = int(self.input_dim)
        hidden = int(context_hidden_dim)
        broad_kernel = int(context_broad_kernel_size)
        if hidden <= 0:
            raise ValueError("context_hidden_dim must be positive")
        if broad_kernel <= 0 or broad_kernel % 2 == 0:
            raise ValueError("context_broad_kernel_size must be a positive odd integer")
        self.context_hidden_dim = hidden
        self.context_broad_kernel_size = broad_kernel
        self.context_local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.context_mid = nn.Conv2d(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.context_mix = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        # Start as the established residual mapper. Context is admitted only
        # when the retrieval objective provides a gradient for it.
        nn.init.zeros_(self.context_mix[-1].weight)
        nn.init.zeros_(self.context_mix[-1].bias)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "radio_spatial_context descriptors require a complete feature map; "
            "use forward_feature_map"
        )

    def forward_rows(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "radio_spatial_context descriptors cannot be projected row-by-row; "
            "use forward_feature_map"
        )

    def fuse_feature_map(self, feature_maps: torch.Tensor) -> torch.Tensor:
        local = self.context_local(feature_maps)
        mid = self.context_mid(feature_maps)
        radius = self.context_broad_kernel_size // 2
        broad = F.avg_pool2d(
            feature_maps,
            kernel_size=self.context_broad_kernel_size,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        context = self.context_mix(torch.cat([local, mid, broad], dim=1))
        pointwise = self.feature_fusion(feature_maps)
        return feature_maps + 0.1 * pointwise + 0.1 * context


def set_radio_spatial_context_base_trainable(
    model: nn.Module,
    *,
    trainable: bool,
) -> None:
    """Freeze/unfreeze all parameters except the spatial context branch."""

    if not isinstance(model, RadioSpatialContextJointModel):
        raise TypeError("context base freezing requires RadioSpatialContextJointModel")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(trainable) or str(name).startswith("context_"))


class _RadioDualAttentionBlock(nn.Module):
    """Small bidirectional decoder block inspired by MATCHA's joint decoder."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        heads = max(1, int(num_heads))
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        self.fine_self_norm = nn.LayerNorm(hidden)
        self.coarse_self_norm = nn.LayerNorm(hidden)
        self.fine_norm_q = nn.LayerNorm(hidden)
        self.fine_norm_kv = nn.LayerNorm(hidden)
        self.coarse_norm_q = nn.LayerNorm(hidden)
        self.coarse_norm_kv = nn.LayerNorm(hidden)
        self.fine_self = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.coarse_self = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fine_cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.coarse_cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fine_mlp = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))
        self.coarse_mlp = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))

    def forward(self, fine_tokens: torch.Tensor, coarse_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fine_update, _ = self.fine_self(
            self.fine_self_norm(fine_tokens),
            self.fine_self_norm(fine_tokens),
            self.fine_self_norm(fine_tokens),
            need_weights=False,
        )
        coarse_update, _ = self.coarse_self(
            self.coarse_self_norm(coarse_tokens),
            self.coarse_self_norm(coarse_tokens),
            self.coarse_self_norm(coarse_tokens),
            need_weights=False,
        )
        fine_tokens = fine_tokens + fine_update
        coarse_tokens = coarse_tokens + coarse_update
        fine_update, _ = self.fine_cross(
            self.fine_norm_q(fine_tokens),
            self.fine_norm_kv(coarse_tokens),
            self.fine_norm_kv(coarse_tokens),
            need_weights=False,
        )
        coarse_update, _ = self.coarse_cross(
            self.coarse_norm_q(coarse_tokens),
            self.coarse_norm_kv(fine_tokens),
            self.coarse_norm_kv(fine_tokens),
            need_weights=False,
        )
        fine_tokens = fine_tokens + fine_update
        coarse_tokens = coarse_tokens + coarse_update
        fine_tokens = fine_tokens + self.fine_mlp(fine_tokens)
        coarse_tokens = coarse_tokens + self.coarse_mlp(coarse_tokens)
        return fine_tokens, coarse_tokens


class RadioDualAttentionFusionJointModel(nn.Module):
    """RADIO-dual MATCHA-style joint model.

    Input maps are channel-concatenated as `[fine_geo, coarse_sem]`. The row
    path remains compatible with the existing sampled dual-softmax losses, while
    the full-map path uses a MATCHA-style bidirectional attention fusion before
    predicting descriptors, heatmaps and offsets.
    """

    def __init__(
        self,
        fine_input_dim: int,
        coarse_input_dim: int,
        output_dim: int = 128,
        residual_hidden_dim: int = 256,
        attention_hidden_dim: int = 256,
        attention_depth: int = 2,
        attention_heads: int = 4,
        attention_patch_size: int = 2,
        attention_upsample_mode: str = "bilinear",
        attention_fusion_mode: str = "legacy",
        group_size: int = 64,
        input_norm_mode: str = "identity",
        gate_mode: str = "residual",
        residual_gate_scale: float = 0.1,
        local_window_fine_mode: str = "mlp",
        measurement_patch_search_radius_px: float = 8.0,
        measurement_patch_context_radius_px: float = 8.0,
        measurement_patch_step_px: float = 1.0,
        measurement_patch_coarse_search_radius_px: float = 0.0,
        measurement_patch_coarse_step_px: float = 0.0,
        measurement_patch_feature_dim: int = 32,
        measurement_patch_hidden_dim: int = 64,
        measurement_patch_encoder_arch: str = "simple",
        measurement_patch_input_mode: str = "rgb",
    ) -> None:
        super().__init__()
        self.local_window_fine_mode = str(local_window_fine_mode)
        self.measurement_patch_config = {
            "search_radius_px": float(measurement_patch_search_radius_px),
            "context_radius_px": float(measurement_patch_context_radius_px),
            "step_px": float(measurement_patch_step_px),
            "coarse_search_radius_px": float(measurement_patch_coarse_search_radius_px),
            "coarse_step_px": float(measurement_patch_coarse_step_px),
            "feature_dim": int(measurement_patch_feature_dim),
            "hidden_dim": int(measurement_patch_hidden_dim),
            "encoder_arch": str(measurement_patch_encoder_arch),
            "input_mode": str(measurement_patch_input_mode),
        }
        self.fine_input_dim = int(fine_input_dim)
        self.coarse_input_dim = int(coarse_input_dim)
        self.output_dim_value = int(output_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.attention_hidden_dim = int(attention_hidden_dim)
        self.attention_depth = int(attention_depth)
        self.attention_heads = int(attention_heads)
        self.attention_patch_size = int(attention_patch_size)
        self.attention_upsample_mode = str(attention_upsample_mode)
        self.attention_fusion_mode = str(attention_fusion_mode)
        hidden = int(attention_hidden_dim)
        if self.attention_upsample_mode not in {"bilinear", "pixel_shuffle"}:
            raise ValueError("attention_upsample_mode must be 'bilinear' or 'pixel_shuffle'")
        if self.attention_fusion_mode not in {"legacy", "matcha_original"}:
            raise ValueError("attention_fusion_mode must be 'legacy' or 'matcha_original'")
        context_channels = hidden if self.attention_upsample_mode == "bilinear" else hidden // (self.attention_patch_size**2)
        if self.attention_upsample_mode == "pixel_shuffle" and hidden % (self.attention_patch_size**2) != 0:
            raise ValueError("attention_hidden_dim must be divisible by attention_patch_size^2 for pixel_shuffle")
        self.adapter = MatchaCoarseFineAdapter(
            input_dim=self.input_dim,
            output_dim=int(output_dim),
            residual_hidden_dim=int(residual_hidden_dim),
            group_size=int(group_size),
            input_norm_mode=str(input_norm_mode),
            gate_mode=str(gate_mode),
            residual_gate_scale=float(residual_gate_scale),
        )
        self.fine_proj = nn.Conv2d(self.fine_input_dim, hidden, 1)
        self.coarse_proj = nn.Conv2d(self.coarse_input_dim, hidden, 1)
        self.attention_blocks = nn.ModuleList(
            [_RadioDualAttentionBlock(hidden, int(attention_heads)) for _ in range(int(attention_depth))]
        )
        self.dec_norm_f = nn.LayerNorm(hidden)
        self.dec_norm_c = nn.LayerNorm(hidden)
        self.fusion = nn.Sequential(
            BasicConvLayer(self.fine_input_dim + self.coarse_input_dim + int(context_channels) * 2, int(residual_hidden_dim), 1, padding=0),
            BasicConvLayer(int(residual_hidden_dim), int(residual_hidden_dim), 3, padding=1),
            nn.Conv2d(int(residual_hidden_dim), int(output_dim), 1),
        )
        self.fusion_c = nn.Sequential(
            BasicConvLayer(self.coarse_input_dim + int(context_channels), int(residual_hidden_dim), 3, padding=1),
            BasicConvLayer(int(residual_hidden_dim), int(residual_hidden_dim), 3, padding=1),
            nn.Conv2d(int(residual_hidden_dim), int(output_dim), 1),
        )
        self.fusion_f = nn.Sequential(
            BasicConvLayer(self.fine_input_dim + int(context_channels), int(residual_hidden_dim), 3, padding=1),
            BasicConvLayer(int(residual_hidden_dim), int(residual_hidden_dim), 3, padding=1),
            nn.Conv2d(int(residual_hidden_dim), int(output_dim), 1),
        )
        self.offset_head_map = nn.Sequential(
            BasicConvLayer(int(output_dim), int(residual_hidden_dim), 1, padding=0),
            nn.Conv2d(int(residual_hidden_dim), 65, 1),
        )
        self.heatmap_head = nn.Sequential(
            BasicConvLayer(int(output_dim), int(residual_hidden_dim), 3, padding=1),
            BasicConvLayer(int(residual_hidden_dim), int(residual_hidden_dim), 1, padding=0),
            nn.Conv2d(int(residual_hidden_dim), 1, 1),
        )
        self.landmark_dustbin_head = nn.Sequential(
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), 1),
        )
        nn.init.zeros_(self.landmark_dustbin_head[-1].weight)
        nn.init.constant_(self.landmark_dustbin_head[-1].bias, 0.7)
        self.rgb_keypoint_detector = MatchaRgbKeypointDetector()
        self.original_fine_matcher = _OriginalMatchaFineMatcher(
            descriptor_dim=int(output_dim),
            hidden_dim=int(residual_hidden_dim),
            output_bins=64,
        )
        self.query_original_fine_matcher = _OriginalMatchaFineMatcher(
            descriptor_dim=int(output_dim),
            hidden_dim=int(residual_hidden_dim),
            output_bins=64,
        )
        self.local_window_query_proj = nn.Linear(int(output_dim), int(residual_hidden_dim))
        self.local_window_render_proj = nn.Linear(int(output_dim), int(residual_hidden_dim))
        self.local_window_score_head = nn.Sequential(
            nn.LayerNorm(int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.local_window_uncertainty_head = nn.Sequential(
            nn.LayerNorm(int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.local_window_logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.patch_corr_fine_head = PatchCorrelationFineHead(
            context_dim=int(output_dim),
            hidden_dim=int(residual_hidden_dim),
            patch_size=32,
            offset_bins=8,
        )
        coarse_radius = float(measurement_patch_coarse_search_radius_px)
        coarse_step = float(measurement_patch_coarse_step_px)
        self.measurement_patch_branch = RGBPatchMeasurementBranch(
            search_radius_px=float(measurement_patch_search_radius_px),
            context_radius_px=float(measurement_patch_context_radius_px),
            step_px=float(measurement_patch_step_px),
            coarse_search_radius_px=None if coarse_radius <= 0.0 else coarse_radius,
            coarse_step_px=None if coarse_step <= 0.0 else coarse_step,
            feature_dim=int(measurement_patch_feature_dim),
            hidden_dim=int(measurement_patch_hidden_dim),
            encoder_arch=str(measurement_patch_encoder_arch),
            input_mode=str(measurement_patch_input_mode),
        )

    @property
    def input_dim(self) -> int:
        return int(self.fine_input_dim + self.coarse_input_dim)

    @property
    def output_dim(self) -> int:
        return int(self.output_dim_value)

    def _split_map(self, feature_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_maps.ndim != 4 or int(feature_maps.shape[1]) != self.input_dim:
            raise ValueError("feature_maps must have shape (B, fine_input_dim + coarse_input_dim, H, W)")
        fine = feature_maps[:, : self.fine_input_dim]
        coarse = feature_maps[:, self.fine_input_dim :]
        return fine, coarse

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter.encode(features)

    def landmark_dustbin_logits(self, query_descriptors: torch.Tensor) -> torch.Tensor:
        return self.landmark_dustbin_head(query_descriptors).reshape(-1)

    def forward_rows(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(features)

    def _attention_contexts(self, feature_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fine, coarse = self._split_map(feature_maps)
        batch, _channels, height, width = fine.shape
        patch = max(1, int(self.attention_patch_size))
        if patch > 1:
            fine_attention_input = F.avg_pool2d(fine, kernel_size=patch, stride=patch, ceil_mode=True)
            coarse_attention_input = F.avg_pool2d(coarse, kernel_size=patch, stride=patch, ceil_mode=True)
        else:
            fine_attention_input = fine
            coarse_attention_input = coarse
        _batch, _channels, attention_height, attention_width = fine_attention_input.shape
        fine_tokens = self.fine_proj(fine_attention_input).flatten(2).transpose(1, 2)
        coarse_tokens = self.coarse_proj(coarse_attention_input).flatten(2).transpose(1, 2)
        for block in self.attention_blocks:
            fine_tokens, coarse_tokens = block(fine_tokens, coarse_tokens)
        hidden = int(self.attention_hidden_dim)
        fine_att = self.dec_norm_f(fine_tokens).transpose(1, 2).reshape(batch, hidden, attention_height, attention_width)
        coarse_att = self.dec_norm_c(coarse_tokens).transpose(1, 2).reshape(batch, hidden, attention_height, attention_width)

        def restore(attention_map: torch.Tensor) -> torch.Tensor:
            if self.attention_upsample_mode == "pixel_shuffle" and patch > 1:
                restored = F.pixel_shuffle(attention_map, patch)
                if int(restored.shape[2]) >= height and int(restored.shape[3]) >= width:
                    return restored[:, :, :height, :width]
                return F.interpolate(restored, size=(height, width), mode="bilinear", align_corners=False)
            if tuple(attention_map.shape[-2:]) != (height, width):
                return F.interpolate(attention_map, size=(height, width), mode="bilinear", align_corners=False)
            return attention_map

        fine_att = restore(fine_att)
        coarse_att = restore(coarse_att)
        return fine, coarse, fine_att, coarse_att

    def forward_fuse_feature(self, feature_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine, coarse, fine_att, coarse_att = self._attention_contexts(feature_maps)
        if self.attention_fusion_mode == "matcha_original":
            coarse_descriptors = F.normalize(self.fusion_c(torch.cat([coarse, coarse_att], dim=1)), dim=1)
            fine_descriptors = F.normalize(self.fusion_f(torch.cat([fine, fine_att], dim=1)), dim=1)
            heatmap_logits = self.heatmap_head(fine_descriptors)
            return coarse_descriptors, fine_descriptors, heatmap_logits
        descriptor_map = F.normalize(self.fusion(torch.cat([fine, coarse, fine_att, coarse_att], dim=1)), dim=1)
        heatmap_logits = self.heatmap_head(descriptor_map)
        return descriptor_map, descriptor_map, heatmap_logits

    def forward_feature_map(self, feature_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_descriptors, fine_descriptors, heatmap_logits = self.forward_fuse_feature(feature_maps)
        descriptor_map = fine_descriptors
        if self.attention_fusion_mode != "matcha_original":
            descriptor_map = coarse_descriptors
        offset_logits = self.offset_head_map(descriptor_map)
        return descriptor_map, heatmap_logits, offset_logits

    def forward_rgb_keypoints(self, images: torch.Tensor) -> torch.Tensor:
        return self.rgb_keypoint_detector(images)

    def pair_confidence_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_confidence_logits(query_descriptors, render_descriptors)

    def pair_fine_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.original_fine_matcher(query_descriptors, render_descriptors)

    def query_pair_fine_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.query_original_fine_matcher(query_descriptors, render_descriptors)

    def pair_fine_uncertainty_log_sigma(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_fine_uncertainty_log_sigma(query_descriptors, render_descriptors)

    def query_pair_fine_uncertainty_log_sigma(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.adapter.pair_fine_uncertainty_log_sigma(render_descriptors, query_descriptors)

    def local_fine_logits_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        batch, channels, qh, qw = query_desc.shape
        _rb, _rc, rh, rw = render_desc.shape
        query_rows = query_desc.permute(0, 2, 3, 1).reshape(batch, qh * qw, channels)
        render_rows = render_desc.permute(0, 2, 3, 1).reshape(batch, rh * rw, channels)
        pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
        qidx = query_indices.long().reshape(-1).clamp(0, qh * qw - 1)
        ridx = render_indices.long().reshape(-1).clamp(0, rh * rw - 1)
        return self.pair_fine_logits(query_rows[pairs, qidx], render_rows[pairs, ridx])

    def local_window_fine_logits_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        query_tokens = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_candidates = _sample_local_window_descriptors(render_desc, pair_indices, render_indices)
        if str(self.local_window_fine_mode) == "correlation":
            logits, _sigma = _score_local_window_correlation_candidates_with_uncertainty(
                query_tokens,
                render_candidates,
                self.local_window_query_proj,
                self.local_window_render_proj,
                self.local_window_uncertainty_head,
                logit_scale=torch.exp(torch.clamp(self.local_window_logit_scale, min=-4.0, max=4.0)),
            )
            return logits
        return _score_local_window_candidates(
            query_tokens,
            render_candidates,
            self.local_window_query_proj,
            self.local_window_render_proj,
            self.local_window_score_head,
        )

    def local_window_fine_logits_uncertainty_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        query_tokens = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_candidates = _sample_local_window_descriptors(render_desc, pair_indices, render_indices)
        if str(self.local_window_fine_mode) == "correlation":
            return _score_local_window_correlation_candidates_with_uncertainty(
                query_tokens,
                render_candidates,
                self.local_window_query_proj,
                self.local_window_render_proj,
                self.local_window_uncertainty_head,
                logit_scale=torch.exp(torch.clamp(self.local_window_logit_scale, min=-4.0, max=4.0)),
            )
        return _score_local_window_candidates_with_uncertainty(
            query_tokens,
            render_candidates,
            self.local_window_query_proj,
            self.local_window_render_proj,
            self.local_window_score_head,
            self.local_window_uncertainty_head,
        )

    def local_window_correlation_logits_uncertainty_from_maps(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        query_tokens = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_candidates = _sample_local_window_descriptors(render_desc, pair_indices, render_indices)
        return _score_local_window_correlation_candidates_with_uncertainty(
            query_tokens,
            render_candidates,
            self.local_window_query_proj,
            self.local_window_render_proj,
            self.local_window_uncertainty_head,
            logit_scale=torch.exp(torch.clamp(self.local_window_logit_scale, min=-4.0, max=4.0)),
        )

    def patch_corr_fine_logits_from_maps_and_rgb(
        self,
        query_feature_maps: torch.Tensor,
        render_feature_maps: torch.Tensor,
        query_rgb_images: torch.Tensor,
        render_rgb_images: torch.Tensor,
        pair_indices: torch.Tensor,
        query_indices: torch.Tensor,
        render_indices: torch.Tensor,
        *,
        query_xy: torch.Tensor,
    ) -> torch.Tensor:
        query_desc, _query_heat, _query_offset = self.forward_feature_map(query_feature_maps)
        render_desc, _render_heat, _render_offset = self.forward_feature_map(render_feature_maps)
        _batch, _channels, rh, rw = render_desc.shape
        query_context = _select_descriptor_rows_from_map(query_desc, pair_indices, query_indices)
        render_context = _select_descriptor_rows_from_map(render_desc, pair_indices, render_indices)
        return self.patch_corr_fine_head(
            query_rgb_images,
            render_rgb_images,
            pair_indices,
            query_xy,
            render_indices,
            render_grid_hw=(int(rh), int(rw)),
            query_context=query_context,
            render_context=render_context,
        )


def _tensor(array: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def _sample_indices(rng: np.random.Generator, count: int, batch_size: int) -> np.ndarray:
    take = min(int(batch_size), int(count))
    return rng.choice(int(count), size=take, replace=False)


def _sample_map_pair_subset(
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    max_pairs: int,
    seed: int,
) -> np.ndarray | None:
    if samples.sample_pair_indices is None:
        if samples.query_feature_maps is None:
            return None
        count = int(samples.query_feature_maps.shape[0])
        pairs = np.arange(count, dtype=np.int64)
    else:
        pairs = np.unique(np.asarray(samples.sample_pair_indices, dtype=np.int64)[indices])
    if pairs.size <= int(max_pairs):
        return np.sort(pairs.astype(np.int64))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(pairs, size=int(max_pairs), replace=False).astype(np.int64))


def _filter_indices_to_pair_subset(
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    pair_subset: np.ndarray | None,
) -> np.ndarray:
    if pair_subset is None or samples.sample_pair_indices is None:
        return indices
    pair_values = np.asarray(samples.sample_pair_indices, dtype=np.int64)[indices]
    keep = np.isin(pair_values, np.asarray(pair_subset, dtype=np.int64))
    return indices[keep]


def _filter_indices_to_positive_matches(samples: MatchaJointTrainingSet, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return values
    keep = np.ones((values.shape[0],), dtype=bool)
    if samples.sample_ignore_mask is not None:
        keep &= ~np.asarray(samples.sample_ignore_mask, dtype=bool)[values]
    if samples.sample_no_match_labels is not None:
        keep &= np.asarray(samples.sample_no_match_labels, dtype=np.int64)[values] == 0
    return values[keep]


def _filter_indices_to_no_matches(samples: MatchaJointTrainingSet, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0 or samples.sample_no_match_labels is None:
        return values[:0]
    keep = np.asarray(samples.sample_no_match_labels, dtype=np.int64)[values] != 0
    if samples.sample_ignore_mask is not None:
        keep &= ~np.asarray(samples.sample_ignore_mask, dtype=bool)[values]
    return values[keep]


def _remap_pair_indices(global_pairs: np.ndarray, pair_subset: np.ndarray | None) -> np.ndarray:
    pairs = np.asarray(global_pairs, dtype=np.int64).reshape(-1)
    if pair_subset is None:
        return pairs
    lookup = {int(pair): int(local) for local, pair in enumerate(np.asarray(pair_subset, dtype=np.int64).reshape(-1))}
    return np.asarray([lookup[int(pair)] for pair in pairs], dtype=np.int64)


def _has_full_map_correspondence_supervision(samples: MatchaJointTrainingSet) -> bool:
    return (
        samples.query_feature_maps is not None
        and samples.render_feature_maps is not None
        and samples.sample_pair_indices is not None
        and samples.query_cell_indices is not None
        and samples.render_cell_indices is not None
    )


def _offset_distribution_loss(logits: torch.Tensor, labels: torch.Tensor, soft_targets: torch.Tensor | None) -> torch.Tensor:
    labels = labels.long().reshape(-1)
    if soft_targets is None:
        return F.cross_entropy(logits, labels)
    if soft_targets.shape != (int(logits.shape[0]), int(logits.shape[1])):
        raise ValueError("offset soft_targets must match logits shape")
    target = soft_targets.float()
    target_sum = torch.sum(target, dim=1, keepdim=True).clamp_min(1e-8)
    target = target / target_sum
    return -torch.sum(target * F.log_softmax(logits, dim=1), dim=1).mean()


def _coarse_fine_loss(
    model: MatchaStyleJointModel,
    samples: MatchaCoarseFineTrainingSet,
    indices: np.ndarray,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    confidence_ignore_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query = _tensor(samples.query_features[indices], dtype=torch.float32, device=device)
    render = _tensor(samples.render_features[indices], dtype=torch.float32, device=device)
    qlabels = _tensor(samples.query_offset_labels[indices], dtype=torch.long, device=device)
    rlabels = _tensor(samples.render_offset_labels[indices], dtype=torch.long, device=device)
    qsoft = (
        None
        if getattr(samples, "query_offset_soft_labels", None) is None
        else _tensor(samples.query_offset_soft_labels[indices], dtype=torch.float32, device=device)
    )
    rsoft = (
        None
        if getattr(samples, "render_offset_soft_labels", None) is None
        else _tensor(samples.render_offset_soft_labels[indices], dtype=torch.float32, device=device)
    )
    confidence_targets = (
        None
        if getattr(samples, "sample_confidence_targets", None) is None
        else _tensor(samples.sample_confidence_targets[indices], dtype=torch.float32, device=device)
    )
    confidence_keep = None
    if confidence_ignore_mask is not None:
        ignore = _tensor(np.asarray(confidence_ignore_mask, dtype=bool).reshape(-1), dtype=torch.bool, device=device)
        if int(ignore.shape[0]) != int(len(indices)):
            raise ValueError("confidence_ignore_mask must contain one value per selected sample")
        confidence_keep = ~ignore
    negatives = _tensor(samples.negative_render_features[indices], dtype=torch.float32, device=device)

    query_z, query_offsets = model.forward_rows(query)
    render_z, render_offsets = model.forward_rows(render)
    loss = query.new_tensor(0.0)
    descriptor_loss, match_confidence = _dual_softmax_descriptor_loss_and_confidence(
        query_z,
        render_z,
        float(config.temperature),
    )
    if float(config.dual_softmax_weight) > 0.0:
        loss = loss + float(config.dual_softmax_weight) * descriptor_loss
    if float(config.offset_loss_weight) > 0.0:
        offset_loss = 0.5 * (
            _offset_distribution_loss(query_offsets, qlabels, qsoft)
            + _offset_distribution_loss(render_offsets, rlabels, rsoft)
        )
        loss = loss + float(config.offset_loss_weight) * offset_loss
    fine_metrics: dict[str, float] = {}
    if float(config.pair_fine_loss_weight) > 0.0:
        pair_fine = model.pair_fine_logits(query_z, render_z)
        pair_sigma = (
            model.pair_fine_uncertainty_log_sigma(query_z, render_z)
            if float(config.fine_uncertainty_loss_weight) > 0.0
            and hasattr(model, "pair_fine_uncertainty_log_sigma")
            else None
        )
        pair_fine_loss, item_metrics = _fine_coordinate_loss_and_metrics(
            pair_fine,
            rlabels,
            soft_targets=rsoft,
            confidence=match_confidence,
            continuous_loss_weight=float(config.fine_continuous_loss_weight),
            uncertainty_log_sigma=pair_sigma,
            uncertainty_loss_weight=float(config.fine_uncertainty_loss_weight),
            loss_mode=str(config.fine_loss_mode),
        )
        fine_metrics.update({f"render_pair_fine_{key}": value for key, value in item_metrics.items()})
        if pair_fine_loss is not None:
            loss = loss + float(config.pair_fine_loss_weight) * pair_fine_loss
    if float(config.query_pair_fine_loss_weight) > 0.0:
        query_pair_fine = model.query_pair_fine_logits(query_z, render_z)
        query_pair_sigma = (
            model.query_pair_fine_uncertainty_log_sigma(query_z, render_z)
            if float(config.fine_uncertainty_loss_weight) > 0.0
            and hasattr(model, "query_pair_fine_uncertainty_log_sigma")
            else None
        )
        query_pair_fine_loss, item_metrics = _fine_coordinate_loss_and_metrics(
            query_pair_fine,
            qlabels,
            soft_targets=qsoft,
            confidence=match_confidence,
            continuous_loss_weight=float(config.fine_continuous_loss_weight),
            uncertainty_log_sigma=query_pair_sigma,
            uncertainty_loss_weight=float(config.fine_uncertainty_loss_weight),
            loss_mode=str(config.fine_loss_mode),
        )
        fine_metrics.update({f"query_pair_fine_{key}": value for key, value in item_metrics.items()})
        if query_pair_fine_loss is not None:
            loss = loss + float(config.query_pair_fine_loss_weight) * query_pair_fine_loss
    negative_z = None
    needs_negative_descriptors = (
        float(config.pair_confidence_loss_weight) > 0.0
        or float(config.hard_negative_weight) > 0.0
        or float(config.coarse_candidate_rank_loss_weight) > 0.0
    )
    if needs_negative_descriptors and negatives.numel() > 0:
        negative_z = model.encode(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
    if float(config.pair_confidence_loss_weight) > 0.0 and negative_z is not None:
        pos_logits = model.pair_confidence_logits(query_z, render_z)
        neg_logits = model.pair_confidence_logits(
            query_z[:, None, :].expand_as(negative_z).reshape(-1, query_z.shape[-1]),
            negative_z.reshape(-1, query_z.shape[-1]),
        )
        pos_targets = torch.ones_like(pos_logits) if confidence_targets is None else torch.clamp(confidence_targets, 0.0, 1.0)
        if confidence_keep is not None:
            if torch.any(confidence_keep):
                confidence_loss = F.binary_cross_entropy_with_logits(pos_logits[confidence_keep], pos_targets[confidence_keep])
            else:
                confidence_loss = pos_logits.new_tensor(0.0)
        else:
            confidence_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_targets)
        confidence_loss = confidence_loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = loss + float(config.pair_confidence_loss_weight) * confidence_loss
    if float(config.hard_negative_weight) > 0.0 and negative_z is not None:
        pos_scores = torch.sum(query_z * render_z, dim=1, keepdim=True)
        neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
        hard_loss = torch.relu(neg_scores - pos_scores + float(config.hard_negative_margin)).mean()
        loss = loss + float(config.hard_negative_weight) * hard_loss
    if float(config.coarse_candidate_rank_loss_weight) > 0.0 and negative_z is not None:
        rank_loss, rank_metrics = _coarse_candidate_rank_loss_and_metrics(
            query_z,
            render_z,
            negative_z,
            margin=float(config.coarse_candidate_rank_margin),
        )
        loss = loss + float(config.coarse_candidate_rank_loss_weight) * rank_loss
    else:
        rank_metrics = {}
    state = {
        "query_z": query_z.detach(),
        "render_z": render_z.detach(),
        "query_offsets": query_offsets.detach(),
        "render_offsets": render_offsets.detach(),
        "query_labels": qlabels.detach(),
        "render_labels": rlabels.detach(),
        "match_confidence": match_confidence.detach(),
    }
    state.update({key: query_z.new_tensor(float(value)) for key, value in fine_metrics.items()})
    state.update(rank_metrics)
    return loss, state


def _coarse_candidate_rank_loss_and_metrics(
    query_z: torch.Tensor,
    render_z: torch.Tensor,
    negative_z: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos_scores = torch.sum(query_z * render_z, dim=1, keepdim=True)
    neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
    hardest_neg_scores = torch.max(neg_scores, dim=1, keepdim=True).values
    rank_loss = torch.relu(hardest_neg_scores - pos_scores + float(margin)).mean()
    return rank_loss, {
        "coarse_candidate_rank_loss": rank_loss.detach(),
        "coarse_candidate_rank_top1_acc": (pos_scores > hardest_neg_scores).float().mean().detach(),
        "coarse_candidate_rank_positive_score_mean": pos_scores.mean().detach(),
        "coarse_candidate_rank_hard_negative_score_mean": hardest_neg_scores.mean().detach(),
    }


def _heatmap_loss(
    model: MatchaStyleJointModel,
    feature_maps: np.ndarray | None,
    targets: np.ndarray | None,
    *,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
) -> torch.Tensor | None:
    if feature_maps is None or targets is None:
        return None
    if pair_subset is not None:
        feature_maps = np.asarray(feature_maps)[pair_subset]
        targets = np.asarray(targets)[pair_subset]
    maps = _tensor(feature_maps, dtype=torch.float32, device=device)
    target = _tensor(targets, dtype=torch.float32, device=device)
    _desc, logits, _offsets = model.forward_feature_map(maps)
    return F.l1_loss(torch.sigmoid(logits[:, 0]), target)


def _no_match_confidence_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    no_match_indices = _filter_indices_to_no_matches(samples, indices)
    if no_match_indices.size and samples.sample_confidence_ignore_mask is not None:
        confidence_ignore = np.asarray(samples.sample_confidence_ignore_mask, dtype=bool)[no_match_indices]
        no_match_indices = no_match_indices[~confidence_ignore]
    if no_match_indices.size == 0:
        return None
    base = samples.coarse_fine_samples
    query = _tensor(base.query_features[no_match_indices], dtype=torch.float32, device=device)
    render = _tensor(base.render_features[no_match_indices], dtype=torch.float32, device=device)
    query_z = model.encode(query)
    render_z = model.encode(render)
    logits = model.pair_confidence_logits(query_z, render_z)
    targets = torch.zeros_like(logits)
    if getattr(base, "sample_confidence_targets", None) is not None:
        targets = _tensor(
            np.asarray(base.sample_confidence_targets, dtype=np.float32)[no_match_indices],
            dtype=torch.float32,
            device=device,
        )
        targets = torch.clamp(targets, 0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, targets)


def _repeatability_loss(
    model: MatchaStyleJointModel,
    images: np.ndarray | None,
    targets: np.ndarray | None,
    *,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
) -> torch.Tensor | None:
    if images is None or targets is None:
        return None
    if pair_subset is not None:
        images = np.asarray(images)[pair_subset]
        targets = np.asarray(targets)[pair_subset]
    image_tensor = _tensor(images, dtype=torch.float32, device=device)
    target = _tensor(targets, dtype=torch.float32, device=device)
    logits = model.forward_rgb_keypoints(image_tensor)
    keypoint_prob = 1.0 - torch.softmax(logits, dim=1)[:, 64]
    return F.binary_cross_entropy(torch.clamp(keypoint_prob, 1e-6, 1.0 - 1e-6), torch.clamp(target, 0.0, 1.0))


def _select_map_rows(
    descriptor_map: torch.Tensor,
    offset_map: torch.Tensor,
    pair_indices: torch.Tensor,
    cell_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = descriptor_map.shape
    pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
    idx = cell_indices.long().reshape(-1).clamp(0, height * width - 1)
    desc_rows = descriptor_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    offset_rows = offset_map.permute(0, 2, 3, 1).reshape(batch, height * width, 65)
    return desc_rows[pairs, idx], offset_rows[pairs, idx]


def _select_descriptor_rows_from_map(
    descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    cell_indices: torch.Tensor,
) -> torch.Tensor:
    if descriptor_map.ndim != 4:
        raise ValueError("descriptor_map must have shape (B, C, H, W)")
    batch, channels, height, width = descriptor_map.shape
    pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
    idx = cell_indices.long().reshape(-1).clamp(0, height * width - 1)
    if pairs.numel() != idx.numel():
        raise ValueError("pair_indices and cell_indices must have the same length")
    rows = descriptor_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    return rows[pairs, idx]


def _select_negative_descriptor_rows_from_map(
    descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    cell_indices: torch.Tensor,
) -> torch.Tensor:
    if descriptor_map.ndim != 4:
        raise ValueError("descriptor_map must have shape (B, C, H, W)")
    cells = cell_indices.long()
    if cells.ndim != 2:
        raise ValueError("cell_indices must have shape (N, K)")
    batch, channels, height, width = descriptor_map.shape
    pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
    if pairs.numel() != cells.shape[0]:
        raise ValueError("pair_indices must contain one value per cell-index row")
    idx = cells.clamp(0, height * width - 1)
    rows = descriptor_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    return rows[pairs[:, None], idx]


def _sample_descriptor_rows_at_image_xy(
    descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    xy: torch.Tensor,
    *,
    image_width: int | torch.Tensor,
    image_height: int | torch.Tensor,
) -> torch.Tensor:
    """Match projected-observation bank sampling with differentiable bilinear sampling."""

    if descriptor_map.ndim != 4:
        raise ValueError("descriptor_map must have shape (B, C, H, W)")
    pairs = pair_indices.long().reshape(-1)
    coordinates = xy.to(device=descriptor_map.device, dtype=descriptor_map.dtype).reshape(-1, 2)
    if pairs.shape[0] != coordinates.shape[0]:
        raise ValueError("pair_indices and xy must contain the same number of observations")
    if pairs.numel() == 0:
        return descriptor_map.new_zeros((0, int(descriptor_map.shape[1])))
    if torch.any(pairs < 0) or torch.any(pairs >= int(descriptor_map.shape[0])):
        raise ValueError("pair_indices contains an out-of-range map index")
    widths = torch.as_tensor(image_width, dtype=descriptor_map.dtype, device=descriptor_map.device).reshape(-1)
    heights = torch.as_tensor(image_height, dtype=descriptor_map.dtype, device=descriptor_map.device).reshape(-1)
    if widths.numel() == 1:
        widths = widths.expand(coordinates.shape[0])
    if heights.numel() == 1:
        heights = heights.expand(coordinates.shape[0])
    if widths.shape[0] != coordinates.shape[0] or heights.shape[0] != coordinates.shape[0]:
        raise ValueError("image dimensions must be scalar or contain one value per observation")
    if torch.any(widths <= 0) or torch.any(heights <= 0):
        raise ValueError("image dimensions must be positive")
    x = 2.0 * coordinates[:, 0] / (widths - 1.0).clamp_min(1.0) - 1.0
    y = 2.0 * coordinates[:, 1] / (heights - 1.0).clamp_min(1.0) - 1.0
    normalized_xy = torch.stack([x, y], dim=1)
    sampled_rows = descriptor_map.new_empty((int(pairs.shape[0]), int(descriptor_map.shape[1])))
    for pair in torch.unique(pairs, sorted=True):
        row_indices = torch.nonzero(pairs == pair, as_tuple=False).reshape(-1)
        grid = normalized_xy[row_indices].reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            descriptor_map[int(pair.item()) : int(pair.item()) + 1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_rows[row_indices] = sampled[0, :, :, 0].T
    return sampled_rows


def _observation_query_group_ids(
    pair_indices: torch.Tensor,
    xy: torch.Tensor,
    *,
    image_width: int | torch.Tensor,
    image_height: int | torch.Tensor,
    grid_width: int,
    grid_height: int,
    image_group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    coordinates = xy.to(device=pair_indices.device, dtype=torch.float32).reshape(-1, 2)
    widths = torch.as_tensor(image_width, dtype=torch.float32, device=pair_indices.device).reshape(-1)
    heights = torch.as_tensor(image_height, dtype=torch.float32, device=pair_indices.device).reshape(-1)
    if widths.numel() == 1:
        widths = widths.expand(coordinates.shape[0])
    if heights.numel() == 1:
        heights = heights.expand(coordinates.shape[0])
    if widths.shape[0] != coordinates.shape[0] or heights.shape[0] != coordinates.shape[0]:
        raise ValueError("image dimensions must be scalar or contain one value per observation")
    col = torch.floor(coordinates[:, 0] / widths.clamp_min(1.0) * float(grid_width)).long()
    row = torch.floor(coordinates[:, 1] / heights.clamp_min(1.0) * float(grid_height)).long()
    col = col.clamp(0, max(int(grid_width) - 1, 0))
    row = row.clamp(0, max(int(grid_height) - 1, 0))
    image_groups = pair_indices.long().reshape(-1) if image_group_ids is None else image_group_ids.long().reshape(-1)
    if image_groups.shape[0] != coordinates.shape[0]:
        raise ValueError("image_group_ids must contain one value per observation")
    return image_groups * int(grid_width * grid_height) + row * int(grid_width) + col


def _sample_landmark_dustbin_descriptors(
    descriptor_map: torch.Tensor,
    *,
    query_heatmap_targets: np.ndarray | None,
    pair_query_group_ids: torch.Tensor,
    positive_pair_indices: torch.Tensor,
    positive_xy: torch.Tensor,
    positive_image_width: torch.Tensor,
    positive_image_height: torch.Tensor,
    samples_per_image: int,
    exclusion_radius_cells: int,
    max_heatmap_target: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sample no-reliable-track cells once per unique query image."""

    if descriptor_map.ndim != 4:
        raise ValueError("descriptor_map must have shape (B, C, H, W)")
    batch, channels, height, width = descriptor_map.shape
    pair_groups = pair_query_group_ids.to(device=descriptor_map.device, dtype=torch.long).reshape(-1)
    if pair_groups.shape[0] != int(batch):
        raise ValueError("pair_query_group_ids must contain one value per descriptor map")
    if int(samples_per_image) <= 0:
        return descriptor_map.new_zeros((0, int(channels))), {
            "landmark_retrieval_dustbin_candidate_count": 0.0,
            "landmark_retrieval_dustbin_sampled_image_count": 0.0,
        }
    positive_pairs = positive_pair_indices.to(device=descriptor_map.device, dtype=torch.long).reshape(-1)
    coordinates = positive_xy.to(device=descriptor_map.device, dtype=torch.float32).reshape(-1, 2)
    widths = positive_image_width.to(device=descriptor_map.device, dtype=torch.float32).reshape(-1)
    heights = positive_image_height.to(device=descriptor_map.device, dtype=torch.float32).reshape(-1)
    if not (
        positive_pairs.shape[0]
        == coordinates.shape[0]
        == widths.shape[0]
        == heights.shape[0]
    ):
        raise ValueError("positive observation arrays must have the same length")
    positive_cols = torch.floor(coordinates[:, 0] / widths.clamp_min(1.0) * float(width)).long()
    positive_rows = torch.floor(coordinates[:, 1] / heights.clamp_min(1.0) * float(height)).long()
    positive_cols = positive_cols.clamp(0, max(int(width) - 1, 0))
    positive_rows = positive_rows.clamp(0, max(int(height) - 1, 0))
    positive_groups = pair_groups[positive_pairs]
    heatmap = None
    if query_heatmap_targets is not None:
        heatmap = torch.as_tensor(
            np.asarray(query_heatmap_targets, dtype=np.float32),
            dtype=torch.float32,
            device=descriptor_map.device,
        )
        if heatmap.shape != (int(batch), int(height), int(width)):
            raise ValueError("query heatmap targets must match descriptor map shape")

    rows = descriptor_map.permute(0, 2, 3, 1).reshape(int(batch), int(height * width), int(channels))
    selected: list[torch.Tensor] = []
    candidate_count = 0
    sampled_image_count = 0
    radius = int(exclusion_radius_cells)
    for group in torch.unique(pair_groups, sorted=True).detach().cpu().tolist():
        pair_indices = torch.nonzero(pair_groups == int(group), as_tuple=False).reshape(-1)
        if pair_indices.numel() == 0:
            continue
        available = torch.ones((int(height), int(width)), dtype=torch.bool, device=descriptor_map.device)
        if heatmap is not None:
            group_heatmap = torch.amax(heatmap[pair_indices], dim=0)
            available &= group_heatmap <= float(max_heatmap_target)
        group_positive = torch.nonzero(positive_groups == int(group), as_tuple=False).reshape(-1)
        for positive_index in group_positive.detach().cpu().tolist():
            row = int(positive_rows[int(positive_index)].item())
            col = int(positive_cols[int(positive_index)].item())
            available[
                max(0, row - radius) : min(int(height), row + radius + 1),
                max(0, col - radius) : min(int(width), col + radius + 1),
            ] = False
        candidates = torch.nonzero(available.reshape(-1), as_tuple=False).reshape(-1)
        candidate_count += int(candidates.numel())
        if candidates.numel() == 0:
            continue
        take = min(int(samples_per_image), int(candidates.numel()))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 104729 * (int(group) + 1))
        order = torch.randperm(int(candidates.numel()), generator=generator)[:take].to(candidates.device)
        selected.append(rows[int(pair_indices[0].item()), candidates[order]])
        sampled_image_count += 1
    output = descriptor_map.new_zeros((0, int(channels))) if not selected else torch.cat(selected, dim=0)
    return output, {
        "landmark_retrieval_dustbin_candidate_count": float(candidate_count),
        "landmark_retrieval_dustbin_sampled_image_count": float(sampled_image_count),
    }


def _forward_coarse_and_fine_feature_maps(
    model: MatchaStyleJointModel,
    feature_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        isinstance(model, RadioDualAttentionFusionJointModel)
        and str(model.attention_fusion_mode) == "matcha_original"
    ):
        coarse_desc, fine_desc, heatmap_logits = model.forward_fuse_feature(feature_maps)
        offset_logits = model.offset_head_map(fine_desc)
        return coarse_desc, fine_desc, heatmap_logits, offset_logits
    desc, heatmap_logits, offset_logits = model.forward_feature_map(feature_maps)
    return desc, desc, heatmap_logits, offset_logits


def _sample_local_window_descriptors(
    descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    cell_indices: torch.Tensor,
    *,
    bins: int = 8,
) -> torch.Tensor:
    if descriptor_map.ndim != 4:
        raise ValueError("descriptor_map must have shape (B, C, H, W)")
    bin_count = int(bins)
    if bin_count <= 0:
        raise ValueError("bins must be positive")
    batch, channels, height, width = descriptor_map.shape
    pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
    idx = cell_indices.long().reshape(-1).clamp(0, height * width - 1)
    if pairs.numel() != idx.numel():
        raise ValueError("pair_indices and cell_indices must have the same length")
    rows = torch.div(idx, int(width), rounding_mode="floor").to(dtype=torch.float32)
    cols = (idx % int(width)).to(dtype=torch.float32)
    bins_1d = torch.arange(bin_count * bin_count, dtype=torch.float32, device=descriptor_map.device)
    bin_x = torch.remainder(bins_1d, bin_count)
    bin_y = torch.div(bins_1d, bin_count, rounding_mode="floor")
    x = cols[:, None] + ((bin_x[None, :] + 0.5) / float(bin_count) - 0.5)
    y = rows[:, None] + ((bin_y[None, :] + 0.5) / float(bin_count) - 0.5)
    x = torch.clamp(x, min=0.0, max=float(width - 1))
    y = torch.clamp(y, min=0.0, max=float(height - 1))
    x0 = torch.floor(x).to(dtype=torch.long)
    y0 = torch.floor(y).to(dtype=torch.long)
    x1 = torch.clamp(x0 + 1, max=int(width - 1))
    y1 = torch.clamp(y0 + 1, max=int(height - 1))
    wx = (x - x0.to(dtype=torch.float32)).to(dtype=descriptor_map.dtype)
    wy = (y - y0.to(dtype=torch.float32)).to(dtype=descriptor_map.dtype)
    descriptor_hwc = descriptor_map.permute(0, 2, 3, 1).contiguous()
    pair_grid = pairs[:, None]
    v00 = descriptor_hwc[pair_grid, y0, x0]
    v01 = descriptor_hwc[pair_grid, y0, x1]
    v10 = descriptor_hwc[pair_grid, y1, x0]
    v11 = descriptor_hwc[pair_grid, y1, x1]
    top = v00 * (1.0 - wx[..., None]) + v01 * wx[..., None]
    bottom = v10 * (1.0 - wx[..., None]) + v11 * wx[..., None]
    return (top * (1.0 - wy[..., None]) + bottom * wy[..., None]).contiguous().reshape(
        int(pairs.numel()),
        bin_count * bin_count,
        channels,
    )


def _score_local_window_candidates(
    query_descriptors: torch.Tensor,
    render_candidates: torch.Tensor,
    query_proj: nn.Module,
    render_proj: nn.Module,
    score_head: nn.Module,
) -> torch.Tensor:
    if query_descriptors.ndim != 2:
        raise ValueError("query_descriptors must have shape (N, C)")
    if render_candidates.ndim != 3:
        raise ValueError("render_candidates must have shape (N, K, C)")
    if int(render_candidates.shape[0]) != int(query_descriptors.shape[0]):
        raise ValueError("query_descriptors and render_candidates must have the same batch size")
    query = F.normalize(query_descriptors, dim=1)
    render = F.normalize(render_candidates, dim=2)
    hidden = torch.tanh(query_proj(query)[:, None, :] + render_proj(render))
    return score_head(hidden).squeeze(-1)


def _score_local_window_candidates_with_uncertainty(
    query_descriptors: torch.Tensor,
    render_candidates: torch.Tensor,
    query_proj: nn.Module,
    render_proj: nn.Module,
    score_head: nn.Module,
    uncertainty_head: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_descriptors.ndim != 2:
        raise ValueError("query_descriptors must have shape (N, C)")
    if render_candidates.ndim != 3:
        raise ValueError("render_candidates must have shape (N, K, C)")
    if int(render_candidates.shape[0]) != int(query_descriptors.shape[0]):
        raise ValueError("query_descriptors and render_candidates must have the same batch size")
    query = F.normalize(query_descriptors, dim=1)
    render = F.normalize(render_candidates, dim=2)
    hidden = torch.tanh(query_proj(query)[:, None, :] + render_proj(render))
    logits = score_head(hidden).squeeze(-1)
    probs = F.softmax(logits, dim=1).detach()
    pooled = torch.sum(hidden * probs[:, :, None], dim=1)
    log_sigma = uncertainty_head(pooled).squeeze(-1)
    return logits, log_sigma


def _score_local_window_correlation_candidates_with_uncertainty(
    query_descriptors: torch.Tensor,
    render_candidates: torch.Tensor,
    query_proj: nn.Module,
    render_proj: nn.Module,
    uncertainty_head: nn.Module,
    *,
    logit_scale: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a local 8x8 window as an explicit query-render correlation volume."""

    if query_descriptors.ndim != 2:
        raise ValueError("query_descriptors must have shape (N, C)")
    if render_candidates.ndim != 3:
        raise ValueError("render_candidates must have shape (N, K, C)")
    if int(render_candidates.shape[0]) != int(query_descriptors.shape[0]):
        raise ValueError("query_descriptors and render_candidates must have the same batch size")
    query_hidden = query_proj(F.normalize(query_descriptors, dim=1))
    render_hidden = render_proj(F.normalize(render_candidates, dim=2))
    query_hidden = F.normalize(query_hidden, dim=1)
    render_hidden = F.normalize(render_hidden, dim=2)
    logits = torch.sum(query_hidden[:, None, :] * render_hidden, dim=2)
    scale = torch.as_tensor(logit_scale, dtype=logits.dtype, device=logits.device)
    logits = logits * torch.clamp(scale, min=1e-3, max=100.0)
    probs = F.softmax(logits, dim=1).detach()
    pooled_render = torch.sum(render_hidden * probs[:, :, None], dim=1)
    pooled = torch.tanh(query_hidden + pooled_render)
    log_sigma = uncertainty_head(pooled).squeeze(-1)
    return logits, log_sigma


def _local_window_logits_from_descriptor_maps(
    model: MatchaStyleJointModel,
    source_descriptor_map: torch.Tensor,
    target_descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    source_tokens = _select_descriptor_rows_from_map(source_descriptor_map, pair_indices, source_indices)
    target_candidates = _sample_local_window_descriptors(target_descriptor_map, pair_indices, target_indices)
    return _score_local_window_candidates(
        source_tokens,
        target_candidates,
        model.local_window_query_proj,
        model.local_window_render_proj,
        model.local_window_score_head,
    )


def _local_window_logits_uncertainty_from_descriptor_maps(
    model: MatchaStyleJointModel,
    source_descriptor_map: torch.Tensor,
    target_descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    mode: str = "mlp",
) -> tuple[torch.Tensor, torch.Tensor]:
    source_tokens = _select_descriptor_rows_from_map(source_descriptor_map, pair_indices, source_indices)
    target_candidates = _sample_local_window_descriptors(target_descriptor_map, pair_indices, target_indices)
    if str(mode) == "correlation":
        return _score_local_window_correlation_candidates_with_uncertainty(
            source_tokens,
            target_candidates,
            model.local_window_query_proj,
            model.local_window_render_proj,
            model.local_window_uncertainty_head,
            logit_scale=torch.exp(torch.clamp(model.local_window_logit_scale, min=-4.0, max=4.0)),
        )
    if str(mode) != "mlp":
        raise ValueError("local window fine mode must be 'mlp' or 'correlation'")
    return _score_local_window_candidates_with_uncertainty(
        source_tokens,
        target_candidates,
        model.local_window_query_proj,
        model.local_window_render_proj,
        model.local_window_score_head,
        model.local_window_uncertainty_head,
    )


def _local_patch_correlation_loss(
    *,
    source_descriptors: torch.Tensor,
    target_descriptor_map: torch.Tensor,
    pair_indices: torch.Tensor,
    target_cell_indices: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor | None, float]:
    if source_descriptors.ndim != 2:
        raise ValueError("source_descriptors must have shape (N, C)")
    if target_descriptor_map.ndim != 4:
        raise ValueError("target_descriptor_map must have shape (B, C, H, W)")
    window = int(window_size)
    if window <= 0 or window % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    batch, channels, height, width = target_descriptor_map.shape
    if int(source_descriptors.shape[1]) != int(channels):
        raise ValueError("source_descriptors channels must match target_descriptor_map")
    pairs = pair_indices.long().reshape(-1).clamp(0, batch - 1)
    indices = target_cell_indices.long().reshape(-1)
    if pairs.numel() != source_descriptors.shape[0] or indices.numel() != source_descriptors.shape[0]:
        raise ValueError("pair_indices and target_cell_indices must contain one value per source descriptor")
    rows = indices // int(width)
    cols = indices % int(width)
    radius = window // 2
    keep = (indices >= 0) & (indices < height * width) & (rows >= radius) & (rows < height - radius) & (cols >= radius) & (cols < width - radius)
    if not torch.any(keep):
        return None, 0.0
    source = F.normalize(source_descriptors[keep], dim=1)
    pairs = pairs[keep]
    rows = rows[keep]
    cols = cols[keep]
    target_hwc = target_descriptor_map.permute(0, 2, 3, 1).contiguous()
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            candidates.append(target_hwc[pairs, rows + int(dy), cols + int(dx)])
    candidate_tensor = F.normalize(torch.stack(candidates, dim=1), dim=2)
    logits = torch.einsum("nc,nkc->nk", source, candidate_tensor)
    labels = torch.full((int(logits.shape[0]),), radius * window + radius, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = float(torch.mean((torch.argmax(logits, dim=1) == labels).float()).detach().cpu().item())
    return loss, acc


def _full_map_correspondence_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
    landmark_memory_bank: LandmarkPrototypeMemoryBank | None = None,
    update_landmark_memory: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not _has_full_map_correspondence_supervision(samples):
        return None, {}
    base = samples.coarse_fine_samples
    indices = _filter_indices_to_pair_subset(samples, indices, pair_subset)
    if indices.size == 0:
        return None, {}
    if pair_subset is None:
        query_feature_maps = samples.query_feature_maps
        render_feature_maps = samples.render_feature_maps
    else:
        query_feature_maps = np.asarray(samples.query_feature_maps)[pair_subset]
        render_feature_maps = np.asarray(samples.render_feature_maps)[pair_subset]
    query_map_inverse = np.arange(int(np.asarray(query_feature_maps).shape[0]), dtype=np.int64)
    unique_query_map_count = int(query_map_inverse.size)
    if samples.pair_query_ids is not None:
        query_names = np.asarray(samples.pair_query_ids, dtype=object).reshape(-1)
        if pair_subset is not None:
            query_names = query_names[pair_subset]
        unique_indices: list[int] = []
        name_to_index: dict[str, int] = {}
        inverse: list[int] = []
        for pair_index, name in enumerate(query_names.tolist()):
            key = str(name)
            unique_index = name_to_index.get(key)
            if unique_index is None:
                unique_index = len(unique_indices)
                name_to_index[key] = int(unique_index)
                unique_indices.append(int(pair_index))
            inverse.append(int(unique_index))
        query_feature_maps = np.asarray(query_feature_maps)[np.asarray(unique_indices, dtype=np.int64)]
        query_map_inverse = np.asarray(inverse, dtype=np.int64)
        unique_query_map_count = int(len(unique_indices))
    query_maps = _tensor(query_feature_maps, dtype=torch.float32, device=device)
    render_maps = _tensor(render_feature_maps, dtype=torch.float32, device=device)
    pairs_global = np.asarray(samples.sample_pair_indices, dtype=np.int64)[indices]
    pairs = _tensor(_remap_pair_indices(pairs_global, pair_subset), dtype=torch.long, device=device)
    qidx = _tensor(np.asarray(samples.query_cell_indices)[indices], dtype=torch.long, device=device)
    ridx = _tensor(np.asarray(samples.render_cell_indices)[indices], dtype=torch.long, device=device)
    qlabels = _tensor(base.query_offset_labels[indices], dtype=torch.long, device=device)
    rlabels = _tensor(base.render_offset_labels[indices], dtype=torch.long, device=device)
    qsoft = (
        None
        if getattr(base, "query_offset_soft_labels", None) is None
        else _tensor(base.query_offset_soft_labels[indices], dtype=torch.float32, device=device)
    )
    rsoft = (
        None
        if getattr(base, "render_offset_soft_labels", None) is None
        else _tensor(base.render_offset_soft_labels[indices], dtype=torch.float32, device=device)
    )

    query_desc_map, query_fine_desc_map, _query_heat, query_offset_map = _forward_coarse_and_fine_feature_maps(
        model,
        query_maps,
    )
    if unique_query_map_count != int(query_map_inverse.size):
        inverse = _tensor(query_map_inverse, dtype=torch.long, device=device)
        query_desc_map = query_desc_map[inverse]
        query_fine_desc_map = query_fine_desc_map[inverse]
        _query_heat = _query_heat[inverse]
        query_offset_map = query_offset_map[inverse]
    render_desc_map, render_fine_desc_map, _render_heat, render_offset_map = _forward_coarse_and_fine_feature_maps(
        model,
        render_maps,
    )
    query_z, query_offsets = _select_map_rows(query_desc_map, query_offset_map, pairs, qidx)
    render_z, render_offsets = _select_map_rows(render_desc_map, render_offset_map, pairs, ridx)
    query_fine_z = _select_descriptor_rows_from_map(query_fine_desc_map, pairs, qidx)
    render_fine_z = _select_descriptor_rows_from_map(render_fine_desc_map, pairs, ridx)

    loss = query_z.new_tensor(0.0)
    descriptor_losses = []
    top1_values = []
    match_confidence = torch.ones((int(query_z.shape[0]),), dtype=query_z.dtype, device=query_z.device)
    for pair in torch.unique(pairs).tolist():
        mask = pairs == int(pair)
        if torch.count_nonzero(mask).item() <= 1:
            continue
        pair_q = query_z[mask]
        pair_r = render_z[mask]
        descriptor_loss, confidence = _dual_softmax_descriptor_loss_and_confidence(
            pair_q,
            pair_r,
            float(config.temperature),
        )
        descriptor_losses.append(descriptor_loss)
        match_confidence[mask] = confidence.detach()
        with torch.no_grad():
            scores = pair_q @ pair_r.T
            labels = torch.arange(scores.shape[0], device=device)
            top1_values.append(torch.mean((torch.argmax(scores, dim=1) == labels).float()))
    if descriptor_losses:
        descriptor_loss = torch.stack(descriptor_losses).mean()
    else:
        descriptor_loss, match_confidence = _dual_softmax_descriptor_loss_and_confidence(
            query_z,
            render_z,
            float(config.temperature),
        )
    if float(config.dual_softmax_weight) > 0.0:
        loss = loss + float(config.dual_softmax_weight) * descriptor_loss
    if float(config.offset_loss_weight) > 0.0:
        offset_loss = 0.5 * (
            _offset_distribution_loss(query_offsets, qlabels, qsoft)
            + _offset_distribution_loss(render_offsets, rlabels, rsoft)
        )
        loss = loss + float(config.offset_loss_weight) * offset_loss
    fine_metrics: dict[str, float] = {}
    if float(config.pair_fine_loss_weight) > 0.0:
        pair_fine = model.pair_fine_logits(query_fine_z, render_fine_z)
        pair_sigma = (
            model.pair_fine_uncertainty_log_sigma(query_fine_z, render_fine_z)
            if float(config.fine_uncertainty_loss_weight) > 0.0
            and hasattr(model, "pair_fine_uncertainty_log_sigma")
            else None
        )
        pair_fine_loss, item_metrics = _fine_coordinate_loss_and_metrics(
            pair_fine,
            rlabels,
            soft_targets=rsoft,
            confidence=match_confidence,
            continuous_loss_weight=float(config.fine_continuous_loss_weight),
            uncertainty_log_sigma=pair_sigma,
            uncertainty_loss_weight=float(config.fine_uncertainty_loss_weight),
            loss_mode=str(config.fine_loss_mode),
        )
        fine_metrics.update({f"map_render_pair_fine_{key}": value for key, value in item_metrics.items()})
        if pair_fine_loss is not None:
            loss = loss + float(config.pair_fine_loss_weight) * pair_fine_loss
    if float(config.query_pair_fine_loss_weight) > 0.0:
        query_pair_fine = model.query_pair_fine_logits(query_fine_z, render_fine_z)
        query_pair_sigma = (
            model.query_pair_fine_uncertainty_log_sigma(query_fine_z, render_fine_z)
            if float(config.fine_uncertainty_loss_weight) > 0.0
            and hasattr(model, "query_pair_fine_uncertainty_log_sigma")
            else None
        )
        query_pair_fine_loss, item_metrics = _fine_coordinate_loss_and_metrics(
            query_pair_fine,
            qlabels,
            soft_targets=qsoft,
            confidence=match_confidence,
            continuous_loss_weight=float(config.fine_continuous_loss_weight),
            uncertainty_log_sigma=query_pair_sigma,
            uncertainty_loss_weight=float(config.fine_uncertainty_loss_weight),
            loss_mode=str(config.fine_loss_mode),
        )
        fine_metrics.update({f"map_query_pair_fine_{key}": value for key, value in item_metrics.items()})
        if query_pair_fine_loss is not None:
            loss = loss + float(config.query_pair_fine_loss_weight) * query_pair_fine_loss
    confidence_loss = None
    if float(config.pair_confidence_loss_weight) > 0.0:
        confidence_targets = torch.ones((int(query_z.shape[0]),), dtype=query_z.dtype, device=query_z.device)
        if getattr(base, "sample_confidence_targets", None) is not None:
            confidence_targets = _tensor(base.sample_confidence_targets[indices], dtype=torch.float32, device=device)
            confidence_targets = torch.clamp(confidence_targets, 0.0, 1.0)
        confidence_keep = torch.ones((int(query_z.shape[0]),), dtype=torch.bool, device=query_z.device)
        if samples.sample_confidence_ignore_mask is not None:
            confidence_keep = ~_tensor(
                np.asarray(samples.sample_confidence_ignore_mask, dtype=bool)[indices],
                dtype=torch.bool,
                device=device,
            )
        if torch.any(confidence_keep):
            confidence_logits = model.pair_confidence_logits(query_z, render_z)
            confidence_loss = F.binary_cross_entropy_with_logits(
                confidence_logits[confidence_keep],
                confidence_targets[confidence_keep],
            )
            loss = loss + float(config.pair_confidence_loss_weight) * confidence_loss
    hard_false_losses = []
    hard_false_count = 0
    if float(config.hard_false_match_weight) > 0.0:
        for pair in torch.unique(pairs).tolist():
            mask = pairs == int(pair)
            count = int(torch.count_nonzero(mask).item())
            if count <= 1:
                continue
            scores = query_z[mask] @ render_z[mask].T
            positive_scores = torch.diagonal(scores)
            false_scores = scores.masked_fill(torch.eye(count, dtype=torch.bool, device=scores.device), -1e9)
            hardest_false = torch.max(false_scores, dim=1).values
            hard_false_losses.append(torch.relu(hardest_false - positive_scores + float(config.hard_false_match_margin)).mean())
            hard_false_count += count
        if hard_false_losses:
            hard_false_loss = torch.stack(hard_false_losses).mean()
            loss = loss + float(config.hard_false_match_weight) * hard_false_loss
    rank_metrics: dict[str, torch.Tensor] = {}
    if float(config.coarse_candidate_rank_loss_weight) > 0.0 and hasattr(base, "negative_render_indices"):
        negative_indices_np = np.asarray(base.negative_render_indices, dtype=np.int64)[indices]
        if negative_indices_np.size > 0 and int(negative_indices_np.shape[1]) > 0:
            negative_indices = _tensor(negative_indices_np, dtype=torch.long, device=device)
            negative_z = _select_negative_descriptor_rows_from_map(render_desc_map, pairs, negative_indices)
            rank_loss, rank_metrics = _coarse_candidate_rank_loss_and_metrics(
                query_z,
                render_z,
                negative_z,
                margin=float(config.coarse_candidate_rank_margin),
            )
            loss = loss + float(config.coarse_candidate_rank_loss_weight) * rank_loss
    landmark_metrics: dict[str, float] = {}
    if float(config.landmark_retrieval_loss_weight) > 0.0:
        required = (
            samples.landmark_sample_pair_indices,
            samples.landmark_query_xy,
            samples.landmark_reference_xy,
            samples.landmark_track_ids,
        )
        if any(value is None for value in required):
            raise ValueError("landmark retrieval loss requires continuous SfM observation supervision")
        landmark_pairs_global = np.asarray(samples.landmark_sample_pair_indices, dtype=np.int64)
        landmark_keep = np.ones((landmark_pairs_global.shape[0],), dtype=bool)
        if pair_subset is not None:
            landmark_keep &= np.isin(landmark_pairs_global, np.asarray(pair_subset, dtype=np.int64))
        if not np.any(landmark_keep):
            raise ValueError("selected full-map pairs contain no landmark retrieval observations")
        landmark_pairs_global = landmark_pairs_global[landmark_keep]
        landmark_pairs = _tensor(
            _remap_pair_indices(landmark_pairs_global, pair_subset),
            dtype=torch.long,
            device=device,
        )
        landmark_query_xy = _tensor(
            np.asarray(samples.landmark_query_xy, dtype=np.float32)[landmark_keep],
            dtype=torch.float32,
            device=device,
        )
        landmark_reference_xy = _tensor(
            np.asarray(samples.landmark_reference_xy, dtype=np.float32)[landmark_keep],
            dtype=torch.float32,
            device=device,
        )
        if samples.pair_query_image_sizes is None or samples.pair_reference_image_sizes is None:
            raise ValueError("continuous landmark retrieval sampling requires original query/reference image sizes")
        query_image_sizes_np = np.asarray(samples.pair_query_image_sizes, dtype=np.int64)
        reference_image_sizes_np = np.asarray(samples.pair_reference_image_sizes, dtype=np.int64)
        if pair_subset is not None:
            query_image_sizes_np = query_image_sizes_np[pair_subset]
            reference_image_sizes_np = reference_image_sizes_np[pair_subset]
        query_image_sizes = _tensor(query_image_sizes_np, dtype=torch.float32, device=device)[landmark_pairs]
        reference_image_sizes = _tensor(reference_image_sizes_np, dtype=torch.float32, device=device)[landmark_pairs]
        pair_query_groups = torch.arange(
            int(query_desc_map.shape[0]),
            dtype=torch.long,
            device=device,
        )
        if samples.pair_query_ids is not None:
            query_names = np.asarray(samples.pair_query_ids, dtype=object).reshape(-1)
            if pair_subset is not None:
                query_names = query_names[pair_subset]
            query_name_to_group = {
                str(name): int(group)
                for group, name in enumerate(sorted({str(value) for value in query_names.tolist()}))
            }
            pair_query_groups = _tensor(
                np.asarray([query_name_to_group[str(value)] for value in query_names.tolist()], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
        landmark_query_image_groups = pair_query_groups[landmark_pairs]
        landmark_query_descriptors = _sample_descriptor_rows_at_image_xy(
            query_desc_map,
            landmark_pairs,
            landmark_query_xy,
            image_width=query_image_sizes[:, 0],
            image_height=query_image_sizes[:, 1],
        )
        landmark_reference_descriptors = _sample_descriptor_rows_at_image_xy(
            render_desc_map,
            landmark_pairs,
            landmark_reference_xy,
            image_width=reference_image_sizes[:, 0],
            image_height=reference_image_sizes[:, 1],
        )
        selected_track_ids = _tensor(
            np.asarray(samples.landmark_track_ids, dtype=np.int64)[landmark_keep],
            dtype=torch.long,
            device=device,
        )
        if torch.any(selected_track_ids < 0):
            raise ValueError("every landmark retrieval observation must have a valid SfM track id")
        known_positive_track_ids = None
        if (
            samples.landmark_known_positive_offsets is not None
            or samples.landmark_known_positive_track_ids is not None
        ):
            if (
                samples.landmark_known_positive_offsets is None
                or samples.landmark_known_positive_track_ids is None
            ):
                raise ValueError("landmark known-positive CSR is partially missing")
            known_offsets, known_tracks = subset_landmark_known_positive_csr(
                samples.landmark_known_positive_offsets,
                samples.landmark_known_positive_track_ids,
                landmark_keep,
            )
            known_positive_track_ids = _tensor(
                _landmark_known_positive_csr_to_padded(known_offsets, known_tracks),
                dtype=torch.long,
                device=device,
            )
        strict_positive_track_ids = None
        if (
            samples.landmark_strict_positive_offsets is not None
            or samples.landmark_strict_positive_track_ids is not None
        ):
            if (
                samples.landmark_strict_positive_offsets is None
                or samples.landmark_strict_positive_track_ids is None
            ):
                raise ValueError("landmark strict-positive CSR is partially missing")
            strict_offsets, strict_tracks = subset_landmark_known_positive_csr(
                samples.landmark_strict_positive_offsets,
                samples.landmark_strict_positive_track_ids,
                landmark_keep,
            )
            strict_positive_track_ids = _tensor(
                _landmark_known_positive_csr_to_padded(strict_offsets, strict_tracks),
                dtype=torch.long,
                device=device,
            )
        coherent_hard_negative_track_ids = None
        coherent_hard_negative_mode_ids = None
        if (
            samples.landmark_coherent_hard_negative_offsets is not None
            or samples.landmark_coherent_hard_negative_track_ids is not None
        ):
            if (
                samples.landmark_coherent_hard_negative_offsets is None
                or samples.landmark_coherent_hard_negative_track_ids is None
            ):
                raise ValueError("landmark coherent hard-negative CSR is partially missing")
            coherent_offsets, coherent_tracks = subset_landmark_known_positive_csr(
                samples.landmark_coherent_hard_negative_offsets,
                samples.landmark_coherent_hard_negative_track_ids,
                landmark_keep,
            )
            coherent_hard_negative_track_ids = _tensor(
                _landmark_known_positive_csr_to_padded(
                    coherent_offsets, coherent_tracks
                ),
                dtype=torch.long,
                device=device,
            )
            if samples.landmark_coherent_hard_negative_mode_ids is not None:
                _mode_offsets, coherent_modes = subset_landmark_known_positive_csr(
                    samples.landmark_coherent_hard_negative_offsets,
                    samples.landmark_coherent_hard_negative_mode_ids,
                    landmark_keep,
                )
                if not np.array_equal(_mode_offsets, coherent_offsets):
                    raise RuntimeError("coherent track/mode CSR offsets diverged")
                coherent_hard_negative_mode_ids = _tensor(
                    _landmark_known_positive_csr_to_padded(
                        coherent_offsets, coherent_modes
                    ),
                    dtype=torch.long,
                    device=device,
                )
        selected_track_xyz = None
        if samples.landmark_track_xyz is not None:
            selected_track_xyz = _tensor(
                np.asarray(samples.landmark_track_xyz, dtype=np.float32)[landmark_keep],
                dtype=torch.float32,
                device=device,
            )
        query_group_ids = _observation_query_group_ids(
            landmark_pairs,
            landmark_query_xy,
            image_width=query_image_sizes[:, 0],
            image_height=query_image_sizes[:, 1],
            grid_width=int(query_desc_map.shape[3]),
            grid_height=int(query_desc_map.shape[2]),
            image_group_ids=landmark_query_image_groups,
        )
        learned_landmark_dustbin = int(config.landmark_dustbin_samples_per_image) > 0
        unmatched_query_descriptors = None
        unmatched_dustbin_logits = None
        landmark_dustbin_logits = None
        dustbin_sampling_metrics: dict[str, float] = {}
        if learned_landmark_dustbin:
            query_heatmap_targets = samples.query_heatmap_targets
            if query_heatmap_targets is not None and pair_subset is not None:
                query_heatmap_targets = np.asarray(query_heatmap_targets)[pair_subset]
            unmatched_query_descriptors, dustbin_sampling_metrics = _sample_landmark_dustbin_descriptors(
                query_desc_map,
                query_heatmap_targets=query_heatmap_targets,
                pair_query_group_ids=pair_query_groups,
                positive_pair_indices=landmark_pairs,
                positive_xy=landmark_query_xy,
                positive_image_width=query_image_sizes[:, 0],
                positive_image_height=query_image_sizes[:, 1],
                samples_per_image=int(config.landmark_dustbin_samples_per_image),
                exclusion_radius_cells=int(config.landmark_dustbin_exclusion_radius_cells),
                max_heatmap_target=float(config.landmark_dustbin_max_heatmap_target),
                seed=int(seed),
            )
            dustbin_valid_input = (
                landmark_query_descriptors.detach()
                if bool(config.landmark_dustbin_detach_descriptors)
                else landmark_query_descriptors
            )
            dustbin_unmatched_input = (
                unmatched_query_descriptors.detach()
                if bool(config.landmark_dustbin_detach_descriptors)
                else unmatched_query_descriptors
            )
            landmark_dustbin_logits = (
                model.landmark_dustbin_logits(dustbin_valid_input)
                / float(config.landmark_retrieval_temperature)
                + float(config.landmark_dustbin_logit)
            )
            unmatched_dustbin_logits = (
                model.landmark_dustbin_logits(dustbin_unmatched_input)
                / float(config.landmark_retrieval_temperature)
                + float(config.landmark_dustbin_logit)
            )
        landmark_loss, landmark_metrics = landmark_retrieval_loss(
            landmark_query_descriptors,
            landmark_reference_descriptors,
            selected_track_ids,
            track_xyz=selected_track_xyz,
            query_group_ids=query_group_ids,
            query_image_group_ids=landmark_query_image_groups,
            known_positive_track_ids=known_positive_track_ids,
            strict_positive_track_ids=strict_positive_track_ids,
            coherent_hard_negative_track_ids=coherent_hard_negative_track_ids,
            coherent_hard_negative_mode_ids=coherent_hard_negative_mode_ids,
            dustbin_logits=landmark_dustbin_logits,
            unmatched_query_descriptors=unmatched_query_descriptors,
            unmatched_dustbin_logits=unmatched_dustbin_logits,
            memory_bank=landmark_memory_bank,
            config=LandmarkRetrievalLossConfig(
                temperature=float(config.landmark_retrieval_temperature),
                prototype_history_mix=float(config.landmark_prototype_history_mix),
                memory_candidate_pool_size=int(config.landmark_memory_candidate_pool_size),
                semantic_hard_negatives_per_query=int(config.landmark_semantic_hard_negatives_per_query),
                geometry_hard_negatives_per_track=int(config.landmark_geometry_hard_negatives_per_track),
                random_negatives=int(config.landmark_random_negatives),
                max_memory_negatives=int(config.landmark_max_memory_negatives),
                memory_negative_merge_policy=str(
                    config.landmark_memory_negative_merge_policy
                ),
                system_hard_negative_margin=float(
                    config.landmark_system_hard_negative_margin
                ),
                system_hard_negative_margin_weight=float(
                    config.landmark_system_hard_negative_margin_weight
                ),
                coherent_hard_negative_margin=float(
                    config.landmark_coherent_hard_negative_margin
                ),
                coherent_hard_negative_margin_weight=float(
                    config.landmark_coherent_hard_negative_margin_weight
                ),
                coherent_hard_negative_min_mode_rows=int(
                    config.landmark_coherent_hard_negative_min_mode_rows
                ),
                dustbin_logit=None if learned_landmark_dustbin else float(config.landmark_dustbin_logit),
                dustbin_loss_weight=float(config.landmark_dustbin_loss_weight),
                dustbin_detach_descriptors=bool(config.landmark_dustbin_detach_descriptors),
                prototype_aggregation_method=str(config.landmark_prototype_aggregation_method),
                prototype_l2_normalize_observations=bool(config.landmark_l2_normalize_observations),
                normalize_final_prototypes=bool(config.landmark_normalize_final_prototypes),
                prototype_min_support_observations=int(config.landmark_min_support_observations),
                positive_prototype_source=str(config.landmark_positive_prototype_source),
                set_valued_cell_positives=bool(config.landmark_set_valued_cell_positives),
                exclude_known_cell_positives_from_memory=bool(
                    config.landmark_exclude_known_cell_positives_from_memory
                ),
                synchronize_memory_updates_across_ranks=bool(
                    config.landmark_memory_sync_ddp
                ),
            ),
            update_memory=bool(update_landmark_memory),
            seed=int(seed),
        )
        if landmark_loss is None:
            raise ValueError("landmark retrieval loss found no valid non-negative SfM track ids")
        landmark_metrics.update(dustbin_sampling_metrics)
        loss = loss + float(config.landmark_retrieval_loss_weight) * landmark_loss
    patch_acc_values = []
    if float(config.patch_correlation_loss_weight) > 0.0:
        query_to_render, query_to_render_acc = _local_patch_correlation_loss(
            source_descriptors=query_z,
            target_descriptor_map=render_desc_map,
            pair_indices=pairs,
            target_cell_indices=ridx,
            window_size=int(config.patch_correlation_window_size),
        )
        render_to_query, render_to_query_acc = _local_patch_correlation_loss(
            source_descriptors=render_z,
            target_descriptor_map=query_desc_map,
            pair_indices=pairs,
            target_cell_indices=qidx,
            window_size=int(config.patch_correlation_window_size),
        )
        patch_losses = [item for item in (query_to_render, render_to_query) if item is not None]
        if patch_losses:
            patch_loss = torch.stack(patch_losses).mean()
            loss = loss + float(config.patch_correlation_loss_weight) * patch_loss
        patch_acc_values = [float(item) for item in (query_to_render_acc, render_to_query_acc)]
    with torch.no_grad():
        if top1_values:
            map_top1 = float(torch.mean(torch.stack(top1_values)).item())
        else:
            scores = query_z @ render_z.T
            labels = torch.arange(scores.shape[0], device=device)
            map_top1 = float(torch.mean((torch.argmax(scores, dim=1) == labels).float()).item())
        metrics = {
            "map_descriptor_top1_acc": map_top1,
            "map_query_pair_count": float(query_map_inverse.size),
            "map_unique_query_forward_count": float(unique_query_map_count),
            "map_query_offset_acc": float(torch.mean((torch.argmax(query_offsets, dim=1) == qlabels).float()).item()),
            "map_render_offset_acc": float(torch.mean((torch.argmax(render_offsets, dim=1) == rlabels).float()).item()),
        }
        metrics.update(fine_metrics)
        if patch_acc_values:
            metrics["patch_correlation_acc"] = float(np.mean(patch_acc_values))
        if confidence_loss is not None:
            metrics["map_pair_confidence_loss"] = float(confidence_loss.detach().cpu().item())
        if hard_false_losses:
            metrics["hard_false_match_loss"] = float(torch.stack(hard_false_losses).mean().detach().cpu().item())
            metrics["hard_false_match_count"] = float(hard_false_count)
        metrics.update({key: float(value.detach().cpu().item()) for key, value in rank_metrics.items()})
        metrics.update(landmark_metrics)
    return loss, metrics


def _rgb_keypoint_loss(
    model: MatchaStyleJointModel,
    images: np.ndarray | None,
    labels: np.ndarray | None,
    *,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    seed: int,
    pair_subset: np.ndarray | None = None,
) -> tuple[torch.Tensor | None, dict[str, float | int]]:
    if images is None or labels is None:
        return None, {}
    if pair_subset is not None:
        images = np.asarray(images)[pair_subset]
        labels = np.asarray(labels)[pair_subset]
    image_tensor = _tensor(images, dtype=torch.float32, device=device)
    labels_tensor = _tensor(labels, dtype=torch.long, device=device)
    logits = model.forward_rgb_keypoints(image_tensor)
    return matcha_alike_distillation_loss(
        logits,
        labels_tensor,
        non_keypoint_divisor=int(config.rgb_non_keypoint_divisor),
        seed=int(seed),
    )


def _local_fine_transformer_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    *,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if samples.query_feature_maps is None or samples.render_feature_maps is None:
        return None, {}
    if samples.query_cell_indices is None or samples.render_cell_indices is None:
        return None, {}
    if samples.sample_pair_indices is None:
        return None, {}
    base = samples.coarse_fine_samples
    indices = _filter_indices_to_pair_subset(samples, indices, pair_subset)
    if indices.size == 0:
        return None, {}
    if pair_subset is None:
        query_feature_maps = samples.query_feature_maps
        render_feature_maps = samples.render_feature_maps
    else:
        query_feature_maps = np.asarray(samples.query_feature_maps)[pair_subset]
        render_feature_maps = np.asarray(samples.render_feature_maps)[pair_subset]
    query_maps = _tensor(query_feature_maps, dtype=torch.float32, device=device)
    render_maps = _tensor(render_feature_maps, dtype=torch.float32, device=device)
    pairs_global = np.asarray(samples.sample_pair_indices, dtype=np.int64)[indices]
    pairs = _tensor(_remap_pair_indices(pairs_global, pair_subset), dtype=torch.long, device=device)
    qidx = _tensor(np.asarray(samples.query_cell_indices)[indices], dtype=torch.long, device=device)
    ridx = _tensor(np.asarray(samples.render_cell_indices)[indices], dtype=torch.long, device=device)
    labels = _tensor(base.render_offset_labels[indices], dtype=torch.long, device=device)
    logits = model.local_fine_logits_from_maps(query_maps, render_maps, pairs, qidx, ridx)
    loss, metrics = _fine_coordinate_loss_and_metrics(logits, labels)
    return loss, metrics


def _local_window_fine_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    *,
    config: MatchaJointTrainingConfig | None = None,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    config = config or MatchaJointTrainingConfig()
    if samples.query_feature_maps is None or samples.render_feature_maps is None:
        return None, {}
    use_dense_fine = samples.fine_sample_pair_indices is not None
    if use_dense_fine:
        pairs_global_np = np.asarray(samples.fine_sample_pair_indices, dtype=np.int64)
        keep = np.ones((pairs_global_np.shape[0],), dtype=bool)
        if pair_subset is not None:
            keep &= np.isin(pairs_global_np, np.asarray(pair_subset, dtype=np.int64))
        weights_all_np = None
        if samples.fine_validity_weight is not None:
            weights_all_np = np.asarray(samples.fine_validity_weight, dtype=np.float32).reshape(-1)
            keep &= np.isfinite(weights_all_np) & (weights_all_np > 0.0)
        if not np.any(keep):
            return None, {}
        pairs_global = pairs_global_np[keep]
        qidx_np = np.asarray(samples.fine_query_cell_indices, dtype=np.int64)[keep]
        ridx_np = np.asarray(samples.fine_render_cell_indices, dtype=np.int64)[keep]
        labels_np = np.asarray(samples.fine_render_offset_labels, dtype=np.int64)[keep]
        query_labels_np = np.asarray(samples.fine_query_offset_labels, dtype=np.int64)[keep]
        weights_np = None if weights_all_np is None else weights_all_np[keep]
        soft_np = (
            None
            if samples.fine_render_offset_soft_labels is None
            else np.asarray(samples.fine_render_offset_soft_labels, dtype=np.float32)[keep]
        )
        query_soft_np = (
            None
            if samples.fine_query_offset_soft_labels is None
            else np.asarray(samples.fine_query_offset_soft_labels, dtype=np.float32)[keep]
        )
    else:
        if samples.query_cell_indices is None or samples.render_cell_indices is None:
            return None, {}
        if samples.sample_pair_indices is None:
            return None, {}
        base = samples.coarse_fine_samples
        indices = _filter_indices_to_pair_subset(samples, indices, pair_subset)
        if indices.size == 0:
            return None, {}
        pairs_global = np.asarray(samples.sample_pair_indices, dtype=np.int64)[indices]
        qidx_np = np.asarray(samples.query_cell_indices, dtype=np.int64)[indices]
        ridx_np = np.asarray(samples.render_cell_indices, dtype=np.int64)[indices]
        labels_np = np.asarray(base.render_offset_labels, dtype=np.int64)[indices]
        query_labels_np = np.asarray(base.query_offset_labels, dtype=np.int64)[indices]
        soft_np = (
            None
            if getattr(base, "render_offset_soft_labels", None) is None
            else np.asarray(base.render_offset_soft_labels, dtype=np.float32)[indices]
        )
        query_soft_np = (
            None
            if getattr(base, "query_offset_soft_labels", None) is None
            else np.asarray(base.query_offset_soft_labels, dtype=np.float32)[indices]
        )
        weights_np = None
    if pair_subset is None:
        query_feature_maps = samples.query_feature_maps
        render_feature_maps = samples.render_feature_maps
    else:
        query_feature_maps = np.asarray(samples.query_feature_maps)[pair_subset]
        render_feature_maps = np.asarray(samples.render_feature_maps)[pair_subset]
    query_maps = _tensor(query_feature_maps, dtype=torch.float32, device=device)
    render_maps = _tensor(render_feature_maps, dtype=torch.float32, device=device)
    pairs = _tensor(_remap_pair_indices(pairs_global, pair_subset), dtype=torch.long, device=device)
    qidx = _tensor(qidx_np, dtype=torch.long, device=device)
    ridx = _tensor(ridx_np, dtype=torch.long, device=device)
    query_desc, _query_heat, _query_offset = model.forward_feature_map(query_maps)
    render_desc, _render_heat, _render_offset = model.forward_feature_map(render_maps)
    labels = _tensor(labels_np, dtype=torch.long, device=device)
    soft_targets = None if soft_np is None else _tensor(soft_np, dtype=torch.float32, device=device)
    confidence = None if weights_np is None else _tensor(weights_np, dtype=torch.float32, device=device)
    render_logits, render_sigma = _local_window_logits_uncertainty_from_descriptor_maps(
        model,
        query_desc,
        render_desc,
        pairs,
        qidx,
        ridx,
        mode=str(config.local_window_fine_mode),
    )
    render_loss, render_metrics = _fine_coordinate_loss_and_metrics(
        render_logits,
        labels,
        soft_targets=soft_targets,
        confidence=confidence,
        continuous_loss_weight=float(config.fine_continuous_loss_weight),
        uncertainty_log_sigma=render_sigma,
        uncertainty_loss_weight=float(config.fine_uncertainty_loss_weight),
        loss_mode=str(config.fine_loss_mode),
    )
    query_labels = _tensor(query_labels_np, dtype=torch.long, device=device)
    query_soft_targets = None if query_soft_np is None else _tensor(query_soft_np, dtype=torch.float32, device=device)
    query_logits, query_sigma = _local_window_logits_uncertainty_from_descriptor_maps(
        model,
        render_desc,
        query_desc,
        pairs,
        ridx,
        qidx,
        mode=str(config.local_window_fine_mode),
    )
    query_loss, query_metrics = _fine_coordinate_loss_and_metrics(
        query_logits,
        query_labels,
        soft_targets=query_soft_targets,
        confidence=confidence,
        continuous_loss_weight=float(config.fine_continuous_loss_weight),
        uncertainty_log_sigma=query_sigma,
        uncertainty_loss_weight=float(config.fine_uncertainty_loss_weight),
        loss_mode=str(config.fine_loss_mode),
    )
    parts: list[tuple[torch.Tensor, float, float]] = []
    for item_loss, item_metrics in ((render_loss, render_metrics), (query_loss, query_metrics)):
        if item_loss is None:
            continue
        valid_count = float(item_metrics.get("valid_count", 0.0))
        if valid_count <= 0.0:
            continue
        parts.append((item_loss, valid_count, float(item_metrics.get("acc", 0.0))))
    if not parts:
        return None, {"valid_count": 0.0, "acc": 0.0}
    total_valid = float(sum(item[1] for item in parts))
    loss = sum(item_loss * (valid_count / total_valid) for item_loss, valid_count, _acc in parts)
    acc = float(sum(acc * valid_count for _loss, valid_count, acc in parts) / total_valid)
    return loss, {
        "valid_count": total_valid,
        "acc": acc,
        "epe_bins": float(
            (
                float(render_metrics.get("epe_bins", 0.0)) * float(render_metrics.get("valid_count", 0.0))
                + float(query_metrics.get("epe_bins", 0.0)) * float(query_metrics.get("valid_count", 0.0))
            )
            / total_valid
        ),
        "uncertainty_bins": float(
            (
                float(render_metrics.get("uncertainty_bins", 0.0)) * float(render_metrics.get("valid_count", 0.0))
                + float(query_metrics.get("uncertainty_bins", 0.0)) * float(query_metrics.get("valid_count", 0.0))
            )
            / total_valid
        ),
        "query_valid_count": float(query_metrics.get("valid_count", 0.0)),
        "query_acc": float(query_metrics.get("acc", 0.0)),
        "query_epe_bins": float(query_metrics.get("epe_bins", 0.0)),
        "query_uncertainty_bins": float(query_metrics.get("uncertainty_bins", 0.0)),
        "query_learned_uncertainty_bins": float(query_metrics.get("learned_uncertainty_bins", 0.0)),
        "query_uncertainty_nll": float(query_metrics.get("uncertainty_nll", 0.0)),
        "query_continuous_epe_bins": float(query_metrics.get("continuous_epe_bins", 0.0)),
        "render_valid_count": float(render_metrics.get("valid_count", 0.0)),
        "render_acc": float(render_metrics.get("acc", 0.0)),
        "render_epe_bins": float(render_metrics.get("epe_bins", 0.0)),
        "render_uncertainty_bins": float(render_metrics.get("uncertainty_bins", 0.0)),
        "render_learned_uncertainty_bins": float(render_metrics.get("learned_uncertainty_bins", 0.0)),
        "render_uncertainty_nll": float(render_metrics.get("uncertainty_nll", 0.0)),
        "render_continuous_epe_bins": float(render_metrics.get("continuous_epe_bins", 0.0)),
        "learned_uncertainty_bins": float(
            (
                float(render_metrics.get("learned_uncertainty_bins", 0.0)) * float(render_metrics.get("valid_count", 0.0))
                + float(query_metrics.get("learned_uncertainty_bins", 0.0)) * float(query_metrics.get("valid_count", 0.0))
            )
            / total_valid
        ),
        "uncertainty_nll": float(
            (
                float(render_metrics.get("uncertainty_nll", 0.0)) * float(render_metrics.get("valid_count", 0.0))
                + float(query_metrics.get("uncertainty_nll", 0.0)) * float(query_metrics.get("valid_count", 0.0))
            )
            / total_valid
        ),
        "continuous_epe_bins": float(
            (
                float(render_metrics.get("continuous_epe_bins", 0.0)) * float(render_metrics.get("valid_count", 0.0))
                + float(query_metrics.get("continuous_epe_bins", 0.0)) * float(query_metrics.get("valid_count", 0.0))
            )
            / total_valid
        ),
        "mode_correlation": 1.0 if str(config.local_window_fine_mode) == "correlation" else 0.0,
    }


def _patch_corr_fine_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    *,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
    sample_seed: int = 0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if samples.query_feature_maps is None or samples.render_feature_maps is None:
        return None, {}
    if samples.query_rgb_images is None or samples.render_rgb_images is None:
        return None, {}
    required = (
        samples.fine_sample_pair_indices,
        samples.fine_query_cell_indices,
        samples.fine_render_cell_indices,
        samples.fine_query_offset_labels,
        samples.fine_render_offset_labels,
    )
    if any(value is None for value in required):
        return None, {}
    pairs_global_np = np.asarray(samples.fine_sample_pair_indices, dtype=np.int64).reshape(-1)
    keep = np.ones((pairs_global_np.shape[0],), dtype=bool)
    if pair_subset is not None:
        keep &= np.isin(pairs_global_np, np.asarray(pair_subset, dtype=np.int64))
    labels_np = np.asarray(samples.fine_render_offset_labels, dtype=np.int64).reshape(-1)
    keep &= (labels_np >= 0) & (labels_np < 64)
    qlabels_all_np = np.asarray(samples.fine_query_offset_labels, dtype=np.int64).reshape(-1)
    keep &= (qlabels_all_np >= 0) & (qlabels_all_np < 64)
    if samples.fine_validity_weight is None:
        weights_np = np.ones((pairs_global_np.shape[0],), dtype=np.float32)
    else:
        weights_np = np.asarray(samples.fine_validity_weight, dtype=np.float32).reshape(-1)
        keep &= np.isfinite(weights_np) & (weights_np > 0.0)
    if not np.any(keep):
        return None, {}
    pairs_global = pairs_global_np[keep]
    qidx_np = np.asarray(samples.fine_query_cell_indices, dtype=np.int64).reshape(-1)[keep]
    ridx_np = np.asarray(samples.fine_render_cell_indices, dtype=np.int64).reshape(-1)[keep]
    qlabels_np = qlabels_all_np[keep]
    labels_np = labels_np[keep]
    weights_np = weights_np[keep]
    max_samples = int(config.patch_corr_fine_max_samples_per_pair)
    if max_samples > 0 and int(labels_np.shape[0]) > max_samples:
        rng = np.random.default_rng(int(sample_seed))
        selected = np.sort(rng.choice(int(labels_np.shape[0]), size=max_samples, replace=False))
        pairs_global = pairs_global[selected]
        qidx_np = qidx_np[selected]
        ridx_np = ridx_np[selected]
        qlabels_np = qlabels_np[selected]
        labels_np = labels_np[selected]
        weights_np = weights_np[selected]

    if pair_subset is None:
        query_feature_maps = samples.query_feature_maps
        render_feature_maps = samples.render_feature_maps
        query_rgb_images = samples.query_rgb_images
        render_rgb_images = samples.render_rgb_images
    else:
        subset = np.asarray(pair_subset, dtype=np.int64)
        query_feature_maps = np.asarray(samples.query_feature_maps)[subset]
        render_feature_maps = np.asarray(samples.render_feature_maps)[subset]
        query_rgb_images = np.asarray(samples.query_rgb_images)[subset]
        render_rgb_images = np.asarray(samples.render_rgb_images)[subset]
    query_rgb_arr = np.asarray(query_rgb_images)
    render_rgb_arr = np.asarray(render_rgb_images)
    if query_rgb_arr.ndim != 4:
        raise ValueError("query_rgb_images must have shape (B, C, H, W)")
    if render_rgb_arr.ndim != 4:
        raise ValueError("render_rgb_images must have shape (B, C, H, W)")
    q_grid_h, q_grid_w = int(np.asarray(query_feature_maps).shape[2]), int(np.asarray(query_feature_maps).shape[3])
    r_grid_h, r_grid_w = int(np.asarray(render_feature_maps).shape[2]), int(np.asarray(render_feature_maps).shape[3])
    q_rgb_h, q_rgb_w = int(query_rgb_arr.shape[2]), int(query_rgb_arr.shape[3])
    r_rgb_h, r_rgb_w = int(render_rgb_arr.shape[2]), int(render_rgb_arr.shape[3])

    def xy_from_cell_labels(
        cell_indices: np.ndarray,
        offset_labels: np.ndarray,
        *,
        grid_h: int,
        grid_w: int,
        rgb_h: int,
        rgb_w: int,
    ) -> np.ndarray:
        cells = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
        labels = np.asarray(offset_labels, dtype=np.int64).reshape(-1)
        cols = cells % max(int(grid_w), 1)
        rows = cells // max(int(grid_w), 1)
        bin_x = labels % 8
        bin_y = labels // 8
        return np.stack(
            [
                (cols.astype(np.float64) + (bin_x.astype(np.float64) + 0.5) / 8.0)
                * (float(rgb_w) / max(float(grid_w), 1.0)),
                (rows.astype(np.float64) + (bin_y.astype(np.float64) + 0.5) / 8.0)
                * (float(rgb_h) / max(float(grid_h), 1.0)),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    query_xy_np = xy_from_cell_labels(
        qidx_np,
        qlabels_np,
        grid_h=q_grid_h,
        grid_w=q_grid_w,
        rgb_h=q_rgb_h,
        rgb_w=q_rgb_w,
    )
    render_xy_np = xy_from_cell_labels(
        ridx_np,
        labels_np,
        grid_h=r_grid_h,
        grid_w=r_grid_w,
        rgb_h=r_rgb_h,
        rgb_w=r_rgb_w,
    )
    pairs = _tensor(_remap_pair_indices(pairs_global, pair_subset), dtype=torch.long, device=device)
    qidx = _tensor(qidx_np, dtype=torch.long, device=device)
    ridx = _tensor(ridx_np, dtype=torch.long, device=device)
    query_xy = _tensor(query_xy_np, dtype=torch.float32, device=device)
    render_xy = _tensor(render_xy_np, dtype=torch.float32, device=device)
    render_labels = _tensor(labels_np, dtype=torch.long, device=device)
    query_labels = _tensor(qlabels_np, dtype=torch.long, device=device)
    weights = _tensor(weights_np, dtype=torch.float32, device=device)
    weights = weights / torch.clamp(torch.sum(weights), min=1e-6)
    query_maps_t = _tensor(query_feature_maps, dtype=torch.float32, device=device)
    render_maps_t = _tensor(render_feature_maps, dtype=torch.float32, device=device)
    query_rgb_t = _tensor(query_rgb_images, dtype=torch.float32, device=device)
    render_rgb_t = _tensor(render_rgb_images, dtype=torch.float32, device=device)
    query_context_all = None
    render_context_all = None
    if bool(config.patch_corr_fine_detach_context) and hasattr(model, "patch_corr_fine_head"):
        with torch.no_grad():
            query_descriptor_maps, _query_heatmap, _query_offsets = model.forward_feature_map(query_maps_t)
            render_descriptor_maps, _render_heatmap, _render_offsets = model.forward_feature_map(render_maps_t)
            query_context_all = _select_descriptor_rows_from_map(query_descriptor_maps, pairs, qidx).detach()
            render_context_all = _select_descriptor_rows_from_map(render_descriptor_maps, pairs, ridx).detach()
    chunk_size = max(int(config.patch_corr_fine_batch_size), 1)

    def directional_loss(
        *,
        source_maps: torch.Tensor,
        target_maps: torch.Tensor,
        source_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        source_xy: torch.Tensor,
        target_labels: torch.Tensor,
        source_context_all: torch.Tensor | None,
        target_context_all: torch.Tensor | None,
        target_grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        item_loss = torch.zeros((), dtype=torch.float32, device=device)
        acc_sum = 0.0
        epe_sum = 0.0
        confidence_sum = 0.0
        entropy_sum = 0.0
        for start in range(0, int(target_labels.numel()), chunk_size):
            end = min(start + chunk_size, int(target_labels.numel()))
            chunk = slice(start, end)
            if source_context_all is not None and target_context_all is not None:
                logits = model.patch_corr_fine_head(
                    source_rgb,
                    target_rgb,
                    pairs[chunk],
                    source_xy[chunk],
                    target_indices[chunk],
                    render_grid_hw=target_grid_hw,
                    query_context=source_context_all[chunk],
                    render_context=target_context_all[chunk],
                )
            else:
                logits = model.patch_corr_fine_logits_from_maps_and_rgb(
                    source_maps,
                    target_maps,
                    source_rgb,
                    target_rgb,
                    pairs[chunk],
                    source_indices[chunk],
                    target_indices[chunk],
                    query_xy=source_xy[chunk],
                )
            chunk_labels = target_labels[chunk]
            chunk_weights = weights[chunk]
            ce = F.cross_entropy(logits, chunk_labels, reduction="none")
            item_loss = item_loss + torch.sum(ce * chunk_weights)
            bins = torch.arange(64, dtype=logits.dtype, device=logits.device)
            bx = torch.remainder(bins, 8.0)
            by = torch.floor(bins / 8.0)
            prob = F.softmax(logits, dim=1)
            pred_x = torch.sum(prob * bx[None, :], dim=1)
            pred_y = torch.sum(prob * by[None, :], dim=1)
            target_x = torch.remainder(chunk_labels.to(dtype=logits.dtype), 8.0)
            target_y = torch.floor(chunk_labels.to(dtype=logits.dtype) / 8.0)
            epe = torch.sqrt(torch.clamp((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2, min=1e-8))
            if float(config.patch_corr_fine_epe_weight) > 0.0:
                item_loss = item_loss + float(config.patch_corr_fine_epe_weight) * torch.sum(epe * chunk_weights)
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                entropy = -torch.sum(prob * torch.log(torch.clamp(prob, min=1e-8)), dim=1)
                confidence = torch.max(prob, dim=1).values
                acc_sum += float(torch.sum((pred == chunk_labels).float() * chunk_weights).detach().cpu().item())
                epe_sum += float(torch.sum(epe.detach() * chunk_weights).cpu().item())
                confidence_sum += float(torch.sum(confidence.detach() * chunk_weights).cpu().item())
                entropy_sum += float(torch.sum(entropy.detach() * chunk_weights).cpu().item())
        return item_loss, {
            "acc": float(acc_sum),
            "valid_count": float(target_labels.numel()),
            "epe_bins": float(epe_sum),
            "confidence": float(confidence_sum),
            "entropy": float(entropy_sum),
        }

    render_loss, render_metrics = directional_loss(
        source_maps=query_maps_t,
        target_maps=render_maps_t,
        source_rgb=query_rgb_t,
        target_rgb=render_rgb_t,
        source_indices=qidx,
        target_indices=ridx,
        source_xy=query_xy,
        target_labels=render_labels,
        source_context_all=query_context_all,
        target_context_all=render_context_all,
        target_grid_hw=(int(render_maps_t.shape[2]), int(render_maps_t.shape[3])),
    )
    query_loss, query_metrics = directional_loss(
        source_maps=render_maps_t,
        target_maps=query_maps_t,
        source_rgb=render_rgb_t,
        target_rgb=query_rgb_t,
        source_indices=ridx,
        target_indices=qidx,
        source_xy=render_xy,
        target_labels=query_labels,
        source_context_all=render_context_all,
        target_context_all=query_context_all,
        target_grid_hw=(int(query_maps_t.shape[2]), int(query_maps_t.shape[3])),
    )
    loss = 0.5 * (render_loss + query_loss)
    total_valid = float(render_metrics["valid_count"] + query_metrics["valid_count"])
    return loss, {
        "acc": float(0.5 * (render_metrics["acc"] + query_metrics["acc"])),
        "valid_count": total_valid,
        "epe_bins": float(0.5 * (render_metrics["epe_bins"] + query_metrics["epe_bins"])),
        "confidence": float(0.5 * (render_metrics["confidence"] + query_metrics["confidence"])),
        "entropy": float(0.5 * (render_metrics["entropy"] + query_metrics["entropy"])),
        "render_acc": float(render_metrics["acc"]),
        "render_valid_count": float(render_metrics["valid_count"]),
        "render_epe_bins": float(render_metrics["epe_bins"]),
        "query_acc": float(query_metrics["acc"]),
        "query_valid_count": float(query_metrics["valid_count"]),
        "query_epe_bins": float(query_metrics["epe_bins"]),
    }


def _cell_center_xy(
    cell_indices: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    rgb_h: int,
    rgb_w: int,
) -> np.ndarray:
    cells = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
    cols = cells % max(int(grid_w), 1)
    rows = cells // max(int(grid_w), 1)
    return np.stack(
        [
            (cols.astype(np.float64) + 0.5) * (float(rgb_w) / max(float(grid_w), 1.0)),
            (rows.astype(np.float64) + 0.5) * (float(rgb_h) / max(float(grid_h), 1.0)),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _subcell_label_xy(
    cell_indices: np.ndarray,
    offset_labels: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    rgb_h: int,
    rgb_w: int,
) -> np.ndarray:
    cells = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(offset_labels, dtype=np.int64).reshape(-1)
    cols = cells % max(int(grid_w), 1)
    rows = cells // max(int(grid_w), 1)
    bin_x = labels % 8
    bin_y = labels // 8
    return np.stack(
        [
            (cols.astype(np.float64) + (bin_x.astype(np.float64) + 0.5) / 8.0)
            * (float(rgb_w) / max(float(grid_w), 1.0)),
            (rows.astype(np.float64) + (bin_y.astype(np.float64) + 0.5) / 8.0)
            * (float(rgb_h) / max(float(grid_h), 1.0)),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _scale_supervision_xy_to_rgb_grid(
    xy: np.ndarray,
    cell_indices: np.ndarray,
    offset_labels: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    rgb_h: int,
    rgb_w: int,
    offset_bins: int = 8,
) -> tuple[np.ndarray, tuple[float, float]]:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cells = np.asarray(cell_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(offset_labels, dtype=np.int64).reshape(-1)
    if coords.shape[0] != cells.shape[0] or coords.shape[0] != labels.shape[0]:
        raise ValueError("xy, cell_indices and offset_labels must contain the same number of rows")
    if coords.shape[0] == 0:
        return coords.astype(np.float32, copy=False), (1.0, 1.0)
    bins = max(int(offset_bins), 1)
    cols = cells % max(int(grid_w), 1)
    rows = cells // max(int(grid_w), 1)
    bin_x = labels % bins
    bin_y = labels // bins
    denom_x = cols.astype(np.float64) + (bin_x.astype(np.float64) + 0.5) / float(bins)
    denom_y = rows.astype(np.float64) + (bin_y.astype(np.float64) + 0.5) / float(bins)

    def estimate_axis_scale(values: np.ndarray, denom: np.ndarray, *, rgb_size: int, grid_size: int) -> float:
        stored_cell = float(rgb_size) / max(float(grid_size), 1.0)
        valid = np.isfinite(values) & np.isfinite(denom) & (denom > 0.25) & (values > 0.0)
        if not np.any(valid):
            return 1.0
        estimated_cell = values[valid] / denom[valid]
        estimated_cell = estimated_cell[np.isfinite(estimated_cell) & (estimated_cell > 1e-6)]
        if estimated_cell.size == 0:
            return 1.0
        original_cell = float(np.median(estimated_cell))
        if original_cell <= 0.0 or not np.isfinite(original_cell):
            return 1.0
        return float(stored_cell / original_cell)

    scale_x = estimate_axis_scale(coords[:, 0], denom_x, rgb_size=int(rgb_w), grid_size=int(grid_w))
    scale_y = estimate_axis_scale(coords[:, 1], denom_y, rgb_size=int(rgb_h), grid_size=int(grid_h))
    scaled = coords.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled.astype(np.float32, copy=False), (float(scale_x), float(scale_y))


@torch.no_grad()
def _crop_rgb_windows_for_pairs(
    images: torch.Tensor,
    pair_indices: torch.Tensor,
    centers_xy: torch.Tensor,
    *,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Crop repeated owners without materializing one full RGB image per measurement row."""

    patches, _ = crop_rgb_windows_by_owner(
        images,
        pair_indices,
        centers_xy,
        radius_px=float(radius_px),
        step_px=float(step_px),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    return patches


def _joint_measurement_patch_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    *,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    pair_subset: np.ndarray | None = None,
    sample_seed: int = 0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if samples.query_feature_maps is None or samples.render_feature_maps is None:
        return None, {}
    if samples.query_rgb_images is None or samples.render_rgb_images is None:
        return None, {}
    required = (
        samples.fine_sample_pair_indices,
        samples.fine_query_cell_indices,
        samples.fine_render_cell_indices,
        samples.fine_query_offset_labels,
        samples.fine_render_offset_labels,
    )
    if any(value is None for value in required):
        return None, {}
    branch = getattr(model, "measurement_patch_branch", None)
    if branch is None:
        return None, {}
    pairs_global_np = np.asarray(samples.fine_sample_pair_indices, dtype=np.int64).reshape(-1)
    keep = np.ones((pairs_global_np.shape[0],), dtype=bool)
    if pair_subset is not None:
        keep &= np.isin(pairs_global_np, np.asarray(pair_subset, dtype=np.int64))
    qlabels_np_all = np.asarray(samples.fine_query_offset_labels, dtype=np.int64).reshape(-1)
    rlabels_np_all = np.asarray(samples.fine_render_offset_labels, dtype=np.int64).reshape(-1)
    keep &= (qlabels_np_all >= 0) & (qlabels_np_all < 64) & (rlabels_np_all >= 0) & (rlabels_np_all < 64)
    if samples.fine_validity_weight is None:
        weights_np_all = np.ones((pairs_global_np.shape[0],), dtype=np.float32)
    else:
        weights_np_all = np.asarray(samples.fine_validity_weight, dtype=np.float32).reshape(-1)
        keep &= np.isfinite(weights_np_all) & (weights_np_all > 0.0)
    if not np.any(keep):
        return None, {}

    pairs_global = pairs_global_np[keep]
    qidx_np = np.asarray(samples.fine_query_cell_indices, dtype=np.int64).reshape(-1)[keep]
    ridx_np = np.asarray(samples.fine_render_cell_indices, dtype=np.int64).reshape(-1)[keep]
    qlabels_np = qlabels_np_all[keep]
    rlabels_np = rlabels_np_all[keep]
    weights_np = weights_np_all[keep]
    max_samples = int(config.measurement_patch_max_samples_per_pair)
    selected: np.ndarray | None = None
    if max_samples > 0:
        rng = np.random.default_rng(int(sample_seed))
        selected_parts = []
        for pair_id in np.unique(pairs_global).tolist():
            pair_indices = np.flatnonzero(pairs_global == int(pair_id))
            if int(pair_indices.size) > max_samples:
                pair_indices = np.sort(rng.choice(pair_indices, size=max_samples, replace=False))
            selected_parts.append(pair_indices)
        selected = (
            np.sort(np.concatenate(selected_parts, axis=0)).astype(np.int64, copy=False)
            if selected_parts
            else np.zeros((0,), dtype=np.int64)
        )
        pairs_global = pairs_global[selected]
        qidx_np = qidx_np[selected]
        ridx_np = ridx_np[selected]
        qlabels_np = qlabels_np[selected]
        rlabels_np = rlabels_np[selected]
        weights_np = weights_np[selected]

    if pair_subset is None:
        query_feature_maps = samples.query_feature_maps
        render_feature_maps = samples.render_feature_maps
        query_rgb_images = samples.query_rgb_images
        render_rgb_images = samples.render_rgb_images
    else:
        subset = np.asarray(pair_subset, dtype=np.int64)
        query_feature_maps = np.asarray(samples.query_feature_maps)[subset]
        render_feature_maps = np.asarray(samples.render_feature_maps)[subset]
        query_rgb_images = np.asarray(samples.query_rgb_images)[subset]
        render_rgb_images = np.asarray(samples.render_rgb_images)[subset]

    query_rgb_arr = np.asarray(query_rgb_images, dtype=np.float32)
    render_rgb_arr = np.asarray(render_rgb_images, dtype=np.float32)
    if query_rgb_arr.ndim != 4 or render_rgb_arr.ndim != 4:
        raise ValueError("query/render RGB images must have shape (B, C, H, W)")
    q_grid_h, q_grid_w = int(np.asarray(query_feature_maps).shape[2]), int(np.asarray(query_feature_maps).shape[3])
    r_grid_h, r_grid_w = int(np.asarray(render_feature_maps).shape[2]), int(np.asarray(render_feature_maps).shape[3])
    q_rgb_h, q_rgb_w = int(query_rgb_arr.shape[2]), int(query_rgb_arr.shape[3])
    r_rgb_h, r_rgb_w = int(render_rgb_arr.shape[2]), int(render_rgb_arr.shape[3])

    query_center_xy_np = _cell_center_xy(qidx_np, grid_h=q_grid_h, grid_w=q_grid_w, rgb_h=q_rgb_h, rgb_w=q_rgb_w)
    query_xy_scale = (1.0, 1.0)
    render_xy_scale = (1.0, 1.0)
    if samples.fine_query_xy is not None:
        query_target_xy_np = np.asarray(samples.fine_query_xy, dtype=np.float32).reshape(-1, 2)[keep]
        if selected is not None:
            query_target_xy_np = query_target_xy_np[selected]
        query_target_xy_np, query_xy_scale = _scale_supervision_xy_to_rgb_grid(
            query_target_xy_np,
            qidx_np,
            qlabels_np,
            grid_h=q_grid_h,
            grid_w=q_grid_w,
            rgb_h=q_rgb_h,
            rgb_w=q_rgb_w,
        )
    else:
        query_target_xy_np = _subcell_label_xy(qidx_np, qlabels_np, grid_h=q_grid_h, grid_w=q_grid_w, rgb_h=q_rgb_h, rgb_w=q_rgb_w)
    if samples.fine_render_xy is not None:
        render_anchor_xy_np = np.asarray(samples.fine_render_xy, dtype=np.float32).reshape(-1, 2)[keep]
        if selected is not None:
            render_anchor_xy_np = render_anchor_xy_np[selected]
        render_anchor_xy_np, render_xy_scale = _scale_supervision_xy_to_rgb_grid(
            render_anchor_xy_np,
            ridx_np,
            rlabels_np,
            grid_h=r_grid_h,
            grid_w=r_grid_w,
            rgb_h=r_rgb_h,
            rgb_w=r_rgb_w,
        )
    else:
        render_anchor_xy_np = _subcell_label_xy(ridx_np, rlabels_np, grid_h=r_grid_h, grid_w=r_grid_w, rgb_h=r_rgb_h, rgb_w=r_rgb_w)
    target_delta_np = (query_target_xy_np - query_center_xy_np).astype(np.float32, copy=False)

    pairs = _tensor(_remap_pair_indices(pairs_global, pair_subset), dtype=torch.long, device=device)
    query_center_xy = _tensor(query_center_xy_np, dtype=torch.float32, device=device)
    render_anchor_xy = _tensor(render_anchor_xy_np, dtype=torch.float32, device=device)
    target_delta = _tensor(target_delta_np, dtype=torch.float32, device=device)
    weights = _tensor(weights_np, dtype=torch.float32, device=device).clamp_min(0.0)
    query_rgb_t = _tensor(query_rgb_arr, dtype=torch.float32, device=device)
    render_rgb_t = _tensor(render_rgb_arr, dtype=torch.float32, device=device)

    chunk_size = max(int(config.measurement_patch_batch_size), 1)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_weight = torch.zeros((), dtype=torch.float32, device=device)
    metric_sums = {
        "center_epe_px": 0.0,
        "mean_epe_px": 0.0,
        "mode_epe_px": 0.0,
        "direct_epe_px": 0.0,
        "gated_epe_px": 0.0,
        "mean_improve_ratio": 0.0,
        "mode_improve_ratio": 0.0,
        "direct_improve_ratio": 0.0,
        "target_in_window_ratio": 0.0,
        "dustbin_probability": 0.0,
        "entropy": 0.0,
    }
    metric_weight = 0.0
    radius = float(branch.measurement_search_radius_px)
    for start in range(0, int(target_delta.shape[0]), chunk_size):
        end = min(start + chunk_size, int(target_delta.shape[0]))
        chunk = slice(start, end)
        pair_chunk = pairs[chunk]
        chunk_weights = weights[chunk]
        query_patch = _crop_rgb_windows_for_pairs(
            query_rgb_t,
            pair_chunk,
            query_center_xy[chunk],
            radius_px=float(branch.crop_radius_px),
            step_px=float(branch.step_px),
            image_width=q_rgb_w,
            image_height=q_rgb_h,
        )
        render_patch = _crop_rgb_windows_for_pairs(
            render_rgb_t,
            pair_chunk,
            render_anchor_xy[chunk],
            radius_px=float(branch.crop_radius_px),
            step_px=float(branch.step_px),
            image_width=r_rgb_w,
            image_height=r_rgb_h,
        )
        delta = target_delta[chunk]
        prior_scale = torch.linalg.norm(delta.detach(), dim=1)
        pred = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=prior_scale)
        spatial_loss, spatial_pred = continuous_offset_nll_with_dustbin(
            pred.logits,
            pred.offsets_xy,
            delta,
            dustbin_logit=pred.dustbin_logit,
            search_radius_px=radius,
            epe_weight=float(config.measurement_patch_epe_weight),
            dustbin_bce_weight=float(config.measurement_patch_dustbin_bce_weight),
            dustbin_positive_weight=float(config.measurement_patch_dustbin_positive_weight),
            target_heatmap_sigma_px=float(config.measurement_patch_target_heatmap_sigma_px),
            sample_weight=chunk_weights,
        )
        direct_loss, _direct_pred = residual_delta_gaussian_nll(
            pred.direct_mean_offset_xy,
            pred.direct_log_sigma_xy,
            delta,
            search_radius_px=radius,
            dustbin_logit=pred.dustbin_logit,
            sample_weight=chunk_weights,
            dustbin_positive_weight=float(config.measurement_patch_dustbin_positive_weight),
        )
        row_weight = torch.sum(chunk_weights).clamp_min(1e-6)
        total_loss = total_loss + (spatial_loss + float(config.measurement_patch_direct_loss_weight) * direct_loss) * row_weight
        total_weight = total_weight + row_weight
        with torch.no_grad():
            center_epe = torch.linalg.norm(delta, dim=1)
            mean_epe = torch.linalg.norm(spatial_pred.mean_offset_xy - delta, dim=1)
            mode_xy = spatial_pred.mode_offset_xy if spatial_pred.mode_offset_xy is not None else spatial_pred.mean_offset_xy
            mode_epe = torch.linalg.norm(mode_xy - delta, dim=1)
            direct_epe = torch.linalg.norm(pred.direct_mean_offset_xy - delta, dim=1)
            gated_xy = pred.gated_mean_offset_xy if pred.gated_mean_offset_xy is not None else spatial_pred.mean_offset_xy
            gated_epe = torch.linalg.norm(gated_xy - delta, dim=1)
            in_window = ((torch.abs(delta[:, 0]) <= radius) & (torch.abs(delta[:, 1]) <= radius)).to(dtype=torch.float32)
            dustbin_prob = (
                torch.sigmoid(pred.dustbin_logit.reshape(-1))
                if pred.dustbin_logit is not None
                else torch.zeros_like(center_epe)
            )
            log_probs = spatial_pred.local_log_probs if spatial_pred.local_log_probs is not None else F.log_softmax(pred.logits, dim=1)
            probs = torch.exp(log_probs)
            entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)), dim=1)
            denom = torch.sum(chunk_weights).clamp_min(1e-6)

            def accumulate(name: str, values: torch.Tensor) -> None:
                metric_sums[name] += float(torch.sum(values.detach() * chunk_weights).cpu().item())

            accumulate("center_epe_px", center_epe)
            accumulate("mean_epe_px", mean_epe)
            accumulate("mode_epe_px", mode_epe)
            accumulate("direct_epe_px", direct_epe)
            accumulate("gated_epe_px", gated_epe)
            accumulate("mean_improve_ratio", (mean_epe < center_epe).to(dtype=torch.float32))
            accumulate("mode_improve_ratio", (mode_epe < center_epe).to(dtype=torch.float32))
            accumulate("direct_improve_ratio", (direct_epe < center_epe).to(dtype=torch.float32))
            accumulate("target_in_window_ratio", in_window)
            accumulate("dustbin_probability", dustbin_prob)
            accumulate("entropy", entropy)
            metric_weight += float(denom.detach().cpu().item())
    if float(total_weight.detach().cpu().item()) <= 0.0:
        return None, {}
    loss = total_loss / total_weight.clamp_min(1e-6)
    normalizer = max(float(metric_weight), 1e-6)
    metrics = {key: float(value / normalizer) for key, value in metric_sums.items()}
    metrics["valid_count"] = float(target_delta.shape[0])
    metrics["query_xy_scale_x"] = float(query_xy_scale[0])
    metrics["query_xy_scale_y"] = float(query_xy_scale[1])
    metrics["render_xy_scale_x"] = float(render_xy_scale[0])
    metrics["render_xy_scale_y"] = float(render_xy_scale[1])
    return loss, metrics


def _rgb_keypoint_position_loss(
    model: MatchaStyleJointModel,
    source_images: np.ndarray | None,
    target_images: np.ndarray | None,
    pair_indices: np.ndarray | None,
    source_cell_indices: np.ndarray | None,
    target_cell_indices: np.ndarray | None,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, float]:
    if source_images is None or target_images is None or source_cell_indices is None or target_cell_indices is None:
        return None, 0.0
    source_tensor = _tensor(source_images, dtype=torch.float32, device=device)
    target_tensor = _tensor(target_images, dtype=torch.float32, device=device)
    source_logits = model.forward_rgb_keypoints(source_tensor)
    target_logits = model.forward_rgb_keypoints(target_tensor)
    batch = int(source_logits.shape[0])
    height, width = int(source_logits.shape[2]), int(source_logits.shape[3])
    src_idx = np.asarray(source_cell_indices, dtype=np.int64).reshape(-1)
    tgt_idx = np.asarray(target_cell_indices, dtype=np.int64).reshape(-1)
    src_labels = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    tgt_labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if not (src_idx.shape[0] == tgt_idx.shape[0] == src_labels.shape[0] == tgt_labels.shape[0]):
        raise ValueError("cell indices and labels must contain one value per correspondence")
    pairs = (
        np.zeros((src_idx.shape[0],), dtype=np.int64)
        if pair_indices is None
        else np.asarray(pair_indices, dtype=np.int64).reshape(-1)
    )
    if pairs.shape[0] != src_idx.shape[0]:
        raise ValueError("pair_indices must contain one value per correspondence")
    keep = (
        (pairs >= 0)
        & (pairs < batch)
        & (src_idx >= 0)
        & (src_idx < height * width)
        & (tgt_idx >= 0)
        & (tgt_idx < height * width)
        & (src_labels >= 0)
        & (src_labels < 64)
        & (tgt_labels >= 0)
        & (tgt_labels < 64)
    )
    if not np.any(keep):
        return None, 0.0
    pairs = pairs[keep]
    src_idx = src_idx[keep]
    tgt_idx = tgt_idx[keep]
    src_labels = src_labels[keep]
    tgt_labels = tgt_labels[keep]

    def points_from_cells(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        rows = indices // int(width)
        cols = indices % int(width)
        x = cols * 8 + (labels % 8)
        y = rows * 8 + (labels // 8)
        return np.stack([x, y], axis=1).astype(np.float32, copy=False)

    source_points = _tensor(points_from_cells(src_idx, src_labels), dtype=torch.float32, device=device)
    target_points = _tensor(points_from_cells(tgt_idx, tgt_labels), dtype=torch.float32, device=device)
    batch_indices = _tensor(pairs, dtype=torch.long, device=device)
    loss, acc, _metrics = matcha_keypoint_position_loss(
        source_logits,
        target_logits,
        source_points,
        target_points,
        point_batch_indices=batch_indices,
    )
    return loss, acc


def _total_loss(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    indices: np.ndarray,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    *,
    seed: int,
    landmark_memory_bank: LandmarkPrototypeMemoryBank | None = None,
    update_landmark_memory: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_indices = _filter_indices_to_positive_matches(samples, indices)
    loss = torch.zeros((), dtype=torch.float32, device=device)
    metrics: dict[str, float] = {}
    use_full_map_geometry_loss = str(config.model_type) in {
        "radio_dual_attention",
        "radio_spatial_context",
    } and _has_full_map_correspondence_supervision(samples)
    if positive_indices.size >= 2 and not use_full_map_geometry_loss:
        confidence_ignore = (
            None
            if samples.sample_confidence_ignore_mask is None
            else np.asarray(samples.sample_confidence_ignore_mask, dtype=bool)[positive_indices]
        )
        coarse_loss, state = _coarse_fine_loss(
            model,
            samples.coarse_fine_samples,
            positive_indices,
            config,
            device,
            confidence_ignore_mask=confidence_ignore,
        )
        loss = loss + coarse_loss
        for key, value in state.items():
            if key.endswith(("_valid_count", "_acc")) or key.startswith("coarse_candidate_rank_"):
                metrics[key] = float(value.detach().cpu().item())
    metrics["positive_match_count"] = float(positive_indices.size)
    no_match_loss = None
    if float(config.pair_confidence_loss_weight) > 0.0:
        if str(config.model_type) == "radio_spatial_context":
            raise ValueError(
                "radio_spatial_context no-match confidence requires a future "
                "full-map implementation; row projection is forbidden"
            )
        no_match_loss = _no_match_confidence_loss(
            model, samples, indices, device=device
        )
    if no_match_loss is not None:
        loss = loss + float(config.pair_confidence_loss_weight) * no_match_loss
        metrics["no_match_confidence_loss"] = float(no_match_loss.detach().cpu().item())
    pair_subset = _sample_map_pair_subset(samples, positive_indices if positive_indices.size else indices, int(config.map_pair_batch_size), int(seed))
    map_loss, map_metrics = _full_map_correspondence_loss(
        model,
        samples,
        positive_indices,
        config,
        device,
        pair_subset=pair_subset,
        landmark_memory_bank=landmark_memory_bank,
        update_landmark_memory=bool(update_landmark_memory),
        seed=int(seed) + 17011,
    )
    if map_loss is not None:
        loss = loss + map_loss
        metrics["map_correspondence_loss"] = float(map_loss.detach().cpu().item())
        metrics.update(map_metrics)
    for prefix, maps, targets in (
        ("query", samples.query_feature_maps, samples.query_heatmap_targets),
        ("render", samples.render_feature_maps, samples.render_heatmap_targets),
    ):
        value = _heatmap_loss(model, maps, targets, device=device, pair_subset=pair_subset)
        if value is not None:
            loss = loss + float(config.dense_heatmap_loss_weight) * value
            metrics[f"{prefix}_heatmap_loss"] = float(value.detach().cpu().item())
    for prefix, images, labels in (
        ("query", samples.query_rgb_images, samples.query_rgb_keypoint_labels),
        ("render", samples.render_rgb_images, samples.render_rgb_keypoint_labels),
    ):
        value, keypoint_metrics = _rgb_keypoint_loss(
            model,
            images,
            labels,
            config=config,
            device=device,
            seed=int(seed),
            pair_subset=pair_subset,
        )
        if value is not None:
            loss = loss + float(config.rgb_keypoint_loss_weight) * value
            metrics[f"{prefix}_rgb_keypoint_loss"] = float(value.detach().cpu().item())
            for key, item in keypoint_metrics.items():
                if isinstance(item, (int, float)):
                    metrics[f"{prefix}_rgb_{key}"] = float(item)
    if float(config.repeatability_loss_weight) > 0.0:
        for prefix, images, targets in (
            ("query", samples.query_rgb_images, samples.query_repeatability_targets),
            ("render", samples.render_rgb_images, samples.render_repeatability_targets),
        ):
            value = _repeatability_loss(model, images, targets, device=device, pair_subset=pair_subset)
            if value is not None:
                loss = loss + float(config.repeatability_loss_weight) * value
                metrics[f"{prefix}_repeatability_loss"] = float(value.detach().cpu().item())
    if float(config.local_fine_transformer_loss_weight) > 0.0:
        value, local_metrics = _local_fine_transformer_loss(model, samples, positive_indices, device=device, pair_subset=pair_subset)
        if value is not None:
            loss = loss + float(config.local_fine_transformer_loss_weight) * value
            metrics["local_fine_transformer_loss"] = float(value.detach().cpu().item())
            metrics["local_fine_transformer_acc"] = float(local_metrics["acc"])
            metrics["local_fine_transformer_valid_count"] = float(local_metrics["valid_count"])
    if float(config.local_window_fine_loss_weight) > 0.0:
        value, local_metrics = _local_window_fine_loss(
            model,
            samples,
            positive_indices,
            config=config,
            device=device,
            pair_subset=pair_subset,
        )
        if value is not None:
            loss = loss + float(config.local_window_fine_loss_weight) * value
            metrics["local_window_fine_loss"] = float(value.detach().cpu().item())
            metrics["local_window_fine_acc"] = float(local_metrics["acc"])
            metrics["local_window_fine_valid_count"] = float(local_metrics["valid_count"])
            for key in (
                "epe_bins",
                "uncertainty_bins",
                "continuous_epe_bins",
                "learned_uncertainty_bins",
                "uncertainty_nll",
                "query_valid_count",
                "query_acc",
                "query_epe_bins",
                "query_uncertainty_bins",
                "query_continuous_epe_bins",
                "query_learned_uncertainty_bins",
                "query_uncertainty_nll",
                "render_valid_count",
                "render_acc",
                "render_epe_bins",
                "render_uncertainty_bins",
                "render_continuous_epe_bins",
                "render_learned_uncertainty_bins",
                "render_uncertainty_nll",
            ):
                if key in local_metrics:
                    metrics[f"local_window_fine_{key}"] = float(local_metrics[key])
    if float(config.patch_corr_fine_loss_weight) > 0.0:
        value, patch_metrics = _patch_corr_fine_loss(
            model,
            samples,
            config=config,
            device=device,
            pair_subset=pair_subset,
            sample_seed=int(seed) + 30000,
        )
        if value is not None:
            loss = loss + float(config.patch_corr_fine_loss_weight) * value
            metrics["patch_corr_fine_loss"] = float(value.detach().cpu().item())
            metrics["patch_corr_fine_acc"] = float(patch_metrics["acc"])
            metrics["patch_corr_fine_valid_count"] = float(patch_metrics["valid_count"])
            metrics["patch_corr_fine_epe_bins"] = float(patch_metrics["epe_bins"])
            metrics["patch_corr_fine_confidence"] = float(patch_metrics["confidence"])
            metrics["patch_corr_fine_entropy"] = float(patch_metrics["entropy"])
            for key in (
                "query_acc",
                "query_valid_count",
                "query_epe_bins",
                "render_acc",
                "render_valid_count",
                "render_epe_bins",
            ):
                if key in patch_metrics:
                    metrics[f"patch_corr_fine_{key}"] = float(patch_metrics[key])
    if float(config.measurement_patch_loss_weight) > 0.0:
        value, measurement_metrics = _joint_measurement_patch_loss(
            model,
            samples,
            config=config,
            device=device,
            pair_subset=pair_subset,
            sample_seed=int(seed) + 31000,
        )
        if value is not None:
            loss = loss + float(config.measurement_patch_loss_weight) * value
            metrics["measurement_patch_loss"] = float(value.detach().cpu().item())
            for key, item in measurement_metrics.items():
                metrics[f"measurement_patch_{key}"] = float(item)
    if float(config.rgb_keypoint_position_loss_weight) > 0.0:
        base = samples.coarse_fine_samples
        for prefix, source_images, target_images, source_cells, target_cells, source_labels, target_labels in (
            (
                "query",
                samples.query_rgb_images,
                samples.render_rgb_images,
                samples.query_cell_indices,
                samples.render_cell_indices,
                base.query_offset_labels,
                base.render_offset_labels,
            ),
            (
                "render",
                samples.render_rgb_images,
                samples.query_rgb_images,
                samples.render_cell_indices,
                samples.query_cell_indices,
                base.render_offset_labels,
                base.query_offset_labels,
            ),
        ):
            selected_source_cells = None if source_cells is None else np.asarray(source_cells)[positive_indices]
            selected_target_cells = None if target_cells is None else np.asarray(target_cells)[positive_indices]
            selected_pairs = None if samples.sample_pair_indices is None else np.asarray(samples.sample_pair_indices)[positive_indices]
            value, acc = _rgb_keypoint_position_loss(
                model,
                source_images,
                target_images,
                selected_pairs,
                selected_source_cells,
                selected_target_cells,
                np.asarray(source_labels)[positive_indices],
                np.asarray(target_labels)[positive_indices],
                device=device,
            )
            if value is not None:
                loss = loss + float(config.rgb_keypoint_position_loss_weight) * value
                metrics[f"{prefix}_rgb_position_loss"] = float(value.detach().cpu().item())
                metrics[f"{prefix}_rgb_position_acc"] = float(acc)
    return loss, metrics


def _sample_eval_indices(sample_count: int, max_count: int, seed: int) -> np.ndarray:
    """Deterministic bounded subset for diagnostic losses on large joint caches."""

    count = int(sample_count)
    limit = max(2, int(max_count))
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(count, size=limit, replace=False).astype(np.int64))


def _build_matcha_joint_model_for_samples(
    samples: MatchaJointTrainingSet,
    config: MatchaJointTrainingConfig,
    device: torch.device,
) -> MatchaStyleJointModel:
    if int(config.output_dim) > samples.coarse_fine_samples.input_dim:
        raise ValueError("output_dim must be <= sample input_dim")
    if (
        str(config.model_type) == "radio_dual_attention"
        and int(config.fine_input_dim) + int(config.coarse_input_dim) != samples.coarse_fine_samples.input_dim
    ):
        raise ValueError("fine_input_dim + coarse_input_dim must match sample input_dim")
    if str(config.model_type) == "radio_dual_attention":
        return RadioDualAttentionFusionJointModel(
            fine_input_dim=int(config.fine_input_dim),
            coarse_input_dim=int(config.coarse_input_dim),
            output_dim=int(config.output_dim),
            residual_hidden_dim=int(config.residual_hidden_dim),
            attention_hidden_dim=int(config.attention_hidden_dim),
            attention_depth=int(config.attention_depth),
            attention_heads=int(config.attention_heads),
            attention_patch_size=int(config.attention_patch_size),
            attention_upsample_mode=str(config.attention_upsample_mode),
            attention_fusion_mode=str(config.attention_fusion_mode),
            group_size=int(config.group_size),
            input_norm_mode=str(config.input_norm_mode),
            gate_mode=str(config.gate_mode),
            residual_gate_scale=float(config.residual_gate_scale),
            local_window_fine_mode=str(config.local_window_fine_mode),
            measurement_patch_search_radius_px=float(config.measurement_patch_search_radius_px),
            measurement_patch_context_radius_px=float(config.measurement_patch_context_radius_px),
            measurement_patch_step_px=float(config.measurement_patch_step_px),
            measurement_patch_coarse_search_radius_px=float(config.measurement_patch_coarse_search_radius_px),
            measurement_patch_coarse_step_px=float(config.measurement_patch_coarse_step_px),
            measurement_patch_feature_dim=int(config.measurement_patch_feature_dim),
            measurement_patch_hidden_dim=int(config.measurement_patch_hidden_dim),
            measurement_patch_encoder_arch=str(config.measurement_patch_encoder_arch),
            measurement_patch_input_mode=str(config.measurement_patch_input_mode),
        ).to(device)
    model_class = (
        RadioSpatialContextJointModel
        if str(config.model_type) == "radio_spatial_context"
        else MatchaStyleJointModel
    )
    model_kwargs = dict(
        input_dim=samples.coarse_fine_samples.input_dim,
        output_dim=int(config.output_dim),
        residual_hidden_dim=int(config.residual_hidden_dim),
        group_size=int(config.group_size),
        input_norm_mode=str(config.input_norm_mode),
        gate_mode=str(config.gate_mode),
        residual_gate_scale=float(config.residual_gate_scale),
        local_window_fine_mode=str(config.local_window_fine_mode),
        measurement_patch_search_radius_px=float(config.measurement_patch_search_radius_px),
        measurement_patch_context_radius_px=float(config.measurement_patch_context_radius_px),
        measurement_patch_step_px=float(config.measurement_patch_step_px),
        measurement_patch_coarse_search_radius_px=float(config.measurement_patch_coarse_search_radius_px),
        measurement_patch_coarse_step_px=float(config.measurement_patch_coarse_step_px),
        measurement_patch_feature_dim=int(config.measurement_patch_feature_dim),
        measurement_patch_hidden_dim=int(config.measurement_patch_hidden_dim),
        measurement_patch_encoder_arch=str(config.measurement_patch_encoder_arch),
        measurement_patch_input_mode=str(config.measurement_patch_input_mode),
    )
    if model_class is RadioSpatialContextJointModel:
        model_kwargs.update(
            context_hidden_dim=int(config.context_hidden_dim),
            context_broad_kernel_size=int(config.context_broad_kernel_size),
        )
    return model_class(**model_kwargs).to(device)


def _model_state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {str(key): value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _build_landmark_memory_bank(
    samples: MatchaJointTrainingSet,
    config: MatchaJointTrainingConfig,
    device: torch.device,
) -> LandmarkPrototypeMemoryBank | None:
    if float(config.landmark_retrieval_loss_weight) <= 0.0:
        return None
    if samples.landmark_track_ids is None:
        raise ValueError("landmark retrieval training requires continuous SfM observation supervision; rebuild the cache")
    if float(config.landmark_coherent_hard_negative_margin_weight) > 0.0:
        if (
            samples.landmark_coherent_hard_negative_offsets is None
            or samples.landmark_coherent_hard_negative_track_ids is None
        ):
            raise ValueError(
                "coherent hard-negative training requires train-only coherent "
                "track CSR supervision"
            )
        if not str(config.landmark_frozen_negative_bank):
            raise ValueError(
                "coherent hard-negative training requires --landmark_frozen_negative_bank"
            )
    track_ids = np.asarray(samples.landmark_track_ids, dtype=np.int64).reshape(-1)
    if track_ids.size == 0 or np.any(track_ids < 0):
        raise ValueError("landmark retrieval training requires valid non-negative SfM track ids for every observation")
    if str(config.landmark_frozen_negative_bank):
        return LandmarkPrototypeMemoryBank.from_projected_landmark_npz(
            Path(config.landmark_frozen_negative_bank),
            device=device,
            expected_descriptor_dim=int(config.output_dim),
        )
    if str(config.landmark_memory_warm_start_bank):
        return LandmarkPrototypeMemoryBank.mutable_from_projected_landmark_npz(
            Path(config.landmark_memory_warm_start_bank),
            device=device,
            expected_descriptor_dim=int(config.output_dim),
            momentum=float(config.landmark_memory_momentum),
        )
    return LandmarkPrototypeMemoryBank(
        capacity=int(config.landmark_memory_capacity),
        descriptor_dim=int(config.output_dim),
        device=device,
        momentum=float(config.landmark_memory_momentum),
    )


def _accumulate_landmark_training_metrics(
    sums: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, float],
) -> None:
    for key, value in metrics.items():
        if not str(key).startswith("landmark_retrieval_") or not np.isfinite(float(value)):
            continue
        sums[str(key)] = float(sums.get(str(key), 0.0) + float(value))
        counts[str(key)] = int(counts.get(str(key), 0) + 1)


def _landmark_training_summary(
    memory_bank: LandmarkPrototypeMemoryBank | None,
    sums: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float | int | bool | str]:
    if memory_bank is None:
        return {"landmark_retrieval_enabled": False}
    output: dict[str, float | int | bool | str] = {
        "landmark_retrieval_enabled": True,
        "landmark_retrieval_memory_final_size": int(len(memory_bank)),
        "landmark_retrieval_training_metric_reduction": "mean_over_training_pairs",
        "landmark_retrieval_memory_scope": (
            "global_frozen_snapshot"
            if bool(memory_bank.frozen)
            else (
                "global_warm_start_ema"
                if str(memory_bank.initialization_mode)
                == "projected_landmark_mutable_warm_start"
                else "rank_local_ema"
            )
        ),
        "landmark_retrieval_memory_source": str(memory_bank.source_path),
        "landmark_retrieval_memory_initialization_mode": str(memory_bank.initialization_mode),
        "landmark_retrieval_memory_allow_new_tracks": bool(memory_bank.allow_new_tracks),
    }
    for key, value in sums.items():
        count = max(1, int(counts.get(key, 0)))
        output[f"train_{key}"] = float(value) / float(count)
    return output


_VALIDATION_RETRIEVAL_METRICS = (
    "landmark_retrieval_loss",
    "landmark_retrieval_positive_loss",
    "landmark_retrieval_system_hard_negative_margin_loss",
    "landmark_retrieval_coherent_hard_negative_margin_loss",
    "landmark_retrieval_recall_at_1",
    "landmark_retrieval_recall_at_5",
    "landmark_retrieval_recall_at_20",
    "landmark_retrieval_strict_valid_recall_at_1",
    "landmark_retrieval_strict_valid_recall_at_5",
    "landmark_retrieval_strict_valid_recall_at_20",
    "landmark_retrieval_positive_score_mean",
    "landmark_retrieval_hardest_negative_score_mean",
    "landmark_retrieval_score_gap_mean",
)


def _loss_and_metrics_for_samples(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    *,
    seed: int,
    landmark_memory_bank: LandmarkPrototypeMemoryBank | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        idx = _sample_eval_indices(samples.coarse_fine_samples.sample_count, int(config.batch_size), int(seed))
        value, metrics = _total_loss(
            model,
            samples,
            idx,
            config,
            device,
            seed=int(seed),
            landmark_memory_bank=landmark_memory_bank,
            update_landmark_memory=False,
        )
    model.train()
    numeric_metrics = {
        str(key): float(metric)
        for key, metric in metrics.items()
        if isinstance(metric, (int, float)) and np.isfinite(float(metric))
    }
    return float(value.detach().cpu().item()), numeric_metrics


def _loss_value_for_samples(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    *,
    seed: int,
    landmark_memory_bank: LandmarkPrototypeMemoryBank | None = None,
) -> float:
    value, _metrics = _loss_and_metrics_for_samples(
        model,
        samples,
        config,
        device,
        seed=int(seed),
        landmark_memory_bank=landmark_memory_bank,
    )
    return float(value)


def _aggregate_validation_evaluations(
    evaluations: Sequence[tuple[float, dict[str, float]]],
    *,
    selection_metric: str,
) -> dict[str, float | str]:
    if not evaluations:
        raise ValueError("validation requires at least one evaluation")
    losses = np.asarray([float(loss) for loss, _metrics in evaluations], dtype=np.float64)
    if not np.all(np.isfinite(losses)):
        raise ValueError("validation total loss contains non-finite values")
    row: dict[str, float | str] = {
        "loss": float(np.mean(losses)),
        "validation_episode_count": float(len(evaluations)),
    }

    map_losses = [
        float(metrics["map_correspondence_loss"])
        for _loss, metrics in evaluations
        if "map_correspondence_loss" in metrics
        and np.isfinite(float(metrics["map_correspondence_loss"]))
    ]
    if map_losses:
        row["map_correspondence_loss"] = float(np.mean(map_losses))

    valid_counts = np.asarray(
        [
            max(0.0, float(metrics.get("landmark_retrieval_valid_count", 0.0)))
            for _loss, metrics in evaluations
        ],
        dtype=np.float64,
    )
    if np.any(valid_counts > 0.0):
        row["landmark_retrieval_valid_count"] = float(np.sum(valid_counts))
        row["landmark_retrieval_valid_count_mean"] = float(np.mean(valid_counts))
    for key in _VALIDATION_RETRIEVAL_METRICS:
        weighted_sum = 0.0
        weight_sum = 0.0
        for (_loss, metrics), weight in zip(evaluations, valid_counts.tolist()):
            if weight <= 0.0 or key not in metrics:
                continue
            value = float(metrics[key])
            if not np.isfinite(value):
                continue
            weighted_sum += float(weight) * value
            weight_sum += float(weight)
        if weight_sum > 0.0:
            row[key] = float(weighted_sum / weight_sum)

    metric_name = str(selection_metric)
    metric_key = "loss" if metric_name == "total_loss" else metric_name
    if metric_key not in row or not np.isfinite(float(row[metric_key])):
        raise ValueError(
            f"validation selection metric {metric_name!r} is unavailable; "
            "enable landmark retrieval supervision with valid query observations"
        )
    row["selection_metric"] = metric_name
    row["selection_value"] = float(row[metric_key])
    return row


def _evaluate(
    model: MatchaStyleJointModel,
    samples: MatchaJointTrainingSet,
    config: MatchaJointTrainingConfig,
    device: torch.device,
    *,
    landmark_memory_bank: LandmarkPrototypeMemoryBank | None = None,
) -> dict[str, float]:
    model.eval()
    base = samples.coarse_fine_samples
    eval_indices = _sample_eval_indices(base.sample_count, int(config.batch_size), int(config.seed) + 991)
    positive_eval_indices = _filter_indices_to_positive_matches(samples, eval_indices)
    pair_subset = _sample_map_pair_subset(samples, positive_eval_indices if positive_eval_indices.size else eval_indices, int(config.map_pair_batch_size), int(config.seed) + 992)
    with torch.no_grad():
        active_indices = positive_eval_indices if positive_eval_indices.size >= 2 else eval_indices
        metrics = {
            "eval_sample_count": int(eval_indices.shape[0]),
            "positive_match_count": int(positive_eval_indices.shape[0]),
        }
        if str(config.model_type) != "radio_spatial_context":
            query = _tensor(base.query_features[active_indices], dtype=torch.float32, device=device)
            render = _tensor(base.render_features[active_indices], dtype=torch.float32, device=device)
            query_z, query_offsets = model.forward_rows(query)
            render_z, render_offsets = model.forward_rows(render)
            scores = query_z @ render_z.T
            labels = torch.arange(scores.shape[0], device=device)
            qlabels = _tensor(base.query_offset_labels[active_indices], dtype=torch.long, device=device)
            rlabels = _tensor(base.render_offset_labels[active_indices], dtype=torch.long, device=device)
            metrics.update(
                {
                    "train_top1_acc": float(torch.mean((torch.argmax(scores, dim=1) == labels).float()).item()),
                    "query_offset_acc": float(torch.mean((torch.argmax(query_offsets, dim=1) == qlabels).float()).item()),
                    "render_offset_acc": float(torch.mean((torch.argmax(render_offsets, dim=1) == rlabels).float()).item()),
                }
            )
            _query_pair_loss, query_pair_metrics = _fine_coordinate_loss_and_metrics(model.query_pair_fine_logits(query_z, render_z), qlabels)
            _render_pair_loss, render_pair_metrics = _fine_coordinate_loss_and_metrics(model.pair_fine_logits(query_z, render_z), rlabels)
            metrics["query_pair_fine_acc"] = float(query_pair_metrics["acc"])
            metrics["query_pair_fine_valid_count"] = float(query_pair_metrics["valid_count"])
            metrics["render_pair_fine_acc"] = float(render_pair_metrics["acc"])
            metrics["render_pair_fine_valid_count"] = float(render_pair_metrics["valid_count"])
        if (
            float(config.coarse_candidate_rank_loss_weight) > 0.0
            and str(config.model_type) != "radio_spatial_context"
        ):
            negatives = _tensor(base.negative_render_features[active_indices], dtype=torch.float32, device=device)
            if negatives.numel() > 0:
                negative_z = model.encode(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
                _rank_loss, rank_metrics = _coarse_candidate_rank_loss_and_metrics(
                    query_z,
                    render_z,
                    negative_z,
                    margin=float(config.coarse_candidate_rank_margin),
                )
                metrics.update({key: float(value.detach().cpu().item()) for key, value in rank_metrics.items()})
        no_match_loss = None
        if (
            float(config.pair_confidence_loss_weight) > 0.0
            and str(config.model_type) != "radio_spatial_context"
        ):
            no_match_loss = _no_match_confidence_loss(
                model, samples, eval_indices, device=device
            )
        if no_match_loss is not None:
            metrics["no_match_confidence_loss"] = float(no_match_loss.detach().cpu().item())
        for prefix, maps, targets in (
            ("query", samples.query_feature_maps, samples.query_heatmap_targets),
            ("render", samples.render_feature_maps, samples.render_heatmap_targets),
    ):
            value = _heatmap_loss(model, maps, targets, device=device, pair_subset=pair_subset)
            if value is not None:
                metrics[f"{prefix}_heatmap_mae"] = float(value.detach().cpu().item())
        for prefix, images, keypoint_labels in (
            ("query", samples.query_rgb_images, samples.query_rgb_keypoint_labels),
            ("render", samples.render_rgb_images, samples.render_rgb_keypoint_labels),
        ):
            _loss_value, keypoint_metrics = _rgb_keypoint_loss(
                model,
                images,
                keypoint_labels,
                config=config,
                device=device,
                seed=int(config.seed) + 10000,
                pair_subset=pair_subset,
            )
            for key, value in keypoint_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}_rgb_keypoint_{key if key != 'loss' else 'ce'}"] = float(value)
        for prefix, images, targets in (
            ("query", samples.query_rgb_images, samples.query_repeatability_targets),
            ("render", samples.render_rgb_images, samples.render_repeatability_targets),
        ):
            value = _repeatability_loss(model, images, targets, device=device, pair_subset=pair_subset)
            if value is not None:
                metrics[f"{prefix}_repeatability_loss"] = float(value.detach().cpu().item())
        if samples.query_cell_indices is not None and samples.render_cell_indices is not None:
            indices = positive_eval_indices
            _map_loss, map_metrics = _full_map_correspondence_loss(
                model,
                samples,
                indices,
                config,
                device,
                pair_subset=pair_subset,
                landmark_memory_bank=landmark_memory_bank,
                update_landmark_memory=False,
                seed=int(config.seed) + 17011,
            )
            metrics.update(map_metrics)
            local_loss, local_metrics = _local_fine_transformer_loss(model, samples, indices, device=device, pair_subset=pair_subset)
            if local_loss is not None:
                metrics["local_fine_transformer_acc"] = float(local_metrics["acc"])
                metrics["local_fine_transformer_valid_count"] = float(local_metrics["valid_count"])
                metrics["local_fine_transformer_loss"] = float(local_loss.detach().cpu().item())
            if float(config.local_window_fine_loss_weight) > 0.0:
                local_window_loss, local_window_metrics = _local_window_fine_loss(
                    model,
                    samples,
                    indices,
                    config=config,
                    device=device,
                    pair_subset=pair_subset,
                )
                if local_window_loss is not None:
                    metrics["local_window_fine_acc"] = float(local_window_metrics["acc"])
                    metrics["local_window_fine_valid_count"] = float(local_window_metrics["valid_count"])
                    metrics["local_window_fine_loss"] = float(local_window_loss.detach().cpu().item())
                    for key in (
                        "query_valid_count",
                        "query_acc",
                        "query_epe_bins",
                        "query_continuous_epe_bins",
                        "query_learned_uncertainty_bins",
                        "query_uncertainty_nll",
                        "render_valid_count",
                        "render_acc",
                        "render_epe_bins",
                        "render_continuous_epe_bins",
                        "render_learned_uncertainty_bins",
                        "render_uncertainty_nll",
                        "continuous_epe_bins",
                        "learned_uncertainty_bins",
                        "uncertainty_nll",
                    ):
                        if key in local_window_metrics:
                            metrics[f"local_window_fine_{key}"] = float(local_window_metrics[key])
            if float(config.patch_corr_fine_loss_weight) > 0.0:
                patch_corr_loss, patch_corr_metrics = _patch_corr_fine_loss(
                    model,
                    samples,
                    config=config,
                    device=device,
                    pair_subset=pair_subset,
                    sample_seed=int(config.seed) + 40000,
                )
                if patch_corr_loss is not None:
                    metrics["patch_corr_fine_acc"] = float(patch_corr_metrics["acc"])
                    metrics["patch_corr_fine_valid_count"] = float(patch_corr_metrics["valid_count"])
                    metrics["patch_corr_fine_loss"] = float(patch_corr_loss.detach().cpu().item())
                    metrics["patch_corr_fine_epe_bins"] = float(patch_corr_metrics["epe_bins"])
                    metrics["patch_corr_fine_confidence"] = float(patch_corr_metrics["confidence"])
                    metrics["patch_corr_fine_entropy"] = float(patch_corr_metrics["entropy"])
            if float(config.measurement_patch_loss_weight) > 0.0:
                measurement_loss, measurement_metrics = _joint_measurement_patch_loss(
                    model,
                    samples,
                    config=config,
                    device=device,
                    pair_subset=pair_subset,
                    sample_seed=int(config.seed) + 41000,
                )
                if measurement_loss is not None:
                    metrics["measurement_patch_loss"] = float(measurement_loss.detach().cpu().item())
                    for key, value in measurement_metrics.items():
                        metrics[f"measurement_patch_{key}"] = float(value)
            if float(config.rgb_keypoint_position_loss_weight) > 0.0:
                for prefix, source_images, target_images, source_cells, target_cells, source_labels, target_labels in (
                    (
                        "query",
                        samples.query_rgb_images,
                        samples.render_rgb_images,
                        samples.query_cell_indices,
                        samples.render_cell_indices,
                        base.query_offset_labels,
                        base.render_offset_labels,
                    ),
                    (
                        "render",
                        samples.render_rgb_images,
                        samples.query_rgb_images,
                        samples.render_cell_indices,
                        samples.query_cell_indices,
                        base.render_offset_labels,
                        base.query_offset_labels,
                    ),
                ):
                    value, acc = _rgb_keypoint_position_loss(
                        model,
                        source_images,
                        target_images,
                        samples.sample_pair_indices,
                        source_cells,
                        target_cells,
                        source_labels,
                        target_labels,
                        device=device,
                    )
                    if value is not None:
                        metrics[f"{prefix}_rgb_position_acc"] = float(acc)
                        metrics[f"{prefix}_rgb_position_loss"] = float(value.detach().cpu().item())
    model.train()
    return metrics


def train_matcha_joint_model(
    samples: MatchaJointTrainingSet,
    config: MatchaJointTrainingConfig | None = None,
    *,
    validation_samples: MatchaJointTrainingSet | None = None,
    validation_interval: int = 0,
    warm_start_model: MatchaStyleJointModel | None = None,
) -> MatchaJointTrainingRun:
    cfg = config or MatchaJointTrainingConfig()
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")
    model = _build_matcha_joint_model_for_samples(samples, cfg, device)
    warm_start_report: dict[str, object] = {}
    if warm_start_model is not None:
        result = model.load_state_dict(warm_start_model.state_dict(), strict=False)
        warm_start_report = {
            "warm_start_loaded": True,
            "warm_start_missing_keys": list(result.missing_keys),
            "warm_start_unexpected_keys": list(result.unexpected_keys),
        }
    if int(cfg.context_freeze_base_steps) > 0:
        set_radio_spatial_context_base_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(cfg.seed))
    landmark_memory_bank = _build_landmark_memory_bank(samples, cfg, device)
    landmark_metric_sums: dict[str, float] = {}
    landmark_metric_counts: dict[str, int] = {}

    def loss_value(step_seed: int) -> float:
        return _loss_value_for_samples(
            model,
            samples,
            cfg,
            device,
            seed=int(step_seed),
            landmark_memory_bank=landmark_memory_bank,
        )

    initial_loss = loss_value(int(cfg.seed))
    validation_history: list[dict[str, float | int | str]] = []
    best_validation_value = float("inf")
    best_validation_total_loss = float("inf")
    best_validation_step = -1
    best_state: dict[str, torch.Tensor] | None = None

    def maybe_validate(step: int) -> None:
        nonlocal best_validation_value, best_validation_total_loss, best_validation_step, best_state
        if validation_samples is None:
            return
        row = _aggregate_validation_evaluations(
            [
                _loss_and_metrics_for_samples(
                    model,
                    validation_samples,
                    cfg,
                    device,
                    seed=int(cfg.seed) + 50000,
                    landmark_memory_bank=landmark_memory_bank,
                )
            ],
            selection_metric=str(cfg.validation_selection_metric),
        )
        row["step"] = int(step)
        validation_history.append(row)
        selection_value = float(row["selection_value"])
        if selection_value < best_validation_value:
            best_validation_value = selection_value
            best_validation_total_loss = float(row["loss"])
            best_validation_step = int(step)
            best_state = _model_state_snapshot(model)

    maybe_validate(0)
    model.train()
    for step in range(int(cfg.steps)):
        if (
            int(cfg.context_freeze_base_steps) > 0
            and int(step) == int(cfg.context_freeze_base_steps)
        ):
            set_radio_spatial_context_base_trainable(model, trainable=True)
        idx = _sample_indices(rng, samples.coarse_fine_samples.sample_count, int(cfg.batch_size))
        loss, step_metrics = _total_loss(
            model,
            samples,
            idx,
            cfg,
            device,
            seed=int(cfg.seed) + int(step),
            landmark_memory_bank=landmark_memory_bank,
            update_landmark_memory=True,
        )
        _accumulate_landmark_training_metrics(landmark_metric_sums, landmark_metric_counts, step_metrics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if validation_samples is not None and int(validation_interval) > 0 and ((int(step) + 1) % int(validation_interval) == 0):
            maybe_validate(int(step) + 1)
    if validation_samples is not None and (not validation_history or int(validation_history[-1]["step"]) != int(cfg.steps)):
        maybe_validate(int(cfg.steps))
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss = loss_value(int(cfg.seed))
    summary = {
        "stage": "matcha_style_joint_training",
        "model_type": str(cfg.model_type),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "sample_count": int(samples.coarse_fine_samples.sample_count),
        "input_dim": int(samples.coarse_fine_samples.input_dim),
        "output_dim": int(cfg.output_dim),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "fine_loss_mode": str(cfg.fine_loss_mode),
        "loss_audit_source": "fixed_training_subset",
    }
    summary.update(warm_start_report)
    summary.update(_landmark_training_summary(landmark_memory_bank, landmark_metric_sums, landmark_metric_counts))
    if validation_samples is not None:
        summary.update(
            {
                "best_validation_loss": float(best_validation_value),
                "best_validation_metric": str(cfg.validation_selection_metric),
                "best_validation_value": float(best_validation_value),
                "best_validation_total_loss": float(best_validation_total_loss),
                "best_validation_loss_compatibility_semantics": "selection_value",
                "best_validation_step": int(best_validation_step),
                "validation_eval_count": int(len(validation_history)),
                "validation_history": validation_history,
                "validation_retrieval_metric_reduction": "weighted_by_landmark_retrieval_valid_count",
            }
        )
    summary.update(
        _evaluate(
            model,
            samples,
            cfg,
            device,
            landmark_memory_bank=landmark_memory_bank,
        )
    )
    return MatchaJointTrainingRun(model=model.cpu().eval(), summary=summary)


def _load_joint_manifest_metadata(path: Path) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    manifest_path = Path(path)
    metadata = json.loads(manifest_path.read_text())
    if str(metadata.get("format", "")) != _JOINT_MANIFEST_FORMAT:
        raise ValueError(f"unsupported MATCHA joint training manifest format in {path}")
    shards = list(metadata.get("shards", []))
    if not shards:
        raise ValueError("MATCHA joint training manifest contains no shards")
    return manifest_path, metadata, [dict(item) for item in shards]


class _LazyJointShardCache:
    def __init__(self, manifest_path: Path, shards: list[dict[str, object]], *, max_size: int = 1) -> None:
        self.manifest_path = Path(manifest_path)
        self.shards = shards
        self.max_size = max(1, int(max_size))
        self.cache: OrderedDict[int, MatchaJointTrainingSet] = OrderedDict()

    def _path_for_index(self, index: int) -> Path:
        shard_path = Path(str(self.shards[int(index)]["path"]))
        if not shard_path.is_absolute():
            shard_path = self.manifest_path.parent / shard_path
        return shard_path

    def get(self, index: int) -> MatchaJointTrainingSet:
        key = int(index)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        sample, _metadata = load_matcha_joint_training_set_npz(self._path_for_index(key))
        self.cache[key] = sample
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return sample


def train_matcha_joint_model_from_manifest(
    manifest_path: Path,
    config: MatchaJointTrainingConfig | None = None,
    *,
    validation_samples: MatchaJointTrainingSet | None = None,
    validation_manifest_path: Path | None = None,
    validation_interval: int = 0,
    shard_cache_size: int = 1,
    steps_per_shard: int = 1,
    warm_start_model: MatchaStyleJointModel | None = None,
) -> MatchaJointTrainingRun:
    cfg = config or MatchaJointTrainingConfig()
    manifest, metadata, shards = _load_joint_manifest_metadata(Path(manifest_path))
    cache = _LazyJointShardCache(manifest, shards, max_size=int(shard_cache_size))
    validation_cache = None
    validation_shards: list[dict[str, object]] = []
    validation_metadata: dict[str, object] = {}
    if validation_manifest_path is not None:
        validation_manifest, validation_metadata, validation_shards = _load_joint_manifest_metadata(Path(validation_manifest_path))
        validation_cache = _LazyJointShardCache(validation_manifest, validation_shards, max_size=max(1, min(int(shard_cache_size), 2)))
    first_samples = cache.get(0)
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")
    model = _build_matcha_joint_model_for_samples(first_samples, cfg, device)
    warm_start_report: dict[str, object] = {}
    if warm_start_model is not None:
        result = model.load_state_dict(warm_start_model.state_dict(), strict=False)
        warm_start_report = {
            "warm_start_loaded": True,
            "warm_start_missing_keys": list(result.missing_keys),
            "warm_start_unexpected_keys": list(result.unexpected_keys),
        }
    if int(cfg.context_freeze_base_steps) > 0:
        set_radio_spatial_context_base_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(cfg.seed))
    landmark_memory_bank = _build_landmark_memory_bank(first_samples, cfg, device)
    landmark_metric_sums: dict[str, float] = {}
    landmark_metric_counts: dict[str, int] = {}
    shard_count = int(len(shards))
    total_sample_count = int(sum(int(item.get("sample_count", 0)) for item in shards))

    def shard_for_step(step: int) -> int:
        epoch = int(step) // max(1, int(steps_per_shard) * shard_count)
        position = (int(step) // max(1, int(steps_per_shard))) % shard_count
        order_rng = np.random.default_rng(int(cfg.seed) + 1009 + int(epoch))
        return int(order_rng.permutation(shard_count)[position])

    initial_loss = _loss_value_for_samples(
        model,
        first_samples,
        cfg,
        device,
        seed=int(cfg.seed),
        landmark_memory_bank=landmark_memory_bank,
    )
    validation_history: list[dict[str, float | int | str]] = []
    best_validation_value = float("inf")
    best_validation_total_loss = float("inf")
    best_validation_step = -1
    best_state: dict[str, torch.Tensor] | None = None

    def maybe_validate(step: int) -> None:
        nonlocal best_validation_value, best_validation_total_loss, best_validation_step, best_state
        if validation_samples is None and validation_cache is None:
            return
        if validation_cache is not None:
            evaluations = []
            for shard_idx in range(len(validation_shards)):
                shard_samples = validation_cache.get(shard_idx)
                evaluations.append(
                    _loss_and_metrics_for_samples(
                        model,
                        shard_samples,
                        cfg,
                        device,
                        seed=int(cfg.seed) + 50000 + int(shard_idx),
                        landmark_memory_bank=landmark_memory_bank,
                    )
                )
        else:
            assert validation_samples is not None
            evaluations = [
                _loss_and_metrics_for_samples(
                    model,
                    validation_samples,
                    cfg,
                    device,
                    seed=int(cfg.seed) + 50000,
                    landmark_memory_bank=landmark_memory_bank,
                )
            ]
        row = _aggregate_validation_evaluations(
            evaluations,
            selection_metric=str(cfg.validation_selection_metric),
        )
        row["step"] = int(step)
        validation_history.append(row)
        selection_value = float(row["selection_value"])
        if selection_value < best_validation_value:
            best_validation_value = selection_value
            best_validation_total_loss = float(row["loss"])
            best_validation_step = int(step)
            best_state = _model_state_snapshot(model)

    maybe_validate(0)
    model.train()
    last_samples = first_samples
    for step in range(int(cfg.steps)):
        if (
            int(cfg.context_freeze_base_steps) > 0
            and int(step) == int(cfg.context_freeze_base_steps)
        ):
            set_radio_spatial_context_base_trainable(model, trainable=True)
        samples = cache.get(shard_for_step(step))
        last_samples = samples
        idx = _sample_indices(rng, samples.coarse_fine_samples.sample_count, int(cfg.batch_size))
        loss, step_metrics = _total_loss(
            model,
            samples,
            idx,
            cfg,
            device,
            seed=int(cfg.seed) + int(step),
            landmark_memory_bank=landmark_memory_bank,
            update_landmark_memory=True,
        )
        _accumulate_landmark_training_metrics(landmark_metric_sums, landmark_metric_counts, step_metrics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (validation_samples is not None or validation_cache is not None) and int(validation_interval) > 0 and ((int(step) + 1) % int(validation_interval) == 0):
            maybe_validate(int(step) + 1)
    if (validation_samples is not None or validation_cache is not None) and (not validation_history or int(validation_history[-1]["step"]) != int(cfg.steps)):
        maybe_validate(int(cfg.steps))
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss = _loss_value_for_samples(
        model,
        first_samples,
        cfg,
        device,
        seed=int(cfg.seed),
        landmark_memory_bank=landmark_memory_bank,
    )
    eval_samples = validation_samples if validation_samples is not None else (validation_cache.get(0) if validation_cache is not None else first_samples)
    summary = {
        "stage": "matcha_style_joint_training",
        "model_type": str(cfg.model_type),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "sample_count": int(total_sample_count),
        "input_dim": int(first_samples.coarse_fine_samples.input_dim),
        "output_dim": int(cfg.output_dim),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "manifest_shard_count": int(shard_count),
        "manifest_sample_count": int(total_sample_count),
        "manifest_lazy_training": True,
        "manifest_shard_cache_size": int(shard_cache_size),
        "manifest_steps_per_shard": int(steps_per_shard),
        "loss_audit_source": "manifest_shard_0_fixed_subset",
    }
    summary.update(warm_start_report)
    summary.update(_landmark_training_summary(landmark_memory_bank, landmark_metric_sums, landmark_metric_counts))
    if validation_samples is not None or validation_cache is not None:
        summary.update(
            {
                "best_validation_loss": float(best_validation_value),
                "best_validation_metric": str(cfg.validation_selection_metric),
                "best_validation_value": float(best_validation_value),
                "best_validation_total_loss": float(best_validation_total_loss),
                "best_validation_loss_compatibility_semantics": "selection_value",
                "best_validation_step": int(best_validation_step),
                "validation_eval_count": int(len(validation_history)),
                "validation_history": validation_history,
                "validation_retrieval_metric_reduction": "weighted_by_landmark_retrieval_valid_count",
            }
        )
    if validation_cache is not None:
        summary.update(
            {
                "validation_manifest_lazy": True,
                "validation_manifest_shard_count": int(len(validation_shards)),
                "validation_manifest_sample_count": int(validation_metadata.get("sample_count", 0)),
            }
        )
    summary.update(
        _evaluate(
            model,
            eval_samples,
            cfg,
            device,
            landmark_memory_bank=landmark_memory_bank,
        )
    )
    return MatchaJointTrainingRun(model=model.cpu().eval(), summary=summary)


def _iter_prefetched_provider_samples(
    get_sample: Callable[[int], MatchaJointTrainingSet],
    indices: Sequence[int],
    *,
    workers: int,
    depth: int,
):
    if int(workers) <= 0 or int(depth) <= 0:
        for index in indices:
            yield get_sample(int(index))
        return
    pending = deque()
    source = iter(int(index) for index in indices)
    max_pending = max(int(depth), int(workers))
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        exhausted = False

        def fill() -> None:
            nonlocal exhausted
            while not exhausted and len(pending) < max_pending:
                try:
                    index = next(source)
                except StopIteration:
                    exhausted = True
                    return
                pending.append(executor.submit(get_sample, int(index)))

        fill()
        while pending:
            future = pending.popleft()
            fill()
            yield future.result()


def _distributed_training_context() -> tuple[int, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0, 1
    return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())


def _provider_training_pair_ordinals(
    *,
    steps: int,
    gradient_accumulation_pairs: int,
    pair_batch_size: int = 1,
    rank: int,
    world_size: int,
) -> list[int]:
    """Return this rank's disjoint slice of the global provider pair stream."""

    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    if int(gradient_accumulation_pairs) <= 0:
        raise ValueError("gradient_accumulation_pairs must be positive")
    if int(pair_batch_size) <= 0:
        raise ValueError("pair_batch_size must be positive")
    if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
        raise ValueError("rank must be in [0, world_size)")
    local_pairs_per_step = int(gradient_accumulation_pairs) * int(pair_batch_size)
    local_count = int(steps) * local_pairs_per_step
    global_pairs_per_step = local_pairs_per_step * int(world_size)
    return [
        (local_index // local_pairs_per_step) * global_pairs_per_step
        + int(rank) * local_pairs_per_step
        + local_index % local_pairs_per_step
        for local_index in range(local_count)
    ]


@torch.no_grad()
def _average_distributed_model_gradients(model: nn.Module, world_size: int) -> None:
    """Average gradients in one dense collective while preserving globally-unused params."""

    if int(world_size) <= 1:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("distributed joint training requires model parameters on one device and dtype")
    active = torch.as_tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.uint8,
        device=parameters[0].device,
    )
    flattened = torch.cat(
        [
            parameter.grad.detach().reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter).reshape(-1)
            for parameter in parameters
        ]
    )
    torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
    flattened.div_(float(world_size))
    torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX)
    offset = 0
    for parameter, is_active in zip(parameters, active.tolist()):
        count = int(parameter.numel())
        if bool(is_active):
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            parameter.grad.copy_(flattened[offset : offset + count].view_as(parameter))
        else:
            parameter.grad = None
        offset += count


@torch.no_grad()
def _synchronize_distributed_model_buffers(model: nn.Module, world_size: int) -> None:
    """Average floating running state and broadcast integral counters from rank zero."""

    if int(world_size) <= 1:
        return
    for buffer in model.buffers():
        if buffer.is_floating_point():
            torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)
            buffer.div_(float(world_size))
        else:
            torch.distributed.broadcast(buffer, src=0)


def _reduce_distributed_training_statistics(
    *,
    landmark_metric_sums: dict[str, float],
    landmark_metric_counts: dict[str, int],
    total_loss_sum: float,
    total_loss_count: int,
    memory_size: int,
) -> tuple[dict[str, float], dict[str, int], float, int, list[int]]:
    rank, world_size = _distributed_training_context()
    if int(world_size) <= 1:
        return (
            dict(landmark_metric_sums),
            dict(landmark_metric_counts),
            float(total_loss_sum),
            int(total_loss_count),
            [int(memory_size)],
        )
    gathered: list[object] = [None for _ in range(int(world_size))]
    torch.distributed.all_gather_object(
        gathered,
        (
            dict(landmark_metric_sums),
            dict(landmark_metric_counts),
            float(total_loss_sum),
            int(total_loss_count),
            int(memory_size),
        ),
    )
    reduced_sums: dict[str, float] = {}
    reduced_counts: dict[str, int] = {}
    reduced_loss_sum = 0.0
    reduced_loss_count = 0
    memory_sizes: list[int] = []
    for item in gathered:
        if item is None:
            raise RuntimeError(f"rank {rank} received an empty distributed statistic payload")
        item_sums, item_counts, item_loss_sum, item_loss_count, item_memory_size = item
        for key, value in item_sums.items():
            reduced_sums[str(key)] = float(reduced_sums.get(str(key), 0.0) + float(value))
        for key, value in item_counts.items():
            reduced_counts[str(key)] = int(reduced_counts.get(str(key), 0) + int(value))
        reduced_loss_sum += float(item_loss_sum)
        reduced_loss_count += int(item_loss_count)
        memory_sizes.append(int(item_memory_size))
    return reduced_sums, reduced_counts, reduced_loss_sum, reduced_loss_count, memory_sizes


def _gather_distributed_memory_state_hashes(
    memory_bank: LandmarkPrototypeMemoryBank | None,
) -> list[str]:
    """Collect full memory-state digests and reject a silent DDP divergence."""

    digest = "" if memory_bank is None else str(memory_bank.state_sha256())
    _rank, world_size = _distributed_training_context()
    if int(world_size) <= 1:
        return [digest]
    gathered: list[object] = [None for _ in range(int(world_size))]
    torch.distributed.all_gather_object(gathered, digest)
    values = [str(value) for value in gathered]
    if memory_bank is not None and len(set(values)) != 1:
        raise RuntimeError(
            "landmark memory banks diverged across DDP ranks; refusing to save an "
            "ambiguous global retrieval checkpoint"
        )
    return values


def _report_distributed_provider_progress(
    *,
    step: int,
    steps: int,
    rank: int,
    world_size: int,
    elapsed_seconds: float,
    global_pair_count: int,
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
    device: torch.device,
) -> None:
    keys = (
        "total_loss",
        "map_correspondence_loss",
        "landmark_retrieval_loss",
        "query_heatmap_loss",
        "render_heatmap_loss",
        "local_window_fine_loss",
        "measurement_patch_loss",
    )
    values = []
    for key in keys:
        values.extend((float(metric_sums.get(key, 0.0)), float(metric_counts.get(key, 0))))
    reduced = torch.as_tensor(values, dtype=torch.float64, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if int(world_size) > 1:
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    memory_values = None
    if device.type == "cuda":
        memory_values = torch.as_tensor(
            [
                float(torch.cuda.memory_allocated(device)),
                float(torch.cuda.memory_reserved(device)),
                float(torch.cuda.max_memory_allocated(device)),
                float(torch.cuda.max_memory_reserved(device)),
            ],
            dtype=torch.float64,
            device=device,
        )
        if int(world_size) > 1:
            torch.distributed.all_reduce(memory_values, op=torch.distributed.ReduceOp.MAX)
    if int(rank) != 0:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        return
    payload: dict[str, float | int | str] = {
        "stage": "provider_training_progress",
        "step": int(step),
        "steps": int(steps),
        "world_size": int(world_size),
        "elapsed_seconds": float(elapsed_seconds),
        "global_pair_count": int(global_pair_count),
        "global_pairs_per_second": float(global_pair_count) / max(float(elapsed_seconds), 1e-9),
    }
    reduced_values = reduced.detach().cpu().tolist()
    for index, key in enumerate(keys):
        value_sum = float(reduced_values[2 * index])
        value_count = int(round(float(reduced_values[2 * index + 1])))
        if value_count > 0:
            payload[f"mean_{key}"] = value_sum / float(value_count)
    if memory_values is not None:
        allocated, reserved, peak_allocated, peak_reserved = memory_values.detach().cpu().tolist()
        payload["cuda_memory_allocated_gib"] = float(allocated / (1024**3))
        payload["cuda_memory_reserved_gib"] = float(reserved / (1024**3))
        payload["cuda_peak_memory_allocated_gib"] = float(peak_allocated / (1024**3))
        payload["cuda_peak_memory_reserved_gib"] = float(peak_reserved / (1024**3))
    print(json.dumps(payload, sort_keys=True), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def train_matcha_joint_model_from_sample_provider(
    sample_count: int,
    get_sample: Callable[[int], MatchaJointTrainingSet],
    config: MatchaJointTrainingConfig | None = None,
    *,
    validation_sample_count: int = 0,
    get_validation_sample: Callable[[int], MatchaJointTrainingSet] | None = None,
    validation_interval: int = 0,
    steps_per_sample: int = 1,
    provider_gradient_accumulation_pairs: int = 1,
    provider_pair_batch_size: int = 1,
    provider_prefetch_workers: int = 0,
    provider_prefetch_depth: int = 0,
    provider_progress_interval_steps: int = 0,
    provider_fixed_audit_interval_steps: int = 0,
    provider_empty_cuda_cache_interval_steps: int = 0,
    warm_start_model: MatchaStyleJointModel | None = None,
    provider_name: str = "sample_provider",
) -> MatchaJointTrainingRun:
    """Train from a lazy per-sample provider without materializing all pairs."""

    cfg = config or MatchaJointTrainingConfig()
    provider_sample_count = int(sample_count)
    if provider_sample_count <= 0:
        raise ValueError("sample provider must contain at least one sample")
    validation_provider_count = int(validation_sample_count) if get_validation_sample is not None else 0
    if validation_provider_count < 0:
        raise ValueError("validation_sample_count must be non-negative")
    gradient_accumulation_pairs = max(1, int(provider_gradient_accumulation_pairs))
    pair_batch_size = max(1, int(provider_pair_batch_size))
    prefetch_workers = max(0, int(provider_prefetch_workers))
    prefetch_depth = max(0, int(provider_prefetch_depth))
    progress_interval_steps = max(0, int(provider_progress_interval_steps))
    fixed_audit_interval_steps = max(0, int(provider_fixed_audit_interval_steps))
    empty_cuda_cache_interval_steps = max(0, int(provider_empty_cuda_cache_interval_steps))
    rank, world_size = _distributed_training_context()
    first_samples = get_sample(0)
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")
    model = _build_matcha_joint_model_for_samples(first_samples, cfg, device)
    warm_start_report: dict[str, object] = {}
    if warm_start_model is not None:
        result = model.load_state_dict(warm_start_model.state_dict(), strict=False)
        warm_start_report = {
            "warm_start_loaded": True,
            "warm_start_missing_keys": list(result.missing_keys),
            "warm_start_unexpected_keys": list(result.unexpected_keys),
        }
    if int(cfg.context_freeze_base_steps) > 0:
        set_radio_spatial_context_base_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(cfg.seed) + int(rank) * 104729)
    landmark_memory_bank = _build_landmark_memory_bank(first_samples, cfg, device)
    if int(rank) == 0:
        print(
            json.dumps(
                {
                    "stage": "provider_initialization",
                    "steps": int(cfg.steps),
                    "provider_sample_count": int(provider_sample_count),
                    "first_pair_sample_count": int(first_samples.coarse_fine_samples.sample_count),
                    "first_pair_landmark_count": int(
                        0
                        if first_samples.landmark_track_ids is None
                        else np.asarray(first_samples.landmark_track_ids).shape[0]
                    ),
                    "landmark_memory_size": int(
                        0 if landmark_memory_bank is None else len(landmark_memory_bank)
                    ),
                    "landmark_memory_candidate_pool_size": int(
                        cfg.landmark_memory_candidate_pool_size
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    landmark_metric_sums: dict[str, float] = {}
    landmark_metric_counts: dict[str, int] = {}
    total_loss_sum = 0.0
    total_loss_count = 0
    progress_metric_sums: dict[str, float] = {}
    progress_metric_counts: dict[str, int] = {}

    def provider_index_for_training_pair(pair_index: int, count: int, *, salt: int) -> int:
        epoch = int(pair_index) // max(1, int(steps_per_sample) * int(count))
        position = (int(pair_index) // max(1, int(steps_per_sample))) % int(count)
        order_rng = np.random.default_rng(int(cfg.seed) + int(salt) + int(epoch))
        return int(order_rng.permutation(int(count))[position])

    initial_loss = _loss_value_for_samples(
        model,
        first_samples,
        cfg,
        device,
        seed=int(cfg.seed),
        landmark_memory_bank=landmark_memory_bank,
    )
    if int(rank) == 0:
        print(
            json.dumps(
                {
                    "stage": "provider_initial_loss",
                    "loss": float(initial_loss),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    fixed_audit_history: list[dict[str, float | int]] = []
    if int(fixed_audit_interval_steps) > 0:
        fixed_audit_history.append({"step": 0, "loss": float(initial_loss)})
    validation_history: list[dict[str, float | int | str]] = []
    best_validation_value = float("inf")
    best_validation_total_loss = float("inf")
    best_validation_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    validation_elapsed_seconds = 0.0
    fixed_audit_elapsed_seconds = 0.0
    training_loop_started: float | None = None

    def maybe_validate(step: int) -> None:
        nonlocal best_validation_value, best_validation_total_loss, best_validation_step, best_state, validation_elapsed_seconds
        if get_validation_sample is None or validation_provider_count <= 0:
            return
        validation_started = time.perf_counter()
        evaluations = []
        for validation_index in range(validation_provider_count):
            samples = get_validation_sample(int(validation_index))
            evaluations.append(
                _loss_and_metrics_for_samples(
                    model,
                    samples,
                    cfg,
                    device,
                    seed=int(cfg.seed) + 50000 + int(validation_index),
                    landmark_memory_bank=landmark_memory_bank,
                )
            )
        row = _aggregate_validation_evaluations(
            evaluations,
            selection_metric=str(cfg.validation_selection_metric),
        )
        row["step"] = int(step)
        validation_history.append(row)
        selection_value = float(row["selection_value"])
        if selection_value < best_validation_value:
            best_validation_value = selection_value
            best_validation_total_loss = float(row["loss"])
            best_validation_step = int(step)
            best_state = _model_state_snapshot(model)
        validation_seconds = float(time.perf_counter() - validation_started)
        if int(rank) == 0:
            print(
                json.dumps(
                    {
                        "stage": "provider_validation",
                        "step": int(step),
                        "steps": int(cfg.steps),
                        **row,
                        "best_validation_metric": str(cfg.validation_selection_metric),
                        "best_validation_value": float(best_validation_value),
                        "best_validation_total_loss": float(best_validation_total_loss),
                        "best_step": int(best_validation_step),
                        "elapsed_seconds": float(validation_seconds),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if training_loop_started is not None:
            validation_elapsed_seconds += validation_seconds

    maybe_validate(0)
    model.train()
    last_samples = first_samples
    training_pair_ordinals = _provider_training_pair_ordinals(
        steps=int(cfg.steps),
        gradient_accumulation_pairs=int(gradient_accumulation_pairs),
        pair_batch_size=int(pair_batch_size),
        rank=int(rank),
        world_size=int(world_size),
    )
    training_indices = [
        provider_index_for_training_pair(pair_ordinal, provider_sample_count, salt=1009)
        for pair_ordinal in training_pair_ordinals
    ]
    training_samples = _iter_prefetched_provider_samples(
        get_sample,
        training_indices,
        workers=int(prefetch_workers),
        depth=int(prefetch_depth),
    )
    consumed_training_pairs = 0
    pair_ordinal_iterator = iter(training_pair_ordinals)
    training_loop_started = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(int(cfg.steps)):
        if (
            int(cfg.context_freeze_base_steps) > 0
            and int(step) == int(cfg.context_freeze_base_steps)
        ):
            set_radio_spatial_context_base_trainable(model, trainable=True)
        optimizer.zero_grad(set_to_none=True)
        for accumulation_index in range(int(gradient_accumulation_pairs)):
            pair_ordinals = [int(next(pair_ordinal_iterator)) for _ in range(int(pair_batch_size))]
            pair_samples = [next(training_samples) for _ in range(int(pair_batch_size))]
            samples = (
                pair_samples[0]
                if len(pair_samples) == 1
                else merge_matcha_joint_training_sets(
                    [_materialize_index_only_joint_training_set(item) for item in pair_samples]
                )
            )
            last_samples = samples
            idx = _sample_indices(rng, samples.coarse_fine_samples.sample_count, int(cfg.batch_size))
            loss, step_metrics = _total_loss(
                model,
                samples,
                idx,
                cfg,
                device,
                seed=int(cfg.seed) + int(pair_ordinals[0]),
                landmark_memory_bank=landmark_memory_bank,
                update_landmark_memory=True,
            )
            _accumulate_landmark_training_metrics(landmark_metric_sums, landmark_metric_counts, step_metrics)
            loss_value = float(loss.detach().cpu().item())
            total_loss_sum += loss_value
            total_loss_count += 1
            progress_metric_sums["total_loss"] = float(progress_metric_sums.get("total_loss", 0.0) + loss_value)
            progress_metric_counts["total_loss"] = int(progress_metric_counts.get("total_loss", 0) + 1)
            for key in (
                "map_correspondence_loss",
                "landmark_retrieval_loss",
                "query_heatmap_loss",
                "render_heatmap_loss",
                "local_window_fine_loss",
                "measurement_patch_loss",
            ):
                value = step_metrics.get(key)
                if value is None or not np.isfinite(float(value)):
                    continue
                progress_metric_sums[key] = float(progress_metric_sums.get(key, 0.0) + float(value))
                progress_metric_counts[key] = int(progress_metric_counts.get(key, 0) + 1)
            (loss / float(gradient_accumulation_pairs)).backward()
            consumed_training_pairs += int(pair_batch_size)
        _average_distributed_model_gradients(model, int(world_size))
        optimizer.step()
        should_fixed_audit = int(fixed_audit_interval_steps) > 0 and (
            (int(step) + 1) % int(fixed_audit_interval_steps) == 0
            or int(step) + 1 == int(cfg.steps)
        )
        if should_fixed_audit:
            fixed_audit_started = time.perf_counter()
            fixed_audit_loss = _loss_value_for_samples(
                model,
                first_samples,
                cfg,
                device,
                seed=int(cfg.seed),
                landmark_memory_bank=landmark_memory_bank,
            )
            fixed_audit_history.append(
                {"step": int(step) + 1, "loss": float(fixed_audit_loss)}
            )
            fixed_audit_elapsed_seconds += float(
                time.perf_counter() - fixed_audit_started
            )
            if int(rank) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "provider_fixed_audit",
                            "step": int(step) + 1,
                            "steps": int(cfg.steps),
                            "loss": float(fixed_audit_loss),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        should_report = int(progress_interval_steps) > 0 and (
            (int(step) + 1) % int(progress_interval_steps) == 0 or int(step) + 1 == int(cfg.steps)
        )
        if should_report:
            active_training_seconds = max(
                float(time.perf_counter() - training_loop_started)
                - float(validation_elapsed_seconds)
                - float(fixed_audit_elapsed_seconds),
                1e-9,
            )
            _report_distributed_provider_progress(
                step=int(step) + 1,
                steps=int(cfg.steps),
                rank=int(rank),
                world_size=int(world_size),
                elapsed_seconds=active_training_seconds,
                global_pair_count=int(consumed_training_pairs) * int(world_size),
                metric_sums=progress_metric_sums,
                metric_counts=progress_metric_counts,
                device=device,
            )
            progress_metric_sums.clear()
            progress_metric_counts.clear()
        if (
            int(empty_cuda_cache_interval_steps) > 0
            and device.type == "cuda"
            and (int(step) + 1) % int(empty_cuda_cache_interval_steps) == 0
        ):
            torch.cuda.empty_cache()
        if get_validation_sample is not None and validation_provider_count > 0 and int(validation_interval) > 0 and ((int(step) + 1) % int(validation_interval) == 0):
            maybe_validate(int(step) + 1)
    training_loop_finished = time.perf_counter()
    if get_validation_sample is not None and validation_provider_count > 0 and (not validation_history or int(validation_history[-1]["step"]) != int(cfg.steps)):
        maybe_validate(int(cfg.steps))
    if best_state is not None:
        model.load_state_dict(best_state)
    _synchronize_distributed_model_buffers(model, int(world_size))
    (
        landmark_metric_sums,
        landmark_metric_counts,
        total_loss_sum,
        total_loss_count,
        landmark_memory_sizes,
    ) = _reduce_distributed_training_statistics(
        landmark_metric_sums=landmark_metric_sums,
        landmark_metric_counts=landmark_metric_counts,
        total_loss_sum=float(total_loss_sum),
        total_loss_count=int(total_loss_count),
        memory_size=0 if landmark_memory_bank is None else int(len(landmark_memory_bank)),
    )
    final_loss = _loss_value_for_samples(
        model,
        first_samples,
        cfg,
        device,
        seed=int(cfg.seed),
        landmark_memory_bank=landmark_memory_bank,
    )
    landmark_memory_state_hashes = _gather_distributed_memory_state_hashes(
        landmark_memory_bank
    )
    training_wall_seconds = float(training_loop_finished - training_loop_started)
    active_training_seconds = max(
        training_wall_seconds
        - float(validation_elapsed_seconds)
        - float(fixed_audit_elapsed_seconds),
        1e-9,
    )
    eval_samples = get_validation_sample(0) if get_validation_sample is not None and validation_provider_count > 0 else first_samples
    summary = {
        "stage": "matcha_style_joint_training",
        "model_type": str(cfg.model_type),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "sample_count": int(provider_sample_count),
        "input_dim": int(first_samples.coarse_fine_samples.input_dim),
        "output_dim": int(cfg.output_dim),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "provider_lazy_training": True,
        "provider_name": str(provider_name),
        "provider_sample_count": int(provider_sample_count),
        "provider_steps_per_sample": int(steps_per_sample),
        "provider_gradient_accumulation_pairs": int(gradient_accumulation_pairs),
        "provider_pair_batch_size": int(pair_batch_size),
        "provider_training_pair_count": int(consumed_training_pairs),
        "provider_global_training_pair_count": int(consumed_training_pairs) * int(world_size),
        "provider_effective_global_gradient_accumulation_pairs": int(gradient_accumulation_pairs) * int(pair_batch_size) * int(world_size),
        "provider_prefetch_workers": int(prefetch_workers),
        "provider_prefetch_depth": int(prefetch_depth),
        "provider_progress_interval_steps": int(progress_interval_steps),
        "provider_fixed_audit_interval_steps": int(fixed_audit_interval_steps),
        "provider_fixed_audit_count": int(len(fixed_audit_history)),
        "provider_fixed_audit_history": fixed_audit_history,
        "provider_empty_cuda_cache_interval_steps": int(empty_cuda_cache_interval_steps),
        "provider_first_pair_sample_count": int(first_samples.coarse_fine_samples.sample_count),
        "provider_last_pair_sample_count": int(last_samples.coarse_fine_samples.sample_count),
        "distributed_training": bool(int(world_size) > 1),
        "distributed_rank": int(rank),
        "distributed_world_size": int(world_size),
        "train_total_loss_mean": float(total_loss_sum) / float(max(1, int(total_loss_count))),
        "training_loop_elapsed_sec": float(active_training_seconds),
        "training_loop_wall_elapsed_sec": float(training_wall_seconds),
        "provider_validation_elapsed_sec": float(validation_elapsed_seconds),
        "provider_fixed_audit_elapsed_sec": float(fixed_audit_elapsed_seconds),
        "training_global_pairs_per_second": (
            float(consumed_training_pairs) * float(world_size)
            / float(active_training_seconds)
        ),
        "loss_audit_source": "provider_index_0_fixed_subset",
    }
    summary.update(warm_start_report)
    summary.update(_landmark_training_summary(landmark_memory_bank, landmark_metric_sums, landmark_metric_counts))
    if landmark_memory_bank is not None:
        summary.update(
            {
                "landmark_retrieval_memory_scope": (
                    "global_frozen_snapshot"
                    if bool(landmark_memory_bank.frozen)
                    else (
                        "ddp_synchronized_global_warm_start_ema"
                        if (
                            str(landmark_memory_bank.initialization_mode)
                            == "projected_landmark_mutable_warm_start"
                            and bool(cfg.landmark_memory_sync_ddp)
                            and int(world_size) > 1
                        )
                        else (
                            "global_warm_start_ema"
                            if str(landmark_memory_bank.initialization_mode)
                            == "projected_landmark_mutable_warm_start"
                            else "rank_local_ema"
                        )
                    )
                ),
                "landmark_retrieval_memory_final_size_by_rank": [int(value) for value in landmark_memory_sizes],
                "landmark_retrieval_memory_state_sha256_by_rank": list(
                    landmark_memory_state_hashes
                ),
                "landmark_retrieval_memory_state_synchronized": bool(
                    len(set(landmark_memory_state_hashes)) <= 1
                ),
            }
        )
    if get_validation_sample is not None and validation_provider_count > 0:
        summary.update(
            {
                "best_validation_loss": float(best_validation_value),
                "best_validation_metric": str(cfg.validation_selection_metric),
                "best_validation_value": float(best_validation_value),
                "best_validation_total_loss": float(best_validation_total_loss),
                "best_validation_loss_compatibility_semantics": "selection_value",
                "best_validation_step": int(best_validation_step),
                "validation_eval_count": int(len(validation_history)),
                "validation_history": validation_history,
                "validation_provider_lazy": True,
                "validation_provider_sample_count": int(validation_provider_count),
                "validation_retrieval_metric_reduction": "weighted_by_landmark_retrieval_valid_count",
            }
        )
    summary.update(
        _evaluate(
            model,
            eval_samples,
            cfg,
            device,
            landmark_memory_bank=landmark_memory_bank,
        )
    )
    return MatchaJointTrainingRun(model=model.cpu().eval(), summary=summary)


def joint_run_as_coarse_fine_adapter_run(run: MatchaJointTrainingRun) -> MatchaCoarseFineTrainingRun:
    """Expose the trained adapter for existing render/matching evaluators."""

    return MatchaCoarseFineTrainingRun(model=run.model.adapter.cpu().eval(), summary=dict(run.summary))


def save_matcha_joint_model(run: MatchaJointTrainingRun, path: Path) -> None:
    """Save the full MATCHA-style joint model, not only its adapter submodule."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = run.model.cpu().eval()
    torch.save(
        {
            "format": _JOINT_MODEL_FORMAT,
            "model_config": {
                "model_type": (
                    "radio_dual_attention"
                    if isinstance(model, RadioDualAttentionFusionJointModel)
                    else "radio_spatial_context"
                    if isinstance(model, RadioSpatialContextJointModel)
                    else "residual_adapter"
                ),
                "input_dim": int(model.input_dim),
                "output_dim": int(model.output_dim),
                "residual_hidden_dim": int(model.adapter.residual_hidden_dim),
                "group_size": int(model.adapter.group_size),
                "input_norm_mode": str(model.adapter.input_norm_mode),
                "gate_mode": str(model.adapter.gate_mode),
                "residual_gate_scale": float(model.adapter.residual_gate_scale),
                "fine_input_dim": int(getattr(model, "fine_input_dim", 0)),
                "coarse_input_dim": int(getattr(model, "coarse_input_dim", 0)),
                "attention_hidden_dim": int(getattr(model, "attention_hidden_dim", 0)),
                "attention_depth": int(getattr(model, "attention_depth", 0)),
                "attention_heads": int(getattr(model, "attention_heads", 0)),
                "attention_patch_size": int(getattr(model, "attention_patch_size", 1)),
                "attention_upsample_mode": str(getattr(model, "attention_upsample_mode", "bilinear")),
                "attention_fusion_mode": str(getattr(model, "attention_fusion_mode", "legacy")),
                "context_hidden_dim": int(getattr(model, "context_hidden_dim", 0)),
                "context_broad_kernel_size": int(
                    getattr(model, "context_broad_kernel_size", 0)
                ),
                "local_window_fine_mode": str(getattr(model, "local_window_fine_mode", "mlp")),
                "measurement_patch_config": dict(getattr(model, "measurement_patch_config", {})),
            },
            "state_dict": model.state_dict(),
            "summary": dict(run.summary),
        },
        output,
    )


def load_matcha_joint_model(path: Path, device: str = "cpu") -> MatchaJointTrainingRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != _JOINT_MODEL_FORMAT:
        raise ValueError(f"unsupported MATCHA joint checkpoint format in {path}")
    cfg = dict(payload["model_config"])
    if str(cfg.get("model_type", "residual_adapter")) == "radio_dual_attention":
        model = RadioDualAttentionFusionJointModel(
            fine_input_dim=int(cfg["fine_input_dim"]),
            coarse_input_dim=int(cfg["coarse_input_dim"]),
            output_dim=int(cfg["output_dim"]),
            residual_hidden_dim=int(cfg["residual_hidden_dim"]),
            attention_hidden_dim=int(cfg.get("attention_hidden_dim", cfg["residual_hidden_dim"])),
            attention_depth=int(cfg.get("attention_depth", 2)),
            attention_heads=int(cfg.get("attention_heads", 4)),
            attention_patch_size=int(cfg.get("attention_patch_size", 1)),
            attention_upsample_mode=str(cfg.get("attention_upsample_mode", "bilinear")),
            attention_fusion_mode=str(cfg.get("attention_fusion_mode", "legacy")),
            group_size=int(cfg["group_size"]),
            input_norm_mode=str(cfg.get("input_norm_mode", "identity")),
            gate_mode=str(cfg.get("gate_mode", "residual")),
            residual_gate_scale=float(cfg.get("residual_gate_scale", 0.1)),
            local_window_fine_mode=str(cfg.get("local_window_fine_mode", "mlp")),
            measurement_patch_search_radius_px=float(dict(cfg.get("measurement_patch_config", {})).get("search_radius_px", 8.0)),
            measurement_patch_context_radius_px=float(dict(cfg.get("measurement_patch_config", {})).get("context_radius_px", 8.0)),
            measurement_patch_step_px=float(dict(cfg.get("measurement_patch_config", {})).get("step_px", 1.0)),
            measurement_patch_coarse_search_radius_px=float(dict(cfg.get("measurement_patch_config", {})).get("coarse_search_radius_px", 0.0)),
            measurement_patch_coarse_step_px=float(dict(cfg.get("measurement_patch_config", {})).get("coarse_step_px", 0.0)),
            measurement_patch_feature_dim=int(dict(cfg.get("measurement_patch_config", {})).get("feature_dim", 32)),
            measurement_patch_hidden_dim=int(dict(cfg.get("measurement_patch_config", {})).get("hidden_dim", 64)),
            measurement_patch_encoder_arch=str(dict(cfg.get("measurement_patch_config", {})).get("encoder_arch", "simple")),
            measurement_patch_input_mode=str(dict(cfg.get("measurement_patch_config", {})).get("input_mode", "rgb")),
        )
    else:
        model_type = str(cfg.get("model_type", "residual_adapter"))
        model_class = (
            RadioSpatialContextJointModel
            if model_type == "radio_spatial_context"
            else MatchaStyleJointModel
        )
        model_kwargs = dict(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            residual_hidden_dim=int(cfg["residual_hidden_dim"]),
            group_size=int(cfg["group_size"]),
            input_norm_mode=str(cfg.get("input_norm_mode", "identity")),
            gate_mode=str(cfg.get("gate_mode", "residual")),
            residual_gate_scale=float(cfg.get("residual_gate_scale", 0.1)),
            local_window_fine_mode=str(cfg.get("local_window_fine_mode", "mlp")),
            measurement_patch_search_radius_px=float(dict(cfg.get("measurement_patch_config", {})).get("search_radius_px", 8.0)),
            measurement_patch_context_radius_px=float(dict(cfg.get("measurement_patch_config", {})).get("context_radius_px", 8.0)),
            measurement_patch_step_px=float(dict(cfg.get("measurement_patch_config", {})).get("step_px", 1.0)),
            measurement_patch_coarse_search_radius_px=float(dict(cfg.get("measurement_patch_config", {})).get("coarse_search_radius_px", 0.0)),
            measurement_patch_coarse_step_px=float(dict(cfg.get("measurement_patch_config", {})).get("coarse_step_px", 0.0)),
            measurement_patch_feature_dim=int(dict(cfg.get("measurement_patch_config", {})).get("feature_dim", 32)),
            measurement_patch_hidden_dim=int(dict(cfg.get("measurement_patch_config", {})).get("hidden_dim", 64)),
            measurement_patch_encoder_arch=str(dict(cfg.get("measurement_patch_config", {})).get("encoder_arch", "simple")),
            measurement_patch_input_mode=str(dict(cfg.get("measurement_patch_config", {})).get("input_mode", "rgb")),
        )
        if model_class is RadioSpatialContextJointModel:
            model_kwargs.update(
                context_hidden_dim=int(cfg["context_hidden_dim"]),
                context_broad_kernel_size=int(cfg["context_broad_kernel_size"]),
            )
        model = model_class(**model_kwargs)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    summary = dict(payload.get("summary", {}))
    missing = [str(item) for item in getattr(incompatible, "missing_keys", [])]
    unexpected = [str(item) for item in getattr(incompatible, "unexpected_keys", [])]
    if missing:
        summary["missing_state_keys"] = missing
    if unexpected:
        summary["unexpected_state_keys"] = unexpected
    return MatchaJointTrainingRun(model=model.to(torch.device(device)).eval(), summary=summary)


def project_feature_map_with_matcha_joint_model(
    model: MatchaStyleJointModel,
    feature_map: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one raw feature map through learned fusion, selector, offsets and heatmap."""

    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    tensor = torch.as_tensor(fmap[None], dtype=torch.float32, device=torch_device)
    with torch.no_grad():
        desc, heatmap_logits, offset_logits = model.forward_feature_map(tensor)
    if was_training:
        model.train()
    selected = desc[0].detach().cpu().numpy().astype(np.float32, copy=False)
    offsets = offset_logits[0].detach().cpu().numpy().astype(np.float32, copy=False)
    heatmap = torch.sigmoid(heatmap_logits[0, 0]).detach().cpu().numpy().astype(np.float32, copy=False)
    return selected, offsets, heatmap


def _descriptor_rows_from_feature_map(feature_map: np.ndarray) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    return fmap.reshape(fmap.shape[0], -1).T.astype(np.float32, copy=False)


def predict_matcha_joint_pair_heads_for_matches(
    model: MatchaStyleJointModel | RadioDualAttentionFusionJointModel,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    matches,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
    fine_target_side: str = "render",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict confidence and MATCHA fine-offset logits using a joint model."""

    values = list(matches)
    if not values:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)
    side = str(fine_target_side)
    if side not in {"render", "query"}:
        raise ValueError("fine_target_side must be 'render' or 'query'")
    query_rows = _descriptor_rows_from_feature_map(query_feature_map)
    render_rows = _descriptor_rows_from_feature_map(render_feature_map)
    query_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
    render_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
    if np.any(query_indices < 0) or np.any(query_indices >= query_rows.shape[0]):
        raise ValueError("match query_index exceeds query feature map size")
    if np.any(render_indices < 0) or np.any(render_indices >= render_rows.shape[0]):
        raise ValueError("match render_index exceeds render feature map size")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    confidences = []
    fine_logits = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            end = min(start + int(batch_size), len(values))
            query = torch.as_tensor(query_rows[query_indices[start:end]], dtype=torch.float32, device=torch_device)
            render = torch.as_tensor(render_rows[render_indices[start:end]], dtype=torch.float32, device=torch_device)
            if int(query.shape[1]) == int(model.output_dim) and int(render.shape[1]) == int(model.output_dim):
                query_z = F.normalize(query, dim=1)
                render_z = F.normalize(render, dim=1)
            elif int(query.shape[1]) == int(model.input_dim) and int(render.shape[1]) == int(model.input_dim):
                query_z = model.encode(query)
                render_z = model.encode(render)
            else:
                raise ValueError("feature maps must contain either raw joint input descriptors or projected descriptors")
            logits = model.pair_confidence_logits(query_z, render_z)
            fine = model.query_pair_fine_logits(query_z, render_z) if side == "query" else model.pair_fine_logits(query_z, render_z)
            confidences.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False))
            fine_logits.append(fine.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    return np.concatenate(confidences, axis=0), np.concatenate(fine_logits, axis=0)


def _optional_npz_value(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    return np.asarray(value)


def _optional_loaded(data, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    return None if value.size == 0 else value


def _save_npz(path: Path, *, compressed: bool, **arrays: object) -> None:
    save = np.savez_compressed if compressed else np.savez
    save(path, **arrays)


def save_matcha_joint_training_set_npz(
    samples: MatchaJointTrainingSet,
    path: Path,
    *,
    compressed: bool = True,
) -> None:
    """Save full MATCHA joint training tensors for reproducible training."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = samples.coarse_fine_samples
    if isinstance(base, IndexOnlyCoarseFineRows):
        _save_npz(
            output,
            compressed=compressed,
            format=np.asarray([_JOINT_INDEX_FORMAT], dtype=object),
            input_dim=np.asarray([int(base.input_dim)], dtype=np.int64),
            sample_count=np.asarray([int(base.sample_count)], dtype=np.int64),
            query_offset_labels=base.query_offset_labels,
            render_offset_labels=base.render_offset_labels,
            query_offset_soft_labels=_optional_npz_value(base.query_offset_soft_labels),
            render_offset_soft_labels=_optional_npz_value(base.render_offset_soft_labels),
            sample_confidence_targets=_optional_npz_value(base.sample_confidence_targets),
            sample_uncertainty_px=_optional_npz_value(base.sample_uncertainty_px),
            negative_render_indices=base.negative_render_indices,
            roundtrip_errors_px=base.roundtrip_errors_px,
            coarse_metadata=np.asarray([dict(base.metadata or {})], dtype=object),
            query_feature_maps=_optional_npz_value(samples.query_feature_maps),
            render_feature_maps=_optional_npz_value(samples.render_feature_maps),
            query_heatmap_targets=_optional_npz_value(samples.query_heatmap_targets),
            render_heatmap_targets=_optional_npz_value(samples.render_heatmap_targets),
            sample_pair_indices=_optional_npz_value(samples.sample_pair_indices),
            query_cell_indices=_optional_npz_value(samples.query_cell_indices),
            render_cell_indices=_optional_npz_value(samples.render_cell_indices),
            fine_sample_pair_indices=_optional_npz_value(samples.fine_sample_pair_indices),
            fine_query_cell_indices=_optional_npz_value(samples.fine_query_cell_indices),
            fine_render_cell_indices=_optional_npz_value(samples.fine_render_cell_indices),
            fine_query_offset_labels=_optional_npz_value(samples.fine_query_offset_labels),
            fine_render_offset_labels=_optional_npz_value(samples.fine_render_offset_labels),
            fine_query_offset_soft_labels=_optional_npz_value(samples.fine_query_offset_soft_labels),
            fine_render_offset_soft_labels=_optional_npz_value(samples.fine_render_offset_soft_labels),
            fine_query_xy=_optional_npz_value(samples.fine_query_xy),
            fine_render_xy=_optional_npz_value(samples.fine_render_xy),
            fine_render_depth=_optional_npz_value(samples.fine_render_depth),
            fine_support_view_count=_optional_npz_value(samples.fine_support_view_count),
            fine_validity_weight=_optional_npz_value(samples.fine_validity_weight),
            query_rgb_images=_optional_npz_value(samples.query_rgb_images),
            render_rgb_images=_optional_npz_value(samples.render_rgb_images),
            query_rgb_keypoint_labels=_optional_npz_value(samples.query_rgb_keypoint_labels),
            render_rgb_keypoint_labels=_optional_npz_value(samples.render_rgb_keypoint_labels),
            pair_type_ids=_optional_npz_value(samples.pair_type_ids),
            pair_type_names=_optional_npz_value(samples.pair_type_names),
            pair_query_ids=_optional_npz_value(samples.pair_query_ids),
            pair_split_names=_optional_npz_value(samples.pair_split_names),
            pair_candidate_ids=_optional_npz_value(samples.pair_candidate_ids),
            pair_translation_errors_m=_optional_npz_value(samples.pair_translation_errors_m),
            pair_rotation_errors_deg=_optional_npz_value(samples.pair_rotation_errors_deg),
            pair_query_image_sizes=_optional_npz_value(samples.pair_query_image_sizes),
            pair_reference_image_sizes=_optional_npz_value(samples.pair_reference_image_sizes),
            sample_no_match_labels=_optional_npz_value(samples.sample_no_match_labels),
            sample_ignore_mask=_optional_npz_value(samples.sample_ignore_mask),
            sample_confidence_ignore_mask=_optional_npz_value(samples.sample_confidence_ignore_mask),
            sample_track_ids=_optional_npz_value(samples.sample_track_ids),
            sample_track_xyz=_optional_npz_value(samples.sample_track_xyz),
            landmark_sample_pair_indices=_optional_npz_value(samples.landmark_sample_pair_indices),
            landmark_query_xy=_optional_npz_value(samples.landmark_query_xy),
            landmark_reference_xy=_optional_npz_value(samples.landmark_reference_xy),
            landmark_track_ids=_optional_npz_value(samples.landmark_track_ids),
            landmark_track_xyz=_optional_npz_value(samples.landmark_track_xyz),
            landmark_support_view_counts=_optional_npz_value(samples.landmark_support_view_counts),
            landmark_known_positive_offsets=_optional_npz_value(samples.landmark_known_positive_offsets),
            landmark_known_positive_track_ids=_optional_npz_value(samples.landmark_known_positive_track_ids),
            landmark_strict_positive_offsets=_optional_npz_value(samples.landmark_strict_positive_offsets),
            landmark_strict_positive_track_ids=_optional_npz_value(samples.landmark_strict_positive_track_ids),
            landmark_coherent_hard_negative_offsets=_optional_npz_value(samples.landmark_coherent_hard_negative_offsets),
            landmark_coherent_hard_negative_track_ids=_optional_npz_value(samples.landmark_coherent_hard_negative_track_ids),
            landmark_coherent_hard_negative_mode_ids=_optional_npz_value(samples.landmark_coherent_hard_negative_mode_ids),
            query_repeatability_targets=_optional_npz_value(samples.query_repeatability_targets),
            render_repeatability_targets=_optional_npz_value(samples.render_repeatability_targets),
        )
        return
    _save_npz(
        output,
        compressed=compressed,
        format=np.asarray([_JOINT_FORMAT], dtype=object),
        query_features=base.query_features,
        render_features=base.render_features,
        query_offset_labels=base.query_offset_labels,
        render_offset_labels=base.render_offset_labels,
        query_offset_soft_labels=_optional_npz_value(base.query_offset_soft_labels),
        render_offset_soft_labels=_optional_npz_value(base.render_offset_soft_labels),
        sample_confidence_targets=_optional_npz_value(base.sample_confidence_targets),
        sample_uncertainty_px=_optional_npz_value(base.sample_uncertainty_px),
        negative_render_features=base.negative_render_features,
        roundtrip_errors_px=base.roundtrip_errors_px,
        coarse_metadata=np.asarray([dict(base.metadata or {})], dtype=object),
        query_feature_maps=_optional_npz_value(samples.query_feature_maps),
        render_feature_maps=_optional_npz_value(samples.render_feature_maps),
        query_heatmap_targets=_optional_npz_value(samples.query_heatmap_targets),
        render_heatmap_targets=_optional_npz_value(samples.render_heatmap_targets),
        sample_pair_indices=_optional_npz_value(samples.sample_pair_indices),
        query_cell_indices=_optional_npz_value(samples.query_cell_indices),
        render_cell_indices=_optional_npz_value(samples.render_cell_indices),
        fine_sample_pair_indices=_optional_npz_value(samples.fine_sample_pair_indices),
        fine_query_cell_indices=_optional_npz_value(samples.fine_query_cell_indices),
        fine_render_cell_indices=_optional_npz_value(samples.fine_render_cell_indices),
        fine_query_offset_labels=_optional_npz_value(samples.fine_query_offset_labels),
        fine_render_offset_labels=_optional_npz_value(samples.fine_render_offset_labels),
        fine_query_offset_soft_labels=_optional_npz_value(samples.fine_query_offset_soft_labels),
        fine_render_offset_soft_labels=_optional_npz_value(samples.fine_render_offset_soft_labels),
        fine_query_xy=_optional_npz_value(samples.fine_query_xy),
        fine_render_xy=_optional_npz_value(samples.fine_render_xy),
        fine_render_depth=_optional_npz_value(samples.fine_render_depth),
        fine_support_view_count=_optional_npz_value(samples.fine_support_view_count),
        fine_validity_weight=_optional_npz_value(samples.fine_validity_weight),
        query_rgb_images=_optional_npz_value(samples.query_rgb_images),
        render_rgb_images=_optional_npz_value(samples.render_rgb_images),
        query_rgb_keypoint_labels=_optional_npz_value(samples.query_rgb_keypoint_labels),
        render_rgb_keypoint_labels=_optional_npz_value(samples.render_rgb_keypoint_labels),
        pair_type_ids=_optional_npz_value(samples.pair_type_ids),
        pair_type_names=_optional_npz_value(samples.pair_type_names),
        pair_query_ids=_optional_npz_value(samples.pair_query_ids),
        pair_split_names=_optional_npz_value(samples.pair_split_names),
        pair_candidate_ids=_optional_npz_value(samples.pair_candidate_ids),
        pair_translation_errors_m=_optional_npz_value(samples.pair_translation_errors_m),
        pair_rotation_errors_deg=_optional_npz_value(samples.pair_rotation_errors_deg),
        pair_query_image_sizes=_optional_npz_value(samples.pair_query_image_sizes),
        pair_reference_image_sizes=_optional_npz_value(samples.pair_reference_image_sizes),
        sample_no_match_labels=_optional_npz_value(samples.sample_no_match_labels),
        sample_ignore_mask=_optional_npz_value(samples.sample_ignore_mask),
        sample_confidence_ignore_mask=_optional_npz_value(samples.sample_confidence_ignore_mask),
        sample_track_ids=_optional_npz_value(samples.sample_track_ids),
        sample_track_xyz=_optional_npz_value(samples.sample_track_xyz),
        landmark_sample_pair_indices=_optional_npz_value(samples.landmark_sample_pair_indices),
        landmark_query_xy=_optional_npz_value(samples.landmark_query_xy),
        landmark_reference_xy=_optional_npz_value(samples.landmark_reference_xy),
        landmark_track_ids=_optional_npz_value(samples.landmark_track_ids),
        landmark_track_xyz=_optional_npz_value(samples.landmark_track_xyz),
        landmark_support_view_counts=_optional_npz_value(samples.landmark_support_view_counts),
        landmark_known_positive_offsets=_optional_npz_value(samples.landmark_known_positive_offsets),
        landmark_known_positive_track_ids=_optional_npz_value(samples.landmark_known_positive_track_ids),
        landmark_strict_positive_offsets=_optional_npz_value(samples.landmark_strict_positive_offsets),
        landmark_strict_positive_track_ids=_optional_npz_value(samples.landmark_strict_positive_track_ids),
        landmark_coherent_hard_negative_offsets=_optional_npz_value(samples.landmark_coherent_hard_negative_offsets),
        landmark_coherent_hard_negative_track_ids=_optional_npz_value(samples.landmark_coherent_hard_negative_track_ids),
        landmark_coherent_hard_negative_mode_ids=_optional_npz_value(samples.landmark_coherent_hard_negative_mode_ids),
        query_repeatability_targets=_optional_npz_value(samples.query_repeatability_targets),
        render_repeatability_targets=_optional_npz_value(samples.render_repeatability_targets),
    )


def load_matcha_joint_training_set_npz(path: Path) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
    """Load a full MATCHA joint training cache."""

    with np.load(Path(path), allow_pickle=True) as data:
        fmt = str(data["format"][0]) if "format" in data else ""
        if fmt == _JOINT_INDEX_FORMAT:
            coarse_metadata = dict(data["coarse_metadata"][0]) if "coarse_metadata" in data else {}
            query_feature_maps = _optional_loaded(data, "query_feature_maps")
            render_feature_maps = _optional_loaded(data, "render_feature_maps")
            query_cell_indices = _optional_loaded(data, "query_cell_indices")
            render_cell_indices = _optional_loaded(data, "render_cell_indices")
            sample_pair_indices = _optional_loaded(data, "sample_pair_indices")
            if query_feature_maps is None or render_feature_maps is None:
                raise ValueError("index-only joint cache requires query/render feature maps")
            if query_cell_indices is None or render_cell_indices is None:
                raise ValueError("index-only joint cache requires query/render cell indices")
            base = IndexOnlyCoarseFineRows(
                query_feature_maps=query_feature_maps,
                render_feature_maps=render_feature_maps,
                query_cell_indices=query_cell_indices,
                render_cell_indices=render_cell_indices,
                negative_render_indices=np.asarray(data["negative_render_indices"], dtype=np.int64),
                query_offset_labels=np.asarray(data["query_offset_labels"], dtype=np.int64),
                render_offset_labels=np.asarray(data["render_offset_labels"], dtype=np.int64),
                roundtrip_errors_px=np.asarray(data["roundtrip_errors_px"], dtype=np.float32),
                query_offset_soft_labels=_optional_loaded(data, "query_offset_soft_labels"),
                render_offset_soft_labels=_optional_loaded(data, "render_offset_soft_labels"),
                sample_confidence_targets=_optional_loaded(data, "sample_confidence_targets"),
                sample_uncertainty_px=_optional_loaded(data, "sample_uncertainty_px"),
                sample_pair_indices=sample_pair_indices,
                metadata=coarse_metadata,
            )
            samples = MatchaJointTrainingSet(
                coarse_fine_samples=base,
                query_feature_maps=query_feature_maps,
                render_feature_maps=render_feature_maps,
                query_heatmap_targets=_optional_loaded(data, "query_heatmap_targets"),
                render_heatmap_targets=_optional_loaded(data, "render_heatmap_targets"),
                sample_pair_indices=sample_pair_indices,
                query_cell_indices=query_cell_indices,
                render_cell_indices=render_cell_indices,
                fine_sample_pair_indices=_optional_loaded(data, "fine_sample_pair_indices"),
                fine_query_cell_indices=_optional_loaded(data, "fine_query_cell_indices"),
                fine_render_cell_indices=_optional_loaded(data, "fine_render_cell_indices"),
                fine_query_offset_labels=_optional_loaded(data, "fine_query_offset_labels"),
                fine_render_offset_labels=_optional_loaded(data, "fine_render_offset_labels"),
                fine_query_offset_soft_labels=_optional_loaded(data, "fine_query_offset_soft_labels"),
                fine_render_offset_soft_labels=_optional_loaded(data, "fine_render_offset_soft_labels"),
                fine_query_xy=_optional_loaded(data, "fine_query_xy"),
                fine_render_xy=_optional_loaded(data, "fine_render_xy"),
                fine_render_depth=_optional_loaded(data, "fine_render_depth"),
                fine_support_view_count=_optional_loaded(data, "fine_support_view_count"),
                fine_validity_weight=_optional_loaded(data, "fine_validity_weight"),
                query_rgb_images=_optional_loaded(data, "query_rgb_images"),
                render_rgb_images=_optional_loaded(data, "render_rgb_images"),
                query_rgb_keypoint_labels=_optional_loaded(data, "query_rgb_keypoint_labels"),
                render_rgb_keypoint_labels=_optional_loaded(data, "render_rgb_keypoint_labels"),
                pair_type_ids=_optional_loaded(data, "pair_type_ids"),
                pair_type_names=_optional_loaded(data, "pair_type_names"),
                pair_query_ids=_optional_loaded(data, "pair_query_ids"),
                pair_split_names=_optional_loaded(data, "pair_split_names"),
                pair_candidate_ids=_optional_loaded(data, "pair_candidate_ids"),
                pair_translation_errors_m=_optional_loaded(data, "pair_translation_errors_m"),
                pair_rotation_errors_deg=_optional_loaded(data, "pair_rotation_errors_deg"),
                pair_query_image_sizes=_optional_loaded(data, "pair_query_image_sizes"),
                pair_reference_image_sizes=_optional_loaded(data, "pair_reference_image_sizes"),
                sample_no_match_labels=_optional_loaded(data, "sample_no_match_labels"),
                sample_ignore_mask=_optional_loaded(data, "sample_ignore_mask"),
                sample_confidence_ignore_mask=_optional_loaded(data, "sample_confidence_ignore_mask"),
                sample_track_ids=_optional_loaded(data, "sample_track_ids"),
                sample_track_xyz=_optional_loaded(data, "sample_track_xyz"),
                landmark_sample_pair_indices=_optional_loaded(data, "landmark_sample_pair_indices"),
                landmark_query_xy=_optional_loaded(data, "landmark_query_xy"),
                landmark_reference_xy=_optional_loaded(data, "landmark_reference_xy"),
                landmark_track_ids=_optional_loaded(data, "landmark_track_ids"),
                landmark_track_xyz=_optional_loaded(data, "landmark_track_xyz"),
                landmark_support_view_counts=_optional_loaded(data, "landmark_support_view_counts"),
                landmark_known_positive_offsets=_optional_loaded(data, "landmark_known_positive_offsets"),
                landmark_known_positive_track_ids=_optional_loaded(data, "landmark_known_positive_track_ids"),
                landmark_strict_positive_offsets=_optional_loaded(data, "landmark_strict_positive_offsets"),
                landmark_strict_positive_track_ids=_optional_loaded(data, "landmark_strict_positive_track_ids"),
                landmark_coherent_hard_negative_offsets=_optional_loaded(data, "landmark_coherent_hard_negative_offsets"),
                landmark_coherent_hard_negative_track_ids=_optional_loaded(data, "landmark_coherent_hard_negative_track_ids"),
                landmark_coherent_hard_negative_mode_ids=_optional_loaded(data, "landmark_coherent_hard_negative_mode_ids"),
                query_repeatability_targets=_optional_loaded(data, "query_repeatability_targets"),
                render_repeatability_targets=_optional_loaded(data, "render_repeatability_targets"),
            )
            return samples, {"format": _JOINT_INDEX_FORMAT, "coarse_metadata": coarse_metadata}
        if fmt != _JOINT_FORMAT:
            raise ValueError(f"unsupported MATCHA joint training-set format in {path}")
        coarse_metadata = dict(data["coarse_metadata"][0]) if "coarse_metadata" in data else {}
        base = MatchaCoarseFineTrainingSet(
            query_features=np.asarray(data["query_features"], dtype=np.float32),
            render_features=np.asarray(data["render_features"], dtype=np.float32),
            query_offset_labels=np.asarray(data["query_offset_labels"], dtype=np.int64),
            render_offset_labels=np.asarray(data["render_offset_labels"], dtype=np.int64),
            negative_render_features=np.asarray(data["negative_render_features"], dtype=np.float32),
            roundtrip_errors_px=np.asarray(data["roundtrip_errors_px"], dtype=np.float32),
            query_offset_soft_labels=_optional_loaded(data, "query_offset_soft_labels"),
            render_offset_soft_labels=_optional_loaded(data, "render_offset_soft_labels"),
            sample_confidence_targets=_optional_loaded(data, "sample_confidence_targets"),
            sample_uncertainty_px=_optional_loaded(data, "sample_uncertainty_px"),
            metadata=coarse_metadata,
        )
        samples = MatchaJointTrainingSet(
            coarse_fine_samples=base,
            query_feature_maps=_optional_loaded(data, "query_feature_maps"),
            render_feature_maps=_optional_loaded(data, "render_feature_maps"),
            query_heatmap_targets=_optional_loaded(data, "query_heatmap_targets"),
            render_heatmap_targets=_optional_loaded(data, "render_heatmap_targets"),
            sample_pair_indices=_optional_loaded(data, "sample_pair_indices"),
            query_cell_indices=_optional_loaded(data, "query_cell_indices"),
            render_cell_indices=_optional_loaded(data, "render_cell_indices"),
            fine_sample_pair_indices=_optional_loaded(data, "fine_sample_pair_indices"),
            fine_query_cell_indices=_optional_loaded(data, "fine_query_cell_indices"),
            fine_render_cell_indices=_optional_loaded(data, "fine_render_cell_indices"),
            fine_query_offset_labels=_optional_loaded(data, "fine_query_offset_labels"),
            fine_render_offset_labels=_optional_loaded(data, "fine_render_offset_labels"),
            fine_query_offset_soft_labels=_optional_loaded(data, "fine_query_offset_soft_labels"),
            fine_render_offset_soft_labels=_optional_loaded(data, "fine_render_offset_soft_labels"),
            fine_query_xy=_optional_loaded(data, "fine_query_xy"),
            fine_render_xy=_optional_loaded(data, "fine_render_xy"),
            fine_render_depth=_optional_loaded(data, "fine_render_depth"),
            fine_support_view_count=_optional_loaded(data, "fine_support_view_count"),
            fine_validity_weight=_optional_loaded(data, "fine_validity_weight"),
            query_rgb_images=_optional_loaded(data, "query_rgb_images"),
            render_rgb_images=_optional_loaded(data, "render_rgb_images"),
            query_rgb_keypoint_labels=_optional_loaded(data, "query_rgb_keypoint_labels"),
            render_rgb_keypoint_labels=_optional_loaded(data, "render_rgb_keypoint_labels"),
            pair_type_ids=_optional_loaded(data, "pair_type_ids"),
            pair_type_names=_optional_loaded(data, "pair_type_names"),
            pair_query_ids=_optional_loaded(data, "pair_query_ids"),
            pair_split_names=_optional_loaded(data, "pair_split_names"),
            pair_candidate_ids=_optional_loaded(data, "pair_candidate_ids"),
            pair_translation_errors_m=_optional_loaded(data, "pair_translation_errors_m"),
            pair_rotation_errors_deg=_optional_loaded(data, "pair_rotation_errors_deg"),
            pair_query_image_sizes=_optional_loaded(data, "pair_query_image_sizes"),
            pair_reference_image_sizes=_optional_loaded(data, "pair_reference_image_sizes"),
            sample_no_match_labels=_optional_loaded(data, "sample_no_match_labels"),
            sample_ignore_mask=_optional_loaded(data, "sample_ignore_mask"),
            sample_confidence_ignore_mask=_optional_loaded(data, "sample_confidence_ignore_mask"),
            sample_track_ids=_optional_loaded(data, "sample_track_ids"),
            sample_track_xyz=_optional_loaded(data, "sample_track_xyz"),
            landmark_sample_pair_indices=_optional_loaded(data, "landmark_sample_pair_indices"),
            landmark_query_xy=_optional_loaded(data, "landmark_query_xy"),
            landmark_reference_xy=_optional_loaded(data, "landmark_reference_xy"),
            landmark_track_ids=_optional_loaded(data, "landmark_track_ids"),
            landmark_track_xyz=_optional_loaded(data, "landmark_track_xyz"),
            landmark_support_view_counts=_optional_loaded(data, "landmark_support_view_counts"),
            landmark_known_positive_offsets=_optional_loaded(data, "landmark_known_positive_offsets"),
            landmark_known_positive_track_ids=_optional_loaded(data, "landmark_known_positive_track_ids"),
            landmark_strict_positive_offsets=_optional_loaded(data, "landmark_strict_positive_offsets"),
            landmark_strict_positive_track_ids=_optional_loaded(data, "landmark_strict_positive_track_ids"),
            landmark_coherent_hard_negative_offsets=_optional_loaded(data, "landmark_coherent_hard_negative_offsets"),
            landmark_coherent_hard_negative_track_ids=_optional_loaded(data, "landmark_coherent_hard_negative_track_ids"),
            landmark_coherent_hard_negative_mode_ids=_optional_loaded(data, "landmark_coherent_hard_negative_mode_ids"),
            query_repeatability_targets=_optional_loaded(data, "query_repeatability_targets"),
            render_repeatability_targets=_optional_loaded(data, "render_repeatability_targets"),
        )
    return samples, {"format": _JOINT_FORMAT, "coarse_metadata": coarse_metadata}


def save_matcha_joint_training_set_manifest(
    items: list[MatchaJointTrainingSet] | tuple[MatchaJointTrainingSet, ...],
    manifest_path: Path,
    *,
    shard_dir: Path | None = None,
) -> dict[str, object]:
    """Save joint training sets as multiple shards plus a lightweight manifest."""

    values = list(items)
    if not values:
        raise ValueError("at least one joint training set is required")
    manifest = Path(manifest_path)
    root = manifest.parent
    output_dir = Path(shard_dir) if shard_dir is not None else root / "shards"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    split_counts: dict[str, int] = {}
    pair_type_counts: dict[str, int] = {}

    def first_string(value: np.ndarray | None, default: str = "") -> str:
        if value is None:
            return default
        arr = np.asarray(value, dtype=object).reshape(-1)
        if arr.shape[0] == 0:
            return default
        return str(arr[0])

    for index, item in enumerate(values):
        shard_path = output_dir / f"shard_{int(index):05d}.npz"
        save_matcha_joint_training_set_npz(item, shard_path)
        try:
            relative_path = shard_path.relative_to(root)
        except ValueError:
            relative_path = shard_path
        pair_count = int(item.query_feature_maps.shape[0]) if item.query_feature_maps is not None else 0
        split_name = first_string(item.pair_split_names)
        pair_type = first_string(item.pair_type_names)
        if split_name:
            split_counts[split_name] = int(split_counts.get(split_name, 0) + max(pair_count, 1))
        if pair_type:
            pair_type_counts[pair_type] = int(pair_type_counts.get(pair_type, 0) + max(pair_count, 1))
        shards.append(
            {
                "path": str(relative_path),
                "sample_count": int(item.coarse_fine_samples.sample_count),
                "pair_count": pair_count,
                "query_id": first_string(item.pair_query_ids),
                "split": split_name,
                "pair_type": pair_type,
                "candidate_id": first_string(item.pair_candidate_ids),
            }
        )
    metadata = {
        "format": _JOINT_MANIFEST_FORMAT,
        "shard_count": int(len(shards)),
        "sample_count": int(sum(int(item["sample_count"]) for item in shards)),
        "split_counts": split_counts,
        "pair_type_counts": pair_type_counts,
        "shards": shards,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_matcha_joint_training_set_manifest(path: Path) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
    """Load and merge a sharded MATCHA joint training cache manifest."""

    manifest_path = Path(path)
    metadata = json.loads(manifest_path.read_text())
    if str(metadata.get("format", "")) != _JOINT_MANIFEST_FORMAT:
        raise ValueError(f"unsupported MATCHA joint training manifest format in {path}")
    shards = list(metadata.get("shards", []))
    if not shards:
        raise ValueError("MATCHA joint training manifest contains no shards")
    loaded = []
    for shard in shards:
        shard_path = Path(str(shard["path"]))
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        sample, _sample_metadata = load_matcha_joint_training_set_npz(shard_path)
        loaded.append(sample)
    merged = merge_matcha_joint_training_sets(loaded)
    output_metadata = dict(metadata)
    output_metadata["format"] = _JOINT_MANIFEST_FORMAT
    output_metadata["shard_count"] = int(len(shards))
    output_metadata["sample_count"] = int(merged.coarse_fine_samples.sample_count)
    return merged, output_metadata


def _materialize_index_only_joint_training_set(samples: MatchaJointTrainingSet) -> MatchaJointTrainingSet:
    """Materialize only bounded coarse rows so a few lazy image pairs can form one GPU batch."""

    rows = samples.coarse_fine_samples
    if not isinstance(rows, IndexOnlyCoarseFineRows):
        return samples
    base = MatchaCoarseFineTrainingSet(
        query_features=np.asarray(rows.query_features[:], dtype=np.float32),
        render_features=np.asarray(rows.render_features[:], dtype=np.float32),
        query_offset_labels=np.asarray(rows.query_offset_labels, dtype=np.int64),
        render_offset_labels=np.asarray(rows.render_offset_labels, dtype=np.int64),
        negative_render_features=np.asarray(rows.negative_render_features[:], dtype=np.float32),
        roundtrip_errors_px=np.asarray(rows.roundtrip_errors_px, dtype=np.float32),
        query_offset_soft_labels=(
            None
            if rows.query_offset_soft_labels is None
            else np.asarray(rows.query_offset_soft_labels, dtype=np.float32)
        ),
        render_offset_soft_labels=(
            None
            if rows.render_offset_soft_labels is None
            else np.asarray(rows.render_offset_soft_labels, dtype=np.float32)
        ),
        sample_confidence_targets=(
            None
            if rows.sample_confidence_targets is None
            else np.asarray(rows.sample_confidence_targets, dtype=np.float32)
        ),
        sample_uncertainty_px=(
            None
            if rows.sample_uncertainty_px is None
            else np.asarray(rows.sample_uncertainty_px, dtype=np.float32)
        ),
        metadata={**dict(rows.metadata), "bounded_lazy_materialization": True},
    )
    return replace(samples, coarse_fine_samples=base)


def merge_matcha_joint_training_sets(
    items: list[MatchaJointTrainingSet] | tuple[MatchaJointTrainingSet, ...],
) -> MatchaJointTrainingSet:
    """Merge one-pair joint caches into a multi-pair cache."""

    values = list(items)
    if not values:
        raise ValueError("at least one joint training set is required")
    if any(isinstance(item.coarse_fine_samples, IndexOnlyCoarseFineRows) for item in values):
        raise ValueError("index-only joint caches must be trained with lazy manifest loading")
    input_dim = values[0].coarse_fine_samples.input_dim
    negative_count = int(values[0].coarse_fine_samples.negative_render_features.shape[1])
    for item in values:
        base = item.coarse_fine_samples
        if int(base.input_dim) != int(input_dim):
            raise ValueError("cannot merge joint sets with different input dimensions")
        if int(base.negative_render_features.shape[1]) != int(negative_count):
            raise ValueError("cannot merge joint sets with different negative counts")

    def stack_base_optional(name: str) -> np.ndarray | None:
        arrays = [getattr(item.coarse_fine_samples, name) for item in values]
        if all(array is None for array in arrays):
            return None
        if any(array is None for array in arrays):
            raise ValueError(f"cannot merge partially missing coarse_fine_samples.{name}")
        return np.concatenate([np.asarray(array) for array in arrays], axis=0)

    base = MatchaCoarseFineTrainingSet(
        query_features=np.concatenate([item.coarse_fine_samples.query_features for item in values], axis=0),
        render_features=np.concatenate([item.coarse_fine_samples.render_features for item in values], axis=0),
        query_offset_labels=np.concatenate([item.coarse_fine_samples.query_offset_labels for item in values], axis=0),
        render_offset_labels=np.concatenate([item.coarse_fine_samples.render_offset_labels for item in values], axis=0),
        negative_render_features=np.concatenate([item.coarse_fine_samples.negative_render_features for item in values], axis=0),
        roundtrip_errors_px=np.concatenate([item.coarse_fine_samples.roundtrip_errors_px for item in values], axis=0),
        query_offset_soft_labels=stack_base_optional("query_offset_soft_labels"),
        render_offset_soft_labels=stack_base_optional("render_offset_soft_labels"),
        sample_confidence_targets=stack_base_optional("sample_confidence_targets"),
        sample_uncertainty_px=stack_base_optional("sample_uncertainty_px"),
        metadata={"merged": True, "source_count": int(len(values))},
    )

    def stack_optional(name: str) -> np.ndarray | None:
        arrays = [getattr(item, name) for item in values]
        if all(array is None for array in arrays):
            return None
        if any(array is None for array in arrays):
            raise ValueError(f"cannot merge partially missing {name}")
        return np.concatenate([np.asarray(array) for array in arrays], axis=0)

    pair_indices = []
    query_indices = []
    render_indices = []
    fine_arrays = [item.fine_sample_pair_indices for item in values]
    landmark_arrays = [item.landmark_sample_pair_indices for item in values]
    if all(array is None for array in landmark_arrays):
        landmark_pair_indices = None
        landmark_query_xy = None
        landmark_reference_xy = None
        landmark_track_ids = None
        landmark_track_xyz = None
        landmark_support_view_counts = None
        landmark_known_positive_offsets = None
        landmark_known_positive_track_ids = None
        landmark_strict_positive_offsets = None
        landmark_strict_positive_track_ids = None
        landmark_coherent_hard_negative_offsets = None
        landmark_coherent_hard_negative_track_ids = None
        landmark_coherent_hard_negative_mode_ids = None
    elif any(array is None for array in landmark_arrays):
        raise ValueError("cannot merge partially missing landmark retrieval supervision")
    else:
        landmark_pair_indices = np.concatenate(
            [
                np.asarray(item.landmark_sample_pair_indices, dtype=np.int64) + int(pair_idx)
                for pair_idx, item in enumerate(values)
            ],
            axis=0,
        )
        landmark_query_xy = stack_optional("landmark_query_xy")
        landmark_reference_xy = stack_optional("landmark_reference_xy")
        landmark_track_ids = stack_optional("landmark_track_ids")
        landmark_track_xyz = stack_optional("landmark_track_xyz")
        landmark_support_view_counts = stack_optional("landmark_support_view_counts")
        known_offsets = [item.landmark_known_positive_offsets for item in values]
        known_track_ids = [item.landmark_known_positive_track_ids for item in values]
        if all(item is None for item in [*known_offsets, *known_track_ids]):
            landmark_known_positive_offsets = None
            landmark_known_positive_track_ids = None
        elif any(item is None for item in [*known_offsets, *known_track_ids]):
            raise ValueError("cannot merge partially missing landmark known-positive CSR")
        else:
            known_counts = np.concatenate(
                [np.diff(np.asarray(item, dtype=np.int64).reshape(-1)) for item in known_offsets],
                axis=0,
            )
            landmark_known_positive_offsets = np.zeros(
                (known_counts.size + 1,), dtype=np.int64
            )
            landmark_known_positive_offsets[1:] = np.cumsum(known_counts)
            landmark_known_positive_track_ids = np.concatenate(
                [np.asarray(item, dtype=np.int64).reshape(-1) for item in known_track_ids],
                axis=0,
            )
        strict_offsets = [item.landmark_strict_positive_offsets for item in values]
        strict_track_ids = [item.landmark_strict_positive_track_ids for item in values]
        if all(item is None for item in [*strict_offsets, *strict_track_ids]):
            landmark_strict_positive_offsets = None
            landmark_strict_positive_track_ids = None
        elif any(item is None for item in [*strict_offsets, *strict_track_ids]):
            raise ValueError("cannot merge partially missing landmark strict-positive CSR")
        else:
            strict_counts = np.concatenate(
                [np.diff(np.asarray(item, dtype=np.int64).reshape(-1)) for item in strict_offsets],
                axis=0,
            )
            landmark_strict_positive_offsets = np.zeros(
                (strict_counts.size + 1,), dtype=np.int64
            )
            landmark_strict_positive_offsets[1:] = np.cumsum(strict_counts)
            landmark_strict_positive_track_ids = np.concatenate(
                [np.asarray(item, dtype=np.int64).reshape(-1) for item in strict_track_ids],
                axis=0,
            )
        coherent_offsets = [
            item.landmark_coherent_hard_negative_offsets for item in values
        ]
        coherent_track_ids = [
            item.landmark_coherent_hard_negative_track_ids for item in values
        ]
        coherent_mode_ids = [
            item.landmark_coherent_hard_negative_mode_ids for item in values
        ]
        if all(item is None for item in [*coherent_offsets, *coherent_track_ids]):
            landmark_coherent_hard_negative_offsets = None
            landmark_coherent_hard_negative_track_ids = None
            landmark_coherent_hard_negative_mode_ids = None
        elif any(item is None for item in [*coherent_offsets, *coherent_track_ids]):
            raise ValueError(
                "cannot merge partially missing landmark coherent hard-negative CSR"
            )
        else:
            if any(item is None for item in coherent_mode_ids) and not all(
                item is None for item in coherent_mode_ids
            ):
                raise ValueError(
                    "cannot merge partially missing coherent hard-mode ids"
                )
            coherent_counts = np.concatenate(
                [
                    np.diff(np.asarray(item, dtype=np.int64).reshape(-1))
                    for item in coherent_offsets
                ],
                axis=0,
            )
            landmark_coherent_hard_negative_offsets = np.zeros(
                (coherent_counts.size + 1,), dtype=np.int64
            )
            landmark_coherent_hard_negative_offsets[1:] = np.cumsum(
                coherent_counts
            )
            landmark_coherent_hard_negative_track_ids = np.concatenate(
                [
                    np.asarray(item, dtype=np.int64).reshape(-1)
                    for item in coherent_track_ids
                ],
                axis=0,
            )
            landmark_coherent_hard_negative_mode_ids = (
                None
                if all(item is None for item in coherent_mode_ids)
                else np.concatenate(
                    [
                        np.asarray(item, dtype=np.int64).reshape(-1)
                        for item in coherent_mode_ids
                    ],
                    axis=0,
                )
            )
    if all(array is None for array in fine_arrays):
        fine_pair_indices = None
        fine_query_indices = None
        fine_render_indices = None
        fine_query_labels = None
        fine_render_labels = None
        fine_query_soft = None
        fine_render_soft = None
        fine_query_xy = None
        fine_render_xy = None
        fine_render_depth = None
        fine_support_view_count = None
        fine_validity_weight = None
    elif any(array is None for array in fine_arrays):
        raise ValueError("cannot merge partially missing dense fine supervision")
    else:
        fine_pair_indices = np.concatenate(
            [
                np.asarray(item.fine_sample_pair_indices, dtype=np.int64) + int(pair_idx)
                for pair_idx, item in enumerate(values)
            ],
            axis=0,
        )
        fine_query_indices = np.concatenate([np.asarray(item.fine_query_cell_indices, dtype=np.int64) for item in values], axis=0)
        fine_render_indices = np.concatenate([np.asarray(item.fine_render_cell_indices, dtype=np.int64) for item in values], axis=0)
        fine_query_labels = np.concatenate([np.asarray(item.fine_query_offset_labels, dtype=np.int64) for item in values], axis=0)
        fine_render_labels = np.concatenate([np.asarray(item.fine_render_offset_labels, dtype=np.int64) for item in values], axis=0)
        fine_query_soft = stack_optional("fine_query_offset_soft_labels")
        fine_render_soft = stack_optional("fine_render_offset_soft_labels")
        fine_query_xy = stack_optional("fine_query_xy")
        fine_render_xy = stack_optional("fine_render_xy")
        fine_render_depth = stack_optional("fine_render_depth")
        fine_support_view_count = stack_optional("fine_support_view_count")
        fine_validity_weight = stack_optional("fine_validity_weight")
    for pair_idx, item in enumerate(values):
        count = int(item.coarse_fine_samples.sample_count)
        pair_indices.append(np.full((count,), int(pair_idx), dtype=np.int64))
        if item.query_cell_indices is not None and item.render_cell_indices is not None:
            query_indices.append(np.asarray(item.query_cell_indices, dtype=np.int64))
            render_indices.append(np.asarray(item.render_cell_indices, dtype=np.int64))
    if query_indices and len(query_indices) != len(values):
        raise ValueError("cannot merge partially missing cell indices")

    return MatchaJointTrainingSet(
        coarse_fine_samples=base,
        query_feature_maps=stack_optional("query_feature_maps"),
        render_feature_maps=stack_optional("render_feature_maps"),
        query_heatmap_targets=stack_optional("query_heatmap_targets"),
        render_heatmap_targets=stack_optional("render_heatmap_targets"),
        sample_pair_indices=np.concatenate(pair_indices, axis=0),
        query_cell_indices=np.concatenate(query_indices, axis=0) if query_indices else None,
        render_cell_indices=np.concatenate(render_indices, axis=0) if render_indices else None,
        fine_sample_pair_indices=fine_pair_indices,
        fine_query_cell_indices=fine_query_indices,
        fine_render_cell_indices=fine_render_indices,
        fine_query_offset_labels=fine_query_labels,
        fine_render_offset_labels=fine_render_labels,
        fine_query_offset_soft_labels=fine_query_soft,
        fine_render_offset_soft_labels=fine_render_soft,
        fine_query_xy=fine_query_xy,
        fine_render_xy=fine_render_xy,
        fine_render_depth=fine_render_depth,
        fine_support_view_count=fine_support_view_count,
        fine_validity_weight=fine_validity_weight,
        query_rgb_images=stack_optional("query_rgb_images"),
        render_rgb_images=stack_optional("render_rgb_images"),
        query_rgb_keypoint_labels=stack_optional("query_rgb_keypoint_labels"),
        render_rgb_keypoint_labels=stack_optional("render_rgb_keypoint_labels"),
        pair_type_ids=stack_optional("pair_type_ids"),
        pair_type_names=stack_optional("pair_type_names"),
        pair_query_ids=stack_optional("pair_query_ids"),
        pair_split_names=stack_optional("pair_split_names"),
        pair_candidate_ids=stack_optional("pair_candidate_ids"),
        pair_translation_errors_m=stack_optional("pair_translation_errors_m"),
        pair_rotation_errors_deg=stack_optional("pair_rotation_errors_deg"),
        pair_query_image_sizes=stack_optional("pair_query_image_sizes"),
        pair_reference_image_sizes=stack_optional("pair_reference_image_sizes"),
        sample_no_match_labels=stack_optional("sample_no_match_labels"),
        sample_ignore_mask=stack_optional("sample_ignore_mask"),
        sample_confidence_ignore_mask=stack_optional("sample_confidence_ignore_mask"),
        sample_track_ids=stack_optional("sample_track_ids"),
        sample_track_xyz=stack_optional("sample_track_xyz"),
        landmark_sample_pair_indices=landmark_pair_indices,
        landmark_query_xy=landmark_query_xy,
        landmark_reference_xy=landmark_reference_xy,
        landmark_track_ids=landmark_track_ids,
        landmark_track_xyz=landmark_track_xyz,
        landmark_support_view_counts=landmark_support_view_counts,
        landmark_known_positive_offsets=landmark_known_positive_offsets,
        landmark_known_positive_track_ids=landmark_known_positive_track_ids,
        landmark_strict_positive_offsets=landmark_strict_positive_offsets,
        landmark_strict_positive_track_ids=landmark_strict_positive_track_ids,
        landmark_coherent_hard_negative_offsets=landmark_coherent_hard_negative_offsets,
        landmark_coherent_hard_negative_track_ids=landmark_coherent_hard_negative_track_ids,
        landmark_coherent_hard_negative_mode_ids=landmark_coherent_hard_negative_mode_ids,
        query_repeatability_targets=stack_optional("query_repeatability_targets"),
        render_repeatability_targets=stack_optional("render_repeatability_targets"),
    )
