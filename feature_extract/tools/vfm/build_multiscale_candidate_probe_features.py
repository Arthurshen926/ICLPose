"""Export frozen, target-free multiscale appearance evidence for top-L tracks.

The artifact is deliberately an input to a later train-only classifier, not a
pose scorer.  It contains no query pose, GT residual, landmark visibility, or
rendered feature.  Each feature compares a detector query node to real
candidate support observations selected by the fixed maplet support index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _verification_points_for_query,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.local_maplet_matching import (
    LocalMapletSupportIndex,
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    canonical_rows_for_track_candidates,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    LocalContextNodes,
    MultiscaleContextSummaries,
    assemble_local_context_nodes,
    cosine_similarity,
    multiscale_per_view_feature_vector,
    summarize_multiscale_context,
)
from feature_extract.vfm.localization.radio_final_context import (
    RadioFinalContextPcaCache,
    load_radio_final_context_pca_cache,
)
from feature_extract.vfm.localization.radio_intermediate_context import (
    RadioIntermediateContextCache,
    load_radio_intermediate_context_cache,
)


ARTIFACT_FORMAT = "multiscale_candidate_probe_features_v1"
SOURCE_ROW_SELECTION_VERSION = "heldout_detector_merit_after_target_free_fit_rows_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--query_context_cache", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--radio_intermediate_cache", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--verification_point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument("--support_view_count", type=int, default=2)
    parser.add_argument("--radius3_px", type=float, default=48.0)
    parser.add_argument("--radius5_px", type=float, default=96.0)
    parser.add_argument("--max_context_nodes", type=int, default=32)
    parser.add_argument("--duplicate_radius_px", type=float, default=2.0)
    parser.add_argument("--support_context_cache_size", type=int, default=4096)
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="diagnostic-only prefix limit; a production S1 artifact uses 0",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, object]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} has no metadata_json")
    return json.loads(str(np.asarray(data["metadata_json"]).item()))


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _load_required_arrays(
    path: Path,
    *,
    keys: Sequence[str],
    context: str,
    require_metadata: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object], set[str]]:
    with np.load(Path(path), allow_pickle=False) as data:
        missing = set(keys) - set(data.files)
        if missing:
            raise ValueError(f"{context} lacks arrays: {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in keys}
        metadata = (
            _metadata(data, context=context)
            if bool(require_metadata)
            else (
                {} if "metadata_json" not in data.files else _metadata(data, context=context)
            )
        )
        names = set(data.files)
    return arrays, metadata, names


def _split_lookup(split_payload: Mapping[str, object]) -> dict[str, str]:
    """Return a disjoint image-to-split mapping from the frozen split manifest."""

    lookup: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        values = split_payload.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split manifest has no non-empty {split} image list")
        for image_id in values:
            value = str(image_id)
            previous = lookup.setdefault(value, split)
            if previous != split:
                raise ValueError(f"query image appears in multiple splits: {value}")
    return lookup


def _validate_metadata_value(
    metadata: Mapping[str, object], *, key: str, expected: object, context: str
) -> None:
    actual = metadata.get(key)
    if str(actual) != str(expected):
        raise ValueError(
            f"{context} lineage mismatch for {key}: expected {expected!r}, got {actual!r}"
        )


def _load_target_free_candidate_rows(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load only the old target-free fit rows used to form held-out nodes."""

    with np.load(Path(path), allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError("candidate artifact with labels cannot select S1 rows")
        metadata = _metadata(data, context="candidate artifact")
        if bool(metadata.get("contains_ground_truth", True)) or str(
            metadata.get("supervision_mode", "")
        ) != "none_inference_only":
            raise ValueError("candidate artifact is not inference-only")
        if "selected_rows" not in data.files:
            raise ValueError("candidate artifact lacks selected_rows")
        rows = np.asarray(data["selected_rows"], dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0 or np.unique(rows).size != len(rows):
        raise ValueError("candidate artifact selected_rows are invalid")
    return rows, metadata


@dataclass
class _ImageDescriptorIndex:
    xy: np.ndarray
    alike: np.ndarray
    intermediate: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        self.alike = np.asarray(self.alike, dtype=np.float32)
        self.intermediate = np.asarray(self.intermediate, dtype=np.float32)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if (
            self.alike.ndim != 2
            or self.intermediate.ndim != 2
            or self.alike.shape[0] != len(self.xy)
            or self.intermediate.shape[0] != len(self.xy)
            or len(self.scores) != len(self.xy)
            or len(self.xy) == 0
        ):
            raise ValueError("image descriptor cache arrays are not node-aligned")
        if not np.isfinite(self.xy).all():
            raise ValueError("image descriptor cache has non-finite coordinates")
        if self.track_ids is not None:
            self.track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
            if self.track_ids.shape != (len(self.xy),):
                raise ValueError("support track IDs do not align with image nodes")
            if len(self.track_ids) > 1 and np.any(self.track_ids[1:] <= self.track_ids[:-1]):
                raise ValueError("support image tracks must be strictly sorted")
        self.tree = cKDTree(self.xy.astype(np.float64, copy=False))

    def local_context(
        self,
        *,
        anchor_xy: np.ndarray,
        anchor_alike: np.ndarray,
        anchor_intermediate: np.ndarray,
        radius_px: float,
        max_nodes: int,
        duplicate_radius_px: float,
    ) -> LocalContextNodes:
        indices = np.asarray(
            self.tree.query_ball_point(
                np.asarray(anchor_xy, dtype=np.float32).reshape(2),
                r=float(radius_px),
            ),
            dtype=np.int64,
        )
        # cKDTree does not promise a semantic order for ties.  Stable source
        # row order makes score ties reproducible before the local selector.
        indices.sort()
        return assemble_local_context_nodes(
            anchor_xy=anchor_xy,
            anchor_alike=anchor_alike,
            anchor_intermediate=anchor_intermediate,
            nearby_xy=self.xy[indices],
            nearby_alike=self.alike[indices],
            nearby_intermediate=self.intermediate[indices],
            nearby_scores=self.scores[indices],
            max_nodes=int(max_nodes),
            duplicate_radius_px=float(duplicate_radius_px),
        )

    def track_position(self, track_id: int) -> int | None:
        if self.track_ids is None:
            raise RuntimeError("query image nodes have no support-track lookup")
        position = int(np.searchsorted(self.track_ids, int(track_id)))
        if position >= len(self.track_ids) or int(self.track_ids[position]) != int(track_id):
            return None
        return position


@dataclass(frozen=True)
class _SupportAppearance:
    context3: LocalContextNodes
    context5: LocalContextNodes
    summaries: MultiscaleContextSummaries
    final_grid4: np.ndarray


class _SupportAppearanceCache:
    """Bounded cache of candidate-local support contexts, never query evidence."""

    def __init__(
        self,
        *,
        geometry: SupportObservationGeometryIndex,
        support_alike: np.ndarray,
        support_scores: np.ndarray,
        radio: RadioIntermediateContextCache,
        final_context: RadioFinalContextPcaCache,
        radius3_px: float,
        radius5_px: float,
        max_context_nodes: int,
        duplicate_radius_px: float,
        max_entries: int,
    ) -> None:
        if int(max_entries) <= 0:
            raise ValueError("support context cache capacity must be positive")
        self.geometry = geometry
        self.support_alike = np.asarray(support_alike, dtype=np.float32)
        self.support_scores = np.asarray(support_scores, dtype=np.float32).reshape(-1)
        self.radio = radio
        self.final_context = final_context
        self.radius3_px = float(radius3_px)
        self.radius5_px = float(radius5_px)
        self.max_context_nodes = int(max_context_nodes)
        self.duplicate_radius_px = float(duplicate_radius_px)
        self.max_entries = int(max_entries)
        self.image_nodes: dict[str, _ImageDescriptorIndex] = {}
        self.appearances: OrderedDict[tuple[int, str], _SupportAppearance] = OrderedDict()
        self.stats = {"image_index_build_count": 0, "appearance_cache_hit_count": 0, "appearance_cache_miss_count": 0}

    def _image_index(self, image_id: str) -> _ImageDescriptorIndex:
        key = str(image_id)
        cached = self.image_nodes.get(key)
        if cached is not None:
            return cached
        image_slice = self.geometry.image_slice(key)
        if image_slice.stop <= image_slice.start:
            raise KeyError(f"support geometry has no image {key!r}")
        source_rows = self.geometry.source_row_indices[image_slice]
        output = _ImageDescriptorIndex(
            xy=self.geometry.xy[image_slice],
            alike=self.support_alike[source_rows],
            intermediate=self.radio.support_descriptors[source_rows],
            scores=self.support_scores[source_rows],
            track_ids=self.geometry.track_ids[image_slice],
        )
        self.image_nodes[key] = output
        self.stats["image_index_build_count"] += 1
        return output

    def get(self, *, track_id: int, image_id: str) -> _SupportAppearance:
        key = (int(track_id), str(image_id))
        cached = self.appearances.get(key)
        if cached is not None:
            self.appearances.move_to_end(key)
            self.stats["appearance_cache_hit_count"] += 1
            return cached
        self.stats["appearance_cache_miss_count"] += 1
        nodes = self._image_index(str(image_id))
        anchor_index = nodes.track_position(int(track_id))
        if anchor_index is None:
            raise ValueError(
                f"support image {image_id!r} does not observe candidate track {track_id}"
            )
        anchor_xy = nodes.xy[anchor_index]
        context3 = nodes.local_context(
            anchor_xy=anchor_xy,
            anchor_alike=nodes.alike[anchor_index],
            anchor_intermediate=nodes.intermediate[anchor_index],
            radius_px=self.radius3_px,
            max_nodes=self.max_context_nodes,
            duplicate_radius_px=self.duplicate_radius_px,
        )
        context5 = nodes.local_context(
            anchor_xy=anchor_xy,
            anchor_alike=nodes.alike[anchor_index],
            anchor_intermediate=nodes.intermediate[anchor_index],
            radius_px=self.radius5_px,
            max_nodes=self.max_context_nodes,
            duplicate_radius_px=self.duplicate_radius_px,
        )
        output = _SupportAppearance(
            context3=context3,
            context5=context5,
            summaries=summarize_multiscale_context(
                context3,
                context5,
                radius3_px=self.radius3_px,
                radius5_px=self.radius5_px,
            ),
            final_grid4=self.final_context.node_grid4_descriptors(
                str(image_id), anchor_xy[None]
            )[0],
        )
        self.appearances[key] = output
        if len(self.appearances) > self.max_entries:
            self.appearances.popitem(last=False)
        return output


def _candidate_support_views(
    *,
    maplet: LocalMapletSupportIndex,
    canonical_row: int,
    query_id: str,
    support_view_count: int,
    view_is_usable: Callable[[str], bool],
) -> tuple[tuple[tuple[str, int], ...], int]:
    """Choose deterministic coverage-ranked support views with explicit fallback."""

    selected: list[tuple[str, int]] = []
    seen: set[str] = set()
    skipped = 0
    rows = np.asarray(maplet.support_image_indices[int(canonical_row)], dtype=np.int64)
    coverage = np.asarray(maplet.support_coverage_counts[int(canonical_row)], dtype=np.int64)
    for image_index, coverage_count in zip(rows.tolist(), coverage.tolist()):
        if int(image_index) < 0:
            continue
        image_id = str(maplet.support_image_ids[int(image_index)])
        if image_id == str(query_id) or image_id in seen or not view_is_usable(image_id):
            skipped += 1
            continue
        selected.append((image_id, int(coverage_count)))
        seen.add(image_id)
        if len(selected) == int(support_view_count):
            break
    return tuple(selected), int(skipped)


def _validate_inputs(
    *,
    proposal_path: Path,
    detector_path: Path,
    candidate_path: Path,
    query_context_path: Path,
    support_feature_path: Path,
    support_geometry_path: Path,
    bank_path: Path,
    maplet_path: Path,
    radio_path: Path,
    final_context_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    SupportObservationGeometryIndex,
    RadioIntermediateContextCache,
    RadioFinalContextPcaCache,
    object,
    LocalMapletSupportIndex,
    dict[str, object],
]:
    """Load and line up every target-free input before any feature is emitted."""

    proposals, _proposal_metadata, _ = _load_required_arrays(
        proposal_path,
        keys=("query_ids", "xy", "candidate_track_ids", "coarse_scores"),
        context="candidate proposals",
        require_metadata=False,
    )
    detector, detector_metadata, _ = _load_required_arrays(
        detector_path,
        keys=(
            "image_ids",
            "offsets",
            "xy",
            "local_descriptors",
            "global_descriptors",
            "detector_scores",
        ),
        context="detector query cache",
    )
    selected_rows, candidate_metadata = _load_target_free_candidate_rows(candidate_path)
    query_context, query_context_metadata, _ = _load_required_arrays(
        query_context_path,
        keys=("image_ids", "offsets", "xy", "local_descriptors", "detector_scores"),
        context="dense query context cache",
    )
    support, support_metadata, _ = _load_required_arrays(
        support_feature_path,
        keys=("track_ids", "descriptors", "detector_scores"),
        context="support observation feature cache",
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        support_geometry_path
    )
    radio = load_radio_intermediate_context_cache(
        radio_path,
        expected_metadata={
            "support_feature_cache_sha256": file_sha256_short(support_feature_path),
            "support_geometry_index_sha256": file_sha256_short(support_geometry_path),
            "query_anchor_cache_sha256": file_sha256_short(detector_path),
            "query_context_cache_sha256": file_sha256_short(query_context_path),
        },
    )
    final_context = load_radio_final_context_pca_cache(final_context_path)
    landmark_bank, bank_metadata = load_landmark_index_npz(bank_path)
    maplet, maplet_metadata = load_local_maplet_support_index_npz(maplet_path)

    proposal_rows = int(len(proposals["query_ids"]))
    offsets = np.asarray(detector["offsets"], dtype=np.int64)
    image_ids = np.asarray(detector["image_ids"]).astype(str)
    if offsets.shape != (len(image_ids) + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != proposal_rows:
        raise ValueError("detector offsets do not span proposal rows")
    expected_query_ids = np.repeat(image_ids, np.diff(offsets))
    if not np.array_equal(expected_query_ids, np.asarray(proposals["query_ids"]).astype(str)):
        raise ValueError("proposal query ownership differs from detector cache")
    if not np.array_equal(
        np.asarray(proposals["xy"], dtype=np.float32),
        np.asarray(detector["xy"], dtype=np.float32),
    ):
        raise ValueError("proposal and detector coordinates differ")
    if np.any(selected_rows < 0) or np.any(selected_rows >= proposal_rows):
        raise ValueError("candidate fit rows are outside detector proposals")
    _validate_metadata_value(
        candidate_metadata,
        key="proposals_sha256",
        expected=file_sha256_short(proposal_path),
        context="candidate artifact",
    )
    _validate_metadata_value(
        candidate_metadata,
        key="detector_query_cache_sha256",
        expected=file_sha256_short(detector_path),
        context="candidate artifact",
    )
    _validate_metadata_value(
        candidate_metadata,
        key="query_context_detector_cache_sha256",
        expected=file_sha256_short(query_context_path),
        context="candidate artifact",
    )
    _validate_metadata_value(
        candidate_metadata,
        key="support_feature_cache_sha256",
        expected=file_sha256_short(support_feature_path),
        context="candidate artifact",
    )
    _validate_metadata_value(
        candidate_metadata,
        key="support_geometry_index_sha256",
        expected=file_sha256_short(support_geometry_path),
        context="candidate artifact",
    )
    _validate_metadata_value(
        candidate_metadata,
        key="projected_landmark_bank_sha256",
        expected=file_sha256_short(bank_path),
        context="candidate artifact",
    )
    _validate_metadata_value(
        candidate_metadata,
        key="maplet_support_index_sha256",
        expected=file_sha256_short(maplet_path),
        context="candidate artifact",
    )
    if str(query_context_metadata.get("format")) != "alike_dense_query_context_cache_v1":
        raise ValueError("unsupported dense query context cache")
    if not np.array_equal(
        np.asarray(query_context["image_ids"]).astype(str), image_ids
    ):
        raise ValueError("dense context image IDs differ from detector image IDs")
    context_offsets = np.asarray(query_context["offsets"], dtype=np.int64)
    if context_offsets.shape != (len(image_ids) + 1,) or int(context_offsets[-1]) != len(query_context["xy"]):
        raise ValueError("dense query context offsets are invalid")
    if detector_metadata.get("image_sha256_by_id") != query_context_metadata.get("image_sha256_by_id"):
        raise ValueError("detector and dense context source-image manifests differ")
    for metadata, context in (
        (detector_metadata, "detector query cache"),
        (query_context_metadata, "dense query context cache"),
        (support_metadata, "support feature cache"),
    ):
        if str(metadata.get("alike_checkpoint_sha256", metadata.get("model_checkpoint_sha256", ""))) != str(
            detector_metadata.get("alike_checkpoint_sha256", "")
        ):
            raise ValueError(f"{context} uses a different ALIKE checkpoint")
    _validate_metadata_value(
        geometry_metadata,
        key="support_feature_cache_sha256",
        expected=file_sha256_short(support_feature_path),
        context="support geometry index",
    )
    if not np.array_equal(
        np.asarray(support["track_ids"], dtype=np.int64)[geometry.source_row_indices],
        geometry.track_ids,
    ):
        raise ValueError("support ALIKE rows and support geometry rows differ")
    if len(radio.support_descriptors) != len(support["track_ids"]):
        raise ValueError("RADIO intermediate support rows do not align with ALIKE")
    if len(radio.query_anchor_descriptors) != proposal_rows or len(
        radio.query_context_descriptors
    ) != len(query_context["xy"]):
        raise ValueError("RADIO intermediate query rows do not align with detector caches")
    if str(final_context.metadata.get("pca_fit_scope")) != "mapping_train_images_only":
        raise ValueError("RADIO final PCA was not fit on mapping train images only")
    if bool(final_context.metadata.get("pose_or_ground_truth_used", True)) or bool(
        final_context.metadata.get("image_retrieval_or_submap_used", True)
    ):
        raise ValueError("RADIO final context cache violates the target-free local protocol")
    if str(final_context.metadata.get("radio_checkpoint_sha256")) != str(
        radio.metadata.get("radio_checkpoint_sha256")
    ):
        raise ValueError("RADIO final and intermediate caches use different checkpoints")
    if str(bank_metadata.get("projection_mode")) != "full_map_projected_observations":
        raise ValueError("landmark bank is not a projected-observation bank")
    descriptor_space = str(bank_metadata.get("descriptor_space_id", ""))
    if not descriptor_space or descriptor_space != str(detector_metadata.get("descriptor_space_id", "")):
        raise ValueError("detector and landmark bank descriptor spaces differ")
    if descriptor_space != str(maplet_metadata.get("source_descriptor_space_id", "")):
        raise ValueError("maplet support index descriptor space differs from landmark bank")
    _validate_metadata_value(
        maplet_metadata,
        key="source_landmark_index_sha256",
        expected=file_sha256_short(bank_path),
        context="maplet support index",
    )
    if not np.array_equal(maplet.anchor_track_ids, landmark_bank.track_ids):
        raise ValueError("maplet support rows do not align with landmark bank tracks")
    if int(detector["global_descriptors"].shape[1]) != int(landmark_bank.features.shape[1]):
        raise ValueError("detector final descriptors and landmark bank dimensions differ")
    if int(detector["local_descriptors"].shape[1]) != int(support["descriptors"].shape[1]):
        raise ValueError("query and support ALIKE dimensions differ")
    if np.intersect1d(image_ids, np.asarray(geometry.image_ids).astype(str)).size:
        raise ValueError("query images unexpectedly overlap the support map")
    return (
        proposals,
        detector,
        selected_rows,
        query_context,
        np.asarray(support["descriptors"], dtype=np.float32),
        np.asarray(support["detector_scores"], dtype=np.float32),
        geometry,
        radio,
        final_context,
        landmark_bank,
        maplet,
        {
            "detector_metadata": detector_metadata,
            "candidate_metadata": candidate_metadata,
            "query_context_metadata": query_context_metadata,
            "support_metadata": support_metadata,
            "geometry_metadata": geometry_metadata,
            "bank_metadata": bank_metadata,
            "maplet_metadata": maplet_metadata,
        },
    )


def _verification_source_rows(
    *,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    fit_rows: np.ndarray,
    point_count: int,
    detector_log_merit_weight: float,
    query_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    selected_rows: list[np.ndarray] = []
    selected_query_ids: list[np.ndarray] = []
    audits: list[dict[str, int]] = []
    for query_id in query_ids:
        points, _purged, audit = _verification_points_for_query(
            str(query_id),
            detector=detector,
            proposals=proposals,
            selected_rows=fit_rows,
            point_count=int(point_count),
            detector_log_merit_weight=float(detector_log_merit_weight),
        )
        rows = np.asarray(points.source_row_indices, dtype=np.int64)
        if len(rows) != int(point_count):
            raise ValueError(
                f"{query_id}: expected {point_count} held-out points, got {len(rows)}"
            )
        selected_rows.append(rows)
        selected_query_ids.append(
            np.full((len(rows),), str(query_id), dtype=f"<U{max(len(str(query_id)), 1)}")
        )
        audits.append({str(key): int(value) for key, value in audit.items()})
    return (
        np.concatenate(selected_rows, axis=0),
        np.concatenate(selected_query_ids, axis=0),
        audits,
    )


def _query_image_index(
    *,
    query_id: str,
    detector: Mapping[str, np.ndarray],
    query_context: Mapping[str, np.ndarray],
    radio: RadioIntermediateContextCache,
) -> _ImageDescriptorIndex:
    image_ids = np.asarray(detector["image_ids"]).astype(str)
    positions = np.flatnonzero(image_ids == str(query_id))
    if positions.size != 1:
        raise ValueError(f"detector cache does not uniquely contain {query_id}")
    position = int(positions[0])
    context_offsets = np.asarray(query_context["offsets"], dtype=np.int64)
    begin, end = int(context_offsets[position]), int(context_offsets[position + 1])
    return _ImageDescriptorIndex(
        xy=np.asarray(query_context["xy"], dtype=np.float32)[begin:end],
        alike=np.asarray(query_context["local_descriptors"], dtype=np.float32)[begin:end],
        intermediate=radio.query_context_descriptors[begin:end],
        scores=np.asarray(query_context["detector_scores"], dtype=np.float32)[begin:end],
    )


def build_multiscale_candidate_probe_features(
    *,
    proposals_path: Path,
    detector_path: Path,
    candidate_path: Path,
    query_context_path: Path,
    support_feature_path: Path,
    support_geometry_path: Path,
    bank_path: Path,
    maplet_path: Path,
    radio_path: Path,
    final_context_path: Path,
    split_json_path: Path,
    output_path: Path,
    summary_path: Path,
    verification_point_count: int,
    detector_log_merit_weight: float,
    support_view_count: int,
    radius3_px: float,
    radius5_px: float,
    max_context_nodes: int,
    duplicate_radius_px: float,
    support_context_cache_size: int,
    max_queries: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Build a frozen S1 feature tensor without loading any pose target."""

    if (
        int(verification_point_count) <= 0
        or int(support_view_count) <= 0
        or int(max_context_nodes) <= 0
        or float(radius3_px) <= 0.0
        or float(radius5_px) < float(radius3_px)
        or float(duplicate_radius_px) < 0.0
    ):
        raise ValueError("multiscale probe export parameters are invalid")
    if output_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if summary_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    start = time.time()
    (
        proposals,
        detector,
        fit_rows,
        query_context,
        support_alike,
        support_scores,
        geometry,
        radio,
        final_context,
        landmark_bank,
        maplet,
        input_metadata,
    ) = _validate_inputs(
        proposal_path=proposals_path,
        detector_path=detector_path,
        candidate_path=candidate_path,
        query_context_path=query_context_path,
        support_feature_path=support_feature_path,
        support_geometry_path=support_geometry_path,
        bank_path=bank_path,
        maplet_path=maplet_path,
        radio_path=radio_path,
        final_context_path=final_context_path,
    )
    split_payload = json.loads(Path(split_json_path).read_text())
    split_by_query = _split_lookup(split_payload)
    all_query_ids = np.asarray(detector["image_ids"]).astype(str)
    if set(all_query_ids.tolist()) != set(split_by_query):
        missing = sorted(set(all_query_ids.tolist()) - set(split_by_query))
        unexpected = sorted(set(split_by_query) - set(all_query_ids.tolist()))
        raise ValueError(
            f"split manifest does not exactly cover detector queries: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    query_ids = all_query_ids if int(max_queries) <= 0 else all_query_ids[: int(max_queries)]
    source_rows, source_query_ids, selection_audits = _verification_source_rows(
        detector=detector,
        proposals=proposals,
        fit_rows=fit_rows,
        point_count=int(verification_point_count),
        detector_log_merit_weight=float(detector_log_merit_weight),
        query_ids=query_ids.tolist(),
    )
    if not np.array_equal(
        source_query_ids, np.asarray(proposals["query_ids"]).astype(str)[source_rows]
    ):
        raise RuntimeError("verification row ownership changed while selecting S1 features")
    split_names = np.asarray(
        [split_by_query[str(query_id)] for query_id in source_query_ids], dtype=np.str_
    )
    candidate_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)[source_rows]
    canonical_rows = canonical_rows_for_track_candidates(
        candidate_tracks, landmark_bank.track_ids
    )
    row_count, candidate_count = candidate_tracks.shape
    view_count = int(support_view_count)
    feature_count = len(MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES)
    features = np.full(
        (row_count, candidate_count, view_count, feature_count), np.nan, dtype=np.float32
    )
    view_valid = np.zeros((row_count, candidate_count, view_count), dtype=bool)
    support_images = np.full(
        (row_count, candidate_count, view_count), "", dtype="<U256"
    )
    support_coverage = np.zeros((row_count, candidate_count, view_count), dtype=np.int32)

    support_cache = _SupportAppearanceCache(
        geometry=geometry,
        support_alike=support_alike,
        support_scores=support_scores,
        radio=radio,
        final_context=final_context,
        radius3_px=float(radius3_px),
        radius5_px=float(radius5_px),
        max_context_nodes=int(max_context_nodes),
        duplicate_radius_px=float(duplicate_radius_px),
        max_entries=int(support_context_cache_size),
    )
    query_indices: dict[str, _ImageDescriptorIndex] = {}
    final_image_ids = set(final_context.image_ids.tolist())
    maplet_support_ids = set(maplet.support_image_ids)
    unavailable_support_images = maplet_support_ids - final_image_ids
    if unavailable_support_images:
        raise ValueError(
            "RADIO final context cache misses support images used by maplets: "
            f"{sorted(unavailable_support_images)[:5]}"
        )
    skipped_view_count = 0
    processed = 0
    for output_row, source_row in enumerate(source_rows.tolist()):
        query_id = str(source_query_ids[output_row])
        query_nodes = query_indices.get(query_id)
        if query_nodes is None:
            query_nodes = _query_image_index(
                query_id=query_id,
                detector=detector,
                query_context=query_context,
                radio=radio,
            )
            query_indices[query_id] = query_nodes
        anchor_xy = np.asarray(detector["xy"], dtype=np.float32)[int(source_row)]
        query_context3 = query_nodes.local_context(
            anchor_xy=anchor_xy,
            anchor_alike=np.asarray(detector["local_descriptors"], dtype=np.float32)[int(source_row)],
            anchor_intermediate=radio.query_anchor_descriptors[int(source_row)],
            radius_px=float(radius3_px),
            max_nodes=int(max_context_nodes),
            duplicate_radius_px=float(duplicate_radius_px),
        )
        query_context5 = query_nodes.local_context(
            anchor_xy=anchor_xy,
            anchor_alike=np.asarray(detector["local_descriptors"], dtype=np.float32)[int(source_row)],
            anchor_intermediate=radio.query_anchor_descriptors[int(source_row)],
            radius_px=float(radius5_px),
            max_nodes=int(max_context_nodes),
            duplicate_radius_px=float(duplicate_radius_px),
        )
        query_summaries = summarize_multiscale_context(
            query_context3,
            query_context5,
            radius3_px=float(radius3_px),
            radius5_px=float(radius5_px),
        )
        query_final_grid4 = final_context.node_grid4_descriptors(
            query_id, anchor_xy[None]
        )[0]
        query_final_anchor = np.asarray(detector["global_descriptors"], dtype=np.float32)[
            int(source_row)
        ]
        for column, (track_id, canonical_row) in enumerate(
            zip(candidate_tracks[output_row].tolist(), canonical_rows[output_row].tolist())
        ):
            if int(track_id) < 0 or int(canonical_row) < 0:
                continue
            selected_views, skipped = _candidate_support_views(
                maplet=maplet,
                canonical_row=int(canonical_row),
                query_id=query_id,
                support_view_count=view_count,
                view_is_usable=lambda image_id: image_id in final_image_ids,
            )
            skipped_view_count += int(skipped)
            if not selected_views:
                continue
            anchor_cosine = cosine_similarity(
                query_final_anchor, landmark_bank.features[int(canonical_row)]
            )
            if not np.isfinite(anchor_cosine):
                raise ValueError("mapped RADIO-final anchor similarity is invalid")
            for view_index, (support_image_id, coverage_count) in enumerate(selected_views):
                appearance = support_cache.get(
                    track_id=int(track_id), image_id=support_image_id
                )
                vector = multiscale_per_view_feature_vector(
                    radio_final_anchor_cosine=anchor_cosine,
                    query_final_grid4=query_final_grid4,
                    support_final_grid4=appearance.final_grid4,
                    query_context3=query_context3,
                    support_context3=appearance.context3,
                    query_context5=query_context5,
                    support_context5=appearance.context5,
                    radius3_px=float(radius3_px),
                    radius5_px=float(radius5_px),
                    query_summaries=query_summaries,
                    support_summaries=appearance.summaries,
                )
                if not np.isfinite(vector).all():
                    raise RuntimeError("multiscale feature vector contains non-finite values")
                features[output_row, column, view_index] = vector
                view_valid[output_row, column, view_index] = True
                support_images[output_row, column, view_index] = support_image_id
                support_coverage[output_row, column, view_index] = int(coverage_count)
        processed += 1
        if processed % 256 == 0 or processed == row_count:
            elapsed = max(time.time() - start, 1e-6)
            print(
                json.dumps(
                    {
                        "progress_rows": processed,
                        "row_count": row_count,
                        "elapsed_seconds": round(elapsed, 1),
                        "rows_per_second": round(processed / elapsed, 2),
                        "valid_candidate_views": int(np.sum(view_valid[:processed])),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not np.all(np.any(view_valid, axis=2)):
        raise ValueError("no candidate received any real support view")
    expected_valid = candidate_tracks >= 0
    missing_candidate_views = expected_valid & ~np.any(view_valid, axis=2)
    if np.any(missing_candidate_views):
        raise ValueError(
            f"{int(np.sum(missing_candidate_views))} valid top-L candidates lack real support views"
        )
    if np.any(~np.isfinite(features[view_valid])):
        raise RuntimeError("valid multiscale feature entries are non-finite")
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "descriptor_feature_names": list(MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),
        "feature_definition": "per_view_candidate_specific_real_image_local_appearance_v1",
        "proposals_sha256": file_sha256_short(proposals_path),
        "detector_query_cache_sha256": file_sha256_short(detector_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "candidate_fit_rows_sha256": _array_sha256_short(fit_rows),
        "query_context_cache_sha256": file_sha256_short(query_context_path),
        "support_feature_cache_sha256": file_sha256_short(support_feature_path),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "maplet_support_index_sha256": file_sha256_short(maplet_path),
        "radio_intermediate_cache_sha256": file_sha256_short(radio_path),
        "radio_final_context_cache_sha256": file_sha256_short(final_context_path),
        "split_json_sha256": file_sha256_short(split_json_path),
        "descriptor_space_id": input_metadata["bank_metadata"].get("descriptor_space_id"),
        "radio_checkpoint_sha256": radio.metadata.get("radio_checkpoint_sha256"),
        "alike_checkpoint_sha256": input_metadata["detector_metadata"].get("alike_checkpoint_sha256"),
        "source_row_selection": SOURCE_ROW_SELECTION_VERSION,
        "verification_point_count": int(verification_point_count),
        "detector_log_merit_weight": float(detector_log_merit_weight),
        "support_view_selection": "fixed_maplet_coverage_rank_with_real_observation_fallback_v1",
        "support_view_count": view_count,
        "radius3_px": float(radius3_px),
        "radius5_px": float(radius5_px),
        "max_context_nodes": int(max_context_nodes),
        "duplicate_radius_px": float(duplicate_radius_px),
        "candidate_top_k": int(candidate_count),
        "query_row_count": int(row_count),
        "query_image_count": int(len(query_ids)),
        "split_row_counts": {
            split: int(np.sum(split_names == split))
            for split in ("train", "validation", "test")
        },
        "diagnostic_max_queries": int(max_queries),
        "input_protocol": {
            "candidate_artifact_supervision_mode": input_metadata["candidate_metadata"].get("supervision_mode"),
            "final_pca_fit_scope": final_context.metadata.get("pca_fit_scope"),
            "final_context_grid": final_context.metadata.get("grid"),
            "intermediate_projection": radio.metadata.get("projection"),
            "intermediate_support_descriptor_source": radio.metadata.get("support_descriptor_source"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_indices=source_rows,
            query_ids=source_query_ids,
            split_names=split_names,
            xy=np.asarray(detector["xy"], dtype=np.float32)[source_rows],
            candidate_track_ids=candidate_tracks,
            candidate_canonical_rows=canonical_rows,
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_images,
            candidate_support_coverage_counts=support_coverage,
            feature_names=np.asarray(MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    temporary_path.replace(output_path)
    summary = {
        "stage": "frozen_multiscale_candidate_specific_appearance_features",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "metadata": metadata,
        "selection_audits": selection_audits,
        "export_audit": {
            "valid_candidate_view_count": int(np.sum(view_valid)),
            "candidate_without_support_view_count": int(np.sum(missing_candidate_views)),
            "support_view_fallback_or_skip_count": int(skipped_view_count),
            "support_cache": dict(support_cache.stats),
            "runtime_seconds": float(time.time() - start),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_multiscale_candidate_probe_features(
        proposals_path=Path(args.proposals),
        detector_path=Path(args.detector_query_cache),
        candidate_path=Path(args.candidate_artifact),
        query_context_path=Path(args.query_context_cache),
        support_feature_path=Path(args.support_feature_cache),
        support_geometry_path=Path(args.support_geometry_index),
        bank_path=Path(args.projected_landmark_bank),
        maplet_path=Path(args.maplet_support_index),
        radio_path=Path(args.radio_intermediate_cache),
        final_context_path=Path(args.radio_final_context_cache),
        split_json_path=Path(args.split_json),
        output_path=Path(args.output),
        summary_path=Path(args.summary_json),
        verification_point_count=int(args.verification_point_count),
        detector_log_merit_weight=float(args.detector_log_merit_weight),
        support_view_count=int(args.support_view_count),
        radius3_px=float(args.radius3_px),
        radius5_px=float(args.radius5_px),
        max_context_nodes=int(args.max_context_nodes),
        duplicate_radius_px=float(args.duplicate_radius_px),
        support_context_cache_size=int(args.support_context_cache_size),
        max_queries=int(args.max_queries),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
