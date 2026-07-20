"""Export frozen large-context per-view candidate evidence for the S1b probe.

The exporter deliberately starts from an already frozen S1 feature layout.  It
therefore keeps held-out query tokens, global top-L candidates, and real
support-view identities bit-for-bit fixed while replacing only the local
appearance representation.  No pose, target residual, visibility label,
rendered feature, image retrieval, or whole-image descriptor is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_multiscale_candidate_probe_features import (
    _ImageDescriptorIndex,
    _split_lookup,
    _validate_inputs,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ContextDescriptorSummary,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    cosine_similarity,
    crop_spatial_grid_context,
    structured_multiscale_per_view_feature_vector,
    summarize_context_descriptors,
)
from feature_extract.vfm.localization.radio_final_context import RadioFinalContextPcaCache
from feature_extract.vfm.localization.radio_intermediate_context import (
    RadioIntermediateContextCache,
)


ARTIFACT_FORMAT = "structured_multiscale_candidate_probe_features_v2"
FROZEN_LAYOUT_FORMAT = "multiscale_candidate_probe_features_v1"
HELDOUT_SOURCE_ROW_SELECTION = "heldout_detector_merit_after_target_free_fit_rows_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
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
    parser.add_argument("--radius7_px", type=float, default=144.0)
    parser.add_argument("--radius11_px", type=float, default=288.0)
    parser.add_argument("--max_context_nodes", type=int, default=128)
    parser.add_argument("--duplicate_radius_px", type=float, default=2.0)
    parser.add_argument("--support_context_cache_size", type=int, default=8192)
    parser.add_argument(
        "--layout_shard_count",
        type=int,
        default=1,
        help="contiguous frozen-layout shard count; use the strict merger before fitting",
    )
    parser.add_argument(
        "--layout_shard_index",
        type=int,
        default=0,
        help="zero-based contiguous frozen-layout shard index",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="diagnostic-only prefix limit; production structured artifacts use 0",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, object]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} has no metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _contiguous_layout_shard_positions(
    row_count: int, *, shard_count: int, shard_index: int
) -> np.ndarray:
    """Return a deterministic contiguous layout shard without reordering rows."""

    if int(row_count) <= 0 or int(shard_count) <= 0 or not 0 <= int(shard_index) < int(
        shard_count
    ):
        raise ValueError("structured layout shard parameters are invalid")
    begin = int(row_count) * int(shard_index) // int(shard_count)
    end = int(row_count) * (int(shard_index) + 1) // int(shard_count)
    if end <= begin:
        raise ValueError("structured layout shard is empty")
    return np.arange(begin, end, dtype=np.int64)


def _load_frozen_layout(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "xy",
        "candidate_track_ids",
        "candidate_canonical_rows",
        "candidate_view_valid",
        "candidate_support_image_ids",
        "candidate_support_coverage_counts",
        "feature_names",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError("frozen S1 layout unexpectedly contains labels")
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"frozen S1 layout lacks arrays: {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
        metadata = _metadata(data, context="frozen S1 layout")
    if metadata.get("format") != FROZEN_LAYOUT_FORMAT:
        raise ValueError("unsupported frozen S1 layout format")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "pose_or_ground_truth_used"
    ) is not False:
        raise ValueError("frozen S1 layout is not target-free")
    if bool(metadata.get("image_retrieval_or_submap_used", True)) or bool(
        metadata.get("whole_image_summary_or_global_used", True)
    ):
        raise ValueError("frozen S1 layout violates the no-retrieval local protocol")
    if metadata.get("source_row_selection") != HELDOUT_SOURCE_ROW_SELECTION:
        raise ValueError("frozen S1 layout does not use held-out verification rows")
    if int(metadata.get("verification_point_count", 0)) <= 0 or not str(
        metadata.get("support_view_selection", "")
    ):
        raise ValueError("frozen S1 layout lacks held-out/support-view protocol")
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32).reshape(-1, 2)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    canonical = np.asarray(arrays["candidate_canonical_rows"], dtype=np.int64)
    view_valid = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    support_ids = np.asarray(arrays["candidate_support_image_ids"]).astype(str)
    support_coverage = np.asarray(arrays["candidate_support_coverage_counts"], dtype=np.int32)
    if (
        len(rows) == 0
        or np.unique(rows).size != len(rows)
        or query_ids.shape != splits.shape != (len(rows),)
        or xy.shape != (len(rows), 2)
        or tracks.ndim != 2
        or canonical.shape != tracks.shape
        or view_valid.shape[:2] != tracks.shape
        or support_ids.shape != view_valid.shape
        or support_coverage.shape != view_valid.shape
        or set(splits.tolist()) - {"train", "validation", "test"}
    ):
        raise ValueError("frozen S1 layout arrays are not aligned")
    if np.any((tracks >= 0) & ~np.any(view_valid, axis=2)):
        raise ValueError("frozen S1 layout has a candidate without a support view")
    if np.any(view_valid & (support_ids == "")):
        raise ValueError("frozen S1 layout has a valid view without an image ID")
    return {
        "source_row_indices": rows,
        "query_ids": query_ids,
        "split_names": splits,
        "xy": xy,
        "candidate_track_ids": tracks,
        "candidate_canonical_rows": canonical,
        "candidate_view_valid": view_valid,
        "candidate_support_image_ids": support_ids,
        "candidate_support_coverage_counts": support_coverage,
    }, metadata


def _validate_layout_lineage(
    *,
    layout: Mapping[str, np.ndarray],
    layout_metadata: Mapping[str, object],
    layout_path: Path,
    proposals: Mapping[str, np.ndarray],
    detector: Mapping[str, np.ndarray],
    landmark_bank: object,
    split_by_query: Mapping[str, str],
    expected_hashes: Mapping[str, str],
) -> None:
    for key, expected in expected_hashes.items():
        actual = layout_metadata.get(key)
        if str(actual) != str(expected):
            raise ValueError(
                f"frozen S1 layout lineage mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= len(proposals["query_ids"])):
        raise ValueError("frozen S1 layout source rows are outside proposals")
    if not np.array_equal(
        np.asarray(layout["query_ids"]).astype(str),
        np.asarray(proposals["query_ids"]).astype(str)[rows],
    ):
        raise ValueError("frozen S1 layout query ownership differs from proposals")
    if not np.allclose(
        np.asarray(layout["xy"], dtype=np.float32),
        np.asarray(detector["xy"], dtype=np.float32)[rows],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("frozen S1 layout coordinates differ from detector rows")
    if not np.array_equal(
        np.asarray(layout["candidate_track_ids"], dtype=np.int64),
        np.asarray(proposals["candidate_track_ids"], dtype=np.int64)[rows],
    ):
        raise ValueError("frozen S1 layout candidate tracks differ from proposals")
    expected_canonical = np.searchsorted(
        np.asarray(landmark_bank.track_ids, dtype=np.int64),
        np.maximum(np.asarray(layout["candidate_track_ids"], dtype=np.int64), 0),
    )
    valid = np.asarray(layout["candidate_track_ids"], dtype=np.int64) >= 0
    bank_tracks = np.asarray(landmark_bank.track_ids, dtype=np.int64)
    if np.any(
        valid
        & (
            (expected_canonical >= len(bank_tracks))
            | (bank_tracks[np.minimum(expected_canonical, len(bank_tracks) - 1)]
               != np.asarray(layout["candidate_track_ids"], dtype=np.int64))
        )
    ):
        raise ValueError("frozen S1 layout has a track absent from the landmark bank")
    expected_canonical = np.where(valid, expected_canonical, -1)
    if not np.array_equal(
        np.asarray(layout["candidate_canonical_rows"], dtype=np.int64), expected_canonical
    ):
        raise ValueError("frozen S1 layout canonical rows differ from landmark bank")
    expected_splits = np.asarray(
        [split_by_query.get(str(query_id), "") for query_id in layout["query_ids"]],
        dtype=np.str_,
    )
    if not np.array_equal(expected_splits, np.asarray(layout["split_names"]).astype(str)):
        raise ValueError("frozen S1 layout split identities differ from frozen manifest")


@dataclass(frozen=True)
class _StructuredSupportAppearance:
    anchor_xy: np.ndarray
    intermediate_anchor: np.ndarray
    alike_anchor: np.ndarray
    intermediate7: ContextDescriptorSummary
    intermediate11: ContextDescriptorSummary
    alike7: ContextDescriptorSummary
    alike11: ContextDescriptorSummary
    final_window3: ContextDescriptorSummary
    final_window5: ContextDescriptorSummary
    final_window7: ContextDescriptorSummary


class _StructuredSupportAppearanceCache:
    """Bounded real-support context cache for structured candidate features."""

    def __init__(
        self,
        *,
        geometry: object,
        support_alike: np.ndarray,
        support_scores: np.ndarray,
        radio: RadioIntermediateContextCache,
        final_context: RadioFinalContextPcaCache,
        radius7_px: float,
        radius11_px: float,
        max_context_nodes: int,
        duplicate_radius_px: float,
        max_entries: int,
    ) -> None:
        if (
            float(radius7_px) <= 0.0
            or float(radius11_px) < float(radius7_px)
            or int(max_context_nodes) <= 0
            or int(max_entries) <= 0
        ):
            raise ValueError("structured support-context cache parameters are invalid")
        self.geometry = geometry
        self.support_alike = np.asarray(support_alike, dtype=np.float32)
        self.support_scores = np.asarray(support_scores, dtype=np.float32).reshape(-1)
        self.radio = radio
        self.final_context = final_context
        self.radius7_px = float(radius7_px)
        self.radius11_px = float(radius11_px)
        self.max_context_nodes = int(max_context_nodes)
        self.duplicate_radius_px = float(duplicate_radius_px)
        self.max_entries = int(max_entries)
        self.image_nodes: dict[str, _ImageDescriptorIndex] = {}
        self.appearances: OrderedDict[tuple[int, str], _StructuredSupportAppearance] = OrderedDict()
        self.anchor_observations: OrderedDict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.stats = {
            "image_index_build_count": 0,
            "appearance_cache_hit_count": 0,
            "appearance_cache_miss_count": 0,
            "anchor_observation_cache_hit_count": 0,
            "anchor_observation_cache_miss_count": 0,
        }

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

    def anchor_observation(
        self, *, track_id: int, image_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the real support anchor without constructing local context grids."""

        key = (int(track_id), str(image_id))
        cached = self.anchor_observations.get(key)
        if cached is not None:
            self.anchor_observations.move_to_end(key)
            self.stats["anchor_observation_cache_hit_count"] += 1
            return cached
        self.stats["anchor_observation_cache_miss_count"] += 1
        image_slice = self.geometry.image_slice(str(image_id))
        image_tracks = np.asarray(
            self.geometry.track_ids[image_slice], dtype=np.int64
        )
        position = int(np.searchsorted(image_tracks, int(track_id)))
        if position >= len(image_tracks) or int(image_tracks[position]) != int(track_id):
            raise ValueError(
                f"support image {image_id!r} does not observe candidate track {track_id}"
            )
        source_row = int(self.geometry.source_row_indices[image_slice][position])
        output = (
            np.asarray(self.geometry.xy[image_slice][position], dtype=np.float32),
            np.asarray(self.radio.support_descriptors[source_row], dtype=np.float32),
            np.asarray(self.support_alike[source_row], dtype=np.float32),
        )
        self.anchor_observations[key] = output
        if len(self.anchor_observations) > self.max_entries:
            self.anchor_observations.popitem(last=False)
        return output

    def get(self, *, track_id: int, image_id: str) -> _StructuredSupportAppearance:
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
        context7 = nodes.local_context(
            anchor_xy=anchor_xy,
            anchor_alike=nodes.alike[anchor_index],
            anchor_intermediate=nodes.intermediate[anchor_index],
            radius_px=self.radius7_px,
            max_nodes=self.max_context_nodes,
            duplicate_radius_px=self.duplicate_radius_px,
        )
        context11 = nodes.local_context(
            anchor_xy=anchor_xy,
            anchor_alike=nodes.alike[anchor_index],
            anchor_intermediate=nodes.intermediate[anchor_index],
            radius_px=self.radius11_px,
            max_nodes=self.max_context_nodes,
            duplicate_radius_px=self.duplicate_radius_px,
        )
        final_grid, image_size = self.final_context.image_grid_descriptors(
            str(image_id), grid_size=8
        )
        output = _StructuredSupportAppearance(
            anchor_xy=np.asarray(anchor_xy, dtype=np.float32),
            intermediate_anchor=np.asarray(nodes.intermediate[anchor_index], dtype=np.float32),
            alike_anchor=np.asarray(nodes.alike[anchor_index], dtype=np.float32),
            intermediate7=summarize_context_descriptors(
                context7,
                descriptor_name="intermediate",
                grid_size=7,
                radius_px=self.radius7_px,
            ),
            intermediate11=summarize_context_descriptors(
                context11,
                descriptor_name="intermediate",
                grid_size=11,
                radius_px=self.radius11_px,
            ),
            alike7=summarize_context_descriptors(
                context7,
                descriptor_name="alike",
                grid_size=7,
                radius_px=self.radius7_px,
            ),
            alike11=summarize_context_descriptors(
                context11,
                descriptor_name="alike",
                grid_size=11,
                radius_px=self.radius11_px,
            ),
            final_window3=crop_spatial_grid_context(
                final_grid, image_size=image_size, xy=anchor_xy, window_size=3
            ),
            final_window5=crop_spatial_grid_context(
                final_grid, image_size=image_size, xy=anchor_xy, window_size=5
            ),
            final_window7=crop_spatial_grid_context(
                final_grid, image_size=image_size, xy=anchor_xy, window_size=7
            ),
        )
        self.appearances[key] = output
        if len(self.appearances) > self.max_entries:
            self.appearances.popitem(last=False)
        return output


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
    offsets = np.asarray(query_context["offsets"], dtype=np.int64)
    begin, end = int(offsets[position]), int(offsets[position + 1])
    return _ImageDescriptorIndex(
        xy=np.asarray(query_context["xy"], dtype=np.float32)[begin:end],
        alike=np.asarray(query_context["local_descriptors"], dtype=np.float32)[begin:end],
        intermediate=radio.query_context_descriptors[begin:end],
        scores=np.asarray(query_context["detector_scores"], dtype=np.float32)[begin:end],
    )


def _query_context_summaries(
    *,
    nodes: _ImageDescriptorIndex,
    detector: Mapping[str, np.ndarray],
    radio: RadioIntermediateContextCache,
    final_context: RadioFinalContextPcaCache,
    query_id: str,
    source_row: int,
    radius7_px: float,
    radius11_px: float,
    max_context_nodes: int,
    duplicate_radius_px: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ContextDescriptorSummary,
    ContextDescriptorSummary,
    ContextDescriptorSummary,
    ContextDescriptorSummary,
    ContextDescriptorSummary,
    ContextDescriptorSummary,
    ContextDescriptorSummary,
]:
    anchor_xy = np.asarray(detector["xy"], dtype=np.float32)[int(source_row)]
    anchor_alike = np.asarray(detector["local_descriptors"], dtype=np.float32)[int(source_row)]
    anchor_intermediate = np.asarray(radio.query_anchor_descriptors[int(source_row)], dtype=np.float32)
    context7 = nodes.local_context(
        anchor_xy=anchor_xy,
        anchor_alike=anchor_alike,
        anchor_intermediate=anchor_intermediate,
        radius_px=float(radius7_px),
        max_nodes=int(max_context_nodes),
        duplicate_radius_px=float(duplicate_radius_px),
    )
    context11 = nodes.local_context(
        anchor_xy=anchor_xy,
        anchor_alike=anchor_alike,
        anchor_intermediate=anchor_intermediate,
        radius_px=float(radius11_px),
        max_nodes=int(max_context_nodes),
        duplicate_radius_px=float(duplicate_radius_px),
    )
    final_grid, image_size = final_context.image_grid_descriptors(str(query_id), grid_size=8)
    return (
        anchor_xy,
        anchor_alike,
        anchor_intermediate,
        summarize_context_descriptors(
            context7, descriptor_name="intermediate", grid_size=7, radius_px=float(radius7_px)
        ),
        summarize_context_descriptors(
            context11, descriptor_name="intermediate", grid_size=11, radius_px=float(radius11_px)
        ),
        summarize_context_descriptors(
            context7, descriptor_name="alike", grid_size=7, radius_px=float(radius7_px)
        ),
        summarize_context_descriptors(
            context11, descriptor_name="alike", grid_size=11, radius_px=float(radius11_px)
        ),
        crop_spatial_grid_context(final_grid, image_size=image_size, xy=anchor_xy, window_size=3),
        crop_spatial_grid_context(final_grid, image_size=image_size, xy=anchor_xy, window_size=5),
        crop_spatial_grid_context(final_grid, image_size=image_size, xy=anchor_xy, window_size=7),
    )


def build_structured_multiscale_candidate_probe_features(
    *,
    frozen_layout_path: Path,
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
    radius7_px: float,
    radius11_px: float,
    max_context_nodes: int,
    duplicate_radius_px: float,
    support_context_cache_size: int,
    layout_shard_count: int = 1,
    layout_shard_index: int = 0,
    max_rows: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Build structured per-view evidence without changing the frozen layout."""

    if (
        float(radius7_px) <= 0.0
        or float(radius11_px) < float(radius7_px)
        or int(max_context_nodes) <= 0
        or float(duplicate_radius_px) < 0.0
        or int(support_context_cache_size) <= 0
        or int(max_rows) < 0
        or int(layout_shard_count) <= 0
        or not 0 <= int(layout_shard_index) < int(layout_shard_count)
    ):
        raise ValueError("structured multiscale export parameters are invalid")
    if output_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if summary_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    started = time.time()
    layout, layout_metadata = _load_frozen_layout(frozen_layout_path)
    (
        proposals,
        detector,
        _fit_rows,
        query_context,
        support_alike,
        support_scores,
        geometry,
        radio,
        final_context,
        landmark_bank,
        _maplet,
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
    split_by_query = _split_lookup(json.loads(Path(split_json_path).read_text()))
    _validate_layout_lineage(
        layout=layout,
        layout_metadata=layout_metadata,
        layout_path=frozen_layout_path,
        proposals=proposals,
        detector=detector,
        landmark_bank=landmark_bank,
        split_by_query=split_by_query,
        expected_hashes={
            "proposals_sha256": file_sha256_short(proposals_path),
            "detector_query_cache_sha256": file_sha256_short(detector_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "query_context_cache_sha256": file_sha256_short(query_context_path),
            "support_feature_cache_sha256": file_sha256_short(support_feature_path),
            "support_geometry_index_sha256": file_sha256_short(support_geometry_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "maplet_support_index_sha256": file_sha256_short(maplet_path),
            "radio_intermediate_cache_sha256": file_sha256_short(radio_path),
        },
    )
    if final_context.grid8_descriptors is None:
        raise ValueError("structured S1b requires RADIO-final grid8 descriptors")
    if 8 not in set(final_context.metadata.get("spatial_grid_sizes", ())):
        raise ValueError("RADIO-final cache manifest does not declare spatial grid8")
    final_image_ids = set(final_context.image_ids.tolist())
    if not set(np.asarray(layout["query_ids"]).astype(str)).issubset(final_image_ids):
        raise ValueError("RADIO-final grid8 cache misses a frozen layout query image")
    valid_support_ids = np.asarray(layout["candidate_support_image_ids"]).astype(str)[
        np.asarray(layout["candidate_view_valid"], dtype=bool)
    ]
    if not set(valid_support_ids.tolist()).issubset(final_image_ids):
        raise ValueError("RADIO-final grid8 cache misses a frozen layout support image")

    full_layout_row_count = int(len(layout["source_row_indices"]))
    full_source_rows_sha256 = _array_sha256_short(layout["source_row_indices"])
    full_candidate_tracks_sha256 = _array_sha256_short(layout["candidate_track_ids"])
    full_support_view_ids_sha256 = _array_sha256_short(
        layout["candidate_support_image_ids"]
    )
    layout_positions = _contiguous_layout_shard_positions(
        full_layout_row_count,
        shard_count=int(layout_shard_count),
        shard_index=int(layout_shard_index),
    )
    layout = {key: np.asarray(value)[layout_positions] for key, value in layout.items()}
    if int(max_rows) > 0:
        limit = min(int(max_rows), len(layout_positions))
        layout_positions = layout_positions[:limit]
        layout = {key: np.asarray(value)[:limit] for key, value in layout.items()}

    rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
    tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    view_valid = np.asarray(layout["candidate_view_valid"], dtype=bool)
    support_ids = np.asarray(layout["candidate_support_image_ids"]).astype(str)
    row_count, candidate_count = tracks.shape
    view_count = int(view_valid.shape[2])
    features = np.full(
        (
            row_count,
            candidate_count,
            view_count,
            len(STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),
        ),
        np.nan,
        dtype=np.float16,
    )
    support_cache = _StructuredSupportAppearanceCache(
        geometry=geometry,
        support_alike=support_alike,
        support_scores=support_scores,
        radio=radio,
        final_context=final_context,
        radius7_px=float(radius7_px),
        radius11_px=float(radius11_px),
        max_context_nodes=int(max_context_nodes),
        duplicate_radius_px=float(duplicate_radius_px),
        max_entries=int(support_context_cache_size),
    )
    query_indices: dict[str, _ImageDescriptorIndex] = {}
    processed = 0
    for output_row, source_row in enumerate(rows.tolist()):
        query_id = str(layout["query_ids"][output_row])
        query_nodes = query_indices.get(query_id)
        if query_nodes is None:
            query_nodes = _query_image_index(
                query_id=query_id,
                detector=detector,
                query_context=query_context,
                radio=radio,
            )
            query_indices[query_id] = query_nodes
        (
            _anchor_xy,
            query_alike_anchor,
            query_intermediate_anchor,
            query_intermediate7,
            query_intermediate11,
            query_alike7,
            query_alike11,
            query_final_window3,
            query_final_window5,
            _query_final_window7,
        ) = _query_context_summaries(
            nodes=query_nodes,
            detector=detector,
            radio=radio,
            final_context=final_context,
            query_id=query_id,
            source_row=int(source_row),
            radius7_px=float(radius7_px),
            radius11_px=float(radius11_px),
            max_context_nodes=int(max_context_nodes),
            duplicate_radius_px=float(duplicate_radius_px),
        )
        for candidate_column, track_id in enumerate(tracks[output_row].tolist()):
            if int(track_id) < 0:
                continue
            canonical_row = int(layout["candidate_canonical_rows"][output_row, candidate_column])
            final_anchor = cosine_similarity(
                np.asarray(detector["global_descriptors"], dtype=np.float32)[int(source_row)],
                landmark_bank.features[canonical_row],
            )
            if not np.isfinite(final_anchor):
                raise ValueError("structured RADIO-final anchor similarity is invalid")
            for view_column in range(view_count):
                if not bool(view_valid[output_row, candidate_column, view_column]):
                    continue
                appearance = support_cache.get(
                    track_id=int(track_id),
                    image_id=str(support_ids[output_row, candidate_column, view_column]),
                )
                vector = structured_multiscale_per_view_feature_vector(
                    radio_final_anchor_cosine=float(final_anchor),
                    radio_intermediate_anchor_cosine=cosine_similarity(
                        query_intermediate_anchor, appearance.intermediate_anchor
                    ),
                    alike_anchor_cosine=cosine_similarity(query_alike_anchor, appearance.alike_anchor),
                    query_final_window3=query_final_window3,
                    support_final_window3=appearance.final_window3,
                    query_final_window5=query_final_window5,
                    support_final_window5=appearance.final_window5,
                    query_intermediate7=query_intermediate7,
                    support_intermediate7=appearance.intermediate7,
                    query_intermediate11=query_intermediate11,
                    support_intermediate11=appearance.intermediate11,
                    query_alike7=query_alike7,
                    support_alike7=appearance.alike7,
                    query_alike11=query_alike11,
                    support_alike11=appearance.alike11,
                )
                if np.any(np.isinf(vector)) or not np.isfinite(vector[:3]).all():
                    raise RuntimeError("structured multiscale feature vector is invalid")
                features[output_row, candidate_column, view_column] = vector.astype(
                    np.float16
                )
        processed += 1
        if processed % 256 == 0 or processed == row_count:
            elapsed = max(time.time() - started, 1e-6)
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
    valid_feature_rows = features[view_valid]
    if np.any(np.isinf(valid_feature_rows)) or np.any(
        ~np.isfinite(valid_feature_rows[:, :3])
    ):
        raise RuntimeError("structured valid feature tensor has invalid anchor values")
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "feature_definition": "per_view_candidate_specific_real_image_large_context_2d_layout_shift_overlap_v2",
        "descriptor_feature_names": list(STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),
        "feature_dtype": "float16_with_nan_for_unobserved_2d_regions",
        "source_row_selection": layout_metadata["source_row_selection"],
        "verification_point_count": int(layout_metadata["verification_point_count"]),
        "support_view_selection": layout_metadata["support_view_selection"],
        "candidate_fit_rows_sha256": layout_metadata.get("candidate_fit_rows_sha256"),
        "detector_log_merit_weight": layout_metadata.get("detector_log_merit_weight"),
        "frozen_layout_features": str(frozen_layout_path),
        "frozen_layout_features_sha256": file_sha256_short(frozen_layout_path),
        "frozen_source_rows_sha256": _array_sha256_short(rows),
        "frozen_candidate_tracks_sha256": _array_sha256_short(tracks),
        "frozen_support_view_ids_sha256": _array_sha256_short(support_ids),
        "full_frozen_layout_row_count": int(full_layout_row_count),
        "full_frozen_source_rows_sha256": full_source_rows_sha256,
        "full_frozen_candidate_tracks_sha256": full_candidate_tracks_sha256,
        "full_frozen_support_view_ids_sha256": full_support_view_ids_sha256,
        "layout_shard_count": int(layout_shard_count),
        "layout_shard_index": int(layout_shard_index),
        "layout_position_count": int(len(layout_positions)),
        "is_complete_frozen_layout": bool(
            int(layout_shard_count) == 1 and int(max_rows) == 0
        ),
        "proposals_sha256": file_sha256_short(proposals_path),
        "detector_query_cache_sha256": file_sha256_short(detector_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
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
        "candidate_top_k": int(candidate_count),
        "support_view_count": int(view_count),
        "query_row_count": int(row_count),
        "diagnostic_max_rows": int(max_rows),
        "query_image_count": int(len(set(np.asarray(layout["query_ids"]).astype(str).tolist()))),
        "split_row_counts": {
            split: int(np.sum(np.asarray(layout["split_names"]).astype(str) == split))
            for split in ("train", "validation", "test")
        },
        "context": {
            "radio_final_grid_size": 8,
            "radio_final_window_sizes": [3, 5],
            "intermediate_alike_grid_sizes": [7, 11],
            "radius7_px": float(radius7_px),
            "radius11_px": float(radius11_px),
            "max_context_nodes": int(max_context_nodes),
            "duplicate_radius_px": float(duplicate_radius_px),
            "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
            "shift_correlation": {
                "maximum_shift_cells": 1,
                "overlap": "common_valid_cells_divided_by_full_grid_area_v1",
            },
        },
        "input_protocol": {
            "frozen_candidate_layout": "heldout_query_rows_fixed_global_topl_fixed_support_views_v1",
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
            source_row_indices=rows,
            layout_positions=layout_positions,
            query_ids=np.asarray(layout["query_ids"]).astype(str),
            split_names=np.asarray(layout["split_names"]).astype(str),
            xy=np.asarray(layout["xy"], dtype=np.float32),
            candidate_track_ids=tracks,
            candidate_canonical_rows=np.asarray(layout["candidate_canonical_rows"], dtype=np.int64),
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=np.asarray(
                layout["candidate_support_coverage_counts"], dtype=np.int32
            ),
            feature_names=np.asarray(
                STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES, dtype=np.str_
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    temporary_path.replace(output_path)
    summary = {
        "stage": "frozen_structured_multiscale_candidate_specific_appearance_features",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "metadata": metadata,
        "export_audit": {
            "valid_candidate_view_count": int(np.sum(view_valid)),
            "support_cache": dict(support_cache.stats),
            "runtime_seconds": float(time.time() - started),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_structured_multiscale_candidate_probe_features(
        frozen_layout_path=Path(args.frozen_layout_features),
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
        radius7_px=float(args.radius7_px),
        radius11_px=float(args.radius11_px),
        max_context_nodes=int(args.max_context_nodes),
        duplicate_radius_px=float(args.duplicate_radius_px),
        support_context_cache_size=int(args.support_context_cache_size),
        layout_shard_count=int(args.layout_shard_count),
        layout_shard_index=int(args.layout_shard_index),
        max_rows=int(args.max_rows),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
