"""Collapse observation RADIO into one view-balanced descriptor per finite plane.

Merged planes can inherit several old observations from the same mapping view.  A raw
maximum over all observations therefore rewards planes merely for having more rows.
This builder first averages rows within a source view, then averages source views with
equal weight.  It consumes mapping RADIO descriptors only and never opens camera poses,
query data, or labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def canonicalize(
    plane_offsets: np.ndarray,
    descriptors: np.ndarray,
    view_names: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    offsets = np.asarray(plane_offsets, np.int64)
    descriptor = np.asarray(descriptors, np.float32)
    names = np.asarray(view_names).astype(str)
    if (
        offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(descriptor)
        or np.any(np.diff(offsets) <= 0) or descriptor.ndim != 2
        or len(names) != len(descriptor)
    ):
        raise ValueError("observation field and visibility rows differ")
    output = []
    view_counts = []
    duplicate_counts = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        lo, hi = int(lo), int(hi)
        per_view = []
        for name in sorted(set(names[lo:hi].tolist())):
            rows = descriptor[lo:hi][names[lo:hi] == name]
            valid = np.linalg.norm(rows, axis=1) > 0
            if np.any(valid):
                per_view.append(_unit(np.mean(rows[valid], axis=0)))
        output.append(
            np.zeros(descriptor.shape[1], np.float32)
            if not per_view else _unit(np.mean(per_view, axis=0))
        )
        view_counts.append(len(per_view))
        duplicate_counts.append((hi - lo) - len(set(names[lo:hi].tolist())))
    counts = np.asarray(view_counts, np.int64)
    duplicates = np.asarray(duplicate_counts, np.int64)
    audit: dict[str, float | int] = {
        "plane_count": len(output),
        "minimum_source_views_per_plane": int(np.min(counts)),
        "median_source_views_per_plane": float(np.median(counts)),
        "maximum_source_views_per_plane": int(np.max(counts)),
        "same_view_duplicate_observation_count": int(np.sum(duplicates)),
    }
    return np.asarray(output, np.float32), audit


def leave_one_view_out_ranks(
    plane_offsets: np.ndarray,
    descriptors: np.ndarray,
    view_names: np.ndarray,
    canonical: np.ndarray,
) -> np.ndarray:
    """Mapping-only identity audit for canonical descriptors."""

    offsets = np.asarray(plane_offsets, np.int64)
    descriptor = np.asarray(descriptors, np.float32)
    names = np.asarray(view_names).astype(str)
    queries = []
    owners = []
    excluded = []
    for plane, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        lo, hi = int(lo), int(hi)
        grouped = []
        for name in sorted(set(names[lo:hi].tolist())):
            rows = descriptor[lo:hi][names[lo:hi] == name]
            valid = np.linalg.norm(rows, axis=1) > 0
            if np.any(valid):
                grouped.append(_unit(np.mean(rows[valid], axis=0)))
        if len(grouped) < 2:
            continue
        grouped_array = np.asarray(grouped, np.float32)
        for index, query in enumerate(grouped_array):
            queries.append(query)
            owners.append(plane)
            excluded.append(_unit(np.mean(np.delete(grouped_array, index, axis=0), axis=0)))
    if not queries:
        return np.zeros(0, np.int64)
    query = np.asarray(queries, np.float32)
    owner = np.asarray(owners, np.int64)
    score = query @ np.asarray(canonical, np.float32).T
    own_score = np.sum(query * np.asarray(excluded, np.float32), axis=1)
    # Stable rank: score descending, then smaller plane row.  Count all strictly
    # better competitors plus tied competitors with a smaller row index.
    plane_rows = np.arange(canonical.shape[0], dtype=np.int64)[None, :]
    return 1 + np.sum(
        (plane_rows != owner[:, None])
        & (
            (score > own_score[:, None])
            | ((score == own_score[:, None]) & (plane_rows < owner[:, None]))
        ),
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation_field", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite canonical plane RADIO field")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    with np.load(args.observation_field, allow_pickle=False) as data:
        source_meta = json.loads(str(data["metadata_json"].item()))
        source_arrays = {
            name: np.asarray(data[name]) for name in (
                "plane_offsets", "observation_plane_rows", "observation_descriptors",
            )
        }
    if (
        source_meta.get("artifact_type") != "goal_maplet_direct_radio_plane_observation_field_v1"
        or source_meta.get("visibility_atlas_content_sha256") != atlas_meta.get("content_sha256")
        or arrays_sha256(source_arrays) != source_meta.get("arrays_sha256")
        or len(atlas.view_names) != len(source_arrays["observation_descriptors"])
    ):
        raise ValueError("source observation field contract differs")
    canonical, audit = canonicalize(
        source_arrays["plane_offsets"], source_arrays["observation_descriptors"], atlas.view_names,
    )
    ranks = leave_one_view_out_ranks(
        source_arrays["plane_offsets"], source_arrays["observation_descriptors"],
        atlas.view_names, canonical,
    )
    plane_count = len(canonical)
    arrays = {
        "plane_offsets": np.arange(plane_count + 1, dtype=np.int64),
        "observation_plane_rows": np.arange(plane_count, dtype=np.int32),
        "observation_descriptors": canonical,
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_radio_plane_observation_field_v1",
        "aggregation": "same_view_mean_then_equal_view_mean_v1",
        "plane_count": plane_count,
        "observation_count": plane_count,
        "source_observation_count": int(len(atlas.view_names)),
        "source_view_count": int(len(set(atlas.view_names.astype(str).tolist()))),
        "minimum_token_pixels": source_meta.get("minimum_token_pixels"),
        "token_grid": source_meta.get("token_grid"),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "source_observation_field_file_sha256": file_sha256(args.observation_field),
        "source_observation_field_content_sha256": source_meta.get("content_sha256"),
        "uses_pose_or_ground_truth": False,
        "mapping_leave_one_view_out_query_count": int(len(ranks)),
        "mapping_leave_one_view_out_recall_at_1": float(np.mean(ranks <= 1)) if len(ranks) else None,
        "mapping_leave_one_view_out_recall_at_5": float(np.mean(ranks <= 5)) if len(ranks) else None,
        "mapping_leave_one_view_out_recall_at_10": float(np.mean(ranks <= 10)) if len(ranks) else None,
        **audit,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
