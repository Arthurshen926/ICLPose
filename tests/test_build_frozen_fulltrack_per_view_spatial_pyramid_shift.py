from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import feature_extract.tools.vfm.build_frozen_fulltrack_per_view_spatial_pyramid_shift as builder
from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_spatial_pyramid_shift_coverage import (
    _validate_spatial_pyramid_metadata_contract,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
)
from feature_extract.vfm.localization.frozen_fulltrack_spatial_pyramid_shift import (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)


def test_spatial_pyramid_builder_preserves_current_csr_and_keeps_masks_out_of_features(
    tmp_path, monkeypatch
) -> None:
    final_path = tmp_path / "final.npz"
    intermediate_path = tmp_path / "intermediate.npz"
    alike_path = tmp_path / "alike.npz"
    raw_path = tmp_path / "raw.npz"
    for path in (final_path, intermediate_path, alike_path, raw_path):
        path.write_bytes(path.name.encode("ascii"))
    geometry_path = tmp_path / "geometry.npz"
    geometry_path.write_bytes(b"geometry")
    rows, candidates = 192, 20
    counts = np.zeros((rows, candidates), dtype=np.int64)
    counts[:, :2] = np.asarray([2, 1], dtype=np.int64)
    offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(counts.reshape(-1)))
    )
    geometry_rows = np.tile(np.asarray([0, 1, 2], dtype=np.int64), rows)
    candidate_tracks = np.full((rows, candidates), -1, dtype=np.int64)
    candidate_tracks[:, :2] = np.asarray([10, 11], dtype=np.int64)
    candidate_probabilities = np.zeros((rows, candidates), dtype=np.float32)
    candidate_probabilities[:, :2] = np.asarray([0.45, 0.45], dtype=np.float32)
    source = {
        "verification_query_ids": np.asarray(["q.png"] * rows),
        "split_names": np.asarray(["train"] * rows),
        "verification_source_row_indices": np.arange(rows, dtype=np.int64),
        "verification_xy": np.full((rows, 2), 8.0, dtype=np.float32),
        "candidate_track_ids": candidate_tracks,
        "candidate_probabilities": candidate_probabilities,
        "null_probabilities": np.full((rows,), 0.1, dtype=np.float32),
    }
    edge_candidates = np.repeat(np.arange(rows * candidates, dtype=np.int64), counts.reshape(-1))
    edges = SimpleNamespace(
        geometry_rows=geometry_rows,
        candidate_shape=(rows, candidates),
        edge_candidate_indices=edge_candidates,
        edge_count=len(geometry_rows),
        candidate_observation_counts=counts,
    )
    geometry = SupportObservationGeometryIndex(
        image_ids=("a.png", "b.png"),
        image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
        track_ids=np.asarray([10, 11, 10], dtype=np.int64),
        xy=np.full((3, 2), 8.0, dtype=np.float32),
        viewing_rays=np.ones((3, 3), dtype=np.float32),
        reprojection_errors=np.zeros((3,), dtype=np.float32),
    )
    source_lineage = {
        "source_frozen_appearance_artifact": str(raw_path),
        "source_frozen_appearance_artifact_sha256": file_sha256_short(raw_path),
        "source_fulltrack_per_view_artifact": str(raw_path),
        "source_fulltrack_per_view_artifact_sha256": file_sha256_short(raw_path),
        "source_edge_candidate_offsets_sha256": "fixture",
        "source_edge_geometry_rows_sha256": "fixture",
        "source_edge_feature_semantics": "raw_aligned_ncc_per_real_sfm_observation_v1",
        "source_kind": "raw_fulltrack_per_view_csr_v1",
    }
    monkeypatch.setattr(
        builder,
        "load_support_observation_geometry_index_npz",
        lambda _path: (geometry, {"coordinate_source": "sfm_observation_xy"}),
    )
    monkeypatch.setattr(
        builder,
        "_load_current_raw_per_view_source",
        lambda **_kwargs: (
            source,
            {"context_cache_sha256": {"radio_final": file_sha256_short(final_path)}},
            edges,
            counts,
            source_lineage,
        ),
    )
    image_ids = np.asarray(["a.png", "b.png", "q.png"])
    sources = tuple(
        SimpleNamespace(
            name=name,
            grid_size=size,
            image_ids=image_ids,
            image_sizes=np.asarray([[32, 32]] * len(image_ids), dtype=np.int64),
            grids=np.ones((len(image_ids), size, size, 2), dtype=np.float32),
            metadata={"source": name},
            cache_path=path,
        )
        for name, size, path in (
            ("radio_final", 16, final_path),
            ("radio_intermediate_pca256", 16, intermediate_path),
            ("alike_fpn", 32, alike_path),
        )
    )
    monkeypatch.setattr(builder, "_load_translation_sources", lambda **_kwargs: sources)
    monkeypatch.setattr(
        builder,
        "_source_metadata",
        lambda source: {"name": source.name, "cache_sha256": source.name},
    )

    def fake_compute_partition(**kwargs):
        count = int(kwargs["end"]) - int(kwargs["begin"])
        width = len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES)
        return (
            np.full((count, width), 0.25, dtype=np.float16),
            np.ones((count, width), dtype=bool),
            np.full((count, width), 0.75, dtype=np.float16),
            np.ones((count, width), dtype=bool),
            {
                "device": kwargs["device_name"],
                "edge_count": count,
                "profile_full_crop_coverage": {},
                "profile_center_coverage": {},
            },
        )

    monkeypatch.setattr(builder, "_compute_partition", fake_compute_partition)
    output = tmp_path / "spatial-pyramid.npz"
    result = builder.build_frozen_fulltrack_per_view_spatial_pyramid_shift(
        source_per_view_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=final_path,
        radio_intermediate_pca256_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        mask_control_output=tmp_path / "mask-control.npz",
        mask_control_summary_json=tmp_path / "mask-control-summary.json",
        devices=("stub:0", "stub:1"),
        batch_size=64,
        force=False,
    )

    assert result["edge_count"] == len(geometry_rows)
    assert result["protocol"]["incomplete_center_anchor_is_unknown"] is True
    with np.load(output, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["candidate_track_ids"], candidate_tracks)
        np.testing.assert_array_equal(
            payload["candidate_probabilities"], candidate_probabilities
        )
        np.testing.assert_array_equal(payload["edge_candidate_offsets"], offsets)
        np.testing.assert_array_equal(payload["edge_geometry_rows"], geometry_rows)
        np.testing.assert_array_equal(
            payload["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES),
        )
        assert payload["edge_profile_valid"].all()
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["format"] == FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT
    assert metadata["per_view_edge_feature_semantics"] == (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
    )
    assert metadata["spatial_pyramid_shift_contract"]["full_crop_required"] is False
    assert (
        metadata["spatial_pyramid_shift_contract"]
        ["explicit_availability_or_neighbor_count_feature"]
        is False
    )
    # The original visual artifact deliberately keeps descriptor/control flags
    # in the strict contract; the audit must not mistake absent duplicate
    # appearance-config fields for a control artifact.
    _validate_spatial_pyramid_metadata_contract(
        paths=(output,),
        metadata_rows=(metadata,),
        profile_names=FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES,
        artifact_role="visual",
    )
    loaded = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(loaded) == (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY
    )
    assert loaded.edge_profile_scores.dtype == np.float16
    second_output = tmp_path / "spatial-pyramid-second-query.npz"
    with np.load(output, allow_pickle=False) as payload:
        second_payload = {name: np.asarray(payload[name]) for name in payload.files}
    second_payload["verification_query_ids"] = np.asarray(["q2.png"] * rows)
    np.savez_compressed(second_output, **second_payload)
    merged = load_frozen_fulltrack_per_view_appearance_features((output, second_output))
    assert merged.edge_profile_scores.dtype == np.float16
    assert merged.candidate_track_ids.shape == (rows * 2, candidates)
    assert merged.edge_profile_scores.shape == (
        len(geometry_rows) * 2,
        len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES),
    )
    assert tuple(np.unique(merged.query_ids)) == ("q.png", "q2.png")
    np.testing.assert_array_equal(
        merged.edge_candidate_offsets[rows * candidates :],
        offsets + len(geometry_rows),
    )
    control_output = tmp_path / "mask-control.npz"
    with np.load(control_output, allow_pickle=False) as payload:
        np.testing.assert_array_equal(
            payload["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES),
        )
        control_metadata = json.loads(str(payload["metadata_json"].item()))
    assert control_metadata["artifact_role"] == "original_crop_mask_overlap_control"
    control = load_frozen_fulltrack_per_view_appearance_features((control_output,))
    assert fulltrack_per_view_feature_granularity(control) == (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY
    )
