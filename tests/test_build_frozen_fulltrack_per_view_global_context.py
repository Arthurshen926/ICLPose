from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multisource_region_context as region_builder
import feature_extract.tools.vfm.build_frozen_fulltrack_per_view_sfm_maplet_transport as maplet_builder
import feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multiscale_translation_mode as translation_builder
import feature_extract.tools.vfm.build_frozen_fulltrack_per_view_absolute_phase as absolute_phase_builder
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_global_context import (
    build_frozen_fulltrack_per_view_global_context,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)


def _write_context_cache(path) -> None:
    np.savez_compressed(
        path,
        image_ids=np.asarray(["a.png", "b.png", "q.png"]),
        global_descriptors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        ),
        summary_descriptors=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "radio_final_context_pca_v1",
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "pca_fit_scope": (
                        "mapping_support_images_excluding_all_query_splits_v1"
                    ),
                }
            )
        ),
    )


def _write_geometry(path) -> None:
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("a.png", "b.png"),
            image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
            source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
            track_ids=np.asarray([10, 11, 10], dtype=np.int64),
            xy=np.asarray(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
            ),
            viewing_rays=np.ones((3, 3), dtype=np.float32),
            reprojection_errors=np.zeros((3,), dtype=np.float32),
        ),
        path,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )


def _write_current_raw_per_view_source(
    path, *, geometry_path, context_path, s0_path
) -> None:
    row_count, candidate_count = 192, 20
    tracks = np.full((row_count, candidate_count), -1, dtype=np.int64)
    tracks[:, :2] = np.asarray([10, 11], dtype=np.int64)
    candidate = np.zeros((row_count, candidate_count), dtype=np.float32)
    candidate[:, :2] = np.asarray([0.45, 0.45], dtype=np.float32)
    counts = np.zeros((row_count, candidate_count), dtype=np.int64)
    counts[:, 0] = 2
    counts[:, 1] = 1
    offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(counts.reshape(-1)))
    )
    # Candidate 10 owns geometry rows a,b and candidate 11 owns a.  Repeating
    # this order makes any accidental support re-enumeration observable.
    geometry_rows = np.tile(np.asarray([0, 2, 1], dtype=np.int64), row_count)
    metadata = {
        "format": "frozen_fulltrack_candidate_per_view_appearance_v1",
        "version": "frozen_fulltrack_candidate_per_view_appearance_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "fixed_candidate_top_k": 20,
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_count_cap": None,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": (
            "raw_aligned_ncc_per_real_sfm_observation_v1"
        ),
        "profiles": [{"name": "raw", "source": "fixture", "window_size": 1}],
        "appearance_config": {"fixture": True},
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "context_cache_sha256": {"radio_final": file_sha256_short(context_path)},
        "source_frozen_appearance_artifact": str(s0_path),
        "source_frozen_appearance_artifact_sha256": file_sha256_short(s0_path),
        "strict_fulltrack_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_posterior_preserved": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "support_view_count_cap": None,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "heldout_s0_verification_rows": True,
        },
    }
    np.savez_compressed(
        path,
        verification_query_ids=np.asarray(["q.png"] * row_count),
        split_names=np.asarray(["train"] * row_count),
        verification_source_row_indices=np.arange(row_count, dtype=np.int64),
        verification_xy=np.full((row_count, 2), 2.0, dtype=np.float32),
        candidate_track_ids=tracks,
        candidate_probabilities=candidate,
        null_probabilities=np.full((row_count,), 0.1, dtype=np.float32),
        candidate_support_observation_counts=counts,
        source_maplet_support_view_counts=np.where(counts > 0, 1, 0),
        profile_names=np.asarray(["raw"], dtype=np.str_),
        edge_candidate_offsets=offsets,
        edge_geometry_rows=geometry_rows,
        edge_profile_scores=np.ones((len(geometry_rows), 1), dtype=np.float16),
        edge_profile_valid=np.ones((len(geometry_rows), 1), dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_builder_preserves_current_raw_csr_and_keeps_per_view_global_context(tmp_path) -> None:
    context_path = tmp_path / "context.npz"
    geometry_path = tmp_path / "geometry.npz"
    s0_path = tmp_path / "s0.npz"
    raw_path = tmp_path / "raw.npz"
    output = tmp_path / "per-view-global.npz"
    _write_context_cache(context_path)
    _write_geometry(geometry_path)
    s0_path.write_bytes(b"immutable-s0-root")
    _write_current_raw_per_view_source(
        raw_path,
        geometry_path=geometry_path,
        context_path=context_path,
        s0_path=s0_path,
    )

    result = build_frozen_fulltrack_per_view_global_context(
        source_per_view_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
    )

    assert result["edge_count"] == 192 * 3
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        output, allow_pickle=False
    ) as built:
        for field in (
            "candidate_track_ids",
            "candidate_probabilities",
            "null_probabilities",
            "candidate_support_observation_counts",
            "edge_candidate_offsets",
            "edge_geometry_rows",
        ):
            np.testing.assert_array_equal(built[field], raw[field])
        np.testing.assert_array_equal(
            built["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES),
        )
        # q/a is a cosine match and q/b is orthogonal under both descriptors.
        np.testing.assert_allclose(
            built["edge_profile_scores"][:3],
            np.asarray([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float16),
        )
        metadata = json.loads(str(built["metadata_json"].item()))

    assert metadata["source_fulltrack_per_view_artifact_sha256"] == file_sha256_short(
        raw_path
    )
    assert metadata["source_frozen_appearance_artifact_sha256"] == file_sha256_short(
        s0_path
    )
    assert metadata["strict_fulltrack_appearance_contract"][
        "source_fulltrack_csr_edges_preserved"
    ] is True
    assert metadata["per_view_edge_feature_semantics"] == (
        FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS
    )

    features = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(features) == (
        FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_FEATURE_GRANULARITY
    )


def test_region_builder_preserves_current_raw_csr_without_early_view_averaging(
    tmp_path, monkeypatch
) -> None:
    context_path = tmp_path / "final-context.npz"
    geometry_path = tmp_path / "geometry.npz"
    s0_path = tmp_path / "s0.npz"
    raw_path = tmp_path / "raw.npz"
    intermediate_path = tmp_path / "intermediate-context.npz"
    alike_path = tmp_path / "alike-context.npz"
    output = tmp_path / "per-view-region.npz"
    _write_context_cache(context_path)
    _write_geometry(geometry_path)
    s0_path.write_bytes(b"immutable-s0-root")
    intermediate_path.write_bytes(b"intermediate-cache")
    alike_path.write_bytes(b"alike-cache")
    _write_current_raw_per_view_source(
        raw_path,
        geometry_path=geometry_path,
        context_path=context_path,
        s0_path=s0_path,
    )
    ids = np.asarray(["a.png", "b.png", "q.png"])
    sources = tuple(
        SimpleNamespace(name=name, image_ids=ids, metadata={"source": name})
        for name in ("radio_final", "radio_intermediate", "alike")
    )

    def fake_load_sources(**_kwargs):
        return sources

    def fake_source_metadata(source):
        return {"name": source.name, "cache_sha256": source.name}

    def fake_compute_partition(**kwargs):
        count = int(kwargs["end"]) - int(kwargs["begin"])
        width = len(FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES)
        values = np.full((count, width), 0.25, dtype=np.float16)
        return values, {"device": kwargs["device_name"], "edge_count": count}

    monkeypatch.setattr(region_builder, "_load_sources", fake_load_sources)
    monkeypatch.setattr(region_builder, "_source_metadata", fake_source_metadata)
    monkeypatch.setattr(region_builder, "_compute_partition", fake_compute_partition)
    result = region_builder.build_frozen_fulltrack_per_view_multisource_region_context(
        source_per_view_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        radio_intermediate_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        devices=("stub:0", "stub:1"),
        batch_size=128,
        force=False,
    )

    assert result["edge_count"] == 192 * 3
    assert result["protocol"]["support_view_features_averaged_before_inference"] is False
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        output, allow_pickle=False
    ) as built:
        for field in (
            "candidate_track_ids",
            "candidate_probabilities",
            "null_probabilities",
            "candidate_support_observation_counts",
            "edge_candidate_offsets",
            "edge_geometry_rows",
        ):
            np.testing.assert_array_equal(built[field], raw[field])
        np.testing.assert_array_equal(
            built["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES),
        )
        assert built["edge_profile_scores"].shape == (
            192 * 3,
            len(FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES),
        )
        assert built["edge_profile_valid"].all()
        metadata = json.loads(str(built["metadata_json"].item()))

    assert metadata["source_fulltrack_per_view_artifact_sha256"] == file_sha256_short(
        raw_path
    )
    assert metadata["source_frozen_appearance_artifact_sha256"] == file_sha256_short(
        s0_path
    )
    assert metadata["per_view_edge_feature_semantics"] == (
        FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
    )
    assert metadata["strict_fulltrack_appearance_contract"][
        "source_fulltrack_csr_edges_preserved"
    ] is True
    features = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(features) == (
        FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_FEATURE_GRANULARITY
    )


def test_sfm_maplet_builder_preserves_raw_csr_and_marks_missing_maplets_unknown(
    tmp_path, monkeypatch
) -> None:
    context_path = tmp_path / "final-context.npz"
    geometry_path = tmp_path / "geometry.npz"
    s0_path = tmp_path / "s0.npz"
    raw_path = tmp_path / "raw.npz"
    intermediate_path = tmp_path / "intermediate-context.npz"
    alike_path = tmp_path / "alike-context.npz"
    output = tmp_path / "per-view-maplet.npz"
    _write_context_cache(context_path)
    _write_geometry(geometry_path)
    s0_path.write_bytes(b"immutable-s0-root")
    intermediate_path.write_bytes(b"intermediate-cache")
    alike_path.write_bytes(b"alike-cache")
    _write_current_raw_per_view_source(
        raw_path,
        geometry_path=geometry_path,
        context_path=context_path,
        s0_path=s0_path,
    )
    ids = np.asarray(["a.png", "b.png", "q.png"])
    sources = tuple(
        SimpleNamespace(
            profile=SimpleNamespace(name=name, grid_size=16),
            image_ids=ids,
            image_sizes=np.asarray([[16, 16]] * len(ids), dtype=np.int64),
            grids=np.ones((len(ids), 16, 16, 2), dtype=np.float32),
            metadata={"source": name},
        )
        for name in ("radio_final", "radio_intermediate", "alike")
    )

    monkeypatch.setattr(maplet_builder, "_load_sources", lambda **_kwargs: sources)
    monkeypatch.setattr(
        maplet_builder,
        "_source_metadata",
        lambda source: {"name": source.profile.name, "cache_sha256": source.profile.name},
    )
    monkeypatch.setattr(
        maplet_builder,
        "_build_maplet_neighbor_topologies",
        lambda **_kwargs: {
            profile.name: np.zeros((3, 4, 4), dtype=np.int64)
            for profile in maplet_builder.SFM_MAPLET_TRANSPORT_PROFILES
        },
    )

    def fake_compute_partition(**kwargs):
        count = int(kwargs["end"]) - int(kwargs["begin"])
        profile_count = len(maplet_builder.SFM_MAPLET_TRANSPORT_PROFILES)
        width = len(FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES)
        values = np.full((count, width), 0.25, dtype=np.float16)
        usable = np.ones((count, profile_count), dtype=bool)
        counts = np.ones((count, profile_count, 4), dtype=np.uint8)
        return values, usable, counts, {"device": kwargs["device_name"], "edge_count": count}

    monkeypatch.setattr(maplet_builder, "_compute_partition", fake_compute_partition)
    result = maplet_builder.build_frozen_fulltrack_per_view_sfm_maplet_transport(
        source_per_view_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        radio_intermediate_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        devices=("stub:0", "stub:1"),
        batch_size=128,
        force=False,
    )

    assert result["edge_count"] == 192 * 3
    assert result["protocol"]["candidate_center_descriptor_excluded"] is True
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        output, allow_pickle=False
    ) as built:
        for field in (
            "candidate_track_ids",
            "candidate_probabilities",
            "null_probabilities",
            "candidate_support_observation_counts",
            "edge_candidate_offsets",
            "edge_geometry_rows",
        ):
            np.testing.assert_array_equal(built[field], raw[field])
        np.testing.assert_array_equal(
            built["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES),
        )
        assert built["edge_profile_valid"].all()
        assert built["edge_maplet_profile_usable"].all()
        assert built["edge_maplet_support_quadrant_counts"].shape == (192 * 3, 4, 4)
        metadata = json.loads(str(built["metadata_json"].item()))

    assert metadata["per_view_edge_feature_semantics"] == (
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
    )
    assert metadata["strict_fulltrack_appearance_contract"]["source_fulltrack_csr_edges_preserved"] is True
    assert metadata["strict_fulltrack_appearance_contract"]["candidate_center_descriptor_excluded"] is True
    features = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(features) == (
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY
    )


def test_sfm_maplet_topology_selects_center_excluded_quadrants_vectorially() -> None:
    geometry = SupportObservationGeometryIndex(
        image_ids=("a.png",),
        image_offsets=np.asarray([0, 5], dtype=np.int64),
        source_row_indices=np.arange(5, dtype=np.int64),
        track_ids=np.arange(10, 15, dtype=np.int64),
        xy=np.asarray(
            [[8.0, 8.0], [6.0, 6.0], [10.0, 6.0], [6.0, 10.0], [10.0, 10.0]],
            dtype=np.float32,
        ),
        viewing_rays=np.ones((5, 3), dtype=np.float32),
        reprojection_errors=np.zeros((5,), dtype=np.float32),
    )
    source = SimpleNamespace(
        profile=SimpleNamespace(name="radio_final", grid_size=16),
        image_ids=np.asarray(["a.png"]),
        image_sizes=np.asarray([[16, 16]], dtype=np.int64),
    )
    neighbors = maplet_builder._maplet_neighbor_array(
        geometry=geometry,
        source=source,
        radius_cells=4,
        max_neighbors_per_quadrant=4,
    )
    np.testing.assert_array_equal(neighbors[0, :, 0], np.asarray([1, 2, 3, 4]))
    assert np.all(neighbors[0, :, 1:] == -1)


def test_translation_mode_builder_preserves_raw_csr_without_view_averaging(
    tmp_path, monkeypatch
) -> None:
    context_path = tmp_path / "final-context.npz"
    geometry_path = tmp_path / "geometry.npz"
    s0_path = tmp_path / "s0.npz"
    raw_path = tmp_path / "raw.npz"
    intermediate_path = tmp_path / "intermediate-pca256.npz"
    alike_path = tmp_path / "alike-context.npz"
    output = tmp_path / "per-view-translation.npz"
    _write_context_cache(context_path)
    _write_geometry(geometry_path)
    s0_path.write_bytes(b"immutable-s0-root")
    intermediate_path.write_bytes(b"intermediate-pca256-cache")
    alike_path.write_bytes(b"alike-cache")
    _write_current_raw_per_view_source(
        raw_path,
        geometry_path=geometry_path,
        context_path=context_path,
        s0_path=s0_path,
    )
    ids = np.asarray(["a.png", "b.png", "q.png"])
    sources = tuple(
        SimpleNamespace(
            name=name,
            grid_size=grid_size,
            image_ids=ids,
            image_sizes=np.asarray([[16, 16]] * len(ids), dtype=np.int64),
            grids=np.ones((len(ids), grid_size, grid_size, 2), dtype=np.float32),
            metadata={"source": name},
            cache_path=tmp_path / f"{name}.npz",
        )
        for name, grid_size in (
            ("radio_final", 16),
            ("radio_intermediate_pca256", 16),
            ("alike_fpn", 64),
        )
    )
    monkeypatch.setattr(
        translation_builder, "_load_translation_sources", lambda **_kwargs: sources
    )
    monkeypatch.setattr(
        translation_builder,
        "_source_metadata",
        lambda source: {"name": source.name, "cache_sha256": source.name},
    )

    def fake_compute_partition(**kwargs):
        count = int(kwargs["end"]) - int(kwargs["begin"])
        width = len(FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES)
        values = np.full((count, width), 0.25, dtype=np.float16)
        valid = np.ones((count, width), dtype=bool)
        return values, valid, {"device": kwargs["device_name"], "edge_count": count}

    monkeypatch.setattr(translation_builder, "_compute_partition", fake_compute_partition)
    result = translation_builder.build_frozen_fulltrack_per_view_multiscale_translation_mode(
        source_per_view_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        radio_intermediate_pca256_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        devices=("stub:0", "stub:1"),
        batch_size=128,
        force=False,
    )

    assert result["edge_count"] == 192 * 3
    assert result["protocol"]["support_view_features_averaged_before_inference"] is False
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        output, allow_pickle=False
    ) as built:
        for field in (
            "candidate_track_ids",
            "candidate_probabilities",
            "null_probabilities",
            "candidate_support_observation_counts",
            "edge_candidate_offsets",
            "edge_geometry_rows",
        ):
            np.testing.assert_array_equal(built[field], raw[field])
        np.testing.assert_array_equal(
            built["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES),
        )
        assert built["edge_profile_valid"].all()
        metadata = json.loads(str(built["metadata_json"].item()))

    assert metadata["per_view_edge_feature_semantics"] == (
        FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
    )
    assert metadata["strict_fulltrack_appearance_contract"]["source_fulltrack_csr_edges_preserved"] is True
    assert metadata["strict_fulltrack_appearance_contract"][
        "explicit_availability_or_neighbor_count_is_not_a_learned_feature"
    ] is True
    features = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(features) == (
        FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY
    )


def test_absolute_phase_builder_preserves_raw_csr_and_separates_position_control(
    tmp_path, monkeypatch
) -> None:
    context_path = tmp_path / "final-context.npz"
    geometry_path = tmp_path / "geometry.npz"
    s0_path = tmp_path / "s0.npz"
    raw_path = tmp_path / "raw.npz"
    intermediate_path = tmp_path / "intermediate-pca256.npz"
    alike_path = tmp_path / "alike-context.npz"
    output = tmp_path / "per-view-absolute-phase.npz"
    _write_context_cache(context_path)
    _write_geometry(geometry_path)
    s0_path.write_bytes(b"immutable-s0-root")
    intermediate_path.write_bytes(b"intermediate-pca256-cache")
    alike_path.write_bytes(b"alike-cache")
    _write_current_raw_per_view_source(
        raw_path,
        geometry_path=geometry_path,
        context_path=context_path,
        s0_path=s0_path,
    )
    ids = np.asarray(["a.png", "b.png", "q.png"])
    sources = tuple(
        SimpleNamespace(
            name=name,
            grid_size=grid_size,
            image_ids=ids,
            image_sizes=np.asarray([[32, 32]] * len(ids), dtype=np.int64),
            grids=np.ones((len(ids), grid_size, grid_size, 2), dtype=np.float32),
            metadata={"source": name},
            cache_path=tmp_path / f"{name}.npz",
        )
        for name, grid_size in (
            ("radio_final", 16),
            ("radio_intermediate_pca256", 16),
            ("alike_fpn", 32),
        )
    )
    monkeypatch.setattr(
        absolute_phase_builder, "_load_translation_sources", lambda **_kwargs: sources
    )
    monkeypatch.setattr(
        absolute_phase_builder,
        "_source_metadata",
        lambda source: {"name": source.name, "cache_sha256": source.name},
    )

    def fake_compute_partition(**kwargs):
        count = int(kwargs["end"]) - int(kwargs["begin"])
        width = len(FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES)
        return (
            np.full((count, width), 0.25, dtype=np.float16),
            np.ones((count, width), dtype=bool),
            {"device": kwargs["device_name"], "edge_count": count},
        )

    monkeypatch.setattr(absolute_phase_builder, "_compute_partition", fake_compute_partition)
    result = absolute_phase_builder.build_frozen_fulltrack_per_view_absolute_phase(
        source_per_view_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        radio_intermediate_pca256_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        devices=("stub:0", "stub:1"),
        batch_size=128,
        force=False,
    )

    assert result["edge_count"] == 192 * 3
    assert result["protocol"]["support_view_features_averaged_before_inference"] is False
    assert result["protocol"]["visual_feature_has_no_coverage_count_or_availability_field"] is True
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        output, allow_pickle=False
    ) as built:
        for field in (
            "candidate_track_ids",
            "candidate_probabilities",
            "null_probabilities",
            "candidate_support_observation_counts",
            "edge_candidate_offsets",
            "edge_geometry_rows",
        ):
            np.testing.assert_array_equal(built[field], raw[field])
        np.testing.assert_array_equal(
            built["profile_names"].astype(str),
            np.asarray(FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES),
        )
        assert built["edge_profile_valid"].all()
        metadata = json.loads(str(built["metadata_json"].item()))

    assert metadata["per_view_edge_feature_semantics"] == (
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS
    )
    assert metadata["strict_fulltrack_appearance_contract"]["source_fulltrack_csr_edges_preserved"] is True
    assert metadata["strict_fulltrack_appearance_contract"]["position_control_is_separate"] is True
    features = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(features) == (
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_FEATURE_GRANULARITY
    )
