from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_fixed_support_view_mass import (
    audit_fixed_support_view_mass,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _write_maplet(path) -> None:
    np.savez_compressed(
        path,
        anchor_track_ids=np.asarray([10, 20], dtype=np.int64),
        support_image_ids=np.asarray(["support-a.png", "support-b.png", "support-c.png"]),
        support_image_indices=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        support_coverage_counts=np.asarray([[3, 1], [1, 1]], dtype=np.int64),
        metadata_json=np.asarray(json.dumps({"format": "local_maplet_support_index_npz"})),
    )


def _write_appearance(path, *, maplet_path, altered_weights: bool = False) -> None:
    count = 192
    tracks = np.tile(np.asarray([[10, 20]], dtype=np.int64), (count, 1))
    probabilities = np.tile(np.asarray([[0.6, 0.3]], dtype=np.float32), (count, 1))
    weights = np.tile(
        np.asarray([[[0.75, 0.25], [0.5, 0.5]]], dtype=np.float32), (count, 1, 1)
    )
    if altered_weights:
        weights[0, 0] = np.asarray([0.5, 0.5], dtype=np.float32)
    strict = {
        "heldout_s0_verification_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed": True,
        "candidate_3d_projection_or_pose_used": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "fixed_candidate_top_k": 20,
    }
    metadata = {
        "format": "frozen_multiscale_candidate_absolute_appearance_v1",
        "query_id": "query.png",
        "split_name": "validation",
        "row_count": count,
        "contains_target_fields": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "strict_frozen_appearance_contract": strict,
        "inputs": {"maplet_support_index": {"sha256": file_sha256_short(maplet_path)}},
    }
    np.savez_compressed(
        path,
        verification_query_ids=np.full((count,), "query.png"),
        split_names=np.full((count,), "validation"),
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=np.full((count,), 0.1, dtype=np.float32),
        candidate_view_weights=weights,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_support_view_mass_audit_conserves_fixed_candidate_view_mass(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    artifact = tmp_path / "appearance.npz"
    output = tmp_path / "audit"
    _write_maplet(maplet)
    _write_appearance(artifact, maplet_path=maplet)

    summary = audit_fixed_support_view_mass(
        appearance_artifacts=[artifact],
        maplet_support_index=maplet,
        output_dir=output,
        pool_sizes=(1, 2, 3),
        coverage_levels=(0.5, 0.9, 0.99),
    )

    # Per row: support-a gets .6*.75=.45, support-b=.6*.25+.3*.5=.30,
    # and support-c=.3*.5=.15. Hence top-1 covers 50% and top-2 5/6.
    coverage = summary["split_summary"]["validation"]["mass_coverage_by_pool_size"]
    assert coverage["1"]["median"] == pytest.approx(0.5)
    assert coverage["2"]["median"] == pytest.approx(5.0 / 6.0)
    assert coverage["3"]["median"] == pytest.approx(1.0)
    needed = summary["split_summary"]["validation"]["support_images_needed_by_coverage"]
    assert needed["0.50"]["median"] == pytest.approx(1.0)
    assert needed["0.90"]["median"] == pytest.approx(3.0)
    assert summary["protocol"]["hypothetical_pool_missing_mass"] == "explicit_unknown_not_renormalized_v1"

    with np.load(output / "per_query_support_view_mass.npz", allow_pickle=False) as payload:
        assert payload["pool_mass_coverage"].shape == (1, 3)
        assert payload["ranked_support_image_ids"][0, :3].tolist() == [
            "support-a.png",
            "support-b.png",
            "support-c.png",
        ]


def test_support_view_mass_audit_rejects_reweighted_maplet_views(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    artifact = tmp_path / "appearance.npz"
    _write_maplet(maplet)
    _write_appearance(artifact, maplet_path=maplet, altered_weights=True)

    with pytest.raises(ValueError, match="view weights differ"):
        audit_fixed_support_view_mass(
            appearance_artifacts=[artifact],
            maplet_support_index=maplet,
            output_dir=tmp_path / "audit",
            pool_sizes=(1,),
            coverage_levels=(0.5,),
        )
