from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.build_multiscale_candidate_probe_features import (
    _candidate_support_views,
    _load_required_arrays,
    _load_target_free_candidate_rows,
    _split_lookup,
    _verification_source_rows,
)


def test_split_lookup_rejects_cross_split_query_reuse() -> None:
    with pytest.raises(ValueError, match="multiple splits"):
        _split_lookup(
            {"train": ["a.png"], "validation": ["a.png"], "test": ["b.png"]}
        )


def test_target_free_candidate_row_loader_rejects_labels(tmp_path) -> None:
    path = tmp_path / "candidate_rows.npz"
    np.savez(
        path,
        selected_rows=np.asarray([2, 4], dtype=np.int64),
        labels=np.asarray([[True]], dtype=bool),
        metadata_json=np.asarray(
            json.dumps(
                {"contains_ground_truth": False, "supervision_mode": "none_inference_only"}
            )
        ),
    )
    with pytest.raises(ValueError, match="with labels"):
        _load_target_free_candidate_rows(path)


def test_allow_listed_proposals_do_not_require_metadata(tmp_path) -> None:
    path = tmp_path / "proposals.npz"
    np.savez(path, query_ids=np.asarray(["query.png"]))
    arrays, metadata, names = _load_required_arrays(
        path,
        keys=("query_ids",),
        context="proposals",
        require_metadata=False,
    )
    assert arrays["query_ids"].tolist() == ["query.png"]
    assert metadata == {}
    assert names == {"query_ids"}


def test_candidate_support_view_selection_falls_back_without_reordering() -> None:
    maplet = SimpleNamespace(
        support_image_indices=np.asarray([[0, 1, 2, -1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[9, 8, 7, 0]], dtype=np.int64),
        support_image_ids=("a.png", "b.png", "c.png"),
    )
    selected, skipped = _candidate_support_views(
        maplet=maplet,
        canonical_row=0,
        query_id="query.png",
        support_view_count=2,
        view_is_usable=lambda image_id: image_id != "b.png",
    )
    assert selected == (("a.png", 9), ("c.png", 7))
    assert skipped == 1


def test_verification_source_rows_preserve_full_query_id(monkeypatch) -> None:
    class Points:
        source_row_indices = np.asarray([3], dtype=np.int64)

    def fake_verification_points(*args, **kwargs):
        return Points(), np.zeros((0,), dtype=np.int64), {"kept_points": 1}

    monkeypatch.setattr(
        "feature_extract.tools.vfm.build_multiscale_candidate_probe_features._verification_points_for_query",
        fake_verification_points,
    )
    rows, query_ids, audits = _verification_source_rows(
        detector={},
        proposals={},
        fit_rows=np.asarray([0], dtype=np.int64),
        point_count=1,
        detector_log_merit_weight=0.0,
        query_ids=["seq12/frame000123.png"],
    )
    assert rows.tolist() == [3]
    assert query_ids.tolist() == ["seq12/frame000123.png"]
    assert audits == [{"kept_points": 1}]
