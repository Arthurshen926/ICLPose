from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_maplet_support_image_manifest import (
    ARTIFACT_FORMAT,
    build_maplet_support_image_manifest,
)


def _write_maplet(path, *, support_ids=("s1.png", "s2.png")) -> None:
    np.savez(
        path,
        anchor_track_ids=np.asarray([11], dtype=np.int64),
        support_image_ids=np.asarray(support_ids),
        support_image_indices=np.asarray([[0, 1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[3, 2]], dtype=np.int32),
        metadata_json=np.asarray(
            json.dumps({"format": "local_maplet_support_index_npz", "max_support_views": 2})
        ),
    )


def _write_split(path, *, train=("q_train.png",)) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "stratified_landmark_query_split_v1",
                "train": list(train),
                "validation": ["q_validation.png"],
                "test": ["q_test.png"],
            }
        )
    )


def test_maplet_support_manifest_excludes_every_query_partition(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    split = tmp_path / "split.json"
    output = tmp_path / "manifest.json"
    summary = tmp_path / "summary.json"
    _write_maplet(maplet)
    _write_split(split)

    result = build_maplet_support_image_manifest(
        maplet_support_index=maplet,
        excluded_query_split_json=split,
        output=output,
        summary_json=summary,
    )

    payload = json.loads(output.read_text())
    assert payload["format"] == ARTIFACT_FORMAT
    assert [row["image_id"] for row in payload["records"]] == ["s1.png", "s2.png"]
    assert payload["metadata"]["support_query_overlap_count"] == 0
    assert result["metadata"]["excluded_query_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_maplet_support_manifest_rejects_query_overlap(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    split = tmp_path / "split.json"
    _write_maplet(maplet)
    _write_split(split, train=("s1.png",))

    with pytest.raises(ValueError, match="overlap"):
        build_maplet_support_image_manifest(
            maplet_support_index=maplet,
            excluded_query_split_json=split,
            output=tmp_path / "manifest.json",
            summary_json=tmp_path / "summary.json",
        )
