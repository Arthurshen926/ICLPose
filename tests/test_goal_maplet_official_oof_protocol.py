from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_extract.vfm.official_oof_protocol import (
    ordered_id_sha256,
    parse_cambridge_image_ids,
    validate_route_folds,
)


def test_parse_cambridge_ids_ignores_headers(tmp_path: Path) -> None:
    pose = tmp_path / "poses.txt"
    pose.write_text(
        "Visual Landmarks\nImageFile, Camera Position [X Y Z W P Q R]\n"
        "seq1/frame00001.png 0 0 0 1 0 0 0\n"
        "seq2/frame00001.png 0 0 0 1 0 0 0\n"
    )
    assert parse_cambridge_image_ids(pose) == [
        "seq1/frame00001.png",
        "seq2/frame00001.png",
    ]


def test_validate_route_folds_partitions_each_image_once() -> None:
    image_ids = [
        "seq1/frame00001.png",
        "seq1/frame00002.png",
        "seq2/frame00001.png",
    ]
    rows = validate_route_folds(image_ids, (("seq1",), ("seq2",)))
    assert [row["query_count"] for row in rows] == [2, 1]
    assert [row["mapping_count"] for row in rows] == [1, 2]
    assert rows[0]["query_image_ids_sha256"] == ordered_id_sha256(
        image_ids[:2]
    )


def test_validate_route_folds_rejects_duplicate_trajectory() -> None:
    with pytest.raises(ValueError, match="multiple folds"):
        validate_route_folds(
            ["seq1/frame00001.png"], (("seq1",), ("seq1",))
        )
