from __future__ import annotations

import pytest

from feature_extract.vfm.matcha_render_query_protocol import (
    PAIR_SOURCE_REAL_GT_RENDER,
    PAIR_SOURCE_REAL_REFERENCE_RENDER,
    build_render_query_metadata,
    validate_render_query_metadata,
)
from feature_extract.vfm.matcha_streaming_manifest import (
    MatchaStreamingPairManifest,
    MatchaStreamingPairRecord,
)


def _records() -> tuple[MatchaStreamingPairRecord, ...]:
    return (
        MatchaStreamingPairRecord(
            query_id="seq/frame00001.png",
            split="train",
            pair_type="A_gt",
            pair_type_id=0,
            record_index=0,
            pair_index=0,
            seed=17,
        ),
        MatchaStreamingPairRecord(
            query_id="seq/frame00002.png",
            split="train",
            pair_type="A_gt",
            pair_type_id=0,
            record_index=1,
            pair_index=0,
            seed=17,
        ),
    )


def test_build_render_query_metadata_records_counts_and_overlap() -> None:
    metadata = build_render_query_metadata(
        pair_source=PAIR_SOURCE_REAL_GT_RENDER,
        source_query_manifest="train_manifest.json",
        query_pose_file="dataset_train.txt",
        train_query_ids=["seq/frame00001.png", "seq/frame00002.png"],
        validation_query_ids=["seq/frame00002.png", "seq/frame00003.png"],
        pair_type_counts={"A_gt": 2},
    )

    assert metadata["pair_source"] == "real_gt_render"
    assert metadata["train_query_count"] == 2
    assert metadata["validation_query_count"] == 2
    assert metadata["train_validation_query_overlap_count"] == 1
    assert metadata["pair_type_counts"] == {"A_gt": 2}


def test_render_query_manifest_validates_required_metadata() -> None:
    manifest = MatchaStreamingPairManifest(
        records=_records(),
        metadata=build_render_query_metadata(
            pair_source=PAIR_SOURCE_REAL_GT_RENDER,
            source_query_manifest="train_manifest.json",
            query_pose_file="dataset_train.txt",
            train_query_ids=["seq/frame00001.png", "seq/frame00002.png"],
            validation_query_ids=["seq/frame00003.png"],
            pair_type_counts={"A_gt": 2},
        ),
    )

    assert manifest.metadata["train_validation_query_overlap_count"] == 0


def test_render_query_metadata_rejects_unknown_pair_source() -> None:
    with pytest.raises(ValueError, match="unsupported render-query pair_source"):
        validate_render_query_metadata(
            {
                "pair_source": "ambiguous_render",
                "source_query_manifest": "train_manifest.json",
                "query_pose_file": "dataset_train.txt",
                "train_query_count": 1,
                "validation_query_count": 0,
                "train_validation_query_overlap_count": 0,
                "pair_type_counts": {"A_gt": 1},
            }
        )


def test_render_query_manifest_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="query_pose_file"):
        MatchaStreamingPairManifest(
            records=_records(),
            metadata={
                "pair_source": PAIR_SOURCE_REAL_GT_RENDER,
                "source_query_manifest": "train_manifest.json",
                "train_query_count": 2,
                "validation_query_count": 0,
                "train_validation_query_overlap_count": 0,
                "pair_type_counts": {"A_gt": 2},
            },
        )


def test_reference_render_metadata_requires_candidate_bank() -> None:
    with pytest.raises(ValueError, match="candidate_bank"):
        build_render_query_metadata(
            pair_source=PAIR_SOURCE_REAL_REFERENCE_RENDER,
            source_query_manifest="train_manifest.json",
            query_pose_file="dataset_train.txt",
            train_query_ids=["seq/frame00001.png"],
            validation_query_ids=[],
            pair_type_counts={"D_reference": 1},
        )
