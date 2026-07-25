import json
from pathlib import Path

from feature_extract.tools.vfm.build_query_disjoint_referenced_manifests import (
    build_query_disjoint_referenced_manifests,
)


def test_query_disjoint_referenced_manifests_remove_all_heldout_images(
    tmp_path: Path,
) -> None:
    records = [
        {
            "query_id": query,
            "reference_image_id": reference,
            "row_indices": [0],
            "row_count": 1,
            "split": "train",
        }
        for query in ("q0", "q1", "q2", "q3")
        for reference in ("s0", "s1", "q0", "q1", "q2")
        if query != reference
    ]
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "format": "vfm_real_radio_joint_referenced_manifest_v1",
                "records": records,
                "record_count": len(records),
                "sample_count": len(records),
            }
        )
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": ["q0"], "validation": ["q1"], "test": ["q2"]})
    )
    paths = {
        role: tmp_path / f"{role}.json"
        for role in ("train", "development", "validation", "test")
    }

    summary = build_query_disjoint_referenced_manifests(
        source_referenced_manifest=source,
        query_split_json=split,
        output_train_manifest=paths["train"],
        output_development_manifest=paths["development"],
        output_validation_manifest=paths["validation"],
        output_test_manifest=paths["test"],
    )

    heldout = {"q0", "q1", "q2"}
    train_records = json.loads(paths["train"].read_text())["records"]
    assert all(record["query_id"] not in heldout for record in train_records)
    assert all(record["reference_image_id"] not in heldout for record in train_records)
    validation_records = json.loads(paths["validation"].read_text())["records"]
    assert {record["query_id"] for record in validation_records} == {"q1"}
    assert all(record["reference_image_id"] not in heldout for record in validation_records)
    assert summary["heldout_query_count"] == 3
    assert summary["contract"]["all_heldout_queries_removed_from_training_both_sides"] is True


def test_query_disjoint_manifest_union_covers_queries_split_across_shards(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = {
        "format": "vfm_real_radio_joint_referenced_manifest_v1",
        "cache_format": "referenced",
        "feature_key": "radio_final",
        "feature_root": "/features",
        "feature_path_template": "{image_token}.npz",
        "image_root": "/images",
    }
    first.write_text(
        json.dumps(
            {
                **common,
                "records": [
                    {"query_id": "q0", "reference_image_id": "s0", "row_indices": [0], "row_count": 1},
                    {"query_id": "q3", "reference_image_id": "s0", "row_indices": [1], "row_count": 1},
                ],
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                **common,
                "records": [
                    {"query_id": "q1", "reference_image_id": "s0", "row_indices": [2], "row_count": 1},
                    {"query_id": "q2", "reference_image_id": "s0", "row_indices": [3], "row_count": 1},
                ],
            }
        )
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["q0"], "validation": ["q1"], "test": ["q2"]}))
    paths = {role: tmp_path / f"{role}.json" for role in ("train", "development", "validation", "test")}

    summary = build_query_disjoint_referenced_manifests(
        source_referenced_manifest=first,
        additional_source_referenced_manifests=(second,),
        query_split_json=split,
        output_train_manifest=paths["train"],
        output_development_manifest=paths["development"],
        output_validation_manifest=paths["validation"],
        output_test_manifest=paths["test"],
    )

    assert summary["outputs"]["development"]["query_count"] == 1
    assert summary["outputs"]["validation"]["query_count"] == 1
    assert summary["outputs"]["test"]["query_count"] == 1
    assert len(summary["source_manifest_union"]) == 2
