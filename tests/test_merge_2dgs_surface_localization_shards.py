from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_extract.tools.vfm.merge_2dgs_surface_localization_shards import (
    main,
)


def _write_shard(root: Path, name: str, image_id: str) -> tuple[Path, Path]:
    result = root / f"{name}.jsonl"
    result.write_text(
        json.dumps(
            {
                "image_id": image_id,
                "success": True,
                "pose_w2c": list(range(16)),
            }
        )
        + "\n"
    )
    summary = root / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "stage": "localize_2dgs_surface_queries_map_only",
                "query_count": 1,
                "config": {"hypothesis_count": 128},
                "production_contract": {"uses_mapping_rgb_at_inference": False},
                "artifacts": {
                    "query_manifest": {"path": f"{name}.json"},
                    "maplets": {"sha256": "fixed"},
                },
            }
        )
    )
    return result, summary


def test_merge_surface_localization_shards_uses_query_list_order(
    tmp_path: Path,
) -> None:
    first_result, first_summary = _write_shard(tmp_path, "s0", "b.png")
    second_result, second_summary = _write_shard(tmp_path, "s1", "a.png")
    query_list = tmp_path / "queries.txt"
    query_list.write_text("a.png\nb.png\n")
    output = tmp_path / "merged.jsonl"
    summary = tmp_path / "merged.json"
    main(
        [
            "--shard_jsonl",
            str(first_result),
            str(second_result),
            "--shard_summary",
            str(first_summary),
            str(second_summary),
            "--expected_query_list",
            str(query_list),
            "--output_jsonl",
            str(output),
            "--output_summary",
            str(summary),
        ]
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["image_id"] for row in rows] == ["a.png", "b.png"]
    assert json.loads(summary.read_text())["success_count"] == 2


def test_merge_surface_localization_shards_rejects_coverage_gap(
    tmp_path: Path,
) -> None:
    result, summary = _write_shard(tmp_path, "s0", "a.png")
    query_list = tmp_path / "queries.txt"
    query_list.write_text("a.png\nb.png\n")
    with pytest.raises(ValueError, match="coverage"):
        main(
            [
                "--shard_jsonl",
                str(result),
                "--shard_summary",
                str(summary),
                "--expected_query_list",
                str(query_list),
                "--output_jsonl",
                str(tmp_path / "merged.jsonl"),
                "--output_summary",
                str(tmp_path / "merged.json"),
            ]
        )


def test_merge_surface_localization_shards_accepts_successful_failure_repair(
    tmp_path: Path,
) -> None:
    result, summary = _write_shard(tmp_path, "s0", "a.png")
    failed = json.loads(result.read_text())
    failed["success"] = False
    failed["pose_w2c"] = None
    result.write_text(json.dumps(failed) + "\n")
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(
        json.dumps(
            {
                "image_id": "a.png",
                "success": True,
                "pose_w2c": list(range(16)),
            }
        )
        + "\n"
    )
    query_list = tmp_path / "queries.txt"
    query_list.write_text("a.png\n")
    output = tmp_path / "merged.jsonl"
    output_summary = tmp_path / "merged.json"
    main(
        [
            "--shard_jsonl",
            str(result),
            "--shard_summary",
            str(summary),
            "--replacement_jsonl",
            str(replacement),
            "--expected_query_list",
            str(query_list),
            "--output_jsonl",
            str(output),
            "--output_summary",
            str(output_summary),
        ]
    )
    assert json.loads(output.read_text())["success"] is True
    assert len(
        json.loads(output_summary.read_text())["failure_replacements"]
    ) == 1
