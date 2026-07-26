from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_extract.tools.vfm.cascade_surface_localization_results import (
    main,
)
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
)


def _manifest(path: Path, image_ids: tuple[str, ...]) -> Path:
    layer = TokenLayerSpec(
        name="radio-final",
        model="radio",
        layer="final",
        channels=4,
        stride=16,
    )
    TokenBankManifest(
        records=tuple(
            TokenBankRecord(
                image_id=image_id,
                token_path=path.parent / f"{image_id}.npz",
                layers=(layer,),
                split="test",
                scene="test",
            )
            for image_id in image_ids
        )
    ).to_json(path)
    return path


def _result(image_id: str, support: int) -> dict[str, object]:
    return {
        "image_id": image_id,
        "success": True,
        "diagnostics": {
            "selected_layout_feature_view_id": "view-a",
            "feature_pose_evidence": [
                {
                    "support_view_id": "view-a",
                    "combined_log_likelihood": -1.0,
                    "feature_supported_anchor_count": support,
                }
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return path


def test_cascade_shards_and_merges_without_ground_truth(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path / "manifest.json",
        ("high", "low-a", "low-b"),
    )
    primary_a = _write_jsonl(
        tmp_path / "primary-a.jsonl",
        [_result("high", 18), _result("low-b", 1)],
    )
    primary_b = _write_jsonl(
        tmp_path / "primary-b.jsonl",
        [_result("low-a", 15)],
    )
    fallback_manifest = tmp_path / "fallback.json"
    summary = tmp_path / "summary.json"
    main(
        [
            "--source_query_manifest",
            str(manifest),
            "--primary_results_jsonl",
            str(primary_a),
            str(primary_b),
            "--minimum_feature_support",
            "16",
            "--fallback_query_manifest",
            str(fallback_manifest),
            "--fallback_shard_count",
            "2",
            "--summary_json",
            str(summary),
        ]
    )
    assert [
        record.image_id
        for record in TokenBankManifest.from_json(
            fallback_manifest
        ).records
    ] == ["low-a", "low-b"]
    shard_ids = [
        [
            record.image_id
            for record in TokenBankManifest.from_json(
                tmp_path / f"fallback_{index:02d}.json"
            ).records
        ]
        for index in range(2)
    ]
    assert shard_ids == [["low-a"], ["low-b"]]

    fallback_a = _write_jsonl(
        tmp_path / "fallback-a.jsonl",
        [_result("low-a", 99)],
    )
    fallback_b = _write_jsonl(
        tmp_path / "fallback-b.jsonl",
        [_result("low-b", 99)],
    )
    output = tmp_path / "merged.jsonl"
    main(
        [
            "--source_query_manifest",
            str(manifest),
            "--primary_results_jsonl",
            str(primary_a),
            str(primary_b),
            "--minimum_feature_support",
            "16",
            "--fallback_query_manifest",
            str(fallback_manifest),
            "--fallback_results_jsonl",
            str(fallback_a),
            str(fallback_b),
            "--output_jsonl",
            str(output),
            "--summary_json",
            str(summary),
        ]
    )
    merged = [
        json.loads(line)
        for line in output.read_text().splitlines()
    ]
    assert [row["image_id"] for row in merged] == [
        "high",
        "low-a",
        "low-b",
    ]
    assert [
        row["diagnostics"]["cascade_used_coverage_fallback"]
        for row in merged
    ] == [False, True, True]


def test_cascade_rejects_overlapping_fallback_shards(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json", ("low",))
    primary = _write_jsonl(
        tmp_path / "primary.jsonl",
        [_result("low", 0)],
    )
    fallback_a = _write_jsonl(
        tmp_path / "fallback-a.jsonl",
        [_result("low", 1)],
    )
    with pytest.raises(ValueError, match="fallback result shards overlap"):
        main(
            [
                "--source_query_manifest",
                str(manifest),
                "--primary_results_jsonl",
                str(primary),
                "--fallback_query_manifest",
                str(tmp_path / "fallback.json"),
                "--fallback_results_jsonl",
                str(fallback_a),
                str(fallback_a),
                "--output_jsonl",
                str(tmp_path / "output.jsonl"),
                "--summary_json",
                str(tmp_path / "summary.json"),
            ]
        )
