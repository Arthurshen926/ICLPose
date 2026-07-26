"""Build and merge a feature-evidence-gated 2DGS localization fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_query_manifest", required=True)
    parser.add_argument(
        "--primary_results_jsonl",
        required=True,
        nargs="+",
    )
    parser.add_argument("--minimum_feature_support", type=int, default=16)
    parser.add_argument("--fallback_query_manifest", required=True)
    parser.add_argument("--fallback_shard_count", type=int, default=1)
    parser.add_argument(
        "--fallback_results_jsonl",
        nargs="*",
        default=(),
    )
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def _read_results(path: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = dict(json.loads(line))
        image_id = str(row["image_id"])
        if image_id in output:
            raise ValueError(f"duplicate localization result: {image_id}")
        output[image_id] = row
    return output


def _selected_feature_support(record: dict[str, object]) -> int:
    diagnostics = dict(record.get("diagnostics") or {})
    selected_view = diagnostics.get("selected_layout_feature_view_id")
    evidence = [
        dict(row)
        for row in diagnostics.get("feature_pose_evidence", [])
        if dict(row).get("support_view_id") == selected_view
    ]
    if not evidence:
        return 0
    selected = max(
        evidence,
        key=lambda row: float(
            row.get("combined_log_likelihood")
            if row.get("combined_log_likelihood") is not None
            else -float("inf")
        ),
    )
    return int(selected.get("feature_supported_anchor_count", 0))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.minimum_feature_support) <= 0:
        raise ValueError("minimum feature support must be positive")
    if int(args.fallback_shard_count) <= 0:
        raise ValueError("fallback shard count must be positive")
    manifest = TokenBankManifest.from_json(
        Path(args.source_query_manifest)
    )
    primary: dict[str, dict[str, object]] = {}
    for path in args.primary_results_jsonl:
        rows = _read_results(Path(path))
        overlap = set(primary) & set(rows)
        if overlap:
            raise ValueError(
                f"primary result shards overlap: {sorted(overlap)[:3]}"
            )
        primary.update(rows)
    source_ids = {record.image_id for record in manifest.records}
    if not set(primary).issubset(source_ids):
        raise ValueError("primary results are absent from source manifest")
    selected_records = tuple(
        record
        for record in manifest.records
        if record.image_id in primary
    )
    manifest_ids = [record.image_id for record in selected_records]
    if set(primary) != set(manifest_ids):
        raise ValueError("primary results and selected manifest differ")
    support_by_image = {
        image_id: _selected_feature_support(primary[image_id])
        for image_id in manifest_ids
    }
    fallback_ids = [
        image_id
        for image_id in manifest_ids
        if support_by_image[image_id]
        < int(args.minimum_feature_support)
    ]
    fallback_set = set(fallback_ids)
    fallback_manifest = TokenBankManifest(
        records=tuple(
            record
            for record in selected_records
            if record.image_id in fallback_set
        )
    )
    fallback_manifest_path = Path(args.fallback_query_manifest)
    fallback_manifest.to_json(fallback_manifest_path)
    fallback_shard_paths: list[str] = []
    if int(args.fallback_shard_count) > 1:
        for shard_index in range(int(args.fallback_shard_count)):
            shard_path = fallback_manifest_path.with_name(
                f"{fallback_manifest_path.stem}_{shard_index:02d}"
                f"{fallback_manifest_path.suffix}"
            )
            TokenBankManifest(
                records=tuple(
                    record
                    for index, record in enumerate(
                        fallback_manifest.records
                    )
                    if index % int(args.fallback_shard_count)
                    == shard_index
                )
            ).to_json(shard_path)
            fallback_shard_paths.append(str(shard_path))
    summary: dict[str, object] = {
        "stage": "cascade_surface_localization_results",
        "query_count": len(manifest_ids),
        "minimum_feature_support": int(args.minimum_feature_support),
        "fallback_query_count": len(fallback_ids),
        "fallback_query_ids": fallback_ids,
        "fallback_query_manifest": str(fallback_manifest_path),
        "fallback_shard_count": int(args.fallback_shard_count),
        "fallback_shard_manifests": fallback_shard_paths,
        "merged": False,
        "production_contract": {
            "gate_uses_ground_truth": False,
            "gate_uses_query_feature_to_2dgs_feature_support": True,
            "uses_mapping_rgb_at_inference": False,
            "uses_pairwise_image_matching": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    }
    if args.fallback_results_jsonl:
        fallback: dict[str, dict[str, object]] = {}
        for path in args.fallback_results_jsonl:
            rows = _read_results(Path(path))
            overlap = set(fallback) & set(rows)
            if overlap:
                raise ValueError(
                    "fallback result shards overlap: "
                    f"{sorted(overlap)[:3]}"
                )
            fallback.update(rows)
        if set(fallback) != fallback_set:
            raise ValueError("fallback results and gated query IDs differ")
        if not str(args.output_jsonl):
            raise ValueError("output_jsonl is required while merging")
        merged = {
            image_id: (
                fallback[image_id]
                if image_id in fallback
                else primary[image_id]
            )
            for image_id in manifest_ids
        }
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            for image_id in manifest_ids:
                row = dict(merged[image_id])
                diagnostics = dict(row.get("diagnostics") or {})
                diagnostics["cascade_used_coverage_fallback"] = (
                    image_id in fallback
                )
                diagnostics[
                    "cascade_primary_feature_supported_anchor_count"
                ] = int(support_by_image[image_id])
                row["diagnostics"] = diagnostics
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary["merged"] = True
        summary["output_jsonl"] = str(output_path)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
