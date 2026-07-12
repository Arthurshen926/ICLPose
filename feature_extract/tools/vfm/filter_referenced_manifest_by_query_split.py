"""Remove every query-split image from both sides of a referenced pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.artifacts import file_sha256_short


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--query_split_json", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()
    input_path = Path(args.input_manifest)
    split_path = Path(args.query_split_json)
    payload = json.loads(input_path.read_text())
    split = json.loads(split_path.read_text())
    excluded = {
        str(image_id)
        for split_name in ("train", "validation", "test")
        for image_id in split.get(split_name, [])
    }
    if not excluded:
        raise ValueError("query split contains no images")
    records = list(payload.get("records", []))
    kept = [
        record
        for record in records
        if str(record.get("query_id", "")) not in excluded
        and str(record.get("reference_image_id", "")) not in excluded
    ]
    if not kept:
        raise ValueError("query filtering removed every referenced pair")
    leaked = [
        record
        for record in kept
        if str(record.get("query_id", "")) in excluded
        or str(record.get("reference_image_id", "")) in excluded
    ]
    if leaked:
        raise RuntimeError("filtered referenced manifest still contains a query image")
    payload["records"] = kept
    payload["record_count"] = int(len(kept))
    payload["sample_count"] = int(len(kept))
    payload["heldout_image_filter"] = {
        "query_split_json": str(split_path),
        "query_split_sha256": file_sha256_short(split_path),
        "heldout_image_count": int(len(excluded)),
        "removed_record_count": int(len(records) - len(kept)),
        "contract": "neither_query_nor_reference_may_be_any_train_validation_test_query_image",
    }
    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "filter_referenced_manifest_by_query_split",
        "input_record_count": int(len(records)),
        "output_record_count": int(len(kept)),
        "removed_record_count": int(len(records) - len(kept)),
        "excluded_image_count": int(len(excluded)),
        "leaked_record_count": 0,
        "inputs": {
            "input_manifest": str(input_path),
            "input_manifest_sha256": file_sha256_short(input_path),
            "query_split_json": str(split_path),
            "query_split_sha256": file_sha256_short(split_path),
        },
        "outputs": {
            "manifest": str(output),
            "manifest_sha256": file_sha256_short(output),
            "summary": str(args.summary_json),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
