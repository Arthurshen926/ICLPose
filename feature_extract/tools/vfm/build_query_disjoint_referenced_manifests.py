"""Split a referenced training manifest by query image without support leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


OUTPUT_FORMAT = "query_disjoint_referenced_manifests_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _manifest_for_records(
    source: Mapping[str, object],
    records: list[dict[str, object]],
    *,
    role: str,
    split_path: Path,
    split_sha256: str,
    heldout_count: int,
) -> dict[str, object]:
    if not records:
        raise ValueError(f"query-disjoint {role} manifest contains no records")
    payload = dict(source)
    payload["records"] = records
    payload["record_count"] = int(len(records))
    payload["sample_count"] = int(len(records))
    payload["selected_pair_count"] = int(len(records))
    payload["split_name"] = str(role)
    payload["internal_query_disjoint_filter"] = {
        "format": OUTPUT_FORMAT,
        "role": str(role),
        "query_split_json": str(split_path),
        "query_split_sha256": str(split_sha256),
        "heldout_image_count": int(heldout_count),
        "contract": "heldout_query_images_never_appear_as_training_query_or_support",
    }
    return payload


def build_query_disjoint_referenced_manifests(
    *,
    source_referenced_manifest: Path,
    query_split_json: Path,
    output_train_manifest: Path,
    output_development_manifest: Path,
    output_validation_manifest: Path,
    output_test_manifest: Path,
) -> dict[str, object]:
    source_path = Path(source_referenced_manifest)
    split_path = Path(query_split_json)
    source = json.loads(source_path.read_text())
    split = json.loads(split_path.read_text())
    records = [dict(record) for record in source.get("records", [])]
    if not records:
        raise ValueError("source referenced manifest contains no records")
    split_ids = {
        name: {str(value) for value in split.get(name, [])}
        for name in ("train", "validation", "test")
    }
    if any(not values for values in split_ids.values()):
        raise ValueError("query split train/validation/test sets must all be non-empty")
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("query split sets overlap")
    heldout = set().union(*split_ids.values())
    source_query_ids = {str(record.get("query_id", "")) for record in records}
    missing = sorted(heldout - source_query_ids)
    if missing:
        raise ValueError(
            "query-disjoint split contains images unavailable as referenced queries: "
            f"count={len(missing)}, preview={missing[:3]!r}"
        )

    train_records = [
        record
        for record in records
        if str(record.get("query_id", "")) not in heldout
        and str(record.get("reference_image_id", "")) not in heldout
    ]
    heldout_records: dict[str, list[dict[str, object]]] = {}
    for role in ("train", "validation", "test"):
        role_records = []
        for source_record in records:
            if str(source_record.get("query_id", "")) not in split_ids[role]:
                continue
            if str(source_record.get("reference_image_id", "")) in heldout:
                continue
            record = dict(source_record)
            record["split"] = str(role)
            role_records.append(record)
        covered = {str(record.get("query_id", "")) for record in role_records}
        missing_role = sorted(split_ids[role] - covered)
        if missing_role:
            raise ValueError(
                f"query-disjoint {role} images have no non-heldout referenced support: "
                f"count={len(missing_role)}, preview={missing_role[:3]!r}"
            )
        heldout_records[role] = role_records

    split_hash = _sha256(split_path)
    outputs = {
        "train": (Path(output_train_manifest), train_records),
        "development": (Path(output_development_manifest), heldout_records["train"]),
        "validation": (Path(output_validation_manifest), heldout_records["validation"]),
        "test": (Path(output_test_manifest), heldout_records["test"]),
    }
    output_summary: dict[str, object] = {}
    for role, (path, role_records) in outputs.items():
        payload = _manifest_for_records(
            source,
            role_records,
            role=role,
            split_path=split_path,
            split_sha256=split_hash,
            heldout_count=len(heldout),
        )
        _write_json(path, payload)
        output_summary[role] = {
            "path": str(path),
            "sha256": _sha256(path),
            "record_count": int(len(role_records)),
            "query_count": int(
                len({str(record.get("query_id", "")) for record in role_records})
            ),
        }

    return {
        "stage": "build_query_disjoint_referenced_manifests",
        "format": OUTPUT_FORMAT,
        "source_referenced_manifest": str(source_path),
        "source_referenced_manifest_sha256": _sha256(source_path),
        "query_split_json": str(split_path),
        "query_split_sha256": split_hash,
        "heldout_query_count": int(len(heldout)),
        "split_query_counts": {name: int(len(values)) for name, values in split_ids.items()},
        "removed_training_record_count": int(len(records) - len(train_records)),
        "outputs": output_summary,
        "contract": {
            "split_query_sets_disjoint": True,
            "all_heldout_queries_removed_from_training_both_sides": True,
            "heldout_reference_images_excluded": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_referenced_manifest", required=True)
    parser.add_argument("--query_split_json", required=True)
    parser.add_argument("--output_train_manifest", required=True)
    parser.add_argument("--output_development_manifest", required=True)
    parser.add_argument("--output_validation_manifest", required=True)
    parser.add_argument("--output_test_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_query_disjoint_referenced_manifests(
        source_referenced_manifest=Path(args.source_referenced_manifest),
        query_split_json=Path(args.query_split_json),
        output_train_manifest=Path(args.output_train_manifest),
        output_development_manifest=Path(args.output_development_manifest),
        output_validation_manifest=Path(args.output_validation_manifest),
        output_test_manifest=Path(args.output_test_manifest),
    )
    _write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
