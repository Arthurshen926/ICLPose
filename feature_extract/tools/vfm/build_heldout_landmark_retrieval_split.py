"""Build a leakage-free landmark-retrieval split from referenced query images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.tokens import TokenBankManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_heldout_landmark_retrieval_split(
    *,
    track_observations_jsonl: Path,
    token_manifest: Path,
    query_referenced_manifest: Path,
    output_support_observations_jsonl: Path,
    output_query_token_manifest: Path,
    output_support_token_manifest: Path | None = None,
    train_referenced_manifest: Path | None = None,
    output_train_referenced_manifest: Path | None = None,
) -> dict[str, object]:
    referenced = json.loads(Path(query_referenced_manifest).read_text())
    records = list(referenced.get("records", []))
    heldout_query_ids = sorted({str(record.get("query_id", "")) for record in records if record.get("query_id")})
    if not heldout_query_ids:
        raise ValueError("query referenced manifest contains no query ids")
    heldout_set = set(heldout_query_ids)
    source_manifest = TokenBankManifest.from_json(Path(token_manifest))
    source_manifest.validate(verify_checksums=False)
    records_by_id = {str(record.image_id): record for record in source_manifest.records}
    missing_tokens = sorted(heldout_set - set(records_by_id))
    if missing_tokens:
        raise ValueError(f"held-out query ids missing from token manifest: {missing_tokens[:10]!r}")
    query_manifest = TokenBankManifest(records=tuple(records_by_id[query_id] for query_id in heldout_query_ids))
    query_manifest.to_json(Path(output_query_token_manifest))
    support_manifest_summary: dict[str, object] = {}
    if output_support_token_manifest is not None:
        support_manifest = TokenBankManifest(
            records=tuple(record for record in source_manifest.records if str(record.image_id) not in heldout_set)
        )
        if not support_manifest.records:
            raise ValueError("held-out filtering removed every support token")
        support_manifest.to_json(Path(output_support_token_manifest))
        support_manifest_summary = {
            "path": str(output_support_token_manifest),
            "record_count": int(len(support_manifest.records)),
            "sha256": _sha256(Path(output_support_token_manifest)),
        }

    input_count = 0
    excluded_count = 0
    support_count = 0
    excluded_images_seen: set[str] = set()
    output_support = Path(output_support_observations_jsonl)
    output_support.parent.mkdir(parents=True, exist_ok=True)
    temporary_support = output_support.with_suffix(output_support.suffix + ".tmp")
    with Path(track_observations_jsonl).open() as source, temporary_support.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            input_count += 1
            item = json.loads(line)
            image_id = str(item.get("image_id", ""))
            if image_id in heldout_set:
                excluded_count += 1
                excluded_images_seen.add(image_id)
                continue
            target.write(line if line.endswith("\n") else line + "\n")
            support_count += 1
    missing_observations = sorted(heldout_set - excluded_images_seen)
    if missing_observations:
        temporary_support.unlink(missing_ok=True)
        raise ValueError(f"held-out query ids have no track observations: {missing_observations[:10]!r}")
    temporary_support.replace(output_support)
    filtered_train_summary: dict[str, object] = {}
    if (train_referenced_manifest is None) != (output_train_referenced_manifest is None):
        raise ValueError("train_referenced_manifest and output_train_referenced_manifest must be provided together")
    if train_referenced_manifest is not None and output_train_referenced_manifest is not None:
        train_payload = json.loads(Path(train_referenced_manifest).read_text())
        train_records = list(train_payload.get("records", []))
        kept_records = [
            record
            for record in train_records
            if str(record.get("query_id", "")) not in heldout_set
            and str(record.get("reference_image_id", "")) not in heldout_set
        ]
        removed_records = int(len(train_records) - len(kept_records))
        if not kept_records:
            raise ValueError("held-out filtering removed every training referenced record")
        train_payload["records"] = kept_records
        train_payload["record_count"] = int(len(kept_records))
        train_payload["sample_count"] = int(len(kept_records))
        train_payload["heldout_image_filter"] = {
            "query_referenced_manifest": str(query_referenced_manifest),
            "heldout_image_count": int(len(heldout_set)),
            "removed_record_count": int(removed_records),
            "contract": "neither_query_nor_reference_may_be_a_heldout_image",
        }
        filtered_output = Path(output_train_referenced_manifest)
        filtered_output.parent.mkdir(parents=True, exist_ok=True)
        filtered_output.write_text(json.dumps(train_payload, indent=2, sort_keys=True) + "\n")
        filtered_train_summary = {
            "input_record_count": int(len(train_records)),
            "output_record_count": int(len(kept_records)),
            "removed_record_count": int(removed_records),
            "path": str(filtered_output),
            "sha256": _sha256(filtered_output),
        }
    return {
        "stage": "heldout_landmark_retrieval_split",
        "split_contract": "query_images_excluded_from_all_prototype_support_observations",
        "heldout_query_count": int(len(heldout_query_ids)),
        "input_observation_count": int(input_count),
        "excluded_query_observation_count": int(excluded_count),
        "support_observation_count": int(support_count),
        "query_token_count": int(len(query_manifest.records)),
        "inputs": {
            "track_observations_jsonl": str(track_observations_jsonl),
            "token_manifest": str(token_manifest),
            "query_referenced_manifest": str(query_referenced_manifest),
        },
        "outputs": {
            "support_observations_jsonl": str(output_support),
            "support_observations_sha256": _sha256(output_support),
            "query_token_manifest": str(output_query_token_manifest),
            "query_token_manifest_sha256": _sha256(Path(output_query_token_manifest)),
            "support_token_manifest": support_manifest_summary,
            "filtered_train_referenced_manifest": filtered_train_summary,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations_jsonl", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--query_referenced_manifest", required=True)
    parser.add_argument("--output_support_observations_jsonl", required=True)
    parser.add_argument("--output_query_token_manifest", required=True)
    parser.add_argument("--output_support_token_manifest", default="")
    parser.add_argument("--train_referenced_manifest", default="")
    parser.add_argument("--output_train_referenced_manifest", default="")
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_heldout_landmark_retrieval_split(
        track_observations_jsonl=Path(args.track_observations_jsonl),
        token_manifest=Path(args.token_manifest),
        query_referenced_manifest=Path(args.query_referenced_manifest),
        output_support_observations_jsonl=Path(args.output_support_observations_jsonl),
        output_query_token_manifest=Path(args.output_query_token_manifest),
        output_support_token_manifest=(
            Path(args.output_support_token_manifest) if str(args.output_support_token_manifest) else None
        ),
        train_referenced_manifest=(Path(args.train_referenced_manifest) if str(args.train_referenced_manifest) else None),
        output_train_referenced_manifest=(
            Path(args.output_train_referenced_manifest) if str(args.output_train_referenced_manifest) else None
        ),
    )
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
