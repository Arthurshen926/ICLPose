"""Build a sequence-balanced, leakage-free landmark-retrieval development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.tokens import TokenBankManifest


SPLIT_FORMAT = "stratified_landmark_query_split_v1"
SPLIT_STRATEGY = "sequence_balanced_temporal_coverage_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_and_frame(image_id: str) -> tuple[str, int]:
    match = re.search(r"(^|/)(?P<seq>seq[^/]+)/frame(?P<frame>\d+)\.[^.]+$", str(image_id))
    if match is None:
        raise ValueError(f"cannot parse sequence/frame from image id: {image_id!r}")
    return str(match.group("seq")), int(match.group("frame"))


def _quantile_positions(count: int, selected_count: int, *, phase: float) -> tuple[int, ...]:
    if selected_count < 0 or selected_count > count:
        raise ValueError("selected_count must be between zero and count")
    if selected_count == 0:
        return ()
    positions = [min(count - 1, int(math.floor((index + float(phase)) * count / selected_count))) for index in range(selected_count)]
    if len(set(positions)) != selected_count:
        raise RuntimeError("quantile selection produced duplicate positions")
    return tuple(positions)


def _temporal_split_labels(
    total_count: int,
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> tuple[str, ...]:
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("train, validation, and test counts must all be positive")
    if train_count + validation_count + test_count != total_count:
        raise ValueError("split counts must sum to total_count")
    labels = ["train"] * total_count
    validation_positions = set(
        _quantile_positions(total_count, validation_count, phase=0.5)
    )
    for position in validation_positions:
        labels[position] = "validation"

    # Stagger test positions after the validation quantiles. This keeps both
    # held-out blocks spread over the sequence without sharing an image.
    test_targets = _quantile_positions(total_count, test_count, phase=1.0)
    available = set(range(total_count)) - validation_positions
    test_positions: list[int] = []
    for target in test_targets:
        candidates = sorted(available, key=lambda value: (abs(value - target), -value))
        if not candidates:
            raise RuntimeError("no temporal position remains for the test split")
        chosen = int(candidates[0])
        available.remove(chosen)
        test_positions.append(chosen)
    for position in test_positions:
        labels[position] = "test"
    counts = Counter(labels)
    expected = {
        "train": int(train_count),
        "validation": int(validation_count),
        "test": int(test_count),
    }
    if dict(counts) != expected:
        raise RuntimeError(f"temporal split allocation differs from requested counts: {dict(counts)!r}")
    return tuple(labels)


def _observation_counts_from_index(
    index_path: Path,
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, int]:
    with np.load(Path(index_path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if str(metadata.get("format", "")) != "sfm_track_observation_index_v1":
            raise ValueError("unsupported track-observation index format")
        if int(metadata.get("source_size", -1)) != int(Path(source_path).stat().st_size):
            raise ValueError("track-observation index source size is stale")
        recorded_hash = str(metadata.get("source_sha256", ""))
        if not recorded_hash or not source_sha256.startswith(recorded_hash):
            raise ValueError("track-observation index source hash is stale")
        image_ids = np.asarray(data["image_ids"]).astype(str)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
    if offsets.shape != (len(image_ids) + 1,) or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid track-observation index offsets")
    return {
        str(image_id): int(count)
        for image_id, count in zip(image_ids.tolist(), np.diff(offsets).tolist())
    }


def _observation_counts_from_jsonl(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = str(item.get("image_id", ""))
            if not image_id:
                raise ValueError("track observation has no image_id")
            counts[image_id] += 1
    return dict(counts)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_stratified_landmark_retrieval_split(
    *,
    track_observations_jsonl: Path,
    token_manifest: Path,
    output_support_observations_jsonl: Path,
    output_query_token_manifest: Path,
    output_support_token_manifest: Path,
    output_query_split_json: Path,
    sequences: Sequence[str] | None = None,
    train_per_sequence: int = 9,
    validation_per_sequence: int = 3,
    test_per_sequence: int = 3,
    min_query_observations: int = 32,
    track_observation_index: Path | None = None,
) -> dict[str, object]:
    source_tracks = Path(track_observations_jsonl)
    source_tokens = Path(token_manifest)
    source_track_sha256 = _sha256(source_tracks)
    source_token_sha256 = _sha256(source_tokens)
    manifest = TokenBankManifest.from_json(source_tokens)
    manifest.validate(verify_checksums=False)

    if track_observation_index is None:
        observation_counts = _observation_counts_from_jsonl(source_tracks)
    else:
        observation_counts = _observation_counts_from_index(
            Path(track_observation_index),
            source_path=source_tracks,
            source_sha256=source_track_sha256,
        )

    records_by_sequence: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for record in manifest.records:
        sequence, frame = _sequence_and_frame(str(record.image_id))
        if int(observation_counts.get(str(record.image_id), 0)) >= int(min_query_observations):
            records_by_sequence[sequence].append((frame, record))
    for records in records_by_sequence.values():
        records.sort(key=lambda item: (item[0], str(item[1].image_id)))

    selected_sequences = tuple(sorted(records_by_sequence)) if sequences is None else tuple(str(value) for value in sequences)
    if not selected_sequences:
        raise ValueError("no sequences were selected")
    if len(set(selected_sequences)) != len(selected_sequences):
        raise ValueError("selected sequences contain duplicates")
    total_per_sequence = int(train_per_sequence) + int(validation_per_sequence) + int(test_per_sequence)
    labels = _temporal_split_labels(
        total_per_sequence,
        train_count=int(train_per_sequence),
        validation_count=int(validation_per_sequence),
        test_count=int(test_per_sequence),
    )
    split_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    sequence_summary: dict[str, object] = {}
    selected_record_by_id: dict[str, object] = {}
    for sequence in selected_sequences:
        eligible = records_by_sequence.get(sequence, [])
        if len(eligible) < total_per_sequence:
            raise ValueError(
                f"sequence {sequence!r} has {len(eligible)} eligible images, "
                f"but {total_per_sequence} are required"
            )
        selected_positions = _quantile_positions(len(eligible), total_per_sequence, phase=0.5)
        selected = [eligible[position] for position in selected_positions]
        per_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
        for label, (_, record) in zip(labels, selected):
            image_id = str(record.image_id)
            split_ids[label].append(image_id)
            per_split[label].append(image_id)
            selected_record_by_id[image_id] = record
        sequence_summary[sequence] = {
            "eligible_image_count": int(len(eligible)),
            "selected_image_count": int(len(selected)),
            "selected": per_split,
            "selected_observation_counts": {
                str(record.image_id): int(observation_counts[str(record.image_id)])
                for _, record in selected
            },
        }

    heldout_ids = [*split_ids["train"], *split_ids["validation"], *split_ids["test"]]
    heldout_set = set(heldout_ids)
    if len(heldout_set) != len(heldout_ids):
        raise RuntimeError("query split contains duplicate image ids")
    query_manifest = TokenBankManifest(records=tuple(selected_record_by_id[value] for value in heldout_ids))
    support_manifest = TokenBankManifest(
        records=tuple(record for record in manifest.records if str(record.image_id) not in heldout_set)
    )
    if not support_manifest.records:
        raise ValueError("held-out filtering removed every support image")
    query_manifest.to_json(Path(output_query_token_manifest))
    support_manifest.to_json(Path(output_support_token_manifest))

    split_payload: dict[str, object] = {
        "format": SPLIT_FORMAT,
        "strategy": SPLIT_STRATEGY,
        "train": split_ids["train"],
        "validation": split_ids["validation"],
        "test": split_ids["test"],
        "per_sequence": sequence_summary,
        "contract": {
            "query_images_excluded_from_support_tokens": True,
            "query_images_excluded_from_support_observations": True,
            "split_sets_are_disjoint": True,
            "temporal_coverage_per_sequence": True,
        },
        "inputs": {
            "token_manifest": str(source_tokens),
            "token_manifest_sha256": source_token_sha256,
            "track_observations_jsonl": str(source_tracks),
            "track_observations_sha256": source_track_sha256,
            "track_observation_index": "" if track_observation_index is None else str(track_observation_index),
        },
        "selection": {
            "sequences": list(selected_sequences),
            "train_per_sequence": int(train_per_sequence),
            "validation_per_sequence": int(validation_per_sequence),
            "test_per_sequence": int(test_per_sequence),
            "min_query_observations": int(min_query_observations),
            "image_sampling": "temporal_quantile_centers",
            "validation_positions": "temporal_bin_centers",
            "test_positions": "staggered_temporal_quantiles_phase_1p0",
        },
    }
    _write_json(Path(output_query_split_json), split_payload)

    input_count = 0
    excluded_count = 0
    support_count = 0
    excluded_images_seen: set[str] = set()
    output_support = Path(output_support_observations_jsonl)
    output_support.parent.mkdir(parents=True, exist_ok=True)
    temporary_support = output_support.with_suffix(output_support.suffix + ".tmp")
    with source_tracks.open() as source, temporary_support.open("w") as target:
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

    return {
        "stage": "stratified_landmark_retrieval_split",
        "format": SPLIT_FORMAT,
        "split_strategy": SPLIT_STRATEGY,
        "query_count": int(len(heldout_ids)),
        "support_image_count": int(len(support_manifest.records)),
        "input_observation_count": int(input_count),
        "excluded_query_observation_count": int(excluded_count),
        "support_observation_count": int(support_count),
        "split_counts": {name: int(len(values)) for name, values in split_ids.items()},
        "sequence_count": int(len(selected_sequences)),
        "per_sequence": sequence_summary,
        "inputs": split_payload["inputs"],
        "outputs": {
            "query_token_manifest": str(output_query_token_manifest),
            "query_token_manifest_sha256": _sha256(Path(output_query_token_manifest)),
            "support_token_manifest": str(output_support_token_manifest),
            "support_token_manifest_sha256": _sha256(Path(output_support_token_manifest)),
            "query_split_json": str(output_query_split_json),
            "query_split_sha256": _sha256(Path(output_query_split_json)),
            "support_observations_jsonl": str(output_support),
            "support_observations_sha256": _sha256(output_support),
        },
    }


def _comma_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations_jsonl", required=True)
    parser.add_argument("--track_observation_index", default="")
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--output_support_observations_jsonl", required=True)
    parser.add_argument("--output_query_token_manifest", required=True)
    parser.add_argument("--output_support_token_manifest", required=True)
    parser.add_argument("--output_query_split_json", required=True)
    parser.add_argument("--sequences", type=_comma_list, default=())
    parser.add_argument("--train_per_sequence", type=int, default=9)
    parser.add_argument("--validation_per_sequence", type=int, default=3)
    parser.add_argument("--test_per_sequence", type=int, default=3)
    parser.add_argument("--min_query_observations", type=int, default=32)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_stratified_landmark_retrieval_split(
        track_observations_jsonl=Path(args.track_observations_jsonl),
        track_observation_index=(Path(args.track_observation_index) if args.track_observation_index else None),
        token_manifest=Path(args.token_manifest),
        output_support_observations_jsonl=Path(args.output_support_observations_jsonl),
        output_query_token_manifest=Path(args.output_query_token_manifest),
        output_support_token_manifest=Path(args.output_support_token_manifest),
        output_query_split_json=Path(args.output_query_split_json),
        sequences=(tuple(args.sequences) if args.sequences else None),
        train_per_sequence=int(args.train_per_sequence),
        validation_per_sequence=int(args.validation_per_sequence),
        test_per_sequence=int(args.test_per_sequence),
        min_query_observations=int(args.min_query_observations),
    )
    _write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
