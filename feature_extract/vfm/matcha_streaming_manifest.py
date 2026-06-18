"""Thin manifests for on-the-fly MATCHA joint training pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from feature_extract.vfm.matcha_render_query_protocol import validate_render_query_metadata


STREAMING_PAIR_MANIFEST_FORMAT = "vfm_matcha_streaming_pair_manifest_v1"


@dataclass(frozen=True)
class MatchaStreamingPairRecord:
    query_id: str
    split: str
    pair_type: str
    pair_type_id: int
    record_index: int
    pair_index: int
    seed: int
    candidate_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": str(self.query_id),
            "split": str(self.split),
            "pair_type": str(self.pair_type),
            "pair_type_id": int(self.pair_type_id),
            "record_index": int(self.record_index),
            "pair_index": int(self.pair_index),
            "seed": int(self.seed),
            "candidate_id": str(self.candidate_id),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatchaStreamingPairRecord":
        return cls(
            query_id=str(data["query_id"]),
            split=str(data["split"]),
            pair_type=str(data["pair_type"]),
            pair_type_id=int(data["pair_type_id"]),
            record_index=int(data["record_index"]),
            pair_index=int(data["pair_index"]),
            seed=int(data["seed"]),
            candidate_id=str(data.get("candidate_id", "")),
        )


@dataclass(frozen=True)
class MatchaStreamingPairManifest:
    records: tuple[MatchaStreamingPairRecord, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        self.validate()

    def validate(self) -> None:
        if not self.records:
            raise ValueError("streaming pair manifest contains no records")
        if "pair_source" in self.metadata:
            validate_render_query_metadata(self.metadata)
        for record in self.records:
            if not record.query_id:
                raise ValueError("streaming pair record has empty query_id")
            if not record.split:
                raise ValueError("streaming pair record has empty split")
            if not record.pair_type:
                raise ValueError("streaming pair record has empty pair_type")

    @property
    def query_count(self) -> int:
        return len({record.query_id for record in self.records})

    @property
    def pair_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.pair_type] = int(counts.get(record.pair_type, 0) + 1)
        return counts

    @property
    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.split] = int(counts.get(record.split, 0) + 1)
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "format": STREAMING_PAIR_MANIFEST_FORMAT,
            "metadata": dict(self.metadata),
            "pair_count": int(len(self.records)),
            "query_count": int(self.query_count),
            "split_counts": self.split_counts,
            "pair_type_counts": self.pair_type_counts,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "MatchaStreamingPairManifest":
        payload = json.loads(Path(path).read_text())
        if str(payload.get("format", "")) != STREAMING_PAIR_MANIFEST_FORMAT:
            raise ValueError(f"unsupported streaming pair manifest format in {path}")
        return cls(
            records=tuple(MatchaStreamingPairRecord.from_dict(item) for item in payload["records"]),
            metadata=dict(payload.get("metadata", {})),
        )


def build_streaming_pair_records(
    query_ids: Sequence[str],
    *,
    split: str,
    pair_types: Sequence[str],
    pair_type_ids: Mapping[str, int],
    seed: int,
    candidate_ids: Mapping[str, str] | None = None,
) -> tuple[MatchaStreamingPairRecord, ...]:
    """Create deterministic pair records without storing image or feature tensors."""

    values: list[MatchaStreamingPairRecord] = []
    candidate_ids = dict(candidate_ids or {})
    for record_index, query_id in enumerate(query_ids):
        for pair_index, pair_type in enumerate(pair_types):
            values.append(
                MatchaStreamingPairRecord(
                    query_id=str(query_id),
                    split=str(split),
                    pair_type=str(pair_type),
                    pair_type_id=int(pair_type_ids[str(pair_type)]),
                    record_index=int(record_index),
                    pair_index=int(pair_index),
                    seed=int(seed),
                    candidate_id=str(candidate_ids.get(str(query_id), "")) if str(pair_type) == "D_reference" else "",
                )
            )
    return tuple(values)
