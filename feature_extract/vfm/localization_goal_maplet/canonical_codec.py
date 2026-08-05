"""Single-code, task-neutral compression for canonical RADIO-final features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .lineage import arrays_sha256, validate_deployment_metadata


SCHEMA = "goal_maplet_canonical_radio_codec_v1"


@dataclass(frozen=True)
class CanonicalRadioCodec:
    mean: np.ndarray
    components: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        components = np.asarray(self.components, dtype=np.float32)
        if components.ndim != 2 or components.shape[1] != mean.size:
            raise ValueError("canonical codec arrays differ")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(components)):
            raise ValueError("canonical codec contains non-finite values")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet canonical RADIO codec")
        validate_deployment_metadata(metadata)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "metadata", metadata)

    @property
    def input_dim(self) -> int:
        return int(self.mean.size)

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({"mean": self.mean, "components": self.components})

    def transform_rows(self, values: np.ndarray, *, chunk_size: int = 65536) -> np.ndarray:
        rows = np.asarray(values, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.input_dim:
            raise ValueError("codec input must have shape [N,input_dim]")
        output = np.empty((rows.shape[0], self.output_dim), dtype=np.float32)
        for start in range(0, rows.shape[0], int(chunk_size)):
            value = rows[start : start + int(chunk_size)]
            value = value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
            projected = (value - self.mean[None]) @ self.components.T
            output[start : start + value.shape[0]] = projected / np.maximum(
                np.linalg.norm(projected, axis=1, keepdims=True), 1e-8
            )
        return output

    def transform_map(self, values: np.ndarray) -> np.ndarray:
        feature = np.asarray(values, dtype=np.float32)
        if feature.ndim != 3 or feature.shape[0] != self.input_dim:
            raise ValueError("codec map input must have shape [C,H,W]")
        rows = feature.transpose(1, 2, 0).reshape(-1, feature.shape[0])
        output = self.transform_rows(rows)
        return output.reshape(feature.shape[1], feature.shape[2], self.output_dim).transpose(2, 0, 1)

    def save_npz(self, path: Path) -> None:
        metadata = {**dict(self.metadata), "artifact_type": SCHEMA, "content_sha256": self.content_sha256}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            mean=self.mean,
            components=self.components,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "CanonicalRadioCodec":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                np.asarray(data["mean"], dtype=np.float32),
                np.asarray(data["components"], dtype=np.float32),
                json.loads(str(np.asarray(data["metadata_json"]).item())),
            )
        declared = str(result.metadata.get("content_sha256", ""))
        if declared and declared != result.content_sha256:
            raise ValueError("canonical RADIO codec hash mismatch")
        return result
