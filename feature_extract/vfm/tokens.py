"""Raw VFM token-bank records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class TokenLayerSpec:
    name: str
    model: str
    layer: str
    channels: int
    stride: int


@dataclass(frozen=True)
class TokenBankRecord:
    image_id: str
    token_path: Path
    layers: Tuple[TokenLayerSpec, ...]
    split: str
    scene: str
    checksum: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_path", Path(self.token_path))
        object.__setattr__(self, "layers", tuple(self.layers))

    def require_raw_tokens(self) -> None:
        for layer in self.layers:
            if layer.channels <= 0 or layer.stride <= 0:
                raise ValueError(f"invalid token layer spec: {layer.name}")

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "token_path": str(self.token_path),
            "layers": [layer.__dict__ for layer in self.layers],
            "split": self.split,
            "scene": self.scene,
            "checksum": self.checksum,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TokenBankRecord":
        return cls(
            image_id=str(data["image_id"]),
            token_path=Path(str(data["token_path"])),
            layers=tuple(TokenLayerSpec(**dict(layer)) for layer in data["layers"]),
            split=str(data["split"]),
            scene=str(data["scene"]),
            checksum=str(data.get("checksum", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TokenBankManifest:
    records: Tuple[TokenBankRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

    def validate(self, verify_checksums: bool = True) -> None:
        seen = set()
        for record in self.records:
            record.require_raw_tokens()
            if record.image_id in seen:
                raise ValueError(f"duplicate image_id in token bank: {record.image_id}")
            seen.add(record.image_id)
            if not record.token_path.exists():
                raise ValueError(f"token file does not exist: {record.token_path}")
            if (
                verify_checksums
                and record.checksum
                and compute_file_sha256(record.token_path) != record.checksum
            ):
                raise ValueError(f"checksum mismatch for token file: {record.token_path}")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [record.to_dict() for record in self.records]}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "TokenBankManifest":
        payload = json.loads(path.read_text())
        return cls(records=tuple(TokenBankRecord.from_dict(item) for item in payload["records"]))


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz_token_record(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
