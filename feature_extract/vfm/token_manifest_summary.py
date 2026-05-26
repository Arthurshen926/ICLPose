"""Summaries for VFM token-bank manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_extract.vfm.tokens import TokenBankManifest


@dataclass(frozen=True)
class TokenManifestSummary:
    scene: str
    split: str
    record_count: int
    total_bytes: int
    layers: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "split": self.split,
            "record_count": self.record_count,
            "total_bytes": self.total_bytes,
            "layers": {
                name: {
                    "shape": list(info["shape"]),
                    "dtype": info["dtype"],
                }
                for name, info in self.layers.items()
            },
        }


def summarize_token_manifest(path: Path) -> TokenManifestSummary:
    manifest = TokenBankManifest.from_json(Path(path))
    manifest.validate()
    if not manifest.records:
        raise ValueError("manifest has no records")
    first = manifest.records[0]
    layers: dict[str, dict[str, object]] = {}
    with np.load(first.token_path) as data:
        for name in data.files:
            layers[name] = {
                "shape": tuple(int(dim) for dim in data[name].shape),
                "dtype": str(data[name].dtype),
            }
    total_bytes = sum(record.token_path.stat().st_size for record in manifest.records)
    return TokenManifestSummary(
        scene=first.scene,
        split=first.split,
        record_count=len(manifest.records),
        total_bytes=int(total_bytes),
        layers=layers,
    )
