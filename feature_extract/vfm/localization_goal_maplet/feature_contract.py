"""Fail-closed query/readout contract for a Goal-Maplet feature field."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .canonical_field import CanonicalSurfaceField
from .lineage import file_sha256


SCHEMA = "goal_maplet_field_feature_contract_v1"


@dataclass(frozen=True)
class FieldFeatureContract:
    canonical_field_sha256: str
    query_readout_type: str
    query_readout_sha256: str
    render_protocol: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.query_readout_type not in {
            "surface_maplet_mapper",
            "canonical_radio_codec",
            "raw_radio_final",
        }:
            raise ValueError("unsupported query readout type")
        if self.query_readout_type != "raw_radio_final" and not self.query_readout_sha256:
            raise ValueError("learned/projected readout requires a SHA-256")
        if not self.canonical_field_sha256 or not self.render_protocol:
            raise ValueError("incomplete field feature contract")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet field feature contract")
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            {
                "canonical_field_sha256": self.canonical_field_sha256,
                "query_readout_type": self.query_readout_type,
                "query_readout_sha256": self.query_readout_sha256,
                "render_protocol": self.render_protocol,
                "metadata": dict(self.metadata),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf8")
        return hashlib.sha256(payload).hexdigest()

    def validate(
        self,
        field: CanonicalSurfaceField,
        *,
        query_readout_path: Path | None = None,
    ) -> None:
        if self.canonical_field_sha256 != field.content_sha256:
            raise ValueError("feature contract and canonical field differ")
        if self.query_readout_type == "raw_radio_final":
            if field.feature_dim != 1280:
                raise ValueError("raw RADIO readout requires a 1280-D field")
            return
        if query_readout_path is None:
            raise ValueError("feature contract requires an explicit query readout artifact")
        if file_sha256(Path(query_readout_path)) != self.query_readout_sha256:
            raise ValueError("query readout artifact SHA-256 mismatch")

    def save_json(self, path: Path) -> None:
        payload = {
            "artifact_type": SCHEMA,
            "canonical_field_sha256": self.canonical_field_sha256,
            "query_readout_type": self.query_readout_type,
            "query_readout_sha256": self.query_readout_sha256,
            "render_protocol": self.render_protocol,
            "metadata": dict(self.metadata),
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> "FieldFeatureContract":
        payload = json.loads(Path(path).read_text())
        result = cls(
            str(payload["canonical_field_sha256"]),
            str(payload["query_readout_type"]),
            str(payload.get("query_readout_sha256", "")),
            str(payload["render_protocol"]),
            payload.get("metadata", {}),
        )
        if str(payload.get("content_sha256", "")) != result.content_sha256:
            raise ValueError("field feature contract content hash mismatch")
        return result
