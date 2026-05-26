"""Small helpers for traceable VFM experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


def file_sha256_short(path: Path, length: int = 16) -> str:
    """Return a short stable file digest for lightweight provenance records."""

    if length <= 0:
        raise ValueError("digest length must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def candidate_bank_artifact_metadata(
    bank: CandidateHypothesisBank,
    bank_path: Path,
    extra_paths: Mapping[str, Path | str | None] | None = None,
) -> dict[str, object]:
    """Build a compact provenance block for reports/checkpoint sidecars."""

    paths = {"candidate_bank": bank_path}
    for name, path in dict(extra_paths or {}).items():
        if path is None or str(path) == "":
            continue
        paths[name] = Path(path)
    return {
        "protocol_name": bank.protocol_name,
        "protocol_kind": bank.protocol_kind.value,
        "protocol_fingerprint": bank.protocol_fingerprint,
        "candidate_count": len(bank.candidates),
        "input_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256_short(Path(path)),
            }
            for name, path in paths.items()
        },
    }


def attach_report_inputs(
    report: Mapping[str, object],
    bank: CandidateHypothesisBank,
    bank_path: Path,
    extra_paths: Mapping[str, Path | str | None] | None = None,
) -> dict[str, object]:
    """Attach provenance to a metric report without changing metric keys."""

    payload = dict(report)
    payload["inputs"] = candidate_bank_artifact_metadata(bank, bank_path, extra_paths)
    return payload
