"""Small helpers for traceable VFM experiment artifacts."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


@lru_cache(maxsize=512)
def _file_sha256_short_cached(
    resolved_path: str,
    file_size: int,
    modified_time_ns: int,
    changed_time_ns: int,
    length: int,
) -> str:
    # File metadata is part of the cache key: an artifact rewrite forces a
    # fresh digest while repeated manifest checks avoid rereading large NPZs.
    del file_size, modified_time_ns, changed_time_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def file_sha256_short(path: Path, length: int = 16) -> str:
    """Return a short stable file digest for lightweight provenance records."""

    if length <= 0:
        raise ValueError("digest length must be positive")
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return _file_sha256_short_cached(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(length),
    )


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
