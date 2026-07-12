from __future__ import annotations

import os

from feature_extract.vfm.artifacts import (
    _file_sha256_short_cached,
    file_sha256_short,
)


def test_file_hash_cache_reuses_unchanged_artifact_and_invalidates_rewrite(
    tmp_path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    first = file_sha256_short(path)
    cached_hits = _file_sha256_short_cached.cache_info().hits
    assert file_sha256_short(path) == first
    assert _file_sha256_short_cached.cache_info().hits == cached_hits + 1

    path.write_bytes(b"second-value")
    os.utime(path, None)
    second = file_sha256_short(path)
    assert second != first
