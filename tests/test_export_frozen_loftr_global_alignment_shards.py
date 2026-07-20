from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.export_frozen_loftr_global_alignment_shards import (
    _cache_query_id,
    _index_paths_by_query_id,
    _source_cache_manifest,
    _source_query_id,
)


def _source(path: Path, query_id: str) -> Path:
    np.savez(
        path,
        verification_query_ids=np.asarray([query_id] * 192, dtype=np.str_),
    )
    return path


def _cache(path: Path, query_id: str) -> Path:
    np.savez(path, query_id=np.asarray([query_id], dtype=np.str_))
    return path


def test_exporter_matches_source_and_cache_by_query_id_not_filename(tmp_path: Path) -> None:
    source_a = _source(tmp_path / "source-b.npz", "seq1/frame00011.png")
    source_b = _source(tmp_path / "source-a.npz", "seq2/frame00011.png")
    cache_a = _cache(tmp_path / "cache-a.npz", "seq2/frame00011.png")
    cache_b = _cache(tmp_path / "cache-b.npz", "seq1/frame00011.png")
    sources = _index_paths_by_query_id((source_a, source_b), kind="source")
    caches = _index_paths_by_query_id((cache_a, cache_b), kind="cache")

    manifest = _source_cache_manifest(
        source_by_query=sources,
        cache_by_query=caches,
    )

    assert [item["query_id"] for item in manifest] == [
        "seq1/frame00011.png",
        "seq2/frame00011.png",
    ]
    assert manifest[0]["source"].endswith("source-b.npz")
    assert manifest[0]["pair_cache"].endswith("cache-b.npz")


def test_exporter_rejects_duplicate_or_incomplete_query_layout(tmp_path: Path) -> None:
    source_a = _source(tmp_path / "source-a.npz", "seq1/frame00011.png")
    source_b = _source(tmp_path / "source-b.npz", "seq1/frame00011.png")
    with pytest.raises(ValueError, match="duplicate source"):
        _index_paths_by_query_id((source_a, source_b), kind="source")

    sources = _index_paths_by_query_id((source_a,), kind="source")
    caches = _index_paths_by_query_id(
        (_cache(tmp_path / "cache.npz", "seq2/frame00011.png"),), kind="cache"
    )
    with pytest.raises(ValueError, match="query sets differ"):
        _source_cache_manifest(source_by_query=sources, cache_by_query=caches)


def test_exporter_requires_one_complete_source_query_and_one_cache_query(tmp_path: Path) -> None:
    bad_source = tmp_path / "bad-source.npz"
    np.savez(bad_source, verification_query_ids=np.asarray(["a.png"] * 191))
    with pytest.raises(ValueError, match="one 192-row query"):
        _source_query_id(bad_source)

    bad_cache = tmp_path / "bad-cache.npz"
    np.savez(bad_cache, query_id=np.asarray(["a.png", "b.png"]))
    with pytest.raises(ValueError, match="exactly one"):
        _cache_query_id(bad_cache)
