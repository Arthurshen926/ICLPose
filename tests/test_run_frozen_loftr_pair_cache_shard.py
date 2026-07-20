from __future__ import annotations

from pathlib import Path

from feature_extract.tools.vfm.run_frozen_loftr_pair_cache_shard import (
    QueryArtifact,
    partition_query_artifacts,
)


def test_loftr_shard_partition_is_disjoint_and_exhaustive() -> None:
    artifacts = tuple(
        QueryArtifact(Path(f"/tmp/{index}.npz"), f"seq/frame{index:02d}.png", "train")
        for index in range(7)
    )
    shards = [
        partition_query_artifacts(artifacts, shard_count=3, shard_index=index)
        for index in range(3)
    ]
    positions = [position for shard in shards for position, _artifact in shard]
    assert sorted(positions) == list(range(7))
    assert len(positions) == len(set(positions))
    assert [position for position, _artifact in shards[1]] == [1, 4]
