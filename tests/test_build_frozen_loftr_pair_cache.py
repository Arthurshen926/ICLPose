from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_frozen_loftr_pair_cache import (
    kornia_pretrained_checkpoint_path,
    _load_mapping_support_manifest,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _write_maplet(path) -> None:
    np.savez_compressed(
        path,
        anchor_track_ids=np.asarray([10], dtype=np.int64),
        support_image_ids=np.asarray(["support-a.png", "support-b.png"]),
        support_image_indices=np.asarray([[0, 1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[1, 1]], dtype=np.int64),
        metadata_json=np.asarray(json.dumps({"format": "local_maplet_support_index_npz"})),
    )


def _write_manifest(path, *, maplet_path, records) -> None:
    payload = {
        "format": "maplet_support_image_manifest_v1",
        "records": [{"image_id": item} for item in records],
        "metadata": {
            "maplet_support_index_sha256": file_sha256_short(maplet_path),
            "support_query_overlap_count": 0,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_or_ground_truth_used": False,
        },
    }
    path.write_text(json.dumps(payload))


def test_mapping_manifest_requires_exact_current_full_maplet_image_set(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    manifest = tmp_path / "support.json"
    _write_maplet(maplet)
    _write_manifest(manifest, maplet_path=maplet, records=["support-a.png", "support-b.png"])
    ids, metadata = _load_mapping_support_manifest(
        path=manifest, maplet_support_index=maplet, query_id="query.png"
    )
    assert ids.tolist() == ["support-a.png", "support-b.png"]
    assert metadata["support_query_overlap_count"] == 0

    _write_manifest(manifest, maplet_path=maplet, records=["support-a.png"])
    with pytest.raises(ValueError, match="exactly"):
        _load_mapping_support_manifest(
            path=manifest, maplet_support_index=maplet, query_id="query.png"
        )


def test_kornia_preset_checkpoint_path_is_bound_to_the_torch_hub_cache(tmp_path) -> None:
    path = kornia_pretrained_checkpoint_path(
        weights="outdoor",
        hub_dir=tmp_path / "hub",
        urls={"outdoor": "https://example.test/models/loftr_outdoor.ckpt"},
    )
    assert path == (tmp_path / "hub" / "checkpoints" / "loftr_outdoor.ckpt").resolve()

    with pytest.raises(ValueError, match="safe checkpoint filename"):
        kornia_pretrained_checkpoint_path(
            weights="outdoor",
            hub_dir=tmp_path / "hub",
            urls={"outdoor": "https://example.test/models/"},
        )
