from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.fit_frozen_multiscale_candidate_appearance_residual import (
    _validate_frozen_loftr_anchor_manifest_audit,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _features(paths, query_ids, split_names):
    return SimpleNamespace(
        paths=tuple(paths),
        query_ids=np.asarray(query_ids),
        split_names=np.asarray(split_names),
    )


def _write_audit(path, *, artifacts, query_ids, split_names):
    records = [
        {
            "query_id": query_id,
            "split_name": split_name,
            "anchor": {
                "path": str(artifact),
                "sha256": file_sha256_short(artifact),
            },
        }
        for artifact, query_id, split_name in zip(artifacts, query_ids, split_names)
    ]
    path.write_text(
        json.dumps(
            {
                "format": "frozen_loftr_anchor_manifest_audit_v1",
                "protocol": {
                    "target_free": True,
                    "full_mapping_pair_cache": True,
                    "fixed_global_top_l": 20,
                    "image_level_selection": False,
                    "image_retrieval_or_submap_used": False,
                    "render": False,
                    "candidate_identity_fixed": True,
                    "source_image_manifest_revalidated": True,
                },
                "expected_query_counts": {"train": 63, "validation": 21},
                "actual_query_counts": {"train": 63, "validation": 21},
                "mapping_support_image_count": 790,
                "implementation": {},
                "records": records,
            }
        )
    )


def test_loftr_residual_fit_requires_the_exact_audited_artifact_set(tmp_path) -> None:
    query_ids = [f"train/{index:03d}.png" for index in range(63)] + [
        f"validation/{index:03d}.png" for index in range(21)
    ]
    split_names = ["train"] * 63 + ["validation"] * 21
    artifacts = []
    for index in range(84):
        artifact = tmp_path / f"anchor_{index:03d}.npz"
        artifact.write_bytes(f"anchor-{index}".encode())
        artifacts.append(artifact)
    audit = tmp_path / "audit.json"
    _write_audit(audit, artifacts=artifacts, query_ids=query_ids, split_names=split_names)

    result = _validate_frozen_loftr_anchor_manifest_audit(
        features=_features(artifacts, query_ids, split_names), path=audit
    )
    assert result["actual_query_counts"] == {"train": 63, "validation": 21}

    artifacts[0].write_bytes(b"rewritten")
    with pytest.raises(ValueError, match="does not match"):
        _validate_frozen_loftr_anchor_manifest_audit(
            features=_features(artifacts, query_ids, split_names), path=audit
        )
