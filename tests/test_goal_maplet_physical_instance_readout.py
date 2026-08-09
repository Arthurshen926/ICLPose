from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    PhysicalInstanceReadout,
    PhysicalInstanceReadoutConfig,
    _region_token_sets,
    encode_physical_instance_regions,
    load_physical_instance_readout,
    save_physical_instance_readout,
    transform_canonical_field_for_role,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField


def test_position_preserving_region_sets_and_roles() -> None:
    rng = np.random.default_rng(4)
    feature = rng.normal(size=(8, 7, 9)).astype(np.float32)
    feature /= np.maximum(np.linalg.norm(feature, axis=0, keepdims=True), 1e-8)
    xy = np.asarray([[0, 0], [4, 3], [8, 6]], dtype=np.float32)
    config = PhysicalInstanceReadoutConfig(feature_dim=8, hidden_dim=16)
    context = _region_token_sets(feature, xy, role="context", config=config)
    local = _region_token_sets(feature, xy, role="local", config=config)
    assert context[0].shape == (3, 81, 8)
    assert local[0].shape == (3, 9, 8)
    assert np.allclose(np.sum(context[2], axis=1), 1.0)
    assert np.allclose(np.sum(local[2], axis=1), 1.0)
    model = PhysicalInstanceReadout(config)
    context_code = encode_physical_instance_regions(model, feature, xy, role="context", device="cpu")
    local_code = encode_physical_instance_regions(model, feature, xy, role="local", device="cpu")
    assert context_code.shape == local_code.shape == (3, 8)
    assert np.allclose(np.linalg.norm(context_code, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(context_code, local_code)


def test_region_sets_mask_missing_rendered_neighbours() -> None:
    rng = np.random.default_rng(8)
    feature = rng.normal(size=(8, 5, 5)).astype(np.float32)
    feature /= np.maximum(np.linalg.norm(feature, axis=0, keepdims=True), 1e-8)
    valid = np.zeros((5, 5), dtype=bool)
    valid[2, 2] = True
    config = PhysicalInstanceReadoutConfig(feature_dim=8, hidden_dim=16)
    tokens, _xy, weight, mask = _region_token_sets(
        feature,
        np.asarray([[2, 2]], dtype=np.int64),
        role="context",
        config=config,
        spatial_valid_mask=valid,
    )
    assert tokens.shape == (1, 81, 8)
    assert int(np.sum(mask)) == 1
    assert np.isclose(np.sum(weight), 1.0)
    model = PhysicalInstanceReadout(config)
    code = encode_physical_instance_regions(
        model,
        feature,
        np.asarray([[2, 2]], dtype=np.int64),
        role="context",
        device="cpu",
        spatial_valid_mask=valid,
    )
    assert np.allclose(code[0], feature[:, 2, 2], atol=1e-5)
    missing = encode_physical_instance_regions(
        model,
        feature,
        np.asarray([[0, 0]], dtype=np.int64),
        role="context",
        device="cpu",
        spatial_valid_mask=np.zeros((5, 5), dtype=bool),
    )
    assert np.all(np.isfinite(missing))
    assert np.allclose(missing, 0.0)


def test_physical_instance_artifact_forbids_teacher_payload(tmp_path: Path) -> None:
    model = PhysicalInstanceReadout(PhysicalInstanceReadoutConfig(feature_dim=8, hidden_dim=16))
    path = tmp_path / "readout.pt"
    save_physical_instance_readout(model, path, metadata={"teacher_roles": {"sam3": "offline"}})
    loaded, metadata = load_physical_instance_readout(path)
    assert isinstance(loaded, PhysicalInstanceReadout)
    assert metadata["stored_map_feature_type_count"] == 1
    assert metadata["stored_downstream_embedding_count"] == 0
    assert metadata["stores_teacher_embeddings"] is False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["stores_teacher_embeddings"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="deployment contract"):
        load_physical_instance_readout(path)


def test_local_role_regenerates_primitive_codes_without_extra_map_payload() -> None:
    field = CanonicalSurfaceField(
        primitive_rows=np.asarray([1, 3]),
        codes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        confidence=np.ones(2, dtype=np.float32),
        uncertainty=np.zeros(2, dtype=np.float32),
        physical_map_sha256="physical",
        metadata={
            "artifact_type": "goal_maplet_canonical_surface_field_v1",
            "stored_downstream_embedding_count": 0,
        },
    )
    model = PhysicalInstanceReadout(PhysicalInstanceReadoutConfig(feature_dim=2, hidden_dim=4))
    local = transform_canonical_field_for_role(model, field, role="local", device="cpu")
    assert local.primitive_rows.tolist() == [1, 3]
    assert local.physical_map_sha256 == field.physical_map_sha256
    assert local.metadata["stored_feature_type_count"] == 1
    assert local.metadata["stored_downstream_embedding_count"] == 0
    assert np.allclose(local.codes, field.codes)
