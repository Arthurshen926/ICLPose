import torch

from feature_extract.vfm.localization_goal_maplet.local_head import (
    ChildLocalReadoutHead,
    load_child_local_head,
    save_child_local_head,
)


def test_child_local_head_roundtrip_stores_no_map_embeddings(tmp_path):
    model = ChildLocalReadoutHead(4)
    path = tmp_path / "head.pt"
    save_child_local_head(path, model, {"physical_map_sha256": "abc"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "map_embeddings" not in payload
    loaded = load_child_local_head(path)
    value = torch.randn(3, 4)
    torch.testing.assert_close(loaded.model.encode_query(value), model.encode_query(value))
    assert loaded.metadata["stored_downstream_embedding_count"] == 0
