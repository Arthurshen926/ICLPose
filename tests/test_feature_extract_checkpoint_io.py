import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.train_impl import safe_torch_load


def test_safe_torch_load_falls_back_for_trusted_local_checkpoint_metadata(tmp_path):
    path = tmp_path / "legacy_metadata_checkpoint.pth"
    torch.save({"tensor": torch.ones(2), "metadata_path": Path("local")}, path)

    loaded = safe_torch_load(path)

    assert torch.equal(loaded["tensor"], torch.ones(2))
    assert loaded["metadata_path"] == Path("local")
