from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.utils.radio_loader import (  # noqa: E402
    ensure_load_state_dict_from_url_weights_only_compatible,
)


def test_load_state_dict_from_url_compatibility_accepts_weights_only(monkeypatch):
    calls = []

    def legacy_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
        calls.append(
            {
                "url": url,
                "model_dir": model_dir,
                "map_location": map_location,
                "progress": progress,
                "check_hash": check_hash,
                "file_name": file_name,
            }
        )
        return {"state_dict": {}}

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", legacy_load_state_dict_from_url)

    ensure_load_state_dict_from_url_weights_only_compatible()
    result = torch.hub.load_state_dict_from_url(
        "https://example.invalid/model.pth",
        map_location="cpu",
        progress=False,
        weights_only=False,
    )

    assert result == {"state_dict": {}}
    assert calls == [
        {
            "url": "https://example.invalid/model.pth",
            "model_dir": None,
            "map_location": "cpu",
            "progress": False,
            "check_hash": False,
            "file_name": None,
        }
    ]
