from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_field import train_impl  # noqa: E402


def test_init_distributed_supports_legacy_process_group_signature(monkeypatch):
    set_device_calls = []
    init_calls = []

    def fake_set_device(local_rank):
        set_device_calls.append(local_rank)

    def fake_init_process_group(backend):
        init_calls.append({"backend": backend})

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(train_impl.torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(train_impl.dist, "init_process_group", fake_init_process_group)

    assert train_impl._init_distributed() == (True, 0, 1, 2)
    assert set_device_calls == [1]
    assert init_calls == [{"backend": "nccl"}]
