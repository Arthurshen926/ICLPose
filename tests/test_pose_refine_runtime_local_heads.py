from __future__ import annotations

import sys
import shutil
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.runtime import build_concat_pose_model, load_concat_pose_model


def _small_corr_wls_config(**overrides):
    cfg = {
        "feature_dim": 4,
        "coarse_feature_dim": 4,
        "hidden_dim": 16,
        "use_corr_wls": True,
        "full_wls": True,
        "use_gru": False,
        "local_radius": 1,
        "proj_dim": 4,
        "proj_mode": "shared_linear",
        "local_flow_head_hidden_dim": 8,
        "local_flow_head_base_flow_mode": "softargmax",
        "local_flow_head_base_temperature": 0.05,
        "local_flow_head_context_mode": "basic",
    }
    cfg.update(overrides)
    return cfg


def test_load_concat_pose_model_can_inject_local_flow_head_from_aux_checkpoint(tmp_path):
    repo_tmp = Path.cwd() / "result" / "tmp_tests" / tmp_path.name
    shutil.rmtree(repo_tmp, ignore_errors=True)
    repo_tmp.mkdir(parents=True)
    try:
        base_model = build_concat_pose_model(_small_corr_wls_config(local_flow_head_enabled=False), "cpu")
        base_ckpt = repo_tmp / "base_pose_refine.pth"
        torch.save({"epoch": 3, "model_state_dict": base_model.state_dict()}, base_ckpt)

        source_model = build_concat_pose_model(_small_corr_wls_config(local_flow_head_enabled=True), "cpu")
        with torch.no_grad():
            for param in source_model.local_flow_head.parameters():
                param.fill_(0.123)
        source_ckpt = repo_tmp / "source_feature_extract.pth"
        torch.save({"model_state_dict": source_model.state_dict()}, source_ckpt)

        model, epoch = load_concat_pose_model(
            {
                "model": _small_corr_wls_config(
                    local_flow_head_enabled=True,
                    local_flow_head_init_checkpoint=str(source_ckpt),
                )
            },
            str(base_ckpt),
            "cpu",
            printer=None,
        )
    finally:
        shutil.rmtree(repo_tmp, ignore_errors=True)

    assert epoch == 3
    assert model.local_flow_head is not None
    first_param = next(model.local_flow_head.parameters())
    assert torch.allclose(first_param, torch.full_like(first_param, 0.123))
