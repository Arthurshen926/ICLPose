from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from feature_extract.students.radio_query_student import DepthAwareLocalMatcher
from pose_refine.tools.eval_corr_wls import load_feature_extract_local_matcher, soft_corr_flow


def test_eval_corr_wls_loads_feature_extract_local_matcher(tmp_path):
    cfg_path = tmp_path / "student.yaml"
    ckpt_path = tmp_path / "student.pth"
    cfg = {
        "model": {
            "local_matcher_enabled": True,
            "local_matcher_radius": 1,
            "local_matcher_hidden_dim": 8,
            "local_matcher_zero_init": False,
            "local_matcher_residual_scale": 1.0,
        }
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    matcher = DepthAwareLocalMatcher(radius=1, hidden_dim=8, zero_init=False, residual_scale=1.0)
    with torch.no_grad():
        matcher.residual_scale.fill_(2.5)
    state = {f"local_matcher.{k}": v.detach().clone() for k, v in matcher.state_dict().items()}
    torch.save({"model_state_dict": state}, ckpt_path)

    loaded = load_feature_extract_local_matcher(str(cfg_path), str(ckpt_path), torch.device("cpu"))

    assert loaded is not None
    assert loaded.radius == 1
    assert torch.allclose(loaded.residual_scale.detach(), torch.tensor(2.5))


def test_soft_corr_flow_applies_optional_matcher():
    rendered = torch.zeros(1, 2, 3, 3)
    query = torch.zeros(1, 2, 3, 3)
    rendered[:, 0] = 1.0
    query[:, 0] = 1.0

    class BiasMatcher(torch.nn.Module):
        def forward(self, corr, depth=None, valid_mask=None):
            out = corr.clone()
            out[:, -1] = out[:, -1] + 10.0
            return out

    flow, confidence = soft_corr_flow(rendered, query, radius=1, temperature=0.01, matcher=BiasMatcher())

    assert flow[:, 0].mean() > 0.9
    assert flow[:, 1].mean() > 0.9
    assert confidence.mean() > 0.9
