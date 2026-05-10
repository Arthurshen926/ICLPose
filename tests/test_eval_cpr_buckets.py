import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools import eval_cpr_buckets


def test_parse_args_accepts_separate_map_checkpoint(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--map-checkpoint",
            "map.pth",
        ],
    )

    try:
        args = eval_cpr_buckets.parse_args()
    except SystemExit as exc:
        raise AssertionError("--map-checkpoint should be accepted by eval_cpr_buckets.py") from exc

    assert args.map_checkpoint == "map.pth"


def test_resolve_map_renderer_state_prefers_separate_map_checkpoint(tmp_path):
    model_state = {"map_renderer_state_dict": {"source": "model"}}
    map_state = {"map_renderer_state_dict": {"source": "map"}}
    map_checkpoint = tmp_path / "map.pth"
    torch.save(map_state, map_checkpoint)
    args = SimpleNamespace(map_checkpoint=str(map_checkpoint))

    resolver = getattr(eval_cpr_buckets, "resolve_map_renderer_state", None)
    assert resolver is not None, "eval_cpr_buckets should expose resolve_map_renderer_state"

    resolved = resolver(model_state, args)

    assert resolved == {"source": "map"}


def test_parse_args_accepts_candidate_render_batch_size(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--candidate-render-batch-size",
            "32",
        ],
    )

    try:
        args = eval_cpr_buckets.parse_args()
    except SystemExit as exc:
        raise AssertionError("--candidate-render-batch-size should be accepted") from exc

    assert args.candidate_render_batch_size == 32


def test_parse_args_accepts_fine_score_stat(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--fine-score-stat",
            "topk_mean",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.fine_score_stat == "topk_mean"


def test_apply_eval_overrides_sets_candidate_render_batch_size():
    cfg = {"map_supervision": {"candidate_render_batch_size": 0}}
    args = SimpleNamespace(candidate_render_batch_size=16)

    apply_overrides = getattr(eval_cpr_buckets, "apply_eval_overrides", None)
    assert apply_overrides is not None, "eval_cpr_buckets should expose apply_eval_overrides"

    apply_overrides(cfg, args)

    assert cfg["map_supervision"]["candidate_render_batch_size"] == 16


def test_fine_prior_adjusted_scores_prefers_near_init_when_scores_tie():
    init_pose = torch.eye(4).view(1, 4, 4)
    selected_pose = init_pose.view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    selected_pose[0, 1, 0, 3] = 0.20
    fine_scores = torch.zeros(1, 2)

    adjusted = eval_cpr_buckets.fine_prior_adjusted_scores(
        fine_scores,
        selected_pose,
        init_pose,
        prior_weight=1.0,
        rot_cost_weight=0.1,
    )

    assert adjusted.argmax(dim=1).item() == 0


def test_fine_prior_adjusted_scores_is_noop_when_weight_zero():
    init_pose = torch.eye(4).view(1, 4, 4)
    selected_pose = init_pose.view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    selected_pose[0, 1, 0, 3] = 0.20
    fine_scores = torch.tensor([[0.1, 0.3]])

    adjusted = eval_cpr_buckets.fine_prior_adjusted_scores(
        fine_scores,
        selected_pose,
        init_pose,
        prior_weight=0.0,
        rot_cost_weight=0.1,
    )

    assert torch.allclose(adjusted, fine_scores)
