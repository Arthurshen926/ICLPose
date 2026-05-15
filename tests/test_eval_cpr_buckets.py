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


def test_parse_args_accepts_jittered_eval_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--init-noise-mode",
            "random",
            "--init-jitter-seed",
            "7",
            "--candidate-jitter-cm",
            "1.5",
            "--candidate-jitter-deg",
            "0.5",
            "--disable-exact-inverse",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.init_noise_mode == "random"
    assert args.init_jitter_seed == 7
    assert args.candidate_jitter_cm == 1.5
    assert args.candidate_jitter_deg == 0.5
    assert args.disable_exact_inverse is True


def test_parse_args_accepts_trainable_fine_selector(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--fine-select",
            "fine_selector",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.fine_select == "fine_selector"


def test_parse_args_accepts_pose_energy_selector_checkpoint(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--fine-select",
            "pose_energy",
            "--pose-energy-checkpoint",
            "stage2.pth",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.fine_select == "pose_energy"
    assert args.pose_energy_checkpoint == "stage2.pth"


def test_parse_args_accepts_fine_selector_adapter_checkpoint(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--fine-select",
            "fine_selector",
            "--fine-selector-adapter-checkpoint",
            "stage1_adapter.pth",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.fine_selector_adapter_checkpoint == "stage1_adapter.pth"


def test_parse_args_accepts_fine_selector_pair_matcher_score_source(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--fine-select",
            "fine_selector",
            "--fine-selector-score-source",
            "pair_matcher_heatmap",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.fine_selector_score_source == "pair_matcher_heatmap"


def test_pose_feature_adapter_projection_applies_query_and_render_domains():
    class DummyAdapter:
        def project_query(self, feat, rgb=None):
            return feat + 1.0

        def project_render(self, feat, rgb=None):
            return feat + 2.0

    query = torch.zeros(1, 2, 3, 4)
    render = torch.zeros(1, 3, 2, 3, 4)

    project = getattr(eval_cpr_buckets, "project_query_render_with_pose_feature_adapter", None)
    assert project is not None, "eval_cpr_buckets should expose pose-feature adapter projection"

    query_out, render_out, query_unc, render_unc = project(DummyAdapter(), query, render)

    assert query_unc is None
    assert render_unc is None
    assert torch.allclose(query_out, torch.ones_like(query))
    assert torch.allclose(render_out, torch.full_like(render, 2.0))


def test_pose_energy_candidate_selection_masks_invalid_candidates():
    outputs = {
        "energy_logits": torch.tensor([[0.1, 3.0, 0.4]]),
        "confidence_logits": torch.zeros(1, 3),
        "residual_delta": torch.zeros(1, 3, 6),
    }
    valid = torch.tensor([[True, False, True]])

    select = getattr(eval_cpr_buckets, "pose_energy_candidate_selection", None)
    assert select is not None, "eval_cpr_buckets should expose pose_energy_candidate_selection"

    result = select(outputs, valid)

    assert int(result["idx"][0]) == 2
    assert result["scores"][0, 1] < -1.0e5
    assert torch.isfinite(result["entropy"]).all()
    assert result["margin"][0] > 0.0


def test_parse_args_accepts_cube_lattice_direction_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cpr_buckets.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--lattice-direction-mode",
            "cube",
        ],
    )

    args = eval_cpr_buckets.parse_args()

    assert args.lattice_direction_mode == "cube"


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


def test_select_fine_pool_indices_rank_uniform_has_no_duplicates_when_enough_valid():
    logits = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    selected = eval_cpr_buckets.select_fine_pool_indices(
        logits,
        valid,
        fine_topk=4,
        mode="rank_uniform",
        rank_topm=1,
    )

    assert selected.shape == (1, 4)
    assert selected[0].unique().numel() == 4


def test_select_fine_pool_indices_rank_pose_hard_keeps_pose_direction_examples():
    logits = torch.tensor([[0.90, 0.10, 0.20, 0.95, 0.05]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    init_pose = torch.eye(4).view(1, 4, 4)
    pose_gt = torch.eye(4).view(1, 4, 4)
    pose_gt[0, 0, 3] = -1.0
    candidate_pose = torch.eye(4).view(1, 1, 4, 4).repeat(1, 5, 1, 1)
    candidate_pose[0, 0, 0, 3] = 0.0
    candidate_pose[0, 1, 0, 3] = -1.0
    candidate_pose[0, 2, 0, 3] = 1.0
    candidate_pose[0, 3, 0, 3] = 2.0
    candidate_pose[0, 4, 1, 3] = -0.5

    selected = eval_cpr_buckets.select_fine_pool_indices(
        logits,
        valid,
        fine_topk=4,
        mode="rank_pose_hard",
        rank_topm=1,
        candidate_pose=candidate_pose,
        init_pose=init_pose,
        pose_gt=pose_gt,
    )

    selected_set = set(selected[0].tolist())
    assert selected.shape == (1, 4)
    assert selected[0].unique().numel() == 4
    assert selected[0, 0].item() == 3
    assert 0 in selected_set
    assert 1 in selected_set
    assert 2 in selected_set
