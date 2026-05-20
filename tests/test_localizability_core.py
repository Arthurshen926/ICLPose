import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.candidate_bank import (
    CandidateBank,
    CandidateBankMetadata,
    candidate_bank_from_npz,
)
from feature_extract.localizability.adapter_bundle import collect_pose_adapter_trainable_parameters
from feature_extract.localizability.losses import (
    basin_bce_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
)
from feature_extract.localizability.mapability import track_feature_variance
from feature_extract.localizability.metrics import ranking_metrics, ranking_row_diagnostics
from feature_extract.localizability.scorer import PoseHypothesisScorer
from feature_extract.localizability.selector import LocalizationFeatureSelector
from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher, PoseFeatureDomainAdapter


def test_localization_feature_selector_outputs_compact_feature_and_gates():
    torch.manual_seed(7)
    selector = LocalizationFeatureSelector(
        in_channels=8,
        out_channels=4,
        group_size=2,
        spatial_utility=True,
        uncertainty=True,
    )
    feature = torch.randn(2, 8, 5, 7)

    outputs = selector(feature)

    assert outputs["z"].shape == (2, 4, 5, 7)
    assert outputs["utility"].shape == (2, 1, 5, 7)
    assert outputs["uncertainty"].shape == (2, 1, 5, 7)
    assert outputs["channel_gate"].shape == (2, 4)
    assert torch.allclose(outputs["z"].norm(dim=1).mean(), torch.tensor(1.0), atol=1.0e-5)
    assert outputs["utility"].min() >= 0.0
    assert outputs["utility"].max() <= 1.0


def test_localization_feature_selector_identity_init_preserves_feature_direction():
    selector = LocalizationFeatureSelector(
        in_channels=8,
        out_channels=8,
        group_size=2,
        spatial_utility=True,
        uncertainty=False,
        identity_init=True,
    )
    feature = torch.randn(2, 8, 5, 7)

    outputs = selector(feature)

    expected = torch.nn.functional.normalize(feature.float(), dim=1, eps=1.0e-6)
    cosine = (outputs["z"] * expected).sum(dim=1).mean()
    assert cosine.item() > 0.999


def test_pose_hypothesis_scorer_prefers_aligned_candidate_with_utility_weight():
    query = torch.zeros(1, 4, 9, 9)
    query[:, :, 4, 4] = 1.0
    render = query[:, None].repeat(1, 3, 1, 1, 1)
    render[:, 1] = torch.roll(render[:, 1], shifts=2, dims=-1)
    render[:, 2] = 0.0
    utility = torch.zeros(1, 1, 9, 9)
    utility[:, :, 4, 4] = 1.0

    scorer = PoseHypothesisScorer(mode="local_corr", radius=1, temperature=0.1)
    scores, aux = scorer(query, render, query_utility=utility)

    assert scores.shape == (1, 3)
    assert int(scores.argmax(dim=1)[0]) == 0
    assert aux["score_maps"].shape[:2] == (1, 3)
    assert aux["valid_mask"].all()


def test_pair_matcher_local_uses_center_offset_evidence_not_shift_invariant_max():
    query = torch.zeros(1, 4, 7, 7)
    query[:, :, 3, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    render = torch.zeros(1, 2, 4, 7, 7)
    render[:, 0, :, 3, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    render[:, 1, :, 3, 4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    utility = torch.zeros(1, 1, 7, 7)
    utility[:, :, 3, 3] = 1.0

    local = PoseHypothesisScorer(mode="local_corr", radius=1, temperature=0.1)
    local_scores, _ = local(query, render, query_utility=utility)
    assert torch.allclose(local_scores[:, 0], local_scores[:, 1], atol=1.0e-5)

    matcher = PairConditionedLocalMatcher(
        channels=4,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=True,
        base_dot_weight=1.0,
    )
    pair = PoseHypothesisScorer(
        mode="pair_matcher_local",
        radius=1,
        temperature=0.1,
        pair_matcher=matcher,
        pair_matcher_stride=1,
        pair_matcher_score_channel=1,
    )
    pair_scores, aux = pair(query, render, query_utility=utility)

    assert int(pair_scores.argmax(dim=1)[0]) == 0
    assert pair_scores[0, 0] > pair_scores[0, 1]
    assert aux["score_maps"].shape[:3] == (1, 2, 3)


def test_collect_pose_adapter_trainable_parameters_keeps_backbone_boundaries_explicit():
    adapter = PoseFeatureDomainAdapter(channels=4, hidden_dim=8, rgb_context_enabled=True, rgb_context_channels=2)
    matcher = PairConditionedLocalMatcher(channels=4, hidden_dim=8, offset_radius=1)

    params = collect_pose_adapter_trainable_parameters(
        adapter,
        matcher,
        train_query_adapter=False,
        train_render_adapter=True,
        train_rgb_context=False,
        train_pair_matcher=True,
    )

    assert params
    assert not any(param.requires_grad for param in adapter.query_adapter.parameters())
    assert all(param.requires_grad for param in adapter.render_adapter.parameters())
    assert not any(param.requires_grad for param in adapter.query_rgb_stem.parameters())
    assert all(param.requires_grad for param in matcher.parameters())


def test_pose_rank_losses_focus_on_geometry_and_online_hard_negative():
    scores = torch.tensor([[0.1, 0.7, 0.2]], requires_grad=True)
    costs = torch.tensor([[0.05, 0.40, 0.12]])
    rank_loss, rank_metrics = pose_distance_soft_rank_loss(scores, costs, temperature_m=0.05)
    hard_loss, hard_metrics = online_score_hard_negative_loss(
        scores,
        costs,
        cost_gap_m=0.10,
        margin=0.05,
    )
    basin_loss = basin_bce_loss(scores, torch.tensor([[True, False, True]]))
    total = rank_loss + hard_loss + basin_loss
    total.backward()

    assert rank_loss.item() > 0.0
    assert hard_loss.item() > 0.0
    assert basin_loss.item() > 0.0
    assert rank_metrics["rank_target_entropy"].item() < 1.1
    assert hard_metrics["online_hard_active"].item() == 1.0
    assert scores.grad is not None


def test_ranking_metrics_report_oracle_gap_spearman_and_basin_recall():
    scores = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.1, 0.2]])
    costs = torch.tensor([[0.30, 0.10, 0.50], [0.40, 0.20, 0.10]])
    basin = costs <= 0.20

    metrics = ranking_metrics(scores, costs, basin_label=basin, topk=(1, 2))

    assert metrics["top1_acc"].item() == 0.5
    assert metrics["pred_cost_m"].item() == 0.25
    assert abs(metrics["oracle_cost_m"].item() - 0.10) < 1.0e-6
    assert abs(metrics["oracle_gap_m"].item() - 0.15) < 1.0e-6
    assert metrics["basin_recall@2"].item() == 1.0
    assert -1.0 <= metrics["spearman"].item() <= 1.0


def test_ranking_row_diagnostics_keeps_selected_oracle_and_basin_fields():
    scores = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.1, 0.2]])
    costs = torch.tensor([[0.30, 0.10, 0.50], [0.40, 0.20, 0.10]])
    basin = costs <= 0.20

    rows = ranking_row_diagnostics(scores, costs, basin_label=basin, sample_names=["a", "b"])

    assert rows[0]["sample_name"] == "a"
    assert rows[0]["selected_idx"] == 1
    assert rows[0]["oracle_idx"] == 1
    assert rows[0]["selected_in_basin"] is True
    assert rows[1]["selected_idx"] == 0
    assert rows[1]["oracle_idx"] == 2
    assert abs(rows[1]["oracle_gap_m"] - 0.3) < 1.0e-6


def test_candidate_bank_from_npz_normalizes_required_fields(tmp_path):
    path = tmp_path / "bank.npz"
    pose_gt = torch.eye(4).view(1, 4, 4).numpy()
    candidates = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1).numpy()
    import numpy as np

    np.savez(
        path,
        sample_names=np.array(["seq/frame.png"]),
        pose_gt=pose_gt,
        candidates=candidates,
        pose_cost_m=np.array([[0.1, 0.5]], dtype=np.float32),
        trans_err_m=np.array([[0.1, 0.5]], dtype=np.float32),
        rot_err_deg=np.array([[1.0, 10.0]], dtype=np.float32),
    )

    bank = candidate_bank_from_npz(path)

    assert isinstance(bank, CandidateBank)
    assert isinstance(bank.metadata, CandidateBankMetadata)
    assert bank.candidate_pose.shape == (1, 2, 4, 4)
    assert bank.basin_label(0.25, 5.0).tolist() == [[True, False]]


def test_track_feature_variance_ignores_invalid_observations():
    features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        ]
    )
    track_ids = torch.tensor([0, 0, 1])
    valid = torch.tensor([True, True, False])

    variance, stats = track_feature_variance(features, track_ids, valid_mask=valid)

    assert variance.item() < 1.0e-6
    assert stats["num_tracks"].item() == 1
