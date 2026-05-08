import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

import feature_extract.train_impl as train_impl
from feature_field.dcff.losses import infonce_contrastive_loss
from feature_field.utils.feature_track_vis import error_to_heatmap_image
from data.radio_loc_retrieval_dataset import save_retrieval_init_entries
from feature_extract import build_records_from_feature_ids as facade_build_records_from_feature_ids
from feature_extract.scripts.export_adaptive_teacher_features import make_feature_filename, resolve_camera_fid
from feature_extract.students.radio_query_student import (
    AnchorPoseInitHead,
    CandidateScoreFusionHead,
    CandidateScoreMapFusionHead,
    DepthAwareLocalFlowHead,
    FeatureBankPoseInitHead,
    LocalCorrDomainAdapter,
    RadioQueryStudent,
)
from feature_extract.train_impl import (
    TeacherFeatureStore,
    TeacherCorrespondenceStore,
    RetrievalTeacherStore,
    build_all_records,
    build_model_param_groups,
    build_records_from_feature_ids,
    compute_map_supervision,
    compute_pose_init_losses,
    compute_w2c_flow,
    depth_observability_weight,
    flow_warp_contrastive_loss,
    flow_warp_feature_alignment_loss,
    _resize_query_flow_valid,
    local_correlation_distribution_loss,
    local_correlation_feature_preprocess,
    local_correlation_joint_losses,
    local_correlation_peak_margin_loss,
    local_correlation_soft_flow_loss,
    local_correlation_subpixel_loss,
    load_pose_candidate_cache_index,
    local_correlation_wls_pose_loss,
    load_model_warmstart,
    maybe_attach_pose_candidate_renders,
    normalize_scene_coord_map,
    candidate_quality_features_from_batch,
    candidate_score_fusion_listwise_loss,
    perturb_w2c_camera_center,
    pose_feature_bank_from_dataset,
    pose_update_gain_loss,
    resolve_perturb_rank_margin,
    resolve_query_feature_dims,
    resolve_safe_num_workers,
    sample_query_feature_by_flow,
    sparse_teacher_correspondence_loss,
    sparse_teacher_local_patch_loss,
    scene_coord_flow_warp_loss,
    scene_coord_regression_loss,
    shifted_local_correlation,
    translation_observability_weight,
    descriptor_pose_retrieval_metrics,
    local_render_score_feature_candidates,
    candidate_local_render_score_nce_loss,
    render_score_candidate_listwise_loss,
    render_score_feature_candidates,
    select_pose_candidate_indices,
    topk_descriptor_candidate_indices,
)
from feature_extract.export_impl import apply_export_feature_dims, build_model as build_export_model
from pose_refine.models.concat_pose_net import local_correlation as reference_local_correlation


def test_radio_query_student_supports_asymmetric_fine_and_coarse_outputs():
    model = RadioQueryStudent(
        feature_dim=64,
        fine_feature_dim=96,
        coarse_feature_dim=32,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
    )

    out = model(torch.randn(2, 3, 64, 80))

    assert out["fine"].shape == (2, 96, 8, 10)
    assert out["coarse"].shape == (2, 32, 4, 5)


def test_local_corr_domain_adapter_projects_query_and_render_with_separate_heads():
    adapter = LocalCorrDomainAdapter(
        feature_dim=4,
        hidden_dim=8,
        output_dim=6,
        zero_init=True,
        l2_normalize=True,
    )
    feat = torch.randn(2, 4, 5, 6)

    query = adapter.project_query(feat)
    render = adapter.project_render(feat)

    assert query.shape == (2, 6, 5, 6)
    assert render.shape == (2, 6, 5, 6)
    assert adapter.query_projector is not adapter.render_projector
    assert torch.allclose(query.norm(dim=1), torch.ones(2, 5, 6), atol=1e-5)
    assert torch.allclose(render.norm(dim=1), torch.ones(2, 5, 6), atol=1e-5)


def test_export_model_builds_local_corr_domain_adapter_checkpoint():
    cfg = {
        "dataset": {
            "feature_hw": [4, 5],
            "coarse_feature_hw": [2, 3],
            "input_hw": [32, 40],
        },
        "model": {
            "feature_dim": 4,
            "fine_feature_dim": 4,
            "coarse_feature_dim": 4,
            "base_channels": 8,
            "stage_dims": [8, 8, 8, 8],
            "local_corr_projector_enabled": True,
            "local_corr_projector_domain_adapter": True,
            "local_corr_projector_hidden_dim": 8,
            "local_corr_projector_output_dim": 4,
        },
    }
    trained = RadioQueryStudent(
        feature_dim=4,
        fine_feature_dim=4,
        coarse_feature_dim=4,
        base_channels=8,
        stage_dims=(8, 8, 8, 8),
        output_hw=(4, 5),
        coarse_output_hw=(2, 3),
        input_hw=(32, 40),
        local_corr_projector_enabled=True,
        local_corr_projector_domain_adapter=True,
        local_corr_projector_hidden_dim=8,
        local_corr_projector_output_dim=4,
    )

    loaded = build_export_model(
        cfg,
        {"model_state_dict": trained.state_dict()},
        torch.device("cpu"),
    )

    assert hasattr(loaded.local_corr_projector, "project_query")


def test_teacher_correspondence_store_loads_and_scales_sparse_points():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.savez(
            root / "seq_a.npz",
            query_xy=np.asarray([[159.0, 79.0], [0.0, 0.0]], dtype=np.float32),
            map_xy=np.asarray([[159.0, 79.0], [0.0, 0.0]], dtype=np.float32),
            confidence=np.asarray([0.2, 0.9], dtype=np.float32),
            query_hw=np.asarray([80, 160], dtype=np.int32),
            map_hw=np.asarray([80, 160], dtype=np.int32),
            coordinate_space=np.asarray("image"),
        )
        store = TeacherCorrespondenceStore(root, feature_hw=(8, 16), max_points=4)

        item = store.load_record({"sample_name": "seq/a.png"})

    assert item["teacher_corr_valid"].sum().item() == 2
    assert torch.allclose(item["teacher_corr_conf"][:2], torch.tensor([0.9, 0.2]))
    assert torch.allclose(item["teacher_corr_query_xy"][0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(item["teacher_corr_query_xy"][1], torch.tensor([15.0, 7.0]))


def test_sparse_teacher_correspondence_loss_prefers_matching_pairs():
    feat = torch.zeros(1, 3, 2, 3)
    feat[0, 0, 0, 0] = 1.0
    feat[0, 1, 1, 2] = 1.0
    query_xy = torch.tensor([[[0.0, 0.0], [2.0, 1.0]]])
    map_xy_good = torch.tensor([[[0.0, 0.0], [2.0, 1.0]]])
    map_xy_bad = torch.tensor([[[2.0, 1.0], [0.0, 0.0]]])
    conf = torch.ones(1, 2)
    valid = torch.ones(1, 2)
    source_hw = torch.tensor([[2.0, 3.0]])

    good, good_metrics = sparse_teacher_correspondence_loss(
        feat,
        feat,
        query_xy,
        map_xy_good,
        conf,
        valid,
        xy_source_hw=source_hw,
        temperature=0.05,
        min_points=2,
    )
    bad, bad_metrics = sparse_teacher_correspondence_loss(
        feat,
        feat,
        query_xy,
        map_xy_bad,
        conf,
        valid,
        xy_source_hw=source_hw,
        temperature=0.05,
        min_points=2,
    )

    assert good.item() < bad.item()
    assert good_metrics["map_teacher_corr_acc"].item() == 1.0
    assert good_metrics["map_teacher_corr_skipped_no_points"].item() == 0.0
    assert bad_metrics["map_teacher_corr_acc"].item() == 0.0


def test_sparse_teacher_correspondence_loss_can_ignore_nearby_false_negatives():
    query_feat = torch.zeros(1, 3, 1, 3)
    map_feat = torch.zeros(1, 3, 1, 3)
    query_feat[0, 0, 0, 0] = 1.0
    query_feat[0, 0, 0, 1] = 1.0
    query_feat[0, 1, 0, 2] = 1.0
    map_feat.copy_(query_feat)
    xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    conf = torch.ones(1, 3)
    valid = torch.ones(1, 3)
    source_hw = torch.tensor([[1.0, 3.0]])

    plain, plain_metrics = sparse_teacher_correspondence_loss(
        query_feat,
        map_feat,
        xy,
        xy,
        conf,
        valid,
        xy_source_hw=source_hw,
        temperature=0.1,
        min_points=3,
    )
    tolerant, tolerant_metrics = sparse_teacher_correspondence_loss(
        query_feat,
        map_feat,
        xy,
        xy,
        conf,
        valid,
        xy_source_hw=source_hw,
        temperature=0.1,
        min_points=3,
        negative_exclusion_px=1.1,
        positive_weight=0.5,
        margin_weight=0.5,
        margin=0.1,
    )

    assert tolerant.item() < plain.item()
    assert tolerant_metrics["map_teacher_corr_gap"].item() > plain_metrics["map_teacher_corr_gap"].item()


def test_sparse_teacher_local_patch_loss_supervises_correct_local_peak():
    query_feat = torch.zeros(1, 3, 5, 5)
    map_feat = torch.zeros(1, 3, 5, 5)
    query_feat[0, 0, 2, 2] = 1.0
    map_feat[0, 0, 2, 2] = 1.0
    map_feat[0, 1, 2, 3] = 1.0
    query_xy = torch.tensor([[[2.0, 2.0]]])
    map_xy_good = torch.tensor([[[2.0, 2.0]]])
    map_xy_bad = torch.tensor([[[3.0, 2.0]]])
    conf = torch.ones(1, 1)
    valid = torch.ones(1, 1)
    source_hw = torch.tensor([[5.0, 5.0]])

    good, good_metrics = sparse_teacher_local_patch_loss(
        query_feat,
        map_feat,
        query_xy,
        map_xy_good,
        conf,
        valid,
        xy_source_hw=source_hw,
        radius=1,
        temperature=0.05,
        min_points=1,
        positive_weight=0.5,
        margin_weight=0.5,
        margin=0.1,
    )
    bad, bad_metrics = sparse_teacher_local_patch_loss(
        query_feat,
        map_feat,
        query_xy,
        map_xy_bad,
        conf,
        valid,
        xy_source_hw=source_hw,
        radius=1,
        temperature=0.05,
        min_points=1,
        positive_weight=0.5,
        margin_weight=0.5,
        margin=0.1,
    )

    assert good.item() < bad.item()
    assert good_metrics["map_teacher_patch_acc"].item() == 1.0
    assert good_metrics["map_teacher_patch_gap"].item() > 0.0
    assert good_metrics["map_teacher_patch_soft_epe"].item() < bad_metrics["map_teacher_patch_soft_epe"].item()
    assert bad_metrics["map_teacher_patch_acc"].item() == 0.0


def test_compute_map_supervision_consumes_teacher_correspondences():
    feat = torch.zeros(1, 3, 2, 3)
    feat[0, 0, 0, 0] = 1.0
    feat[0, 1, 1, 2] = 1.0
    coarse = torch.randn(1, 2, 1, 2)
    xy = torch.tensor([[[0.0, 0.0], [2.0, 1.0]]])
    batch = {
        "rgb": torch.zeros(1, 3, 8, 12),
        "rendered_map_fine": feat.clone(),
        "rendered_map_coarse": coarse.clone(),
        "rendered_map_mask": torch.ones(1, 1, 2, 3),
        "teacher_fine": feat.clone(),
        "teacher_coarse": coarse.clone(),
        "teacher_corr_query_xy": xy,
        "teacher_corr_map_xy": xy,
        "teacher_corr_conf": torch.ones(1, 2),
        "teacher_corr_valid": torch.ones(1, 2),
        "teacher_corr_hw": torch.tensor([[2.0, 3.0]]),
    }
    outputs = {"fine": feat.clone(), "coarse": coarse.clone()}
    cfg = {
        "loss": {},
        "map_supervision": {
            "enabled": True,
            "teacher_corr_weight": 1.0,
            "teacher_corr_local_patch_weight": 1.0,
            "teacher_corr_local_patch_radius": 1,
            "teacher_corr_min_points": 2,
            "teacher_corr_temperature": 1.0,
        },
    }

    total, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"))

    assert total.item() > 0.0
    assert metrics["map_teacher_corr_missing"].item() == 0.0
    assert metrics["map_teacher_corr_skipped_no_points"].item() == 0.0
    assert metrics["map_teacher_corr_acc"].item() == 1.0
    assert metrics["map_teacher_patch_skipped_no_points"].item() == 0.0
    assert metrics["map_teacher_patch_acc"].item() == 1.0
    assert metrics["map_teacher_patch_self_acc"].item() == 1.0
    assert metrics["map_teacher_patch_radio_acc"].item() == 1.0


def test_local_correlation_wls_ignores_flow_outside_search_window():
    height, width = 4, 5
    rendered = F.normalize(torch.randn(1, 4, height, width), dim=1)
    query = F.normalize(torch.randn(1, 4, height, width), dim=1)
    flow_gt = torch.full((1, 2, height, width), 5.0)
    valid = torch.ones(1, 1, height, width)
    depth = torch.ones(1, height, width) * 3.0
    pose = torch.eye(4).unsqueeze(0)

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow_gt,
        valid,
        radius=1,
        temperature=0.05,
        compute_wls_pose=True,
        depth=depth,
        pose_ref=pose,
        pose_gt=pose,
        intrinsics={
            "fx": 40.0,
            "fy": 40.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
        },
    )

    assert result["metrics"]["map_corr_wls_conf_cov"].item() == 0.0
    assert result["metrics"]["map_corr_wls_delta_trans_mm"].item() == 0.0


def test_local_correlation_wls_min_conf_cov_gates_sparse_confidence():
    class SparseBadFlowHead(torch.nn.Module):
        def forward(self, corr, depth=None, valid_mask=None):
            batch, _channels, height, width = corr.shape
            flow = torch.ones(batch, 2, height, width, device=corr.device, dtype=corr.dtype)
            confidence = torch.zeros(batch, 1, height, width, device=corr.device, dtype=corr.dtype)
            confidence[:, :, 0, 0] = 1.0
            return {"flow": flow, "confidence": confidence}

    height, width = 4, 5
    rendered = F.normalize(torch.randn(1, 4, height, width), dim=1)
    query = F.normalize(torch.randn(1, 4, height, width), dim=1)
    flow_gt = torch.zeros(1, 2, height, width)
    valid = torch.ones(1, 1, height, width)
    depth = torch.ones(1, height, width) * 3.0
    pose = torch.eye(4).unsqueeze(0)

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow_gt,
        valid,
        radius=1,
        temperature=0.05,
        compute_wls_pose=True,
        flow_head=SparseBadFlowHead(),
        wls_min_conf_cov=0.5,
        depth=depth,
        pose_ref=pose,
        pose_gt=pose,
        intrinsics={
            "fx": 40.0,
            "fy": 40.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
        },
    )

    assert result["metrics"]["map_corr_wls_conf_cov"].item() == 0.0
    assert result["metrics"]["map_corr_wls_delta_trans_mm"].item() == 0.0


def test_descriptor_pose_retrieval_metrics_reports_topk_pose_errors():
    query_desc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bank_desc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    query_pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    bank_pose = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    query_pose[1, 0, 3] = -1.0
    bank_pose[1, 0, 3] = -1.0
    bank_pose[2, 0, 3] = 4.0

    metrics = descriptor_pose_retrieval_metrics(
        query_desc,
        query_pose,
        bank_desc,
        bank_pose,
        topk=(1, 2),
    )

    assert metrics["retrieval_top1_trans_mm"].item() == 0.0
    assert metrics["retrieval_top2_best_trans_mm"].item() == 0.0
    assert metrics["retrieval_top1_recall_5deg_1000mm"].item() == 100.0


def test_select_pose_candidate_indices_supports_score_and_uniform_limits():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.0])

    top_idx = select_pose_candidate_indices(5, limit=2, scores=scores, strategy="score")
    uniform_idx = select_pose_candidate_indices(5, limit=3, strategy="uniform")

    assert top_idx.tolist() == [1, 3]
    assert uniform_idx.tolist() == [0, 2, 4]


def test_topk_descriptor_candidate_indices_uses_cosine_similarity():
    query_desc = torch.tensor([[1.0, 0.0]])
    bank_desc = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]])

    idx, scores = topk_descriptor_candidate_indices(query_desc, bank_desc, k=2)

    assert idx.tolist() == [[1, 2]]
    assert scores[0, 0] > scores[0, 1]


def test_feature_bank_pose_init_head_can_use_retrieval_descriptor_space():
    head = FeatureBankPoseInitHead(
        token_dim=3,
        anchor_centers=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        anchor_rotmats=torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
        anchor_descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        fine_dim=4,
        coarse_dim=2,
        hypotheses=1,
        projector="identity",
        feature_source="retrieval",
    )

    out = head(torch.tensor([[0.0, 1.0, 0.0]]), fine=None, coarse=None)

    assert out["anchor_indices"].item() == 1


def test_pose_feature_bank_from_dataset_can_use_retrieval_teacher_descriptors():
    class RetrievalStore:
        def load_record(self, record):
            return torch.tensor([float(record["teacher_idx"]), 1.0])

    class Dataset:
        records = [
            {"sample_name": "seq/a.png", "teacher_idx": 2},
            {"sample_name": "seq/b.png", "teacher_idx": 3},
        ]
        retrieval_teacher_store = RetrievalStore()
        name_to_pose = {}
        basename_to_pose = {}

    pose_a = torch.eye(4)
    pose_b = torch.eye(4)
    pose_b[0, 3] = -1.0
    Dataset.name_to_pose = {"seq/a.png": pose_a, "seq/b.png": pose_b}

    centers, rotmats, desc = pose_feature_bank_from_dataset(
        Dataset(),
        teacher_store=None,
        num_anchors=10,
        sampling="train_all",
        feature_source="retrieval",
    )

    assert centers.shape == (2, 3)
    assert rotmats.shape == (2, 3, 3)
    assert torch.equal(desc, torch.tensor([[2.0, 1.0], [3.0, 1.0]]))


def test_render_score_feature_candidates_selects_best_masked_match():
    query = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]
    )
    candidates = torch.stack(
        [
            torch.tensor([[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]]),
            query.squeeze(0),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]),
        ],
        dim=0,
    )
    mask = torch.ones(3, 1, 2, 2)

    result = render_score_feature_candidates(query, candidates, mask=mask)

    assert result["best_idx"].item() == 1
    assert torch.allclose(result["scores"], torch.tensor([0.0, 1.0, 0.5]))
    assert result["score_margin"].item() == 0.5


def test_render_score_feature_candidates_can_spatial_center_before_cosine():
    query = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 3.0], [2.0, 1.0]],
        ]
    )
    good = query + torch.tensor([10.0, -5.0]).view(2, 1, 1)
    bad = -query + torch.tensor([10.0, -5.0]).view(2, 1, 1)
    candidates = torch.stack([bad, good], dim=0)

    result = render_score_feature_candidates(query, candidates, preprocess="spatial_center")

    assert result["best_idx"].item() == 1
    assert result["scores"][1] > 0.99
    assert result["scores"][0] < -0.99


def test_local_render_score_feature_candidates_tolerates_small_pixel_shift():
    query = torch.zeros(2, 3, 3)
    query[0, 1, 1] = 1.0
    good = torch.zeros(2, 3, 3)
    good[0, 1, 2] = 1.0
    bad = torch.zeros(2, 3, 3)
    bad[1, 1, 2] = 1.0
    candidates = torch.stack([bad, good], dim=0)
    mask = torch.zeros(1, 3, 3)
    mask[:, 1, 1] = 1.0

    result = local_render_score_feature_candidates(query, candidates, mask=mask, radius=1)

    assert result["best_idx"].item() == 1
    assert result["scores"][1] > 0.99
    assert result["scores"][0] < 0.01


def test_local_render_score_feature_candidates_can_return_correlation_volume_score_map():
    radius = 2
    height, width = 4, 5
    query = F.normalize(torch.randn(3, height, width), dim=0)
    candidates = F.normalize(torch.randn(2, 3, height, width), dim=1)

    result = local_render_score_feature_candidates(
        query,
        candidates,
        radius=radius,
        score_map_mode="volume",
    )

    assert result["score_map"].shape == (2, (2 * radius + 1) ** 2, height, width)
    assert torch.allclose(result["scores"], result["score_map"].max(dim=1).values.mean(dim=(1, 2)))


def test_local_render_score_feature_candidates_can_return_volume_plus_peak_offset():
    radius = 1
    height, width = 3, 4
    query = F.normalize(torch.randn(2, height, width), dim=0)
    candidates = F.normalize(torch.randn(2, 2, height, width), dim=1)

    result = local_render_score_feature_candidates(
        query,
        candidates,
        radius=radius,
        score_map_mode="volume_plus_peak_offset",
    )

    query_n = F.normalize(query.unsqueeze(0).expand(2, -1, -1, -1), dim=1)
    candidate_n = F.normalize(candidates, dim=1)
    corr = shifted_local_correlation(query_n, candidate_n, radius=radius)
    peak_offset = train_impl._local_correlation_peak_score_map(corr, radius=radius)
    expected_channels = (2 * radius + 1) ** 2 + 3
    assert result["score_map"].shape == (2, expected_channels, height, width)
    assert torch.allclose(result["score_map"][:, -3:], peak_offset)


def test_retrieval_teacher_store_supports_summary_descriptors():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_dir = root / "summary"
        summary_dir.mkdir()
        torch.save(torch.arange(3).float(), summary_dir / "seq1_frame000_summary_3.pt")

        store = RetrievalTeacherStore(root, subdir="summary")

        assert store.feature_dim == 3
        assert torch.equal(store.load("seq1/frame000.png"), torch.arange(3).float())


def test_retrieval_teacher_store_can_fallback_to_feature_id_records():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_dir = root / "summary"
        summary_dir.mkdir()
        torch.save(torch.arange(4).float(), summary_dir / "rgb_7_summary_4.pt")

        store = RetrievalTeacherStore(root, subdir="summary")

        loaded = store.load_record({"sample_name": "seq2/frame00154.png", "teacher_idx": 7})
        assert torch.equal(loaded, torch.arange(4).float())


def test_pose_init_metrics_report_score_selected_and_oracle_topk_recall():
    outputs = {
        "fine": torch.zeros(1, 1, 1, 1),
        "pose_init": {
            "center": torch.tensor([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "rotmat": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "scores": torch.tensor([[4.0, 1.0]]),
        },
    }
    batch = {"pose_gt": torch.eye(4).unsqueeze(0)}

    _loss, metrics = compute_pose_init_losses(
        outputs,
        batch,
        {
            "pose_init": {
                "enabled": True,
                "trans_weight": 1.0,
                "rot_weight": 0.0,
                "score_weight": 0.0,
            }
        },
    )

    assert metrics["pose_init_best_trans_mm"].item() == 0.0
    assert metrics["pose_init_pred_trans_mm"].item() == 2000.0
    assert metrics["pose_init_best_joint_5deg_1000mm"].item() == 100.0
    assert metrics["pose_init_pred_joint_5deg_1000mm"].item() == 0.0


def test_anchor_pose_init_head_exposes_all_anchor_poses_for_teacher_forced_loss():
    head = AnchorPoseInitHead(
        token_dim=4,
        hidden_dim=8,
        hypotheses=1,
        anchor_centers=torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        anchor_rotmats=torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
        residual_scale=1.0,
    )

    out = head(torch.zeros(1, 4))

    assert out["center"].shape == (1, 1, 3)
    assert out["all_center"].shape == (1, 2, 3)
    assert out["all_rotmat"].shape == (1, 2, 3, 3)


def test_pose_init_all_anchor_pose_loss_can_supervise_target_outside_topk():
    outputs = {
        "fine": torch.zeros(1, 1, 1, 1),
        "pose_init": {
            "center": torch.tensor([[[5.0, 0.0, 0.0]]]),
            "rotmat": torch.eye(3).view(1, 1, 3, 3),
            "scores": torch.tensor([[4.0]]),
            "all_center": torch.tensor([[[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "all_rotmat": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "anchor_logits": torch.tensor([[4.0, 1.0]]),
            "anchor_indices": torch.tensor([[0]]),
            "anchor_centers": torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "anchor_rotmats": torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
        },
    }
    batch = {"pose_gt": torch.eye(4).unsqueeze(0)}

    _loss, metrics = compute_pose_init_losses(
        outputs,
        batch,
        {
            "pose_init": {
                "enabled": True,
                "trans_weight": 1.0,
                "rot_weight": 0.0,
                "score_weight": 0.0,
                "anchor_weight": 0.0,
                "use_all_anchor_pose_loss": True,
            }
        },
    )

    assert metrics["pose_init_best_trans_mm"].item() == 0.0
    assert metrics["pose_init_pred_trans_mm"].item() == 5000.0


def test_resolve_safe_num_workers_disables_workers_for_full_resolution_inputs():
    cfg = {"num_workers": 2}
    dataset_cfg = {"input_hw": [1088, 1920]}

    assert resolve_safe_num_workers(cfg, dataset_cfg) == 0


def test_resolve_safe_num_workers_preserves_small_inputs_and_explicit_override():
    assert resolve_safe_num_workers({"num_workers": 2}, {"input_hw": [256, 455]}) == 2
    assert (
        resolve_safe_num_workers(
            {"num_workers": 2, "allow_highres_num_workers": True},
            {"input_hw": [1088, 1920]},
        )
        == 2
    )


def test_radio_query_student_fine_low_level_skip_receives_gradients():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        fine_low_level_skip=True,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["fine"].square().mean()
    loss.backward()

    assert out["fine"].shape == (2, 16, 8, 10)
    assert model.fine_low_fuse is not None
    assert model.fine_low_scale.grad is not None
    assert any(p.grad is not None for p in model.fine_low_fuse.parameters())


def test_radio_query_student_fine_highres_skip_receives_gradients():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        fine_highres_skip=True,
        fine_highres_source="stage2",
        fine_highres_init=0.25,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["fine"].square().mean()
    loss.backward()

    assert out["fine"].shape == (2, 16, 16, 20)
    assert model.fine_highres_fuse is not None
    assert model.fine_highres_scale.grad is not None
    assert any(p.grad is not None for p in model.fine_highres_fuse.parameters())


def test_radio_query_student_fine_highres_zero_init_starts_as_residual_noop():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        fine_highres_skip=True,
        fine_highres_source="stage2",
        fine_highres_init=1.0,
        fine_highres_zero_init=True,
    )

    assert torch.count_nonzero(model.fine_highres_fuse[-1].weight).item() == 0
    assert torch.count_nonzero(model.fine_highres_fuse[-1].bias).item() == 0

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["fine"].square().mean()
    loss.backward()

    assert model.fine_highres_fuse[-1].weight.grad is not None


def test_radio_query_student_global_context_zero_init_starts_as_residual_noop():
    torch.manual_seed(7)
    baseline = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        l2_normalize=False,
    )
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        l2_normalize=False,
        global_context_enabled=True,
        global_context_zero_init=True,
    )
    model.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(2, 3, 64, 80)
    baseline_out = baseline(x)
    out = model(x)
    loss = out["fine"].square().mean() + out["coarse"].square().mean()
    loss.backward()

    assert model.global_context is not None
    assert torch.allclose(out["fine"], baseline_out["fine"], atol=1e-6)
    assert torch.allclose(out["coarse"], baseline_out["coarse"], atol=1e-6)
    assert model.global_context[1].weight.grad is not None
    assert model.global_context[1].weight.grad.abs().sum().item() > 0


def test_radio_query_student_window_attention_zero_init_starts_as_residual_noop():
    torch.manual_seed(11)
    baseline = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        l2_normalize=False,
    )
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        l2_normalize=False,
        window_attention_layers=2,
        window_attention_heads=4,
        window_attention_size=3,
        window_attention_zero_init=True,
    )
    model.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(2, 3, 64, 80)
    baseline_out = baseline(x)
    out = model(x)
    loss = out["fine"].square().mean() + out["coarse"].square().mean()
    loss.backward()

    assert model.window_attention is not None
    assert len(model.window_attention) == 2
    assert torch.allclose(out["fine"], baseline_out["fine"], atol=1e-6)
    assert torch.allclose(out["coarse"], baseline_out["coarse"], atol=1e-6)
    assert model.window_attention[0].attn.out_proj.weight.grad is not None
    assert model.window_attention[0].attn.out_proj.weight.grad.abs().sum().item() > 0


def test_radio_query_student_local_corr_projector_starts_as_noop_and_trains():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        local_corr_projector_enabled=True,
        local_corr_projector_zero_init=True,
    )

    feat = torch.randn(2, 16, 8, 10)
    out = model.local_corr_projector(feat)
    loss = out.square().mean()
    loss.backward()

    assert torch.allclose(out, torch.nn.functional.normalize(feat, dim=1), atol=1e-6)
    assert model.local_corr_projector.refine[-1].weight.grad is not None


def test_radio_query_student_teacher_fine_condition_starts_as_noop_and_receives_gradients():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        teacher_fine_condition=True,
        teacher_fine_init=1.0,
        teacher_fine_zero_init=True,
        teacher_fine_detach=True,
    )

    rgb = torch.randn(2, 3, 64, 80)
    teacher_fine = torch.randn(2, 16, 8, 10)
    base = model(rgb)
    out = model(rgb, teacher_fine=teacher_fine)
    loss = out["fine"].square().mean()
    loss.backward()

    assert torch.allclose(out["fine"], base["fine"], atol=1e-6)
    assert model.teacher_fine_fuse is not None
    assert model.teacher_fine_fuse[-1].weight.grad is not None


def test_build_model_param_groups_can_boost_scene_coord_head_lr():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        scene_coord_head=True,
        scene_coord_use_pixel_grid=True,
        scene_coord_global_context=True,
    )

    groups = build_model_param_groups(
        model,
        base_lr=1e-4,
        weight_decay=1e-5,
        lr_scales={
            "scene_coord_head.": 10.0,
            "scene_context_proj.": 10.0,
        },
    )

    lr_by_param_id = {
        id(param): group["lr"]
        for group in groups
        for param in group["params"]
    }
    scene_lr = {
        lr_by_param_id[id(param)]
        for name, param in model.named_parameters()
        if name.startswith("scene_coord_head.") or name.startswith("scene_context_proj.")
    }
    stem_lr = {
        lr_by_param_id[id(param)]
        for name, param in model.named_parameters()
        if name.startswith("stem.")
    }

    assert scene_lr == {1e-3}
    assert stem_lr == {1e-4}


def test_radio_query_student_depth_aware_local_matcher_starts_as_noop_and_trains():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        local_matcher_enabled=True,
        local_matcher_radius=1,
        local_matcher_hidden_dim=12,
        local_matcher_zero_init=True,
    )
    corr = torch.randn(2, 9, 8, 10, requires_grad=True)
    depth = torch.ones(2, 1, 8, 10)

    refined = model.local_matcher(corr, depth=depth)
    loss = refined.square().mean()
    loss.backward()

    assert torch.allclose(refined, corr, atol=1e-6)
    assert model.local_matcher.refine[-1].weight.grad is not None
    assert model.local_matcher.refine[-1].weight.grad.abs().sum().item() > 0


def test_depth_aware_local_flow_head_outputs_flow_and_confidence_with_gradients():
    head = DepthAwareLocalFlowHead(radius=2, hidden_dim=16, zero_init=True, max_flow=2.0)
    corr = torch.randn(2, 25, 6, 7)
    depth = torch.ones(2, 1, 6, 7)

    pred = head(corr, depth=depth)
    loss = pred["flow"].square().mean() + pred["confidence"].mean()
    loss.backward()

    assert pred["flow"].shape == (2, 2, 6, 7)
    assert pred["confidence"].shape == (2, 1, 6, 7)
    assert pred["confidence"].min().item() >= 0.0
    assert pred["confidence"].max().item() <= 1.0
    assert torch.allclose(pred["flow"], torch.zeros_like(pred["flow"]), atol=1e-6)
    assert head.predict[-1].weight.grad is not None


def test_depth_aware_local_flow_head_exposes_correlation_prior_context():
    head = DepthAwareLocalFlowHead(radius=1, hidden_dim=12, zero_init=True, base_flow_mode="none")
    corr = torch.zeros(1, 9, 4, 5)
    corr[:, 5] = 2.0

    pred = head(corr)

    assert head.predict[0].block[0].in_channels == 17
    assert torch.allclose(pred["flow"], torch.zeros_like(pred["flow"]), atol=1e-6)


def test_depth_aware_local_flow_head_can_start_from_softargmax_flow():
    head = DepthAwareLocalFlowHead(
        radius=2,
        hidden_dim=16,
        zero_init=True,
        max_flow=2.0,
        base_flow_mode="softargmax",
        base_temperature=0.01,
    )
    corr = torch.zeros(1, 25, 3, 4)
    dx, dy = 1, -1
    channel = (dy + 2) * 5 + (dx + 2)
    corr[:, channel] = 10.0

    pred = head(corr)

    assert torch.allclose(pred["flow"][:, 0].mean(), torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(pred["flow"][:, 1].mean(), torch.tensor(-1.0), atol=1e-3)


def test_radio_query_student_can_build_local_flow_head():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        local_flow_head_enabled=True,
        local_flow_head_radius=3,
        local_flow_head_hidden_dim=16,
    )

    assert model.local_flow_head is not None
    assert model.local_flow_head.radius == 3


def test_warmstart_non_strict_skips_shape_mismatched_layers():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(3, 3))
    initial_state = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint_state = {key: value.clone() for key, value in initial_state.items()}
    checkpoint_state["0.weight"] = torch.full_like(checkpoint_state["0.weight"], 3.0)
    checkpoint_state["1.weight"] = torch.ones(4, 3)
    checkpoint_state["unexpected.weight"] = torch.ones(1)

    result = load_model_warmstart(
        model,
        {"model_state_dict": checkpoint_state},
        strict=False,
    )

    assert torch.allclose(model.state_dict()["0.weight"], torch.full_like(initial_state["0.weight"], 3.0))
    assert torch.allclose(model.state_dict()["1.weight"], initial_state["1.weight"])
    assert result["skipped_mismatched"]["1.weight"] == ((4, 3), (3, 3))
    assert "unexpected.weight" in result["skipped_unexpected"]


def test_warmstart_non_strict_can_skip_prefixes():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False), torch.nn.Linear(3, 3, bias=False))
    initial_state = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint_state = {
        "0.weight": torch.full_like(initial_state["0.weight"], 3.0),
        "1.weight": torch.full_like(initial_state["1.weight"], 4.0),
    }

    result = load_model_warmstart(
        model,
        {"model_state_dict": checkpoint_state},
        strict=False,
        skip_prefixes=["0."],
    )

    assert torch.allclose(model.state_dict()["0.weight"], initial_state["0.weight"])
    assert torch.allclose(model.state_dict()["1.weight"], torch.full_like(initial_state["1.weight"], 4.0))
    assert "0.weight" in result["skipped_by_prefix"]


def test_load_config_can_inherit_from_base_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.yaml"
        child = root / "child.yaml"
        base.write_text(
            "exp_name: base_exp\n"
            "model:\n"
            "  feature_dim: 96\n"
            "  local_flow_head_base_temperature: 0.03\n",
            encoding="utf-8",
        )
        child.write_text(
            "base_config: base.yaml\n"
            "exp_name: child_exp\n"
            "model:\n"
            "  local_flow_head_base_temperature: 0.01\n",
            encoding="utf-8",
        )

        cfg = train_impl.load_config(child)
        facade_cfg = train_impl.load_feature_extract_config(str(child))

    assert cfg["exp_name"] == "child_exp"
    assert cfg["model"]["feature_dim"] == 96
    assert cfg["model"]["local_flow_head_base_temperature"] == 0.01
    assert facade_cfg["exp_name"] == "child_exp"
    assert facade_cfg["model"]["feature_dim"] == 96
    assert facade_cfg["model"]["local_flow_head_base_temperature"] == 0.01


def test_local_correlation_joint_loss_supports_discrete_ce_target():
    height, width = 5, 6
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    dx, dy = 1, -1
    valid = torch.zeros(1, 1, height, width)
    for y in range(height):
        for x in range(width):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            yq, xq = y + dy, x + dx
            if 0 <= yq < height and 0 <= xq < width:
                query[0, c, yq, xq] = 1.0
                valid[0, 0, y, x] = 1.0
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = dx
    flow[:, 1] = dy

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.05,
        compute_ce=True,
        ce_temperature=0.05,
    )

    assert result["losses"]["ce"].item() < 1e-3
    assert result["metrics"]["map_query_corr_ce_acc"].item() > 0.99


def test_local_correlation_joint_loss_can_ignore_near_zero_flow_targets():
    height, width = 5, 6
    channels = height * width
    rendered = torch.randn(1, channels, height, width)
    query = torch.randn_like(rendered)
    valid = torch.ones(1, 1, height, width)
    flow = torch.zeros(1, 2, height, width)

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.05,
        compute_ce=True,
        min_flow_px=0.5,
    )

    assert torch.isfinite(result["losses"]["ce"])
    assert result["metrics"]["map_query_corr_min_flow_cov"].item() == 0.0
    assert result["metrics"]["map_query_corr_ce_cov"].item() == 0.0


def test_local_correlation_joint_loss_can_ignore_too_large_flow_targets():
    height, width = 5, 6
    rendered = torch.randn(1, 4, height, width)
    query = torch.randn_like(rendered)
    valid = torch.ones(1, 1, height, width)
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 5.0

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        compute_ce=True,
        max_flow_px=2.0,
    )

    assert torch.isfinite(result["losses"]["ce"])
    assert result["metrics"]["map_query_corr_max_flow_cov"].item() == 0.0
    assert result["metrics"]["map_query_corr_ce_cov"].item() == 0.0


def test_local_correlation_joint_loss_reports_peak_diagnostics_without_peak_loss():
    rendered = torch.zeros(1, 2, 3, 3)
    query = torch.zeros_like(rendered)
    rendered[:, 0] = 1.0
    query[:, 0] = 1.0
    flow = torch.zeros(1, 2, 3, 3)
    valid = torch.ones(1, 1, 3, 3)

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=1,
        compute_flow=True,
        compute_peak=False,
        low_peak_gap_threshold=0.01,
    )

    assert "map_query_corr_peak_gap_diag" in result["metrics"]
    assert "map_query_corr_peak_low_frac" in result["metrics"]
    assert "map_query_corr_peak_low_threshold" in result["metrics"]


def test_local_correlation_joint_loss_reports_flow_magnitudes():
    height, width = 5, 6
    rendered = torch.randn(1, 4, height, width)
    query = torch.randn_like(rendered)
    valid = torch.ones(1, 1, height, width)
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 1.0

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.05,
        compute_flow=True,
    )

    assert result["metrics"]["map_query_corr_gt_flow_mag_px"].item() > 0.99
    assert result["metrics"]["map_query_corr_pred_flow_mag_px"].item() >= 0.0
    assert "map_query_corr_flow_cosine" in result["metrics"]


def test_local_correlation_joint_loss_can_penalize_wrong_flow_direction():
    class OppositeFlowHead:
        def __call__(self, corr, depth=None, valid_mask=None):
            flow = corr.new_zeros(corr.shape[0], 2, corr.shape[2], corr.shape[3])
            flow[:, 0] = -1.0
            confidence = corr.new_ones(corr.shape[0], 1, corr.shape[2], corr.shape[3])
            return {"flow": flow, "confidence": confidence}

    height, width = 5, 6
    rendered = torch.randn(1, 4, height, width)
    query = torch.randn_like(rendered)
    valid = torch.ones(1, 1, height, width)
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 1.0

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        compute_flow_cosine=True,
        flow_head=OppositeFlowHead(),
    )

    assert result["losses"]["flow_cosine"].item() > 1.9
    assert result["metrics"]["map_query_corr_flow_cosine"].item() < -0.99
    assert "map_query_corr_flow_cosine_loss" in result["metrics"]


def test_local_correlation_joint_loss_passes_intrinsics_to_context_heads():
    class RecordingMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_intrinsics = None

        def forward(self, corr, depth=None, valid_mask=None, intrinsics=None):
            self.seen_intrinsics = intrinsics
            return corr

    class RecordingFlowHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_intrinsics = None

        def forward(self, corr, depth=None, valid_mask=None, intrinsics=None):
            self.seen_intrinsics = intrinsics
            flow = corr.new_zeros(corr.shape[0], 2, corr.shape[2], corr.shape[3])
            confidence = corr.new_ones(corr.shape[0], 1, corr.shape[2], corr.shape[3])
            return {"flow": flow, "confidence": confidence}

    height, width = 4, 5
    matcher = RecordingMatcher()
    flow_head = RecordingFlowHead()
    intrinsics = torch.tensor([[20.0, 21.0, 2.0, 1.5]])

    result = local_correlation_joint_losses(
        torch.randn(1, 4, height, width),
        torch.randn(1, 4, height, width),
        torch.zeros(1, 2, height, width),
        torch.ones(1, 1, height, width),
        radius=1,
        compute_flow=True,
        matcher=matcher,
        flow_head=flow_head,
        depth=torch.ones(1, height, width),
        intrinsics=intrinsics,
    )

    assert torch.isfinite(result["losses"]["flow"])
    assert matcher.seen_intrinsics is intrinsics
    assert flow_head.seen_intrinsics is intrinsics


def test_flow_direction_loss_has_bounded_gradient_from_zero_flow():
    class LearnableZeroFlowHead:
        def __init__(self, height, width):
            self.flow = torch.zeros(1, 2, height, width, requires_grad=True)

        def __call__(self, corr, depth=None, valid_mask=None):
            confidence = corr.new_ones(corr.shape[0], 1, corr.shape[2], corr.shape[3])
            return {"flow": self.flow, "confidence": confidence}

    height, width = 5, 6
    rendered = torch.randn(1, 4, height, width)
    query = torch.randn_like(rendered)
    valid = torch.ones(1, 1, height, width)
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 1.0
    head = LearnableZeroFlowHead(height, width)

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        compute_flow_cosine=True,
        flow_head=head,
    )
    result["losses"]["flow_cosine"].backward()

    assert torch.isfinite(head.flow.grad).all()
    assert head.flow.grad.abs().max().item() < 1.0


def test_resize_query_flow_valid_does_not_rescale_flow_already_at_target_resolution():
    query = torch.randn(1, 4, 8, 10)
    flow = torch.ones(1, 2, 4, 5)
    valid = torch.ones(1, 1, 4, 5)

    query_r, flow_r, valid_r = _resize_query_flow_valid(query, flow, valid, (4, 5))

    assert query_r.shape[-2:] == (4, 5)
    assert valid_r.shape[-2:] == (4, 5)
    assert torch.allclose(flow_r, flow)


def test_local_correlation_feature_preprocess_can_concat_highpass_channels():
    feat = torch.zeros(1, 2, 5, 5)
    feat[:, 0, :, 2:] = 1.0
    feat[:, 1, 2:, :] = 2.0

    processed = local_correlation_feature_preprocess(
        feat,
        mode="concat_highpass",
        highpass_kernel=3,
        highpass_scale=2.0,
    )

    assert processed.shape == (1, 4, 5, 5)
    assert torch.allclose(processed[:, :2], feat)
    assert processed[:, 2:].abs().sum().item() > 0.0


def test_local_correlation_joint_loss_accepts_concat_highpass_preprocess():
    height, width = 5, 6
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    dx, dy = 1, 0
    valid = torch.zeros(1, 1, height, width)
    for y in range(height):
        for x in range(width):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            xq = x + dx
            if 0 <= xq < width:
                query[0, c, y, xq] = 1.0
                valid[0, 0, y, x] = 1.0
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = dx

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        compute_ce=True,
        feature_preprocess="concat_highpass",
        highpass_kernel=3,
        highpass_scale=1.0,
    )

    assert result["metrics"]["map_query_corr_ce_acc"].item() > 0.99


def test_local_correlation_distribution_loss_prefers_map_self_distribution():
    height, width = 5, 6
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    for y in range(height):
        for x in range(width):
            rendered[0, y * width + x, y, x] = 1.0
    query_good = rendered.clone()
    query_bad = torch.roll(rendered, shifts=1, dims=-1)
    valid = torch.ones(1, 1, height, width)

    good_loss, good_metrics = local_correlation_distribution_loss(
        rendered,
        query_good,
        valid,
        radius=2,
        student_temperature=0.05,
        target_temperature=0.05,
    )
    bad_loss, bad_metrics = local_correlation_distribution_loss(
        rendered,
        query_bad,
        valid,
        radius=2,
        student_temperature=0.05,
        target_temperature=0.05,
    )

    assert good_loss.item() < bad_loss.item()
    assert good_metrics["map_query_corr_distill_argmax_agree"].item() > bad_metrics[
        "map_query_corr_distill_argmax_agree"
    ].item()


def test_local_correlation_joint_loss_can_decode_argmax_st_flow():
    height, width = 5, 6
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    dx, dy = 1, -1
    valid = torch.zeros(1, 1, height, width)
    for y in range(height):
        for x in range(width):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            yq, xq = y + dy, x + dx
            if 0 <= yq < height and 0 <= xq < width:
                query[0, c, yq, xq] = 1.0
                valid[0, 0, y, x] = 1.0
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = dx
    flow[:, 1] = dy

    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.05,
        compute_flow=True,
        flow_decode_mode="argmax_st",
    )

    assert result["metrics"]["map_query_corr_flow_epe"].item() < 1e-4
    assert result["metrics"]["map_query_corr_flow_cosine"].item() > 0.99


def test_local_correlation_joint_loss_can_use_explicit_flow_head():
    height, width = 5, 6
    rendered = torch.randn(1, 4, height, width)
    query = torch.randn_like(rendered)
    valid = torch.ones(1, 1, height, width)
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 1.0

    class ConstantFlowHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow_bias = torch.nn.Parameter(torch.zeros(2))
            self.conf_bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, corr, depth=None, valid_mask=None):
            b, _c, h, w = corr.shape
            return {
                "flow": self.flow_bias.view(1, 2, 1, 1).expand(b, -1, h, w),
                "confidence": torch.sigmoid(self.conf_bias).view(1, 1, 1, 1).expand(b, -1, h, w),
                "confidence_logits": self.conf_bias.view(1, 1, 1, 1).expand(b, -1, h, w),
            }

    head = ConstantFlowHead()
    result = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.05,
        compute_flow=True,
        flow_head=head,
        flow_head_conf_weight=0.1,
    )
    total = result["losses"]["flow"] + result["losses"]["flow_conf"]
    total.backward()

    assert result["metrics"]["map_query_corr_flow_source_explicit"].item() == 1.0
    assert head.flow_bias.grad is not None
    assert head.flow_bias.grad.abs().sum().item() > 0
    assert head.conf_bias.grad is not None


def test_radio_query_student_scene_coord_head_outputs_metric_channels_and_receives_gradients():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        scene_coord_head=True,
        scene_coord_zero_init=True,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["scene_coord"].square().mean()
    loss.backward()

    assert out["scene_coord"].shape == (2, 3, 8, 10)
    assert torch.count_nonzero(out["scene_coord"]).item() == 0
    assert model.scene_coord_head[-1].weight.grad is not None


def test_radio_query_student_scene_coord_head_can_use_grid_and_global_context():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        scene_coord_head=True,
        scene_coord_zero_init=False,
        scene_coord_use_pixel_grid=True,
        scene_coord_global_context=True,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["scene_coord"].abs().mean()
    loss.backward()

    assert out["scene_coord"].shape == (2, 3, 8, 10)
    assert model.scene_context_proj is not None
    assert model.scene_context_proj.weight.grad is not None


def test_teacher_feature_store_reports_asymmetric_feature_dims_and_resolutions():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fine_dir = root / "fine_geo"
        coarse_dir = root / "coarse_sem"
        fine_dir.mkdir()
        coarse_dir.mkdir()
        torch.save(torch.randn(96, 8, 10), fine_dir / "rgb_0_fine_geo_96x8x10.pt")
        torch.save(torch.randn(32, 4, 5), coarse_dir / "rgb_0_coarse_sem_32x4x5.pt")

        store = TeacherFeatureStore(root)

        assert store.fine_feature_dim == 96
        assert store.coarse_feature_dim == 32
        assert store.feature_dim == 96
        assert store.feature_hw == (8, 10)
        assert store.coarse_feature_hw == (4, 5)


def test_query_records_can_match_teacher_ids_as_colmap_image_ids():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "seq9").mkdir()
        image_path = root / "seq9" / "frame00001.png"
        image_path.write_bytes(b"not used")

        records = build_records_from_feature_ids(
            {
                "source_dir": str(root),
                "image_patterns": ["seq*/*.png"],
            },
            teacher_indices=[42],
            image_id_to_name={42: "seq9/frame00001.png"},
        )

        assert records == [
            {
                "teacher_idx": 42,
                "image_path": str(image_path),
                "sample_name": "seq9/frame00001.png",
                "normalized_name": "seq9/frame00001.png",
            }
        ]
        assert facade_build_records_from_feature_ids(
            {
                "source_dir": str(root),
                "image_patterns": ["seq*/*.png"],
            },
            teacher_indices=[42],
            image_id_to_name={42: "seq9/frame00001.png"},
        ) == records


def test_build_records_prefers_feature_export_index_over_sorted_or_colmap_ids():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        images_root = root / "images"
        (images_root / "seq1").mkdir(parents=True)
        (images_root / "seq8").mkdir(parents=True)
        sorted_first = images_root / "seq1" / "frame00001.png"
        exported_first = images_root / "seq8" / "frame00110.png"
        sorted_first.write_bytes(b"not used")
        exported_first.write_bytes(b"not used")

        feature_root = root / "features"
        fine_dir = feature_root / "fine_geo"
        coarse_dir = feature_root / "coarse_sem"
        fine_dir.mkdir(parents=True)
        coarse_dir.mkdir(parents=True)
        torch.save(torch.randn(8, 2, 2), fine_dir / "rgb_0_fine_geo_8x2x2.pt")
        torch.save(torch.randn(4, 1, 1), coarse_dir / "rgb_0_coarse_sem_4x1x1.pt")
        (feature_root / "export_index.json").write_text(
            '[{"teacher_idx": 0, "sample_name": "seq8/frame00110.png"}]\n',
            encoding="utf-8",
        )

        records = build_all_records(
            {
                "source_dir": str(images_root),
                "feature_dir": str(feature_root),
                "image_patterns": ["seq*/*.png"],
            },
            TeacherFeatureStore(feature_root),
        )

        assert records[0]["teacher_idx"] == 0
        assert records[0]["sample_name"] == "seq8/frame00110.png"
        assert records[0]["image_path"] == str(exported_first)


def test_map_supervision_accepts_low_resolution_coarse_query_map_alignment():
    batch = {
        "rendered_map_fine": torch.randn(2, 96, 8, 10),
        "rendered_map_fine_raw": torch.randn(2, 96, 8, 10),
        "rendered_map_coarse": torch.randn(2, 32, 8, 10),
        "rendered_map_mask": torch.ones(2, 1, 8, 10),
        "teacher_fine": torch.randn(2, 96, 8, 10),
        "teacher_coarse": torch.randn(2, 32, 4, 5),
    }
    outputs = {
        "fine": torch.randn(2, 96, 8, 10, requires_grad=True),
        "coarse": torch.randn(2, 32, 4, 5, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 16},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.5,
            "query_coarse_weight": 0.5,
            "rendered_teacher_fine_weight": 0.25,
            "rendered_teacher_coarse_weight": 0.25,
            "query_fine_infonce_weight": 0.1,
            "query_coarse_infonce_weight": 0.1,
            "query_variance_weight": 0.01,
            "map_variance_weight": 0.01,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs["fine"].grad is not None
    assert outputs["coarse"].grad is not None
    assert "map_query_fine_nce_loss" in metrics
    assert "map_query_coarse_nce_loss" in metrics
    assert "map_query_variance_loss" in metrics
    assert "map_variance_loss" in metrics


def test_map_supervision_coarse_pose_rank_prefers_gt_map_pose_over_perturbed_pose():
    query_coarse = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
    rendered_pos = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
    rendered_neg = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True)
    batch = {
        "rendered_map_fine": torch.zeros(1, 2, 1, 1),
        "rendered_map_fine_raw": torch.zeros(1, 2, 1, 1),
        "rendered_map_coarse": rendered_pos,
        "rendered_map_coarse_neg": rendered_neg,
        "rendered_map_mask": torch.ones(1, 1, 1, 1),
        "rendered_map_mask_neg": torch.ones(1, 1, 1, 1),
        "teacher_fine": torch.zeros(1, 2, 1, 1),
        "teacher_coarse": torch.zeros(1, 2, 1, 1),
    }
    outputs = {
        "fine": torch.zeros(1, 2, 1, 1, requires_grad=True),
        "coarse": query_coarse,
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "coarse_start_epoch": 0,
            "coarse_pose_rank_weight": 1.0,
            "coarse_pose_rank_temperature": 0.1,
        },
    }

    good_loss, good_metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    good_loss.backward()

    bad_batch = dict(batch)
    bad_batch["rendered_map_coarse"] = rendered_neg.detach()
    bad_batch["rendered_map_coarse_neg"] = rendered_pos.detach()
    bad_loss, bad_metrics = compute_map_supervision(
        bad_batch,
        {"fine": outputs["fine"].detach(), "coarse": query_coarse.detach()},
        cfg,
        torch.device("cpu"),
        epoch=0,
    )

    assert good_loss.item() < bad_loss.item()
    assert good_metrics["map_coarse_pose_rank_acc"].item() == 1.0
    assert bad_metrics["map_coarse_pose_rank_acc"].item() == 0.0
    assert query_coarse.grad is not None
    assert rendered_pos.grad is not None
    assert rendered_neg.grad is not None


def test_map_supervision_global_coarse_pose_energy_uses_multiple_rendered_pose_negatives():
    query_coarse = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], requires_grad=True)
    rendered_pos = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], requires_grad=True)
    rendered_neg = torch.tensor(
        [[
            [[[0.0]], [[1.0]], [[0.0]]],
            [[[0.0]], [[0.0]], [[1.0]]],
        ]],
        requires_grad=True,
    )
    batch = {
        "rendered_map_fine": torch.zeros(1, 3, 1, 1),
        "rendered_map_fine_raw": torch.zeros(1, 3, 1, 1),
        "rendered_map_coarse": rendered_pos,
        "rendered_map_coarse_global_neg": rendered_neg,
        "rendered_map_mask": torch.ones(1, 1, 1, 1),
        "rendered_map_mask_global_neg": torch.ones(1, 2, 1, 1, 1),
        "teacher_fine": torch.zeros(1, 3, 1, 1),
        "teacher_coarse": torch.zeros(1, 3, 1, 1),
    }
    outputs = {
        "fine": torch.zeros(1, 3, 1, 1, requires_grad=True),
        "coarse": query_coarse,
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "coarse_start_epoch": 0,
            "coarse_pose_energy_weight": 1.0,
            "coarse_pose_energy_temperature": 0.1,
        },
    }

    good_loss, good_metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    good_loss.backward()

    bad_batch = dict(batch)
    bad_batch["rendered_map_coarse"] = rendered_neg[:, 0].detach()
    bad_batch["rendered_map_coarse_global_neg"] = torch.stack(
        [rendered_pos.detach(), rendered_neg[:, 1].detach()],
        dim=1,
    )
    bad_loss, bad_metrics = compute_map_supervision(
        bad_batch,
        {"fine": outputs["fine"].detach(), "coarse": query_coarse.detach()},
        cfg,
        torch.device("cpu"),
        epoch=0,
    )

    assert good_loss.item() < bad_loss.item()
    assert good_metrics["map_coarse_pose_energy_acc"].item() == 1.0
    assert bad_metrics["map_coarse_pose_energy_acc"].item() == 0.0
    assert query_coarse.grad is not None
    assert rendered_pos.grad is not None
    assert rendered_neg.grad is not None


def test_candidate_local_render_score_distinguishes_spatially_swapped_candidates_with_same_global_mean():
    query_coarse = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    rendered_pos = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    rendered_neg = torch.tensor([[[[[0.0, 1.0]], [[1.0, 0.0]]]]], requires_grad=True)

    loss, metrics = candidate_local_render_score_nce_loss(
        query_coarse,
        rendered_pos,
        rendered_neg,
        pos_mask=torch.ones(1, 1, 1, 2),
        neg_mask=torch.ones(1, 1, 1, 1, 2),
        temperature=0.1,
        radius=0,
    )
    loss.backward()

    assert metrics["map_coarse_pose_local_energy_acc"].item() == 1.0
    assert metrics["map_coarse_pose_local_energy_gap"].item() > 0.9
    assert query_coarse.grad is not None
    assert rendered_pos.grad is not None
    assert rendered_neg.grad is not None


def test_map_supervision_coarse_pose_local_energy_prefers_gt_candidate_over_same_global_mean_negative():
    query_coarse = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    rendered_pos = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    rendered_neg = torch.tensor([[[[[0.0, 1.0]], [[1.0, 0.0]]]]], requires_grad=True)
    batch = {
        "rendered_map_fine": torch.zeros(1, 2, 1, 2),
        "rendered_map_fine_raw": torch.zeros(1, 2, 1, 2),
        "rendered_map_coarse": rendered_pos,
        "rendered_map_coarse_global_neg": rendered_neg,
        "rendered_map_mask": torch.ones(1, 1, 1, 2),
        "rendered_map_mask_global_neg": torch.ones(1, 1, 1, 1, 2),
        "teacher_fine": torch.zeros(1, 2, 1, 2),
        "teacher_coarse": torch.zeros(1, 2, 1, 2),
    }
    outputs = {
        "fine": torch.zeros(1, 2, 1, 2, requires_grad=True),
        "coarse": query_coarse,
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "coarse_start_epoch": 0,
            "coarse_pose_local_energy_weight": 1.0,
            "coarse_pose_local_energy_temperature": 0.1,
            "coarse_pose_local_energy_radius": 0,
        },
    }

    good_loss, good_metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    good_loss.backward()

    bad_batch = dict(batch)
    bad_batch["rendered_map_coarse"] = rendered_neg[:, 0].detach()
    bad_batch["rendered_map_coarse_global_neg"] = rendered_pos.detach().unsqueeze(1)
    bad_loss, bad_metrics = compute_map_supervision(
        bad_batch,
        {"fine": outputs["fine"].detach(), "coarse": query_coarse.detach()},
        cfg,
        torch.device("cpu"),
        epoch=0,
    )

    assert good_loss.item() < bad_loss.item()
    assert good_metrics["map_coarse_pose_local_energy_acc"].item() == 1.0
    assert bad_metrics["map_coarse_pose_local_energy_acc"].item() == 0.0
    assert query_coarse.grad is not None
    assert rendered_pos.grad is not None
    assert rendered_neg.grad is not None


def _pose_with_center(center):
    pose = torch.eye(4)
    pose[:3, 3] = -torch.tensor(center, dtype=torch.float32)
    return pose


def test_render_score_candidate_listwise_loss_targets_nearest_pose_and_scores_features():
    query = torch.tensor(
        [
            [[[[1.0, 0.0]], [[0.0, 1.0]]]],
            [[[[0.0, 1.0]], [[1.0, 0.0]]]],
        ]
    ).squeeze(1).clone().detach().requires_grad_(True)
    candidates = torch.tensor(
        [
            [
                [[[0.0, 1.0]], [[1.0, 0.0]]],
                [[[1.0, 0.0]], [[0.0, 1.0]]],
                [[[0.5, 0.5]], [[0.5, 0.5]]],
            ],
            [
                [[[1.0, 0.0]], [[0.0, 1.0]]],
                [[[0.5, 0.5]], [[0.5, 0.5]]],
                [[[0.0, 1.0]], [[1.0, 0.0]]],
            ],
        ],
        requires_grad=True,
    )
    candidate_pose = torch.stack(
        [
            torch.stack([
                _pose_with_center([1.0, 0.0, 0.0]),
                _pose_with_center([0.0, 0.0, 0.0]),
                _pose_with_center([2.0, 0.0, 0.0]),
            ]),
            torch.stack([
                _pose_with_center([2.0, 0.0, 0.0]),
                _pose_with_center([1.0, 0.0, 0.0]),
                _pose_with_center([0.0, 0.0, 0.0]),
            ]),
        ]
    )
    pose_gt = torch.stack([
        _pose_with_center([0.0, 0.0, 0.0]),
        _pose_with_center([0.0, 0.0, 0.0]),
    ])

    loss, metrics = render_score_candidate_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        mode="global",
        temperature=0.1,
    )
    loss.backward()

    assert metrics["map_candidate_render_score_acc"].item() == 1.0
    assert metrics["map_candidate_render_score_target_idx"].item() == 1.5
    assert metrics["map_candidate_render_score_pred_idx"].item() == 1.5
    assert query.grad is not None
    assert candidates.grad is not None


def test_render_score_candidate_listwise_loss_supports_local_mode_and_mask_bank():
    query = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]], requires_grad=True)
    candidates = torch.tensor(
        [[
            [[[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]],
        ]],
        requires_grad=True,
    )
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.0, 0.0, 0.0]),
            _pose_with_center([1.0, 0.0, 0.0]),
        ])
    ])
    pose_gt = _pose_with_center([0.0, 0.0, 0.0]).unsqueeze(0)
    mask = torch.ones(1, 2, 1, 1, 3)

    loss, metrics = render_score_candidate_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        mask=mask,
        mode="local",
        radius=1,
        temperature=0.1,
    )
    loss.backward()

    assert metrics["map_candidate_render_score_acc"].item() == 1.0
    assert metrics["map_candidate_render_score_pred_idx"].item() == 0.0
    assert query.grad is not None
    assert candidates.grad is not None


def test_map_feature_renderer_attach_pose_candidate_renders_shapes():
    class FakeRenderer:
        device = torch.device("cpu")
        name_to_intr = {
            "a.png": {"fx": 10.0, "fy": 11.0, "cx": 1.0, "cy": 2.0},
            "b.png": {"fx": 12.0, "fy": 13.0, "cx": 3.0, "cy": 4.0},
        }

        def _render_pose(self, sample_name, pose, require_grad=False, feature="all"):
            scale = float(pose[0, 3].item())
            fine = torch.full((1, 2, 2, 3), scale)
            coarse = torch.full((1, 3, 1, 2), scale + 1.0)
            mask = torch.ones(1, 1, 1, 2)
            zeros_f = torch.zeros_like(fine)
            alpha = torch.ones(1, 1, 1, 2)
            rgb = torch.zeros(1, 3, 1, 2)
            depth = torch.ones(1, 1, 1, 2)
            pos = torch.zeros(1, 3, 1, 2)
            return zeros_f, fine, coarse, mask, alpha, rgb, depth, pos

    renderer = FakeRenderer()
    poses = torch.stack([
        torch.stack([_pose_with_center([0.0, 0.0, 0.0]), _pose_with_center([1.0, 0.0, 0.0])]),
        torch.stack([_pose_with_center([2.0, 0.0, 0.0]), _pose_with_center([3.0, 0.0, 0.0])]),
    ])
    batch = {"sample_name": ["a.png", "b.png"]}

    train_impl.MapFeatureRenderer.attach_pose_candidate_renders(renderer, batch, poses, require_grad=False)

    assert batch["rendered_map_candidate_pose"].shape == (2, 2, 4, 4)
    assert batch["rendered_map_candidate_fine"].shape == (2, 2, 2, 2, 3)
    assert batch["rendered_map_candidate_coarse"].shape == (2, 2, 3, 1, 2)
    assert batch["rendered_map_candidate_mask"].shape == (2, 2, 1, 1, 2)
    assert batch["rendered_map_candidate_depth"].shape == (2, 2, 1, 1, 2)
    assert batch["rendered_map_candidate_intrinsics"].shape == (2, 2, 4)
    assert torch.allclose(batch["rendered_map_candidate_pose"], poses)


def test_map_feature_renderer_attach_pose_candidate_renders_supports_coarse_only():
    class FakeRenderer:
        device = torch.device("cpu")
        name_to_intr = {"a.png": {"fx": 10.0, "fy": 10.0, "cx": 0.0, "cy": 0.0}}

        def __init__(self):
            self.requested_features = []

        def _render_pose(self, sample_name, pose, require_grad=False, feature="all"):
            self.requested_features.append(feature)
            scale = float(pose[0, 3].item())
            coarse = torch.full((1, 3, 1, 2), scale + 1.0)
            mask = torch.ones(1, 1, 1, 2)
            alpha = torch.ones(1, 1, 1, 2)
            rgb = torch.zeros(1, 3, 1, 2)
            depth = torch.ones(1, 1, 1, 2)
            pos = torch.zeros(1, 3, 1, 2)
            return None, None, coarse, mask, alpha, rgb, depth, pos

    renderer = FakeRenderer()
    poses = torch.stack([torch.stack([_pose_with_center([0.0, 0.0, 0.0])])])
    batch = {"sample_name": ["a.png"]}

    train_impl.MapFeatureRenderer.attach_pose_candidate_renders(
        renderer,
        batch,
        poses,
        require_grad=True,
        feature="coarse",
    )

    assert renderer.requested_features == ["coarse"]
    assert "rendered_map_candidate_fine" not in batch
    assert batch["rendered_map_candidate_coarse"].shape == (1, 1, 3, 1, 2)
    assert batch["rendered_map_candidate_depth"].shape == (1, 1, 1, 1, 2)


def test_pose_candidate_cache_index_matches_sample_name_variants(tmp_path):
    cache_path = tmp_path / "candidates.npz"
    pose0 = _pose_with_center([0.0, 0.0, 0.0]).numpy().astype(np.float32)
    pose1 = _pose_with_center([1.0, 0.0, 0.0]).numpy().astype(np.float32)
    entries = [
        {
            "query_img_id": 1,
            "query_image_name": "seq8/frame00110.png",
            "query_image_stem": "seq8_frame00110",
            "pose_init": pose0,
            "init_source": "test",
            "retrieval_frame_id": 2,
            "retrieval_image_name": "seq1/frame00001.png",
            "retrieval_score": 1.0,
            "pose_init_candidates": np.stack([pose0, pose1], axis=0),
            "candidate_valid_mask": np.asarray([True, False]),
            "retrieval_frame_ids_candidates": np.asarray([2, 3], dtype=np.int64),
            "retrieval_image_names_candidates": np.asarray(["a.png", "b.png"]),
            "retrieval_scores_candidates": np.asarray([1.0, 0.5], dtype=np.float32),
        }
    ]
    save_retrieval_init_entries(entries, {}, str(cache_path))

    index = load_pose_candidate_cache_index(str(cache_path))

    assert index["seq8/frame00110.png"]["pose_init_candidates"].shape == (2, 4, 4)
    assert index["frame00110.png"]["candidate_valid_mask"].tolist() == [True, False]
    assert index["seq8_frame00110"]["retrieval_scores_candidates"].tolist() == [1.0, 0.5]


def test_joint_radio_dataset_loads_pose_candidate_quality_fields(tmp_path):
    class FakeTeacherStore:
        def load_pair(self, _idx):
            return torch.zeros(2, 1, 2), torch.zeros(2, 1, 2)

    cache_path = tmp_path / "candidates_quality.npz"
    pose0 = _pose_with_center([0.0, 0.0, 0.0]).numpy().astype(np.float32)
    pose1 = _pose_with_center([1.0, 0.0, 0.0]).numpy().astype(np.float32)
    entries = [
        {
            "query_img_id": 1,
            "query_image_name": "seq8/frame00110.png",
            "query_image_stem": "seq8_frame00110",
            "pose_init": pose0,
            "init_source": "test",
            "retrieval_frame_id": 2,
            "retrieval_image_name": "a.png",
            "retrieval_score": 1.0,
            "pose_init_candidates": np.stack([pose0, pose1], axis=0),
            "candidate_valid_mask": np.asarray([True, True]),
            "retrieval_frame_ids_candidates": np.asarray([2, 3], dtype=np.int64),
            "retrieval_image_names_candidates": np.asarray(["a.png", "b.png"]),
            "retrieval_scores_candidates": np.asarray([100.0, 10.0], dtype=np.float32),
            "retrieval_pnp_num_inliers_candidates": np.asarray([40.0, 5.0], dtype=np.float32),
            "retrieval_pnp_reproj_median_candidates": np.asarray([0.5, 10.0], dtype=np.float32),
            "retrieval_pnp_inlier_ratio_candidates": np.asarray([0.8, 0.1], dtype=np.float32),
        }
    ]
    save_retrieval_init_entries(entries, {}, str(cache_path))
    dataset = train_impl.JointRADIOQueryDataset(
        [{"image_path": None, "teacher_idx": 0, "sample_name": "seq8/frame00110.png"}],
        FakeTeacherStore(),
        input_hw=(4, 4),
        feature_hw=(1, 2),
        synthetic_rgb=True,
        pose_candidate_cache_index=load_pose_candidate_cache_index(str(cache_path)),
        pose_candidate_topk=1,
    )

    item = dataset[0]

    assert item["pose_init_candidates"].shape == (1, 4, 4)
    assert item["retrieval_pnp_num_inliers_candidates"].tolist() == [40.0]
    assert item["retrieval_pnp_reproj_median_candidates"].tolist() == [0.5]
    assert torch.allclose(item["retrieval_pnp_inlier_ratio_candidates"], torch.tensor([0.8]))


def test_candidate_quality_features_sanitize_and_normalize_priors():
    batch = {
        "retrieval_scores_candidates": torch.tensor([[100.0, 10.0, 1.0]]),
        "retrieval_pnp_success_candidates": torch.tensor([[1.0, 1.0, 0.0]]),
        "retrieval_pnp_num_inliers_candidates": torch.tensor([[40.0, 5.0, 0.0]]),
        "retrieval_pnp_reproj_median_candidates": torch.tensor([[0.5, 10.0, float("inf")]]),
        "retrieval_pnp_inlier_ratio_candidates": torch.tensor([[0.8, 0.1, 0.0]]),
    }
    valid = torch.tensor([[True, True, False]])

    features, names = candidate_quality_features_from_batch(batch, valid_mask=valid)

    assert features.shape[:2] == (1, 3)
    assert "retrieval_pnp_reproj_median_quality" in names
    assert torch.isfinite(features).all()
    median_quality_idx = names.index("retrieval_pnp_reproj_median_quality")
    assert features[0, 0, median_quality_idx] > features[0, 1, median_quality_idx]
    assert torch.all(features[0, 2] == 0)


def test_candidate_score_fusion_listwise_loss_uses_priors_and_backprops_to_features():
    query = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
    candidates = torch.tensor(
        [[
            [[[0.0]], [[1.0]]],
            [[[1.0]], [[0.0]]],
        ]],
        requires_grad=True,
    )
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.0, 0.0, 0.0]),
            _pose_with_center([1.0, 0.0, 0.0]),
        ])
    ])
    pose_gt = _pose_with_center([0.0, 0.0, 0.0]).unsqueeze(0)
    batch = {
        "retrieval_pnp_reproj_median_candidates": torch.tensor([[0.1, 20.0]]),
        "retrieval_pnp_inlier_ratio_candidates": torch.tensor([[0.9, 0.1]]),
    }
    valid = torch.ones(1, 2, dtype=torch.bool)
    prior_features, prior_names = candidate_quality_features_from_batch(batch, valid_mask=valid)
    scorer = CandidateScoreFusionHead(input_dim=3 + prior_features.shape[-1], hidden_dim=0)
    with torch.no_grad():
        scorer.linear.weight.zero_()
        scorer.linear.bias.zero_()
        scorer.linear.weight[0, 0] = 0.25
        scorer.linear.weight[0, 3 + prior_names.index("retrieval_pnp_reproj_median_quality")] = 1.0
        scorer.linear.weight[0, 3 + prior_names.index("retrieval_pnp_inlier_ratio")] = 1.0

    loss, metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        scorer,
        batch=batch,
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
    )
    loss.backward()

    assert metrics["map_candidate_score_fusion_acc"].item() == 1.0
    assert metrics["map_candidate_score_fusion_render_acc"].item() == 0.0
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0
    assert scorer.linear.weight.grad is not None and scorer.linear.weight.grad.abs().sum() > 0


def test_candidate_score_fusion_soft_target_rewards_near_pose_candidates():
    query = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
    candidates = torch.tensor(
        [[
            [[[1.0]], [[0.0]]],
            [[[0.8]], [[0.2]]],
            [[[0.0]], [[1.0]]],
        ]],
        requires_grad=True,
    )
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.00, 0.0, 0.0]),
            _pose_with_center([0.08, 0.0, 0.0]),
            _pose_with_center([1.00, 0.0, 0.0]),
        ])
    ])
    pose_gt = _pose_with_center([0.0, 0.0, 0.0]).unsqueeze(0)
    valid = torch.ones(1, 3, dtype=torch.bool)
    scorer = CandidateScoreFusionHead(input_dim=12, hidden_dim=0)
    with torch.no_grad():
        scorer.linear.weight.zero_()
        scorer.linear.bias.zero_()
        scorer.linear.weight[0, 0] = 2.0

    loss, metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        scorer,
        batch={},
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
        target_mode="soft",
        target_temperature_m=0.20,
    )
    loss.backward()

    assert metrics["map_candidate_score_fusion_soft_target_entropy"].item() > 0.0
    assert metrics["map_candidate_score_fusion_pred_trans_mm"].item() < 100.0
    assert query.grad is not None and query.grad.abs().sum() > 0


def test_candidate_score_fusion_cost_regression_rewards_full_candidate_ordering():
    class FixedLogitScorer(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

        def forward(self, features):
            return self.logits.unsqueeze(0).expand(features.shape[0], -1)

    query = torch.tensor([[[[1.0]], [[0.0]]]])
    candidates = torch.tensor(
        [[
            [[[1.0]], [[0.0]]],
            [[[0.8]], [[0.2]]],
            [[[0.0]], [[1.0]]],
        ]]
    )
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.00, 0.0, 0.0]),
            _pose_with_center([0.20, 0.0, 0.0]),
            _pose_with_center([1.00, 0.0, 0.0]),
        ])
    ])
    pose_gt = _pose_with_center([0.0, 0.0, 0.0]).unsqueeze(0)
    valid = torch.ones(1, 3, dtype=torch.bool)
    good_scorer = FixedLogitScorer([2.0, 0.0, -2.0])
    bad_scorer = FixedLogitScorer([-2.0, 0.0, 2.0])

    good_loss, good_metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        good_scorer,
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
        target_mode="soft",
        target_temperature_m=0.20,
        cost_regression_weight=1.0,
        cost_regression_temperature_m=0.20,
    )
    bad_loss, bad_metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        bad_scorer,
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
        target_mode="soft",
        target_temperature_m=0.20,
        cost_regression_weight=1.0,
        cost_regression_temperature_m=0.20,
    )
    good_loss.backward()

    assert good_metrics["map_candidate_score_fusion_cost_regression_loss"] < bad_metrics[
        "map_candidate_score_fusion_cost_regression_loss"
    ]
    assert good_loss.item() < bad_loss.item()
    assert good_scorer.logits.grad is not None and good_scorer.logits.grad.abs().sum() > 0


def test_candidate_score_fusion_pairwise_rank_rewards_cost_ordering():
    class FixedLogitScorer(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

        def forward(self, features):
            return self.logits.unsqueeze(0).expand(features.shape[0], -1)

    query = torch.tensor([[[[1.0]], [[0.0]]]])
    candidates = torch.tensor(
        [[
            [[[1.0]], [[0.0]]],
            [[[0.7]], [[0.3]]],
            [[[0.1]], [[0.9]]],
        ]]
    )
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.00, 0.0, 0.0]),
            _pose_with_center([0.15, 0.0, 0.0]),
            _pose_with_center([0.80, 0.0, 0.0]),
        ])
    ])
    pose_gt = _pose_with_center([0.0, 0.0, 0.0]).unsqueeze(0)
    valid = torch.ones(1, 3, dtype=torch.bool)
    good_scorer = FixedLogitScorer([2.0, 0.0, -2.0])
    bad_scorer = FixedLogitScorer([-2.0, 0.0, 2.0])

    good_loss, good_metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        good_scorer,
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
        target_mode="soft",
        target_temperature_m=0.20,
        pairwise_rank_weight=1.0,
        pairwise_rank_temperature=1.0,
        pairwise_rank_min_gap_m=0.05,
    )
    bad_loss, bad_metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        bad_scorer,
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
        target_mode="soft",
        target_temperature_m=0.20,
        pairwise_rank_weight=1.0,
        pairwise_rank_temperature=1.0,
        pairwise_rank_min_gap_m=0.05,
    )
    good_loss.backward()

    assert good_metrics["map_candidate_score_fusion_pairwise_rank_loss"] < bad_metrics[
        "map_candidate_score_fusion_pairwise_rank_loss"
    ]
    assert good_loss.item() < bad_loss.item()
    assert good_scorer.logits.grad is not None and good_scorer.logits.grad.abs().sum() > 0


def test_candidate_score_map_fusion_head_can_compare_candidates_with_context():
    torch.manual_seed(7)
    scorer = CandidateScoreMapFusionHead(
        vector_dim=2,
        score_map_channels=3,
        map_channels=2,
        grid_size=2,
        hidden_dim=0,
        context_layers=1,
        context_heads=2,
    )
    score_maps = torch.randn(1, 3, 3, 5, 6)
    vector_features = torch.randn(1, 3, 2)

    logits = scorer(score_maps, vector_features)
    changed_score_maps = score_maps.clone()
    changed_score_maps[:, 1] = changed_score_maps[:, 1] + 2.0
    changed_logits = scorer(changed_score_maps, vector_features)
    logits.sum().backward()

    assert logits.shape == (1, 3)
    assert not torch.allclose(logits[:, 0], changed_logits[:, 0])
    assert scorer.linear.weight.grad is not None and scorer.linear.weight.grad.abs().sum() > 0


def test_candidate_score_map_fusion_context_residual_preserves_calibrated_prior():
    torch.manual_seed(9)
    vector_weights = [0.5, -0.25]
    scorer = CandidateScoreMapFusionHead(
        vector_dim=2,
        score_map_channels=3,
        map_channels=2,
        grid_size=1,
        hidden_dim=0,
        initial_vector_weights=vector_weights,
        initial_bias=0.1,
        context_layers=1,
        context_heads=1,
        context_residual=True,
    )
    score_maps = torch.randn(2, 4, 3, 5, 5)
    vector_features = torch.randn(2, 4, 2)

    logits = scorer(score_maps, vector_features)
    expected = vector_features @ torch.tensor(vector_weights, dtype=vector_features.dtype) + 0.1

    assert torch.allclose(logits, expected, atol=1e-6)


def test_candidate_score_fusion_wls_soft_target_uses_refined_pose_costs():
    query = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]], requires_grad=True)
    candidates = torch.stack(
        [
            query.detach().squeeze(0),
            torch.flip(query.detach().squeeze(0), dims=(-1,)),
        ],
        dim=0,
    ).unsqueeze(0).requires_grad_()
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.0, 0.0, 0.0]),
            _pose_with_center([0.5, 0.0, 0.0]),
        ])
    ])
    pose_gt = torch.stack([_pose_with_center([0.0, 0.0, 0.0])])
    valid = torch.ones(1, 2, dtype=torch.bool)
    batch = {
        "rendered_map_candidate_depth": torch.ones(1, 2, 1, 2, 2),
        "rendered_map_candidate_intrinsics": torch.tensor([[[20.0, 20.0, 0.5, 0.5], [20.0, 20.0, 0.5, 0.5]]]),
    }
    scorer = CandidateScoreFusionHead(input_dim=12, hidden_dim=0)

    loss, metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        scorer,
        batch=batch,
        candidate_valid_mask=valid,
        mode="global",
        temperature=1.0,
        target_mode="wls_soft",
        target_temperature_m=0.20,
        radius=0,
        wls_downsample=2,
    )
    loss.backward()

    assert metrics["map_candidate_score_fusion_wls_target_trans_mm"].item() < 1.0
    assert "map_candidate_score_fusion_wls_pred_trans_mm" in metrics
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0


def test_masked_score_map_stats_capture_peakiness_beyond_mean():
    score_map = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.25, 0.25], [0.25, 0.25]]],
        ]
    )

    stats = train_impl._masked_score_map_stats(score_map)

    assert torch.allclose(stats["mean"], torch.tensor([0.25, 0.25]))
    assert stats["max"][0] > stats["max"][1]
    assert stats["topk_mean"][0] > stats["topk_mean"][1]
    assert stats["peakiness"][0] > stats["peakiness"][1]


def test_candidate_score_fusion_rich_render_features_match_head_dim_and_backprop():
    query = F.normalize(torch.randn(1, 4, 4, 4), dim=1).requires_grad_()
    candidates = F.normalize(torch.randn(1, 2, 4, 4, 4), dim=2).requires_grad_()
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.0, 0.0, 0.0]),
            _pose_with_center([0.5, 0.0, 0.0]),
        ])
    ])
    pose_gt = torch.stack([_pose_with_center([0.0, 0.0, 0.0])])
    scorer = CandidateScoreFusionHead(input_dim=18, hidden_dim=0)

    loss, metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        scorer,
        render_feature_mode="rich",
    )
    loss.backward()

    assert metrics["map_candidate_score_fusion_loss"].item() > 0
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0
    assert scorer.linear.weight.grad is not None and scorer.linear.weight.grad.abs().sum() > 0


def test_candidate_score_map_fusion_head_uses_spatial_score_maps():
    score_maps = torch.randn(2, 3, 3, 5, 7, requires_grad=True)
    vector_features = torch.randn(2, 3, 4, requires_grad=True)
    scorer = CandidateScoreMapFusionHead(
        vector_dim=4,
        score_map_channels=3,
        map_channels=4,
        grid_size=2,
        hidden_dim=8,
    )

    logits = scorer(score_maps, vector_features)
    logits.sum().backward()

    assert logits.shape == (2, 3)
    assert score_maps.grad is not None and score_maps.grad.abs().sum() > 0
    assert vector_features.grad is not None and vector_features.grad.abs().sum() > 0


def test_candidate_score_map_fusion_head_can_initialize_vector_prior():
    score_maps = torch.randn(1, 2, 1, 3, 3)
    vector_features = torch.tensor([[[2.0, 4.0], [1.0, -1.0]]])
    scorer = CandidateScoreMapFusionHead(
        vector_dim=2,
        score_map_channels=1,
        map_channels=2,
        grid_size=1,
        hidden_dim=0,
        initial_vector_weights=[1.5, -0.5],
        initial_bias=0.25,
    )

    logits = scorer(score_maps, vector_features)

    assert torch.allclose(logits, torch.tensor([[[1.25], [2.25]]]).squeeze(-1))


def test_candidate_score_fusion_listwise_loss_can_use_score_map_head():
    query = F.normalize(torch.randn(1, 4, 4, 5), dim=1).requires_grad_()
    candidates = F.normalize(torch.randn(1, 2, 4, 4, 5), dim=2).requires_grad_()
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.0, 0.0, 0.0]),
            _pose_with_center([0.4, 0.0, 0.0]),
        ])
    ])
    pose_gt = torch.stack([_pose_with_center([0.0, 0.0, 0.0])])
    scorer = CandidateScoreMapFusionHead(
        vector_dim=12,
        score_map_channels=3,
        map_channels=4,
        grid_size=2,
        hidden_dim=8,
    )

    loss, metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        scorer,
        mode="local",
        radius=1,
    )
    loss.backward()

    assert metrics["map_candidate_score_fusion_loss"].item() > 0
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0


def test_candidate_score_fusion_listwise_loss_can_use_correlation_volume_score_map_head():
    query = F.normalize(torch.randn(1, 4, 4, 5), dim=1).requires_grad_()
    candidates = F.normalize(torch.randn(1, 2, 4, 4, 5), dim=2).requires_grad_()
    candidate_pose = torch.stack([
        torch.stack([
            _pose_with_center([0.0, 0.0, 0.0]),
            _pose_with_center([0.4, 0.0, 0.0]),
        ])
    ])
    pose_gt = torch.stack([_pose_with_center([0.0, 0.0, 0.0])])
    scorer = CandidateScoreMapFusionHead(
        vector_dim=12,
        score_map_channels=25,
        map_channels=4,
        grid_size=2,
        hidden_dim=8,
    )

    loss, metrics = candidate_score_fusion_listwise_loss(
        query,
        candidates,
        candidate_pose,
        pose_gt,
        scorer,
        mode="local",
        radius=2,
        score_map_mode="volume",
    )
    loss.backward()

    assert metrics["map_candidate_score_fusion_loss"].item() > 0
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0


def test_candidate_score_fusion_head_supports_explicit_linear_initialization():
    scorer = CandidateScoreFusionHead(
        input_dim=4,
        hidden_dim=0,
        initial_weights=[0.1, 0.2, 0.3, 0.4],
        initial_bias=-0.5,
    )

    assert torch.allclose(scorer.linear.weight, torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    assert torch.allclose(scorer.linear.bias, torch.tensor([-0.5]))


# Removed: test_compute_map_supervision_uses_candidate_score_fusion_head
# — candidate scorer disabled in training path for CPR Phase 1

def test_gather_candidate_bank_selects_arbitrary_topm_and_mask():
    values = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    indices = torch.tensor([[2, 0], [3, 1]])

    gathered = train_impl.gather_candidate_bank(values, indices)

    assert torch.allclose(gathered[0, 0], values[0, 2])
    assert torch.allclose(gathered[0, 1], values[0, 0])
    assert torch.allclose(gathered[1, 0], values[1, 3])
    assert torch.allclose(gathered[1, 1], values[1, 1])


def test_select_candidate_stage2_indices_can_teacher_force_oracle_candidates():
    valid = torch.tensor([[True, True, False, True]])
    pose_cost = torch.tensor([[0.30, 0.10, 0.01, 0.20]])

    selected = train_impl.select_candidate_stage2_indices(
        valid=valid,
        topm=2,
        selection="oracle",
        pose_cost=pose_cost,
    )

    assert selected.tolist() == [[1, 3]]


def test_maybe_attach_pose_candidate_renders_can_use_batch_candidate_cache():
    class FakeRenderer:
        device = torch.device("cpu")

        def attach_pose_candidate_renders(self, batch, candidate_poses, **kwargs):
            batch["rendered_map_candidate_pose"] = candidate_poses
            batch["rendered_map_candidate_valid_mask"] = kwargs.get("candidate_valid_mask")
            batch["rendered_map_candidate_require_grad"] = torch.tensor(float(kwargs.get("require_grad", False)))
            return batch

    poses = torch.stack([torch.stack([_pose_with_center([0.0, 0.0, 0.0])])])
    valid = torch.ones(1, 1, dtype=torch.bool)
    batch = {"pose_init_candidates": poses, "candidate_valid_mask": valid}
    cfg = {
        "map_supervision": {
            "enabled": True,
            "candidate_render_score_weight": 1.0,
            "candidate_render_pose_source": "batch",
            "candidate_render_score_train_map": False,
        }
    }

    out = maybe_attach_pose_candidate_renders(
        batch,
        outputs={},
        cfg=cfg,
        map_renderer=FakeRenderer(),
        require_grad=True,
        epoch=0,
    )

    assert torch.allclose(out["rendered_map_candidate_pose"], poses)
    assert out["rendered_map_candidate_valid_mask"] is valid
    assert out["rendered_map_candidate_require_grad"].item() == 0.0


def test_maybe_attach_pose_candidate_renders_can_use_online_pose_init_scores():
    class FakeRenderer:
        device = torch.device("cpu")

        def attach_pose_candidate_renders(self, batch, candidate_poses, **kwargs):
            batch["rendered_map_candidate_pose"] = candidate_poses
            batch["rendered_map_candidate_valid_mask"] = kwargs.get("candidate_valid_mask")
            return batch

    poses = torch.stack(
        [
            torch.stack(
                [
                    _pose_with_center([0.0, 0.0, 0.0]),
                    _pose_with_center([0.2, 0.0, 0.0]),
                ]
            )
        ]
    )
    scores = torch.tensor([[2.0, -1.0]])
    batch = {"sample_name": ["seq/frame.png"]}
    outputs = {"pose_init": {"pose_w2c": poses, "scores": scores}}
    cfg = {
        "map_supervision": {
            "enabled": True,
            "candidate_score_fusion_weight": 1.0,
            "candidate_render_pose_source": "feature_bank_online",
            "candidate_render_score_train_map": False,
        }
    }

    out = maybe_attach_pose_candidate_renders(
        batch,
        outputs=outputs,
        cfg=cfg,
        map_renderer=FakeRenderer(),
        require_grad=True,
        epoch=0,
    )

    assert torch.allclose(out["rendered_map_candidate_pose"], poses)
    assert torch.equal(out["rendered_map_candidate_valid_mask"], torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(out["retrieval_original_scores_candidates"], scores)
    assert torch.allclose(out["retrieval_scores_candidates"], torch.tensor([[3.0, 0.0]]))


def test_should_preattach_pose_candidate_renders_only_for_batch_sources():
    cfg = {
        "map_supervision": {
            "enabled": True,
            "candidate_render_pose_source": "batch",
            "candidate_render_score_weight": 0.0,
            "candidate_score_fusion_weight": 1.0,
        }
    }

    assert train_impl.should_preattach_pose_candidate_renders(cfg, epoch=0)

    cfg["map_supervision"]["candidate_render_pose_source"] = "pose_init"
    assert not train_impl.should_preattach_pose_candidate_renders(cfg, epoch=0)

    cfg["map_supervision"]["candidate_render_pose_source"] = "batch"
    cfg["map_supervision"]["candidate_score_fusion_weight"] = 0.0
    assert not train_impl.should_preattach_pose_candidate_renders(cfg, epoch=0)


def test_query_student_can_use_higher_localization_resolution_than_teacher():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fine_dir = root / "fine_geo"
        coarse_dir = root / "coarse_sem"
        fine_dir.mkdir()
        coarse_dir.mkdir()
        torch.save(torch.randn(64, 8, 10), fine_dir / "rgb_0_fine_geo_64x8x10.pt")
        torch.save(torch.randn(64, 8, 10), coarse_dir / "rgb_0_coarse_sem_64x8x10.pt")
        store = TeacherFeatureStore(root)
        cfg = {
            "model": {"feature_dim": 64},
            "dataset": {
                "student_feature_hw": [16, 20],
                "student_coarse_feature_hw": [8, 10],
            },
        }

        resolve_query_feature_dims(cfg, store)

        assert cfg["dataset"]["teacher_feature_hw"] == [8, 10]
        assert cfg["dataset"]["feature_hw"] == [16, 20]
        assert cfg["dataset"]["coarse_feature_hw"] == [8, 10]


def test_export_preserves_student_resolution_when_teacher_is_lower_resolution():
    class Store:
        fine_feature_dim = 64
        coarse_feature_dim = 64
        feature_hw = (8, 10)
        coarse_feature_hw = (4, 5)

    cfg = {
        "model": {"feature_dim": 64},
        "dataset": {
            "student_feature_hw": [16, 20],
            "student_coarse_feature_hw": [8, 10],
        },
    }

    apply_export_feature_dims(cfg, Store())

    assert cfg["dataset"]["teacher_feature_hw"] == [8, 10]
    assert cfg["dataset"]["teacher_coarse_feature_hw"] == [4, 5]
    assert cfg["dataset"]["feature_hw"] == [16, 20]
    assert cfg["dataset"]["coarse_feature_hw"] == [8, 10]


def test_depth_observability_weight_prioritizes_near_valid_pixels():
    depth = torch.tensor([[[1.0, 4.0]]])
    mask = torch.ones(1, 1, 1, 2)

    weight = depth_observability_weight(depth, mask=mask, strength=1.0, power=1.0)

    assert weight.shape == mask.shape
    assert weight[0, 0, 0, 0] > weight[0, 0, 0, 1]
    assert torch.isclose((weight * mask).sum() / mask.sum(), torch.tensor(1.0), atol=1e-5)


def test_translation_observability_weight_emphasizes_z_parallax_pixels():
    depth = torch.ones(1, 1, 5, 5)
    mask = torch.ones_like(depth)
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 2.0, "cy": 2.0}

    weight = translation_observability_weight(
        depth,
        intrinsics,
        mask=mask,
        strength=1.0,
        mode="z",
        power=1.0,
    )

    center = weight[0, 0, 2, 2]
    corner = weight[0, 0, 0, 0]
    assert corner > center
    assert torch.isclose((weight * mask).sum() / mask.sum(), torch.tensor(1.0), atol=1e-5)


def test_translation_observability_xy_matches_inverse_depth_for_constant_depth():
    depth = torch.ones(1, 1, 3, 3) * 2.0
    intrinsics = {"fx": 4.0, "fy": 6.0, "cx": 1.0, "cy": 1.0}

    weight = translation_observability_weight(depth, intrinsics, strength=1.0, mode="xy")

    assert torch.allclose(weight, torch.ones_like(weight), atol=1e-5)


def test_perturb_w2c_camera_center_moves_center_in_world_frame():
    pose = torch.eye(4)
    offset = torch.tensor([0.05, 0.0, 0.0])

    perturbed = perturb_w2c_camera_center(pose, offset, frame="world")
    center = -(perturbed[:3, :3].T @ perturbed[:3, 3])

    assert torch.allclose(center, offset, atol=1e-6)


def test_perturb_w2c_camera_center_moves_center_in_camera_frame():
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    offset_cam = torch.tensor([0.05, 0.0, 0.0])

    perturbed = perturb_w2c_camera_center(pose, offset_cam, frame="camera")
    center = -(perturbed[:3, :3].T @ perturbed[:3, 3])
    expected_world = pose[:3, :3].T @ offset_cam

    assert torch.allclose(center, expected_world, atol=1e-6)


def test_perturb_rank_margin_scales_with_translation_distance():
    margin = resolve_perturb_rank_margin(
        {
            "perturb_margin": 0.0002,
            "perturb_margin_per_m": 0.08,
        },
        {"rendered_map_neg_dist_m": torch.tensor([0.01, 0.05])},
        torch.device("cpu"),
    )

    assert torch.isclose(margin, torch.tensor(0.0026), atol=1e-7)


def test_compute_w2c_flow_is_zero_for_identical_poses():
    pose = torch.eye(4).unsqueeze(0)
    depth = torch.ones(1, 3, 4)
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 1.5, "cy": 1.0}

    flow, valid = compute_w2c_flow(pose, pose, depth, intrinsics)

    assert flow.shape == (1, 2, 3, 4)
    assert valid.shape == (1, 1, 3, 4)
    assert torch.allclose(flow, torch.zeros_like(flow), atol=1e-6)
    assert valid.sum().item() == 12


def test_local_correlation_subpixel_loss_prefers_gt_offset_peak():
    rendered = torch.zeros(1, 5, 5, 5)
    query = torch.zeros_like(rendered)
    for x in range(5):
        rendered[:, x, :, x] = 1.0
    # At rendered pixel x, the correct query correspondence is x + 1.
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    flow = torch.ones(1, 2, 5, 5)
    flow[:, 1] = 0.0
    valid = torch.ones(1, 1, 5, 5)
    valid[:, :, :, -1] = 0.0

    loss, metrics = local_correlation_subpixel_loss(
        rendered,
        query,
        flow,
        valid,
        radius=1,
        temperature=0.01,
    )

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_subpx_cov"] > 0
    assert metrics["map_query_corr_subpx_flow_epe"] < 0.5


def test_shifted_local_correlation_matches_reference_without_unfold():
    fmap1 = torch.randn(2, 3, 4, 5)
    fmap2 = torch.randn(2, 3, 4, 5)

    actual = shifted_local_correlation(fmap1, fmap2, radius=2)
    expected = reference_local_correlation(fmap1, fmap2, radius=2)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_joint_local_correlation_losses_match_individual_losses():
    torch.manual_seed(7)
    rendered = torch.randn(1, 5, 4, 6)
    query = torch.randn(1, 5, 4, 6, requires_grad=True)
    flow = torch.zeros(1, 2, 4, 6)
    flow[:, 0] = 0.75
    flow[:, 1] = -0.25
    valid = torch.ones(1, 1, 4, 6)
    valid[:, :, 0] = 0.0
    valid[:, :, :, -1] = 0.0

    subpx_loss, subpx_metrics = local_correlation_subpixel_loss(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.07,
    )
    flow_loss, flow_metrics = local_correlation_soft_flow_loss(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.07,
        huber_delta=0.3,
    )
    peak_loss, peak_metrics = local_correlation_peak_margin_loss(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        margin=0.04,
    )

    joint = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=2,
        temperature=0.07,
        huber_delta=0.3,
        peak_margin=0.04,
        compute_subpixel=True,
        compute_flow=True,
        compute_peak=True,
    )
    total = (
        joint["losses"]["subpixel"]
        + joint["losses"]["flow"]
        + joint["losses"]["peak"]
    )
    total.backward()

    assert torch.allclose(joint["losses"]["subpixel"], subpx_loss, atol=1e-6)
    assert torch.allclose(joint["losses"]["flow"], flow_loss, atol=1e-6)
    assert torch.allclose(joint["losses"]["peak"], peak_loss, atol=1e-6)
    assert torch.allclose(
        joint["metrics"]["map_query_corr_subpx_flow_epe"],
        subpx_metrics["map_query_corr_subpx_flow_epe"],
        atol=1e-6,
    )
    assert torch.allclose(
        joint["metrics"]["map_query_corr_flow_epe"],
        flow_metrics["map_query_corr_flow_epe"],
        atol=1e-6,
    )
    assert torch.allclose(
        joint["metrics"]["map_query_corr_peak_gap"],
        peak_metrics["map_query_corr_peak_gap"],
        atol=1e-6,
    )
    assert query.grad is not None


def test_joint_local_correlation_losses_can_use_trainable_depth_aware_matcher():
    rendered = torch.randn(1, 4, 5, 5)
    query = torch.randn(1, 4, 5, 5)
    flow = torch.zeros(1, 2, 5, 5)
    flow[:, 0] = 1.0
    valid = torch.ones(1, 1, 5, 5)

    class BiasMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(9))

        def forward(self, corr, depth=None, valid_mask=None):
            return corr + self.bias.view(1, 9, 1, 1)

    matcher = BiasMatcher()
    joint = local_correlation_joint_losses(
        rendered,
        query,
        flow,
        valid,
        radius=1,
        compute_peak=True,
        matcher=matcher,
        depth=torch.ones(1, 1, 5, 5),
    )
    joint["losses"]["peak"].backward()

    assert torch.isfinite(joint["losses"]["peak"])
    assert matcher.bias.grad is not None
    assert matcher.bias.grad.abs().sum().item() > 0


def test_map_supervision_applies_depth_aware_local_correlation_loss():
    rendered = torch.zeros(1, 6, 6, 6)
    for x in range(6):
        rendered[:, x, :, x] = 1.0
    query = torch.zeros_like(rendered)
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    query = query.clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": torch.randn(1, 6, 6, 6),
        "rendered_map_fine_raw": torch.randn(1, 6, 6, 6),
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, 6, 6),
        "rendered_map_fine_neg": rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, 6, 6),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, 6, 6), torch.zeros(1, 1, 6, 6)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, 6, 6),
        "teacher_fine": torch.randn(1, 6, 6, 6),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "local_corr_projector_enabled": True,
            "query_corr_subpixel_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_subpx_loss"].item() > 0
    assert query.grad is not None


def test_map_supervision_reports_query_corr_skip_when_negatives_missing():
    h, w = 6, 6
    query = torch.randn(1, 6, h, w, requires_grad=True)
    batch = {
        "rendered_map_fine": torch.randn(1, 6, h, w),
        "rendered_map_fine_raw": torch.randn(1, 6, h, w),
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_corr_subpixel_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_skipped_missing_negatives"].item() == 1.0
    assert metrics["map_query_corr_missing_rendered_map_fine_neg"].item() == 1.0
    assert metrics["map_query_corr_missing_rendered_map_flow_neg_to_gt"].item() == 1.0


def test_map_supervision_applies_query_corr_distribution_distillation():
    h, w = 6, 6
    neg_rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        neg_rendered[:, x, :, x] = 1.0
    gt_rendered = torch.zeros_like(neg_rendered)
    gt_rendered[:, :, :, 1:] = neg_rendered[:, :, :, :-1]
    query = gt_rendered.clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": gt_rendered,
        "rendered_map_fine_raw": gt_rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": neg_rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": gt_rendered,
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_corr_distill_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_distill_loss"].item() > 0
    assert metrics["map_query_corr_distill_argmax_agree"].item() > 0.99
    assert query.grad is not None


def test_map_supervision_applies_shared_local_corr_projector():
    h, w = 6, 6
    rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        rendered[:, x, :, x] = 1.0
    query = torch.zeros_like(rendered)
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    query = query.clone().requires_grad_(True)

    projector = torch.nn.Conv2d(6, 6, kernel_size=1, bias=False)
    with torch.no_grad():
        projector.weight.zero_()
        for c in range(6):
            projector.weight[c, c, 0, 0] = 1.0

    batch = {
        "rendered_map_fine": torch.randn(1, 6, h, w),
        "rendered_map_fine_raw": torch.randn(1, 6, h, w),
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "local_corr_projector_enabled": True,
            "query_corr_subpixel_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(
        batch,
        outputs,
        cfg,
        torch.device("cpu"),
        epoch=0,
        local_corr_projector=projector,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_projector_active"].item() == 1.0
    assert projector.weight.grad is not None
    assert projector.weight.grad.abs().sum().item() > 0


def test_validation_visuals_use_configured_student_fine_key_for_localization_branch():
    captured = {}
    original = train_impl.save_feature_track_visual

    def fake_save_feature_track_visual(output_path, **kwargs):
        captured["student_fine"] = kwargs["student_fine"].clone()

    train_impl.save_feature_track_visual = fake_save_feature_track_visual
    try:
        with tempfile.TemporaryDirectory() as tmp:
            batch = {
                "rgb": torch.zeros(1, 3, 8, 8),
                "teacher_fine": torch.zeros(1, 4, 2, 2),
                "teacher_coarse": torch.zeros(1, 2, 2, 2),
                "sample_name": ["seq/frame.png"],
            }
            outputs = {
                "fine": torch.zeros(1, 4, 2, 2),
                "fine_loc": torch.ones(1, 4, 2, 2),
                "coarse": torch.zeros(1, 2, 2, 2),
            }
            train_impl.save_validation_visuals(
                batch,
                outputs,
                qual_dir=Path(tmp) / "qual",
                feature_track_root=Path(tmp) / "track",
                step=1,
                limit=1,
                student_fine_key="fine_loc",
            )
    finally:
        train_impl.save_feature_track_visual = original

    assert torch.equal(captured["student_fine"], torch.ones(4, 2, 2))


def test_local_correlation_peak_margin_loss_is_low_for_correct_peak():
    height, width = 5, 7
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    flow_dy, flow_dx = -1, 2
    for y in range(height):
        for x in range(width):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            yq, xq = y + flow_dy, x + flow_dx
            if 0 <= yq < height and 0 <= xq < width:
                query[0, c, yq, xq] = 1.0
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = flow_dx
    flow[:, 1] = flow_dy
    valid = torch.zeros(1, 1, height, width)
    valid[:, :, 1:4, 0:5] = 1.0

    loss, metrics = local_correlation_peak_margin_loss(
        rendered,
        query,
        flow,
        valid,
        radius=3,
        margin=0.05,
    )

    assert loss.item() < 1e-4
    assert metrics["map_query_corr_peak_acc"].item() > 0.99


def test_local_correlation_wls_pose_loss_empty_mask_is_finite():
    height, width = 6, 8
    rendered = torch.randn(1, 4, height, width)
    query = torch.randn(1, 4, height, width)
    depth = torch.ones(1, height, width)
    pose = torch.eye(4).unsqueeze(0)
    intrinsics = {
        "fx": 20.0,
        "fy": 21.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    empty_mask = torch.zeros(1, 1, height, width)

    loss, metrics = local_correlation_wls_pose_loss(
        rendered,
        query,
        depth,
        pose,
        pose,
        intrinsics,
        valid_mask=empty_mask,
        radius=2,
        temperature=0.05,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["map_corr_wls_trans_err_mm"])


def test_pose_update_gain_loss_penalizes_updates_without_margin_gain():
    pose_gt = torch.eye(4).unsqueeze(0)
    pose_ref = pose_gt.clone()
    pose_ref[:, 0, 3] = -0.10
    pose_good = pose_gt.clone()
    pose_good[:, 0, 3] = -0.02
    pose_bad = pose_gt.clone()
    pose_bad[:, 0, 3] = -0.095

    good_loss, good_metrics = pose_update_gain_loss(
        pose_good,
        pose_ref,
        pose_gt,
        trans_margin_m=0.02,
        rot_margin_deg=0.0,
        rot_weight=0.0,
        trans_weight=1.0,
    )
    bad_loss, bad_metrics = pose_update_gain_loss(
        pose_bad,
        pose_ref,
        pose_gt,
        trans_margin_m=0.02,
        rot_margin_deg=0.0,
        rot_weight=0.0,
        trans_weight=1.0,
    )

    assert good_loss.item() == 0.0
    assert bad_loss.item() > 0.01
    assert good_metrics["map_corr_pose_gain_trans_margin_mm"].item() == 20.0
    assert bad_metrics["map_corr_pose_gain_trans_loss_mm"].item() > 10.0


def test_flow_warp_feature_alignment_loss_uses_depth_flow_correspondence():
    h, w = 5, 7
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    rendered = torch.stack([x, y, x * x, y * y], dim=0).unsqueeze(0)
    query = torch.zeros_like(rendered)
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    flow = torch.cat([torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)], dim=1)
    valid = torch.ones(1, 1, h, w)
    valid[:, :, :, -1] = 0.0

    good, good_metrics = flow_warp_feature_alignment_loss(rendered, query, flow, valid)
    bad, _ = flow_warp_feature_alignment_loss(rendered, query, torch.zeros_like(flow), valid)

    assert good < bad
    assert good_metrics["map_query_flow_warp_cov"].item() > 0


def test_flow_warp_contrastive_loss_penalizes_wrong_subpixel_peak():
    h, w = 5, 7
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    rendered = torch.stack([x, y, x * x, y * y], dim=0).unsqueeze(0)
    query = torch.zeros_like(rendered)
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    flow = torch.cat([torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)], dim=1)
    valid = torch.ones(1, 1, h, w)
    valid[:, :, :, -1] = 0.0

    good, good_metrics = flow_warp_contrastive_loss(rendered, query, flow, valid, margin=0.05)
    bad, _ = flow_warp_contrastive_loss(rendered, query, torch.zeros_like(flow), valid, margin=0.05)

    assert good < bad
    assert good_metrics["map_query_flow_warp_hard_gap"].item() > -0.1


def test_scene_coord_losses_use_rendered_depth_position_targets():
    h, w = 5, 7
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    position = torch.stack([x, y, x + y], dim=0).unsqueeze(0)
    center = torch.zeros(1, 3, 1, 1)
    scale = 2.0
    pred = normalize_scene_coord_map(position, center, scale).clone().requires_grad_(True)
    mask = torch.ones(1, 1, h, w)

    reg_loss, reg_metrics = scene_coord_regression_loss(pred, position, mask, center, scale)
    flow = torch.cat([torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)], dim=1)
    shifted_position = torch.zeros_like(position)
    shifted_position[:, :, :, :-1] = position[:, :, :, 1:]
    valid = torch.ones(1, 1, h, w)
    valid[:, :, :, -1] = 0.0
    warp_loss, warp_metrics = scene_coord_flow_warp_loss(
        pred,
        shifted_position,
        flow,
        valid,
        center,
        scale,
    )
    (reg_loss + warp_loss).backward()

    assert reg_loss.item() < 1e-6
    assert warp_loss.item() < 1e-6
    assert reg_metrics["map_query_scene_coord_err_cm"].item() < 1e-3
    assert warp_metrics["map_query_scene_coord_warp_err_cm"].item() < 1e-3
    assert pred.grad is not None


def test_map_supervision_applies_flow_warp_alignment_loss():
    h, w = 6, 6
    rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        rendered[:, x, :, x] = 1.0
    query = torch.zeros_like(rendered)
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    query = query.clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": torch.randn(1, 6, h, w),
        "rendered_map_fine_raw": torch.randn(1, 6, h, w),
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_flow_warp_weight": 1.0,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_flow_warp_loss"].item() > 0
    assert query.grad is not None


def test_map_supervision_applies_scene_coord_supervision_and_position_augmented_corr():
    h, w = 6, 6
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    position = torch.stack([x, y, x + y], dim=0).unsqueeze(0)
    neg_position = torch.zeros_like(position)
    neg_position[:, :, :, :-1] = position[:, :, :, 1:]
    scene_pred = (position / 2.0).clone().requires_grad_(True)
    rendered = torch.randn(1, 6, h, w)
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_position": position,
        "rendered_map_fine_neg": rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_position_neg": neg_position,
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": torch.randn(1, 6, h, w, requires_grad=True),
        "scene_coord": scene_pred,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_scene_coord_weight": 1.0,
            "query_scene_coord_warp_weight": 1.0,
            "scene_coord_center": [0.0, 0.0, 0.0],
            "scene_coord_scale": 2.0,
            "query_corr_subpixel_weight": 0.5,
            "query_corr_scene_coord_weight": 3.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_scene_coord_err_cm"].item() < 1e-3
    assert metrics["map_query_scene_coord_warp_err_cm"].item() < 1e-3
    assert metrics["map_query_corr_scene_coord_weight"].item() == 3.0
    assert scene_pred.grad is not None


def test_map_supervision_applies_flow_warp_contrastive_loss():
    h, w = 6, 6
    rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        rendered[:, x, :, x] = 1.0
    query = torch.zeros_like(rendered)
    query[:, :, :, 1:] = rendered[:, :, :, :-1]
    query = query.clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": torch.randn(1, 6, h, w),
        "rendered_map_fine_raw": torch.randn(1, 6, h, w),
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_flow_warp_contrastive_weight": 1.0,
            "query_flow_warp_contrastive_margin": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_flow_warp_contrastive_loss"].item() >= 0
    assert query.grad is not None


def test_map_supervision_applies_map_self_flow_warp_contrastive_loss():
    h, w = 6, 6
    neg_rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        neg_rendered[:, x, :, x] = 1.0
    gt_rendered = torch.zeros_like(neg_rendered)
    gt_rendered[:, :, :, 1:] = neg_rendered[:, :, :, :-1]
    gt_rendered = gt_rendered.clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": gt_rendered,
        "rendered_map_fine_raw": gt_rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": neg_rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": torch.randn(1, 6, h, w, requires_grad=True),
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "map_self_flow_warp_contrastive_weight": 1.0,
            "query_flow_warp_contrastive_margin": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_self_flow_warp_contrastive_loss"].item() >= 0
    assert gt_rendered.grad is not None


def test_map_supervision_applies_map_self_local_correlation_losses():
    h, w = 6, 6
    neg_rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        neg_rendered[:, x, :, x] = 1.0
    gt_rendered = torch.zeros_like(neg_rendered)
    gt_rendered[:, :, :, 1:] = neg_rendered[:, :, :, :-1]
    gt_rendered = gt_rendered.clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": gt_rendered,
        "rendered_map_fine_raw": gt_rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": neg_rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": torch.randn(1, 6, h, w, requires_grad=True),
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "map_self_corr_subpixel_weight": 1.0,
            "map_self_corr_flow_weight": 0.5,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_self_corr_subpx_loss"].item() > 0
    assert metrics["map_self_corr_flow_loss"].item() > 0
    assert metrics["map_self_corr_flow_epe"].item() < 0.5
    assert gt_rendered.grad is not None


def test_map_supervision_map_self_corr_uses_shared_projector():
    h, w = 6, 6
    neg_rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        neg_rendered[:, x, :, x] = 1.0
    gt_rendered = torch.zeros_like(neg_rendered)
    gt_rendered[:, :, :, 1:] = neg_rendered[:, :, :, :-1]
    projector = torch.nn.Conv2d(6, 6, kernel_size=1, bias=False)
    with torch.no_grad():
        projector.weight.zero_()
        for c in range(6):
            projector.weight[c, c, 0, 0] = 1.0
    batch = {
        "rendered_map_fine": gt_rendered,
        "rendered_map_fine_raw": gt_rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": neg_rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": torch.randn(1, 6, h, w, requires_grad=True),
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "local_corr_projector_enabled": True,
            "map_self_corr_subpixel_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(
        batch,
        outputs,
        cfg,
        torch.device("cpu"),
        epoch=0,
        local_corr_projector=projector,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_projector_active"].item() == 1.0
    assert metrics["map_self_corr_subpx_loss"].item() > 0
    assert "map_self_corr_argmax_flow_epe" in metrics
    assert projector.weight.grad is not None
    assert projector.weight.grad.abs().sum().item() > 0


def test_map_supervision_applies_projected_query_map_identity_anchor():
    h, w = 5, 6
    rendered = torch.randn(1, 4, h, w)
    query = torch.randn(1, 4, h, w, requires_grad=True)
    projector = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)
    with torch.no_grad():
        projector.weight.zero_()
        for c in range(4):
            projector.weight[c, c, 0, 0] = 1.0
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 4, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 4, h, w),
        "teacher_coarse": torch.randn(1, 4, 3, 3),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 4, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "local_corr_projector_enabled": True,
            "query_projected_fine_weight": 1.0,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(
        batch,
        outputs,
        cfg,
        torch.device("cpu"),
        epoch=0,
        local_corr_projector=projector,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_projected_fine_loss"].item() > 0
    assert metrics["map_query_projected_fine_cosine"].item() < 1.0
    assert query.grad is not None
    assert query.grad.abs().sum().item() > 0


def test_map_supervision_projected_anchor_can_project_map_only():
    h, w = 5, 6
    rendered = torch.randn(1, 4, h, w)
    projector = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)
    with torch.no_grad():
        projector.weight.zero_()
        projector.weight[0, 1, 0, 0] = 1.0
        projector.weight[1, 2, 0, 0] = -1.0
        projector.weight[2, 3, 0, 0] = 0.5
        projector.weight[3, 0, 0, 0] = 2.0
    query = projector(rendered).detach().clone().requires_grad_(True)
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 4, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 4, h, w),
        "teacher_coarse": torch.randn(1, 4, 3, 3),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 4, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "local_corr_projector_enabled": True,
            "local_corr_projector_apply_to_query": False,
            "query_projected_fine_weight": 1.0,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(
        batch,
        outputs,
        cfg,
        torch.device("cpu"),
        epoch=0,
        local_corr_projector=projector,
    )

    assert torch.isfinite(loss)
    assert metrics["map_query_projected_fine_loss"].item() < 1e-5
    assert metrics["map_query_projected_fine_cosine"].item() > 0.999


def test_map_supervision_map_self_corr_can_use_ce_and_peak_losses():
    h, w = 6, 6
    neg_rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        neg_rendered[:, x, :, x] = 1.0
    gt_rendered = torch.zeros_like(neg_rendered)
    gt_rendered[:, :, :, 1:] = neg_rendered[:, :, :, :-1]
    batch = {
        "rendered_map_fine": gt_rendered,
        "rendered_map_fine_raw": gt_rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": neg_rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.cat(
            [torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)],
            dim=1,
        ),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    batch["rendered_map_flow_valid_neg_to_gt"][:, :, :, -1] = 0.0
    outputs = {
        "fine": torch.randn(1, 6, h, w, requires_grad=True),
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "map_self_corr_ce_weight": 1.0,
            "map_self_corr_peak_margin_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(
        batch,
        outputs,
        cfg,
        torch.device("cpu"),
        epoch=0,
    )

    assert torch.isfinite(loss)
    assert metrics["map_self_corr_ce_loss"].item() > 0
    assert "map_self_corr_peak_gap" in metrics


def test_map_supervision_applies_query_identity_local_correlation_loss():
    h, w = 6, 6
    rendered = torch.randn(1, 6, h, w)
    query = torch.randn(1, 6, h, w, requires_grad=True)
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 6, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 3, 3),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_identity_corr_ce_weight": 1.0,
            "query_identity_corr_peak_margin_weight": 1.0,
            "query_identity_corr_distill_weight": 0.5,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_query_identity_corr_ce_loss"].item() > 0
    assert metrics["map_query_identity_corr_peak_margin_loss"].item() >= 0
    assert metrics["map_query_identity_corr_distill_loss"].item() > 0
    assert "map_query_identity_corr_distill_argmax_agree" in metrics
    assert "map_query_identity_corr_peak_gap" in metrics
    assert query.grad is not None


def test_map_supervision_query_identity_corr_uses_native_map_resolution():
    h, w = 4, 5
    qh, qw = 8, 10
    rendered = torch.randn(1, 6, h, w)
    query = torch.randn(1, 6, qh, qw, requires_grad=True)
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 6, 2, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 2, 3),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 2, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_identity_corr_ce_weight": 1.0,
            "query_corr_radius": 1,
            "query_corr_temperature": 0.05,
            "coarse_start_epoch": 0,
        },
    }
    calls = []
    original = train_impl.local_correlation_joint_losses

    def fake_joint(rendered_feat, query_feat, flow_gt, valid_mask, **kwargs):
        calls.append(
            (
                tuple(rendered_feat.shape[-2:]),
                tuple(query_feat.shape[-2:]),
                tuple(flow_gt.shape[-2:]),
                tuple(valid_mask.shape[-2:]),
            )
        )
        zero = rendered_feat.sum() * 0.0 + query_feat.sum() * 0.0
        return {
            "losses": {
                "ce": zero + 1.0,
                "subpixel": zero,
                "flow": zero,
                "peak": zero,
                "wls_pose": zero,
                "pose_gain": zero,
                "flow_conf": zero,
            },
            "metrics": {"map_query_corr_ce_loss": zero + 1.0},
        }

    try:
        train_impl.local_correlation_joint_losses = fake_joint
        loss, _metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    finally:
        train_impl.local_correlation_joint_losses = original

    assert torch.isfinite(loss)
    assert calls == [((h, w), (qh, qw), (h, w), (h, w))]


def test_map_supervision_can_decouple_reconstruction_and_localization_fine_heads():
    h, w = 4, 5
    rendered = torch.randn(1, 6, h, w)
    fine = torch.zeros(1, 6, h, w, requires_grad=True)
    fine_loc = torch.ones(1, 6, h, w, requires_grad=True)
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 6, 2, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 2, 3),
    }
    outputs = {
        "fine": fine,
        "fine_loc": fine_loc,
        "coarse": torch.randn(1, 6, 2, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_key": "fine",
            "query_local_fine_key": "fine_loc",
            "query_fine_weight": 1.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_identity_corr_ce_weight": 1.0,
            "query_identity_corr_radius": 1,
            "coarse_start_epoch": 0,
        },
    }
    captured_query_means = []
    original = train_impl.local_correlation_joint_losses

    def fake_joint(rendered_feat, query_feat, flow_gt, valid_mask, **kwargs):
        captured_query_means.append(float(query_feat.detach().mean()))
        zero = rendered_feat.sum() * 0.0 + query_feat.sum() * 0.0
        return {
            "losses": {
                "ce": zero + 1.0,
                "subpixel": zero,
                "flow": zero,
                "peak": zero,
                "wls_pose": zero,
                "pose_gain": zero,
                "flow_conf": zero,
            },
            "metrics": {"map_query_corr_ce_loss": zero + 1.0},
        }

    try:
        train_impl.local_correlation_joint_losses = fake_joint
        loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    finally:
        train_impl.local_correlation_joint_losses = original

    loss.backward()

    assert torch.isfinite(loss)
    assert captured_query_means == [1.0]
    assert fine.grad is not None
    assert fine_loc.grad is not None
    assert "map_query_local_fine_cosine" in metrics


def test_map_supervision_query_corr_uses_localization_fine_head():
    h, w = 4, 5
    rendered = torch.randn(1, 6, h, w)
    neg_rendered = torch.randn(1, 6, h, w)
    fine = torch.zeros(1, 6, h, w, requires_grad=True)
    fine_loc = torch.ones(1, 6, h, w, requires_grad=True)
    batch = {
        "rendered_map_fine": rendered,
        "rendered_map_fine_raw": rendered,
        "rendered_map_coarse": torch.randn(1, 6, 2, 3),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": neg_rendered,
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_neg_to_gt": torch.zeros(1, 2, h, w),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "teacher_fine": torch.randn(1, 6, h, w),
        "teacher_coarse": torch.randn(1, 6, 2, 3),
    }
    outputs = {
        "fine": fine,
        "fine_loc": fine_loc,
        "coarse": torch.randn(1, 6, 2, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_key": "fine",
            "query_local_fine_key": "fine_loc",
            "query_fine_weight": 1.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_corr_ce_weight": 1.0,
            "query_corr_radius": 1,
            "coarse_start_epoch": 0,
        },
    }
    captured_query_means = []
    original = train_impl.local_correlation_joint_losses

    def fake_joint(rendered_feat, query_feat, flow_gt, valid_mask, **kwargs):
        captured_query_means.append(float(query_feat.detach().mean()))
        zero = rendered_feat.sum() * 0.0 + query_feat.sum() * 0.0
        return {
            "losses": {
                "ce": zero + 1.0,
                "subpixel": zero,
                "flow": zero,
                "peak": zero,
                "wls_pose": zero,
                "pose_gain": zero,
                "flow_conf": zero,
            },
            "metrics": {"map_query_corr_ce_loss": zero + 1.0},
        }

    try:
        train_impl.local_correlation_joint_losses = fake_joint
        loss, _metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    finally:
        train_impl.local_correlation_joint_losses = original

    loss.backward()

    assert torch.isfinite(loss)
    assert captured_query_means == [1.0]
    assert fine.grad is not None
    assert fine_loc.grad is not None


def test_map_supervision_applies_feature_metric_pose_loss():
    h, w = 8, 8
    v, u = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    rendered = torch.stack([u, v, u * u, v * v, torch.sin(u), torch.cos(v)], dim=0).unsqueeze(0)
    query = rendered.clone().requires_grad_(True)
    pose_gt = torch.eye(4).unsqueeze(0)
    pose_neg = perturb_w2c_camera_center(pose_gt, torch.tensor([[0.02, 0.0, 0.0]]), frame="camera")
    batch = {
        "rendered_map_fine": rendered.clone(),
        "rendered_map_fine_raw": rendered.clone(),
        "rendered_map_coarse": torch.randn(1, 6, 4, 4),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": rendered.clone(),
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_depth_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "rendered_map_intrinsics": torch.tensor([[20.0, 20.0, (w - 1) / 2.0, (h - 1) / 2.0]]),
        "rendered_map_pose_gt": pose_gt,
        "rendered_map_pose_neg": pose_neg,
        "teacher_fine": rendered.clone(),
        "teacher_coarse": torch.randn(1, 6, 4, 4),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 4, 4, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "feature_metric_pose_weight": 1.0,
            "feature_metric_pose_trans_weight": 10.0,
            "feature_metric_pose_rot_weight": 0.0,
            "feature_metric_pose_normalize": False,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_feature_metric_pose_loss"].item() > 0
    assert metrics["map_feature_metric_init_trans_err_mm"].item() > 0
    assert query.grad is not None


def test_feature_metric_pose_loss_can_use_local_corr_projector():
    class ChannelSliceProjector(torch.nn.Module):
        def forward(self, feat):
            return feat[:, :3] * 2.0

    h, w = 8, 8
    rendered = torch.randn(1, 6, h, w)
    query = torch.randn(1, 6, h, w, requires_grad=True)
    pose_gt = torch.eye(4).unsqueeze(0)
    pose_neg = perturb_w2c_camera_center(pose_gt, torch.tensor([[0.02, 0.0, 0.0]]), frame="camera")
    batch = {
        "rendered_map_fine": rendered.clone(),
        "rendered_map_fine_raw": rendered.clone(),
        "rendered_map_coarse": torch.randn(1, 6, 4, 4),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": rendered.clone(),
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_depth_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "rendered_map_intrinsics": torch.tensor([[20.0, 20.0, (w - 1) / 2.0, (h - 1) / 2.0]]),
        "rendered_map_pose_gt": pose_gt,
        "rendered_map_pose_neg": pose_neg,
        "teacher_fine": rendered.clone(),
        "teacher_coarse": torch.randn(1, 6, 4, 4),
    }
    outputs = {
        "fine": query,
        "coarse": torch.randn(1, 6, 4, 4, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "local_corr_projector_enabled": True,
            "feature_metric_use_projector": True,
            "feature_metric_pose_weight": 1.0,
            "feature_metric_pose_trans_weight": 10.0,
            "feature_metric_pose_rot_weight": 0.0,
            "feature_metric_pose_normalize": False,
            "coarse_start_epoch": 0,
        },
    }
    captured_shapes = []
    original = train_impl.feature_metric_localization_loss

    def fake_feature_metric(query_feat, rendered_feat, *args, **kwargs):
        captured_shapes.append((tuple(query_feat.shape), tuple(rendered_feat.shape)))
        zero = query_feat.sum() * 0.0 + rendered_feat.sum() * 0.0
        return zero + 1.0, {
            "map_feature_metric_pose_loss": zero + 1.0,
            "map_feature_metric_init_trans_err_mm": zero + 10.0,
            "map_feature_metric_trans_err_mm": zero + 9.0,
            "map_feature_metric_trans_gain_mm": zero + 1.0,
            "map_feature_metric_delta_trans_mm": zero,
            "map_feature_metric_rot_err_deg": zero,
            "map_feature_metric_init_rot_err_deg": zero,
            "map_feature_metric_residual_l1": zero,
        }

    try:
        train_impl.feature_metric_localization_loss = fake_feature_metric
        loss, metrics = compute_map_supervision(
            batch,
            outputs,
            cfg,
            torch.device("cpu"),
            epoch=0,
            local_corr_projector=ChannelSliceProjector(),
        )
    finally:
        train_impl.feature_metric_localization_loss = original

    assert torch.isfinite(loss)
    assert metrics["map_feature_metric_pose_loss"].item() == 1.0
    assert captured_shapes == [((1, 3, h, w), (1, 3, h, w))]


def test_map_supervision_applies_map_self_feature_metric_pose_loss():
    h, w = 8, 8
    v, u = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    rendered = torch.stack([u, v, u * u, v * v, torch.sin(u), torch.cos(v)], dim=0).unsqueeze(0)
    rendered_gt = rendered.clone().requires_grad_(True)
    pose_gt = torch.eye(4).unsqueeze(0)
    pose_neg = perturb_w2c_camera_center(pose_gt, torch.tensor([[0.02, 0.0, 0.0]]), frame="camera")
    batch = {
        "rendered_map_fine": rendered_gt,
        "rendered_map_fine_raw": rendered_gt,
        "rendered_map_coarse": torch.randn(1, 6, 4, 4),
        "rendered_map_mask": torch.ones(1, 1, h, w),
        "rendered_map_fine_neg": rendered.clone(),
        "rendered_map_mask_neg": torch.ones(1, 1, h, w),
        "rendered_map_depth_neg": torch.ones(1, 1, h, w),
        "rendered_map_flow_valid_neg_to_gt": torch.ones(1, 1, h, w),
        "rendered_map_intrinsics": torch.tensor([[20.0, 20.0, (w - 1) / 2.0, (h - 1) / 2.0]]),
        "rendered_map_pose_gt": pose_gt,
        "rendered_map_pose_neg": pose_neg,
        "teacher_fine": rendered.clone(),
        "teacher_coarse": torch.randn(1, 6, 4, 4),
    }
    outputs = {
        "fine": rendered.clone().requires_grad_(True),
        "coarse": torch.randn(1, 6, 4, 4, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "map_self_feature_metric_pose_weight": 1.0,
            "feature_metric_pose_trans_weight": 10.0,
            "feature_metric_pose_rot_weight": 0.0,
            "feature_metric_pose_normalize": False,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["map_self_feature_metric_pose_loss"].item() > 0
    assert metrics["map_self_feature_metric_init_trans_err_mm"].item() > 0
    assert rendered_gt.grad is not None


def test_main_loss_can_apply_fine_feature_gradient_anchor():
    teacher_fine = torch.zeros(1, 8, 6, 6)
    teacher_fine[:, :, :, 3:] = 1.0
    teacher_coarse = torch.zeros(1, 4, 3, 3)
    outputs = {
        "fine": torch.zeros_like(teacher_fine, requires_grad=True),
        "coarse": teacher_coarse.clone().requires_grad_(True),
    }
    batch = {"teacher_fine": teacher_fine, "teacher_coarse": teacher_coarse}
    cfg = {
        "loss": {
            "fine_l1_weight": 0.0,
            "fine_cos_weight": 0.0,
            "fine_channel_std_weight": 0.0,
            "coarse_l1_weight": 0.0,
            "coarse_cos_weight": 0.0,
            "coarse_channel_std_weight": 0.0,
            "fine_grad_weight": 1.0,
            "coarse_grad_weight": 0.0,
        }
    }

    from feature_extract.train_impl import compute_main_losses

    loss, metrics = compute_main_losses(outputs, batch, cfg)
    loss.backward()

    assert loss.item() > 0
    assert metrics["fine_grad_loss"].item() > 0
    assert outputs["fine"].grad is not None


def test_map_supervision_can_apply_query_fine_gradient_alignment():
    target = torch.zeros(1, 8, 6, 6)
    target[:, :, :, 3:] = 1.0
    batch = {
        "rendered_map_fine": target,
        "rendered_map_fine_raw": target,
        "rendered_map_coarse": torch.zeros(1, 4, 3, 3),
        "rendered_map_mask": torch.ones(1, 1, 6, 6),
        "teacher_fine": target,
        "teacher_coarse": torch.zeros(1, 4, 3, 3),
    }
    outputs = {
        "fine": torch.zeros_like(target, requires_grad=True),
        "coarse": torch.zeros(1, 4, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_weight": 0.0,
            "query_coarse_weight": 0.0,
            "rendered_teacher_fine_weight": 0.0,
            "rendered_teacher_coarse_weight": 0.0,
            "query_fine_grad_weight": 1.0,
            "rendered_teacher_fine_grad_weight": 0.0,
            "coarse_start_epoch": 0,
        },
    }

    loss, metrics = compute_map_supervision(batch, outputs, cfg, torch.device("cpu"), epoch=0)
    loss.backward()

    assert loss.item() > 0
    assert metrics["map_query_fine_grad_loss"].item() > 0
    assert outputs["fine"].grad is not None


def test_infonce_can_use_cross_batch_pixel_negatives():
    pred = torch.randn(2, 8, 4, 5)
    target = torch.randn(2, 8, 4, 5)
    mask = torch.ones(2, 1, 4, 5)

    loss = infonce_contrastive_loss(
        pred,
        target,
        mask=mask,
        n_samples=12,
        cross_batch=True,
    )

    assert torch.isfinite(loss)


def test_adaptive_teacher_exporter_resolves_da3_feature_ids_and_names():
    class Cam:
        image_name = "seq1/frame0001.png"

    fid = resolve_camera_fid(Cam(), {"seq1/frame0001.png": 17, "frame0001.png": 18})
    filename = make_feature_filename(fid, "fine_geo", 96, 68, 120)

    assert fid == 17
    assert filename == "rgb_17_fine_geo_96x68x120.pt"


def test_feature_track_error_visual_resizes_adaptive_coarse_map():
    pred = torch.randn(32, 68, 120)
    target = torch.randn(32, 34, 60)
    mask = torch.ones(1, 68, 120)

    image = error_to_heatmap_image(pred, target, mask=mask)

    assert image.size == (60, 34)


if __name__ == "__main__":
    test_radio_query_student_supports_asymmetric_fine_and_coarse_outputs()
    test_radio_query_student_fine_low_level_skip_receives_gradients()
    test_radio_query_student_fine_highres_skip_receives_gradients()
    test_radio_query_student_fine_highres_zero_init_starts_as_residual_noop()
    test_radio_query_student_global_context_zero_init_starts_as_residual_noop()
    test_radio_query_student_window_attention_zero_init_starts_as_residual_noop()
    test_radio_query_student_local_corr_projector_starts_as_noop_and_trains()
    test_radio_query_student_teacher_fine_condition_starts_as_noop_and_receives_gradients()
    test_radio_query_student_depth_aware_local_matcher_starts_as_noop_and_trains()
    test_depth_aware_local_flow_head_outputs_flow_and_confidence_with_gradients()
    test_depth_aware_local_flow_head_can_start_from_softargmax_flow()
    test_radio_query_student_can_build_local_flow_head()
    test_radio_query_student_scene_coord_head_outputs_metric_channels_and_receives_gradients()
    test_radio_query_student_scene_coord_head_can_use_grid_and_global_context()
    test_teacher_feature_store_reports_asymmetric_feature_dims_and_resolutions()
    test_query_records_can_match_teacher_ids_as_colmap_image_ids()
    test_build_records_prefers_feature_export_index_over_sorted_or_colmap_ids()
    test_map_supervision_accepts_low_resolution_coarse_query_map_alignment()
    test_query_student_can_use_higher_localization_resolution_than_teacher()
    test_export_preserves_student_resolution_when_teacher_is_lower_resolution()
    test_warmstart_non_strict_can_skip_prefixes()
    test_load_config_can_inherit_from_base_config()
    test_depth_observability_weight_prioritizes_near_valid_pixels()
    test_translation_observability_weight_emphasizes_z_parallax_pixels()
    test_translation_observability_xy_matches_inverse_depth_for_constant_depth()
    test_perturb_w2c_camera_center_moves_center_in_world_frame()
    test_perturb_w2c_camera_center_moves_center_in_camera_frame()
    test_perturb_rank_margin_scales_with_translation_distance()
    test_compute_w2c_flow_is_zero_for_identical_poses()
    test_local_correlation_subpixel_loss_prefers_gt_offset_peak()
    test_local_correlation_joint_loss_can_ignore_near_zero_flow_targets()
    test_local_correlation_joint_loss_can_ignore_too_large_flow_targets()
    test_local_correlation_joint_loss_reports_flow_magnitudes()
    test_resize_query_flow_valid_does_not_rescale_flow_already_at_target_resolution()
    test_local_correlation_feature_preprocess_can_concat_highpass_channels()
    test_local_correlation_joint_loss_accepts_concat_highpass_preprocess()
    test_local_correlation_distribution_loss_prefers_map_self_distribution()
    test_local_correlation_joint_loss_can_decode_argmax_st_flow()
    test_local_correlation_joint_loss_can_use_explicit_flow_head()
    test_shifted_local_correlation_matches_reference_without_unfold()
    test_joint_local_correlation_losses_can_use_trainable_depth_aware_matcher()
    test_map_supervision_applies_depth_aware_local_correlation_loss()
    test_map_supervision_applies_query_corr_distribution_distillation()
    test_map_supervision_applies_shared_local_corr_projector()
    test_validation_visuals_use_configured_student_fine_key_for_localization_branch()
    test_local_correlation_peak_margin_loss_is_low_for_correct_peak()
    test_local_correlation_wls_pose_loss_empty_mask_is_finite()
    test_flow_warp_feature_alignment_loss_uses_depth_flow_correspondence()
    test_flow_warp_contrastive_loss_penalizes_wrong_subpixel_peak()
    test_scene_coord_losses_use_rendered_depth_position_targets()
    test_map_supervision_applies_flow_warp_alignment_loss()
    test_map_supervision_applies_scene_coord_supervision_and_position_augmented_corr()
    test_map_supervision_applies_flow_warp_contrastive_loss()
    test_map_supervision_applies_map_self_flow_warp_contrastive_loss()
    test_map_supervision_applies_map_self_local_correlation_losses()
    test_map_supervision_map_self_corr_uses_shared_projector()
    test_map_supervision_applies_projected_query_map_identity_anchor()
    test_map_supervision_projected_anchor_can_project_map_only()
    test_map_supervision_map_self_corr_can_use_ce_and_peak_losses()
    test_map_supervision_applies_query_identity_local_correlation_loss()
    test_map_supervision_query_identity_corr_uses_native_map_resolution()
    test_map_supervision_applies_feature_metric_pose_loss()
    test_map_supervision_applies_map_self_feature_metric_pose_loss()
    test_main_loss_can_apply_fine_feature_gradient_anchor()
    test_map_supervision_can_apply_query_fine_gradient_alignment()
    test_infonce_can_use_cross_batch_pixel_negatives()
    test_adaptive_teacher_exporter_resolves_da3_feature_ids_and_names()
    test_feature_track_error_visual_resizes_adaptive_coarse_map()
    print("adaptive_joint_query_map tests passed")
