import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_field.dcff.losses import infonce_contrastive_loss
from feature_field.utils.feature_track_vis import error_to_heatmap_image
from feature_extract import build_records_from_feature_ids as facade_build_records_from_feature_ids
from feature_extract.scripts.export_adaptive_teacher_features import make_feature_filename, resolve_camera_fid
from feature_extract.students.radio_query_student import RadioQueryStudent
from feature_extract.train_impl import (
    TeacherFeatureStore,
    build_all_records,
    build_records_from_feature_ids,
    compute_map_supervision,
    compute_w2c_flow,
    depth_observability_weight,
    flow_warp_contrastive_loss,
    flow_warp_feature_alignment_loss,
    local_correlation_joint_losses,
    local_correlation_peak_margin_loss,
    local_correlation_soft_flow_loss,
    local_correlation_subpixel_loss,
    local_correlation_wls_pose_loss,
    normalize_scene_coord_map,
    perturb_w2c_camera_center,
    resolve_perturb_rank_margin,
    resolve_query_feature_dims,
    sample_query_feature_by_flow,
    scene_coord_flow_warp_loss,
    scene_coord_regression_loss,
    shifted_local_correlation,
    translation_observability_weight,
)
from feature_extract.export_impl import apply_export_feature_dims
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


def test_radio_query_student_fine_loc_head_starts_as_noop_and_receives_gradients():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        fine_loc_head=True,
        fine_loc_zero_init=True,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["fine_loc"].square().mean()
    loss.backward()

    assert torch.allclose(out["fine_loc"], out["fine"], atol=1e-6)
    assert model.fine_loc_head is not None
    assert model.fine_loc_head[-1].weight.grad is not None


def test_radio_query_student_fine_loc_can_detach_base_descriptor():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        fine_loc_head=True,
        fine_loc_zero_init=True,
        fine_loc_detach_base=True,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["fine_loc"].square().mean()
    loss.backward()

    assert model.fine_loc_head[-1].weight.grad is not None
    assert model.fine_head[-1].weight.grad is None


def test_radio_query_student_fine_loc_highres_branch_starts_as_noop_and_receives_gradients():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(16, 20),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        fine_loc_head=True,
        fine_loc_zero_init=True,
        fine_loc_detach_base=True,
        fine_loc_highres_source="stage2",
        fine_loc_highres_init=1.0,
        fine_loc_highres_zero_init=True,
        fine_loc_highres_detach=True,
    )

    out = model(torch.randn(2, 3, 64, 80))
    loss = out["fine_loc"].square().mean()
    loss.backward()

    assert torch.allclose(out["fine_loc"], out["fine"], atol=1e-6)
    assert model.fine_loc_highres_fuse is not None
    assert model.fine_loc_highres_fuse[-1].weight.grad is not None
    assert model.fine_head[-1].weight.grad is None


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


def test_map_supervision_can_use_fine_loc_for_localization_losses():
    h, w = 6, 6
    rendered = torch.zeros(1, 6, h, w)
    for x in range(w):
        rendered[:, x, :, x] = 1.0
    fine = torch.randn(1, 6, h, w, requires_grad=True)
    fine_loc = torch.zeros_like(rendered)
    fine_loc[:, :, :, 1:] = rendered[:, :, :, :-1]
    fine_loc = fine_loc.clone().requires_grad_(True)
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
        "fine": fine,
        "fine_loc": fine_loc,
        "coarse": torch.randn(1, 6, 3, 3, requires_grad=True),
    }
    cfg = {
        "loss": {"infonce_temperature": 0.07, "infonce_samples": 8},
        "map_supervision": {
            "enabled": True,
            "query_fine_key": "fine_loc",
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
    grad_loc, grad_fine = torch.autograd.grad(loss, [fine_loc, fine], allow_unused=True)

    assert torch.isfinite(loss)
    assert metrics["map_query_corr_subpx_loss"].item() > 0
    assert grad_loc is not None
    assert grad_fine is None


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
    test_radio_query_student_fine_loc_head_starts_as_noop_and_receives_gradients()
    test_radio_query_student_fine_loc_can_detach_base_descriptor()
    test_radio_query_student_fine_loc_highres_branch_starts_as_noop_and_receives_gradients()
    test_radio_query_student_teacher_fine_condition_starts_as_noop_and_receives_gradients()
    test_radio_query_student_scene_coord_head_outputs_metric_channels_and_receives_gradients()
    test_radio_query_student_scene_coord_head_can_use_grid_and_global_context()
    test_teacher_feature_store_reports_asymmetric_feature_dims_and_resolutions()
    test_query_records_can_match_teacher_ids_as_colmap_image_ids()
    test_build_records_prefers_feature_export_index_over_sorted_or_colmap_ids()
    test_map_supervision_accepts_low_resolution_coarse_query_map_alignment()
    test_query_student_can_use_higher_localization_resolution_than_teacher()
    test_export_preserves_student_resolution_when_teacher_is_lower_resolution()
    test_depth_observability_weight_prioritizes_near_valid_pixels()
    test_translation_observability_weight_emphasizes_z_parallax_pixels()
    test_translation_observability_xy_matches_inverse_depth_for_constant_depth()
    test_perturb_w2c_camera_center_moves_center_in_world_frame()
    test_perturb_w2c_camera_center_moves_center_in_camera_frame()
    test_perturb_rank_margin_scales_with_translation_distance()
    test_compute_w2c_flow_is_zero_for_identical_poses()
    test_local_correlation_subpixel_loss_prefers_gt_offset_peak()
    test_shifted_local_correlation_matches_reference_without_unfold()
    test_map_supervision_applies_depth_aware_local_correlation_loss()
    test_map_supervision_can_use_fine_loc_for_localization_losses()
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
    test_map_supervision_applies_feature_metric_pose_loss()
    test_map_supervision_applies_map_self_feature_metric_pose_loss()
    test_main_loss_can_apply_fine_feature_gradient_anchor()
    test_map_supervision_can_apply_query_fine_gradient_alignment()
    test_infonce_can_use_cross_batch_pixel_negatives()
    test_adaptive_teacher_exporter_resolves_da3_feature_ids_and_names()
    test_feature_track_error_visual_resizes_adaptive_coarse_map()
    print("adaptive_joint_query_map tests passed")
