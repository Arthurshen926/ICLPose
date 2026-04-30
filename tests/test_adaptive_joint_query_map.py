import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_field.dcff.losses import infonce_contrastive_loss
from feature_field.utils.feature_track_vis import error_to_heatmap_image
from feature_extract.scripts.export_adaptive_teacher_features import make_feature_filename, resolve_camera_fid
from feature_extract.students.radio_query_student import RadioQueryStudent
from feature_extract.train_impl import (
    TeacherFeatureStore,
    build_records_from_feature_ids,
    compute_map_supervision,
)


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
    test_teacher_feature_store_reports_asymmetric_feature_dims_and_resolutions()
    test_query_records_can_match_teacher_ids_as_colmap_image_ids()
    test_map_supervision_accepts_low_resolution_coarse_query_map_alignment()
    test_infonce_can_use_cross_batch_pixel_negatives()
    test_adaptive_teacher_exporter_resolves_da3_feature_ids_and_names()
    test_feature_track_error_visual_resizes_adaptive_coarse_map()
    print("adaptive_joint_query_map tests passed")
