from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import TexturePatchEncoder
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import RGBPatchMeasurementBranch
from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import (
    _likelihood_stats,
    apply_rgb_patch_measurements_to_match_table,
    apply_rgb_patch_measurements_to_rows,
    load_rgb_patch_measurement_branch,
)


def _write_query_image(path: Path, *, size: tuple[int, int] = (16, 16)) -> None:
    w, h = size
    xx = np.linspace(0, 255, w, dtype=np.uint8).reshape(1, w)
    yy = np.linspace(0, 255, h, dtype=np.uint8).reshape(h, 1)
    arr = np.stack(
        [
            np.broadcast_to(xx, (h, w)),
            np.broadcast_to(yy, (h, w)),
            ((np.broadcast_to(xx, (h, w)).astype(np.uint16) + np.broadcast_to(yy, (h, w)).astype(np.uint16)) // 2).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(arr, mode="RGB").save(path)


def _write_render_cache(path: Path, *, size: tuple[int, int] = (16, 16)) -> None:
    w, h = size
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    rgb = np.stack(
        [
            np.broadcast_to(xx, (h, w)),
            np.broadcast_to(yy, (h, w)),
            0.5 * (np.broadcast_to(xx, (h, w)) + np.broadcast_to(yy, (h, w))),
        ],
        axis=-1,
    )
    np.savez(path, rgb=rgb.astype(np.float32), depth=np.ones((h, w), dtype=np.float32))


def _model() -> RGBPatchMeasurementBranch:
    torch.manual_seed(0)
    model = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
    )
    model.eval()
    return model


def test_load_rgb_patch_measurement_branch_preserves_two_stage_config(tmp_path: Path) -> None:
    model = RGBPatchMeasurementBranch(
        coarse_search_radius_px=2.0,
        coarse_step_px=1.0,
        search_radius_px=0.5,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
    )
    checkpoint = tmp_path / "two_stage.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "coarse_search_radius_px": 2.0,
                "coarse_step_px": 1.0,
                "search_radius_px": 0.5,
                "context_radius_px": 1.0,
                "step_px": 0.5,
                "feature_dim": 4,
                "hidden_dim": 8,
                "input_mode": "rgb",
            },
        },
        checkpoint,
    )

    loaded = load_rgb_patch_measurement_branch(checkpoint, device=torch.device("cpu"))

    assert isinstance(loaded, RGBPatchMeasurementBranch)
    assert loaded.coarse_search_radius_px == 2.0
    assert loaded.coarse_step_px == 1.0
    assert loaded.measurement_search_radius_px == 2.5
    assert loaded.crop_radius_px == 3.5


def test_rgb_patch_match_table_fusion_only_refines_query_side(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)
    rows = [
        {
            "query_id": "q0.png",
            "match_index": "5",
            "query_center_x": "8.0",
            "query_center_y": "8.0",
            "render_x": "7.0",
            "render_y": "6.0",
            "render_depth": "4.5",
            "world_x": "1.0",
            "world_y": "2.0",
            "world_z": "3.0",
            "radio_match_score": "0.42",
            "query_gt_x": "8.25",
            "query_gt_y": "7.75",
        }
    ]

    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        rows,
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=_model(),
        image_width=16,
        image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert summary["row_count"] == 1
    row = fused_rows[0]
    assert row["render_x"] == "7.0"
    assert row["render_y"] == "6.0"
    assert row["render_depth"] == "4.5"
    assert row["depth_valid"] is True
    assert row["world_x"] == "1.0"
    assert row["world_y"] == "2.0"
    assert row["world_z"] == "3.0"
    assert float(row["query_refined_x"]) == float(row["query_center_x"]) + float(row["measurement_dx"])
    assert float(row["query_refined_y"]) == float(row["query_center_y"]) + float(row["measurement_dy"])
    assert 0.0 <= float(row["measurement_valid_prob"]) <= 1.0
    assert float(row["measurement_cov_xx"]) > 0.0
    assert float(row["measurement_cov_yy"]) > 0.0
    assert float(row["measurement_sigma_px"]) > 0.0
    assert 0.0 <= float(row["local_cost_entropy"]) <= 1.0
    assert float(row["measurement_search_radius_px"]) == 1.0
    assert float(row["measurement_context_radius_px"]) == 1.0
    assert float(row["measurement_step_px"]) == 1.0
    assert row["rgb_patch_prediction_head"] == "center"
    assert summary["prediction_head"] == "center"
    assert float(row["measurement_dx"]) == 0.0
    assert float(row["measurement_dy"]) == 0.0
    assert float(row["query_refined_x"]) == float(row["query_center_x"])
    assert float(row["query_refined_y"]) == float(row["query_center_y"])
    assert row["measurement_mean_dx"] != ""
    assert row["measurement_mean_dy"] != ""
    assert summary["measurement_improve_pair_count"] == 1
    assert summary["center_within_measurement_window_pair_count"] == 1
    assert summary["center_within_measurement_window_rate"] == 1.0
    assert summary["depth_valid_rate"] == 1.0


def test_rgb_patch_match_table_fusion_mode_head_uses_cost_volume_peak(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)

    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        [
            {
                "query_id": "q0.png",
                "match_index": "5",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
                "render_depth": "4.5",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
            }
        ],
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=_model(),
        image_width=16,
        image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
        prediction_head="mode",
    )

    row = fused_rows[0]
    assert summary["prediction_head"] == "likelihood_mode"
    assert row["rgb_patch_prediction_head"] == "likelihood_mode"
    assert float(row["measurement_dx"]) == float(row["measurement_peak_dx"])
    assert float(row["measurement_dy"]) == float(row["measurement_peak_dy"])
    assert float(row["measurement_dx"]) == float(row["measurement_mode_dx"])
    assert float(row["measurement_dy"]) == float(row["measurement_mode_dy"])
    assert row["measurement_mean_dx"] != ""
    assert row["measurement_mean_dy"] != ""


def test_rgb_patch_match_table_fusion_records_total_search_radius_for_coarse_to_fine(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)
    model = RGBPatchMeasurementBranch(
        search_radius_px=0.5,
        context_radius_px=1.0,
        step_px=0.5,
        coarse_search_radius_px=2.0,
        coarse_step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
    )

    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        [
            {
                "query_id": "q0.png",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "query_gt_x": "9.0",
                "query_gt_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
            }
        ],
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=model,
        image_width=16,
        image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert float(fused_rows[0]["measurement_search_radius_px"]) == 2.5
    assert float(fused_rows[0]["measurement_fine_search_radius_px"]) == 0.5
    assert float(fused_rows[0]["measurement_coarse_search_radius_px"]) == 2.0
    assert summary["center_within_measurement_window_rate"] == 1.0


def test_rgb_patch_match_table_fusion_likelihood_mean_head_is_explicit(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)

    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        [
            {
                "query_id": "q0.png",
                "match_index": "5",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
                "render_depth": "4.5",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
            }
        ],
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=_model(),
        image_width=16,
        image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
        prediction_head="likelihood_mean",
    )

    row = fused_rows[0]
    assert summary["prediction_head"] == "likelihood_mean"
    assert row["rgb_patch_prediction_head"] == "likelihood_mean"
    assert float(row["measurement_dx"]) == float(row["measurement_mean_dx"])
    assert float(row["measurement_dy"]) == float(row["measurement_mean_dy"])
    assert row["measurement_mode_dx"] != ""
    assert row["measurement_mode_dy"] != ""


def test_rgb_patch_match_table_fusion_can_use_gated_head(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)

    fused_rows, summary = apply_rgb_patch_measurements_to_rows(
        [
            {
                "query_id": "q0.png",
                "match_index": "5",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
                "render_depth": "4.5",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
            }
        ],
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=_model(),
        image_width=16,
        image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
        prediction_head="gated",
    )

    row = fused_rows[0]
    assert summary["prediction_head"] == "gated"
    assert row["rgb_patch_prediction_head"] == "gated"
    assert float(row["measurement_dx"]) == float(row["measurement_gated_dx"])
    assert float(row["measurement_dy"]) == float(row["measurement_gated_dy"])
    assert row["measurement_gate_prob"] != ""


def test_likelihood_stats_accepts_batched_offsets_from_data_parallel() -> None:
    logits = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=torch.float32)
    offsets = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )

    stats = _likelihood_stats(logits, offsets, covariance_floor_px2=1e-4)

    assert stats["mean"].shape == (2, 2)
    assert stats["cov"].shape == (2, 2, 2)
    assert torch.allclose(stats["peak"], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_rgb_patch_match_table_fusion_cli_accepts_gated_prediction_head() -> None:
    from feature_extract.tools.vfm.apply_rgb_patch_measurement_to_match_table import parse_args

    args = parse_args(
        [
            "--match_table_csv",
            "matches.csv",
            "--render_cache_manifest_csv",
            "render_cache.csv",
            "--image_root",
            "images",
            "--checkpoint",
            "model.pt",
            "--output_dir",
            "out",
            "--prediction_head",
            "gated",
            "--data_parallel_device_ids",
            "0",
            "1",
        ]
    )

    assert args.prediction_head == "gated"
    assert args.data_parallel_device_ids == [0, 1]


def test_rgb_patch_match_table_fusion_cli_writes_dense_depth_schema(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    input_csv = tmp_path / "match_table.csv"
    with input_csv.open("w", newline="") as handle:
        fieldnames = [
            "query_id",
            "match_index",
            "query_center_x",
            "query_center_y",
            "render_x",
            "render_y",
            "render_depth",
            "world_x",
            "world_y",
            "world_z",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "match_index": "0",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
                "render_depth": "4.5",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
            }
        )
    checkpoint = tmp_path / "rgb_patch_measurement_branch.pt"
    model = _model()
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "search_radius_px": 1.0,
                "context_radius_px": 1.0,
                "step_px": 1.0,
                "feature_dim": 4,
                "hidden_dim": 8,
                "input_mode": "rgb",
                "template_scale_factors": [1.0],
            },
        },
        checkpoint,
    )
    output_dir = tmp_path / "out"

    summary = apply_rgb_patch_measurements_to_match_table(
        match_table_csv=input_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=output_dir,
        image_width=16,
        image_height=16,
        batch_size=1,
        device="cpu",
        base_dir=tmp_path,
        data_parallel_device_ids=[0, 1],
    )

    assert summary["row_count"] == 1
    assert summary["requested_data_parallel_device_ids"] == [0, 1]
    assert summary["active_data_parallel_device_ids"] == []
    assert Path(summary["outputs"]["match_table_csv"]).exists()
    assert Path(summary["outputs"]["match_table_jsonl"]).exists()
    assert json.loads(Path(summary["outputs"]["summary"]).read_text())["row_count"] == 1
    with Path(summary["outputs"]["match_table_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["query_refined_x"] != ""
    assert rows[0]["measurement_valid_prob"] != ""
    assert rows[0]["measurement_search_radius_px"] == "1.0"
    assert rows[0]["measurement_context_radius_px"] == "1.0"
    assert rows[0]["measurement_step_px"] == "1.0"
    assert rows[0]["render_depth"] == "4.5"


def test_rgb_patch_match_table_fusion_supports_different_query_and_render_sizes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png", size=(16, 16))
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache, size=(8, 8))
    rows = [
        {
            "query_id": "q0.png",
            "match_index": "0",
            "query_center_x": "8.0",
            "query_center_y": "8.0",
            "render_x": "4.0",
            "render_y": "4.0",
            "render_depth": "4.5",
            "world_x": "1.0",
            "world_y": "2.0",
            "world_z": "3.0",
        }
    ]

    fused_rows, _summary = apply_rgb_patch_measurements_to_rows(
        rows,
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=_model(),
        query_image_width=16,
        query_image_height=16,
        render_image_width=8,
        render_image_height=8,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert fused_rows[0]["render_x"] == "4.0"
    assert fused_rows[0]["query_refined_x"] != ""


def test_rgb_patch_checkpoint_loader_accepts_legacy_missing_empty_prior_scale_buffer(tmp_path: Path) -> None:
    model = _model()
    state = model.state_dict()
    state.pop("prior_scale_expert_centers")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": state,
            "config": {
                "search_radius_px": 1.0,
                "context_radius_px": 1.0,
                "step_px": 1.0,
                "feature_dim": 4,
                "hidden_dim": 8,
                "input_mode": "rgb",
                "template_scale_factors": [1.0],
            },
        },
        checkpoint,
    )

    loaded = load_rgb_patch_measurement_branch(checkpoint, device=torch.device("cpu"))

    assert isinstance(loaded, RGBPatchMeasurementBranch)
    assert loaded.search_radius_px == 1.0


def test_rgb_patch_checkpoint_loader_accepts_cached_texture_projection_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    projection = TexturePatchEncoder(feature_dim=4, hidden_dim=8, input_mode="rgb_graygrad")
    checkpoint = tmp_path / "cached_projection.pt"
    torch.save(
        {
            "model": projection.state_dict(),
            "config": {
                "projection_type": "texture_rgb_graygrad",
                "input_dim": 3,
                "hidden_dim": 8,
                "output_dim": 4,
                "search_radius_px": 1.0,
                "context_radius_px": 1.0,
                "step_px": 1.0,
                "temperature": 0.2,
                "crop_before_projection": True,
            },
        },
        checkpoint,
    )

    loaded = load_rgb_patch_measurement_branch(checkpoint, device=torch.device("cpu"))
    query_patch = torch.rand((2, 3, 5, 5), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 5, 5), dtype=torch.float32)
    pred = loaded.forward_from_patches(query_patch, render_patch)

    assert loaded.search_radius_px == 1.0
    assert loaded.context_radius_px == 1.0
    assert loaded.step_px == 1.0
    assert loaded.crop_radius_px == 2.0
    assert pred.logits.shape == (2, 9)
    assert pred.offsets_xy.shape == (9, 2)
    assert pred.dustbin_logit.shape == (2,)


def test_rgb_patch_checkpoint_loader_accepts_joint_measurement_branch_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
    )
    checkpoint = tmp_path / "joint.pt"
    torch.save(
        {
            "format": "matcha_joint_model_v1",
            "model_config": {
                "measurement_patch_config": {
                    "search_radius_px": 1.0,
                    "context_radius_px": 1.0,
                    "step_px": 1.0,
                    "coarse_search_radius_px": 0.0,
                    "coarse_step_px": 0.0,
                    "feature_dim": 4,
                    "hidden_dim": 8,
                    "input_mode": "rgb",
                    "encoder_arch": "simple",
                }
            },
            "state_dict": {
                f"measurement_patch_branch.{key}": value
                for key, value in model.state_dict().items()
            },
        },
        checkpoint,
    )

    loaded = load_rgb_patch_measurement_branch(checkpoint, device=torch.device("cpu"))
    query_patch = torch.rand((2, 3, 5, 5), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 5, 5), dtype=torch.float32)
    pred = loaded.forward_from_patches(query_patch, render_patch)

    assert isinstance(loaded, RGBPatchMeasurementBranch)
    assert loaded.measurement_model_type == "joint_measurement_patch_branch"
    assert loaded.search_radius_px == 1.0
    assert pred.logits.shape == (2, 9)


def test_rgb_patch_match_table_fusion_can_apply_cached_texture_projection_checkpoint(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    input_csv = tmp_path / "match_table.csv"
    with input_csv.open("w", newline="") as handle:
        fieldnames = [
            "query_id",
            "match_index",
            "query_center_x",
            "query_center_y",
            "render_x",
            "render_y",
            "render_depth",
            "world_x",
            "world_y",
            "world_z",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "match_index": "0",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
                "render_depth": "4.5",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
            }
        )
    projection = TexturePatchEncoder(feature_dim=4, hidden_dim=8, input_mode="rgb_graygrad")
    checkpoint = tmp_path / "cached_projection.pt"
    torch.save(
        {
            "model": projection.state_dict(),
            "config": {
                "projection_type": "texture_rgb_graygrad",
                "input_dim": 3,
                "hidden_dim": 8,
                "output_dim": 4,
                "search_radius_px": 1.0,
                "context_radius_px": 1.0,
                "step_px": 1.0,
                "temperature": 0.2,
                "crop_before_projection": True,
            },
        },
        checkpoint,
    )

    summary = apply_rgb_patch_measurements_to_match_table(
        match_table_csv=input_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=tmp_path / "out_cached_projection",
        image_width=16,
        image_height=16,
        batch_size=1,
        device="cpu",
        base_dir=tmp_path,
    )

    with Path(summary["outputs"]["match_table_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["measurement_model_type"] == "cached_projection"
    assert rows[0]["query_refined_x"] != ""
    assert rows[0]["measurement_valid_prob"] == "0.5"
    assert rows[0]["rgb_patch_prediction_head"] == "center"
