from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from feature_extract.tools.vfm.train_vfm_highres_geometry_head import (
    HighResGeometryDataset,
    _parse_args,
)
from feature_extract.vfm.vfm_highres_geometry_head import (
    RadioHighResGeometryHead,
    geometry_confidence_loss,
    masked_geometry_loss,
    masked_log_depth_gradient_loss,
    masked_log_depth_l1,
    masked_scale_invariant_log_depth_loss,
    predict_radio_geometry_map,
)
from feature_extract.vfm.localization_goal_maplet.surface_refiner import (
    dense_scale_invariant_geometry_evidence,
)


def test_geometry_head_fixed_epoch_parser_allows_final_fit_without_eval() -> None:
    args = _parse_args([
        "--train_geometry_manifest", "train.json",
        "--output_dir", "out",
        "--checkpoint_protocol", "fixed_epoch_no_selection",
    ])
    assert args.eval_geometry_manifest == ""
    assert args.checkpoint_protocol == "fixed_epoch_no_selection"


def test_radio_highres_geometry_head_decodes_to_requested_resolution() -> None:
    model = RadioHighResGeometryHead(in_channels=8, hidden_channels=16)
    tokens = torch.randn(2, 8, 6, 10)

    output = model(tokens, output_size=(24, 40))

    assert output.depth.shape == (2, 24, 40)
    assert output.normal.shape == (2, 3, 24, 40)
    assert output.confidence.shape == (2, 24, 40)
    assert torch.isfinite(output.depth).all()
    assert torch.isfinite(output.normal).all()
    assert torch.allclose(torch.linalg.norm(output.normal, dim=1), torch.ones(2, 24, 40), atol=1e-4)


def test_radio_geometry_map_prediction_has_image_aligned_layout() -> None:
    model = RadioHighResGeometryHead(in_channels=8, hidden_channels=16)
    prediction = predict_radio_geometry_map(
        model, np.zeros((8, 3, 5), dtype=np.float32), output_size=(7, 11),
    )
    assert prediction.depth.shape == (7, 11)
    assert prediction.normal.shape == (7, 11, 3)
    assert prediction.confidence.shape == (7, 11)


def test_radio_highres_geometry_head_supports_separate_decoders() -> None:
    model = RadioHighResGeometryHead(in_channels=8, hidden_channels=16, architecture="separate_decoders")
    tokens = torch.randn(1, 8, 6, 10)

    output = model(tokens, output_size=(18, 30))

    assert output.depth.shape == (1, 18, 30)
    assert output.normal.shape == (1, 3, 18, 30)
    assert model.depth_refine is not model.normal_refine


def test_radio_highres_geometry_head_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        RadioHighResGeometryHead(in_channels=8, hidden_channels=16, architecture="unknown")


def test_masked_geometry_loss_uses_depth_and_normal_terms() -> None:
    pred_depth = torch.ones(1, 4, 5) * 4.0
    pred_normal = torch.zeros(1, 3, 4, 5)
    pred_normal[:, 2] = -1.0
    target_depth = pred_depth.clone()
    target_normal = pred_normal.clone()
    valid = torch.ones(1, 4, 5, dtype=torch.bool)

    loss = masked_geometry_loss(pred_depth, pred_normal, target_depth, target_normal, valid)

    assert torch.isfinite(loss)
    assert float(loss.detach().cpu()) < 1e-6


def test_masked_depth_loss_keeps_gradient_when_valid_mask_is_empty() -> None:
    pred_depth = torch.ones(1, 4, 5, requires_grad=True)
    target_depth = torch.ones(1, 4, 5)
    valid = torch.zeros(1, 4, 5, dtype=torch.bool)

    loss = masked_log_depth_l1(pred_depth, target_depth, valid)
    loss.backward()

    assert pred_depth.grad is not None
    assert torch.all(pred_depth.grad == 0)


def test_geometry_confidence_loss_prefers_validity_aligned_probability() -> None:
    valid = torch.tensor([[[True, False], [True, False]]])
    aligned = torch.tensor([[[0.9, 0.1], [0.8, 0.2]]])
    reversed_probability = 1.0 - aligned

    assert geometry_confidence_loss(aligned, valid) < geometry_confidence_loss(
        reversed_probability, valid,
    )


def test_scale_invariant_depth_loss_ignores_global_scale_but_gradient_detects_shape() -> None:
    target = torch.tensor([[[2.0, 3.0, 5.0], [2.5, 4.0, 6.0]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    scaled = 1.8 * target
    distorted = scaled.clone()
    distorted[:, :, -1] *= 1.4

    assert float(masked_scale_invariant_log_depth_loss(scaled, target, valid)) < 1e-6
    assert masked_scale_invariant_log_depth_loss(distorted, target, valid) > 0.01
    assert float(masked_log_depth_gradient_loss(scaled, target, valid)) < 1e-6
    assert masked_log_depth_gradient_loss(distorted, target, valid) > 0.01


def test_geometry_dataset_rejects_missing_token_layer(tmp_path) -> None:
    token_path = tmp_path / "tokens.npz"
    geometry_path = tmp_path / "geometry.npz"
    np.savez_compressed(token_path, other_layer=np.zeros((8, 2, 3), dtype=np.float32))
    np.savez_compressed(
        geometry_path,
        depth=np.ones((4, 6), dtype=np.float32),
        normal_cam=np.zeros((4, 6, 3), dtype=np.float32),
        valid=np.ones((4, 6), dtype=bool),
    )
    manifest_path = tmp_path / "geometry_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_id": "frame.png",
                        "token_path": str(token_path),
                        "geometry_path": str(geometry_path),
                    }
                ]
            }
        )
    )
    dataset = HighResGeometryDataset(manifest_path, layer_name="radio_final")

    with pytest.raises(KeyError, match="radio_final"):
        _ = dataset[0]


def test_dense_geometry_evidence_marginalizes_scale_and_uses_fixed_denominator() -> None:
    depth = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    normal = np.zeros((2, 2, 3), dtype=np.float64)
    normal[..., 2] = 1.0
    full = SimpleNamespace(
        depth=depth,
        normal=normal,
        mask=np.ones((2, 2), dtype=bool),
    )
    evidence = dense_scale_invariant_geometry_evidence(
        full, np.eye(4), 2.5 * depth, normal, np.ones((2, 2)),
    )
    assert evidence["score"] > 0.99
    partial = SimpleNamespace(
        depth=depth,
        normal=normal,
        mask=np.asarray([[True, True], [False, False]]),
    )
    partial_evidence = dense_scale_invariant_geometry_evidence(
        partial, np.eye(4), 2.5 * depth, normal, np.ones((2, 2)),
    )
    assert np.isclose(partial_evidence["coverage"], 0.5)
    assert partial_evidence["score"] < 0.51
