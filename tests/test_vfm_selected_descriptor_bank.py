import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from feature_extract.vfm.selected_descriptor_bank import build_selected_descriptor_bank
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _selector_checkpoint(path):
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.bias.zero_()
        selector.uncertainty_head.weight.zero_()
        selector.uncertainty_head.bias.zero_()
    torch.save(selector.state_dict(), path)


def _record(tmp_path, image_id, feature):
    path = tmp_path / f"{image_id}.npz"
    np.savez_compressed(path, radio_final=np.asarray(feature, dtype=np.float32))
    return TokenBankRecord(
        image_id=image_id,
        token_path=path,
        layers=(TokenLayerSpec("radio_final", "radio", "final", 2, 14),),
        split="test",
        scene="synthetic",
    )


def test_build_selected_descriptor_bank_batches_selector_forward(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "a", [[[1.0, 1.0]], [[0.0, 0.0]]]),
            _record(tmp_path, "b", [[[0.0, 0.0]], [[2.0, 2.0]]]),
        )
    )
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0

    bank = build_selected_descriptor_bank(
        manifest=manifest,
        selector=selector,
        layer_name="radio_final",
        device="cpu",
        batch_size=2,
    )

    assert bank.image_ids == ("a", "b")
    np.testing.assert_allclose(bank.descriptors, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert bank.pooling == "selected_mean"
    assert bank.metadata["batch_size"] == 2


def test_build_selected_descriptor_bank_can_use_utility_weighted_pooling(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "a", [[[1.0, 0.0]], [[0.0, 1.0]]]),
        )
    )
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.weight[0, 0, 0, 0] = 8.0
        selector.utility_head.bias.zero_()

    unweighted = build_selected_descriptor_bank(
        manifest=manifest,
        selector=selector,
        layer_name="radio_final",
        device="cpu",
        batch_size=1,
        utility_weighted_pooling=False,
    )
    weighted = build_selected_descriptor_bank(
        manifest=manifest,
        selector=selector,
        layer_name="radio_final",
        device="cpu",
        batch_size=1,
        utility_weighted_pooling=True,
    )

    np.testing.assert_allclose(unweighted.descriptors, np.asarray([[0.7071, 0.7071]], dtype=np.float32), atol=1e-3)
    assert weighted.descriptors[0, 0] > weighted.descriptors[0, 1]
    assert weighted.pooling == "selected_utility_weighted_mean"
    assert weighted.metadata["utility_weighted_pooling"] is True


def test_build_selected_descriptor_bank_can_mask_high_or_low_spatial_utility(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "a", [[[1.0, 0.0]], [[0.0, 1.0]]]),
        )
    )
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.weight[0, 0, 0, 0] = 8.0
        selector.utility_head.bias.zero_()

    high_removed = build_selected_descriptor_bank(
        manifest=manifest,
        selector=selector,
        layer_name="radio_final",
        device="cpu",
        batch_size=1,
        utility_spatial_mask="high",
        utility_spatial_mask_fraction=0.5,
    )
    low_removed = build_selected_descriptor_bank(
        manifest=manifest,
        selector=selector,
        layer_name="radio_final",
        device="cpu",
        batch_size=1,
        utility_spatial_mask="low",
        utility_spatial_mask_fraction=0.5,
    )

    np.testing.assert_allclose(high_removed.descriptors, np.asarray([[0.0, 1.0]], dtype=np.float32), atol=1e-3)
    np.testing.assert_allclose(low_removed.descriptors, np.asarray([[1.0, 0.0]], dtype=np.float32), atol=1e-3)
    assert high_removed.metadata["utility_spatial_mask"] == "high"
    assert high_removed.metadata["utility_spatial_mask_fraction"] == pytest.approx(0.5)


def test_build_selected_descriptor_bank_cli_writes_npz(tmp_path):
    manifest = TokenBankManifest(records=(_record(tmp_path, "a", [[[1.0]], [[0.0]]]),))
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    checkpoint = tmp_path / "selector.pt"
    output = tmp_path / "selected_desc.npz"
    _selector_checkpoint(checkpoint)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_selected_descriptor_bank",
            "--manifest",
            str(manifest_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--layer_name",
            "radio_final",
            "--device",
            "cpu",
            "--batch_size",
            "4",
            "--output",
            str(output),
        ],
        check=True,
    )

    bank = TokenDescriptorBank.from_npz(output)
    assert bank.image_ids == ("a",)
    assert bank.metadata["selector_checkpoint"] == str(checkpoint)
    assert bank.metadata["batch_size"] == 4
    np.testing.assert_allclose(bank.descriptors, np.asarray([[1.0, 0.0]], dtype=np.float32))
