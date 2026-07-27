import numpy as np
import torch

from feature_extract.vfm.localization.highres_surface_metric_decoder import (
    HighresSurfaceMetricDecoder,
    HighresSurfaceMetricDecoderConfig,
    decode_highres_surface_metric,
    load_highres_surface_metric_decoder,
    save_highres_surface_metric_decoder,
)


def test_highres_decoder_shapes_and_contract_roundtrip(tmp_path):
    model = HighresSurfaceMetricDecoder(
        HighresSurfaceMetricDecoderConfig(
            radio_dim=6, hidden_dim=8, rgb_dim=8, output_dim=4
        )
    ).eval()
    radio = torch.randn(1, 6, 2, 3)
    rgb = torch.randn(1, 3, 32, 48)
    output = model(radio, rgb)
    assert output["coarse"].shape == (1, 4, 2, 3)
    assert output["middle"].shape == (1, 4, 4, 6)
    assert output["fine"].shape == (1, 4, 8, 12)
    assert output["matchability"].shape == (1, 1, 8, 12)

    path = tmp_path / "decoder.pt"
    metadata = {
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
    }
    save_highres_surface_metric_decoder(path, model, metadata)
    loaded, loaded_metadata = load_highres_surface_metric_decoder(path)
    decoded = decode_highres_surface_metric(
        loaded,
        radio[0].numpy(),
        np.zeros((32, 48, 3), dtype=np.float32),
        device="cpu",
    )
    assert decoded["fine"].shape == (4, 8, 12)
    assert loaded_metadata == metadata
