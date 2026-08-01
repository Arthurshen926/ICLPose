from types import SimpleNamespace

import numpy as np
import torch

from feature_extract.tools.vfm.build_surface_retrieval_maplet_bank import (
    _full_map_mapped_view_descriptors,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    SurfaceMapletMapper,
    SurfaceMapletMapperConfig,
    pool_radio_final_context_torch,
)
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
)


def test_mapping_descriptors_project_full_map_before_context_pooling(
    tmp_path,
) -> None:
    torch.manual_seed(7)
    raw = np.asarray(
        [
            [[1.0, 0.0, 2.0], [0.5, 3.0, -1.0], [2.0, 1.0, 0.0]],
            [[0.0, 2.0, 1.0], [4.0, -0.5, 2.0], [1.0, 3.0, 0.5]],
        ],
        dtype=np.float32,
    )
    token_path = tmp_path / "tokens.npz"
    np.savez_compressed(token_path, radio_final=raw)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="seq1/frame.png",
                token_path=token_path,
                layers=(
                    TokenLayerSpec(
                        name="radio_final",
                        model="radio",
                        layer="final",
                        channels=2,
                        stride=16,
                    ),
                ),
                split="mapping",
                scene="scene",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    model = SurfaceMapletMapper(
        SurfaceMapletMapperConfig(
            input_dim=2,
            hidden_dim=3,
            output_dim=2,
        )
    ).eval()
    mapper = SimpleNamespace(model=model)
    source = SimpleNamespace(
        view_image_ids=("seq1/frame.png",),
        view_token_xy=np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    metadata = {"pool_sizes": [3], "pool_weights": [1.0]}

    observed = _full_map_mapped_view_descriptors(
        source,
        mapper,
        metadata,
        manifest_path,
        layer_name="radio_final",
        selected_rows=np.asarray([0]),
        device="cpu",
    )

    with torch.no_grad():
        mapped = model(torch.from_numpy(raw)[None])[0]
        expected = pool_radio_final_context_torch(
            mapped,
            torch.asarray([[1.0, 1.0]]),
            pool_sizes=(3,),
            pool_weights=(1.0,),
        ).numpy()
        pooled_raw = pool_radio_final_context_torch(
            torch.from_numpy(raw),
            torch.asarray([[1.0, 1.0]]),
            pool_sizes=(3,),
            pool_weights=(1.0,),
        )
        post_pool_baseline = model(
            pooled_raw[:, :, None, None]
        )[:, :, 0, 0].numpy()

    np.testing.assert_allclose(observed, expected, atol=1e-6)
    assert not np.allclose(observed, post_pool_baseline, atol=1e-4)
