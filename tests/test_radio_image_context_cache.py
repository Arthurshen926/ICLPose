import torch

from feature_extract.tools.vfm.build_radio_image_context_cache import (
    multiscale_context_descriptors,
)


def test_multiscale_context_shapes_and_normalization() -> None:
    generator = torch.Generator().manual_seed(3)
    summary = torch.randn(2, 8, generator=generator)
    local = torch.randn(2, 6, 5, 7, generator=generator)

    summary_out, global_out, grid2, grid4 = multiscale_context_descriptors(
        summary, local
    )

    assert summary_out.shape == (2, 8)
    assert global_out.shape == (2, 6)
    assert grid2.shape == (2, 4, 6)
    assert grid4.shape == (2, 16, 6)
    for values in (summary_out, global_out, grid2, grid4):
        torch.testing.assert_close(
            torch.linalg.vector_norm(values, dim=-1),
            torch.ones(values.shape[:-1]),
        )
