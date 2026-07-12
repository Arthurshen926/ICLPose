from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_identity_geometry_fusions import (
    geometric_probability_fusion,
)


def test_geometric_probability_fusion_preserves_invalid_entries() -> None:
    fused = geometric_probability_fusion(
        np.asarray([[0.25, 1.0, -np.inf]], dtype=np.float32),
        np.asarray([[1.0, 0.25, 0.8]], dtype=np.float32),
    )

    np.testing.assert_allclose(fused[0, :2], [0.5, 0.5], atol=1e-7)
    assert fused[0, 2] == -np.inf


def test_geometric_probability_fusion_rejects_non_probabilities() -> None:
    with pytest.raises(ValueError, match="not probabilities"):
        geometric_probability_fusion(
            np.asarray([[1.2]], dtype=np.float32),
            np.asarray([[0.5]], dtype=np.float32),
        )
