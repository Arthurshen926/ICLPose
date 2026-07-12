from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.descriptor_path_audit import (
    audit_full_map_observation_projection_parity,
)
from feature_extract.vfm.localization.schemas import MappedFeatureMap
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


class _IdentityMapper:
    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        values = np.asarray(feature_map, dtype=np.float32)
        return MappedFeatureMap(
            coarse_descriptors=values,
            measurement_context=values,
            offset_logits=np.zeros((65, *values.shape[1:]), dtype=np.float32),
            heatmap=np.ones(values.shape[1:], dtype=np.float32),
        )


def test_full_map_observation_projection_parity(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.npz"
    np.savez(token_path, radio_final=np.arange(24, dtype=np.float32).reshape(3, 2, 4))
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="image.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "c-radio_v4-h", "final", 3, 16),),
                split="train",
                scene="scene",
            ),
        )
    )
    observations = [
        ColmapTrackObservation(
            track_id=1,
            image_id="image.png",
            point2d_idx=0,
            xy=(2.5, 1.5),
            xyz=np.asarray([0.0, 0.0, 1.0]),
            track_length=2,
            reprojection_error=0.1,
            image_width=8,
            image_height=4,
        )
    ]

    result = audit_full_map_observation_projection_parity(
        observations,
        manifest,
        _IdentityMapper(),
        max_observations=1,
    )

    assert result["passed"] is True
    assert result["min_cosine"] > 0.99999
    assert result["max_abs_difference"] == 0.0
