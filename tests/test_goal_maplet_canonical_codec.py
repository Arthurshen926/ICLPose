import numpy as np

from feature_extract.vfm.localization_goal_maplet.canonical_codec import CanonicalRadioCodec


def test_canonical_codec_map_matches_row_transform():
    codec = CanonicalRadioCodec(
        mean=np.zeros((4,), dtype=np.float32),
        components=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        metadata={"artifact_type": "goal_maplet_canonical_radio_codec_v1"},
    )
    feature = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + 1.0
    mapped = codec.transform_map(feature)
    rows = feature.transpose(1, 2, 0).reshape(-1, 4)
    expected = codec.transform_rows(rows).reshape(2, 3, 2).transpose(2, 0, 1)
    np.testing.assert_allclose(mapped, expected, atol=1e-6)
