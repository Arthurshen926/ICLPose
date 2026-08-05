import numpy as np

from feature_extract.vfm.localization_goal_maplet.detector_radio_refiner import sample_radio_at_pixels


def test_sample_radio_at_token_centers_is_exact():
    field = np.zeros((2, 2, 2), dtype=np.float32)
    field[:, 0, 0] = [3.0, 4.0]
    value = sample_radio_at_pixels(field, np.asarray([[0.0, 0.0]]), image_width=2, image_height=2)
    assert np.allclose(value[0], [0.6, 0.8])
