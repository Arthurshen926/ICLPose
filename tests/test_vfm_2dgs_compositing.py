import numpy as np

from feature_extract.vfm.vfm_2dgs_mapping import _composite_sorted_packed_hits


def _reference(packed):
    pixels, rows, weights = [], [], []
    current_pixel, transmittance = -1, 1.0
    for pixel_f, _depth_f, row_f, alpha_f in packed.tolist():
        pixel = int(pixel_f)
        if pixel != current_pixel:
            current_pixel, transmittance = pixel, 1.0
        alpha = float(np.clip(alpha_f, 0.0, 0.999))
        weight = transmittance * alpha
        if weight > 1e-12:
            pixels.append(pixel)
            rows.append(int(row_f))
            weights.append(weight)
        transmittance *= max(1.0 - alpha, 0.0)
        if transmittance <= 1e-4:
            transmittance = 0.0
    return np.asarray(pixels), np.asarray(rows), np.asarray(weights, dtype=np.float32)


def test_vectorized_compositing_matches_reference_early_stop():
    rng = np.random.default_rng(19)
    packed = []
    for pixel, count in enumerate((1, 4, 20, 100)):
        depth = np.sort(rng.uniform(0.1, 10.0, size=count))
        alpha = rng.uniform(0.001, 0.999, size=count)
        for index in range(count):
            packed.append([pixel, depth[index], 1000 * pixel + index, alpha[index]])
    packed = np.asarray(packed, dtype=np.float64)
    expected = _reference(packed)
    actual = _composite_sorted_packed_hits(packed)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], rtol=2e-6, atol=1e-8)
