import numpy as np

from feature_extract.vfm.localization_goal_maplet.latent_selector import (
    FEATURE_NAMES,
    LatentSafetySelectorArtifact,
)


class _ProbabilityEstimator:
    def __init__(self, column):
        self.column = np.asarray(column, dtype=np.float64)

    def predict_proba(self, value):
        probability = self.column[: np.asarray(value).shape[0]]
        return np.stack([1.0 - probability, probability], axis=1)


def test_latent_safety_selector_is_bounded_residual_with_optional_veto():
    artifact = LatentSafetySelectorArtifact(
        _ProbabilityEstimator([0.1, 0.9, 0.8]),
        _ProbabilityEstimator([0.9, 0.1, 0.8]),
        {
            "artifact_type": "goal_maplet_latent_safety_selector_v1",
            "feature_names": list(FEATURE_NAMES),
            "residual_scale": 0.25,
            "catastrophic_veto_probability": 0.5,
        },
    )
    feature = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    base = np.asarray([0.50, 0.49, 0.48])
    residual, _, catastrophic = artifact.score_candidates(feature, base, safety_veto=False)
    safety, _, _ = artifact.score_candidates(feature, base, safety_veto=True)
    assert np.max(np.abs(residual - base)) <= 0.25 * np.subtract(*np.percentile(base, [75, 25]))
    assert catastrophic[0] > 0.5
    assert not np.isfinite(safety[0])
    assert int(np.argmax(safety)) == 1
