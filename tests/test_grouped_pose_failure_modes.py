import numpy as np

from feature_extract.tools.vfm.audit_grouped_pose_failure_modes import (
    _classify,
    _coherent_shift,
)


def test_coherent_shift_requires_repeated_consistent_3d_displacement() -> None:
    coherent, norm, dispersion = _coherent_shift(
        np.asarray([[0.5, 0.0, 0.0], [0.52, 0.01, 0.0], [0.48, -0.01, 0.0]]),
        minimum_pairs=3,
        minimum_norm_m=0.1,
        maximum_dispersion_m=0.2,
        maximum_relative_dispersion=0.35,
    )

    assert coherent
    assert np.isclose(norm, 0.5)
    assert dispersion < 0.03


def test_failure_class_prioritizes_missing_proposals_before_identity_modes() -> None:
    category = _classify(
        translation_error_m=0.3,
        success_threshold_m=0.1,
        available_fraction=0.25,
        minimum_available_fraction=0.5,
        identity_fraction=0.75,
        minimum_identity_fraction=0.5,
        degenerate=True,
        coherent_shift=True,
        maplet_mismatch_fraction=1.0,
    )

    assert category == "E_correct_candidate_absent_from_sample_pool"
