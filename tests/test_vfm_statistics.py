import pytest

from feature_extract.vfm.statistics import (
    mcnemar_exact_pvalue,
    paired_bootstrap_delta_ci,
    wilcoxon_signed_rank,
)


def test_paired_bootstrap_delta_ci_has_expected_direction():
    low_cost = [0.1, 0.2, 0.1, 0.3]
    high_cost = [0.4, 0.5, 0.6, 0.7]

    estimate, lo, hi = paired_bootstrap_delta_ci(low_cost, high_cost, resamples=500, seed=0)

    assert estimate < 0.0
    assert lo < estimate < hi
    assert hi < 0.0


def test_mcnemar_exact_pvalue_detects_asymmetric_discordance():
    pvalue = mcnemar_exact_pvalue(
        baseline_success=[False, False, False, False, True],
        method_success=[True, True, True, True, True],
    )

    assert pvalue < 0.25


def test_wilcoxon_signed_rank_reports_signed_direction():
    result = wilcoxon_signed_rank([0.1, 0.2, 0.1], [0.3, 0.4, 0.5])

    assert result.signed_rank < 0.0
    assert result.nonzero_count == 3
    assert result.normal_approx_z == pytest.approx(-1.603567, rel=1e-5)
