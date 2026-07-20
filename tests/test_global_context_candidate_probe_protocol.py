from __future__ import annotations

from feature_extract.tools.vfm.audit_multiscale_candidate_probe import (
    _allowed_soft_global_context as audit_allows_soft_global_context,
)
from feature_extract.tools.vfm.fit_multiscale_candidate_probe import (
    _allowed_soft_global_context as fit_allows_soft_global_context,
)


def _manifest() -> dict[str, object]:
    return {
        "format": "global_context_candidate_probe_features_v1",
        "whole_image_summary_or_global_used": True,
        "soft_global_context_factor": True,
        "global_context_usage": "fixed_candidate_support_view_soft_radio_final_global_factor_v1",
        "global_context_hard_retrieval_or_candidate_reselection": False,
    }


def test_fit_and_audit_accept_the_same_narrow_soft_global_manifest() -> None:
    manifest = _manifest()
    assert fit_allows_soft_global_context(manifest) is True
    assert audit_allows_soft_global_context(manifest) is True

    support8 = {
        **manifest,
        "format": "global_context_support8_candidate_probe_features_v1",
        "global_context_usage": "fixed_maplet_support8_view_soft_radio_final_global_factor_v1",
    }
    assert fit_allows_soft_global_context(support8) is True
    assert audit_allows_soft_global_context(support8) is True


def test_soft_global_manifest_rejects_hard_reselection_or_untyped_global_use() -> None:
    hard = _manifest()
    hard["global_context_hard_retrieval_or_candidate_reselection"] = True
    assert fit_allows_soft_global_context(hard) is False
    assert audit_allows_soft_global_context(hard) is False

    untyped = _manifest()
    untyped["global_context_usage"] = "whole_image_retrieval"
    assert fit_allows_soft_global_context(untyped) is False
    assert audit_allows_soft_global_context(untyped) is False
