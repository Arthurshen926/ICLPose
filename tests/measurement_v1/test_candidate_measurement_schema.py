from __future__ import annotations

from feature_extract.vfm.measurement_v1.candidate_measurement_schema import (
    CandidateIdentityKey,
    CandidateMeasurementCacheKey,
    CandidateScoreBundle,
    support_view_set_id,
    validate_unique_cache_keys,
)


def _identity(**overrides: object) -> CandidateIdentityKey:
    values: dict[str, object] = {
        "query_id": "seq1/frame00001.png",
        "source_query_row": 12,
        "track_id": 101,
        "prototype_id": 0,
        "support_view_set_id": support_view_set_id(("a.png", "b.png")),
    }
    values.update(overrides)
    return CandidateIdentityKey(**values)  # type: ignore[arg-type]


def test_candidate_identity_changes_for_every_semantic_axis() -> None:
    base = _identity()
    variants = (
        _identity(source_query_row=13),
        _identity(track_id=102),
        _identity(prototype_id=1),
        _identity(support_view_set_id=support_view_set_id(("a.png", "c.png"))),
        _identity(crop_geometry_version="different_crop_v2"),
    )
    assert len({base.digest, *(value.digest for value in variants)}) == len(variants) + 1


def test_measurement_cache_key_binds_support_image_and_checkpoint() -> None:
    identity = _identity()
    keys = (
        CandidateMeasurementCacheKey(identity, "a.png", "0123456789abcdef"),
        CandidateMeasurementCacheKey(identity, "b.png", "0123456789abcdef"),
        CandidateMeasurementCacheKey(identity, "a.png", "fedcba9876543210"),
    )
    assert len({key.digest for key in keys}) == 3
    validate_unique_cache_keys(keys)


def test_duplicate_measurement_cache_key_is_rejected() -> None:
    key = CandidateMeasurementCacheKey(_identity(), "a.png", "0123456789abcdef")
    try:
        validate_unique_cache_keys((key, key))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate cache key should fail")


def test_score_bundle_rejects_probability_similarity_confusion() -> None:
    CandidateScoreBundle(0.8, 0.4, 0.7, 0.6)
    try:
        CandidateScoreBundle(0.8, -0.1, 0.7, 0.6)
    except ValueError as exc:
        assert "assignment_probability" in str(exc)
    else:
        raise AssertionError("invalid probability should fail")

