from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_complement import _summarize


def _row(name, inlier, candidate, translation, rotation, usable=True):
    return {
        "name": name,
        "usable": usable,
        "pnp_inlier_count": inlier,
        "candidate_correspondence_count": candidate,
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
    }


def test_complement_summary_separates_raw_recall_and_selective_precision():
    rows = [
        _row("a", 30, 100, 0.2, 1.0),
        _row("b", 10, 100, 9.0, 90.0),
        _row("c", 14, 100, 0.3, 2.0),
        _row("d", 0, 1, 0.0, 0.0, usable=False),
    ]
    result = _summarize(rows, 0.15)
    assert result["raw_recall_2m45"] == 0.5
    assert result["accepted_count"] == 1
    assert result["accepted_precision_2m45"] == 1.0
    assert result["selective_system_recall_2m45"] == 0.25
    assert result["rejected_good_2m45_count"] == 1
    assert result["rejected_bad_2m45_count"] == 1
