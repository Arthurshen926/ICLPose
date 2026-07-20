from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_frozen_mast3r_rank_fusion import (
    _validate_alpha_zero_matches_baseline,
    choose_train_alpha,
    descending_ranks,
    parse_alpha_grid,
    rank_percentiles,
    score_selection,
    summarize_selection,
)


def _keys() -> tuple[tuple[str, str, str, int], ...]:
    return (
        ("train", "policy", "train.png", 0),
        ("train", "policy", "train.png", 1),
        ("validation", "policy", "validation.png", 0),
        ("validation", "policy", "validation.png", 1),
    )


def test_rank_percentiles_and_descending_ranks_have_explicit_tie_break() -> None:
    values = np.asarray([1.0, 1.0, 2.0])
    frozen_orders = np.asarray([9, 3, 7])

    assert np.allclose(rank_percentiles(values, frozen_orders), [0.25, 0.25, 1.0])
    assert descending_ranks(values, frozen_orders).tolist() == [3, 2, 1]


def test_parse_alpha_grid_requires_zero_and_unique_nonnegative_values() -> None:
    assert parse_alpha_grid("0.5,0,0.25") == (0.0, 0.25, 0.5)
    for bad in ("0.1", "0,0", "0,-0.1", "0,nan"):
        try:
            parse_alpha_grid(bad)
        except ValueError:
            pass
        else:  # pragma: no cover - keeps each invalid contract explicit
            raise AssertionError(f"expected invalid grid: {bad}")


def test_train_alpha_refuses_a_nonzero_weight_that_worsens_the_tail() -> None:
    keys = _keys()
    baseline = np.asarray([1.0, 0.0, 1.0, 0.0])
    mast3r = np.asarray([0.0, 1.0, 0.0, 1.0])
    # A sufficiently large optional weight selects a catastrophic alternative;
    # the train-only search must retain alpha zero despite that nonzero score.
    translation = np.asarray([0.20, 1.20, 0.20, 0.20])
    rotation = np.zeros((4,))
    rows = {
        alpha: score_selection(
            keys=keys,
            baseline_scores=baseline,
            mast3r_scores=mast3r,
            baseline_tie_break_orders=np.arange(4),
            translation_m=translation,
            rotation_deg=rotation,
            alpha=alpha,
        )
        for alpha in (0.0, 2.0)
    }

    chosen, report = choose_train_alpha(rows_by_alpha=rows, fit_split="train")

    assert chosen == 0.0
    assert report["grid"]["2.0"]["tail_safe_train_candidate"] is False


def test_train_alpha_selects_safe_nonzero_complementary_evidence() -> None:
    keys = _keys()
    baseline = np.asarray([0.0, 1.0, 0.0, 1.0])
    mast3r = np.asarray([1.0, 0.0, 1.0, 0.0])
    # Candidate zero is the train oracle and has the same safe pose residual.
    translation = np.asarray([0.10, 0.10, 0.10, 0.10])
    rotation = np.zeros((4,))
    rows = {
        alpha: score_selection(
            keys=keys,
            baseline_scores=baseline,
            mast3r_scores=mast3r,
            baseline_tie_break_orders=np.arange(4),
            translation_m=translation,
            rotation_deg=rotation,
            alpha=alpha,
        )
        for alpha in (0.0, 1.0)
    }

    chosen, report = choose_train_alpha(rows_by_alpha=rows, fit_split="train")

    assert chosen == 1.0
    assert report["chosen_alpha"] == 1.0
    assert summarize_selection(rows[1.0])['median_oracle_score_rank'] == 1.0


def test_alpha_zero_must_match_the_source_baseline_selection() -> None:
    keys = _keys()
    baseline = {
        "query_ids": np.asarray(["train.png", "train.png", "validation.png", "validation.png"]),
        "split_names": np.asarray(["train", "train", "validation", "validation"]),
        "evaluation_labels": np.asarray(["policy", "policy", "policy", "policy"]),
        "hypothesis_indices": np.asarray([0, 1, 0, 1]),
        "independent_score_top1": np.asarray([False, True, False, True]),
    }
    rows = score_selection(
        keys=keys,
        baseline_scores=np.asarray([0.0, 1.0, 0.0, 1.0]),
        mast3r_scores=np.asarray([1.0, 0.0, 1.0, 0.0]),
        baseline_tie_break_orders=np.arange(4),
        translation_m=np.asarray([0.1, 0.1, 0.1, 0.1]),
        rotation_deg=np.zeros((4,)),
        alpha=0.0,
    )
    _validate_alpha_zero_matches_baseline(
        baseline=baseline, keys=keys, alpha_zero_rows=rows
    )
    baseline["independent_score_top1"] = np.asarray([True, False, False, True])
    try:
        _validate_alpha_zero_matches_baseline(
            baseline=baseline, keys=keys, alpha_zero_rows=rows
        )
    except ValueError as error:
        assert "alpha zero" in str(error)
    else:  # pragma: no cover - documents the strict source-top1 invariant
        raise AssertionError("expected mismatched baseline selection to fail")
