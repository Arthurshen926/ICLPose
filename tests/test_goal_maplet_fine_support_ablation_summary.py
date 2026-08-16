import pytest

from feature_extract.tools.vfm.summarize_goal_maplet_fine_support_ablation import (
    _bootstrap_interval,
    _paired_bootstrap_delta,
)


def test_block_bootstrap_is_deterministic_and_sequence_aware():
    rows = []
    for sequence, values in {"seq1": [0.1, 0.2, 0.3], "seq2": [0.7, 0.8]}.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "image_id": f"{sequence}/frame{index:05d}.png",
                    "surface_sets": {
                        "child_returned_set": {"exact_visible_mass_recall": value}
                    },
                }
            )
    left = _bootstrap_interval(
        rows, "exact_visible_mass_recall", block_length=2,
        repetitions=100, seed=9,
    )
    right = _bootstrap_interval(
        rows, "exact_visible_mass_recall", block_length=2,
        repetitions=100, seed=9,
    )
    assert left == right
    assert left["mean"] == pytest.approx(0.42)
    assert left["block_bootstrap_95_low"] <= left["mean"]
    assert left["block_bootstrap_95_high"] >= left["mean"]


def test_paired_bootstrap_uses_query_aligned_differences():
    baseline = []
    variant = []
    for index, value in enumerate([0.1, 0.2, 0.3, 0.4]):
        image_id = f"seq1/frame{index:05d}.png"
        baseline.append({
            "image_id": image_id,
            "surface_sets": {"child_returned_set": {"metric": value}},
        })
        variant.append({
            "image_id": image_id,
            "surface_sets": {"child_returned_set": {"metric": value + 0.05}},
        })
    result = _paired_bootstrap_delta(
        baseline, variant, "metric", block_length=2, repetitions=50, seed=3,
    )
    assert result["mean"] == pytest.approx(0.05)
    assert result["block_bootstrap_95_low"] == pytest.approx(0.05)
    assert result["block_bootstrap_95_high"] == pytest.approx(0.05)
