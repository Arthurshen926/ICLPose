from feature_extract.tools.vfm.plan_goal_maplet_route_balanced_mapping_views import (
    plan_nested_views,
)


def _records():
    return [
        {"image_id": f"{route}/frame{index:05d}.png"}
        for route, count in (("seq1", 12), ("seq2", 8), ("seq3", 10))
        for index in range(count)
    ]


def test_route_balanced_plan_is_nested_balanced_and_keeps_seed() -> None:
    seeds = ["seq1/frame00000.png", "seq1/frame00011.png", "seq2/frame00000.png"]
    selected, _ = plan_nested_views(
        _records(), routes=("seq1", "seq2", "seq3"), required_names=seeds,
        maximum_count=18,
    )
    assert selected[: len(seeds)] == seeds
    assert len(selected) == len(set(selected)) == 18
    for count in (9, 12, 18):
        prefix = selected[:count]
        route_counts = [sum(name.startswith(route + "/") for name in prefix) for route in ("seq1", "seq2", "seq3")]
        assert max(route_counts) - min(route_counts) <= 1
    assert set(selected[:9]).issubset(selected[:12])
    assert set(selected[:12]).issubset(selected[:18])


def test_route_progression_spreads_added_views() -> None:
    selected, _ = plan_nested_views(
        _records(), routes=("seq1", "seq2", "seq3"),
        required_names=["seq1/frame00000.png", "seq2/frame00000.png", "seq3/frame00000.png"],
        maximum_count=6,
    )
    assert "seq1/frame00011.png" in selected
    assert "seq2/frame00007.png" in selected
    assert "seq3/frame00009.png" in selected
