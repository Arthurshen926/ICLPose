from feature_extract.tools.vfm.build_goal_maplet_fold_colmap_dataset import (
    _chart_indices,
    _flattened_name,
    _proportional_subsample,
)


def test_fold_geometry_subsample_is_exact_route_balanced_and_deterministic():
    values = [f"seq1/frame{i:05d}.png" for i in range(8)] + [
        f"seq2/frame{i:05d}.png" for i in range(4)
    ]
    selected = _proportional_subsample(values, 6)
    assert len(selected) == len(set(selected)) == 6
    assert sum(value.startswith("seq1/") for value in selected) == 4
    assert sum(value.startswith("seq2/") for value in selected) == 2
    assert selected == _proportional_subsample(list(reversed(values)), 6)


def test_fold_geometry_image_names_are_flat_and_reversible():
    assert _flattened_name("seq12/frame00001.png") == "seq12__frame00001.png"


def test_chart_subset_indices_keep_all_dense_supervision_images():
    dataset = ["seq1/a.png", "seq1/b.png", "seq2/c.png", "seq2/d.png"]
    charts = ["seq1/b.png", "seq2/d.png"]
    assert _chart_indices(dataset, charts) == [1, 3]
    assert len(dataset) > len(charts)
