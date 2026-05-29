import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.make_linear_selector_ablation import ablate_transform, main
from feature_extract.vfm.feature_compression import FeatureCompressionTransform


def _transform() -> FeatureCompressionTransform:
    matrix = np.zeros((8, 2), dtype=np.float32)
    matrix[0:2] = 4.0
    matrix[2:4] = 1.0
    matrix[4:6] = 0.1
    matrix[6:8] = 2.0
    return FeatureCompressionTransform(
        method="learned_linear",
        input_dim=8,
        output_dim=2,
        mean=np.zeros((8,), dtype=np.float32),
        matrix=matrix,
        l2_normalize=True,
    )


def test_ablate_transform_zeroes_top_energy_group() -> None:
    ablated, summary = ablate_transform(
        _transform(),
        policy="top",
        group_size=2,
        remove_fraction=0.25,
        seed=0,
    )

    assert summary["removed_groups"] == [0]
    assert np.allclose(ablated.matrix[0:2], 0.0)
    assert not np.allclose(ablated.matrix[6:8], 0.0)


def test_make_linear_selector_ablation_cli_writes_transform(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "ablated.npz"
    summary = tmp_path / "summary.json"
    _transform().to_npz(source)

    main(
        [
            "--input_transform",
            str(source),
            "--policy",
            "bottom",
            "--group_size",
            "2",
            "--remove_fraction",
            "0.25",
            "--output_transform",
            str(output),
            "--summary_json",
            str(summary),
        ]
    )

    loaded = FeatureCompressionTransform.from_npz(output)
    report = json.loads(summary.read_text())
    assert report["policy"] == "bottom"
    assert report["removed_groups"] == [2]
    assert loaded.method == "learned_linear_ablate_bottom"
