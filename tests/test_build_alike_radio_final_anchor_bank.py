from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_alike_radio_final_anchor_bank import main
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)


def test_radio_only_bank_stores_one_anonymous_prototype(
    tmp_path: Path,
) -> None:
    base = AnchorLocalDescriptorBank(
        anchor_ids=np.asarray([7], dtype=np.int64),
        descriptor_offsets=np.asarray([0, 2], dtype=np.int64),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        support_image_ids=("view_a", "view_b"),
        descriptor_quality=np.asarray([1.0, 1.0], dtype=np.float32),
        support_view_directions=np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        metadata={
            "representation": "feature_aligned_2dgs_anchor_alike",
            "local_feature": "alike_anchor",
        },
    )
    base_path = tmp_path / "base.npz"
    base.save_npz(base_path)
    replay = tmp_path / "replay"
    replay.mkdir()
    for image_id, descriptor in (
        ("view_a", [1.0, 0.0, 0.0]),
        ("view_b", [0.0, 1.0, 0.0]),
    ):
        np.savez_compressed(
            replay / f"{image_id}.npz",
            image_id=np.asarray(image_id),
            target_anchor_ids=np.asarray([7], dtype=np.int64),
            descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            vfm_descriptors=np.asarray([descriptor], dtype=np.float32),
            scores=np.asarray([1.0], dtype=np.float32),
        )
    output_path = tmp_path / "radio.npz"
    summary_path = tmp_path / "summary.json"
    main(
        [
            "--alike_descriptor_bank",
            str(base_path),
            "--augmented_replay_dir",
            str(replay),
            "--output_descriptor_bank",
            str(output_path),
            "--summary_json",
            str(summary_path),
            "--output_feature",
            "radio_final",
            "--collapse_radio_final_prototype",
        ]
    )
    output = AnchorLocalDescriptorBank.load_npz(output_path)
    assert output.descriptors.shape == (1, 3)
    np.testing.assert_allclose(
        output.descriptors,
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        / np.sqrt(2.0),
        atol=1e-6,
    )
    assert output.support_image_ids == (
        "__anonymous_radio_final_prototype__",
    )
    assert output.metadata["local_feature"] == (
        "radio_final_at_alike_detection"
    )
    assert output.metadata["alike_descriptor_used_for_identity"] is False
    assert output.metadata["stores_mapping_image_ids"] is False
