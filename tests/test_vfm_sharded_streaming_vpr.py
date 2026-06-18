from __future__ import annotations

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    CambridgePoseRecord,
    pose_w2c_from_center_rotation,
    write_cambridge_pose_file,
)
from feature_extract.vfm.sharded_streaming_vpr import (
    StreamingVPRTopK,
    build_streaming_vpr_candidate_bank,
    iter_virtual_reference_grid_records,
)


def _record(image_id: str, center: tuple[float, float, float]) -> CambridgePoseRecord:
    rotation = np.eye(3, dtype=np.float64)
    camera_center = np.asarray(center, dtype=np.float64)
    return CambridgePoseRecord(
        image_id=image_id,
        camera_center=camera_center,
        rotation_w2c=rotation,
        pose_w2c=pose_w2c_from_center_rotation(camera_center, rotation),
    )


def test_streaming_vpr_topk_keeps_best_candidates_across_shards() -> None:
    topk = StreamingVPRTopK(
        query_ids=("q0.png", "q1.png"),
        query_descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        top_k=2,
    )

    topk.update(
        records=(_record("weak_for_q0.png", (2.0, 0.0, 0.0)), _record("best_for_q1.png", (1.0, 0.0, 0.0))),
        descriptors=np.asarray([[0.6, 0.4], [0.1, 1.0]], dtype=np.float32),
        start_ordinal=0,
    )
    topk.update(
        records=(_record("best_for_q0.png", (0.0, 0.0, 0.0)), _record("second_for_q1.png", (1.2, 0.0, 0.0))),
        descriptors=np.asarray([[1.0, 0.0], [0.2, 0.9]], dtype=np.float32),
        start_ordinal=2,
    )

    assert [item.record.image_id for item in topk.results_for_query("q0.png")] == [
        "best_for_q0.png",
        "weak_for_q0.png",
    ]
    assert [item.record.image_id for item in topk.results_for_query("q1.png")] == [
        "best_for_q1.png",
        "second_for_q1.png",
    ]


def test_streaming_vpr_candidate_bank_uses_gt_only_for_pose_labels(tmp_path) -> None:
    query_pose_file = tmp_path / "queries.txt"
    write_cambridge_pose_file(
        [
            _record("q0.png", (0.0, 0.0, 0.0)),
            _record("q1.png", (1.0, 0.0, 0.0)),
        ],
        query_pose_file,
    )
    topk = StreamingVPRTopK(
        query_ids=("q0.png", "q1.png"),
        query_descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        top_k=1,
    )
    topk.update(
        records=(_record("c0.png", (0.2, 0.0, 0.0)), _record("c1.png", (1.5, 0.0, 0.0))),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        start_ordinal=10,
    )

    bank = build_streaming_vpr_candidate_bank(
        topk,
        query_pose_file=query_pose_file,
        protocol_name="streaming_vpr",
        descriptor_pooling="mean",
    )

    assert len(bank.candidates) == 2
    assert bank.candidates[0].candidate_type == "sharded_streaming_virtual_vpr"
    assert bank.candidates[0].metadata["candidate_uses_gt"] is False
    assert bank.candidates[0].metadata["pose_label_uses_gt"] is True
    assert bank.candidates[0].pose_error.translation_m == 0.2
    assert bank.candidates[1].pose_error.translation_m == 0.5


def test_virtual_reference_grid_iterator_supports_start_and_max_records(tmp_path) -> None:
    reference_pose_file = tmp_path / "refs.txt"
    write_cambridge_pose_file([_record("ref.png", (0.0, 2.0, 0.0))], reference_pose_file)

    records = list(
        iter_virtual_reference_grid_records(
            reference_pose_file=reference_pose_file,
            grid_step_m=1.0,
            grid_margin_m=0.0,
            height_mode="nearest",
            height_knn=1,
            height_offsets_m=(0.0, 0.5),
            orientation_knn=1,
            yaw_offsets_deg=(-5.0, 0.0),
            image_prefix="virtual_grid",
            start_ordinal=1,
            max_records=2,
        )
    )

    assert len(records) == 2
    assert records[0].ordinal == 1
    assert records[0].record.image_id == "virtual_grid/x000_z000_h000_y001.png"
    assert records[1].record.image_id == "virtual_grid/x000_z000_h001_y000.png"
    np.testing.assert_allclose(records[1].record.camera_center, [0.0, 2.5, 0.0], atol=1e-6)
