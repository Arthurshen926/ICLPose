import json
import subprocess
import sys

import numpy as np
import pytest

from feature_extract.vfm.cambridge_pose_lattice import (
    build_init_pose_lattice_bank,
    build_cambridge_pose_lattice_bank,
    build_cambridge_reference_pose_neighbor_bank,
    build_virtual_reference_pose_grid_records,
    build_virtual_reference_pose_records,
    fine_lattice_world_offsets,
    parse_cambridge_pose_file,
    q_level_world_offsets,
    write_cambridge_pose_file,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind


def test_parse_cambridge_pose_file_converts_camera_center_to_world_to_camera(tmp_path):
    poses = tmp_path / "dataset_test.txt"
    poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 1.0 2.0 3.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    records = parse_cambridge_pose_file(poses)

    assert len(records) == 1
    assert records[0].image_id == "seq/frame0001.png"
    np.testing.assert_allclose(
        np.asarray(records[0].pose_w2c),
        np.asarray(
            [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def test_build_cambridge_pose_lattice_bank_marks_gt_centered_candidates(tmp_path):
    poses = tmp_path / "dataset_test.txt"
    poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 1.0 2.0 3.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    bank = build_cambridge_pose_lattice_bank(
        pose_file=poses,
        protocol_name="synthetic_gt_lattice",
        offsets=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
    )

    assert bank.protocol_kind == ProtocolKind.CONTROLLED_LATTICE
    assert len(bank.candidates) == 2
    assert bank.candidates[0].query_id == "seq/frame0001.png"
    assert bank.candidates[0].candidate_type == "rendered_pose_lattice"
    assert bank.candidates[0].pose_error.translation_m == pytest.approx(0.0)
    assert bank.candidates[1].pose_error.translation_m == pytest.approx(0.25)
    assert bank.candidates[1].metadata["candidate_uses_gt"] is True
    assert bank.candidates[1].metadata["offset_world_m"] == [0.25, 0.0, 0.0]


def test_build_cambridge_pose_lattice_cli_writes_jsonl(tmp_path):
    poses = tmp_path / "dataset_test.txt"
    output = tmp_path / "bank.jsonl"
    poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_cambridge_pose_lattice",
            "--pose_file",
            str(poses),
            "--protocol_name",
            "synthetic_gt_lattice",
            "--offsets",
            "0,0,0;0.5,0,0",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "controlled_lattice"
    assert len(lines) == 3
    assert lines[2]["pose_error"]["translation_m"] == pytest.approx(0.5)


def test_build_cambridge_reference_pose_neighbor_bank_excludes_self_match(tmp_path):
    poses = tmp_path / "dataset_train.txt"
    poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "seq/frame0002.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "seq/frame0003.png 3.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    bank = build_cambridge_reference_pose_neighbor_bank(
        query_pose_file=poses,
        reference_pose_file=poses,
        protocol_name="train_pose_neighbors",
        top_k=2,
        exclude_same_image=True,
    )

    first_query = [candidate for candidate in bank.candidates if candidate.query_id == "seq/frame0001.png"]
    assert bank.protocol_kind == ProtocolKind.REFERENCE_POSE
    assert len(first_query) == 2
    assert first_query[0].reference_image == "seq/frame0002.png"
    assert first_query[0].pose_error.translation_m == pytest.approx(1.0)
    assert first_query[0].metadata["candidate_uses_gt"] is True
    assert first_query[0].metadata["candidate_generator"] == "pose_nearest_reference"
    assert first_query[1].reference_image == "seq/frame0003.png"


def test_build_cambridge_reference_pose_neighbor_cli_writes_jsonl(tmp_path):
    poses = tmp_path / "dataset_train.txt"
    output = tmp_path / "bank.jsonl"
    poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "seq/frame0002.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_cambridge_reference_pose_neighbors",
            "--query_pose_file",
            str(poses),
            "--reference_pose_file",
            str(poses),
            "--protocol_name",
            "train_pose_neighbors",
            "--top_k",
            "1",
            "--exclude_same_image",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "reference_pose"
    assert len(lines) == 3
    assert lines[1]["query_id"] == "seq/frame0001.png"
    assert lines[1]["reference_image"] == "seq/frame0002.png"


def test_virtual_reference_pose_records_expand_reference_database_without_query_gt(tmp_path):
    references = tmp_path / "dataset_train.txt"
    queries = tmp_path / "dataset_test.txt"
    virtual_pose_file = tmp_path / "virtual_dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    queries.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq8/frame0001.png 0.5 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    virtual_records = build_virtual_reference_pose_records(
        reference_pose_file=references,
        offsets=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        yaw_offsets_deg=(0.0,),
        image_prefix="virtual_oldhospital",
    )
    write_cambridge_pose_file(virtual_records, virtual_pose_file)
    real_bank = build_cambridge_reference_pose_neighbor_bank(
        query_pose_file=queries,
        reference_pose_file=references,
        protocol_name="real_reference_oracle",
        top_k=1,
    )
    virtual_bank = build_cambridge_reference_pose_neighbor_bank(
        query_pose_file=queries,
        reference_pose_file=virtual_pose_file,
        protocol_name="virtual_reference_oracle",
        top_k=1,
    )

    assert [record.image_id for record in virtual_records] == [
        "virtual_oldhospital/seq1__frame0001/t000_y000.png",
        "virtual_oldhospital/seq1__frame0001/t001_y000.png",
    ]
    assert real_bank.candidates[0].pose_error.translation_m == pytest.approx(0.5)
    assert virtual_bank.candidates[0].reference_image == "virtual_oldhospital/seq1__frame0001/t001_y000.png"
    assert virtual_bank.candidates[0].pose_error.translation_m == pytest.approx(0.0)


def test_build_virtual_reference_pose_database_cli_writes_cambridge_pose_file(tmp_path):
    references = tmp_path / "dataset_train.txt"
    virtual_pose_file = tmp_path / "virtual_dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 1.0 2.0 3.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_virtual_reference_pose_database",
            "--reference_pose_file",
            str(references),
            "--offsets",
            "0,0,0;0.25,0,0",
            "--yaw_offsets_deg",
            "0,10",
            "--image_prefix",
            "virtual_oldhospital",
            "--output_pose_file",
            str(virtual_pose_file),
        ],
        check=True,
    )

    records = parse_cambridge_pose_file(virtual_pose_file)
    assert len(records) == 4
    assert records[0].image_id == "virtual_oldhospital/seq1__frame0001/t000_y000.png"
    assert records[-1].image_id == "virtual_oldhospital/seq1__frame0001/t001_y001.png"
    np.testing.assert_allclose(records[0].camera_center, [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(records[2].camera_center, [1.25, 2.0, 3.0], atol=1e-6)


def test_virtual_reference_pose_grid_records_fill_between_sparse_references(tmp_path):
    references = tmp_path / "dataset_train.txt"
    queries = tmp_path / "dataset_test.txt"
    virtual_pose_file = tmp_path / "virtual_grid_dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "seq1/frame0002.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    queries.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq8/frame0001.png 0.5 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    virtual_records = build_virtual_reference_pose_grid_records(
        reference_pose_file=references,
        grid_step_m=0.5,
        yaw_offsets_deg=(0.0,),
        image_prefix="virtual_grid",
    )
    write_cambridge_pose_file(virtual_records, virtual_pose_file)
    virtual_bank = build_cambridge_reference_pose_neighbor_bank(
        query_pose_file=queries,
        reference_pose_file=virtual_pose_file,
        protocol_name="virtual_grid_reference_oracle",
        top_k=1,
    )

    assert [record.image_id for record in virtual_records] == [
        "virtual_grid/x000_z000_y000.png",
        "virtual_grid/x001_z000_y000.png",
        "virtual_grid/x002_z000_y000.png",
    ]
    np.testing.assert_allclose(virtual_records[1].camera_center, [0.5, 0.0, 0.0], atol=1e-6)
    assert virtual_bank.candidates[0].reference_image == "virtual_grid/x001_z000_y000.png"
    assert virtual_bank.candidates[0].pose_error.translation_m == pytest.approx(0.0)


def test_virtual_reference_pose_grid_records_can_interpolate_height(tmp_path):
    references = tmp_path / "dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "seq1/frame0002.png 1.0 2.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    virtual_records = build_virtual_reference_pose_grid_records(
        reference_pose_file=references,
        grid_step_m=0.5,
        yaw_offsets_deg=(0.0,),
        image_prefix="virtual_grid",
        height_mode="idw",
        height_knn=2,
    )

    np.testing.assert_allclose(virtual_records[1].camera_center, [0.5, 1.0, 0.0], atol=1e-6)


def test_virtual_reference_pose_grid_records_can_add_height_offsets(tmp_path):
    references = tmp_path / "dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 0.0 1.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    virtual_records = build_virtual_reference_pose_grid_records(
        reference_pose_file=references,
        grid_step_m=0.5,
        yaw_offsets_deg=(0.0,),
        image_prefix="virtual_grid",
        height_offsets_m=(-0.25, 0.0, 0.25),
    )

    assert [record.image_id for record in virtual_records] == [
        "virtual_grid/x000_z000_h000_y000.png",
        "virtual_grid/x000_z000_h001_y000.png",
        "virtual_grid/x000_z000_h002_y000.png",
    ]
    np.testing.assert_allclose(virtual_records[0].camera_center, [0.0, 0.75, 0.0], atol=1e-6)
    np.testing.assert_allclose(virtual_records[1].camera_center, [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(virtual_records[2].camera_center, [0.0, 1.25, 0.0], atol=1e-6)


def test_virtual_reference_pose_grid_records_can_add_orientation_knn(tmp_path):
    references = tmp_path / "dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 0.0 1.0 0.0 1.0 0.0 0.0 0.0",
                "seq1/frame0002.png 0.0 1.0 0.0 0.70710678 0.0 0.70710678 0.0",
            ]
        )
        + "\n"
    )

    virtual_records = build_virtual_reference_pose_grid_records(
        reference_pose_file=references,
        grid_step_m=0.5,
        yaw_offsets_deg=(0.0,),
        image_prefix="virtual_grid",
        orientation_knn=2,
    )

    assert [record.image_id for record in virtual_records] == [
        "virtual_grid/x000_z000_o000_y000.png",
        "virtual_grid/x000_z000_o001_y000.png",
    ]
    assert not np.allclose(virtual_records[0].rotation_w2c, virtual_records[1].rotation_w2c)


def test_build_virtual_reference_pose_database_cli_writes_grid_pose_file(tmp_path):
    references = tmp_path / "dataset_train.txt"
    virtual_pose_file = tmp_path / "virtual_grid_dataset_train.txt"
    references.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq1/frame0001.png 0.0 1.0 0.0 1.0 0.0 0.0 0.0",
                "seq1/frame0002.png 1.0 2.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_virtual_reference_pose_database",
            "--reference_pose_file",
            str(references),
            "--mode",
            "grid",
            "--grid_step_m",
            "0.5",
            "--grid_height_mode",
            "idw",
            "--grid_height_knn",
            "2",
            "--grid_orientation_knn",
            "2",
            "--yaw_offsets_deg",
            "0,10",
            "--image_prefix",
            "virtual_grid",
            "--output_pose_file",
            str(virtual_pose_file),
        ],
        check=True,
    )

    records = parse_cambridge_pose_file(virtual_pose_file)
    assert len(records) == 12
    assert records[0].image_id == "virtual_grid/x000_z000_o000_y000.png"
    assert records[1].image_id == "virtual_grid/x000_z000_o000_y001.png"
    np.testing.assert_allclose(records[4].camera_center, [0.5, 1.5, 0.0], atol=1e-6)


def test_build_init_pose_lattice_bank_uses_init_pose_and_gt_only_for_labels(tmp_path):
    gt_poses = tmp_path / "dataset_test.txt"
    gt_poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    init_pose = np.eye(4, dtype=np.float64)
    init_pose[:3, 3] = np.asarray([-0.2, 0.0, 0.0], dtype=np.float64)
    init_bank = CandidateHypothesisBank.from_candidates(
        protocol_name="retrieval_init",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="seq/frame0001.png",
                candidate_id="frame0001:retrieval:000",
                candidate_type="real_retrieval",
                pose=init_pose.tolist(),
                reference_image="seq/reference.png",
                prior_score=10.0,
                metadata={"retrieval_rank": 1},
            )
        ],
    )

    bank = build_init_pose_lattice_bank(
        init_bank=init_bank,
        gt_pose_file=gt_poses,
        protocol_name="init_centered_lattice",
        offsets=((0.0, 0.0, 0.0), (-0.2, 0.0, 0.0)),
        max_inits_per_query=1,
    )

    assert bank.protocol_kind == ProtocolKind.RENDERED_POSE
    assert len(bank.candidates) == 2
    assert bank.candidates[0].query_id == "seq/frame0001.png"
    assert bank.candidates[0].candidate_type == "rendered_pose_lattice"
    assert bank.candidates[0].pose_error.translation_m == pytest.approx(0.2)
    assert bank.candidates[1].pose_error.translation_m == pytest.approx(0.0)
    assert bank.candidates[0].reference_image == "seq/reference.png"
    assert bank.candidates[0].metadata["candidate_uses_gt"] is False
    assert bank.candidates[0].metadata["source_candidate_id"] == "frame0001:retrieval:000"
    assert bank.candidates[1].metadata["offset_world_m"] == [-0.2, 0.0, 0.0]


def test_fine_lattice_world_offsets_builds_local_xz_grid_with_height_offsets() -> None:
    offsets = fine_lattice_world_offsets(radius_m=0.1, step_m=0.1, height_offsets_m=(-0.05, 0.0))

    assert (0.0, -0.05, 0.0) in offsets
    assert (0.1, 0.0, 0.0) in offsets
    assert (-0.1, 0.0, 0.0) in offsets
    assert (0.0, 0.0, 0.1) in offsets
    assert (0.1, 0.0, 0.1) not in offsets


def test_build_init_pose_lattice_bank_can_add_yaw_offsets(tmp_path):
    gt_poses = tmp_path / "dataset_test.txt"
    gt_poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 0.996194698 0.0 0.087155743 0.0",
            ]
        )
        + "\n"
    )
    init_bank = CandidateHypothesisBank.from_candidates(
        protocol_name="stage1_vpr",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="seq/frame0001.png",
                candidate_id="init0",
                candidate_type="descriptor_retrieval_reference_pose",
                pose=np.eye(4, dtype=np.float64).tolist(),
                pose_error=PoseCost(translation_m=0.0, rotation_deg=10.0),
                reference_image="stage1/r0.png",
                metadata={"retrieval_rank": 1},
            )
        ],
    )

    bank = build_init_pose_lattice_bank(
        init_bank=init_bank,
        gt_pose_file=gt_poses,
        protocol_name="stage2_fine_lattice",
        offsets=((0.0, 0.0, 0.0),),
        max_inits_per_query=1,
        yaw_offsets_deg=(-10.0, 0.0, 10.0),
    )

    assert len(bank.candidates) == 3
    assert [candidate.metadata["yaw_offset_deg"] for candidate in bank.candidates] == [-10.0, 0.0, 10.0]
    assert bank.candidates[0].pose_error.rotation_deg == pytest.approx(0.0, abs=1e-3)
    assert bank.candidates[0].candidate_id == "seq__frame0001:init_lattice:000:000:000"


def test_build_init_pose_lattice_bank_can_add_free_orientation_offsets(tmp_path):
    gt_poses = tmp_path / "dataset_test.txt"
    gt_poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 0.996194698 0.087155743 0.0 0.0",
            ]
        )
        + "\n"
    )
    init_bank = CandidateHypothesisBank.from_candidates(
        protocol_name="stage1_vpr",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="seq/frame0001.png",
                candidate_id="init0",
                candidate_type="descriptor_retrieval_reference_pose",
                pose=np.eye(4, dtype=np.float64).tolist(),
                pose_error=PoseCost(translation_m=0.0, rotation_deg=10.0),
                reference_image="stage1/r0.png",
                metadata={"retrieval_rank": 1},
            )
        ],
    )

    bank = build_init_pose_lattice_bank(
        init_bank=init_bank,
        gt_pose_file=gt_poses,
        protocol_name="stage2_free_orientation_lattice",
        offsets=((0.0, 0.0, 0.0),),
        max_inits_per_query=1,
        yaw_offsets_deg=(0.0,),
        pitch_offsets_deg=(-10.0, 0.0, 10.0),
        roll_offsets_deg=(-5.0, 0.0, 5.0),
    )

    assert len(bank.candidates) == 9
    best = min(bank.candidates, key=lambda candidate: candidate.pose_error.rotation_deg)
    assert best.pose_error.rotation_deg == pytest.approx(0.0, abs=1e-3)
    assert best.metadata["pitch_offset_deg"] == -10.0
    assert best.metadata["roll_offset_deg"] == 0.0
    assert best.metadata["orientation_lattice_kind"] == "world_yaw_pitch_roll"


def test_init_pose_lattice_candidate_ids_are_unique_across_sequences(tmp_path):
    gt_poses = tmp_path / "dataset_test.txt"
    gt_poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq4/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "seq8/frame0001.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    init_bank = CandidateHypothesisBank.from_candidates(
        protocol_name="retrieval_init",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="seq4/frame0001.png",
                candidate_id="seq4_init",
                candidate_type="real_retrieval",
                pose=np.eye(4, dtype=np.float64).tolist(),
            ),
            CandidateHypothesis(
                query_id="seq8/frame0001.png",
                candidate_id="seq8_init",
                candidate_type="real_retrieval",
                pose=np.asarray(
                    [
                        [1.0, 0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ).tolist(),
            ),
        ],
    )

    bank = build_init_pose_lattice_bank(
        init_bank=init_bank,
        gt_pose_file=gt_poses,
        protocol_name="init_centered_lattice",
        offsets=((0.0, 0.0, 0.0),),
        max_inits_per_query=1,
    )

    candidate_ids = [candidate.candidate_id for candidate in bank.candidates]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert candidate_ids == [
        "seq4__frame0001:init_lattice:000:000",
        "seq8__frame0001:init_lattice:000:000",
    ]


def test_build_init_pose_lattice_cli_writes_rendered_pose_bank(tmp_path):
    gt_poses = tmp_path / "dataset_test.txt"
    init_path = tmp_path / "init.jsonl"
    output = tmp_path / "rendered_pose_bank.jsonl"
    gt_poses.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq/frame0001.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    init_pose = np.eye(4, dtype=np.float64)
    init_pose[:3, 3] = np.asarray([-0.1, 0.0, 0.0], dtype=np.float64)
    CandidateHypothesisBank.from_candidates(
        protocol_name="retrieval_init",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="seq/frame0001.png",
                candidate_id="frame0001:retrieval:000",
                candidate_type="real_retrieval",
                pose=init_pose.tolist(),
                pose_error=PoseCost(0.1, 0.0),
            )
        ],
    ).to_jsonl(init_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_init_pose_lattice",
            "--init_bank",
            str(init_path),
            "--gt_pose_file",
            str(gt_poses),
            "--protocol_name",
            "init_centered_lattice",
            "--offsets",
            "0,0,0;-0.1,0,0",
            "--yaw_offsets_deg=-5,0,5",
            "--pitch_offsets_deg=-2.5,0,2.5",
            "--roll_offsets_deg",
            "0",
            "--max_inits_per_query",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "rendered_pose"
    assert len(lines) == 19
    assert lines[1]["metadata"]["candidate_uses_gt"] is False
    assert lines[2]["metadata"]["pitch_offset_deg"] == 0.0
    assert lines[18]["pose_error"]["translation_m"] == pytest.approx(0.0)


def test_q_level_world_offsets_include_center_half_and_full_radius():
    offsets = q_level_world_offsets("q25")

    assert (0.0, 0.0, 0.0) in offsets
    assert (0.125, 0.0, 0.0) in offsets
    assert (-0.25, 0.0, 0.0) in offsets
    assert len(offsets) == 13
    assert max(float(np.linalg.norm(offset)) for offset in offsets) == pytest.approx(0.25)
