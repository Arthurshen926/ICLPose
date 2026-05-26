import json
import subprocess
import sys

import numpy as np
import pytest

from feature_extract.vfm.cambridge_pose_lattice import (
    build_init_pose_lattice_bank,
    build_cambridge_pose_lattice_bank,
    build_cambridge_reference_pose_neighbor_bank,
    parse_cambridge_pose_file,
    q_level_world_offsets,
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
            "--max_inits_per_query",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "rendered_pose"
    assert len(lines) == 3
    assert lines[1]["metadata"]["candidate_uses_gt"] is False
    assert lines[2]["pose_error"]["translation_m"] == pytest.approx(0.0)


def test_q_level_world_offsets_include_center_half_and_full_radius():
    offsets = q_level_world_offsets("q25")

    assert (0.0, 0.0, 0.0) in offsets
    assert (0.125, 0.0, 0.0) in offsets
    assert (-0.25, 0.0, 0.0) in offsets
    assert len(offsets) == 13
    assert max(float(np.linalg.norm(offset)) for offset in offsets) == pytest.approx(0.25)
