import json
import subprocess
import sys

import numpy as np
import pytest

from feature_extract.vfm.descriptor_retrieval_candidates import (
    build_descriptor_retrieval_reference_pose_bank,
)
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def _write_poses(path):
    path.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "q0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "r0.png 5.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "r1.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )


def test_descriptor_retrieval_bank_uses_descriptor_order_and_pose_labels(tmp_path):
    pose_file = tmp_path / "poses.txt"
    _write_poses(pose_file)
    query_descriptors = TokenDescriptorBank(
        image_ids=("q0.png",),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    )
    map_descriptors = TokenDescriptorBank(
        image_ids=("r0.png", "r1.png"),
        descriptors=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    )

    bank = build_descriptor_retrieval_reference_pose_bank(
        query_descriptors=query_descriptors,
        map_descriptors=map_descriptors,
        query_pose_file=pose_file,
        reference_pose_file=pose_file,
        protocol_name="descriptor_retrieval_train",
        top_k=2,
    )

    assert bank.protocol_kind == ProtocolKind.REFERENCE_POSE
    assert [candidate.reference_image for candidate in bank.candidates] == ["r0.png", "r1.png"]
    assert bank.candidates[0].prior_score == pytest.approx(0.9 / np.linalg.norm([0.9, 0.1]))
    assert bank.candidates[0].pose_error.translation_m == pytest.approx(5.0)
    assert bank.candidates[1].pose_error.translation_m == pytest.approx(1.0)
    assert bank.candidates[0].metadata["candidate_generator"] == "descriptor_retrieval"
    assert bank.candidates[0].metadata["candidate_uses_gt"] is False
    assert bank.candidates[0].metadata["pose_label_uses_gt"] is True


def test_descriptor_retrieval_bank_supports_chunked_topk_scoring(tmp_path):
    pose_file = tmp_path / "poses.txt"
    _write_poses(pose_file)
    query_descriptors = TokenDescriptorBank(
        image_ids=("q0.png",),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="vlad",
    )
    map_descriptors = TokenDescriptorBank(
        image_ids=("r0.png", "r1.png"),
        descriptors=np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
        layer_name="radio_final",
        pooling="vlad",
    )

    bank = build_descriptor_retrieval_reference_pose_bank(
        query_descriptors=query_descriptors,
        map_descriptors=map_descriptors,
        query_pose_file=pose_file,
        reference_pose_file=pose_file,
        protocol_name="descriptor_retrieval_train",
        top_k=2,
        score_block_size=1,
    )

    assert [candidate.reference_image for candidate in bank.candidates] == ["r1.png", "r0.png"]
    assert bank.candidates[0].metadata["descriptor_pooling"] == "vlad"
    assert bank.candidates[0].metadata["score_block_size"] == 1


def test_descriptor_retrieval_bank_cli_writes_jsonl(tmp_path):
    pose_file = tmp_path / "poses.txt"
    query_npz = tmp_path / "query.npz"
    map_npz = tmp_path / "map.npz"
    output = tmp_path / "bank.jsonl"
    _write_poses(pose_file)
    TokenDescriptorBank(
        image_ids=("q0.png",),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    ).to_npz(query_npz)
    TokenDescriptorBank(
        image_ids=("r0.png", "r1.png"),
        descriptors=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    ).to_npz(map_npz)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_descriptor_retrieval_bank",
            "--query_descriptors",
            str(query_npz),
            "--map_descriptors",
            str(map_npz),
            "--query_pose_file",
            str(pose_file),
            "--reference_pose_file",
            str(pose_file),
            "--protocol_name",
            "descriptor_retrieval_train",
            "--top_k",
            "1",
            "--score_block_size",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "reference_pose"
    assert lines[1]["reference_image"] == "r0.png"


def test_descriptor_retrieval_bank_can_filter_invalid_pose_centers(tmp_path):
    pose_file = tmp_path / "poses.txt"
    pose_file.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "q0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "q_bad.png 999999999.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "r0.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "r_bad.png 999999999.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    query_descriptors = TokenDescriptorBank(
        image_ids=("q0.png", "q_bad.png"),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    )
    map_descriptors = TokenDescriptorBank(
        image_ids=("r0.png", "r_bad.png"),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
    )

    bank = build_descriptor_retrieval_reference_pose_bank(
        query_descriptors=query_descriptors,
        map_descriptors=map_descriptors,
        query_pose_file=pose_file,
        reference_pose_file=pose_file,
        protocol_name="descriptor_retrieval_train",
        top_k=2,
        max_abs_pose_center=100.0,
    )

    assert [candidate.query_id for candidate in bank.candidates] == ["q0.png"]
    assert [candidate.reference_image for candidate in bank.candidates] == ["r0.png"]
