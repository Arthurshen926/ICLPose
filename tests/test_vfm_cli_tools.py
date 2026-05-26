import json
import subprocess
import sys

import numpy as np


def test_manifest_existing_tokens_cli(tmp_path):
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    np.savez_compressed(token_dir / "q0.npz", radio_final=np.ones((4, 2, 2), dtype=np.float32))
    layer_spec = tmp_path / "layers.json"
    layer_spec.write_text(
        json.dumps(
            [
                {
                    "name": "radio_final",
                    "model": "c-radio_v4-h",
                    "layer": "final",
                    "channels": 4,
                    "stride": 16,
                }
            ]
        )
    )
    output = tmp_path / "manifest.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.extract_tokens",
            "--existing_npz_dir",
            str(token_dir),
            "--layer_spec",
            str(layer_spec),
            "--scene",
            "OldHospital",
            "--split",
            "test",
            "--output_manifest",
            str(output),
        ],
        check=True,
    )

    manifest = json.loads(output.read_text())
    assert manifest["records"][0]["image_id"] == "q0"
    assert manifest["records"][0]["checksum"]


def test_build_hypothesis_bank_and_eval_score_table_cli(tmp_path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "c0",
                    "candidate_type": "reference_pose",
                    "prior_score": 0.7,
                    "pose_error": {"translation_m": 0.1, "rotation_deg": 2.0},
                }
            ]
        )
    )
    bank = tmp_path / "bank.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_hypothesis_bank",
            "--protocol_name",
            "synthetic",
            "--protocol_kind",
            "real_retrieval",
            "--candidates",
            str(candidates),
            "--output",
            str(bank),
        ],
        check=True,
    )
    assert "candidate" in bank.read_text()

    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "query_id": "q0",
                    "candidate_id": "c0",
                    "score": 1.0,
                    "cost_m": 0.1,
                    "basin_label": True,
                    "protocol_kind": "real_retrieval",
                    "method": "selected_feature",
                },
                {
                    "query_id": "q0",
                    "candidate_id": "c1",
                    "score": 0.0,
                    "cost_m": 0.5,
                    "basin_label": False,
                    "protocol_kind": "real_retrieval",
                    "method": "selected_feature",
                },
            ]
        )
    )
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.eval_score_table",
            "--rows",
            str(rows),
            "--output_json",
            str(report),
        ],
        check=True,
    )

    data = json.loads(report.read_text())
    assert data["mean_top1_acc"] == 1.0
    assert data["mean_pred_cost_m"] == 0.1


def test_build_token_descriptor_bank_cli(tmp_path):
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    np.savez_compressed(
        token_dir / "q0.npz",
        radio_final=np.asarray([[[1.0, 3.0]], [[0.0, 0.0]]], dtype=np.float32),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_id": "q0",
                        "token_path": str(token_dir / "q0.npz"),
                        "layers": [
                            {
                                "name": "radio_final",
                                "model": "c-radio_v4-h",
                                "layer": "final",
                                "channels": 2,
                                "stride": 16,
                            }
                        ],
                        "split": "test",
                        "scene": "Synthetic",
                        "checksum": "",
                        "metadata": {},
                    }
                ]
            }
        )
    )
    output = tmp_path / "descriptors.npz"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_token_descriptor_bank",
            "--manifest",
            str(manifest),
            "--layer_name",
            "radio_final",
            "--output",
            str(output),
        ],
        check=True,
    )

    with np.load(output, allow_pickle=True) as data:
        assert data["image_ids"].tolist() == ["q0"]
        np.testing.assert_allclose(data["descriptors"], np.asarray([[1.0, 0.0]], dtype=np.float32))


def test_score_candidate_bank_descriptors_cli(tmp_path):
    query_descriptors = tmp_path / "query_desc.npz"
    map_descriptors = tmp_path / "map_desc.npz"
    np.savez_compressed(
        query_descriptors,
        image_ids=np.asarray(["q0"], dtype=object),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "layer_name": "radio_final",
                    "pooling": "mean",
                    "normalized": True,
                    "metadata": {},
                }
            )
        ),
    )
    np.savez_compressed(
        map_descriptors,
        image_ids=np.asarray(["good", "bad"], dtype=object),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "layer_name": "radio_final",
                    "pooling": "mean",
                    "normalized": True,
                    "metadata": {},
                }
            )
        ),
    )
    candidates = tmp_path / "bank.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "header",
                        "protocol_name": "synthetic",
                        "protocol_kind": "reference_pose",
                        "protocol_fingerprint": "",
                    }
                ),
                json.dumps(
                    {
                        "record_type": "candidate",
                        "query_id": "q0",
                        "candidate_id": "bad",
                        "candidate_type": "reference_pose",
                        "reference_image": "bad",
                        "pose_error": {"translation_m": 1.0, "rotation_deg": 20.0},
                    }
                ),
                json.dumps(
                    {
                        "record_type": "candidate",
                        "query_id": "q0",
                        "candidate_id": "good",
                        "candidate_type": "reference_pose",
                        "reference_image": "good",
                        "pose_error": {"translation_m": 0.1, "rotation_deg": 2.0},
                    }
                ),
            ]
        )
        + "\n"
    )
    report = tmp_path / "report.json"
    rows = tmp_path / "rows.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.score_candidate_bank_descriptors",
            "--bank",
            str(candidates),
            "--query_descriptors",
            str(query_descriptors),
            "--map_descriptors",
            str(map_descriptors),
            "--translation_threshold_m",
            "0.25",
            "--rotation_threshold_deg",
            "5.0",
            "--output_rows",
            str(rows),
            "--output_report",
            str(report),
        ],
        check=True,
    )

    data = json.loads(report.read_text())
    assert data["mean_top1_acc"] == 1.0
    assert data["mean_pred_cost_m"] == 0.1
    assert data["inputs"]["protocol_name"] == "synthetic"
    assert data["inputs"]["input_files"]["candidate_bank"]["sha256"]
    assert data["inputs"]["input_files"]["query_descriptors"]["sha256"]
