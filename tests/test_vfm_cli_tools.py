import json
import subprocess
import sys
from pathlib import Path

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


def test_build_token_descriptor_bank_cli_supports_vlad_pooling(tmp_path):
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    np.savez_compressed(
        token_dir / "q0.npz",
        radio_final=np.asarray([[[1.0, 0.4]], [[0.0, 0.6]]], dtype=np.float32),
    )
    np.savez_compressed(
        token_dir / "q1.npz",
        radio_final=np.asarray([[[0.0, 0.6]], [[1.0, 0.4]]], dtype=np.float32),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_id": "q0",
                        "token_path": str(token_dir / "q0.npz"),
                        "layers": [{"name": "radio_final", "model": "c-radio_v4-h", "layer": "final", "channels": 2, "stride": 16}],
                        "split": "test",
                        "scene": "Synthetic",
                        "checksum": "",
                        "metadata": {},
                    },
                    {
                        "image_id": "q1",
                        "token_path": str(token_dir / "q1.npz"),
                        "layers": [{"name": "radio_final", "model": "c-radio_v4-h", "layer": "final", "channels": 2, "stride": 16}],
                        "split": "test",
                        "scene": "Synthetic",
                        "checksum": "",
                        "metadata": {},
                    },
                ]
            }
        )
    )
    output = tmp_path / "vlad_descriptors.npz"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_token_descriptor_bank",
            "--manifest",
            str(manifest),
            "--layer_name",
            "radio_final",
            "--pooling",
            "vlad",
            "--vlad_clusters",
            "2",
            "--vlad_iterations",
            "4",
            "--output",
            str(output),
        ],
        check=True,
    )

    with np.load(output, allow_pickle=True) as data:
        descriptors = np.asarray(data["descriptors"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].tolist()))
    assert descriptors.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), np.ones(2), atol=1e-5)
    assert float(descriptors[0] @ descriptors[1]) < 0.5
    assert metadata["pooling"] == "vlad"
    assert metadata["metadata"]["vlad_clusters"] == 2


def test_build_token_descriptor_bank_cli_reuses_vlad_codebook(tmp_path):
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    for image_id, values in {
        "db0": np.asarray([[[1.0, 0.4]], [[0.0, 0.6]]], dtype=np.float32),
        "db1": np.asarray([[[0.0, 0.6]], [[1.0, 0.4]]], dtype=np.float32),
        "q0": np.asarray([[[0.9, 0.5]], [[0.1, 0.5]]], dtype=np.float32),
    }.items():
        np.savez_compressed(token_dir / f"{image_id}.npz", radio_final=values)

    def write_manifest(path: Path, image_ids) -> None:
        path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "image_id": image_id,
                            "token_path": str(token_dir / f"{image_id}.npz"),
                            "layers": [{"name": "radio_final", "model": "c-radio_v4-h", "layer": "final", "channels": 2, "stride": 16}],
                            "split": "test",
                            "scene": "Synthetic",
                            "checksum": "",
                            "metadata": {},
                        }
                        for image_id in image_ids
                    ]
                }
            )
        )

    db_manifest = tmp_path / "db_manifest.json"
    query_manifest = tmp_path / "query_manifest.json"
    write_manifest(db_manifest, ["db0", "db1"])
    write_manifest(query_manifest, ["q0"])
    codebook = tmp_path / "vlad_codebook.npz"
    db_output = tmp_path / "db_vlad.npz"
    query_output = tmp_path / "query_vlad.npz"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_token_descriptor_bank",
            "--manifest",
            str(db_manifest),
            "--layer_name",
            "radio_final",
            "--pooling",
            "vlad",
            "--vlad_clusters",
            "2",
            "--vlad_codebook_output",
            str(codebook),
            "--vlad_tokens_per_image",
            "1",
            "--output",
            str(db_output),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_token_descriptor_bank",
            "--manifest",
            str(query_manifest),
            "--layer_name",
            "radio_final",
            "--pooling",
            "vlad",
            "--vlad_codebook_input",
            str(codebook),
            "--vlad_tokens_per_image",
            "1",
            "--output",
            str(query_output),
        ],
        check=True,
    )

    with np.load(codebook) as data:
        assert data["centroids"].shape == (2, 2)
    with np.load(db_output, allow_pickle=True) as data:
        db_meta = json.loads(str(data["metadata_json"].tolist()))
    with np.load(query_output, allow_pickle=True) as data:
        query_meta = json.loads(str(data["metadata_json"].tolist()))
    assert db_meta["metadata"]["vlad_codebook_output"] == str(codebook)
    assert db_meta["metadata"]["vlad_tokens_per_image"] == 1
    assert query_meta["metadata"]["vlad_codebook_input"] == str(codebook)
    assert query_meta["metadata"]["vlad_tokens_per_image"] == 1


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
