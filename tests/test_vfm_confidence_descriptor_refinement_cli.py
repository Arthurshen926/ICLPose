import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.train_stage_c29_confidence_descriptor_refinement import main
from tests.test_vfm_confidence_descriptor_refinement import _write_fixture


def test_train_stage_c29_confidence_descriptor_refinement_cli_writes_summary(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path = _write_fixture(tmp_path)
    summary_path = tmp_path / "summary.json"
    model_path = tmp_path / "model.pt"

    main(
        [
            "--match_jsonl",
            str(match_path),
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--output_model",
            str(model_path),
            "--summary_json",
            str(summary_path),
            "--steps",
            "20",
            "--batch_size",
            "2",
            "--lr",
            "0.1",
            "--output_dim",
            "4",
        ]
    )

    summary = json.loads(summary_path.read_text())
    assert summary["stage"] == "stage_c29_confidence_supervised_descriptor_refinement"
    assert summary["sample_summary"]["sample_count"] == 2
    assert summary["training"]["output_dim"] == 4
    assert np.isfinite(summary["training"]["train_top1_acc"])
    assert model_path.exists()
