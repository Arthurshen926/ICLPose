from __future__ import annotations

import csv
import json

from feature_extract.tools.vfm.train_coarse_candidate_ranker import main


def test_train_coarse_candidate_ranker_writes_model_and_summary(tmp_path) -> None:
    csv_path = tmp_path / "matches.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "similarity",
                "similarity_margin",
                "confidence",
                "coarse_rank",
                "coarse_score",
                "coarse_score_gap",
                "mutual_rank",
                "gt_reproj_error_stride",
                "patch_correct",
            ],
        )
        writer.writeheader()
        for query_idx in range(4):
            writer.writerow(
                {
                    "query_id": f"q{query_idx}",
                    "similarity": "0.9",
                    "similarity_margin": "0.2",
                    "confidence": "0.7",
                    "coarse_rank": "0",
                    "coarse_score": "0.9",
                    "coarse_score_gap": "0.0",
                    "mutual_rank": "0",
                    "gt_reproj_error_stride": "0.4",
                    "patch_correct": "True",
                }
            )
            writer.writerow(
                {
                    "query_id": f"q{query_idx}",
                    "similarity": "0.5",
                    "similarity_margin": "0.02",
                    "confidence": "0.1",
                    "coarse_rank": "4",
                    "coarse_score": "0.5",
                    "coarse_score_gap": "0.4",
                    "mutual_rank": "3",
                    "gt_reproj_error_stride": "4.0",
                    "patch_correct": "False",
                }
            )

    out_dir = tmp_path / "ranker"
    main(
        [
            "--match_csv",
            str(csv_path),
            "--output_dir",
            str(out_dir),
            "--feature_sets",
            "coarse",
            "--eval_on_train",
            "--max_iter",
            "50",
        ]
    )

    summary = json.loads((out_dir / "coarse_candidate_ranker_summary.json").read_text())
    assert summary["row_count"] == 8
    assert summary["models"]["coarse"]["status"] == "trained"
    assert (out_dir / "coarse_model.json").exists()
