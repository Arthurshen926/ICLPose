from __future__ import annotations

import json
from pathlib import Path

from feature_extract.tools.vfm.convert_render_match_table_to_jsonl import main


def test_convert_render_match_table_csv_to_typed_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "match_table.csv"
    jsonl_path = tmp_path / "match_table.jsonl"
    csv_path.write_text(
        "query_id,similarity,patch_correct,pnp_inlier,ignore_label,token_match_rank,baseline_reproj_residual_px,xy\n"
        "q1,0.75,True,False,,3,4.25,\"[10.0, 20.0]\"\n"
    )

    main(["--match_csv", str(csv_path), "--output_jsonl", str(jsonl_path)])

    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows == [
        {
            "query_id": "q1",
            "similarity": 0.75,
            "patch_correct": True,
            "pnp_inlier": False,
            "ignore_label": None,
            "token_match_rank": 3,
            "baseline_reproj_residual_px": 4.25,
            "xy": [10.0, 20.0],
        }
    ]
