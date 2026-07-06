from __future__ import annotations

from pathlib import Path

from feature_extract.tools.vfm import augment_measurement_rows_with_local_affine as cli


def test_local_geometry_augment_cli_dispatches_homography(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_homography(**kwargs):
        calls.append(dict(kwargs))
        return {"geometry_model": "homography", "output_rows": 7}

    monkeypatch.setattr(cli, "augment_measurement_rows_with_local_homography_from_jsonl", fake_homography)

    rows_csv = tmp_path / "rows.csv"
    observations_jsonl = tmp_path / "tracks.jsonl"
    output_csv = tmp_path / "out.csv"
    cli.main(
        [
            "--geometry_model",
            "homography",
            "--rows_csv",
            str(rows_csv),
            "--track_observations_jsonl",
            str(observations_jsonl),
            "--output_rows_csv",
            str(output_csv),
            "--image_width",
            "1920",
            "--image_height",
            "1080",
            "--local_radius_px",
            "32",
            "--min_points",
            "8",
            "--max_points",
            "64",
            "--max_rmse_px",
            "6",
            "--max_rows",
            "128",
        ]
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["rows_csv"] == rows_csv
    assert call["track_observations_jsonl"] == observations_jsonl
    assert call["output_rows_csv"] == output_csv
    assert call["image_width"] == 1920
    assert call["image_height"] == 1080
    assert call["local_radius_px"] == 32.0
    assert call["min_points"] == 8
    assert call["max_points"] == 64
    assert call["max_rmse_px"] == 6.0
    assert call["max_rows"] == 128
    assert '"geometry_model": "homography"' in capsys.readouterr().out
