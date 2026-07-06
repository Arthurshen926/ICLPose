from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from feature_extract.vfm.measurement_v1.radio_cache import materialize_radio_cache_for_d0_rows


class _FakeRadioExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract_local(self, rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        image = np.asarray(rgb, dtype=np.float32)
        return np.moveaxis(image / 255.0, -1, 0)[None]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_materialize_radio_cache_fills_render_paths_and_reuses_existing_query_cache(tmp_path: Path) -> None:
    query_cache = tmp_path / "query" / "q0_4x4.npz"
    query_cache.parent.mkdir(parents=True)
    np.savez_compressed(query_cache, radio_dual=np.ones((1, 3, 4, 4), dtype=np.float32))
    render_rgb_cache = tmp_path / "render_rgb_depth" / "q0_4x4.npz"
    render_rgb_cache.parent.mkdir(parents=True)
    np.savez_compressed(
        render_rgb_cache,
        rgb=np.full((4, 4, 3), 128, dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        alpha=np.ones((4, 4), dtype=np.float32),
    )
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(manifest, [{"query_id": "q0.png", "rgb_depth_cache_path": str(render_rgb_cache)}])
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_radio_dual_feature_cache_path": str(query_cache),
                "render_radio_dual_feature_cache_path": "",
            }
        ],
    )

    extractor = _FakeRadioExtractor()
    summary = materialize_radio_cache_for_d0_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "rows_out.csv",
        render_cache_manifest_csv=manifest,
        render_radio_cache_dir=tmp_path / "render_radio",
        extractor=extractor,
        image_width=4,
        image_height=4,
    )

    out_rows = list(csv.DictReader((tmp_path / "rows_out.csv").open()))
    render_path = Path(out_rows[0]["render_radio_dual_feature_cache_path"])
    assert summary["render_cache_written_count"] == 1
    assert summary["query_cache_present_count"] == 1
    assert render_path.exists()
    with np.load(render_path) as data:
        assert data["radio_dual"].shape == (1, 3, 4, 4)
        assert data["radio_dual"].dtype == np.float32


def test_materialize_radio_cache_can_write_float16_render_features(tmp_path: Path) -> None:
    render_rgb_cache = tmp_path / "render_rgb_depth" / "q0_4x4.npz"
    render_rgb_cache.parent.mkdir(parents=True)
    np.savez_compressed(
        render_rgb_cache,
        rgb=np.full((4, 4, 3), 128, dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        alpha=np.ones((4, 4), dtype=np.float32),
    )
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(manifest, [{"query_id": "q0.png", "rgb_depth_cache_path": str(render_rgb_cache)}])
    rows_csv = tmp_path / "rows.csv"
    _write_csv(rows_csv, [{"query_id": "q0.png"}])

    summary = materialize_radio_cache_for_d0_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "rows_out.csv",
        render_cache_manifest_csv=manifest,
        render_radio_cache_dir=tmp_path / "render_radio",
        extractor=_FakeRadioExtractor(),
        image_width=4,
        image_height=4,
        output_dtype="float16",
    )

    out_rows = list(csv.DictReader((tmp_path / "rows_out.csv").open()))
    render_path = Path(out_rows[0]["render_radio_dual_feature_cache_path"])
    assert summary["output_dtype"] == "float16"
    with np.load(render_path) as data:
        assert data["radio_dual"].dtype == np.float16


def test_materialize_radio_cache_reports_missing_render_rgb_without_fabricating_cache(tmp_path: Path) -> None:
    rows_csv = tmp_path / "rows.csv"
    _write_csv(rows_csv, [{"query_id": "missing.png"}])
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(manifest, [{"query_id": "other.png", "rgb_depth_cache_path": str(tmp_path / "other.npz")}])

    summary = materialize_radio_cache_for_d0_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "rows_out.csv",
        render_cache_manifest_csv=manifest,
        render_radio_cache_dir=tmp_path / "render_radio",
        extractor=_FakeRadioExtractor(),
        image_width=4,
        image_height=4,
    )

    assert summary["missing_render_rgb_depth_cache_count"] == 1
    assert summary["render_cache_written_count"] == 0
