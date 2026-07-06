"""Materialize RADIO feature caches for measurement-v1 D0 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _extract_radio_feature_from_rgb
from feature_extract.vfm.measurement_v1.radio_cache import (
    materialize_radio_cache_for_d0_rows,
    write_radio_cache_summary,
)


class _RadioExtractorAdapter:
    def __init__(self, *, version: str, device: str, radio_repo: str, feature_key: str) -> None:
        from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor

        self.extractor = RADIOFeatureExtractor(version=version, device=device, radio_repo=radio_repo)
        self.feature_key = str(feature_key)

    def extract_local(self, rgb: np.ndarray) -> np.ndarray:
        if self.feature_key == "radio_dual":
            import torch

            image = np.asarray(rgb, dtype=np.float32)
            if image.max(initial=0.0) > 1.0:
                image = image / 255.0
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            return self.extractor.extract_dual(tensor)["dual"].detach().cpu().numpy().astype(np.float32, copy=False)
        return _extract_radio_feature_from_rgb(rgb, self.extractor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--render_cache_manifest_csv", required=True)
    parser.add_argument("--render_radio_cache_dir", required=True)
    parser.add_argument("--query_radio_cache_dir", default="")
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--feature_key", default="radio_dual")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--output_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument("--summary_json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    extractor = _RadioExtractorAdapter(
        version=str(args.radio_version),
        device=str(args.device),
        radio_repo=str(args.radio_repo),
        feature_key=str(args.feature_key),
    )
    summary = materialize_radio_cache_for_d0_rows(
        rows_csv=Path(args.rows_csv),
        output_rows_csv=Path(args.output_rows_csv),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv),
        render_radio_cache_dir=Path(args.render_radio_cache_dir),
        extractor=extractor,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        key=str(args.feature_key),
        base_dir=Path(args.base_dir),
        query_radio_cache_dir=Path(args.query_radio_cache_dir) if str(args.query_radio_cache_dir) else None,
        skip_existing=not bool(args.no_skip_existing),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        output_dtype=str(args.output_dtype),
    )
    summary_path = Path(args.summary_json) if str(args.summary_json) else Path(args.output_rows_csv).with_name("radio_cache_summary.json")
    write_radio_cache_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
