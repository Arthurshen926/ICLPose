from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from feature_extract.tools.vfm.eval_real_radio_closed_loop import parse_args, resolve_runtime_device
from feature_extract.vfm.localization import CoarseProposal, MappedFeatureMap, MeasurementResult
from feature_extract.vfm.localization.pipeline import (
    RealRadioLocalizationPair,
    _load_feature_map,
    _load_rgb_chw,
    load_real_radio_localization_pairs_csv,
    run_real_radio_localization_pairs,
)


def _write_rgb(path: Path, *, value: int = 0, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), int(value), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_load_feature_map_accepts_npy_and_single_array_npz(tmp_path: Path) -> None:
    feature = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    npy = tmp_path / "feature.npy"
    npz = tmp_path / "feature.npz"
    np.save(npy, feature)
    np.savez(npz, radio_dual=feature)

    np.testing.assert_allclose(_load_feature_map(npy), feature)
    np.testing.assert_allclose(_load_feature_map(npz), feature)
    np.testing.assert_allclose(_load_feature_map(npz, key="radio_dual"), feature)


def test_load_rgb_chw_returns_normalized_chw_tensor(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    _write_rgb(image, value=128, size=(4, 2))

    rgb = _load_rgb_chw(image)

    assert rgb.shape == (3, 2, 4)
    assert rgb.dtype == np.float32
    assert np.allclose(rgb, 128.0 / 255.0)


def test_load_real_radio_pairs_csv_groups_multiple_tracks_per_pair(tmp_path: Path) -> None:
    rows_csv = tmp_path / "pairs.csv"
    with rows_csv.open("w", newline="") as handle:
        fieldnames = ["query_id", "support_image_id", "query_gt_x", "query_gt_y", "support_x", "support_y"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "seq/q.png",
                "support_image_id": "seq/r.png",
                "query_gt_x": "1",
                "query_gt_y": "2",
                "support_x": "3",
                "support_y": "4",
            }
        )
        writer.writerow(
            {
                "query_id": "seq/q.png",
                "support_image_id": "seq/r.png",
                "query_gt_x": "5",
                "query_gt_y": "6",
                "support_x": "7",
                "support_y": "8",
            }
        )

    pairs = load_real_radio_localization_pairs_csv(rows_csv, feature_path_template="{image_stem}.npy")

    assert len(pairs) == 1
    assert pairs[0].reference_image_id == "seq/r.png"
    assert pairs[0].query_feature_path == Path("seq_q.npy")
    assert pairs[0].reference_feature_path == Path("seq_r.npy")
    assert len(pairs[0].ground_truth) == 2


def test_load_real_radio_pairs_csv_supports_radio_token_feature_template(tmp_path: Path) -> None:
    rows_csv = tmp_path / "pairs.csv"
    with rows_csv.open("w", newline="") as handle:
        fieldnames = ["query_id", "support_image_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"query_id": "seq9/frame00010.png", "support_image_id": "seq9/frame00006.png"})

    pairs = load_real_radio_localization_pairs_csv(rows_csv, feature_path_template="{image_token}.npz")

    assert pairs[0].query_feature_path == Path("seq9__frame00010.png.npz")
    assert pairs[0].reference_feature_path == Path("seq9__frame00006.png.npz")


def test_run_real_radio_localization_pairs_writes_coarse_and_measurement_rows(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    feature_root = tmp_path / "features"
    _write_rgb(image_root / "query.png", value=10)
    _write_rgb(image_root / "reference.png", value=20)
    feature_root.mkdir(parents=True)
    query_feature = np.zeros((2, 1, 1), dtype=np.float32)
    reference_feature = np.zeros((2, 1, 1), dtype=np.float32)
    np.save(feature_root / "query.npy", query_feature)
    np.save(feature_root / "reference.npy", reference_feature)

    class FakeFeatureMapper:
        def project(self, feature_map):
            return MappedFeatureMap(
                coarse_descriptors=np.asarray(feature_map, dtype=np.float32),
                measurement_context=np.asarray(feature_map, dtype=np.float32),
            )

    class FakeCoarseMatcher:
        def match(self, query_descriptors, reference_descriptors, *, query_image_size, reference_image_size):
            del query_descriptors, reference_descriptors, query_image_size, reference_image_size
            return [
                CoarseProposal(
                    query_index=0,
                    reference_index=0,
                    query_xy=np.asarray([5.0, 6.0], dtype=np.float32),
                    reference_xy=np.asarray([7.0, 8.0], dtype=np.float32),
                    score=0.9,
                    confidence=0.8,
                    rank=0,
                    metadata={"mutual": True},
                )
            ]

    class FakeMeasurement:
        def measure(self, query_rgb, reference_rgb, proposals, *, mapped_query=None, mapped_reference=None):
            assert query_rgb.shape == (3, 16, 16)
            assert reference_rgb.shape == (3, 16, 16)
            assert mapped_query is not None
            assert mapped_reference is not None
            return [
                MeasurementResult(
                    proposal=proposals[0],
                    measured_query_xy=np.asarray([5.5, 6.0], dtype=np.float32),
                    measured_reference_xy=proposals[0].reference_xy,
                    confidence=0.75,
                    uncertainty_px=0.25,
                )
            ]

    pairs = [
        RealRadioLocalizationPair(
            query_id="query.png",
            reference_image_id="reference.png",
            query_feature_path=Path("query.npy"),
            reference_feature_path=Path("reference.npy"),
            query_gt_xy=np.asarray([5.5, 6.0], dtype=np.float32),
            reference_gt_xy=np.asarray([7.0, 8.0], dtype=np.float32),
        )
    ]

    summary = run_real_radio_localization_pairs(
        pairs,
        image_root=image_root,
        feature_root=feature_root,
        output_dir=tmp_path / "out",
        feature_mapper=FakeFeatureMapper(),
        coarse_matcher=FakeCoarseMatcher(),
        measurement_branch=FakeMeasurement(),
        gt_reference_radius_px=1.0,
    )

    assert summary["pair_count"] == 1
    assert summary["proposal_count"] == 1
    assert summary["measurement_count"] == 1
    assert summary["gt_matched_count"] == 1
    assert summary["measurement_epe_median_px"] == 0.0
    assert summary["measurement_improve_ratio"] == 1.0
    with Path(summary["outputs"]["proposals_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["query_id"] == "query.png"
    assert rows[0]["reference_image_id"] == "reference.png"
    assert rows[0]["measured_query_x"] == "5.5"
    assert rows[0]["measurement_confidence"] == "0.75"
    assert rows[0]["gt_query_x"] == "5.5"


def test_eval_real_radio_closed_loop_cli_uses_joint_mapper_topk_and_measurement() -> None:
    args = parse_args(
        [
            "--pairs_csv",
            "pairs.csv",
            "--image_root",
            "images",
            "--feature_root",
            "features",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
            "--k_per_query",
            "3",
            "--prediction_head",
            "gated",
            "--measurement_batch_size",
            "32",
            "--device",
            "cpu",
        ]
    )

    assert args.matcha_joint_checkpoint == "joint.pt"
    assert args.k_per_query == 3
    assert args.prediction_head == "gated"
    assert args.measurement_batch_size == 32
    assert not hasattr(args, "render_cache_manifest_csv")


def test_eval_real_radio_closed_loop_resolves_runtime_device(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert str(resolve_runtime_device("cuda")) == "cpu"
    assert str(resolve_runtime_device("cpu")) == "cpu"
