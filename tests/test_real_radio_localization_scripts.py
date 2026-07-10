from __future__ import annotations

from pathlib import Path

from feature_extract.tools.vfm import eval_real_radio_localization
from feature_extract.tools.vfm import train_real_radio_localization

parse_eval_args = eval_real_radio_localization.parse_args
parse_train_args = train_real_radio_localization.parse_args


def test_train_real_radio_localization_cli_defaults_to_real_pair_rows() -> None:
    args = parse_train_args(
        [
            "--train_rows_csv",
            "train.csv",
            "--val_rows_csv",
            "val.csv",
            "--image_root",
            "images",
            "--output_dir",
            "out",
            "--image_width",
            "640",
            "--image_height",
            "480",
            "--steps",
            "10",
            "--batch_size",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert args.train_rows_csv == "train.csv"
    assert args.val_rows_csv == "val.csv"
    assert args.query_source == "real_pair"
    assert args.render_patch_augmentation == "none"
    assert not hasattr(args, "render_cache_manifest_csv")
    assert not hasattr(args, "train_render_cache_manifest_csv")


def test_eval_real_radio_localization_cli_uses_real_reference_rows() -> None:
    args = parse_eval_args(
        [
            "--rows_csv",
            "rows.csv",
            "--image_root",
            "images",
            "--checkpoint",
            "model.pt",
            "--output_dir",
            "out",
            "--image_width",
            "640",
            "--image_height",
            "480",
            "--prediction_head",
            "gated",
            "--device",
            "cpu",
        ]
    )

    assert args.rows_csv == "rows.csv"
    assert args.reference_source == "real_pair"
    assert args.prediction_head == "gated"
    assert not hasattr(args, "render_cache_manifest_csv")


def test_train_real_radio_localization_main_invokes_real_pair_training(monkeypatch) -> None:
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"stage": "fake", "outputs": {}}

    monkeypatch.setattr(train_real_radio_localization, "train_rgb_patch_measurement_branch", fake_train)

    train_real_radio_localization.main(
        [
            "--train_rows_csv",
            "train.csv",
            "--image_root",
            "images",
            "--output_dir",
            "out",
            "--image_width",
            "640",
            "--image_height",
            "480",
            "--reference_image_width",
            "320",
            "--reference_image_height",
            "240",
            "--steps",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert captured["rows_csv"] == Path("train.csv")
    assert captured["render_cache_manifest_csv"] is None
    assert captured["val_render_cache_manifest_csv"] is None
    assert captured["query_source"] == "real_pair"
    assert captured["render_patch_augmentation"] == "none"
    assert captured["render_image_width"] == 320
    assert captured["render_image_height"] == 240


def test_eval_real_radio_localization_main_invokes_real_pair_fusion(monkeypatch) -> None:
    captured = {}

    def fake_apply(**kwargs):
        captured.update(kwargs)
        return {"stage": "fake", "outputs": {}}

    monkeypatch.setattr(eval_real_radio_localization, "apply_rgb_patch_measurements_to_real_pair_rows", fake_apply)

    eval_real_radio_localization.main(
        [
            "--rows_csv",
            "rows.csv",
            "--image_root",
            "images",
            "--checkpoint",
            "model.pt",
            "--output_dir",
            "out",
            "--image_width",
            "640",
            "--image_height",
            "480",
            "--reference_image_width",
            "320",
            "--reference_image_height",
            "240",
            "--device",
            "cpu",
        ]
    )

    assert captured["rows_csv"] == Path("rows.csv")
    assert captured["image_root"] == Path("images")
    assert captured["checkpoint"] == Path("model.pt")
    assert captured["reference_image_width"] == 320
    assert captured["reference_image_height"] == 240
