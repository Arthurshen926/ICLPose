from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT,
    RAW_FORMAT,
    main,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
    save_spatial_image_context_cache,
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def test_intermediate_spatial_pca_uses_only_manifest_training_images(tmp_path) -> None:
    generator = np.random.default_rng(8)
    raw = tmp_path / "raw.npz"
    ids = np.asarray(["a.png", "b.png", "c.png", "d.png"])
    values = _unit_rows(generator.normal(size=(4, 16, 4)).astype(np.float32))
    save_spatial_image_context_cache(
        SpatialImageContextCache(
            image_ids=ids,
            image_sizes=np.asarray([[80, 40]] * 4, dtype=np.int64),
            grids={4: values},
            metadata={
                "format": RAW_FORMAT,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "spatial_grid_sizes": [4],
                "radio_checkpoint_sha256": "radio-test",
                "source_image_manifest_sha256": "image-test",
                "image_source_contract": {
                    "version": 1,
                    "resolved_image_root": "/tmp/images",
                    "image_count": 4,
                    "image_ids_sha256": "ids-test",
                    "sampled_content_manifest_sha256": "image-test",
                    "source_image_dimensions": {"80x40": 4},
                    "sampled_bytes_per_file_end": 32768,
                },
                "intermediate_index": -6,
            },
        ),
        raw,
    )
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"records": [{"image_id": "a.png"}, {"image_id": "b.png"}]}))
    output = tmp_path / "pca.npz"
    summary = tmp_path / "summary.json"

    assert main(
        [
            "--source_context",
            str(raw),
            "--pca_training_manifest",
            str(manifest),
            "--output_cache",
            str(output),
            "--summary_json",
            str(summary),
            "--projection_dim",
            "2",
            "--seed",
            "0",
        ]
    ) == 0

    cache = load_spatial_image_context_cache(output, expected_format=PCA_FORMAT)
    assert cache.grids[4].shape == (4, 16, 2)
    assert cache.metadata["pca_fit_scope"] == "mapping_train_images_only"
    assert cache.metadata["pca_training_image_count"] == 2
    assert cache.metadata["radio_checkpoint_sha256"] == "radio-test"
    assert cache.metadata["image_source_contract"]["image_ids_sha256"] == "ids-test"
    with np.load(output, allow_pickle=False) as data:
        assert data["grid4_pca_mean"].shape == (4,)
        assert data["grid4_pca_components"].shape == (2, 4)
        assert np.isfinite(data["grid4_pca_mean"]).all()
        assert np.isfinite(data["grid4_pca_components"]).all()


def test_intermediate_spatial_pca_rejects_legacy_source_without_image_contract(
    tmp_path,
) -> None:
    raw = tmp_path / "raw.npz"
    values = _unit_rows(np.ones((2, 16, 4), dtype=np.float32))
    save_spatial_image_context_cache(
        SpatialImageContextCache(
            image_ids=np.asarray(["a.png", "b.png"]),
            image_sizes=np.asarray([[80, 40]] * 2, dtype=np.int64),
            grids={4: values},
            metadata={
                "format": RAW_FORMAT,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "spatial_grid_sizes": [4],
                "source_image_manifest_sha256": "legacy-only",
            },
        ),
        raw,
    )
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"records": [{"image_id": "a.png"}]}))

    with np.testing.assert_raises_regex(ValueError, "image source contract lacks fields"):
        main(
            [
                "--source_context",
                str(raw),
                "--pca_training_manifest",
                str(manifest),
                "--output_cache",
                str(tmp_path / "pca.npz"),
                "--summary_json",
                str(tmp_path / "summary.json"),
                "--projection_dim",
                "2",
            ]
        )


def test_intermediate_spatial_pca_preserves_disjoint_support_fit_scope(tmp_path) -> None:
    generator = np.random.default_rng(4)
    raw = tmp_path / "raw.npz"
    values = _unit_rows(generator.normal(size=(3, 16, 4)).astype(np.float32))
    contract = {
        "version": 1,
        "resolved_image_root": "/tmp/images",
        "image_count": 3,
        "image_ids_sha256": "ids-test",
        "sampled_content_manifest_sha256": "image-test",
        "source_image_dimensions": {"80x40": 3},
        "sampled_bytes_per_file_end": 32768,
    }
    save_spatial_image_context_cache(
        SpatialImageContextCache(
            image_ids=np.asarray(["a.png", "b.png", "c.png"]),
            image_sizes=np.asarray([[80, 40]] * 3, dtype=np.int64),
            grids={4: values},
            metadata={
                "format": RAW_FORMAT,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "spatial_grid_sizes": [4],
                "source_image_manifest_sha256": "image-test",
                "image_source_contract": contract,
                "intermediate_index": -6,
            },
        ),
        raw,
    )
    manifest = tmp_path / "support.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [{"image_id": "a.png"}, {"image_id": "b.png"}],
                "metadata": {
                    "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1"
                },
            }
        )
    )
    output = tmp_path / "pca.npz"
    assert main(
        [
            "--source_context",
            str(raw),
            "--pca_training_manifest",
            str(manifest),
            "--output_cache",
            str(output),
            "--summary_json",
            str(tmp_path / "summary.json"),
            "--projection_dim",
            "2",
        ]
    ) == 0
    cache = load_spatial_image_context_cache(output, expected_format=PCA_FORMAT)
    assert (
        cache.metadata["pca_fit_scope"]
        == "mapping_support_images_excluding_all_query_splits_v1"
    )
