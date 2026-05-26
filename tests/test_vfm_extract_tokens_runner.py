import numpy as np
from PIL import Image

from feature_extract.tools.vfm.extract_tokens import (
    discover_image_files,
    extract_token_manifest_from_images,
)
from feature_extract.vfm.tokens import TokenLayerSpec


class FakeExtractor:
    layer_specs = (TokenLayerSpec("fake_final", "fake-vfm", "final", 3, 4),)

    def __init__(self):
        self.calls = 0

    def extract(self, image_path):
        self.calls += 1
        return {"fake_final": np.ones((3, 2, 2), dtype=np.float32)}


class FakeBatchExtractor:
    layer_specs = (TokenLayerSpec("fake_batch", "fake-vfm", "final", 2, 4),)

    def __init__(self):
        self.batch_sizes = []

    def extract(self, image_path):
        raise AssertionError("batch path should be used")

    def extract_batch(self, image_paths):
        self.batch_sizes.append(len(image_paths))
        return [
            {"fake_batch": np.full((2, 1, 1), idx, dtype=np.float32)}
            for idx, _ in enumerate(image_paths)
        ]


def test_discover_image_files_uses_split_file(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "seq0" / "frame.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    split_file = tmp_path / "split.txt"
    split_file.write_text("seq0/frame.png\n")

    files = discover_image_files(image_root=image_root, split_file=split_file)

    assert files == (image_path,)


def test_extract_token_manifest_from_images_writes_npz_and_manifest(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "seq0" / "frame.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)

    manifest = extract_token_manifest_from_images(
        image_paths=(image_path,),
        image_root=image_root,
        output_root=tmp_path / "tokens",
        scene="Synthetic",
        split="train",
        extractors=(FakeExtractor(),),
        storage_dtype="float16",
    )

    assert manifest.records[0].image_id == "seq0/frame.png"
    assert manifest.records[0].checksum
    with np.load(manifest.records[0].token_path) as data:
        assert data["fake_final"].shape == (3, 2, 2)
        assert data["fake_final"].dtype == np.float16


def test_extract_token_manifest_uses_batch_extractor_when_available(tmp_path):
    image_root = tmp_path / "images"
    paths = []
    for idx in range(3):
        path = image_root / "seq0" / f"frame{idx}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
        paths.append(path)
    extractor = FakeBatchExtractor()

    manifest = extract_token_manifest_from_images(
        image_paths=tuple(paths),
        image_root=image_root,
        output_root=tmp_path / "tokens",
        scene="Synthetic",
        split="train",
        extractors=(extractor,),
        storage_dtype="float32",
        batch_size=2,
    )

    assert extractor.batch_sizes == [2, 1]
    assert len(manifest.records) == 3


def test_extract_token_manifest_skip_existing_reuses_npz_without_extractor_call(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "seq0" / "frame.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    token_path = tmp_path / "tokens" / "seq0__frame.png.npz"
    token_path.parent.mkdir(parents=True)
    np.savez_compressed(token_path, fake_final=np.zeros((3, 2, 2), dtype=np.float16))
    extractor = FakeExtractor()

    manifest = extract_token_manifest_from_images(
        image_paths=(image_path,),
        image_root=image_root,
        output_root=tmp_path / "tokens",
        scene="Synthetic",
        split="train",
        extractors=(extractor,),
        storage_dtype="float16",
        skip_existing=True,
    )

    assert extractor.calls == 0
    assert manifest.records[0].token_path == token_path
    assert manifest.records[0].checksum
