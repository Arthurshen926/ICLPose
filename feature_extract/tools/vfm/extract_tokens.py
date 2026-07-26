"""Create a VFM token-bank manifest or extract raw VFM tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import yaml

from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class TokenExtractor(Protocol):
    layer_specs: tuple[TokenLayerSpec, ...]

    def extract(self, image_path: Path) -> Mapping[str, np.ndarray]:
        ...


def _looks_like_image_reference(token: str) -> bool:
    return Path(token).suffix in IMAGE_SUFFIXES


def _load_layer_specs(path: Path) -> tuple[TokenLayerSpec, ...]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("layer spec must be a non-empty JSON list")
    return tuple(TokenLayerSpec(**dict(item)) for item in payload)


def build_manifest_from_npz_dir(
    npz_dir: Path,
    layer_specs: tuple[TokenLayerSpec, ...],
    scene: str,
    split: str,
) -> TokenBankManifest:
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"no .npz token files found in {npz_dir}")
    records = [
        TokenBankRecord(
            image_id=path.stem,
            token_path=path,
            layers=layer_specs,
            split=split,
            scene=scene,
            checksum=compute_file_sha256(path),
        )
        for path in files
    ]
    manifest = TokenBankManifest(records=tuple(records))
    manifest.validate()
    return manifest


def discover_image_files(
    image_root: Path,
    split_file: Path | None = None,
    max_images: int | None = None,
) -> tuple[Path, ...]:
    image_root = Path(image_root)
    if split_file is not None:
        files = []
        for line in Path(split_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rel = line.split()[0]
            if not _looks_like_image_reference(rel):
                continue
            path = Path(rel)
            files.append(path if path.is_absolute() else image_root / path)
    else:
        files = sorted(path for path in image_root.rglob("*") if path.suffix in IMAGE_SUFFIXES)

    if max_images is not None:
        files = files[:max_images]
    if not files:
        raise ValueError(f"no images found under {image_root}")
    missing = [path for path in files if not path.exists()]
    if missing:
        raise ValueError(f"split references missing image: {missing[0]}")
    return tuple(files)


def _image_id(image_root: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(image_root).as_posix()
    except ValueError:
        return image_path.name


def _token_path_for_image(output_root: Path, image_id: str) -> Path:
    safe_name = image_id.replace("/", "__").replace("\\", "__")
    return output_root / f"{safe_name}.npz"


def extract_token_manifest_from_images(
    image_paths: Sequence[Path],
    image_root: Path,
    output_root: Path,
    scene: str,
    split: str,
    extractors: Sequence[TokenExtractor],
    storage_dtype: str = "float32",
    batch_size: int = 1,
    skip_existing: bool = False,
) -> TokenBankManifest:
    if not extractors:
        raise ValueError("at least one extractor is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    layer_specs: list[TokenLayerSpec] = []
    for extractor in extractors:
        layer_specs.extend(extractor.layer_specs)
    if len({layer.name for layer in layer_specs}) != len(layer_specs):
        raise ValueError("extractor layer names must be unique")

    records = []
    total = len(image_paths)
    for batch_start in range(0, total, batch_size):
        raw_batch_paths = tuple(Path(path) for path in image_paths[batch_start: batch_start + batch_size])
        if skip_existing:
            batch_paths = []
            for image_path in raw_batch_paths:
                image_id = _image_id(Path(image_root), image_path)
                output_path = _token_path_for_image(Path(output_root), image_id)
                if output_path.exists():
                    records.append(
                        TokenBankRecord(
                            image_id=image_id,
                            token_path=output_path,
                            layers=tuple(layer_specs),
                            split=split,
                            scene=scene,
                            checksum=compute_file_sha256(output_path),
                        )
                    )
                else:
                    batch_paths.append(image_path)
            batch_paths = tuple(batch_paths)
            if not batch_paths:
                print(f"processed {min(batch_start + batch_size, total)}/{total} images")
                continue
        else:
            batch_paths = raw_batch_paths
        batch_arrays = [dict() for _ in batch_paths]
        for extractor in extractors:
            outputs = _extract_batch_or_single(extractor, batch_paths)
            if len(outputs) != len(batch_paths):
                raise ValueError("extractor returned a mismatched batch length")
            for item, output in zip(batch_arrays, outputs):
                for name, array in output.items():
                    if name in item:
                        raise ValueError(f"duplicate token array name: {name}")
                    item[name] = np.asarray(array, dtype=np.dtype(storage_dtype))
        for image_path, arrays in zip(batch_paths, batch_arrays):
            records.append(
                _write_token_record(
                    image_path=image_path,
                    image_root=Path(image_root),
                    output_root=Path(output_root),
                    scene=scene,
                    split=split,
                    arrays=arrays,
                    layer_specs=tuple(layer_specs),
                )
            )
        print(f"processed {min(batch_start + batch_size, total)}/{total} images")
    manifest = TokenBankManifest(records=tuple(records))
    manifest.validate()
    return manifest


def _extract_batch_or_single(
    extractor: TokenExtractor,
    image_paths: Sequence[Path],
) -> list[Mapping[str, np.ndarray]]:
    batch_method = getattr(extractor, "extract_batch", None)
    if callable(batch_method):
        return list(batch_method(tuple(image_paths)))
    return [extractor.extract(path) for path in image_paths]


def _write_token_record(
    image_path: Path,
    image_root: Path,
    output_root: Path,
    scene: str,
    split: str,
    arrays: Mapping[str, np.ndarray],
    layer_specs: tuple[TokenLayerSpec, ...],
) -> TokenBankRecord:
    image_id = _image_id(Path(image_root), Path(image_path))
    output_path = _token_path_for_image(Path(output_root), image_id)
    write_npz_token_record(output_path, arrays)
    return TokenBankRecord(
        image_id=image_id,
        token_path=output_path,
        layers=layer_specs,
        split=split,
        scene=scene,
        checksum=compute_file_sha256(output_path),
    )


class RadioTokenExtractor:
    def __init__(
        self,
        version: str,
        device: str,
        radio_repo: str,
        array_name: str = "radio_final",
        input_width: int = 0,
        input_height: int = 0,
    ):
        from feature_extract.extractors import RADIOFeatureExtractor

        self.array_name = array_name
        if (int(input_width) > 0) != (int(input_height) > 0):
            raise ValueError("RADIO input_width and input_height must be set together")
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.extractor = RADIOFeatureExtractor(
            version=version,
            device=device,
            radio_repo=radio_repo,
        )
        self.layer_specs = (
            TokenLayerSpec(
                name=array_name,
                model=version,
                layer="final",
                channels=1280,
                stride=16,
            ),
        )

    def extract(self, image_path: Path) -> Mapping[str, np.ndarray]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if self.input_width > 0:
            image = image.resize((self.input_width, self.input_height), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        outputs = self.extractor.extract(tensor)
        return {self.array_name: outputs["local"].detach().cpu().numpy()}

    def extract_batch(self, image_paths: Sequence[Path]) -> list[Mapping[str, np.ndarray]]:
        import torch
        from PIL import Image

        tensors = []
        shapes = set()
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            if self.input_width > 0:
                image = image.resize((self.input_width, self.input_height), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor)
            shapes.add(tuple(tensor.shape))
        if len(shapes) != 1:
            return [self.extract(path) for path in image_paths]
        batch = torch.stack(tensors, dim=0)
        outputs = self.extractor.extract_batch(batch)
        local = outputs["local"].detach().cpu().numpy()
        return [{self.array_name: local[idx]} for idx in range(local.shape[0])]


def _dinov2_channels(model_name: str) -> int:
    if "vitg" in model_name:
        return 1536
    if "vitl" in model_name:
        return 1024
    if "vitb" in model_name:
        return 768
    return 384


class DinoTokenExtractor:
    def __init__(
        self,
        model_name: str,
        stride: int,
        device: str,
        layer: int = 11,
        facet: str = "token",
        array_name: str = "dinov2_final",
    ):
        from feature_extract.extractors import ViTExtractor

        self.model_name = model_name
        self.stride = stride
        self.layer = layer
        self.facet = facet
        self.array_name = array_name
        self.extractor = ViTExtractor(model_type=model_name, stride=stride, device=device)
        self.layer_specs = (
            TokenLayerSpec(
                name=array_name,
                model=model_name,
                layer=str(layer),
                channels=_dinov2_channels(model_name),
                stride=stride,
            ),
        )

    def extract(self, image_path: Path) -> Mapping[str, np.ndarray]:
        batch, _ = self.extractor.preprocess(image_path, patch_size=self.stride)
        desc = self.extractor.extract_descriptors(
            batch.to(self.extractor.device),
            layer=self.layer,
            facet=self.facet,
            include_cls=False,
        )
        tokens = desc.squeeze(0).squeeze(0).detach().cpu().numpy()
        height, width = self.extractor.num_patches
        spatial = tokens.reshape(height, width, tokens.shape[-1]).transpose(2, 0, 1)
        return {self.array_name: spatial.astype(np.float32)}


def _load_data_config(path: Path) -> dict:
    return dict(yaml.safe_load(path.read_text()))


def build_model_extractors(
    model_names: Sequence[str],
    device: str,
    radio_repo: str,
    radio_version: str,
    dinov2_model: str,
    dinov2_stride: int,
    radio_input_width: int = 0,
    radio_input_height: int = 0,
) -> tuple[TokenExtractor, ...]:
    extractors: list[TokenExtractor] = []
    for name in model_names:
        lower = name.lower()
        if "radio" in lower:
            extractors.append(
                RadioTokenExtractor(
                    version=radio_version if name == "radio" else name,
                    device=device,
                    radio_repo=radio_repo,
                    input_width=int(radio_input_width),
                    input_height=int(radio_input_height),
                )
            )
        elif "dino" in lower:
            extractors.append(
                DinoTokenExtractor(
                    model_name=dinov2_model,
                    stride=dinov2_stride,
                    device=device,
                )
            )
        else:
            raise ValueError(f"unsupported VFM model name: {name}")
    return tuple(extractors)


def _models_from_config(config: Mapping[str, object]) -> tuple[str, ...]:
    raw = dict(config.get("raw_vfm_tokens", {}))
    models = raw.get("models", [])
    return tuple(str(model["name"]) for model in models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VFM token-bank manifest")
    parser.add_argument("--existing_npz_dir", default=None, help="Directory containing per-image NPZ tokens")
    parser.add_argument("--layer_spec", default=None, help="JSON list of token layer specs")
    parser.add_argument("--data_config", default=None, help="VFM data config for model-backed extraction")
    parser.add_argument("--image_root", default=None, help="Override image root")
    parser.add_argument("--split_file", default=None, help="Override split file")
    parser.add_argument("--output_root", default=None, help="Override token output root")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_input_width", type=int, default=0)
    parser.add_argument("--radio_input_height", type=int, default=0)
    parser.add_argument("--dinov2_model", default="dinov2_vits14")
    parser.add_argument("--dinov2_stride", type=int, default=14)
    parser.add_argument("--storage_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()

    if args.existing_npz_dir is not None:
        if args.layer_spec is None or args.scene is None:
            raise ValueError("--layer_spec and --scene are required with --existing_npz_dir")
        layer_specs = _load_layer_specs(Path(args.layer_spec))
        manifest = build_manifest_from_npz_dir(
            npz_dir=Path(args.existing_npz_dir),
            layer_specs=layer_specs,
            scene=args.scene,
            split=args.split,
        )
        manifest.to_json(Path(args.output_manifest))
        return

    if args.data_config is None:
        raise ValueError("either --existing_npz_dir or --data_config is required")
    config = _load_data_config(Path(args.data_config))
    scene = args.scene or str(config["scene"])
    image_root = Path(args.image_root or str(config["image_root"]))
    split_file = Path(args.split_file) if args.split_file else Path(dict(config["splits"])[args.split])
    output_root = Path(
        args.output_root
        or str(dict(config["raw_vfm_tokens"])["output_root"])
    ) / args.split
    model_names = (
        tuple(item.strip() for item in args.models.split(",") if item.strip())
        if args.models
        else _models_from_config(config)
    )
    image_paths = discover_image_files(
        image_root=image_root,
        split_file=split_file,
        max_images=args.max_images,
    )
    manifest = extract_token_manifest_from_images(
        image_paths=image_paths,
        image_root=image_root,
        output_root=output_root,
        scene=scene,
        split=args.split,
        extractors=build_model_extractors(
            model_names=model_names,
            device=args.device,
            radio_repo=args.radio_repo,
            radio_version=args.radio_version,
            dinov2_model=args.dinov2_model,
            dinov2_stride=args.dinov2_stride,
            radio_input_width=int(args.radio_input_width),
            radio_input_height=int(args.radio_input_height),
        ),
        storage_dtype=args.storage_dtype,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
    manifest.to_json(Path(args.output_manifest))


if __name__ == "__main__":
    main()
