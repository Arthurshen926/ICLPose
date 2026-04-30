#!/usr/bin/env python3
from __future__ import annotations

"""Export adaptive RADIO teacher targets in the cached dual-feature format.

The query-student trainer consumes cached ``fine_geo`` and ``coarse_sem``
features. Adaptive DCFF checkpoints keep the learned RADIO compressor inside
``projection_state``; this script materializes those online targets so Stage A
query-map training can use the same 96d/32d space without running RADIO in the
dataloader.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_field.utils.checkpoint_io import safe_torch_load
from feature_field.utils.dcff_eval_targets import (
    build_teacher_target_provider,
    default_images_dir,
    infer_feature_dims,
)
from feature_field.utils.scene_colmap import build_da3_image_order, load_scene_colmap


def make_feature_filename(fid: int, scale_name: str, channels: int, height: int, width: int) -> str:
    return f"rgb_{int(fid)}_{scale_name}_{int(channels)}x{int(height)}x{int(width)}.pt"


def resolve_camera_fid(cam, name_to_fid: dict[str, int]) -> int | None:
    image_name = str(cam.image_name).replace("\\", "/")
    candidates = [
        image_name,
        Path(image_name).as_posix(),
        Path(image_name).name,
    ]
    if image_name.startswith("images/"):
        candidates.append(image_name[len("images/"):])
    if image_name.startswith("processed/"):
        candidates.append(image_name[len("processed/"):])
    for candidate in candidates:
        if candidate in name_to_fid:
            return int(name_to_fid[candidate])
    return None


def _load_config_and_checkpoint(config_path: str | None, checkpoint_path: str | None) -> tuple[dict, dict]:
    ckpt = {}
    cfg = None
    if checkpoint_path:
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
        cfg = copy.deepcopy(ckpt.get("config", {}))
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            file_cfg = yaml.safe_load(handle) or {}
        cfg = file_cfg if cfg is None else _deep_merge(cfg, file_cfg)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("A valid --checkpoint or --config is required")
    return cfg, ckpt


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _select_cameras(source_dir: str, images_subdir: str, split: str):
    train_cams, test_cams, _, _, _ = load_scene_colmap(source_dir, images_subdir)
    if split == "train":
        return train_cams
    if split == "test":
        return test_cams
    if split == "all":
        return train_cams + test_cams
    return test_cams if test_cams else train_cams


def _resized_hw(cam, longest_edge: int | None) -> tuple[int, int]:
    with Image.open(cam.image) as image:
        width, height = image.size
    if longest_edge and longest_edge > 0:
        scale = float(longest_edge) / max(width, height)
        if scale < 1.0:
            width, height = int(width * scale), int(height * scale)
    return int(height), int(width)


def _camera_batches(cams, batch_size: int, longest_edge: int | None):
    """Yield batches with matching resized image shapes."""
    pending = []
    pending_hw = None
    for cam in cams:
        hw = _resized_hw(cam, longest_edge)
        if pending and (len(pending) >= batch_size or hw != pending_hw):
            yield pending
            pending = []
            pending_hw = None
        pending.append(cam)
        pending_hw = hw
    if pending:
        yield pending


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Adaptive DCFF checkpoint with projection_state")
    parser.add_argument("--config", default=None, help="Optional DCFF config override")
    parser.add_argument("--source_dir", default="/hy-tmp/Cambridge_stdloc/OldHospital")
    parser.add_argument("--images_subdir", default="processed")
    parser.add_argument("--feature_dir", default=None, help="Optional source feature dir for cached/PCA init fallback")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_split", choices=["train", "test", "all", "auto"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input_longest_edge", type=int, default=None,
                        help="Override adaptive teacher input_longest_edge; use 1920 for full Cambridge resolution.")
    parser.add_argument("--coarse_downsample", choices=["auto", "true", "false"], default="auto",
                        help="Downsample coarse targets by 2. auto follows checkpoint training.coarse_downsample.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--strict_fids", action="store_true", help="Fail instead of skipping cameras missing DA3 ids.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg, ckpt = _load_config_and_checkpoint(args.config, args.checkpoint)
    if args.feature_dir:
        cfg.setdefault("dataset", {})["feature_dir"] = args.feature_dir
    if args.input_longest_edge is not None:
        cfg.setdefault("teacher", {})["input_longest_edge"] = int(args.input_longest_edge)
        cfg.setdefault("training", {})["longest_edge"] = int(args.input_longest_edge)
    if args.coarse_downsample == "auto":
        coarse_downsample = bool(cfg.get("training", {}).get("coarse_downsample", False))
    else:
        coarse_downsample = args.coarse_downsample == "true"

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    images_dir = default_images_dir(args.source_dir, args.images_subdir)
    name_to_fid = build_da3_image_order(images_dir)
    cameras = _select_cameras(args.source_dir, args.images_subdir, args.camera_split)

    selected = []
    skipped_missing = []
    for cam in cameras:
        fid = resolve_camera_fid(cam, name_to_fid)
        if fid is None:
            skipped_missing.append(cam.image_name)
            continue
        selected.append((cam, fid))
    if args.strict_fids and skipped_missing:
        raise KeyError(f"Missing DA3 ids for cameras: {skipped_missing[:8]}")
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    if not selected:
        raise RuntimeError("No cameras selected for adaptive teacher export")

    provider = build_teacher_target_provider(
        cfg,
        ckpt,
        source_dir=args.source_dir,
        feature_dir=cfg.get("dataset", {}).get("feature_dir"),
        device=device,
        images_subdir=args.images_subdir,
    )
    fine_dim, coarse_dim = infer_feature_dims(cfg)

    output_dir = Path(args.output_dir)
    fine_dir = output_dir / "fine_geo"
    coarse_dir = output_dir / "coarse_sem"
    fine_dir.mkdir(parents=True, exist_ok=True)
    coarse_dir.mkdir(parents=True, exist_ok=True)

    export_index = []
    exported = 0
    longest_edge = getattr(provider, "input_longest_edge", args.input_longest_edge)
    cam_only = [cam for cam, _ in selected]
    fid_by_name = {str(cam.image_name).replace("\\", "/"): fid for cam, fid in selected}

    for batch_cams in _camera_batches(cam_only, max(1, int(args.batch_size)), longest_edge):
        fine_batch, coarse_batch, _ = provider.get_batch(batch_cams)
        if coarse_downsample:
            coarse_batch = F.interpolate(
                coarse_batch.float(),
                size=(fine_batch.shape[-2] // 2, fine_batch.shape[-1] // 2),
                mode="bilinear",
                align_corners=False,
            )
        fine_batch = fine_batch.detach().float().cpu()
        coarse_batch = coarse_batch.detach().float().cpu()
        for cam, fine, coarse in zip(batch_cams, fine_batch, coarse_batch):
            image_name = str(cam.image_name).replace("\\", "/")
            fid = fid_by_name[image_name]
            c_f, h_f, w_f = fine.shape
            c_c, h_c, w_c = coarse.shape
            if int(c_f) != fine_dim or int(c_c) != coarse_dim:
                raise ValueError(
                    f"Unexpected teacher dims for {image_name}: fine={c_f}, coarse={c_c}; "
                    f"expected {fine_dim}/{coarse_dim}"
                )
            fine_path = fine_dir / make_feature_filename(fid, "fine_geo", c_f, h_f, w_f)
            coarse_path = coarse_dir / make_feature_filename(fid, "coarse_sem", c_c, h_c, w_c)
            if not (args.skip_existing and fine_path.exists() and coarse_path.exists()):
                for stale_path in fine_dir.glob(f"rgb_{fid}_fine_geo_*.pt"):
                    if stale_path != fine_path:
                        stale_path.unlink()
                for stale_path in coarse_dir.glob(f"rgb_{fid}_coarse_sem_*.pt"):
                    if stale_path != coarse_path:
                        stale_path.unlink()
                torch.save(fine.half(), fine_path)
                torch.save(coarse.half(), coarse_path)
            export_index.append(
                {
                    "teacher_idx": int(fid),
                    "sample_name": image_name,
                    "colmap_image_id": None,
                    "fine_shape": [int(c_f), int(h_f), int(w_f)],
                    "coarse_shape": [int(c_c), int(h_c), int(w_c)],
                }
            )
            exported += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "config": str(Path(args.config).resolve()) if args.config else None,
        "source_dir": str(Path(args.source_dir).resolve()),
        "images_subdir": args.images_subdir,
        "images_dir": str(Path(images_dir).resolve()),
        "camera_split": args.camera_split,
        "num_records": exported,
        "skipped_missing_fids": skipped_missing,
        "teacher_mode": getattr(provider, "mode", None),
        "input_longest_edge": int(longest_edge) if longest_edge is not None else None,
        "coarse_downsample": bool(coarse_downsample),
        "feature_hw": export_index[0]["fine_shape"][1:] if export_index else None,
        "coarse_feature_hw": export_index[0]["coarse_shape"][1:] if export_index else None,
        "fine_feature_dim": fine_dim,
        "coarse_feature_dim": coarse_dim,
    }
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "export_index.json").write_text(json.dumps(export_index, indent=2) + "\n", encoding="utf-8")
    print(
        f"Exported {exported} adaptive teacher samples to {output_dir} "
        f"(dims={fine_dim}/{coarse_dim}, hw={metadata['feature_hw']}/{metadata['coarse_feature_hw']}, "
        f"mode={metadata['teacher_mode']})"
    )
    if skipped_missing:
        print(f"Skipped {len(skipped_missing)} cameras without DA3 ids")


if __name__ == "__main__":
    main()
