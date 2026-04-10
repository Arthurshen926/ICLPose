#!/usr/bin/env python3
"""
Extract and save RADIO features for all training images in a scene.

Usage:
    python radio_gs/scripts/extract_radio_features.py \
        --scene room_0 \
        --image_dir dataset/room_0/Sequence_1/rgb/ \
        --output_dir output/radio_features/room_0/ \
        --radio_repo /root/RADIO \
        --radio_version c-radio_v4-h \
        --batch_size 4 \
        --extract_adaptors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.data.benchmark_paths import extract_feature_frame_index

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---- image loading helpers ------------------------------------------------

def _collect_image_paths(image_dir: str) -> tuple[list[Path], str]:
    """Return image paths with numeric frame-order when indices are parseable."""
    paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    indexed: list[tuple[int, Path]] = []
    for path in paths:
        try:
            indexed.append((extract_feature_frame_index(path), path))
        except ValueError:
            indexed = []
            break

    if indexed:
        indexed.sort(key=lambda item: item[0])
        return [path for _, path in indexed], "numeric"

    return sorted(paths), "lexicographic"


def _nearest_radio_resolution(h: int, w: int, patch_size: int = 16) -> tuple[int, int]:
    """Round h, w to nearest multiples of *patch_size*."""
    return (
        max(patch_size, round(h / patch_size) * patch_size),
        max(patch_size, round(w / patch_size) * patch_size),
    )


def _load_and_preprocess(
    paths: list[Path],
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Load images, resize, normalize to [0, 1] and stack → (B, 3, H, W)."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0)
        tensors.append(t)
    return torch.stack(tensors).to(device)


# ---- model loading --------------------------------------------------------

def _load_radio_model(
    radio_repo: str,
    radio_version: str,
    adaptor_names: list[str] | None,
    device: torch.device,
):
    """Load RADIO via torch.hub and return (model, input_conditioner)."""
    kwargs: dict = {"source": "local"}
    if adaptor_names:
        kwargs["adaptor_names"] = adaptor_names

    model = torch.hub.load(radio_repo, "radio_model", version=radio_version, **kwargs)
    model = model.to(device).eval()

    input_conditioner = model.make_preprocessor_external()

    return model, input_conditioner


# ---- PCA statistics -------------------------------------------------------

def _compute_pca_stats(
    all_features: list[torch.Tensor],
    n_components: int = 64,
) -> dict[str, torch.Tensor]:
    """Compute mean, std, and top-*n_components* PCA components from features.

    Args:
        all_features: list of (C, Hp, Wp) tensors (float32).
        n_components: number of PCA components to keep.

    Returns:
        dict with 'mean' (C,), 'std' (C,), 'components_64' (n_components, C).
    """
    # Stack all spatial tokens: (N_total, C)
    pixels = torch.cat([f.reshape(f.shape[0], -1).T for f in all_features], dim=0)
    mean = pixels.mean(dim=0)
    std = pixels.std(dim=0).clamp(min=1e-6)
    centered = pixels - mean
    # Economy SVD — only need top components
    k = min(n_components, centered.shape[0], centered.shape[1])
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:k]  # (k, C)
    return {"mean": mean, "std": std, "components_64": components}


# ---- main extraction loop -------------------------------------------------

@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine adaptor setup
    adaptor_names: list[str] | None = None
    if args.extract_adaptors:
        adaptor_names = ["siglip2-g", "sam3"]
        print(f"[RADIO] Extracting with adaptors: {adaptor_names}")

    # Load model
    print(f"[RADIO] Loading {args.radio_version} from {args.radio_repo} …")
    model, conditioner = _load_radio_model(
        args.radio_repo, args.radio_version, adaptor_names, device
    )

    # Collect images
    image_paths, image_sort_mode = _collect_image_paths(args.image_dir)
    print(f"[RADIO] Found {len(image_paths)} images in {args.image_dir}")
    print(f"[RADIO] Image ordering: {image_sort_mode}")

    # Probe resolution from first image
    probe_img = Image.open(image_paths[0])
    orig_w, orig_h = probe_img.size
    target_h, target_w = _nearest_radio_resolution(orig_h, orig_w, patch_size=16)
    patch_h, patch_w = target_h // 16, target_w // 16
    print(f"[RADIO] Input resolution: {orig_h}×{orig_w} → padded {target_h}×{target_w}")
    print(f"[RADIO] Feature grid: {patch_h}×{patch_w}")

    # Prepare output dirs
    subdirs = ["backbone", "summary"]
    if args.extract_adaptors:
        subdirs += ["siglip2", "sam3"]
    for sd in subdirs:
        os.makedirs(os.path.join(args.output_dir, sd), exist_ok=True)

    pca_accumulator: list[torch.Tensor] = []
    frame_manifest: list[dict[str, object]] = []
    total_bytes: int = 0
    t0 = time.time()

    # Process in batches
    n = len(image_paths)
    for start in tqdm(range(0, n, args.batch_size), desc="Extracting RADIO features"):
        batch_paths = image_paths[start : start + args.batch_size]
        imgs = _load_and_preprocess(batch_paths, target_h, target_w, device)

        # Apply RADIO-specific normalization
        imgs = conditioner(imgs)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(imgs)

        # Unpack backbone features
        if isinstance(output, tuple) and len(output) == 2:
            summary, spatial = output  # summary: (B, D_sum), spatial: (B, N, D)
            adaptor_outputs = {}
        else:
            # With adaptors: output is (summary, spatial, adaptor_dict)
            summary, spatial, adaptor_outputs = output

        B, N, D = spatial.shape
        spatial_2d = spatial.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)

        # Save per-frame
        for i in range(B):
            source_path = batch_paths[i]
            source_rank = start + i
            try:
                frame_idx = extract_feature_frame_index(source_path)
            except ValueError:
                frame_idx = source_rank
            stem = f"rgb_{frame_idx}"
            frame_manifest.append(
                {
                    "source_rank": source_rank,
                    "frame_idx": frame_idx,
                    "source_file": source_path.name,
                    "saved_stem": stem,
                }
            )

            # Backbone: float16
            bb = spatial_2d[i].cpu().half()
            bb_path = os.path.join(args.output_dir, "backbone", f"{stem}.pt")
            torch.save(bb, bb_path)
            total_bytes += bb.nelement() * bb.element_size()

            # Summary: float32
            sm = summary[i].cpu().float()
            sm_path = os.path.join(args.output_dir, "summary", f"{stem}.pt")
            torch.save(sm, sm_path)
            total_bytes += sm.nelement() * sm.element_size()

            # Accumulate for PCA (convert back to float32)
            pca_accumulator.append(bb.float())

            # Adaptor features
            if args.extract_adaptors and adaptor_outputs:
                for name in ["siglip2-g", "sam3"]:
                    if name not in adaptor_outputs:
                        continue
                    ad_out = adaptor_outputs[name]
                    ad_spatial = ad_out.get("spatial", ad_out) if isinstance(ad_out, dict) else ad_out
                    if ad_spatial.ndim == 3:
                        # (B, N, D_ad) → (B, D_ad, Hp, Wp)
                        D_ad = ad_spatial.shape[-1]
                        ad_2d = ad_spatial.permute(0, 2, 1).reshape(B, D_ad, patch_h, patch_w)
                    else:
                        ad_2d = ad_spatial
                    short_name = name.replace("-g", "").replace("-", "")  # siglip2, sam3
                    ad_frame = ad_2d[i].cpu().half()
                    ad_path = os.path.join(args.output_dir, short_name, f"{stem}.pt")
                    torch.save(ad_frame, ad_path)
                    total_bytes += ad_frame.nelement() * ad_frame.element_size()

    # PCA statistics
    print("[RADIO] Computing PCA statistics …")
    pca_stats = _compute_pca_stats(pca_accumulator, n_components=64)
    pca_path = os.path.join(args.output_dir, "pca_stats.pt")
    torch.save(pca_stats, pca_path)

    manifest_path = Path(args.output_dir) / "frame_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scene": args.scene,
                "image_dir": str(Path(args.image_dir).resolve()),
                "image_sort_mode": image_sort_mode,
                "num_frames": len(frame_manifest),
                "frames": frame_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Summary
    elapsed = time.time() - t0
    disk_mb = total_bytes / (1024 * 1024)
    print("=" * 60)
    print(f"  Scene       : {args.scene}")
    print(f"  Frames      : {n}")
    print(f"  Backbone dim: {D} × {patch_h}×{patch_w}")
    print(f"  Summary dim : {summary.shape[-1]}")
    print(f"  Disk usage  : {disk_mb:.1f} MB  (float16 spatial + float32 summary)")
    print(f"  PCA saved   : {pca_path}")
    print(f"  Manifest    : {manifest_path}")
    print(f"  Time        : {elapsed:.1f}s  ({elapsed / n:.2f}s/frame)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RADIO features for all images in a scene."
    )
    parser.add_argument("--scene", type=str, default="room_0", help="Scene name")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory of RGB images (.png/.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save extracted features",
    )
    parser.add_argument(
        "--radio_repo",
        type=str,
        default="/root/RADIO",
        help="Local path to RADIO repo for torch.hub",
    )
    parser.add_argument(
        "--radio_version",
        type=str,
        default="c-radio_v4-h",
        help="RADIO model version string",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Images per batch")
    parser.add_argument(
        "--extract_adaptors",
        action="store_true",
        help="Also extract SigLIP2 and SAM3 adaptor features",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device (default: cuda)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)",
    )

    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
