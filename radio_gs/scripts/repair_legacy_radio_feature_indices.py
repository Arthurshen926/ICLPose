#!/usr/bin/env python3
"""Repair legacy RADIO feature directories created with lexicographic image order.

Legacy bug pattern:
1) image paths were sorted lexicographically (rgb_1, rgb_10, rgb_100, ...)
2) extracted features were saved as rgb_{rank}.pt using the loop rank, not
   the true numeric frame id from the source filename.

This script reconstructs the old lexicographic image order and remaps each
saved feature file rank to the true frame id of the source image.

By default it runs in dry-run mode and prints example remappings.
Use --apply to rename files in-place.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.data.benchmark_paths import extract_feature_frame_index

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FEATURE_SUBDIRS = ("backbone", "summary", "siglip2", "sam3")


def collect_legacy_image_order(image_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def collect_feature_files(feature_dir: Path, subdir: str) -> list[Path]:
    root = feature_dir / subdir
    if not root.exists():
        return []
    return sorted(root.glob("rgb_*.pt"), key=extract_feature_frame_index)


def build_mapping(image_dir: Path, feature_dir: Path) -> list[dict[str, object]]:
    image_paths = collect_legacy_image_order(image_dir)
    backbone_files = collect_feature_files(feature_dir, "backbone")
    if not backbone_files:
        raise FileNotFoundError(f"No rgb_*.pt files found in {feature_dir / 'backbone'}")
    if len(backbone_files) != len(image_paths):
        raise ValueError(
            f"Count mismatch: {len(image_paths)} images vs {len(backbone_files)} backbone features"
        )

    mapping: list[dict[str, object]] = []
    for rank, (img_path, feat_path) in enumerate(zip(image_paths, backbone_files)):
        true_frame_idx = extract_feature_frame_index(img_path)
        saved_frame_idx = extract_feature_frame_index(feat_path)
        mapping.append(
            {
                "rank": rank,
                "source_file": img_path.name,
                "saved_file": feat_path.name,
                "saved_frame_idx": saved_frame_idx,
                "true_frame_idx": true_frame_idx,
                "needs_rename": saved_frame_idx != true_frame_idx,
            }
        )
    return mapping


def apply_mapping(feature_dir: Path, mapping: list[dict[str, object]]) -> None:
    rename_pairs = [
        (int(item["saved_frame_idx"]), int(item["true_frame_idx"]))
        for item in mapping
        if bool(item["needs_rename"])
    ]
    if not rename_pairs:
        return

    for subdir in FEATURE_SUBDIRS:
        files = collect_feature_files(feature_dir, subdir)
        if not files:
            continue
        root = feature_dir / subdir

        temp_pairs: list[tuple[Path, Path]] = []
        final_pairs: list[tuple[Path, Path]] = []
        for saved_idx, true_idx in rename_pairs:
            src = root / f"rgb_{saved_idx}.pt"
            if not src.exists():
                continue
            tmp = root / f"__tmp_repair_rgb_{saved_idx}.pt"
            dst = root / f"rgb_{true_idx}.pt"
            temp_pairs.append((src, tmp))
            final_pairs.append((tmp, dst))

        for src, tmp in temp_pairs:
            src.rename(tmp)
        for tmp, dst in final_pairs:
            if dst.exists():
                raise FileExistsError(f"Destination already exists during repair: {dst}")
            tmp.rename(dst)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_repaired_copy(feature_dir: Path, output_dir: Path, mapping: list[dict[str, object]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    rename_map = {
        int(item["saved_frame_idx"]): int(item["true_frame_idx"])
        for item in mapping
    }

    for item in feature_dir.iterdir():
        if item.is_file():
            link_or_copy(item, output_dir / item.name)

    for subdir in FEATURE_SUBDIRS:
        src_root = feature_dir / subdir
        if not src_root.exists():
            continue
        dst_root = output_dir / subdir
        dst_root.mkdir(parents=True, exist_ok=False)

        for src in collect_feature_files(feature_dir, subdir):
            saved_idx = extract_feature_frame_index(src)
            true_idx = rename_map[saved_idx]
            dst = dst_root / f"rgb_{true_idx}.pt"
            if dst.exists():
                raise FileExistsError(f"Destination already exists during copy repair: {dst}")
            link_or_copy(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair legacy RADIO feature frame indices")
    parser.add_argument("--image_dir", required=True, help="Directory containing source rgb_*.png images")
    parser.add_argument("--feature_dir", required=True, help="Feature directory containing backbone/, summary/, ...")
    parser.add_argument("--apply", action="store_true", help="Actually rename files in-place")
    parser.add_argument(
        "--output_dir",
        help="Write a repaired copy to a new directory instead of renaming the source in-place",
    )
    parser.add_argument("--show", type=int, default=20, help="Number of mapping rows to print")
    args = parser.parse_args()
    if args.apply and args.output_dir:
        parser.error("--apply and --output_dir are mutually exclusive")

    image_dir = Path(args.image_dir)
    feature_dir = Path(args.feature_dir)
    mapping = build_mapping(image_dir, feature_dir)
    renamed = [item for item in mapping if bool(item["needs_rename"])]
    output_dir = Path(args.output_dir) if args.output_dir else None

    print("=" * 60)
    print("LEGACY RADIO FEATURE INDEX AUDIT")
    print("=" * 60)
    print(f"image_dir   : {image_dir}")
    print(f"feature_dir : {feature_dir}")
    print(f"frames      : {len(mapping)}")
    print(f"needs_rename: {len(renamed)}")
    print()
    print("Sample remappings:")
    for item in renamed[: args.show]:
        print(
            f"  rank={item['rank']:>4}  "
            f"{item['saved_file']} <- {item['source_file']}  "
            f"rename_to=rgb_{item['true_frame_idx']}.pt"
        )

    if output_dir is not None:
        write_repaired_copy(feature_dir, output_dir, mapping)
        manifest_root = output_dir
    else:
        manifest_root = feature_dir

    manifest_path = manifest_root / "legacy_repair_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "image_dir": str(image_dir.resolve()),
                "feature_dir": str(feature_dir.resolve()),
                "output_dir": str(output_dir.resolve()) if output_dir is not None else None,
                "num_frames": len(mapping),
                "num_renames": len(renamed),
                "mapping": mapping,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Manifest written to: {manifest_path}")

    if output_dir is not None:
        print(f"Wrote repaired copy to: {output_dir}")
        return

    if not args.apply:
        print("Dry run only. Re-run with --apply to rename files in-place.")
        return

    apply_mapping(feature_dir, mapping)
    print("Applied in-place feature index repair.")


if __name__ == "__main__":
    main()
