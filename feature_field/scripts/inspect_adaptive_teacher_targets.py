#!/usr/bin/env python3
from __future__ import annotations

"""Inspect online/adaptive RADIO teacher target quality at different sizes."""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_field.dcff.radio_teacher import OnlineRadioTeacher
from feature_field.utils.checkpoint_io import safe_torch_load
from feature_field.utils.scene_colmap import load_scene_colmap
from feature_field.visualize_feature_comparison import target_basis_pca_colorize


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _load_config(config_path: str | None, checkpoint_path: str | None) -> tuple[dict, dict]:
    ckpt: dict = {}
    if checkpoint_path:
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
        cfg = dict(ckpt.get("config", {}))
    elif config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError("Provide --config or --checkpoint")
    if not isinstance(cfg, dict):
        raise ValueError("Could not load a valid DCFF config")
    return cfg, ckpt


def _select_cameras(source_dir: str, images_subdir: str, split: str, indices: list[int]):
    train_cams, test_cams, _, _, _ = load_scene_colmap(source_dir, images_subdir)
    if split == "train":
        cams = train_cams
    elif split == "test":
        cams = test_cams
    elif split == "all":
        cams = train_cams + test_cams
    else:
        cams = test_cams if test_cams else train_cams
    return [cams[i] for i in indices if 0 <= i < len(cams)]


def _resize_image_tensor(cam, longest_edge: int, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(cam.image).convert("RGB")
    width, height = img.size
    scale = float(longest_edge) / max(width, height)
    if scale < 1.0:
        width, height = int(width * scale), int(height * scale)
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return tensor, (height, width)


def _load_display_image(cam, max_edge: int = 640):
    img = Image.open(cam.image).convert("RGB")
    width, height = img.size
    scale = float(max_edge) / max(width, height)
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return img


def _build_teacher(cfg: dict, ckpt: dict, device: torch.device) -> OnlineRadioTeacher:
    teacher_cfg = cfg.get("teacher", {})
    compress_cfg = teacher_cfg.get("compress", {})
    model_cfg = cfg.get("model", {})
    feature_dir = cfg.get("dataset", {}).get("feature_dir", "")
    pca_init_dir = teacher_cfg.get("pca_init_dir")
    if pca_init_dir is None:
        candidate = Path(feature_dir) / "pca_params"
        pca_init_dir = str(candidate) if candidate.is_dir() else None

    fallback_dim = int(model_cfg.get("feature_dim", 64))
    fine_dim = int(model_cfg.get("fine_feature_dim", fallback_dim))
    coarse_dim = int(model_cfg.get("coarse_feature_dim", fallback_dim))
    teacher = OnlineRadioTeacher(
        target_dim=fallback_dim,
        fine_dim=fine_dim,
        coarse_dim=coarse_dim,
        bottleneck=(teacher_cfg.get("mode", "cached") == "online_bottleneck"),
        shallow_block=teacher_cfg.get("shallow_block", 10),
        radio_repo=teacher_cfg.get("radio_repo", "feature_extract/checkpoints/RADIO"),
        pca_init_dir=pca_init_dir,
        compress_hidden_dim=int(compress_cfg.get("hidden_dim", 256)),
        sample_pixels=int(compress_cfg.get("sample_pixels", 1024)),
        recon_chunk_pixels=int(compress_cfg.get("recon_chunk_pixels", 4096)),
        recon_cos_weight=float(compress_cfg.get("recon_cos_weight", 1.0)),
        recon_l1_weight=float(compress_cfg.get("recon_l1_weight", 0.25)),
        min_spatial_std=float(compress_cfg.get("min_spatial_std", 0.0)),
        std_weight=float(compress_cfg.get("std_weight", 0.0)),
        decorrelation_weight=float(compress_cfg.get("decorrelation_weight", 0.0)),
        fine_raw_highpass_kernel=int(compress_cfg.get("fine_raw_highpass_kernel", 0)),
        coarse_raw_highpass_kernel=int(compress_cfg.get("coarse_raw_highpass_kernel", 0)),
        fine_adapter_highpass_kernel=int(compress_cfg.get("fine_adapter_highpass_kernel", 0)),
        coarse_adapter_highpass_kernel=int(compress_cfg.get("coarse_adapter_highpass_kernel", 0)),
        normalize_output=bool(compress_cfg.get("normalize_output", True)),
    ).to(device)
    if "projection_state" in ckpt:
        teacher.load_projection_state(ckpt["projection_state"])
    teacher.eval()
    return teacher


def _feature_stats(feat: torch.Tensor) -> dict[str, float]:
    x = F.normalize(feat.float(), p=2, dim=1)
    flat = x.flatten(2)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return {
        "channel_spatial_std": float(flat.std(dim=-1, unbiased=False).mean().item()),
        "grad_l1": float(0.5 * (dx.abs().mean().item() + dy.abs().mean().item())),
        "grad_l2": float(0.5 * (dx.norm(dim=1).mean().item() + dy.norm(dim=1).mean().item())),
        "norm_mean": float(feat.float().norm(dim=1).mean().item()),
    }


def _save_edge_figure(
    output_path: Path,
    edge: int,
    rows: list[dict],
    fine_vis: list[torch.Tensor],
    coarse_vis: list[torch.Tensor],
) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13, max(3, 3.2 * n)))
    if n == 1:
        axes = np.asarray([axes])

    for row_idx, (row, fine_rgb, coarse_rgb) in enumerate(zip(rows, fine_vis, coarse_vis)):
        axes[row_idx, 0].imshow(_load_display_image(row["cam"]))
        axes[row_idx, 0].set_title(f"RGB {row['image_name']}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(fine_rgb.permute(1, 2, 0).clamp(0, 1).cpu())
        axes[row_idx, 1].set_title(
            f"Fine target {row['fine_shape'][3]}x{row['fine_shape'][2]}\n"
            f"std={row['fine_stats']['channel_spatial_std']:.3f} grad={row['fine_stats']['grad_l2']:.3f}"
        )
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(coarse_rgb.permute(1, 2, 0).clamp(0, 1).cpu())
        axes[row_idx, 2].set_title(
            f"Coarse target {row['coarse_shape'][3]}x{row['coarse_shape'][2]}\n"
            f"std={row['coarse_stats']['channel_spatial_std']:.3f} grad={row['coarse_stats']['grad_l2']:.3f}"
        )
        axes[row_idx, 2].axis("off")

    fig.suptitle(f"Adaptive teacher targets, input_longest_edge={edge}", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Adaptive DCFF checkpoint with projection_state")
    parser.add_argument("--config", default=None, help="Config used when no checkpoint is provided")
    parser.add_argument("--source_dir", default="/hy-tmp/Cambridge_stdloc/OldHospital")
    parser.add_argument("--images_subdir", default="processed")
    parser.add_argument("--camera_split", default="train", choices=["train", "test", "all", "auto"])
    parser.add_argument("--camera_indices", default="0,20,50,100,200,400")
    parser.add_argument("--input_longest_edges", default="640,1280")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, ckpt = _load_config(args.config, args.checkpoint)
    teacher = _build_teacher(cfg, ckpt, device)
    cameras = _select_cameras(
        args.source_dir,
        args.images_subdir,
        args.camera_split,
        _parse_int_list(args.camera_indices),
    )
    if not cameras:
        raise RuntimeError("No cameras selected for teacher target inspection")

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "camera_split": args.camera_split,
        "num_cameras": len(cameras),
        "edges": {},
    }

    for edge in _parse_int_list(args.input_longest_edges):
        rows = []
        fine_targets = []
        coarse_targets = []
        for cam in cameras:
            image, (height, width) = _resize_image_tensor(cam, edge, device)
            teacher.set_image_size(height, width)
            with torch.no_grad():
                fine_raw, coarse_raw = teacher.extract_raw(image)
                fine, coarse = teacher.project(fine_raw, coarse_raw)
            row = {
                "cam": cam,
                "image_name": cam.image_name,
                "resized_hw": [height, width],
                "radio_hw": list(getattr(teacher, "_radio_input_size", (fine.shape[-2] * teacher.patch_size, fine.shape[-1] * teacher.patch_size))),
                "fine_shape": list(fine.shape),
                "coarse_shape": list(coarse.shape),
                "fine_stats": _feature_stats(fine),
                "coarse_stats": _feature_stats(coarse),
            }
            rows.append(row)
            fine_targets.append(fine.squeeze(0).detach().cpu())
            coarse_targets.append(coarse.squeeze(0).detach().cpu())
            del image, fine_raw, coarse_raw, fine, coarse
            torch.cuda.empty_cache()

        _, fine_vis = target_basis_pca_colorize(fine_targets, fine_targets)
        _, coarse_vis = target_basis_pca_colorize(coarse_targets, coarse_targets)
        _save_edge_figure(
            output_dir / f"adaptive_teacher_targets_edge{edge}.png",
            edge,
            rows,
            fine_vis,
            coarse_vis,
        )

        edge_rows = []
        for row in rows:
            cleaned = {k: v for k, v in row.items() if k != "cam"}
            edge_rows.append(cleaned)
        summary["edges"][str(edge)] = {
            "fine_shape": edge_rows[0]["fine_shape"],
            "coarse_shape": edge_rows[0]["coarse_shape"],
            "fine_grad_l2_mean": float(np.mean([r["fine_stats"]["grad_l2"] for r in edge_rows])),
            "coarse_grad_l2_mean": float(np.mean([r["coarse_stats"]["grad_l2"] for r in edge_rows])),
            "fine_std_mean": float(np.mean([r["fine_stats"]["channel_spatial_std"] for r in edge_rows])),
            "coarse_std_mean": float(np.mean([r["coarse_stats"]["channel_spatial_std"] for r in edge_rows])),
            "rows": edge_rows,
        }
        print(
            f"[edge={edge}] fine={edge_rows[0]['fine_shape']} "
            f"coarse={edge_rows[0]['coarse_shape']} "
            f"fine_grad={summary['edges'][str(edge)]['fine_grad_l2_mean']:.4f} "
            f"coarse_grad={summary['edges'][str(edge)]['coarse_grad_l2_mean']:.4f}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved teacher target inspection to {output_dir}")


if __name__ == "__main__":
    main()
