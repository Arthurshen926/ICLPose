"""
Visualize RADIO-GS rendered features via PCA decomposition.

Produces side-by-side comparisons and a summary grid:
  GT RGB | GT RADIO features (PCA) | Rendered features (PCA) | Cosine map

Usage:
    python radio_gs/scripts/visualize_features.py \
        --config radio_gs/configs/replica_explicit_v9.yaml \
        --checkpoint output/radio_gs/room0_explicit_v9/checkpoints/best.pth \
        --num_views 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.config import load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer


def features_to_pca_rgb(features: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """Convert high-dim feature map to RGB via PCA.

    Args:
        features: [C, H, W] feature tensor.

    Returns:
        [H, W, 3] uint8 numpy array.
    """
    C, H, W = features.shape
    feat_flat = features.reshape(C, -1).T.cpu().numpy()  # [HW, C]

    mean = feat_flat.mean(axis=0)
    centered = feat_flat - mean

    # Fast PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_proj = U[:, :n_components] * S[:n_components]  # [HW, 3]

    # Normalize to [0, 1]
    for i in range(n_components):
        vmin, vmax = pca_proj[:, i].min(), pca_proj[:, i].max()
        if vmax - vmin > 1e-8:
            pca_proj[:, i] = (pca_proj[:, i] - vmin) / (vmax - vmin)
        else:
            pca_proj[:, i] = 0.5

    rgb = (pca_proj.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb


def shared_pca_colorize(features_list, n_components=3):
    """Apply a SHARED PCA to multiple feature maps for comparable visualization."""
    all_flat = []
    shapes = []
    for feat in features_list:
        C, H, W = feat.shape
        shapes.append((H, W))
        all_flat.append(feat.reshape(C, -1).T.cpu().numpy())  # [HW, C]

    stacked = np.concatenate(all_flat, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, S_all, Vt_all = np.linalg.svd(centered, full_matrices=False)
    basis = Vt_all[:n_components]  # [3, C]

    results = []
    offset = 0
    for (H, W) in shapes:
        n = H * W
        proj = (centered[offset:offset + n] @ basis.T)  # [HW, 3]
        offset += n
        for c in range(n_components):
            vmin, vmax = proj[:, c].min(), proj[:, c].max()
            if vmax - vmin > 1e-6:
                proj[:, c] = (proj[:, c] - vmin) / (vmax - vmin)
            else:
                proj[:, c] = 0.5
        results.append((proj.reshape(H, W, n_components) * 255).astype(np.uint8))
    return results


def cosine_heatmap(cos_map: np.ndarray) -> np.ndarray:
    """Convert cosine similarity map [H,W] → colorized [H,W,3]."""
    cos_clipped = np.clip(cos_map, 0, 1)
    heatmap = cv2.applyColorMap((cos_clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def load_pipeline(config_path, checkpoint_path, device):
    config = load_config(config_path)
    model = ExplicitFeatureGaussian(latent_dim=getattr(config, "latent_dim", 64))
    model.load_from_ply(getattr(config, "ply_path", ""))
    model = model.to(device).eval()

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
    ).to(device).eval()

    renderer = FeatureFieldRenderer(
        image_height=getattr(config, "feature_height", 30),
        image_width=getattr(config, "feature_width", 40),
        fx=getattr(config, "fx", 320.0) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=getattr(config, "latent_dim", 64),
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    if getattr(config, "use_refiner", False):
        extra_ch = 3 if getattr(config, "refiner_rgb_guide", False) else 0
        refiner = ScreenSpaceRefiner(
            latent_dim=getattr(config, "latent_dim", 64),
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
        ).to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)

    return model, codec, renderer, sharpener, refiner, config


def main():
    parser = argparse.ArgumentParser(description="RADIO-GS Feature Visualization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--scale", type=int, default=6, help="Upscale factor for display")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, codec, renderer, sharpener, refiner, config = load_pipeline(
        args.config, args.checkpoint, device
    )

    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    split_name = (
        getattr(config, "val_split", "Sequence_2")
        if args.split == "val"
        else getattr(config, "train_split", "Sequence_1")
    )
    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)
    feature_size = (fH, fW)
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)

    output_dir = Path(args.output_dir or f"output/radio_gs/visualizations/{scene}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load poses
    pose_file = scene_root / split_name / "traj_w_c.txt"
    all_poses = np.loadtxt(str(pose_file)).reshape(-1, 4, 4).astype(np.float32)
    all_w2c = np.linalg.inv(all_poses)
    total = len(all_w2c)

    gt_feat_dir = Path(f"output/radio_features_1280d/room_0/{split_name}/backbone")
    rgb_dir = scene_root / split_name / "rgb"

    indices = list(range(0, total, max(1, total // args.num_views)))[:args.num_views]
    print(f"Visualizing {len(indices)} frames from {split_name}")
    print(f"Output: {output_dir}")

    # Render and collect features
    gt_feats_1280 = []
    rendered_feats_1280 = []
    gt_rgbs = []
    cosine_maps = []

    with torch.no_grad():
        for i in indices:
            # GT feature
            gt = torch.load(gt_feat_dir / f"rgb_{i}.pt").float()
            if gt.dim() == 2:
                gt = gt.reshape(fH, fW, -1).permute(2, 0, 1)
            gt_feats_1280.append(gt)

            # Render + decode
            pose = torch.from_numpy(all_w2c[i:i + 1]).to(device)
            result = renderer.render_features_batch(model, pose)
            rendered_compact = sharpener(result["feature_map"])

            if refiner is not None:
                guide = None
                if rgb_guide_enabled:
                    rgb_path = rgb_dir / f"rgb_{i}.png"
                    if rgb_path.exists():
                        img = cv2.imread(str(rgb_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (fW, fH))
                        guide = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                rendered_compact = refiner(rendered_compact, guide=guide)

            decoded = codec.decoder(rendered_compact).squeeze(0).cpu()  # [1280, H, W]
            rendered_feats_1280.append(decoded)

            # Per-pixel cosine similarity
            cos = F.cosine_similarity(
                decoded.reshape(-1, fH * fW), gt.reshape(-1, fH * fW), dim=0
            ).reshape(fH, fW).numpy()
            cosine_maps.append(cos)

            # GT RGB
            rgb = cv2.imread(str(rgb_dir / f"rgb_{i}.png"))
            if rgb is not None:
                gt_rgbs.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            else:
                gt_rgbs.append(np.zeros((480, 640, 3), dtype=np.uint8))

    # Shared PCA for comparable visualization
    print("Computing shared PCA across GT + rendered features...")
    all_for_pca = gt_feats_1280 + rendered_feats_1280
    pca_imgs = shared_pca_colorize(all_for_pca, n_components=3)
    gt_pcas = pca_imgs[:len(gt_feats_1280)]
    rendered_pcas = pca_imgs[len(gt_feats_1280):]

    # Per-frame comparisons + summary grid
    S = args.scale
    dH, dW = fH * S, fW * S
    n_grid = min(8, len(indices))
    grid = np.ones((n_grid * (dH + 2) + 30, dW * 4 + 14, 3), dtype=np.uint8) * 255

    # Header
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["GT RGB", "GT Feature (PCA)", "Rendered (PCA)", "Cosine Map"]
    for col, lbl in enumerate(labels):
        x = col * (dW + 4) + 4
        cv2.putText(grid, lbl, (x, 18), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    for j in range(len(indices)):
        i = indices[j]
        cos = cosine_maps[j]

        # Upscale panels
        rgb_small = cv2.resize(gt_rgbs[j], (fW, fH))
        cos_heat = cosine_heatmap(cos)

        panels = [rgb_small, gt_pcas[j], rendered_pcas[j], cos_heat]

        # Save individual comparison
        row = np.concatenate([
            cv2.resize(p, (dW, dH), interpolation=cv2.INTER_NEAREST)
            for p in panels
        ], axis=1)
        comp_path = output_dir / f"comp_{args.split}_{i:04d}_cos{cos.mean():.3f}.png"
        cv2.imwrite(str(comp_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

        # Add to grid
        if j < n_grid:
            for col, panel in enumerate(panels):
                up = cv2.resize(panel, (dW, dH), interpolation=cv2.INTER_NEAREST)
                y = j * (dH + 2) + 30
                x = col * (dW + 4) + 2
                grid[y:y + dH, x:x + dW] = up

    grid_path = output_dir / f"grid_{args.split}.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Grid saved: {grid_path}")

    # Stats
    mean_cos = np.mean([c.mean() for c in cosine_maps])
    min_cos = np.mean([c.min() for c in cosine_maps])
    print(f"\nCosine stats over {len(indices)} frames:")
    print(f"  Mean: {mean_cos:.4f}  |  Min (avg): {min_cos:.4f}")
    for j, i in enumerate(indices):
        print(f"  Frame {i:4d}: mean={cosine_maps[j].mean():.4f}  min={cosine_maps[j].min():.4f}")


if __name__ == "__main__":
    main()
