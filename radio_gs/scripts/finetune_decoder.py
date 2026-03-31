"""Stage 2: Fine-tune the HCD decoder on rendered features from trained Gaussians.

After latent-space training (V3) or end-to-end training (V5) produces good
Gaussian features, this script freezes the Gaussians (and optional refiner)
and fine-tunes the decoder to reconstruct 1280d features from rendered 64d
features (bridging distribution gap).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radio_gs.config import RadioGSConfig, load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer


class PairedDataset(Dataset):
    """Loads rendered 64d features (from 3DGS) + GT 1280d features."""
    def __init__(self, rendered_dir, gt_1280_dir):
        self.rendered_dir = Path(rendered_dir)
        self.gt_dir = Path(gt_1280_dir) / "backbone"
        if not self.gt_dir.exists():
            self.gt_dir = Path(gt_1280_dir)
        self.files = sorted(self.gt_dir.glob("rgb_*.pt"),
                            key=lambda p: int(p.stem.split("_")[1]))
        self.rendered_files = sorted(self.rendered_dir.glob("rgb_*.pt"),
                                     key=lambda p: int(p.stem.split("_")[1]))
        assert len(self.files) == len(self.rendered_files), \
            f"Mismatch: {len(self.files)} GT vs {len(self.rendered_files)} rendered"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        gt = torch.load(self.files[idx], map_location="cpu").float()
        rendered = torch.load(self.rendered_files[idx], map_location="cpu").float()
        return {"gt_1280": gt, "rendered_64": rendered}


def render_all_features(model, renderer, sharpener, pose_file, output_dir, device, refiner=None):
    """Pre-render all 64d features from trained Gaussians."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    poses = np.loadtxt(pose_file).reshape(-1, 4, 4).astype(np.float32)
    w2c = np.linalg.inv(poses)
    
    model.eval()
    print(f"Rendering {len(w2c)} frames...")
    with torch.no_grad():
        for i in tqdm(range(len(w2c)), desc="Rendering"):
            pose = torch.from_numpy(w2c[i:i+1]).to(device)
            result = renderer.render_features_batch(model, pose)
            feat = sharpener(result["feature_map"])  # [1, 64, H, W]
            if refiner is not None:
                feat = refiner(feat)
            torch.save(feat.squeeze(0).cpu(), output_dir / f"rgb_{i}.pt")
    print(f"Saved {len(w2c)} rendered features to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune decoder on rendered features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="V3 best checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    config = load_config(args.config)

    # Build components
    latent_dim = getattr(config, "latent_dim", 64)
    model = ExplicitFeatureGaussian(latent_dim=latent_dim)
    ply_path = getattr(config, "ply_path", "")
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device)
    
    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
    ).to(device)
    
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
        feature_dim=latent_dim,
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device)

    # Optional screen-space refiner
    refiner = None
    if getattr(config, "use_refiner", False):
        refiner = ScreenSpaceRefiner(
            latent_dim=latent_dim,
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
        ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Freeze Gaussians and refiner
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if refiner is not None:
        for p in refiner.parameters():
            p.requires_grad = False
        refiner.eval()

    # Pre-render all features for train+val
    output_base = Path(getattr(config, "output_dir", "output/radio_gs"))
    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    
    for split_name, split in [("train", getattr(config, "train_split", "Sequence_1")),
                               ("val", getattr(config, "val_split", "Sequence_2"))]:
        rendered_dir = output_base / "rendered_features" / split_name
        if not (rendered_dir / "rgb_0.pt").exists():
            render_all_features(
                model, renderer, sharpener,
                str(scene_root / split / "traj_w_c.txt"),
                str(rendered_dir), device, refiner=refiner
            )
        else:
            print(f"Using cached rendered features: {rendered_dir}")

    # Build datasets
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    gt_1280_base = getattr(config, "feature_dir", "").replace("64d", "1280d")
    gt_1280_val = gt_1280_base.replace(train_split, val_split)
    
    train_ds = PairedDataset(
        output_base / "rendered_features" / "train",
        gt_1280_base,
    )
    val_ds = PairedDataset(
        output_base / "rendered_features" / "val",
        gt_1280_val,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # Train only decoder
    optimizer = optim.AdamW(codec.decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    best_cos = -1.0
    out_dir = output_base / "decoder_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        codec.decoder.train()
        train_cos = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"E{epoch:03d}", leave=False):
            rendered = batch["rendered_64"].to(device)
            gt = batch["gt_1280"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                decoded = codec.decoder(rendered)
                if decoded.shape[-2:] != gt.shape[-2:]:
                    gt = F.interpolate(gt, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
                l2 = F.mse_loss(decoded, gt)
                cos = F.cosine_similarity(decoded.flatten(2), gt.flatten(2), dim=1).mean()
                loss = l2 + 2.0 * (1.0 - cos)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_cos += cos.item()
            n += 1
        
        scheduler.step()
        avg_train_cos = train_cos / max(n, 1)

        if epoch % 5 == 0 or epoch == 1:
            codec.decoder.eval()
            val_cos = 0.0
            nv = 0
            with torch.no_grad():
                for batch in val_loader:
                    rendered = batch["rendered_64"].to(device)
                    gt = batch["gt_1280"].to(device)
                    decoded = codec.decoder(rendered)
                    if decoded.shape[-2:] != gt.shape[-2:]:
                        gt = F.interpolate(gt, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
                    cos = F.cosine_similarity(decoded.flatten(2), gt.flatten(2), dim=1).mean()
                    val_cos += cos.item()
                    nv += 1
            avg_val_cos = val_cos / max(nv, 1)
            is_best = avg_val_cos > best_cos
            if is_best:
                best_cos = avg_val_cos
                # Save updated codec + model + refiner state
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "codec_state_dict": codec.state_dict(),
                    "sharpener_state_dict": sharpener.state_dict(),
                    "epoch": epoch,
                    "val_cosine": best_cos,
                }
                if refiner is not None:
                    save_dict["refiner_state_dict"] = refiner.state_dict()
                torch.save(save_dict, out_dir / "best.pth")
            marker = " ★" if is_best else ""
            print(f"[E{epoch:03d}] train_cos={avg_train_cos:.4f} val_cos={avg_val_cos:.4f}{marker}")
        else:
            print(f"[E{epoch:03d}] train_cos={avg_train_cos:.4f}")

    print(f"\nBest val cosine: {best_cos:.4f}")
    print(f"Saved to {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
