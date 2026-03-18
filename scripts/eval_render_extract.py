#!/usr/bin/env python3
"""Evaluate with render-then-extract inference.

Instead of warping stored features from another frame, render RGB from
3DGS at the estimated pose, extract SD+DINOv2 features from the rendered
image, apply PCA compression, and use those as reference features.

This ELIMINATES view-dependence because both query and reference features
are extracted at the SAME viewpoint.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_render_extract.py \
        --config configs/exp059_oh_depth_warp.yaml \
        --checkpoint output/exp059_oh_depth_warp/checkpoints/best.pth \
        --pca_dir output/features_multiscale_pca/OldHospital_indexed/pca_params \
        --iters 1 3 5
"""
import sys, yaml, torch, math, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4

# Monkey-patch for load_scene
import feature_3dgs.train_2dgs_geometry as _tmod
class _FakeArgs:
    init_from_depth = False
    sensor_depth_dir = None
_tmod.args = _FakeArgs()

from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS
from gsplat import rasterization_2dgs


class RenderExtractor:
    """Render RGB from 3DGS and extract SD+DINOv2 features + PCA compress."""

    def __init__(self, ply_path, pca_dir, device='cuda',
                 fx=1663.12, fy=1663.12, cx=960.0, cy=540.0,
                 img_width=1920, img_height=1080, dino_stride=7):
        self.device = device
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.img_width, self.img_height = img_width, img_height

        # Load 3DGS for RGB rendering
        print("[RenderExtractor] Loading 3DGS model...")
        self.gaussians = GaussianModel2DGS(sh_degree=3)
        self.gaussians.load_ply(ply_path)

        self.K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=torch.float32, device=device
        )

        # Pre-extract Gaussian properties (avoid repeated property calls)
        self._cache_gaussian_props()

        # Load feature extractor
        print("[RenderExtractor] Loading feature extractor (SD + DINOv2)...")
        from feature_extraction.multiscale_extractor import MultiScaleFeatureExtractor
        self.extractor = MultiScaleFeatureExtractor(
            device=device, dino_stride=dino_stride
        )

        # Load PCA parameters
        print("[RenderExtractor] Loading PCA parameters...")
        self.pca = {}
        for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
            d = np.load(f"{pca_dir}/{scale}_pca.npz")
            self.pca[scale] = {
                'mean': torch.from_numpy(d['mean']).float().to(device),
                'components': torch.from_numpy(d['components']).float().to(device),
            }
        print("[RenderExtractor] Ready.")

    def _cache_gaussian_props(self):
        """Cache Gaussian properties for fast repeated rendering."""
        self._means = self.gaussians.get_xyz
        self._opacity = self.gaussians.get_opacity.squeeze(-1)
        scales_2d = self.gaussians.get_scaling
        self._scales = torch.cat([
            scales_2d,
            torch.ones(scales_2d.shape[0], 1, device=self.device)
        ], dim=-1)
        self._rotations = self.gaussians.get_rotation
        self._colors = self.gaussians.get_features  # SH coefficients
        self._sh_degree = self.gaussians.active_sh_degree

    def render_rgb(self, pose_w2c):
        """Render RGB from 3DGS at given pose.

        Args:
            pose_w2c: (4, 4) world-to-camera matrix
        Returns:
            PIL Image of rendered RGB
        """
        with torch.no_grad():
            render_colors, *_ = rasterization_2dgs(
                means=self._means,
                quats=self._rotations,
                scales=self._scales,
                opacities=self._opacity,
                colors=self._colors,
                viewmats=pose_w2c[None],
                Ks=self.K[None],
                width=self.img_width,
                height=self.img_height,
                packed=False,
                sh_degree=self._sh_degree,
                backgrounds=torch.zeros(1, 4, device=self.device),
                near_plane=0.01, far_plane=500,
                render_mode="RGB+ED",
            )

        rgb = render_colors[0, :, :, :3]  # [H, W, 3]
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(rgb_np)

    def extract_and_compress(self, pil_image):
        """Extract features from PIL image and apply PCA compression.

        Args:
            pil_image: PIL Image
        Returns:
            dict of {scale: (1, D_pca, H, W)} PCA-compressed features
        """
        with torch.no_grad():
            ms = self.extractor.extract(pil_image)

        result = {}
        for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
            feat = getattr(ms, scale).to(self.device)  # [C, H, W]
            C, H, W = feat.shape

            # PCA transform
            mean = self.pca[scale]['mean']       # [C]
            comp = self.pca[scale]['components']  # [D_pca, C]

            flat = feat.reshape(C, -1).T          # [HW, C]
            projected = (flat - mean) @ comp.T    # [HW, D_pca]
            D_pca = projected.shape[1]
            result[scale] = projected.T.reshape(1, D_pca, H, W)  # [1, D_pca, H, W]

        return result

    def get_ref_features(self, pose_w2c):
        """Render + extract + compress for a single pose.

        Args:
            pose_w2c: (4, 4) world-to-camera matrix
        Returns:
            dict of {scale: (1, D_pca, H, W)}
        """
        pil_img = self.render_rgb(pose_w2c)
        return self.extract_and_compress(pil_img)


def build_model(cfg, device):
    mc = cfg['model']
    rc = cfg.get('renderer', {})
    intrinsics = {
        'fx': rc.get('fx', 320.0),
        'fy': rc.get('fy', 320.0),
        'cx': rc.get('cx', 319.5),
        'cy': rc.get('cy', 239.5),
    }
    model = MSFlowPoseNet(
        hidden_dim=mc.get('hidden_dim', 128),
        decode_dim=mc.get('decode_dim', 64),
        local_radius=mc.get('local_radius', 4),
        damping=mc.get('damping', 0.001),
        coarse_hw=tuple(mc.get('coarse_hw', [7, 10])),
        mid_hw=tuple(mc.get('mid_hw', [15, 20])),
        fine_hw=tuple(mc.get('fine_hw', [35, 46])),
        fine_iters=mc.get('fine_iters', 4),
        mid_iters=mc.get('mid_iters', 1),
        corr_temperature=mc.get('corr_temperature', 1.0),
        intrinsics=intrinsics,
        img_hw=(rc.get('img_height', 480), rc.get('img_width', 640)),
        coarse_in_dim=mc.get('coarse_in_dim', 512),
        mid_in_dim=mc.get('mid_in_dim', 512),
        fine_sd_in_dim=mc.get('fine_sd_in_dim', 512),
        fine_dino_in_dim=mc.get('fine_dino_in_dim', 768),
    ).to(device)
    return model


def build_renderer(cfg, device):
    rc = cfg['renderer']
    return MultiScaleRenderer(
        ply_path=rc['ply_path'],
        scale_model_paths=rc['scale_model_paths'],
        device=device,
        img_height=rc.get('img_height', 480),
        img_width=rc.get('img_width', 640),
        fx=rc.get('fx', 320.0),
        fy=rc.get('fy', 320.0),
        cx=rc.get('cx', 319.5),
        cy=rc.get('cy', 239.5),
    )


def evaluate_render_extract(model, renderer, render_extractor, val_loader,
                            num_iters, device, reextract_interval=1):
    """Evaluate using render-then-extract.

    For each outer iteration:
    1. Render RGB from 3DGS at estimated pose
    2. Extract SD+DINOv2 features and PCA compress
    3. Render depth from 3DGS
    4. Forward model with query features + extracted ref features + depth
    5. Update pose

    Args:
        reextract_interval: re-extract features every N iterations (1 = every iter)
    """
    model.eval()
    all_rot, all_trans = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"render-extract iters={num_iters}"):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)
            B = pose_cur.shape[0]

            ref_feats_cache = None

            for oi in range(num_iters):
                # Re-extract features at intervals
                should_extract = (ref_feats_cache is None or
                                  oi % reextract_interval == 0)

                if should_extract:
                    # Extract features for each sample in batch
                    batch_ref_feats = {scale: [] for scale in qf.keys()}
                    for b in range(B):
                        ref_feats_b = render_extractor.get_ref_features(
                            pose_cur[b].float()
                        )
                        for scale in ref_feats_b:
                            if scale in batch_ref_feats:
                                batch_ref_feats[scale].append(ref_feats_b[scale])

                    ref_feats_cache = {
                        scale: torch.cat(batch_ref_feats[scale], dim=0)
                        for scale in batch_ref_feats
                    }

                # Render depth at current estimated pose
                depth = renderer.render_depth_batch(pose_cur.float())

                # Forward model
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, ref_feats_cache, depth)

                if oi < num_iters - 1 and 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            # Final pose
            if 'delta_xi' in pred:
                T = se3_exp(pred['delta_xi'].float())
                pp = torch.bmm(T, pose_cur.float())

                Rr = torch.bmm(
                    pp[:, :3, :3].transpose(1, 2),
                    pose_gt.float()[:, :3, :3]
                )
                tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                re = torch.acos(ca) * 180 / math.pi
                te = torch.norm(
                    pp[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1
                ) * 1000
                all_rot.extend(re.cpu().tolist())
                all_trans.extend(te.cpu().tolist())

    r = np.array(all_rot)
    t = np.array(all_trans)
    print(f"  render-extract iters={num_iters}: "
          f"rot={np.nanmean(r):.2f}° (med {np.nanmedian(r):.2f}°)  "
          f"trans={np.nanmean(t):.1f}mm  "
          f"<1°={np.mean(r < 1.0) * 100:.1f}%  "
          f"joint@1°/50mm={np.mean((r < 1.0) & (t < 50.0)) * 100:.1f}%")
    return {
        'rot_mean': float(np.nanmean(r)),
        'rot_med': float(np.nanmedian(r)),
        'trans_mean': float(np.nanmean(t)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pca_dir', required=True,
                        help='Directory with PCA params (coarse_pca.npz etc.)')
    parser.add_argument('--iters', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--reextract_interval', type=int, default=1,
                        help='Re-extract features every N iterations (1=every iter)')
    parser.add_argument('--dino_stride', type=int, default=7)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda')

    # Build model + load checkpoint
    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

    # Build depth renderer
    renderer = build_renderer(cfg, device)

    # Build render-extract pipeline
    rc = cfg['renderer']
    render_extractor = RenderExtractor(
        ply_path=rc['ply_path'],
        pca_dir=args.pca_dir,
        device=str(device),
        fx=rc.get('fx', 1663.12),
        fy=rc.get('fy', 1663.12),
        cx=rc.get('cx', 960.0),
        cy=rc.get('cy', 540.0),
        img_width=rc.get('img_width', 1920),
        img_height=rc.get('img_height', 1080),
        dino_stride=args.dino_stride,
    )

    # Build val loader
    dc = cfg['data']
    full_ds = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc['train_traj_path'],
        noise_rot_deg=dc.get('val_noise_rot_deg', 10.0),
        noise_trans_m=dc.get('val_noise_trans_m', 0.5),
        is_train=True,
    )
    n_val = max(1, int(len(full_ds) * dc.get('val_split_ratio', 0.1)))
    n_train = len(full_ds) - n_val
    _, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # batch_size=1 for render-extract (sequential rendering)
        shuffle=False,
        num_workers=0,  # No multiprocessing (CUDA in workers)
        collate_fn=collate_v4,
        pin_memory=False,
    )

    print(f"\n{'='*60}")
    print(f"Render-Then-Extract Evaluation")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Re-extract interval: {args.reextract_interval}")
    print(f"{'='*60}")

    for n_iter in args.iters:
        evaluate_render_extract(
            model, renderer, render_extractor, val_loader,
            n_iter, device,
            reextract_interval=args.reextract_interval,
        )


if __name__ == '__main__':
    main()
