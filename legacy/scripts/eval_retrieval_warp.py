#!/usr/bin/env python3
"""Evaluate with retrieval-based warp inference.

Instead of 3DGS feature rendering, warp stored features from the
nearest training frame to the estimated pose using depth-based warping.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_retrieval_warp.py \
        --config configs/exp059_oh_depth_warp.yaml \
        --checkpoint output/exp059_oh_depth_warp/checkpoints/best.pth \
        --iters 1 3 5
"""
import sys, yaml, torch, math, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.depth_warp import backward_warp_features
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4, load_poses_c2w, c2w_to_w2c
from torch.utils.data import DataLoader
from tqdm import tqdm


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


class TrainingFeatureIndex:
    """Index of training features for nearest-neighbor retrieval."""

    def __init__(self, feature_dir, traj_path, train_indices, device='cpu'):
        """Load all training features and poses into memory."""
        from data.dataset_v4 import PoseDatasetV4

        # Load ALL features (train set only)
        ds = PoseDatasetV4(
            feature_base_dir=feature_dir,
            traj_path=traj_path,
            frame_indices=train_indices,
            noise_rot_deg=0.0,  # no noise for reference features
            noise_trans_m=0.0,
            is_train=False,
        )

        self.poses_w2c = torch.from_numpy(ds.poses_w2c).float()  # (N, 4, 4)
        self.positions = self.poses_w2c[:, :3, 3]  # (N, 3) camera positions in w2c

        # Precompute positions in world frame for distance computation
        poses_c2w = load_poses_c2w(traj_path)
        if train_indices is not None:
            poses_c2w = poses_c2w[train_indices]
        self.positions_world = torch.from_numpy(
            poses_c2w[:, :3, 3].astype(np.float32)
        )  # (N, 3)

        # Load all features into memory
        print(f"[RetrievalIndex] Caching {len(ds)} frames of features...")
        self.features = []  # list of {scale: (C, H, W)}
        for i in tqdm(range(len(ds)), desc="Loading features"):
            item = ds[i]
            self.features.append({
                k: v for k, v in item['query_feats'].items()
            })
        # Store w2c poses
        self.train_poses_w2c = self.poses_w2c.clone()
        print(f"[RetrievalIndex] Cached {len(self.features)} frames")

    def find_nearest(self, pose_w2c, k=1):
        """Find k nearest training frames by position distance.

        Args:
            pose_w2c: (B, 4, 4) estimated w2c poses
        Returns:
            indices: (B, k) nearest training frame indices
        """
        # Convert w2c to world position (camera center in world frame)
        # For w2c: T_wc, camera center = -R^T @ t
        R = pose_w2c[:, :3, :3]  # (B, 3, 3)
        t = pose_w2c[:, :3, 3]  # (B, 3)
        cam_pos = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # Compute distances to all training positions
        dists = torch.cdist(cam_pos.cpu(), self.positions_world)  # (B, N)
        _, indices = dists.topk(k, largest=False)  # (B, k)
        return indices

    def get_features(self, idx, device):
        """Get features for a single training frame index.

        Args:
            idx: scalar index
        Returns:
            {scale: (1, C, H, W)} on device
        """
        feats = self.features[idx]
        return {k: v.unsqueeze(0).to(device) for k, v in feats.items()}

    def get_nearest_features_batch(self, pose_w2c, device, k=1):
        """Get features from nearest training frame for each batch item.

        Args:
            pose_w2c: (B, 4, 4)
            k: number of nearest neighbors per batch item
        Returns:
            if k==1: ref_feats: {scale: (B, C, H, W)}, ref_poses: (B, 4, 4)
            if k>1:  list of k tuples (ref_feats, ref_poses)
        """
        indices = self.find_nearest(pose_w2c, k=k)  # (B, k)
        B = pose_w2c.shape[0]

        if k == 1:
            ref_feats_list = {scale: [] for scale in self.features[0].keys()}
            ref_poses = []
            for b in range(B):
                idx = indices[b, 0].item()
                feats = self.features[idx]
                for scale in feats:
                    ref_feats_list[scale].append(feats[scale])
                ref_poses.append(self.train_poses_w2c[idx])
            ref_feats = {
                scale: torch.stack(ref_feats_list[scale]).to(device)
                for scale in ref_feats_list
            }
            ref_poses_tensor = torch.stack(ref_poses).to(device)
            return ref_feats, ref_poses_tensor
        else:
            # Return list of k (ref_feats, ref_poses) tuples
            result = []
            for ki in range(k):
                ref_feats_list = {scale: [] for scale in self.features[0].keys()}
                ref_poses = []
                for b in range(B):
                    idx = indices[b, ki].item()
                    feats = self.features[idx]
                    for scale in feats:
                        ref_feats_list[scale].append(feats[scale])
                    ref_poses.append(self.train_poses_w2c[idx])
                ref_feats = {
                    scale: torch.stack(ref_feats_list[scale]).to(device)
                    for scale in ref_feats_list
                }
                ref_poses_tensor = torch.stack(ref_poses).to(device)
                result.append((ref_feats, ref_poses_tensor))
            return result


def evaluate_retrieval_warp(model, renderer, val_loader, feat_index,
                            scale_intrinsics, num_iters, device):
    """Evaluate using retrieval-based warp instead of 3DGS rendering."""
    model.eval()
    all_rot, all_trans = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"retrieval-warp iters={num_iters}", leave=False):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)

            for oi in range(num_iters):
                # 1. Find nearest training frame
                ref_feats, ref_poses = feat_index.get_nearest_features_batch(
                    pose_cur, device)

                # 2. Render depth at current estimated pose
                depth = renderer.render_depth_batch(pose_cur.float())

                # 3. Warp reference features from ref pose to estimated pose
                warped_feats = backward_warp_features(
                    ref_feats, depth,
                    ref_poses.float(),  # "GT" pose = reference frame's known pose
                    pose_cur.float(),   # estimated pose
                    scale_intrinsics,
                )

                # 4. Run model
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, warped_feats, depth)

                if oi < num_iters - 1 and 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            # Final pose
            if 'delta_xi' in pred:
                T = se3_exp(pred['delta_xi'].float())
                pp = torch.bmm(T, pose_cur.float())

                Rr = torch.bmm(pp[:, :3, :3].transpose(1, 2), pose_gt.float()[:, :3, :3])
                tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                re = torch.acos(ca) * 180 / math.pi
                te = torch.norm(pp[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000
                all_rot.extend(re.cpu().tolist())
                all_trans.extend(te.cpu().tolist())

    r = np.array(all_rot)
    t = np.array(all_trans)
    print(f"  retrieval-warp iters={num_iters}: "
          f"rot={np.nanmean(r):.2f}° (med {np.nanmedian(r):.2f}°)  "
          f"trans={np.nanmean(t):.1f}mm  "
          f"<1°={np.mean(r < 1.0) * 100:.1f}%  "
          f"joint@1°/50mm={np.mean((r < 1.0) & (t < 50.0)) * 100:.1f}%")
    return {'rot_mean': float(np.nanmean(r)), 'rot_med': float(np.nanmedian(r)),
            'trans_mean': float(np.nanmean(t))}


def evaluate_multiframe_warp(model, renderer, val_loader, feat_index,
                             scale_intrinsics, num_iters, num_frames, device):
    """Evaluate using multi-frame aggregated feature warping.

    For each iteration, warp features from K nearest neighbors and average them.
    This reduces view-dependent noise from any single frame.
    """
    model.eval()
    all_rot, all_trans = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader,
                         desc=f"multiframe-warp K={num_frames} iters={num_iters}",
                         leave=False):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)

            for oi in range(num_iters):
                # 1. Find K nearest training frames
                knn_list = feat_index.get_nearest_features_batch(
                    pose_cur, device, k=num_frames)

                # 2. Render depth at current estimated pose
                depth = renderer.render_depth_batch(pose_cur.float())

                # 3. Warp features from ALL K neighbors and aggregate
                warped_accum = None
                valid_count = None
                for ki, (ref_feats_k, ref_poses_k) in enumerate(knn_list):
                    warped_k = backward_warp_features(
                        ref_feats_k, depth,
                        ref_poses_k.float(), pose_cur.float(),
                        scale_intrinsics,
                    )
                    if warped_accum is None:
                        warped_accum = {s: w.clone() for s, w in warped_k.items()}
                        valid_count = {s: (w.abs().sum(1, keepdim=True) > 1e-6).float()
                                       for s, w in warped_k.items()}
                    else:
                        for s in warped_accum:
                            mask = (warped_k[s].abs().sum(1, keepdim=True) > 1e-6).float()
                            warped_accum[s] = warped_accum[s] + warped_k[s]
                            valid_count[s] = valid_count[s] + mask

                # Average over valid contributions
                warped_feats = {}
                for s in warped_accum:
                    count = valid_count[s].clamp(min=1.0)
                    warped_feats[s] = warped_accum[s] / count

                # 4. Run model
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, warped_feats, depth)

                if oi < num_iters - 1 and 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            # Final pose
            if 'delta_xi' in pred:
                T = se3_exp(pred['delta_xi'].float())
                pp = torch.bmm(T, pose_cur.float())

                Rr = torch.bmm(pp[:, :3, :3].transpose(1, 2), pose_gt.float()[:, :3, :3])
                tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                re = torch.acos(ca) * 180 / math.pi
                te = torch.norm(pp[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000
                all_rot.extend(re.cpu().tolist())
                all_trans.extend(te.cpu().tolist())

    r = np.array(all_rot)
    t = np.array(all_trans)
    print(f"  multiframe-warp K={num_frames} iters={num_iters}: "
          f"rot={np.nanmean(r):.2f}° (med {np.nanmedian(r):.2f}°)  "
          f"trans={np.nanmean(t):.1f}mm  "
          f"<1°={np.mean(r < 1.0) * 100:.1f}%  "
          f"joint@1°/50mm={np.mean((r < 1.0) & (t < 50.0)) * 100:.1f}%")
    return {'rot_mean': float(np.nanmean(r)), 'rot_med': float(np.nanmedian(r)),
            'trans_mean': float(np.nanmean(t))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--iters', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--multi-frames', type=int, nargs='+', default=None,
                        help='Number of nearest frames for multi-frame aggregation (e.g., 3 5 10)')
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

    # Build renderer (for depth only)
    renderer = build_renderer(cfg, device)
    scale_intrinsics = renderer.get_scale_intrinsics()

    # Build val loader (split from train)
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
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=dc.get('val_batch_size', 4),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_v4,
        pin_memory=True,
    )

    # Build training feature index (for retrieval)
    # Use training indices from the split
    train_indices = sorted(train_ds.indices)
    feat_index = TrainingFeatureIndex(
        feature_dir=dc['train_feature_dir'],
        traj_path=dc['train_traj_path'],
        train_indices=train_indices,
        device='cpu',
    )

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Retrieval-Based Warp Evaluation")
    print(f"{'='*60}")
    for n_iter in args.iters:
        evaluate_retrieval_warp(
            model, renderer, val_loader, feat_index,
            scale_intrinsics, n_iter, device)

    # Multi-frame aggregation evaluation
    if args.multi_frames:
        print(f"\n{'='*60}")
        print(f"Multi-Frame Aggregated Warp Evaluation")
        print(f"{'='*60}")
        for K in args.multi_frames:
            for n_iter in args.iters:
                evaluate_multiframe_warp(
                    model, renderer, val_loader, feat_index,
                    scale_intrinsics, n_iter, K, device)


if __name__ == '__main__':
    main()
