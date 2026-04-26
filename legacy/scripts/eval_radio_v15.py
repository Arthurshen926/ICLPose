#!/usr/bin/env python3
"""Fixed-seed evaluation of RadioPoseNet v15 — test N=0,1,3,5 outer iters."""
import sys, os, math, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ic_models.radio_pose_net import RadioPoseNet
from scripts.train_radio_pose import RadioRenderer, RadioPoseDataset

torch.set_grad_enabled(False)
torch.manual_seed(42)
np.random.seed(42)

# Load config
cfg = yaml.safe_load(open('configs/radio_pose_oh_v15.yaml'))
rc = cfg['renderer']
mc = cfg['model']
dc = cfg['data']
device = torch.device('cuda')

# Setup renderer
renderer = RadioRenderer(
    rc['ply_path'], rc['feature_model_path'], device,
    tuple(rc['img_hw']), rc['fx'], rc['fy'], rc['cx'], rc['cy']
)

# Build model
intrinsics = {k: rc[k] for k in ('fx', 'fy', 'cx', 'cy')}
model = RadioPoseNet(
    feat_dim=mc.get('feat_dim', 64),
    hidden_dim=mc.get('hidden_dim', 128),
    n_heads=mc.get('n_heads', 4),
    n_attn_layers=mc.get('n_attn_layers', 2),
    ffn_dim=mc.get('ffn_dim', 128),
    local_radius=mc.get('local_radius', 4),
    fine_iters=mc.get('fine_iters', 4),
    damping=mc.get('damping', 0.001),
    coarse_hw=tuple(mc.get('coarse_hw', [17, 30])),
    fine_hw=tuple(mc.get('fine_hw', [34, 60])),
    intrinsics=intrinsics,
    img_hw=tuple(rc.get('img_hw', [1080, 1920])),
    conf_floor=mc.get('conf_floor', 0.0),
    detach_conf_in_solver=mc.get('detach_conf_in_solver', False),
    solver_hw=tuple(mc['solver_hw']) if 'solver_hw' in mc else None,
    sequential_solve=mc.get('sequential_solve', False),
    solver_trans_scale=mc.get('solver_trans_scale', 1.0),
    shared_projection=mc.get('shared_projection', False),
).to(device).eval()

# Load best checkpoint
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'output/radio_pose_oh_v15/checkpoints/best.pth'
if not os.path.exists(ckpt_path):
    ckpt_path = 'output/radio_pose_oh_v15/checkpoints/latest.pth'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Loaded {ckpt_path}, epoch={ckpt.get('epoch', '?')}")

# Val dataset — with noise (is_train=True)
val_indices = None
idx_file = dc.get('val_index_file', dc.get('test_index_file'))
if idx_file and os.path.exists(idx_file):
    val_indices = torch.load(idx_file)
    if isinstance(val_indices, dict):
        val_indices = val_indices.get('test', val_indices.get('val'))
    val_indices = list(val_indices) if val_indices is not None else None

noise_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
noise_trans = noise_deg / 32.0  # scale: 8° → 0.25m, 3° → 0.094m

val_ds = RadioPoseDataset(
    dc['feature_dir'], dc['traj_path'],
    frame_indices=val_indices,
    noise_rot_deg=noise_deg,
    noise_trans_m=noise_trans,
    is_train=True,  # adds random noise
)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

def rot_error_deg(R1, R2):
    R_rel = R1 @ R2.T
    trace = R_rel[0,0] + R_rel[1,1] + R_rel[2,2]
    cos_a = ((trace - 1) / 2).clamp(-1, 1)
    return torch.acos(cos_a).item() * 180 / math.pi

def run_eval(n_iters):
    """Run full val set with n_iters outer iterations, return rot/trans errors."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    rot_init, rot_final, trans_init, trans_final = [], [], [], []
    pred_flow_mags, gt_flow_mags = [], []
    
    for batch in val_loader:
        query_feat = batch['query_feat'].to(device)
        pose_gt = batch['pose_gt'].to(device)
        pose_init = batch['initial_pose'].to(device)
        B = pose_gt.shape[0]

        # Init errors
        for b in range(B):
            c2w_i = torch.linalg.inv(pose_init[b])
            c2w_g = torch.linalg.inv(pose_gt[b])
            rot_init.append(rot_error_deg(c2w_i[:3,:3], c2w_g[:3,:3]))
            trans_init.append((c2w_i[:3,3] - c2w_g[:3,3]).norm().item() * 100)
        
        if n_iters == 0:
            # No refinement — copy init as final
            for b in range(B):
                c2w_i = torch.linalg.inv(pose_init[b])
                c2w_g = torch.linalg.inv(pose_gt[b])
                rot_final.append(rot_error_deg(c2w_i[:3,:3], c2w_g[:3,:3]))
                trans_final.append((c2w_i[:3,3] - c2w_g[:3,3]).norm().item() * 100)
            continue

        # Iterative refinement
        from modules.lie_algebra import se3_exp
        pose_cur = pose_init.clone()
        for it in range(n_iters):
            render_feat = renderer.render_features(pose_cur, (68, 120))
            depth = renderer.render_depth(pose_cur, (68, 120))
            result = model(query_feat, render_feat, depth)
            delta_xi = result['delta_xi']
            delta_T = se3_exp(delta_xi)
            pose_cur = (delta_T @ pose_cur).detach()
            
            # Flow stats on first iteration only
            if it == 0 and n_iters <= 3:
                pred_flow = result.get('flow_fine', result.get('flow_coarse'))
                if pred_flow is not None:
                    pred_flow_mags.append(pred_flow.norm(dim=1).mean().item())
                gt_flow = result.get('gt_flow_fine')
                if gt_flow is not None:
                    gt_flow_mags.append(gt_flow.norm(dim=1).mean().item())

        # Final errors
        for b in range(B):
            c2w_f = torch.linalg.inv(pose_cur[b])
            c2w_g = torch.linalg.inv(pose_gt[b])
            rot_final.append(rot_error_deg(c2w_f[:3,:3], c2w_g[:3,:3]))
            trans_final.append((c2w_f[:3,3] - c2w_g[:3,3]).norm().item() * 100)
    
    rot_init = np.array(rot_init)
    rot_final = np.array(rot_final)
    trans_init = np.array(trans_init)
    trans_final = np.array(trans_final)
    
    return {
        'rot_init_med': np.median(rot_init),
        'rot_final_med': np.median(rot_final),
        'rot_delta': np.median(rot_final) - np.median(rot_init),
        'trans_init_med': np.median(trans_init),
        'trans_final_med': np.median(trans_final),
        'trans_delta': np.median(trans_final) - np.median(trans_init),
        'n_samples': len(rot_init),
        'pred_flow_mag': np.mean(pred_flow_mags) if pred_flow_mags else 0,
    }

print("\n" + "=" * 70)
print(f"FIXED-SEED EVALUATION ({noise_deg}° noise, {len(val_ds)} val frames)")
print("=" * 70)

for n in [0, 1, 3, 5]:
    r = run_eval(n)
    print(f"\nN={n} iters ({r['n_samples']} samples):")
    print(f"  Rotation:    init={r['rot_init_med']:.2f}° → final={r['rot_final_med']:.2f}° (Δ={r['rot_delta']:+.2f}°)")
    print(f"  Translation: init={r['trans_init_med']:.1f}cm → final={r['trans_final_med']:.1f}cm (Δ={r['trans_delta']:+.1f}cm)")
    if r['pred_flow_mag'] > 0:
        print(f"  Pred flow magnitude: {r['pred_flow_mag']:.3f}px")
